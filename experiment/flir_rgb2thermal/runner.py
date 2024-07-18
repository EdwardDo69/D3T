import time
from pathlib import Path

import numpy as np
import torch.nn.functional as F
import torchvision
import wandb
from dataset import build_train_loader
from ema import ModelEMA
from loguru import logger
from losses import HMfocalLoss

from cvpods.engine.runner import (RUNNERS, DefaultCheckpointer, DefaultRunner,
                                  DistributedDataParallel, Infinite,
                                  auto_scale_config, comm, get_bn_modules,
                                  hooks, maybe_convert_module, torch)
from cvpods.layers import cat
from cvpods.modeling.losses import iou_loss
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.structures import Boxes, pairwise_iou
import copy

@RUNNERS.register()
class SemiRunner(DefaultRunner):
    def __init__(self, cfg, build_model):

        if cfg.WANDB and comm.is_main_process():
            wandb.init(project='HT', entity='xxx')
            wandb.config.update(cfg)
            wandb.run.name = Path(cfg.OUTPUT_DIR).stem
            wandb.define_metric("AP50_tea", summary="max")
            wandb.define_metric("AP50_stu", summary="max")

        self.ema_start = cfg.TRAINER.EMA.START_STEPS
        self.step2_start = cfg.TRAINER.STEP2
        self.iter_min = cfg.TRAINER.STEP2
        self.rgb_ir_change_total = cfg.TRAINER.RGB_IR_CHANGE_TOTAL
        self.rgb_ir_change_rgb = cfg.TRAINER.RGB_IR_CHANGE_RGB
        self.rgb_ir_change_ir = self.rgb_ir_change_total - self.rgb_ir_change_rgb
        self.rgb_ir_change_increase = cfg.TRAINER.RGB_IR_CHANGE_INCREASE

        self._hooks = []

        self.data_loader = build_train_loader(cfg)
        self.best_ap50 = 0.

        model = build_model(cfg)
        # Convert SyncBN to BatchNorm if only 1 GPU exists
        self.model = maybe_convert_module(model)
        logger.info(f"Model: \n{self.model}")

        self.optimizer = self.build_optimizer(cfg, self.model)

        # Don't modify code below unless you are expert
        if True:
            if cfg.TRAINER.FP16.ENABLED:
                self.mixed_precision = True
                if cfg.TRAINER.FP16.TYPE == "APEX":
                    from apex import amp
                    self.model, self.optimizer = amp.initialize(
                        self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                    )
            else:
                self.mixed_precision = False

            # For training, wrap with DDP. But don't need this for inference.
            if comm.get_world_size() > 1:
                torch.cuda.set_device(comm.get_local_rank())
                if cfg.MODEL.DDP_BACKEND == "torch":
                    self.model = DistributedDataParallel(
                        self.model,
                        device_ids=[comm.get_local_rank()],
                        broadcast_buffers=False,
                        find_unused_parameters=True
                    )
                elif cfg.MODEL.DDP_BACKEND == "apex":
                    from apex.parallel import \
                        DistributedDataParallel as ApexDistributedDataParallel
                    self.model = ApexDistributedDataParallel(self.model)
                else:
                    raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))

            if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
                epoch_iters = -1
            else:
                epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
                logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

            auto_scale_config(cfg, self.data_loader)

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        self.model.train()
        self._data_loader_iter = iter(self.data_loader)

        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.WINDOW_SIZE

        self.cfg = cfg

        self.decay_factor = cfg.TRAINER.EMA.DECAY_FACTOR
        self.burn_in_steps = cfg.TRAINER.SSL.BURN_IN_STEPS
        self.ema_update_steps = cfg.TRAINER.EMA.UPDATE_STEPS

        self.ema_model = ModelEMA(self.model, self.decay_factor)
        self.ema_model.model.eval()
        self.ema_model_ir = ModelEMA(self.model, self.decay_factor)
        self.ema_model_ir.model.eval()
        logger.info("EMA model built!")

        self.checkpointer = DefaultCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema_model.model,
            ema_ir = self.ema_model_ir.model
        )

        self.alpha = cfg.TRAINER.DISTILL.HM.ALPHA
        self.beta = cfg.TRAINER.DISTILL.HM.BETA

        self.vfl_loss = HMfocalLoss(
            use_sigmoid=cfg.MODEL.FCOS.VFL.USE_SIGMOID,
            alpha=cfg.MODEL.FCOS.VFL.ALPHA,
            gamma=cfg.MODEL.FCOS.VFL.GAMMA,
            weight_type=cfg.MODEL.FCOS.VFL.WEIGHT_TYPE,
            loss_weight=cfg.MODEL.FCOS.VFL.LOSS_WEIGHT
        ).cuda()

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg

        ret = [
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))

        def test_and_save_results():
            logger.info('####################Evaluating: Student####################')
            self._last_eval_results = self.test(self.cfg, self.model)

            if comm.is_main_process():
                best_model = False
                if self.best_ap50 < self._last_eval_results['bbox']['AP50']:
                    best_model = True
                    self.best_ap50 = self._last_eval_results['bbox']['AP50']
                    logger.info("Best mAP: {}". format(self._last_eval_results['bbox']['AP50']))
                    logger.info("Saving checkpoint to {}".format('best'))
                    self.checkpointer.save('best')
                if self.cfg.WANDB:
                    metric = {key + '_stu': value for key, value in self._last_eval_results['bbox'].items()}
                    wandb.log(metric, step=self.iter)
                    if best_model:
                        self.checkpointer.save('best')
                        self.best_ap50 = self._last_eval_results['bbox']['AP50']
                        log_dict = {
                            'model': 'student',
                            'iter': self.iter,
                        }
                        log_dict.update(self._last_eval_results['bbox'])
                        wandb.run.summary["best_map50"] = log_dict
            return self._last_eval_results

        def test_and_save_ema_results():
            if self.iter > self.ema_start:
                logger.info('####################Evaluating: EMA RGB ####################')
                results = self.test(self.cfg, self.ema_model.model)
                if comm.is_main_process():
                    best_model = False
                    if self.best_ap50 < results['bbox']['AP50']:
                        best_model = True
                        self.best_ap50 = results['bbox']['AP50']
                        logger.info("Best mAP: {}". format(results['bbox']['AP50']))
                        logger.info("Saving checkpoint to {}".format('best'))
                        self.checkpointer.save('best')
                        ###self.checkpointer.ema_model_save('ema_best')

                    if self.cfg.WANDB:
                        metric = {key + '_tea': value for key, value in results['bbox'].items()}
                        if best_model:
                            log_dict = {
                                'model': 'student',
                                'iter': self.iter,
                            }
                            log_dict.update(results['bbox'])
                            wandb.run.summary["best_map50"] = log_dict
                        wandb.log(metric, step=self.iter)
                return results
            else:
                logger.info(f'Evaluating: EMA, EMA will start at iter {self.ema_start}')

        def test_and_save_ir_ema_results():
            if self.iter > self.step2_start:
                logger.info('#################### Evaluating: EMA_Thermal ####################')
                results = self.test(self.cfg, self.ema_model_ir.model)
                if comm.is_main_process():
                    best_model = False
                    if self.best_ap50 < results['bbox']['AP50']:
                        best_model = True
                        self.best_ap50 = results['bbox']['AP50']
                        logger.info("Best mAP: {}". format(results['bbox']['AP50']))
                        logger.info("Saving checkpoint to {}".format('best'))
                        self.checkpointer.save('best')
                        ###self.checkpointer.ema_model_save('ema_best')

                    if self.cfg.WANDB:
                        metric = {key + '_tea': value for key, value in results['bbox'].items()}
                        if best_model:
                            log_dict = {
                                'model': 'student',
                                'iter': self.iter,
                            }
                            log_dict.update(results['bbox'])
                            wandb.run.summary["best_map50"] = log_dict
                        wandb.log(metric, step=self.iter)
                return results
            else:
                logger.info(f'Evaluating: EMA, EMA will start at iter {self.ema_start}')

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_ema_results))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_ir_ema_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=self.cfg.GLOBAL.LOG_INTERVAL
            ))
        return ret

    def resume_or_load(self, resume=True):
        self.checkpointer.resume = resume

        if not getattr(self.cfg.TRAINER.EMA, 'FAKE', False):
            # self.ema_checkpointer.resume = resume
            if resume:
                self.start_iter = (self.checkpointer.resume_or_load(
                    self.cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
            else:
                self.start_iter = (self.checkpointer.resume_or_load(
                    self.cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1)
        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length

        self.scheduler.last_epoch = self.start_iter - 1

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[IterRunner] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self.epoch += 1
            if hasattr(self.data_loader.sampler, 'set_epoch'):
                self.data_loader.sampler.set_epoch(self.epoch)
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        unsup_weak, unsup_strong, sup_weak, sup_strong = [], [], [], []
        for d in data:
            uw, us, sw, ss = d
            unsup_weak.append(uw)
            unsup_strong.append(us)
            sup_weak.append(sw)
            sup_strong.append(ss)

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        ### Step1 Burn-in step with RGB = IR


        if self.iter <= self.step2_start:
            # Forward Logic
            sup_weak.extend(sup_strong)
            loss_dict_sup = self.model(sup_weak)

            loss_dict_sup = {k: v * self.cfg.TRAINER.DISTILL.SUP_WEIGHT for k,
                             v in loss_dict_sup.items()}
            losses_sup = sum([
                metrics_value for metrics_value in loss_dict_sup.values()
                if metrics_value.requires_grad
            ])

            losses_sup.backward()
            losses = losses_sup.detach()
            loss_dict = loss_dict_sup
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Teacher Update Logic
            if self.iter == self.ema_start:
                # Start
                logger.info('EMA started.')
                self.ema_model.update(self.model, decay=0.)
                self.ema_model_ir.update(self.model, decay=0.)
            elif self.iter > self.ema_start and (self.iter - self.ema_start) % self.ema_update_steps == 0:
                self.ema_model.update(self.model)
                self.ema_model_ir.update(self.model)
        # ############################################################################################################
        # ### Step 2 . 1 train IR teacher
        else:
            if (self.iter > self.iter_min) and (self.iter <= self.iter_min + self.rgb_ir_change_ir ):
                if (self.rgb_ir_change_ir >1) and (self.iter == self.iter_min + 1):
                    logger.info('Iter {}: Change train RGB --> Train IR'.format(self.iter))

                unsup_weight = self.cfg.TRAINER.DISTILL.UNSUP_WEIGHT
                if self.cfg.TRAINER.DISTILL.SUPPRESS == 'exp':
                    target = self.burn_in_steps + 2000
                    if self.iter <= target:
                        scale = np.exp((self.iter - target) / 1000)
                        unsup_weight *= scale
                elif self.cfg.TRAINER.DISTILL.SUPPRESS == 'step':
                    target = self.burn_in_steps * 2
                    if self.iter <= target:
                        unsup_weight *= 0.25
                elif self.cfg.TRAINER.DISTILL.SUPPRESS == 'linear':
                    target = self.burn_in_steps * 2
                    if self.iter <= target:
                        unsup_weight *= (self.iter - self.burn_in_steps) / self.burn_in_steps

                # 2.1.1 RGB weak --> IR strong
                student_logits_RGB2IR, student_deltas_RGB2IR, student_quality_RGB2IR, box_xyxy_RGB2IR = self.model(unsup_strong,
                                                                                                   get_data=True)
                with torch.no_grad():
                    teacher_logits_RGB2IR, teacher_deltas_RGB2IR, teacher_quality_RGB2IR, tea_box_xyxy_RGB2IR = self.ema_model.model(
                        unsup_weak,
                        is_teacher=True)
                loss_dict_unsup_RGB2IR = self.get_distill_loss(student_logits_RGB2IR, student_deltas_RGB2IR, student_quality_RGB2IR,
                                                           teacher_logits_RGB2IR, teacher_deltas_RGB2IR, teacher_quality_RGB2IR,
                                                           box_xyxy=box_xyxy_RGB2IR, tea_box_xyxy=tea_box_xyxy_RGB2IR,
                                                           name="_RGB2IR")
                distill_weights_RGB2IR = {
                    "distill_loss_logits_RGB2IR": self.cfg.TRAINER.DISTILL.WEIGHTS.LOGITS,
                    "distill_loss_deltas_RGB2IR": self.cfg.TRAINER.DISTILL.WEIGHTS.DELTAS,
                    "distill_loss_quality_RGB2IR": self.cfg.TRAINER.DISTILL.WEIGHTS.QUALITY,
                    "loss_uhl_RGB2IR": self.cfg.TRAINER.DISTILL.WEIGHTS.UHL,
                    "fore_ground_sum_RGB2IR": 1.,
                }
                unsup_weight_RGB2IR = unsup_weight * self.cfg.TRAINER.DISTILL.UNSUP_WEIGHT_STEP2
                loss_dict_unsup_RGB2IR = {k: (v * unsup_weight_RGB2IR) if v.requires_grad else v for k, v in
                                      loss_dict_unsup_RGB2IR.items()}
                loss_dict_unsup_RGB2IR = {k: v * distill_weights_RGB2IR[k] for k, v in loss_dict_unsup_RGB2IR.items()}
                losses_unsup_RGB2IR = sum([
                    metrics_value for metrics_value in loss_dict_unsup_RGB2IR.values()
                    if metrics_value.requires_grad
                ])
                losses_unsup_RGB2IR.backward()
                loss_dict = loss_dict_unsup_RGB2IR
                losses = losses_unsup_RGB2IR.detach()

                ## 2.1.2 IR weak --> IR trong
                student_logits_IR, student_deltas_IR, student_quality_IR, box_xyxy_IR = self.model(unsup_strong,
                                                                                                   get_data=True)
                with torch.no_grad():
                    teacher_logits_IR, teacher_deltas_IR, teacher_quality_IR, tea_box_xyxy_IR = self.ema_model_ir.model(
                        unsup_weak,
                        is_teacher=True)
                loss_dict_unsup_IR = self.get_distill_loss(student_logits_IR, student_deltas_IR, student_quality_IR,
                                                           teacher_logits_IR, teacher_deltas_IR, teacher_quality_IR,
                                                           box_xyxy=box_xyxy_IR, tea_box_xyxy=tea_box_xyxy_IR,
                                                           name="_IR")
                distill_weights_IR = {
                    "distill_loss_logits_IR": self.cfg.TRAINER.DISTILL.WEIGHTS.LOGITS,
                    "distill_loss_deltas_IR": self.cfg.TRAINER.DISTILL.WEIGHTS.DELTAS,
                    "distill_loss_quality_IR": self.cfg.TRAINER.DISTILL.WEIGHTS.QUALITY,
                    "loss_uhl_IR": self.cfg.TRAINER.DISTILL.WEIGHTS.UHL,
                    "fore_ground_sum_IR": 1.,
                }
                unsup_weight_IR = unsup_weight * self.cfg.TRAINER.DISTILL.UNSUP_WEIGHT_STEP2
                loss_dict_unsup_IR = {k: (v * unsup_weight_IR) if v.requires_grad else v for k, v in
                                      loss_dict_unsup_IR.items()}
                loss_dict_unsup_IR = {k: v * distill_weights_IR[k] for k, v in loss_dict_unsup_IR.items()}
                losses_unsup_IR = sum([
                    metrics_value for metrics_value in loss_dict_unsup_IR.values()
                    if metrics_value.requires_grad
                ])
                losses_unsup_IR.backward()
                loss_dict.update( loss_dict_unsup_IR)
                losses += losses_unsup_IR.detach()

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.iter > self.ema_start and (self.iter - self.ema_start) % self.ema_update_steps == 0:
                    self.ema_model_ir.update(self.model)

            ############################################################################################################
            ### Step 2 . 2 train RGB student
            if (self.iter > self.iter_min + self.rgb_ir_change_ir) and (self.iter <= self.iter_min + self.rgb_ir_change_total):
                if (self.rgb_ir_change_rgb >1) and (self.iter == self.iter_min + self.rgb_ir_change_ir + 1):
                        logger.info('Iter {}: Change train IR --> Train RGB'.format(self.iter))

                ## 2.2.2 RGB lable --> RGB weak, strong
                sup_weak_strong = copy.deepcopy(sup_weak)
                sup_weak_strong.extend(sup_strong)
                loss_dict_sup = self.model(sup_weak_strong)

                loss_dict_sup = {k: v * self.cfg.TRAINER.DISTILL.SUP_WEIGHT_STEP2 for k,
                v in loss_dict_sup.items()}
                losses_sup = sum([
                    metrics_value for metrics_value in loss_dict_sup.values()
                    if metrics_value.requires_grad
                ])
                losses_sup.backward()
                losses = losses_sup.detach()
                loss_dict = loss_dict_sup

                # Forward Logic
                unsup_weight = self.cfg.TRAINER.DISTILL.UNSUP_WEIGHT
                if self.cfg.TRAINER.DISTILL.SUPPRESS == 'exp':
                    target = self.burn_in_steps + 2000
                    if self.iter <= target:
                        scale = np.exp((self.iter - target) / 1000)
                        unsup_weight *= scale
                elif self.cfg.TRAINER.DISTILL.SUPPRESS == 'step':
                    target = self.burn_in_steps * 2
                    if self.iter <= target:
                        unsup_weight *= 0.25
                elif self.cfg.TRAINER.DISTILL.SUPPRESS == 'linear':
                    target = self.burn_in_steps * 2
                    if self.iter <= target:
                        unsup_weight *= (self.iter - self.burn_in_steps) / self.burn_in_steps

                ## 2.2.1 IR weak --> RGB strong
                student_logits_IR2RGB, student_deltas_IR2RGB, student_quality_IR2RGB, box_xyxy_IR2RGB = self.model(sup_strong,
                                                                                                       get_data=True)
                with torch.no_grad():
                    teacher_logits_IR2RGB, teacher_deltas_IR2RGB, teacher_quality_IR2RGB, tea_box_xyxy_IR2RGB = self.ema_model_ir.model(
                        sup_weak,
                        is_teacher=True)
                loss_dict_unsup_IR2RGB = self.get_distill_loss(student_logits_IR2RGB, student_deltas_IR2RGB, student_quality_IR2RGB,
                                                            teacher_logits_IR2RGB, teacher_deltas_IR2RGB, teacher_quality_IR2RGB,
                                                            box_xyxy=box_xyxy_IR2RGB, tea_box_xyxy=tea_box_xyxy_IR2RGB,
                                                            name="_IR2RGB")
                distill_weights_IR2RGB = {
                    "distill_loss_logits_IR2RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.LOGITS,
                    "distill_loss_deltas_IR2RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.DELTAS,
                    "distill_loss_quality_IR2RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.QUALITY,
                    "loss_uhl_IR2RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.UHL,
                    "fore_ground_sum_IR2RGB": 1.,
                }
                unsup_weight_IR2RGB = unsup_weight * self.cfg.TRAINER.DISTILL.UNSUP_WEIGHT_STEP2
                loss_dict_unsup_IR2RGB = {k: (v * unsup_weight_IR2RGB) if v.requires_grad else v for k, v in
                                       loss_dict_unsup_IR2RGB.items()}
                loss_dict_unsup_IR2RGB = {k: v * distill_weights_IR2RGB[k] for k, v in loss_dict_unsup_IR2RGB.items()}
                losses_unsup_IR2RGB = sum([
                    metrics_value for metrics_value in loss_dict_unsup_IR2RGB.values()
                    if metrics_value.requires_grad
                ])
                losses_unsup_IR2RGB.backward()
                loss_dict.update(loss_dict_unsup_IR2RGB)
                losses += losses_unsup_IR2RGB.detach()

                ### 2.2.3 RGB weak --> RGB strong
                student_logits_RGB, student_deltas_RGB, student_quality_RGB, box_xyxy_RGB = self.model(sup_strong,
                                                                                       get_data=True)
                with torch.no_grad():
                    teacher_logits_RGB, teacher_deltas_RGB, teacher_quality_RGB, tea_box_xyxy_RGB = self.ema_model.model(
                        sup_weak,
                        is_teacher=True)
                loss_dict_unsup_RGB = self.get_distill_loss(student_logits_RGB, student_deltas_RGB, student_quality_RGB,
                                                        teacher_logits_RGB, teacher_deltas_RGB, teacher_quality_RGB,
                                                        box_xyxy=box_xyxy_RGB, tea_box_xyxy=tea_box_xyxy_RGB, name="_RGB")
                distill_weights_RGB = {
                    "distill_loss_logits_RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.LOGITS,
                    "distill_loss_deltas_RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.DELTAS,
                    "distill_loss_quality_RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.QUALITY,
                    "loss_uhl_RGB": self.cfg.TRAINER.DISTILL.WEIGHTS.UHL,
                    "fore_ground_sum_RGB": 1.,
                }
                unsup_weight_RGB = unsup_weight * self.cfg.TRAINER.DISTILL.UNSUP_WEIGHT_STEP2
                loss_dict_unsup_RGB = {k: (v * unsup_weight_RGB) if v.requires_grad else v for k, v in
                                   loss_dict_unsup_RGB.items()}
                loss_dict_unsup_RGB = {k: v * distill_weights_RGB[k] for k, v in loss_dict_unsup_RGB.items()}
                losses_unsup_RGB = sum([
                    metrics_value for metrics_value in loss_dict_unsup_RGB.values()
                    if metrics_value.requires_grad
                ])
                losses_unsup_RGB.backward()
                loss_dict.update(loss_dict_unsup_RGB)
                losses += losses_unsup_RGB.detach()

                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.iter > self.ema_start and (self.iter - self.ema_start) % self.ema_update_steps == 0:
                    self.ema_model.update(self.model)

                if self.iter ==30000:
                    self.rgb_ir_change_ir += self.rgb_ir_change_increase
                    self.rgb_ir_change_rgb -= self.rgb_ir_change_increase
                elif self.iter ==20000:
                    self.rgb_ir_change_ir += self.rgb_ir_change_increase
                    self.rgb_ir_change_rgb -= self.rgb_ir_change_increase

                if self.iter == self.iter_min + self.rgb_ir_change_total:
                    self.iter_min = self.iter_min + self.rgb_ir_change_total

        self._detect_anomaly(losses, loss_dict)
        self._write_metrics(loss_dict, data_time)

        if self.cfg.WANDB and comm.is_main_process() and self.iter % self.cfg.GLOBAL.LOG_INTERVAL == 0:
            log_dict = {
                        'lr': self.optimizer.param_groups[0]["lr"]
                    }
            log_dict.update(loss_dict)
            wandb.log(log_dict, step=self.iter)

        self.step_outputs = {
            "loss_for_backward": losses,
        }

        self.inner_iter += 1

    def get_distill_loss(self,
                         student_logits, student_deltas, student_quality,
                         teacher_logits, teacher_deltas, teacher_quality,
                         box_xyxy=None, tea_box_xyxy=None, name=""):
        num_classes = self.cfg.MODEL.FCOS.NUM_CLASSES

        student_logits = torch.cat([
            permute_to_N_HWA_K(x, num_classes) for x in student_logits
        ], dim=1).view(-1, num_classes)
        teacher_logits = torch.cat([
            permute_to_N_HWA_K(x, num_classes) for x in teacher_logits
        ], dim=1).view(-1, num_classes)

        student_deltas = torch.cat([
            permute_to_N_HWA_K(x, 4) for x in student_deltas
        ], dim=1).view(-1, 4)
        teacher_deltas = torch.cat([
            permute_to_N_HWA_K(x, 4) for x in teacher_deltas
        ], dim=1).view(-1, 4)

        student_quality = torch.cat([
            permute_to_N_HWA_K(x, 1) for x in student_quality
        ], dim=1).view(-1, 1)
        teacher_quality = torch.cat([
            permute_to_N_HWA_K(x, 1) for x in teacher_quality
        ], dim=1).view(-1, 1)

        with torch.no_grad():
            # Region Selection
            ratio = self.cfg.TRAINER.DISTILL.RATIO
            count_num = int(teacher_logits.size(0) * ratio)
            # print("count_num: ", count_num)
            teacher_probs = teacher_logits.sigmoid()

            # harmony measure
            cls_prob = torch.max(teacher_probs, 1)[0]
            iou = teacher_quality.sigmoid().squeeze()
            hm = (cls_prob ** self.alpha) * (iou ** self.beta)
            max_vals = hm

            sorted_vals, sorted_inds = torch.topk(max_vals, teacher_logits.size(0))
            mask = torch.zeros_like(max_vals)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        delta_weghts = torch.ones_like(teacher_quality).squeeze()
        cls_prob = torch.max(teacher_probs, 1)[0]
        iou = teacher_quality.sigmoid().squeeze()
        hm = (cls_prob ** self.alpha) * (iou ** self.beta)
        loss_uncertainty = (1 - hm) / self.cfg.TRAINER.DISTILL.UN_REGULAR_ALPHA

        loss_weight = torch.exp(-loss_uncertainty.detach())
        loss_logits = QFLv2(
            student_logits.sigmoid(),
            teacher_probs,
            weight=loss_weight,
            reduction="sum",
        ) / fg_num

        loss_deltas = (iou_loss(
            student_deltas,
            teacher_deltas,
            box_mode="ltrb",
            loss_type='giou',
            reduction="none",
        ) * delta_weghts * loss_weight).mean()

        loss_quality = F.binary_cross_entropy(
            student_quality.sigmoid(),
            teacher_quality.sigmoid(),
            weight=loss_weight.unsqueeze(1),
            reduction='mean'
        )
        loss_dict = {
            "distill_loss_logits"+name: loss_logits,
            "distill_loss_quality"+name: loss_quality,
            "distill_loss_deltas"+name: loss_deltas,
            "fore_ground_sum"+name: fg_num,
        }

        N = len(box_xyxy)
        n_b_mask = b_mask.clone().view(N, -1)

        n_student_logits = student_logits.view(N, -1, num_classes)

        logits = []
        targets = []
        for idx, box_xyxy_single_sample in enumerate(box_xyxy):
            if n_b_mask[idx].sum() <= 1:
                continue
            pos_stu_box_xyxy = box_xyxy_single_sample[n_b_mask[idx]]
            pos_box = Boxes(pos_stu_box_xyxy.detach())
            ious = pairwise_iou(pos_box, pos_box)
            ious.fill_diagonal_(0)
            ious_max, _ = ious.max(dim=1)
            loss_mask = ious_max > 0
            cls_iou_targets = torch.zeros_like(n_student_logits[idx][n_b_mask[idx]])
            teacher_probs = teacher_logits.view(N, -1, num_classes)[idx].sigmoid()
            pseudo_labels = torch.max(teacher_probs, 1)[1]
            pseudo_labels = pseudo_labels[n_b_mask[idx]]

            cls_iou_targets[loss_mask, pseudo_labels[loss_mask]] = ious_max[loss_mask]
            logits.append(n_student_logits[idx][n_b_mask[idx]][loss_mask])
            targets.append(cls_iou_targets[loss_mask])

        logits = torch.cat(logits)
        targets = torch.cat(targets)
        loss_uhl = self.vfl_loss(logits, targets)
        loss_dict.update({'loss_uhl'+name: loss_uhl})
        return loss_dict


def QFLv2(pred_sigmoid,          # (n, 80)
          teacher_sigmoid,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss
