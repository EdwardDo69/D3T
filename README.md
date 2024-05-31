
<div align="center">

# D3T: Distinctive Dual-Domain Teacher Zigzagging Across RGB-Thermal Gap for Domain-Adaptive Object Detection
[![PWC](https://cvpr.thecvf.com/static/core/img/cvpr-navbar-logo.svg)](https://arxiv.org/abs/2403.09359)

[Dinh Phat Do](https://github.com/EdwardDo69), [Taehoon Kim](https://scholar.google.com/citations?user=RrKoTX4AAAAJ&hl=en), [Jaemin Na](https://github.com/NaJaeMin92), Jiwon Kim, Keonho Lee, Kyunghwan Cho, [Wonjun Hwang](https://scholar.google.co.uk/citations?user=-I8AfBAAAAAJ&hl=en)<br>

### [ArXiv](https://arxiv.org/abs/2402.12974) | [Youtube](https://youtu.be/NPbjykByfRA?si=MbQUXWQ28rrT86AT)

> **Abstract:** *Domain adaptation for object detection typically entails transferring knowledge from one visible domain to another visible domain. However, there are limited studies on adapting from the visible to the thermal domain, because the domain gap between the visible and thermal domains is much larger than expected, and traditional domain adaptation can not successfully facilitate learning in this situation. To overcome this challenge, we propose a Distinctive Dual-Domain Teacher (D3T) framework that employs distinct training paradigms for each domain. Specifically, we segregate the source and target training sets for building dual-teachers and successively deploy exponential moving average to the student model to individual teachers of each domain. The framework further incorporates a zigzag learning method between dual teachers, facilitating a gradual transition from the visible to thermal domains during training. We validate the superiority of our method through newly designed experimental protocols with well-known thermal datasets, i.e., FLIR and KAIST.*

</br>

<p align="center">
<img src="/image/Figure2.jpg">
</p>

**Overview of D3T:** Our D3T model consists of two stages. **Burn-in Stage:** We initiate the training of the object detector using
labeled data from the RGB domain. **Zigzag Learning Stage:** Comprises two distinct and interleaved training components for the Thermal
domain and the RGB domain, respectively. During each step of training, the student model utilizes images from a single domain for
training but leverages knowledge from two teachers for enhanced learning effectiveness. In each step, only one teacher model is updated
corresponding to the trained domain.

</br>






<div align="center">
  
![Demo](/image/Demo.gif)

</div>



<br>
<br>
<br>

# We will clean and release the code soon.

For inquiries, please contact: phatai@ajou.ac.kr
