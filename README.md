# DualStyleGAN - Official PyTorch Implementation

<img src="./doc_images/overview.jpg" width="96%" height="96%">

This repository provides the official PyTorch implementation for the following paper:

**Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**<br>
[Shuai Yang](https://williamyang1991.github.io/), [Liming Jiang](https://liming-jiang.com/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
In CVPR 2022.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/dualstylegan/) | [**Paper**]() (coming soon)
> **Abstract:** *Recent studies on StyleGAN show high performance on artistic portrait generation by transfer learning with limited data. In this paper, we explore more challenging exemplar-based high-resolution portrait style transfer by introducing a novel <b>DualStyleGAN</b> with flexible control of dual styles of the original face domain and the extended artistic portrait domain. Different from StyleGAN, DualStyleGAN provides a natural way of style transfer by characterizing the content and style of a portrait with an <b>intrinsic style path</b> and a new <b>extrinsic style path</b>, respectively. The delicately designed extrinsic style path enables our model to modulate both the color and complex structural styles hierarchically to precisely pastiche the style example. Furthermore, a novel progressive fine-tuning scheme is introduced to smoothly transform the generative space of the model to the target domain, even with the above modifications on the network architecture. Experiments demonstrate the superiority of DualStyleGAN over state-of-the-art methods in high-quality portrait style transfer and flexible style control.*

## Updates

- [03/2022] This website is created.

## Code

- We are cleaning our code. Coming soon. 

## Results

#### Exemplar-based cartoon style trasnfer

https://user-images.githubusercontent.com/18130694/158047991-77c31137-c077-415e-bae2-865ed3ec021f.mp4

#### Exemplar-based caricature style trasnfer

https://user-images.githubusercontent.com/18130694/158048107-7b0aa439-5e3a-45a9-be0e-91ded50e9136.mp4

#### Exemplar-based anime style trasnfer

https://user-images.githubusercontent.com/18130694/158048114-237b8b81-eff3-4033-89f4-6e8a7bbf67f7.mp4


## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{yang2022Pastiche,
  title={Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer},
  author={Yang, Shuai and Jiang, Liming and Liu, Ziwei and Loy, Chen Change},
  booktitle={CVPR},
  year={2022}
}
```
