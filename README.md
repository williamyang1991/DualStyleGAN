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

## Dataset Preparation

Cartoon, Caricature and Anime datasets can be downloaded from their official pages.
We also provide the script to build new datasets.

| Dataset | Description |
| :--- | :--- |
| [Cartoon](https://mega.nz/file/HslSXS4a#7UBanJTjJqUl_2Z-JmAsreQYiJUKC-8UlZDR0rUsarw) | 317 cartoon face images from [Toonify](https://github.com/justinpinkney/toonify). |
| Caricature | 199 images from [WebCaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm). Please refer to [dataset preparation](./data_preparation/readme.md#caricature-dataset) for more details. |
| Anime | 174 images from [Danbooru Portraits](https://www.gwern.net/Crops#danbooru2019-portraits). Please refer to  [dataset preparation](./data_preparation/readme.md#anime-dataset) for more details. |
| Other styles | Please refer to  [dataset preparation](./data_preparation/readme.md#build-your-own-dataset) for the way of building new datasets. |

## Inference for Style Transfer and Artistic Portrait Generation

Pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GZQ6Gs5AzJq9lUL-ldIQexi0JYPKNy8b?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1sOpPszHfHSgFsgw47S6aAA ) (access code: cvpr):

| Model | Description |
| :--- | :--- |
| [encoder](https://drive.google.com/file/d/1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej/view?usp=sharing) | Pixel2style2pixel encoder that embeds FFHQ images into StyleGAN2 Z+ latent code |
| [cartoon](https://drive.google.com/drive/folders/1xPo8PcbMXzcUyvwe5liJrfbA5yx4OF1j?usp=sharing) | DualStyleGAN and sampling models trained on Cartoon dataset, 317 (refined) extrinsic style codes |
| [caricature](https://drive.google.com/drive/folders/1BwLXWkSyWDApblBPvaHKsRCTqnhiHxUZ?usp=sharing) | DualStyleGAN and sampling models trained on Caricature dataset, 199 (refined) extrinsic style codes |
| [anime](https://drive.google.com/drive/folders/1YvFj33Bfum4YuBeqNNCYLfiBrD4tpzg7?usp=sharing) | DualStyleGAN and sampling models trained on Anime dataset, 174 (refined) extrinsic style codes |
| [arcane](https://drive.google.com/drive/folders/1-MYwaEQthhAJ_ScWVb0LOQiVkKeSzpBm?usp=sharing) | DualStyleGAN and sampling models trained on Arcane dataset, 100 extrinsic style codes |
| [comic](https://drive.google.com/drive/folders/1qC2onFGs2R-XCXRQTP_yyNbY1fT0BdZG?usp=sharing) | DualStyleGAN and sampling models trained on Comic dataset, 101 extrinsic style codes |
| [pixar](https://drive.google.com/drive/folders/1ve4P8Yb4EZ9g_sRy_RCw3N74p46tNpeW?usp=sharing) | DualStyleGAN and sampling models trained on Pixar dataset, 122 extrinsic style codes |
| [slamdunk](https://drive.google.com/drive/folders/1X345yn_YbMEHBcj7K91O-oQZ2YjVpAcI?usp=sharing) | DualStyleGAN and sampling models trained on Slamdunk dataset, 120 extrinsic style codes |

The saved checkpoints is under the following folder structure:
```
Checkpoint
|--encoder.pt                     % Pixel2style2pixel model
|--cartoon
    |--generator.pt               % DualStyleGAN model
    |--sampler.pt                 % The extrinsic style code sampling model
    |--exstyle_code.npy           % extrinsic style codes of Cartoon dataset
    |--refined_exstyle_code.npy   % refined extrinsic style codes of Cartoon dataset
|--caricature
    % the same files as in Cartoon
...
```

### Artistic Portrait Generation
Generate random Cartoon face images: (Results are saved in the `./output/` folder)
```python
python generate.py 
```

Specify the style type with `--style` and the filename of the saved images with `--name`
```python
python generate.py --style arcane --name arcane_generate
```

Specify the weight to adjust the degree of style with `--weight.

Keep the intrinsic style code, extrinsic color code or extrinsic strcture code fixed using `--fix_content`, `--fix_color` and `--fix_structure`, respectively.
```python
python generate.py --style caricature --name caricature_generate --weight 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 --fix_content
```

Find more options via `python generate.py -h`


## Training Code

- We are cleaning our code. Coming soon. 

| [stylegan](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) | StyleGAN model trained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch), provided from [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel#pretrained-models). |

## Results

#### Exemplar-based cartoon style trasnfer

https://user-images.githubusercontent.com/18130694/158047991-77c31137-c077-415e-bae2-865ed3ec021f.mp4

#### Exemplar-based caricature style trasnfer

https://user-images.githubusercontent.com/18130694/158048107-7b0aa439-5e3a-45a9-be0e-91ded50e9136.mp4

#### Exemplar-based anime style trasnfer

https://user-images.githubusercontent.com/18130694/158048114-237b8b81-eff3-4033-89f4-6e8a7bbf67f7.mp4

#### Other styles

<img src="https://user-images.githubusercontent.com/18130694/158049559-5450568f-170d-4847-88e1-d9bd12901966.jpg" width="48%"><img src="https://user-images.githubusercontent.com/18130694/158049562-e9971b49-ebd9-4300-bd08-34fc2473729f.jpg" width="48%">
<img src="https://user-images.githubusercontent.com/18130694/158049563-72718807-4bef-472d-8875-71eee22ae934.jpg" width="48%"><img src="https://user-images.githubusercontent.com/18130694/158049565-0322a005-c402-40bc-8bef-9b22a8ca3fd4.jpg" width="48%">

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

## Acknowledgments

The code is developed based on [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel).
