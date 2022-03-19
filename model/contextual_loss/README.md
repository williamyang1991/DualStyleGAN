# Contextual Loss
PyTorch implementation of Contextual Loss (CX) and Contextual Bilateral Loss (CoBi).

Fork from [https://github.com/S-aiueo32/contextual_loss_pytorch](https://github.com/S-aiueo32/contextual_loss_pytorch)

## Introduction
There are many image transformation tasks whose spatially aligned data is hard to capture in the wild.
Pixel-to-pixel or global loss functions can NOT be directly applied such unaligned data.
CX is a loss function to defeat the problem.
The key idea of CX is interpreting images as sets of feature points that don't have spatial coordinates.
If you want to know more about CX, please refer the original [paper](https://arxiv.org/abs/1803.02077), [repo](https://github.com/roimehrez/contextualLoss) and examples in [./doc](./doc) directory.

## Requirements
-  Python3.7+
-  `torch` & `torchvision`

## Installation
```
pip install git+https://github.com/S-aiueo32/contextual_loss_pytorch.git
```

## Usage
You can use it like PyTorch APIs.
```python
import torch

import contextual_loss as cl
import contextual_loss.fuctional as F


# input features
img1 = torch.rand(1, 3, 96, 96)
img2 = torch.rand(1, 3, 96, 96)

# contextual loss
criterion = cl.ContextualLoss()
loss = criterion(img1, img2)

# functional call
loss = F.contextual_loss(img1, img2, band_width=0.1, loss_type='cosine')

# comparing with VGG features
# if `use_vgg` is set, VGG model will be created inside of the criterion
criterion = cl.ContextualLoss(use_vgg=True, vgg_layer='relu5_4')
loss = criterion(img1, img2)

```

## Reference
### Papers
1. Mechrez, Roey, Itamar Talmi, and Lihi Zelnik-Manor. "The contextual loss for image transformation with non-aligned data." Proceedings of the European Conference on Computer Vision (ECCV). 2018.  
2. Mechrez, Roey, et al. "Maintaining natural image statistics with the contextual loss." Asian Conference on Computer Vision. Springer, Cham, 2018.
### Implementations
Thanks to the owners of the following awesome implementations.
- Original Repository: https://github.com/roimehrez/contextualLoss
- Simple PyTorch Implemantation: https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da
- CoBi: https://github.com/ceciliavision/zoom-learn-zoom
