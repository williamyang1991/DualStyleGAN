Code from [rosinality-stylegan2-pytorch-cp](https://github.com/senior-sigan/rosinality-stylegan2-pytorch-cpu)

Scripts to convert rosinality/stylegan2-pytorch to the CPU compatible format

If you would like to use CPU for testing, please make the following changes:

1. Change `model.stylegan.op`  to `model.stylegan.op_cpu`
https://github.com/williamyang1991/DualStyleGAN/blob/6ab383bc47ed36e9f57f0be6ab22f1ebe9c21dec/util.py#L10

2. Change `model.stylegan.op`  to `model.stylegan.op_cpu`
https://github.com/williamyang1991/DualStyleGAN/blob/6ab383bc47ed36e9f57f0be6ab22f1ebe9c21dec/model/stylegan/model.py#L11

If using GPUs, changing the above lines back to the original ones.
 
