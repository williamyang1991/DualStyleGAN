import os

import numpy as np
import torch
from torch import optim
from util import save_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from PIL import Image
from tqdm import tqdm
import math

from model.dualstylegan import DualStyleGAN
from model.stylegan import lpips
from model.encoder.psp import pSp
from model.encoder.criteria import id_loss
import model.contextual_loss.functional as FCX
from model.vgg import VGG19

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Refine Extrinsic Style Codes")
        self.parser.add_argument("style", type=str, help="target style type")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--ckpt", type=str, default=None, help="path to the saved dualstylegan model")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path to the saved extrinsic style codes")
        self.parser.add_argument("--instyle_path", type=str, default=None, help="path to the saved intrinsic style codes")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--iter", type=int, default=100, help="total training iterations")
        self.parser.add_argument("--batch", type=int, default=1, help="batch size")
        self.parser.add_argument("--lr_color", type=float, default=0.01, help="learning rate for color parts")
        self.parser.add_argument("--lr_structure", type=float, default=0.005, help="learning rate for structure parts")
        self.parser.add_argument("--model_name", type=str, default='refined_exstyle_code.npy', help="name to save the refined extrinsic style codes")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.ckpt is None:
            self.opt.ckpt = os.path.join(self.opt.model_path, self.opt.style, 'generator.pt') 
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(self.opt.model_path, self.opt.style, 'exstyle_code.npy')    
        if self.opt.instyle_path is None:
            self.opt.instyle_path = os.path.join(self.opt.model_path, self.opt.style, 'instyle_code.npy')          
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)
        
if __name__ == "__main__":
    device = "cuda"

    parser = TrainOptions()
    args = parser.parse()
    print('*'*50)
    
    if not os.path.exists("log/%s/refine_exstyle/"%(args.style)):
        os.makedirs("log/%s/refine_exstyle/"%(args.style))
        
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6).to(device)
    generator.eval()

    ckpt = torch.load(args.ckpt)
    generator.load_state_dict(ckpt["g_ema"])
    noises_single = generator.make_noise()

    percept = lpips.PerceptualLoss(model="net-lin", net="vgg", use_gpu=device.startswith("cuda"))
    vggloss = VGG19().to(device).eval()

    print('Load models successfully!')
    
    datapath = os.path.join(args.data_path, args.style, 'images/train')
    exstyles_dict = np.load(args.exstyle_path, allow_pickle='TRUE').item()
    instyles_dict = np.load(args.instyle_path, allow_pickle='TRUE').item()
    files = list(exstyles_dict.keys())

    dict = {}
    for ii in range(0,len(files),args.batch):
        batchfiles = files[ii:ii+args.batch]
        imgs = []
        exstyles = []
        instyles = []
        for file in batchfiles:
            img = transform(Image.open(os.path.join(datapath, file)).convert("RGB"))
            imgs.append(img)
            exstyles.append(torch.tensor(exstyles_dict[file]))
            instyles.append(torch.tensor(instyles_dict[file]))
        imgs = torch.stack(imgs, 0).to(device)
        exstyles = torch.cat(exstyles, dim=0).to(device)
        instyles = torch.cat(instyles, dim=0).to(device)
        with torch.no_grad():  
            real_feats = vggloss(imgs)

        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True  
            
        # color code
        exstyles_c = exstyles[:,7:].detach().clone()
        exstyles_c.requires_grad = True
        # structure code
        exstyles_s = exstyles[:,0:7].detach().clone()
        exstyles_s.requires_grad = True

        optimizer = optim.Adam([{'params':exstyles_c,'lr':args.lr_color}, 
                                {'params':exstyles_s,'lr':args.lr_structure}, 
                                {'params':noises,'lr':0.1}])

        pbar = tqdm(range(args.iter), smoothing=0.01, dynamic_ncols=False, ncols=100)
        
        for i in pbar:       

            latent = torch.cat((exstyles_s, exstyles_c), dim=1)
            latent = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

            img_gen, _ = generator([instyles], latent, noise=noises, use_res=True, z_plus_latent=True)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            if i == 0:
                img_gen0 = img_gen.detach().clone()

            Lperc = percept(img_gen, imgs).sum()
            Lnoise = noise_regularize(noises)

            fake_feats = vggloss(img_gen)
            LCX = FCX.contextual_loss(fake_feats[2], real_feats[2].detach(), band_width=0.2, loss_type='cosine')

            loss = Lperc + LCX + 1e5 * Lnoise

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            noise_normalize_(noises)

            pbar.set_description(
                (
                    f"[{ii * args.batch:03d}/{len(files):03d}]"
                    f" Lperc: {Lperc.item():.3f}; Lnoise: {Lnoise.item():.3f};"
                    f" LCX: {LCX.item():.3f}"               
                )
            )


        with torch.no_grad():
            latent = torch.cat((exstyles_s, exstyles_c), dim=1)
            for j in range(imgs.shape[0]):
                vis = torchvision.utils.make_grid(torch.cat([imgs[j:j+1], img_gen0[j:j+1], img_gen[j:j+1].detach()], dim=0), 3, 1)
                save_image(torch.clamp(vis.cpu(),-1,1), os.path.join("./log/%s/refine_exstyle/"%(args.style), batchfiles[j]))
                dict[batchfiles[j]] = latent[j:j+1].cpu().numpy()

    np.save(os.path.join(args.model_path, args.style, args.model_name), dict) 
    
    print('Refinement done!')
    