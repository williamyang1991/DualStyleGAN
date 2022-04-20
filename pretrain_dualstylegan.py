import argparse
from argparse import Namespace
import math
import random
import os

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from util import data_sampler, requires_grad, accumulate, sample_data, d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize, make_noise, mixing_noise, set_grad_none

from model.dualstylegan import DualStyleGAN
from model.stylegan.model import Discriminator
from model.encoder.psp import pSp
from model.vgg import VGG19

try:
    import wandb

except ImportError:
    wandb = None


from model.stylegan.dataset import MultiResolutionDataset
from model.stylegan.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from model.stylegan.non_leaking import augment, AdaptiveAugment
from model.stylegan.model import Generator, Discriminator

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Pretrain DualStyleGAN")
        self.parser.add_argument("path", type=str, help="path to the lmdb dataset")
        self.parser.add_argument("--iter", type=int, default=3000, help="total training iterations")
        self.parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
        self.parser.add_argument("--n_sample", type=int, default=9, help="number of the samples generated during training")
        self.parser.add_argument("--size", type=int, default=1024, help="image sizes for the model")
        self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        self.parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        self.parser.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        self.parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
        self.parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
        self.parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        self.parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
        self.parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
        self.parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1")
        self.parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        self.parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        self.parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
        self.parser.add_argument("--augment_p", type=float, default=0, help="probability of applying augmentation. 0 = use adaptive augmentation")
        self.parser.add_argument("--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation")
        self.parser.add_argument("--ada_length", type=int, default=500 * 1000, help="target duraing to reach augmentation probability for adaptive augmentation")
        self.parser.add_argument("--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation")
        self.parser.add_argument("--save_every", type=int, default=3000, help="interval of saving a checkpoint")
        self.parser.add_argument("--subspace_freq", type=int, default=4, help="how often to use Gaussian style code")
        self.parser.add_argument("--model_name", type=str, default='generator-pretrain', help="saved model name")
        self.parser.add_argument("--encoder_path", type=str, default=None, help="path to the encoder model")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")


    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.encoder_path is None:
            self.opt.encoder_path = os.path.join(self.opt.model_path, 'encoder.pt')
        if self.opt.ckpt is None:
            self.opt.ckpt = os.path.join(self.opt.model_path, 'stylegan2-ffhq-config-f.pt')        
        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s: %s' % (str(name), str(value)))
        return self.opt


def pretrain(args, loader, generator, discriminator, g_optim, d_optim, g_ema, encoder, vggloss, device, inject_index=5, savemodel=True):
    loader = sample_data(loader)
    vgg_weights = [0.5, 0.5, 0.5, 0.0, 0.0]
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, ncols=140, dynamic_ncols=False, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_zs = mixing_noise(args.n_sample, args.latent, 1.0, device)
    with torch.no_grad():
        source_img, _ = generator([sample_zs[0]], None, input_is_latent=False, z_plus_latent=False, use_res=False)
        source_img = source_img.detach()
        target_img, _ = generator(sample_zs, None, input_is_latent=False, z_plus_latent=False, inject_index=inject_index, use_res=False)
        target_img = target_img.detach()
        style_img, _ = generator([sample_zs[1]], None, input_is_latent=False, z_plus_latent=False, use_res=False)        
        _, sample_style = encoder(F.adaptive_avg_pool2d(style_img, 256), randomize_noise=False, 
                                      return_latents=True, z_plus_latent=True, return_z_plus_latent=False)
        sample_style = sample_style.detach()
        if get_rank() == 0:
            utils.save_image(
                F.adaptive_avg_pool2d(source_img, 256),
                f"log/%s-instyle.jpg"%(args.model_name),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1))          
            utils.save_image(
                F.adaptive_avg_pool2d(target_img, 256),
                f"log/%s-target.jpg"%(args.model_name),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1))
            utils.save_image(
                F.adaptive_avg_pool2d(style_img, 256),
                f"log/%s-exstyle.jpg"%(args.model_name),
                nrow=int(args.n_sample ** 0.5),
                normalize=True,
                range=(-1, 1))        
        
    for idx in pbar:
        i = idx + args.start_iter
        
        which = i % args.subspace_freq 

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)
        
        # real_zs contains z1 and z2
        real_zs =  mixing_noise(args.batch, args.latent, 1.0, device)
        with torch.no_grad():
            # g(z^+_l) with l=inject_index
            target_img, _ = generator(real_zs, None, input_is_latent=False, z_plus_latent=False, inject_index=inject_index, use_res=False)
            target_img = target_img.detach()
            # g(z2)
            style_img, _ = generator([real_zs[1]], None, input_is_latent=False, z_plus_latent=False, use_res=False)
            style_img = style_img.detach()
            # E(g(z2))
            _, pspstyle = encoder(F.adaptive_avg_pool2d(style_img, 256), randomize_noise=False, 
                                      return_latents=True, z_plus_latent=True, return_z_plus_latent=False)
            pspstyle = pspstyle.detach()

        
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        if which > 0:
            # set z~_2 = z2
            noise = [real_zs[0]]
            externalstyle = g_module.get_latent(real_zs[1]).detach()
            z_plus_latent = False
        else:
            # set z~_2 = E(g(z2))
            noise = [real_zs[0].unsqueeze(1).repeat(1, g_module.n_latent, 1)]
            externalstyle = pspstyle
            z_plus_latent = True
            
        fake_img, _ = generator(noise, externalstyle, use_res=True, z_plus_latent=z_plus_latent)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred) * 0.1

        loss_dict["d"] = d_loss # Ladv
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        if which > 0:
            # set z~_2 = z2
            noise = [real_zs[0]]
            externalstyle = g_module.get_latent(real_zs[1]).detach()
            z_plus_latent = False
        else:
            # set z~_2 = E(g(z2))
            noise = [real_zs[0].unsqueeze(1).repeat(1, g_module.n_latent, 1)]
            externalstyle = pspstyle
            z_plus_latent = True
            
        fake_img, _ = generator(noise, externalstyle, use_res=True, z_plus_latent=z_plus_latent)
        
        real_feats = vggloss(F.adaptive_avg_pool2d(target_img, 256).detach())
        fake_feats = vggloss(F.adaptive_avg_pool2d(fake_img, 256))
        gr_loss = torch.tensor(0.0).to(device)
        for ii, weight in enumerate(vgg_weights):
            if weight > 0:
                gr_loss += F.l1_loss(fake_feats[ii], real_feats[ii].detach()) * weight
        
        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred) * 0.1

        loss_dict["g"] = g_loss # Ladv
        loss_dict["gr"] = gr_loss # L_perc
        
        g_loss += gr_loss
        
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()
 
        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            externalstyle = torch.randn(path_batch_size, 512, device=device)
            externalstyle = g_module.get_latent(externalstyle).detach()        
            fake_img, latents = generator(noise, externalstyle, return_latents=True, use_res=True, 
                                          z_plus_latent=False)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        
        accumulate(g_ema.res, g_module.res, accum)
        
        loss_reduced = reduce_loss_dict(loss_dict)
        
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        gr_loss_val = loss_reduced["gr"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"iter: {i:d}; d: {d_loss_val:.3f}; g: {g_loss_val:.3f}; gr: {gr_loss_val:.3f}; r1: {r1_val:.3f}; "
                    f"path: {path_loss_val:.3f}; mean path: {mean_path_length_avg:.3f}; "
                    f"augment: {ada_aug_p:.1f}"
                )
            )

            if i % 300 == 0 or (i+1) == args.iter:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_zs[0].unsqueeze(1).repeat(1, g_module.n_latent, 1)], 
                                      sample_style, use_res=True, z_plus_latent=True)   
                    sample = F.interpolate(sample,256)
                    utils.save_image(
                        sample,
                        f"log/%s-%06d.jpg"%(args.model_name, i),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if savemodel and ((i+1) % args.save_every == 0 or (i+1) == args.iter):
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"%s/%s-%06d.pt"%(args.model_path, args.model_name, i+1),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = TrainOptions()
    args = parser.parse()
    if args.local_rank == 0:
        print('*'*98)
        
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = DualStyleGAN(args.size, args.latent, args.n_mlp, 
                             channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = DualStyleGAN(args.size, args.latent, args.n_mlp, 
                         channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(generator.res.parameters()) + list(generator.style.parameters()),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.generator.load_state_dict(ckpt["g_ema"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.generator.load_state_dict(ckpt["g_ema"])
        
        if "g_optim" in ckpt:
            g_optim.load_state_dict(ckpt["g_optim"])
        if "d_optim" in ckpt:
            d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="pretrain dualstylegan")
    
    ckpt = torch.load(args.encoder_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = args.encoder_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = True
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    encoder = pSp(opts).to(device).eval()
    encoder.latent_avg = encoder.latent_avg.to(device)
    vggloss = VGG19().to(device).eval()

    print('Models successfully loaded!')
    
    full_iter = args.iter
    args.iter = full_iter // 10
    pretrain(args, loader, generator, discriminator, g_optim, d_optim, g_ema, encoder, vggloss, device, inject_index=7, savemodel=False)
    args.iter = full_iter // 10
    pretrain(args, loader, generator, discriminator, g_optim, d_optim, g_ema, encoder, vggloss, device, inject_index=6, savemodel=False)
    args.iter = full_iter
    pretrain(args, loader, generator, discriminator, g_optim, d_optim, g_ema, encoder, vggloss, device, inject_index=5)
