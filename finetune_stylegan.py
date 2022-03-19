import argparse
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

        self.parser = argparse.ArgumentParser(description="Train StyleGAN")
        self.parser.add_argument("path", type=str, help="path to the lmdb dataset")
        self.parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
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
        self.parser.add_argument("--save_every", type=int, default=10000, help="interval of saving a checkpoint")
        self.parser.add_argument("--style", type=str, default='cartoon', help="style type")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path to save the model")

    def parse(self):
        self.opt = self.parser.parse_args()     
        args = vars(self.opt)
        if self.opt.local_rank == 0:
            print('Load options')
            for name, value in sorted(args.items()):
                print('%s: %s' % (str(name), str(value)))
        return self.opt

def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

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

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
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

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

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

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"iter: {i:05d}; d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 100 == 0 or (i+1) == args.iter:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    sample = F.interpolate(sample,256)
                    utils.save_image(
                        sample,
                        f"log/%s/finetune-%06d.jpg"%(args.style, i),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if (i+1) % args.save_every == 0 or (i+1) == args.iter:
                torch.save(
                    {
                        #"g": g_module.state_dict(),
                        #"d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        #"g_optim": g_optim.state_dict(),
                        #"d_optim": d_optim.state_dict(),
                        #"args": args,
                        #"ada_aug_p": ada_aug_p,
                    },
                    f"%s/%s/fintune-%06d.pt"%(args.model_path, args.style, i+1),
                )
            
if __name__ == "__main__":
    device = "cuda"

    parser = TrainOptions()
    args = parser.parse()
    if args.local_rank == 0:
        print('*'*98)

    if not os.path.exists("log/%s/"%(args.style)):
        os.makedirs("log/%s/"%(args.style))
    if not os.path.exists("%s/%s/"%(args.model_path, args.style)):
        os.makedirs("%s/%s/"%(args.model_path, args.style))    
    
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    #if args.arch == 'stylegan2':
        #from model.stylegan.model import Generator, Discriminator

    #elif args.arch == 'swagan':
        #from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
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

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        
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
        wandb.init(project="stylegan 2")

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)
