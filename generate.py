import os
import numpy as np
import torch
from util import save_image
import argparse
from model.dualstylegan import DualStyleGAN
from model.sampler.icp import ICPTrainer

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Random Artistic Portrait Generation")
        self.parser.add_argument("--batch", type=int, default=8, help="number of generated images")
        self.parser.add_argument("--style", type=str, default='cartoon', help="target style type")
        self.parser.add_argument("--truncation", type=float, default=0.5, help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="weight of the extrinsic style")
        self.parser.add_argument("--name", type=str, default='cartoon_generate', help="filename to save the generated images")
        self.parser.add_argument("--fix_content", action="store_true", help="using a fixed intrinsic style code (content)")
        self.parser.add_argument("--fix_color", action="store_true", help="using a fixed extrinsic color code (style)")
        self.parser.add_argument("--fix_structure", action="store_true", help="using a fixed extrinsic structure code (style")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="name of the saved dualstylegan")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--sampler_name", type=str, default='sampler.pt', help="name of the saved sampling network")

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


if __name__ == "__main__":
    device = "cuda"

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    generator.eval()
    icptc = ICPTrainer(np.empty([0,512*11]), 128)
    icpts = ICPTrainer(np.empty([0,512*7]), 128)

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.sampler_name), map_location=lambda storage, loc: storage)
    icptc.icp.netT.load_state_dict(ckpt['color'])
    icpts.icp.netT.load_state_dict(ckpt['structure'])
    icptc.icp.netT = icptc.icp.netT.to(device)
    icpts.icp.netT = icpts.icp.netT.to(device)

    print('Load models successfully!')
    
    with torch.no_grad():
        instyle = torch.randn(args.batch, 512).to(device)
        # sample structure codes
        res_in = icpts.icp.netT(torch.randn(args.batch,128).to(device)).reshape(-1,7,512)
        # sample color codes
        ada_in = icptc.icp.netT(torch.randn(args.batch,128).to(device)).reshape(-1,11,512)

        if args.fix_content:
            instyle = instyle[0:1].repeat(args.batch, 1)
        if args.fix_color:
            ada_in = ada_in[0:1].repeat(args.batch, 1, 1)
        if args.fix_structure:
            res_in = res_in[0:1].repeat(args.batch, 1, 1)
        # concatenate two codes to form the complete extrinsic style code
        latent = torch.cat((res_in, ada_in), dim=1)
        # map into W+ space
        exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)

        img_gen, _ = generator([instyle], exstyle, input_is_latent=False, truncation=args.truncation, truncation_latent=0, 
                               use_res=True, interp_weights=args.weight)

        img_gen = torch.clamp(img_gen.detach(), -1, 1)

    print('Generate images successfully!')
    
    for i in range(args.batch):
        save_image(img_gen[i].cpu(), os.path.join(args.output_path, args.name+'_%02d.jpg'%(i)))
    
    print('Save images successfully!')
