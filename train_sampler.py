import numpy as np
import torch 
import argparse
from model.sampler.icp import ICPTrainer
import os

class TrainOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Train Sampler")
        self.parser.add_argument("style", type=str, help="target style type")
        self.parser.add_argument("--iter", type=int, default=500000, help="iterations")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path to the extrinsic style codes")
        self.parser.add_argument("--model_name", type=str, default='sampler.pt', help="name of the saved model")

    def parse(self):
        self.opt = self.parser.parse_args()     
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

if __name__ == "__main__":
    device = "cuda"

    parser = TrainOptions()
    args = parser.parse()
    print('*'*50)
    
    if args.exstyle_path is None:
        if os.path.exists(os.path.join(args.model_path, args.style, 'refined_exstyle_code.npy')):
            exstyles_dict = np.load(os.path.join(args.model_path, args.style, 'refined_exstyle_code.npy'),allow_pickle='TRUE').item()
        else:
            exstyles_dict = np.load(os.path.join(args.model_path, args.style, 'exstyle_code.npy'),allow_pickle='TRUE').item()
    else:
        exstyles_dict = np.load(args.exstyle_path,allow_pickle='TRUE').item()
        
    exstyles = []
    for k in exstyles_dict.keys():
        exstyles += [torch.tensor(exstyles_dict[k])]
    exstyles = torch.cat(exstyles, dim=0).reshape(-1,18*512)
    
    # augment extrinsic style codes to about 1000 by duplicate and small jittering
    W = torch.normal(exstyles.repeat(1000//exstyles.shape[0], 1), 0.05)
    # color code
    WC = W[:,512*7:].detach().cpu().numpy()
    # style code
    WS = W[:,0:512*7].detach().cpu().numpy()
    
    print('Load extrinsic tyle codes successfully!')
    # train color code sampler 
    icptc = ICPTrainer(WC, 128)
    icptc.icp.netT = icptc.icp.netT.to(device)
    icptc.train_icp(int(500000/WC.shape[0]))
    # train structure code sampler 
    icpts = ICPTrainer(WS, 128)
    icpts.icp.netT = icpts.icp.netT.to(device)
    icpts.train_icp(int(500000/WS.shape[0]))
    
    torch.save(
        {
            "color": icptc.icp.netT.state_dict(),
            "structure": icpts.icp.netT.state_dict(),
        },
        f"%s/%s/%s"%(args.model_path, args.style, args.model_name),
    )
    
    print('Training done!')
    
