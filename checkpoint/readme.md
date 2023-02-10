## saved models

folder structure:
```
Checkpoint
|--encoder.pt                             % Pixel2style2pixel model
|--encoder_wplus.pt                       % Pixel2style2pixel model (optional)
|--shape_predictor_68_face_landmarks.dat  % Face alignment model
|--stylegan2-ffhq-config-f.pt             % (only for training) StyleGAN model
|--model_ir_se50.pth                      % (only for training) Pretrained IR-SE50 model for ID loss 
|--generator-pretrain.pt                  % (only for training) Pretrained DualStyleGAN on FFHQ
|--cartoon
    |--generator.pt                       % DualStyleGAN model
    |--sampler.pt                         % The extrinsic style code sampling model
    |--exstyle_code.npy                   % Extrinsic style codes of Cartoon dataset
    |--refined_exstyle_code.npy           % Refined extrinsic style codes of Cartoon dataset
    |--instyle_code.npy                   % (only for training) Intrinsic style codes of Cartoon dataset
    |--finetune-000600.pt                 % (only for training) StyleGAN fine-tuned on Cartoon dataset
|--caricature
    % the same files as in Cartoon
...
```
