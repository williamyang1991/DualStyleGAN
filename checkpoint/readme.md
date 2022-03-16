## saved models

folder structure:
```
Checkpoint
|--stylegan2-ffhq-config-f.pt             % StyleGAN model
|--encoder.pt                             % Pixel2style2pixel model
|--shape_predictor_68_face_landmarks.dat  % Face alignment model
|--cartoon
    |--generator.pt                       % DualStyleGAN model
    |--sampler.pt                         % The extrinsic style code sampling model
    |--exstyle_code.npy                   % extrinsic style codes of Cartoon dataset
    |--refined_exstyle_code.npy           % refined extrinsic style codes of Cartoon dataset
|--caricature
    % the same files as in Cartoon
...
```
