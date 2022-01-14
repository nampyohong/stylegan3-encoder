# stylegan3-encoder

## Introduction
Training e4e version encoder is being delayed because there is no available gpu for me 😢
It will be released no later than January 28th, so please wait!

Encoder implementation for image inversion task of stylegan3 generator ([Alias Free GAN](https://github.com/NVlabs/stylegan3)).  
The neural network architecture and hyper-parameter settings of the base configuration is almost the same as that of [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel), and various settings of improved encoder architecture will be added in the future.  
For fast training, pytorch DistibutedDataParallel is used.  

## Installation

### GPU and NVIDIA driver info
* GeForce RTX 3090 x 8
* NVIDIA driver version: 460.91.03

### Docker build
```
$ sh build_img.sh
$ sh build_container.sh [container-name]
```

### Install package
```
$ docker start [container-name]
$ docker attach [container-name]
$ pip install -v -e .
```

### Pretrained weights
![tree](./imgs/tree_pretrained.png)
- [encoder pretrained, base configuration](https://drive.google.com/file/d/1dog6vajt_1zUwh_hopxSvQ2ZSWALz71T/view?usp=sharing)
- [stylegan3, vgg, inception](https://ngc.nvidia.com/catalog/models/nvidia:research:stylegan3)
- [dlib landmarks detector](https://drive.google.com/file/d/1HKmjg6iXsWr4aFPuU0gBXPGR83wqMzq7/view?usp=sharing)
- [IR-SE50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing)

### Prepare dataset
![tree2](./imgs/tree_data.png)
- [ffhq](https://github.com/NVlabs/ffhq-dataset)
- [ffhqs - 1000 images sampled from FFHQ, for test](https://drive.google.com/drive/folders/1taHKxS66YKJNhdhiGcEdM6nnE5W9zBb1?usp=sharing)
- [celeba-hq](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [celeba-hq-samples](https://drive.google.com/file/d/1IRIQTaTDn3NGuTauyultlQdYHlIntsBD/view?usp=sharing)

### Train
```
python train.py \
    --outdir exp/[exp_name] \
    --encoder [encoder_type] \
    --data data/[dataset_name] \
    --gpus [num_gpus] \
    --batch [total_batch_size] \
    --generator [generator_pkl]
```

### Test
```
python test.py \
    --testdir exp/[train_exp]/[train_exp_subdir] \
    --data data/[dataset_name] \
    --gpus [num_gpus] \
    --batch [total_batch_size]
```

## Experiments
### Base configuration
**Train options**
```
{
  "model_architecture": "base",
  "dataset_dir": "data/ffhq",
  "num_gpus": 8,
  "batch_size": 32,
  "batch_gpu": 4,
  "generator_pkl": "pretrained/stylegan3-t-ffhq-1024x1024.pkl",
  "val_dataset_dir": null,
  "training_steps": 100001,
  "val_steps": 10000,
  "print_steps": 50,
  "tensorboard_steps": 50,
  "image_snapshot_steps": 100,
  "network_snapshot_steps": 5000,
  "learning_rate": 0.001,
  "l2_lambda": 1.0,
  "lpips_lambda": 0.8,
  "id_lambda": 0.1,
  "reg_lambda": 0.0,
  "gan_lambda": 0.0,
  "edit_lambda": 0.0,
  "random_seed": 0,
  "num_workers": 3,
  "resume_pkl": null,
  "run_dir": "exp/base/00000-base-ffhq-gpus8-batch32"
}
```
**Learning Curve**
![l2loss](./imgs/train_l2.png)
![lpipsloss](./imgs/train_lpips.png)
![idloss](./imgs/train_id.png)
![idimprove](./imgs/train_id_improve.png)

**Trainset examples**  
Real image batch X
![real1](./imgs/real_batch_1.png)
![real2](./imgs/real_batch_2.png)
![real3](./imgs/real_batch_3.png)
Encoded image batch G.synthesis(E(X))
![encoded1](./imgs/encoded_batch_1.png)
![encoded2](./imgs/encoded_batch_2.png)
![encoded3](./imgs/encoded_batch_3.png)

**Testset examples(celeba-hq)**  
Target image
![target](./imgs/target.png)
Encoded image
![encoded](./imgs/encoded.png)
Encoded image, transform x=0.2, y=0
![x02y00](./imgs/encoded_transform_x0.2_y0.0.png)
Encoded image, transform x=0.2, y=0.1
![x02y01](./imgs/encoded_transform_x0.2_y0.1.png)
Encoded image, transform x=-0.2, y=0.1
![x-02y01](./imgs/encoded_transform_x-0.2_y0.1.png)
Encoded image, transform x=-0.2, y=-0.1
![x-02y-01](./imgs/encoded_transform_x-0.2_y-0.1.png)

## TODO
 - [x] Refactoring configuration system
 - [x] Implement resume checkpoint
 - [x] Apply [Transformer](https://arxiv.org/abs/1706.03762) encoder instead of convs in GradualStyleBlock(Config-a) -> CNN GradualStyleBlock is better than transformer // discarded
 - [x] Taining delta w from avg latent w_avg (G.mapping.w_avg) -> Training Config-b now, not use regularization loss, same as psp paper
 - [ ] Add L2 delta-regularization loss and GAN loss(latent discriminator), [e4e](https://arxiv.org/abs/2102.02766)
 - [ ] Add metrics for test, FID, [Precision and Recall](https://arxiv.org/abs/1806.00035), [Fidelity and Diversity](https://arxiv.org/abs/2002.09797)
 - [ ] GPU memory optimization
 - [ ] Colab demo
 - [ ] Apply [hyperstyle](https://github.com/yuval-alaluf/hyperstyle)
 - [ ] Train encoder for stylegan3-r generator

## References
1. [stylegan3](https://github.com/NVlabs/stylegan3)
2. [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
3. [e4e](https://github.com/omertov/encoder4editing)
