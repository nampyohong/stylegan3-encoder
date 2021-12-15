# stylegan3-encoder

## References
1. [stylegan3](https://github.com/NVlabs/stylegan3)
2. [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

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

### Train
```
python train.py \
    --outdir exp/[exp_name] \
    --cfg [cfg_name] \
    --data data/[dataset_name] \
    --gpus [num_gpus] \
    --batch [total_batch_size]
```

## Experiments
### Base configuration
**Train options**
```
{
  "dataset": "ffhq",
  "generator_pkl": "pretrained/stylegan3-t-ffhq-1024x1024.pkl",
  "encoder_pkl": null
  "num_gpus": 8,
  "learning_rate": 0.001,
  "lambda1": 1.0,
  "lambda2": 0.8,
  "lambda3": 0.1,
  "batch_size": 32,
  "batch_gpu": 4,
  "random_seed": 0,
  "num_workers": 3,
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

**Encoder checkpoint** will be available in few days

### TODO
 - [ ] Implement demo script
 - [ ] Refactoring configuration system
 - [ ] Implement resume checkpoint
 - [ ] Implement scripts for Validation, test dataset
 - [ ] Fix legacy script to handle encoder snapshot pkl
 - [ ] Train encoder for stylegan3-r generator
 - [ ] Model architecture over parametrization using [Transformer](https://arxiv.org/abs/1706.03762)
 - [ ] Apply L2 delta-regularization loss and GAN loss(latent discriminator), [e4e](https://arxiv.org/abs/2102.02766)
 - [ ] Apply [hyperstyle](https://github.com/yuval-alaluf/hyperstyle)

## References
1. [stylegan3](https://github.com/NVlabs/stylegan3)
2. [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)
