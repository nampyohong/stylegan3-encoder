python train.py \
    --outdir exp/base \
    --encoder base \
    --data data/ffhq \
    --gpus 8 \
    --batch 32 \
    --generator pretrained/stylegan3-t-ffhq-1024x1024.pkl \
