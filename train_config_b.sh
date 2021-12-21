# train delta_w instead of w, not apply reg loss
python train.py \
    --outdir exp/config-b \
    --encoder base \
    --data data/ffhq \
    --gpus 8 \
    --batch 32 \
    --generator pretrained/stylegan3-t-ffhq-1024x1024.pkl \
    --w_avg \
