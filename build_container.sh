nvidia-docker run -ti \
-v $(pwd):/workspace/stylegan3-encoder/ \
--ipc=host \
--net=host \
--name=$1 \
stylegan3 \
/bin/bash
