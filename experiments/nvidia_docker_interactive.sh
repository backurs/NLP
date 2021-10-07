nvidia-docker run -it \
--ipc=host \
--gpus 'device=0' \
-v '/home/abackurs/GitHub/NLP:/workspace/NLP' \
-v '/home/abackurs/data:/workspace/data' \
--rm \
-p 6006:6006 \
--name profiling_interactive \
exp_nvidia \
bash
