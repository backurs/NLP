nvidia-docker run -it \
--ipc=host \
--gpus 'device=0' \
-v '/home/abackurs/GitHub/NLP:/workspace/NLP' \
-v '/home/abackurs/data:/workspace/data' \
--rm \
--name test \
exp_nvidia \
python /workspace/NLP/train.py
