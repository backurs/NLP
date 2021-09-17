docker run -it \
--gpus all \
-v '/home/abackurs/GitHub/NLP:/workspace/NLP' \
-v '/home/abackurs/data:/workspace/data' \
--rm \
experiments \
python /workspace/NLP/train.py

