FROM cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py310:0.0.36

RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --no-cache-dir --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /requirements.txt

RUN pip install  --no-cache-dir -r /requirements.txt

