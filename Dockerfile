FROM docker.io/jeffersonqin/yuzumarker.fontdetection.huggingfacespace.base:latest

WORKDIR /workspace

COPY . .

CMD ["python", "demo.py", "-d", "-1", "-c", "huggingface://gyrojeff/YuzuMarker.FontDetection/commit=bc0f7fc-epoch=26-step=261954.ckpt", "-m", "resnet50", "-z", "512", "-p", "7860"]
