FROM docker.io/jeffersonqin/yuzumarker.fontdetection.huggingfacespace.base:latest

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
USER root
RUN mv /workspace/font_demo_cache.bin $HOME/app/font_demo_cache.bin
RUN mv /workspace/demo_fonts $HOME/app/demo_fonts

USER user
COPY --chown=user detector $HOME/app/detector
COPY --chown=user font_dataset $HOME/app/font_dataset
COPY --chown=user utils $HOME/app/utils
COPY --chown=user configs $HOME/app/configs
COPY --chown=user demo.py $HOME/app/demo.py

CMD ["python", "demo.py", "-d", "-1", "-c", "huggingface://gyrojeff/YuzuMarker.FontDetection/name=4x-epoch=18-step=368676.ckpt", "-m", "resnet50", "-z", "512", "-p", "7860", "-a", "0.0.0.0"]
