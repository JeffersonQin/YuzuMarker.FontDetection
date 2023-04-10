FROM docker.io/pytorch/pytorch:latest

COPY ./requirements.txt /workspace/requirements.txt

RUN pip install -r /workspace/requirements.txt

COPY ./demo_fonts /workspace/demo_fonts
COPY ./font_demo_cache.bin /workspace/font_demo_cache.bin
