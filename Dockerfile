FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN cd /workspace

COPY . /workspace
RUN pip install -r /workspace/pct_requirements.txt