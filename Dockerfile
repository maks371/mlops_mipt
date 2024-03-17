FROM python:3.10-bookworm

COPY train.py infer.py requirements.txt ./
RUN pip install --upgrade pip && pip install --default-timeout=1000 -r requirements.txt
