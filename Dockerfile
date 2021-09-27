# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY car_racing car_racing

