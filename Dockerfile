FROM python:3.10.6-buster

COPY api api
COPY requirements.txt requirements.txt
COPY model.pkl model.pkl

RUN pip install -r requirements.txt

CMD uvicorn api.main:app --host 0.0.0.0
