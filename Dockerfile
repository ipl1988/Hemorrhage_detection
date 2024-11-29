#FROM python:3.10.6-slim

#COPY api api
#COPY requirements.txt requirements.txt
#COPY model.pkl model.pkl

#RUN python -m pip install --upgrade pip
#RUN pip install -r requirements.txt

#CMD uvicorn api.main:app --host 0.0.0.0




FROM python:3.10

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
