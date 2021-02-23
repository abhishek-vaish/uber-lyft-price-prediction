FROM python:3.7

WORKDIR /app

RUN mkdir src

COPY ./model/requirements.txt  ./src

RUN cd src \
    && pip install -r requirements.txt

RUN mkdir data

COPY ./data ./data

COPY ./model ./src

WORKDIR /app/src

CMD ["python3", "app.py"]