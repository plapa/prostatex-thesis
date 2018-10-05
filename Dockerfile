FROM tensorflow/tensorflow:latest-gpu-py3

ADD requirements.txt /p/

RUN pip install --upgrade pip
RUN pip install -r /p/requirements.txt



RUN mkdir working

ENV PYTHONPATH /working:$PYTHONPATH

WORKDIR /working

CMD [ "bash" ]

