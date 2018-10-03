FROM tensorflow/tensorflow:latest-gpu-py3

ADD requirements.txt /p/

ADD ./data/processed /p/data/processed
ADD ./src/ /p/src/


RUN pip install --upgrade pip
RUN pip install -r /p/requirements.txt

WORKDIR /p
CMD [ "bash" ]

