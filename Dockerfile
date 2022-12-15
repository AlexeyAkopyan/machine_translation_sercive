FROM pytorch/pytorch:latest
COPY ./requirements.txt /nmt_service/
RUN pip install -r /nmt_service/requirements.txt
COPY . /nmt_service
#CMD ["python", "/nmt_service/test.py"]