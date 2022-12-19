FROM pytorch/pytorch:latest
COPY ./requirements.txt /workspace/nmt_service/
RUN pip install -r /workspace/nmt_service/requirements.txt
#COPY . /workspace/nmt_service
RUN apt-get -y update
RUN apt-get install -y ca-certificates curl gnupg lsb-release git
RUN mkdir -p /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
RUN apt-get -y update
RUN apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
#RUN export MLFLOW_TRACKING_URI="./nmt_service/mlruns"
#RUN echo $(ls  ./)
#CMD ["docker container ls"]