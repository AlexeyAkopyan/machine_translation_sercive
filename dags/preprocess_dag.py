from datetime import datetime

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.decorators import task
from airflow.utils.edgemodifier import Label

# Docker library from PIP
import docker

# Simple DAG
with DAG(
    "preprocess", 
    schedule_interval="@daily", 
    start_date=datetime(2022, 1, 1), 
    catchup=False, 
    tags=['nmt']
) as dag:


    @task(task_id='run_translation')
    def run_gpu_translation(**kwargs):

        # get the docker params from the environment
        client = docker.from_env()
          
            
        # run the container
        command = (
            "python preprocess.py --train-src data/raw/train.ru_en.ru "
            "--train-trg data/raw/train.ru_en.en "
            "--val-src data/raw/valid.ru_en.ru "
            "--val-trg data/raw/valid.ru_en.en "
            "--test-src data/raw/test.ru_en.ru "
            "--test-trg data/raw/test.ru_en.en "
            "--src-lang ru "
            "--trg-lang en "
            "--src-vocab-size 4000 "
            "--trg-vocab-size 4000 "
            "--save-data-dir ./data/processed/ "
            "--max-seq-len 128 "
            "--src-tokenizer-path ./weights_models/ru_tokenizer.model "
            "--trg-tokenizer-path ./weights_models/en_tokenizer.model")






        response = client.containers.run(

             # The container you wish to call
             'nmt_service:latest',

             # The command to run inside the container
             command,

             # Passing the GPU access
             device_requests=[
                 docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
             ], 
             
             # Give the proper system volume mount point
             volumes=[
                 '/home/iref/Repos/machine_translation_sercive/data:/nmt_service/data'
             ]
        )

        return str(response)

    run_translation = run_gpu_translation()


    # Dummy functions
    start = DummyOperator(task_id='start')
    end   = DummyOperator(task_id='end')


    # Create a simple workflow
    start >> run_translation >> end