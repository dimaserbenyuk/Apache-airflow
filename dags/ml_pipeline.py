from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime

default_args = {"start_date": datetime(2023, 1, 1), "retries": 1}

with DAG("ml_pipeline", default_args=default_args, schedule_interval="@daily", catchup=False) as dag:

    wait_for_etl = ExternalTaskSensor(
        task_id="wait_for_etl",
        external_dag_id="etl_pipeline",
        external_task_id="load",
        poke_interval=30,
        timeout=600,
        mode="poke"
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/dags/repo/scripts/train.py"
    )

    wait_for_etl >> train
