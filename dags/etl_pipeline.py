from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {"start_date": datetime(2023, 1, 1), "retries": 1}

with DAG("etl_pipeline", default_args=default_args, schedule_interval="@daily", catchup=False) as dag:

    extract = BashOperator(
        task_id="extract",
        bash_command="python /opt/airflow/dags/repo/scripts/extract.py"
    )

    transform = BashOperator(
        task_id="transform",
        bash_command="python /opt/airflow/dags/repo/scripts/transform.py"
    )

    load = BashOperator(
        task_id="load",
        bash_command="python /opt/airflow/dags/repo/scripts/load.py"
    )

    extract >> transform >> load
