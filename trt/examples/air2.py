from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.sensors import ExternalTaskSensor
from airflow.operators.python_operator import PythonOperator

def print_execution_date(ds):
    print(ds)

ds = ''
start_date = datetime(2020, 11, 30)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': start_date
}

with DAG(
    dag_id='DAG_B', 
    schedule_interval='0 0 * * *',
    default_args=default_args) as dag:

    sensor = ExternalTaskSensor(
        task_id='wait_for_task_2',
        external_dag_id='DAG_A',
        external_task_id='Task_2',
        start_date=start_date,
        execution_date_fn=lambda x: x,
        mode='reschedule',
        timeout=3600,
    )

    task_3 = PythonOperator(
        dag=dag,
        task_id='Task_3',
        python_callable=print_execution_date,
        op_kwargs={'ds': ds},
    )

    sensor >> task_3