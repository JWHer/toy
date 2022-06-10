import asyncio
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.api.client.json_client import Client
from datetime import datetime
import sys, os

sys.path.append(os.path.join(sys.path[0], '/home/jwher/build/tx2'))
from core.app.service.base.restful_service import Restful

def create_dag(dag_id,
               schedule,
               dag_number,
               default_args):

    def hello_world_py(*args):
        print('Hello World')
        print('This is DAG: {}'.format(str(dag_number)))

    dag = DAG(dag_id,
              schedule_interval=schedule,
              default_args=default_args)

    with dag:
        t1 = PythonOperator(
            task_id='hello_world',
            python_callable=hello_world_py)

    return dag

async def main():
    c = Restful('localhost', 18080, api_ver='/api/v1', id='airflow', pw='airflow', default_header={"Content-Type": "application/json"})
    dag_id = "hello"
    data = await c.request(f'/dags/{dag_id}/dagRuns',method='post',payload={"dag_run_id": "python test"})
    print(data)

if __name__ == "__main__":
    # register dag
    # https://stackoverflow.com/questions/63420667/can-airflow-persist-access-to-metadata-of-short-lived-dynamically-generated-task/63421315#63421315
    # n=1
    # dag_id = 'loop_hello_world_{}'.format(str(n))

    # default_args = {'owner': 'airflow',
    #                 'start_date': datetime(2021, 1, 1)
    #                 }

    # schedule = '@once'
    # dag_number = n

    # globals()[dag_id] = create_dag(dag_id,
    #                               schedule,
    #                               dag_number,
    #                               default_args)

    # execute dag
    # c = Client(api_base_url='http://localhost:18080', auth=None, session=None)
    # dag_id = "hello"
    # c.trigger_dag(dag_id=dag_id)#, run_id=f'test_run_id', conf={})

    asyncio.run(main())
