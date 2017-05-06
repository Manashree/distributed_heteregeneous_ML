"""
Code that goes along with the Airflow located at:
http://airflow.readthedocs.org/en/latest/tutorial.html
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'Team Sayian',
    'depends_on_past': False,
    'start_date': datetime(2017, 5, 5),
    'email': ['animhan@indiana.edu'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG(
    'allstate', default_args=default_args, schedule_interval=timedelta(1))

# t1, t2 and t3 are examples of tasks created by instantiating operators
t1 = BashOperator(
    task_id='Preprocessing Data',
    bash_command='python allstate-factorize.py',
    dag=dag)

t2 = BashOperator(
    task_id='Run Model 1 - RFR',
    bash_command='python allstate-final.py classifier=rfr',
    retries=3,
    dag=dag)

t3 = BashOperator(
    task_id='Run Model 2 - ENCV',
    bash_command='python allstate-final.py classifier=encv',
    dag=dag)

t4 = BashOperator(
    task_id='Combine Submissions-MetaClassifier',
    bash_command='python combine_submission.py files=2 w1=0.9 w2=0.1 f1=rfr_predictions.csv f2=encv_predictions.csv output=submission_1.csv',
    retries=3,
    dag=dag)

t2.set_upstream(t1)
t3.set_upstream(t1)
t4.set_upstream(t2)
t4.set_upstream(t3)
