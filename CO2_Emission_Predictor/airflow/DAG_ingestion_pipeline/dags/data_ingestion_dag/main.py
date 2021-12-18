from datetime import timedelta, datetime
from pathlib import Path

from airflow import DAG
# We need to import the operators used in our tasks
from airflow.operators.python_operator import PythonOperator
# We then import the days_ago function
from airflow.utils.dates import days_ago

import pandas as pd
import sqlite3
import os

# get dag directory path
dag_path = os.getcwd()


def execution_date_to_millis(execution_date):
    """converts execution date (in DAG timezone) to epoch millis
    Args:
        date (execution date): %Y-%m-%d
    Returns:
        milliseconds
    """
    date = datetime.strptime(execution_date, "%Y-%m-%d")
    epoch = datetime.utcfromtimestamp(0)
    return (date - epoch).total_seconds() * 1000.0


def transform_data(exec_date):
    try:
        print(f"Ingesting data for date: {exec_date}")
        date = datetime.strptime(exec_date, '%Y-%m-%d %H')
        file_date_path = f"{date.strftime('%Y-%m-%d')}/{date.hour}"

        data = pd.read_csv(f"{dag_path}/raw_data/{file_date_path}/raw.csv", low_memory=False)

        data = preprocess(data)

        # load processed data
        output_dir = Path(f'{dag_path}/processed_data/{file_date_path}')
        output_dir.mkdir(parents=True, exist_ok=True)
        # processed_data/2021-08-15/12/2021-08-15_12.csv
        data.to_csv(output_dir / f"{file_date_path}.csv".replace("/", "_"), index=False, mode='a')

    except ValueError as e:
        print("datetime format should match %Y-%m-%d %H", e)
        raise e

def preprocess(X, models_dir, training_mode):
    X_preprocessed = data_cleaning_orchestrator(X)
    numeric_columns = get_numerical_columns(X_preprocessed)
    categorical_columns = get_categorical_columns(X_preprocessed)
    features = feature_selection(numeric_columns,categorical_columns)
    X_preprocessed = X_preprocessed[features]
    return X_preprocessed

def data_cleaning_orchestrator(X):
    X_preprocessed = handle_inconsist_data(X)
    X_preprocessed = column_rename(X_preprocessed)
    X_preprocessed  = preprocess_missing_values(X_preprocessed)

    return X_preprocessed


def column_rename(X):
    X.columns = X.columns.str.replace(r"\(.*?\)", "")
    X.columns = X.columns.str.rstrip(' ')
    X.columns = X.columns.str.replace(' ', '_')
    return X

def preprocess_missing_values(X):

    X_preprocessed_missing_values = X.apply(lambda x: x.fillna(x.value_counts().index[0]))
    return X_preprocessed_missing_values

def handle_inconsist_data(X):
    X_preprocessed = X.drop(['Fuel Consumption Comb (mpg)'], axis=1)
    return X_preprocessed


def get_numerical_columns(X):
    numeric_columns =  X.select_dtypes(include=['number']).columns.tolist()
    return numeric_columns

def get_categorical_columns(X):
    categorical_columns =  X.select_dtypes(include=['object']).columns.tolist()
    return categorical_columns


def feature_selection(numeric_columns,categorical_columns):
    exclude_feature = ['Make', 'Model', 'Vehicle_Class']
    categorical_features = [col for col in categorical_columns if col not in exclude_feature]
    exclude_feature = exclude_feature + categorical_features
    features = [col for col in numeric_columns if col not in exclude_feature]

    return features

def load_data(exec_date):
    print(f"Loading data for date: {exec_date}")
    date = datetime.strptime(exec_date, '%Y-%m-%d %H')
    file_date_path = f"{date.strftime('%Y-%m-%d')}/{date.hour}"

    conn = sqlite3.connect("/usr/local/airflow/db/datascience.db")
    c = conn.cursor()
    c.execute('''
                CREATE TABLE IF NOT EXISTS prediction (
                    engine_size FLOAT NOT NULL,
                    cylinders FLOAT NOT NULL,
                    fuel_consumption_city FLOAT NOT NULL,
                    fuel_consumption_hwy FLOAT NOT NULL,
                    fuel_consumption_comb FLOAT NOT NULL,
                    co2_emissions FLOAT NOT NULL
                    
                );
             ''')
    processed_file = f"{dag_path}/processed_data/{file_date_path}/{file_date_path.replace('/', '_')}.csv"
    records = pd.read_csv(processed_file)
    records.to_sql('prediction', conn, index=False, if_exists='append')


# initializing the default arguments that we'll pass to our DAG
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(5)
}

ingestion_dag = DAG(
    'CO2_Emission_ingestion',
    default_args=default_args,
    description='CO2 Emission records for data analysis',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    user_defined_macros={'date_to_millis': execution_date_to_millis}
)

task_1 = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    op_args=["{{ ds }} {{ execution_date.hour }}"],
    dag=ingestion_dag,
)

task_2 = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_args=["{{ ds }} {{ execution_date.hour }}"],
    dag=ingestion_dag,
)


task_1 >> task_2