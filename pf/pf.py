import argparse, os, json
import shutil

import redis
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame

from time import process_time
from memory_profiler import profile
# from typing import Generator

client:redis.Redis = None
def redis_client(host='127.0.0.1', port=6379) -> redis.Redis:
    global client
    if client is None:
        try:
            client = redis.Redis(host=host, port=port, db=0)
        except Exception as e:
            print(f"redis error! {e}")
        finally:
            client.close()
    return client

_spark_client:SparkSession = None
def spark_client(app_name='Profile') -> SparkSession:
    global _spark_client
    if _spark_client is None:
        try:
            _spark_client = SparkSession.builder.appName(app_name).getOrCreate()
            _spark_client.conf.set("spark.sql.execution.arrow.enabled", "true")
        except Exception as e:
            print(f"spakr error! {e}")
        finally:
            pass
            # _spark_client.stop()
    return _spark_client

def load_file(filename):
    with open(filename, 'r') as file:
        js = json.load(file)
    return js

def set_mem(key:str, data:dict):
    pass

def get_mem(key:str) -> dict:
    pass

def save_file(filename, data:dict):
    with open(filename, 'w') as file:
        json.dump(data, file) #, indent=2) spark not allow indent

def load_redis(key:str, data:dict):
    bson = json.dumps(data).encode('utf-8')
    redis_client().set(key, bson)

def set_redis(key:str, data:dict):
    set_mem(key, data)

def get_redis(key:str):
    get_mem(key)

def save_redis(key:str):
    bson = redis_client().get(key)
    data = json.loads(bson.decode('utf-8'))
    return data

def load_spark(filename):
    return spark_client().read.format('json').option("inferSchema", "true").load(filename)

def set_spark(key:str, data:dict):
    set_mem(key, data)
    # pdf = pd.DataFrame.from_dict(data)
    # df = spark_client().createDataFrame(pdf)

def get_spark(key:str):
    get_mem(key)
    # spark_client().sparkContext.parallelize()

def save_spark(filename, data:SparkDataFrame):
    data.write.format('json').mode('overwrite').save(filename)

def time_profile(fn):
    def wrapper_func(*args, **kwargs):
        t_start = process_time()
        fn(*args, **kwargs)
        t_stop = process_time()
        # print("Elapsed time:", t_stop, t_start)
        print(f"Elapsed time during the whole\
    [{fn.__name__}] in seconds:\t{t_stop-t_start}")
    return wrapper_func

@time_profile
@profile
def test_file(file_name:str, repeat:int):
    for _ in range(repeat):
        js = load_file(file_name)
        set_mem(file_name, js)
        get_mem(file_name)
        save_file(file_name, js)

@time_profile
@profile
def test_redis(file_name:str, repeat:int):
    # initial load
    js = load_file(file_name)
    for _ in range(repeat):
        load_redis(file_name, js)
        set_redis(file_name, js)
        get_redis(file_name)
        save_redis(file_name)

@time_profile
@profile
def test_spark(file_name:str, repeat:int):
    df = load_spark(file_name)
    dir_name = os.path.join(os.path.dirname(file_name), 'spark')
    file_name = os.path.join(dir_name, os.path.basename(file_name))
    save_spark(file_name+f'.0', df)
    for idx in range(repeat):
        df = load_spark(file_name+f'.{idx}')
        # shutil.rmtree(dir_name)
        set_spark(file_name, df)
        get_spark(file_name)
        save_spark(file_name+f'.{idx+1}', df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare File IO and Redis")
    parser.add_argument('rw_num', type=int, help='An Integer for number of r/w')
    parser.add_argument('--file', action='store_true')
    parser.add_argument('--redis', action='store_true')
    parser.add_argument('--spark', action='store_true')
    parser.add_argument('--file_name', type=str, help='A json file for rw', default='testfile.json')
    args = parser.parse_args()

    if not args.file and not args.redis and not args.spark:
        test_file(args.file_name, args.rw_num)
        test_redis(args.file_name, args.rw_num)
        test_spark(args.file_name, args.rw_num)

    if args.file:
        test_file(args.file_name, args.rw_num)
    if args.redis:
        test_redis(args.file_name, args.rw_num)
        client.close()
    if args.spark:
        test_spark(args.file_name, args.rw_num)
        _spark_client.stop()
