import argparse
import os, json, redis
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

def load_file(filename):
    with open(filename, 'r') as file:
        js = json.load(file)
    return js

def set_mem(key:str, data:dict):
    pass

def get_mem(key:str) -> dict:
    pass

def set_redis(key:str, data:dict):
    bson = json.dumps(data).encode('utf-8')
    redis_client().set(key, bson)

def get_redis(key:str):
    bson = redis_client().get(key)
    data = json.loads(bson.decode('utf-8'))
    return data

def save_file(filename, data:dict):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

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
def test_wo_redis(file_name:str, repeat:int):
    for _ in range(repeat):
        js = load_file(file_name)
        set_mem(file_name, js)
        get_mem(file_name)
        save_file(file_name, js)

@time_profile
@profile
def test_w_redis(file_name:str, repeat:int):
    js = load_file(file_name)
    for _ in range(repeat):
        set_redis(file_name, js)
        get_redis(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare File IO and Redis")
    parser.add_argument('rw_num', type=int, help='An Integer for number of r/w')
    parser.add_argument('--redis', action='store_true')
    parser.add_argument('--file', action='store_true')
    parser.add_argument('--file_name', type=str, help='A json file for rw', default='testfile2.json')
    args = parser.parse_args()

    if not args.redis and not args.file:
        test_wo_redis(args.file_name, args.rw_num)
        test_w_redis(args.file_name, args.rw_num)
    elif args.file:
        test_wo_redis(args.file_name, args.rw_num)
    elif args.redis:
        test_w_redis(args.file_name, args.rw_num)
