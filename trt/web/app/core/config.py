import hashlib
import json, logging.config, os
from pydantic import BaseSettings
import logging

from app import APP_ROOT_PATH

logger = logging.getLogger(__name__)

ENV_PATH = 'docker/.env'

class Settings(BaseSettings):
    ########## Dot Env ##########

    # Host path
    # <MOUNT_ROOT>/<STORGAE_DIR>         <SHARED_PATH><MINIO>
    # ├<AIRFLOW_DIR>                    <AIRFLOW>
    # ├<DATA_DIR>                       
    # └<MLFLOW_DIR>                     <MLFLOW>
    MOUNT_STORAGE_ROOT: str = '/volumes'
    STORAGE_DIR: str = 'bucket'
    AIRFLOW_DIR: str = 'airflow'
    DATA_DIR: str = 'data'
    MLFLOW_DIR: str = 'mlflow'

    # Each containers
    SHARED_VOLUME_DIR_STORAGE: str = '/data'                   # minio
    SHARED_VOLUME_DIR_MLFLOW: str = '/mlruns'                  # mlflow

    # Hosts
    MINIO_HOST: str = 'localhost'
    MINIO_PORT: str = '9000'
    MINIO_ACCESS_KEY: str = 'minio'
    MINIO_SECRET_KEY: str = 'minio123'

    MLFLOW_HOST: str = 'localhost'
    MLFLOW_PORT: str = '5000'

    SQLALCHEMY_DATABASE_URI: str = 'sqlite:///database.db'
    HOST: str = '0.0.0.0'
    PORT: int = 8000

    ARCHITECTURE: str = 'compute86'
    EXTERNAL_DOMAIN: str = 'localhost'

    TZ = 'asia/seoul'
    TIMEZONE: int = 9
    TIMEFORMAT: str = '%Y-%m-%dT%H:%M:%S.%f'
    ARCHIVE_FORMAT = 'zip' # tar gztar bztar xztar
    
    ########## CONST ##########

    API_V1_STR: str = "/api/v1"
    
    PROJECT_NAME: str = 'Auto TRT'
    LOG_CONF: str = '.log_conf.json'

    TIMEFORMAT_COCO: str = '%Y-%m-%d %H:%M:%S'

    SETTABLE = ['TZ', 'TIMEZONE', 'ARCHIVE_FORMAT']

    @property
    def value(self) -> dict:
        return { key:getattr(self, key) for key in self.SETTABLE}

    # setter will not work. See github.com/samuelcolvin/pydantic/issues/1577
    # @value.setter
    def set(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.SETTABLE: setattr(self, key, value)
            else: logger.warning(f"Key[{key}] is invalid setting")

    def tmp(self, uri:str) -> str:
        tmp_dir = f'/tmp/{hashlib.md5(uri.encode("utf8")).hexdigest()}'
        if not os.path.isdir(tmp_dir): os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    class Config:
        env_file = ENV_PATH
        env_file_encoding = 'utf-8'
        case_sensitive = True

# try:
#     load_dotenv(ENV_PATH)
# except Exception as e:
#     logger.warning('Env File is not Defined')

settings = Settings()

with open(settings.LOG_CONF) as json_file:
    conf = json.load(json_file)
    logging.config.dictConfig(conf)
