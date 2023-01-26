import asyncio, yaml
import logging, os, json, sys
from datetime import datetime
from urllib.parse import urlparse
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID, uuid4

from app import crud
from app.api import deps
from app.models.model import Model
from app.service.base.defines import Status
from app.schemas.model import ModelDB, ModelCreateUser, ModelCreate, ModelUpdate
from app.service.base.worker import worker
# from app.service.storage.storage_service import storage_service
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self) -> None:
        worker.register('build', self._build)

    def get_items(self, db: Session, skip=0, limit=100) -> List[ModelDB]:
        """
        Models를 획득한다.
        
        :param db: DB Session의 인스턴스
        :param skip (int, optional):  offset 수. Defaults to 0.
        :param limit (int, optional): limit 개수. Defaults to 100.
        :return Models
        """
        items : List[Model] = crud.model.get_multi(db=db, skip=skip, limit=limit)
        return [ ModelDB.from_orm(item) for item in items ]

    def get(self, db: Session, id) -> ModelDB:
        """
        Model을 획득한다.
        
        :param db: DB Session의 인스턴스
        :param id (Any): 획득할 Model의 id
        :return: Model
        """
        item : Model = crud.model.get(db=db, id=id)
        if item is None: raise KeyError(f'Model[{id}] not found')
        return ModelDB.from_orm(item)
        
    def create(self, db: Session, obj: ModelCreateUser) -> ModelDB:
        """
        Model을 생성한다.
        
        :param db: DB Session의 인스턴스
        :param item_in: 생성할 Model 정보
        :return: Model
        """
        base_id = obj.base_id
        del obj.base_id
        id = uuid4()
        if not obj.production_id:
            obj.production_id = id
        if not obj.location:
            obj.location=f'file://{settings.train}/0/{str(id).replace("-","")}/artifacts'
        if not obj.capacity:
            obj.capacity=1
        obj.tags.append(str(base_id))

        obj = ModelCreate(
                id=id,
                status=Status.ALIVE,
                **obj.dict()
            )
        created = crud.model.create(db, obj_in=obj)
        
        base_model = crud.model.get(db, id=base_id)
        if not base_model:
            logger.warning(f'Base Model[{base_id}] not exist.')
        else:
            base_db = ModelDB.from_orm(base_model)

            new_location = created.location
            origin_location = f'{settings.train}/0/{str(base_db.id).replace("-","")}/artifacts/model/data' if 'base' not in base_db.tags else urlparse(base_db.location).path

            model_cfgs = [ cfg for cfg in os.listdir(origin_location) if '.json' in cfg]
            for model_cfg in model_cfgs:

                with open(f'{origin_location}/{model_cfg}', 'r') as cfg_file:
                    model_cfg_dict = json.load(cfg_file)
                model_cfg_dict['classes'] = created.classes
                model_cfg_dict['num_classes']=len(created.classes)

                dest_uri = new_location+'/model/data/'
                storage_service.create(dest_uri, None)
                with open(urlparse(dest_uri+model_cfg).path, 'w') as cfg_file:
                    json.dump(model_cfg_dict, cfg_file, indent=2)
            storage_service.copy(f'file://{origin_location}/model.pth', f'file://{new_location}/model/data/model.pth')

        return ModelDB.from_orm(created)

    def update(self, db: Session, id, obj: ModelUpdate) -> ModelDB:
        """
        Model을 업데이트한다.
        
        :param db: DB Session의 인스턴스
        :param item_in: 업데이트할 Model 정보
        :return: Model
        """
        item : Model = crud.model.get(db=db, id=id)
        if item is None: raise KeyError(f'Model[{id}] not found')
        updated = crud.model.update(db=db, db_obj=item, obj_in=obj)
        return ModelDB.from_orm(updated)

    def delete(self, db: Session, id) -> ModelDB:
        """
        Model을 삭제한다.
        
        :param db: DB Session의 인스턴스
        :param id: 삭제할 Model id
        :return: Model
        """
        removed = crud.model.remove(db=db, id=id)
        if removed is None: raise KeyError(f'Model[{id}] not found')
        return ModelDB.from_orm(removed)
        
    def delete_all(self, db: Session) -> List[ModelDB]:
        """
        Models를 삭제한다.
        
        :param db: DB Session의 인스턴스
        :return: Models
        """
        items: List[Model] = crud.model.remove_multi(db=db)
        return [ModelDB.from_orm(item) for item in items]
    
    # TODO : project, storage service 구현 필요
    def is_downloadable(self, db: Session, id) -> ModelDB:
        """
        Model이 Download가 가능한 상태인지 확인한다.

        :param db: DB Session의 인스턴스
        :param item_in: 업데이트할 Model 정보
        :return: Model
        """
        raise NotImplementedError('Not Implemented')
        # model: Model = curd.model.get(db, id)
        # train: Train = crud.train.get(db, id)
        # project = crud.project.get(db=db, id=train.project_id)

        # run_id = str(train.id).replace('-', '')
        # path = f'{project.serial}/{run_id}/artifacts/model/'

        # obj = ModelUpdate(
        #     status=Status.ALIVE if os.path.isdir(path) else Status.FAILED,
        #     **model
        # )
        # updated = crud.model.update(db, db_obj=model, obj_in=obj)
        # return ModelDB.from_orm(updated)

    def build(self, db: Session, id, onnx_config: dict=None, archiver_config: dict = None) -> ModelDB:
        model = crud.model.get(db, id)
        if model is None:
            raise KeyError(f'Model[{id}] not found')
        model_db = ModelDB.from_orm(model)
        if urlparse(model_db.location).scheme not in storage_service.map.keys():
            raise ValueError(f'Model[{id}] has already been built.')
        building = crud.model.update(db, db_obj=model, obj_in={"status": Status.RUNNING})
        worker.run_task('build', args=[model_db, onnx_config, archiver_config])
        return ModelDB.from_orm(building)


    # TODO Proejct service 구현 필요
    async def _build(self, model_db:ModelDB, onnx_config: dict=None, archiver_config: dict = None) -> ModelDB:
        model_dir = f"{settings.train}/{urlparse(model_db.location).path}/model/"
        model_cfg = f"{settings.train}/{urlparse(model_db.location).path}/model/data/model.json"

        # 1. make onnx
        default_onnx_config = self._default_onnx_config(model_dir, model_cfg)
        if onnx_config: default_onnx_config.update(onnx_config)
        onnx_config = default_onnx_config
        onnx_cmd = self._make_cmd('/opt/model/tools/export_onnx.py', onnx_config)

        onnx_proc = await asyncio.create_subprocess_shell(
            onnx_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await onnx_proc.communicate()
        if stdout:
            logger.info(f'[onnx][stdout] : {stdout.decode()}')
        if stderr:
            logger.error(f'[onnx][stderr] : {stderr.decode()}')

        if onnx_proc.returncode!=0:
            with deps.get_conn() as db:
                model = crud.model.get(db, model_db.id)
                updated = crud.model.update(db, db_obj=model, obj_in={"status": Status.FAILED})
            raise Exception(f'Model[{model_db.id}] onnx build failed.')

        # 2. build engine
        default_archiver_config = self._default_archiver_config(model_dir, model_db.classes)
        if archiver_config: default_archiver_config.update(archiver_config)
        archiver_config = default_archiver_config
        archive_config_file = model_dir + 'archive.yaml'
        with open(archive_config_file, 'w') as acf:
            yaml.dump(archiver_config, acf)
        archived_zip_path = os.path.normpath(model_dir+"model_archive.zip")
        archive_cmd = f'python ./archiver/main.py --archive-file {archived_zip_path} --archive-config-file {archive_config_file}'
        archive_cmd = archive_cmd.split(' ')
        
        archive_proc = await asyncio.create_subprocess_exec(
            *archive_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await archive_proc.communicate()
        if stdout:
            logger.info(f'[archive][stdout] : {stdout.decode()}')
        if stderr:
            logger.error(f'[archive][stderr] : {stderr.decode()}')

        if archive_proc.returncode!=0:
            with deps.get_conn() as db:
                model = crud.model.get(db, model_db.id)
                updated = crud.model.update(db, db_obj=model, obj_in={"status": Status.FAILED})
            raise Exception(f'Model[{model_db.id}] engine build failed.')

        # 3. add model.conf
        version = f"{model_db.name}_{datetime.utcnow().strftime('%y%m%d')}-{str(model_db.id)[-8:]}_{settings.ARCHITECTURE}"
        model_conf = {
            "id": str(model_db.id),
            "name": model_db.name,
            "version": version,  #{name}_{version}-{revision}_{architecture}
            "platform": "dgpu",
            "framework": "deepstream==6.0",
            "capacity": onnx_config['batch_size'],
            "precision": "FP16",    # ??
            "archive file": archived_zip_path,
            "description": model_db.desc
        }
        model_conf_path = os.path.normpath(model_dir+"model.conf")
        with open(model_conf_path, 'w') as mcf:
            json.dump(model_conf, mcf)

        with deps.get_conn() as db:
            model = crud.model.get(db, model_db.id)
            uri = f'file://{archived_zip_path}'
            updated = crud.model.update(db, db_obj=model, obj_in=ModelUpdate(
                capacity=onnx_config['batch_size'],
                status=Status.ALIVE,
                version=version,
                platform=model_conf['platform'],
                framework=model_conf['framework'],
                precision=model_conf['precision'],
                # location=f"http://{settings.EXTERNAL_DOMAIN}/api/v1/storage/model/{model.id}"
                location=f"http://{settings.EXTERNAL_DOMAIN}/api/v1/storage/{uri}"
            ))

        return ModelDB.from_orm(updated)

    def _make_cmd(self, filename:str, pydict:dict):
        cmd = f"PYTHONPATH='/opt/model/' python {filename}"
        for key, value in pydict.items():
            if value and not (isinstance(value, bool) and value):
                cmd+=f' --{key} {value}'
            else:
                cmd+=f' --{key}'
        return cmd

    def _default_onnx_config(self, model_dir:str, model_cfg:str=None) -> dict:
        onnx_config = {
            'output_name': os.path.normpath(model_dir+"data/model.onnx"),
            'batch_size': 1,
            'opset': 11,
            'input_size': 512, #TODO take from model.json
            # 'no-onnxsim': True,
            'ckpt': os.path.normpath(model_dir+"data/model.pth")
        }
        
        if model_cfg:
            onnx_config['model_cfg'] = model_cfg
        return onnx_config

    def _default_archiver_config(self, model_dir:str, classes:list) -> dict:
        archive_file = "file://" + os.path.normpath(model_dir+"data/model.onnx")
        engine_file = "file://" + os.path.normpath(model_dir+"data/model.engine")
        label_file = os.path.normpath(model_dir+"data/labels.txt")
        with open(label_file, 'w') as lf:
            lf.write("\n".join(classes))
        label_file = "file://" + label_file
        custom_lib_path = f"file:///opt/archiver/resources/tmp/PeopleNet_v1.1.0.parser.so" #TODO change

        return {
                "model":{
                    "sources":{
                        "infer@model.onnx":{
                            "url": archive_file,
                        },
                        "trt@model.tensorrt":{
                            "from": "#model.sources.infer",
                            "url": engine_file,
                            "builder":{
                                "max_batch_size":1
                            },
                            "config":{
                                "max_workspace_size":1,     # GB
                                "flag":1                    # FP16
                            },
                            "profile":{
                                "dynamic_batch_size": [1,1,1]
                            }
                        }
                    },
                    "runtime":{
                        "framework":{
                            "infer@framework.deepstream.nvinfer":{
                                "configs":{
                                    "property":{
                                        "labelfile-path": label_file,
                                        "custom-lib-path": custom_lib_path,
                                        "parse-bbox-func-name":"NvDsInferParseModel",
                                        "model-engine-file": "#model.sources.trt.url",
                                        "gpu-id" : "0",
                                        "net-scale-factor" : "0.0039215697906911373",
                                        "model-color-format" : "0",
                                        "num-detected-classes" : "1",
                                        "interval" : "0",
                                        "gie-unique-id" : "1",
                                        "batch-size" : "1",
                                        "network-mode" : "2",
                                        "process-mode" : "1",
                                        "network-type" : "0",
                                        "symmetric-padding" : "1",
                                        "infer-dims" : "3;512;512",
                                        "cluster-mode" : "3",
                                        "network-input-order" : "0"
                                    },
                                    "class-attrs-all":{
                                        "threshold" : "0.5",
                                        "dbscan-min-score" : "0.7",
                                        "nms-iou-threshold" : "0.7"
                                    }
                                },
                                "attributes":{
                                    "class_type":"auto"
                                }
                            }
                        }
                    }
                }
            }

model_service = ModelService()