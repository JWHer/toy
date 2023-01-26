from operator import mod
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse
from typing import Any, List, Dict
from uuid import UUID
import logging, io

from app.api import deps
from app.api.exception_handler import exception_handler
from app.core.config import settings
from app.schemas.storage.storage import Storage, StorageUpdate
from app.service.model_service import model_service
from app.service.project_service import project_service
from app.service.storage.storage_service import storage_service
from app.service.train.train_service import train_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/model/{id}")
@exception_handler
def read_model_storage(
    id:UUID,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Archive 된 모델을 가져온다

    :param id: 가져올 모델 id
    :return StreamingResponse
    :raises HTTPException: 500
    """
    model = model_service.get(db, id)

    # TODO check if is build
    uri = f'file://{model.model_dir}'

    item, mime = storage_service.get_zip(uri)
    return StreamingResponse(io.BytesIO(item), media_type=mime)


@router.get("/list/{dir_path:path}", response_model=List[Storage])
@exception_handler
def list_storages(
    dir_path: str,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Storages를 획득한다.

    :param dir_path: 조회할 경로
    :param skip (int, optional): offset 수. Defaults to 0.
    :param limit (int, optional): limit 개수. Defaults to 100.
    :return Storages
    """
    items = storage_service.get_items(dir_path, skip, limit)
    return items


@router.get("/{path:path}")
@exception_handler
def read_storage(
    path: str
) -> Any: 
    """
    Storage을 획득한다.
    
    :param path: 획득할 파일
    :return StreamingResponse
    :raises HTTPException: 500
    """
    item, mime = storage_service.get(path)
    return StreamingResponse(io.BytesIO(item), media_type=mime)


# @router.post("/", response_model=Storage)
# def create_storage(

# ) -> Any:
#     """
#     Storage을 생성한다.
    
#     :return: Storage
#     """
#     try:
#         item = storage_service.create(path, file)
#         return item
#     except Exception as e:
#         err_str = f'StorageService - create error, msg={e}'
#         logger.error(err_str)
#         raise HTTPException(
#             status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
#             detail=err_str
#         )


# @router.put("/{id}", response_model=Storage)
# def update_storage(

#     item_in: StorageUpdate
# ) -> Any:
#     """
#     Storage을 업데이트한다.
    
#     :param db: DB Session의 인스턴스
#     :param id: 업데이트할 Storage id
#     :param item_in: 업데이트할 Storage 정보
#     :return: Storage
#     :raises HTTPException: 500
#     """
#     try:
#         item = storage_service.update(path, obj=item_in)
#         return item
#     except KeyError as e:
#         err_str = str(e)
#         logger.error(err_str)
#         raise HTTPException(
#             status_code=HTTPStatus.NOT_FOUND.value,
#             detail=err_str
#         )
#     except Exception as e:
#         err_str = f'StorageService - update error, msg={e}'
#         logger.error(err_str)
#         raise HTTPException(
#             status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
#             detail=err_str
#         )


# @router.delete("/{path:path}", response_model=List[Storage])
# def delete_storage(
#     path: str
# ) -> Any:
#     """
#     Storage을 삭제한다.
    
#     :param path: 삭제할 경로
#     :return: List[Storage]
#     :raises HTTPException: 500
#     """
#     try:
#         items = storage_service.delete(path)
#         return items
#     except KeyError as e:
#         err_str = str(e)
#         logger.error(err_str)
#         raise HTTPException(
#             status_code=HTTPStatus.NOT_FOUND.value,
#             detail=err_str
#         )
#     except Exception as e:
#         err_str = f'StorageService - delete error, msg={e}'
#         logger.error(err_str)
#         raise HTTPException(
#             status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
#             detail=err_str
#         )
