from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Any, List, Dict
from uuid import UUID
import logging

from app.api import deps
from app.api.exception_handler import exception_handler
from app.schemas.model import ModelDB, ModelCreateUser, ModelUpdate
from app.service.model_service import model_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[ModelDB])
@exception_handler
def read_models(
    db: Session = Depends(deps.get_db),
    *,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Models를 획득한다.

    :param db: DB Session의 인스턴스
    :param skip (int, optional): offset 수. Defaults to 0.
    :param limit (int, optional): limit 개수. Defaults to 100.
    :return Models
    """
    items = model_service.get_items(db, skip, limit)
    return items


@router.get("/{id}", response_model=ModelDB)
@exception_handler
def read_model(
    db: Session = Depends(deps.get_db),
    *,
    id: UUID,
) -> Any: 
    """
    Model을 획득한다.
    
    :param db: DB Session의 인스턴스
    :param id: 획득할 Model id
    :return: Model
    :raises HTTPException: 500
    """
    item = model_service.get(db, id)
    return item


@router.post("/", response_model=ModelDB)
@exception_handler
def create_model(
    db: Session = Depends(deps.get_db),
    *,
    item_in: ModelCreateUser
) -> Any:
    """
    Model을 생성한다.
    
    :param db: DB Session의 인스턴스
    :param item_in: 생성할 Model 정보
    :return: Model
    """
    item = model_service.create(db, obj=item_in)
    return item


@router.put("/{id}", response_model=ModelDB)
@exception_handler
def update_model(
    db: Session = Depends(deps.get_db),
    *,
    id: UUID,
    item_in: ModelUpdate
) -> Any:
    """
    Model을 업데이트한다.
    
    :param db: DB Session의 인스턴스
    :param id: 업데이트할 Model id
    :param item_in: 업데이트할 Model 정보
    :return: Model
    :raises HTTPException: 500
    """
    item = model_service.update(db, id=id, obj=item_in)
    return item


@router.delete("/{id}", response_model=ModelDB)
@exception_handler
def delete_model(
    db: Session = Depends(deps.get_db),
    *,
    id: UUID
) -> Any:
    """
    Model을 삭제한다.
    
    :param db: DB Session의 인스턴스
    :param id: 삭제할 Model id
    :return: Model
    :raises HTTPException: 500
    """
    item = model_service.delete(db, id=id)
    return item


@router.delete("/", response_model=List[ModelDB])
@exception_handler
def delete_models(
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Models를 삭제한다.
    
    :param db: DB Session의 인스턴스
    :return: Models
    """
    items = model_service.delete_all(db=db)
    return items


@router.post("/{id}/build", response_model=ModelDB)
@exception_handler
def build_model(
    db: Session = Depends(deps.get_db),
    *,
    id: UUID,
    onnx_config: dict=None,
    archiver_config: dict=None,
) -> Any:
    """
    Build Model

    Args:
        onnx_config: dict
        archiver_config: dict

    Returns:
        Model
    """
    item = model_service.build(
        db=db, id=id,
        onnx_config=onnx_config,
        archiver_config=archiver_config
    )
    return item
