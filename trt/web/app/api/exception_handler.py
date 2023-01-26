import asyncio
from inspect import iscoroutinefunction
import logging
from fastapi import HTTPException
from functools import wraps
from http import HTTPStatus


logger = logging.getLogger(__name__)

def exception_handler(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_inner_function(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except KeyError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.NOT_FOUND.value,
                    detail=err_str
                )
            except ValueError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
                    detail=err_str
                )    
            except AttributeError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.CONFLICT.value,
                    detail=err_str
                )
            except NotImplementedError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.NOT_IMPLEMENTED.value,
                    detail=err_str
                )            
            except Exception as e:
                err_str = f'Internal error, msg={e}'
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    detail=err_str
                )
        return async_inner_function
    else:
        @wraps(func)
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.NOT_FOUND.value,
                    detail=err_str
                )
            except ValueError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY.value,
                    detail=err_str
                )    
            except AttributeError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.CONFLICT.value,
                    detail=err_str
                )
            except NotImplementedError as e:
                err_str = str(e)
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.NOT_IMPLEMENTED.value,
                    detail=err_str
                )            
            except Exception as e:
                err_str = f'Internal error, msg={e}'
                logger.error(err_str)
                raise HTTPException(
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    detail=err_str
                )
        return inner_function
