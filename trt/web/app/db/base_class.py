from typing import Any

from sqlalchemy.ext.declarative import as_declarative, declared_attr, declarative_base


@as_declarative()
class _Base(object):
    id: Any
    __name__: str
    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(self) -> str:
        return self.__name__.lower()

Base = declarative_base(cls=_Base)