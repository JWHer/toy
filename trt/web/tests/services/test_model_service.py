from unittest import TestCase
from faker import Faker
from sqlalchemy.orm import Session
from sqlalchemy import MetaData

from app.api import deps 
from app.db.base_class import Base
from app.db.session import engine
from app.models.model import Model
from app.schemas.model import ModelDB, ModelCreateUser, ModelUpdate
from app.service.model_service import model_service

fake = Faker()
my_var = {}

def init_data():
    pass
        
def clear_data(session: Session, metadata: MetaData):
    for table in reversed(metadata.sorted_tables):
        session.execute(table.delete())
    session.commit()


class TestModelService(TestCase):

    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        init_data()

    @classmethod
    def tearDownClass(cls) -> None:
        Base.metadata.drop_all(bind=engine, checkfirst=True)

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test1_create(self) -> None:
        my_model = {
            "production_id": fake.uuid4(),
            "name": fake.name(),
            "classes": fake.pylist(4, True, [str]),
            "desc": fake.sentence(),
            "location": fake.url(),
            "capacity": fake.pyint(1,8),
            "version": "person-net_1.0-d7d3aa59_compute86",
            "platform": ["dgpu", "jetson", "trition"][fake.pyint(max_value=2)],
            "framework": ["deepstream==6.0", "deepstream==5.1"][fake.pyint(max_value=1)],
            "precision": ["FP16", "INT8"][fake.pyint(max_value=1)]
        }

        with deps.get_conn() as db:
            model_db = model_service.create(db, ModelCreateUser(**my_model))
            my_var['id'] = model_db.id
        print(f'created id={model_db.id}')
        assert model_db is not None

    def test2_read(self):
        with deps.get_conn() as db:
            model_db = model_service.get(db, id=my_var['id'])
        print(f'read id={model_db.id}')
        assert model_db is not None

    def test3_read_all(self):
        with deps.get_conn() as db:
            model_dbs = model_service.get_items(db)
        print(f'read all ids={[model.id for model in model_dbs]}')
        for model in model_dbs:
            assert model is not None
    
    def test4_update(self) -> None:
        my_update = {
            "name": fake.name(),
            "classes": fake.pylist(4, True, [str]),
            "desc": fake.sentence(),
            "location": fake.url(),
            "status": ["CREATED", "ALIVE", "FAILED", "DELETED"][fake.pyint(max_value=3)],
            "capacity": fake.pyint(1,8),
            "version": "person-net_1.0-d7d3aa59_compute86",
            "platform": ["dgpu", "jetson", "trition"][fake.pyint(max_value=2)],
            "framework": ["deepstream==6.0", "deepstream==5.1"][fake.pyint(max_value=1)],
            "precision": ["FP16", "INT8"][fake.pyint(max_value=1)]
        }
        with deps.get_conn() as db:
            model = model_service.update(
                    db, id = my_var['id'],
                    obj=ModelUpdate(**my_update)
                )
        print(f'updated id={model.id}')
        for key, value in my_update.items():
            assert getattr(model, key) == value
        
    def test4a_update_no_exist_id(self) -> None:
        my_update = {
            "name": fake.name(),
            "classes": fake.pylist(4, True, [str]),
            "desc": fake.sentence(),
            "location": fake.url(),
            "status": ["CREATED", "ALIVE", "FAILED", "DELETED"][fake.pyint(max_value=3)],
            "capacity": fake.pyint(1,8),
            "version": "person-net_1.0-d7d3aa59_compute86",
            "platform": ["dgpu", "jetson", "trition"][fake.pyint(max_value=2)],
            "framework": ["deepstream==6.0", "deepstream==5.1"][fake.pyint(max_value=1)],
            "precision": ["FP16", "INT8"][fake.pyint(max_value=1)]
        }
        id = fake.uuid4()
        with deps.get_conn() as db:
            self.assertRaises(
                KeyError,
                model_service.update,
                db, id,
                obj=ModelUpdate(**my_update)
            )
        print(f'update no exist id={id}')
            
    def test5_delete(self) -> None:
        with deps.get_conn() as db:
            model = model_service.delete(db, id=my_var['id'])
        print(f'deleted id={model.id}')
        assert model is not None
        
    def test5a_delete_no_exist_id(self) -> None:
        id = fake.uuid4()
        with deps.get_conn() as db:
            self.assertRaises(
                    Exception,
                    model_service.delete,
                    db, id
                )
            # model = model_service.delete(db, id)
        print(f'delete no exist id={id}')

    def test6_delete_all(self):
        with deps.get_conn() as db:
            model_dbs = model_service.delete_all(db)
        print(f'delete all ids={[model.id for model in model_dbs]}')
        assert model_dbs is not None
    