"""Initianlize db

Revision ID: cfb2f627341c

"""
import os, pathlib, json
from alembic import op
from sqlalchemy.sql import func
from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime, TIMESTAMP
from app.utils.guid_type import GUID

# revision identifiers, used by Alembic.
revision = 'cfb2f627341c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    model_tbl = op.create_table(
        'model',
        Column('id', GUID, primary_key=True, index=True),
        Column('production_id', GUID, nullable=False),
        Column('name', String),
        Column('classes', JSON),
        Column('desc', String, nullable=True),
        Column('tags', JSON),
        Column('location', String, nullable=True),
        Column('status', String, nullable=False),
        Column('capacity', Integer),
        Column('version', String),
        Column('platform', String),
        Column('framework', String),
        Column('precision', String),
        Column('updated', DateTime(timezone=True), server_default=func.now(), onupdate=func.now()),
        Column('created', DateTime(timezone=True), server_default=func.now()),    
    )
        
def downgrade():
    op.drop_table("model")
