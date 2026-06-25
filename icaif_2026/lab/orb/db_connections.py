
from datetime import datetime

from sqlalchemy import (
    Column, String, Boolean, Integer, Float,
    Date, DateTime, JSON, UniqueConstraint, ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship


DATABASE_URL= "postgresql://postgres:postgres@localhost:5432/trading"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800
)