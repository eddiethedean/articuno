"""
Comprehensive tests for SQLAlchemy ↔ Pydantic conversions.
"""

import pytest
from datetime import datetime, date, time
from articuno.sqlalchemy_infer import (
    infer_pydantic_model_from_sqlalchemy,
    pydantic_to_sqlalchemy,
)
from articuno import infer_pydantic_model

# Optional imports
try:
    from sqlalchemy import (
        Column,
        Integer,
        String,
        Float,
        Boolean,
        DateTime,
        Date,
        Time,
    )
    from sqlalchemy.orm import DeclarativeBase

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not installed"
)


def test_sqlalchemy_to_pydantic_basic_types():
    """Test conversion from SQLAlchemy model to Pydantic with basic types."""

    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String(255), nullable=False)
        age = Column(Integer, nullable=True)
        score = Column(Float, nullable=False)
        active = Column(Boolean, default=True)

    PydanticModel = infer_pydantic_model_from_sqlalchemy(User, model_name="UserModel")

    instance = PydanticModel(id=1, name="Alice", age=30, score=95.5, active=True)
    assert instance.id == 1
    assert instance.name == "Alice"
    assert instance.age == 30
    assert instance.score == 95.5
    assert instance.active is True


def test_sqlalchemy_to_pydantic_nullable_fields():
    """Test that nullable SQLAlchemy fields become Optional in Pydantic."""

    class Base(DeclarativeBase):
        pass

    class Product(Base):
        __tablename__ = "products"
        id = Column(Integer, primary_key=True)
        name = Column(String(255), nullable=False)
        description = Column(String(1000), nullable=True)
        price = Column(Float, nullable=False)

    PydanticModel = infer_pydantic_model_from_sqlalchemy(
        Product, model_name="ProductModel"
    )

    # Can create with None for nullable field
    instance = PydanticModel(id=1, name="Widget", description=None, price=19.99)
    assert instance.description is None

    # Can also provide value
    instance2 = PydanticModel(
        id=2, name="Gadget", description="A cool gadget", price=29.99
    )
    assert instance2.description == "A cool gadget"


def test_sqlalchemy_to_pydantic_temporal_types():
    """Test conversion of datetime, date, and time types."""

    class Base(DeclarativeBase):
        pass

    class Event(Base):
        __tablename__ = "events"
        id = Column(Integer, primary_key=True)
        event_date = Column(Date, nullable=False)
        event_time = Column(Time, nullable=True)
        created_at = Column(DateTime, nullable=False)

    PydanticModel = infer_pydantic_model_from_sqlalchemy(Event, model_name="EventModel")

    instance = PydanticModel(
        id=1,
        event_date=date(2024, 1, 15),
        event_time=time(10, 30),
        created_at=datetime(2024, 1, 15, 10, 30),
    )
    assert instance.event_date == date(2024, 1, 15)
    assert instance.event_time == time(10, 30)
    assert instance.created_at == datetime(2024, 1, 15, 10, 30)


def test_sqlalchemy_to_pydantic_force_optional():
    """Test force_optional parameter."""

    class Base(DeclarativeBase):
        pass

    class Item(Base):
        __tablename__ = "items"
        id = Column(Integer, primary_key=True)
        name = Column(String(255), nullable=False)
        value = Column(Integer, nullable=False)

    PydanticModel = infer_pydantic_model_from_sqlalchemy(
        Item, model_name="ItemModel", force_optional=True
    )

    # All fields should be Optional
    instance = PydanticModel(id=None, name=None, value=None)
    assert instance.id is None
    assert instance.name is None
    assert instance.value is None


def test_pydantic_to_sqlalchemy_basic_types():
    """Test conversion from Pydantic model to SQLAlchemy."""
    from pydantic import BaseModel

    class UserModel(BaseModel):
        id: int
        name: str
        age: int | None = None
        score: float
        active: bool = True

    SQLAlchemyModel = pydantic_to_sqlalchemy(
        UserModel, model_name="User", table_name="users"
    )

    # Verify it's a SQLAlchemy model
    assert hasattr(SQLAlchemyModel, "__tablename__")
    assert SQLAlchemyModel.__tablename__ == "users"
    assert hasattr(SQLAlchemyModel, "id")
    assert hasattr(SQLAlchemyModel, "name")
    assert hasattr(SQLAlchemyModel, "age")
    assert hasattr(SQLAlchemyModel, "score")
    assert hasattr(SQLAlchemyModel, "active")


def test_pydantic_to_sqlalchemy_optional_fields():
    """Test that Optional Pydantic fields become nullable in SQLAlchemy."""
    from pydantic import BaseModel
    from typing import Optional

    class ProductModel(BaseModel):
        id: int
        name: str
        description: Optional[str] = None
        price: float

    SQLAlchemyModel = pydantic_to_sqlalchemy(ProductModel, model_name="Product")

    # Check that description column is nullable
    from sqlalchemy import inspect

    mapper = inspect(SQLAlchemyModel)
    desc_column = mapper.columns.get("description")
    assert desc_column is not None
    assert desc_column.nullable is True


def test_pydantic_to_sqlalchemy_temporal_types():
    """Test conversion of datetime, date, and time types."""
    from pydantic import BaseModel

    class EventModel(BaseModel):
        id: int
        event_date: date
        event_time: time | None = None
        created_at: datetime

    SQLAlchemyModel = pydantic_to_sqlalchemy(EventModel, model_name="Event")

    # Verify temporal columns exist
    assert hasattr(SQLAlchemyModel, "event_date")
    assert hasattr(SQLAlchemyModel, "event_time")
    assert hasattr(SQLAlchemyModel, "created_at")


def test_infer_pydantic_model_with_sqlalchemy():
    """Test that infer_pydantic_model works with SQLAlchemy models."""

    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String(255), nullable=False)
        email = Column(String(255), nullable=True)

    # Test using the unified infer_pydantic_model function
    PydanticModel = infer_pydantic_model(User, model_name="UserModel")

    instance = PydanticModel(id=1, name="Alice", email="alice@example.com")
    assert instance.id == 1
    assert instance.name == "Alice"
    assert instance.email == "alice@example.com"


def test_sqlalchemy_model_name_validation():
    """Test that invalid model names raise ValueError."""

    class Base(DeclarativeBase):
        pass

    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        name = Column(String(255))

    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model_from_sqlalchemy(User, model_name="123Invalid")
