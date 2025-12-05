"""
Comprehensive tests for SQLModel ↔ Pydantic conversions.
"""

import pytest
import uuid
from datetime import datetime, date
from articuno.sqlmodel_infer import (
    infer_pydantic_model_from_sqlmodel,
    pydantic_to_sqlmodel,
)
from articuno import infer_pydantic_model

# Optional imports
try:
    from sqlmodel import SQLModel, Field

    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False


pytestmark = pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not installed")


def _unique_table_name(base_name: str) -> str:
    """Generate a unique table name for test isolation."""
    return f"{base_name}_{uuid.uuid4().hex[:8]}"


def test_sqlmodel_to_pydantic_basic_types():
    """Test conversion from SQLModel to Pydantic with basic types."""

    class User(SQLModel, table=True):
        __tablename__ = "users_basic_types"
        id: int | None = Field(default=None, primary_key=True)
        name: str
        age: int | None = None
        score: float
        active: bool = True

    PydanticModel = infer_pydantic_model_from_sqlmodel(User, model_name="UserModel")

    instance = PydanticModel(id=1, name="Alice", age=30, score=95.5, active=True)
    assert instance.id == 1
    assert instance.name == "Alice"
    assert instance.age == 30
    assert instance.score == 95.5
    assert instance.active is True


def test_sqlmodel_to_pydantic_optional_fields():
    """Test that optional SQLModel fields are preserved in Pydantic."""

    class Product(SQLModel, table=True):
        __tablename__ = "products"
        id: int | None = Field(default=None, primary_key=True)
        name: str
        description: str | None = None
        price: float

    PydanticModel = infer_pydantic_model_from_sqlmodel(
        Product, model_name="ProductModel"
    )

    # Can create with None for optional field
    instance = PydanticModel(id=1, name="Widget", description=None, price=19.99)
    assert instance.description is None

    # Can also provide value
    instance2 = PydanticModel(
        id=2, name="Gadget", description="A cool gadget", price=29.99
    )
    assert instance2.description == "A cool gadget"


def test_sqlmodel_to_pydantic_temporal_types():
    """Test conversion of datetime and date types."""

    class Event(SQLModel, table=True):
        __tablename__ = "events"
        id: int | None = Field(default=None, primary_key=True)
        event_date: date
        created_at: datetime

    PydanticModel = infer_pydantic_model_from_sqlmodel(Event, model_name="EventModel")

    instance = PydanticModel(
        id=1, event_date=date(2024, 1, 15), created_at=datetime(2024, 1, 15, 10, 30)
    )
    assert instance.event_date == date(2024, 1, 15)
    assert instance.created_at == datetime(2024, 1, 15, 10, 30)


def test_sqlmodel_to_pydantic_force_optional():
    """Test force_optional parameter."""

    class Item(SQLModel, table=True):
        __tablename__ = "items"
        id: int | None = Field(default=None, primary_key=True)
        name: str
        value: int

    PydanticModel = infer_pydantic_model_from_sqlmodel(
        Item, model_name="ItemModel", force_optional=True
    )

    # All fields should be Optional
    instance = PydanticModel(id=None, name=None, value=None)
    assert instance.id is None
    assert instance.name is None
    assert instance.value is None


def test_pydantic_to_sqlmodel_basic_types():
    """Test conversion from Pydantic model to SQLModel."""
    from pydantic import BaseModel

    class UserModel(BaseModel):
        id: int | None = None
        name: str
        age: int | None = None
        score: float
        active: bool = True

    table_name = _unique_table_name("users_basic_types")
    SQLModelClass = pydantic_to_sqlmodel(
        UserModel, model_name="User", table_name=table_name
    )

    # Verify it's a SQLModel
    assert issubclass(SQLModelClass, SQLModel)
    assert hasattr(SQLModelClass, "__tablename__")
    assert SQLModelClass.__tablename__ == table_name
    assert hasattr(SQLModelClass, "id")
    assert hasattr(SQLModelClass, "name")
    assert hasattr(SQLModelClass, "age")
    assert hasattr(SQLModelClass, "score")
    assert hasattr(SQLModelClass, "active")


def test_pydantic_to_sqlmodel_optional_fields():
    """Test that Optional Pydantic fields are preserved in SQLModel."""
    from pydantic import BaseModel

    class ProductModel(BaseModel):
        id: int | None = None
        name: str
        description: str | None = None
        price: float

    table_name = _unique_table_name("products_optional")
    SQLModelClass = pydantic_to_sqlmodel(
        ProductModel, model_name="Product", table_name=table_name
    )

    # Verify optional field exists
    assert hasattr(SQLModelClass, "description")

    # Can create instance with None
    instance = SQLModelClass(id=1, name="Widget", description=None, price=19.99)
    assert instance.description is None


def test_pydantic_to_sqlmodel_temporal_types():
    """Test conversion of datetime and date types."""
    from pydantic import BaseModel

    class EventModel(BaseModel):
        id: int | None = None
        event_date: date
        created_at: datetime

    table_name = _unique_table_name("events_temporal")
    SQLModelClass = pydantic_to_sqlmodel(
        EventModel, model_name="Event", table_name=table_name
    )

    # Verify temporal fields exist
    assert hasattr(SQLModelClass, "event_date")
    assert hasattr(SQLModelClass, "created_at")

    instance = SQLModelClass(
        id=1, event_date=date(2024, 1, 15), created_at=datetime(2024, 1, 15, 10, 30)
    )
    assert instance.event_date == date(2024, 1, 15)
    assert instance.created_at == datetime(2024, 1, 15, 10, 30)


def test_infer_pydantic_model_with_sqlmodel():
    """Test that infer_pydantic_model works with SQLModel classes."""

    class User(SQLModel, table=True):
        __tablename__ = "users_infer_test"
        id: int | None = Field(default=None, primary_key=True)
        name: str
        email: str | None = None

    # Test using the unified infer_pydantic_model function
    PydanticModel = infer_pydantic_model(User, model_name="UserModel")

    instance = PydanticModel(id=1, name="Alice", email="alice@example.com")
    assert instance.id == 1
    assert instance.name == "Alice"
    assert instance.email == "alice@example.com"


def test_sqlmodel_model_name_validation():
    """Test that invalid model names raise ValueError."""

    class User(SQLModel, table=True):
        __tablename__ = "users_validation_test"
        id: int | None = Field(default=None, primary_key=True)
        name: str

    with pytest.raises(ValueError, match="Invalid model name"):
        infer_pydantic_model_from_sqlmodel(User, model_name="123Invalid")
