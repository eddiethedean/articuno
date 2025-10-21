"""
Tests for realistic, real-world usage scenarios.
"""

import pytest
from datetime import datetime, date
from articuno import infer_pydantic_model, df_to_pydantic
from articuno.codegen import generate_class_code

# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


def test_ecommerce_order():
    """Test e-commerce order with nested items."""
    orders = [
        {
            "order_id": 1001,
            "customer": {
                "id": 501,
                "name": "Alice Johnson",
                "email": "alice@example.com"
            },
            "items": [
                {"product": "Laptop", "quantity": 1, "price": 999.99},
                {"product": "Mouse", "quantity": 2, "price": 29.99}
            ],
            "total": 1059.97,
            "status": "shipped",
            "created_at": datetime(2024, 1, 15, 10, 30)
        },
        {
            "order_id": 1002,
            "customer": {
                "id": 502,
                "name": "Bob Smith",
                "email": "bob@example.com"
            },
            "items": [
                {"product": "Keyboard", "quantity": 1, "price": 79.99}
            ],
            "total": 79.99,
            "status": "pending",
            "created_at": datetime(2024, 1, 16, 14, 45)
        }
    ]
    
    Model = infer_pydantic_model(orders, model_name="EcommerceOrder")
    instances = list(df_to_pydantic(orders, model=Model))
    
    assert len(instances) == 2
    assert instances[0].order_id == 1001
    assert instances[0].customer.name == "Alice Johnson"
    assert len(instances[0].items) == 2
    assert instances[0].total == 1059.97
    assert instances[1].status == "pending"


def test_user_profile_with_optionals():
    """Test user profile with timestamps and optional fields."""
    users = [
        {
            "user_id": 1,
            "username": "alice",
            "email": "alice@example.com",
            "full_name": "Alice Johnson",
            "bio": "Software developer",
            "avatar_url": "https://example.com/alice.jpg",
            "created_at": datetime(2023, 1, 1, 0, 0),
            "last_login": datetime(2024, 1, 15, 10, 30),
            "is_verified": True,
            "subscription_tier": "premium"
        },
        {
            "user_id": 2,
            "username": "bob",
            "email": "bob@example.com",
            "full_name": None,  # Optional
            "bio": None,  # Optional
            "avatar_url": None,  # Optional
            "created_at": datetime(2023, 6, 15, 0, 0),
            "last_login": None,  # Never logged in
            "is_verified": False,
            "subscription_tier": "free"
        }
    ]
    
    Model = infer_pydantic_model(users, model_name="UserProfile")
    instances = list(df_to_pydantic(users, model=Model))
    
    assert len(instances) == 2
    assert instances[0].username == "alice"
    assert instances[0].is_verified is True
    assert instances[1].full_name is None
    assert instances[1].last_login is None


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_timeseries_data():
    """Test time-series data with datetime index."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "temperature": [20 + i * 0.1 for i in range(100)],
        "humidity": [50 + i * 0.05 for i in range(100)],
        "pressure": [1013 - i * 0.01 for i in range(100)]
    })
    
    Model = infer_pydantic_model(df, model_name="WeatherData")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert len(instances) == 100
    assert instances[0].temperature == 20.0
    assert instances[99].humidity > 50


def test_api_response_nested_json():
    """Test API response-like nested JSON structure."""
    api_responses = [
        {
            "status": "success",
            "code": 200,
            "data": {
                "id": 123,
                "type": "user",
                "attributes": {
                    "name": "Alice",
                    "age": 30,
                    "roles": ["admin", "user"]
                },
                "relationships": {
                    "organization": {
                        "id": 456,
                        "name": "Acme Corp"
                    }
                }
            },
            "meta": {
                "request_id": "abc-123",
                "timestamp": datetime(2024, 1, 15, 10, 30)
            }
        },
        {
            "status": "success",
            "code": 200,
            "data": {
                "id": 124,
                "type": "user",
                "attributes": {
                    "name": "Bob",
                    "age": 25,
                    "roles": ["user"]
                },
                "relationships": {
                    "organization": {
                        "id": 789,
                        "name": "Tech Inc"
                    }
                }
            },
            "meta": {
                "request_id": "def-456",
                "timestamp": datetime(2024, 1, 15, 11, 00)
            }
        }
    ]
    
    Model = infer_pydantic_model(api_responses, model_name="APIResponse")
    instances = list(df_to_pydantic(api_responses, model=Model))
    
    assert len(instances) == 2
    assert instances[0].status == "success"
    # Nested dicts are represented differently - may be nested models or dicts
    data = instances[0].data
    
    # Access attributes - could be a dict or nested model
    if hasattr(data, 'attributes'):
        attrs = data.attributes
    else:
        attrs = data["attributes"]
    
    # Check the name - attrs could be a dict or model
    if hasattr(attrs, 'name'):
        assert attrs.name == "Alice"
        assert len(attrs.roles) == 2
    else:
        assert attrs["name"] == "Alice"
        assert len(attrs["roles"]) == 2
    
    # Check organization
    if hasattr(data, 'relationships'):
        rels = data.relationships
    else:
        rels = data["relationships"]
    
    if hasattr(rels, 'organization'):
        org = rels.organization
    else:
        org = rels["organization"]
    
    if hasattr(org, 'name'):
        assert org.name == "Acme Corp"
    else:
        assert org["name"] == "Acme Corp"


def test_sql_query_result_simulation():
    """Test SQL query result-like data."""
    query_results = [
        {
            "employee_id": 1,
            "first_name": "Alice",
            "last_name": "Johnson",
            "department": "Engineering",
            "salary": 95000.00,
            "hire_date": date(2020, 1, 15),
            "manager_id": None,  # Top-level manager
            "is_active": True
        },
        {
            "employee_id": 2,
            "first_name": "Bob",
            "last_name": "Smith",
            "department": "Engineering",
            "salary": 85000.00,
            "hire_date": date(2021, 3, 20),
            "manager_id": 1,
            "is_active": True
        },
        {
            "employee_id": 3,
            "first_name": "Charlie",
            "last_name": "Brown",
            "department": "Sales",
            "salary": 75000.00,
            "hire_date": date(2019, 6, 1),
            "manager_id": None,
            "is_active": False
        }
    ]
    
    Model = infer_pydantic_model(query_results, model_name="Employee")
    instances = list(df_to_pydantic(query_results, model=Model))
    
    assert len(instances) == 3
    assert instances[0].employee_id == 1
    assert instances[0].manager_id is None
    assert instances[1].manager_id == 1
    assert instances[2].is_active is False


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_csv_like_mixed_types():
    """Test CSV-like data with mixed types."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [30, 25, 35, 28, 32],
        "city": ["New York", "Los Angeles", "Chicago", None, "Boston"],
        "salary": [95000.50, 85000.00, 105000.75, 78000.00, 92000.25],
        "hire_date": pd.to_datetime([
            "2020-01-15", "2021-03-20", "2019-06-01", "2022-09-10", "2020-11-05"
        ]),
        "is_manager": [True, False, True, False, False]
    })
    
    Model = infer_pydantic_model(df, model_name="CSVData")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert len(instances) == 5
    assert instances[0].name == "Alice"
    assert instances[3].city is None
    assert instances[2].is_manager is True


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_large_dataset_performance():
    """Test with larger dataset (1000+ rows)."""
    # Create a larger dataset
    df = pl.DataFrame({
        "id": list(range(1000)),
        "name": [f"User{i}" for i in range(1000)],
        "value": [i * 1.5 for i in range(1000)],
        "category": [f"Cat{i % 10}" for i in range(1000)]
    })
    
    Model = infer_pydantic_model(df, model_name="LargeDataset")
    
    # Convert first 100 to instances
    instances = list(df_to_pydantic(df.head(100), model=Model))
    
    assert len(instances) == 100
    assert instances[0].id == 0
    assert instances[99].id == 99


def test_end_to_end_with_code_generation():
    """Test end-to-end: infer model, create instances, generate code."""
    data = [
        {
            "product_id": 1,
            "name": "Laptop",
            "price": 999.99,
            "specs": {
                "cpu": "Intel i7",
                "ram": 16,
                "storage": 512
            },
            "in_stock": True,
            "last_updated": datetime(2024, 1, 15)
        },
        {
            "product_id": 2,
            "name": "Mouse",
            "price": 29.99,
            "specs": {
                "cpu": None,
                "ram": None,
                "storage": None
            },
            "in_stock": True,
            "last_updated": datetime(2024, 1, 16)
        }
    ]
    
    # Infer model
    Model = infer_pydantic_model(data, model_name="Product")
    
    # Create instances
    instances = list(df_to_pydantic(data, model=Model))
    assert len(instances) == 2
    assert instances[0].name == "Laptop"
    
    # Generate code
    code = generate_class_code(Model)
    assert isinstance(code, str)
    assert len(code) > 0
    
    # Code should be valid Python
    compile(code, "<string>", "exec")


def test_streaming_data_generator():
    """Test with streaming/generator data source."""
    def data_stream():
        """Simulate streaming data."""
        for i in range(50):
            yield {
                "sensor_id": i % 5,
                "reading": 20.0 + i * 0.1,
                "timestamp": datetime(2024, 1, 15, 10, i % 60),
                "status": "ok" if i % 10 != 0 else "warning"
            }
    
    Model = infer_pydantic_model(data_stream(), model_name="SensorData", max_scan=10)
    
    # Use model with new stream
    instances = list(df_to_pydantic(data_stream(), model=Model, max_scan=20))
    
    assert len(instances) == 50
    assert instances[0].sensor_id == 0
    assert instances[10].status == "warning"


def test_multilevel_nested_structure():
    """Test deeply nested structure."""
    data = [
        {
            "id": 1,
            "level1": {
                "name": "Level 1",
                "level2": {
                    "name": "Level 2",
                    "level3": {
                        "name": "Level 3",
                        "value": 42
                    }
                }
            }
        }
    ]
    
    Model = infer_pydantic_model(data, model_name="DeepNested")
    instance = Model(**data[0])
    
    assert instance.id == 1
    # Check if nested models or dicts
    level1 = instance.level1
    if hasattr(level1, 'name'):
        assert level1.name == "Level 1"
        level2 = level1.level2
        if hasattr(level2, 'name'):
            assert level2.name == "Level 2"
            assert level2.level3.value == 42
        else:
            assert level2["name"] == "Level 2"
            assert level2["level3"]["value"] == 42
    else:
        # Dict access fallback
        assert level1["name"] == "Level 1"
        level2 = level1["level2"]
        if hasattr(level2, 'name'):
            assert level2.name == "Level 2"
        else:
            assert level2["name"] == "Level 2"
            assert level2["level3"]["value"] == 42


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_financial_transaction_data():
    """Test financial transaction-like data."""
    df = pd.DataFrame({
        "transaction_id": [1001, 1002, 1003, 1004],
        "account_id": [5001, 5002, 5001, 5003],
        "amount": [1500.00, -250.50, 3000.00, -100.00],
        "currency": ["USD", "USD", "EUR", "GBP"],
        "transaction_type": ["deposit", "withdrawal", "deposit", "withdrawal"],
        "timestamp": pd.to_datetime([
            "2024-01-15 10:30:00",
            "2024-01-15 11:45:00",
            "2024-01-15 14:20:00",
            "2024-01-15 16:00:00"
        ]),
        "status": ["completed", "completed", "pending", "completed"],
        "description": ["Salary", "ATM Withdrawal", "Transfer", None]
    })
    
    Model = infer_pydantic_model(df, model_name="Transaction")
    instances = list(df_to_pydantic(df, model=Model))
    
    assert len(instances) == 4
    assert instances[0].amount == 1500.00
    assert instances[1].amount < 0  # Withdrawal
    assert instances[2].status == "pending"
    assert instances[3].description is None


def test_sparse_data_with_many_optionals():
    """Test very sparse data where most fields are optional."""
    data = [
        {"id": 1, "a": 1, "b": None, "c": None, "d": None},
        {"id": 2, "a": None, "b": 2, "c": None, "d": None},
        {"id": 3, "a": None, "b": None, "c": 3, "d": None},
        {"id": 4, "a": None, "b": None, "c": None, "d": 4},
    ]
    
    Model = infer_pydantic_model(data, model_name="SparseData")
    instances = list(df_to_pydantic(data, model=Model))
    
    assert len(instances) == 4
    assert instances[0].a == 1
    assert instances[1].b == 2
    assert instances[2].c == 3
    assert instances[3].d == 4

