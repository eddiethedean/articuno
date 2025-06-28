import re
from pydantic import BaseModel, create_model
import pytest
from articuno.codegen import generate_pydantic_class_code


def test_generate_static_model():
    class User(BaseModel):
        id: int
        name: str

    code = generate_pydantic_class_code(User)

    assert "class User(BaseModel):" in code
    assert "id: int" in code
    assert "name: str" in code


def test_generate_dynamic_model():
    DynamicModel = create_model('DynamicModel', age=(int, 0), active=(bool, ...))

    code = generate_pydantic_class_code(DynamicModel)

    assert "class DynamicModel(BaseModel):" in code
    assert re.search(r"age: int\s*=\s*0", code)
    assert re.search(r"active: bool", code)


def test_generate_with_custom_model_name():
    class Person(BaseModel):
        first_name: str
        last_name: str

    code = generate_pydantic_class_code(Person, model_name="CustomPerson")

    assert "class CustomPerson(BaseModel):" in code
    assert "first_name: str" in code
    assert "last_name: str" in code


def test_output_file_written(tmp_path):
    class Item(BaseModel):
        sku: str
        quantity: int

    output_file = tmp_path / "item_model.py"
    code = generate_pydantic_class_code(Item, output_path=output_file)

    assert output_file.exists()
    content = output_file.read_text()
    assert content == code
    assert "class Item(BaseModel):" in content
    assert "sku: str" in content
    assert "quantity: int" in content
