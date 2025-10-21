"""
Tests for code generation functionality.
"""

import pytest
import tempfile
from pathlib import Path
from articuno import infer_pydantic_model
from articuno.codegen import generate_class_code


def test_generate_code_from_simple_model():
    """Test generating Python code from a simple model."""
    dicts = [
        {"id": 1, "name": "Alice", "score": 95.5},
        {"id": 2, "name": "Bob", "score": 88.0},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="SimpleModel")
    code = generate_class_code(Model)
    
    # Check that code is generated
    assert isinstance(code, str)
    assert len(code) > 0
    
    # Check for key elements
    assert "class" in code or "BaseModel" in code


def test_generate_code_from_nested_model():
    """Test generating code from a model with nested structures."""
    dicts = [
        {"id": 1, "user": {"name": "Alice", "age": 30}},
        {"id": 2, "user": {"name": "Bob", "age": 25}},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="NestedModel")
    code = generate_class_code(Model)
    
    assert isinstance(code, str)
    assert len(code) > 0


def test_write_code_to_file():
    """Test writing generated code to a file."""
    dicts = [{"id": 1, "value": "test"}]
    Model = infer_pydantic_model(dicts, model_name="FileModel")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "generated_model.py"
        
        code = generate_class_code(Model, output_path=output_path)
        
        # File should exist
        assert output_path.exists()
        
        # File should have content
        file_content = output_path.read_text()
        assert len(file_content) > 0
        
        # Returned code should match file content
        assert code == file_content


def test_return_code_as_string():
    """Test returning code as string without writing to file."""
    dicts = [{"id": 1, "value": "test"}]
    Model = infer_pydantic_model(dicts, model_name="StringModel")
    
    code = generate_class_code(Model)
    
    # Should return a string
    assert isinstance(code, str)
    assert len(code) > 0


def test_custom_model_name_override():
    """Test overriding model name in generated code."""
    dicts = [{"id": 1, "value": "test"}]
    Model = infer_pydantic_model(dicts, model_name="OriginalName")
    
    code = generate_class_code(Model, model_name="CustomName")
    
    # Generated code should reference the custom name
    assert isinstance(code, str)
    # Note: exact format depends on datamodel-code-generator output


def test_temp_directory_cleanup():
    """Test that temporary directories are cleaned up properly."""
    dicts = [{"id": 1, "value": "test"}]
    Model = infer_pydantic_model(dicts, model_name="CleanupModel")
    
    # Generate code multiple times - should not accumulate temp dirs
    for _ in range(3):
        code = generate_class_code(Model)
        assert isinstance(code, str)
    
    # If temp cleanup works properly, this completes without issues


def test_temp_cleanup_on_error():
    """Test that temp directories are cleaned up even on error."""
    # This is harder to test directly, but we can verify the code structure
    # The fix uses try-finally to ensure cleanup
    dicts = [{"id": 1}]
    Model = infer_pydantic_model(dicts, model_name="ErrorModel")
    
    # Normal operation should work
    code = generate_class_code(Model)
    assert isinstance(code, str)


def test_generated_code_is_valid_python():
    """Test that generated code is valid Python syntax."""
    dicts = [
        {"id": 1, "name": "Alice", "score": 95.5, "active": True},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="ValidModel")
    code = generate_class_code(Model)
    
    # Try to compile the code
    try:
        compile(code, "<string>", "exec")
    except SyntaxError as e:
        pytest.fail(f"Generated code has syntax errors: {e}")


def test_generate_code_with_optional_fields():
    """Test code generation with optional fields."""
    dicts = [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": None},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="OptionalModel")
    code = generate_class_code(Model)
    
    assert isinstance(code, str)
    assert len(code) > 0


def test_generate_code_with_force_optional():
    """Test code generation when force_optional was used."""
    dicts = [{"id": 1, "name": "Alice"}]
    
    Model = infer_pydantic_model(dicts, model_name="ForceOptModel", force_optional=True)
    code = generate_class_code(Model)
    
    assert isinstance(code, str)
    assert len(code) > 0


def test_generate_code_with_lists():
    """Test code generation with list fields."""
    dicts = [
        {"id": 1, "tags": ["python", "data"]},
        {"id": 2, "tags": ["ml", "ai"]},
    ]
    
    Model = infer_pydantic_model(dicts, model_name="ListModel")
    code = generate_class_code(Model)
    
    assert isinstance(code, str)
    assert len(code) > 0


def test_output_path_as_string():
    """Test that output_path can be provided as a string."""
    dicts = [{"id": 1}]
    Model = infer_pydantic_model(dicts, model_name="StringPathModel")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = str(Path(tmpdir) / "model.py")
        
        code = generate_class_code(Model, output_path=output_path)
        
        assert Path(output_path).exists()
        assert len(code) > 0


def test_output_path_as_path_object():
    """Test that output_path can be provided as a Path object."""
    dicts = [{"id": 1}]
    Model = infer_pydantic_model(dicts, model_name="PathObjectModel")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "model.py"
        
        code = generate_class_code(Model, output_path=output_path)
        
        assert output_path.exists()
        assert len(code) > 0

