# 📓 Articuno Examples

This directory contains comprehensive Jupyter notebooks demonstrating all articuno features.

## 🚀 Core Examples

Learn the basics and get started quickly:

### [01_quick_start.ipynb](01_quick_start.ipynb)
**Basic usage with all backends**
- Polars DataFrame to Pydantic
- Pandas DataFrame to Pydantic
- Dict iterables to Pydantic
- Generator behavior
- Quick introduction to the main API

### [02_temporal_types.ipynb](02_temporal_types.ipynb)
**Working with dates and times**
- `datetime.datetime` support
- `datetime.date` support
- `datetime.timedelta` support
- Temporal types across all backends (Polars, Pandas, dicts)
- Time-series data processing

### [03_pyarrow_support.ipynb](03_pyarrow_support.ipynb)
**High-performance PyArrow-backed Pandas**
- `int64[pyarrow]`, `string[pyarrow]`, `bool[pyarrow]`
- `timestamp[pyarrow]` → datetime
- `date[pyarrow]` → date
- `duration[pyarrow]` → timedelta
- Performance benefits of PyArrow

### [04_nested_structures.ipynb](04_nested_structures.ipynb)
**Complex nested data**
- Nested dictionaries
- Multi-level nesting
- Lists of objects
- Pandas with nested dict columns
- Deep hierarchical structures

---

## 🎯 Advanced Examples

Master advanced features and real-world use cases:

### [05_force_optional_and_max_scan.ipynb](05_force_optional_and_max_scan.ipynb)
**Performance and flexibility**
- `force_optional=True` for sparse data
- `max_scan` parameter for large datasets
- Combining options for optimal performance
- Memory-efficient schema inference

### [06_code_generation.ipynb](06_code_generation.ipynb)
**Generate production-ready code**
- Simple model code generation
- Nested model code generation
- Saving to files
- Custom model name override
- Exporting for other projects

### [07_api_responses.ipynb](07_api_responses.ipynb)
**REST API integration**
- Simple API responses
- Paginated responses
- Error response handling
- JSON:API format
- Real API integration patterns

### [08_database_results.ipynb](08_database_results.ipynb)
**SQL query results**
- Converting database rows to Pydantic
- JOIN query results
- Aggregate query results (GROUP BY)
- NULL handling from databases
- Shopping cart and e-commerce data

### [09_advanced_features.ipynb](09_advanced_features.ipynb)
**Power user features**
- Generator-based processing for large datasets
- Mixed data sources with same model
- Sparse data with automatic optional detection
- Type precedence (bool vs int)
- Unicode and international character support

---

## 📚 Legacy Examples

Original notebooks from testing (still useful for learning):

- **[polars_inference.ipynb](polars_inference.ipynb)** - Polars-specific inference demonstrations
- **[articuno_pandas_pyarrow_example.ipynb](articuno_pandas_pyarrow_example.ipynb)** - Pandas with PyArrow types
- **[pandas_nested.ipynb](pandas_nested.ipynb)** - Nested structures in Pandas
- **[optional_nested_example.ipynb](optional_nested_example.ipynb)** - Optional nested fields
- **[articuno_inference_demo.ipynb](articuno_inference_demo.ipynb)** - General inference demonstrations

---

## 🎓 Learning Path

**If you're new to articuno:**
1. Start with `01_quick_start.ipynb`
2. Then explore `02_temporal_types.ipynb` for datetime support
3. Check `04_nested_structures.ipynb` for complex data
4. Review `07_api_responses.ipynb` or `08_database_results.ipynb` for your use case

**For advanced users:**
- Jump to `09_advanced_features.ipynb` for power features
- See `06_code_generation.ipynb` for production deployment
- Explore `05_force_optional_and_max_scan.ipynb` for performance tuning

---

## 🏃 Running the Examples

All notebooks can be run with:

```bash
jupyter notebook examples/
```

Or individually:

```bash
jupyter notebook examples/01_quick_start.ipynb
```

---

## 📋 Requirements

Install articuno with all dependencies:

```bash
pip install articuno[polars,pandas]
pip install pyarrow  # For PyArrow examples
pip install jupyter  # To run notebooks
```

---

## 💡 Tips

- All examples are self-contained and can run independently
- Examples use realistic data scenarios
- Code snippets can be copied directly into your projects
- Each notebook includes explanations and best practices

---

## 🤝 Contributing

Found an issue or want to add an example? Contributions welcome!

