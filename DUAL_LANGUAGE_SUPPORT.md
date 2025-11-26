# Dual Language Support: Python/PySpark & R/dbplyr

The AI Code Generator now supports **both Python/PySpark and R/dbplyr** code generation based on actual patterns from the BHFDSC standard-pipeline repository.

## Overview

This enhancement provides authentic standard-pipeline templates for both:
- **Python/PySpark**: Original patterns from the standard-pipeline
- **R/dbplyr**: Equivalent R implementations

## Python Templates (PYTHON_TEMPLATES)

Based on actual code from https://github.com/BHFDSC/standard-pipeline:

### 1. **table_loader**
- Source: `functions/table_management.py`
- Loads and standardizes tables with archive filtering
- Includes person_id standardization and column name cleaning

### 2. **cohort_pipeline**
- Source: Databricks notebooks (D04-D11 pattern)
- Complete pipeline with notebook structure
- Includes markdown cells, config loading, and parameter injection

### 3. **inclusion_criteria**
- Source: `functions/cohort_construction.py`
- Applies inclusion criteria with flowchart tracking
- Uses the `apply_inclusion_criteria` function pattern

### 4. **phenotyping_algorithm**
- Pattern from phenotyping notebooks (D05, D06, D09b)
- Codelist-based event identification
- Temporal filtering and aggregation

### 5. **privacy_control**
- Source: `functions/data_privacy.py`
- Count rounding and low-count suppression
- Implements disclosure control standards

### 6. **data_quality_check**
- Comprehensive quality validation
- Checks for missing columns, nulls, duplicates
- Returns structured results dictionary

### 7. **test_function**
- pytest-based test template
- Includes schema definition and test data
- Covers normal cases and edge cases

## R Templates (R_TEMPLATES)

Equivalent R/dbplyr implementations of Python patterns:

### Maintained Templates
All existing R templates remain available:
- `cohort_pipeline`: R version with dbplyr operations
- `table_loader`: R function with roxygen2 docs
- `demographics_join`: dplyr join pattern
- `codelist_filter`: Inner join with codelists
- `phenotyping_algorithm`: R phenotyping function
- `data_quality_check`: R quality checks
- `aggregate_with_privacy`: Privacy-preserving aggregation
- `test_function`: testthat test template

## Usage

### Programmatic

```python
from r_code_generator import RCodeGenerator

generator = RCodeGenerator()

# Generate Python code
py_code = generator.generate_template(
    "table_loader",
    {},
    language="python"
)

# Generate R code
r_code = generator.generate_template(
    "table_loader",
    {},
    language="r"
)
```

### UI

1. Navigate to **AI Code Gen** tab in Streamlit
2. Select **Python/PySpark** or **R/dbplyr** using the radio button
3. Choose a template type (automatically filtered by language)
4. Fill in parameters
5. Generate code with appropriate syntax highlighting

## Template Availability

### Python-Only Templates
- `inclusion_criteria`: Uses cohort_construction.py patterns
- `privacy_control`: From data_privacy.py

### R-Only Templates
- `demographics_join`: Simple join pattern
- `codelist_filter`: Codelist filtering
- `aggregate_with_privacy`: Privacy aggregation

### Available in Both
- `cohort_pipeline`
- `table_loader`
- `phenotyping_algorithm`
- `data_quality_check`
- `test_function`

## Key Features

### 1. Authentic Patterns
Python templates are extracted directly from standard-pipeline source code:
- Actual function signatures
- Real PySpark operations
- Databricks notebook structure
- Standard-pipeline conventions

### 2. Language-Aware UI
- Dynamic template list based on selected language
- Appropriate syntax highlighting (python vs r)
- Correct file extensions (.py vs .R)
- Language-specific descriptions

### 3. Equivalent Functionality
R templates provide equivalent functionality using:
- dbplyr instead of PySpark
- dplyr verbs instead of DataFrame operations
- R idioms (piping with `%>%`, `!!` for unquoting)
- roxygen2 instead of docstrings

## Standard-Pipeline Alignment

### Python Templates Match:

**Notebook Structure** (D04-D11):
```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Title

# COMMAND ----------
# MAGIC %run ./project_config
# MAGIC %run ./parameters

# COMMAND ----------
from pyspark.sql import functions as f
from functions.table_management import load_table
```

**Function Patterns** (functions/*.py):
```python
def load_table(spark, table_name, archive_date=None, standardize=True):
    '''Docstring following standard format'''
    df = spark.table(table_name)
    if archive_date is not None:
        df = df.filter(f.col('archive_date') == archive_date)
    return df
```

**Privacy Control** (data_privacy.py):
```python
def round_counts_to_multiple(df, columns, multiple=5):
    '''Round to nearest multiple for disclosure control'''
    for col in columns:
        df = df.withColumn(col, f.round(f.col(col) / multiple) * multiple)
    return df
```

## Testing

Both languages tested and verified:
```bash
$ python -c "from r_code_generator import RCodeGenerator; ..."
✓ Python templates work!
✓ R templates work!
```

## Future Enhancements

1. **More Templates**: Add D07-D10 patterns (geographic, measurements, outcomes)
2. **Notebook Conversion**: Full .ipynb to .Rmd conversion
3. **Codelist Management**: Templates for codelist creation
4. **Config Templates**: Parameter and project_config equivalents

## References

- Standard-Pipeline: https://github.com/BHFDSC/standard-pipeline
- table_management.py: Table loading and standardization
- cohort_construction.py: Inclusion criteria and flowcharts
- data_privacy.py: Disclosure control functions
- D04-D11 notebooks: Sequential pipeline steps
