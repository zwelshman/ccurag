# AI Code Generation for R/dbplyr

Automated R code generation based on patterns from the [BHFDSC standard-pipeline](https://github.com/BHFDSC/standard-pipeline).

## Overview

This feature enables AI-powered code generation for R using dbplyr, providing equivalent functionality to the Python/PySpark standard-pipeline. It learns from your indexed codebase to generate context-aware suggestions.

## Features

### 1. Template Generation 📋

Generate boilerplate R/dbplyr code based on organizational patterns.

**Available Templates:**
- **Cohort Pipeline**: Complete pipeline for cohort construction
- **Table Loader**: Reusable function for loading and standardizing tables
- **Demographics Join**: Standard pattern for joining demographics data
- **Codelist Filter**: Apply medical codelist filtering (SNOMED, ICD-10)
- **Phenotyping Algorithm**: Template for clinical phenotyping
- **Data Quality Check**: Validation function for data quality
- **Aggregate with Privacy**: Privacy-preserving aggregation with disclosure control

**Example:**
```r
# Generated cohort pipeline template
library(dplyr)
library(dbplyr)
library(DBI)

# Connect to database
con <- DBI::dbConnect(odbc::odbc(), dsn = config$database$dsn)

# Load base cohort
cohort <- tbl(con, "hds_curated_assets__demographics") %>%
  filter(age >= 18 & !is.na(person_id)) %>%
  # Apply additional filtering
  left_join(tbl(con, "clinical_events"), by = "person_id")

# Save results
compute(cohort, name = "cohort_final", temporary = FALSE)
```

### 2. Code Completion 💡

Get context-aware suggestions for your R/dbplyr code as you write.

**How it works:**
1. Paste your partial code
2. System analyzes the context
3. Suggests likely next operations

**Example:**
```r
# Your code:
demographics <- tbl(con, "hds_curated_assets__demographics") %>%

# Suggestions:
# 1. filter({condition}) - Filter rows based on condition
# 2. select(person_id, age, sex) - Select specific columns
# 3. left_join({other_table}, by = "person_id") - Join with another table
```

### 3. Test Generation 🧪

Automatically create testthat test cases for R functions.

**Features:**
- Normal/expected inputs
- Edge cases (NA values, empty data, boundaries)
- Error cases (invalid inputs)
- Uses testthat framework

**Example:**
```r
# Original function:
calculate_age <- function(date_of_birth, reference_date = Sys.Date()) {
  age_years <- as.numeric(difftime(reference_date, date_of_birth, units = "days")) / 365.25
  floor(age_years)
}

# Generated tests:
test_that("calculate_age works correctly", {
  # Normal case
  dob <- as.Date("1980-01-01")
  ref <- as.Date("2024-01-01")
  expect_equal(calculate_age(dob, ref), 44)

  # Edge case: leap year
  dob <- as.Date("2000-02-29")
  ref <- as.Date("2024-02-28")
  expect_equal(calculate_age(dob, ref), 23)

  # Error case: future date of birth
  expect_error(calculate_age(Sys.Date() + 365))
})
```

### 4. Documentation Generation 📝

Generate roxygen2 documentation for R functions and README files.

**Supports:**
- Function documentation (roxygen2 format)
- README files (markdown)
- File-level documentation comments

**Example:**
```r
#' Load and Filter Clinical Events
#'
#' Retrieves clinical events from the database for specified persons, with
#' optional filtering by event type and date range.
#'
#' @param con Database connection object
#' @param person_ids Vector of person IDs to filter
#' @param event_type Optional event type filter
#' @param start_date Optional start date (inclusive)
#' @param end_date Optional end date (inclusive)
#'
#' @return A lazy tibble containing filtered clinical events
#'
#' @examples
#' \dontrun{
#'   events <- load_and_filter_events(
#'     con,
#'     person_ids = c(1, 2, 3),
#'     event_type = "diagnosis",
#'     start_date = as.Date("2020-01-01")
#'   )
#' }
#'
#' @export
load_and_filter_events <- function(con, person_ids, event_type,
                                   start_date = NULL, end_date = NULL) {
  # Function implementation...
}
```

### 5. Refactoring Suggestions 🔧

Get suggestions to improve your R/dbplyr code.

**Focus Areas:**
- **Performance**: Optimize for large datasets (avoid collect(), use compute())
- **Readability**: Improve code clarity (use across(), consistent naming)
- **dbplyr Best Practices**: Leverage lazy evaluation, proper joins

**Example:**
```r
# Original (problematic):
data <- tbl(con, "large_table") %>%
  filter(year == 2023) %>%
  collect()  # Loads everything into memory!

result <- data %>%
  group_by(age_group) %>%
  summarise(count = n())

# Suggested refactoring:
result <- tbl(con, "large_table") %>%
  filter(year == 2023) %>%
  group_by(age_group) %>%
  summarise(count = n()) %>%
  compute(name = "temp_summary", temporary = TRUE)  # Keep in database
```

### 6. Python to R Translation 🔄

Convert Python/PySpark code from standard-pipeline to R/dbplyr equivalent.

**Useful for:**
- Migrating from PySpark to R/dbplyr
- Learning R equivalents of Python patterns
- Adapting standard-pipeline code to R

**Example:**
```python
# Python/PySpark:
def load_table(spark, table_name, archive_date=None):
    df = spark.table(table_name)
    if archive_date is not None:
        df = df.filter(f.col('archive_date') == archive_date)
    return df
```

Translates to:

```r
# R/dbplyr:
load_table <- function(con, table_name, archive_date = NULL) {
  tbl_data <- tbl(con, table_name)

  if (!is.null(archive_date)) {
    tbl_data <- tbl_data %>%
      filter(archive_date == !!archive_date)
  }

  return(tbl_data)
}
```

## Usage

### Via Streamlit UI

1. Navigate to **AI Code Gen** page
2. Select the feature tab (Templates, Completion, Tests, etc.)
3. Enter your parameters or code
4. Click the generate button
5. Download or copy the generated code

### Programmatic Usage

```python
from r_code_generator import RCodeGenerator
from hybrid_retriever import HybridRetriever
from code_analyzer import CodeAnalyzer

# Initialize with RAG components (optional but recommended)
analyzer = CodeAnalyzer()
retriever = HybridRetriever(vector_store)
generator = RCodeGenerator(
    hybrid_retriever=retriever,
    code_analyzer=analyzer
)

# Generate a template
suggestion = generator.generate_template(
    template_type="cohort_pipeline",
    params={
        "pipeline_name": "MI Study Cohort",
        "purpose": "Identify MI patients",
        "base_table": "hds_curated_assets__demographics",
        "initial_filters": "age >= 18",
        "pipeline_steps": "filter(condition) %>% select(columns)",
        "output_table": "cohort_mi"
    }
)

print(suggestion.code)

# Get code completions
completions = generator.suggest_completion(
    code_context="""
    demographics <- tbl(con, "hds_curated_assets__demographics") %>%
    """
)

for comp in completions:
    print(f"{comp.explanation}: {comp.code}")

# Generate tests
test_suggestion = generator.generate_tests(
    function_code="calculate_age <- function(dob, ref) { ... }",
    function_name="calculate_age"
)

# Translate Python to R
translation = generator.translate_python_to_r(
    python_code="def load_table(spark, name): ..."
)
```

## Learning from Codebase

The generator learns R patterns from your indexed repositories:

```python
# Analyze codebase to learn patterns
stats = generator.learn_from_codebase()

print(f"R files analyzed: {stats['r_files_analyzed']}")
print(f"Top R functions: {stats['top_functions']}")
print(f"Top R packages: {stats['top_imports']}")
```

This enables:
- Custom template generation based on your organization's patterns
- Context-aware completions using actual code examples
- Refactoring suggestions aligned with your coding standards

## Configuration

### Requirements

- **ANTHROPIC_API_KEY**: Required for AI-powered features (set in `.env`)
- **Indexed codebase**: Optional but enhances suggestions (run `build_metadata_index.py`)
- **BM25 index**: Optional for better RAG retrieval (build in Q&A tab)

### Template vs. AI Mode

**Template Mode** (no API key):
- Uses pre-defined templates
- Pattern-based completions
- Basic refactoring rules
- No translation capability

**AI Mode** (with API key):
- Custom template generation
- Context-aware completions
- Intelligent refactoring
- Python-to-R translation
- Documentation generation

## Standard-Pipeline Patterns

Based on BHFDSC standard-pipeline, the generator understands these patterns:

### 1. Modular Pipeline Structure
Sequential processing steps (D01-D11 pattern) with clear separation of concerns.

### 2. Codelist-Driven Filtering
Using CSV codelists for clinical concept definitions (SNOMED-CT, ICD-10, BNF codes).

### 3. Privacy-by-Design
Disclosure control functions, count rounding, suppression of small numbers.

### 4. Configuration Separation
Parameters and environment settings isolated from analysis logic.

### 5. Reusable Functions
Common operations abstracted into domain-specific modules.

## Examples

### Example 1: Generate MI Cohort Pipeline

```python
suggestion = generator.generate_template(
    "cohort_pipeline",
    params={
        "pipeline_name": "Myocardial Infarction Study",
        "purpose": "Identify MI patients and extract covariates",
        "base_table": "hds_curated_assets__demographics",
        "initial_filters": "age >= 18 & age <= 100",
        "pipeline_steps": """
  # Load MI diagnosis codes
  inner_join(tbl(con, "hes_apc_diagnosis"), by = "person_id") %>%
  filter(diagnosis_code %in% mi_codes) %>%

  # Add comorbidities
  left_join(comorbidities, by = "person_id") %>%

  # Calculate follow-up
  mutate(follow_up_days = as.numeric(end_date - index_date))
""",
        "output_table": "cohort_mi_final"
    }
)
```

### Example 2: Generate Phenotyping Algorithm

```python
suggestion = generator.generate_template(
    "phenotyping_algorithm",
    params={
        "phenotype_name": "Chronic Kidney Disease",
        "phenotype_id": "ckd",
        "condition_description": "chronic kidney disease stages 3-5 based on eGFR measurements"
    }
)
```

### Example 3: Translate Python Function

```python
python_code = """
def apply_codelist(df, codelist_path, code_col='code'):
    codes = pd.read_csv(codelist_path)
    return df.join(codes, df[code_col] == codes['code'], 'inner')
"""

suggestion = generator.translate_python_to_r(python_code)
```

## Best Practices

1. **Review Generated Code**: Always review AI-generated code before use
2. **Test Thoroughly**: Run generated tests and add custom cases
3. **Customize Templates**: Modify templates to match your organization's standards
4. **Learn from Codebase**: Index your repositories to improve suggestions
5. **Provide Context**: More context = better completions and suggestions
6. **Iterate**: Use refactoring suggestions to continuously improve code

## Limitations

- AI-generated code may need manual adjustments
- Translation quality depends on code complexity
- Custom patterns require indexed codebase
- Template parameters must be provided correctly

## Troubleshooting

### "AI client not available"
- Set `ANTHROPIC_API_KEY` in your `.env` file
- Restart the application after adding the key

### "No suggestions found"
- Provide more code context
- Build BM25 index for better RAG retrieval
- Index more R repositories

### "Template type not found"
- Use one of the standard template types
- Or provide custom description with RAG enabled

## Future Enhancements

- **More Templates**: Add templates for specific health data patterns
- **Custom Pattern Learning**: Learn templates from your own codebase
- **Multi-file Generation**: Generate entire R packages
- **Interactive Refinement**: Iteratively improve generated code
- **Integration with RStudio**: IDE plugin for in-editor generation

## Related

- [BHFDSC standard-pipeline](https://github.com/BHFDSC/standard-pipeline)
- [FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md) - See "AI-Powered Code Generation" section
- [example_r_code_generation.py](example_r_code_generation.py) - Programmatic examples

## Support

For issues or questions about R code generation:
- Check this documentation
- Run `python example_r_code_generation.py` for examples
- Open an issue on GitHub
