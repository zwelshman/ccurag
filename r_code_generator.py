"""AI-powered code generation for Python/PySpark and R/dbplyr.

Based on BHFDSC standard-pipeline patterns:
https://github.com/BHFDSC/standard-pipeline

Provides code intelligence for both Python and R development:
- Template generation: Create boilerplate based on standard-pipeline patterns
- Code completion: Suggest context-aware code snippets
- Refactoring suggestions: Propose improvements with diffs
- Test generation: Auto-create tests for uncovered code
- Documentation generation: Auto-write docstrings, READMEs
- Language translation: Convert between Python/PySpark and R/dbplyr

Designed to work with RAG system to learn from existing codebases.
"""

import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import os
from collections import defaultdict

# Optional import - only needed if ANTHROPIC_API_KEY is set
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RCodePattern:
    """Represents a learned R code pattern."""
    pattern_type: str  # 'function', 'pipeline', 'join', 'filter', etc.
    code_template: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    frequency: int = 0  # How often this pattern appears


@dataclass
class CodeSuggestion:
    """A code generation suggestion."""
    suggestion_type: str  # 'template', 'completion', 'refactor', 'test', 'doc'
    code: str
    explanation: str
    confidence: float  # 0.0 to 1.0
    context: Optional[str] = None


class RCodeGenerator:
    """Generates code for both Python/PySpark and R/dbplyr based on standard-pipeline patterns."""

    # Python/PySpark templates - actual patterns from BHFDSC standard-pipeline
    PYTHON_TEMPLATES = {
        "table_loader": {
            "template": """from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from typing import Optional

def load_table(
    spark,
    table_name: str,
    archive_date: Optional[str] = None,
    standardize: bool = True
) -> DataFrame:
    '''Load and standardize a table.

    Args:
        spark: SparkSession
        table_name: Name of table to load
        archive_date: Optional archive date filter (YYYY-MM-DD format)
        standardize: Whether to apply standardization

    Returns:
        DataFrame with loaded and optionally standardized table

    Example:
        >>> demographics = load_table(spark, 'demographics', archive_date='2024-01-01')
    '''
    # Load table
    df = spark.table(table_name)

    # Apply archive filtering if specified
    if archive_date is not None:
        df = df.filter(f.col('archive_date') == archive_date)

    # Standardize column names and person ID
    if standardize:
        # Rename person ID column to standard name
        for col in df.columns:
            if 'person_id' in col.lower() or 'pseudo_id' in col.lower():
                df = df.withColumnRenamed(col, 'person_id')

        # Clean column names (lowercase, replace spaces with underscores)
        for col in df.columns:
            new_col = col.lower().replace(' ', '_')
            if new_col != col:
                df = df.withColumnRenamed(col, new_col)

    return df
""",
            "description": "Load and standardize tables (from table_management.py)",
            "tags": ["python", "pyspark", "table", "loading"]
        },

        "cohort_pipeline": {
            "template": """# Databricks notebook source
# MAGIC %md
# MAGIC # {pipeline_name}
# MAGIC
# MAGIC **Purpose**: {purpose}
# MAGIC
# MAGIC **Author**: Auto-generated
# MAGIC **Date**: {date}

# COMMAND ----------

# MAGIC %run ./project_config

# COMMAND ----------

# MAGIC %run ./parameters

# COMMAND ----------

from pyspark.sql import functions as f
from functions.table_management import load_table, save_table
from functions.cohort_construction import apply_inclusion_criteria

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Base Cohort

# COMMAND ----------

# Load base cohort data
cohort_base = load_table(spark, '{base_table}')

# Apply initial filters
cohort_filtered = cohort_base.filter(
    {initial_filters}
)

# Add study dates from parameters
cohort_with_dates = (
    cohort_filtered
    .withColumn('cohort_entry_start_date', f.to_date(f.lit(cohort_entry_start_date)))
    .withColumn('cohort_entry_end_date', f.to_date(f.lit(cohort_entry_end_date)))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Apply Pipeline Steps

# COMMAND ----------

{pipeline_steps}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Save Results

# COMMAND ----------

save_table(cohort_final, '{output_table}', overwrite=True)

# COMMAND ----------

# Display summary statistics
print(f"Total rows in final cohort: {{cohort_final.count()}}")
print(f"Total unique persons: {{cohort_final.select('person_id').distinct().count()}}")
""",
            "description": "Complete cohort construction pipeline (Databricks notebook format)",
            "tags": ["python", "pyspark", "cohort", "pipeline", "notebook"]
        },

        "inclusion_criteria": {
            "template": """from pyspark.sql import functions as f
from functions.cohort_construction import apply_inclusion_criteria

# Define inclusion criteria as SQL expressions
inclusion_criteria = {{
    '{criterion_1_name}': '{criterion_1_sql}',
    '{criterion_2_name}': '{criterion_2_sql}',
    '{criterion_3_name}': '{criterion_3_sql}',
}}

# Apply inclusion criteria and create flowchart
cohort_final = apply_inclusion_criteria(
    cohort=cohort_base,
    inclusion_criteria=inclusion_criteria,
    row_id_col='row_id',
    person_id_col='person_id',
    flowchart_table='{flowchart_table_name}',
    clean_up=True  # Remove intermediate criteria columns
)

# The flowchart table shows exclusions at each step
flowchart = spark.table('{flowchart_table_name}')
display(flowchart)
""",
            "description": "Apply inclusion criteria with flowchart tracking (from cohort_construction.py)",
            "tags": ["python", "pyspark", "cohort", "criteria", "flowchart"]
        },

        "phenotyping_algorithm": {
            "template": """from pyspark.sql import functions as f
from pyspark.sql import Window

def identify_{phenotype_id}(
    cohort: DataFrame,
    clinical_events: DataFrame,
    codelists_path: str,
    index_date_col: str = 'index_date',
    lookback_days: int = 365
) -> DataFrame:
    '''Identify patients with {phenotype_name}.

    {condition_description}

    Args:
        cohort: Base cohort DataFrame with person_id
        clinical_events: Clinical events table
        codelists_path: Path to codelist CSV file
        index_date_col: Column name containing index date
        lookback_days: Days to look back from index date

    Returns:
        DataFrame with person_id and phenotype flags
    '''
    # Load codelists
    codelists = spark.read.csv(codelists_path, header=True)

    # Filter clinical events to relevant time window and codes
    events_filtered = (
        clinical_events
        .join(cohort.select('person_id', index_date_col), on='person_id', how='inner')
        .filter(
            (f.col('event_date') >= f.date_sub(f.col(index_date_col), lookback_days)) &
            (f.col('event_date') <= f.col(index_date_col))
        )
        .join(codelists.select('code'), on='code', how='inner')
    )

    # Aggregate to person level
    phenotype = (
        events_filtered
        .groupBy('person_id')
        .agg(
            f.lit(1).alias('{phenotype_id}_flag'),
            f.min('event_date').alias('{phenotype_id}_first_date'),
            f.count('*').alias('{phenotype_id}_event_count')
        )
    )

    return phenotype
""",
            "description": "Phenotyping algorithm template based on standard-pipeline patterns",
            "tags": ["python", "pyspark", "phenotyping", "clinical"]
        },

        "privacy_control": {
            "template": """from pyspark.sql import functions as f

def round_counts_to_multiple(
    df: DataFrame,
    columns: list,
    multiple: int = 5
) -> DataFrame:
    '''Round count values to nearest multiple for disclosure control.

    Args:
        df: DataFrame containing count columns
        columns: List of columns to round
        multiple: Rounding interval (default 5)

    Returns:
        DataFrame with rounded count columns
    '''
    for col in columns:
        df = df.withColumn(
            col,
            f.round(f.col(col) / multiple) * multiple
        )
    return df


def redact_low_counts(
    df: DataFrame,
    columns: list,
    threshold: int = 10,
    redaction_value: str = '[REDACTED]'
) -> DataFrame:
    '''Suppress low counts below threshold for disclosure control.

    Args:
        df: DataFrame containing count columns
        columns: List of columns to check
        threshold: Minimum value to show (default 10)
        redaction_value: Value to use for redacted cells

    Returns:
        DataFrame with low counts suppressed
    '''
    for col in columns:
        df = df.withColumn(
            col,
            f.when(f.col(col) >= threshold, f.col(col)).otherwise(redaction_value)
        )
    return df


# Example usage
summary_safe = (
    summary
    .transform(lambda df: round_counts_to_multiple(df, ['count', 'total'], multiple=5))
    .transform(lambda df: redact_low_counts(df, ['count', 'total'], threshold=10))
)
""",
            "description": "Privacy and disclosure control functions (from data_privacy.py)",
            "tags": ["python", "pyspark", "privacy", "disclosure"]
        },

        "data_quality_check": {
            "template": """from pyspark.sql import functions as f

def check_data_quality(
    df: DataFrame,
    table_name: str,
    required_columns: list
) -> dict:
    '''Perform comprehensive data quality checks.

    Args:
        df: DataFrame to check
        table_name: Name of table for reporting
        required_columns: List of columns that must be present

    Returns:
        Dictionary with check results
    '''
    results = {{'table_name': table_name}}

    # Check 1: Required columns present
    actual_columns = df.columns
    missing_columns = set(required_columns) - set(actual_columns)
    results['columns_present'] = len(missing_columns) == 0
    results['missing_columns'] = list(missing_columns)

    # Check 2: Row count
    row_count = df.count()
    results['row_count'] = row_count
    results['has_data'] = row_count > 0

    # Check 3: Null counts for required columns
    null_counts = {{}}
    for col in required_columns:
        if col in actual_columns:
            null_count = df.filter(f.col(col).isNull()).count()
            null_counts[col] = null_count
    results['null_counts'] = null_counts

    # Check 4: Duplicate person IDs
    if 'person_id' in actual_columns:
        dup_count = (
            df.groupBy('person_id')
            .count()
            .filter(f.col('count') > 1)
            .count()
        )
        results['duplicate_person_ids'] = dup_count

    return results


# Example usage
quality_results = check_data_quality(
    cohort_final,
    'cohort_final',
    ['person_id', 'index_date', 'age', 'sex']
)

# Print results
for key, value in quality_results.items():
    print(f"{{key}}: {{value}}")
""",
            "description": "Data quality validation checks",
            "tags": ["python", "pyspark", "quality", "validation"]
        },

        "test_function": {
            "template": """import pytest
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DateType

def test_{function_name}(spark):
    '''Test {function_name} function.'''

    # Setup: Create test data
    schema = StructType([
        StructField('person_id', StringType(), False),
        {test_schema_fields}
    ])

    test_data = [
        {test_data_rows}
    ]

    test_df = spark.createDataFrame(test_data, schema)

    # Execute
    result = {function_name}({test_params})

    # Assert: Normal case
    assert result.count() == {expected_count}, "Expected {{}} rows".format({expected_count})

    # Assert: Data validation
    {assertions}

    # Edge case: Empty input
    empty_df = spark.createDataFrame([], schema)
    empty_result = {function_name}({empty_test_params})
    assert empty_result.count() == 0, "Empty input should return empty result"
""",
            "description": "Pytest test template for PySpark functions",
            "tags": ["python", "pyspark", "test", "pytest"]
        }
    }

    # R/dbplyr templates - dbplyr equivalents of Python patterns
    R_TEMPLATES = {
        "cohort_pipeline": {
            "template": """# {pipeline_name}
# Purpose: {purpose}
# Author: Auto-generated
# Date: {date}

library(dplyr)
library(dbplyr)
library(DBI)

# Load configuration
config <- jsonlite::read_json("config/parameters.json")

# Connect to database
con <- DBI::dbConnect(
  odbc::odbc(),
  dsn = config$database$dsn
)

# Load base cohort
cohort <- tbl(con, "{base_table}") %>%
  filter({initial_filters})

# Apply inclusion criteria
cohort_final <- cohort %>%
  {pipeline_steps}

# Save results
compute(cohort_final, name = "{output_table}", temporary = FALSE)

# Disconnect
DBI::dbDisconnect(con)
""",
            "description": "Standard cohort construction pipeline using dbplyr",
            "tags": ["cohort", "pipeline", "dbplyr"]
        },

        "table_loader": {
            "template": """#' Load and standardize a table
#'
#' @param con Database connection
#' @param table_name Name of the table to load
#' @param archive_date Optional archive date filter
#' @param standardize Whether to apply standardization
#' @return A lazy tibble
#' @export
load_table <- function(con, table_name, archive_date = NULL, standardize = TRUE) {{
  # Load table
  tbl_data <- tbl(con, table_name)

  # Apply archive filtering if specified
  if (!is.null(archive_date)) {{
    tbl_data <- tbl_data %>%
      filter(archive_date == !!archive_date)
  }}

  # Standardize column names
  if (standardize) {{
    tbl_data <- tbl_data %>%
      rename_with(tolower) %>%
      rename_with(~ gsub("\\\\s+", "_", .x))
  }}

  return(tbl_data)
}}
""",
            "description": "Reusable table loading function with standardization",
            "tags": ["function", "table", "loader"]
        },

        "demographics_join": {
            "template": """# Join demographics data
demographics <- tbl(con, "hds_curated_assets__demographics") %>%
  select(
    person_id,
    date_of_birth,
    sex,
    ethnicity,
    lsoa
  )

cohort_with_demographics <- cohort %>%
  left_join(demographics, by = "person_id")
""",
            "description": "Standard demographics table join pattern",
            "tags": ["join", "demographics", "hds"]
        },

        "codelist_filter": {
            "template": """# Apply codelist filtering
codelists <- read_csv("{codelist_path}")

{table_name}_filtered <- {table_name} %>%
  inner_join(
    codelists %>% select(code, description),
    by = c("{code_column}" = "code")
  ) %>%
  filter({additional_filters})
""",
            "description": "Filter data using medical codelists",
            "tags": ["filter", "codelist", "snomed", "icd10"]
        },

        "test_function": {
            "template": """test_that("{function_name} works correctly", {{
  # Setup
  {setup_code}

  # Execute
  result <- {function_name}({test_params})

  # Assert
  expect_equal({expected_assertions})
  expect_true({validation_checks})

  # Edge cases
  {edge_case_tests}
}})
""",
            "description": "Test template for R functions using testthat",
            "tags": ["test", "testthat", "quality"]
        },

        "phenotyping_algorithm": {
            "template": """#' {phenotype_name} Phenotyping Algorithm
#'
#' Identifies patients with {condition_description}
#'
#' @param con Database connection
#' @param index_date Reference date for phenotyping
#' @param lookback_days Days to look back from index date
#' @return Tibble with person_id and phenotype flags
#' @export
identify_{phenotype_id} <- function(con, index_date, lookback_days = 365) {{

  # Load codelists
  codes_{phenotype_id} <- read_csv("codelists/{phenotype_id}_codes.csv")

  # Query clinical events
  events <- tbl(con, "clinical_events") %>%
    filter(
      event_date >= !!index_date - !!lookback_days,
      event_date <= !!index_date
    ) %>%
    inner_join(codes_{phenotype_id}, by = c("code" = "snomed_code"))

  # Aggregate to person level
  phenotype <- events %>%
    group_by(person_id) %>%
    summarise(
      {phenotype_id}_flag = 1L,
      {phenotype_id}_first_date = min(event_date, na.rm = TRUE),
      {phenotype_id}_event_count = n()
    ) %>%
    ungroup()

  return(phenotype)
}}
""",
            "description": "Template for creating phenotyping algorithms",
            "tags": ["phenotyping", "clinical", "algorithm"]
        },

        "data_quality_check": {
            "template": """#' Perform data quality checks
#'
#' @param tbl_data Table to check
#' @param required_cols Required column names
#' @return List with check results
check_data_quality <- function(tbl_data, required_cols) {{

  results <- list()

  # Check 1: Required columns present
  actual_cols <- colnames(tbl_data)
  missing_cols <- setdiff(required_cols, actual_cols)
  results$columns_present <- length(missing_cols) == 0
  results$missing_columns <- missing_cols

  # Check 2: Row count
  row_count <- tbl_data %>% count() %>% pull(n)
  results$row_count <- row_count
  results$has_data <- row_count > 0

  # Check 3: Null counts for key columns
  null_counts <- tbl_data %>%
    summarise(across(all_of(required_cols), ~ sum(is.na(.)), .names = "null_{.col}")) %>%
    collect()
  results$null_counts <- null_counts

  # Check 4: Duplicate IDs
  if ("person_id" %in% actual_cols) {{
    dup_count <- tbl_data %>%
      group_by(person_id) %>%
      filter(n() > 1) %>%
      count() %>%
      pull(n)
    results$duplicate_ids <- dup_count
  }}

  return(results)
}}
""",
            "description": "Data quality validation function",
            "tags": ["quality", "validation", "checks"]
        },

        "aggregate_with_privacy": {
            "template": """# Aggregate with disclosure control
{table_name}_summary <- {table_name} %>%
  group_by({grouping_vars}) %>%
  summarise(
    count = n(),
    {aggregations}
  ) %>%
  ungroup() %>%
  # Apply disclosure control (round to nearest 5, suppress <10)
  mutate(
    count_rounded = round(count / 5) * 5,
    count_suppressed = if_else(count < 10, NA_real_, count_rounded)
  ) %>%
  filter(!is.na(count_suppressed))  # Remove suppressed rows
""",
            "description": "Privacy-preserving aggregation with disclosure control",
            "tags": ["privacy", "aggregation", "disclosure"]
        }
    }

    def __init__(self, hybrid_retriever=None, code_analyzer=None):
        """Initialize R code generator.

        Args:
            hybrid_retriever: HybridRetriever instance for semantic search
            code_analyzer: CodeAnalyzer instance for pattern learning
        """
        self.hybrid_retriever = hybrid_retriever
        self.code_analyzer = code_analyzer
        self.anthropic_client = None

        # Initialize Anthropic client if API key is available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and ANTHROPIC_AVAILABLE:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        else:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("anthropic package not installed - AI features will be limited")
            elif not api_key:
                logger.warning("ANTHROPIC_API_KEY not found - some features will be limited")

        # Learned patterns from codebase
        self.learned_patterns: List[RCodePattern] = []

    def generate_template(self,
                         template_type: str,
                         params: Dict[str, Any],
                         language: str = "r") -> CodeSuggestion:
        """Generate code from a template.

        Args:
            template_type: Type of template (e.g., 'cohort_pipeline', 'table_loader')
            params: Parameters to fill in template
            language: Target language ('python' or 'r', default 'r')

        Returns:
            CodeSuggestion with generated code
        """
        # Select appropriate template dictionary
        templates = self.PYTHON_TEMPLATES if language.lower() == "python" else self.R_TEMPLATES

        if template_type not in templates:
            # Try to find similar patterns using RAG
            if self.hybrid_retriever and self.anthropic_client:
                return self._generate_custom_template(template_type, params, language)
            else:
                return CodeSuggestion(
                    suggestion_type="template",
                    code="",
                    explanation=f"Template type '{template_type}' not found and RAG not available",
                    confidence=0.0
                )

        template_info = templates[template_type]
        template_str = template_info["template"]

        # Fill in template parameters
        try:
            # Add default date if not provided
            if "date" not in params:
                from datetime import datetime
                params["date"] = datetime.now().strftime("%Y-%m-%d")

            code = template_str.format(**params)

            return CodeSuggestion(
                suggestion_type="template",
                code=code,
                explanation=f"Generated {language.upper()} {template_type} template: {template_info['description']}",
                confidence=0.95,
                context=f"Language: {language.upper()} | Tags: {', '.join(template_info['tags'])}"
            )
        except KeyError as e:
            return CodeSuggestion(
                suggestion_type="template",
                code="",
                explanation=f"Missing required parameter: {e}",
                confidence=0.0
            )

    def _generate_custom_template(self,
                                  template_type: str,
                                  params: Dict[str, Any],
                                  language: str = "r") -> CodeSuggestion:
        """Generate custom template using RAG and Claude.

        Args:
            template_type: Description of what to generate
            params: Parameters for the template
            language: Target language ('python' or 'r')

        Returns:
            CodeSuggestion with AI-generated code
        """
        # Search for similar examples in codebase
        lang_specific = "Python PySpark" if language.lower() == "python" else "R dbplyr"
        query = f"{template_type} {lang_specific} example"
        similar_docs = self.hybrid_retriever.similarity_search(query, k=5)

        # Build context from similar examples
        context = "\n\n---\n\n".join([
            f"# Example from {doc.metadata.get('repo', 'unknown')}/{doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in similar_docs
        ])

        # Construct prompt for Claude
        if language.lower() == "python":
            prompt = f"""Based on the following Python/PySpark code examples from the codebase, generate a new Python/PySpark code template for: {template_type}

Parameters to include:
{chr(10).join(f'- {k}: {v}' for k, v in params.items())}

Examples from codebase:
{context}

Generate clean, well-documented Python code following the patterns shown in the examples. Include:
1. Proper docstrings (Google/NumPy style)
2. PySpark DataFrame operations
3. Error handling
4. Comments explaining key steps
5. Type hints where appropriate

Output only the Python code, no explanations."""
        else:
            prompt = f"""Based on the following R code examples from the codebase, generate a new R/dbplyr code template for: {template_type}

Parameters to include:
{chr(10).join(f'- {k}: {v}' for k, v in params.items())}

Examples from codebase:
{context}

Generate clean, well-documented R code following the patterns shown in the examples. Include:
1. Proper roxygen2 documentation comments
2. dbplyr pipeline patterns
3. Error handling
4. Comments explaining key steps

Output only the R code, no explanations."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            generated_code = response.content[0].text

            return CodeSuggestion(
                suggestion_type="template",
                code=generated_code,
                explanation=f"AI-generated template for '{template_type}' based on {len(similar_docs)} similar examples",
                confidence=0.8,
                context=f"Learned from: {', '.join(set(doc.metadata.get('repo', 'unknown') for doc in similar_docs))}"
            )
        except Exception as e:
            logger.error(f"Failed to generate custom template: {e}")
            return CodeSuggestion(
                suggestion_type="template",
                code="",
                explanation=f"Failed to generate template: {str(e)}",
                confidence=0.0
            )

    def suggest_completion(self,
                          code_context: str,
                          cursor_position: Optional[int] = None) -> List[CodeSuggestion]:
        """Suggest code completions based on context.

        Args:
            code_context: The code written so far
            cursor_position: Position in code where completion is needed

        Returns:
            List of CodeSuggestion objects ranked by relevance
        """
        suggestions = []

        # Pattern 1: Detect common dbplyr pipeline patterns
        if "tbl(con," in code_context and code_context.strip().endswith("%>%"):
            # Suggest common next steps in pipeline
            pipeline_suggestions = [
                ("filter", "filter({condition})", "Filter rows based on condition"),
                ("select", "select({columns})", "Select specific columns"),
                ("mutate", "mutate({new_column} = {expression})", "Create or modify columns"),
                ("group_by", "group_by({grouping_vars})", "Group data for aggregation"),
                ("left_join", "left_join({other_table}, by = c(\"{key}\" = \"{other_key}\"))", "Join with another table"),
                ("summarise", "summarise({aggregation})", "Aggregate grouped data"),
            ]

            for op_name, code, desc in pipeline_suggestions:
                suggestions.append(CodeSuggestion(
                    suggestion_type="completion",
                    code=f"  {code}",
                    explanation=desc,
                    confidence=0.85
                ))

        # Pattern 2: Detect table references to HDS curated assets
        if "tbl(con," in code_context:
            if self.code_analyzer:
                # Suggest commonly used tables from the codebase
                all_tables = self.code_analyzer.get_all_tables()
                hds_tables = [t for t in all_tables if t.startswith("hds_curated_assets")]

                for table in hds_tables[:5]:  # Top 5
                    suggestions.append(CodeSuggestion(
                        suggestion_type="completion",
                        code=f'tbl(con, "{table}")',
                        explanation=f"Load {table}",
                        confidence=0.75,
                        context=f"Used in {len(self.code_analyzer.table_to_repos.get(table, []))} repos"
                    ))

        # Pattern 3: Use AI for context-aware completion
        if self.anthropic_client and len(suggestions) < 3:
            ai_suggestions = self._get_ai_completions(code_context)
            suggestions.extend(ai_suggestions)

        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)

        return suggestions[:10]  # Return top 10

    def _get_ai_completions(self, code_context: str) -> List[CodeSuggestion]:
        """Get AI-powered code completions using Claude.

        Args:
            code_context: Current code context

        Returns:
            List of AI-generated suggestions
        """
        # Search for similar code in the indexed codebase
        similar_docs = []
        if self.hybrid_retriever:
            # Extract last few lines as query
            lines = code_context.strip().split('\n')
            query_text = '\n'.join(lines[-5:]) if len(lines) >= 5 else code_context
            similar_docs = self.hybrid_retriever.similarity_search(
                f"R code similar to: {query_text}",
                k=3
            )

        # Build context
        examples_context = "\n\n---\n\n".join([
            f"# From {doc.metadata.get('repo', 'unknown')}\n{doc.page_content}"
            for doc in similar_docs
        ]) if similar_docs else "No similar examples found."

        prompt = f"""Given this R code context, suggest 2-3 likely next lines of code.

Current code:
```r
{code_context}
```

Similar examples from codebase:
{examples_context}

Provide 2-3 short, practical completions that would naturally follow this code.
Format your response as a JSON array:
[
  {{"code": "suggestion 1", "explanation": "why this makes sense"}},
  {{"code": "suggestion 2", "explanation": "why this makes sense"}}
]"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            suggestions_data = json.loads(response.content[0].text)

            return [
                CodeSuggestion(
                    suggestion_type="completion",
                    code=item["code"],
                    explanation=item["explanation"],
                    confidence=0.7
                )
                for item in suggestions_data[:3]
            ]
        except Exception as e:
            logger.error(f"Failed to get AI completions: {e}")
            return []

    def generate_tests(self,
                      function_code: str,
                      function_name: Optional[str] = None) -> CodeSuggestion:
        """Generate test cases for R function.

        Args:
            function_code: The R function code to test
            function_name: Name of function (extracted if not provided)

        Returns:
            CodeSuggestion with test code
        """
        # Extract function name if not provided
        if not function_name:
            match = re.search(r'(\w+)\s*<-\s*function', function_code)
            if match:
                function_name = match.group(1)
            else:
                return CodeSuggestion(
                    suggestion_type="test",
                    code="",
                    explanation="Could not extract function name from code",
                    confidence=0.0
                )

        # Extract parameters
        param_match = re.search(r'function\s*\((.*?)\)', function_code, re.DOTALL)
        params = []
        if param_match:
            params_str = param_match.group(1)
            # Simple parameter extraction (could be improved)
            params = [p.strip().split('=')[0].strip() for p in params_str.split(',') if p.strip()]

        if not self.anthropic_client:
            # Use template-based test generation (R version)
            template = self.R_TEMPLATES["test_function"]
            test_code = template["template"].format(
                function_name=function_name,
                setup_code="  # TODO: Setup test data",
                test_params=", ".join(params) if params else "",
                expected_assertions="# TODO: Add assertions",
                validation_checks="TRUE  # TODO: Add validation",
                edge_case_tests="  # TODO: Add edge case tests"
            )

            return CodeSuggestion(
                suggestion_type="test",
                code=test_code,
                explanation=f"Template-based test for {function_name}",
                confidence=0.6,
                context="Fill in TODO sections with actual test logic"
            )

        # AI-powered test generation
        prompt = f"""Generate comprehensive testthat test cases for this R function.

Function code:
```r
{function_code}
```

Create tests that cover:
1. Normal/expected inputs
2. Edge cases (empty data, NA values, boundary conditions)
3. Error cases (invalid inputs)

Use the testthat framework. Include setup code and clear assertions.
Output only the R test code."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            test_code = response.content[0].text

            # Clean up markdown code blocks if present
            test_code = re.sub(r'^```r?\n', '', test_code)
            test_code = re.sub(r'\n```$', '', test_code)

            return CodeSuggestion(
                suggestion_type="test",
                code=test_code,
                explanation=f"AI-generated tests for {function_name} covering normal, edge, and error cases",
                confidence=0.85
            )
        except Exception as e:
            logger.error(f"Failed to generate tests: {e}")
            return CodeSuggestion(
                suggestion_type="test",
                code="",
                explanation=f"Failed to generate tests: {str(e)}",
                confidence=0.0
            )

    def generate_documentation(self,
                             code: str,
                             doc_type: str = "function") -> CodeSuggestion:
        """Generate documentation for R code.

        Args:
            code: The R code to document
            doc_type: Type of documentation ('function', 'file', 'readme')

        Returns:
            CodeSuggestion with documentation
        """
        if not self.anthropic_client:
            return CodeSuggestion(
                suggestion_type="documentation",
                code="",
                explanation="AI client not available for documentation generation",
                confidence=0.0
            )

        if doc_type == "function":
            prompt = f"""Generate roxygen2 documentation for this R function.

Function code:
```r
{code}
```

Include:
- @description
- @param for each parameter with type and description
- @return describing return value
- @examples with runnable examples
- @export if appropriate

Output only the roxygen2 comments (starting with #')."""

        elif doc_type == "readme":
            prompt = f"""Generate a README.md for this R code.

Code:
```r
{code}
```

Include:
- Overview of what the code does
- Installation/setup instructions
- Usage examples
- Key functions/features
- Dependencies

Output only the markdown."""

        else:  # file-level documentation
            prompt = f"""Generate file-level documentation comment for this R code.

Code:
```r
{code}
```

Create a header comment block explaining:
- Purpose of this file
- Main functions/capabilities
- How it fits into the larger project

Output only the comment block."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            documentation = response.content[0].text

            return CodeSuggestion(
                suggestion_type="documentation",
                code=documentation,
                explanation=f"AI-generated {doc_type} documentation",
                confidence=0.85
            )
        except Exception as e:
            logger.error(f"Failed to generate documentation: {e}")
            return CodeSuggestion(
                suggestion_type="documentation",
                code="",
                explanation=f"Failed to generate documentation: {str(e)}",
                confidence=0.0
            )

    def suggest_refactoring(self,
                          code: str,
                          focus: Optional[str] = None) -> List[CodeSuggestion]:
        """Suggest refactoring improvements for R code.

        Args:
            code: The R code to analyze
            focus: Specific area to focus on ('performance', 'readability', 'dbplyr')

        Returns:
            List of refactoring suggestions with diffs
        """
        suggestions = []

        # Rule-based refactoring suggestions

        # Suggestion 1: Replace collect() with compute() for large data
        if "collect()" in code and "tbl(con," in code:
            suggestions.append(CodeSuggestion(
                suggestion_type="refactor",
                code=code.replace("collect()", "compute(name = 'temp_table', temporary = TRUE)"),
                explanation="Use compute() instead of collect() to keep data in database and avoid memory issues",
                confidence=0.75,
                context="Performance improvement for large datasets"
            ))

        # Suggestion 2: Use across() for multiple column operations
        if code.count("mutate(") > 1:
            suggestions.append(CodeSuggestion(
                suggestion_type="refactor",
                code="# Consider using across() to apply operations to multiple columns:\n# mutate(across(c(col1, col2), ~ operation(.x)))",
                explanation="Consolidate multiple mutate() calls using across() for cleaner code",
                confidence=0.7,
                context="Readability improvement"
            ))

        # Suggestion 3: Add indexes for commonly filtered columns
        if "filter(" in code:
            filter_cols = re.findall(r'filter\([^)]*?(\w+)\s*[=<>!]', code)
            if filter_cols:
                suggestions.append(CodeSuggestion(
                    suggestion_type="refactor",
                    code=f"# Consider adding database indexes on: {', '.join(set(filter_cols))}",
                    explanation="Adding indexes on filtered columns can significantly improve query performance",
                    confidence=0.65,
                    context="Performance optimization"
                ))

        # AI-powered refactoring suggestions
        if self.anthropic_client:
            ai_suggestions = self._get_ai_refactorings(code, focus)
            suggestions.extend(ai_suggestions)

        return suggestions

    def _get_ai_refactorings(self,
                           code: str,
                           focus: Optional[str] = None) -> List[CodeSuggestion]:
        """Get AI-powered refactoring suggestions.

        Args:
            code: Code to refactor
            focus: Refactoring focus area

        Returns:
            List of AI-generated refactoring suggestions
        """
        focus_text = f" focusing on {focus}" if focus else ""

        prompt = f"""Analyze this R/dbplyr code and suggest 2-3 refactoring improvements{focus_text}.

Code:
```r
{code}
```

For each suggestion, provide:
1. What to change
2. Why it's better
3. The refactored code

Consider:
- dbplyr best practices (lazy evaluation, compute vs collect)
- Code readability and maintainability
- Performance optimizations
- Error handling

Format as JSON array:
[
  {{
    "improvement": "description",
    "reason": "why it's better",
    "code": "refactored code snippet"
  }}
]"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            refactorings = json.loads(response.content[0].text)

            return [
                CodeSuggestion(
                    suggestion_type="refactor",
                    code=item["code"],
                    explanation=f"{item['improvement']}: {item['reason']}",
                    confidence=0.8
                )
                for item in refactorings[:3]
            ]
        except Exception as e:
            logger.error(f"Failed to get AI refactorings: {e}")
            return []

    def translate_python_to_r(self,
                            python_code: str) -> CodeSuggestion:
        """Translate Python/PySpark code to R/dbplyr.

        Useful for converting standard-pipeline patterns to R.

        Args:
            python_code: Python code to translate

        Returns:
            CodeSuggestion with R translation
        """
        if not self.anthropic_client:
            return CodeSuggestion(
                suggestion_type="translation",
                code="",
                explanation="AI client not available for translation",
                confidence=0.0
            )

        prompt = f"""Translate this Python/PySpark code to equivalent R/dbplyr code.

Python code:
```python
{python_code}
```

Requirements:
1. Use dbplyr for database operations (equivalent to PySpark)
2. Follow R naming conventions (snake_case)
3. Use tidyverse/dplyr idioms
4. Include roxygen2 documentation if translating functions
5. Preserve the logic and structure

Output only the R code with comments explaining key translations."""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )

            r_code = response.content[0].text

            # Clean up markdown
            r_code = re.sub(r'^```r?\n', '', r_code)
            r_code = re.sub(r'\n```$', '', r_code)

            return CodeSuggestion(
                suggestion_type="translation",
                code=r_code,
                explanation="Translated Python/PySpark code to R/dbplyr equivalent",
                confidence=0.8,
                context="Review translations for accuracy, especially complex operations"
            )
        except Exception as e:
            logger.error(f"Failed to translate code: {e}")
            return CodeSuggestion(
                suggestion_type="translation",
                code="",
                explanation=f"Translation failed: {str(e)}",
                confidence=0.0
            )

    def learn_from_codebase(self) -> Dict[str, Any]:
        """Analyze indexed codebase to learn R patterns.

        Returns:
            Statistics about learned patterns
        """
        if not self.code_analyzer:
            logger.warning("No code analyzer available for learning")
            return {"status": "error", "message": "Code analyzer not available"}

        logger.info("Learning R patterns from codebase...")

        # Analyze R files
        r_files = []
        for repo, files in self.code_analyzer.metadata.items():
            for file_path, metadata in files.items():
                if metadata.language in ["r", "rmarkdown"]:
                    r_files.append({
                        "repo": repo,
                        "file": file_path,
                        "metadata": metadata
                    })

        # Extract common patterns
        function_patterns = defaultdict(int)
        import_patterns = defaultdict(int)

        for file_info in r_files:
            metadata = file_info["metadata"]

            # Count function usage
            for func in metadata.function_calls:
                function_patterns[func] += 1

            # Count imports
            for imp in metadata.imports:
                import_patterns[imp] += 1

        # Store top patterns
        top_functions = sorted(function_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
        top_imports = sorted(import_patterns.items(), key=lambda x: x[1], reverse=True)[:20]

        stats = {
            "status": "success",
            "r_files_analyzed": len(r_files),
            "total_repos_with_r": len(set(f["repo"] for f in r_files)),
            "top_functions": dict(top_functions),
            "top_imports": dict(top_imports),
            "patterns_learned": len(self.learned_patterns)
        }

        logger.info(f"✓ Learned patterns from {len(r_files)} R files across {stats['total_repos_with_r']} repos")

        return stats
