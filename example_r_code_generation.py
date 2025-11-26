"""Example usage of R code generation features.

Demonstrates:
1. Template generation for common R/dbplyr patterns
2. Code completion suggestions
3. Test generation
4. Documentation generation
5. Refactoring suggestions
6. Python to R translation
"""

import os
from r_code_generator import RCodeGenerator, CodeSuggestion
from code_analyzer import CodeAnalyzer
from hybrid_retriever import HybridRetriever
from vector_store_pinecone import PineconeVectorStore


def print_suggestion(suggestion: CodeSuggestion, title: str = "Suggestion"):
    """Pretty print a code suggestion."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"Type: {suggestion.suggestion_type}")
    print(f"Confidence: {suggestion.confidence:.2f}")
    print(f"\nExplanation: {suggestion.explanation}")
    if suggestion.context:
        print(f"Context: {suggestion.context}")
    print(f"\nGenerated Code:")
    print("-" * 80)
    print(suggestion.code)
    print("-" * 80)


def example_template_generation():
    """Example 1: Generate code from templates."""
    print("\n" + "="*80)
    print("EXAMPLE 1: TEMPLATE GENERATION")
    print("="*80)

    generator = RCodeGenerator()

    # Example 1a: Generate a cohort pipeline
    print("\n📋 Generating cohort construction pipeline...")
    suggestion = generator.generate_template(
        template_type="cohort_pipeline",
        params={
            "pipeline_name": "MI Study Cohort",
            "purpose": "Identify patients with myocardial infarction and extract relevant features",
            "base_table": "hds_curated_assets__demographics",
            "initial_filters": "age >= 18 & !is.na(person_id)",
            "pipeline_steps": """
  # Join clinical events
  left_join(tbl(con, "clinical_events"), by = "person_id") %>%
  # Filter to MI events
  filter(diagnosis_code %in% mi_codes) %>%
  # Calculate follow-up time
  mutate(follow_up_days = as.numeric(end_date - index_date))
""".strip(),
            "output_table": "cohort_mi_final"
        }
    )
    print_suggestion(suggestion, "Cohort Pipeline Template")

    # Example 1b: Generate a table loader function
    print("\n📋 Generating reusable table loader function...")
    suggestion = generator.generate_template(
        template_type="table_loader",
        params={}
    )
    print_suggestion(suggestion, "Table Loader Function")

    # Example 1c: Generate a phenotyping algorithm
    print("\n📋 Generating phenotyping algorithm...")
    suggestion = generator.generate_template(
        template_type="phenotyping_algorithm",
        params={
            "phenotype_name": "Type 2 Diabetes",
            "phenotype_id": "t2dm",
            "condition_description": "type 2 diabetes mellitus based on diagnosis codes and prescriptions"
        }
    )
    print_suggestion(suggestion, "Phenotyping Algorithm Template")


def example_code_completion():
    """Example 2: Get code completion suggestions."""
    print("\n" + "="*80)
    print("EXAMPLE 2: CODE COMPLETION")
    print("="*80)

    generator = RCodeGenerator()

    # Example: Complete a pipeline
    code_context = """
library(dplyr)
library(dbplyr)

# Load demographics
demographics <- tbl(con, "hds_curated_assets__demographics") %>%
"""

    print("\n💡 Getting completion suggestions for pipeline...")
    print(f"Context:\n{code_context}")

    suggestions = generator.suggest_completion(code_context)

    print(f"\n✓ Found {len(suggestions)} suggestions:\n")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion.code.strip()}")
        print(f"   → {suggestion.explanation} (confidence: {suggestion.confidence:.2f})")
        print()


def example_test_generation():
    """Example 3: Generate tests for R functions."""
    print("\n" + "="*80)
    print("EXAMPLE 3: TEST GENERATION")
    print("="*80)

    generator = RCodeGenerator()

    # Example function to test
    function_code = """
calculate_age <- function(date_of_birth, reference_date = Sys.Date()) {
  # Calculate age in years
  age_years <- as.numeric(difftime(reference_date, date_of_birth, units = "days")) / 365.25

  # Round down to whole years
  floor(age_years)
}
"""

    print("\n🧪 Generating tests for age calculation function...")
    print(f"Function:\n{function_code}")

    suggestion = generator.generate_tests(function_code, "calculate_age")
    print_suggestion(suggestion, "Generated Tests")


def example_documentation_generation():
    """Example 4: Generate documentation."""
    print("\n" + "="*80)
    print("EXAMPLE 4: DOCUMENTATION GENERATION")
    print("="*80)

    generator = RCodeGenerator()

    # Example function to document
    function_code = """
load_and_filter_events <- function(con, person_ids, event_type,
                                  start_date = NULL, end_date = NULL) {
  events <- tbl(con, "clinical_events") %>%
    filter(person_id %in% !!person_ids)

  if (!is.null(event_type)) {
    events <- events %>% filter(type == !!event_type)
  }

  if (!is.null(start_date)) {
    events <- events %>% filter(event_date >= !!start_date)
  }

  if (!is.null(end_date)) {
    events <- events %>% filter(event_date <= !!end_date)
  }

  return(events)
}
"""

    print("\n📝 Generating roxygen2 documentation...")
    print(f"Function:\n{function_code}")

    suggestion = generator.generate_documentation(function_code, doc_type="function")
    print_suggestion(suggestion, "Generated Documentation")


def example_refactoring_suggestions():
    """Example 5: Get refactoring suggestions."""
    print("\n" + "="*80)
    print("EXAMPLE 5: REFACTORING SUGGESTIONS")
    print("="*80)

    generator = RCodeGenerator()

    # Code that could be improved
    code = """
# Load and process data
data <- tbl(con, "large_table") %>%
  filter(year == 2023) %>%
  mutate(age_group = case_when(
    age < 18 ~ "child",
    age >= 18 & age < 65 ~ "adult",
    age >= 65 ~ "senior"
  )) %>%
  collect()  # This loads everything into memory!

# Process in R
result <- data %>%
  group_by(age_group) %>%
  summarise(count = n(), avg_value = mean(value))
"""

    print("\n🔧 Getting refactoring suggestions...")
    print(f"Original code:\n{code}")

    suggestions = generator.suggest_refactoring(code, focus="performance")

    print(f"\n✓ Found {len(suggestions)} refactoring suggestions:\n")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion.explanation}")
        print(f"   Confidence: {suggestion.confidence:.2f}")
        if suggestion.context:
            print(f"   Context: {suggestion.context}")


def example_python_to_r_translation():
    """Example 6: Translate Python code to R."""
    print("\n" + "="*80)
    print("EXAMPLE 6: PYTHON TO R TRANSLATION")
    print("="*80)

    generator = RCodeGenerator()

    # Python/PySpark code from standard-pipeline
    python_code = """
def load_table(spark, table_name, archive_date=None, standardize=True):
    '''Load and standardize a table.

    Args:
        spark: SparkSession
        table_name: Name of table to load
        archive_date: Optional archive date filter
        standardize: Whether to apply standardization

    Returns:
        DataFrame
    '''
    # Load table
    df = spark.table(table_name)

    # Apply archive filtering
    if archive_date is not None:
        df = df.filter(f.col('archive_date') == archive_date)

    # Standardize column names
    if standardize:
        for col in df.columns:
            df = df.withColumnRenamed(col, col.lower().replace(' ', '_'))

    return df
"""

    print("\n🔄 Translating Python/PySpark to R/dbplyr...")
    print(f"Python code:\n{python_code}")

    suggestion = generator.translate_python_to_r(python_code)
    print_suggestion(suggestion, "R Translation")


def example_with_rag_integration():
    """Example 7: Use RAG integration for learning from codebase."""
    print("\n" + "="*80)
    print("EXAMPLE 7: RAG INTEGRATION (Learning from Codebase)")
    print("="*80)

    # Check if we have indexed data
    code_analyzer = CodeAnalyzer()

    if not code_analyzer.metadata:
        print("\n⚠️  No indexed codebase found. Run build_metadata_index.py first.")
        print("   This example shows how the generator learns from your actual R code.")
        return

    # Initialize with RAG components
    try:
        vector_store = PineconeVectorStore()
        hybrid_retriever = HybridRetriever(vector_store)

        # Load BM25 index if available
        import os
        if os.path.exists(".cache/bm25_index.pkl"):
            docs = vector_store.get_all_documents()
            hybrid_retriever.build_bm25_index(docs)
            print("✓ Loaded BM25 index")

        generator = RCodeGenerator(
            hybrid_retriever=hybrid_retriever,
            code_analyzer=code_analyzer
        )

        # Learn from codebase
        print("\n🧠 Learning R patterns from indexed codebase...")
        stats = generator.learn_from_codebase()

        print(f"\nLearning Statistics:")
        print(f"  R files analyzed: {stats.get('r_files_analyzed', 0)}")
        print(f"  Repos with R code: {stats.get('total_repos_with_r', 0)}")

        if stats.get('top_functions'):
            print(f"\n  Top R functions used:")
            for func, count in list(stats['top_functions'].items())[:5]:
                print(f"    - {func}: {count} times")

        if stats.get('top_imports'):
            print(f"\n  Top R packages imported:")
            for pkg, count in list(stats['top_imports'].items())[:5]:
                print(f"    - {pkg}: {count} times")

        # Try custom template generation using learned patterns
        print("\n\n💡 Generating custom template based on learned patterns...")
        suggestion = generator.generate_template(
            template_type="data quality validation function for health data",
            params={
                "dataset_type": "HES",
                "required_columns": ["person_id", "event_date", "diagnosis_code"]
            }
        )
        print_suggestion(suggestion, "Custom Template (RAG-powered)")

    except Exception as e:
        print(f"\n⚠️  Could not initialize RAG components: {e}")
        print("   Make sure Pinecone is configured and indexed.")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("R CODE GENERATION EXAMPLES")
    print("Demonstrating AI-powered code generation for R/dbplyr")
    print("="*80)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not found in environment")
        print("   Some features will use template-based generation instead of AI")
        print("   Set your API key in .env to enable full AI capabilities\n")

    examples = [
        ("Template Generation", example_template_generation),
        ("Code Completion", example_code_completion),
        ("Test Generation", example_test_generation),
        ("Documentation Generation", example_documentation_generation),
        ("Refactoring Suggestions", example_refactoring_suggestions),
        ("Python to R Translation", example_python_to_r_translation),
        ("RAG Integration", example_with_rag_integration),
    ]

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("✓ All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
