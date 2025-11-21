"""Test the CodeAnalyzer with mock data to demonstrate functionality."""

import logging
from code_analyzer import CodeAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Mock code examples that would be found in BHFDSC repos
MOCK_DOCUMENTS = [
    {
        "content": """
# COVID-19 Phenotyping Algorithm

import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Load COVID positive cases
covid_df = spark.table("hds_curated_assets__covid_positive")

# Load demographics
demographics_df = spark.table("hds_curated_assets__demographics")

# Join datasets
result = covid_df.join(demographics_df, on="person_id", how="left")

# Save results
result.write.saveAsTable("ccu_covid_phenotype")
""",
        "metadata": {
            "source": "CCU002_01/phenotypes/covid_phenotype.py",
            "repo": "BHFDSC/CCU002_01",
            "type": "file"
        }
    },
    {
        "content": """
-- Cardiovascular outcomes analysis
-- Extract MI diagnoses from HES APC

SELECT
    p.person_id,
    p.age,
    d.diagnosis_code,
    d.diagnosis_date
FROM hds_curated_assets__hes_apc_diagnosis d
INNER JOIN hds_curated_assets__demographics p
    ON d.person_id = p.person_id
WHERE d.diagnosis_code LIKE 'I21%'  -- MI codes
""",
        "metadata": {
            "source": "CCU003_01/sql/mi_extraction.sql",
            "repo": "BHFDSC/CCU003_01",
            "type": "file"
        }
    },
    {
        "content": """
# Smoking Status Algorithm
# Identifies smoking status from GP records

library(dplyr)
library(hds)

# Load required data
demographics <- dbGetQuery(conn, "SELECT * FROM hds_curated_assets__demographics")
ethnicity <- spark.table("hds_curated_assets__ethnicity_multisource")

# Apply HDS phenotyping function
smoking_status <- hds::phenotype_smoking(demographics, gp_records)

# Save results
write.csv(smoking_status, "outputs/smoking_phenotype.csv")
""",
        "metadata": {
            "source": "phenotypes/smoking_algorithm.R",
            "repo": "BHFDSC/phenotyping-algorithms",
            "type": "file"
        }
    },
    {
        "content": """
# Diabetes Phenotyping
# Similar to smoking algorithm but for diabetes

import hds_functions as hds
from pyspark.sql import functions as F

# Load curated assets
demographics = spark.table("hds_curated_assets__demographics")
hes_diagnosis = spark.table("hds_curated_assets__hes_apc_diagnosis")

# Apply diabetes phenotyping
diabetes_cases = hds.phenotype_diabetes(demographics, hes_diagnosis)

# Save
diabetes_cases.write.parquet("outputs/diabetes_phenotype.parquet")
""",
        "metadata": {
            "source": "phenotypes/diabetes_algorithm.py",
            "repo": "BHFDSC/phenotyping-algorithms",
            "type": "file"
        }
    },
    {
        "content": """
# Death registry linkage
# Links COVID cases to death records

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Load datasets
covid = spark.table("hds_curated_assets__covid_positive")
deaths = spark.table("hds_curated_assets__deaths_single")
deaths_causes = spark.table("hds_curated_assets__deaths_cause_of_death")

# Join to find COVID-related deaths
covid_deaths = (
    covid
    .join(deaths, on="person_id", how="inner")
    .join(deaths_causes, on="person_id", how="left")
    .filter(F.col("death_date") >= F.col("covid_positive_date"))
)

result_count = covid_deaths.count()
print(f"Found {result_count} COVID-related deaths")
""",
        "metadata": {
            "source": "CCU002_01/analysis/covid_deaths.py",
            "repo": "BHFDSC/CCU002_01",
            "type": "file"
        }
    },
    {
        "content": """
# Data curation script for HES APC
# Processes raw HES data into curated assets

from pyspark.sql import functions as F
import hds_curation_functions as hds_curate

# Read raw HES data
raw_hes = spark.table("raw.hes_apc")

# Apply curation
curated_episodes = hds_curate.curate_hes_episodes(raw_hes)
curated_spells = hds_curate.curate_hes_spells(curated_episodes)

# Save to curated assets
curated_episodes.write.saveAsTable("hds_curated_assets__hes_apc_cips_episodes")
curated_spells.write.saveAsTable("hds_curated_assets__hes_apc_provider_spells")
""",
        "metadata": {
            "source": "curation/hes_curation.py",
            "repo": "BHFDSC/hds-curation",
            "type": "file"
        }
    }
]


def test_analyzer():
    """Test the analyzer with mock documents."""
    print("\n" + "="*80)
    print("TESTING CODE ANALYZER WITH MOCK DATA")
    print("="*80 + "\n")

    # Initialize analyzer
    analyzer = CodeAnalyzer()

    # Clear any existing cache to start fresh
    analyzer.clear_cache()

    # Index the mock documents
    logger.info(f"Indexing {len(MOCK_DOCUMENTS)} mock documents...")
    analyzer.index_documents(MOCK_DOCUMENTS, force_rebuild=True)

    # Print statistics
    stats = analyzer.get_stats()
    print("\n" + "-"*80)
    print("OVERALL STATISTICS")
    print("-"*80)
    print(f"Total repositories: {stats['total_repos']}")
    print(f"Total files analyzed: {stats['total_files']}")
    print(f"Unique tables found: {stats['total_unique_tables']}")
    print(f"Unique functions found: {stats['total_unique_functions']}")
    print(f"\nFile types:")
    for file_type, count in stats['file_types'].items():
        print(f"  {file_type}: {count} files")

    # Test 1: Table usage queries
    print("\n" + "="*80)
    print("TEST 1: TABLE USAGE QUERIES")
    print("="*80 + "\n")

    test_tables = [
        "hds_curated_assets__demographics",
        "hds_curated_assets__covid_positive",
        "hds_curated_assets__deaths_single",
        "hds_curated_assets__hes_apc_diagnosis"
    ]

    for table in test_tables:
        print(f"\n📊 Table: {table}")
        print("-" * 80)
        usage = analyzer.get_table_usage(table)

        if usage['total_repos'] > 0:
            print(f"✓ Used in {usage['total_repos']} repositories")
            print(f"  Repositories: {', '.join(usage['repos'])}")
            print(f"  Total files: {usage['total_files']}")
            print(f"  File types: {list(usage['files_by_type'].keys())}")
        else:
            print("✗ Not found in any repositories")

    # Test 2: Function usage queries
    print("\n" + "="*80)
    print("TEST 2: FUNCTION USAGE QUERIES")
    print("="*80 + "\n")

    usage = analyzer.get_function_usage("hds")
    print(f"Found {usage['total_functions_found']} HDS functions\n")

    if usage['total_functions_found'] > 0:
        for func_name, func_data in sorted(usage['functions'].items()):
            print(f"📦 {func_name}")
            print(f"   Used in {func_data['total_repos']} repos: {', '.join(func_data['repos'])}")
    else:
        print("No HDS functions found")

    # Test 3: Cross-analysis
    print("\n" + "="*80)
    print("TEST 3: CROSS-ANALYSIS - COVID + DEATHS")
    print("="*80 + "\n")

    covid_usage = analyzer.get_table_usage("hds_curated_assets__covid_positive")
    deaths_usage = analyzer.get_table_usage("hds_curated_assets__deaths_single")

    covid_repos = set(covid_usage['repos'])
    deaths_repos = set(deaths_usage['repos'])
    both = covid_repos & deaths_repos

    print(f"Repos using COVID data: {len(covid_repos)}")
    print(f"  {list(covid_repos)}")
    print(f"\nRepos using Deaths data: {len(deaths_repos)}")
    print(f"  {list(deaths_repos)}")
    print(f"\nRepos using BOTH: {len(both)}")
    if both:
        print(f"  {list(both)}")

    # Test 4: File classification
    print("\n" + "="*80)
    print("TEST 4: FILE CLASSIFICATION")
    print("="*80 + "\n")

    curation_files = []
    analysis_files = []
    phenotyping_files = []

    for repo, files in analyzer.metadata.items():
        for file_path, meta in files.items():
            if meta.file_type == "curation":
                curation_files.append(f"{repo}/{file_path}")
            elif meta.file_type == "analysis":
                analysis_files.append(f"{repo}/{file_path}")
            elif meta.file_type == "phenotyping":
                phenotyping_files.append(f"{repo}/{file_path}")

    print(f"Curation files ({len(curation_files)}):")
    for f in curation_files:
        print(f"  - {f}")

    print(f"\nAnalysis files ({len(analysis_files)}):")
    for f in analysis_files:
        print(f"  - {f}")

    print(f"\nPhenotyping files ({len(phenotyping_files)}):")
    for f in phenotyping_files:
        print(f"  - {f}")

    # Test 5: Detailed file inspection
    print("\n" + "="*80)
    print("TEST 5: DETAILED FILE INSPECTION")
    print("="*80 + "\n")

    # Pick one file and show all extracted metadata
    sample_repo = "BHFDSC/CCU002_01"
    sample_file = "CCU002_01/analysis/covid_deaths.py"

    if sample_repo in analyzer.metadata and sample_file in analyzer.metadata[sample_repo]:
        meta = analyzer.metadata[sample_repo][sample_file]
        print(f"File: {sample_file}")
        print(f"  Language: {meta.language}")
        print(f"  Type: {meta.file_type}")
        print(f"  Tables used: {meta.tables_used}")
        print(f"  Imports: {meta.imports}")
        print(f"  Function calls: {meta.function_calls}")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED!")
    print("="*80 + "\n")

    print("Next steps:")
    print("1. Run 'python build_metadata_index.py' to analyze actual BHFDSC repos")
    print("2. Run 'python example_analyzer_usage.py' to query the real data")


if __name__ == "__main__":
    test_analyzer()
