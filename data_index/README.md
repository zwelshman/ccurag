# Data Index Folder

This folder contains committed cache files for the BHFDSC Q&A system that can be shared across deployments.

## Files

- `bm25_index.pkl` - BM25 keyword search index for hybrid retrieval
- `code_metadata.json` - Structured metadata extracted from code analysis

## Purpose

Unlike the `.cache` folder which is excluded from version control, this folder is committed to Git. This allows pre-built indices to be shared across different environments and deployments without needing to rebuild them from scratch.

## Usage

The application will automatically check this folder first before looking in `.cache`, providing a seamless experience across deployments.

## Updating Indices

To update these files:

1. Build the indices using the Streamlit app or build scripts
2. Copy the generated files from `.cache/` to `data_index/`
3. Commit and push the changes to Git
