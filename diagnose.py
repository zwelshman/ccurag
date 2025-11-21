"""Diagnostic script to check the current state of the indexing system."""

import logging
import json
from pathlib import Path
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run diagnostics."""
    print("=" * 60)
    print("CCURAG DIAGNOSTIC REPORT")
    print("=" * 60)

    # Check configuration
    print("\n📋 Configuration:")
    print(f"  - Organization: {Config.GITHUB_ORG}")
    print(f"  - Vector Store Backend: {Config.VECTOR_STORE_BACKEND}")
    print(f"  - Sample Repos: {Config.SAMPLE_REPOS or 'All'}")
    print(f"  - Use Test Repos: {Config.USE_TEST_REPOS}")
    if Config.USE_TEST_REPOS:
        print(f"  - Test Repos: {Config.TEST_REPOS}")

    # Check checkpoint
    print("\n💾 Checkpoint Status:")
    checkpoint_file = Path(".checkpoint.json")
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"  ✅ Checkpoint exists")
            print(f"  - Organization: {checkpoint.get('organization')}")
            print(f"  - Total Repos: {checkpoint.get('total_repos')}")
            print(f"  - Completed: {checkpoint.get('completed_repos')}")
            print(f"  - Documents Collected: {checkpoint.get('documents_collected')}")
            print(f"  - Last Updated: {checkpoint.get('last_updated')}")

            # Show sample of processed repos
            processed = checkpoint.get('processed_repos', [])
            if processed:
                print(f"  - Sample Processed Repos: {', '.join(processed[:5])}")
                if len(processed) > 5:
                    print(f"    ...and {len(processed) - 5} more")
        except Exception as e:
            print(f"  ❌ Error reading checkpoint: {e}")
    else:
        print("  ⚠️  No checkpoint file found")

    # Check vector store
    print("\n🗄️  Vector Store Status:")
    if Config.VECTOR_STORE_BACKEND == "pinecone":
        try:
            from vector_store_pinecone import PineconeVectorStore
            pinecone_store = PineconeVectorStore()

            # Check if index exists
            existing_indexes = [index.name for index in pinecone_store.pc.list_indexes()]
            print(f"  - Pinecone Index Name: {Config.PINECONE_INDEX_NAME}")

            if Config.PINECONE_INDEX_NAME in existing_indexes:
                print(f"  ✅ Index exists")

                # Load and get stats
                pinecone_store.load_vectorstore()
                stats = pinecone_store.get_stats()
                vector_count = stats.get('total_vector_count', 0)
                namespaces = stats.get('namespaces', {})

                print(f"  - Total Vectors: {vector_count}")
                if vector_count == 0:
                    print("  ⚠️  WARNING: Index has 0 vectors!")
                    print("     This means documents were never uploaded to Pinecone.")
                    print("     Run force_reindex.py to fix this.")
                else:
                    print(f"  ✅ Index is populated with {vector_count} vectors")

                if namespaces:
                    print(f"  - Namespaces: {list(namespaces.keys())}")
            else:
                print(f"  ❌ Index does not exist")
                print(f"     Available indexes: {existing_indexes}")
                print("     Run index_repos.py or force_reindex.py to create it.")

        except Exception as e:
            print(f"  ❌ Error checking Pinecone: {e}")

    elif Config.VECTOR_STORE_BACKEND == "chroma":
        db_path = Path(Config.CHROMA_DB_DIR)
        if db_path.exists():
            print(f"  ✅ ChromaDB directory exists at {Config.CHROMA_DB_DIR}")
            # Count files
            files = list(db_path.rglob("*"))
            print(f"  - Files in DB: {len(files)}")
        else:
            print(f"  ❌ ChromaDB directory does not exist at {Config.CHROMA_DB_DIR}")

    # Summary and recommendations
    print("\n" + "=" * 60)
    print("📊 SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)

    if checkpoint_file.exists() and Config.VECTOR_STORE_BACKEND == "pinecone":
        try:
            from vector_store_pinecone import PineconeVectorStore
            pinecone_store = PineconeVectorStore()
            existing_indexes = [index.name for index in pinecone_store.pc.list_indexes()]

            if Config.PINECONE_INDEX_NAME in existing_indexes:
                pinecone_store.load_vectorstore()
                stats = pinecone_store.get_stats()
                vector_count = stats.get('total_vector_count', 0)

                if vector_count == 0:
                    print("\n⚠️  ISSUE DETECTED: Checkpoint exists but Pinecone has 0 vectors")
                    print("\n✅ SOLUTION:")
                    print("   1. Run: python force_reindex.py")
                    print("      This will re-fetch all documents and upload them to Pinecone")
                    print("\n   2. Or delete checkpoint and re-run:")
                    print("      rm .checkpoint.json && python index_repos.py")
                else:
                    print("\n✅ System looks healthy!")
                    print(f"   - {vector_count} vectors in Pinecone")
                    with open(checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                    print(f"   - {checkpoint.get('completed_repos', 0)} repos indexed")
            else:
                print("\n⚠️  Pinecone index does not exist")
                print("\n✅ SOLUTION:")
                print("   Run: python index_repos.py")
        except Exception as e:
            print(f"\n❌ Error during diagnosis: {e}")
    elif not checkpoint_file.exists():
        print("\n✅ No checkpoint - ready for fresh indexing")
        print("\n   Run: python index_repos.py")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
