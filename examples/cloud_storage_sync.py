"""
Example implementation: Cloud Storage + ChromaDB Sync

This shows how to sync ChromaDB with cloud storage (S3, GCS, or Azure Blob)
with minimal changes to existing code.

Choose ONE of the providers below based on your cloud platform.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class CloudStorageSync:
    """Sync local ChromaDB directory with cloud storage."""

    def __init__(
        self,
        provider: Literal["s3", "gcs", "azure"],
        bucket_name: str,
        local_path: str = "chroma_db",
        remote_path: str = "chroma_db"
    ):
        """
        Initialize cloud storage sync.

        Args:
            provider: Cloud provider ("s3", "gcs", or "azure")
            bucket_name: Name of the storage bucket/container
            local_path: Local directory for ChromaDB
            remote_path: Remote path/prefix in cloud storage
        """
        self.provider = provider
        self.bucket_name = bucket_name
        self.local_path = Path(local_path)
        self.remote_path = remote_path

    def download(self) -> bool:
        """
        Download ChromaDB from cloud storage to local directory.

        Returns:
            True if download successful or files already exist locally, False otherwise
        """
        # Skip if local database already exists
        if self.local_path.exists() and any(self.local_path.iterdir()):
            logger.info(f"Local database exists at {self.local_path}, skipping download")
            return True

        logger.info(f"Downloading ChromaDB from {self.provider}://{self.bucket_name}/{self.remote_path}")

        try:
            if self.provider == "s3":
                return self._download_s3()
            elif self.provider == "gcs":
                return self._download_gcs()
            elif self.provider == "azure":
                return self._download_azure()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error downloading from cloud storage: {e}")
            return False

    def upload(self) -> bool:
        """
        Upload local ChromaDB to cloud storage.

        Returns:
            True if upload successful, False otherwise
        """
        if not self.local_path.exists():
            logger.warning(f"Local database at {self.local_path} does not exist, nothing to upload")
            return False

        logger.info(f"Uploading ChromaDB to {self.provider}://{self.bucket_name}/{self.remote_path}")

        try:
            if self.provider == "s3":
                return self._upload_s3()
            elif self.provider == "gcs":
                return self._upload_gcs()
            elif self.provider == "azure":
                return self._upload_azure()
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error uploading to cloud storage: {e}")
            return False

    # ==================== AWS S3 ====================

    def _download_s3(self) -> bool:
        """Download from AWS S3."""
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client('s3')

        try:
            # List all objects in the remote path
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=self.remote_path)

            self.local_path.mkdir(parents=True, exist_ok=True)

            for page in pages:
                if 'Contents' not in page:
                    logger.warning(f"No files found at s3://{self.bucket_name}/{self.remote_path}")
                    return False

                for obj in page['Contents']:
                    key = obj['Key']
                    # Calculate local file path
                    relative_path = key[len(self.remote_path):].lstrip('/')
                    if not relative_path:  # Skip the directory itself
                        continue
                    local_file = self.local_path / relative_path

                    # Create parent directories
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    # Download file
                    logger.info(f"Downloading {key} -> {local_file}")
                    s3.download_file(self.bucket_name, key, str(local_file))

            logger.info("S3 download complete")
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.error(f"Bucket {self.bucket_name} does not exist")
            else:
                logger.error(f"S3 error: {e}")
            return False

    def _upload_s3(self) -> bool:
        """Upload to AWS S3."""
        import boto3

        s3 = boto3.client('s3')

        # Walk through local directory and upload all files
        for local_file in self.local_path.rglob('*'):
            if local_file.is_file():
                # Calculate S3 key
                relative_path = local_file.relative_to(self.local_path)
                s3_key = f"{self.remote_path}/{relative_path}".replace('\\', '/')

                logger.info(f"Uploading {local_file} -> s3://{self.bucket_name}/{s3_key}")
                s3.upload_file(str(local_file), self.bucket_name, s3_key)

        logger.info("S3 upload complete")
        return True

    # ==================== Google Cloud Storage ====================

    def _download_gcs(self) -> bool:
        """Download from Google Cloud Storage."""
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        # List all blobs with prefix
        blobs = bucket.list_blobs(prefix=self.remote_path)

        self.local_path.mkdir(parents=True, exist_ok=True)
        found_files = False

        for blob in blobs:
            # Calculate local file path
            relative_path = blob.name[len(self.remote_path):].lstrip('/')
            if not relative_path:  # Skip the directory itself
                continue

            found_files = True
            local_file = self.local_path / relative_path

            # Create parent directories
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            logger.info(f"Downloading {blob.name} -> {local_file}")
            blob.download_to_filename(str(local_file))

        if not found_files:
            logger.warning(f"No files found at gs://{self.bucket_name}/{self.remote_path}")
            return False

        logger.info("GCS download complete")
        return True

    def _upload_gcs(self) -> bool:
        """Upload to Google Cloud Storage."""
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)

        # Walk through local directory and upload all files
        for local_file in self.local_path.rglob('*'):
            if local_file.is_file():
                # Calculate blob name
                relative_path = local_file.relative_to(self.local_path)
                blob_name = f"{self.remote_path}/{relative_path}".replace('\\', '/')

                blob = bucket.blob(blob_name)
                logger.info(f"Uploading {local_file} -> gs://{self.bucket_name}/{blob_name}")
                blob.upload_from_filename(str(local_file))

        logger.info("GCS upload complete")
        return True

    # ==================== Azure Blob Storage ====================

    def _download_azure(self) -> bool:
        """Download from Azure Blob Storage."""
        from azure.storage.blob import BlobServiceClient

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(self.bucket_name)

        # List all blobs with prefix
        blobs = container_client.list_blobs(name_starts_with=self.remote_path)

        self.local_path.mkdir(parents=True, exist_ok=True)
        found_files = False

        for blob in blobs:
            # Calculate local file path
            relative_path = blob.name[len(self.remote_path):].lstrip('/')
            if not relative_path:  # Skip the directory itself
                continue

            found_files = True
            local_file = self.local_path / relative_path

            # Create parent directories
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            logger.info(f"Downloading {blob.name} -> {local_file}")
            blob_client = container_client.get_blob_client(blob.name)
            with open(local_file, "wb") as f:
                f.write(blob_client.download_blob().readall())

        if not found_files:
            logger.warning(f"No files found at {self.bucket_name}/{self.remote_path}")
            return False

        logger.info("Azure Blob download complete")
        return True

    def _upload_azure(self) -> bool:
        """Upload to Azure Blob Storage."""
        from azure.storage.blob import BlobServiceClient

        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(self.bucket_name)

        # Walk through local directory and upload all files
        for local_file in self.local_path.rglob('*'):
            if local_file.is_file():
                # Calculate blob name
                relative_path = local_file.relative_to(self.local_path)
                blob_name = f"{self.remote_path}/{relative_path}".replace('\\', '/')

                blob_client = container_client.get_blob_client(blob_name)
                logger.info(f"Uploading {local_file} -> {self.bucket_name}/{blob_name}")
                with open(local_file, "rb") as f:
                    blob_client.upload_blob(f, overwrite=True)

        logger.info("Azure Blob upload complete")
        return True


# ==================== Integration Example ====================

def example_usage_in_streamlit():
    """
    Example of how to integrate this into your Streamlit app.

    Add this to the beginning of app.py:
    """
    import streamlit as st
    from config import Config

    # Initialize sync (choose your provider)
    # For S3:
    sync = CloudStorageSync(
        provider="s3",
        bucket_name="my-chroma-db-bucket",
        local_path=Config.CHROMA_DB_DIR
    )

    # For GCS:
    # sync = CloudStorageSync(
    #     provider="gcs",
    #     bucket_name="my-chroma-db-bucket",
    #     local_path=Config.CHROMA_DB_DIR
    # )

    # For Azure:
    # sync = CloudStorageSync(
    #     provider="azure",
    #     bucket_name="my-chroma-db-container",
    #     local_path=Config.CHROMA_DB_DIR
    # )

    # Download database on app startup
    @st.cache_resource
    def download_vectorstore():
        """Download vector store from cloud on startup."""
        success = sync.download()
        if success:
            st.success("Vector database downloaded from cloud storage")
        return success

    # Call this at the start of your app
    if not Path(Config.CHROMA_DB_DIR).exists() or not any(Path(Config.CHROMA_DB_DIR).iterdir()):
        with st.spinner("Downloading vector database from cloud storage..."):
            download_vectorstore()

    # Upload database after indexing
    # In your run_indexing() function, add this at the end:
    def upload_after_indexing():
        """Upload vector store to cloud after indexing."""
        with st.spinner("Uploading vector database to cloud storage..."):
            success = sync.upload()
            if success:
                st.success("Vector database uploaded to cloud storage")
            else:
                st.error("Failed to upload vector database to cloud storage")


# ==================== Configuration Example ====================

def setup_instructions():
    """
    Setup instructions for each cloud provider.
    """
    print("""
    # Setup Instructions

    ## AWS S3
    1. Create an S3 bucket: `aws s3 mb s3://my-chroma-db-bucket`
    2. Set AWS credentials:
       - Option A: AWS CLI (`aws configure`)
       - Option B: Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
       - Option C: IAM role (for EC2/ECS)
    3. Install: `pip install boto3`

    ## Google Cloud Storage
    1. Create a GCS bucket: `gsutil mb gs://my-chroma-db-bucket`
    2. Set up authentication:
       - Option A: `gcloud auth application-default login`
       - Option B: Service account key (GOOGLE_APPLICATION_CREDENTIALS env var)
    3. Install: `pip install google-cloud-storage`

    ## Azure Blob Storage
    1. Create a storage account and container via Azure Portal
    2. Get connection string from Azure Portal
    3. Set environment variable: `AZURE_STORAGE_CONNECTION_STRING`
    4. Install: `pip install azure-storage-blob`

    ## Add to .env file:
    # For AWS
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    AWS_DEFAULT_REGION=us-east-1

    # For GCS
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

    # For Azure
    AZURE_STORAGE_CONNECTION_STRING=your_connection_string

    ## Add to requirements.txt:
    boto3>=1.28.0  # For AWS S3
    google-cloud-storage>=2.10.0  # For GCS
    azure-storage-blob>=12.17.0  # For Azure
    """)


if __name__ == "__main__":
    # Show setup instructions
    setup_instructions()
