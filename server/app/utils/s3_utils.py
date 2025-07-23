
import os
import boto3
import zipfile
import tempfile
import shutil
from fastapi import HTTPException, status
from app.utils import logger
from threading import Lock
from app.config.settings import settings



log_lock = Lock()

BUCKET_NAME = 'knowledgebase-s3bucket-342494841005-eu-west-2'
REGION_NAME = 'eu-west-2'

# Initialize boto3 S3 client using settings
try:
    print(f"[DEBUG] S3 Client Region: {REGION_NAME}")
    print(f"[DEBUG] S3 Bucket Name: {BUCKET_NAME}")
    s3_client = boto3.client('s3', region_name=REGION_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to initialize S3 client: {str(e)}")

async def download_s3_folder(folder_name: str) -> str:
    """
    Download files from an S3 bucket folder and extract to a temporary directory.
    Returns the path to the extracted directory.
    Raises an HTTPException on failure.
    """
    # Print AWS identity for debugging
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        print(f"[DEBUG] AWS Identity: {identity}")
    except Exception as e:
        print(f"[DEBUG] Could not get AWS identity: {e}")
    print(f"[DEBUG] Downloading folder: {folder_name} from bucket: {BUCKET_NAME}")
    with log_lock:
        logger.info(f"Downloading S3 folder: {folder_name} from bucket: {BUCKET_NAME}")

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    folder_name = folder_name.rstrip('/') + '/'
    
    try:
        # List and download all objects in the S3 folder using paginator (handles >1000 files and deep hierarchy)
        paginator = s3_client.get_paginator('list_objects_v2')
        found_any = False
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=folder_name):
            if 'Contents' not in page or not page['Contents']:
                continue
            for obj in page['Contents']:
                s3_key = obj['Key']
                if s3_key.endswith('/'):
                    continue  # Skip directory entries
                found_any = True
                relative_path = os.path.relpath(s3_key, folder_name)
                local_path = os.path.join(temp_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"[DEBUG] Downloading {s3_key} → {local_path}")
                with log_lock:
                    logger.debug(f"Downloading S3 object: {s3_key} to {local_path}")
                s3_client.download_file(BUCKET_NAME, s3_key, local_path)
        if not found_any:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No files found in S3 folder: {folder_name}"
            )

        # If the folder contains a ZIP file, extract it
        zip_files = [f for f in os.listdir(temp_dir) if f.endswith('.zip')]
        if zip_files:
            zip_path = os.path.join(temp_dir, zip_files[0])
            extract_path = os.path.join(temp_dir, 'extracted')
            os.makedirs(extract_path, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                with log_lock:
                    logger.debug(f"Extracted ZIP: {zip_path} to {extract_path}")
                # Find the root directory in the extracted contents
                extracted_dirs = [d for d in os.listdir(extract_path) 
                                if os.path.isdir(os.path.join(extract_path, d))]
                if extracted_dirs:
                    return os.path.join(extract_path, extracted_dirs[0])
                return extract_path
            except zipfile.BadZipFile:
                shutil.rmtree(extract_path, ignore_errors=True)
                with log_lock:
                    logger.error(f"Invalid ZIP file: {zip_path}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid ZIP file in S3 folder: {zip_files[0]}"
                )

        # If no ZIP, return the temp directory
        return temp_dir

    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        with log_lock:
            logger.error(f"Failed to download S3 folder {folder_name} from bucket {BUCKET_NAME}: {str(e)}")
        print(f"[DEBUG] Exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading S3 folder: {str(e)}"
        )
