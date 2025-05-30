
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

# Initialize boto3 S3 client using settings
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize S3 client: {str(e)}")

async def download_s3_folder(folder_name: str) -> str:
    """
    Download files from an S3 bucket folder and extract to a temporary directory.
    Returns the path to the extracted directory.
    Raises an HTTPException on failure.
    """
    with log_lock:
        logger.info(f"Downloading S3 folder: {folder_name} from bucket: {settings.bucket_name}")

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    folder_name = folder_name.rstrip('/') + '/'
    
    try:
        # List objects in the S3 folder
        response = s3_client.list_objects_v2(Bucket=settings.bucket_name, Prefix=folder_name)
        if 'Contents' not in response or not response['Contents']:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No files found in S3 folder: {folder_name}"
            )

        # Download each file
        for obj in response['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith('/'):  # Skip directory entries
                continue
            local_path = os.path.join(temp_dir, s3_key[len(folder_name):])
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with log_lock:
                logger.debug(f"Downloading S3 object: {s3_key} to {local_path}")
            s3_client.download_file(settings.bucket_name, s3_key, local_path)

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
            logger.error(f"Failed to download S3 folder {folder_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading S3 folder: {str(e)}"
        )
