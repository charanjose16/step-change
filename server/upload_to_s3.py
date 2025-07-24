import os
import boto3
from botocore.exceptions import ClientError

BUCKET_NAME = 'knowledgebase-s3bucket-342494841005-eu-west-2'
REGION_NAME = 'eu-west-2'

def upload_folder_to_s3(local_folder, bucket_name=BUCKET_NAME, s3_folder='stepchange-testing-repo/', region_name=REGION_NAME):
    # Use IAM role attached to the VM or credentials from env
    s3 = boto3.client('s3', region_name=region_name)

    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            # Compute S3 key relative to the local_folder root
            relative_path = os.path.relpath(local_path, local_folder)
            s3_key = os.path.join(s3_folder, relative_path).replace("\\", "/")
            print(f"Uploading {local_path} → s3://{bucket_name}/{s3_key}")
            try:
                s3.upload_file(local_path, bucket_name, s3_key)
            except ClientError as e:
                print(f"❌ AWS Client error: {e}")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    print("✅ Upload complete.")

if __name__ == "__main__":
    # Set these variables as needed
    local_folder = "C:/Users/charanarivarasan/Documents/stepchange-testing-repo"  # Local folder to upload
    upload_folder_to_s3(local_folder) 