import argparse
import logging
import os
from datetime import datetime, timedelta

import boto3
from tqdm import tqdm

logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("s3transfer").setLevel(logging.CRITICAL)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Download images from S3 bucket for a specific date or date range.")
    parser.add_argument("--bucket", required=True, help="The name of the S3 bucket.")
    parser.add_argument("--start_date", required=True, help="The start date in YYYY-MM-DD format.")
    parser.add_argument(
        "--end_date", help="The end date in YYYY-MM-DD format. If not provided, only the start_date will be used."
    )
    parser.add_argument("--output_folder", required=True, help="The folder where the images will be downloaded.")
    parser.add_argument("--boto_output", action="store_true", help="Enable boto3 output.")
    return parser.parse_args()


def download_images_by_prefix(s3_client, bucket, prefix, output_folder, boto_output):
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" in response:
        for obj in tqdm(response["Contents"], desc=f"Downloading images for {prefix}"):
            key = obj["Key"]
            file_name = os.path.join(output_folder, os.path.basename(key))
            if boto_output:
                print(f"Downloading {file_name} from {key}")
            s3_client.download_file(bucket, key, file_name)
            if boto_output:
                print(f"Downloaded {file_name}")


def main():
    args = parse_arguments()

    bucket = args.bucket
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else start_date
    output_folder = args.output_folder
    boto_output = args.boto_output

    s3_client = boto3.client("s3")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_date = start_date
    while current_date <= end_date:
        prefix = f"raw-uploads/{current_date.year}/{current_date.month}/{current_date.day}/"
        download_images_by_prefix(s3_client, bucket, prefix, output_folder, boto_output)
        current_date += timedelta(days=1)


if __name__ == "__main__":
    main()
