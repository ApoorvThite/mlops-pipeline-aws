import boto3

s3_bucket_name = "apoorvawsbucket321"  # Replace this
s3_client = boto3.client("s3")

s3_client.upload_file("data/titanic.csv", s3_bucket_name, "data/training/titanic.csv")
s3_client.upload_file("src/code.zip", s3_bucket_name, "code/code.zip")

print("Upload complete.")