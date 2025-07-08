import boto3

bucket = "apoorvawsbucket321"  # Replace this
s3 = boto3.client("s3")

s3.upload_file("data/titanic.csv", bucket, "data/training/titanic.csv")
s3.upload_file("src/code.zip", bucket, "code/code.zip")

print("Upload complete.")