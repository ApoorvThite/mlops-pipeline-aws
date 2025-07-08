import sagemaker
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import boto3

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::339713091238:role/apoorv_admin"  # Replace with your IAM Role ARN
bucket = sagemaker_session.default_bucket()

# Name of the model tarball from the previous training step
#model_artifact = sagemaker_session.list_models()[-1].model_data_url if sagemaker_session.list_models() else \
#    's3://apoorvawsbucket321/sagemaker-scikit-learn-2025-07-08-01-12-08-327/output/model.tar.gz'  # fallback
model_artifact = "s3://apoorvawsbucket321/sagemaker-scikit-learn-2025-07-08-01-12-08-327/output/model.tar.gz"

print(f"Using model artifact: {model_artifact}")

# Create a SageMaker model
sklearn_model = Model(
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=boto3.Session().region_name,
        version="0.23-1"
    ),
    model_data=model_artifact,
    role=role,
    sagemaker_session=sagemaker_session,
    entry_point="inference.py",  # This file should define a `predict_fn` or similar logic
    source_dir="."               # Folder containing inference.py
)

# Deploy the model
predictor = sklearn_model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer()
)

print(f"\n✅ Model deployed! Endpoint name: {predictor.endpoint_name}")