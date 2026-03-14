import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model import Model
#from sagemaker import DataCaptureConfig
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()

# 🔁 UPDATE these values:
s3_bucket_name = "apoorvawsbucket321"  # 👈 Your actual S3 bucket
role = "arn:aws:iam::339713091238:role/apoorv_admin"  # 👈 Your IAM Role ARN

# Setup training job
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir="src",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    output_path=f"s3://{s3_bucket_name}/model-artifacts/",
    sagemaker_session=sagemaker_session
)

# Train the model
sklearn_estimator.fit({
    "train": f"s3://{s3_bucket_name}/data/training/titanic.csv"
})

# Deploy the trained model to a real-time endpoint
trained_model = sklearn_estimator.create_model()

endpoint_predictor = trained_model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
    data_capture_config=DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=f"s3://{s3_bucket_name}/inference-capture/"
    )
)

print("✅ Model deployed to SageMaker endpoint!")