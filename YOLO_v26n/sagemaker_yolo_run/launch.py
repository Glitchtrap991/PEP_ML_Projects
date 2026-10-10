import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


# ============================================================
# CONFIGURATION
# ============================================================

REGION = "ap-south-1"

ROLE_ARN = (
    "arn:aws:iam::215572962108:"
    "role/service-role/AmazonSageMakerAdminIAMExecutionRole"
)

DATASET_S3 = "s3://yolov26-m-run/dataset"


# ============================================================
# AWS SESSION
# ============================================================

boto_session = boto3.Session(region_name=REGION)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session
)

print("Region:", boto_session.region_name)
print("Role:", ROLE_ARN)
print("Dataset:", DATASET_S3)


# ============================================================
# TRAINING JOB
# ============================================================

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src",

    role=ROLE_ARN,

    framework_version="2.6",
    py_version="py312",

    instance_type="ml.g5.12xlarge",
    instance_count=1,

    volume_size=100,

    max_run=12 * 60 * 60,

    sagemaker_session=sagemaker_session,
)


# ============================================================
# START JOB
# ============================================================

estimator.fit(
    {
        "training": DATASET_S3
    },
    wait=True,
    logs=True
)