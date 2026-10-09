from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core import image_uris
from sagemaker.core.shapes import InstancePreference
from sagemaker.core.training.configs import (
    Compute,
    SourceCode,
    InputData,
    OutputDataConfig,
    StoppingCondition,
)
from sagemaker.train.model_trainer import ModelTrainer


# ============================================================
# CONFIG
# ============================================================

REGION = "ap-south-1"

# CHANGE THESE
DATASET_S3_URI = "s3://yolov26-m-run/dataset"
OUTPUT_S3_URI = "s3://yolov26-m-run/deformable-detr-output/"

BASE_JOB_NAME = "deformable-detr-iedxray"

ROLE_ARN = (
    "arn:aws:iam::215572962108:"
    "role/service-role/AmazonSageMakerAdminIAMExecutionRole"
)


# ============================================================
# SAGEMAKER SESSION
# ============================================================

session = Session()

region = session.boto_region_name

if region != REGION:
    raise RuntimeError(
        f"SageMaker session is using region '{region}', "
        f"but this launcher expects '{REGION}'."
    )

role = ROLE_ARN

print("=" * 60)
print("SageMaker Deformable DETR Training")
print("=" * 60)
print(f"Region:  {region}")
print(f"Role:    {role}")
print(f"Dataset: {DATASET_S3_URI}")
print(f"Output:  {OUTPUT_S3_URI}")


# ============================================================
# INSTANCE PREFERENCE LIST
# ============================================================
#
# Ordered from most desirable -> fallback.
#
# IMPORTANT:
# Keep these to single-GPU instance configurations because
# detr_impl.py currently performs single-GPU training.
#
# SageMaker will try to obtain capacity from this list.
# ============================================================

compute = Compute(
    instance_preferences=[
        InstancePreference(
            instance_type="ml.g6e.8xlarge"
        ),
        InstancePreference(
            instance_type="ml.g6.8xlarge"
        ),
        InstancePreference(
            instance_type="ml.g5.8xlarge"
        ),
        InstancePreference(
            instance_type="ml.g5.12xlarge"
        ),
        InstancePreference(
            instance_type="ml.g6.4xlarge"
        )
    ],

    # One instance, whichever type wins
    instance_count=1,

    # Enough space for dataset + checkpoints + temporary files
    volume_size_in_gb=100,
)


# ============================================================
# PYTORCH TRAINING IMAGE
# ============================================================
#
# The container image is fixed before SageMaker chooses which
# GPU instance from the preference list will actually run.
#
# All candidates above are NVIDIA GPU instances.
# ============================================================

training_image = image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="2.6.0",
    py_version="py312",
    instance_type="ml.g6e.8xlarge",
    image_scope="training",
)

print(f"Training image: {training_image}")

source_code = SourceCode(
    source_dir="./Deformable DETR",
    entry_script="deformable_train.py",
    requirements="requirements.txt",

    ignore_patterns=[
        "__pycache__",
        "*.pyc",
        ".git",
        ".ipynb_checkpoints",
        ".cache",
    ],
)


# ============================================================
# DATASET INPUT
# ============================================================
#
# Inside the training container this becomes:
#
# /opt/ml/input/data/dataset/
#
# and SageMaker exposes:
#
# SM_CHANNEL_DATASET
#
# ============================================================

dataset_input = InputData(
    channel_name="dataset",
    data_source=DATASET_S3_URI,
)


# ============================================================
# OUTPUT
# ============================================================

output_config = OutputDataConfig(
    s3_output_path=OUTPUT_S3_URI
)


# ============================================================
# STOPPING CONDITION
# ============================================================
#
# max_runtime:
#   Training is allowed to run for up to 24 hours.
#
# max_pending:
#   Instance preference list can wait up to 12 hours for
#   capacity before SageMaker gives up.
#
# ============================================================

stopping_condition = StoppingCondition(
    max_runtime_in_seconds=24 * 60 * 60,
    max_pending_time_in_seconds=12 * 60 * 60,
)


# ============================================================
# MODEL TRAINER
# ============================================================

trainer = ModelTrainer(
    base_job_name=BASE_JOB_NAME,

    role=role,
    sagemaker_session=session,

    training_image=training_image,

    source_code=source_code,
    compute=compute,

    output_data_config=output_config,
    stopping_condition=stopping_condition,

    # Dataset should be downloaded to local instance storage
    training_input_mode="File",
)


# ============================================================
# START JOB
# ============================================================

print()
print("=" * 60)
print("Submitting training job...")
print("=" * 60)

trainer.train(
    input_data_config=[
        dataset_input
    ],

    # Don't block this launcher for potentially many hours.
    wait=True,
    logs=True,
)


# ============================================================
# JOB INFORMATION
# ============================================================

training_job = trainer._latest_training_job

print()
print("Training job submitted successfully.")
print(f"Job name: {training_job.training_job_name}")

training_job.refresh()

resource_config = training_job.resource_config

print()
print("Instance preference order:")

for i, preference in enumerate(
    resource_config.instance_preferences,
    start=1
):
    print(
        f"  {i}. {preference.instance_type}"
    )

print()
print(
    "Selected instance:",
    resource_config.selected_instance_type
)

print(
    "Selected count:",
    resource_config.selected_instance_count
)

if resource_config.selected_instance_type is None:
    print()
    print(
        "No instance selected yet. "
        "SageMaker is waiting for capacity from the preference list."
    )