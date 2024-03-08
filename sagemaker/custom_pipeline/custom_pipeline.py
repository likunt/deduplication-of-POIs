from __future__ import annotations

import datetime

from airflow.decorators import dag
from airflow.models import Variable
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTransformOperator
from airflow.providers.amazon.aws.sensors.glue import GlueJobSensor
from airflow.utils.task_group import TaskGroup
from dedup_custom_pipeline.config import *
from hooks.callbacks import FailureCallback, SuccessCallback
from operators.glue import AWSGlueCatalogGetLatestTableOperator
from operators.sagemaker import GetTransformConfigOperator
from utils.version import get_version

S3_BUCKET_BASE = Variable.get("BASE_POIS_S3_BUCKET")
S3_BUCKET = Variable.get("DEDUP_S3_BUCKET")
DATABASE_NAME = Variable.get("BASE_POIS_GLUE_CATALOG_DATABASE")

PREFIX = "dedup"
METHOD = "Custom"

default_args: dict[str, str] = {}


@dag(
    default_args=default_args,
    start_date=datetime.datetime(2024, 2, 20),
    tags=["dedup", "custom"],
    on_success_callback=SuccessCallback,
    on_failure_callback=FailureCallback,
)
def dedup_custom_pipeline():
    """An Airflow DAG for Dedup using custom-trained model."""

    # Get the table name
    table_name = AWSGlueCatalogGetLatestTableOperator(
        task_id="get_base_poi_dataset_version",
        database_name=DATABASE_NAME,
    )

    # Get run version
    run_version = get_version()

    with TaskGroup("process_individual_poi_types") as process_individual_poi_types:
        for poi_type in POI_TYPES:
            #  I. GeneratePairsGlueJob
            generate_pairs = GlueJobOperator(
                task_id=f"pair_generation_{poi_type}",
                job_name="dedup_ml_generatepairs",
                script_args={
                    "--INPUT_CATALOG_DATABASE": DATABASE_NAME,
                    "--INPUT_CATALOG_PREDICATE": f"type='{poi_type}'",
                    "--INPUT_CATALOG_TABLE": table_name.output,
                    "--OUTPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/pairs/{poi_type}/{run_version}/",
                },
                wait_for_completion=False,
            )

            wait_for_pair_generation = GlueJobSensor(
                task_id=f"wait_for_pair_generation_{poi_type}",
                job_name="dedup_ml_generatepairs",
                run_id=generate_pairs.output,
            )

            # II. PrefilterGlueJob
            if poi_type == "dining":
                STOPWORDS = STOPWORDS_COMMON+STOPWORDS_DINING
            elif poi_type == "experience":
                STOPWORDS = STOPWORDS_COMMON+STOPWORDS_EXPERIENCE
            elif poi_type == "lodging":
                STOPWORDS = STOPWORDS_COMMON+STOPWORDS_LODGING

            prefiltering = GlueJobOperator(
                task_id=f"prefiltering_{poi_type}",
                job_name="dedup_ml_prefilter",
                script_args={
                    "--INPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/pairs/{poi_type}/{run_version}/",
                    "--OUTPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/prefiltering/{poi_type}/{run_version}/",
                    "--STOPWORDS": STOPWORDS,
                    "--THRESHOLD": THRESHOLD, 
                    "--DIST_TOL": DIST_TOL,
                    "--ADDRESS_ABBREVIATIONS": ADDRESS_ABBREVIATIONS,
                },
                wait_for_completion=False,
            )

            wait_for_prefiltering = GlueJobSensor(
                task_id=f"wait_for_prefiltering_{poi_type}",
                job_name="dedup_ml_prefilter",
                run_id=prefiltering.output,
            )

            # III. AddFeaturesGlueJob
            addfeatures = GlueJobOperator(
                task_id=f"addfeatures_{poi_type}",
                job_name="dedup_ml_addfeatures",
                script_args={
                    "--INPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/prefiltering/{poi_type}/{run_version}/",
                    "--OUTPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/addfeatures/{poi_type}/{run_version}/",
                    "--FILL_VALUE": FILL_VALUE,
                },
                wait_for_completion=False,
            )

            wait_for_addfeatures = GlueJobSensor(
                task_id=f"wait_for_addfeatures_{poi_type}",
                job_name="dedup_ml_addfeatures",
                run_id=addfeatures.output,
            )

            # IV. Sagemaker Batch Transform Job

            config_dedup = GetTransformConfigOperator(
                task_id=f"get_transform_config_{poi_type}",
                model_name="dedup-model-name-gradient",
                run_version=run_version,
                input_path=f"s3://{S3_BUCKET}/{PREFIX}/addfeatures/{poi_type}/{run_version}",
                output_path=f"s3://{S3_BUCKET}/{PREFIX}/inference/{poi_type}/{run_version}",
                instance_count=4,
                instance_type="ml.m5.xlarge",
                accept="text/csv",
                content_type="text/csv",
                strategy="MultiRecord",
            )

            transform_dedup = SageMakerTransformOperator(
                task_id=f"sagemaker_transform_{poi_type}",
                config=config_dedup.output,
                aws_conn_id="aws_default",
            )

            # V. InferenceGlueJob
            inference = GlueJobOperator(
                task_id=f"inference_{poi_type}",
                job_name="dedup_ml_inference",
                script_args={
                    "--INPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/inference/{poi_type}/{run_version}/",
                    "--OUTPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/postinference/{poi_type}/{run_version}/",
                },
                wait_for_completion=False,
            )

            wait_for_inference = GlueJobSensor(
                task_id=f"wait_for_inference_{poi_type}",
                job_name="dedup_ml_inference",
                run_id=inference.output,
            )

            # VI. MarkDuplicatesGlueJob
            markduplicates = GlueJobOperator(
                task_id=f"markduplicates_{poi_type}",
                job_name="dedup_ml_markduplicates",
                script_args={
                    "--INPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/postinference/{poi_type}/{run_version}/",
                    "--OUTPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/markduplicates/{METHOD}/{poi_type}/{run_version}/",
                    "--INPUT_CATALOG_DATABASE": DATABASE_NAME,
                    "--INPUT_CATALOG_PREDICATE": f"type='{poi_type}'",
                    "--INPUT_CATALOG_TABLE": table_name.output,
                },
                wait_for_completion=False,
            )

            wait_for_markduplicates = GlueJobSensor(
                task_id=f"wait_for_markduplicates_{poi_type}",
                job_name="dedup_ml_markduplicates",
                run_id=markduplicates.output,
            )

            (
                generate_pairs
                >> wait_for_pair_generation
                >> prefiltering
                >> wait_for_prefiltering
                >> addfeatures
                >> wait_for_addfeatures
                >> transform_dedup
                >> inference
                >> wait_for_inference
                >> markduplicates
                >> wait_for_markduplicates
            )

    table_name >> run_version >> process_individual_poi_types


dedup_custom_pipeline = dedup_custom_pipeline()
