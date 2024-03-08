from __future__ import annotations

import datetime

from airflow.decorators import dag
from airflow.models import Variable
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.sensors.glue import GlueJobSensor
from airflow.utils.task_group import TaskGroup
from dedup_findmatches_pipeline.config import *
from hooks.callbacks import FailureCallback, SuccessCallback
from operators.glue import AWSGlueCatalogGetLatestTableOperator
from utils.version import get_version

S3_BUCKET_BASE = Variable.get("BASE_POIS_S3_BUCKET")
S3_BUCKET = Variable.get("DEDUP_S3_BUCKET")
TRANSFORM_ID = Variable.get("DEDUP_TRANSFORM_ID")
DATABASE_NAME = Variable.get("BASE_POIS_GLUE_CATALOG_DATABASE")

PREFIX = "dedup"
METHOD = "Findmatches"

default_args: dict[str, str] = {}


@dag(
    default_args=default_args,
    start_date=datetime.datetime(2024, 2, 20),
    tags=["dedup", "findmatches"],
    on_success_callback=SuccessCallback,
    on_failure_callback=FailureCallback,
)
def dedup_findmatches_pipeline():
    """An Airflow DAG for Dedup using AWS Findmatches."""

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

            # III. FindmatchesGlueJob
            findmatches = GlueJobOperator(
                task_id=f"findmatches_{poi_type}",
                job_name="dedup_ml_findmatches",
                script_args={
                    "--INPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/prefiltering/{poi_type}/{run_version}/",
                    "--OUTPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/findmatches/{poi_type}/{run_version}/",
                    "--TRANSFORM_ID": TRANSFORM_ID,
                },
                wait_for_completion=False,
            )

            wait_for_findmatches = GlueJobSensor(
                task_id=f"wait_for_findmatches_{poi_type}",
                job_name="dedup_ml_findmatches",
                run_id=findmatches.output,
            )

            # IV. MarkDuplicatesGlueJob
            markduplicates = GlueJobOperator(
                task_id=f"markduplicates_{poi_type}",
                job_name="dedup_ml_markduplicates",
                script_args={
                    "--INPUT_S3_URI": f"s3://{S3_BUCKET}/{PREFIX}/findmatches/{poi_type}/{run_version}/",
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
                >> findmatches
                >> wait_for_findmatches
                >> markduplicates
                >> wait_for_markduplicates
            )

    table_name >> run_version >> process_individual_poi_types


dedup_findmatches_pipeline = dedup_findmatches_pipeline()
