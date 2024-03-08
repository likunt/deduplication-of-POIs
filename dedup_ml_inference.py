import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext


class InferenceGlueJob:
    def __init__(self) -> None:
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "INPUT_S3_URI",
                "OUTPUT_S3_URI",
            ],
        )

        self.input_s3_uri = args["INPUT_S3_URI"]
        self.output_s3_uri = args["OUTPUT_S3_URI"]

        sc = SparkContext()
        self.context = GlueContext(sc)
        self.spark = self.context.spark_session
        self.job = Job(self.context)
        self.job.init(args["JOB_NAME"], args)

    @staticmethod
    def generate_match_id(row: dict) -> dict:
        """
        Generate match_id as concatenated string of compact_id0 and compact_id1

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with match_id
        """

        match_id = None

        if row.get("compact_id0") and row.get("compact_id1"):
            key0, key1 = row["compact_id0"], row["compact_id1"]
            if key1 < key0:
                key0, key1 = key1, key0
            match_id = key0 + "*" + key1

        row["match_id"] = match_id
        return row

    def run(self) -> None:
        """ML-based Dedup Pipeline.

        This module is the fourth step of the custom pipeline, that is to postprocess output from batch inference

        Input:
            compact_id0, compact_id1, match

        Output:
            match_id in the form of (compact_id0, compact_id1)
        """

        # 0. Get the input data
        dyf = self.context.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={"paths": [self.input_s3_uri]},
            format="csv",
            format_options={"withHeader": True},
        )

        # I. Filter matched pairs
        dyf = dyf.filter(lambda row: row["match"] == "1")

        # II. Generate match_id
        dyf = dyf.map(
            f=InferenceGlueJob.generate_match_id,
        )

        dyf = dyf.select_fields(paths=["match_id"])

        # III. Write to S3
        self.context.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="s3",
            format="json",
            connection_options={"path": self.output_s3_uri},
        )

        self.job.commit()


if __name__ == "__main__":
    InferenceGlueJob().run()
