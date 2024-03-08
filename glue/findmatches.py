import sys

import pyspark.sql.functions as F
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsglueml.transforms import FindMatches
from pyspark.context import SparkContext
from pyspark.sql import Window


class FindmatchesGlueJob:
    def __init__(self) -> None:
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "INPUT_S3_URI",
                "OUTPUT_S3_URI",
                "TRANSFORM_ID",
            ],
        )

        self.input_s3_uri = args["INPUT_S3_URI"]
        self.output_s3_uri = args["OUTPUT_S3_URI"]
        self.transform_id = args["TRANSFORM_ID"]

        sc = SparkContext()
        self.context = GlueContext(sc)
        self.spark = self.context.spark_session
        self.job = Job(self.context)
        self.job.init(args["JOB_NAME"], args)

    @staticmethod
    def add_keys(row: dict) -> dict:
        """
        Add key for each pair of POIs.

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with unique key
        Notes:
            Findmatches require unique key for each POI. Define key as a concatenated string
            of compact_ids of POI pair.
        """
        key0, key1 = None, None
        if row.get("compact_id0") and row.get("compact_id1"):
            key0 = row["compact_id0"] + "," + row["compact_id1"]
            key1 = row["compact_id1"] + "," + row["compact_id0"]

        row["key0"], row["key1"] = key0, key1
        return row

    @staticmethod
    def generate_match_id(row: dict) -> dict:
        """
        Generate new match_id based on Findmatches-generated match_id and key

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with new match_id
        Notes:
            new match_id contains information of original id of POIs and Findmatches-generated match_id.
            It will be used when splitting POI pairs into list of single POIs
        """
        match_id = None

        if row.get("key"):
            key0, key1 = row["key"].split(",")
            if key1 < key0:
                key0, key1 = key1, key0

            match_id = f"{key0}*{key1}*{row['match_id']}"

        row["match_id"] = match_id
        return row

    def run(self) -> None:
        """ML-based Dedup Pipeline.

        This module is the third step of the findmatches pipeline, that is to predict duplicate pairs using Findmatches.
        There are three main steps:
        Step I: Convert input data to data format required by Findmatches
        Step II. Apply Findmatches transform
        Step III. Refine duplicates based on paris of POIs

        Input:
            POI paris with attributes compact_id0, name0, addr0, lat0, lon0,
                                    compact_id1, name1, addr1, lat1, lon1,
                                    pair_id, name_jaro, addr_jaro
            where name_jaro and addr_jaro are similarity scores based on jaro_winkler.

        Output:
            POIs with attribute match_id
        """

        # 0. Get the input data
        dyf = self.context.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={"paths": [self.input_s3_uri]},
            format="json",
            format_options={
                "multiline": True,
            },
        )

        # I. Convert to data format required by Findmatches

        # Findmatches require a list of POIs with the format:
        # name, addr, lat, lon, key
        # where key is a unique id for each POI

        # Note:
        #     Although we have compact_id in input data, which can be a candidate for key. However,
        #     since we want to predict if any pair of POIs is identical, we will create a key that
        #     containscompact_id of both POIs within a pair, and recover compact_id from key.

        #     For example, if input data have
        #     compact_id0, compact_id1
        #     123, 345
        #     123, 489
        #     489, 567

        #     After adding keys,
        #     compact_id0, compact_id1, key0, key1
        #     123, 345, 123-345, 345-123
        #     123, 489, 123-489, 489-123
        #     489, 567, 489-567, 567-489

        #     Then convert to dataformat below for Findmatches:
        #     compact_id key
        #     123 123-345
        #     345 345-123
        #     123 123-489
        #     489 489-123
        #     489 489-567
        #     567 567-489

        # Add key
        dyf = dyf.map(
            f=FindmatchesGlueJob.add_keys,
        )

        # Generate input data for FindMatches
        dyf0 = dyf.select_fields(paths=["name0", "addr0", "lat0", "lon0", "key0"])
        dyf0 = (
            dyf0.rename_field("name0", "name")
            .rename_field("addr0", "addr")
            .rename_field("lat0", "lat")
            .rename_field("lon0", "lon")
            .rename_field("key0", "key")
        )

        dyf1 = dyf.select_fields(paths=["name1", "addr1", "lat1", "lon1", "key1"])
        dyf1 = (
            dyf1.rename_field("name1", "name")
            .rename_field("addr1", "addr")
            .rename_field("lat1", "lat")
            .rename_field("lon1", "lon")
            .rename_field("key1", "key")
        )

        df = dyf0.toDF().union(dyf1.toDF())
        dyf = DynamicFrame.fromDF(df, self.context, "dyf")

        # II. Apply FindMatches transform

        # Findmatches will give match_id besides the origianl schema. For example, if Findmatches predict
        # compact_id 123, 345, 567 are identical, the result will look like
        #     compact_id key match_id
        #     123 123-345 A
        #     345 345-123 A
        #     123 123-489 A
        #     489 489-123 B
        #     489 489-567 B
        #     567 567-489 A
        # However, the pair (123, 567) should not be considered. So we add the sorted pair of compact_id to match_id, i.e.,
        #     compact_id key match_id
        #     123 123-345 123-345-A
        #     345 345-123 123-345-A
        #     123 123-489 123-489-A
        #     489 489-123 123-489-B
        #     489 489-567 489-567-B
        #     567 567-489 489-567-A
        #     This new definition of match_id will only suggest 123 and 345 are identical.
        dyf_marked_duplicates = FindMatches.apply(
            dyf,
            transformId=self.transform_id,
            transformation_ctx="dyf_marked_duplicates",
            computeMatchConfidenceScores=False,
        )

        dyf_marked_duplicates = dyf_marked_duplicates.map(
            f=FindmatchesGlueJob.generate_match_id,
        )

        # III. Select Duplicate Pairs (i.e., match_id count >1)
        w = Window.partitionBy("match_id")
        df = dyf_marked_duplicates.toDF().select(
            "match_id", F.count("match_id").over(w).alias("count")
        )
        df = df.filter(df["count"] > 1).drop("count").distinct()

        # VII. Write to s3
        dyf = DynamicFrame.fromDF(df, self.context, "dyf")
        self.context.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="s3",
            format="json",
            connection_options={"path": self.output_s3_uri},
        )

        self.job.commit()


if __name__ == "__main__":
    FindmatchesGlueJob().run()
