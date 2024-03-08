import sys

import pyspark.sql.functions as F
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from graphframes import GraphFrame
from pyspark.context import SparkContext


@F.udf()
def get_key0(match_id) -> str:
    key0 = match_id.split("*")[0]
    return key0


@F.udf()
def get_key1(match_id) -> str:
    key1 = match_id.split("*")[1]
    return key1


class MarkDuplicatesGlueJob:
    def __init__(self) -> None:
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "INPUT_S3_URI",
                "OUTPUT_S3_URI",
                "INPUT_CATALOG_DATABASE",
                "INPUT_CATALOG_TABLE",
                "INPUT_CATALOG_PREDICATE",
            ],
        )

        self.input_s3_uri = args["INPUT_S3_URI"]
        self.output_s3_uri = args["OUTPUT_S3_URI"]
        self.input_catalog_database = args["INPUT_CATALOG_DATABASE"]
        self.input_catalog_table = args["INPUT_CATALOG_TABLE"]
        self.input_catalog_predicate = args["INPUT_CATALOG_PREDICATE"]

        sc = SparkContext()
        sc.setCheckpointDir("s3a://aws-glue-shuffle-us-east-1-769026081114/checkpoint/")
        self.context = GlueContext(sc)
        self.spark = self.context.spark_session
        self.job = Job(self.context)
        self.job.init(args["JOB_NAME"], args)

    @staticmethod
    def generate_id(row: dict) -> dict:
        """
        Generate compact_id as data source code + id.

        Note:
            - When applying FindMatches on POIs, it will assign a unique match_id to duplicate POIs.
            However, we need to exclude the predicted duplicates that are not in pairs. compact_id is
            used for definition of POI pairs and refining the results from FindMatches.

            - For internal model, we also need a unique id for each POI when generating pairs for inference and
            recovering POI id
        """

        if not row.get("id"):
            row["compact_id"] = None
            return row

        if row["id"].get("tripadvisor"):
            idx = "1" + str(row["id"]["tripadvisor"])
        elif row["id"].get("stb"):
            idx = "2" + str(row["id"]["stb"])
        elif row["id"].get("osm_way"):
            idx = "3" + str(row["id"]["osm_way"])
        elif row["id"].get("osm_node"):
            idx = "4" + str(row["id"]["osm_node"])
        elif row["id"].get("expedia") or row["id"].get("hcom") or row["id"].get("vrbo"):
            idx = "5" + str(row["id"]["expedia"])
        elif row["id"].get("viator"):
            idx = "6" + str(row["id"]["viator"])

        row["compact_id"] = idx

        return row

    @staticmethod
    def assign_root(row: dict) -> dict:
        """
        Use root as match_id for each POI

        Note:
            For leaf nodes that share the same root, we regard them as duplicates.
        """
        if row.get("root") and row.get("match_id"):
            row["match_id"] = row["root"]

        return row

    def run(self) -> None:
        """ML-based Dedup Pipeline.

        This module is the fourth step of the pipeline, that is to merge Findmatches results with base dataset

        Input:
            POIs with attribute match_id

        Output:
            base dataset schema + match_id
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
        df = dyf.toDF()

        # I. Find root for all nodes

        # Suppose now we have all the duplicate pairs and df takes a form as
        # match_id
        # 123-345-A
        # 123-489-A
        # 223-256-C
        # 256-335-D

        # Then 123, 345 and 489 are identical, and 223, 256 and 335 are identical. We should assign all the
        # identical POIs in a cluster with the same match_id

        # recover key from match_id
        df = df.withColumn("key0", get_key0(df.match_id))
        df = df.withColumn("key1", get_key1(df.match_id))

        # use GraphFrame to find root for all nodes
        df_v = df.select("key0").union(df.select("key1")).distinct()
        df_v = df_v.withColumnRenamed("key0", "id")
        df = df.withColumnRenamed("key0", "src").withColumnRenamed("key1", "dst")

        # Create a GraphFrame
        g = GraphFrame(df_v, df)
        result = g.connectedComponents()
        df_id_root = result.select("id", "component").withColumnRenamed(
            "component", "root"
        )

        # define a column root ("r"+"component")
        df_id_root = df_id_root.withColumn(
            "root", F.concat(F.lit("r"), df_id_root["root"])
        )
        df_id_root = df_id_root.withColumnRenamed("id", "node")

        # II. Load base_dataset and assign compact_id as match_id
        dyf_base = self.context.create_dynamic_frame.from_catalog(
            database=self.input_catalog_database,
            table_name=self.input_catalog_table,
            push_down_predicate=self.input_catalog_predicate,
        )

        # drop existing match_id
        dyf_base = dyf_base.drop_fields(paths=["match_id"])

        dyf_base = dyf_base.map(
            f=MarkDuplicatesGlueJob.generate_id,
        )

        dyf_base = dyf_base.rename_field("compact_id", "match_id")
        df_with_index = dyf_base.toDF()

        # III. Join df_id_root with base dataset
        df_with_index = df_with_index.join(
            df_id_root, df_with_index.match_id == df_id_root.node, "outer"
        )
        dyf_marked_duplicates = DynamicFrame.fromDF(
            df_with_index, self.context, "dyf_marked_duplicates"
        )

        # update match_id to root
        dyf_marked_duplicates = dyf_marked_duplicates.map(
            f=MarkDuplicatesGlueJob.assign_root,
        )

        dyf_marked_duplicates = dyf_marked_duplicates.drop_fields(
            paths=["root", "node"]
        )

        # IV. Write to s3
        self.context.write_dynamic_frame.from_options(
            frame=dyf_marked_duplicates,
            connection_type="s3",
            format="parquet",
            connection_options={"path": self.output_s3_uri},
        )

        self.job.commit()


if __name__ == "__main__":
    MarkDuplicatesGlueJob().run()
