import sys

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from h3 import h3
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql.functions import PandasUDFType, pandas_udf
from sklearn.neighbors import KDTree

FEATURES = ["compact_id", "name", "addr", "lat", "lon"]
FEATURES_STR = ["name", "addr", "compact_id"]
FEATURES_FLOAT = ["lat", "lon"]
K = 30  # nearest neighbors


def generate_feature_struct(number_features: int) -> T.StructType:
    struc_features = []
    for i in range(number_features):
        for feature in FEATURES:
            if feature in FEATURES_STR:
                struc_features.append(
                    T.StructField(f"{feature}{i}", T.StringType(), True)
                )
            elif feature in FEATURES_FLOAT:
                struc_features.append(
                    T.StructField(f"{feature}{i}", T.DoubleType(), True)
                )
    return T.StructType(struc_features)


SCHEMA_FEATURES = generate_feature_struct(2)

DIC1, DIC2 = {}, {}
for feature in FEATURES:
    DIC1[feature] = feature + "0"
    DIC2[feature] = feature + "1"


@pandas_udf(SCHEMA_FEATURES, PandasUDFType.GROUPED_MAP)
def generate_pairs(pdf):
    """
    Given a list of POIs, find K-nearest neighbors of each POI based on lat and lon.
    Return all pairs of POIs with FEATURES.

    Args:
        pdf (pandas DataFrame): POI dataset
    Returns:
        pandas DataFrame: pairs of POIs with FEATURES
    """
    size = min(K, len(pdf))
    X = pdf[["lat", "lon"]]
    tree = KDTree(X, leaf_size=30)
    _, indices = tree.query(X, k=size)

    arr = np.empty((0, 2))
    for i in range(1, size):
        tmp = np.array(list(zip(indices[:, 0], indices[:, i])))
        arr = np.concatenate([arr, tmp])
    df1 = pdf[FEATURES].rename(columns=DIC1).iloc[arr[:, 0]].reset_index(drop=True)
    df2 = pdf[FEATURES].rename(columns=DIC2).iloc[arr[:, 1]].reset_index(drop=True)

    return pd.concat([df1, df2], axis=1)


class GeneratePairsGlueJob:
    def __init__(self) -> None:
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "INPUT_CATALOG_DATABASE",
                "INPUT_CATALOG_TABLE",
                "INPUT_CATALOG_PREDICATE",
                "OUTPUT_S3_URI",
            ],
        )

        self.input_catalog_database = args["INPUT_CATALOG_DATABASE"]
        self.input_catalog_table = args["INPUT_CATALOG_TABLE"]
        self.input_catalog_predicate = args["INPUT_CATALOG_PREDICATE"]
        self.output_s3_uri = args["OUTPUT_S3_URI"]

        sc = SparkContext()
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
    def update_address(row: dict) -> dict:
        """
        Extract address from source data.

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with extracted address
        Notes:
            base dataset does not have the attribute address. Address is extracted from subfield of struct data.
        """

        address = None

        if row.get("address") and row["address"].get("address1"):
            address = row["address"]["address1"]

        elif row.get("additional_fields") and row["additional_fields"].get(
            "tripadvisor"
        ):
            if row["additional_fields"]["tripadvisor"].get("address"):
                for dic in row["additional_fields"]["tripadvisor"]["address"]:
                    if dic["lang"] == "en":
                        address = dic["value"]
                        address = address.split(",")[0] if address else None

        row["addr"] = address
        return row

    @staticmethod
    def generate_h3(row: dict) -> dict:
        """
        Add h3_index based on lat, lon and resolution.

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with h3_index
        Notes:
            h3 index is used to divide global datasets into block and easily search for nearest neighbors
        """
        h3_idx = None

        if row.get("lat") and row.get("lon"):
            h3_idx = h3.geo_to_h3(row["lat"], row["lon"], 3)

        row["h3_index"] = h3_idx
        return row

    @staticmethod
    def add_pair_ids(row: dict) -> dict:
        """
        Add pair_id based on compact_id for detecting duplicate POI pairs.
        For example, (POI1, POI3) and (POI3, POI1).

        Note:
            pair_id is for temporary use and will be deleted.
        """

        pair_id = None

        if row.get("compact_id0") and row.get("compact_id1"):
            if row["compact_id0"] < row["compact_id1"]:
                pair_id = row["compact_id0"] + "-" + row["compact_id1"]
            else:
                pair_id = row["compact_id1"] + "-" + row["compact_id0"]

        row["pair_id"] = pair_id

        return row

    def run(self) -> None:
        """ML-based Dedup Pipeline.

        This module is the first step of the findmatches/custom pipeline, that is to generate
        pairs of POIs of k-nearest neighors

        Input:
            raw data in mobi.source.data.batch/base/marked_duplicates

        Output:
        - pairs/:
            POI paris with attributes compact_id0, name0, addr0, lat0, lon0,
                        compact_id1, name1, addr1, lat1, lon1, pair_id
            where compact_id (str) is a unique id of POI and pair_id (str) is sorted string compact_id0-compact_id1

        - base/:
            base dataset + compact_id (str) + h3_index (str)
        """

        # 0. Get the input data
        dyf = self.context.create_dynamic_frame.from_catalog(
            database=self.input_catalog_database,
            table_name=self.input_catalog_table,
            push_down_predicate=self.input_catalog_predicate,
        )

        # I. Add address
        dyf = dyf.map(
            f=GeneratePairsGlueJob.update_address,
        )

        # II. Add h3_index
        dyf = dyf.map(
            f=GeneratePairsGlueJob.generate_h3,
        )

        # III. Add compact_id and select only related attributes
        dyf = dyf.map(
            f=GeneratePairsGlueJob.generate_id,
        )
        df_with_index = dyf.toDF().select(
            ["h3_index", "name", "addr", "lat", "lon", "compact_id"]
        )

        # IV. Remove h3 block whose number of POIs is less than K
        w = Window.partitionBy("h3_index")
        df_with_index = df_with_index.select(
            "h3_index",
            "name",
            "addr",
            "lat",
            "lon",
            "compact_id",
            F.count("h3_index").over(w).alias("count"),
        )

        df_with_index = df_with_index.filter(df_with_index["count"] >= 2).drop("count")

        # V. Generate POI pairs
        df_pairs = df_with_index.groupby("h3_index").apply(generate_pairs)
        dyf = DynamicFrame.fromDF(df_pairs, self.context, "dyf")

        # VI. Remove duplicate pairs
        dyf = dyf.map(
            f=GeneratePairsGlueJob.add_pair_ids,
        )
        df_pairs = dyf.toDF().dropDuplicates(["pair_id"])

        # VII. Write to s3
        dyf = DynamicFrame.fromDF(df_pairs, self.context, "dyf")
        self.context.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="s3",
            format="json",
            connection_options={"path": self.output_s3_uri},
        )

        self.job.commit()


if __name__ == "__main__":
    GeneratePairsGlueJob().run()
