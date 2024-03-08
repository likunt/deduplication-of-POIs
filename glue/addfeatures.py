import string
import sys
from difflib import SequenceMatcher
from typing import Callable

import numpy as np
import pyspark.sql.functions as F
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from geopy.distance import geodesic
from Levenshtein import distance, jaro_winkler
from pyspark.context import SparkContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

FEATURES_STR = ["name0", "name1", "addr0", "addr1"]
FEATURES_DIST = ["dist_eucd", "dist_geod"]

class AddFeaturesGlueJob:
    def __init__(self) -> None:
        args = getResolvedOptions(
            sys.argv,
            [
                "JOB_NAME",
                "INPUT_S3_URI",
                "OUTPUT_S3_URI",
                "FILL_VALUE",
            ],
        )

        self.input_s3_uri = args["INPUT_S3_URI"]
        self.output_s3_uri = args["OUTPUT_S3_URI"]
        self.fill_value = args["FILL_VALUE"]
        # Create TfidfVectorizer object
        token_pattern = r"(?u)\b\w+\b|\b\d+\b"
        self.vectorizer = TfidfVectorizer(token_pattern=token_pattern)
        
        sc = SparkContext()
        self.context = GlueContext(sc)
        self.spark = self.context.spark_session
        self.job = Job(self.context)
        self.job.init(args["JOB_NAME"], args)

    def add_name_features(self) -> Callable[[dict], dict]:
        """
        Add similarity scores of name strings

        Args:
            self: instance of AddFeaturesGlueJob class
        Returns:
            Callable[[dict], dict]: a function that maps current row to a row with additional similarity scores of name
        """
        vec = self.vectorizer

        def mapping_function(row: dict) -> dict:
            if not (row.get("name0") and row.get("name1")):
                row.update(
                    {
                        "name_gesh": None,
                        "name_leven": None,
                        "name_jaro": None,
                        "name_tfidf": None,
                    }
                )
                return row

            # gesh
            row["name_gesh"] = SequenceMatcher(None, row["name0"], row["name1"]).ratio()
            # leven
            maxlen = max(len(row["name0"]), len(row["name1"]))
            row["name_leven"] = float(
                maxlen - distance(row["name0"], row["name1"])
            ) / float(maxlen)
            # jaro
            row["name_jaro"] = jaro_winkler(row["name0"], row["name1"])
            # tfidf
            if row["name0"] == row["name1"]:
                row["name_tfidf"] = 1.0
            else:
                m = vec.fit_transform([row["name0"], row["name1"]])
                cosine_sim = cosine_similarity(m)
                row["name_tfidf"] = float(cosine_sim[0][1])

            return row

        return mapping_function

    @staticmethod
    def add_distance_features(row: dict) -> dict:
        """
        Add distance measures of lat/lon

        Args:
            row (dict): the row to format
        Returns:
            dict: the modified row, with euclidean and geodesic distances
        """
        if not (
            row.get("lat0") and row.get("lat1") and row.get("lon0") and row.get("lon1")
        ):
            row.update(
                {
                    "dist_eucd": None,
                    "dist_geod": None,
                }
            )
            return row

        # euclidean distance
        P1 = np.array((row["lat0"], row["lon0"]))
        P2 = np.array((row["lat1"], row["lon1"]))
        # subtracting both the vectors
        row["dist_eucd"] = (
            (row["lat0"] - row["lat1"]) ** 2 + (row["lon0"] - row["lon1"]) ** 2
        ) ** 0.5
        row["dist_geod"] = geodesic(P1, P2).km

        return row

    def add_address_features(self) -> Callable[[dict], dict]:
        """
        Add similarity scores of address strings

        Args:
            self: instance of AddFeaturesGlueJob class
        Returns:
            Callable[[dict], dict]: a function that maps current row to a row with additional similarity scores of address
        """
        vec = self.vectorizer
        const = self.fill_value

        def mapping_function(row: dict) -> dict:
            if not (row.get("addr0") and row.get("addr1")):
                row.update(
                    {
                        "addr_gesh": const,
                        "addr_leven": const,
                        "addr_jaro": const,
                        "addr_tfidf": const,
                    }
                )
                return row

            # gesh
            row["addr_gesh"] = SequenceMatcher(None, row["addr0"], row["addr1"]).ratio()
            # leven
            maxlen = max(len(row["addr0"]), len(row["addr1"]))
            row["addr_leven"] = float(
                maxlen - distance(row["addr0"], row["addr1"])
            ) / float(maxlen)
            # jaro
            row["addr_jaro"] = jaro_winkler(row["addr0"], row["addr1"])
            # tfidf
            if row["addr0"] == row["addr1"]:
                row["addr_tfidf"] = 1.0
            else:
                m = vec.fit_transform([row["addr0"], row["addr1"]])
                cosine_sim = cosine_similarity(m)
                row["addr_tfidf"] = float(cosine_sim[0][1])

            return row

        return mapping_function

    def run(self) -> None:
        """ML-based Dedup Pipeline.

        This module is the third step of the custom pipeline, that is to calculate similarity measures of name and address.

        Input:
            POI pairs with attributes compact_id0, name0, addr0, lat0, lon0,
                                    compact_id1, name1, addr1, lat1, lon1,
                                    pair_id, name_jaro, addr_jaro
            where name_jaro and addr_jaro are similarity scores based on jaro_winkler.

        Output:
            POI paris with attributes compact_id0, compact_id1, dist_geod, dist_eucd,
                                    name_gesh, name_leven, name_jaro, name_lcss, name_tfidf,
                                    addr_gesh, addr_leven, addr_jaro, addr_lcss, addr_tfidf
        """

        # 0. get the input data
        dyf = self.context.create_dynamic_frame.from_options(
            connection_type="s3",
            connection_options={"paths": [self.input_s3_uri]},
            format="json",
            format_options={
                "multiline": True,
            },
        )

        dyf = dyf.drop_fields(paths=["pair_id", "name_jaro", "addr_jaro"])

        # I. Add name features
        dyf = dyf.map(
            f=self.add_name_features(),
        )

        # II. Add distance features
        dyf = dyf.map(
            f=self.add_distance_features,
        )

        # III. Add addr features
        dyf = dyf.map(
            f=self.add_address_features(),
        )

        dyf = dyf.drop_fields(
            paths=["name0", "addr0", "lat0", "lon0", "name1", "addr1", "lat1", "lon1"]
        )

        # IV. Feature Engineering of distance features
        df = dyf.toDF()

        # 1. winsorizer
        for col in FEATURES_DIST:
            quantiles = df.stat.approxQuantile(col, [0.25, 0.75], 0.05)
            q1, q3 = quantiles
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            df = df.withColumn(
                col, F.when(F.col(col) < lower_bound, lower_bound).otherwise(F.col(col))
            )
            df = df.withColumn(
                col, F.when(F.col(col) > upper_bound, upper_bound).otherwise(F.col(col))
            )

        # 2. minmaxscaler
        for col in FEATURES_DIST:
            mi, mx = df.select(F.min(col), F.max(col)).first()
            df = df.withColumn(col, (F.col(col) - mi) / (mx - mi))

        # V. Write to S3
        dyf = DynamicFrame.fromDF(df, self.context, "dyf")
        dyf = dyf.repartition(10000)

        self.context.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="s3",
            format="csv",
            connection_options={"path": self.output_s3_uri},
        )

        self.job.commit()


if __name__ == "__main__":
    AddFeaturesGlueJob().run()
