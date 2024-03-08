import string
import sys
from typing import Callable

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from Levenshtein import jaro_winkler
from pyspark.context import SparkContext
from unidecode import unidecode
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

STOPWORDS_COMMON = ["the", "and", "by"]
STOPWORDS_LODGING = ["hotel", "hostel", "hostal", "inn", "motel", "resort", "lodge", "villa"]
FEATURES_STR = ["name0", "name1", "addr0", "addr1"]
FEATURES_ADDR = ["addr0", "addr1"]
THRESHOLD = 0.7
DIST_TOL = 0.01

ADDRESS_ABBREVIATIONS = {
    'rd': 'road',
    'st': 'street',
    'dr': 'drive',
    'ave': 'avenue',
    'blvd': 'boulevard',
    'ln': 'lane',
    'ct': 'court',
    'pl': 'place',
    'ter': 'terrace',
    'pkwy': 'parkway',
    'cir': 'circle',
    'jr': 'junior',
    'ne': 'northeast',
    'nw': 'northwest',
    'se': 'southeast',
    'sw': 'southwest',
    'e': 'east',
    'w': 'west',
    's': 'south',
    'n': 'north',
}

class PrefilterGlueJob:
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
        # Initialize the Porter Stemmer
        self.porter = PorterStemmer()

        sc = SparkContext()
        self.context = GlueContext(sc)
        self.spark = self.context.spark_session
        self.job = Job(self.context)
        self.job.init(args["JOB_NAME"], args)

    @staticmethod
    def preprocess_row(self) -> Callable[[dict], dict]:
        """
        Format name and address strings test

        Args:
            row (dict): the row to format
        Returns:
            dict: the modified row, with correct formatting.
        """
        porter = self.porter

        def mapping_function(row: dict) -> dict:
            for s in FEATURES_STR:
                if not row.get(s):
                    row[s] = None
                    continue

                s_preprocessed = unidecode(row[s])
                s_preprocessed = s_preprocessed.lower()
                s_preprocessed = "".join(
                    [char for char in s_preprocessed if char not in string.punctuation]
                )

                words = word_tokenize(s_preprocessed)
                lst = [porter.stem(w) for w in words]

                lst = [w for w in lst if w not in STOPWORDS]
                s_preprocessed = " ".join(lst)

                row[s] = s_preprocessed

            return row
        return mapping_function

    @staticmethod       
    def preprocess_address(row: dict) -> dict:
        """ 
        Format address strings

        Args:
            row (dict): the row to format
        Returns:
            dict: the modified row, with correct formatting.
        """
        for s in FEATURES_ADDR:
            if not row.get(s):
                row[s] = None
                continue

            # reorder number and street name
            numbers = re.findall(r"\d+", s)
            string_without_numbers = re.sub(r"\d+", "", s)
            s_preprocessed = " ".join(numbers) + " " + string_without_numbers

            # convert abbreviations
            words = s_preprocessed.split()
            expanded_words = [ADDRESS_ABBREVIATIONS.get(w.lower(), w) for w in words]

            row[s] =  " ".join(expanded_words)

        return row

    @staticmethod
    def get_dist_diff(row: dict) -> dict:
        """
        Calculate distance difference

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with absolute difference of lat/lon.
        """

        lat_diff, lon_diff = None, None

        if row.get("lat0") and row.get("lat1"):
            lat_diff = abs(row["lat1"] - row["lat0"])

        if row.get("lon0") and row.get("lon1"):
            lon_diff = abs(row["lon1"] - row["lon0"])

        row["lat_diff"], row["lon_diff"] = lat_diff, lon_diff

        return row

    @staticmethod
    def add_name_features(row: dict) -> dict:
        """
        Add jaro_winkler score of name strings

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with jaro_winkler score of name strings.
        """

        name_jaro = None

        if row.get("name0") and row.get("name1"):
            name_jaro = jaro_winkler(row["name0"], row["name1"])

        row["name_jaro"] = name_jaro

        return row

    @staticmethod
    def add_address_features(row: dict) -> dict:
        """
        Add jaro_winkler score of name strings

        Args:
            row (dict): current row
        Returns:
            dict: the modified row, with jaro_winkler score of address strings.
        """

        addr_jaro = None

        if row.get("addr0") and row.get("addr1"):
            addr_jaro = jaro_winkler(row["addr0"], row["addr1"])

        row["addr_jaro"] = addr_jaro

        return row

    def run(self) -> None:
        """ML-based Dedup Pipeline.

        This module is the second step of the findmatches/custom pipeline, that is to filter out
        pairs of POI with low similarity scores of name and address.

        Input:
        - pairs/:
            POI paris with attributes compact_id0, name0, addr0, lat0, lon0,
                                      compact_id1, name1, addr1, lat1, lon1,
                                    pair_id
            where compact_id is a unique id of POI and pair_id is sorted string compact_id0-compact_id1

        Output:
        - prefiltering/:
            POI paris with attributes compact_id0, name0, addr0, lat0, lon0,
                                    compact_id1, name1, addr1, lat1, lon1,
                                    pair_id, name_jaro, addr_jaro
            where name_jaro and addr_jaro are similarity scores based on jaro_winkler.
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

        # I. Preprocess name and address strings
        dyf = dyf.map(
            f=PrefilterGlueJob.preprocess_row,
        )

        dyf = dyf.map(
            f=PrefilterGlueJob.preprocess_address,
        )

        # II. Add similarity scores of name and address
        dyf = dyf.map(
            f=PrefilterGlueJob.add_name_features,
        )

        dyf = dyf.map(
            f=PrefilterGlueJob.add_address_features,
        )

        # III. Filter out POI pairs with similarity scores lower than THRESHOLD.
        dyf1 = dyf.filter(
            lambda row: row["name_jaro"] >= THRESHOLD and row["addr_jaro"] >= THRESHOLD
        )
        dyf2 = dyf.filter(
            lambda row: row["name_jaro"] >= THRESHOLD and row["addr_jaro"] is None
        )
        df = dyf1.toDF().union(dyf2.toDF())
        dyf = DynamicFrame.fromDF(df, self.context, "dyf")

        # IV. Filter out POI pairs with distance tolerance higher than DIST_TOL.
        dyf = dyf.map(
            f=PrefilterGlueJob.get_dist_diff,
        )

        dyf = dyf.filter(
            lambda row: row["lat_diff"] <= DIST_TOL and row["lon_diff"] <= DIST_TOL
        ).drop_fields(paths=["lat_diff", "lon_diff"])

        # V. Write to s3
        self.context.write_dynamic_frame.from_options(
            frame=dyf,
            connection_type="s3",
            format="json",
            connection_options={"path": self.output_s3_uri},
        )

        self.job.commit()


if __name__ == "__main__":
    PrefilterGlueJob().run()
