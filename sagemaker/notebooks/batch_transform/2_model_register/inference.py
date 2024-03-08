import pandas as pd
import os
# from sklearn.ensemble import RandomForestClassifier
import joblib
from io import StringIO
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import make_pipeline


# if __name__ =='__main__':
#     # Load data from the CSV file
#     df = pd.read_csv('/opt/ml/input/data/train/train.csv')
#     X = df.drop('match', axis=1)
#     y = df['match']

#     # Train the model
#     num_imputer = SimpleImputer(strategy="median")
#     model = make_pipeline(num_imputer, RandomForestClassifier(max_depth=2, random_state=0))
#     model.fit(X, y)

#     # Save the model to the /opt/ml/model directory
#     joblib.dump(model, '/opt/ml/model/model.joblib')


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(input_data, content_type):
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        features = ['name_gesh', 'name_jaro', 'name_tfidf', 'dist_geod', 'addr_gesh', 'addr_jaro', 'addr_tfidf']
        df = pd.read_csv(StringIO(input_data))
        df = df[features]
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))

"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    res = model.predict(input_data)
    return res

def output_fn(prediction, content_type):
    """
    Format the output of the model to include column names.
    """
    if content_type == "text/csv":
        # Assuming a single column output here. Modify as needed.
        result = pd.DataFrame(prediction, columns=['match'])
        return result.to_csv(index=False)
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
