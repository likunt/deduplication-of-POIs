import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
from io import StringIO
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


if __name__ =='__main__':
    # Load data from the CSV file
    df = pd.read_csv('/opt/ml/input/data/train/train.csv')
    X = df.drop('match', axis=1)
    y = df['match']

    # Train the model
    num_imputer = SimpleImputer(strategy="median")
   # clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf = GradientBoostingClassifier(loss='exponential', n_estimators=290, max_features='sqrt', subsample=0.4333521558952111, 
                                     max_depth=35, min_samples_split=6, min_samples_leaf=1, random_state=0)
    model = make_pipeline(num_imputer, clf)
    model.fit(X, y)

    # Save the model to the /opt/ml/model directory
    joblib.dump(model, '/opt/ml/model/model.joblib')


