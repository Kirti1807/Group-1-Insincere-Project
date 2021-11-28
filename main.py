from typing import Text
import pandas as pd

from TextPreprocessing import TextPreprocessing
from DataAnalysis import DataAnalysis
from FeatureEngineering import FeatureEngineering
from model_training import ModelTraining
from sklearn.model_selection import train_test_split
from PrepareData import DataPrepration
from evaluate import EvaluateModel
'''  

Group 1 Insinscere project!

All group members can definitely contribute to the project. Please be careful for merge conflicts though! If 2 or more
people are working on the exact same file and try to push to github, there will be problems with the way Github views
the changed files!

            File information below.

            1) train.csv - 1,306,122 rows & 3 columns.

'''



''' The real file with a million plus rows is called 'train.csv', but 'customData.csv' is placed in for quick testing.
    Text feature/column name is 'question_text'. '''
df = pd.read_csv('customData.csv', encoding='latin-1') 

'''
    all data preprocessing and feature engineering done in DataPrepration class and return finale training and testing dat
'''
dp = DataPrepration(df)

x_train , x_test , y_train , y_test = dp.data_prepration(df)

'''

Step 4 - Model training. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

'''


# Get the training samples ready, instantiate the class as well.
mt = ModelTraining(x_train, y_train)
# ============================================ XGboost Model =========================================
XGboostModel = mt.Xgboost_model(fine_tuning=False)
predict = XGboostModel.predict(x_test)
# check the unique values in predict
# ============================================ XGboost Model =========================================
evaluate = EvaluateModel(x_test, y_test, XGboostModel)
evaluate.evaluate_model()
evaluate.plot_confusion_matrix(y_test, XGboostModel.predict(x_test))
evaluate.plot_roc_curve(y_test, XGboostModel.predict_proba(x_test)[: , 1])