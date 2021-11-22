from typing import Text
import pandas as pd

from TextPreprocessing import TextPreprocessing
from DataAnalysis import DataAnalysis
from FeatureEngineering import FeatureEngineering
from model_training import ModelTraining


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

Step 1 - Data Analysis. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

'''
# da = DataAnalysis()
# da.explore_dataset(True)
# da.plot_wordcloud()




'''

Step 2 - Text Preprocessing. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

'''
tp = TextPreprocessing(df, 'question_text')
df = tp.GetDataFrame()






'''

Step 3 - Feature Engineering. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

'''
fe = FeatureEngineering(df)
df = fe.add_more_features(df)

# Shuffle the dataframe.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features with class functions now. Train and test sets are passed with slicing.
x_train, x_test = fe.extract_features(df[:3500], df[3500:])



'''

Step 4 - Model training. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

'''
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Get the training samples ready, instantiate the class as well.
mt = ModelTraining(x_train, x_test)

logR = mt.logistic()