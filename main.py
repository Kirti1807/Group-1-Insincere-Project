from typing import Text
import pandas as pd

from TextPreprocessing import TextPreprocessing
from DataAnalysis import DataAnalysis
from FeatureEngineering import FeatureEngineering
from model_training import ModelTraining
from sklearn.model_selection import train_test_split


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

# remove qid column and split the df into X , Y
df = df.drop('qid' , axis=1)
X = df.drop('target' , axis=1)
Y = df['target']

# spliting the X , Y in training and testing set
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2 , random_state=1)

# Extract features with class functions now. Train and test sets are passed with slicing.
x_train_fe, x_test_fe = fe.extract_features(x_train , x_test)

# print(x_train_fe.head())
# print(x_test_fe.head())





# print(x_train_fe.shape , x_test_fe.shape , y_train.shape , y_test.shape)

'''

Step 4 - Model training. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

'''


# Get the training samples ready, instantiate the class as well.
mt = ModelTraining(x_train_fe, y_train)

logR = mt.logistic()
