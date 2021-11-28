from typing import Text
import pandas as pd

from TextPreprocessing import TextPreprocessing
from DataAnalysis import DataAnalysis
from FeatureEngineering import FeatureEngineering
from model_training import ModelTraining
from sklearn.model_selection import train_test_split


class DataPrepration:
    def __init__(self , df):
        self.df = df

    def data_prepration(self , df):

        df_copy = df.copy()
        '''
        Step 1 - Data Analysis. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)
        '''

        # da = DataAnalysis()
        # da.explore_dataset(True)
        # da.plot_wordcloud()
        '''
        Step 2 - Text Preprocessing. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

        '''
        tp = TextPreprocessing(df_copy, 'question_text')
        df_copy = tp.GetDataFrame()

        '''

        Step 3 - Feature Engineering. (Highlight the portion of code below and Use Ctrl + / to uncomment it and see results.)

        '''
        fe = FeatureEngineering(df_copy)
        df_copy = fe.add_more_features(df_copy)

        # remove qid column and split the df into X , Y
        df_copy = df_copy.drop('qid' , axis=1)
        X = df_copy.drop('target' , axis=1)
        Y = df_copy['target']

        # spliting the X , Y in training and testing set
        x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size=0.2 , random_state=1)

        # Extract features with class functions now. Train and test sets are passed with slicing.
        x_train, x_test = fe.extract_features(x_train , x_test)

        # print(x_train_fe.head())
        # print(x_test_fe.head())

        return x_train , x_test , y_train , y_test
