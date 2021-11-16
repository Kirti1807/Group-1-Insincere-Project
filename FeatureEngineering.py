''' 

https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 

3rd bullet point in project proposal file, which mentions pre processing for the text.

'''

import pandas as pd
import nltk

class FeatureEngineering:
    #just pass dataframe so that entire class can work on it
    def __init__(self , df):
        self.df = df

    '''
        This function is for extend of dataset. 
        We will add more features so that training of model becomre easier and 
        it can prevent from underfitting due to lack of features.
    '''
    def add_more_features(self , df):
        df_copy = df.copy()

        ''' 
            This feature calculates the number of sentences in each data point. 
            It is not much useful here because due to removal of panctuation in data preprocessing all words count in one sentence.
        '''
        df_copy['total_sentences'] = df_copy['question_text'].apply( lambda text: len(nltk.sent_tokenize(text)))


        ''' 
            This feature counts the number of words in our text.
        '''
        df_copy['Number_of_words'] = df_copy['question_text'].apply( lambda text: len(nltk.word_tokenize(text)))


        ''' 
            This feature counts the number of all unique words in our text. 
        '''
        df_copy['Number_of_unique_words'] = df_copy['question_text'].apply( lambda text: len(set(nltk.word_tokenize(text))))


        ''' 
            This feature counts the total number of characters in text it's mean it calculates the total length of text.
        '''
        df_copy['number_of_characters'] = df_copy['question_text'].apply(lambda text: len(text))

        ''' 
            This feature calculates the avarage word length of each word in text.
        '''
        df_copy['number of characters per words'] = df_copy['number_of_characters'] / df_copy['Number_of_words']

        return df_copy
        

