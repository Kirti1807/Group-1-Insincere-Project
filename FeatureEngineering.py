''' 
https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 
3rd bullet point in project proposal file, which mentions pre processing for the text.
'''

from numpy import vectorize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
            It is not much useful here because due to removal of punctuation in data preprocessing all words count in one sentence.
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


    # This function extracting features by using TF-IDF vector
    def extract_features(self , x_train , x_test):
        #initilize the tf-idf vector
        vectorizer = TfidfVectorizer()

        # Assigning the vector ID's
        extracted_data = list(
            vectorizer.fit_transform(x_train['question_text']).toarray()
        )

        # Creating dataframe from vector ID's 
        extracted_data = pd.DataFrame(extracted_data)
        extracted_data.head()

        # Assign the column name to the features
        extracted_data.columns = vectorizer.get_feature_names()


        vocab = vectorizer.vocabulary_
        mapping = vectorizer.get_feature_names()
        keys = list(vocab.keys())


        Modified_df = extracted_data.copy()
        print(Modified_df.shape)
        Modified_df.head()

        # Reset the index of new training data or old training set and then concatinate both of them
        Modified_df.reset_index(drop=True, inplace=True)
        x_train.reset_index(drop=True, inplace=True)

        # Create finel training set by concatinate both data sets
        Final_Training_data = pd.concat([x_train, Modified_df], axis=1)

        Final_Training_data.head()
        print(Final_Training_data.shape)

        # Remove the question_text column from final data set
        Final_Training_data.drop(["question_text"], axis=1, inplace=True)
        Final_Training_data.head()

        # storing the final training data set into new csv file
        Final_Training_data.to_csv("Final_Training_vectorized", index=False)

        # vectorizing the test data set
        dff_test = list(vectorizer.transform(x_test["question_text"]).toarray())

        #access the features name 
        vocab_test = vectorizer.vocabulary_
        keys_test = list(vocab_test.keys())

        # create data frame of vector ID's of testing set 
        dff_test_df = pd.DataFrame(dff_test, columns=keys_test)

        # Reset the index of new testing data or old testing set  
        dff_test_df.reset_index(drop=True, inplace=True)
        x_test.reset_index(drop=True, inplace=True)

        # Create final testing set by concatinate both data sets
        Final_Test = pd.concat([x_test, dff_test_df], axis=1)

        # Remove the question_text column fronm finale testing data 
        Final_Test.drop(["question_text"], axis=1, inplace=True)

        # Storing the fianl testing test into new csv file
        Final_Test.to_csv("Final_Test_vectorized", index=False)

        # returning final training and testing data set
        return Final_Training_data , Final_Test