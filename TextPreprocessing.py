import string

''' 

https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 

4th bullet point in project proposal file, which mentions pre processing for the text.

'''

class TextPreprocessing:
    # Just pass in the dataframe itself so this entire class can work on it.
    def __init__(self, df, textColumnName):
        self.m_df = df
        self.m_textColumnName = textColumnName

        # Run all the steps to get the text clean.
        self.LowerText()
        self.CleanPunctuation()
    
    def GetDataFrame(self):
        return self.m_df
    
    def LowerText(self):
        ''' In the class constructor m_df is set to the dataframe of course. Also m_textColumnName is exactly that,
            the name of the feature/column that this function will try and access. 
            
            The train.csv file the dataset (https://www.kaggle.com/c/quora-insincere-questions-classification/data?select=train.csv)
            has a feature/column named question_text. That is passed in as a 2nd argument into the constructor.
            
            Using list comprehension, condense a typical for loop and its return value will be new entries in the 
            dataframe.  '''
        self.m_df[self.m_textColumnName] = [text.lower() for text in self.m_df[self.m_textColumnName]]

    def CleanPunctuation(self):
        pass

