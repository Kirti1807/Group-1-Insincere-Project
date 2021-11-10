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
    
    def GetDataFrame(self):
        return self.m_df
    
    def LowerText(self):
        self.m_df[self.m_textColumnName] = [text.lower() for text in self.m_df[self.m_textColumnName]]

    def CleanPunctuation(self):
        pass

