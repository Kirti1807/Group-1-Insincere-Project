import pandas as pd
from TextPreprocessing import TextPreprocessing

'''  

Group 1 Insinscere project!

All group members can definitely contribute to the project. Please be careful for merge conflicts though! If 2 or more
people are working on the exact same file and try to push to github, there will be problems with the way Github views
the changed files!

            File information below.

            1) train.csv - 1,306,122 rows & 3 columns.

'''

df = pd.read_csv('train.csv', encoding='latin-1') # Text feature/column name is 'question_text'

''' Instruction #4 in the project proposal, text preprocessing. Give dataframe along with 
    the feature/column name. '''
tp = TextPreprocessing(df, 'question_text')
print(tp.GetDataFrame().head())