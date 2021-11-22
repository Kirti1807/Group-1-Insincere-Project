''' 

https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 

2nd bullet point in project proposal file, which mentions data analysis

'''

''' 

https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 

2nd bullet point in project proposal file, which mentions data analysis

'''

from nltk import collocations
import numpy as np
from numpy.core.records import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
# nltk.download("stopwords")
# stopwords = set(STOPWORDS)
# calling the dataset
# df_train = pd.read_csv("datasets/train.csv")
# df_test = pd.read_csv("datasets/test.csv")

# understanding the dataset insights
# print("Understanding the testing dataset:->")
# df_test.shape
# df_test.info()

# checking for imbalance dataset
# print("Checking for see if the dataset is balance or not:")
# df_train["target"].value_counts()
# df_train["target"].value_counts().plot(kind = "bar", color = ["Green", "Blue"])

# len_sincere=0
# len_no_sincere  = 0
# aray_labels = 0
# image_color=False
# mask = ""
# text = ""


class DataAnalysis:
    ''' Function updates (from Omar):
    
        1) The previous code had certain train and test datasets such as the following:
            "df_train = pd.read_csv("datasets/train.csv")
             df_test = pd.read_csv("datasets/test.csv")"

            Not sure what TEST.CSV was, and even the default train.csv file to use for this
            project has a lot of rows. Too much for the task for DataAnalysis. So use some 
            the custom data on this, it's a small dataset but will work just fine.

        2) Previous variables such as len_sincere, len_no_sincere, etc are defined in the class
            instead of outside the class with the naming convention "self.m_(rest of name)". 
            Where "m" is for "member" meaning this variable is a member of the class.
    '''
    def __init__(self, imageColorBool=True):
        self.m_df = pd.read_csv('customData.csv', encoding='latin-1')

        self.m_len_sincere = 0
        self.m_len_no_sincere  = 0
        self.m_aray_labels = 0
        self.m_image_color = imageColorBool
        self.m_mask = ""
        self.m_text = ""

    def explore_dataset(self, visualizing): 
        print("Head of dataset is", self.m_df.head())
        print("Shape of dataset is:", self.m_df.shape)
        if visualizing:
            self.visualize_data()


    """
    Function updates (from Omar):

        1) self - This is a class function and it was missing the self keyword.

        2) Updated the variable names to the class variables. Ex: from len_sincere 
            to self.m_len_sincere.

        3) show_word_cloud_with_labels - This argument was never used in the function
            despite it having a default value of false like "show_word_cloud_with_labels = False",
            So it's been removed.

    
    Drawing a pie chart to help visualize the difference in the two labesl we have
    in the dataset
    value_counts: is used to check the number of sincere and unsincere questions we have to
    labels is printing the two labesl we have
    
    """
    def visualize_data(self):
        self.m_len_sincere = self.m_df["target"].value_counts()
        self.m_len_no_sincere = self.m_df["target"].value_counts()
        aray_labels = np.array(self.m_len_sincere, self.m_len_no_sincere)
        labels = ["Sincere Questions", "Non Sincere Question"]
        print("Number of labels", labels)
        print("Number of sincere questions are", self.m_len_sincere)
        print("Number of non-sicere questions are", self.m_len_no_sincere)
        plt.pie(aray_labels, labels=labels,autopct="%1.1f%%",  startangle=90)
        plt.show()


        # working on the wordcloud samples


    ''' Function updates (from Omar):
    
        1) Indentation - The code indentation for the function was a bit off. Always be sure
            function code is about 1 tab of space are creating the function.


        2) Variables - There were a lot of self variables declared here. Was a bit unnecessary
            due to the fact they wouldn't be used anywhere else in the class. See below:

                self.text = text
                self.mask = mask
                self.max_words = max_words
                self.max_font_size =max_font_size
                self.figure_size =figure_size
                self.title = title
                self.title_size =title_size
                self.image_color = image_color


        3) stopwords - They were imported in a previous line, outside the class in the old implementation.
            But even doing so, no language (english, spanish, etc) was specified.
    '''
    def plot_wordcloud(self, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), title = None, title_size=40):
        from nltk.corpus import stopwords

        # Stop words must be in a set data structure to use the union function to update current stop words.
        stop_words = set(stopwords.words('english'))
        more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
        stop_words = stop_words.union(more_stopwords)

        # Create an ndarray of a few text values to show a word cloud.
        textArray = self.m_df['question_text'][:5].values
        print(f'Text type: {type(textArray)}.\nText: {textArray}')

        # Start the word cloud off with the following values and immediately geenrate the ndarray in str format.
        wordcloud = WordCloud(background_color='white',
                    stopwords = stop_words,
                    # max_words = max_words,
                    # max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400
                    ).generate(str(textArray))
        
        plt.figure(figsize=(24.0,16.0))

        """
        plotting the wordcloud 
        
        """
        if self.m_image_color == False:
            # Issue here.
            image_colors = ImageColorGenerator(mask)
            plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
            plt.title("Word Cloud of Questions", fontdict={'size': 40,  
                                    'verticalalignment': 'bottom'})
        else:
            plt.imshow(wordcloud)
            plt.title("Word Cloud of Questions", fontdict={'size': 40, 'color': 'black', 
                                    'verticalalignment': 'bottom'})
        plt.axis('off')
        plt.tight_layout()  
    
        # df_train["question_text"], title="Word Cloud of Questions"