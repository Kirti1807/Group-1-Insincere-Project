''' 

https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 

2nd bullet point in project proposal file, which mentions data analysis

'''

''' 

https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 

2nd bullet point in project proposal file, which mentions data analysis

'''

import numpy as np
from numpy.core.records import array
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download("stopwords")
stopwords = set(STOPWORDS)
# calling the dataset
df_train = pd.read_csv("datasets/train.csv")
df_test = pd.read_csv("datasets/test.csv")

# understanding the dataset insights
print("Understanding the testing dataset:->")
df_test.shape
df_test.info()

# checking for imbalance dataset
print("Checking for see if the dataset is balance or not:")
df_train["target"].value_counts()
df_train["target"].value_counts().plot(kind = "bar", color = ["Green", "Blue"])

len_sincere=0
len_no_sincere  = 0
aray_labels = 0
image_color=False
mask = ""
text = ""


class DataAnalasys:
    def __init__(self):
        self.df_train = df_train
    def explore_dataset(self, visualizing): 
        print("Head of dataset is", df_train.head())
        print("Shape of dataset is:",df_train.shape)
        if visualizing:
            self.visualize_data()


    """
    Drawing a pie chart to help visualize the difference in the two labesl we have
    in the dataset
    value_counts: is used to check the number of sincere and unsincere questions we have to
    labels is printing the two labesl we have
    
    
    """


    def visualize_data(show_word_cloud_with_labels = False):
        len_sincere = df_train["target"].value_counts()
        len_no_sincere = df_train["target"].value_counts()
        aray_labels = np.array(len_sincere, len_no_sincere)
        labels = ["Sincere Questions", "Non Sincere Question"]
        print("Number of labels", labels)
        print("Number of sincere questions are", len_sincere)
        print("Number of non-sicere questions are", len_no_sincere)
        plt.pie(aray_labels, labels=labels,autopct="%1.1f%%",  startangle=90)
        plt.show()


        # working on the wordcloud samples

    def plot_wordcloud(self,text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, ):
                   self.text = text
                   self.mask = mask
                   self.max_words = max_words
                   self.max_font_size =max_font_size
                   self.figure_size =figure_size
                   self.title = title
                   self.title_size =title_size
                   self.image_color = image_color
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    # max_words = max_words,
                    # max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400
                    
                    )
    # wordcloud.generate(str(text))
    
    plt.figure(figsize=(24.0,16.0))
    """
    plotting the wordcloud 
    
    """
    if image_color == False:
        image_colors = wordcloud.ImageColorGenerator(mask)
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
        plt.title("Word Cloud of Questions", fontdict={'size': 40,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud)
        plt.title("Word Cloud of Questions", fontdict={'size': 40, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off')
    plt.tight_layout()  
    
df_train["question_text"], title="Word Cloud of Questions"



    