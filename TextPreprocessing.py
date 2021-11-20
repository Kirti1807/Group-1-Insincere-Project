import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

''' 

https://docs.google.com/document/d/1ep6Ki92F81lRiojOdqv5otBdDlfY6ZWcqjVsQc0LVjg/edit 

4th bullet point in project proposal file, which mentions pre processing for the text.

'''

class TextPreprocessing:
    # Just pass in the dataframe itself so this entire class can work on it.
    def __init__(self, df, textColumnName):
        self.m_df = df
        self.m_textColumnName = textColumnName
        self.m_stopWords = list(stopwords.words('english'))
        self.m_lemmatizer = WordNetLemmatizer()

        ''' Every step is necessary to clean the text. 
            1) LowerText - This is a simple step, all it really requires is the .lower() function to be used.
                Done with list comprehension instead of a for loop and it will update the dataframe.
                
            2) CleanPunctuation - Characters like '.', ',', '?', etc. They add nothing to sentiment.
            
            3) CleanStopWords - Stop words like 'in', 'a', or 'so', just like punctuation, add nothing
                to the sentiment. 
                
                    Key Note: This part of the code was running EXTREMELY slow. Because the list of stop
                        words (as seen above) was initialized everytime the function was called. That
                        initialization is now of course in the constructor so therefore the 
                        .apply(CleanStopWords) runs much faster. Potentially it can still be improved speed
                        wise. Will return to this soon '''
        self.LowerText()

        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanPunctuation)
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanStopWords)
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanUrls)
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanHashTags)
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.CleanNumbers)
        self.m_df[textColumnName] = self.m_df[textColumnName].apply(self.LemmatizeText)
    
    def GetDataFrame(self):
        return self.m_df
    
    def LowerText(self):
        self.m_df[self.m_textColumnName] = [text.lower() for text in self.m_df[self.m_textColumnName]]

    def CleanPunctuation(self, text):
        ''' In order to easily clean punctuation, first we have to make a translation table
        with the given text which is simple. NORMALLY the first 2 arguments in the function
        indicate what you'd like to replace. Example: If myStr = 'Hello People', and then
        the goal is to replace 'l' with '@', simply do something like 'myTable = 
        myStr.maketrans('l', '@'). THEN use the translate function on the string itself
        and pass in the table so it can begin mapping.
        
        However in the case of removing punctuation, ignore the 2 arguments with default 
        values like ''. and pass in a 3rd argument. It indicates values to REMOVE from the 
        text. All those values are stored in string.punctuation. '''
        translationTable = text.maketrans('', '', string.punctuation)
        return text.translate(translationTable)
    
    def CleanStopWords(self, text):
        ''' Stop words are just those words that give nothing to the overall sentiment
        like 'to', 'a', 'is', 'of', etc. Start by using the desired language.
        
        Break the sentence up by words in a list with the split function. Why? 
        Because its a must to check every single word IN the sentence to see if it's
        a stop word. 
        
        Using list comprehension, if a word is NOT a stop word, put it into a new list.
        Then afterwords just use the join function, why? Because ' '.join() will take
        an iterable and put all the values together, and by giving it a space such as ' '
        it's ensured the new string or sentence has spaces between words. '''

        wordsInText = text.split()
        wordsInText = [w for w in wordsInText if not w in self.m_stopWords]
        return ' '.join(wordsInText)

    
    def CleanUrls(self, text):
        ''' Of course, urls are useless in text sentiment as well. Using regex, the 
            urls will be removed. The '\S+' means it will match any string with a 
            non whitespace character. See the link: https://www.programiz.com/python-programming/regex
            So the http\S+ should find any bit of text starting with http and take care 
            of the rest of the link. '''
        return re.sub("http\S+", "", text)

    
    def CleanHashTags(self, text):
        ''' Using regex just like the CleanUrls function, except this time it's for hash tags. '''
        return re.sub("#\S+", "", text)

    
    def CleanNumbers(self, text):
        ''' Numbers give nothing to sentiment too, so using regex again, let's remove them. '''
        return re.sub("\d+", "", text)

    def LemmatizeText(self, text):
        ''' Lemmatization is far better than Stemming because although Stemmming gets a word to
            it's ROOT form, that doesn't guarantee the word will make any logical sense. That's why
            Lemmatization is there. 
            
            Quite a few things are going on here. 

            1) ' '.join() - Remember this function join takes in an iterable, in this case a list
                since the line of code is using list comprehension, and will create a string from
                all values IN that list. 
            
            2) text.split() - This splits the text up, word by word. Why? Each individual word
                must be lemmatized to get its root form of course.

            3) .lemmatize() - Of course the lemmatizer must be used at some point. So just pass it
                the word. '''
        return ' '.join([self.m_lemmatizer.lemmatize(word) for word in text.split()])