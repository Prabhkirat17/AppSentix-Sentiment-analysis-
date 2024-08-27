!pip install google-play-scraper

from google_play_scraper import app
from google.colab import files
import pandas as pd
import numpy as np

from google_play_scraper import Sort, reviews_all

Total_reviews = reviews_all(
    #'com.androbaby.game2048', #Original App
    #'com.inspiredsquare.number', #Similar App 1
    #'merge.blocks.drop.number.puzzle.games', #Similar App 2
    #'com.xoxj.card2048', #Competitor App 1
    'com.appcraft.number.puzzle', #Competitor App 2

    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    #country='us', # defaults to 'us'
    sort=Sort.NEWEST, # defaults to Sort.MOST_RELEVANT
)

# Extracting the relevant data for the csv file
data = [
    {
        'Package name': review['reviewId'],
        'Reviewer name': review['userName'],
        'Review': review['content'],
        'Rating': review['score'],
    }
    for review in Total_reviews
]

df_busu = pd.DataFrame(data)

df_busu.head(1000) #Limiting the number of reviews to 1000

#Downloading the table as a .csv file
#df_busu.to_csv('2048.csv', index=False) # Original app
#df_busu.to_csv('Match the Number - 2048 Game.csv', index=False) # Similar App 1
#df_busu.to_csv('2048 Merge Games - M2 Blocks.csv', index=False) # Similar App 2
#df_busu.to_csv('2048 solitaire.csv', index=False) # Competitor App 1
df_busu.to_csv('Numberzilla - Number Puzzle.csv', index=False) # Competitor App 2

#files.download('2048.csv') # Original App
#files.download('Match the Number - 2048 Game.csv') # Similar App 1
#files.download('2048 Merge Games - M2 Blocks.csv') # Similar App 2
#files.download('2048 solitaire.csv') # Competitor App 1
files.download('Numberzilla - Number Puzzle.csv') # Competitor App 2

"""TASK 2: PREPROCESS YOUR TEXT"""

!pip install nltk

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re

# Now preprocessing the data: Remove punctuations,Remove special characters and emojis,Turn numbers into text,Remove extra white spaces, Turn all words into lowercase,
# Remove stop words,Lemmatize the reviews,Output 15 sample pre-processed reviews in your notebook.
def preprocess_review(review_process):
    # For the punctutation removal
    review_process = review_process.translate(str.maketrans("", "", string.punctuation))

     # Removing any extra white spaces in the reviews
    review_process = ' '.join(review_process.split())

    # Converting all the letters in lowercase
    review_process = review_process.lower()

    # Turning all the numbers into text
    review_process = re.sub(r'\d+', 'number', review_process)

    # Removing any speacial characters and emojis
    review_process = re.sub(r'[^\w\s]', '', review_process)

    # Remmoving any stop words
    stop_words = set(stopwords.words('english'))
    review_process = ' '.join([word for word in review_process.split() if word not in stop_words])

    # Lemmatizing the reviews
    lemmatizer = WordNetLemmatizer()
    review_process = ' '.join([lemmatizer.lemmatize(word) for word in review_process.split()])

    return review_process

preprocessed_reviews = df_busu
preprocessed_reviews['Review'] = preprocessed_reviews['Review'].apply(preprocess_review)
df_preprocessed_only = preprocessed_reviews.head(15)
df_preprocessed_only_set = df_preprocessed_only['Review']
print(df_preprocessed_only_set.head(15))

"""TASK 3: SENTIMENT ANALYSIS

Part 1: Using TextBlob for sentimental analysis
"""

from textblob import TextBlob

# Using TextBlob for sentimental analysis
def sentiment_polarity(review_data):
    analysis_data = TextBlob(review_data)
    return analysis_data.sentiment.polarity

# Calculating the sentiment polarity
preprocessed_reviews['TB_polarity'] = preprocessed_reviews['Review'].apply(sentiment_polarity)

df_out = preprocessed_reviews.head(250)

df_out_filtered = df_out[['Package name', 'Review', 'TB_polarity']]
df_out_filtered.head(250)

"""Part 2: Using Vader for calculating Reviews Sentiment"""

!pip install vaderSentiment

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Using VADER for sentiment analysis
analyzer = SentimentIntensityAnalyzer()
def vader_sentiment(review_data):

    sentiment_scores = analyzer.polarity_scores(review_data)
    return sentiment_scores['compound']  # Using the compound score as an overall sentiment polarity

# Calculating VADER sentiment polarity
#df_out['VADER_polarity'] = df_out['Review'].apply(vader_sentiment)
preprocessed_reviews['Vader_polarity'] = preprocessed_reviews['Review'].apply(vader_sentiment)

df_out = preprocessed_reviews.head(250)

# Filter out the DataFrame to include only 'Package name,' 'Rating,' and 'VADER_polarity'
df_out_filtered = df_out[['Package name', 'Review', 'Vader_polarity']]

df_out_filtered.head(250)

"""How does the sentiments retrieved by Textblob and Vader compare with each
other? How do they compare with the rating of the app? How do you interpret the
similarities/differences?

When comparing sentiments extracted by TextBlob with VADER, the polarity ratings provided to each reviews had to be examined in order to compare their sentimental analysis. TextBlob and VADER both assigned a polarity score to each review, indicating the text's sentiments. Example: Vader Sentiment Analysis:
Positive Sentiments:

"love challenge" - Positive sentiment (0.6705),
"surprisingly addictive enjoyable game" - Positive sentiment (0.5423),
"good cognitive" - Positive sentiment (0.4404),
"enjoying game" - Positive sentiment (0.5267),
"great game enough power ups many ad" - Positive sentiment (0.6249),

Negative Sentiments:

"deleted way many ad cant even enjoy game" - Negative sentiment (-0.3875),
"distressed longer match diagonally" - Negative sentiment (-0.4215)

TextBlob Sentiment Analysis:
Positive Sentiments:

"love challenge" - Positive sentiment (0.5),
"surprisingly addictive enjoyable game" - Positive sentiment (0.04375),
"good cognitive" - Positive sentiment (0.7),
"enjoying game" - Positive sentiment (0.05),
"great game enough power ups many ad" - Positive sentiment (0.225),
Negative Sentiments:

"deleted way many ad cant even enjoy game" - Positive sentiment (0.16667),
"distressed longer match diagonally" - Neutral sentiment (0.0) . TextBlob and VADER both agree that feedback like "love challenge," "good cognitive," and "enjoying game" are typically positive.
These two tools also show negative feedback in reviews, such as "deleted way many ads cant even enjoy game."
There are several interpretations of sentiments. According to VADER (0.5423), "surprisingly addictive enjoyable game" has a higher positive sentiment than TextBlob (0.04375).
VADER (-0.4215) interprets "distressed longer match diagonally" negatively, but TextBlob (0.0) interprets it neutrally. Sentiment analysis is subjective, and depending on the underlying models and training data, various techniques may produce somewhat different findings.In general, the overall positive sentiments found by TextBlob and VADER are consistent with the app's high rating of 4.3, showing good customer feedback towards the app.

TASK4: SUPERVISED LEARNING
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score

file_path = '/content/drive/MyDrive/EECS 4312/reviews_classified_lemmatized.csv'
#file_path1 = '/content/drive/MyDrive/EECS 4312/Match the Number - 2048 Game.csv'
#file_path2 = '/content/drive/MyDrive/EECS 4312/Numberzilla - Number Puzzle.csv'
def load_data():
    data = pd.read_csv(file_path)
    return data

def data_set_encode(data_set):
    label_encoder = LabelEncoder()
    data_set = data_set.apply(lambda col: label_encoder.fit_transform(col))
    return data_set

def split(dataframe):
    y = dataframe['Review category']
    x = dataframe.drop(columns=['Review category'])
    return train_test_split(x, y, test_size=0.3, random_state=0)

#def split_2(dataframe):
   # y = dataframe['Review']
   # x = dataframe.drop(columns=['Review'])
   # return train_test_split(x, y, test_size=0.3, random_state=0)

def plot_confusion_matrix(cm,clf):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()

def create_classifier(algorithm):
    if algorithm == "SVM":
        return SVC(kernel='linear', C=1.0)
    elif algorithm   == "Logistic Regression":
        return LogisticRegression(C=1.0, penalty='l2', max_iter=500)
    elif algorithm == "Random Forest":
        return RandomForestClassifier(n_estimators=500, max_depth=5, bootstrap=True, n_jobs=-1)
    else:
        raise ValueError("Invalid algorithm choice. Supported options: 'SVM' or 'Logistic Regression'")

data_set = load_data()
fr_data_encoded = data_set_encode(data_set[data_set['Review category'].str.contains('f')])

x_train, x_test, y_train, y_test = split(fr_data_encoded)

# Create the classifier (SVM/Logistic Regression/Random Forest)
clf = create_classifier("Random Forest")

# train the classifier on the training data
clf.fit(x_train, y_train)

# make predictions on the test data
y_pred = clf.predict(x_test)

# evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# a classification report for more detailed metrics
print(classification_report(y_test, y_pred,zero_division=0.0))

# plotting the confusion metrics
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plot_confusion_matrix(cm,clf)

#data_set = load_data()
br_data_encoded = data_set_encode(data_set[data_set['Review category'].str.contains('b')])

bx_train, bx_test, by_train, by_test = split(br_data_encoded)

# Create the classifier (SVM/Logistic Regression/Random Forest)
br_clf = create_classifier("Random Forest")

# train the classifier on the training data
br_clf.fit(bx_train, by_train)

# make predictions on the test data
by_pred = br_clf.predict(bx_test)

# evaluate the model
accuracy = accuracy_score(by_test, by_pred)
print("Accuracy:", accuracy)

# a classification report for more detailed metrics
print(classification_report(by_test, by_pred,zero_division=0.0))

# plotting the confusion metrics
B_cm = confusion_matrix(by_test, by_pred, labels=br_clf.classes_)
plot_confusion_matrix(B_cm,br_clf)

"""Recommend two changes that the developers should apply to their app and
explain your rational. Specify these changes are recommended for which app and
why.

These changes are of the app Numberzilla: Number Match Game.
By analysing the features, app reviews and bugs for this app I have two major recommendations for this app:


1.   Adding new game modes or adding variety to attract the users: By adding new. New modes for games might include varying levels of difficulty, puzzle variants, or time-limited challenges. This not only increases consumer satisfaction but also increases the replay value of the game.
2.   Innovative Tutorials and efficient onboarding experience: User feedback indicates that the app's onboarding procedure and instruction are difficult or confusing. Making the instructions and onboarding process more straightforward, informative and simple makes the user experience more effecient. A simple and engaging onboarding process may increase user happiness, decrease dissatisfaction and lead to a favourable initial experience of the app.

The recommendations are based on feedback from customers and desires which is consistent with a user-centric approach to app development.The addition of new game modes and the enhancement of the tutorial/onboarding process directly contribute to an improved user experience. A pleasant and users-friendly experience encourages customer loyalty and favourable feedback.

Explain the risk and uncertainty associated with the recommendations you made
in Task III.

Individual user preferences may influence the efficacy of the recommendations. It is critical to continuously monitor user input in order to respond to changing user expectations. Technical issues may arise while implementing new features and enhancing onboarding. Thorough testing is essential for identifying and addressing any possible problems. It is critical to balance new features with bug fixes and technological enhancements.

Task 5
"""

import pprint
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

text_corpus = preprocessed_reviews['Review']
#Numberzilla

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)

from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)

#pprint.pprint(dictionary.token2id)

bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)

from gensim import models

tfidf = models.TfidfModel(bow_corpus) # initialize a model

corpus_tfidf = tfidf[bow_corpus]
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=15)  # initialize an LSI transformation
corpus_lsi = lsi_model[corpus_tfidf]

topics_with_limit = lsi_model.print_topics(15, num_words=7)

for topic in topics_with_limit:
    print(topic)

for doc, as_text in zip(corpus_lsi, text_corpus):
    print(doc, as_text)

"""How do these review topics (i.e. summarized user feedback) compare to the
software features you extracted in TASK4?

Topics 1 and 8: Use positive words like "great," "love," and "good." Users like the game and appreciate the personalization choices. If consumers find the themes appealing and engaging, these favourable feelings may be tied to theme personalization. Topic 5: Mentions the words "many" and "ad." Users express worries about the frequency of advertisements and might be dissatisfied with the amount of ads they face while playing the game.
Topic 11 focuses on the words "cool" and "game." Users show good feelings regarding the game's coolness, suggesting overall enjoyment.
Positive sentiments throughout numerous subjects indicate overall app appreciation. Advertisement-related remarks in Topic 5 suggest a possible area for improvement. Addressing consumer concerns about ad frequency may improve the overall user experience.

How do these review topics (i.e. summarized user feedback) compare to the bug
reports you extracted in TASK4?

The analysis themes derived from user reviews reveal a variety of moods and elements about user experiences. While some subjects convey overall viewpoints, they may not address particular bug-related issues specifically. Topics such as "great," "love," and "challenging" reflect good user feedback.
Users admire the game's "addictive" gameplay and think it's "cool" and "interesting."
Some users have expressed the concern with the crashes and UI/UX being difficult to understand and these things are very import for an app's overall success. SO, if these should be addressed on a priority basis. Comparing to the bug reports generated in task 4, it is crutial to analyze that these bug fixes are important to fix.

How do you explain the differences between TASK 4 and TASK 5 in terms of
accuracy of the model and the size of data for each of your recommendations?

In Task 4, a supervised learning strategy is used to construct Random Forest classifiers that determine whether a user review is a feature request or a bug report. This method includes training the model on labelled data, with each data point representing a different type (feature request or problem report). Patterns and correlations between characteristics and class labels are discovered by the model.
For topic modelling in Task 5, an unsupervised learning strategy based on Latent Dirichlet Allocation (LDA) is used. Without explicit class labels, LDA is a probabilistic model that detects subjects in a collection of text documents. It identifies latent topics based on word distribution throughout documents.
For training and assessment, Task 4 relies on labelled data with specific class labels.
Task 5 does not need labelled data; instead, it discovers latent subjects in the absence of established classifications.
The number and quality of labelled training data impact model performance in Task 4.
The overall size and structure of the text corpus impact topic modelling in Task 5.
"""

