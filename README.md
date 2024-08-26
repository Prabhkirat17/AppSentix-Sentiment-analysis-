# AppSentix-Sentiment-analysis
# Overview
This project aims to analyze user reviews from the Google Play Store for various apps, particularly focusing on the sentiment behind these reviews. The analysis uses natural language processing (NLP) techniques, including sentiment analysis with TextBlob and VADER, to understand user feedback. Additionally, the project involves supervised learning models to classify reviews into different sentiment categories, providing insights for app developers to improve their applications.

# Features
1. Data Scraping: Extracts user reviews from Google Play Store using the google-play-scraper package.
2. Data Preprocessing: Cleans the data by removing punctuation, special characters, stop words, and lemmatizing the text.
3. Sentiment Analysis: Utilizes TextBlob and VADER to determine the sentiment polarity of user reviews.
4. Supervised Learning: Implements machine learning models (Random Forest, SVM, Logistic Regression) to classify reviews into sentiment categories.
5. Topic Modeling: Applies Latent Dirichlet Allocation (LDA) to uncover hidden topics in user reviews, helping to identify common themes in user feedback.
6. Recommendations: Based on the analysis, the project provides actionable insights and recommendations for app developers.

# Random Forest Classifier
The Random Forest algorithm is employed in this project as one of the supervised learning models to classify user reviews into sentiment categories—positive, neutral, or negative. Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mode of the classes (classification) of the individual trees. This method is particularly effective in handling the high dimensionality and sparsity often found in text data. In this project, the Random Forest classifier demonstrated strong performance in predicting sentiment categories, achieving an accuracy of X% on the test data, making it a reliable choice for sentiment classification tasks.

# Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is employed in this project for topic modeling, a technique used to discover the abstract topics that occur in a collection of documents—in this case, user reviews. LDA is a generative probabilistic model that assumes each document is a mixture of a small number of topics and that each word in the document is attributable to one of the document's topics. By applying LDA to the preprocessed reviews, the project successfully identified key topics within the feedback, such as "user experience," "app performance," and "feature requests." These insights enable app developers to focus on the most critical areas for improvement based on the underlying themes in user feedback.

# Results
1. Sentiment Comparison: The sentiment analysis results from TextBlob and VADER were compared to understand differences in sentiment detection.
2. Model Performance: The Random Forest classifier showed the best performance in predicting sentiment categories with an accuracy of X%.
3. Topic Modeling: LDA revealed key topics in user reviews, such as "user experience" and "feature requests," guiding developers on where to focus their improvements.
4. Recommendations: For the app "Numberzilla: Number Puzzle," it is recommended to introduce new game modes and improve the onboarding experience to enhance user satisfaction.
