### Spam Message Classifier
This is a simple spam message classifier that uses Natural Language Processing (NLP) techniques to classify messages as either spam or not spam. The app is built using Streamlit, a Python library for building web apps for machine learning and data science.

### Requirements
Python 3.8 or higher

Enter your message in the text area and click the "Predict" button to see if the message is spam or not.

### How it works
The app uses a machine learning model that has been trained on a dataset of SMS messages labeled as either spam or not spam. When the user enters a message, the app first preprocesses the text by converting it to lowercase, tokenizing it into words, removing any non-alphanumeric tokens, and stemming each word. Then, the app vectorizes the processed text using a term frequency-inverse document frequency (tf-idf) vectorizer and predicts whether the message is spam or not using a pre-trained machine learning model.

If the message is predicted to be spam, the app displays a "This message seems like Spam" header and an image of spam. If the message is predicted to not be spam, the app displays a "This message does not seem like a Spam" header and an image of a non-spam message.
