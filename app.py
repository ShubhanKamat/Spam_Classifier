import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

if not os.path.isdir(nltk_data_path):
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)

    
    
class SpamClassifier:
    def __init__(self):
        self.ps = PorterStemmer() # Initialize Porter Stemmer for text processing
        self.stop_words = set(stopwords.words('english')) # Load stop words from NLTK corpus
        self.tfidf = None # Initialize tf-idf vectorizer to None
        self.model = None # Initialize model to None

    def transform_text(self, text):
        text = text.lower() # Convert all text to lowercase
        tokens = word_tokenize(text) # Tokenize the text into words
        tokens = [i for i in tokens if i.isalnum()] # Remove any non-alphanumeric tokens
        tokens = (self.ps.stem(i) for i in tokens if i not in self.stop_words and i not in string.punctuation) # Stem each word and remove stop words and punctuation
        return " ".join(tokens) # Join the list of processed tokens into a string

    def load_model(self):
        self.tfidf = pickle.load(open('vectorizer.pkl','rb')) # Load tf-idf vectorizer from a saved file
        self.model = pickle.load(open('model.pkl','rb')) # Load model from a saved file

    def predict(self, input_sms):
        transformed_sms = self.transform_text(input_sms) # Transform the input text using the transform_text method
        vector_input = self.tfidf.transform([transformed_sms]) # Vectorize the transformed text using tf-idf vectorizer
        result = self.model.predict(vector_input)[0] # Predict the result using the loaded model
        if result == 1:
            return "This message seems like Spam"
        else:
            return "This message does not seem like a Spam"

def main():
    classifier = SpamClassifier() # Initialize SpamClassifier object
    classifier.load_model() # Load model and vectorizer

    st.title("Spam message classifier")
    st.subheader("This app will tell you whether your message is spam or not spam")
    input_sms = st.text_area("Enter the message") # Add a text area for user to input message
    logging.info('Message entered')
    try:
        if st.button('Predict'): # Add a button to trigger the prediction
            result = classifier.predict(input_sms) # Predict the result using the input message
            if result == "This message seems like Spam": # If result is Spam
                st.header("This message seems like Spam")
                st.image("spam_image.png") # Display the spam image
            else:
                st.header("This message does not seem like a Spam")
                st.image("not_spam.jpg") # Else, display the not-spam image
    except Exception as e:
            raise CustomException(e,sys)

if __name__ == '__main__':
    main()
