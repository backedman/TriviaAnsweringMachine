import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

# Helper function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN

class NLPModel:
    def __init__(self):  # Initialize the model with necessary parameters
        # Initialize model components (preprocessing, training, etc.)
        #self.model
        self.training_tagged = []
        self.flattened_sentences = []
        self.tfidf = TfidfVectorizer(tokenizer=self.tokenize, lowercase=False)
        
        self.training_tfidf = None
        

        pass

    def tokenize(self, text):
        # Your tokenization logic goes here
        return text  # No tokenization needed, return the input as-is

    def preprocess_text(self, text):
    # Tokenization
        sentences = sent_tokenize(text)
        
        preprocessed_sentences = []
        for sentence in sentences:
            # Tokenization
            words = word_tokenize(sentence)
            
            # Filtering Stop Words
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            
            # Stemming
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in filtered_words]
            
            # Tagging Parts of Speech
            pos_tags = nltk.pos_tag(stemmed_words)
            
            # Lemmatizing
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
            
            # Named Entity Recognition (NER)
            entities = nltk.chunk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)))
            
            preprocessed_sentences.append(lemmatized_words)
            
        return preprocessed_sentences
    
    def process_data(self, data):
        # Train the NLP model using the provided training data and labels

        for data in data_json:
            text = data["text"]
            answer = data["answer"]

            preprocessed_sentences = self.preprocess_text(text)

            for sentence in preprocessed_sentences:
                
                self.training_tagged.extend((sentence, answer))
                self.flattened_sentences.append(sentence)

        #flattened_sentences = [sentence[0] for sentence in training_tagged]
        self.training_tagged.extend(training_tagged)

    def train_model():
        # Fit and transform the TF-IDF vectorizer
        self.training_tfidf = self.tfidf.fit_transform(self.flattened_sentences)

            


    def save(self, file_path):
        model_data = {
            'training_tagged': self.training_tagged,
            'tfidf': self.tfidf,
            'training_tfidf': self.training_tfidf
        }
        with open(file_path, 'wb') as f:
            joblib.dump(model_data, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            model_data = joblib.load(f)
            self.training_tagged = model_data['training_tagged']
            self.tfidf = model_data['tfidf']
            self.training_tfidf = model_data['training_tfidf']

    
    def predict(self, input_data):
        # Use the trained model to make predictions on input_data
        # Return predictions and confidence levels

        similarities = []
        print(len(self.training_tagged))

        new_text_processed = self.preprocess_text(input_data)
        print(new_text_processed)
        training_text_tfidf = self.training_tfidf

        for sentence in new_text_processed:
            sentence_tfidf = self.tfidf.transform([sentence])

            similarities.append(cosine_similarity(sentence_tfidf,training_text_tfidf))

        sentences = np.mean(similarities, axis=0)

        closest_index = sentences.argmax()
        
        return self.training_tagged[closest_index][1]



        
    
    def evaluate(self, test_data, labels):
        # Evaluate the performance of the model on test data
        # Return evaluation metrics
        pass
    
    # Additional functions for model tuning, hyperparameter optimization, etc.

if __name__ == "__main__":
    # Train a simple model on QB data, save it to a file
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--predict', type=str)

    flags = parser.parse_args()

    model = NLPModel()

    if flags.data:
        with open(flags.data, 'r') as data_file:
            data_json = json.load(data_file)

            model.train_model(data_json)
            print(model.predict("My name is bobby, bobby newport. your name is jeff?"))
            model.save("model.pkl")

    if flags.model:
        model.load(flags.model)
    
    if flags.predict:
        print(model.predict(flags.predict))




        
