import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np

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
        self.training_tagged = {}
        self.sentences = []
        self.tfidf = TfidfVectorizer(tokenizer=lambda i:i, lowercase=False)
        self.training_tfidf = None
        

        pass

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
    
    def train_model(self, data):
        # Train the NLP model using the provided training data and labels

        from sklearn.feature_extraction.text import TfidfVectorizer

        # Tokenize all sentences at once
        
        training_tagged = {}
        sentences = []

        for data in data_json:
            text = data["text"]
            answer = data["answer"]

            preprocessed_sentences = self.preprocess_text(text)

            for sentence in preprocessed_sentences:
                training_tagged.append((sentence, answer))

        flattened_sentences = [sentence[0] for sentence in training_tagged]

        # Fit and transform the TF-IDF vectorizer
        self.training_tfidf = self.tfidf.fit_transform(flattened_sentences)

            


    def save(self, file_path):
        pass

    
    def predict(self, input_data):
        # Use the trained model to make predictions on input_data
        # Return predictions and confidence levels

        similarities = []

        new_text_processed = self.preprocess_text(input_data)
        training_text_tfidf = self.training_tfidf

        for sentence in new_text_processed:
            sentence_tfidf = self.tfidf.transform(sentence)
            similarities.append(cosine_similarity(new_text_tfidf, sentence_tfidf))


        sentences = np.mean(similarities)

        closest_index = similarities.argmax()
        




        pass
    
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

    flags = parser.parse_args()

    model = NLPModel()

    if flags.data:
        with open(flags.data, 'r') as data_file:
            data_json = json.load(data_file)

            model.train_model(data_json)




        
