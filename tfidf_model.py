import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import math
import pickle
import joblib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm  # Import tqdm for progress tracking
from collections import defaultdict


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')


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

        self.tfidf = TfidfVectorizer(tokenizer=self.tokenize, lowercase=False)
        
        self.training_tfidf = None
        
        #self.manager = multiprocessing.Manager()
        
        self.flattened_sentences = []
        self.training_tagged = []
        self.answers = []
        
        
        
    def tokenize(self, text):
        # Your tokenization logic goes here
        return text  # No tokenization needed, return the input as-is

    def preprocess_text(self, text):
        # Tokenization
        sentences = sent_tokenize(text)
        
        preprocessed_sentences = []
        batch_size = 50  # Adjust the batch size based on your system's capabilities
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_words = [word_tokenize(sentence) for sentence in batch_sentences]

            # Filtering Stop Words
            stop_words = set(stopwords.words('english'))
            filtered_words = [[word for word in words if word.lower() not in stop_words] for words in batch_words]

            # Stemming
            stemmer = PorterStemmer()
            stemmed_words = [[stemmer.stem(word) for word in words] for words in filtered_words]

            # Tagging Parts of Speech
            pos_tags = [nltk.pos_tag(words) for words in stemmed_words]

            # Lemmatizing
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [[lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos] for pos in pos_tags]

            preprocessed_sentences.extend(lemmatized_words)

        return preprocessed_sentences
    
    def process_data(self, data_json):
        #print("Processing data in parallel...")
        batch_size = 10000  # Experiment with different batch sizes
        num_processes = int(multiprocessing.cpu_count()/2)  # Utilize more processes

        batches = [data_json[i:i + batch_size] for i in range(0, len(data_json), batch_size)]
        
        #print('batches')

        #training_tagged = []  # Initialize or clear self.training_tagged
        sentence_answers = []

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(executor.map(self.process_data_batch, batches), total=len(batches)))
            
        #with multiprocessing.Pool() as pool:
        #results = []
        #for batch in batches:
        #results.append(self.process_data_batch(batch))

        for batch_result in results:
            for result in batch_result:
                sentence_answers.extend(result)
            #print("here")
            
        # Create a dictionary to group sentences by answer
        answer_groups = defaultdict(list)

        # Iterate through each (sentence, answer) pair in batch_results
        for sentence, answer in sentence_answers:
            answer_groups[answer].extend(sentence)
            
        #print(list(answer_groups.items())[0])

        # Create a new list with sentences grouped by answer
        sentence_answers.extend([(sentence,answer) for answer, sentence in answer_groups.items()])

        self.flattened_sentences.extend([x[0] for x in sentence_answers])
        self.training_tagged.extend([x[1] for x in sentence_answers])
        
        
        
        #print("Data processing complete.")

    def process_data_batch(self, batch):
        batch_results = []
        
        
        
        for data in batch:
            text = data["text"]
            answer = data["answer"]
            preprocessed_sentences = self.preprocess_text(text)
            training_tagged = [(sentence, answer) for sentence in preprocessed_sentences]
            
            
            
            #print(training_tagged)
            batch_results.append(training_tagged)
            
        #create another list where instead, the "sentence" of elements with the same answer are appended with each other
            
        return batch_results

    def train_model(self):
        # Fit and transform the TF-IDF vectorizer
        
        #print(self.flattened_sentences)
        if(self.flattened_sentences):
            self.training_tfidf = self.tfidf.fit_transform(self.flattened_sentences)
            self.flattened_sentences = []
            #self.
            
        #print(self.training_tfidf)
        #print(self.training_tagged)

            


    def save(self, file_path):
        model_data = {
            'training_tagged': list(self.training_tagged),
            'tfidf': self.tfidf,
            'training_tfidf': self.training_tfidf
        }
        #print(model_data)
        with open(file_path, 'wb') as f:
            joblib.dump(model_data, f)

    def load(self, file_path):
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print(os.path.exists(file_path))
                model_data = joblib.load(file_path)
                self.training_tagged = list(model_data['training_tagged'])
                self.tfidf = model_data['tfidf']
                print(self.tfidf)
                self.training_tfidf = model_data['training_tfidf']
            
        return self

    def predict(self, input_data):
        # Preprocess input data
        new_text_processed = self.preprocess_text(input_data)
        new_text_processed_tfidf = self.tfidf.transform(new_text_processed)
        training_tfidf = self.training_tfidf
    
        # Calculate sentence similarities
        sentence_similarities = cosine_similarity(new_text_processed_tfidf, training_tfidf)
    
        # Initialize data structures
        similarities_max = {}
        answers = []
    
        # Iterate over sentence similarities
        for similarity_row in sentence_similarities:
            for answer, similarity in zip(self.training_tagged, similarity_row):
                if isinstance(answer, list):
                    continue
                # Update similarities_max only when the new similarity is greater
                if answer not in similarities_max or similarity > similarities_max[answer]:
                    similarities_max[answer] = similarity
    
            if not answers:
                answers.extend(similarities_max.keys())
    
        # Calculate total similarity for each answer and find the maximum similarity and its index
        total_similarities = np.array([similarities_max[answer] for answer in answers])
        closest_index = np.argmax(total_similarities)
        closest_answer = answers[closest_index]
    
        return total_similarities[closest_index], closest_answer




        
        
                        
    #return (sentences.max(),self.training_tagged[closest_index])



        
    
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

            model.process_data(data_json)
            model.train_model()
            print(model.predict("My name is bobby, bobby newport. your name is jeff?"))
            model.save("model.pkl")

    if flags.model:
        model.load(flags.model)
    
    if flags.predict:
        print(model.predict(flags.predict))




        
