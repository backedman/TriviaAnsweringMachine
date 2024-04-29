from typing import List, Tuple
import nltk
import sklearn
import question_categorizer as qc
from question_categorizer import TextClassificationModel
from tfidf_model import NLPModel
import tfidf_model
import transformers
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict





class QuizBowlModel:

    def __init__(self, clear = False):
        """
        Load your model(s) and whatever else you need in this function.

        Do NOT load your model or resources in the guess_and_buzz() function, 
        as it will increase latency severely. 
        """
        
        self.categories = ['Geography', 'Religion', 'Philosophy', 'Trash','Mythology', 'Literature','Science', 'Social Science', 'History', 'Current Events', 'Fine Arts', 'ALL']
        self.tfidf_models = [None for _ in range(len(self.categories) + 1)]
        self.qc_model = qc.TextClassificationModel.load_model("models/categorizer")
                
        self.load_tfidf_models(clear=clear)



        

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        This function accepts a list of question strings, and returns a list of tuples containing
        strings representing the guess and corresponding booleans representing 
        whether or not to buzz. 

        So, guess_and_buzz(["This is a question"]) should return [("answer", False)]

        If you are using a deep learning model, try to use batched prediction instead of 
        iterating using a for loop.
        """
        
        guesses = []
        curr_question = ""
        
        for question in question_text:
            curr_question += question + "."
            
            confidence,answer = self.predict(curr_question)
            
            confidence = True if confidence > 0.5 else False

            guesses.append((confidence,answer))
            
        return guesses
    
    def load_tfidf_models(self, clear=False):
        
        print("loading tfidf models")
        
        # Create respective model if not exist
        if not clear:
            for category in range(len(self.categories)):
                if self.tfidf_models[category] is None:
                    self.tfidf_models[category] = NLPModel().load(f"models/{self.categories[category]}_tfidf.pkl")
                
            self.tfidf_models[-1] = NLPModel().load(f"models/{'ALL'}_tfidf.pkl")
        else:
            for category in range(len(self.categories)):
                if self.tfidf_models[category] is None:
                    self.tfidf_models[category] = NLPModel()
                    
            self.tfidf_models[-1] = NLPModel()

        
    
    def train(self, data):
        
        
        print("Category-tagging Data...")
        
        # Create n empty lists, each index associated with the index of the category
        training_data = [[] for _ in range(len(self.categories) + 1)]

        # Create a tqdm progress bar for data processing
        '''with tqdm(total=len(data)) as pbar:
            # Go through data
            for data_point in data:
                text = data_point["text"]
                answer = data_point["answer"]
                categories = data_point["category"]
                
                for category in categories:
                    
                
                if(text == ""):
                    continue
                
                try:
                    # Get category from qc_model
                    prediction = self.qc_model.predict(text)
                    predictions = np.argwhere(prediction >= 1.5)[1]
                except:
                    print(text)

                for prediction_ind in predictions:
                    # Store data in array with respective index
                    training_data[prediction_ind.item()].append({"text": text, "answer": answer})

                # Update progress bar
                pbar.update(1)

        print("Category-tagging complete.")'''
        
        with tqdm(total=len(data)) as pbar:
            for data_point in data:
                text = data_point["text"]
                answer = data_point["answer"]
                categories = data_point["category"]
                #print(categories)
                #print(data_point)
                
                for category in categories:
                    
                    category_ind = self.categories.index(category)
                            
                    training_data[category_ind].append({"text": text, "answer": answer})
                    
                    if(len(training_data[category_ind]) == 10000):
                        #print("here")
                        self.tfidf_models[category_ind].process_data(training_data[category_ind])        
                        
                        training_data[category_ind] = []
                    
                # Update progress bar
                pbar.update(1)
                    
        for ind,data in enumerate(training_data):
            
            self.tfidf_models[ind].process_data(data)
                        
            training_data[self.categories.index(category)] = []
        
        print("TRAINING DATA")
        with tqdm(total=len(self.categories)) as pbar:
            for category in range(len(self.categories)):
                # Train model
                self.tfidf_models[category].train_model()
                
                # Save model
                self.tfidf_models[category].save(f"models/{self.categories[category]}_tfidf.pkl")
                
                # Unload model
                self.tfidf_models[category] = None
                training_data[category] = None
                
                pbar.update(1)

        '''# Create a tqdm progress bar for model training
        with tqdm(total=len(self.categories)) as pbar:
            # Iterate over categories
            for category, category_data in enumerate(training_data[:-1]):
                
                #if(self.categories[category] != 'Mythology'):
                #    continue
                
                #print(training_data)
                
                
                
                print(f"Training model for category: {self.categories[category]}")
                
                # Process data
                self.tfidf_models[category].process_data(category_data)
                
                # Train model
                self.tfidf_models[category].train_model()
                
                # Save model
                self.tfidf_models[category].save(f"models/{self.categories[category]}_tfidf.pkl")
                
                # Unload model
                self.tfidf_models[category] = None
                training_data[category] = None
                
                #garbage collection/free processes
                
                # Update progress bar
                pbar.update(1)'''

        print("Training complete.")

    
        
    def predict(self, input_data, confidence_threshold=1.5):
        # Get category confidence scores from qc_model
        category_confidences = self.qc_model.predict(input_data)
        print("Category confidences:", category_confidences)
        
        # Find the indices of categories with confidence scores above the threshold
        confident_indices = (category_confidences > confidence_threshold).nonzero()[:,1]
        
        print(confident_indices)
        
        max_confidence = 0
        max_answer = None
        max_category = 0
        for category in confident_indices:
            print(category)
            confidence,answer = self.tfidf_models[category].predict(input_data)
            
            if(confidence > max_confidence):
                max_confidence = confidence
                max_answer = answer
                max_category = category
                
            break
            
        #max_confidence, max_answer = selected_model.predict(input_data)
        print("Prediction for category", self.categories[category], ":", max_answer, "with confidence", max_confidence)
        
        return (max_confidence, max_answer)
        
        
        

        
        
        
        
        
        
if __name__ == "__main__":
    # Train a simple model on QB data, save it to a file
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--predict', type=str)
    parser.add_argument('--clear', action='store_const', const=True, default=False)

    flags = parser.parse_args()
    model = None
    
    print(flags.clear)

    if flags.clear:

        model = QuizBowlModel(clear=True)
        
    else:

        model = QuizBowlModel()

        

    if flags.data:
        with open(flags.data, 'r') as data_file:
            data_json = json.load(data_file)

            model.train(data_json)
            #print(model.predict("My name is bobby, bobby newport. your name is jeff?"))
            #model.save("model.pkl")

    if flags.model:
        model.load(flags.model)
    
    if flags.predict:
        print(model.predict(flags.predict))