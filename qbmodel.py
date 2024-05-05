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
        self.tfidf_models = [None for _ in range(len(self.categories))]
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
                    
            print(self.tfidf_models)
                    
        
    
    def train(self, data):
                
        # Create n empty lists, each index associated with the index of the category
        training_data = [[] for _ in range(len(self.categories))]
        
        with tqdm(total=len(data)) as pbar:
            for data_point in data:
                text = data_point["text"]
                answer = data_point["answer"]
                categories = data_point["category"]
                
                for category in categories:
                    
                    category_ind = self.categories.index(category)
                            
                    training_data[category_ind].append({"text": text, "answer": answer})
                    
                pbar.update(1)
                    

        for ind,data in enumerate(training_data):
            
            self.tfidf_models[ind].process_data(data)
            
            # Train model
            self.tfidf_models[ind].train_model()
                
            # Save model
            self.tfidf_models[ind].save(f"models/{self.categories[ind]}_tfidf.pkl")
            self.tfidf_models[ind] = None
                
                        
            training_data[ind] = []
            
            #Update progress bar
            #pbar.update(1)
        
        print("TRAINING DATA")
        '''with tqdm(total=len(self.categories)) as pbar:
            for category in range(len(self.categories)):
                
                # Train model
                self.tfidf_models[category].train_model()
                
                # Save model
                self.tfidf_models[category].save(f"models/{self.categories[category]}_tfidf.pkl")
                
                # Unload model
                #print(f'category {self.categories[category]} gets unloaded')
                self.tfidf_models[category] = None
                training_data[category] = None
                
                pbar.update(1)'''
                
        print("Training complete.")

    
        
    def predict(self, input_data, confidence_threshold=1.5):
        # Get category confidence scores from qc_model
        category_confidences = self.qc_model.predict(input_data)
        #print("Category confidences:", category_confidences)
        
        # Find the indices of categories with confidence scores above the threshold
        confident_indices = (category_confidences > confidence_threshold).nonzero()[:,1]
        
        #print(confident_indices)
        
        max_confidence = 0
        max_answer = None
        max_category = 0
        for category in confident_indices:
            #print(category)
            confidence,answer = self.tfidf_models[category].predict(input_data)
            
            if(confidence > max_confidence):
                max_confidence = confidence
                max_answer = answer
                max_category = category
            
        #max_confidence, max_answer = selected_model.predict(input_data)
        #print("Prediction for category", self.categories[category], ":", max_answer, "with confidence", max_confidence)
        
        return (np.tanh(max_confidence), max_answer)
        
    def evaluate(self, input_data):
        correct = 0
        count = 0
        
        with tqdm(total=len(input_data)) as pbar: 
          for data_point in input_data:
              print(count % 10)
              count += 1
              text = data_point["text"]
              answer = data_point["answer"]
          
              answer_predict = self.predict(text)[1]
              
              if(answer == answer_predict):
                  correct += 1
                  print(correct)
                
              if(count % 10 == 0):
                  average = float(correct)/count
                  print(f'rolling average: {average}')
                
              pbar.update(1)
        
          
          accuracy = correct/len(input_data)
          
          return accuracy
        
            
        

        
        
        
        
        
        
if __name__ == "__main__":
    # Train a simple model on QB data, save it to a file
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--predict', type=str)
    parser.add_argument('--clear', action='store_const', const=True, default=False)
    parser.add_argument('--evaluate', type=str)

    flags = parser.parse_args()
    model = None
    
    print(flags.clear)

    if flags.clear:

        model = QuizBowlModel(clear=True)
        
    else:

        model = QuizBowlModel()

        

    if flags.data:
        
        data_json = []
        
        for data in flags.data:
            with open(flags.data, 'r') as data_file:
                data_json.extend(json.load(data_file))

                model.train(data_json)
            #print(model.predict("My name is bobby, bobby newport. your name is jeff?"))
            #model.save("model.pkl")

    if flags.model:
        model.load(flags.model)
    
    if flags.predict:
        print(model.predict(flags.predict))
        
    if flags.evaluate:
        with open(flags.evaluate, 'r') as data_file:
          data_json = json.load(data_file)
          print(f'accuracy: {model.evaluate(data_json)}')
        