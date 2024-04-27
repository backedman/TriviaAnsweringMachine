from typing import List, Tuple
import nltk
import sklearn
import question_categorizer as qc
import tfidf_model
import transformers
import numpy as np
import pandas as pd





class QuizBowlModel:

    def __init__(self):
        """
        Load your model(s) and whatever else you need in this function.

        Do NOT load your model or resources in the guess_and_buzz() function, 
        as it will increase latency severely. 
        """
        
        self.categories = ['Geography', 'Religion', 'Philosophy', 'Trash', 'Mythology', 'Literature', 'Science', 'Social Science', 'History', 'Current Events', 'Fine Arts']
        self.tfidf_models = ['tfidf_Geography', 'tfidf_Religion', 'tfidf_Philosophy', 'tfidf_Trash', 'tfidf_Mythology', 'tfidf_Literature', 'tfidf_Science', 'tfidf_Social_Science', 'tfidf_History', 'tfidf_Current_Events', 'tfidf_Fine_Arts']
        self.qc_model = qc.TextClassificationModel.load_model("categorizer")



        pass

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        This function accepts a list of question strings, and returns a list of tuples containing
        strings representing the guess and corresponding booleans representing 
        whether or not to buzz. 

        So, guess_and_buzz(["This is a question"]) should return [("answer", False)]

        If you are using a deep learning model, try to use batched prediction instead of 
        iterating using a for loop.
        """
        pass
    
    def train(self, data):
        
        #go through data
        for data in data_json:
            text = data["text"]
            answer = data["answer"]
        
            #get category from qc_model
        
        #store in array with respective index
        
        #create respective model if not exist
        
        #toss in data from array to respective model's process_data
        
        #call train_model() on each
        
        #call save_model() on each
        
        
