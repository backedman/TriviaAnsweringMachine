import json
from bs4 import BeautifulSoup
import re
from tqdm import tqdm  # Import tqdm for progress tracking
import sys
import question_categorizer as qc
import numpy as np
from question_categorizer import TextClassificationModel

qc_model = qc.TextClassificationModel.load_model("models/categorizer")

categories = ['Geography', 'Religion', 'Philosophy', 'Trash','Mythology', 'Literature','Science', 'Social Science', 'History', 'Current Events', 'Fine Arts']

def remove_newline(string):
    return re.sub('\n+', ' ', string)

def clean_text(text, answer):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    #text = re.sub(r'?','.',text)
    text = text.replace('?','.')
    
    # Clean the text further
    text = re.sub(r'[^a-zA-Z.\s-]', '', text)
    
    
    
    # Remove answer from text
    try:
        # Preprocess the answer to replace underscores with spaces
        processed_answer = answer.replace('_', ' ')
        
        # Remove parentheses from the processed answer
        processed_answer = re.sub(r'\([^)]*\)', '', processed_answer)
        
        # Replace all instances of the processed answer with an empty string, ignoring case
        text = re.sub(re.escape(processed_answer), '', text, flags=re.IGNORECASE)
    except Exception as e:
        print("An error occurred during text cleaning:", e)
        print("Text:", text)
        print("Answer:", answer)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_data():
    with open("data/JEOPARDY_QUESTIONS1.json", "r") as f:
        jeopardy_data = json.load(f)

    wiki_files = [
        "wiki_page_text.json",
        "wiki_text_2.json"
    ]
    
    question_files = ["JEOPARDY_QUESTIONS1.json",
        "qadata4.json",
        "qadata.json",
        "qadata2.json",
        "qadata5.json",
        "quizbowl_2021_and_prior_RAW.json",
        "qadata3.json",
        "qadata6.json",
        "quizbowl_2021_standardized_answers.json"]
    
    wiki_data = []
    
    for file_path in wiki_files:
        with open('data/' + file_path, "r") as f:
            wiki_data.extend(json.load(f))
            
    for file_path in question_files:
        with open('data/' + file_path, "r") as f:
            jeopardy_data.extend(json.load(f))

    with open("data/training_data.json", "w") as f:
        training_data = []

        # Process Jeopardy data
        print("Processing Jeopardy data...")
        for entry in tqdm(jeopardy_data):
            question = entry["question"]
            answer = str(entry["answer"])

            # Preprocess the text
            soup = BeautifulSoup(question, 'html.parser')
            clean_question = ''.join(soup.findAll(text=True, recursive=False))
            
            question_category = []
            
            # Get category from qc_model
            prediction = qc_model.predict(question)
            predictions = np.argwhere(prediction >= 1.5)[1]
            
            for prediction_ind in predictions:
                # Store data in array with respective index
                question_category.append(categories[prediction_ind])
                
            question_category.append('ALL')
            
            

            training_entry = {
                "text": clean_question,
                "answer": answer,#,
                # Mohit, put categorizing code here
                "category": question_category
            }

            training_data.append(training_entry)

        # Process Wikipedia data
        print("Processing Wikipedia data...")
        for entry in tqdm(wiki_data):
            page = str(entry["page"])
            text = entry["text"]
            
            if(text == ""):
                continue
            
            text = remove_newline(text)
            text = clean_text(text, page)
            
            question_category = []
            
            # Get category from qc_model
            prediction = qc_model.predict(text)
            predictions = np.argwhere(prediction >= 1.5)[1]
            
            for prediction_ind in predictions:
                # Store data in array with respective index
                question_category.append(categories[prediction_ind])
                
            question_category.append('ALL')
            


            training_entry = {
                "text": text,
                "answer": page,
                # Mohit, put categorizing code here
                "category": question_category
            }

            training_data.append(training_entry)

        json.dump(training_data, f, indent=4)
        
process_data()
