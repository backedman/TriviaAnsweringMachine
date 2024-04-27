import json
from bs4 import BeautifulSoup
import re
from tqdm import tqdm  # Import tqdm for progress tracking
import sys

def remove_newline(string):
    return re.sub('\n+', ' ', string)

def clean_text(text, answer):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    #text = re.sub(r'[^a-zA-Z.-\s]', '', text)
    
    # Remove answer from text
    try:
        # Preprocess the answer to replace underscores with spaces
        processed_answer = answer.replace('_', ' ')
        
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
        "data/wiki_page_text.json",
        "data/wiki_text_2.json"
    ]
    wiki_data = []
    for file_path in wiki_files:
        with open(file_path, "r") as f:
            wiki_data.extend(json.load(f))

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

            training_entry = {
                "text": clean_question,
                "answer": answer#,
                # Mohit, put categorizing code here
                #"category": "Unknown"
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
            


            training_entry = {
                "text": text,
                "answer": page#,
                # Mohit, put categorizing code here
                #"category": "Unknown"
            }

            training_data.append(training_entry)

        json.dump(training_data, f, indent=4)
        
process_data()
