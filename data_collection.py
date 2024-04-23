import json
import xml.etree.ElementTree as ET

# Function to extract Wikipedia page title, text content, and category from a data dump file
def extract_wikipedia_pages(data_dump_file):
    with open(data_dump_file, 'r', encoding='utf-8') as f:
        tree = ET.iterparse(f)
        for _, elem in tree:
            if elem.tag.endswith('title'):
                title = elem.text
            elif elem.tag.endswith('text'):
                text = elem.text
                category = determine_category(title)
                yield {
                    "text": text,
                    "answer": title,
                    "category": category
                }
            elem.clear()

# Function to determine the category of a Wikipedia page based on its title
def determine_category(title):
    for category, mappings in category_mappings.items():
        if any(mapping.lower() in title.lower() for mapping in mappings):
            return category
    return None

# Load the provided data mappings
with open('category_mappings.json', 'r') as f:
    category_mappings = json.load(f)

# Example usage
data_dump_file = 'path/to/your/data/dump.xml'
output_file = "wikipedia_data.json"

# Open output file
with open(output_file, "w") as out_file:
    out_file.write("[")  # Start of the list
    first_entry = True
    # Iterate through Wikipedia data and write to JSON file as it is yielded
    for page_data in extract_wikipedia_pages(data_dump_file):
        if not first_entry:
            out_file.write(",")  # Add comma between entries except for the first one
        json.dump(page_data, out_file)
        first_entry = False
    out_file.write("]")  # End of the list

print("Wikipedia data has been written to", output_file)
