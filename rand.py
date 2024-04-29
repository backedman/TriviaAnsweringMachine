from constituent_treelib import ConstituentTree

# First, we have to provide a sentence that should be parsed
sentence = "I've got a machine learning task involving a large amount of text data."

# Then, we define the language that should be considered with respect to the underlying models 
language = ConstituentTree.Language.English

# You can also specify the desired model for the language ("Small" is selected by default)
spacy_model_size = ConstituentTree.SpacyModelSize.Medium

# Next, we must create the neccesary NLP pipeline. 
# If you wish, you can instruct the library to download and install the models automatically
nlp = ConstituentTree.create_pipeline(language, spacy_model_size) # , download_models=True

# Now, we can instantiate a ConstituentTree object and pass it the sentence and the NLP pipeline
tree = ConstituentTree(sentence, nlp)

# Finally, we can extract the phrases
tree.extract_all_phrases()