import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def remove_stopwords(text):
    # Parse the text using spaCy
    doc = nlp(text)

    # Create a list of tokens without stopwords
    tokens_without_stopwords = [token.text for token in doc if not token.is_stop]

    # Reconstruct the text without stopwords
    filtered_text = ' '.join(tokens_without_stopwords)

    return filtered_text

# Example usage:
input_text = "This is an example sentence with some stopwords that we want to remove."
filtered_text = remove_stopwords(input_text)
print(filtered_text)


#OUTPUT :
#example sentence stopwords want remove
