import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    # Parse the text using spaCy
    doc = nlp(text)

    # Lemmatize each token in the document
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    return lemmatized_text

# Example usage:
input_text = "Lemmatization is a process of reducing words to their base or dictionary form."
lemmatized_text = lemmatize_text(input_text)
print(lemmatized_text)


#OUTPUT :
#lemmatization be a process of reduce word to their base or dictionary form
