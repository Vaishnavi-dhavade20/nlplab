import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

custom_about_text = (
    "hello vaishnavi is this side"
    " uploaded token file "
    " thank you"
    
)
nlp = spacy.load("en_core_web_sm")
about_doc = nlp(custom_about_text)
print([token for token in about_doc if not token.is_stop])

#OUTPUT :
#[hello, vaishnavi, uploaded, token, file,  , thank]