'''
Assignment No: 03 (Spacy)
Name: vaishnavi dhavade
Roll: 19
Batch: B1
Title: Implements Named Entity Recognition(NER) on textual data using SpaCy or NLTK library.
'''



import spacy

raw_text=("The Board of Control for Cricket in India (BCCI) is the governing body for cricket in India"
          " and is under the jurisdiction of Ministry of Youth Affairs and Sports, Government of India. "
          "[2] The board was formed in December 1928 as a society, registered under the Tamil Nadu Societies "
          "Registration Act. It is a consortium of state cricket associations and the state associations select "
          "their representatives who in turn elect the BCCI Chief. Its headquarters are in Wankhede Stadium, Mumbai."
          "Grant Govan was its first president and Anthony De Mello its first secretary.")

NER = spacy.load("en_core_web_sm",disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

text = NER(raw_text)

for x in text.ents:
    print(x.text,x.label_)

#spacy.displacy.render(text, style="ent", jupyter=True)

# spacy.explain(u"NORP")


'''
Output

The Board of Control for Cricket ORG
India GPE
BCCI ORG
India GPE
Ministry of Youth Affairs ORG
Sports, Government of India ORG
2 CARDINAL
December 1928 DATE
the Tamil Nadu Societies Registration Act ORG
BCCI ORG
Wankhede Stadium FAC
Mumbai GPE
Grant Govan PERSON
first ORDINAL
Anthony De Mello PERSON
first ORDINAL
'''