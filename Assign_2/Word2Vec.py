from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count

print("\nWord2vec \n")

data = [
    "this is a sentence",
    "another example sentence",
    "word embeddings are useful",
    "word2vec is a popular model",
]

# Tokenize the text data (split sentences into words)
tokenized_data = [sentence.split() for sentence in data]

# Create a Word2Vec model
w2v_model = Word2Vec(tokenized_data, min_count=0, workers=cpu_count())

# Find the most similar words to 'word'
similar_words = w2v_model.wv.most_similar('word')

for word, score in similar_words:
    print(f"{word}: {score}")

'''
Output:

Word2vec 

sentence: 0.21617141366004944
embeddings: 0.044689226895570755
example: 0.015025208704173565
useful: 0.0019510718993842602
is: -0.03284316137433052
this: -0.04568909481167793
another: -0.0742427185177803
are: -0.09326908737421036
a: -0.09575342386960983
word2vec: -0.10513807833194733
'''