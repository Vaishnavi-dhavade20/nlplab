from gensim.utils import simple_preprocess
from gensim import corpora, models
import numpy as np
text = [
        "The weather today is sunny and warm, perfect for outdoor activities.",
        "I have a lot of work to do, and the deadline is approaching fast.",
        "I enjoy reading books in my free time, especially science fiction novels."
        ]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

print("\nTerm Frequency – Inverse Document Frequency (TF-ID)")
print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("\n TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])


'''
Output:

Term Frequency - Inverse Document Frequency (TF-ID)
Dictionary : 
[['activities', 1], ['and', 1], ['for', 1], ['is', 1], ['outdoor', 1], ['perfect', 1], ['sunny', 1], ['the', 1], ['today', 1], ['warm', 1], ['weather', 1]]
[['and', 1], ['is', 1], ['the', 1], ['approaching', 1], ['deadline', 1], ['do', 1], ['fast', 1], ['have', 1], ['lot', 1], ['of', 1], ['to', 1], ['work', 1]]
[['books', 1], ['enjoy', 1], ['especially', 1], ['fiction', 1], ['free', 1], ['in', 1], ['my', 1], ['novels', 1], ['reading', 1], ['science', 1], ['time', 1]]

 TF-IDF Vector:
[['activities', 0.34], ['and', 0.17], ['for', 0.34], ['is', 0.17], ['outdoor', 0.34], ['perfect', 0.34], ['sunny', 0.34], ['the', 0.17], ['today', 0.34], ['warm', 0.34], ['weather', 0.34]]
[['and', 0.16], ['is', 0.16], ['the', 0.16], ['approaching', 0.32], ['deadline', 0.32], ['do', 0.32], ['fast', 0.32], ['have', 0.32], ['lot', 0.32], ['of', 0.32], ['to', 0.32], ['work', 0.32]]     
[['books', 0.3], ['enjoy', 0.3], ['especially', 0.3], ['fiction', 0.3], ['free', 0.3], ['in', 0.3], ['my', 0.3], ['novels', 0.3], ['reading', 0.3], ['science', 0.3], ['time', 0.3]]
'''