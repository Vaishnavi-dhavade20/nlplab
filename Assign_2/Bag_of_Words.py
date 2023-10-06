from gensim.utils import simple_preprocess
from gensim import corpora

text = open('sample_text.txt', encoding='utf-8')

tokens = []
for line in text.read().split('.'):
    tokens.append(simple_preprocess(line, deacc=True))

g_dict = corpora.Dictionary(tokens)

print("The dictionary has: " + str(len(g_dict)) + " tokens\n")
print(g_dict.token2id)

g_bow = [g_dict.doc2bow(token, allow_update=True) for token in tokens]
print("Bag of Words : ", g_bow)

'''
Output:

The dictionary has: 57 tokens

{'analyzing': 0, 'and': 1, 'branch': 2, 'consists': 3, 'data': 4, 'deriving': 5, 'efficient': 6, 'for': 7, 'from': 8, 'in': 9, 'information': 10, 'is': 11, 'manner': 12, 'nlp': 13, 'of': 14, 'processes': 15, 'science': 16, 'smart': 17, 'systematic': 18, 'text': 19, 'that': 20, 'the': 21, 'understanding': 22, 'analysis': 23, 'as': 24, 'automated': 25, 'automatic': 26, 'by': 27, 'can': 28, 'chunks': 29, 'components': 30, 'entity': 31, 'etc': 32, 'extraction': 33, 'its': 34, 'machine': 35, 'massive': 36, 'named': 37, 'numerous': 38, 'one': 39, 'organize': 40, 'perform': 41, 'problems': 42, 'range': 43, 'recognition': 44, 'relationship': 45, 'segmentation': 46, 'sentiment': 47, 'solve': 48, 'speech': 49, 'such': 50, 'summarization': 51, 'tasks': 52, 'topic': 53, 'translation': 54, 'utilizing': 55, 'wide': 56}
Bag of Words :  [[(0, 1), (1, 2), (2, 1), (3, 1), (4, 2), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1), (14, 2), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1)], [(1, 3), (4, 1), (13, 1), (14, 2), (19, 1), (21, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1), (29, 1), (30, 1), (31, 1), (32, 1), (33, 1), (34, 1), (35, 1), (36, 1), (37, 1), (38, 1), (39, 1), (40, 1), (41, 1), (42, 1), (43, 1), (44, 2), (45, 1), (46, 1), (47, 1), (48, 1), (49, 1), (50, 1), (51, 1), (52, 1), (53, 1), (54, 1), (55, 1), (56, 1)], []]
'''
