#import word2vec
import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
from gensim.models.word2vec import Word2Vec


'''word2vec.word2phrase('/home/kumud/Desktop/TheanoPrep/11', '/home/kumud/Desktop/TheanoPrep/text-phrases', verbose=True)

word2vec.word2vec('/home/kumud/Desktop/TheanoPrep/11', '/home/kumud/Desktop/TheanoPrep/text.bin', size=100, verbose=True)

word2vec.word2clusters('/home/kumud/Desktop/TheanoPrep/11', '/home/kumud/Desktop/TheanoPreps/text-clusters.txt', 100, verbose=True)

model = word2vec.load('/home/kumud/Desktop/TheanoPrep/text.bin')

model.vocab
model.vectors.shape
#print model'''

#print model.vectors
#cPickle.dump([model.vectors], open("vec.p", "wb"))
#np.savetxt('vvv.txt', model.vectors,delimiter=" ", fmt="%s")
#np.savetxt('mv.txt', model.vocab,delimiter=" ", fmt="%s")

#model['ke'].shape

model = Word2Vec.load_word2vec_format('text.bin', binary=True)
#model.most_similar(['eka'])
print model.vocab
