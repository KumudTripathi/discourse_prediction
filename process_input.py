import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import csv

def build_data_cv(data_folder, cv=8, clean_string=True):
   """
   Loads data and split into 10 folds.
   """
   revs = [] ;t=[]
   D_file = data_folder[0]
   I_file = data_folder[1]
   N_file = data_folder[2]
   vocab = defaultdict(float)
   with open('homes.txt', 'w') as ff:#print vocab
    with open(D_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
                #t.append((vocab))
                #print type(word)
                #print "vocab ==="+ str(vocab)
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
            t = np.array(t)
            #np.savetxt('v.txt', vocab, delimiter=" ", fmt="%s")
            '''cPickle.dump([vocab], open("vocab.p", "wb"))
            words = np.array(words)
            ff.write("%s\n"%words)#print vocab['eka']#words'''
    with open(I_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(N_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":2, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
            ff.write("%s\n"%vocab)
    return revs, vocab
    
    
def get_W(word_vecs, k=30):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    wrd,wid=[],[]
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        wrd.append(word)
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        wid.append(word_idx_map[word])
        i += 1
    wrd = np.array(wrd)
    wid = np.array(wid,dtype="int")
    np.savetxt('w2v_feature300_278/wrd.txt', wrd,delimiter=" ", fmt="%s")
    np.savetxt('w2v_feature300_278/wid.txt', wid)
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        print "binary_len:" + str(binary_len)
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               #print "word : "+ str(word)
               #print "vocab : " + str(vocab)
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=30):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    #w2v_file = sys.argv[1]     
    data_folder = ["D_278.pok", "I_278.pok","N_278.pok"]    
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, cv=6, clean_string=True)
    t=[]
    for word in vocab.keys():
       t.append((word))#print word
    #np.savetxt('v1_cv8.txt', t, delimiter=" ", fmt="%s")#np.savetxt('filter3/vocab.txt', vocab, delimiter=" ", fmt="%s")
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors..."
    #w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    #print "num words already in word2vec: " + str(len(w2v))
    #add_unknown_words(w2v, vocab)
    #W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, word_idx_map = get_W(rand_vecs)
    
    #print vocab
    cPickle.dump([revs, W2, word_idx_map, vocab], open("new_278_cv8_W2.p", "wb"))
    cPickle.dump([vocab], open("vocab.p", "wb"))
   # np.savetxt('filter3/words.txt', words, delimiter=" ", fmt="%s")
    print "dataset created!"
    #print vocab.shape,W2.shape
    
    
