# CNN for discourse_prediction in a storytelling style speech
Python code for the paper "Sentence Based Discourse Classification for Hindi Story Text-to-Speech (TTS) System"

Runs the model on Hindi Story dataset (created by us). Please cite the original paper when using the data.

# Steps for running the model

1- python process_data.py

2- # Any one of them 

   THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python convolution_net_sentence.py -nonstatic -rand
   THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -static -word2vec
   THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -nonstatic -word2vec 
   
