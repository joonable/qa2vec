# Doc2Vec Model
#---------------------------------------
#
# In this example, we will download and preprocess the movie
#  review data.
#
# From this data set we will compute/fit a Doc2Vec model to get
# Document vectors.  From these document vectors, we will split the
# documents into train/test and use these doc vectors to do sentiment
# analysis on the movie review dataset.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import pickle
import string
import requests
import collections
import io
import tarfile
import urllib.request
import json
import text_helpers

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Make a saving directory if it doesn't exist
data_folder_name = 'temp'
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

# Start a graph session
sess = tf.Session()

with open('./config.json') as json_file:
    conf = json.load(json_file)

# Declare model parameters
batch_size = conf['batch_size']
vocab_size = conf['vocab_size']
iterations = conf['iterations']
lr = conf['lr']

# embedding_size = 200   # Word embedding size
word_emb_size = conf['word_emb_size']
doc_emb_size = conf['doc_emb_size']   # Document embedding size
concatenated_size = word_emb_size + doc_emb_size

num_sampled = int(batch_size/2)    # Number of negative examples to sample.
window_size = conf['window_size']       # How many words to consider to the left.

# Add checkpoints to training
save_embeddings_every = conf['save_embeddings_every']
print_valid_every = conf['print_valid_every']
print_loss_every = conf['print_loss_every']

epoch=100


# Declare stop words
#stops = stopwords.words('english')
stops = []

# We pick a few test words for validation.
valid_words = ['고객', '배송', '주문', '결제', '환불', '방송']
# Later we will have to transform these into indices

# Load the movie review data
print('Loading Data')
# texts, target = text_helpers.load_movie_data()

texts, target = text_helpers.load_dataset()

# Normalize text
print('Normalizing Text Data')
print(texts[0])
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(dm=0, window=window_size, min_count=5, workers=8, vector_size=doc_emb_size, dm_concat=1)
model.build_vocab(documents)

for i in range(100):
    print('epoch {0}'.format(i))
    model.train(documents, total_examples=model.corpus_count, epochs=model.iter)

model.save('./temp/doc2vec_gensim.model')
print("Model Saved")
