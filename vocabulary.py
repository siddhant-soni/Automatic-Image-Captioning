# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:16:31 2019

@author: siddh
"""
import string
import os
import nltk
import pickle
from collections import Counter


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(threshold):
    """Build a simple vocabulary wrapper."""
    #coco = COCO(json)
    counter = Counter()
    #ids = coco.anns.keys()
    images_path = 'datasets/Flicker8k_Dataset/'
    image_text_path = 'datasets/Flickr8k_text/Flickr8k.token.txt'
    jpgs = os.listdir(images_path)
    print("No. of images: {}".format(len(jpgs)))
    with open(image_text_path, 'r') as f:
        text = f.read()
        
    descriptions = dict()
    for line in text.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)
    
    print("No. of desc: {}".format(len(descriptions)))
    
    table = str.maketrans('','',string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] =  ' '.join(desc)
        descriptions[key] = desc_list
    ids = descriptions.keys()
    for i, id in enumerate(ids):
        caption = descriptions[id]
        for cap in caption:
            tokens = nltk.tokenize.word_tokenize(cap.lower())
            counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main():
    vocab = build_vocab(4)
    vocab_path = 'vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))