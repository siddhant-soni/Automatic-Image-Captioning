# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:42:19 2019

@author: siddh
"""

import torch
import matplotlib.pyplot as plt
import numpy as np 
import pickle
from torchvision import transforms 
from vocabulary import Vocabulary
from models import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_path = 'vocab.pkl'
embed_size = 256
hidden_size = 512
num_layers = 1
encoder_path = 'models/encoder-10-200.ckpt'
decoder_path = 'models/decoder-10-200.ckpt'
#encoder_path = 'models/encoder-10-300.ckpt'
#decoder_path = 'models/decoder-10-300.ckpt'
image_path = 'umair.jpeg'

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])

# Load vocabulary wrapper
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Build models
encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
encoder = encoder.to(device)
decoder = decoder.to(device)

# Load the trained model parameters
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

# Prepare an image
image = load_image(image_path, transform)
image_tensor = image.to(device)

# Generate an caption from the image
feature = encoder(image_tensor)
sampled_ids = decoder.sample(feature)
sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

# Convert word_ids to words
sampled_caption = []
for word_id in sampled_ids:
    word = vocab.idx2word[word_id]
    sampled_caption.append(word)
    if word == '<end>':
        break
sentence = ' '.join(sampled_caption)

# Print out the image and the generated caption
image = Image.open(image_path)
plt.imshow(np.asarray(image))
plt.title(sentence)
print ('\n\n\n\n\n\n')
print (sentence)