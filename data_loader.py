# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:05:14 2019

@author: siddh
"""

import torch
#import torchvision.transforms as transforms
import torch.utils.data as data
import os
#import pickle
#import numpy as np
import nltk
from PIL import Image
#from build_vocab import Vocabulary

class FlickrDataset(data.Dataset):
    """Flickr Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, desc_path, vocab, transform=None, size=256):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.desc = self.create_descriptions(desc_path)
        self.ids = list(range(len(self.desc)))
        #self.ids=i.keys() for i in desc]
        self.vocab = vocab
        self.transform = transform
        self.size = [size, size]

    def create_descriptions(self, desc_path):
        with open(desc_path, 'r') as f:
            data = f.readlines()
        desc = list()
        for line in data:
            d = dict()
            d['image_id'] = line.split()[0]
            d['caption'] = ' '.join(line.split()[1:])
            desc.append(d)
        return desc
    
    def resize_image(self, image, size):
        return image.resize(size, Image.ANTIALIAS)
    
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        desc = self.desc
        vocab = self.vocab
        ann_id = self.ids[index]
        #caption = coco.anns[ann_id]['caption']
        caption = desc[ann_id]['caption']
        img_id = desc[ann_id]['image_id']
        #path = coco.loadImgs(img_id)[0]['file_name']
        path = str(img_id)+'.jpg'

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = self.resize_image(image, self.size)
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, desc_path, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom Flickr dataset."""
    # Flickr caption dataset
    flickr = FlickrDataset(root=root,
                       desc_path=desc_path,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for Flickr dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=flickr, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader