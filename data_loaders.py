from collections import defaultdict
import random
from spacy.lang.en.stop_words import STOP_WORDS
import re
import torch
import numpy as np

# function to prepare captions
def preprocessing(text, length = 20):
  prepared = []
  for sent in text:
    sent = re.sub('\n', '', sent)
    sent = re.sub("\r", "", sent)
    sent = re.sub('[^\w\s]+', '', sent)
    #in the beginning delete too long sentences
    sent = [i for i in sent.split() if i not in STOP_WORDS and len(i) > 1][:length + 1]
    #add start and end
    sent  = ["<START>"] + sent + ["<END>"]
    sent = " ".join(sent)
    prepared.append(sent)
  return prepared

#class for all of the data
class Vocabulary:
  def __init__(self):
    self.word_to_idx = {"<START>" : 0, "<END>": 1, "<PAD>": 2}
    self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}

  def build_vocab(self, sentences, condition):
    counter = defaultdict(int)
    cur_idx = len(self.word_to_idx) + 1
    for group in sentences:
      for sent in group:
        for word in sent.split():
          counter[word] += 1
          if counter[word] >= condition and word not in self.word_to_idx.keys():
            self.word_to_idx[word] = cur_idx
            self.idx_to_word[cur_idx] = word
            cur_idx += 1

  # get numeric transformation
  def tokenize(self, sentences):
    return [[self.word_to_idx[word] for word in sent.split() if word in self.word_to_idx] for sent in sentences]

  # get word transformation
  def reverse_tokenizer(self, sent):
    return [self.idx_to_word[word] for word in sent if word in self.idx_to_word.keys()]


class CaptionDataset:
  def __init__(self, data_text, data_pictures, test_mode=False):
    self.tokenized = data_text
    self.data_pictures = data_pictures
    self.test_mode = test_mode

  def __len__(self):
    return len(self.data_pictures)

  def __getitem__(self, idx):
    if not self.test_mode:
      text = random.choice(self.tokenized[idx])
    else:
      text = self.tokenized[idx]
    image = self.data_pictures[idx]

    return {"caption": text,
            "image": image}

# get sentences padded
def get_padded(values):
    max_len = 0
    for value in values:
        if len(value) > max_len:
            max_len = len(value)
    padded = np.array([value + [0]*(max_len-len(value)) for value in values])
    return padded

# creating a batch - collect images and get text padded
def collacate_fn(batch):
    imgs = [item["image"] for item in batch]

    description = [item["caption"] for item in batch]
    description = get_padded(description)
    return {"images": torch.tensor(imgs), "caption": torch.tensor(description)}
