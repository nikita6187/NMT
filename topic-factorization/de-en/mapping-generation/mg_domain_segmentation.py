import numpy as np
import gzip
import pickle

# Paths, only on target training dataset!
VOCAB_FILE = "/work/smt2/bahar/experiment/data-raw/vocabs-bpe20k/target.vocab.pkl"

TRAIN_ECB = "/work/smt2/bahar/experiment/data-raw/train-bpe20k/train.ECB.de-en.en.pp.tc.bpe.20k.gz"
TRAIN_EMEA = "/work/smt2/bahar/experiment/data-raw/train-bpe20k/train.EMEA.de-en.en.pp.tc.bpe.20k.gz"
TRAIN_JRC = "/work/smt2/bahar/experiment/data-raw/train-bpe20k/train.JRC-Acquis.de-en.en.pp.tc.bpe.20k.gz"
TRAIN_KDE4 = "/work/smt2/bahar/experiment/data-raw/train-bpe20k/train.KDE4.de-en.en.pp.tc.bpe.20k.gz"
TRAIN_NEWS = "/work/smt2/bahar/experiment/data-raw/train-bpe20k/train.News-Commentary.de-en.en.pp.tc.bpe.20k.gz"
TRAIN_TED = "/work/smt2/bahar/experiment/data-raw/train-bpe20k/train.TED2013.de-en.en.pp.tc.bpe.20k.gz"

# Import vocab
with open(VOCAB_FILE, "rb") as w:
    vocab = pickle.load(w)
print("Fin Vocab")

# Import individual texts
with gzip.open(TRAIN_ECB, 'rb') as f:
    LINES_ECB = f.readlines()
print("Fin ECB")

with gzip.open(TRAIN_EMEA, 'rb') as f:
    LINES_EMEA = f.readlines()
print("Fin EMEA")

with gzip.open(TRAIN_JRC, 'rb') as f:
    LINES_JRC = f.readlines()
print("Fin JRC")

with gzip.open(TRAIN_KDE4, 'rb') as f:
    LINES_KDE4 = f.readlines()
print("Fin KDE4")

with gzip.open(TRAIN_NEWS, 'rb') as f:
    LINES_NEWS = f.readlines()
print("Fin NEWS")

with gzip.open(TRAIN_TED, 'rb') as f:
    LINES_TED = f.readlines()
print("Fin TED")

all_lines = [LINES_ECB, LINES_EMEA, LINES_JRC, LINES_KDE4, LINES_NEWS, LINES_TED]
vocab_distribution = [[] for _ in range(len(all_lines))]  # TODO: see if this resuslts in side-effects

# Go through all texts, and segment where each vocabulary item ends up
for text, idx in zip(all_lines, range(len(all_lines))):
    # text now contains each line
    for line in text:
        # Get individual tokens
        tokens = line.split()

        for token in tokens:
            if token in vocab:
                vocab_distribution[idx].append(token)

print("Fin segmentation")

# Debug for now
print([len(dis) for dis in vocab_distribution])




