import numpy as np
import gzip
import pickle
import copy
import sys
import json
import math

# USAGE: python3 mg_domain_segmentation_subdivide.py <amount to subdivide> <save path>

# USAGE: if you want to save the mapping, provide the save path as the second argument

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
print("Len of vocab: " + str(len(vocab.keys())))
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
amount_of_domains = len(all_lines)
vocab_distribution = [[] for _ in range(amount_of_domains)]

# Go through all texts, and segment where each vocabulary item ends up
for text, idx in zip(all_lines, range(amount_of_domains)):
    # text now contains each line
    for line in text:
        # Get individual tokens
        tokens = line.split()
        for token in tokens:
            token = token.decode("utf-8")
            if token in vocab:
                vocab_distribution[idx].append(token)

vocab_distribution = [list(set(v)) for v in vocab_distribution]
print("Fin segmentation")


# Sub segment
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


temp_dis = []
for v in vocab_distribution:
    temp_dis.extend(list(chunks(v, int(sys.argv[1]))))
vocab_distribution = temp_dis

print("Amount of vocab in each topic: ")
print([len(dis) for dis in vocab_distribution])

# Visualize overlap
overlap = np.zeros((amount_of_domains, amount_of_domains))
for t1 in range(amount_of_domains):
    for t2 in range(amount_of_domains):
        overlap[t1, t2] = len(set(vocab_distribution[t1]).intersection(set(vocab_distribution[t2])))
print(overlap)


# Create topics out of distributions
topics = copy.deepcopy(vocab_distribution)

# add <S>, </S> and <UNK> to each
to_add = ['<S>', '</S>', '<UNK>']
for v in topics:
    v.extend(to_add)
topics = [list(set(v)) for v in topics]

print("Amount of vocab in each topic: ")
print([len(set(dis)) for dis in topics])

# convert to indices
topic_dic = {}
for t, idx in zip(topics, range(len(topics))):
    for word in t:
        if vocab[word] not in topic_dic.keys():
            topic_dic[vocab[word]] = (word, [idx])
        else:
            topic_dic[vocab[word]][1].append(idx)
            topic_dic[vocab[word]] = (topic_dic[vocab[word]][0], list(set(topic_dic[vocab[word]][1])))


print(topic_dic[0])
print(topic_dic[5])
print(topic_dic[100])

# save topic_dic in json
if len(sys.argv) == 3:
    print("Saving json")
    with open(sys.argv[2], 'w') as fp:
        json.dump(topic_dic, fp)


