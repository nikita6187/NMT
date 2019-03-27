import numpy as np
import gzip
import pickle
import copy
import sys
import json
import math

# USAGE: python3 mg_random.py <max_size_of_segment> <save path>

# USAGE: if you want to save the mapping, provide the save path as the second argument

# Paths, only on target training dataset!
VOCAB_FILE = "/work/smt2/bahar/experiment/data-raw/vocabs-bpe20k/target.vocab.pkl"

# Import vocab
with open(VOCAB_FILE, "rb") as w:
    vocab = pickle.load(w)
print("Len of vocab: " + str(len(vocab.keys())))
print("Fin Vocab")


# generate vocab_distribution randomly
def chunks(l, n):
    m = n
    r = [l[i:i+m] for i in range(0, len(l), m)]
    p = max(range(0, len(l), m)) + m
    print("---")
    print(p)
    print(len(l))
    if p < len(l)-1:
        r.append(l[p:])
    return r


vocab_distribution = list(vocab.values())
np.random.shuffle(vocab_distribution)
vocab_distribution = chunks(vocab_distribution, int(sys.argv[1]))

# Create topics out of distributions
topics = copy.deepcopy(vocab_distribution)

# add <S>, </S> and <UNK> to each
to_add = ['<S>', '</S>', '<UNK>']
all_words_non_training = [v.extend(to_add) for v in topics]

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


