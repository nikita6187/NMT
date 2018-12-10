import math
import pandas as pd
from collections import Counter
import operator
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


def read_file(file_path, max_amount=math.inf):
    """
    Reads a number of lines from file and return list of lines
    :param file_path:
    :param max_amount:
    :return: list[str]
    """
    line_list = []

    with open(file=file_path) as f:
        curr_idx = 0
        while curr_idx < max_amount:
            line_list.append(f.readline())
            curr_idx += 1

    return line_list


def process_word(word):
    """
    Returns process word
    :param word:
    :return: str word
    """
    # lower
    word = word.lower()

    # Remove stop
    if word in en_stop:
        return None

    # Remove symbols
    if len(word) < 3:
        return None

    # Get morph
    t_word = wn.morphy(word)
    if t_word is not None:
        word = t_word

    # Get lemma
    word = WordNetLemmatizer().lemmatize(word)
    return word


def process_bpe(line_list, seq="@@", rm_words=["\n"]):
    """
    Transforms line list form read_file to list[list[str=words]], and bpe words merged together
    :param line_list:
    :param seq:
    :param rm_words:
    :return: ist[list[str=words]]
    """
    full_word_list = []
    for line in line_list:
        temp_words = line.split()
        full_words = []
        idx = 0
        while idx < len(temp_words):
            if temp_words[idx].endswith(seq):
                temp_str = ""
                while temp_words[idx].endswith(seq):
                    temp_str += temp_words[idx][:-(len(seq))]
                    idx += 1
                temp_str += temp_words[idx]
                if temp_str not in rm_words:
                    w = process_word(temp_str)
                    if w is not None:
                        full_words.append(w)
                idx += 1
            else:
                if temp_words[idx] not in rm_words:
                    w = process_word(temp_words[idx])
                    if w is not None:
                        full_words.append(w)
                idx += 1
        full_word_list.append(full_words)
    return full_word_list


def get_word_count(word_list):
    """
    Returns word count.
    :param word_list:
    :return: List[(str=word, int=count)]
    """

    word_count = {}
    for line in word_list:
        for word, val in Counter(line).most_common():
            if word in word_count:
                word_count[word] += val
            else:
                word_count[word] = val

    return sorted(word_count.items(), key=operator.itemgetter(1))


def get_topics_lda(word_list):
    from gensim import corpora
    dictionary = corpora.Dictionary(word_list)
    corpus = [dictionary.doc2bow(text) for text in word_list]

    import gensim
    NUM_TOPICS = 5
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    topics = ldamodel.print_topics(num_words=10)
    for topic in topics:
        print(topic)

    import pyLDAvis.gensim
    lda_display = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False, sort=False)
    pyLDAvis.display(lda_display)

    return topics, ldamodel


l = read_file('./trg.shuf', 10000)
wl = process_bpe(line_list=l)
print(get_word_count(word_list=wl))
get_topics_lda(word_list=wl)


