import pickle


def convert_vocab_mt_to_lm(source_file, target_file):
    with open(source_file, "rb") as f:
        vocab = pickle.load(f)
        lines = []
        for word in vocab:
            lines.append(word + ' ' + str(vocab[word]) + '\n')
        with open(target_file, "w") as wf:
            wf.writelines(lines)


def convert_vocab_lm_to_mt(source_file, target_file):
    with open(source_file, "rb") as f:
        vocab_lines = f.readlines()
    vocab = {}
    for line in vocab_lines:
        d = line.split()
        vocab[d[0]] = int(d[1])
    pickle.dump(vocab, open(target_file, "wb"))


convert_vocab_lm_to_mt('./lm.vocab.txt', './newvocab.target.vocab.pkl')

#convert_vocab('./target.vocab.pkl', './lm.vocab.txt')
