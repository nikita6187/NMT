import pickle


def convert_vocab(source_file, target_file):
    with open(source_file, "rb") as f:
        vocab = pickle.load(f)
        lines = []
        for word in vocab:
            lines.append(word + ' ' + str(vocab[word]) + '\n')
        with open(target_file, "w") as wf:
            wf.writelines(lines)


convert_vocab('./target.vocab.pkl', './lm.vocab.txt')
