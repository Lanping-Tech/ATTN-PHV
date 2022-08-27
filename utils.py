
alphabets = ['', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']

def num_word(word_size=3):
    return len(alphabets) ** word_size - toID([alphabets[1] for _ in range(word_size)]) + 1

def toID(word):
    word_id = 0
    for alphabet in word:
        assert alphabet in alphabets
        word_id = word_id * len(alphabets) + alphabets.index(alphabet)
    return word_id

def word2id(word):
    startID = toID([alphabets[1] for _ in range(len(word))])
    return toID(word) - startID + 1

def id2word(word_id, word_size=3):
    word_id = word_id + toID([alphabets[1] for _ in range(word_size)]) - 1
    word = ''
    while word_id > 0:
        word = alphabets[word_id % len(alphabets)] + word
        word_id = word_id // len(alphabets)
    return word

def seq2ids(seq, word_size=3, sride=3):
    word_ids = []
    for i in range(0, len(seq) - word_size + 1, sride):
        word_ids.append(word2id(seq[i:i+word_size]))
    return word_ids

def ids2seq(word_ids):
    seq = ''
    for id in word_ids:
        seq += id2word(id)
    return seq
