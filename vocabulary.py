"""
Shared Vocabulary class — imported by both train.py and chatbot.py
so pickle can find it correctly in both contexts.
"""

from collections import Counter

PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
MAX_LEN = 256


def tokenize(text):
    return text.lower().strip().split()


class Vocabulary:
    def __init__(self):
        self.token2idx = {PAD: 0, BOS: 1, EOS: 2, UNK: 3}
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.freq = Counter()

    def build(self, sentences, min_freq=1):
        for sent in sentences:
            self.freq.update(tokenize(sent))
        for token, freq in self.freq.items():
            if freq >= min_freq and token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
        print(f"[Vocab] Size: {len(self.token2idx)}")

    def encode(self, text, max_len=MAX_LEN):
        tokens = [BOS] + tokenize(text)[:max_len - 2] + [EOS]
        return [self.token2idx.get(t, self.token2idx[UNK]) for t in tokens]

    def decode(self, indices):
        result = []
        for i in indices:
            t = self.idx2token.get(i, UNK)
            if t == EOS:
                break
            if t not in (BOS, PAD):
                result.append(t)
        return " ".join(result)

    def __len__(self):
        return len(self.token2idx)
