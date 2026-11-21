import numpy as np
from collections import Counter

class BPETokenizer:
    def __init__(self,vocab_size):
        self.pref_vocab_size = vocab_size
        self.vocab = {}
        self.rev_vocab = {}
        self.merges = []

    def train(self,text):
        chars = sorted(set(text))
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.reverse_vocab = {i: char for char, i in self.vocab.items()}
        tokens = [self.vocab[char] for char in text]
        while len(self.vocab) < self.pref_vocab_size:
            pairs = Counter(zip(tokens,tokens[1:]))

            if not pairs:
                break

            (id1,id2),frequency = pairs.most_common(1)[0]

            if frequency < 2:
                break

            idnew = len(self.vocab)

            new_token = self.reverse_vocab[id1] + self.reverse_vocab[id2]
            self.vocab[idnew] = new_token

            self.reverse_vocab[idnew] = new_token
            self.merges.append((id1,id2,idnew))

            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == id1
                    and tokens[i + 1] == id2
                ):
                    new_tokens.append(idnew)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens


    def encode(self,text):
        tokens = [self.vocab[char] for char in text]

        for id1, id2, idnew in self.merges:
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == id1
                    and tokens[i + 1] == id2
                ):
                    new_tokens.append(idnew)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens
    def decode(self,tokens):
        return "".join(self.reverse_vocab[token] for token in tokens)
    

        


training_data = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"

Tokenizer = BPETokenizer(18)

Tokenizer.train(training_data)

print(Tokenizer.encode("lowest"))
print(Tokenizer.decode(Tokenizer.encode("lowest newest widest")))