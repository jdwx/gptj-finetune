import numpy as np
import os

from typing import Iterable


def expand_path(path: str, acceptable_extensions: tuple[str] = None) -> Iterable[str]:
    if acceptable_extensions is None:
        acceptable_extensions = tuple(".txt")
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file.endswith(acceptable_extensions):
                yield os.path.join(root, file)
        for subdir in subdirs:
            yield from expand_path(os.path.join(root, subdir), acceptable_extensions)


class TextDataset:

    def __init__(self, tokenizer):
        self.texts = list()
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng()
        self.token_count = 0

        # Learn the model's max input size and BOS/EOS tokens rather than hard-coding
        self.input_size = self.tokenizer.model_max_length
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return self.token_count

    def add_file(self, file_path: str, verbose: bool = True) -> int:
        with open(file_path, "r") as ff:
            text = ff.read()
        tokens = self.add_text(text)
        if verbose:
            print(f"{file_path} added {tokens} tokens")
        return tokens

    def add_recursive(self, base_path: str, verbose: bool = True) -> int:
        total_added = 0
        for input_file in expand_path(base_path, (".txt",)):
            file_added = self.add_file(input_file)
            total_added += file_added
        if verbose:
            print(f"Added {total_added} total tokens from: {base_path}")
        return total_added

    def add_text(self, text: str) -> int:
        # Convert the text to tokens.
        tokens = self.tokenizer(self.bos_token + text + self.eos_token)

        token_count = len(tokens['input_ids'])
        self.token_count += token_count
        self.texts.append({
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
        })

        return token_count

    def clone_labels(self):
        for text in self.texts:
            text['labels'] = text['input_ids']

    def split_train_and_test(self, train_split: float = None, test_split: float = None):
        if train_split is None and test_split is None:
            train_split = 0.8
            test_split = 1.0 - train_split
        elif train_split is None:
            train_split = 1.0 - test_split
        else:
            test_split = 1.0 - train_split

        #
        # To avoid accidentally creating relationships between test data
        # and training data, we're going to split on texts, not chunks.
        # As a result, the requested split ratio is somewhat of a
        # best-effort attempt.
        #
        # We go through the texts greedily in random order.  If more than half
        # of the text's tokens fit into the space allocated for training, it
        # becomes a training text. Otherwise, it's a test text.
        #

        order = self.rng.permutation(len(self.texts))
        train_target = int(round((self.token_count * train_split)))
        train_tokens = 0
        test_tokens = 0
        total_tokens = 0
        train = []
        test = []
        for ii in order:
            text = self.texts[ii]
            # noinspection PyTypeChecker
            text_tokens = len(text['input_ids'])
            total_tokens += text_tokens
            train_max = int(train_target + (text_tokens / 2))
            if train_tokens + text_tokens <= train_max:
                train.append(text)
                train_tokens += text_tokens
            else:
                test.append(text)
                test_tokens += text_tokens

        print(f"Split train/test target: {train_split*100:.1f}/{test_split*100:.1f},",
              f"{train_target} / {self.token_count -train_target}")
        print(f"Split train/test actual: {train_tokens/total_tokens*100:.1f}/{test_tokens/total_tokens*100:.1f},",
              f"{train_tokens} / {test_tokens}")

        return train, test
