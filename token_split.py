# This program takes a list of one or more directories on the command line.
# For each directory, it collects all the text files (those with .txt
# extension) and tokenizes them. It then randomly splits the texts into
# training and test sets.
#
# The output is two pickle files in the current directory:
# train_texts.pkl - Training texts in random order
# test_texts.pkl - Test texts in random order
#
# Each file contains a sequence of tokenized texts. Each tokenized text is
# represented as a list of integers representing the text's token IDs.
#
# This program *does not* chunk the texts into fixed-size pieces.  To do that,
# see the chunking.py program.  To use them for other purposes, see the
# get_data_from_pickle() function in lib/pickler.py.

import pickle
import sys


from transformers import AutoTokenizer
from lib import TextDataset


def main(paths: list[str]) -> None:

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    dataset = TextDataset(tokenizer)
    for path in paths:
        dataset.add_recursive(path)
    if len(dataset) == 0:
        print("I've got nothing here.")
        return

    # The default split is 80% train, 20% test. If this is not suitable, you can
    # change it here by passing train_split or test_split as floating point values
    # between 0 and 1. (E.g., train_split=0.9 would give 90% train, 10% test.)
    train_texts, test_texts = dataset.split_train_and_test()

    with open("train_texts.pkl", "wb") as f:
        for text in train_texts:
            pickle.dump(text, f)

    with open("test_texts.pkl", "wb") as f:
        for text in test_texts:
            pickle.dump(text, f)

    print("Processed:", len(train_texts), "training texts",
          len(test_texts), "test texts")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main(["text/"])
