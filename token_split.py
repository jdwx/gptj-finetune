import numpy as np
import pickle
import sys


from transformers import AutoTokenizer
from lib import LeftPadChunker, TextDataset


class TextChunker:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.input_chunker = LeftPadChunker(tokenizer.model_max_length, pad_with=tokenizer.eos_token_id)
        self.attention_chunker = LeftPadChunker(tokenizer.model_max_length, pad_with=0)

    def __call__(self, texts):
        for text in texts:
            input_chunks = self.input_chunker(text['input_ids'])
            attention_chunks = self.attention_chunker(text['attention_mask'])
            for input_chunk, attention_chunk in zip(input_chunks, attention_chunks):
                yield {
                    'input_ids': input_chunk,
                    'attention_mask': attention_chunk,
                }


def main(paths: list[str]) -> None:

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    dataset = TextDataset(tokenizer)
    for path in paths:
        dataset.add_recursive(path)
    if len(dataset) == 0:
        print("I've got nothing here.")
        return
    train_texts, test_texts = dataset.split_train_and_test()

    chunker = TextChunker(tokenizer)
    train_set = list(chunker(train_texts))
    test_set = list(chunker(test_texts))

    print("Chunked:", len(train_set), "training chunks", len(test_set), "test chunks")
    print("Training chunk sizes:", np.unique([len(x['input_ids']) for x in train_set]))
    print("Test chunk sizes:", np.unique([len(x['input_ids']) for x in test_set]))

    with open("train_set.pkl", "wb") as f:
        for chunk in train_set:
            pickle.dump(chunk, f)

    with open("test_set.pkl", "wb") as f:
        for chunk in test_set:
            pickle.dump(chunk, f)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main(["text/"])
