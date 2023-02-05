# This program takes pickle files containing train and test texts and
# chunks them into 2048-token chunks. It then saves the chunks in new
# pickle files. Each chunk pickle file contains a sequence of
# chunks of text where each chunk is a list of token IDs. Texts are
# left-padded to the tokenizer's max length with EOS tokens so files
# do not overlap. Left padding is used since for language modelling it
# is more likely to have a short input at the beginning rather than the
# end.

import pickle
import sys


from transformers import AutoTokenizer
from lib import TextChunker, get_data_from_pickle


def main(in_path: str, out_path: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    chunker = TextChunker(tokenizer)
    text_count = 0
    chunk_count = 0
    with open(out_path, "wb") as out_file:
        for text in get_data_from_pickle(in_path):
            text_count += 1
            for chunk in chunker(text):
                pickle.dump(chunk, out_file)
                chunk_count += 1

    print(f"Chunked {text_count} texts in {in_path} into {chunk_count} chunks in {out_path}.")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main("train_texts.pkl", "train_chunks.pkl")
        main("test_texts.pkl", "test_chunks.pkl")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2])
        main(sys.argv[3], sys.argv[4])
    else:
        print(f"Usage: {sys.argv[0]} [texts_in_path chunks_out_path [texts_in_path chunks_out_path]]")
