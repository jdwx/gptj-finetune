import pickle


def get_data_from_pickle(file_path: str):
    with open(file_path, "rb") as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError:
            pass


def get_data_from_pickle_with_labels(file_path: str):
    for chunk in get_data_from_pickle(file_path):
        chunk['labels'] = chunk['input_ids']
        yield chunk
