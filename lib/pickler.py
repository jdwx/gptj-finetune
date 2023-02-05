import pickle


def get_data_from_pickle(file_path: str):
    with open(file_path, "rb") as f:
        try:
            while True:
                chunk = pickle.load(f)
                chunk['labels'] = chunk['input_ids']
                yield chunk
        except EOFError:
            pass
