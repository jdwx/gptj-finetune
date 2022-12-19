from typing import Iterable


class BaseChunker:

    offset = 0

    def __init__(self, chunk_size: int,
                 chunk_from_right: bool = False,
                 pad_with=0):
        self.chunk_size = chunk_size
        self.chunk_from_right = chunk_from_right
        self.padding = [pad_with] * (self.chunk_size - 1)

    def _chunks(self, data: list):
        if self.chunk_from_right:
            offset = len(data) % self.chunk_size
            if offset > 0:
                yield data[0:offset]
        else:
            offset = 0
        while offset < len(data):
            chunk = data[offset:offset + self.chunk_size]
            offset += self.chunk_size
            yield chunk

    def __call__(self, data: list) -> Iterable:
        yield from self._chunks(data)


class TruncateChunker(BaseChunker):

    def __call__(self, data: list) -> Iterable:
        for chunk in self._chunks(data):
            if len(chunk) == self.chunk_size:
                yield chunk


class LeftPadChunker(BaseChunker):

    def __init__(self, chunk_size: int,
                 chunk_from_right: bool = True,
                 pad_with=0):
        super().__init__(chunk_size, chunk_from_right, pad_with)

    def __call__(self, data: list) -> Iterable:
        for chunk in self._chunks(data):
            yield (self.padding + chunk)[-self.chunk_size:]


class RightPadChunker(BaseChunker):

    def __call__(self, data: list) -> Iterable:
        for chunk in self._chunks(data):
            yield (chunk + self.padding)[:self.chunk_size]


class DictChunker:

    def __init__(self, chunker):
        self.chunker = chunker

    def __call__(self, data):
        work = dict()
        keys = []
        length = 0
        for key in data:
            work[key] = list(self.chunker(data[key]))
            if length == 0:
                length = len(work[key])
            keys.append(key)
        for ii in range(length):
            out = dict()
            for key in keys:
                out[key] = work[key][ii]
            yield out
