__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import mmap
from typing import Iterator, Optional

import numpy as np

from . import BaseKVIndexer
from ..compound import CompoundExecutor


class BinaryPbIndexer(BaseKVIndexer):
    class WriteHandler:
        def __init__(self, path, mode):
            self.body = open(path, mode)
            self.header = open(path + '.head', mode)

        def close(self):
            self.body.close()
            self.header.close()

        def flush(self):
            self.body.flush()
            self.header.flush()

    class ReadHandler:
        def __init__(self, path):
            with open(path + '.head', 'rb') as fp:
                tmp = np.frombuffer(fp.read(), dtype=np.int64).reshape([-1, 4])
                self.header = {r[0]: r[1:] for r in tmp}
            self._body = open(path, 'r+b')
            self.body = self._body.fileno()

        def close(self):
            self._body.close()

    def get_add_handler(self):
        # keep _start position as in pickle serialization
        handler = self.WriteHandler(self.index_abspath, 'ab')
        return handler

    def get_create_handler(self):
        self._start = 0  # override _start position
        return self.WriteHandler(self.index_abspath, 'wb')

    def get_query_handler(self):
        return self.ReadHandler(self.index_abspath)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._total_byte_len = 0
        self._start = 0
        self._page_size = mmap.ALLOCATIONGRANULARITY

    def add(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        for key, value in zip(keys, values):
            l = len(value)  #: the length
            p = int(self._start / self._page_size) * self._page_size  #: offset of the page
            r = self._start % self._page_size  #: the remainder, i.e. the start position given the offset
            self.write_handler.header.write(
                np.array(
                    (key, p, r, r + l),
                    dtype=np.int64
                ).tobytes()
            )
            self._start += l
            self.write_handler.body.write(value)
            self._size += 1
            # print(f'l: {l} p: {p} r: {r} r+l: {r + l} size: {self._size}')

    def query(self, key: int) -> Optional[bytes]:
        # FIXME this shouldn't be required
        if self.query_handler is None:
            self.query_handler = self.get_query_handler()
        pos_info = self.query_handler.header.get(key, None)
        if pos_info is not None:
            p, r, l = pos_info
            with mmap.mmap(self.query_handler.body, offset=p, length=l) as m:
                return m[r:]

    def update(self, keys: Iterator[int], values: Iterator[bytes], *args, **kwargs):
        new_values = []
        new_keys = []
        for key, new_value in zip(keys, values):
            pos_info = self.query_handler.header.get(key, None)
            if pos_info is not None:
                p, r, l = pos_info
                with mmap.mmap(self.query_handler.body, offset=p, length=l) as m:
                    old_value = m[r:]
                    if old_value != new_value:
                        new_keys.append(key)
                        new_values.append(new_value)

        if new_keys:
            adder = self.get_add_handler()
            _start = adder.body.tell()
            for key, new_value in zip(new_keys, new_values):
                l = len(new_value)  #: the length
                p = int(_start / self._page_size) * self._page_size  #: offset of the page
                r = _start % self._page_size  #: the remainder, i.e. the start position given the offset
                adder.header.write(
                    np.array(
                        (key, p, r, r + l),
                        dtype=np.int64
                    ).tobytes()
                )
                _start += l
                adder.body.write(new_value)

            # FIXME none of these fix the .save() error
            adder.flush()
            adder.close()
            del adder
            # FIXME this is required in order to avoid the warning 'no update...'. Maybe there's a better way?
            self.touch()

            # rebuild
            # TODO is this required?
            self.query_handler = self.get_query_handler()

    def delete(self, keys: Iterator[int], *args, **kwargs):
        raise NotImplementedError


class DataURIPbIndexer(BinaryPbIndexer):
    """Shortcut for :class:`DocPbIndexer` equipped with ``requests.on`` for storing doc-level protobuf and data uri info,
    differ with :class:`ChunkPbIndexer` only in ``requests.on`` """


class UniquePbIndexer(CompoundExecutor):
    """A frequently used pattern for combining a :class:`BaseKVIndexer` and a :class:`DocIDCache` """
