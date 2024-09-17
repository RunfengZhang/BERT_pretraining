import numpy as np
import h5py


class H5Writer(object):
    def __init__(self, path, mode='a', chunk_size=4096):
        self.path = path
        self.mode = mode
        self.chunk_size = chunk_size

    def __enter__(self):
        self.hf = h5py.File(self.path, self.mode)
        return self

    def add_batch(self, data_dict):
        for key, value in data_dict.items():
            value = np.array(value)
            is_str=False
            if value.dtype.type is np.unicode_:
                value = np.array([sent.encode('utf-8') for sent in value])
                is_str = True

            if key in self.hf:
                dataset = self.hf[key]
                batch_size = len(value)
                dataset.resize(dataset.shape[0] + batch_size, axis=0)
                dataset[-batch_size:] = value
            else:
                value = np.array(value)
                shape = (None,) + value.shape[1:]
                dtype = value.dtype if not is_str else h5py.special_dtype(vlen=str)
                chunks = (self.chunk_size,) + value.shape[1:]
                self.hf.create_dataset(key, data=value, chunks=chunks, maxshape=shape, dtype=dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        self.hf.close()