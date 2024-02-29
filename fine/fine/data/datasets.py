"""
The code is based on https://github.com/rusty1s/pytorch_geometric/blob/76d61eaa9fc8702aa25f29dfaa5134a169d0f1f6/torch_geometric/data/dataset.py#L19
and https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/in_memory_dataset.py

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import copy
import os.path as osp
import re
from abc import ABC
from itertools import product, repeat
from typing import Tuple

import torch
from data.complex import Cochain, Complex
from data.utils import convert_graph_dataset_with_rings
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.datasets import ZINC


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())


class ComplexDataset(Dataset, ABC):
    """Base class for cochain complex datasets.

    This class mirrors
    https://github.com/rusty1s/pytorch_geometric/blob/76d61eaa9fc8702aa25f29dfaa5134a169d0f1f6/torch_geometric/data/dataset.py#L19
    """

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_dim: int = None,
        num_classes: int = None,
        init_method: str = 'sum',
        cellular: bool = False,
    ):
        # These have to be initialised before calling the super class.
        self._max_dim = max_dim
        self._num_features = [None for _ in range(max_dim + 1)]
        self._init_method = init_method
        self._cellular = cellular

        super(ComplexDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self._num_classes = num_classes
        self.train_ids = None
        self.val_ids = None
        self.test_ids = None

    @property
    def max_dim(self):
        return self._max_dim

    @max_dim.setter
    def max_dim(self, value):
        self._max_dim = value

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        prefix = "cell_" if self._cellular else ""
        return osp.join(self.root, f'{prefix}complex_dim{self.max_dim}_{self._init_method}')

    def num_features_in_dim(self, dim):
        if dim > self.max_dim:
            raise ValueError(
                '`dim` {} larger than max allowed dimension {}.'.format(dim, self.max_dim)
            )
        if self._num_features[dim] is None:
            self._look_up_num_features()
        return self._num_features[dim]

    def _look_up_num_features(self):
        for complex in self:
            for dim in range(complex.dimension + 1):
                if self._num_features[dim] is None:
                    self._num_features[dim] = complex.cochains[dim].num_features
                else:
                    assert self._num_features[dim] == complex.cochains[dim].num_features

    def get_idx_split(self):
        idx_split = {'train': self.train_ids, 'valid': self.val_ids, 'test': self.test_ids}
        return idx_split


class InMemoryComplexDataset(ComplexDataset):
    """Wrapper around ComplexDataset with functionality such as batching and storing the dataset.

    This class mirrors
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/in_memory_dataset.py
    """
    # NOOOOOO: You don't need "Complex"Dataset, just look into standard Dataset and InMemoryDataset classes from pytorch geometric
    @property
    def raw_file_names(self):
        r"""The name of the files to find in the :obj:`self.raw_dir` folder in
        order to skip the download."""
        raise NotImplementedError

    @property
    def processed_file_names(self):
        r"""The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing."""
        raise NotImplementedError

    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_dim: int = None,
        num_classes: int = None,
        include_down_adj=False,
        init_method=None,
        cellular: bool = False,
    ):
        self.include_down_adj = include_down_adj
        super(InMemoryComplexDataset, self).__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            max_dim,
            num_classes,
            init_method=init_method,
            cellular=cellular,
        )
        self.data, self.slices = None, None
        self.__data_list__ = None

    def len(self):
        for dim in range(self.max_dim + 1):
            for item in self.slices[dim].values():
                return len(item) - 1
        return 0

    def get(self, idx):
        if hasattr(self, '__data_list__'):
            if self.__data_list__ is None:
                self.__data_list__ = self.len() * [None]
            else:
                data = self.__data_list__[idx]
                if data is not None:
                    return copy.copy(data)

        retrieved = [self._get_cochain(dim, idx) for dim in range(0, self.max_dim + 1)]
        cochains = [r[0] for r in retrieved if not r[1]]

        targets = self.data['labels']
        start, end = idx, idx + 1
        if torch.is_tensor(targets):
            s = list(repeat(slice(None), targets.dim()))
            cat_dim = 0
            s[cat_dim] = slice(start, end)
        else:
            # TODO: come up with a better method to handle this
            assert targets[start] is None
            s = start

        target = targets[s]

        dim = self.data['dims'][idx].item()
        assert dim == len(cochains) - 1
        data = Complex(*cochains, y=target)

        if hasattr(self, '__data_list__'):
            self.__data_list__[idx] = copy.copy(data)

        return data

    def _get_cochain(self, dim, idx) -> Tuple[Cochain, bool]:
        if dim < 0 or dim > self.max_dim:
            raise ValueError(f'The current dataset does not have cochains at dimension {dim}.')

        cochain_data = self.data[dim]
        cochain_slices = self.slices[dim]
        data = Cochain(dim)
        if cochain_data.__num_cells__[idx] is not None:
            data.num_cells = cochain_data.__num_cells__[idx]
        if cochain_data.__num_cells_up__[idx] is not None:
            data.num_cells_up = cochain_data.__num_cells_up__[idx]
        if cochain_data.__num_cells_down__[idx] is not None:
            data.num_cells_down = cochain_data.__num_cells_down__[idx]
        elif dim == 0:
            data.num_cells_down = None

        for key in cochain_data.keys:
            item, slices = cochain_data[key], cochain_slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            data[key] = None
            if start != end:
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    cat_dim = cochain_data.__cat_dim__(key, item)
                    if cat_dim is None:
                        cat_dim = 0
                    s[cat_dim] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                data[key] = item[s]
        empty = data.num_cells is None

        return data, empty

    @staticmethod
    def collate(data_list, max_dim):
        r"""Collates a python list of data objects to the internal storage
        format of :class:`InMemoryComplexDataset`."""

        def init_keys(dim, keys):
            cochain = Cochain(dim)
            for key in keys[dim]:
                cochain[key] = []
            cochain.__num_cells__ = []
            cochain.__num_cells_up__ = []
            cochain.__num_cells_down__ = []
            slc = {key: [0] for key in keys[dim]}
            return cochain, slc

        def collect_keys(data_list, max_dim):
            keys = {dim: set() for dim in range(0, max_dim + 1)}
            for complex in data_list:
                for dim in keys:
                    if dim not in complex.cochains:
                        continue
                    cochain = complex.cochains[dim]
                    keys[dim] |= set(cochain.keys)
            return keys

        keys = collect_keys(data_list, max_dim)
        types = {}
        cat_dims = {}
        tensor_dims = {}
        data = {'labels': [], 'dims': []}
        slices = {}
        for dim in range(0, max_dim + 1):
            data[dim], slices[dim] = init_keys(dim, keys)

        for complex in data_list:
            # Collect cochain-wise items
            for dim in range(0, max_dim + 1):
                # Get cochain, if present
                cochain = None
                if dim in complex.cochains:
                    cochain = complex.cochains[dim]

                # Iterate on keys
                for key in keys[dim]:
                    if cochain is not None and hasattr(cochain, key) and cochain[key] is not None:
                        data[dim][key].append(cochain[key])
                        if isinstance(cochain[key], Tensor) and cochain[key].dim() > 0:
                            cat_dim = cochain.__cat_dim__(key, cochain[key])
                            cat_dim = 0 if cat_dim is None else cat_dim
                            s = slices[dim][key][-1] + cochain[key].size(cat_dim)
                            if key not in cat_dims:
                                cat_dims[key] = cat_dim
                            else:
                                assert cat_dim == cat_dims[key]
                            if key not in tensor_dims:
                                tensor_dims[key] = cochain[key].dim()
                            else:
                                assert cochain[key].dim() == tensor_dims[key]
                        else:
                            s = slices[dim][key][-1] + 1
                        if key not in types:
                            types[key] = type(cochain[key])
                        else:
                            assert type(cochain[key]) is types[key]
                    else:
                        s = slices[dim][key][-1] + 0
                    slices[dim][key].append(s)

                # Handle non-keys
                # TODO: could they be considered as keys as well?
                num = None
                num_up = None
                num_down = None
                if cochain is not None:
                    if hasattr(cochain, '__num_cells__'):
                        num = cochain.__num_cells__
                    if hasattr(cochain, '__num_cells_up__'):
                        num_up = cochain.__num_cells_up__
                    if hasattr(cochain, '__num_cells_down__'):
                        num_down = cochain.__num_cells_down__
                data[dim].__num_cells__.append(num)
                data[dim].__num_cells_up__.append(num_up)
                data[dim].__num_cells_down__.append(num_down)

            # Collect complex-wise label(s) and dims
            if not hasattr(complex, 'y'):
                complex.y = None
            if isinstance(complex.y, Tensor):
                assert complex.y.size(0) == 1
            data['labels'].append(complex.y)
            data['dims'].append(complex.dimension)

        # Pack lists into tensors

        # Cochains
        for dim in range(0, max_dim + 1):
            for key in keys[dim]:
                if types[key] is Tensor and len(data_list) > 1:
                    if tensor_dims[key] > 0:
                        cat_dim = cat_dims[key]
                        data[dim][key] = torch.cat(data[dim][key], dim=cat_dim)
                    else:
                        data[dim][key] = torch.stack(data[dim][key])
                elif types[key] is Tensor:  # Don't duplicate attributes...
                    data[dim][key] = data[dim][key][0]
                elif types[key] is int or types[key] is float:
                    data[dim][key] = torch.tensor(data[dim][key])

                slices[dim][key] = torch.tensor(slices[dim][key], dtype=torch.long)

        # Labels and dims
        item = data['labels'][0]
        if isinstance(item, Tensor) and len(data_list) > 1:
            if item.dim() > 0:
                cat_dim = 0
                data['labels'] = torch.cat(data['labels'], dim=cat_dim)
            else:
                data['labels'] = torch.stack(data['labels'])
        elif isinstance(item, Tensor):
            data['labels'] = data['labels'][0]
        elif isinstance(item, int) or isinstance(item, float):
            data['labels'] = torch.tensor(data['labels'])
        data['dims'] = torch.tensor(data['dims'])

        return data, slices

    def copy(self, idx=None):
        if idx is None:
            data_list = [self.get(i) for i in range(len(self))]
        else:
            data_list = [self.get(i) for i in idx]
        dataset = copy.copy(self)
        dataset.__indices__ = None
        dataset.__data_list__ = data_list
        dataset.data, dataset.slices = self.collate(data_list)

        return dataset

    def get_split(self, split):
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split {split}.')
        idx = self.get_idx_split()[split]
        if idx is None:
            raise AssertionError("No split information found.")
        # if self.__indices__ is not None:
        #     raise AssertionError("Cannot get the split for a subset of the original dataset.")
        return self[idx]


class ZincDataset(InMemoryComplexDataset):
    """This is ZINC from the Benchmarking GNNs paper. This is a graph regression task."""

    def __init__(
        self,
        root,
        max_ring_size,
        use_edge_features=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        subset=True,
        include_down_adj=True,
        n_jobs=2,
    ):
        self.name = 'ZINC'
        self._max_ring_size = max_ring_size
        self._use_edge_features = use_edge_features
        self._subset = subset
        self._n_jobs = n_jobs
        super(ZincDataset, self).__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            max_dim=2,
            cellular=True,
            include_down_adj=include_down_adj,
            num_classes=1,
        )

        self.data, self.slices, idx = self.load_dataset()
        self.train_ids = idx[0]
        self.val_ids = idx[1]
        self.test_ids = idx[2]

        self.num_node_type = 28
        self.num_edge_type = 4

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        name = self.name
        return [f'{name}_complex.pt', f'{name}_idx.pt']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        ZINC(self.raw_dir, subset=self._subset)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        print(f"Processing cell complex dataset for {self.name}")
        train_data = ZINC(self.raw_dir, subset=self._subset, split='train')
        val_data = ZINC(self.raw_dir, subset=self._subset, split='val')
        test_data = ZINC(self.raw_dir, subset=self._subset, split='test')

        data_list = []
        idx = []
        start = 0
        print("Converting the train dataset to a cell complex...")
        train_complexes, _, _ = convert_graph_dataset_with_rings(
            train_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=True,  # was false
            n_jobs=self._n_jobs,
        )
        data_list += train_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print("Converting the validation dataset to a cell complex...")
        val_complexes, _, _ = convert_graph_dataset_with_rings(
            val_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=True,  # was False
            n_jobs=self._n_jobs,
        )
        data_list += val_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print("Converting the test dataset to a cell complex...")
        test_complexes, _, _ = convert_graph_dataset_with_rings(
            test_data,
            max_ring_size=self._max_ring_size,
            include_down_adj=self.include_down_adj,
            init_edges=self._use_edge_features,
            init_rings=True,  # was False
            n_jobs=self._n_jobs,
        )
        data_list += test_complexes
        idx.append(list(range(start, len(data_list))))

        path = self.processed_paths[0]
        print(f'Saving processed dataset in {path}....')
        torch.save(self.collate(data_list, 2), path)

        path = self.processed_paths[1]
        print(f'Saving idx in {path}....')
        torch.save(idx, path)

    @property
    def processed_dir(self):
        """Overwrite to change name based on edges"""
        directory = super(ZincDataset, self).processed_dir
        suffix0 = "_full" if self._subset is False else ""
        suffix1 = f"_{self._max_ring_size}rings" if self._cellular else ""
        suffix2 = "-E" if self._use_edge_features else ""
        return directory + suffix0 + suffix1 + suffix2


def load_zinc_graph_dataset(root, subset=True):
    raw_dir = osp.join(root, 'ZINC', 'raw')

    train_data = ZINC(raw_dir, subset=subset, split='train')
    val_data = ZINC(raw_dir, subset=subset, split='val')
    test_data = ZINC(raw_dir, subset=subset, split='test')
    data = train_data + val_data + test_data

    if subset:
        assert len(train_data) == 10000
        assert len(val_data) == 1000
        assert len(test_data) == 1000
    else:
        assert len(train_data) == 220011
        assert len(val_data) == 24445
        assert len(test_data) == 5000

    idx = []
    start = 0
    idx.append(list(range(start, len(train_data))))
    start = len(train_data)
    idx.append(list(range(start, start + len(val_data))))
    start = len(train_data) + len(val_data)
    idx.append(list(range(start, start + len(test_data))))

    return data, idx[0], idx[1], idx[2]
