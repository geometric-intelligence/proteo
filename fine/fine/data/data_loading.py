import os

import torch
from six.moves import collections_abc as container_abcs
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data

string_classes = (str, bytes)
int_classes = (int,)
from data.complex import Cochain, CochainBatch, Complex, ComplexBatch
from data.datasets import ComplexDataset, ZincDataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Collater(object):
    """Object that converts python lists of objects into the appropiate storage format.

    Args:
        follow_batch: Creates assignment batch vectors for each key in the list.
        max_dim: The maximum dimension of the cochains considered from the supplied list.
    """

    def __init__(self, follow_batch, max_dim=2):
        self.follow_batch = follow_batch
        self.max_dim = max_dim

    def collate(self, batch):
        """Converts a data list in the right storage format."""
        elem = batch[0]
        if isinstance(elem, Cochain):
            return CochainBatch.from_cochain_list(batch, self.follow_batch)
        elif isinstance(elem, Complex):
            return ComplexBatch.from_complex_list(batch, self.follow_batch, max_dim=self.max_dim)
        elif isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError("DataLoader found invalid type: {}".format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges cochain complexes into to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
        max_dim (int): The maximum dimension of the chains to be used in the batch.
            (default: 2)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], max_dim=2, **kwargs):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch

        super(DataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=Collater(follow_batch, max_dim), **kwargs
        )


def load_dataset(
    name,
    root=os.path.join(ROOT_DIR, "datasets"),
    max_dim=2,
    fold=0,
    init_method="sum",
    n_jobs=2,
    **kwargs,
) -> ComplexDataset:
    """Returns a ComplexDataset with the specified name. and initialised with the given params."""

    if name == "ZINC":
        dataset = ZincDataset(
            os.path.join(root, name),
            max_ring_size=kwargs["max_ring_size"],
            include_down_adj=kwargs["include_down_adj"],
            use_edge_features=kwargs["use_edge_features"],
            n_jobs=n_jobs,
        )
    else:
        raise NotImplementedError(name)

    return dataset
