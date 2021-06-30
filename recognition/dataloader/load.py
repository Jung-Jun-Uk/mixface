from .kface import KFace
from .face import Face
from .merge import Merge

def load_datasets(dataset_name, config, batch_size, test_batch_size, cuda, workers, rank):
    if dataset_name == 'kface':
        dataset = KFace(config, batch_size, test_batch_size, cuda, workers, rank)
    elif dataset_name == 'face':
        dataset = Face(config, batch_size, test_batch_size, cuda, workers, rank)
    elif dataset_name == 'merge':
        dataset = Merge(config, batch_size, test_batch_size, cuda, workers, rank)
    return dataset
