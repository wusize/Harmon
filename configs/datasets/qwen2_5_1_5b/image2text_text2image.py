from mmengine.config import read_base
from xtuner.dataset import ConcatDataset
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler
from src.datasets.collate_functions import (collate_func_gen,
                                            collate_func_und, CollateConcat)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


with read_base():
    from .image2text import dataset as und_data
    from .text2image import dataset as gen_data
    from .processors import *   #


dataset = dict(
    type=ConcatDataset,
    datasets=[und_data, gen_data]
)

group_keys = ['image2text', 'text2image']
repeat = [1, 4]
batch_size = 32
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=FixedBatchMultiSourceSampler,
                 repeat=repeat,
                 batch_size=batch_size,    # fixed batch size for all sources
                 shuffle=True),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=collate_func_und,
                                      pad_index=pad_index),
                                 dict(type=collate_func_gen,
                                      pad_index=pad_index),
                                 ],
                    keys=group_keys
                    )
)
