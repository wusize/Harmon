from src.datasets.understanding.caption_datasets import CaptionDataset
from mmengine.config import read_base
from src.datasets.collate_functions import collate_func_und, CollateConcat
# from mmengine.dataset import DefaultSampler
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler

from xtuner.dataset.map_fns import template_map_fn_factory


with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index, image_length


max_length = 512


dataset = dict(type=CaptionDataset,
               data_path='data/cc3m/cc3m_densecaps.json',
               local_folder='data/cc3m/raw',
               image_size=image_size,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               template_map_fn=dict(
                   type=template_map_fn_factory, template=prompt_template),
               max_length=max_length,
               image_length=image_length,)


group_keys = ['image2text',]
repeat = [1]
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
                                 # dict(type=collate_func_gen,
                                 #      pad_index=pad_index),
                                 ],
                    keys=group_keys
                    )
)
