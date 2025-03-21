from src.datasets.text2image.text2image import LargeText2ImageDataset
from mmengine.config import read_base
from src.datasets.collate_functions import collate_func_gen, CollateConcat
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler

with read_base():
    from .processors import prompt_template, tokenizer, image_processor, pad_index


max_length = 128


dataset = dict(type=LargeText2ImageDataset,
               cap_folder='data/laion8m/laion6m_shortcaps',
               data_path='data/laion8m/laion6m_shortcaps.json',
               local_folder='data/laion8m/images',
               unconditional=0.1,
               prompt_template=prompt_template,
               image_processor=image_processor,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)


group_keys = ['text2image']
repeat = [1,]
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
                    collate_fns=[
                                 # dict(type=collate_func_und,
                                 #      pad_index=pad_index),
                                 dict(type=collate_func_gen,
                                      pad_index=pad_index),
                                 ],
                    keys=group_keys
                    )
)
