# from https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py
import copy
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import math
import random
import numpy as np
from PIL import Image
from xtuner.registry import BUILDER
from src.datasets.imagenet.classes_v2 import CLASS_NAMES
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class CustomImageNet(Dataset):
    def __init__(self, data_path, image_dir, is_train, image_size,
                 debug=False,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                 tokenizer=None,
                 prompt_template=None,
                 unconditional=0.1,
                 data_type='text2image'
                 ):
        super().__init__()
        self.image_dir = image_dir
        self.data_path = data_path
        self.is_train = is_train
        self.image_size = image_size
        self.debug = debug
        if is_train:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std, inplace=True)
            ])

        with open(data_path, 'r') as f:
            self.data_list = f.readlines()

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.unconditional = unconditional
        self.class_names = copy.deepcopy(CLASS_NAMES)
        self.data_type = data_type

    def __len__(self):
        return len(self.data_list)

    def _get_raw_data(self, idx):
        if self.debug:
            idx = idx % 100
        # if idx == -1:
        #     return {}
        data = self.data_list[idx].strip()
        filename, class_id = data.split()

        data_dict = dict(class_id=int(class_id),
                         image_dir=self.image_dir, image_file=filename)

        image = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
        image = self.transform(image)
        data_dict['pixel_values'] = image

        return data_dict

    def _process_text(self, class_id):
        text = self.class_names[class_id]
        text = random.choice(text.split(',')).strip()
        if random.uniform(0, 1) < self.unconditional:
            class_id = -1
            prompt = "Generate an image."
        else:
            prompt = f"Generate an image: {text.strip()}."

        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')[0]

        return dict(input_ids=input_ids, class_id=class_id, type='text2image')

    def __getitem__(self, idx):
        data = self._get_raw_data(idx)
        class_id = data.get('class_id')
        # class_name = self.class_names[class_id]
        data.update(self._process_text(class_id))
        data['type'] = self.data_type

        return data
