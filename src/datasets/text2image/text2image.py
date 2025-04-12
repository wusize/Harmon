from torch.utils.data import Dataset
from PIL import Image
import os
import json
import random
import torch
import numpy as np
from einops import rearrange
from xtuner.registry import BUILDER
from src.datasets.utils import crop2square
from glob import glob


class Text2ImageDataset(Dataset):
    def __init__(self,
                 data_path,
                 local_folder,
                 image_size,
                 unconditional=0.1,
                 tokenizer=None,
                 prompt_template=None,
                 max_length=1024,
                 crop_image=True,
                 cap_source='caption',
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.unconditional = unconditional
        self.local_folder = local_folder
        self.cap_source = cap_source

        self.image_size = image_size

        self.tokenizer = BUILDER.build(tokenizer)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.crop_image = crop_image

    def _load_data(self, data_path):
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        image = Image.open(os.path.join(self.local_folder, image_file))
        assert image.width > 8 and image.height > 8, f"Image: {image.size}"
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image

    def _process_text(self, text):
        if random.uniform(0, 1) < self.unconditional:
            prompt = "Generate an image."
        else:
            prompt = f"Generate an image: {text.strip()}"
        prompt = self.prompt_template['INSTRUCTION'].format(input=prompt)
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')[0]

        return dict(input_ids=input_ids[:self.max_length])

    def _process_image(self, image):
        data = dict()

        if self.crop_image:
            image = crop2square(image)
        else:
            target_size = max(image.size)
            image = image.resize(size=(target_size, target_size))

        image = image.resize(size=(self.image_size, self.image_size))
        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')

        data.update(pixel_values=pixel_values)

        return data

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')

            caption = data_sample[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class LargeText2ImageDataset(Text2ImageDataset):
    # self.data_list only contains paths of images and captions

    def __init__(self, cap_folder=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_folder = self.local_folder if cap_folder is None else cap_folder

    def _load_data(self, data_path):      # image path and annotation path are saved in a json file
        if data_path.endswith(".json"):
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        else:
            self.data_list = []
            json_files = glob(f'{data_path}/*.json')
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    self.data_list += json.load(f)

        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            image = self._read_image(data_sample['image']).convert('RGB')
            with open(f"{self.cap_folder}/{data_sample['annotation']}", 'r') as f:
                caption = json.load(f)[self.cap_source]
            data = self._process_image(image)
            data.update(self._process_text(caption))
            data.update(type='text2image')
            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{data_sample}: {e}", flush=True)
            return self._retry()
