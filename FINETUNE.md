### Finetune

Please install [xtuner](https://github.com/InternLM/xtuner). Here is an example of finetuning Harmon.

```shell
cd /path/to/Harmon
export PYTHONPATH=./:$PYTHONPATH
export LAUNCHER="torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "

export CMD="scripts/train.py \
configs/examples/qwen2_5_1_5b_kl16_mar_h_train_example.py \
--launcher pytorch \
--deepspeed deepspeed_zero2"

echo $LAUNCHER
echo $CMD

bash -c "$LAUNCHER $CMD"

sleep 60s

```
The data should be formatted as:

```
data
├── YOUR_DATASET
    ├── data_info.json
    ├── local_folder
        ├── 000000
            ├── 0000001.jpg
    ├── cap_folder
        ├── 000000
            ├── 0000001.json
```


The `data/YOUR_DATASET/cap_folder/000000/0000001.json` looks as:
```
{'caption': 'xxxxxxxx'}
```

The `data/YOUR_DATASET/data_info.json` looks as:
```
[{'image': '000000/0000001.jpg', 'annotation': '000000/0000001.json'},
{'image': '000000/0000002.jpg', 'annotation': '000000/0000002.json'},
]
```


To instantiate an image caption dataset:

```
from src.datasets.understanding.caption_datasets import CaptionDataset
dataset = CaptionDataset(
               data_path='data/YOUR_DATASET/data_info.json',
               local_folder='data/YOUR_DATASET/local_folder',
               cap_folder='data/YOUR_DATASET/cap_folder',
               image_size=image_size,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               template_map_fn=dict(
                   type=template_map_fn_factory, template=prompt_template),
               max_length=max_length,
               image_length=image_length,)

```

To instantiate a text-to-image dataset:
```
from src.datasets.text2image.text2image import LargeText2ImageDataset
dataset = LargeText2ImageDataset(
               data_path='data/YOUR_DATASET/data_info.json',
               local_folder='data/YOUR_DATASET/local_folder',
               cap_folder='data/YOUR_DATASET/cap_folder',
               unconditional=0.1,
               prompt_template=prompt_template,
               image_processor=image_processor,
               ceph_folder=None,
               ceph_config=None,
               tokenizer=tokenizer,
               max_length=max_length)

```
