### Checkpoints
Obtain checkpoints from [wusize/harmon](https://huggingface.co/wusize/harmon), and arrange the model checkpoints as
```text
Harmon/
├── checkpoints
    ├── kl16.ckpt
    ├── harmon_0.5b.pth
    ├── harmon_1.5b.pth
    ├── harmon_1.5b-o.pth  # Fine-tuned model on BLIP3o-60k
```
It is recommended to use the following command to download the checkpoints
```bash
huggingface-cli download wusize/harmon  --local-dir checkpoints --repo-type model
```

### 🔄 Update
We fine-tuned Harmon-1.5B using [BLIP3o-60k](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k) dataset. During fine-tuning, we only updated the parameters of the MAR decoder. The fine-tuned model achieves **0.85** on GenEval. The model checkpoint is available at [harmon_1.5b-o.pth](https://huggingface.co/wusize/harmon/blob/main/harmon_1.5b-o.pth).


### Image-to-text 


```shell
export PYTHONPATH=./:$PYTHONPATH
python scripts/image2text.py configs/models/qwen2_5_1_5b_kl16_mar_h.py \
         --checkpoint checkpoints/harmon_1.5b.pth  --image_size 512 \
         --image data/view.jpg --prompt "Describe the image in detail."
```

### Text-to-Image

```shell
export PYTHONPATH=./:$PYTHONPATH
python scripts/text2image.py configs/models/qwen2_5_1_5b_kl16_mar_h.py \
         --checkpoint checkpoints/harmon_1.5b.pth  --image_size 512 \
         --prompt 'a dog on the left and a cat on the right.'  --output output.jpg
```

**To generate a list of images based on prompts in a json file.**
```shell
export PYTHONPATH=./:$PYTHONPATH
accelerate launch scripts/batch_text2image.py configs/models/qwen2_5_1_5b_kl16_mar_h.py \
       --checkpoint checkpoints/harmon_1.5b.pth  --image_size 512 \
       --data path/to/xxx.json --output output
```
The json file should look like:

```json
[
  {
   "prompt": "a dog on the left and a cat on the right."
  }
]
```


### train

```shell
cd /path/to/Harmon
source activate /your/env
export NCCL_DEBUG=DEBUG
export MKL_SERVICE_FORCE_INTEL=1


MASTER_ADDR=$MLP_WORKER_0_HOST
echo $MASTER_ADDR

MASTER_PORT=$MLP_WORKER_0_PORT
NNODES=8
GPUS_PER_NODE=8
export NCCL_DEBUG=INFO

NODE_RANK=$(grep -oP 'worker-\K\d+' /etc/hosts | head -n 1)
echo $NODE_RANK

export PYTHONPATH=./:$PYTHONPATH
export LAUNCHER="torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    "

export CMD="scripts/train.py \
configs/train_example/qwen2_5_0_5b_kl16_mar_b_train_example.py \
--launcher pytorch \
--deepspeed deepspeed_zero2"

echo $LAUNCHER
echo $CMD

bash -c "$LAUNCHER $CMD"

sleep 60s

```
