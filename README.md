### Checkpoints
Obtain checkpoints from [wusize/harmon](https://huggingface.co/wusize/harmon), and arrange the model checkpoints as
```text
Harmon/
├── checkpoints
    ├── kl16.ckpt
    ├── harmon_0.5b.pth
    ├── harmon_1.5b.pth
```
It is recommended to use the following command to download the checkpoints
```bash
huggingface-cli download wusize/harmon  --local-dir checkpoints --repo-type model
```


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
python src/models/mar_llms/text2image.py configs/models/qwen2_5_1_5b_kl16_mar_h.py \
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