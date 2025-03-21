from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoTokenizer, SiglipImageProcessor


llm_name_or_path = 'Qwen/Qwen2.5-0.5B-Instruct'
visual_encoder_name_or_path = 'google/siglip-base-patch16-256'
prompt_template = PROMPT_TEMPLATE.qwen_chat
pad_index = 151645
image_length = 1024 + 64
image_size = 512

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=SiglipImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
    size={"height": image_size, "width": image_size}
)
