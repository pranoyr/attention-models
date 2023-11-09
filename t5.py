import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from typing import List

transformers.logging.set_verbosity_error()

def exists(val):
	return val is not None

# config
MAX_LENGTH = 77
DEFAULT_T5_NAME = 'google/t5-v1_1-base'
T5_CONFIGS = {}

# singleton globals

def get_tokenizer(name):
	tokenizer = T5Tokenizer.from_pretrained(name)
	return tokenizer

def get_model(name):
	print(name)
	model = T5EncoderModel.from_pretrained(name)
	return model

def get_model_and_tokenizer(name):
	global T5_CONFIGS

	if name not in T5_CONFIGS:
		T5_CONFIGS[name] = dict()
	if "model" not in T5_CONFIGS[name]:
		T5_CONFIGS[name]["model"] = get_model(name)
	if "tokenizer" not in T5_CONFIGS[name]:
		T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)


	return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim(name):
	if name not in T5_CONFIGS:
		# avoids loading the model if we only want to get the dim
		config = T5Config.from_pretrained(name)
		T5_CONFIGS[name] = dict(config=config)
	elif "config" in T5_CONFIGS[name]:
		config = T5_CONFIGS[name]["config"]
	elif "model" in T5_CONFIGS[name]:
		config = T5_CONFIGS[name]["model"].config
	else:
		assert False
	return config.d_model

# encoding text

class TextEncoder(torch.nn.Module):
	def __init__(self, name = DEFAULT_T5_NAME):
		super().__init__()
		self.name = name
	   
		self.t5, self.tokenizer = get_model_and_tokenizer(name)
		self.t5.eval()
	
	def forward(self, texts: List[str], device = None):
	 
		if torch.cuda.is_available():
		    device = "cuda"

		encoded = self.tokenizer.batch_encode_plus(
			texts,
			return_tensors = "pt",
			padding = 'max_length',
			max_length = MAX_LENGTH,
			truncation = True
		)

		input_ids = encoded.input_ids.to(device)
		attn_mask = encoded.attention_mask.to(device)


		with torch.no_grad():
			output = self.t5(input_ids = input_ids, attention_mask = attn_mask)
			encoded_text = output.last_hidden_state.detach()

		attn_mask = attn_mask.bool()

		# if not exists(device):
		# 	return encoded_text, attn_mask

		encoded_text.to(device)
		attn_mask.to(device)

		return encoded_text

if __name__== '__main__':
    text = ["hello world", "hi there", "how are you doing?"]
    T5_Encoder = TextEncoder().cuda()
    encoded_text = T5_Encoder(text)
    print(encoded_text.shape)
    # print(attn_mask.shape)



