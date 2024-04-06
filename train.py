from transformers import AutoConfig, AutoTokenizer, GPT2Model

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_config = AutoConfig.from_pretrained("gpt2")

model = GPT2Model(model_config)