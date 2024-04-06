from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel

model_aliases = {
    "GPT2_SMALL": "gpt2", 
    "GPT2_MEDIUM": "gpt2-medium", 
    "GPT2_LARGE": "gpt2-large", 
    "GPT2_XL": "gpt2-xl"
}

tokenisers = []
models = []
for model, alias in model_aliases.items():
    tokenizer = AutoTokenizer.from_pretrained(alias)
    model_config = AutoConfig.from_pretrained(alias)

    language_model = GPT2LMHeadModel(model_config)

    tokenisers += [tokenizer]
    models += [language_model]

print(f"I initialised {len(tokenisers)} tokenisers and {len(models)} models...")
print(tokenisers)
print(models)