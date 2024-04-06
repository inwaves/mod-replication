import torch as t

from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
from torch.datasets import DataLoader
from utils import model_stats

def load_models():
    model_aliases = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

    tokenisers = {}
    models = {}
    for alias in model_aliases:
        tokenizer = AutoTokenizer.from_pretrained(alias)
        model_config = AutoConfig.from_pretrained(alias)

        language_model = GPT2LMHeadModel(model_config)

        tokenisers[alias] = tokenizer
        models[alias] = language_model

        num_params, total_size = model_stats(language_model)
        print(f"Initialised model: {alias}, Number of parameters: {num_params/1e6}M, Total size: {total_size/1e6:.2f} MB")

def preprocess_data():
    pass

def train(model):
    dataset = preprocess_data() 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


def main():
    load_models()
    
if __name__ == "__main__":
    main()