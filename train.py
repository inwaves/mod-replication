import numpy as np
import time
import mlflow.pytorch
import torch as t
import random

from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchinfo import summary
from gpt_mod import GPT2LMHeadModel_MixtureOfDepths
from utils import model_stats


seed = 42
t.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


@dataclass
class Parameters:
    batch_size: int = 32
    batch_size_dataset_loader: int = 32
    epochs: int = 10
    experiment_name: str = "/Users/inwaves@live.com/{0}"
    mod_capacity_budget: float = 0.125


def load_model(model_alias, is_mod):
    tokenizer = AutoTokenizer.from_pretrained(model_alias)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel_MixtureOfDepths() if is_mod else GPT2LMHeadModel()

    optimizer = t.optim.AdamW(model.parameters(), lr=5e-5)
    num_params, total_size = model_stats(model)
    print(f"Initialised model: {model_alias}, Number of parameters: {num_params/1e6}M, Total size: {total_size/1e6:.2f} MB")

    return model, tokenizer, optimizer

def collate_fn(batch, tokenizer):
    texts = [item['text'] for item in batch]  # Extracting text data from the batch

    # Tokenize the text data
    batch_encoding = tokenizer(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")

    # You no longer need to manually pad or convert lists to tensors since the tokenizer does this for you
    return {
        "input_ids": batch_encoding['input_ids'],
        "attention_mask": batch_encoding['attention_mask'],
        "labels": batch_encoding['input_ids'],  # Assuming you want to use the input IDs as labels for some sort of language modeling
    }
def preprocess_data():
    iterable_dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
    dataloader = DataLoader(iterable_dataset, batch_size=Parameters.batch_size_dataset_loader, collate_fn=collate_fn)
    return dataloader


def train_loop(model, tokenizer, optimizer, dataloader, selected_model):
    mlflow.set_experiment(Parameters.experiment_name.format(selected_model))

    start_time = time.time()
    for epoch in range(Parameters.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            # inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True, max_length=1024)

            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels = batch["input_ids"].to("cuda")

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            mlflow.log_metric("loss", loss.item())
            mlflow.log_metric("perplexity", t.exp(loss).item())

            if step % 100 == 0:
                mlflow.log_metric("loss", loss.item(), step=step)
                duration = time.time() - start_time
                mlflow.log_metric("duration", duration, step=step)
                print(f"Epoch: {epoch}/{Parameters.epochs}, Batch: {step}, {duration}, Loss: {loss.item()}")

def log_parameters_and_artifacts(model):
    mlflow.log_params(Parameters.__dict__)
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

def train(selected_model, is_mod):

    model, tokenizer, optimizer = load_model(selected_model, is_mod)
    dataloader = preprocess_data()
    log_parameters_and_artifacts(model)

    mlflow.end_run()
    with mlflow.start_run() as run:
        train_loop(model, tokenizer, optimizer, dataloader, selected_model)
    mlflow.pytorch.log_model(model, "model")


def main():
    model_aliases = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    selected_model = model_aliases[0]
    is_mod = False
    train(selected_model, is_mod)
    
if __name__ == "__main__":
    main()