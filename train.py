import argparse
import logging
import numpy as np
import time
import mlflow.pytorch
import torch as t
import random

from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchinfo import summary
from gpt_mod import GPT2LMHeadModel_MixtureOfDepths
from utils import model_stats

MODEL_ALIASES = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]

logger = logging.getLogger("__name__")

seed = 42
t.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# TODO: merge this & cl args
@dataclass
class Parameters:
    batch_size: int = 32
    batch_size_dataset_loader: int = 32
    epochs: int = 10
    experiment_name: str = "/Users/inwaves@live.com/{0}"
    mod_capacity_budget: float = 0.125

def preprocess_data(batch, tokeniser):
    texts = [item['text'] for item in batch]  # Extracting text data from the batch

    # Tokenize the text data
    batch_encoding = tokeniser(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt")

    # You no longer need to manually pad or convert lists to tensors since the tokeniser does this for you
    return {
        "input_ids": batch_encoding['input_ids'],
        "attention_mask": batch_encoding['attention_mask'],
        "labels": batch_encoding['input_ids'],  # Assuming you want to use the input IDs as labels for some sort of language modeling
    }


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model name, one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl.", default="gpt2")
    parser.add_argument("-c", "--capacity", type=float, help="Model capacity as fraction of total; float in [0, 1].")
    parser.add_argument("-b", "--batch_size", type=int, help="Training batch size.")
    args = parser.parse_args()

    if is_mod:
        config = GPT2Config(capacity=args.capacity)
        model = GPT2LMHeadModel_MixtureOfDepths()
    else:
        config = GPT2Config()
        model = GPT2LMHeadModel(config)

    num_params, total_size = model_stats(model)
    logger.info(f"Initialised model: {model_alias}, Number of parameters: {num_params/1e6}M, Total size: {total_size/1e6:.2f} MB")
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    tokeniser.pad_token = tokeniser.eos_token

    optimizer = t.optim.AdamW(model.parameters(), lr=5e-5)
    iterable_dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")

    dataloader = DataLoader(iterable_dataset, batch_size=args.batch_size)

    return model, tokeniser, optimizer, dataloader


def train_loop(model, tokeniser, optimizer, dataloader, selected_model):
    mlflow.set_experiment(Parameters.experiment_name.format(selected_model))

    start_time = time.time()
    for epoch in range(Parameters.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            batch = preprocess_data(batch, tokeniser)

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
                logger.info(f"Epoch: {epoch}/{Parameters.epochs}, Batch: {step}, {duration}, Loss: {loss.item()}")

def log_parameters_and_artifacts(model):
    mlflow.log_params(Parameters.__dict__)
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

def train(model_name, is_mod):
    log_parameters_and_artifacts(model)

    mlflow.end_run()
    with mlflow.start_run() as run:
        train_loop(model, tokeniser, optimizer, dataloader, model_name)
    mlflow.pytorch.log_model(model, "model")


def main():
    model, tokeniser, optimizer, dataloader = setup()
    train(model)
    
if __name__ == "__main__":
    main()