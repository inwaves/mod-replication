import argparse
import logging
import os
import random
import time

import mlflow.pytorch
import numpy as np
import torch as t
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchinfo import summary
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from gpt_mod import GPT2LMHeadModel_MixtureOfDepths
from utils import log_parameters_and_artifacts, model_stats, preprocess_data

EXPERIMENT_DIR = os.getenv("EXPERIMENT_DIR")

device = "cuda" if t.cuda.is_available() else "cpu"
if device != "cuda":
    raise ValueError("CUDA not available")
logger = logging.getLogger("__name__")

seed = 42
t.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--use_mod", type=bool, help="Use MoD", default=False)
    parser.add_argument("-b", "--batch_size", type=int, help="Training batch size.")
    parser.add_argument(
        "-c",
        "--capacity_fraction",
        type=float,
        help="Model capacity as fraction of total; float in [0, 1].",
    )
    parser.add_argument("-e", "--epochs", type=int, help="Training batch size.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Model name, one of: gpt2, gpt2-medium, gpt2-large, gpt2-xl.",
        default="gpt2",
    )
    parser.add_argument(
        "-l",
        "--log_every_n_steps",
        type=int,
        help="How often should we log?",
        default=100,
    )
    args = parser.parse_args()

    if args.use_mod:
        config = GPT2Config(capacity_fraction=args.capacity_fraction)
        model = GPT2LMHeadModel_MixtureOfDepths(config)
    else:
        config = GPT2Config()
        model = GPT2LMHeadModel(config)

    model.to(device)
    num_params, total_size = model_stats(model)
    logger.info(
        f"Initialised model: {args.model}, Number of parameters: {num_params/1e6}M, Total size: {total_size/1e6:.2f} MB"
    )
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    tokeniser.pad_token = tokeniser.eos_token

    optimiser = t.optim.AdamW(model.parameters(), lr=5e-5)
    iterable_dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")

    dataloader = DataLoader(iterable_dataset, batch_size=args.batch_size)

    return model, tokeniser, optimiser, dataloader, args


def train_loop(model, tokeniser, optimiser, dataloader, args):
    mlflow.set_experiment(EXPERIMENT_DIR + args.model)

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            batch = preprocess_data(batch, tokeniser)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].to(device)

            optimiser.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimiser.step()

            mlflow.log_metric("loss", loss.item())
            mlflow.log_metric("perplexity", t.exp(loss).item())

            if step % args.log_every_n_steps == 0:
                mlflow.log_metric("loss", loss.item(), step=step)
                duration = time.time() - start_time
                mlflow.log_metric("duration", duration, step=step)
                logger.info(
                    f"Epoch: {epoch}/{args.epochs}, Batch: {step}, {duration}, Loss: {loss.item()}"
                )


def train(model, tokeniser, optimiser, dataloader, args):
    # TODO: do we need two functions here?
    log_parameters_and_artifacts(model, args)

    mlflow.end_run()
    with mlflow.start_run() as run:
        train_loop(model, tokeniser, optimiser, dataloader, args)
    mlflow.pytorch.log_model(model, "model")


def main():
    model, tokeniser, optimiser, dataloader, args = setup()
    train(model, tokeniser, optimiser, dataloader, args)


if __name__ == "__main__":
    main()
