from dataclasses import dataclass

from torchinfo import summary
import mlflow.pytorch
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as t
from thop import profile

from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
from torch.datasets import DataLoader
from utils import model_stats

@dataclass
class Parameters:
    batch_size: int = 32
    batch_size_dataset_loader: int = 32
    epochs: int = 10
    experiment_name: str = "os.getenv("EXPERIMENT_DIR")"



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

def create_model():
    model = ""
    optimizer = ""
    return model, optimizer

def preprocess_data():
    iterable_dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
    dataloader = DataLoader(iterable_dataset, batch_size=Parameters.batch_size_dataset_loader, shuffle=True)
    return dataloader


def train_loop(model, optimizer, dataloader, epochs, experiment_name):
    mlflow.login()
    mlflow.set_experiment(experiment_name)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(data, target)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            mlflow.log_metric("loss", loss.item())
            mlflow.log_metric("perplexity", t.exp(loss).item())

            if batch_idx % 100 == 0:
                mlflow.log_metric("loss", loss.item(), step=(batch_idx // 100))

def log_parameters_and_artifacts(model):
    mlflow.log_params(Parameters.__dict__)
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

def train():
    dataloader = preprocess_data(Parameters.batch_size)
    model, optimizer = create_model()
    log_parameters_and_artifacts(model)

    with mlflow.start_run() as run:
        train_loop(model, optimizer, dataloader, Parameters.epochs, Parameters.experiment_name)
    mlflow.pytorch.log_model(model, "model")


def main():
    load_models()
    
if __name__ == "__main__":
    main()