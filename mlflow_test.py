import os

import mlflow.pytorch
import torch
from mlflow import MlflowClient
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchmetrics import Accuracy

# Set the MLflow experiment
mlflow.login()
experiment_name = "os.getenv("EXPERIMENT_DIR")"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

# Define a synthetic dataset
class SyntheticMNISTDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Generate random images (28x28 pixels) and labels (10 classes)
        self.data = torch.randn(num_samples, 1, 28, 28)
        self.targets = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        # Specify the task as "multiclass" for the Accuracy metric
        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Initialize the synthetic dataset and data loader
dataset = SyntheticMNISTDataset(num_samples=1000)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
model = MNISTModel()

# Configure the optimizer
optimizer = model.configure_optimizers()

# Training loop
def train(model, data_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):  # Add this outer loop for epochs
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model.training_step((data, target), batch_idx)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            
            # Optionally, log loss every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}/{epochs} Batch: {batch_idx} \tLoss: {loss.item()}')
                mlflow.log_metric("loss", f"{loss:3f}", step=(batch_idx // 100))


epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
metric_fn = Accuracy(task="multiclass", num_classes=10)

with mlflow.start_run() as run:
    params = {
        "epochs": epochs,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "Adam",
    }
    # Log training parameters.
    mlflow.log_params(params)
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
    train(model, data_loader, optimizer, epochs)
    mlflow.pytorch.log_model(model, "model")



def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

# Example of manually logging metrics after the run
print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))