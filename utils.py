import mlflow.pytorch

from torchinfo import summary


def model_stats(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    return num_parameters, total_size


def preprocess_data(batch, tokeniser):
    texts = batch["text"]  # Extracting text data from the batch

    # Tokenize the text data
    batch_encoding = tokeniser(
        texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"
    )

    # You no longer need to manually pad or convert lists to tensors since the tokeniser does this for you
    return {
        "input_ids": batch_encoding["input_ids"],
        "attention_mask": batch_encoding["attention_mask"],
        "labels": batch_encoding[
            "input_ids"
        ],  # Assuming you want to use the input IDs as labels for some sort of language modeling
    }


def log_parameters_and_artifacts(model, args):
    mlflow.log_params(args.__dict__)
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")
