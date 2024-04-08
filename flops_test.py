from transformers import AutoConfig, GPT2LMHeadModel, GPT2Tokenizer
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn


class GPT2Wrapper(nn.Module):
    def __init__(self, config):
        super(GPT2Wrapper, self).__init__()
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    @property
    def transformer(self):
        return self.model.transformer

class BlockWrapper(nn.Module):
    def __init__(self, block, embedding):
        super().__init__()
        self.block = block
        self.embedding = embedding

    def forward(self, input_ids, attention_mask=None):
        # Simulating input processing through embedding and then through one transformer block
        x = self.embedding(input_ids)
        x = self.block(x)[0]
        return x
    
def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Initialize the model with the given configuration
    model = GPT2Wrapper(config)
    return tokenizer, model

def log_model_flops(model, tokenizer, model_alias):
    inputs = tokenizer("I love hamsters.", return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)

    model_inputs = (input_ids,) if attention_mask is None else (input_ids, attention_mask)

    flops = FlopCountAnalysis(model, model_inputs)
    total_flops = flops.total()

    transformer = model.transformer
    embedding = model.transformer.wte

    num_blocks = len(transformer.h)

    first_block = transformer.h[0]
    block_wrapper = BlockWrapper(first_block, embedding)

    flops = FlopCountAnalysis(block_wrapper, (input_ids,))
    block_flops = flops.total()

    flops_other = total_flops - (num_blocks * block_flops)

    mod_capacity_budget = 0.125
    flops_budget = 1e18
    batch_size = 32

    flop_per_block_with_budget = block_flops * mod_capacity_budget

    num_budget_blocks = num_blocks // 2
    num_normal_blocks = num_blocks - num_budget_blocks

    normal_blocks_flops = num_normal_blocks * block_flops
    budget_blocks_flops = num_budget_blocks * flop_per_block_with_budget

    total_flops_with_budget = normal_blocks_flops + budget_blocks_flops + flops_other
    total_flops_with_budget_per_batch = total_flops_with_budget / batch_size
    num_forward_passes = flops_budget / total_flops_with_budget_per_batch

    total_flops_without_budget = num_blocks * block_flops + flops_other
    total_flops_without_budget_per_batch = total_flops_without_budget / batch_size
    num_forward_passes_without_budget = flops_budget / total_flops_without_budget_per_batch

    return {
        "model": model_alias,
        "total_flops": f"{total_flops:.1e}",
        "num_blocks": num_blocks,
        "block_flops": f"{block_flops:.1e}",
        "flop_per_block_with_budget": f"{flop_per_block_with_budget:.1e}",
        "flops_other": f"{flops_other:.1e}",
        "num_forward_passes_without_budget": f"{int(num_forward_passes_without_budget):,}",
        "num_forward_passes": f"{int(num_forward_passes):,}",
    }

def create_md_table(data):
    # Now print the table in a transposed format
    headers = ["Metric"] + [row["model"] for row in data]
    rows = [
        ["Total FLOPs"] + [row["total_flops"] for row in data],
        ["# of Transformer Blocks"] + [row["num_blocks"] for row in data],
        ["FLOPs for Standard Block"] + [row["block_flops"] for row in data],
        ["FLOPs for MOT Block"] + [row["flop_per_block_with_budget"] for row in data],
        ["FLOPs for All Non-Transformer Layers"] + [row["flops_other"] for row in data],
        ["# of 32B Forward Passes w/o MOT"] + [row["num_forward_passes_without_budget"] for row in data],
        ["# of 32B Forward Passes w/ MOT"] + [row["num_forward_passes"] for row in data],
    ]

    # Formatting and printing the table
    print("|" + "|".join(headers) + "|")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("|" + "|".join(map(str, row)) + "|")

if __name__ == "__main__":
    model_aliases = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    table_results = []
    for model_alias in model_aliases:
        tokenizer, model = load_model(model_alias)
        row = log_model_flops(model, tokenizer, model_alias)
        table_results.append(row)
    create_md_table(table_results)


    