def model_stats(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size = sum(
        p.numel() * p.element_size() for p in model.parameters() if p.requires_grad
    )
    return num_parameters, total_size
