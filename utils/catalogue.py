def init_catalogue(model, wide_strategy, deep_strategy):
    catalogue = {}
    catalogue["baseline"] = []
    catalogue["topk"] = []
    catalogue["qsgd"] = []

    for name, param in model.named_parameters():
        if "wide" in name:
            catalogue[wide_strategy].append(name)
        if "deep" in name:
            catalogue[deep_strategy].append(name)

    return catalogue

def show_catalogue(catalogue, log):
    for key, value in catalogue.items():
        log.logger.info(f"{key} layer: {value}")

def evaluate_sparsity():
    """
    To be completed.
    :return:
    """
    return