import torch
from copy import deepcopy


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data


def find_quantile(arr, perc):
    arr_sorted = torch.sort(arr).values
    frac_idx = perc * (len(arr_sorted) - 1)
    frac_part = frac_idx - int(frac_idx)
    low_idx = int(frac_idx)
    high_idx = low_idx + 1
    quant = arr_sorted[low_idx] + (arr_sorted[high_idx] - arr_sorted[low_idx]) * frac_part  # linear interpolation

    return quant
