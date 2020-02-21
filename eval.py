import torch

def cal_metrics(pred, landmark, size, original_size):
    maximum_x, coords_x = pred.max(dim=-1)
    maximum_y, coords_y = maximum_x.max(dim=-1)
    import ipdb; ipdb.set_trace()

