"""Opt-in B2 classification selector fix for singleton/empty class subsets."""


def balanced_classification_loss(pred, label):
    """Retain upstream 0.5/0.5 class balancing, never discard a singleton.

    Ignored labels (-1/-2) remain ignored. Empty subsets return a differentiable
    zero on the prediction device. No CUDA requirement is introduced.
    """
    import torch
    import torch.nn.functional as functional

    if pred.ndim < 1 or pred.shape[-1] != 2 or pred.numel() // 2 != label.numel():
        raise ValueError("prediction rows and label count must match")
    pred = pred.reshape(-1, 2)
    label = label.reshape(-1)
    if not torch.isfinite(label).all() or not torch.isin(label, label.new_tensor([-2, -1, 0, 1])).all():
        raise ValueError("labels must be -2/-1 ignore or binary 0/1")
    # Unlike pred.sum()*0, an empty sum never reads ignored NaN/Inf values.
    total = pred.reshape(-1)[:0].sum()
    for class_id in (1, 0):
        select = label.eq(class_id).nonzero(as_tuple=False).flatten().to(pred.device)
        if select.numel():
            selected_pred = pred.index_select(0, select)
            selected_label = label.to(pred.device).index_select(0, select).long()
            true_log_probability = selected_pred.gather(1, selected_label[:, None])
            if not torch.isfinite(true_log_probability).all():
                raise ValueError("active true-class log probabilities must be finite")
            term = functional.nll_loss(selected_pred, selected_label)
            if not torch.isfinite(term):
                raise ValueError("active classification reduction overflow")
            total = total + 0.5 * term
    return total
