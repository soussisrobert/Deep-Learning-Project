"""Opt-in B2 classification selector fix for singleton/empty class subsets."""


def balanced_classification_loss(pred, label):
    """Retain upstream 0.5/0.5 class balancing, never discard a singleton.

    Ignored labels (-1/-2) remain ignored. Empty subsets return a differentiable
    zero on the prediction device. No CUDA requirement is introduced.
    """
    import torch.nn.functional as functional

    pred = pred.reshape(-1, 2)
    label = label.reshape(-1)
    total = pred.sum() * 0
    for class_id in (1, 0):
        select = label.eq(class_id).nonzero(as_tuple=False).flatten().to(pred.device)
        if select.numel():
            selected_pred = pred.index_select(0, select)
            selected_label = label.to(pred.device).index_select(0, select).long()
            total = total + 0.5 * functional.nll_loss(selected_pred, selected_label)
    return total
