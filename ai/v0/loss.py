import pdb

from amends import _torch
import torch



def error(y_, y):
    stdv = torch.exp(.5 * y_[:,[1]])
    a = (y - y_[:,[0]]) / stdv
    return a
def abs_error(y_, y):
    return torch.abs(error(y_, y))

def simp_nll(y_, y, alpha=1, beta=1):
    mu = y_[...,[0]]
    log_var = y_[...,[1]]
    r = alpha * log_var + beta * ((y - mu)**2 / torch.exp(log_var))
    return r.mean()
def mse(y_, y):
    assert y.shape == y_.shape
    r = y - y_
    r = r * r
    return r.mean()
def mse_adapted(y_, y):
    r = mse(y_[:,0].reshape(y.shape), y)
    return r
def huber(y_, y, delta=10):
    a = abs_error(y_, y)
    normal_case_mask = a <= delta
    a[normal_case_mask] = .5 * (a[normal_case_mask] * a[normal_case_mask])
    a[~normal_case_mask] = delta * (a[~normal_case_mask] - .5 * delta)
    return a.mean()
def l2(params):
    return _torch.flat_sum(params, key_f=lambda parameter: (parameter**2).sum())


by_name = {
    'simp_nll': simp_nll,
    'mse': mse_adapted,
    'huber': huber,
}


def criterion(
    model,
    lammda=.0001,
    loss_name='simp_nll',
    loss_f=None
):
    def _(y_, y):
        loss_f_ = by_name.get(loss_name, loss_f)
        r = loss_f_(y_, y[0])
        if lammda is not None:
            print(f'adding l2 term')
            r += lammda * l2(model.parameters())
        return r
    return _