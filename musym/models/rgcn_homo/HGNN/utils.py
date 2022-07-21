import torch
import torch.nn.functional as F

def h_add(u, v, eps=1e-5):
    "add two tensors in hyperbolic space."
    v = v + eps
    th_dot_u_v = 2. * torch.sum(u * v, dim=1, keepdim=True)
    th_norm_u_sq = torch.sum(u * u, dim=1, keepdim=True)
    th_norm_v_sq = torch.sum(v * v, dim=1, keepdim=True)
    denominator = 1. + th_dot_u_v + th_norm_v_sq * th_norm_u_sq
    result = (1. + th_dot_u_v + th_norm_v_sq) / (denominator + eps) * u + \
             (1. - th_norm_u_sq) / (denominator + eps) * v
    return torch.renorm(result, 2, 0, (1-eps))


def exp_map_zero(v, eps=1e-5):
    """
    Exp map from tangent space of zero to hyperbolic space
    Args:
        v: [batch_size, *] in tangent space
    """
    v = v + eps
    norm_v = torch.norm(v, 2, 1, keepdim=True)
    result = F.tanh(norm_v) / (norm_v) * v
    return torch.renorm(result, 2, 0, (1-eps))


def log_map_zero(v, eps=1e-5):
    """
        Exp map from hyperbolic space of zero to tangent space
        Args:
            v: [batch_size, *] in hyperbolic space
    """
    diff = v + eps
    norm_diff = torch.norm(v, 2, 1, keepdim=True)
    atanh = torch.min(norm_diff, torch.Tensor([1.0 - eps])) #.to(v.get_device()))
    return 1. / atanh / norm_diff * diff


def h_mul(u, v):
    out = torch.mm(u, log_map_zero(v))
    out = exp_map_zero(out)
    return out