import torch

def normalize_input(input_tensor):
    # copy of the defensive normalization added in FieldDataset
    t = input_tensor
    if torch.cuda.is_available():
        t = t.cuda() if not t.is_cuda else t
    if t.dim() == 4 and t.shape[1] == 1:
        t = t.squeeze(1)
    elif t.dim() == 5:
        if t.shape[0] == 1 and t.shape[2] == 1:
            t = t.squeeze(0).squeeze(1)
        elif t.shape[2] == 1:
            t = t.squeeze(2)
    return t

def normalize_target(target_tensor):
    t = target_tensor
    if t.dim() == 4:
        C0, D1, H, W = t.shape
        if C0 <= 16 and D1 > 1:
            t = t.permute(1, 0, 2, 3)  # assume [C,T,H,W] -> [T,C,H,W]
    elif t.dim() == 5:
        if t.shape[0] == 1:
            s = t.squeeze(0)  # [C, T, H, W]
            t = s.permute(1, 0, 2, 3)
    return t

shapes = [
    torch.randn(1, 3, 1, 128, 128),   # [B_sim, C, T, H, W] (common BVTS batch)
    torch.randn(3, 1, 128, 128),      # [C, T, H, W] (where T==1)
    torch.randn(3, 2, 128, 128),      # [C, T, H, W] (T>1)
    torch.randn(1, 3, 2, 128, 128),   # [B_sim, C, T, H, W] (T>1)
]

for s in shapes:
    inp = normalize_input(s)
    tgt = normalize_target(s)
    print("orig:", tuple(s.shape), "-> input_norm:", tuple(inp.shape), "target_norm:", tuple(tgt.shape))