import torch
from src.data.tensor_dataset import TensorDataset


class FakeDataManager:
    def __init__(self, tensor_map):
        self._tensor_map = tensor_map

    def get_or_load_simulation(self, sim_index, field_names=None, num_frames=None):
        return {"tensor_data": {name: self._tensor_map[name] for name in field_names}}

    def is_cached(self, sim_index: int) -> bool:
        return True


def test_tensor_dataset_shapes():
    # Build BVTS tensors for two fields
    B, C1, C2, T, H, W = 1, 1, 2, 6, 8, 8

    f1 = torch.zeros((B, C1, T, H, W))
    f2 = torch.zeros((B, C2, T, H, W))

    dm = FakeDataManager({"f1": f1, "f2": f2})

    ds = TensorDataset(data_manager=dm, sim_indices=[0], field_names=["f1", "f2"], num_frames=None, num_predict_steps=2, augmentation_config=None)

    initial_state, rollout_targets = ds[0]

    # After concatenation C_all = C1 + C2
    C_all = C1 + C2

    assert isinstance(initial_state, torch.Tensor)
    assert isinstance(rollout_targets, torch.Tensor)

    # initial_state expected shape: [C_all, 1, H, W]
    assert initial_state.shape == (C_all, 1, H, W)

    # rollout_targets expected shape: [C_all, num_predict_steps, H, W]
    assert rollout_targets.shape == (C_all, ds.num_predict_steps, H, W)
