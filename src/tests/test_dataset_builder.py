import torch
from src.data.dataset_utilities import DatasetBuilder


class FakeDataManager:
    def __init__(self, tensor):
        self._tensor = tensor

    def get_or_load_simulation(self, sim_index, field_names=None, num_frames=None):
        return {"tensor_data": {field_names[0]: self._tensor}}
    
    def is_cached(self, sim_index: int) -> bool:
        """Minimal cached check for tests: pretend the simulation is already cached.

        DatasetBuilder only queries this to decide whether to call
        get_or_load_simulation for caching. Returning True keeps the test focused
        on frame detection logic.
        """
        return True


def test_setup_cache_detects_bvts_layout():
    # BVTS layout: [B, C, T, H, W]
    B, C, T, H, W = 1, 3, 10, 8, 8
    tensor = torch.zeros((B, C, T, H, W))
    dm = FakeDataManager(tensor)
    builder = DatasetBuilder(dm)

    num_frames = builder.setup_cache([0], ["field"], num_frames=None, num_predict_steps=2)
    assert num_frames == T
 
