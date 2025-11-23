
import time
import torch
from phi.torch.flow import *
from phiml import math
from phiml import nn as phiml_nn
from src.training.synthetic.trainer import SyntheticTrainer

print(f"CUDA available: {torch.cuda.is_available()}")


# Mock config
config = {
    'trainer': {
        'synthetic': {
            'epochs': 1,
            'learning_rate': 1e-3,
            'rollout_steps': 2,
            'scheduler': 'cosine'
        },
        'batch_size': 2,
        'rollout_steps': 2
    },
    'model': {
        'synthetic': {
            'model_path': 'models/synthetic',
            'model_save_name': 'test_model'
        }
    }
}

# Mock model
class MockModel:
    def __init__(self):
        self.network = phiml_nn.u_net(
            in_channels=1,
            out_channels=1,
            levels=4,
            filters=32,
            activation='ReLU'
        )
    
    def __call__(self, x):
        return math.native_call(self.network, x)
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass

# Mock dataset
class MockBatch:
    def __init__(self):
        # [Batch, X, Y, Vector]
        self.initial_state = math.random_uniform(math.batch(batch=2), math.spatial(x=32, y=32), math.channel(vector=1))
        # [Batch, Time, X, Y, Vector]
        self.targets = math.random_uniform(math.batch(batch=2), math.spatial(time=2), math.spatial(x=32, y=32), math.channel(vector=1))

class MockDataset:
    def iterate_batches(self, batch_size, shuffle=True):
        for _ in range(5):
            yield MockBatch()

def run_benchmark():
    print("Initializing trainer...")
    model = MockModel()
    trainer = SyntheticTrainer(config, model)
    dataset = MockDataset()
    
    print(f"Optimizer type: {type(trainer.optimizer)}")
    print(f"Optimizer dir: {dir(trainer.optimizer)}")
    
    print("Starting training loop...")
    start_time = time.time()
    trainer.train(dataset, num_epochs=10, verbose=True)
    end_time = time.time()
    
    print(f"Training finished in {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
