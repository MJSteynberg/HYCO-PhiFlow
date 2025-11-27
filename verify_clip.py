
from phi.flow import *
from phi import math

def verify_clip():
    t = math.tensor([-1.0, 0.5, 2.0])
    clipped = math.clip(t, 0.0, 1.0)
    print(f"Original: {t}")
    print(f"Clipped: {clipped}")
    
    # Check if it works with tensors as bounds
    min_val = math.tensor(0.0)
    max_val = math.tensor(1.0)
    clipped_tensor = math.clip(t, min_val, max_val)
    print(f"Clipped with tensor bounds: {clipped_tensor}")

if __name__ == "__main__":
    verify_clip()
