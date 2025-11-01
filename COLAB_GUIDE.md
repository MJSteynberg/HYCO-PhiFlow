# Running HYCO-PhiFlow on Google Colab

This guide explains how to run your experiments on Google Colab for free GPU access.

## Quick Start

1. **Upload to Google Drive:**
   - Upload your entire `HYCO-PhiFlow` folder to Google Drive
   - Note the path (e.g., `/MyDrive/HYCO-PhiFlow`)

2. **Open the Colab Notebook:**
   - Upload `colab_setup.ipynb` to Colab or open it from Drive
   - Or create a new notebook and copy the cells

3. **Update the Path:**
   - In cell #3, change `GDRIVE_PROJECT_PATH` to match your Drive location
   - Example: `GDRIVE_PROJECT_PATH = "/content/drive/MyDrive/University/HYCO-PhiFlow"`

4. **Run All Cells:**
   - Go to `Runtime → Run all` or run cells sequentially
   - The notebook will automatically setup everything

## How Cache Storage Works

### Default (Without Colab)
```yaml
# In conf/data/smoke_256.yaml
data_dir: 'data/'           # Relative to project root
cache_dir: 'data/cache'     # Relative to project root
```

### On Colab (Automatic in Notebook)
The notebook automatically uses fast local storage:
```bash
data_dir: '/content/data/smoke_256'      # Local SSD (fast!)
cache_dir: '/content/cache/smoke_256'    # Local SSD (fast!)
```

### Manual Override (if not using notebook)
You can override paths when running commands:
```bash
python run.py --config-name=smoke_experiment \
    data.data_dir=/content/data/smoke_256 \
    data.cache_dir=/content/cache/smoke_256
```

## Key Hydra Overrides for Colab

### Change Storage Locations
```bash
# Use local storage (fast)
python run.py --config-name=smoke_experiment \
    data.data_dir=/content/data/smoke_256 \
    data.cache_dir=/content/cache/smoke_256

# Use Google Drive (persistent but slower)
python run.py --config-name=smoke_experiment \
    data.data_dir=/content/drive/MyDrive/HYCO-PhiFlow/data/smoke_256 \
    data.cache_dir=/content/drive/MyDrive/HYCO-PhiFlow/data/cache/smoke_256
```

### Adjust Training Parameters
```bash
# Reduce batch size if OOM
python run.py --config-name=smoke_experiment \
    trainer_params.batch_size=4

# Train for fewer epochs (testing)
python run.py --config-name=smoke_experiment \
    trainer_params.epochs=50

# Change which simulations to use
python run.py --config-name=smoke_experiment \
    trainer_params.train_sim=[0,1,2,3,4]
```

### Run Different Modes
```bash
# Generate data only
python run.py --config-name=smoke_experiment \
    run_params.mode=[generate]

# Train only
python run.py --config-name=smoke_experiment \
    run_params.mode=[train]

# Train then evaluate
python run.py --config-name=smoke_experiment \
    run_params.mode=[train,evaluate]
```

## Storage Strategy

### Recommended Approach (used in notebook)
1. **Local Storage (`/content/`):**
   - Store raw simulation data
   - Store cached tensors
   - Fast I/O during training
   - **Deleted when session ends**

2. **Google Drive:**
   - Store source code
   - Store trained models
   - Store evaluation results
   - **Persists forever**

### Storage Layout
```
/content/                          (local, fast, temporary)
├── HYCO-PhiFlow/                 (code copied from Drive)
├── data/
│   └── smoke_256/                (raw simulation data)
│       ├── sim_000000/
│       ├── sim_000001/
│       └── ...
└── cache/
    └── smoke_256/                (cached tensors)
        ├── sim_000000.pt
        ├── sim_000001.pt
        └── ...

/content/drive/MyDrive/            (Google Drive, slower, persistent)
└── HYCO-PhiFlow/
    ├── src/                      (source code)
    ├── conf/                     (configs)
    ├── run.py
    └── colab_results_*/          (saved after training)
        ├── results/              (trained models)
        └── outputs/              (logs)
```

## Why Use Local Storage?

### Speed Comparison
| Storage | Read Speed | Write Speed | Use Case |
|---------|-----------|-------------|----------|
| `/content/` | **Fast** (~500 MB/s) | **Fast** | Training data, cache |
| Google Drive | Slow (~50 MB/s) | Very Slow | Code, final results |

### When Each is Loaded
- **Setup:** Code copied from Drive → `/content/` (once)
- **Generation:** Data created in `/content/data/` (once)
- **Caching:** Tensors saved to `/content/cache/` (once per sim)
- **Training:** Read from `/content/cache/` (thousands of times) ⚡
- **Save Results:** Copy `/content/results/` → Drive (once at end)

## Common Issues & Solutions

### 1. Session Timeout
**Problem:** Training interrupted after 12 hours

**Solution:**
- Save checkpoints regularly (modify trainer to save every N epochs)
- Use the save cell (#10) periodically during training
- Consider Colab Pro for longer sessions

### 2. Out of Memory (OOM)
**Problem:** CUDA out of memory error

**Solution:**
```bash
# Reduce batch size
python run.py --config-name=smoke_experiment \
    trainer_params.batch_size=4

# Use fewer prediction steps
python run.py --config-name=smoke_experiment \
    trainer_params.num_predict_steps=2

# Use smaller resolution
python run.py --config-name=smoke_quick_test  # Uses 128x128
```

### 3. Disk Space Full
**Problem:** `/content/` runs out of space

**Solution:**
- Free tier: ~70GB disk space
- Delete unnecessary files: `!rm -rf /content/data/` after caching
- Generate fewer simulations
- Use Google Drive for cache (slower but more space)

### 4. Cache Not Found After Restart
**Problem:** Session restarted, cache is gone

**Solution:**
- Cache will regenerate automatically (takes time)
- Or copy cache from previous results if saved:
```python
# Copy cache from Drive if you saved it
!cp -r /content/drive/MyDrive/HYCO-PhiFlow/saved_cache /content/cache
```

### 5. Import Errors
**Problem:** Module not found errors

**Solution:**
```python
# Reinstall dependencies
!pip install hydra-core omegaconf phiflow torch torchvision

# Check Python path
import sys
sys.path.insert(0, '/content/HYCO-PhiFlow')
```

## Tips for Long Training Sessions

1. **Monitor GPU Usage:**
```python
!nvidia-smi  # Check GPU utilization
!watch -n 1 nvidia-smi  # Monitor continuously
```

2. **Prevent Disconnection:**
   - Keep the Colab tab active
   - Use browser extensions to prevent sleep
   - Or use Colab Pro (background execution)

3. **Save Progress Regularly:**
```python
# Add this to your training loop or run periodically
import shutil
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = f"/content/drive/MyDrive/HYCO-PhiFlow/checkpoint_{timestamp}"
shutil.copytree("results", f"{backup_dir}/results")
```

4. **Use TensorBoard for Monitoring:**
```python
%load_ext tensorboard
%tensorboard --logdir outputs
```

## Available Experiments

Run any of these experiments on Colab:

```bash
# Smoke experiments
python run.py --config-name=smoke_experiment       # 256x256, full training
python run.py --config-name=smoke_quick_test      # 128x128, quick test

# Burgers equation
python run.py --config-name=burgers_experiment
python run.py --config-name=burgers_quick_test

# Heat equation  
python run.py --config-name=heat_physical_experiment

# Physical experiments
python run.py --config-name=smoke_physical_experiment
python run.py --config-name=burgers_physical_experiment
```

## Cost Comparison

| Option | GPU | RAM | Session | Cost |
|--------|-----|-----|---------|------|
| **Colab Free** | T4 (16GB) | 12GB | ~12hr | Free |
| **Colab Pro** | T4/P100/V100 | 32GB | 24hr | $10/mo |
| **Colab Pro+** | V100/A100 | 52GB | Background | $50/mo |
| **Local (your PC)** | ? | ? | Unlimited | $0 |

## Next Steps

1. Upload your project to Google Drive
2. Open `colab_setup.ipynb` in Colab
3. Update the `GDRIVE_PROJECT_PATH` in cell #3
4. Run all cells and wait for training to complete
5. Download results from the generated `colab_results_*` folder

Happy training! 🚀
