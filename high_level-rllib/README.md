## Quick Start Guide

### Step 1: Navigate to Working Directory
```bash
cd high_level-rllib
```

---

### Step 2: Setup & Installation
```bash
# Create conda environment from the provided rllib.yml file
conda env create -f ./_docker/rllib.yml

# Activate the environment
conda activate rllib
```

---

### Step 3: Prepare Training Data
The module expects graphs in subdirectories:
- `Station_graphs/` - Static graphs with base stations
- `NoStation_graphs/` - Static graphs without base stations
- `Dynamic_graphs/` - Graphs with time-varying priorities

Each graph directory should contain pickled graph objects and associated metadata.

---

### Step 4: Configure Training (ppo_config.json)
Edit hyperparameters:
```json
{
    "training": {
        "lr": 3e-4,
        "training_iter": 500,
        "checkpoint_freq": 50
    },
    "resource": {
        "num_env_runners": 4,
        "num_gpus": 1
    }
}
```

---

### Step 5: Train a Model
```bash
# Train PPO on 25-node graphs
python rllib_ppo.py 25

# Outputs:
# - Checkpoints in tuner_logs_*/
# - Training logs and metrics
# - Configuration saved with model
```

---

### Step 6: Evaluate Trained Model
```bash
# Evaluate using pure RL policy
# Note: Replace <checkpoint_path> with the actual path from tuner_logs_*/ (Ray Tune generates timestamped paths)
python eval_ppo.py RL 150 --checkpoint <checkpoint_path> --hidden_size 32 --graph 25N-3A-15T-RP-RD-GRID-1

# Outputs CSV files with metrics:
# - eval_logs_NS-general/results-150steps/<graph_name>/summary/RL.csv - Aggregated statistics
# - eval_logs_NS-general/results-150steps/<graph_name>/detailed/RL.csv - Per-instance results
```

---

### Step 7: Compare with Baselines
```bash
# Run CBLS baseline
python CBLS_runner.py ./NoStation_graphs 25N-3A-15T-RP-RD-GRID-1
```