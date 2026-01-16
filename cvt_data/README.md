
# Build Env
``` bash
# create an env "cvt" includes lerobot v0.3.3
cd cvt_data
uv sync --python 3.10

# enter "cvt" and install kuavo_msgs
cd your/path/of/kuavo-manip-lightly/kuavo_msgs
uv pip install -e .
```

# Run Cvt
```bash
# make sure you have modified "source_dirs"&"target_dir" in script before running
uv run python cvt_rosbag2lerobot.py
```

# Viz Data
```bash
uv run python -m lerobot.scripts.visualize_dataset --repo-id /your/dataset/path  --episode-index 0
```