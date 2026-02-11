# rootml-bridge

A Python toolkit for bridging ROOT and machine learning workflows in high-energy physics. Export ROOT files to Parquet, train ML models, and attach predictions back to ROOT files.

## Features

- **Export ROOT → Parquet**: Convert ROOT TTree data to Parquet format with metadata preservation
- **Train ML Models**: Built-in support for XGBoost with configurable training pipelines
- **Attach Predictions**: Add ML scores back to ROOT files as new branches
- **Chunked Processing**: Memory-efficient handling of large datasets
- **Provenance Tracking**: Automatic metadata capture (git commit, timestamps, config)

## Installation

```bash
# Clone the repository
git clone https://github.com/zagraywolf/rootml-bridge.git
cd rootml-bridge

# Install dependencies
pip install --break-system-packages \
    ROOT \
    pandas \
    pyarrow \
    xgboost \
    scikit-learn \
    pyyaml \
    typer
```

## Quick Start

### 1. Generate Synthetic Data (Optional)

```bash
python examples/make_synthetic_root.py
```

This creates `synthetic.root` with 50,000 events containing:
- Features: `x1`, `x2`, `x3`
- Label: `label` (binary classification)
- Event identifiers: `run`, `lumi`, `event`
- Event weights: `weight`

### 2. Export ROOT to Parquet

```bash
python -m rootml.cli.main export \
    --config configs/export.yaml \
    --out data.parquet
```

**Export config** (`configs/export.yaml`):
```yaml
input_files:
  - synthetic.root

tree: Events

features:
  - x1
  - x2
  - x3

label: label
weight: weight

event_id:
  - run
  - lumi
  - event

selection: null  # Optional ROOT selection string
chunk_size: 10000  # Rows per chunk
```

### 3. Train ML Model

```bash
python -m rootml.cli.main train \
    --data data.parquet \
    --config configs/train.yaml \
    --out outputs/train_run_1
```

**Training config** (`configs/train.yaml`):
```yaml
model: xgboost

target: label
weight: weight

features:
  - x1
  - x2
  - x3

test_size: 0.2
val_size: 0.1
seed: 42

xgb_params:
  max_depth: 4
  n_estimators: 200
  learning_rate: 0.1
  subsample: 0.8
```

**Outputs**:
- `model.json`: Trained XGBoost model
- `metrics.json`: Test AUC and other metrics
- `scores.parquet`: Full dataset with ML predictions

### 4. Attach Scores to ROOT File

```bash
python -m rootml.cli.main attach \
    --input-root synthetic.root \
    --tree Events \
    --scores outputs/train_run_1/scores.parquet \
    --out synthetic_with_scores.root
```

This creates a new ROOT file with an additional `ml_score` branch.

## Workflow Overview

```
ROOT File (TTree)
    ↓
    └─ Export (chunked) → Parquet
                            ↓
                            └─ Train ML → Model + Scores
                                            ↓
                                            └─ Attach → ROOT File + ML Branch
```

## Configuration

### Export Configuration

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `input_files` | List[str] | ROOT files to process | Yes |
| `tree` | str | TTree name | Yes |
| `features` | List[str] | Feature columns | Yes |
| `label` | str | Target variable | Yes |
| `event_id` | List[str] | Event identifier columns | Yes |
| `weight` | str | Event weight column | No |
| `selection` | str | ROOT selection string | No |
| `chunk_size` | int | Rows per chunk (default: 100k) | No |

### Training Configuration

| Field | Type | Description |
|-------|------|-------------|
| `model` | str | Model type (currently: `xgboost`) |
| `target` | str | Target variable name |
| `weight` | str | Weight column name |
| `features` | List[str] | Feature columns |
| `test_size` | float | Test set fraction |
| `val_size` | float | Validation set fraction |
| `seed` | int | Random seed |
| `xgb_params` | dict | XGBoost hyperparameters |

## Advanced Usage

### Manual Attachment (Python API)

```python
from rootml.attach import attach_scores

attach_scores(
    input_root="input.root",
    tree_name="Events",
    scores_path="scores.parquet",
    event_col="event",
    score_col="ml_score",
    output_root="output.root"
)
```

### Custom Selection Filters

Apply ROOT selection strings during export:

```yaml
selection: "pt > 30 && abs(eta) < 2.4"
```

### Metadata Access

The exported Parquet file contains provenance metadata:

```python
import pyarrow.parquet as pq

metadata = pq.read_metadata('data.parquet')
custom_metadata = metadata.schema.metadata

print(custom_metadata[b'rootml_export_time'].decode())
print(custom_metadata[b'git_commit'].decode())
print(custom_metadata[b'config'].decode())
```

## Architecture

### Module Structure

```
rootml/
├── __init__.py
├── attach.py          # Attach scores to ROOT files
├── config.py          # Configuration loading/validation
├── export.py          # ROOT → Parquet export
├── cli/
│   └── main.py        # Command-line interface
└── train/
    ├── __init__.py
    ├── base.py        # BaseTrainer abstract class
    ├── run.py         # Training dispatcher
    └── xgb_trainer.py # XGBoost implementation
```

### Key Design Choices

1. **Chunked Processing**: Uses RDataFrame's entry index (`rdfentry_`) to process large files in chunks without loading everything into memory

2. **Implicit Multi-Threading Disabled**: `ROOT.ROOT.DisableImplicitMT()` is called to ensure deterministic chunking behavior

3. **Event ID Preservation**: Maintains run/lumi/event identifiers throughout the pipeline for accurate score attachment

4. **Flattened Arrays**: ROOT-exported columns (stored as 1-element arrays) are automatically flattened during training

## Common Issues

### Array Flattening

ROOT exports numeric columns as 1-element arrays. The training pipeline automatically flattens these:

```python
# Automatic flattening in xgb_trainer.py
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].apply(lambda x: x[0] if hasattr(x, "__len__") else x)
```

### Memory Management

For very large files (>100M events), adjust `chunk_size` in the export config:

```yaml
chunk_size: 50000  # Smaller chunks = less memory
```

### Missing Event IDs

If attachment fails with missing events, verify that:
1. Event IDs are unique in the ROOT file
2. The same events exist in both the ROOT file and scores Parquet
3. Event ID column names match between config and data

## Extending the Framework

### Adding a New Model Type

1. Create a new trainer in `rootml/train/`:

```python
from rootml.train.base import BaseTrainer

class MyTrainer(BaseTrainer):
    def train(self, data_path, config, out_dir):
        # Your training logic
        pass
```

2. Register in `rootml/train/run.py`:

```python
def run_training(data_path, config, out_dir):
    model_type = config["model"]
    
    if model_type == "xgboost":
        trainer = XGBTrainer()
    elif model_type == "mymodel":
        trainer = MyTrainer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

## Performance

Benchmarks on a typical HEP dataset (10M events, 20 features):

| Task | Time | Memory |
|------|------|--------|
| Export (100k chunks) | ~5 min | ~2 GB |
| Training (XGBoost) | ~3 min | ~4 GB |
| Attachment | ~2 min | ~1 GB |

## Contributing

Contributions welcome! Areas for improvement:
- Additional model types (LightGBM, neural networks)
- GPU acceleration for training
- Distributed processing support
- More sophisticated feature engineering

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{rootml_bridge,
  author = {Singh, Kunwar Abhuday},
  title = {rootml-bridge: ROOT and ML Integration Toolkit},
  year = {2026},
  url = {https://github.com/zagraywolf/rootml-bridge}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.
