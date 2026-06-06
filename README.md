# ResNet-18 CIFAR-10 Benchmark

This project contains a set of PyTorch experiments for training and profiling a ResNet-18 style model on the CIFAR-10 image classification dataset.

The experiments focus on practical benchmarking questions: baseline training speed, data loading performance, CPU versus GPU runtime, optimizer behavior, batch normalization impact, and model parameter count.

## Project Contents

| File | Purpose |
| --- | --- |
| `lab2.py` | Main command-line runner for all experiments. |
| `c1.py` | Baseline ResNet-18 training experiment. |
| `c2.py` | Optimized training loop experiment. |
| `c3.py` | DataLoader and I/O worker performance test. |
| `c4.py` | Worker count comparison during training. |
| `c5.py` | CPU versus GPU training performance comparison. |
| `c6.py` | SGD versus Adam optimizer comparison. |
| `c7.py` | Batch normalization comparison. |
| `q3.py` | ResNet-18 parameter counting utility. |
| `trace.json` | Generated profiler trace output, when profiling is enabled. |

## Requirements

Use Python 3 with PyTorch and TorchVision installed.

```bash
pip install torch torchvision
```

CUDA is optional, but several experiments are designed to benchmark GPU execution. If CUDA is unavailable, run CPU-compatible experiments with `--device cpu`.

## Running Experiments

Run experiments through `lab2.py`.

```bash
python lab2.py --exercise c1
```

Available exercise names:

```text
c1, c2, c3, c4, c5, c6, c7, q3
```

Run every experiment in sequence:

```bash
python lab2.py --run_all
```

## Common Options

The main runner supports these options:

| Option | Default | Description |
| --- | --- | --- |
| `--exercise` | none | Selects one experiment to run. |
| `--run_all` | false | Runs all experiments sequentially. |
| `--epochs` | `5` | Number of training epochs. |
| `--batch_size` | `128` | Batch size used by the DataLoader. |
| `--num_workers` | `8` | Number of DataLoader worker processes. |
| `--lr` | `0.1` | Learning rate. |
| `--device` | `cuda` | Device to use: `cuda` or `cpu`. |
| `--optimizer` | `sgd` | Optimizer to use: `sgd` or `adam`. |
| `--profile` | false | Enables PyTorch profiling where supported. |

Example with custom training settings:

```bash
python lab2.py --exercise c5 --epochs 10 --batch_size 64 --num_workers 4 --lr 0.01 --optimizer adam
```

Run on CPU:

```bash
python lab2.py --exercise c5 --device cpu
```

Run on CUDA:

```bash
python lab2.py --exercise c5 --device cuda
```

## Profiling

Some experiments support PyTorch profiling with `--profile`.

```bash
python lab2.py --exercise c1 --profile
```

Profiling can create `trace.json` or trace logs, depending on the experiment. These files are generated outputs and may be replaced by later profiling runs.

## Dataset

The scripts use TorchVision's CIFAR-10 dataset loader. On the first run, the dataset is downloaded into a local `data` directory. Later runs reuse the local copy.

## Notes

- GPU experiments require a working CUDA-enabled PyTorch installation.
- DataLoader timing can vary between machines and even between runs on the same machine.
- Higher `num_workers` values are not always faster; the best value depends on CPU, storage, memory, and operating system scheduling.
- `trace.json` is a generated profiling artifact, not source code.
