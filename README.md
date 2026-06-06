# ResNet-18 Performance Optimization & Training Benchmarking

This project develops and benchmarks deep learning training pipelines for CIFAR-10 image classification using PyTorch and a ResNet-18 style convolutional neural network.

The goal is to evaluate how different training and systems-level choices affect model performance, training throughput, resource utilization, and convergence efficiency. The experiments compare SGD and Adam optimizers, CPU versus GPU execution, multi-worker data loading strategies, batch normalization behavior, and profiling output from PyTorch Profiler.

## Technical Highlights

- Implemented ResNet-18 style residual blocks for CIFAR-10 image classification in PyTorch.
- Built repeatable experiment scripts for benchmarking training speed, accuracy, optimizer behavior, and hardware execution paths.
- Compared SGD and Adam to observe differences in convergence, runtime, and training dynamics.
- Benchmarked CPU and CUDA execution to measure the impact of GPU acceleration on training throughput.
- Evaluated `DataLoader` worker counts to identify input pipeline bottlenecks and improve batch loading efficiency.
- Used PyTorch Profiler to capture runtime traces and inspect performance hotspots.
- Measured model parameter count to connect architecture size with compute and memory requirements.
- Organized experiments behind a single CLI runner for consistent configuration across benchmark runs.

## Benchmarking Scope

This repository is focused on performance engineering for scalable AI training workflows. It explores:

- Training loop efficiency
- Data loading and preprocessing overhead
- GPU acceleration versus CPU execution
- Optimizer tradeoffs between SGD and Adam
- Batch normalization impact on model behavior
- Runtime profiling and bottleneck identification
- Practical computer vision model experimentation on CIFAR-10

## Project Contents

| File | Purpose |
| --- | --- |
| `lab2.py` | Main command-line runner for all experiments. |
| `c1.py` | Baseline ResNet-18 training experiment. |
| `c2.py` | Optimized training loop experiment for throughput comparison. |
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

Run on CPU/GPU:

```bash
python lab2.py --exercise c5 --device cpu #cuda
```

Run on CUDA:

```bash
python lab2.py --exercise c5 --device cuda
```

## Key Learning Outcomes

This project provided hands-on experience with:

- Deep learning model training and evaluation in PyTorch
- Residual network architecture design for computer vision
- CUDA-based GPU acceleration
- Optimizer selection and convergence analysis
- Input pipeline tuning with multi-worker data loading
- PyTorch Profiler based performance diagnosis
- Training metrics analysis for throughput and accuracy
- Practical performance engineering for AI systems

## Dataset

The scripts use TorchVision's CIFAR-10 dataset loader. On the first run, the dataset is downloaded into a local `data` directory. Later runs reuse the local copy.