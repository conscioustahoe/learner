# Learner for PyTorch

A powerful and flexible training framework for PyTorch that provides a clean, extensible interface for training neural networks. This implementation offers advanced features like callbacks, visualization, and model interpretation through a unified API.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Features](#features)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Callbacks](#callbacks)
- [Visualization](#visualization)
- [Model Interpretation](#model-interpretation)
- [Examples](#examples)

## Installation

### 1. Create Virtual Environment

```bash
# Create a new virtual environment
python -m venv learner_env

# Activate the environment
source learner_env/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU only:
# pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy matplotlib pandas seaborn scikit-learn

# Optional: Install Jupyter for notebooks
pip install jupyter notebook ipykernel
```

### 3. Save the Learner Code

Save the learner implementation as `learner.py` in your project directory.

## Quick Start

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from learner import Learner, Accuracy, ProgressCallback

# Create a simple model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create dummy data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
X_valid = torch.randn(200, 784)
y_valid = torch.randint(0, 10, (200,))

# Create DataLoaders
train_ds = TensorDataset(X_train, y_train)
valid_ds = TensorDataset(X_valid, y_valid)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=32)

# Create Learner
learn = Learner(
    dls=[train_dl, valid_dl],
    model=model,
    loss_func=nn.CrossEntropyLoss(),
    metrics=[Accuracy()],
    cbs=[ProgressCallback()]
)

# Train
learn.fit(n_epoch=5, lr=0.01)
```

## Core Concepts

### The Learner

The `Learner` class is the central abstraction that combines:
- **Model**: Any PyTorch neural network
- **Data**: DataLoaders for training and validation
- **Loss Function**: How to compute the loss
- **Optimizer**: How to update weights
- **Callbacks**: Extensible hooks into the training loop
- **Metrics**: What to track during training

### Why This Design?

The Learner pattern solves several problems in deep learning development:

1. **Separation of Concerns**: Training logic is separate from model architecture
2. **Reusability**: Same training code works for any model/task
3. **Extensibility**: Add features via callbacks without changing core code
4. **Debugging**: Centralized place to monitor training
5. **Best Practices**: Encodes good training practices by default

### Callbacks

Callbacks are the key to the Learner's flexibility. They can:
- Monitor training progress
- Modify the training loop
- Log metrics
- Implement training techniques (scheduling, early stopping, etc.)
- Visualize results

**Callback Philosophy:**
- Callbacks can see and modify any part of training state
- Multiple callbacks can work together
- Order of execution is controlled
- Can cancel operations at any point

### Training Loop

The training loop follows this flow:
1. `before_fit` → Start of training
2. For each epoch:
   - `before_epoch`
   - Training phase: `before_train` → batches → `after_train`
   - Validation phase: `before_validate` → batches → `after_validate`
   - `after_epoch`
3. `after_fit` → End of training

## Understanding Key Features

### How Learning Rate Finder Works

The learning rate finder helps you find an optimal learning rate by training your model with exponentially increasing learning rates and recording the loss at each step.

**The Algorithm:**
1. Start with a very small learning rate (e.g., 1e-7)
2. Train for one batch and record the loss
3. Increase the learning rate exponentially
4. Repeat until the loss explodes or reaches a maximum learning rate
5. Plot loss vs learning rate on a log scale

**What to Look For:**
- The steepest downward slope indicates the fastest learning
- Choose a learning rate slightly before the minimum point
- Avoid the region where loss starts increasing rapidly

```python
# How it works internally:
# 1. Save model state
# 2. Train with increasing LR: lr = start_lr * (end_lr/start_lr)^(iteration/num_iterations)
# 3. Record losses
# 4. Restore model state
# 5. Plot and find steepest descent
```

### One-Cycle Training Explained

One-cycle training is a learning rate scheduling technique that can dramatically improve training speed and final model performance.

**The Concept:**
Instead of using a fixed learning rate or only decreasing it, one-cycle training:

1. **Warmup Phase** (30% of training):
   - Learning rate increases from `lr/div_factor` to `max_lr`
   - Momentum decreases from 0.95 to 0.85
   - Allows model to explore the loss landscape

2. **Annealing Phase** (70% of training):
   - Learning rate decreases from `max_lr` to `lr/final_div`
   - Momentum increases from 0.85 back to 0.95
   - Helps model settle into a good minimum

**Why It Works:**
- Initial low LR helps stable start
- High LR in middle enables faster training and escaping local minima
- Final low LR allows fine-tuning
- Inverse momentum relationship helps stabilization

```python
# Example schedule visualization:
# LR:    /\_____ (increases then decreases)
# Mom:   \_/‾‾‾‾ (decreases then increases)
```

### Mixed Precision Training Deep Dive

Mixed precision training uses both 16-bit (half) and 32-bit (single) floating-point precision to speed up training while maintaining model accuracy.

**How It Works:**

1. **Forward Pass**: Compute in FP16 (faster)
2. **Loss Scaling**: Scale the loss to prevent gradient underflow
3. **Backward Pass**: Compute gradients in FP16
4. **Gradient Unscaling**: Unscale gradients before optimizer step
5. **Master Weights**: Keep FP32 copies of weights for updates

**Benefits:**
- 2-3x faster training on modern GPUs (V100, RTX 30XX, A100)
- 50% less memory usage
- Larger batch sizes possible
- Minimal accuracy loss

**The Process:**
```python
# Pseudocode of what happens:
with autocast():  # Convert to FP16
    output = model(input)
    loss = loss_fn(output, target)

# Scale loss to prevent underflow
scaled_loss = loss * scale_factor
scaled_loss.backward()

# Unscale gradients
for param in model.parameters():
    param.grad /= scale_factor

# Update with FP32 master weights
optimizer.step()
```

### Gradient Clipping Explained

Gradient clipping prevents the "exploding gradient" problem by limiting the magnitude of gradients during backpropagation.

**Types of Clipping:**

1. **Value Clipping**: Clip each gradient component
   ```python
   if gradient > threshold:
       gradient = threshold
   elif gradient < -threshold:
       gradient = -threshold
   ```

2. **Norm Clipping** (used in our implementation):
   ```python
   total_norm = sqrt(sum(grad^2 for all gradients))
   clip_coef = max_norm / (total_norm + 1e-6)
   if clip_coef < 1:
       gradient *= clip_coef
   ```

**When to Use:**
- RNNs and LSTMs (prone to exploding gradients)
- Deep networks
- When you see NaN losses
- Training instability

### Early Stopping Mechanism

Early stopping prevents overfitting by monitoring a metric and stopping training when it stops improving.

**How It Works:**
1. Monitor a metric (usually validation loss)
2. Track the best value seen
3. Count epochs without improvement
4. Stop if no improvement for `patience` epochs

**Implementation Details:**
```python
if current_metric better than best_metric:
    best_metric = current_metric
    num_bad_epochs = 0
else:
    num_bad_epochs += 1
    if num_bad_epochs >= patience:
        stop_training()
```

### Callback System Architecture

The callback system allows modular extensions to the training loop without modifying core code.

**Design Pattern:**
- **Observer Pattern**: Callbacks observe training events
- **Chain of Responsibility**: Multiple callbacks can handle same event
- **Ordered Execution**: Callbacks run in priority order

**Callback Lifecycle:**
```
Training Start
    ↓
before_fit() ──→ [setup resources]
    ↓
For each epoch:
    before_epoch() ──→ [epoch setup]
        ↓
    Training:
        before_train() ──→ [switch to train mode]
        For each batch:
            before_batch() ──→ [prepare batch]
            forward pass
            after_pred() ──→ [process predictions]
            compute loss
            after_loss() ──→ [process loss]
            backward pass
            after_backward() ──→ [process gradients]
            optimizer step
            after_step() ──→ [after weight update]
            after_batch() ──→ [batch cleanup]
        after_train() ──→ [training cleanup]
        ↓
    Validation:
        [similar structure with before_validate/after_validate]
        ↓
    after_epoch() ──→ [log metrics, save models]
    ↓
after_fit() ──→ [final cleanup]
```

### Metrics System

Metrics accumulate values during training/validation to compute epoch-level statistics.

**Architecture:**
```python
class Metric:
    def reset(self):  # Called at epoch start
    def accumulate(self, learn):  # Called after each batch
    def value(self):  # Return computed metric
```

**Difference from Loss:**
- Loss is used for optimization (needs gradients)
- Metrics are for monitoring (no gradients needed)
- Metrics can be non-differentiable (e.g., accuracy)

### Hook System for Model Introspection

PyTorch hooks allow you to inspect or modify intermediate values during forward/backward passes.

**Types of Hooks:**
1. **Forward Hooks**: Access layer inputs/outputs
2. **Backward Hooks**: Access gradients

**Use Cases:**
- Visualizing activations
- Debugging vanishing/exploding gradients
- Feature extraction
- Gradient modification

```python
def forward_hook(module, input, output):
    # module: the layer
    # input: tuple of inputs to the layer
    # output: layer output
    # Can modify output by returning new value
    
def backward_hook(module, grad_input, grad_output):
    # grad_input: gradients w.r.t. layer inputs
    # grad_output: gradients w.r.t. layer outputs
    # Can modify gradients by returning new values
```

## Features

### Core Training Features
- Automatic train/eval mode switching
- Device management (CPU/GPU)
- Gradient accumulation
- Mixed precision training
- Learning rate scheduling
- Model checkpointing

### Visualization
- Training/validation loss plots
- Metric tracking and plotting
- Learning rate visualization
- Activation and gradient statistics
- Model architecture summary

### Model Interpretation
- Confusion matrices
- Top loss analysis
- Most confused predictions
- Classification reports

## Basic Usage

### Creating a Learner

```python
learn = Learner(
    dls=[train_dl, valid_dl],    # DataLoaders
    model=model,                  # PyTorch model
    loss_func=nn.MSELoss(),      # Loss function
    opt_func=partial(torch.optim.SGD, momentum=0.9),  # Optimizer
    lr=0.01,                     # Learning rate
    metrics=[Accuracy()],        # Metrics to track
    cbs=[ProgressCallback()]     # Callbacks
)
```

### Training

```python
# Basic training
learn.fit(n_epoch=10)

# With different learning rate
learn.fit(n_epoch=10, lr=0.001)

# With temporary callbacks
from learner import EarlyStoppingCallback
learn.fit(n_epoch=20, cbs=[EarlyStoppingCallback(patience=3)])
```

### Making Predictions

```python
# Single prediction
pred = learn.predict(torch.randn(1, 784))

# Batch predictions
preds, targets = learn.get_preds()  # On validation set
preds = learn.get_preds(dl=test_dl)  # On specific dataloader
```

### Saving and Loading

```python
# Save model
learn.save('my_model')

# Load model
learn.load('my_model')
```

## Advanced Features

### Learning Rate Finder

Find the optimal learning rate automatically:

```python
# Run learning rate finder
lr_finder = learn.lr_find(start_lr=1e-7, end_lr=1, num_it=100)
# This will plot a graph showing loss vs learning rate
```

The learning rate finder trains your model with exponentially increasing learning rates to find the optimal value. Look for the steepest downward slope in the plot. See [Understanding Key Features](#understanding-key-features) for detailed explanation.

### One-Cycle Training

Use the 1cycle learning rate policy for faster convergence:

```python
from learner import OneCycleLR

learn.fit(n_epoch=20, lr=0.01, cbs=[OneCycleLR(max_lr=0.01)])
```

This implements Leslie Smith's one-cycle policy which can train models 5-10x faster. The learning rate increases then decreases, while momentum does the opposite. See [Understanding Key Features](#understanding-key-features) for why this works.

### Mixed Precision Training

Train with 16-bit precision for 2-3x faster training:

```python
from learner import MixedPrecision

learn.fit(n_epoch=10, cbs=[MixedPrecision()])
```

Mixed precision uses FP16 computation with FP32 master weights, providing significant speedup on modern GPUs while maintaining accuracy. See [Understanding Key Features](#understanding-key-features) for technical details.

### Gradient Clipping

Prevent gradient explosion in deep or recurrent networks:

```python
from learner import GradientClipping

learn.fit(n_epoch=10, cbs=[GradientClipping(max_norm=1.0)])
```

This clips gradient norms to prevent training instability. Essential for RNNs and very deep networks. See [Understanding Key Features](#understanding-key-features) for how it works.

## Callbacks

### Built-in Callbacks

1. **Recorder** (added by default)
   - Records losses, metrics, learning rates

2. **ProgressCallback**
   ```python
   ProgressCallback()  # Shows training progress
   ```

3. **EarlyStoppingCallback**
   ```python
   EarlyStoppingCallback(monitor='valid_loss', patience=3, mode='min')
   ```

4. **SaveModelCallback**
   ```python
   SaveModelCallback(monitor='valid_loss', name='best_model')
   ```

5. **CSVLogger**
   ```python
   CSVLogger(filename='training_history.csv')
   ```

6. **ActivationStats**
   ```python
   # Monitor activations during training
   act_cb = ActivationStats(every=100)
   learn.fit(10, cbs=[act_cb])
   act_cb.plot_stats()
   ```

7. **GradientStats**
   ```python
   # Monitor gradients during training
   grad_cb = GradientStats(every=100)
   learn.fit(10, cbs=[grad_cb])
   grad_cb.plot_stats()
   ```

### Creating Custom Callbacks

```python
from learner import Callback

class MyCallback(Callback):
    order = 20  # When to run (lower = earlier)
    
    def before_fit(self):
        print(f"Starting training for {self.n_epoch} epochs")
        
    def after_epoch(self):
        print(f"Epoch {self.epoch} complete!")
```

## Visualization

### Understanding Training Dynamics

Visualization is crucial for debugging and optimizing neural network training. Here's what each plot tells you:

### Plot Training History

```python
# Plot losses
learn.plot_losses()
```

**What to Look For:**
- **Decreasing training loss**: Model is learning
- **Gap between train/valid loss**: Indicates overfitting
- **Volatile loss**: Learning rate might be too high
- **Plateauing loss**: May need LR reduction or model capacity increase

```python
# Plot metrics
learn.plot_metrics()
```

**Interpreting Metrics:**
- Compare metric trends with loss trends
- Metrics plateauing while loss decreases suggests overfitting
- Sudden metric drops might indicate numerical instability

```python
# Plot learning rate schedule
learn.plot_lr()
```

**Why Monitor LR:**
- Verify your schedule is working correctly
- Correlate LR changes with loss behavior
- Debug convergence issues

### Model Summary

```python
# Print model architecture with parameter counts
learn.summary()
```

**Key Information:**
- Total parameters vs trainable parameters
- Memory requirements
- Layer output shapes (helps debug dimension mismatches)

### Activation Analysis

```python
from learner import ActivationStats

# Train with activation monitoring
act_stats = ActivationStats()
learn.fit(5, cbs=[act_stats])

# Plot activation statistics
act_stats.plot_stats()
```

**Diagnosing Issues:**
- **Mean ≈ 0, Std ≈ 1**: Well-behaved activations
- **Mean → 0, Std → 0**: Dying neurons (vanishing gradient)
- **Mean or Std exploding**: Activation explosion
- **Std very different across layers**: Poor initialization

### Gradient Flow Analysis

```python
from learner import GradientStats

# Monitor gradients
grad_stats = GradientStats()
learn.fit(5, cbs=[grad_stats])
grad_stats.plot_stats()
```

**What Healthy Gradients Look Like:**
- Similar magnitude across layers
- No layers with near-zero gradients
- Gradients should be ~100x smaller than weights
- Earlier layers shouldn't have vanishing gradients

## Model Interpretation

### For Classification Tasks

```python
from learner import ClassificationInterpretation

# Create interpretation object
interp = ClassificationInterpretation(learn)

# Plot confusion matrix
interp.plot_confusion_matrix(figsize=(8, 8))

# Show most confused predictions
interp.most_confused(min_val=5)

# Plot top losses (worst predictions)
interp.plot_top_losses(k=9)
```

## Examples

### Example 1: Image Classification

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32)

# Create model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 8 * 8, 10)
)

# Create learner with multiple callbacks
learn = Learner(
    dls=[trainloader, testloader],
    model=model,
    loss_func=nn.CrossEntropyLoss(),
    metrics=[Accuracy()],
    cbs=[
        ProgressCallback(),
        EarlyStoppingCallback(patience=5),
        SaveModelCallback(name='best_cifar10')
    ]
)

# Find learning rate
learn.lr_find()

# Train with one-cycle
learn.fit(20, lr=0.01, cbs=[OneCycleLR(max_lr=0.01)])

# Visualize results
learn.plot_losses()
learn.plot_metrics()

# Interpret
interp = ClassificationInterpretation(learn)
interp.plot_confusion_matrix()
```

### Example 2: Custom Metric and Callback

```python
from learner import Metric, Callback

# Custom F1 Score metric
class F1Score(Metric):
    def __init__(self):
        super().__init__()
        self.tp = self.fp = self.fn = 0
        
    def accumulate(self, learn):
        pred = learn.pred.argmax(dim=-1)
        y = learn.yb
        self.tp += ((pred == 1) & (y == 1)).sum().item()
        self.fp += ((pred == 1) & (y == 0)).sum().item()
        self.fn += ((pred == 0) & (y == 1)).sum().item()
        
    @property
    def value(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Custom learning rate warmup callback
class LRWarmup(Callback):
    def __init__(self, warmup_epochs=5):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        
    def before_fit(self):
        self.start_lr = self.learn.lr / self.warmup_epochs
        
    def before_batch(self):
        if self.epoch < self.warmup_epochs and self.train:
            # Linear warmup
            lr = self.start_lr + (self.learn.lr - self.start_lr) * (
                self.epoch * len(self.dl) + self.iter) / (self.warmup_epochs * len(self.dl))
            for pg in self.opt.param_groups:
                pg['lr'] = lr

# Use custom components
learn = Learner(
    dls=[train_dl, valid_dl],
    model=model,
    metrics=[Accuracy(), F1Score()],
    cbs=[ProgressCallback(), LRWarmup(warmup_epochs=3)]
)

learn.fit(10)
```

## Extending the Learner

### Creating Custom Callbacks

The callback system is designed for easy extension. Here's the pattern:

```python
class MyCallback(Callback):
    order = 10  # Lower numbers run first
    
    def __init__(self, param):
        super().__init__()
        self.param = param
    
    def before_fit(self):
        # Setup code
        self.state = []
    
    def after_batch(self):
        # Access any learner attribute
        if self.training:  # Only during training
            self.state.append(self.loss.item())
    
    def after_fit(self):
        # Cleanup or final processing
        self.plot_results()
```

**Key Points:**
- Override only the methods you need
- Access learner state via `self.` (delegated)
- Return `True` to skip remaining callbacks
- Raise `CancelXException` to skip phases

### Creating Custom Metrics

Metrics accumulate values over an epoch:

```python
class CustomMetric(Metric):
    def reset(self):
        # Called at start of epoch
        self.total = 0
        self.count = 0
    
    def accumulate(self, learn):
        # Called after each batch
        # Access: learn.pred, learn.yb, learn.loss
        self.total += compute_something(learn.pred, learn.yb)
        self.count += learn.yb.size(0)
    
    @property
    def value(self):
        # Return final metric value
        return self.total / self.count
```

### Advanced Callback Patterns

**1. State Machine Callback:**
```python
class StateMachineCallback(Callback):
    def __init__(self):
        super().__init__()
        self.state = 'init'
    
    def after_loss(self):
        if self.state == 'init' and self.loss < 0.5:
            self.state = 'low_loss'
            self.learn.lr *= 0.1  # Reduce LR
```

**2. Conditional Callback:**
```python
class ConditionalCallback(Callback):
    def __init__(self, condition_fn):
        super().__init__()
        self.condition_fn = condition_fn
    
    def after_epoch(self):
        if self.condition_fn(self.learn):
            # Do something
            pass
```

**3. Composite Callback:**
```python
class CompositeCallback(Callback):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks
    
    def before_fit(self):
        for cb in self.callbacks:
            cb.learn = self.learn
            cb.before_fit()
```

## Tips and Best Practices

### Training Strategy

1. **Start Simple, Then Complexify**
   - Begin with a small model that can overfit
   - Gradually add complexity (layers, regularization)
   - Ensure each component works before adding more

2. **Learning Rate is Key**
   - Always run learning rate finder first
   - When in doubt, use a lower learning rate
   - Consider one-cycle training for faster convergence

3. **Monitor Everything**
   ```python
   learn = Learner(
       dls=[train_dl, valid_dl],
       model=model,
       cbs=[
           ProgressCallback(),
           ActivationStats(),      # Watch for dying neurons
           GradientStats(),        # Detect vanishing/exploding gradients
           CSVLogger(),           # Keep training history
           SaveModelCallback()    # Don't lose good models
       ]
   )
   ```

4. **Validation Strategy**
   - Always use a validation set (even if small)
   - Watch the gap between train/valid metrics
   - Use early stopping to prevent overfitting

5. **Debugging Workflow**
   - Can you overfit a single batch? (sanity check)
   - Can you overfit the full training set? (capacity check)
   - Does validation loss improve? (generalization check)

### Performance Optimization

1. **GPU Utilization**
   - Maximize batch size (memory permitting)
   - Use mixed precision on modern GPUs
   - Profile to find bottlenecks

2. **Data Pipeline**
   ```python
   DataLoader(
       dataset,
       batch_size=64,
       num_workers=4,          # Parallel data loading
       pin_memory=True,        # Faster GPU transfer
       persistent_workers=True # Keep workers alive
   )
   ```

3. **Callback Order Matters**
   - Callbacks execute in order of their `order` attribute
   - Put measurement callbacks early
   - Put modification callbacks in correct sequence

### Model Design Principles

1. **Initialization Matters**
   - Poor initialization → vanishing/exploding gradients
   - Use appropriate initialization for your activation
   - Monitor initial forward pass statistics

2. **Normalization Helps**
   - Batch norm for CNNs
   - Layer norm for transformers
   - Helps with deeper networks

3. **Residual Connections**
   - Enable training very deep networks
   - Help gradient flow
   - Often improve performance

### Common Patterns

```python
# Standard classification setup
learn = Learner(
    dls=[train_dl, valid_dl],
    model=model,
    loss_func=nn.CrossEntropyLoss(),
    metrics=[Accuracy()],
    cbs=[
        ProgressCallback(),
        OneCycleLR(max_lr=1e-2),
        MixedPrecision(),
        EarlyStoppingCallback(patience=5),
        SaveModelCallback('best_model')
    ]
)

# Find best LR
lr_finder = learn.lr_find()

# Train
learn.fit(30)

# Analyze results
learn.plot_losses()
interp = ClassificationInterpretation(learn)
interp.plot_confusion_matrix()
```

## Troubleshooting

### CUDA Out of Memory
**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. **Reduce batch size** (most effective)
   ```python
   # Halve your batch size
   train_dl = DataLoader(dataset, batch_size=16)  # was 32
   ```

2. **Use gradient accumulation**
   ```python
   # Accumulate gradients over multiple steps
   class GradientAccumulation(Callback):
       def __init__(self, n_acc=4):
           self.n_acc = n_acc
       
       def before_fit(self):
           self.count = 0
           
       def after_backward(self):
           self.count += 1
           if self.count % self.n_acc != 0:
               self.learn.cancel_step = True
   ```

3. **Enable mixed precision** (saves ~50% memory)
   ```python
   learn.fit(n_epoch=10, cbs=[MixedPrecision()])
   ```

4. **Gradient checkpointing** for very deep models

### Training Not Converging

**Symptoms:** Loss not decreasing, metrics not improving

**Diagnostic Steps:**

1. **Check learning rate**
   ```python
   # Too high: loss explodes or oscillates
   # Too low: loss decreases very slowly
   lr_finder = learn.lr_find()
   ```

2. **Verify data preprocessing**
   ```python
   # Check a batch
   xb, yb = next(iter(learn.train_dl))
   print(f"Input shape: {xb.shape}, range: [{xb.min()}, {xb.max()}]")
   print(f"Target shape: {yb.shape}, unique values: {yb.unique()}")
   ```

3. **Monitor gradients**
   ```python
   # Look for vanishing or exploding gradients
   grad_stats = GradientStats()
   learn.fit(1, cbs=[grad_stats])
   grad_stats.plot_stats()
   ```

4. **Check for numerical instability**
   - Add small epsilon to denominators
   - Use log-sum-exp trick for stability
   - Check for NaN/Inf in activations

5. **Simplify the problem**
   ```python
   # Can you overfit a single batch?
   single_batch = next(iter(train_dl))
   single_dl = DataLoader([single_batch], batch_size=1)
   learn.fit(100, lr=0.1)  # Should get near-zero loss
   ```

### Slow Training

**Symptoms:** Each epoch takes too long

**Profile your training:**
```python
import time

class TimingCallback(Callback):
    def before_fit(self):
        self.times = {'data': 0, 'forward': 0, 'backward': 0, 'step': 0}
    
    def before_batch(self):
        self.t = time.time()
    
    def after_pred(self):
        self.times['data'] += time.time() - self.t
        self.t = time.time()
    
    def after_loss(self):
        self.times['forward'] += time.time() - self.t
        self.t = time.time()
    
    def after_backward(self):
        self.times['backward'] += time.time() - self.t
        self.t = time.time()
    
    def after_step(self):
        self.times['step'] += time.time() - self.t
```

**Common Solutions:**
- **Data loading bottleneck**: Increase `num_workers` in DataLoader
- **Small batch size**: GPU underutilized
- **CPU-GPU transfer**: Keep data on GPU, use pinned memory
- **Inefficient model**: Profile with PyTorch profiler

### Overfitting

**Symptoms:** Train loss decreases but validation loss increases

**Solutions:**
1. **Regularization callbacks**
   ```python
   class DropoutCallback(Callback):
       def __init__(self, p=0.5):
           self.p = p
       
       def before_batch(self):
           if self.training:
               # Apply dropout to activations
               pass
   ```

2. **Data augmentation**
3. **Early stopping**
4. **Reduce model capacity**
5. **Increase training data**

### Gradient Problems

**Vanishing Gradients:**
- Use ReLU instead of sigmoid/tanh
- Batch normalization
- Residual connections
- Better initialization (Xavier/He)

**Exploding Gradients:**
- Gradient clipping
- Smaller learning rate
- Batch normalization
- Check for numerical overflow

## Quick Reference

### Available Callbacks
- `Recorder` - Records training statistics (added by default)
- `ProgressCallback` - Shows training progress
- `OneCycleLR` - One-cycle learning rate schedule
- `MixedPrecision` - FP16 training
- `GradientClipping` - Clip gradient norms
- `EarlyStoppingCallback` - Stop when metric stops improving
- `SaveModelCallback` - Save best model checkpoint
- `ActivationStats` - Monitor activation statistics
- `GradientStats` - Monitor gradient statistics
- `CSVLogger` - Log metrics to CSV
- `WandbCallback` - Structured logging for visualization

### Built-in Metrics
- `Accuracy` - Classification accuracy
- `TopKAccuracy` - Top-K classification accuracy

### Key Methods
- `learn.fit(n_epoch, lr)` - Train model
- `learn.lr_find()` - Find optimal learning rate
- `learn.predict(x)` - Make prediction
- `learn.get_preds()` - Get all predictions
- `learn.save/load(name)` - Save/load model
- `learn.plot_losses()` - Visualize training
- `learn.summary()` - Model architecture summary

### Interpretation (Classification)
- `interp.plot_confusion_matrix()` - Confusion matrix
- `interp.plot_top_losses()` - Worst predictions
- `interp.most_confused()` - Most confused classes

## Credit

To Jeremy Howard 🙏🏼  [@jeremyphoward](https://twitter.com/jeremyphoward) | [@jph00](https://github.com/jph00)