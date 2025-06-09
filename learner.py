import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from functools import partial, wraps
from collections import defaultdict, OrderedDict
import pandas as pd
from typing import Optional, Callable, List, Tuple, Dict, Any, Union
import warnings
from contextlib import contextmanager
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
from pathlib import Path
import json

# Base Callback System
class Callback:
    """Base callback class"""
    order = 0  # Callbacks are sorted by order
    
    def __init__(self): 
        self.learn = None
        
    def __repr__(self): 
        return f'{self.__class__.__name__}'
    
    def __getattr__(self, name):
        # Delegate attribute access to learner
        if name.startswith('_'): raise AttributeError(name)
        return getattr(self.learn, name)
    
    # Callback methods - return True to skip rest of callbacks
    def before_fit(self): pass
    def before_epoch(self): pass
    def before_train(self): pass
    def before_batch(self): pass
    def after_pred(self): pass
    def after_loss(self): pass
    def before_backward(self): pass
    def after_backward(self): pass
    def before_step(self): pass
    def after_step(self): pass
    def after_batch(self): pass
    def after_train(self): pass
    def before_validate(self): pass
    def after_validate(self): pass
    def after_epoch(self): pass
    def after_fit(self): pass
    
    # Special methods
    def cleanup_fit(self): pass
    def skip_batch(self): return False
    def skip_epoch(self): return False
    def skip_train(self): return False
    def skip_validate(self): return False


class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass
class CancelTrainException(Exception): pass
class CancelValidException(Exception): pass
class CancelFitException(Exception): pass


# Core Learner Class
class Learner:
    """Central training abstraction combining model, data, loss, optimizer, and callbacks"""
    
    def __init__(self, dls, model, loss_func=None, opt_func=None, lr=1e-3, cbs=None, 
                 metrics=None, path='.', model_dir='models', wd=None, wd_bn_bias=False,
                 train_bn=True, moms=(0.95, 0.85, 0.95)):
        # Store data
        self.dls = dls if isinstance(dls, (list, tuple)) else [dls]
        self.train_dl, self.valid_dl = self.dls[0], self.dls[1] if len(self.dls) > 1 else None
        
        # Model and training components
        self.model = model
        self.loss_func = loss_func or nn.CrossEntropyLoss()
        self.opt_func = opt_func or partial(torch.optim.Adam, lr=lr)
        self.lr = lr
        self.wd = wd or 0.
        self.wd_bn_bias = wd_bn_bias
        self.train_bn = train_bn
        self.moms = moms
        
        # Paths
        self.path = Path(path)
        self.model_dir = self.path/model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Callbacks and metrics
        self.cbs = [] if cbs is None else list(cbs)
        self.metrics = [] if metrics is None else list(metrics)
        
        # Add recorder callback by default
        self.add_cb(Recorder())
        
        # State variables
        self.epoch = 0
        self.n_epoch = 1
        self.loss = None
        self.opt = None
        self.train = True
        self.cancel_train = False
        self.cancel_valid = False
        self.cancel_epoch = False
        self.cancel_fit = False
        
        # Get device
        self.device = next(model.parameters()).device
        
        # Hooks storage
        self._hooks = []
        
    def add_cb(self, cb):
        """Add callback to learner"""
        cb.learn = self
        self.cbs.append(cb)
        self.cbs = sorted(self.cbs, key=lambda x: x.order)
        
    def remove_cb(self, cb_type):
        """Remove callbacks of given type"""
        self.cbs = [cb for cb in self.cbs if not isinstance(cb, cb_type)]
        
    def __call__(self, name):
        """Call method on all callbacks"""
        for cb in self.cbs:
            method = getattr(cb, name, None)
            if method is not None:
                if method(): return True
        return False
    
    @contextmanager
    def no_logging(self):
        """Context manager to disable logging"""
        # Remove recorder temporarily
        recorder = self.recorder
        self.remove_cb(Recorder)
        yield
        self.add_cb(recorder)
    
    @contextmanager
    def no_bar(self):
        """Context manager to disable progress bar"""
        progress = None
        for cb in self.cbs:
            if isinstance(cb, ProgressCallback):
                progress = cb
                break
        if progress:
            self.remove_cb(ProgressCallback)
        yield
        if progress:
            self.add_cb(progress)
    
    @property
    def recorder(self):
        """Get recorder callback"""
        for cb in self.cbs:
            if isinstance(cb, Recorder):
                return cb
        return None
    
    def _split_kwargs(self, kwargs):
        """Split kwargs between optimizer and step function"""
        opt_kwargs = {}
        step_kwargs = {}
        for k, v in kwargs.items():
            if k in ['lr', 'mom', 'beta', 'wd']:
                step_kwargs[k] = v
            else:
                opt_kwargs[k] = v
        return opt_kwargs, step_kwargs
    
    def create_opt(self):
        """Create optimizer"""
        self.opt = self.opt_func(self.model.parameters())
        
    def one_batch(self):
        """Train or evaluate on one batch"""
        self.pred = self.model(self.xb)
        self('after_pred')
        if self.train:
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.cancel_backward:
                self.loss.backward()
                self('after_backward')
                if not self.cancel_step:
                    self.opt.step()
                    self('after_step')
                    self.opt.zero_grad()
                    
    def all_batches(self):
        """Train or evaluate on all batches"""
        self.n_iter = len(self.dl)
        for self.iter, self.batch in enumerate(self.dl):
            if self('skip_batch'): continue
            try:
                self._split_batch()
                self('before_batch')
                self.cancel_backward = False
                self.cancel_step = False
                self.one_batch()
                self('after_batch')
            except CancelBatchException: pass
            
    def _split_batch(self):
        """Split batch into x and y"""
        self.batch = tuple(t.to(self.device) for t in self.batch)
        self.xb, self.yb = self.batch[0], self.batch[1]
        
    def one_epoch(self, train=True):
        """Run one epoch of training or validation"""
        self.model.train(train)
        self.train = train
        self.dl = self.train_dl if train else self.valid_dl
        if not self.dl: return
        
        try:
            self(f'before_{"train" if train else "validate"}')
            if not self(f'skip_{"train" if train else "validate"}'):
                self.all_batches()
            self(f'after_{"train" if train else "validate"}')
        except (CancelTrainException, CancelValidException): pass
        
    def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):
        """Main training loop"""
        # Add temporary callbacks
        if cbs: 
            for cb in cbs: self.add_cb(cb)
            
        self.n_epoch = n_epoch
        self.epoch = start_epoch
        if lr: self.lr = lr
        if wd is not None: self.wd = wd
        
        try:
            self('before_fit')
            if reset_opt or self.opt is None: self.create_opt()
            
            for self.epoch in range(start_epoch, n_epoch):
                try:
                    self('before_epoch')
                    if not self('skip_epoch'):
                        self.one_epoch(True)
                        with torch.no_grad():
                            self.one_epoch(False)
                    self('after_epoch')
                except CancelEpochException: pass
                
            self('after_fit')
        except CancelFitException: pass
        finally:
            # Cleanup
            self('cleanup_fit')
            # Remove temporary callbacks
            if cbs:
                for cb in cbs: 
                    self.cbs.remove(cb)
                    
    # Learning Rate Finder
    def lr_find(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, show_plot=True):
        """Learning rate finder"""
        # Save current state
        state = self.model.state_dict()
        opt_state = self.opt.state_dict() if self.opt else None
        
        # Use special LR finder callback
        lr_finder = LRFinder(start_lr, end_lr, num_it, stop_div)
        with self.no_logging():
            self.fit(1, cbs=[lr_finder])
            
        # Restore state
        self.model.load_state_dict(state)
        if opt_state and self.opt:
            self.opt.load_state_dict(opt_state)
            
        if show_plot:
            lr_finder.plot()
            
        return lr_finder
    
    # Prediction methods
    @torch.no_grad()
    def predict(self, x):
        """Predict on a single item"""
        self.model.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(self.device)
        if len(x.shape) == len(self.dls[0].dataset[0][0].shape):
            x = x.unsqueeze(0)
        return self.model(x)
    
    @torch.no_grad() 
    def get_preds(self, dl=None, with_input=False, with_target=False, with_decoded=False):
        """Get predictions on a dataloader"""
        if dl is None: dl = self.valid_dl
        self.model.eval()
        
        preds, targets, inputs = [], [], []
        for batch in dl:
            batch = tuple(t.to(self.device) for t in batch)
            xb, yb = batch[0], batch[1]
            pred = self.model(xb)
            preds.append(pred.cpu())
            if with_target: targets.append(yb.cpu())
            if with_input: inputs.append(xb.cpu())
            
        preds = torch.cat(preds)
        res = [preds]
        if with_target: res.append(torch.cat(targets))
        if with_input: res.append(torch.cat(inputs))
        if with_decoded: res = [o.argmax(dim=-1) if len(o.shape) > 1 else o for o in res]
        
        return res[0] if len(res) == 1 else tuple(res)
    
    # Save/Load
    def save(self, name, with_opt=True):
        """Save model and optionally optimizer"""
        state = {'model': self.model.state_dict()}
        if with_opt and self.opt:
            state['opt'] = self.opt.state_dict()
        torch.save(state, self.model_dir/f'{name}.pth')
        
    def load(self, name, with_opt=True):
        """Load model and optionally optimizer"""
        state = torch.load(self.model_dir/f'{name}.pth')
        self.model.load_state_dict(state['model'])
        if with_opt and self.opt and 'opt' in state:
            self.opt.load_state_dict(state['opt'])
            
    # Hook management
    def register_hook(self, hook_func, module=None, on='forward'):
        """Register a hook on module"""
        if module is None: module = self.model
        if on == 'forward':
            handle = module.register_forward_hook(hook_func)
        else:
            handle = module.register_backward_hook(hook_func)
        self._hooks.append(handle)
        return handle
    
    def remove_hooks(self):
        """Remove all hooks"""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
    # Visualization and interpretation
    def plot_losses(self, skip_start=5, skip_end=0):
        """Plot training and validation losses"""
        return self.recorder.plot_losses(skip_start, skip_end)
    
    def plot_metrics(self, skip_start=5, skip_end=0):
        """Plot metrics"""
        return self.recorder.plot_metrics(skip_start, skip_end)
    
    def plot_lr(self):
        """Plot learning rate schedule"""
        return self.recorder.plot_lr()
    
    def summary(self):
        """Print model summary"""
        return model_summary(self.model, self.dls[0])


# Essential Callbacks
class Recorder(Callback):
    """Records losses, metrics, and other statistics during training"""
    order = 10  # Run early
    
    def __init__(self):
        super().__init__()
        self.reset()
        
    def reset(self):
        self.lrs = []
        self.losses = []
        self.val_losses = []
        self.metrics = []
        self.epochs = []
        self.train_losses = []
        self.iters = []
        self.nb_batches = []
        
    def before_fit(self):
        self.reset()
        
    def before_epoch(self):
        self.epoch_losses = []
        self.epoch_start_time = time.time()
        
    def after_batch(self):
        if self.train:
            self.lrs.append(self.opt.param_groups[0]['lr'])
            self.losses.append(self.loss.item())
            self.epoch_losses.append(self.loss.item())
            self.iters.append(self.epoch * self.n_iter + self.iter)
            
    def after_epoch(self):
        # Record epoch stats
        self.epochs.append(self.epoch)
        if self.epoch_losses:
            self.train_losses.append(np.mean(self.epoch_losses))
        
        # Validation loss
        if not self.train and hasattr(self, 'loss') and self.loss is not None:
            self.val_losses.append(self.loss.item())
            
        # Metrics
        if self.metrics:
            self.metrics.append([m.value for m in self.learn.metrics])
            
        self.epoch_time = time.time() - self.epoch_start_time
        
    def plot_losses(self, skip_start=5, skip_end=0):
        """Plot training and validation losses"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Training losses
        train_losses = self.losses[skip_start:len(self.losses)-skip_end]
        iters = self.iters[skip_start:len(self.iters)-skip_end] 
        ax.plot(iters, train_losses, label='Train Loss', alpha=0.7)
        
        # Epoch average losses
        if self.train_losses:
            epoch_iters = [e * self.n_iter for e in self.epochs]
            ax.plot(epoch_iters[:len(self.train_losses)], self.train_losses, 
                   'r-', label='Train Loss (Avg)', linewidth=2)
        
        # Validation losses  
        if self.val_losses:
            val_iters = [(e+1) * self.n_iter for e in self.epochs[:len(self.val_losses)]]
            ax.plot(val_iters, self.val_losses, 'g-', label='Valid Loss', linewidth=2)
            
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Losses')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_metrics(self, skip_start=0, skip_end=0):
        """Plot metrics"""
        if not self.metrics:
            print("No metrics to plot")
            return
            
        metrics = np.array(self.metrics)
        epochs = self.epochs[:len(metrics)]
        
        n_metrics = metrics.shape[1]
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics), squeeze=False)
        
        for i, (ax, metric) in enumerate(zip(axes.flat, self.learn.metrics)):
            vals = metrics[skip_start:len(metrics)-skip_end, i]
            eps = epochs[skip_start:len(epochs)-skip_end]
            ax.plot(eps, vals, 'o-', label=metric.__class__.__name__)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.__class__.__name__)
            ax.legend()
            ax.grid(True)
            
        plt.tight_layout()
        plt.show()
        return fig
        
    def plot_lr(self):
        """Plot learning rate over iterations"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.iters, self.lrs)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        return fig


class ProgressCallback(Callback):
    """Shows training progress with bar and metrics"""
    order = 15
    
    def __init__(self, plot_losses=False):
        super().__init__()
        self.plot_losses = plot_losses
        
    def before_fit(self):
        self.epochs = self.learn.n_epoch
        print(f"Epoch\tTrain Loss\tValid Loss\tTime")
        
    def before_epoch(self):
        self.start_time = time.time()
        
    def after_epoch(self):
        # Gather stats
        train_loss = self.learn.recorder.train_losses[-1] if self.learn.recorder.train_losses else float('nan')
        valid_loss = self.learn.recorder.val_losses[-1] if self.learn.recorder.val_losses else float('nan')
        elapsed = time.time() - self.start_time
        
        # Print stats
        stats = f"{self.epoch+1}/{self.epochs}\t{train_loss:.4f}\t\t{valid_loss:.4f}\t\t{elapsed:.2f}s"
        
        # Add metrics
        if self.learn.metrics and self.learn.recorder.metrics:
            for metric, value in zip(self.learn.metrics, self.learn.recorder.metrics[-1]):
                stats += f"\t{value:.4f}"
                
        print(stats)


# Learning Rate Finder
class LRFinder(Callback):
    """Callback for learning rate finder"""
    order = 1
    
    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        super().__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.num_it, self.stop_div = num_it, stop_div
        
    def before_fit(self):
        self.lrs, self.losses = [], []
        self.best_loss = float('inf')
        self.scheduler = ExponentialLR(self.learn.opt, self.start_lr, self.end_lr, self.num_it)
        
    def before_batch(self):
        if not self.train: raise CancelValidException()
        
    def after_batch(self):
        # Record
        self.lrs.append(self.scheduler.get_lr())
        loss = self.smooth_loss = self.loss.item()
        self.losses.append(loss)
        
        # Stop if loss exploded
        if self.stop_div and len(self.losses) > 3:
            if loss > self.best_loss * 4:
                raise CancelFitException()
        if loss < self.best_loss: self.best_loss = loss
        
        # Step scheduler
        self.scheduler.step()
        
        # Stop after num_it
        if self.iter >= self.num_it:
            raise CancelFitException()
            
    def plot(self):
        """Plot learning rate vs loss"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.lrs, self.losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Rate Finder')
        ax.grid(True)
        
        # Find and plot suggested LR (steepest slope)
        if len(self.losses) > 10:
            # Simple heuristic: point of steepest descent
            gradients = np.gradient(self.losses)
            min_grad_idx = np.argmin(gradients[5:-5]) + 5  # Skip very start/end
            suggested_lr = self.lrs[min_grad_idx]
            ax.axvline(x=suggested_lr, color='red', linestyle='--', 
                      label=f'Suggested LR: {suggested_lr:.2e}')
            ax.legend()
            
        plt.tight_layout()
        plt.show()
        return fig


# Schedulers
class ExponentialLR:
    """Exponential learning rate scheduler"""
    def __init__(self, optimizer, start_lr, end_lr, num_it):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.current_it = 0
        
    def get_lr(self):
        return self.start_lr * (self.end_lr / self.start_lr) ** (self.current_it / self.num_it)
    
    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_it += 1


# Metrics
class Metric:
    """Base class for metrics"""
    def __init__(self): self.reset()
    def reset(self): self.total = self.count = 0
    def accumulate(self, learn): pass
    @property
    def value(self): return self.total / self.count if self.count != 0 else None
    @property 
    def name(self): return self.__class__.__name__


class Accuracy(Metric):
    """Classification accuracy metric"""
    def accumulate(self, learn):
        pred = learn.pred.argmax(dim=-1)
        self.total += (pred == learn.yb).float().sum().item()
        self.count += learn.yb.numel()


class TopKAccuracy(Metric):
    """Top-K classification accuracy"""
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        
    def accumulate(self, learn):
        pred = learn.pred.topk(self.k, dim=-1)[1]
        targs = learn.yb.unsqueeze(1).expand_as(pred)
        self.total += (pred == targs).float().sum().item()
        self.count += learn.yb.numel()


# Model Introspection and Visualization
class ActivationStats(Callback):
    """Records activation statistics during training"""
    order = 20
    
    def __init__(self, modules=None, every=100):
        super().__init__()
        self.modules = modules
        self.every = every
        self.stats = defaultdict(lambda: {'mean': [], 'std': [], 'hist': []})
        
    def before_fit(self):
        # Register hooks
        self.hooks = []
        modules = self.modules or [m for m in self.model.modules() 
                                  if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))]
        
        for i, module in enumerate(modules):
            hook = self._make_hook(f'layer_{i}_{module.__class__.__name__}')
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)
            
    def _make_hook(self, name):
        def hook(module, input, output):
            if self.iter % self.every == 0:
                self.stats[name]['mean'].append(output.detach().mean().item())
                self.stats[name]['std'].append(output.detach().std().item())
        return hook
    
    def after_fit(self):
        # Remove hooks
        for h in self.hooks:
            h.remove()
            
    def plot_stats(self):
        """Plot activation statistics"""
        n_layers = len(self.stats)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for name, stats in self.stats.items():
            iters = list(range(0, len(stats['mean']) * self.every, self.every))
            axes[0].plot(iters, stats['mean'], label=name)
            axes[1].plot(iters, stats['std'], label=name)
            
        axes[0].set_title('Activation Means')
        axes[0].set_ylabel('Mean')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True)
        
        axes[1].set_title('Activation Standard Deviations')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Std Dev')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig


class GradientStats(Callback):
    """Records gradient statistics during training"""
    order = 20
    
    def __init__(self, modules=None, every=100):
        super().__init__()
        self.modules = modules
        self.every = every
        self.stats = defaultdict(lambda: {'mean': [], 'std': []})
        
    def before_fit(self):
        self.hooks = []
        modules = self.modules or [m for m in self.model.modules() 
                                  if isinstance(m, (nn.Conv2d, nn.Linear))]
        
        for i, module in enumerate(modules):
            if hasattr(module, 'weight') and module.weight.requires_grad:
                hook = self._make_hook(f'layer_{i}_{module.__class__.__name__}', module)
                handle = module.register_backward_hook(hook)
                self.hooks.append(handle)
                
    def _make_hook(self, name, module):
        def hook(module, grad_input, grad_output):
            if self.iter % self.every == 0 and hasattr(module, 'weight'):
                grad = module.weight.grad
                if grad is not None:
                    self.stats[name]['mean'].append(grad.detach().mean().item())
                    self.stats[name]['std'].append(grad.detach().std().item())
        return hook
    
    def after_fit(self):
        for h in self.hooks:
            h.remove()
            
    def plot_stats(self):
        """Plot gradient statistics"""
        n_layers = len(self.stats)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        for name, stats in self.stats.items():
            if stats['mean']:  # Only plot if we have data
                iters = list(range(0, len(stats['mean']) * self.every, self.every))
                axes[0].plot(iters, stats['mean'], label=name)
                axes[1].plot(iters, stats['std'], label=name) 
                
        axes[0].set_title('Gradient Means')
        axes[0].set_ylabel('Mean')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True)
        
        axes[1].set_title('Gradient Standard Deviations')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Std Dev')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig


# Model Interpretation
class ClassificationInterpretation:
    """Interpretation methods for classification models"""
    
    def __init__(self, learn):
        self.learn = learn
        self.preds, self.targs = learn.get_preds(with_decoded=True, with_target=True)
        
    def plot_confusion_matrix(self, figsize=(8, 8), normalize=False):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.targs, self.preds)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_top_losses(self, k=9, figsize=(12, 8)):
        """Plot items with highest loss"""
        # Get losses for each prediction
        losses = []
        dl = self.learn.valid_dl
        self.learn.model.eval()
        
        with torch.no_grad():
            for batch in dl:
                xb, yb = batch[0].to(self.learn.device), batch[1].to(self.learn.device)
                preds = self.learn.model(xb)
                loss = F.cross_entropy(preds, yb, reduction='none')
                losses.extend(loss.cpu().numpy())
                
        losses = np.array(losses)
        idxs = np.argsort(losses)[-k:]  # Get indices of highest losses
        
        # Plot
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        for i, (idx, ax) in enumerate(zip(idxs, axes.flat)):
            # Get the item from dataset
            x, y = dl.dataset[idx]
            pred = self.preds[idx]
            loss_val = losses[idx]
            
            # Plot image if it's image data
            if len(x.shape) == 3:  # Assuming CHW format
                ax.imshow(x.permute(1, 2, 0))
            ax.set_title(f'Loss: {loss_val:.3f}\nActual: {y}, Pred: {pred}')
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        return fig
    
    def most_confused(self, min_val=1):
        """Show most confused class pairs"""
        cm = confusion_matrix(self.targs, self.preds)
        np.fill_diagonal(cm, 0)  # Remove correct predictions
        
        # Get most confused pairs
        confused = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] >= min_val:
                    confused.append((cm[i, j], i, j))
                    
        confused.sort(reverse=True, key=lambda x: x[0])
        
        print("Most confused pairs (count, actual, predicted):")
        for count, actual, pred in confused[:10]:
            print(f"{count:4d}  {actual} -> {pred}")
            
        return confused


# Additional callbacks
class EarlyStoppingCallback(Callback):
    """Stop training when monitored metric stops improving"""
    order = 30
    
    def __init__(self, monitor='valid_loss', patience=3, min_delta=0, mode='min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.num_bad_epochs = 0
        
    def after_epoch(self):
        # Get current value
        if self.monitor == 'valid_loss':
            current = self.learn.recorder.val_losses[-1] if self.learn.recorder.val_losses else None
        else:
            # Assume it's a metric
            idx = None
            for i, m in enumerate(self.learn.metrics):
                if m.__class__.__name__ == self.monitor:
                    idx = i
                    break
            if idx is not None and self.learn.recorder.metrics:
                current = self.learn.recorder.metrics[-1][idx]
            else:
                current = None
                
        if current is None: return
        
        # Check if improved
        if self.best is None:
            self.best = current
            self.num_bad_epochs = 0
        else:
            if self.mode == 'min':
                improved = current < self.best - self.min_delta
            else:
                improved = current > self.best + self.min_delta
                
            if improved:
                self.best = current
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                
        # Stop if no improvement
        if self.num_bad_epochs >= self.patience:
            print(f'Early stopping: {self.monitor} did not improve for {self.patience} epochs')
            raise CancelFitException()


class SaveModelCallback(Callback):
    """Save model when monitored metric improves"""
    order = 31
    
    def __init__(self, monitor='valid_loss', name='best_model', mode='min'):
        super().__init__()
        self.monitor = monitor
        self.name = name
        self.mode = mode
        self.best = None
        
    def after_epoch(self):
        # Get current value (similar to EarlyStoppingCallback)
        if self.monitor == 'valid_loss':
            current = self.learn.recorder.val_losses[-1] if self.learn.recorder.val_losses else None
        else:
            idx = None
            for i, m in enumerate(self.learn.metrics):
                if m.__class__.__name__ == self.monitor:
                    idx = i
                    break
            if idx is not None and self.learn.recorder.metrics:
                current = self.learn.recorder.metrics[-1][idx]
            else:
                current = None
                
        if current is None: return
        
        # Check if improved
        save = False
        if self.best is None:
            save = True
        else:
            if self.mode == 'min':
                save = current < self.best
            else:
                save = current > self.best
                
        if save:
            self.best = current
            self.learn.save(self.name)
            print(f'Better model found at epoch {self.epoch}, {self.monitor}={current:.4f}. Saved as {self.name}.pth')


class GradientClipping(Callback):
    """Clip gradients during training"""
    order = 11
    
    def __init__(self, max_norm=1.0):
        super().__init__()
        self.max_norm = max_norm
        
    def after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.max_norm)


class OneCycleLR(Callback):
    """One cycle learning rate schedule"""
    order = 1
    
    def __init__(self, max_lr, epochs=None, steps_per_epoch=None, pct_start=0.3,
                 anneal_strategy='cos', div_factor=25., final_div=1e4):
        super().__init__()
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div = final_div
        
    def before_fit(self):
        epochs = self.epochs or self.learn.n_epoch
        steps_per_epoch = self.steps_per_epoch or len(self.learn.train_dl)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.learn.opt,
            max_lr=self.max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,
            div_factor=self.div_factor,
            final_div=self.final_div
        )
        
    def after_batch(self):
        if self.train:
            self.scheduler.step()


# Utility functions
def model_summary(model, dl):
    """Print model summary with layer sizes"""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["output_shape"] = list(output.size())
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            
        if (not isinstance(module, nn.Sequential) and 
            not isinstance(module, nn.ModuleList) and 
            not (module == model)):
            hooks.append(module.register_forward_hook(hook))
            
    summary = OrderedDict()
    hooks = []
    
    model.apply(register_hook)
    
    # Make a forward pass
    x, _ = next(iter(dl))
    model(x.to(next(model.parameters()).device))
    
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Print summary
    print("-" * 80)
    print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
    print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for layer in summary:
        print(f"{layer:<25} {str(summary[layer]['output_shape']):<25} "
              f"{summary[layer]['nb_params']:<15,}")
        total_params += summary[layer]["nb_params"]
        
    # Count trainable params
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
            
    print("=" * 80)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print("-" * 80)
    
    return summary


# Mixed Precision Training
class MixedPrecision(Callback):
    """Mixed precision training using PyTorch's autocast"""
    order = 9
    
    def __init__(self):
        super().__init__()
        
    def before_fit(self):
        self.scaler = torch.cuda.amp.GradScaler()
        
    def before_batch(self):
        self.learn.pred = None  # Clear pred before autocast
        
    def after_pred(self):
        if self.train:
            with torch.cuda.amp.autocast():
                self.learn.pred = self.learn.model(self.learn.xb)
                
    def after_loss(self):
        if self.train:
            with torch.cuda.amp.autocast():
                self.learn.loss = self.learn.loss_func(self.learn.pred, self.learn.yb)
                
    def after_backward(self):
        self.scaler.scale(self.learn.loss).backward()
        
    def after_step(self):
        self.scaler.step(self.learn.opt)
        self.scaler.update()
        self.learn.opt.zero_grad()


# CSV Logger
class CSVLogger(Callback):
    """Log metrics to CSV file"""
    order = 40
    
    def __init__(self, filename='history.csv'):
        super().__init__()
        self.filename = filename
        
    def before_fit(self):
        self.file = open(self.learn.path/self.filename, 'w')
        self.writer = None
        
    def after_epoch(self):
        # Gather metrics
        metrics = OrderedDict()
        metrics['epoch'] = self.epoch
        metrics['train_loss'] = self.learn.recorder.train_losses[-1] if self.learn.recorder.train_losses else None
        metrics['valid_loss'] = self.learn.recorder.val_losses[-1] if self.learn.recorder.val_losses else None
        
        # Add custom metrics
        if self.learn.metrics and self.learn.recorder.metrics:
            for metric, value in zip(self.learn.metrics, self.learn.recorder.metrics[-1]):
                metrics[metric.__class__.__name__] = value
                
        # Write to CSV
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=metrics.keys())
            self.writer.writeheader()
            
        self.writer.writerow(metrics)
        self.file.flush()
        
    def after_fit(self):
        self.file.close()
        

# Weights & Biases style visualization
class WandbCallback(Callback):
    """Log metrics for visualization (similar to W&B)"""
    order = 50
    
    def __init__(self, project_name='fastai_run', log_model=False):
        super().__init__()
        self.project_name = project_name
        self.log_model = log_model
        self.run_data = defaultdict(list)
        
    def before_fit(self):
        self.run_path = self.learn.path/f'runs/{self.project_name}'
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime('%Y%m%d_%H%M%S')
        
    def after_batch(self):
        if self.train:
            self.run_data['loss'].append(self.loss.item())
            self.run_data['lr'].append(self.opt.param_groups[0]['lr'])
            self.run_data['iteration'].append(self.epoch * self.n_iter + self.iter)
            
    def after_epoch(self):
        # Log epoch metrics
        self.run_data['epoch'].append(self.epoch)
        self.run_data['train_loss_avg'].append(self.learn.recorder.train_losses[-1])
        
        if self.learn.recorder.val_losses:
            self.run_data['valid_loss'].append(self.learn.recorder.val_losses[-1])
            
        if self.learn.metrics and self.learn.recorder.metrics:
            for metric, value in zip(self.learn.metrics, self.learn.recorder.metrics[-1]):
                self.run_data[f'metric_{metric.__class__.__name__}'].append(value)
                
    def after_fit(self):
        # Save run data
        with open(self.run_path/f'{self.run_id}.json', 'w') as f:
            json.dump(dict(self.run_data), f)
            
        # Save model if requested
        if self.log_model:
            self.learn.save(f'{self.project_name}_{self.run_id}')