# Bisection Projection

A neural network-based approach for constrained optimization problems that trains networks to predict interior points with maximized margin to constraint boundaries, then uses bisection to project infeasible points onto the feasible region.

## Overview

The method consists of two main phases:
- **Training**: Train a neural network (IPNN) to predict interior points with maximized margin to the constrained boundary
- **Inference**: Use bisection between the predicted interior points and input infeasible points to find feasible projections

## Main Scripts

### 0. `0.BP_toy_example.py` - Toy Example Training
Trains a neural network on 2D toy constraint problems for demonstration and experimentation.

**Key features:**
- Uses simple 2D constraint sets (Complex_Constraints, Disconnected_Ball)
- Trains NoiseResNet model with unsupervised learning
- Configurable parameters for margin, learning rate, and network architecture
- Saves trained models and training records

**Usage:**
```bash
python 0.BP_toy_example.py
```

### 1. `1.Vis_BP_toy.ipynb` - Visualization Notebook  
Interactive Jupyter notebook for visualizing toy example results and bisection projection behavior.

**Visualizations include:**
- Training loss curves and penalty evolution
- Eccentricity distributions and feasibility rates  
- Bisection projection trajectories for different test inputs
- Sensitivity analysis for bisection step sizes

**Usage:**
```bash
jupyter notebook 1.Vis_BP_toy.ipynb
```

### 2. `2.BP_main.py` - Full Implementation
Main script handling real optimization problems with comprehensive training, testing, and analysis capabilities.

**Supported problem types:**
- Quadratic Programming (QP)
- Convex Quadratically Constrained Quadratic Programming (QCQP)  
- Second-Order Cone Programming (SOCP)
- Semidefinite Programming (SDP)
- AC Optimal Power Flow (ACOPF)
- Joint Chance-Constrained Inventory Management (JCCIM)

**Key functions:**
- `train_all()`: Train models on multiple problem types
- `test_all()`: Comprehensive testing and evaluation

**Usage:**
```bash
python 2.BP_main.py
```

## Quick Start

1. **Run toy example**: `python 0.BP_toy_example.py`
2. **Visualize results**: Open `1.Vis_BP_toy.ipynb` in Jupyter
3. **Train full models**: Modify `2.BP_main.py` and run for your specific problem type

## Dependencies

- PyTorch
- NumPy  
- Matplotlib
- Jupyter (for visualization notebook)

## Results

Training results, models, and visualizations are saved in the `results/` and `models/` directories, organized by problem type and configuration.

