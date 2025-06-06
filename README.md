### Bisection Projection

#### Non-convex Toy example

1. Case 1: $\mathcal{K}_{\omega}=\{x\mid x^{\top}Qx +q^{\top}x + b\le 0 \}$, where $\omega=\{Q,q,b\}$.
2. Case 2: $\mathcal{C}_{\theta}=\cup_{i=1}^4 \mathcal{B}(x_i,r_i)$, where $\theta=\{x_i,r_i\}_{i=1}^4$.

Run BP_toy_example.py to visualize MEIPs predictions and BP trajectory with MEIPs.

#### Constraint Optimization

1. Convex problems: QCQP, SOCP.
2. Non-convex problems: ACOPF, JCCIM.

Run training_all.py to train NN predictor, IPNN for MEIPs and BP for infeasible predictions.






