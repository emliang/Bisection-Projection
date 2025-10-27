import matplotlib.pyplot as plt
import numpy as np

# Calculate the limits based on all data
x_min = min(min(nn_opt_gap_1*100), min(nn_opt_gap_2*100))
x_max = max(max(nn_opt_gap_1*100), max(nn_opt_gap_2*100))
y_min = min(min(bis_opt_gap_wo_gamma*100), min(bis_opt_gap_with_gamma*100))
y_max = max(max(bis_opt_gap_wo_gamma*100), max(bis_opt_gap_with_gamma*100))

# Add some padding to the limits
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05
x_min = max(0, x_min - x_padding)
x_max = x_max + x_padding
y_min = max(0, y_min - y_padding)
y_max = y_max + y_padding

plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(nn_opt_gap_1*100, nn_opt_gap_1*100, label='Initial NN solution', c='C0', linestyle='-', linewidth=2)
plt.scatter(nn_opt_gap_1*100, bis_opt_gap_wo_gamma*100, label='Projected solution (w/o $\gamma$)', c='C3', s=10, alpha=0.3)
plt.xlabel('Initial Optimal gap (%)', fontsize=16)
plt.ylabel('Projected Optimal gap (%)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.legend(fontsize=14, loc='lower right')

plt.subplot(1,2,2)
plt.plot(nn_opt_gap_2*100, nn_opt_gap_2*100, label='Initial NN solution', c='C0', linestyle='-', linewidth=2)
plt.scatter(nn_opt_gap_2*100, bis_opt_gap_with_gamma*100, label='Projected solution (with $\gamma$)', c='C3', s=10, alpha=0.3)
plt.xlabel('Initial Optimal gap (%)', fontsize=16)
plt.ylabel('Projected Optimal gap (%)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.legend(fontsize=14, loc='lower right')

plt.tight_layout()
plt.show() 