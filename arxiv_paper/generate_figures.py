import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
})

# Load the structured trainer states
phases = ['kahwa', 'loubna', 'rawi']
phase_data = {}
for p in phases:
    with open(f'arxiv_paper/figures_data/trainer_state_{p}.json', 'r') as f:
        phase_data[p] = json.load(f)

# Offset calculation for cumulative steps
step_offsets = {
    'kahwa': 0,
    'loubna': 9886,
    'rawi': 9886 + 20650 # 30536
}

# --- Extract WER evaluations ---
wer_points = []
loss_eval_points = []
train_loss_points = []

for p in phases:
    lh = phase_data[p].get('log_history', [])
    offset = step_offsets[p]
    for item in lh:
        step = item.get('step', 0) + offset
        if 'eval_wer' in item:
            wer_points.append((step, item['eval_wer'], p))
        if 'eval_loss' in item:
            loss_eval_points.append((step, item['eval_loss'], p))
        if 'loss' in item:
            train_loss_points.append((step, item['loss'], p))

# ==========================================================
# FIGURE 1: Validation WER Convergence Across Phases
# ==========================================================
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=300)

phase_colors = {
    'kahwa': '#1f77b4',
    'loubna': '#2ca02c',
    'rawi': '#d62728'
}
phase_labels = {
    'kahwa': 'Phase 1: Kahwa Podcast (Spontaneous)',
    'loubna': 'Phase 2: Loubna Stories (Expressive)',
    'rawi': 'Phase 3: Rawi (Cultural Folklore)'
}

# Add background phase shading
ax.axvspan(0, 9886, color='#1f77b4', alpha=0.07, label='_nolegend_')
ax.axvspan(9886, 30536, color='#2ca02c', alpha=0.07, label='_nolegend_')
ax.axvspan(30536, 31661, color='#d62728', alpha=0.07, label='_nolegend_')

# Boundary lines
ax.axvline(9886, color='gray', linestyle=':', linewidth=1.2)
ax.axvline(30536, color='gray', linestyle=':', linewidth=1.2)

# Plot by phase
for p in phases:
    pts = [(s, w) for s, w, ph in wer_points if ph == p]
    pts.sort()
    xs = [x[0] for x in pts]
    ys = [x[1] for x in pts]
    ax.plot(xs, ys, color=phase_colors[p], linewidth=2.0, marker='o', markersize=3.5, label=phase_labels[p])

# Baseline markers (Whisper Small)
ax.axhline(34.85, 0.02, 9886/31661, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.7, label='Small Baseline (Kahwa: 34.85%)')
ax.axhline(14.87, 9886/31661, 30536/31661, color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.7, label='Small Baseline (Loubna: 14.87%)')
ax.axhline(27.54, 30536/31661, 1.0, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.7, label='Small Baseline (Rawi: 27.54%)')

# Annotations for best points
ax.annotate(r'$\mathbf{0.68\%}$ WER' + '\n(-98.05%)', xy=(9886, 0.68), xytext=(5500, 15),
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
            fontsize=8.5, fontweight='bold', color='#1f77b4',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#1f77b4', alpha=0.9))

ax.annotate(r'$\mathbf{0.34\%}$ WER' + '\n(-97.71%)', xy=(30536, 0.34), xytext=(22000, 10),
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2),
            fontsize=8.5, fontweight='bold', color='#2ca02c',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#2ca02c', alpha=0.9))

ax.annotate(r'$\mathbf{0.95\%}$' + '\n(-96.55%)', xy=(31661, 0.95), xytext=(27000, 25),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2),
            fontsize=8.5, fontweight='bold', color='#d62728',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#d62728', alpha=0.9))

# Labels and limits
ax.set_xlabel('Cumulative Training Optimization Steps')
ax.set_ylabel('Validation Word Error Rate (WER %)')
ax.set_title('Whisper Medium Validation WER Convergence across Curriculum Phases')
ax.set_xlim(-500, 32500)
ax.set_ylim(-1, 50)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
ax.legend(loc='upper right', framealpha=0.92)

plt.tight_layout()
fig.savefig('arxiv_paper/fig_wer_convergence.pdf', format='pdf', bbox_inches='tight')
fig.savefig('arxiv_paper/fig_wer_convergence.png', format='png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Generated fig_wer_convergence.pdf and fig_wer_convergence.png')

# ==========================================================
# FIGURE 2: Training & Evaluation Loss Trajectory
# ==========================================================
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=300)

# Training loss (filter extreme initial spike for clean visualization)
train_pts = [(s, l) for s, l, p in train_loss_points if l is not None]
train_pts.sort()
tx = [x[0] for x in train_pts]
ty = [x[1] for x in train_pts]

# Evaluation loss
eval_pts = [(s, l) for s, l, p in loss_eval_points if l is not None]
eval_pts.sort()
ex = [x[0] for x in eval_pts]
ey = [x[1] for x in eval_pts]

# Add background phase shading
ax.axvspan(0, 9886, color='#1f77b4', alpha=0.07)
ax.axvspan(9886, 30536, color='#2ca02c', alpha=0.07)
ax.axvspan(30536, 31661, color='#d62728', alpha=0.07)
ax.axvline(9886, color='gray', linestyle=':', linewidth=1.2)
ax.axvline(30536, color='gray', linestyle=':', linewidth=1.2)

# Text labels for phases
ax.text(4943, 2.8, 'Phase 1: Kahwa', ha='center', fontsize=9, fontweight='bold', color='#1f77b4')
ax.text(20211, 2.8, 'Phase 2: Loubna', ha='center', fontsize=9, fontweight='bold', color='#2ca02c')
ax.text(31098, 2.8, 'Phase 3: Rawi', ha='center', fontsize=8, fontweight='bold', color='#d62728', rotation=90)

ax.plot(tx, ty, color='#4a6fa5', alpha=0.45, linewidth=1.0, label='Training Loss (Batch)')
# Rolling average of training loss
window = 20
if len(ty) > window:
    smoothed = np.convolve(ty, np.ones(window)/window, mode='valid')
    ax.plot(tx[window-1:], smoothed, color='#0b3c5d', linewidth=1.8, label=f'Smoothed Training Loss (MA {window})')

ax.plot(ex, ey, color='#e63946', marker='s', markersize=4, linewidth=2.0, label='Validation Loss (Eval Split)')

# Best final loss annotation
ax.annotate(r'Final Loss = $\mathbf{0.00612}$', xy=(31661, 0.00612), xytext=(21000, 1.2),
            arrowprops=dict(arrowstyle='->', color='#e63946', lw=1.2),
            fontsize=9, fontweight='bold', color='#e63946',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#e63946', alpha=0.9))

ax.set_xlabel('Cumulative Training Optimization Steps')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Whisper Medium Cross-Entropy Loss Trajectory across Curriculum')
ax.set_xlim(-500, 32500)
ax.set_ylim(-0.05, 3.2) # focused on active convergence region
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
ax.legend(loc='upper right', framealpha=0.92)

plt.tight_layout()
fig.savefig('arxiv_paper/fig_loss_trajectory.pdf', format='pdf', bbox_inches='tight')
fig.savefig('arxiv_paper/fig_loss_trajectory.png', format='png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Generated fig_loss_trajectory.pdf and fig_loss_trajectory.png')
