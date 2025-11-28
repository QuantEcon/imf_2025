import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Get default matplotlib colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_state = colors[0]  # First color for states
color_policy = colors[1]  # Second color for policy
color_action = colors[2]  # Third color for actions

# Draw State Space ellipse
state_ellipse = Ellipse((3, 4.5), 4, 6,
                        edgecolor=color_state,
                        facecolor=color_state,
                        linewidth=3,
                        alpha=0.15)
ax.add_patch(state_ellipse)
ax.text(3, 7.8, 'State Space S', fontsize=16, ha='center', va='center',
        weight='bold', color=color_state)

# Draw states inside the ellipse
states = ['s₁', 's₂', 's₃', 's₄', 's₅', 's₆']
state_positions = [(2.5, 6.5), (3.5, 6.5), (2, 5), (4, 5), (2.5, 3.5), (3.5, 3.5)]

for i, (state, pos) in enumerate(zip(states, state_positions)):
    # Highlight s₄ as the current state
    if state == 's₄':
        circle = plt.Circle(pos, 0.4, color=color_state, linewidth=3, fill=True, alpha=0.9)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state, fontsize=14, ha='center', va='center',
                weight='bold', color='white')
    else:
        circle = plt.Circle(pos, 0.35, color=color_state, linewidth=2, fill=False, alpha=0.6)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state, fontsize=12, ha='center', va='center',
                color=color_state, alpha=0.7)

# Draw Action Space ellipse
action_ellipse = Ellipse((11, 4.5), 4, 6,
                         edgecolor=color_action,
                         facecolor=color_action,
                         linewidth=3,
                         alpha=0.15)
ax.add_patch(action_ellipse)
ax.text(11, 7.8, 'Action Space A', fontsize=16, ha='center', va='center',
        weight='bold', color=color_action)

# Draw actions inside the ellipse
actions = ['a₁', 'a₂', 'a₃', 'a₄']
action_positions = [(10.5, 6.2), (11.5, 6.2), (10.5, 4.5), (11.5, 4.5)]

for i, (action, pos) in enumerate(zip(actions, action_positions)):
    # Highlight a₃ as the selected action
    if action == 'a₃':
        square = FancyBboxPatch((pos[0]-0.35, pos[1]-0.35), 0.7, 0.7,
                               boxstyle="round,pad=0.05",
                               edgecolor=color_action,
                               facecolor=color_action,
                               linewidth=3,
                               alpha=0.9)
        ax.add_patch(square)
        ax.text(pos[0], pos[1], action, fontsize=14, ha='center', va='center',
                weight='bold', color='white')
    else:
        square = FancyBboxPatch((pos[0]-0.3, pos[1]-0.3), 0.6, 0.6,
                               boxstyle="round,pad=0.05",
                               edgecolor=color_action,
                               facecolor='none',
                               linewidth=2,
                               alpha=0.6)
        ax.add_patch(square)
        ax.text(pos[0], pos[1], action, fontsize=12, ha='center', va='center',
                color=color_action, alpha=0.7)

# Draw single arrow from s₄ to a₃
arrow = FancyArrowPatch((4.4, 5), (10.15, 4.5),
                        arrowstyle='->', mutation_scale=25,
                        color='black', linewidth=3.5,
                        connectionstyle="arc3,rad=0.0")
ax.add_patch(arrow)

# Add sigma label above the arrow
ax.text(7.25, 5.2, 'σ', fontsize=28, ha='center', va='center',
        weight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.9))

plt.tight_layout()
plt.savefig('/home/john/temp/policy_function_direct.pdf', dpi=300, bbox_inches='tight')
plt.show()
