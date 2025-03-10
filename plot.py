import matplotlib.pyplot as plt
import datetime

# Sample data dictionary
score_dict = {
    "looping": {
        "Clip Score": {32: 0.93, 48: 0.91, 64: 0.89, 96: 0.87, 128: 0.85},
        "Image Quality": {32: 0.92, 48: 0.90, 64: 0.88, 96: 0.86, 128: 0.84},
        "Dynamic Degree": {32: 0.91, 48: 0.89, 64: 0.87, 96: 0.85, 128: 0.83},
        "Motion Smoothness": {32: 0.90, 48: 0.88, 64: 0.86, 96: 0.84, 128: 0.82},
        "Temporal Flickering": {32: 0.89, 48: 0.87, 64: 0.85, 96: 0.83, 128: 0.81}
    },
    "regular": {
        "Clip Score": {32: 0.90, 48: 0.88, 64: 0.86, 96: 0.84, 128: 0.82},
        "Image Quality": {32: 0.89, 48: 0.87, 64: 0.85, 96: 0.83, 128: 0.81},
        "Dynamic Degree": {32: 0.88, 48: 0.86, 64: 0.84, 96: 0.82, 128: 0.80},
        "Motion Smoothness": {32: 0.87, 48: 0.85, 64: 0.83, 96: 0.81, 128: 0.79},
        "Temporal Flickering": {32: 0.86, 48: 0.84, 64: 0.82, 96: 0.80, 128: 0.78}
    }
}

# Create figure with subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
metrics = ["Clip Score", "Image Quality", "Dynamic Degree", "Motion Smoothness", "Temporal Flickering"]
frame_lengths = [32, 48, 64, 96, 128]

# Create evenly spaced positions for x-axis
x_positions = list(range(len(frame_lengths)))

# Plot each metric
for ax, metric in zip(axes, metrics):
    # Plot looping line (red)
    looping_values = [score_dict["looping"][metric][length] for length in frame_lengths]
    ax.plot(x_positions, looping_values, 'r-', label='looping')
    
    # Plot regular line (green)
    regular_values = [score_dict["regular"][metric][length] for length in frame_lengths]
    ax.plot(x_positions, regular_values, 'g-', label='regular')
    
    # Customize plot
    ax.set_title(metric)
    ax.set_xlabel('Frame Length')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(frame_lengths)
    ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Generate filename with current timestamp
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
filename = f"plot-{current_time}.png"

# Save plot
plt.savefig(filename)
plt.close()