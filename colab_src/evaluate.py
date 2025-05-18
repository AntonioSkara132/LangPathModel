import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from textEncoders import TextEncoder
from transformers import AutoTokenizer
from nn import TrajectoryModel

d_model = 128
model = TrajectoryModel(d_model=d_model, num_heads_decoder=8, num_decoder_layers=2)
model.eval()

# Load the trained model
model.load_state_dict(torch.load('/home/antonio/Workspace/Seminar/LangPathModel/colab_src/model_state_dict.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))

# Move model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.positional_encoding = model.positional_encoding.to(device)

# Initialize tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_encoder = TextEncoder(output_dim=d_model)

# Encode text
text = "bottom circle"
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

txt = encoded['input_ids'].to(device)
txt_mask = (encoded['attention_mask'] == 0).to(device)

path_mask = torch.Tensor([[1]]).to(device)

# Initialize starting point
start = torch.Tensor([[[0.1, 0.9, 0, 0]]]).to(device)
tgt = start
positions = [start[0, 0, :2].clone().cpu().numpy()]  # Store initial position

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Generated Path with Binned Actions (0=blue, 1=red)')

# Scatter plot objects
scatter_action_0 = ax.scatter([], [], color='blue', label='Action = 0', s=50)
scatter_action_1 = ax.scatter([], [], color='red', label='Action = 1', s=50)
ax.legend()
ax.grid(True)
ax.set_aspect('equal')

# Function to update the plot with new points
def update(i):
    global start, tgt

    with torch.no_grad():
        # Generate the next prediction
        prediction = model(text=txt, path=start, tgt=tgt, text_mask=txt_mask, path_mask=path_mask)

    next_point = prediction[:, -1, :]
    positions.append(next_point[0, :2].cpu().numpy())  # Save (x, y)

    # Append next_point to tgt for next prediction
    tgt = torch.cat([tgt, next_point.unsqueeze(1)], dim=1)

    # Binned actions
    actions = tgt[0, :, 2].cpu().numpy()
    binned_actions = (actions >= 0.5).astype(int)

    # Update the scatter plots with the new position and action
    if binned_actions[-1] == 0:
        scatter_action_0.set_offsets(positions)  # Add new point to action 0 scatter
    else:
        scatter_action_1.set_offsets(positions)  # Add new point to action 1 scatter

    if next_point[0, 3] > 0.5:
        return scatter_action_0, scatter_action_1

    return scatter_action_0, scatter_action_1

# Create the animation and assign it to a variable
from IPython.display import HTML

ani = FuncAnimation(fig, update, frames=200, interval=100, repeat=False)

# Display as HTML5 video
plt.show()
