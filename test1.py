

import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from src.textEncoders import TextEncoder
from transformers import AutoTokenizer
from src.nn import TrajectoryModel

# Step 1: Initialize the model

d_model = 128
model = TrajectoryModel(d_model=d_model, num_heads_decoder=8, num_decoder_layers=2)

# Step 2: Load the saved state dict
model.load_state_dict(torch.load('/home/antonio/Workspace/Seminar/LangPathModel/models/model_state_dict_dec_2_head_8_sq_and_ci.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))

# Step 3: Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.positional_encoding = model.positional_encoding.to(device)  # This line was added

# Initialize tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text_encoder = TextEncoder(output_dim=d_model)

# Text and encoding
text = "circling"
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# **Instead of directly using text_encoder output, use encoded['input_ids'] and encoded['attention_mask']**
# txt = text_encoder(encoded['input_ids'], encoded['attention_mask'])
txt = encoded['input_ids'].to(device)
txt_mask = (encoded['attention_mask'] == 0).to(device)


path_mask = torch.Tensor([[1]]).to(device)

# Iniptialize starting point
start = torch.Tensor([[[0.1, 0.9, 0, 0]]]).to(device)  # (1, 1, 4)
#tgt = torch.empty([1, 1, 4]).to(device)  # (1, 1, 4)
tgt = start
# Store predictions
positions = [start[0, 0, :2].clone().cpu().numpy()]

# Loop to generate 100 predictions
for i in range(200):
    with torch.no_grad():
        prediction = model(
            text=txt,
            path=start,
            tgt=tgt,
            text_mask=txt_mask,
            path_mask=path_mask
        )  # Output shape: (1, seq_len+1, 4
    next_point = prediction[:, -1, :]  # Get the last predicted point
    #print(next_point)
    positions.append(next_point[0, :2].cpu().numpy())  # Save (x, y)

    # Append next_point to tgt for next prediction
    tgt = torch.cat([tgt, next_point.unsqueeze(1)], dim=1)
    if next_point[0, 3] > 0.12: break
    print(next_point[0, 3])

# Convert predictions to numpy array
positions = np.array(positions)  # shape: (num_points, 2)
actions = tgt[0, :, 2].cpu().numpy()  # shape: (num_points,) - only 'a' values
#print(positions)
# Bin actions
binned_actions = (actions >= 0.5).astype(int)  # 0 if a < 0.5, 1 otherwise

# Plot
plt.figure(figsize=(8, 6))
for i in range(len(positions)):
    x, y = positions[i]
    if binned_actions[i] == 0:
        plt.scatter(x, y,color='blue', label='Action = 0' if i == 0 else "", s=50)
    else:
        plt.scatter(x, y, color='red', label='Action = 1' if i == 0 else "", s=50)

plt.title("Generated Path with Binned Actions (0=blue, 1=red)")
plt.xlabel("X")
plt.ylabel("Y")

plt.legend()
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()
