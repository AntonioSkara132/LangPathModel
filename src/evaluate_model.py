import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from transformers import AutoTokenizer

# ---- Your own modules ---- #
from textEncoders import TextEncoder  # update if module path differs
from nn import TrajectoryModel        # update if module path differs


def build_model(d_model: int, num_heads: int, num_decoder_layers: int, model_path: str, device: torch.device):
    """Instantiate TrajectoryModel and load weights."""
    model = TrajectoryModel(
        d_model=d_model,
        num_heads_decoder=num_heads,
        num_decoder_layers=num_decoder_layers,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    model.positional_encoding = model.positional_encoding.to(device)  # ensure PE on the same device
    return model


def generate_animation(model: TrajectoryModel, tokenizer, text_prompt: str, frames: int, interval: int, device: torch.device, save_path: str | None):
    """Generate and optionally save an animated trajectory."""
    # Encode text
    encoded = tokenizer(text_prompt, padding=True, truncation=True, return_tensors="pt")
    txt = encoded["input_ids"].to(device)
    txt_mask = (encoded["attention_mask"] == 0).to(device)
    path_mask = torch.tensor([[1]], device=device)

    # Initialise starting point
    start = torch.tensor([[[0.1, 0.9, 0.0, 0.0]]], device=device)  # shape (1,1,4)
    tgt = start.clone()
    positions = [start[0, 0, :2].clone().cpu().numpy()]

    # Matplotlib setup
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Generated Path with Binned Actions (0=blue, 1=red)")
    ax.grid(True)
    ax.set_aspect("equal")

    scatter_action_0 = ax.scatter([], [], color="blue", label="Action = 0", s=50)
    scatter_action_1 = ax.scatter([], [], color="red", label="Action = 1", s=50)
    ax.legend()

    def update(_):
        nonlocal tgt, positions
        with torch.no_grad():
            prediction = model(text=txt, path=start, tgt=tgt, text_mask=txt_mask, path_mask=path_mask)
        next_point = prediction[:, -1, :]
        positions.append(next_point[0, :2].cpu().numpy())
        tgt = torch.cat([tgt, next_point.unsqueeze(1)], dim=1)
        actions = tgt[0, :, 2].cpu().numpy()
        binned_actions = (actions >= 0.5).astype(int)
        if binned_actions[-1] == 0:
            scatter_action_0.set_offsets(positions)
        else:
            scatter_action_1.set_offsets(positions)
        return scatter_action_0, scatter_action_1

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)

    if save_path:
        ani.save(save_path)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()


# ------------------------------ CLI ------------------------------ #

def get_args():
    parser = argparse.ArgumentParser(description="Generate and visualise a trajectory from text using TrajectoryModel")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model state dict (.pth)")
    parser.add_argument("--text", type=str, default="bottom circle", help="Text prompt describing the desired trajectory")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the decoder")
    parser.add_argument("--num_decoder_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--frames", type=int, default=200, help="Number of animation frames")
    parser.add_argument("--interval", type=int, default=100, help="Delay between frames in milliseconds")
    parser.add_argument("--save", type=str, default=None, help="Optional output file (e.g., path/animation.mp4 or .gif). If omitted, animation is displayed interactively.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = build_model(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_decoder_layers,
        model_path=args.model_path,
        device=device,
    )

    generate_animation(
        model=model,
        tokenizer=tokenizer,
        text_prompt=args.text,
        frames=args.frames,
        interval=args.interval,
        device=device,
        save_path=args.save,
    )
