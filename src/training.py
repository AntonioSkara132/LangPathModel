import argparse
import torch
from torch.utils.data import DataLoader
from nn import LangPathModel
import sys
sys.path.append("/content/LangPathModel")
from data.data_utils import PathDataset, collate_fn

t
def get_args():
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Train TrajectoryModel with configurable hyper‑parameters")

    # Training hyper‑parameters
    parser.add_argument("--niter", type=int, default=50, help="Number of training iterations/epochs")
    parser.add_argument("--start_lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularisation)")
    parser.add_argument("--lr_step", type=int, default=15, help="Step size for LR scheduler (if any)")
    parser.add_argument("--gamma", type=int, default=0.1, help="Scheduler factor")


    # Model architecture
    parser.add_argument("--d_model", type=int, default=128, help="Transformer hidden size (d_model)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the decoder")
    parser.add_argument("--num_decoder_layers", type=int, default=5, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Dataloader / misc
    parser.add_argument("--dataset_path", type=str, default="cats_and_dogs.pth", help="Where is data")
    parser.add_argument("--batch_size", type=int, default=500, help="Mini‑batch size")
    parser.add_argument("--output_path", type=str, default="cats_and_dogs.pth", help="Where to save the trained model")

    return parser.parse_args()


def main():
    args = get_args()
    
    dataset = PathDataset(args.dataset_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    # Instantiate model with cmd‑line hyper‑parameters
    model = LangPathModel(
        d_model=args.d_model,
        num_heads_decoder=args.num_heads,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
    )

    #  Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for name, param in model.named_parameters():
        print(f"{name}: {param.device}")
        

    train(
        model=model,
        niter=args.niter,
        dataloader=dataloader,
        device=device,
        start_lr=args.start_lr,
        gamma = args.gamma,
        weight_decay=args.weight_decay,
        step=args.lr_step,
    )

    #Save the model
    model.to("cpu")
    torch.save(model.state_dict(), args.output_path)
    print(f"\nModel saved to {args.output_path}")


if __name__ == "__main__":
    main()

