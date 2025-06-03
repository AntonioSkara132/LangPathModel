import time
from datetime import datetime
import torch
from torch.optim.lr_scheduler import StepLR

# ------------------------------------------------------------
# assumes you have already done:
#   from LangPathModel.colab_src.nn          import TrajectoryModel
#   from LangPathModel.colab_src.textEncoders import TextEncoder
#   (plus dataloader creation etc.)
# ------------------------------------------------------------

def train(
        model,
        dataloader,
        niter,
        device,
        start_lr: float = 1e-4,
        step: int = 10,
        gamma: float = 0.1,
        weight_decay: float = 0.0,
        checkpoint: str | None = None,
        ckpt_every: int = 10,
        ckpt_dir: str = "/checkpoints/"
):
    """Train `model` for `niter` epochs and time each epoch."""

    # optional, load a checkpoint
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded checkpoint: {checkpoint}")

    model.to(device)
    model.train()                                          
    model.positional_encoding = model.positional_encoding.to(device)

    criterion  = model.get_loss
    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=start_lr,
                                  weight_decay=weight_decay)
    scheduler  = StepLR(optimizer, step_size=10, gamma=0.5)

    num_batches = len(dataloader)

    for epoch in range(1, niter + 1):
        epoch_start = time.time()
        running_loss = 0.0

        for batch_paths, batch_texts, batch_path_masks in dataloader:

            batch_paths      = batch_paths.to(device).float()
            path_masks       = batch_path_masks.to(device).float()

            # teacher-forcing split
            decoder_input    = batch_paths[:, :-1]            # all except last token
            target_output    = batch_paths[:, 1:]             # all except first token

            # text to device + masks
            text_ids   = batch_texts["input_ids"].to(device).long()
            text_mask  = (batch_texts["attention_mask"] == 0).to(device)  # True where padding

            # forward/backward
            optimizer.zero_grad(set_to_none=True)

            preds = model(text=text_ids,
                          tgt=decoder_input,
                          path_mask = path_masks,
                          text_mask=text_mask)  

            loss = criterion(preds, target_output, path_masks[:, 1:])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

	# epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss   = running_loss / num_batches
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"Epoch {epoch:03d}/{niter} | "
              f"Loss: {avg_loss:8.4f} | "
              f"Time: {epoch_time:6.2f}s")

        # checkpointing
        if epoch % ckpt_every == 0 or epoch == niter:
            ckpt_path = f"{ckpt_dir}/model_state_epoch_{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ saved checkpoint to {ckpt_path}")



    
    
