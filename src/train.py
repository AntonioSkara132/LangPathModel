import torch
from nn import TrajectoryModel

#data = [Batch, sequence, (batch input, batch target)]

def train(model, dataloader, niter, device):
    criterion = model.get_loss()  # assumes it returns CrossEntropyLoss with ignore_index for padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model = TrajectoryModel()
    model.train()

    for epoch in range(niter):
        total_loss = 0
        for batch_texts, batch_paths, batch_masks in dataloader():
            batch_paths = batch_paths.to(device).float()
            batch_masks = batch_masks.to(device).long()

            # Shift target for teacher forcing
            decoder_input = batch_paths[:, :-1]      # all except last token
            target_output = batch_texts[:, 1:]        # all except first token
            encoder_input = batch_paths[:, 0]
            encoder_input_mask = torch.Tensor([1])

            optimizer.zero_grad()
            predictions = model(text = batch_texts, path = encoder_input, path_mask = encoder_input_mask)  # shape: [B, T]
            # Reshape for loss: CrossEntropy wants [B*T, vocab_size] vs [B*T]

            #predictions = predictions.reshape(-1, predictions.size(-1))
            target_output = target_output.reshape(-1)

            loss = criterion(predictions, target_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")



    
    