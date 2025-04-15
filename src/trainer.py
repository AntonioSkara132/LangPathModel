import torch
from nn import TrajectoryModel

#data = [Batch, sequence, (batch input, batch target)]

def train(model, dataloader, niter, device):
    criterion = model.get_loss  # assumes it returns CrossEntropyLoss with ignore_index for padding
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    #Scheduler = torch.optim.LRScheduler(optimizer, gamma=0.9)

    model = TrajectoryModel()
    model.train()

    for epoch in range(niter):
        total_loss = 0
        for batch_paths, batch_texts in dataloader:
            #print(f"paths: {batch_paths[0]}")
            batch_paths = batch_paths.to(device).float()
            #print(type(batch_paths))
            # Shift target for teacher forcing
            decoder_input = batch_paths[:, :-1]      # all except last token
            target_output = batch_paths[:, 1:]        # all except first token
            encoder_input = batch_paths[:, 0]
            encoder_input_mask = (encoder_input.abs().sum(dim=-1) != 0).int().reshape(-1, 1)

            optimizer.zero_grad()
            #print(f"encoder input mask: {encoder_input_mask.shape}")
            #print(f"encoder_input: {encoder_input.shape}")
            predictions = model(text = batch_texts, path = encoder_input, path_mask = encoder_input_mask, tgt = decoder_input)  # shape: [B, T]
            # Reshape for loss: CrossEntropy wants [B*T, vocab_size] vs [B*T]

            #predictions = predictions.reshape(-1, predictions.size(-1))
            #print("predictions:", predictions.shape)        # should be [32, 199, 512]
            #print("target_output:", target_output.shape)    # should be the same

            loss = criterion(predictions, target_output)#fix this
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(loss.item())

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")



    
    
