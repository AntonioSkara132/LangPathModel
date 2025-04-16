import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from textEncoders import TextEncoder


class TrajectoryModel(nn.Module):
    def __init__(self, 
                 d_traj = 4, 
                 d_model=512, 
                 num_heads_encoder=8,
                 num_heads_decoder=8,
                 num_decoder_layers=5,
                 num_encoder_layers=1,
                 hidden_dim=512, 
                 dropout = 0, 
                 max_length=1000):
        super(TrajectoryModel, self).__init__()

        self.d_model = d_model
        self.num_encoders = num_encoder_layers
        self.num_decoders = num_decoder_layers
        
        # Embedding layer for input and output
        self.input_embedding = nn.Linear(d_traj, d_model)
        #self.output_embedding = nn.Linear(d_traj, d_model)
        #self.text_encoder = TextEncoder(output_dim = d_model)
        
        # Positional encoding to add positional information
        self.positional_encoding = self.get_positional_encoding(max_length, d_model)
        self.text_encoder = TextEncoder(output_dim=d_model)

        encoderLayer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads_encoder,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first = True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=encoderLayer, num_layers=1)

        decoderLayer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads_decoder,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first = True
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=5)
        # Define M encoder layers
        #self.encoder_layers = nn.ModuleList(
        #    [EncoderLayer(d_model=d_model, num_heads=num_heads_encoder, dff=hidden_dim, dropout=dropout) for _ in range(num_encoder_layers)]
        #)

        # Define N decoder layers
        #self.decoder_layers = nn.ModuleList(
        #    [DecoderLayer(d_model=d_model, num_heads=num_heads_decoder, dff=hidden_dim, dropout=dropout) for _ in range(num_decoder_layers)]
        #)
        self.output_layer = nn.Linear(d_model, d_traj)

    def forward(self, path, text, path_mask, tgt, text_mask):
        batch_size, path_len = path.size(0), path.size(1)
        tgt_len = tgt.size(1)
        #print(path_len)
        #print(path.shape) 
        #print(tgt.shape)
        #print(tgt.size(1))
        # Embedding
        #print(f"encoding: {self.positional_encoding[:path_len].permute(1, 0, 2).shape}")
        #print(f"encoding: {self.positional_encoding[:tgt_len].unsqueeze(0).shape}")
        emb_src = self.input_embedding(path)
        emb_tgt = self.input_embedding(tgt) 

        #emb_src = emb_src.unsqueeze(0)
        #print(emb_src.shape)
        #print(emb_tgt.shape)
        emb_src = emb_src + self.positional_encoding[:path_len].permute(1, 0, 2)
        emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
        emb_text = self.text_encoder(text, text_mask)
        #print(f"e: {emb_src.shape}")
        #print(f"tgt: {emb_tgt.shape}")
        # Combine
        
        #emb_src = emb_src.transpose(0, 1)
        #emb_tgt = emb_tgt.transpose(0, 1)
        # Masks # Convert to bool padding mask
        #print(f"combined mask{combined_mask}")

        # Decoder target mask (causal)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=path.device) * float('-inf'), diagonal=1).bool()
        #print(f"emb_tgt: {emb_tgt.shape}")

        # Encode
        #print(emb_src.shape)
        #print(f"combined mask: {combined_mask.shape}")
        enc_output = self.encoder(emb_src, src_key_padding_mask=path_mask)

        combined_mask = torch.cat([path_mask, text_mask], dim=1).bool()  # [B, S]

        memory = torch.cat([enc_output, emb_text], dim=1)
        # Decode
        out = self.decoder(emb_tgt, memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask = combined_mask)
        out = self.output_layer(out)
        return out
    
    def get_positional_encoding(self, max_length, d_model):
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_loss(self, predictions, targets, loss_type='mse'):
        """
        Compute the loss for the model, including MSE loss and a length penalty.

        Args:
        - predictions: Tensor of predicted outputs, shape [batch_size, seq_length, d_traj]
        - targets: Tensor of ground truth values, shape [batch_size, seq_length, d_traj]
        - predicted_length: Tensor of the predicted sequence length, shape [batch_size]
        - target_length: Target length for the sequence, scalar or tensor of shape [batch_size]
        - lambda_penalty: Hyperparameter to control the weight of the length penalty
        - loss_type: Type of loss to use. Options are 'mse' (Mean Squared Error)

        Returns:
        - loss: Scalar tensor representing the computed loss value
        """
        
        # Mean Squared Error Loss
        if loss_type == 'mse':
            criterion = nn.MSELoss()
            mse_loss = criterion(predictions, targets)
        
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Penalizing based on the difference between predicted length and target length
        #length_penalty = torch.abs(length).float()
        #length_penalty = length_penalty.mean()  # average penalty across the batch
        
        # Total loss: MSE loss + lambda * length penalty
        loss = mse_loss
        
        return loss

# class EncoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dff, dropout=0.0):
#         super(EncoderLayer, self).__init__()
#         self.layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=num_heads,
#             dim_feedforward=dff,
#             dropout=dropout
#         )

#     def forward(self, x):
#         return self.layer(x)

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dff, dropout=0.0):
#         super(DecoderLayer, self).__init__()
#         self.layer = nn.TransformerDecoderLayer(
#             d_model=d_model,
#             nhead=num_heads,
#             dim_feedforward=dff,
#             dropout=dropout
#         )

#     def forward(self, x, memory, tgt_mask=None):
#         return self.layer(x, memory, tgt_mask=tgt_mask)

#     def forward(self, x, enc_output, mask=None):
#         # Apply the transformer decoder layer
#         return self.decoder_layer(x, enc_output, memory_key_padding_mask=mask[1], tgt_key_padding_mask=mask[0])

"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', d_model=128):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Project BERT's hidden size to desired model size
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden_size, d_model)

        # Optionally freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, text_input, attention_mask=None):
        
        :param text_input: Tensor of shape (batch_size, seq_length)
        :param attention_mask: Optional mask tensor of shape (batch_size, seq_length)
        :return: Tensor of shape (batch_size, seq_length, d_model)
        
        outputs = self.bert(input_ids=text_input, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape: (batch, seq_len, hidden_size)
        projected = self.linear(hidden_states)     # shape: (batch, seq_len, d_model)
        return self.dropout(projected)

"""



