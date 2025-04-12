import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence

class TrajectoryModel(nn.Module):
    def __init__(self, 
                 d_traj = 3, 
                 d_model=512, 
                 num_heads_encoder=8,
                 num_heads_decoder=8,
                 num_decoder_layers=5,
                 num_encoder_layers=1,
                 hidden_dim=512, 
                 dropout = 0, 
                 max_length=100):
        super(TrajectoryModel, self).__init__()

        self.d_model = d_model
        self.num_encoders = num_encoder_layers
        self.num_decoders = num_decoder_layers
        
        # Embedding layer for input and output
        self.input_embedding = nn.Linear(d_traj, d_model)
        self.output_embedding = nn.Linear(d_traj, d_model)
        self.text_encoder = TextEncoder(d_model = d_model)
        
        # Positional encoding to add positional information
        self.positional_encoding = self._get_positional_encoding(max_length, d_model)
        
        # Define M encoder layers
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model=d_model, num_heads=num_heads_encoder, dff=hidden_dim, dropout=dropout) for _ in range(num_encoder_layers)]
        )

        # Define N decoder layers
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model=d_model, num_heads=num_heads_decoder, dff=hidden_dim, dropout=dropout) for _ in range(num_decoder_layers)]
        )
        self.output_layer = nn.Linear(d_model, d_traj)

    def forward(self, path, text, path_mask, tgt):
        #path tensor [batch_size, seq_size(padded), 4]
        #text tesor [batch_size, seq_size(padded), 512]

        batch_size, path_len, _ = path.size()
        text_len = text.size(1)

        emb_src = self.input_embedding(path) + self.positional_encoding[:path.size(0), :] # [batch_size, seq_length_traj, d_model]
        
        emb_tgt = self.input_embedding(tgt) + self.positional_encoding[:tgt.size(0), :] # [batch_size, seq_length_traj, d_model]
        emb_text = self.text_encoder(text)  # [batch_size, seq_length_text, d_model]

        emb_src  = torch.cat([emb_src, emb_text], dim=1)  
        
        #padding mask for encoder
        text_mask = torch.zeros(batch_size, text_len, dtype=torch.bool, device=path.device)
        combined_mask = torch.cat([path_mask, text_mask], dim=1)  # [batch_size, seq_len_traj + seq_len_text]
        
        T = tgt.shape[1]
        tgt_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

        enc_output = emb_src
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output, combined_mask)
        
        out = emb_tgt
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(out, enc_output, tgt_mask = tgt_mask)
        out = self.output_layer(out)
        return out
    
    def get_positional_encoding(self, max_length, d_model):
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_loss(self, predictions, targets, length, lambda_penalty=0.1, loss_type='mse'):
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
        length_penalty = torch.abs(length).float()
        length_penalty = length_penalty.mean()  # average penalty across the batch
        
        # Total loss: MSE loss + lambda * length penalty
        loss = mse_loss + lambda_penalty * length_penalty
        
        return loss

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout
        )

    def forward(self, x):
        return self.layer(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout
        )

    def forward(self, x, memory, tgt_mask=None):
        return self.layer(x, memory, tgt_mask=tgt_mask)

    def forward(self, x, enc_output, mask=None):
        # Apply the transformer decoder layer
        return self.decoder_layer(x, enc_output, memory_key_padding_mask=mask[1], tgt_key_padding_mask=mask[0])
    
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
        """
        :param text_input: Tensor of shape (batch_size, seq_length)
        :param attention_mask: Optional mask tensor of shape (batch_size, seq_length)
        :return: Tensor of shape (batch_size, seq_length, d_model)
        """
        outputs = self.bert(input_ids=text_input, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # shape: (batch, seq_len, hidden_size)
        projected = self.linear(hidden_states)     # shape: (batch, seq_len, d_model)
        return self.dropout(projected)

        



