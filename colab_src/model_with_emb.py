import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from LangPathModel.colab_src.textEncoders import TextEncoder


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
        
        # Embedding layer for input and output
        self.embedding = nn.EmbeddingTable(25*25, d_model, max_norm = 1)
        
        # Positional encoding to add positional information
        self.positional_encoding = self.get_positional_encoding(max_length, d_model)
        self.text_encoder = TextEncoder(output_dim=d_model)

        decoderLayer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads_decoder,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first = True
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=num_decoder_layers)

    def forward(self, text, path_mask, tgt):
        emb_tgt = self.input_embedding(tgt) 

        emb_src = emb_src + self.positional_encoding[:path_len].permute(1, 0, 2)
        emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
        emb_text = self.text_encoder(text, text_mask)
      
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=path.device) * float('-inf'), diagonal=1).bool()

        out = self.decoder(emb_tgt, memory=emb_text, tgt_mask=tgt_mask)
        out = out @ self.embedding..weights.T
        return out
    
    def get_positional_encoding(self, max_length, d_model):
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

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
