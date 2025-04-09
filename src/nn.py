import torch
import torch.nn as nn
from transformers import BertModel

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
        self.text_encoder = TextEncoder(d_model = 112)

        
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

    def forward(self, src, text, tgt):
        emb_src = self.input_embedding(src) + self.positional_encoding[:src.size(0), :] # [batch_size, seq_length_traj, d_model]
        emb_tgt = self.input_embedding(tgt) + self.positional_encoding[:src.size(0), :] # [batch_size, seq_length_traj, d_model]
        text_embedd = self.text_encoder(text)  # [batch_size, seq_length_text, d_model]
        emb = torch.cat([emb_src, text_embedd], dim=1)  

        enc_output = emb_src
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_output)
        
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(emb_tgt, enc_output)
        out = self.output_layer(out)
        return out
    
    def get_positional_encoding(self, max_length, d_model):
        position = torch.arange(0, max_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dff=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout
        )

    def forward(self, x, mask=None):
        # Apply the transformer encoder layer
        return self.encoder_layer(x, src_key_padding_mask=mask)


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dff=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=dropout
        )

    def forward(self, x, enc_output, mask=None):
        # Apply the transformer decoder layer
        return self.decoder_layer(x, enc_output, memory_key_padding_mask=mask[1], tgt_key_padding_mask=mask[0])



class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', d_model=512):
        super(TextEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_input):
        """
        Forward pass for processing textual data
        :param text_input: Tensor of shape (batch_size, seq_length) representing tokenized text
        :return: Tensor of shape (batch_size, seq_length, d_model) representing processed text embeddings
        """
        bert_output = self.bert(text_input)
        hidden_states = bert_output.last_hidden_state  
        transformed_states = self.linear(hidden_states)
        transformed_states = self.dropout(transformed_states)
        return transformed_states
        


