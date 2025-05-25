import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pad_sequence
from src.textEncoders import TextEncoder

"""LangPathModel
This file contains definition of basic LangPathModel
"""


class LangPathModel(nn.Module):
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
		self.decoder = torch.nn.TransformerDecoder(decoder_layer=decoderLayer, num_layers=5)
		self.output_layer = nn.Linear(d_model, d_traj)

	def forward(self, path_mask, tgt, text_mask):
		batch_size, path_len = path.size(0), path.size(1)
		tgt_len = tgt.size(1)

		emb_tgt = self.input_embedding(tgt) 

		emb_tgt = emb_tgt + self.positional_encoding[:tgt_len].permute(1, 0, 2)
		emb_text = self.text_encoder(text, text_mask)      

		tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=path.device) * float('-inf'), diagonal=1).bool()

		out = self.decoder(emb_tgt, memory=emb_text, tgt_mask=tgt_mask)
		out = self.output_layer(out)
		return out

	def get_positional_encoding(self, max_length, d_model):
		position = torch.arange(0, max_length).unsqueeze(1).float()
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
		pe = torch.zeros(max_length, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		return pe

	def get_loss(
		self,
		predictions: torch.Tensor,      # [B, T, d_traj]
		targets:     torch.Tensor,      # [B, T, d_traj]
		mask:        torch.Tensor,      # [B, T]  (1 = keep, 0 = pad)
		length_penalty: torch.Tensor | None = None,  # e.g. abs(pred_len - tgt_len)
		lambda_penalty: float = 0.1,
		loss_type: str = "mse",
	):
		"""Just and MSE loss between target output and predictions, bu we need to keep in mind to mask extra elements"""


		mask_exp = mask.unsqueeze(-1).float()

		if loss_type == "mse":
			err = (predictions - targets) ** 2
		elif loss_type == "l1":
			err = torch.abs(predictions - targets)
		else:
			raise ValueError(f"Unsupported loss type: {loss_type}")

		# apply mask + safe normalisation 
		masked_err   = err * mask_exp                      # zero out ignored steps
		total_tokens = mask_exp.sum()                      # scalar
		loss_core    = masked_err.sum() / total_tokens.clamp_min(1.0)

		# optional sequence-length penalty 
		if length_penalty is not None:
			loss_penalty = length_penalty.float().mean()
			loss = loss_core + lambda_penalty * loss_penalty
		else:
			loss = loss_core

		return loss

    






