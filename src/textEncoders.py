import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from transformers import AutoModel

class SmallTextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-small")
        
        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Optional projection layer (not needed if downstream uses d_model=512)
        # self.linear = nn.Linear(512, 512)  # Can skip this if not modifying
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Take CLS token
        cls_output = output.last_hidden_state[:, 0]
        return self.dropout(cls_output)


class TinyTextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained("prajjwal1/bert-tiny")  # TinyBERT
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim, bias = False)
        
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        return self.linear(self.dropout(pooled_output))

class TextEncoder(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', output_dim=128):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Project BERT's hidden size to desired model size
        hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(hidden_size, output_dim)

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
        return projected
