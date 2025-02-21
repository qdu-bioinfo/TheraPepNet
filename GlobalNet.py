import torch
import torch.nn as nn
import numpy as np


class RoPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RoPE, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
    
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        self.position = position
        self.div_term = div_term

    def forward(self, x):
        seq_len = x.size(0)
        position = self.position[:seq_len]
        div_term = self.div_term
        sin = torch.sin(position * div_term)
        cos = torch.cos(position * div_term)
        x_ = x.view(seq_len, -1, self.d_model)
        x_ = x_ * cos + x_[:, :, 1::2] * sin
        
        return x_.view(seq_len, -1)


class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        q = self.query(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out(context)
        return out, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.linear1(x)))
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = SelfAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2, _ = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.pe = PositionalEncoding(encoder_layer.self_attn.d_model)

    def forward(self, src):
        src = self.pe(src)
        for layer in self.layers:
            src = layer(src)
        return src


class GlobalNet(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(GlobalNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = EncoderLayer(d_model, nhead)
        self.encoder = Encoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embedding(x)  
        x = self.encoder(x)
        x = x.mean(dim=1)  
        x = self.fc(x)
        return x
