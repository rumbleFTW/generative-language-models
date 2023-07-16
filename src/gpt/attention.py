import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_head = self.d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()

        x = x.view(batch_size, seq_len, self.num_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.size()

        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_head).float()
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, q, k, v, mask=None):
        qs = self.wq(q)
        ks = self.wk(k)
        vs = self.wv(v)

        qs = self.split_heads(qs)
        ks = self.split_heads(ks)
        vs = self.split_heads(vs)

        output, attn_weights = self.scaled_dot_product_attention(qs, ks, vs, mask)
        output = self.merge_heads(output)

        return self.dense(output), attn_weights
