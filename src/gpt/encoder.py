import torch
import torch.nn as nn

from attention import MultiHeadSelfAttention


class EncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, hidden_dim: int, dropout_rate=0.1
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model)
        )

        self.dropout1 = nn.Dropout1d(dropout_rate)
        self.dropout2 = nn.Dropout1d(dropout_rate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X, mask):
        mhsa_output, attn_weights = self.mhsa(X, X, X, mask)
        mhsa_output = self.dropout1(mhsa_output)
        mhsa_output = self.norm1(X + mhsa_output)

        ffn_output = self.ffn(mhsa_output)
        ffn_output = self.dropout2(ffn_output)

        output = self.norm2(mhsa_output + ffn_output)

        return output, attn_weights


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        num_heads,
        hidden_dim,
        src_vocab_size,
        max_seq_len,
        dropout_rate=0.1,
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(src_vocab_size, self.d_model)
        self.pos_embed = nn.Embedding(max_seq_len, self.d_model)

        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(d_model, num_heads, hidden_dim, dropout_rate)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, input, training, mask):
        token_embeds = self.token_embed(input)

        num_pos = input.size(0) * self.max_seq_len
        pos_idx = (
            torch.arange(self.max_seq_len)
            .repeat(num_pos)
            .reshape(input.size())
            .to(input.device)
        )
        pos_embeds = self.pos_embed(pos_idx)

        x = self.dropout(token_embeds + pos_embeds)

        weights = None
        for block in self.blocks:
            x, weights = block(x, training, mask)

        return x, weights
