import torch
import torch.nn as nn

from attention import MultiHeadSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()

        self.mhsa1 = MultiHeadSelfAttention(num_heads, d_model)
        self.mhsa2 = MultiHeadSelfAttention(num_heads, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model)
        )

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, encoder_output, target, training, decoder_mask, memory_mask):
        mhsa_output1, attn_weights = self.mhsa1(target, target, target, decoder_mask)
        mhsa_output1 = self.dropout1(mhsa_output1)
        mhsa_output1 = self.layernorm1(mhsa_output1 + target)

        mhsa_output2, attn_weights = self.mhsa2(
            mhsa_output1, encoder_output, encoder_output, memory_mask
        )
        mhsa_output2 = self.dropout2(mhsa_output2)
        mhsa_output2 = self.layernorm2(mhsa_output2 + mhsa_output1)

        ffn_output = self.ffn(mhsa_output2)
        ffn_output = self.dropout3(ffn_output)
        output = self.layernorm3(ffn_output + mhsa_output2)

        return output, attn_weights


class Decoder(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        num_heads,
        hidden_dim,
        target_vocab_size,
        max_seq_len,
        dropout_rate=0.1,
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(target_vocab_size, self.d_model)
        self.pos_embed = nn.Embedding(max_seq_len, self.d_model)

        self.dropout = nn.Dropout(dropout_rate)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, hidden_dim, dropout_rate)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, encoder_output, target, training, decoder_mask, memory_mask):
        token_embeds = self.token_embed(target)

        num_pos = target.size(0) * self.max_seq_len
        pos_idx = (
            torch.arange(self.max_seq_len)
            .repeat(num_pos)
            .reshape(target.size())
            .to(target.device)
        )
        pos_embeds = self.pos_embed(pos_idx)

        x = self.dropout(token_embeds + pos_embeds)

        weights = None
        for block in self.blocks:
            x, weights = block(encoder_output, x, training, decoder_mask, memory_mask)

        return x, weights
