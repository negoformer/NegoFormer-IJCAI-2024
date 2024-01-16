from typing import Union
import torch
import torch.nn as nn
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
from .layers.Embed import TokenEmbedding


class Model(nn.Module):
    """
        Autoformer is the first method to achieve the series-wise connection,
        with inherent O(LlogL) complexity
    """

    def __init__(self, enc_in: int, output_length: int, d_model: int = 512, dropout: float = 0.1, n_heads: int = 8,
                 factor: int = 5,
                 e_layers: int = 2, d_layers: int = 1, activation: str = "gelu", d_ff: Union[int, None] = None,
                 distil: bool = False):
        super(Model, self).__init__()

        self.pred_length = output_length
        self.dropout = dropout

        if d_ff is None:
            d_ff = d_model * 4

        # Decomp
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)
        self.output_attention = False

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        # self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        # self.dec_embedding = DataEmbedding_wo_pos(dec_in, d_model, embed, freq, dropout)

        self.enc_embedding = TokenEmbedding(enc_in, d_model)
        self.dec_embedding = TokenEmbedding(enc_in, d_model)
        self.time_embedding = nn.Linear(1, d_model, bias=False)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    moving_avg=25,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    1,
                    d_ff,
                    moving_avg=25,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, 1, bias=True)
        )

    def forward(self, x_enc, x_dec, time, target_time,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_length, 1)
        zeros = torch.zeros([x_dec.shape[0], target_time.shape[1], x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, 0:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, 0:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc)

        time_out = self.time_embedding(time)

        enc_out = enc_out + time_out

        enc_out = nn.Dropout(p=self.dropout)(enc_out)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec

        time_dec = torch.cat([time, target_time], dim=1)

        dec_out = self.dec_embedding(seasonal_init)

        time_out_dec = self.time_embedding(time_dec)

        dec_out = dec_out + time_out_dec

        dec_out = nn.Dropout(p=self.dropout)(dec_out)

        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_length:, :], attns
        else:
            return dec_out[:, -self.pred_length:, :]  # [B, L, D]
