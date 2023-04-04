from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

from .pvtv2 import pvt_v2_b2

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        # Compute attention
        attn = th.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = th.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module for Hyperspectral Pansharpening (Image Fusion) '''

    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()
        #Parameters
        self.n_head         = n_head        #No of heads
        self.in_pixels      = in_pixels     #No of pixels in the input image
        self.linear_dim     = linear_dim    #Dim of linear-layer (outputs)

        #Linear layers
        self.w_qs   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for queries
        self.w_ks   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for keys
        self.w_vs   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for values
        self.fc     = nn.Linear(n_head * linear_dim, in_pixels, bias=False) #Final fully connected layer

        #Scaled dot product attention
        self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)

        #Batch normalization layer
        self.OutBN = nn.BatchNorm2d(num_features=num_features)

    def forward(self, v, k, q, mask=None):
        # Reshaping matrixes to 2D
        # q = b, c_q, h*w
        # k = b, c_k, h*w
        # v = b, c_v, h*w
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head          = self.n_head
        linear_dim      = self.linear_dim

        # Reshaping K, Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        #Save V
        output = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(v_attn)
        
        
        output = output + v_attn
        #output  = v_attn

        #Reshape output to original image format
        output = output.view(b, c, h, w)

        #We can consider batch-normalization here,,,
        #Will complete it later
        output = self.OutBN(output)
        return output



class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,  # 128
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        # self.edge_bb = pvt_v2_b2()
        
        
        # for noise feature extraction using Unet
        time_embed_dim = model_channels * 4  # 128 * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    # print(f'in encoder curr ds {ds}, level{level}, curr attention res{attention_resolutions}')
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample( 
                            ch, conv_resample, dims=dims, out_channels=out_ch # spatial res/2
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    # print(f'in decoder curr ds {ds}, level{level}, curr attention res{attention_resolutions}')
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

  

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)


        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)



class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
'''a –> the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
   mode –> either 'fan_in' (default) or 'fan_out'. 
Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. 
Choosing 'fan_out' preserves the magnitudes in the backwards pass.
'''

class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class DimensionalExtention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalExtention, self).__init__()
        self.extend = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.extend(x)

class EdgeDeducingModule(nn.Module):
    def __init__(self, in_channel):
        super(EdgeDeducingModule, self).__init__()
        self.reduce = DimensionalReduction(in_channel=in_channel, out_channel=int(in_channel/2))  # [128, 320, 512]
        
        self.cbr1 = nn.Sequential(ConvBR(int(in_channel/2), 64, 3, padding=1))
        self.cbr2 = nn.Sequential(ConvBR(64, 32, 3, padding=1))
                                   
        self.out_conv = nn.Conv2d(32+64, 1, 1)

    def forward(self, edge_bb_feature):
        size = edge_bb_feature.size()[2:]
        edge_bb_feature = self.reduce(edge_bb_feature)
        # edge_bb_feature = F.interpolate(edge_bb_feature, size, mode='bilinear', align_corners=False)
        # print('reduced edge_bb_feature', edge_bb_feature.size())
        cat_feature = self.cbr1(edge_bb_feature)
        # print('cbr1', cat_feature.size())
        out = self.cbr2(cat_feature)
        # print('cbr2', out.size())
        out = th.cat([out, cat_feature], dim=1)
        # print('cat feature size', out.size())
        out = self.out_conv(out)
        return out


class PriorGuidedFeatureRefinement(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PriorGuidedFeatureRefinement, self).__init__()
        self.edge_deduce_module = EdgeDeducingModule(in_channel)
        self.de = DimensionalExtention(1, in_channel)
        self.dr = DimensionalReduction(in_channel, out_channel)
        
        
    def forward(self, edge_bb_feature):
        edge_out = self.edge_deduce_module(edge_bb_feature)
        feature_out = self.de(edge_out)
        feature_out = th.mul(edge_bb_feature,  feature_out)
        feature_out = self.dr(feature_out)
        return feature_out, edge_out



class CrossDomainFeatureFusion(nn.Module):  # [128, 320, 512]
    def __init__(self, cat_channel, out_channel, N=[4, 8, 16]):
        super(CrossDomainFeatureFusion, self).__init__()
        # grouping method is the only difference here
        
        self.g_conv1 = nn.Conv2d(cat_channel, cat_channel, kernel_size=1, groups=N[0], bias=False)
        self.g_conv2 = nn.Conv2d(cat_channel, cat_channel, kernel_size=1, groups=N[1], bias=False)
        self.g_conv3 = nn.Conv2d(cat_channel, cat_channel, kernel_size=1, groups=N[2], bias=False)
        self.dr = DimensionalReduction(cat_channel, out_channel)
        self.cbr1 = ConvBR(out_channel, out_channel, 3, padding=1)
        self.cbr2 = ConvBR(out_channel, out_channel, 3, padding=1)
        self.upsample = Upsample(out_channel, False, dims=2)


    def forward(self, noise_f, edge_f):
        x = th.cat([noise_f, edge_f], dim=1)
        x = self.g_conv1(x) + self.g_conv2(x) + self.g_conv3(x) + x
        x_res = self.dr(x)

        x = self.cbr1(x_res)
        x = self.cbr2(x)
        x = x_res + x
        x = self.upsample(x)
        
        return x



class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class IntegratedUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.edge_bb = pvt_v2_b2()
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.out_layer5 =  nn.ModuleList([])
        layer5 = [  ResBlock(
                        128*2,    # channel
                        time_embed_dim,
                        dropout,
                        out_channels=256,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
                    ResBlock(
                        128*2,
                        time_embed_dim,
                        dropout,
                        out_channels=256,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
                    ResBlock(
                            128*2,
                            time_embed_dim,
                            dropout,
                            out_channels=128,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                ]
        self.out_layer5.append(TimestepEmbedSequential(*layer5))
        
        self.out_layer6 =  nn.ModuleList([])
        layer6 = [  ResBlock(
                        128*2,    # channel
                        time_embed_dim,
                        dropout,
                        out_channels=256,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
                    ResBlock(
                        128*2,
                        time_embed_dim,
                        dropout,
                        out_channels=256,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
        self.out_layer6.append(TimestepEmbedSequential(*layer6))
        
        
        self.out = nn.Sequential(
            normalization(256),
            nn.SiLU(),
            zero_module(conv_nd(2, 256, 2, 3, padding=1)),
        )


        path = './results/pvt_v2_b2.pth'
        save_model = th.load(path)
        model_dict = self.edge_bb.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.edge_bb.load_state_dict(model_dict)

        self.pgfr1 = PriorGuidedFeatureRefinement(in_channel=512, out_channel=512)
        self.pgfr2 = PriorGuidedFeatureRefinement(in_channel=320+512, out_channel=320)    # curr PGFR + last PGFR
        self.pgfr3 = PriorGuidedFeatureRefinement(in_channel=128+320, out_channel=128)
        self.pgfr4 = PriorGuidedFeatureRefinement(in_channel=64+128, out_channel=64)    

        self.dr2 = DimensionalReduction(in_channel=512*2, out_channel=320)    #  U-net + U-net(after DR)
        self.dr3 = DimensionalReduction(in_channel=320+256, out_channel=128)
        self.dr4 = DimensionalReduction(in_channel=128+256, out_channel=64) 


        self.transformer_encoder1 = MultiHeadAttention(4, 11**2, 32, 512) # 11^2
        self.transformer_encoder2 = MultiHeadAttention(8, 22**2, 64, 320) # 11^2
        self.transformer_encoder3 = MultiHeadAttention(16, 44**2, 128, 128) # 11^2
        self.transformer_encoder4 = MultiHeadAttention(32, 88**2, 256, 64) # 11^2
        self.dimension_extension = DimensionalExtention(64, 128)
        
        # self.cdff1 = CrossDomainFeatureFusion(cat_channel=512*2, out_channel=512)    #  PGFR + U-net(after DR)
        # self.cdff2 = CrossDomainFeatureFusion(cat_channel=320*2, out_channel=320)
        # self.cdff3 = CrossDomainFeatureFusion(cat_channel=128*2, out_channel=128)
        # self.cdff4 = CrossDomainFeatureFusion(cat_channel=64*2, out_channel=128)    

        self.pgfr1_up =  Upsample(512, False, dims=2)
        self.pgfr2_up =  Upsample(320, False, dims=2)
        self.pgfr3_up =  Upsample(128, False, dims=2)
        # self.dr1 = DimensionalReduction(in_channel=512, out_channel=512)
        
        self.upsample_s1 = Upsample(512, False, dims=2)
        self.upsample_s2 = Upsample(320, False, dims=2)
        self.upsample_s3 = Upsample(128, False, dims=2)
        self.upsample_s4 = Upsample(128, False, dims=2)

        
        self.upsample_32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)



    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # if self.num_classes is not None:
        #     assert y.shape == (x.shape[0],)
        #     emb = emb + self.label_emb(y)

        # for edge feature extraction 
        # in_channel_list = [64, 128, 320, 512]
        # channel = 32
        pvt = self.edge_bb(x[:, :-1, ...])
        fb1 = pvt[0]
        fb2 = pvt[1]
        fb3 = pvt[2]
        fb4 = pvt[3]

        h = x.type(self.dtype)
        for idx, module in enumerate(self.input_blocks):
            h = module(h, emb)
            if (idx-1)%(self.num_res_blocks+1)==0:
                hs.append(h)
        hs.pop()

        h = self.middle_block(h, emb)
        


        
        pgfr1_out, edge1 = self.pgfr1(fb4)
        V1, K1, Q1 = h, h, pgfr1_out
        h = self.transformer_encoder1(V1, K1, Q1)
        h = self.upsample_s1(h)
        h = self.dr2(th.cat([h, hs.pop()], dim=1))
        pgfr1_out = self.pgfr1_up(pgfr1_out)

        pgfr2_out, edge2 = self.pgfr2(th.cat([fb3, pgfr1_out], dim=1))
        V2, K2, Q2 = h, h, pgfr2_out
        h = self.transformer_encoder2(V2, K2, Q2)
        h = self.upsample_s2(h)
        h = self.dr3(th.cat([h, hs.pop()], dim=1))
        pgfr2_out = self.pgfr2_up(pgfr2_out)

        pgfr3_out, edge3 = self.pgfr3(th.cat([fb2, pgfr2_out], dim=1))
        V3, K3, Q3 = h, h, pgfr3_out
        h = self.transformer_encoder3(V3, K3, Q3)
        h = self.upsample_s3(h)
        h = self.dr4(th.cat([h, hs.pop()], dim=1))
        pgfr3_out = self.pgfr3_up(pgfr3_out)

        pgfr4_out, edge4 = self.pgfr4(th.cat([fb1, pgfr3_out], dim=1))
        V4, K4, Q4 = h, h, pgfr4_out
        h = self.transformer_encoder4(V4, K4, Q4)
        h = self.dimension_extension(h)
        h = self.upsample_s4(h)
        
        

        h = th.cat([h, hs.pop()], dim=1)
        for module in self.out_layer5:
            h = module(h, emb)

        h = th.cat([h, hs.pop()], dim=1)
        for module in self.out_layer6:
            h = module(h, emb)
        
        out = self.out(h)
        
        return out, (self.upsample_32(edge1), self.upsample_16(edge2), self.upsample_8(edge3), self.upsample_4(edge4))


