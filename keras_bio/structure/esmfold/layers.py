"""
Neural network layers for ESMFold protein folding architecture.

Implements Triangular Multiplicative Updates, Triangular Attention, Invariant Point
Attention (IPA), Angle ResNet, Structure Module, and Evoformer-style Folding Trunk.
"""

import math
from typing import Any, Optional

import keras
from keras import layers, ops

import keras_bio.structure.esmfold.geometry as geometry
import keras_bio.structure.esmfold.residue_constants as rc
import keras_bio.structure.esmfold.feats as feats


class EsmFoldRelativePosition(layers.Layer):
    """
    Encodes relative 1D sequence distance between residue pairs into pairwise embedding space.
    """

    def __init__(self, position_bins: int = 32, pairwise_state_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.position_bins = position_bins
        self.pairwise_state_dim = pairwise_state_dim
        self.embedding = layers.Embedding(
            input_dim=2 * position_bins + 2,
            output_dim=pairwise_state_dim,
            name="embedding",
        )

    def call(self, residue_index, mask=None):
        # diff = j - i
        diff = ops.expand_dims(residue_index, axis=1) - ops.expand_dims(residue_index, axis=2)
        diff = ops.clip(diff, -self.position_bins, self.position_bins)
        diff = diff + self.position_bins + 1  # 0 is reserved for masked

        if mask is not None:
            pair_mask = ops.expand_dims(mask, axis=1) * ops.expand_dims(mask, axis=2)
            diff = ops.where(pair_mask > 0, diff, ops.zeros_like(diff))

        return self.embedding(diff)


class EsmFoldAttention(layers.Layer):
    """
    Multi-head attention supporting arbitrary broadcastable external pairwise biases and gating.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = layers.Dense(c_hidden * no_heads, use_bias=False, name="linear_q")
        self.linear_k = layers.Dense(c_hidden * no_heads, use_bias=False, name="linear_k")
        self.linear_v = layers.Dense(c_hidden * no_heads, use_bias=False, name="linear_v")
        self.linear_o = layers.Dense(c_q, use_bias=True, name="linear_o")
        if gating:
            self.linear_g = layers.Dense(c_hidden * no_heads, use_bias=True, name="linear_g")
        else:
            self.linear_g = None

    def call(self, q_x, kv_x, biases=None):
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # Reshape to [*, Q/K, H, C_hidden]
        q_shape = ops.shape(q)
        kv_shape = ops.shape(k)
        new_q_shape = tuple(q_shape[:-1]) + (self.no_heads, self.c_hidden)
        new_kv_shape = tuple(kv_shape[:-1]) + (self.no_heads, self.c_hidden)

        q = ops.reshape(q, new_q_shape)
        k = ops.reshape(k, new_kv_shape)
        v = ops.reshape(v, new_kv_shape)

        # Transpose to [*, H, Q/K, C_hidden]
        q = ops.swapaxes(q, -2, -3)
        k = ops.swapaxes(k, -2, -3)
        v = ops.swapaxes(v, -2, -3)

        q = q * (1.0 / math.sqrt(self.c_hidden))

        # [*, H, Q, K]
        attn_scores = ops.matmul(q, ops.swapaxes(k, -1, -2))
        if biases is not None:
            for b in biases:
                attn_scores = attn_scores + b

        attn_probs = ops.softmax(attn_scores, axis=-1)

        # [*, H, Q, C_hidden]
        output = ops.matmul(attn_probs, v)
        # [*, Q, H, C_hidden]
        output = ops.swapaxes(output, -2, -3)

        if self.linear_g is not None:
            g = ops.sigmoid(self.linear_g(q_x))
            g = ops.reshape(g, new_q_shape)
            output = output * g

        out_shape = tuple(ops.shape(output)[:-2]) + (self.no_heads * self.c_hidden,)
        output = ops.reshape(output, out_shape)
        return self.linear_o(output)


class EsmFoldSelfAttention(layers.Layer):
    """
    Self-attention layer with pairwise external attention bias and gating.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_width: int,
        gated: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width
        self.gated = gated
        self.rescale_factor = 1.0 / math.sqrt(head_width)

        self.proj = layers.Dense(embed_dim * 3, use_bias=False, name="proj")
        self.o_proj = layers.Dense(embed_dim, use_bias=True, name="o_proj")
        if gated:
            self.g_proj = layers.Dense(embed_dim, use_bias=True, name="g_proj")
        else:
            self.g_proj = None

    def call(self, x, mask=None, bias=None):
        batch_dim = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        qkv = self.proj(x)
        qkv = ops.reshape(qkv, (batch_dim, seq_len, self.num_heads, 3 * self.head_width))
        qkv = ops.transpose(qkv, (0, 2, 1, 3))  # [B, H, L, 3 * head_width]

        q, k, v = ops.split(qkv, 3, axis=-1)
        q = q * self.rescale_factor

        a = ops.matmul(q, ops.swapaxes(k, -1, -2))  # [B, H, L, L]

        if bias is not None:
            # bias: [B, L, L, H] -> [B, H, L, L]
            a = a + ops.transpose(bias, (0, 3, 1, 2))

        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L]
            m = ops.cast(ops.expand_dims(ops.expand_dims(mask, axis=1), axis=1), dtype=a.dtype)
            a = a + (1.0 - m) * -1e9

        attn_probs = ops.softmax(a, axis=-1)
        o = ops.matmul(attn_probs, v)  # [B, H, L, head_width]
        o = ops.transpose(o, (0, 2, 1, 3))  # [B, L, H, head_width]
        o = ops.reshape(o, (batch_dim, seq_len, self.embed_dim))

        if self.g_proj is not None:
            g = ops.sigmoid(self.g_proj(x))
            o = o * g

        return self.o_proj(o), attn_probs


class EsmFoldSequenceToPair(layers.Layer):
    """
    Projects sequence representations into residue-residue pair representations.
    """

    def __init__(self, sequence_state_dim: int, inner_dim: int, pairwise_state_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layernorm")
        self.proj = layers.Dense(inner_dim * 2, use_bias=True, name="proj")
        self.o_proj = layers.Dense(pairwise_state_dim, use_bias=True, name="o_proj")

    def call(self, sequence_state):
        s = self.layernorm(sequence_state)
        s = self.proj(s)
        q, k = ops.split(s, 2, axis=-1)

        # Outer product: [B, 1, L, inner_dim] and [B, L, 1, inner_dim]
        prod = ops.expand_dims(q, axis=1) * ops.expand_dims(k, axis=2)
        diff = ops.expand_dims(q, axis=1) - ops.expand_dims(k, axis=2)

        x = ops.concatenate([prod, diff], axis=-1)
        return self.o_proj(x)


class EsmFoldPairToSequence(layers.Layer):
    """
    Projects pair representations into attention bias for sequence self-attention.
    """

    def __init__(self, pairwise_state_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layernorm")
        self.linear = layers.Dense(num_heads, use_bias=False, name="linear")

    def call(self, pairwise_state):
        z = self.layernorm(pairwise_state)
        return self.linear(z)


class EsmFoldResidueMLP(layers.Layer):
    """
    Feed-forward transition MLP for residue and pair representations.
    """

    def __init__(self, embed_dim: int, inner_dim: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.layernorm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="0")
        self.linear_1 = layers.Dense(inner_dim, use_bias=True, name="1")
        self.relu = layers.Activation("relu")
        self.linear_2 = layers.Dense(embed_dim, use_bias=True, name="3")
        self.dropout = layers.Dropout(dropout) if dropout > 0 else None

    def call(self, x):
        h = self.layernorm(x)
        h = self.linear_1(h)
        h = self.relu(h)
        h = self.linear_2(h)
        if self.dropout is not None:
            h = self.dropout(h)
        return x + h


class EsmFoldTriangleMultiplicativeUpdate(layers.Layer):
    """
    Triangular multiplicative edge update (outgoing or incoming).
    """

    def __init__(self, pairwise_state_dim: int, outgoing: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.pairwise_state_dim = pairwise_state_dim
        self.outgoing = outgoing

        self.layer_norm_in = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm_in")
        self.linear_a_p = layers.Dense(pairwise_state_dim, use_bias=True, name="linear_a_p")
        self.linear_a_g = layers.Dense(pairwise_state_dim, use_bias=True, name="linear_a_g")
        self.linear_b_p = layers.Dense(pairwise_state_dim, use_bias=True, name="linear_b_p")
        self.linear_b_g = layers.Dense(pairwise_state_dim, use_bias=True, name="linear_b_g")
        self.linear_g = layers.Dense(pairwise_state_dim, use_bias=True, name="linear_g")
        self.linear_z = layers.Dense(pairwise_state_dim, use_bias=True, name="linear_z")
        self.layer_norm_out = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm_out")

    def call(self, z, mask=None):
        if mask is None:
            mask = ops.ones(ops.shape(z)[:-1], dtype=z.dtype)

        mask_exp = ops.expand_dims(mask, axis=-1)

        z_norm = self.layer_norm_in(z)
        a = mask_exp * ops.sigmoid(self.linear_a_g(z_norm)) * self.linear_a_p(z_norm)
        b = mask_exp * ops.sigmoid(self.linear_b_g(z_norm)) * self.linear_b_p(z_norm)

        if self.outgoing:
            a_perm = ops.transpose(a, (0, 3, 1, 2))  # [B, C, I, K]
            b_perm = ops.transpose(b, (0, 3, 2, 1))  # [B, C, K, J]
        else:
            a_perm = ops.transpose(a, (0, 3, 2, 1))  # [B, C, K, I]
            b_perm = ops.transpose(b, (0, 3, 1, 2))  # [B, C, J, K]

        p = ops.matmul(a_perm, b_perm)
        p = ops.transpose(p, (0, 2, 3, 1))  # [B, I, J, C]

        p = self.layer_norm_out(p)
        p = self.linear_z(p)
        g = ops.sigmoid(self.linear_g(z_norm))
        return p * g


class EsmFoldTriangleAttention(layers.Layer):
    """
    Triangular self-attention (starting node or ending node) with pairwise bias.
    """

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        no_heads: int,
        starting: bool = True,
        inf: float = 1e9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm")
        self.linear = layers.Dense(no_heads, use_bias=False, name="linear")
        self.mha = EsmFoldAttention(c_in, c_in, c_in, c_hidden, no_heads, gating=True, name="mha")

    def call(self, x, mask=None):
        if mask is None:
            mask = ops.ones(ops.shape(x)[:-1], dtype=x.dtype)

        if not self.starting:
            x = ops.swapaxes(x, -2, -3)
            mask = ops.swapaxes(mask, -1, -2)

        x_norm = self.layer_norm(x)

        mask = ops.cast(mask, dtype=x.dtype)
        # [*, I, 1, 1, J]
        mask_bias = ops.expand_dims(ops.expand_dims(self.inf * (mask - 1.0), axis=-2), axis=-2)

        # [*, H, I, J]
        tri_bias = self.linear(x_norm)
        # Permute to [*, H, I, J]
        tri_bias = ops.transpose(tri_bias, (0, 3, 1, 2))
        tri_bias = ops.expand_dims(tri_bias, axis=-4)  # [*, 1, H, I, J]

        biases = [mask_bias, tri_bias]
        out = self.mha(x_norm, x_norm, biases=biases)

        if not self.starting:
            out = ops.swapaxes(out, -2, -3)

        return out


class EsmFoldTriangularSelfAttentionBlock(layers.Layer):
    """
    Complete Evoformer-style Folding Trunk Block for pairwise and sequence interaction.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.config = config

        # Sequence state updates
        self.pair_to_sequence = EsmFoldPairToSequence(
            config.pairwise_state_dim,
            config.sequence_state_dim // config.sequence_head_width,
            name="pair_to_sequence",
        )
        self.layernorm_1 = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layernorm_1")
        self.seq_attention = EsmFoldSelfAttention(
            config.sequence_state_dim,
            config.sequence_state_dim // config.sequence_head_width,
            config.sequence_head_width,
            gated=True,
            name="seq_attention",
        )
        self.mlp_seq = EsmFoldResidueMLP(
            config.sequence_state_dim,
            config.sequence_state_dim * 4,
            dropout=config.dropout,
            name="mlp_seq",
        )

        # Pairwise state updates
        self.sequence_to_pair = EsmFoldSequenceToPair(
            config.sequence_state_dim,
            config.pairwise_state_dim // 2,
            config.pairwise_state_dim,
            name="sequence_to_pair",
        )
        self.tri_mul_out = EsmFoldTriangleMultiplicativeUpdate(
            config.pairwise_state_dim, outgoing=True, name="tri_mul_out"
        )
        self.tri_mul_in = EsmFoldTriangleMultiplicativeUpdate(
            config.pairwise_state_dim, outgoing=False, name="tri_mul_in"
        )
        pairwise_num_heads = config.pairwise_state_dim // config.pairwise_head_width
        self.tri_att_start = EsmFoldTriangleAttention(
            config.pairwise_state_dim,
            config.pairwise_head_width,
            pairwise_num_heads,
            starting=True,
            name="tri_att_start",
        )
        self.tri_att_end = EsmFoldTriangleAttention(
            config.pairwise_state_dim,
            config.pairwise_head_width,
            pairwise_num_heads,
            starting=False,
            name="tri_att_end",
        )
        self.mlp_pair = EsmFoldResidueMLP(
            config.pairwise_state_dim,
            config.pairwise_state_dim * 4,
            dropout=config.dropout,
            name="mlp_pair",
        )

    def call(self, sequence_state, pairwise_state, mask=None):
        # 1. Update sequence state
        bias = self.pair_to_sequence(pairwise_state)
        y = self.layernorm_1(sequence_state)
        attn_out, _ = self.seq_attention(y, mask=mask, bias=bias)
        sequence_state = sequence_state + attn_out
        sequence_state = self.mlp_seq(sequence_state)

        # 2. Update pairwise state
        pairwise_state = pairwise_state + self.sequence_to_pair(sequence_state)

        tri_mask = None
        if mask is not None:
            tri_mask = ops.expand_dims(mask, axis=2) * ops.expand_dims(mask, axis=1)

        pairwise_state = pairwise_state + self.tri_mul_out(pairwise_state, mask=tri_mask)
        pairwise_state = pairwise_state + self.tri_mul_in(pairwise_state, mask=tri_mask)
        pairwise_state = pairwise_state + self.tri_att_start(pairwise_state, mask=tri_mask)
        pairwise_state = pairwise_state + self.tri_att_end(pairwise_state, mask=tri_mask)
        pairwise_state = self.mlp_pair(pairwise_state)

        return sequence_state, pairwise_state


class EsmFoldBackboneUpdate(layers.Layer):
    """
    Predicts 6D rigid frame update vector (quaternion update + 3D translation).
    """

    def __init__(self, sequence_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.linear = layers.Dense(6, use_bias=True, name="linear")

    def call(self, s):
        return self.linear(s)


class EsmFoldAngleResnetBlock(layers.Layer):
    """Residual block inside Angle ResNet."""

    def __init__(self, resnet_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.linear_1 = layers.Dense(resnet_dim, use_bias=True, name="linear_1")
        self.linear_2 = layers.Dense(resnet_dim, use_bias=True, name="linear_2")
        self.relu = layers.Activation("relu")

    def call(self, a):
        s_initial = a
        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)
        return a + s_initial


class EsmFoldAngleResnet(layers.Layer):
    """
    Predicts backbone and sidechain torsion angles from sequence representations.
    """

    def __init__(
        self,
        sequence_dim: int,
        resnet_dim: int,
        num_resnet_blocks: int,
        num_angles: int = 7,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_angles = num_angles
        self.epsilon = epsilon

        self.linear_in = layers.Dense(resnet_dim, use_bias=True, name="linear_in")
        self.linear_initial = layers.Dense(resnet_dim, use_bias=True, name="linear_initial")
        self.blocks = [
            EsmFoldAngleResnetBlock(resnet_dim, name=f"layers_{i}")
            for i in range(num_resnet_blocks)
        ]
        self.linear_out = layers.Dense(num_angles * 2, use_bias=True, name="linear_out")
        self.relu = layers.Activation("relu")

    def call(self, s, s_initial):
        s_init = self.relu(s_initial)
        s_init = self.linear_initial(s_init)
        s_feat = self.relu(s)
        s_feat = self.linear_in(s_feat)
        s_feat = s_feat + s_init

        for block in self.blocks:
            s_feat = block(s_feat)

        s_feat = self.relu(s_feat)
        s_out = self.linear_out(s_feat)

        out_shape = tuple(ops.shape(s_out)[:-1]) + (self.num_angles, 2)
        s_angles = ops.reshape(s_out, out_shape)
        unnormalized_angles = s_angles

        # Normalize (sin, cos) to unit circle
        norm = ops.sqrt(ops.maximum(ops.sum(ops.square(s_angles), axis=-1, keepdims=True), self.epsilon))
        normalized_angles = s_angles / norm

        return unnormalized_angles, normalized_angles


class EsmFoldStructureModuleTransitionLayer(layers.Layer):
    """Transition layer inside StructureModule."""

    def __init__(self, sequence_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.linear_1 = layers.Dense(sequence_dim, use_bias=True, name="linear_1")
        self.linear_2 = layers.Dense(sequence_dim, use_bias=True, name="linear_2")
        self.linear_3 = layers.Dense(sequence_dim, use_bias=True, name="linear_3")
        self.relu = layers.Activation("relu")

    def call(self, s):
        s_init = s
        s = self.relu(self.linear_1(s))
        s = self.relu(self.linear_2(s))
        s = self.linear_3(s)
        return s + s_init


class EsmFoldStructureModuleTransition(layers.Layer):
    """Transition module inside StructureModule."""

    def __init__(
        self,
        sequence_dim: int,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transition_layers = [
            EsmFoldStructureModuleTransitionLayer(sequence_dim, name=f"layers_{i}")
            for i in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm")

    def call(self, s):
        for l in self.transition_layers:
            s = l(s)
        if self.dropout is not None:
            s = self.dropout(s)
        return self.layer_norm(s)


class EsmFoldInvariantPointAttention(layers.Layer):
    """
    Invariant Point Attention (IPA) combining scalar self-attention, 3D point geometric
    attention in residue reference frames, and pairwise representation bias.
    """

    _allow_non_tensor_positional_args = True

    def __init__(
        self,
        sequence_dim: int = 384,
        pairwise_dim: int = 128,
        ipa_dim: int = 16,
        num_heads: int = 12,
        num_qk_points: int = 4,
        num_v_points: int = 8,
        inf: float = 1e5,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._allow_non_tensor_positional_args = True
        self.sequence_dim = sequence_dim
        self.pairwise_dim = pairwise_dim
        self.ipa_dim = ipa_dim
        self.num_heads = num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.inf = inf
        self.epsilon = epsilon

        # Scalar projections
        self.linear_q = layers.Dense(num_heads * ipa_dim, use_bias=False, name="linear_q")
        self.linear_kv = layers.Dense(2 * num_heads * ipa_dim, use_bias=False, name="linear_kv")

        # Point projections
        self.linear_q_points = layers.Dense(num_heads * num_qk_points * 3, use_bias=False, name="linear_q_points")
        self.linear_kv_points = layers.Dense(num_heads * (num_qk_points + num_v_points) * 3, use_bias=False, name="linear_kv_points")

        # Pairwise bias projection
        self.linear_b = layers.Dense(num_heads, use_bias=False, name="linear_b")

        # Output projection
        concat_out_dim = num_heads * (pairwise_dim + ipa_dim + num_v_points * 4)
        self.linear_out = layers.Dense(sequence_dim, use_bias=True, name="linear_out")

    def build(self, input_shape):
        self.head_weights = self.add_weight(
            name="head_weights",
            shape=(self.num_heads,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, s, z, r: geometry.Rigid, mask=None):
        batch_dim = ops.shape(s)[0]
        n_res = ops.shape(s)[1]

        # 1. Scalar queries, keys, values
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        q = ops.reshape(q, (batch_dim, n_res, self.num_heads, self.ipa_dim))
        kv = ops.reshape(kv, (batch_dim, n_res, self.num_heads, 2 * self.ipa_dim))
        k, v = ops.split(kv, 2, axis=-1)

        # 2. Point queries, keys, values
        # [B, N, H * P_q * 3]
        q_pts = self.linear_q_points(s)
        q_pts = ops.reshape(q_pts, (batch_dim, n_res, self.num_heads * self.num_qk_points, 3))
        # Transform local points into global space using rigid frames
        # r[..., None]: [B, N, 1]
        q_pts = r[..., None].apply(q_pts)
        q_pts = ops.reshape(q_pts, (batch_dim, n_res, self.num_heads, self.num_qk_points, 3))

        kv_pts = self.linear_kv_points(s)
        kv_pts = ops.reshape(kv_pts, (batch_dim, n_res, self.num_heads * (self.num_qk_points + self.num_v_points), 3))
        kv_pts = r[..., None].apply(kv_pts)
        kv_pts = ops.reshape(kv_pts, (batch_dim, n_res, self.num_heads, self.num_qk_points + self.num_v_points, 3))

        k_pts = kv_pts[..., :self.num_qk_points, :]
        v_pts = kv_pts[..., self.num_qk_points:, :]

        # 3. Scalar attention: [B, H, N, N]
        q_t = ops.transpose(q, (0, 2, 1, 3))
        k_t = ops.transpose(k, (0, 2, 1, 3))
        v_t = ops.transpose(v, (0, 2, 1, 3))

        a_scalar = ops.matmul(q_t, ops.swapaxes(k_t, -1, -2))
        a_scalar = a_scalar * math.sqrt(1.0 / (3.0 * self.ipa_dim))

        # Pair bias: [B, N, N, H] -> [B, H, N, N]
        if z is not None:
            b = self.linear_b(z)
            b_t = ops.transpose(b, (0, 3, 1, 2))
            a_scalar = a_scalar + math.sqrt(1.0 / 3.0) * b_t

        # Point distance attention: ||q_i - k_j||^2
        # q_pts: [B, N, 1, H, P_q, 3], k_pts: [B, 1, N, H, P_q, 3]
        q_pts_exp = ops.expand_dims(q_pts, axis=2)
        k_pts_exp = ops.expand_dims(k_pts, axis=1)
        pt_dist_sq = ops.sum(ops.square(q_pts_exp - k_pts_exp), axis=-1)  # [B, N, N, H, P_q]

        head_w = ops.softplus(self.head_weights)
        head_w = head_w * math.sqrt(1.0 / (3.0 * (self.num_qk_points * 9.0 / 2.0)))
        # reshape head_w to [1, 1, 1, H, 1]
        head_w = ops.reshape(head_w, (1, 1, 1, self.num_heads, 1))

        pt_att = pt_dist_sq * head_w
        pt_att = ops.sum(pt_att, axis=-1) * (-0.5)  # [B, N, N, H]
        pt_att = ops.transpose(pt_att, (0, 3, 1, 2))  # [B, H, N, N]

        a = a_scalar + pt_att

        if mask is not None:
            # mask: [B, N] -> [B, 1, N, N]
            square_mask = ops.expand_dims(mask, axis=1) * ops.expand_dims(mask, axis=2)
            square_mask = ops.cast(square_mask, dtype=a.dtype)
            mask_bias = self.inf * (square_mask - 1.0)
            a = a + ops.expand_dims(mask_bias, axis=1)

        attn_weights = ops.softmax(a, axis=-1)  # [B, H, N, N]

        # 4. Aggregations
        # Scalar output: [B, H, N, ipa_dim] -> [B, N, H * ipa_dim]
        o_scalar = ops.matmul(attn_weights, v_t)
        o_scalar = ops.transpose(o_scalar, (0, 2, 1, 3))
        o_scalar = ops.reshape(o_scalar, (batch_dim, n_res, self.num_heads * self.ipa_dim))

        # Point output:
        # attn_weights: [B, H, N_q, N_k]
        # v_pts: [B, N_k, H, P_v, 3] -> [B, H, 3, N_k, P_v]
        v_pts_perm = ops.transpose(v_pts, (0, 2, 4, 1, 3))  # [B, H, 3, N_k, P_v]
        # attn_weights expand: [B, H, 1, N_q, N_k, 1]
        # v_pts expand: [B, H, 3, 1, N_k, P_v]
        # Weighted sum over N_k (axis=-2):
        attn_exp = ops.expand_dims(ops.expand_dims(attn_weights, axis=2), axis=-1)
        v_exp = ops.expand_dims(v_pts_perm, axis=3)
        o_pt_global = ops.sum(attn_exp * v_exp, axis=-2)  # [B, H, 3, N_q, P_v]
        # Permute to [B, N_q, H, P_v, 3]
        o_pt_global = ops.transpose(o_pt_global, (0, 3, 1, 4, 2))

        # Invert apply through local residue frames:
        # r[..., None, None]: [B, N, 1, 1]
        o_pt_local = r[..., None, None].invert_apply(o_pt_global)

        # Norms: [B, N, H * P_v]
        o_pt_norm = ops.sqrt(ops.sum(ops.square(o_pt_local), axis=-1) + self.epsilon)
        o_pt_norm = ops.reshape(o_pt_norm, (batch_dim, n_res, self.num_heads * self.num_v_points))

        # Flatten point coordinates: [B, N, H * P_v * 3]
        o_pt_flat = ops.reshape(o_pt_local, (batch_dim, n_res, self.num_heads * self.num_v_points * 3))

        # Pair representation aggregation:
        # attn_weights: [B, H, N_q, N_k]
        # z: [B, N_q, N_k, C_z]
        if z is not None:
            o_pair = ops.einsum("bhqk,bqkc->bqhc", attn_weights, z)
            o_pair = ops.reshape(o_pair, (batch_dim, n_res, self.num_heads * self.pairwise_dim))
        else:
            o_pair = ops.zeros((batch_dim, n_res, self.num_heads * self.pairwise_dim), dtype=s.dtype)

        # 5. Concat all representations & project
        concat_feats = ops.concatenate([o_scalar, o_pt_flat, o_pt_norm, o_pair], axis=-1)
        return self.linear_out(concat_feats)


class EsmFoldStructureModule(layers.Layer):
    """
    Iterative 3D coordinate generation module using Invariant Point Attention,
    Backbone Update, Angle ResNet, and frame-to-atom coordinate projection.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.config = config

        self.layer_norm_s = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm_s")
        self.layer_norm_z = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm_z")
        self.linear_in = layers.Dense(config.sequence_dim, use_bias=True, name="linear_in")

        self.ipa = EsmFoldInvariantPointAttention(
            sequence_dim=config.sequence_dim,
            pairwise_dim=config.pairwise_dim,
            ipa_dim=config.ipa_dim,
            num_heads=config.num_heads_ipa,
            num_qk_points=config.num_qk_points,
            num_v_points=config.num_v_points,
            inf=config.inf,
            epsilon=config.epsilon,
            name="ipa",
        )
        self.layer_norm_ipa = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm_ipa")
        self.transition = EsmFoldStructureModuleTransition(
            config.sequence_dim,
            num_layers=config.num_transition_layers,
            dropout_rate=config.dropout_rate,
            name="transition",
        )
        self.bb_update = EsmFoldBackboneUpdate(config.sequence_dim, name="bb_update")
        self.angle_resnet = EsmFoldAngleResnet(
            config.sequence_dim,
            config.resnet_dim,
            config.num_resnet_blocks,
            num_angles=config.num_angles,
            epsilon=config.epsilon,
            name="angle_resnet",
        )

        # Preload standard chemical frame constants as non-trainable weights
        self.default_frames = ops.convert_to_tensor(rc.restype_rigid_group_default_frame, dtype="float32")
        self.group_idx = ops.convert_to_tensor(rc.restype_atom14_to_rigid_group, dtype="int32")
        self.atom_mask = ops.convert_to_tensor(rc.restype_atom14_mask, dtype="float32")
        self.lit_positions = ops.convert_to_tensor(rc.restype_atom14_rigid_group_positions, dtype="float32")

    def call(self, evoformer_output_dict, aatype, mask=None):
        s = evoformer_output_dict["single"]
        z = evoformer_output_dict["pair"]

        if mask is None:
            mask = ops.ones(ops.shape(s)[:-1], dtype=s.dtype)

        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)

        s_initial = s
        s = self.linear_in(s)

        # Initialize identity rigid body transformations
        rigids = geometry.Rigid.identity(ops.shape(s)[:-1], dtype=s.dtype)

        outputs_frames = []
        outputs_sidechain_frames = []
        outputs_unnormalized_angles = []
        outputs_angles = []
        outputs_positions = []
        outputs_states = []

        for i in range(self.config.num_blocks):
            # 1. Invariant Point Attention
            s = s + self.ipa(s, z, rigids, mask=mask)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # 2. Backbone update
            rigids = rigids.compose_q_update_vec(self.bb_update(s))

            # 3. Convert to rotation matrix and scale translation
            backb_to_global = geometry.Rigid(
                geometry.Rotation(rot_mats=rigids.get_rots().get_rot_mats(), quats=None),
                rigids.get_trans(),
            )
            backb_to_global = backb_to_global.scale_translation(self.config.trans_scale_factor)

            # 4. Torsion angle prediction
            unnorm_angles, angles = self.angle_resnet(s, s_initial)

            # 5. Global frames & atomic positions
            all_frames_to_global = feats.torsion_angles_to_frames(
                backb_to_global, angles, aatype, self.default_frames
            )
            pred_xyz = feats.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global,
                aatype,
                self.default_frames,
                self.group_idx,
                self.atom_mask,
                self.lit_positions,
            )

            scaled_rigids = rigids.scale_translation(self.config.trans_scale_factor)

            outputs_frames.append(scaled_rigids.to_tensor_7())
            outputs_sidechain_frames.append(all_frames_to_global.to_tensor_4x4())
            outputs_unnormalized_angles.append(unnorm_angles)
            outputs_angles.append(angles)
            outputs_positions.append(pred_xyz)
            outputs_states.append(s)

            # Stop rotation gradient between blocks
            rigids = rigids.stop_rot_gradient()

        return {
            "frames": ops.stack(outputs_frames, axis=1),
            "sidechain_frames": ops.stack(outputs_sidechain_frames, axis=1),
            "unnormalized_angles": ops.stack(outputs_unnormalized_angles, axis=1),
            "angles": ops.stack(outputs_angles, axis=1),
            "positions": ops.stack(outputs_positions, axis=1),
            "states": ops.stack(outputs_states, axis=1),
            "single": s,
        }


class EsmFoldingTrunk(layers.Layer):
    """
    Folding Trunk combining recycling, triangular attention blocks, and Structure Module.
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.config = config
        self.max_recycles = config.max_recycles
        self.recycle_bins = config.recycle_bins

        self.pairwise_positional_embedding = EsmFoldRelativePosition(
            position_bins=config.position_bins,
            pairwise_state_dim=config.pairwise_state_dim,
            name="pairwise_positional_embedding",
        )
        self.blocks = [
            EsmFoldTriangularSelfAttentionBlock(config, name=f"blocks_{i}")
            for i in range(config.num_blocks)
        ]

        self.recycle_s_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="recycle_s_norm")
        self.recycle_z_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="recycle_z_norm")
        self.recycle_disto = layers.Embedding(
            input_dim=config.recycle_bins,
            output_dim=config.pairwise_state_dim,
            name="recycle_disto",
        )

        self.trunk2sm_s = layers.Dense(config.structure_module.sequence_dim, use_bias=True, name="trunk2sm_s")
        self.trunk2sm_z = layers.Dense(config.structure_module.pairwise_dim, use_bias=True, name="trunk2sm_z")

        self.structure_module = EsmFoldStructureModule(config.structure_module, name="structure_module")

    @staticmethod
    def distogram(coords, min_bin=3.375, max_bin=21.375, num_bins=15):
        # coords: [..., 3, 3] corresponding to N, CA, C
        boundaries = ops.linspace(min_bin, max_bin, num_bins - 1)
        boundaries = ops.square(boundaries)

        N = coords[..., 0, :]
        CA = coords[..., 1, :]
        C = coords[..., 2, :]

        # Infer CB coordinates
        b = CA - N
        c = C - CA
        # Cross product: b x c
        a = ops.cross(b, c)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        # Pairwise distance matrix
        dists = ops.sum(ops.square(ops.expand_dims(CB, axis=-2) - ops.expand_dims(CB, axis=-3)), axis=-1, keepdims=True)
        # Sum over conditions dists > boundaries
        bins = ops.sum(ops.cast(dists > boundaries, dtype="int32"), axis=-1)
        return bins

    def call(self, seq_feats, pair_feats, true_aa, residx, mask=None, no_recycles=None):
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        if no_recycles is None:
            no_recycles = self.max_recycles
        no_recycles = no_recycles + 1

        def trunk_iter(s, z, residx_in, mask_in):
            z = z + self.pairwise_positional_embedding(residx_in, mask=mask_in)
            for block in self.blocks:
                s, z = block(s, z, mask=mask_in)
            return s, z

        s_s = s_s_0
        s_z = s_z_0
        recycle_s = ops.zeros_like(s_s)
        recycle_z = ops.zeros_like(s_z)
        recycle_bins = ops.zeros(ops.shape(s_z)[:-1], dtype="int32")

        structure = {}
        for recycle_idx in range(no_recycles):
            rec_s_norm = self.recycle_s_norm(recycle_s)
            rec_z_norm = self.recycle_z_norm(recycle_z) + self.recycle_disto(recycle_bins)

            s_s, s_z = trunk_iter(s_s_0 + rec_s_norm, s_z_0 + rec_z_norm, residx, mask)

            structure = self.structure_module(
                {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                true_aa,
                mask=mask,
            )

            recycle_s = s_s
            recycle_z = s_z
            # Distogram of last block N, CA, C coordinates
            last_positions = structure["positions"][:, -1]
            recycle_bins = self.distogram(last_positions[..., :3, :], num_bins=self.recycle_bins)

        structure["s_s"] = s_s
        structure["s_z"] = s_z
        return structure


class LDDTHead(layers.Layer):
    """Predicts per-residue per-atom categorical lDDT distribution."""

    def __init__(self, sequence_dim: int, hidden_dim: int, lddt_bins: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.lddt_bins = lddt_bins
        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-5, name="0")
        self.linear_1 = layers.Dense(hidden_dim, use_bias=True, name="1")
        self.linear_2 = layers.Dense(hidden_dim, use_bias=True, name="2")
        self.linear_3 = layers.Dense(37 * lddt_bins, use_bias=True, name="3")

    def call(self, states):
        # states: [num_blocks, B, L, sequence_dim]
        h = self.layer_norm(states)
        h = self.linear_1(h)
        h = self.linear_2(h)
        h = self.linear_3(h)
        shape = tuple(ops.shape(states)[:-1]) + (37, self.lddt_bins)
        return ops.reshape(h, shape)


def categorical_lddt(logits, bins: int = 50):
    """
    Computes expected value of lDDT scores across bins (0..1).
    """
    bin_tensor = ops.linspace(0.0, 1.0, bins + 1)
    v_bins = (bin_tensor[:-1] + bin_tensor[1:]) / 2.0  # [bins]
    probs = ops.softmax(logits, axis=-1)  # [..., 37, bins]
    v_bins = ops.reshape(v_bins, (1,) * (len(ops.shape(probs)) - 1) + (bins,))
    return ops.sum(probs * v_bins, axis=-1)
