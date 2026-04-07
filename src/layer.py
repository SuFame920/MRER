
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import warnings
from itertools import accumulate

class EnhancedLSTM(torch.nn.Module):
    """
    A wrapper for different recurrent dropout implementations, which
    pytorch currently doesn't support nativly.

    Uses multilayer, bidirectional lstms with dropout between layers
    and time steps in a variational manner.

    "allen" reimplements a lstm with hidden to hidden dropout, thus disabling
    CUDNN. Can only be used in bidirectional mode.
    `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`

    "drop_connect" uses default implemetation, but monkey patches the hidden to hidden
    weight matrices instead.
    `Regularizing and Optimizing LSTM Language Models
        <https://arxiv.org/abs/1708.02182>`

    "native" ignores dropout and uses the default implementation.
    """

    def __init__(self,
                 lstm_type,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.lstm_type = lstm_type

        if lstm_type == "drop_connect":
            self.provider = WeightDropLSTM(
                input_size,
                hidden_size,
                num_layers,
                ff_dropout,
                recurrent_dropout,
                bidirectional=bidirectional)
        elif lstm_type == "native":
            self.provider = torch.nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True)
        else:
            raise Exception(lstm_type + " is an invalid lstm type")

    # Expects unpacked inputs in format (batch, seq, features)
    def forward(self, inputs, hidden, lengths):
        seq_len = inputs.shape[1]
        if self.lstm_type in ["allen", "native"]:
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True)

            output, _ = self.provider(packed, hidden)

            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

            return output
        elif self.lstm_type == "drop_connect":
            return self.provider(inputs, lengths, seq_len)


class WeightDropLSTM(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.locked_dropout = LockedDropout()
        self.lstms = [
            torch.nn.LSTM(
                input_size
                if l == 0 else hidden_size * (1 + int(bidirectional)),
                hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True) for l in range(num_layers)
        ]
        if recurrent_dropout:
            self.lstms = [
                WeightDrop(lstm, ['weight_hh_l0'], dropout=recurrent_dropout)
                for lstm in self.lstms
            ]

        self.lstms = torch.nn.ModuleList(self.lstms)
        self.ff_dropout = ff_dropout
        self.num_layers = num_layers

    def forward(self, input, lengths, seq_len):
        """Expects input in format (batch, seq, features)"""
        output = input
        for lstm in self.lstms:
            output = self.locked_dropout(
                output, batch_first=True, p=self.ff_dropout)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                output, lengths, batch_first=True, enforce_sorted=False)
            output, _ = lstm(packed, None)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len)

        return output

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_first=False, p=0.5):
        if not self.training or not p:
            return x
        mask_shape = (x.size(0), 1, x.size(2)) if batch_first else (1,
                                                                    x.size(1),
                                                                    x.size(2))

        mask = x.data.new(*mask_shape).bernoulli_(1 - p).div_(1 - p)
        return mask * x



class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        if hasattr(module, "bidirectional") and module.bidirectional:
            self.weights.extend(
                [weight + "_reverse" for weight in self.weights])

        self.dropout = dropout
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')

            w = None
            mask = torch.ones(1, raw_w.size(1))
            if raw_w.is_cuda: mask = mask.to(raw_w.device)
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout, training=self.training)
            w = mask.expand_as(raw_w) * raw_w
            self.module._parameters[name_w] = w

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # Ignore lack of flattening warning
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1) / self.n_in ** self.scale

        return s


class RoEmbedding(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()
        self.cfg = cfg
        self.rope_qk_dim = int(cfg.get("rope_qk_dim", cfg.get("inner_dim", 256)))
        # 绗竴涓?浠ｈ〃锛屽垏鍒嗘垚涓ゅ潡锛涚浜屼釜2浠ｈ〃锛屼竴鍏辨湁2涓爣绛撅紙姝?璐燂級
        self.dense = nn.Linear(input_size, self.rope_qk_dim * 2 * 2)

    def custom_sinusoidal_position_embedding(self, token_index, pos_type=1):
        """
        See RoPE paper: https://arxiv.org/abs/2104.09864
        """
        output_dim = self.rope_qk_dim
        position_ids = token_index.unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float).to(self.cfg.device)
        if pos_type == 0:
            indices = torch.pow(10000, -2 * indices / output_dim)
        else:
            indices = torch.pow(15, -2 * indices / output_dim)  #15
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((1, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (1, len(token_index), output_dim))
        embeddings = embeddings.squeeze(0)
        return embeddings
    
    def get_ro_embedding(self, qw: torch.Tensor, kw: torch.Tensor, token_index):
        """_summary_
        Parameters
        ----------
        qw : torch.Tensor, (seq_len, class_nums, hidden_size)
        kw : torch.Tensor, (seq_len, class_nums, hidden_size)
        """

        x, y = token_index, token_index

        x_pos_emb = self.custom_sinusoidal_position_embedding(x)
        y_pos_emb = self.custom_sinusoidal_position_embedding(y)

        x_pos_emb = x_pos_emb.unsqueeze(0)
        y_pos_emb = y_pos_emb.unsqueeze(0)

        x_cos_pos = x_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        x_sin_pos = x_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        cur_qw2 = cur_qw2.reshape(qw.shape)
        cur_qw = qw * x_cos_pos + cur_qw2 * x_sin_pos

        y_cos_pos = y_pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        y_sin_pos = y_pos_emb[...,  None, ::2].repeat_interleave(2, dim=-1)
        cur_kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        cur_kw2 = cur_kw2.reshape(kw.shape)
        cur_kw = kw * y_cos_pos + cur_kw2 * y_sin_pos

        # 恢复原版: 保持不进行 / math.sqrt(d) 归一化
        # 隐式地利用高维点积放大随机初始化层的梯度，弥补全局微小学习率的不足。
        pred_logits = torch.einsum('bmhd,bnhd->bmnh', cur_qw, cur_kw).contiguous()
        return pred_logits


    def _project_qk(self, sequence_outputs):
        outputs = self.dense(sequence_outputs)
        outputs = torch.split(outputs, self.rope_qk_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        q_utterance, k_utterance = torch.split(outputs, self.rope_qk_dim, dim=-1)
        return q_utterance, k_utterance

    def classify_matrix(self, sequence_outputs, input_labels, masks, key_sequence_outputs=None):
        if key_sequence_outputs is None:
            key_sequence_outputs = sequence_outputs

        q_utterance, _ = self._project_qk(sequence_outputs)
        _, k_utterance = self._project_qk(key_sequence_outputs)

        # q_utterance: batch_size, seq_len, class_nums, rope_qk_dim

        token_index = torch.arange(0, sequence_outputs.shape[1]).to(self.cfg.device)

        # ro_logits = []
        ro_logits = self.get_ro_embedding(q_utterance, k_utterance, token_index)

        pred_logits = ro_logits

        # ROP pair binary class weight: neg fixed at 1.0, pos from rop_pair_pos_weight.
        # Keep backward compatibility for old fields.
        legacy_weight = self.cfg.get("pair_pos_weight", None)
        if legacy_weight is None:
            legacy_weight = self.cfg.get("loss_weight", 1.0)
        if isinstance(legacy_weight, dict):
            legacy_weight = legacy_weight.get("rop", legacy_weight.get("pair", 1.0))
        pair_pos_weight = float(getattr(self.cfg, "rop_pair_pos_weight", legacy_weight))
        criterion = nn.CrossEntropyLoss(
            weight=sequence_outputs.new_tensor([1.0, pair_pos_weight])
        )

        active_loss = masks.view(-1) == 1
        active_logits = pred_logits.view(-1, pred_logits.shape[-1])[active_loss]
        active_labels = input_labels.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)
        if torch.isnan(loss):
            loss = sequence_outputs.new_tensor(0.0)
        elif torch.isinf(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

        return loss, pred_logits 


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        杈撳叆:
            features: 杈撳叆鏍锋湰鐨勭壒寰侊紝灏哄涓?[batch_size, hidden_dim].
            labels: 姣忎釜鏍锋湰鐨刧round truth鏍囩锛屽昂瀵告槸[batch_size].
            mask: 鐢ㄤ簬瀵规瘮瀛︿範鐨刴ask锛屽昂瀵镐负 [batch_size, batch_size], 濡傛灉鏍锋湰i鍜宩灞炰簬鍚屼竴涓猯abel锛岄偅涔坢ask_{i,j}=1
        杈撳嚭:
            loss鍊?        """
        # device = (torch.device('cuda')
                #   if features.is_cuda
                #   else torch.device('cpu'))
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # labels and mask cannot be both specified.
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # Unsupervised mode: only self-positive pairs on diagonal.
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Supervised mode: positives are samples sharing the same label.
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        绀轰緥:
        labels:
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 涓や釜鏍锋湰i,j鐨刲abel鐩哥瓑鏃讹紝mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]])
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 璁＄畻涓や袱鏍锋湰闂寸偣涔樼浉浼煎害
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits鏄痑nchor_dot_contrast鍑忓幓姣忎竴琛岀殑鏈€澶у€煎緱鍒扮殑鏈€缁堢浉浼煎害
        绀轰緥: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])
        '''
        # 鏋勫缓mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(mask.device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        浣嗘槸瀵逛簬璁＄畻Loss鑰岃█锛?i,i)浣嶇疆琛ㄧず鏍锋湰鏈韩鐨勭浉浼煎害锛屽Loss鏄病鐢ㄧ殑锛屾墍浠ヨmask鎺?        # 绗琲nd琛岀ind浣嶇疆濉厖涓?
        寰楀埌logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 闄や簡鑷繁涔嬪锛屾鏍锋湰鐨勪釜鏁? [2 0 2 2]
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")


        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        璁＄畻姝ｆ牱鏈钩鍧囩殑log-likelihood
        鑰冭檻鍒颁竴涓被鍒彲鑳藉彧鏈変竴涓牱鏈紝灏辨病鏈夋鏍锋湰浜?姣斿鎴戜滑labels鐨勭浜屼釜绫诲埆 labels[1,2,1,1]
        鎵€浠ヨ繖閲屽彧璁＄畻姝ｆ牱鏈釜鏁?0鐨?        '''
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

# Build heterograph.
def build_hgraph(utterance_num=None):
    if utterance_num is None:
        utterance_num = 2
        #0鏄痑ll锛?鏄痵ub
        # edges = torch.tensor([ [[0, 0], [0, 1]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    # else:
    aa_edges0 = torch.Tensor([[i, i + 1] for i in range(utterance_num - 1)])
    aa_edges1 = torch.Tensor([[i + 1, i] for i in range(utterance_num - 1)])
    aa_edges2 = torch.Tensor([[i, i] for i in range(utterance_num)])
    aa_edges = torch.cat((aa_edges0, aa_edges1, aa_edges2), dim=0)

    as_edges = torch.cat((
            torch.tensor([[i, 0 * utterance_num + i] for i in range(utterance_num)]),
            torch.tensor([[i, 1 * utterance_num + i] for i in range(utterance_num)]),
            torch.tensor([[i, 2 * utterance_num + i] for i in range(utterance_num)]),
    ), dim=0)
    sa_edges = as_edges[:, [1, 0]]
    edges = [aa_edges, as_edges, sa_edges]

    # print('nn', edges[0].shape)
    g = dgl.heterograph({
        ('all', 'dependency0', 'all'): edges[0].tolist(),
        ('sub', 'dependency1', 'all'): edges[2].tolist(),
        ('all', 'dependency2', 'sub'): edges[1].tolist(),
    })

# To print nodes and edges, you have to specify the node/edge type
    # for ntype in g.ntypes:
        # print(f"Nodes of type '{ntype}': {g.nodes(ntype)}")

    # for etype in g.etypes:
        # src, dst = g.edges(etype=etype)
        # print(f"Edges of type '{etype}': {src} -> {dst}")
    return g
# build_hgraph(2)




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm variant (no mean subtraction)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding with dynamic max-length extension."""

    def __init__(self, dim, max_len=512):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("pe", self._build(max_len, self.dim), persistent=False)

    @staticmethod
    def _build(max_len, dim):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Keep behavior stable for odd dimensions by trimming the last cosine term.
        if dim % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        return pe

    def _ensure_max_len(self, needed_len, device):
        if needed_len <= self.pe.size(0):
            return
        new_len = self.pe.size(0)
        while new_len < needed_len:
            new_len *= 2
        self.pe = self._build(new_len, self.dim).to(device)

    def forward(self, position_ids):
        # position_ids: [B, L]
        max_pos = int(position_ids.max().item()) + 1
        self._ensure_max_len(max_pos, position_ids.device)
        out = self.pe[position_ids]
        return out


def _build_norm(norm_type, dim, eps):
    norm_type = str(norm_type).lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    if norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def _get_activation(name):
    name = str(name).lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported ffn activation: {name}")


class ARISEMultiHeadAttention(nn.Module):
    """
    Standard Transformer-style MHA used by ARISE.
    head_dim is derived as d_model // n_heads (identical to BERT / GPT).
    """

    def __init__(self, d_model, n_heads, attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        self.head_dim = self.d_model // self.n_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x, mask=None):
        # x: [B, L, D]
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-2, -1))
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.attn_dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.resid_dropout(self.out_proj(out))
        return out, attn


class ARISEBlock(nn.Module):
    """
    Standard Transformer block for ARISE:
    MHA + AddNorm, FFN + AddNorm with configurable pre/post norm.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        ffn_mult=2.0,
        ffn_act="gelu",
        attn_dropout=0.1,
        ffn_dropout=0.1,
        resid_dropout=0.1,
        norm_type="layernorm",
        norm_order="pre",
        norm_eps=1e-6,
    ):
        super().__init__()
        self.norm_order = str(norm_order).lower()
        if self.norm_order not in {"pre", "post"}:
            raise ValueError(f"Unsupported norm_order: {norm_order}")

        self.attn = ARISEMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )

        d_ff = max(1, int(round(float(ffn_mult) * int(d_model))))
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.ffn_act = _get_activation(ffn_act)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        self.norm1 = _build_norm(norm_type, d_model, norm_eps)
        self.norm2 = _build_norm(norm_type, d_model, norm_eps)

    def _forward_ffn(self, x):
        x = self.ffn_in(x)
        x = self.ffn_act(x)
        x = self.ffn_dropout(x)
        x = self.ffn_out(x)
        return self.resid_dropout(x)

    def forward(self, x, mask=None, return_attn=False):
        if self.norm_order == "pre":
            attn_out, attn_w = self.attn(self.norm1(x), mask=mask)
            x = x + attn_out
            x = x + self._forward_ffn(self.norm2(x))
            if return_attn:
                return x, attn_w
            return x

        attn_out, attn_w = self.attn(x, mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self._forward_ffn(x))
        if return_attn:
            return x, attn_w
        return x


class ARISEEncoder(nn.Module):
    """
    Arbitrary Relation-view Interaction for Single-and-mixed modality Encoder.
    Runs independent Transformer stacks for global/speaker/local relations.
    """

    def __init__(
        self,
        input_dim,
        n_heads,
        num_layers=2,
        ffn_mult=2.0,
        ffn_act="gelu",
        attn_dropout=0.1,
        ffn_dropout=0.1,
        resid_dropout=0.1,
        norm_type="layernorm",
        norm_order="pre",
        norm_eps=1e-6,
        use_pe=False,
        pe_max_utt=128,
        view_fuse="max",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.n_heads = int(n_heads)
        # Standard Transformer: head_dim = d_model // n_heads
        self.model_dim = self.input_dim
        self.num_layers = int(num_layers)
        self.use_pe = bool(use_pe)
        self.view_fuse = str(view_fuse).lower()
        if self.view_fuse not in {"max", "mean"}:
            raise ValueError(f"Unsupported arise_view_fuse: {view_fuse}")

        self.in_proj = nn.Identity()
        self.out_proj = nn.Identity()
        self.pe = SinusoidalPositionalEmbedding(self.model_dim, max_len=max(2, int(pe_max_utt)))

        def _make_stack():
            return nn.ModuleList(
                [
                    ARISEBlock(
                        d_model=self.model_dim,
                        n_heads=self.n_heads,
                        ffn_mult=ffn_mult,
                        ffn_act=ffn_act,
                        attn_dropout=attn_dropout,
                        ffn_dropout=ffn_dropout,
                        resid_dropout=resid_dropout,
                        norm_type=norm_type,
                        norm_order=norm_order,
                        norm_eps=norm_eps,
                    )
                    for _ in range(self.num_layers)
                ]
            )

        self.global_layers = _make_stack()
        self.speaker_layers = _make_stack()
        self.local_layers = _make_stack()

    @staticmethod
    def _synced_position_ids(seq_len, device, batch_size):
        # For [mix, t, a, v], each utterance shares one position index.
        utt_len = seq_len // 4
        base = torch.arange(utt_len, device=device)
        pos = base.repeat(4)
        if pos.numel() < seq_len:
            # Guard for any non-4U edge case; should rarely happen.
            tail = torch.arange(seq_len - pos.numel(), device=device) + utt_len
            pos = torch.cat([pos, tail], dim=0)
        return pos.unsqueeze(0).expand(batch_size, -1)

    def _run_stack(self, x, layers, mask, return_attn=False):
        out = x
        last_attn = None
        for idx, layer in enumerate(layers):
            is_last = (idx == len(layers) - 1)
            if return_attn and is_last:
                out, last_attn = layer(out, mask=mask, return_attn=True)
            else:
                result = layer(out, mask=mask, return_attn=False)
                # return_attn=False always returns tensor directly
                out = result if not isinstance(result, tuple) else result[0]
        return out, last_attn

    def forward(self, x, gmasks=None, smasks=None, lmasks=None, return_attn=False):
        # x: [B, 4U, D_in]
        bsz, seq_len, _ = x.shape
        x = self.in_proj(x)
        if self.use_pe:
            pos_ids = self._synced_position_ids(seq_len, x.device, bsz)
            x = x + self.pe(pos_ids)

        global_out, g_attn = self._run_stack(x, self.global_layers, gmasks, return_attn=return_attn)
        speaker_out, s_attn = self._run_stack(x, self.speaker_layers, smasks, return_attn=return_attn)
        local_out, l_attn = self._run_stack(x, self.local_layers, lmasks, return_attn=return_attn)

        if self.view_fuse == "mean":
            fused = (global_out + speaker_out + local_out) / 3.0
        else:
            fused = torch.max(
                torch.stack([global_out, speaker_out, local_out], dim=0), dim=0
            )[0]
        fused = self.out_proj(fused)
        if return_attn:
            attn_dict = {"global": g_attn, "speaker": s_attn, "local": l_attn}
            return fused, attn_dict
        return fused


# Backward-compatible aliases.
TRIEMultiHeadAttention = ARISEMultiHeadAttention
TRIEBlock = ARISEBlock
TRIEEncoder = ARISEEncoder


class FusionGate(nn.Module):
    def __init__(self, hid_size):
        super(FusionGate, self).__init__()
        self.fuse_weight = nn.Parameter(torch.Tensor(hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fuse_weight)

    def forward(self, a, b):
        # Compute fusion coefficients
        fusion_coef = torch.sigmoid(self.fuse_weight)
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor


class NewFusionGate(nn.Module):
    def __init__(self, hid_size):
        super(NewFusionGate, self).__init__()
        self.fuse = nn.Linear(hid_size * 2, hid_size)

    def forward(self, a, b):
        # Concatenate a and b along the last dimension
        concat_ab = torch.cat([a, b], dim=-1)
        # Apply the linear layer
        fusion_coef = torch.sigmoid(self.fuse(concat_ab))
        # Fuse tensors a and b
        fused_tensor = fusion_coef * a + (1 - fusion_coef) * b
        return fused_tensor


class AttentionFusion(nn.Module):
    """
    澶氳瑙掓敞鎰忓姏铻嶅悎妯″潡銆?
    杈撳叆鑻ュ共涓悓缁村害瑙嗚鐗瑰緛 [B, L, D]锛屽湪姣忎釜鏃堕棿姝ヤ笂瀵硅瑙掔淮杩涜娉ㄦ剰鍔涘姞鏉冿紝
    杈撳嚭铻嶅悎鍚庣殑搴忓垪 [B, L, D]銆?    """

    def __init__(
            self,
            hid_size,
            num_views=3,
            attn_dim=None,
            use_scale=True,
            use_residual=True,
            residual_mode='max',
            use_layernorm=True,
            dropout=0.0):
        super(AttentionFusion, self).__init__()
        self.hid_size = hid_size
        self.num_views = num_views
        self.attn_dim = attn_dim if attn_dim is not None else hid_size

        # 鏄惁鍚敤缂╂斁鐐圭Н锛堥櫎浠?sqrt(attn_dim)锛?        self.use_scale = use_scale
        # 鏄惁鍙犲姞娈嬪樊锛堟潵鑷師濮嬪瑙嗚鐨?max 鎴?mean锛?        self.use_residual = use_residual
        self.residual_mode = residual_mode
        # 鏄惁鍦ㄨ緭鍑虹鍋?LayerNorm
        self.use_layernorm = use_layernorm

        if self.residual_mode not in ('max', 'mean'):
            raise ValueError("residual_mode must be 'max' or 'mean'")

        # 鐢ㄦ嫾鎺ュ悗鐨勫瑙嗚淇℃伅鐢熸垚 query锛屾洿瀹规槗寤烘ā鈥滆瑙掗棿浜掕ˉ鍏崇郴鈥?        self.q_proj = nn.Linear(hid_size * num_views, self.attn_dim)
        # 姣忎釜瑙嗚鍒嗗埆鏄犲皠鎴?key/value
        self.k_proj = nn.Linear(hid_size, self.attn_dim)
        self.v_proj = nn.Linear(hid_size, hid_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hid_size)

    def forward(self, *views, view_mask=None, return_attn=False):
        """
        Args:
            *views: 澶氫釜瑙嗚寮犻噺锛屾瘡涓舰鐘?[B, L, D]
            view_mask: 鍙€夛紝褰㈢姸 [B, L, V]锛? 琛ㄧず鍙敤锛? 琛ㄧず灞忚斀
            return_attn: 鏄惁杩斿洖娉ㄦ剰鍔涙潈閲?        Returns:
            fused: [B, L, D]
            attn_weights(鍙€?: [B, L, V]
        """
        if len(views) == 0:
            raise ValueError('AttentionFusion expects at least one view.')
        if len(views) != self.num_views:
            raise ValueError(
                f'View count mismatch: expect {self.num_views}, got {len(views)}')

        ref_shape = views[0].shape
        if len(ref_shape) != 3:
            raise ValueError('Each view must be a 3D tensor [B, L, D].')

        for idx, x in enumerate(views):
            if x.shape != ref_shape:
                raise ValueError(
                    f'All views must share the same shape. view0={ref_shape}, '
                    f'view{idx}={x.shape}')

        # X: [B, L, V, D]
        x = torch.stack(views, dim=2)
        bsz, seq_len, num_views, hid = x.size()

        # 1) 鍩轰簬鎵€鏈夎瑙掓嫾鎺ュ緱鍒?query: [B, L, A]
        # 2) 姣忎釜瑙嗚鍒嗗埆寰楀埌 key/value: [B, L, V, A], [B, L, V, D]
        q = self.q_proj(x.reshape(bsz, seq_len, num_views * hid))
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 鐐圭Н娉ㄦ剰鍔涘垎鏁? [B, L, V]
        attn_logits = torch.einsum('bla,blva->blv', q, k)
        if self.use_scale:
            attn_logits = attn_logits / (self.attn_dim ** 0.5)

        # 鍙€夌殑瑙嗚鎺╃爜锛氬皢涓嶅彲鐢ㄨ瑙掓墦鍒版瀬灏忓€硷紝閬垮厤鍒嗗埌姒傜巼
        if view_mask is not None:
            if view_mask.shape != attn_logits.shape:
                raise ValueError(
                    f'view_mask shape mismatch: expect {attn_logits.shape}, got {view_mask.shape}')
            attn_logits = attn_logits.masked_fill(view_mask == 0, -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 铻嶅悎杈撳嚭: [B, L, D]
        fused = torch.einsum('blv,blvd->bld', attn_weights, v)

        # 娈嬪樊鏉ヨ嚜鈥滃師濮嬭瑙掔壒寰佽仛鍚堚€濓紝绋冲畾璁粌骞朵繚鐣欏熀纭€淇℃伅
        if self.use_residual:
            if self.residual_mode == 'max':
                residual = torch.max(x, dim=2)[0]
            else:
                residual = torch.mean(x, dim=2)
            fused = fused + residual

        if self.use_layernorm:
            fused = self.layer_norm(fused)

        if return_attn:
            return fused, attn_weights
        return fused

