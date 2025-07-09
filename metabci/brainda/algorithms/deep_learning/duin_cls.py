import numpy as np
import torch
from einops import repeat
from torch import nn, distributed
import copy as cp
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os
import torch.distributed as dist


def init_distributed():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = args.local_rank

    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = args.world_size

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=local_rank)

class SubjectLayer(nn.Module):
    """
    Subject Layer used to transform embeddings with specified subj id.
    """
    def __init__(self, d_input, n_subjects, d_output, use_bias=False):
        super().__init__()
        # Initialize parameters
        self.d_input = d_input
        self.d_output = d_output
        self.n_subjects = n_subjects
        self.use_bias = use_bias

        # Initialize the model
        self.W = nn.Linear(in_features=self.n_subjects,
                           out_features=(self.d_input*self.d_output),
                           bias=False)
        self.B = nn.Linear(in_features=self.n_subjects,
                           out_features=self.d_output,
                           bias=False) if self.use_bias else None

        # Initialize the weight
        nn.init.trunc_normal_(self.W.weight, mean=0, std=0.02)
        if self.W.bias is not None:
            nn.init.zeros_(self.W.bias)
        if self.B is not None:
            nn.init.constant_(self.B.weight, val=0.)
            if self.B.bias is not None:
                nn.init.zeros_((self.B.bias))

    def forward(self, inputs):
        """

        Parameters
        ----------
        inputs: [X, subj_id]

        Returns
        -------

        """
        X = inputs[0]
        subj_id = inputs[1]                 # (batch_size, n_subjects)
        W_s = torch.reshape(self.W(subj_id), shape=(-1, self.d_input,self.d_output))        #(batch_size, d_input, d_output)
        Z = torch.reshape(torch.matmul(
            torch.reshape(X, shape=(X.shape[0], -1, X.shape[-1])),W_s
        ), shape=(*X.shape[:-1], W_s.shape[-1]))
        Z = torch.reshape((torch.reshape(Z, shape=(Z.shape[0], -1, Z.shape[-1])) + torch.unsqueeze(self.B(subj_id),dim=-2)
                           ), shape=Z.shape) if self.use_bias else Z
        return Z


class SubjectBlock(nn.Module):
    """
    Subject Block used to transform embeddings with specified subj id.
    """
    def __init__(self, params):
        super().__init__()
        self.params = cp.deepcopy(params)

        # Initialize the model
        self.subj_layer = SubjectLayer(d_input=self.params.d_input, n_subjects=self.params.n_subjects,
                                       d_output=self.params.d_output, use_bias=self.params.use_bias)
        self.proj_layer = nn.Linear(in_features=self.params.d_output, out_features=self.params.d_output, bias=True) if self.params.use_proj else None

        # Initialize the weight
        if self.proj_layer is not None:
            nn.init.trunc_normal_(self.proj_layer.weight, mean=0., std=0.02)
            if self.proj_layer.bias is not  None:
                nn.init.constant_(self.proj_layer.bias, val=0.)

    def forward(self, inputs):
        """

        Parameters
        ----------
        inputs

        Returns
        -------

        """
        X = inputs[0]
        subj_id = inputs[1]
        emb = self.subj_layer((X, subj_id))
        emb = self.proj_layer(emb) if self.params.use_proj else emb
        return emb


class PatchTokenizer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = cp.deepcopy(params)

        # Initialize the model
        self.conv_blocks = nn.Sequential()
        for conv_idx in range(len(self.params.n_filters)):
            n_channels = self.params.n_filters[conv_idx-1] if conv_idx > 0 else self.params.d_neural
            n_filters = self.params.n_filters[conv_idx]
            kernel_size = self.params.kernel_sizes[conv_idx]
            n_strides = self.params.n_strides[conv_idx]
            dilation = self.params.dilation_rates[conv_idx]
            pool_size = self.params.pool_sizes[conv_idx]
            use_bn = self.params.use_bn[conv_idx]
            ues_res = self.params.use_res[conv_idx]
            self.conv_blocks.append(PatchTokenizer._make_conv_block(
                n_channels=n_channels, n_filters=n_filters,kernel_size=kernel_size,n_strides=n_strides,
                dilation_rate=dilation, pool_size=pool_size, use_bn=use_bn,use_res=ues_res
            ))

        # Initialize the weight
        for mudule_i in self.conv_blocks.modules():
            if isinstance(mudule_i, nn.BatchNorm1d):
                if mudule_i.weight is not None:
                    nn.init.ones_(mudule_i.weight)
                if mudule_i.bias is not None:
                    nn.init.zeros_(mudule_i.bias)

    def _make_conv_block(n_channels, n_filters, kernel_size, n_strides,
                         dilation_rate, pool_size=1, use_bn=False, use_res=False, **kwargs):
        """
        Add [Conv1d,BatchNorm1d,AvgPool1d]

        Parameters
        ----------
        n_channels
        n_filters
        kernel_size
        n_strides
        dilation_rate
        pool_size
        use_bn
        use_res
        kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns
        -------

        """
        conv_block = nn.Sequential(**kwargs)
        padding = "same" if n_strides==1 else int((dilation_rate * (kernel_size - 1)) / 2)
        conv_block.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.Conv1d(
                in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size,
                stride=n_strides, padding=padding, dilation=dilation_rate,
                groups=1, bias=True, padding_mode="zeros"
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0, 2, 1])))
        ))

        if use_bn:
            conv_block.append(nn.Sequential(
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0, 2, 1]))),
                nn.BatchNorm1d(num_features=n_filters,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0, 2, 1])))
            ))

        if use_res:
            resconv_block = ResidualConnection(module=conv_block, residual_scales=[1., 1.])
            conv_block = nn.Sequential(**kwargs)
            conv_block.append(resconv_block)

        if pool_size > 1:
            conv_block.append(nn.Sequential(
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
                nn.AvgPool1d(kernel_size=kernel_size,padding=0,ceil_mode=False, count_include_pad=True),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0, 2, 1])))
            ))
        return conv_block

    def forward(self, X):
        batch_size, n_channels, seq_len,  = X.shape
        n_segs = seq_len // self.params.seg_len
        X = torch.reshape(X, shape=(-1, self.params.seg_len, n_channels))
        T = torch.reshape(self.conv_blocks(X), shape=(batch_size, n_segs, -1))
        return T


class ResidualConnection(nn.Module):
    def __init__(self, module, residual_scales=[1., 1.], **kwargs):
        super().__init__()

        # Initialize the parameters
        self.module = module
        self.residual_scales = residual_scales

    def forward(self, emb, *args, **kwargs):
        return (self.residual_scales[0]*emb) + (self.residual_scales[1]*self.module(emb, *args, **kwargs))



class LambdaLayer(nn.Module):
    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class TimeEmbedding(nn.Module):
    def __init__(self, d_model, max_len, mode=None, **kwargs):
        super().__init__()
        assert d_model%2 == 0, ("Error: The dimensions of model embedding ({:d}) must be a multiples of 2").format(d_model)
        assert mode in [None, "zero", "zeros", "normal", "uniform", "sincos"],(
            "ERROR: Get unknown time embedding mode {} in layers.TimeEmbedding.".format(mode)
        )
        self.d_model = d_model
        self.max_len = max_len
        self.mode = mode

        # Initialize model architecture according to `mode`.
        mode = str(self.mode).lower()

        if mode in [None, "zeros"]:
            time_encodings = np.random.uniform(low=-2e-2, high=2e-2, size=(self.max_len, self.d_model))
        if mode == "zero":
            time_encodings = np.random.uniform(low=-2e-2, high=2e-2, size=(self.max_len, 1))
        if mode == "normal":
            time_encodings = np.random.normal(loc=0., scale=1., size=(self.max_len, self.d_model))
        if mode == "uniform":
            time_encodings = np.random.uniform(low=0., high=1., size=(self.mode, self.d_model))
        if mode =="sincos":
            time_encodings = np.zeros((self.max_len, self.d_model), dtype=np.float32)
            time_idxs = np.expand_dims(np.arange(0, self.max_len, dtype=np.float32), axis=-1)
            div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(np.log(1e4) / self.d_model))
            time_encodings[:, 0::2] = np.sin(time_idxs*div_term)
            time_encodings[:, 1::2] = np.cos(time_idxs * div_term)

        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)

        if mode in [None, "sincos"]:
            self.time_encodings = nn.Parameter(time_encodings, requires_grad=False)
        else:
            self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)

    def forward(self, emb):
        batch_size, seq_len, _ = emb.shape
        time_emb = self.time_encodings[:emb.shape[-2], :].expand(batch_size, seq_len, -1)
        emb = emb + time_emb
        return emb

class RotaryEmbedding(nn.Module):
    def __init__(self, d_model, theta=1e4, **kwargs):
        super().__init__()
        assert d_model % 2 == 0, (
            "ERROR: The dimensions of model embedding ({:d}) must be a multiples of 2 in layers.RotaryEmbedding."
        ).format(d_model)
        freqs = 1. / (theta ** (np.arange(0, d_model, 2)[:(d_model // 2)] / d_model))
        self.freqs = nn.Parameter(torch.tensor(freqs, dtype=torch.float32), requires_grad=False)

    def forward(self, emb):
        position_idxs = torch.arange(emb.shape[-2], dtype=emb.dtype).to(device=emb.device)
        freqs = torch.einsum("..., f -> ... f", position_idxs, self.freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        emb = torch.cos(freqs) * emb + torch.sin(freqs) * torch.reshape((
            torch.flip(torch.reshape(emb, shape=(*emb.shape[:-1], emb.shape[-1] // 2, 2)), dims=[-1]) *\
            torch.tensor([-1., 1.], dtype=emb.dtype).to(device =emb.device)
        ), shape=emb.shape)
        return emb

class MHAMatrix(nn.Module):
    def __init__(self, d_model, n_heads,d_head, use_bias=True, **kwargs):
        super().__init__()

        self.W = nn.Linear(in_features=d_model, out_features=(n_heads*d_head), bias=use_bias)
        self.n_heads = n_heads
        self.d_head = d_head

        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not  None:
                    nn.init.constant_(module_i.bias, val=0.)

    def forward(self, emb):
        head_shape = emb.shape[:-1]
        emb = torch.reshape(self.W(emb), shape=(*head_shape, self.n_heads, self.d_head))
        return emb

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head, attn_dropout=0., scale_trainable=False, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(p=attn_dropout, inplace=False)
        self.scale = nn.Parameter(torch.tensor(1. / np.sqrt(d_head), dtype=torch.float32), requires_grad=scale_trainable)

    def forward(self, embs, attn_score=None, attn_mask=None, key_padding_mask=None):
        emb_q, emb_k, emb_v = embs
        attn_score = (torch.matmul(emb_q, torch.permute(emb_k, dims=[0,1,3,2])) * self.scale) if attn_score is None else \
            (torch.matmul(emb_q, torch.permute(emb_k, dims=[0,1,3,2])) * self.scale) + attn_score
        if attn_mask is not None:
            attn_score = torch.where(torch.unsqueeze(torch.unsqueeze(attn_mask, dim=0), dim=0), -np.inf, attn_score)
        if key_padding_mask is not None:
            attn_score = torch.where(torch.unsqueeze(torch.unsqueeze(key_padding_mask, dim=1), dim=2), -np.inf, attn_score)
        attn_weight = self.dropout(torch.softmax(attn_score, dim=-1))
        emb = torch.matmul(attn_weight, emb_v)
        return emb, attn_weight, attn_score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, attn_dropout=0., proj_dropout=0., emb_rotary=None, use_bias=True, **kwargs):
        super().__init__()

        self.emb_rotary = emb_rotary

        # Initialize the model
        self.W_q = MHAMatrix(d_model=d_model, n_heads=n_heads, d_head=d_head, use_bias=use_bias)
        self.W_k = MHAMatrix(d_model=d_model, n_heads=n_heads, d_head=d_head, use_bias=use_bias)
        self.W_v = MHAMatrix(d_model=d_model, n_heads=n_heads, d_head=d_head, use_bias=use_bias)

        self.norm_q = nn.LayerNorm(normalized_shape=(d_head,), eps=1e-5, elementwise_affine=True)
        self.norm_k = nn.LayerNorm(normalized_shape=(d_head,), eps=1e-5, elementwise_affine=True)

        self.attention = ScaledDotProductAttention(d_head=d_head, attn_dropout=attn_dropout, scale_trainable=False)
        self.proj = nn.Sequential(
            nn.Linear(in_features=(n_heads * d_head), out_features=d_model, bias=True),
            nn.Dropout(p=proj_dropout, inplace=False)
        )

        #Initialize the weight
        for module_i in self.norm_q.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None:
                    nn.init.ones_(module_i.weight)
                if module_i.bias is not None:
                    nn.init.zeros_(module_i.bias)
        for module_i in self.norm_k.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None:
                    nn.init.ones_(module_i.weight)
                if module_i.bias is not None:
                    nn.init.zeros_(module_i.bias)
        for module_i in self.proj.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0.,std=0.02)
                if module_i.bias is not None:
                    nn.init.constant_(module_i.bias, val=0.)

    def forward(self, embs, attn_score=None, attn_mask=None, key_padding_mask=None):
        emb_q, emb_k, emb_v = embs
        emb_q = self.norm_q(torch.permute(self.W_q(emb_q), dims=[0,2,1,3]))
        emb_k = self.norm_k(torch.permute(self.W_k(emb_k), dims=[0,2,1,3]))
        emb_v = torch.permute(self.W_v(emb_v), dims=[0,2,1,3])
        if self.emb_rotary is not None:
            emb_q = self.emb_rotary(emb_q)
            emb_k = self.emb_rotary(emb_k)
        emb, attn_weight, attn_score = self.attention((emb_q,emb_k,emb_v),
                                                      attn_score, attn_mask, key_padding_mask)
        emb = torch.permute(emb, dims=[0,2,1,3])
        emb = torch.reshape(emb, shape=(*emb.shape[:-2], -1))
        emb = self.proj(emb)
        return emb, attn_weight, attn_score

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, ff_dropout, use_bias=[True,True], use_bias_gate=None, **kwargs):
        super().__init__()

        self.is_gated = (use_bias_gate is not None)

        # Initialize the model
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff, bias=use_bias[0]),
            nn.ReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=d_ff, out_features=d_model, bias=use_bias[1])
        )
        self.dropout1 = nn.Dropout(p=ff_dropout[0], inplace=False)
        self.dropout2 = nn.Dropout(p=ff_dropout[1], inplace=False)
        self.gate = nn.Linear(in_features=d_model, out_features=d_ff, bias=use_bias_gate) if use_bias_gate else None

        # Initialize the weight
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None:
                    nn.init.constant_(module_i.bias, val=0.)

    def forward(self, emb):
        emb = self.fc1(emb) * self.gate(emb) if self.is_gated else self.fc1(emb)
        emb = self.dropout1(emb)
        emb = self.fc2(emb)
        emb = self.dropout2(emb)
        return emb



class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_head, attn_dropout, proj_dropout,
                 d_ff, ff_dropout, rot_theta=None, norm_first=False, **kwargs):
        """
        Initialize `TransformerBlock` object.

        Args:
            d_model: int - The dimensions of model embedding.
            n_heads: int - The number of attention heads in `mha` block.
            d_head: int - The dimensions of attention head in `mha` block.
            attn_dropout: float - The dropout probability of attention score in `mha` block.
            proj_dropout: float - The dropout probability of projection in `mha` block.
            d_ff: int - The dimensions of the hidden layer in `ffn` block.
            ff_dropout: (2[list],) - The dropout probabilities in `ffn` block.
            rot_theta: float - The power base of rotation angle, default as `None`.
            norm_first: bool - The flag that indicates whether normalize data first.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        super().__init__()
        self.rot_theta = rot_theta
        self.norm_first = norm_first

        # Initialize the model
        emb_rotary = RotaryEmbedding(d_model=d_head, theta=self.rot_theta) if self.rot_theta is not  None else None
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_head=d_head, attn_dropout=attn_dropout,
                                      proj_dropout=proj_dropout, emb_rotary=emb_rotary, use_bias=True)
        self.norm_mha = nn.LayerNorm(normalized_shape=(d_model,), eps=1e-5, elementwise_affine=True)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, ff_dropout=ff_dropout) if d_ff is not None else None
        self.norm_ffn = nn.LayerNorm(normalized_shape=(d_model,), eps=1e-5, elementwise_affine=True)

        # Initialize the weight
        for module_i in self.norm_mha.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None:
                    nn.init.ones_(module_i.weight)
                if module_i.bias is not None:
                    nn.init.zeros_(module_i.bias)
        for module_i in self.norm_ffn.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None:
                    nn.init.ones_(module_i.weight)
                if module_i.bias is not None:
                    nn.init.zeros_(module_i.bias)

    def forward(self, emb, attn_score=None, attn_mask=None, key_padding_mask=None):
        if self.norm_first:
            emb = self.norm_mha(emb) if self.norm_mha is not None else emb
        attn_emb, attn_weight, attn_score = self.mha((emb,emb,emb), attn_score=attn_score, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        emb = attn_emb + emb
        if not self.norm_first:
            emb = self.norm_mha(emb) if self.norm_mha is not None else emb

        if self.norm_first:
            emb = self.norm_ffn(emb) if self.norm_ffn is not None else emb
        emb = self.ffn(emb) + emb if self.ffn is not None else emb
        if not self.norm_first:
            emb = self.norm_ffn(emb) if self.norm_ffn is not None else emb

        return emb, attn_weight, attn_score

class TransformerStack(nn.Module):
    def __init__(self, params,  **kwargs):
        super(TransformerStack, self).__init__()
        self.params = cp.deepcopy(params)

        # Initialize the model: (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.xfmr_blocks = nn.ModuleList(modules=[TransformerBlock(d_model=self.params.d_model, n_heads=self.params.n_heads, d_head=self.params.d_head,
                                                                   attn_dropout=self.params.attn_dropout, proj_dropout=self.params.proj_dropout,
                                                                   d_ff=self.params.d_ff, ff_dropout=self.params.ff_dropout, rot_theta=self.params.rot_theta,
                                                                   norm_first=self.params.norm_first) for block_idx in range(self.params.n_blocks) ])

        # Initialize the weight
        for block_idx, xfmr_block_i in enumerate(self.xfmr_blocks):
            for module_i in xfmr_block_i.modules():
                if isinstance(module_i, nn.Linear):
                    module_i.weight.data.div_(np.sqrt(2. * (block_idx + 1)))

    def forward(self, emb, attn_score=None, attn_mask=None, key_padding_mask=None):
        attn_weight = torch.softmax(attn_score, dim=-1) if attn_score is not None else None
        for block_idx in range(len(self.xfmr_blocks)):
            emb, attn_weight, attn_score = self.xfmr_blocks[block_idx](emb,
                                                                       attn_score=attn_score if self.params.res_attn else None,
                                                                       attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return emb, attn_weight,attn_score


def kmeans_sklearn(samples, n_clusters, n_iters=10, use_cossim=False):
    """
    K-means clustering using scikit-learn.

    Args:
        samples: (n_samples, d_emb) - The input samples.
        n_clusters: int - The number of clusters, i.e., K.
        n_iters: int - The number of K-means iterations.
        use_cossim: bool - The flag that indicates whether cosine similarity.

    Returns:
        means: (n_clusters, d_emb) - The K-means cluster means.
        bins: (n_clusters,) - The sample counts of each cluster.
    """
    samples_np = samples.detach().numpy()

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=n_iters, random_state=42)
    kmeans.fit(samples_np)

    means = torch.tensor(kmeans.cluster_centers_, dtype=samples.dtype, device=samples.device)
    bins = torch.tensor(np.bincount(kmeans.labels_, minlength=n_clusters), dtype=torch.long, device=samples.device)

    if use_cossim:
        means = torch.nn.functional.normalize(means, p=2, dim=-1, eps=1e-12)

    return means, bins


class EMAEmbedding(nn.Module):
    def __init__(self, n_embs, d_emb, decay=0.99, init_kmeans=True, **kwargs):
        super().__init__()
        self.n_embs=n_embs
        # Initialize the model
        embeddings = F.normalize(torch.randn((n_embs, d_emb), dtype=torch.float32), p=2, dim=-1, eps=1e-12)
        self.embeddings = nn.Parameter(embeddings, requires_grad=False)
        self.initted = nn.Parameter(torch.Tensor([not init_kmeans]), requires_grad=False)
        if distributed.is_available() and distributed.is_initialized():
            print("INFO: DDP is enabled, use ddp_reduce to sync across multi-GPUs!")

    def init_emb(self, emb_data):
        if self.initted: return

        print("INFO: Perform K-means initialization for embeddings in layers.LaBraMVectorQuantizer.EMAEmbedding.")
        emb_init, _ = kmeans_sklearn(samples=emb_data, n_clusters=self.n_embs, n_iters=10, use_cossim=True)
        if distributed.is_available() and distributed.is_initialized():
            distributed.broadcast(tensor=emb_init, src=0)
        emb_init = F.normalize(emb_init, p=2, dim=-1, eps=1e-12)
        self.embeddings.data.copy_(emb_init)
        self.initted.copy_(torch.Tensor([True]))

    def forward(self, index):
        emb = F.embedding(input=index, weight=self.embeddings, padding_idx=None, max_norm=None,
                          norm_type=2, scale_grad_by_freq=False, sparse=False)
        return emb

class LaBraMVectorQuantizer(nn.Module):
    def __init__(self, d_model, codex_size, d_codex, beta=1., decay=0.99, init_kmeans=True, **kwargs):
        super().__init__()

        self.beta = beta
        self.codex_size = codex_size
        self.decay = decay

        # Initialize the model
        self.pre_proj = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=d_model, out_features=d_codex, bias=True)
        )
        self.codex = EMAEmbedding(n_embs=codex_size, d_emb=d_codex, decay=decay, init_kmeans=init_kmeans)
        counts = torch.zeros((codex_size,), dtype=torch.float32)
        self.counts = nn.Parameter(counts, requires_grad=False)
        if distributed.is_available() and distributed.is_initialized():
            print("INFO: DDP is enabled, use ddp_reduce to sync across multi-GPUs!")
        self.layernorm = nn.LayerNorm(normalized_shape=(d_codex,), eps=1e-5, elementwise_affine=True)
        self.post_proj = nn.Sequential(nn.Linear(in_features=d_codex, out_features=d_model,bias=True))

        # Initialize the weight
        for module_i in self.pre_proj.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)
        for module_i in self.layernorm.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)
        for module_i in self.post_proj.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    def init_counts(self):
        self.counts.data.copy_(torch.zeros(self.counts.shape))

    def get_counts(self):
        return self.counts.detach().cpu().numpy()

    def loss(self, Z_e, Z_q):
        loss_commitment = F.mse_loss(input=Z_e, target=Z_q.detach(), size_average=None, reduce=None, reduction="mean")
        loss = self.beta * loss_commitment
        return loss

    def forward(self, Z):
        Z_e = self.pre_proj(Z)
        Z_e = F.normalize(Z_e, p=2, dim=-1, eps=1e-12)
        self.codex.init_emb(torch.reshape(Z_e, shape=(-1, Z_e.shape[-1])))
        codex = self.codex.embeddings
        codex_dists = (
            torch.sum(torch.reshape(Z_e, shape=(-1, Z_e.shape[-1])).pow(2), dim=-1, keepdim=True)
            + torch.unsqueeze(torch.sum(codex.pow(2), dim=-1), dim=0)
            - 2. * torch.einsum("nd,cd->nc", torch.reshape(Z_e, shape=(-1, Z_e.shape[-1])),codex))
        codex_dists = torch.reshape(codex_dists, shape=(*Z_e.shape[:-1], -1))
        codex_idxs = torch.argmin(codex_dists, dim=-1)
        codex_probs = F.one_hot(codex_idxs, num_classes=self.codex_size).to(dtype=codex_dists.dtype)
        Z_q = torch.reshape(self.codex(codex_idxs), shape=Z_e.shape)
        loss = self.loss(Z_e, Z_q)
        Z_q = Z_e + (Z_q - Z_e).detach()
        perplexity = torch.tensor(0., dtype=torch.float32)

        if self.training:
            counts = torch.sum(torch.reshape(codex_probs, shape=(-1, codex_probs.shape[-1])), dim=0)
            distributed.all_reduce(tensor=counts, op=distributed.ReduceOp.SUM) if distributed.is_available() and distributed.is_initialized() else None
            ema_inplace(self.counts, counts, decay=self.decay, use_norm=False)
            zero_mask = (counts == 0)
            counts = counts.masked_fill(zero_mask, value=1.)
            codex_n = torch.matmul(torch.permute(torch.reshape(codex_probs, shape=(-1, codex_probs.shape[-1])), dims=[1,0]),
                                   torch.reshape(Z_e, shape=(-1, Z_e.shape[-1])))
            distributed.all_reduce(tensor=codex_n, op=distributed.ReduceOp.SUM) if distributed.is_available() and distributed.is_initialized() else None
            codex_n = F.normalize(codex / torch.unsqueeze(counts, dim=-1), p=2, dim=-1, eps=1e-12)
            codex_n = torch.where(zero_mask[...,None], codex, codex_n)
            ema_inplace(self.codex.embeddings, codex_n, decay=self.decay, use_norm=True)
        else:
            with torch.no_grad():
                counts = torch.sum(torch.reshape(codex_probs, shape=(-1, codex_probs.shape[-1])), dim=0)
                distributed.all_reduce(tensor=counts, op=distributed.ReduceOp.SUM) if distributed.is_available() and distributed.is_initialized() else None
                ema_inplace(self.counts, counts, decay=self.decay, use_norm=False)

        Z_q = self.layernorm(Z_q)
        Z_q = self.post_proj(Z_q)
        return Z_q, loss, codex_probs

def ema_inplace(weight, value, decay=0.99, use_norm=False):
    weight.data.mul_(decay).add_(value, alpha=(1. - decay))
    if use_norm:
        weight.data.copy_(F.normalize(weight.data, p=2, dim=-1, eps=1e-12))

class ContrastiveBlock(nn.Module):
    def __init__(self, d_model, d_contra, loss_mode, **kwargs):
        super().__init__()

        assert loss_mode in ["clip", "clip_orig", "unicl"], ("ERROR: unknown loss mode {}").format(loss_mode)
        self.loss_mode = loss_mode

        # Initialize the model
        if loss_mode == "clip":
            self.tau = nn.Parameter(torch.tensor(0.25, dtype=torch.float32), requires_grad=False)
        if loss_mode == "clip_orig":
            self.t = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=False)
        if loss_mode == "unicl":
            self.t = nn.Parameter(torch.tensor(2.0, dtype=torch.float32), requires_grad=False)

        self.proj_z = nn.Sequential()
        if d_contra is not None:
            self.proj_z.append(nn.Linear(in_features=d_model, out_features=d_contra, bias=True))
        self.proj_z.append(nn.Flatten(start_dim=1, end_dim=-1))

        self.proj_y = nn.Sequential()
        if d_contra is not None:
            self.proj_y.append(nn.Linear(in_features=d_model, out_features=d_contra,bias=True))
        self.proj_y.append(nn.Flatten(start_dim=1, end_dim=-1))

        for module_i in self.proj_z.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)
        for module_i in self.proj_y.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    def forward(self, inputs):
        X_f, y_true = inputs
        Z, Y = X_f
        label_z, labei_y = y_true
        emb_z = F.normalize(self.proj_z(Z), p=2, dim=-1, eps=1e-12)
        emb_y = F.normalize(self.proj_z(Y), p=2, dim=-1, eps=1e-12)
        loss_matrix = None
        if self.loss_mode == "clip":
            loss_matrix = torch.exp(torch.matmul(emb_z, torch.permute(emb_y, dims=[1,0])) / self.tau)
            labels = torch.eye(loss_matrix.shape[0], dtype=loss_matrix.dtype)
            loss_z = torch.squeeze(torch.subtract(
                torch.log(torch.sum(loss_matrix, dim=0, keepdim=True)),
                torch.log(torch.sum(torch.multiply(loss_matrix, labels), dim=0,keepdim=True))
            ))
            loss_y = torch.squeeze(torch.subtract(
                torch.log(torch.sum(loss_matrix, dim=0, keepdim=True)),
                torch.log(torch.sum(torch.multiply(loss_matrix, labels), dim=1,keepdim=True))
            ))
            loss = (torch.mean(loss_z) + torch.mean(loss_y)) / 2
        elif self.loss_mode == "clip_orig":
            loss_matrix = torch.matmul(emb_z, torch.permute(emb_y, dims=[1,0])) * torch.exp(self.t)
            labels = torch.eye(loss_matrix.shape[0], dtype=loss_matrix.dtype)
            loss_z = -torch.sum(labels * torch.log(torch.softmax(loss_matrix, dim=-1) + 1e-12), dim=-1, keepdim=True)
            loss_y = -torch.sum(labels * torch.log(torch.softmax(loss_matrix, dim=0) + 1e-12), dim=0, keepdim=True)
            loss = (torch.mean(loss_z) + torch.mean(loss_y)) / 2
        elif self.loss_mode == "unicl":
            loss_matrix = torch.matmul(emb_z, torch.permute(emb_y, dims=[1,0])) * torch.exp(self.t)
            labels = torch.matmul(label_z, torch.permute(labei_y, dims=[1,0]))
            loss_z = -torch.sum(labels * torch.log(torch.softmax(loss_matrix, dim=-1) + 1e-12), dim=-1, keepdim=True)
            loss_y = -torch.sum(labels * torch.log(torch.softmax(loss_matrix, dim=0) + 1e-12), dim=0, keepdim=True)
            loss = (torch.mean(loss_z) + torch.mean(loss_y)) / 2

        return loss, loss_matrix

class LabelCLSHead(nn.Module):
    def __init__(self, params, **kwargs):
        super().__init__()
        self.params = cp.deepcopy(params)

        # Initialize the model
        self.cls_head = nn.Sequential()
        self.cls_head.append(nn.Flatten(start_dim=1, end_dim=-1))
        self.cls_head.append(nn.AdaptiveAvgPool1d(self.params.d_feature))
        for hidden_idx in range(len(self.params.d_hidden)):
            self.cls_head.append(nn.Sequential(
                nn.Linear(in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.d_feature),
                          out_features=self.params.d_hidden[hidden_idx], bias=True),
                nn.ReLU(inplace=False)
            ))
        if self.params.dropout > 0.:
            self.cls_head.append(nn.Dropout(p=self.params.dropout, inplace=False))
        self.cls_head.append(nn.Sequential(
            nn.Linear(in_features=(self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.d_feature),
                      out_features=self.params.n_labels, bias=True),
            nn.Sigmoid()
        ))

        # Initialize the weight
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    def forward(self, E):
        return self.cls_head(E)

class DotDict:
    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)

class DuIN(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = cp.deepcopy(params)
        self.params.subj = DotDict(self.params.subj)
        self.params.tokenizer = DotDict(self.params.tokenizer)
        self.params.encoder = DotDict(self.params.encoder)
        self.params.vq = DotDict(self.params.vq)
        self.params.contra = DotDict(self.params.contra)
        self.params.cls = DotDict(self.params.cls)
        d_token_out = self.params.tokenizer.seg_len * self.params.tokenizer.n_filters[-1]

        # Initialize the model
        self.subj_block = SubjectBlock(params=self.params.subj)             #(batch_size, seq_len, n_channels) -> (batch_size, seq_len, d_neural)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)       #(batch_size, seq_len, d_neural) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.proj_token = nn.Linear(d_token_out, self.params.encoder.d_model)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder),
            LambdaLayer(func=(lambda x: x[0])))
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_means)
        self.contra_block = ContrastiveBlock(d_model=self.params.contra.d_model,
            d_contra=self.params.contra.d_contra, loss_mode=self.params.contra.loss_mode)
        self.cls_block = LabelCLSHead(params=self.params.cls)

        nn.init.trunc_normal_(self.proj_token.weight, mean=0., std=0.02)
        if self.proj_token.bias is not None:
            nn.init.constant_(self.proj_token.bias, 0.)

    def forward(self, inputs):
        X, y_true, subj_id = inputs
        X_h = self.subj_block((X, subj_id))
        T = self.tokenizer(X_h)
        E = self.proj_token(T)
        E = self.encoder(E)
        E_vq, loss_vq, _ = self.vq_block(E)
        loss_contra, _ = self.contra_block(((E, E), (y_true, y_true)))
        y_pred = self.cls_block(E)
        loss_cls = F.cross_entropy(input=y_pred, target=y_true)
        loss_total = (self.params.cls_loss_scale * loss_cls + self.params.contra_loss_scale * loss_contra)
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
            "contra": loss_contra
        })
        return y_pred, loss


if __name__ == "__main__":
    params = {
        "subj": {
            "d_input": 2100,              # 输入维度(seq_len)
            "n_subjects": 1,            # 受试者数量
            "d_output": 256,             # 输出维度
            "use_bias": True,            # 是否使用偏置
            "use_proj": True             # 是否使用投影层 (新增关键参数)
        },
        "tokenizer": {
            "d_neural": 20,             # 神经数据维度
            "n_filters": [64, 128],      # 卷积层过滤器数量列表
            "kernel_sizes": [3, 3],      # 卷积核尺寸列表
            "n_strides": [1, 1],         # 卷积步长列表
            "dilation_rates": [1, 1],    # 空洞率列表
            "pool_sizes": [1, 1],        # 池化尺寸列表
            "use_bn": [True, True],      # 是否使用批归一化列表
            "use_res": [False, False],   # 是否使用残差连接列表
            "seg_len": 64                # 分段长度 (新增关键参数)
        },
        "encoder": {
            "d_model": 128,              # 模型维度
            "n_heads": 4,                # 注意力头数
            "d_head": 32,                # 每个注意力头的维度
            "attn_dropout": 0.1,         # 注意力dropout率
            "proj_dropout": 0.1,         # 投影dropout率
            "d_ff": 512,                # 前馈网络维度
            "ff_dropout": [0.1, 0.1],    # 前馈网络dropout率
            "rot_theta": None,           # 旋转嵌入的theta值
            "norm_first": True,          # 是否先进行归一化
            "n_blocks": 4,               # Transformer块数量
            "emb_len": 512,              # 嵌入最大长度 (新增关键参数)
            "res_attn": False            # 是否保留注意力分数 (新增关键参数)
        },
        "vq": {
            "d_model": 128,              # 输入维度
            "codex_size": 128,           # 码本大小
            "d_codex": 128,              # 码本维度
            "beta": 0.25,                # VQ损失权重
            "decay": 0.99,               # EMA衰减率
            "init_means": True            # 是否初始化均值
        },
        "contra": {
            "d_model": 128,              # 输入维度
            "d_contra": 64,             # 对比学习维度
            "loss_mode": "clip"           # 损失模式 ("clip", "clip_orig", "unicl")
        },
        "cls": {
            "d_feature": 128,            # 输入特征维度
            "d_hidden": [64],       # 隐藏层维度列表
            "n_labels": 4,               # 标签数量
            "dropout": 0.1                # dropout率
        },
        "cls_loss_scale": 0.8,           # 分类损失权重
        "contra_loss_scale": 0.2         # 对比损失权重
    }

    params = DotDict(params)

    model = DuIN(params)

    batch_size = 32
    seq_len = 2100
    n_channels = 20
    X = torch.randn(batch_size, 20, seq_len)
    y_true = torch.randint(0, 4, (batch_size,))
    subj_id = torch.eye(1)[torch.randint(0, 1, (batch_size,))]
    y_pred, loss = model((X, y_true, subj_id))

    print(f"input.shape {X.shape}")
    print(f"out.shape {y_pred.shape}")