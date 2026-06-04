import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadCrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, hidden_dim, out_dim=1, num_heads=1, use_layernorm=True, use_learned_query=True):
        super().__init__()
        assert hidden_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim # // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_layernorm = use_layernorm
        self.use_learned_query = use_learned_query

        # Linear projections
        self.q_proj = nn.Linear(q_dim, num_heads * hidden_dim)
        # self.q_proj = val_block(input_size=q_dim,
        #                         hidden_size=hidden_dim,
        #                         output_size=num_heads * hidden_dim)
        self.k_proj = nn.Linear(k_dim, num_heads * hidden_dim)
        # self.k_proj = val_block(input_size=k_dim,
        #                         hidden_size=hidden_dim,
        #                         output_size=num_heads * hidden_dim)
        self.v_proj = nn.Linear(k_dim, num_heads * hidden_dim)
        # self.v_proj = val_block(input_size=k_dim,
        #                         hidden_size=hidden_dim,
        #                         output_size=num_heads * hidden_dim)

        self.out_proj = nn.Linear(num_heads * hidden_dim, hidden_dim)
        # self.out_proj = val_block(input_size=num_heads * hidden_dim,
        #                           hidden_size=num_heads * hidden_dim,
        #                           output_size=hidden_dim)

        self.final_out = nn.Linear(hidden_dim, out_dim)
        # self.final_out = val_block(input_size=hidden_dim,
        #                           hidden_size=hidden_dim,
        #                           output_size=out_dim)

        # Layer norms
        if use_layernorm:
            self.key_norm = nn.LayerNorm(k_dim)
            self.value_norm = nn.LayerNorm(k_dim)
            self.query_norm = nn.LayerNorm(q_dim)
            self.out_norm = nn.LayerNorm(hidden_dim)

        # Optional learned query
        if use_learned_query:
            self.learned_query = nn.Parameter(torch.randn(1, 1, q_dim))

    def forward(self, k, q=None, mask=None):
        """
        key:   [B, T_kv, D_kv]
        value: [B, T_kv, D_kv]
        query: [B, T_q, D_q] or None → becomes [B, 1, D_q] via learned vector
        mask:  [B, T_q, T_kv] or [B, 1, T_kv] -- True for masked positions
        """
        if q == None:
            key, value = k.float(), k.float()
        else:
            query, key, value = q.float(), k.float(), k.float()


        B, T_kv, _ = key.shape

        # Optional norm
        if self.use_layernorm:
            key = self.key_norm(key)
            value = self.value_norm(value)

        # Handle query
        if q is None:
            if self.use_learned_query:
                query = self.learned_query.expand(B, -1, -1)  # [B, 1, D_q]
            else:
                raise ValueError("query is None and use_learned_query=False")

        else:
            if self.use_layernorm:
                query = self.query_norm(query)

        T_q = query.shape[1]

        # Linear projections
        Q = self.q_proj(query)  # [B, T_q, E]
        K = self.k_proj(key)    # [B, T_kv, E]
        V = self.v_proj(value)  # [B, T_kv, E]

        # Reshape to multi-head
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_q, D_h]
        K = K.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, T_kv, D_h]
        V = V.view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, T_kv, D_h]

        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, T_q, T_kv]

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, T_q, T_kv]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T_q, T_kv]
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T_q, D_h]

        # Combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, -1)  # [B, T_q, E]
        out = self.out_proj(attn_output)

        # Residual (handle dim mismatch)
        if query.shape[-1] != out.shape[-1]:
            residual = self.q_proj(query)
        else:
            residual = query

        out = out + residual

        if self.use_layernorm:
            out = self.out_norm(out)

        out = self.final_out(out)

        return out

class AFTFullMultiHead(nn.Module):
    def __init__(self, max_seqlen, k_dim, q_dim=None, hidden_dim=64, num_heads=1, out_dim=1):
        super().__init__()
        '''
        Vectorized Multi-head AFT-Full (parallelized across heads)
        - Each head has same hidden_dim (not split)
        - num_heads and hidden_dim are independent
        '''
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.max_seqlen = max_seqlen
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Unified projection layers (batchified for parallel heads)
        if q_dim is not None:
            # self.to_q = val_block(input_size=self.q_dim,
            #                       hidden_size=self.hidden_dim,
            #                       output_size=num_heads * hidden_dim)
            self.to_q = nn.Linear(self.q_dim, num_heads * hidden_dim)
        # self.to_k = val_block(input_size=self.k_dim,
        #                       hidden_size=self.hidden_dim,
        #                       output_size=num_heads * hidden_dim)
        self.to_k = nn.Linear(self.k_dim, num_heads * hidden_dim)
        # self.to_v = val_block(input_size=self.k_dim,
        #                       hidden_size=self.hidden_dim,
        #                       output_size=num_heads * hidden_dim)
        self.to_v = nn.Linear(self.k_dim, num_heads * hidden_dim)

        # Final projection: concatenate all heads → project to output dim
        # self.project = val_block(input_size=num_heads * hidden_dim,
        #                          hidden_size=self.hidden_dim,
        #                          output_size=out_dim)
        self.project = nn.Linear(num_heads * hidden_dim, out_dim)
        # Batched bias: one per head [H, T_q, T_k]
        self.wbias = nn.Parameter(torch.empty(num_heads, max_seqlen, max_seqlen))
        nn.init.xavier_uniform_(self.wbias)

    def forward(self, q=None, k=None, mask=None):
        B, T_k, _ = k.shape
        T_q = T_k if q is None else q.shape[1]

        # Projections (Q is optional)
        K = self.to_k(k).view(B, T_k, self.num_heads, self.hidden_dim).transpose(1, 2)  # [B, H, T_k, D]
        V = self.to_v(k).view(B, T_k, self.num_heads, self.hidden_dim).transpose(1, 2)  # [B, H, T_k, D]

        if q is not None:
            Q = self.to_q(q).view(B, T_q, self.num_heads, self.hidden_dim).transpose(1, 2)  # [B, H, T_q, D]
        else:
            Q = None

        # Compute attention weights
        wb = self.wbias[:, :T_q, :T_k]  # [H, T_q, T_k]
        weights = torch.abs(wb).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, T_q, T_k]

        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(1) == 0, -1e9)
            weights = torch.abs(weights)

        exp_K = torch.abs(K)  # [B, H, T_k, D]
        KV = exp_K * V  # [B, H, T_k, D]

        # AFT computation
        numerator = torch.matmul(weights, KV)        # [B, H, T_q, D]
        denominator = torch.matmul(weights, exp_K)   # [B, H, T_q, D]
        output = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

        # Modulate with Q (if present)
        if Q is not None:
            Q_sigmoid = torch.sigmoid(Q)  # [B, H, T_q, D]
            Yt = Q_sigmoid * output       # [B, H, T_q, D]
            Yt = Yt.transpose(1, 2).contiguous().view(B, T_q, self.num_heads * self.hidden_dim)  # [B, T_q, H*D]
        else:
            Yt = output.mean(dim=2, keepdim=True)  # average over time dim
            Yt = Yt.transpose(1, 2).contiguous().view(B, 1, self.num_heads * self.hidden_dim)  # [B, T_q, H*D]

        Yt = self.project(Yt)  # [B, T_q, output_dim]
        return Yt
        

class val_block(nn.Module):
    '''PyTorch implementation of the Decoder'''

    def __init__(self, **kwargs):
        nn.Module.__init__(self)

        input_size = kwargs['input_size']
        hidden_size = kwargs['hidden_size']
        output_size = kwargs['output_size']
        self.hidden_size = hidden_size
        self.output_size = output_size

        # self.out_1 = nn.Linear(input_size, self.hidden_size)
        # self.out_2 = nn.Linear(input_size, self.hidden_size)
        # self.out_3 = nn.Linear(input_size, self.hidden_size)
        # self.out = nn.Linear(self.hidden_size, output_size)

        self.out_1 = nn.Linear(input_size, self.hidden_size)
        # self.out_2 = nn.Linear(self.hidden_size, self.hidden_size)
        # self.out_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, decoder_input):  # , decoder_h0):
        # r_t = F.sigmoid(self.out_1(decoder_input))
        # z_t = F.sigmoid(self.out_2(decoder_input))
        # n_t = F.tanh(self.out_3(decoder_input) + r_t)
        # h_t = (1-z_t)*(n_t)
        # output = self.out(h_t)

        x = F.gelu(self.out_1(decoder_input))
        # x = self.out_2(x)
        # x = F.gelu(self.out_3(x))

        output = self.out(x)

        return output


class BatchedAFTFullConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads

        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.out_per_head = out_channels // heads

        self.lin_q = nn.Linear(in_channels, heads * hidden_channels)
        self.lin_k = nn.Linear(in_channels, heads * hidden_channels)
        self.lin_v = nn.Linear(in_channels, heads * hidden_channels)

        self.lin_out = nn.Linear(heads * hidden_channels, out_channels)
        self.lin_skip = nn.Linear(in_channels, out_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None, edge_mask=None, node_mask=None):
        """
        x:           [B, N, F]
        edge_index:  [B, 2, E]
        edge_weight: [B, E] (optional)
        edge_mask:   [B, E]
        node_mask:   [B, N]
        """
        B, N, _ = x.shape
        _, _, E = edge_index.shape
        H, D = self.heads, self.hidden_channels

        q = self.lin_q(x).view(B, N, H, D)  # [B, N, H, D]
        k = self.lin_k(x).view(B, N, H, D)
        v = self.lin_v(x).view(B, N, H, D)

        # Reshape edge index to [B * E]
        src = edge_index[:, 0, :]  # [B, E]
        tgt = edge_index[:, 1, :]  # [B, E]
        batch_idx = torch.arange(B, device=x.device).view(-1, 1).expand(-1, E)  # [B, E]

        # Gather k and v at source nodes
        k_src = k[batch_idx, src]  # [B, E, H, D]
        v_src = v[batch_idx, src]  # [B, E, H, D]
        q_tgt = q[batch_idx, tgt]  # [B, E, H, D]

        # Bias from structural edge weights (broadcasted to match [B, E, H, 1])
        if edge_weight is not None:
            bias = edge_weight.unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
        else:
            bias = torch.zeros(B, E, 1, 1, device=x.device)

        logits = k_src + bias  # [B, E, H, D]
        logits = logits.clamp(min=-30, max=30)  # for stability

        weights = logits.exp()  # [B, E, H, D]
        weighted_v = weights * v_src  # [B, E, H, D]

        if edge_mask is not None:
            mask = edge_mask.unsqueeze(-1).unsqueeze(-1)  # [B, E, 1, 1]
            weighted_v = weighted_v * mask
            weights = weights * mask

        # Aggregate to nodes: sum over all incoming edges
        out_num = torch.zeros(B, N, H, D, device=x.device)
        out_den = torch.zeros(B, N, H, D, device=x.device)

        tgt_flat = tgt + torch.arange(B, device=x.device).view(-1, 1) * N  # [B, E]
        tgt_flat = tgt_flat.view(-1)  # [B * E]

        out_num_flat = out_num.view(B * N, H, D)
        out_den_flat = out_den.view(B * N, H, D)

        scatter_indices = (batch_idx * N + tgt).view(-1)

        out_num_flat.index_add_(0, scatter_indices, weighted_v.view(-1, H, D))
        out_den_flat.index_add_(0, scatter_indices, weights.view(-1, H, D))

        out = out_num / (out_den + 1e-8)  # avoid divide-by-zero
        out = out.view(B, N, H * D)
        out = self.lin_out(out) + self.lin_skip(x)

        if node_mask is not None:
            # print(f"*********************************************************")
            # print(f"out.shape: {out.shape}, node_mask.unsqueeze(-1).shape: {node_mask.unsqueeze(-1).shape}")
            # print(f"*********************************************************")

            out = out * node_mask.unsqueeze(-1)

        return out
