import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from utils import aggregate

class SHINE_BERT(nn.Module):
    """
    SHINE with BERT-TINY(2 layers) replacing GCN.
    - For each node type i=1..type_num-1:
        in_proj[i]: in_dim[i] -> hidden_size
        BERT(2L) with adjacency-based attention mask
        out_proj[i]: hidden_size -> out_dim[i]
        + optional concat with self.feature['word_emb'] when i==1 and concat_word_emb=True
    """
    def __init__(self, adj_dict, features_dict, in_features_dim, out_features_dim, params,
                 hidden_size: int = 128, num_heads: int = 4, intermediate_size: int = 512):
        super().__init__()
        self.adj = adj_dict
        self.feature = features_dict
        self.in_features_dim = in_features_dim
        self.out_features_dim = out_features_dim
        self.type_num = len(params.type_num_node)
        self.drop_out = params.drop_out
        self.concat_word_emb = params.concat_word_emb
        self.device = params.device

        # per-type projections (types indexed from 1..type_num-1 in original code)
        self.in_projs = nn.ModuleList()
        self.out_projs = nn.ModuleList()
        for i in range(1, self.type_num):
            self.in_projs.append(nn.Linear(self.in_features_dim[i], hidden_size))
            self.out_projs.append(nn.Linear(hidden_size, self.out_features_dim[i]))

        # BERT-TINY (2 layers)
        config = BertConfig(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            is_decoder=False,
            add_cross_attention=False
        )
        self.bert = BertModel(config).to(self.device)

    @staticmethod
    def _build_attn_mask_from_adj(adj: torch.Tensor, device, dtype):
        """
        adj: [N, N] (0/1). Ensures self-loop. Returns 4D additive mask: [1, 1, N, N]
        where allowed=0, blocked=-1e4 (added to attention scores).
        """
        N = adj.size(0)
        adj = adj.clone()
        adj.fill_diagonal_(1)                     # self-loops
        allow = (adj > 0).to(dtype=dtype, device=device)  # 1.0 for allowed
        mask = (1.0 - allow) * -1e4               # blocked -> large negative
        return mask.unsqueeze(0).unsqueeze(0)     # [1,1,N,N]

    def _encode_one_type(self, i: int):
        """
        Encode node type (1-based to match original): i in [1, type_num-1]
        """
        key = str(i)                               # features key: "1","2","3",...
        adj_key = f"{i}{i}"                        # intra-type adj key: "11","22",...
        x = self.feature[key].to(self.device)      # [N, in_dim_i]
        adj = self.adj[adj_key].to(self.device)    # [N, N]

        # proj in -> hidden
        x = self.in_projs[i-1](x)                  # [N, hidden]
        x = x.unsqueeze(0)                         # [1, N, hidden] for BERT

        # 4D additive attention mask from adjacency
        attn_mask_4d = self._build_attn_mask_from_adj(
            adj, device=self.device, dtype=x.dtype
        )                                          # [1,1,N,N]

        # BERT forward with inputs_embeds
        out = self.bert(inputs_embeds=x, attention_mask=attn_mask_4d).last_hidden_state
        out = out.squeeze(0)                       # [N, hidden]

        # proj hidden -> out_dim
        out = self.out_projs[i-1](out)             # [N, out_dim_i]
        out = F.dropout(out, p=self.drop_out, training=self.training)

        # optional concat with external word embeddings (match original i==1 branch)
        if (i == 2) and self.concat_word_emb:      # original code's i==1 branch maps to type index==2 here?
            # 주의: 원래 코드에서 i==1은 range(type_num-1) 기준 두 번째 타입.
            # 여기선 실제 타입 인덱스 i가 2일 때가 동일한 지점이 된다.
            word_emb = self.feature['word_emb'].to(self.device)  # [N, D_word]
            out = torch.cat([out, word_emb], dim=-1)             # [N, out_dim_i + D_word]
        return out

    def embed_component(self, norm=True):
        outputs = []
        # original loops i in range(self.type_num-1) over 0-based;
        # here we use real type indices 1..type_num-1 for clarity
        for t in range(1, self.type_num):
            outputs.append(self._encode_one_type(t))

        # aggregate over types (kept as in original)
        refined = aggregate(self.adj, outputs, self.type_num - 1)

        if norm:
            refined_norm = []
            for x in refined:
                refined_norm.append(x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-9))
            return refined_norm
        return refined

    def forward(self, epoch=None):
        return self.embed_component()
