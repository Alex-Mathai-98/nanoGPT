import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

########## Leave untouched ##########
@dataclass
class GPTConfig:
    block_size : int = 1024 # I like to call this sequence length
    vocab_size : int =  50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout:float = 0.0
    bias:bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
#####################################

class MLP(nn.Module) :

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module) :

    def __init__(self, config):
        super().__init__()
        self.config = config

        # pre-norm helps greatly stabilize gradients
        self.pre_norm = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.mha = MultiHeadAttention(config)
        # post-norm helps stabilize gradients
        self.post_norm = nn.LayerNorm(config.n_embed, bias=config.bias)

        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs) :

        # pre-norm helps stabilize gradients
        inputs = self.pre_norm(inputs)

        # runs the normalized inputs into each attention head 
        # of a multi-head attention block.
        mha_output = self.mha(inputs)

        # add skip-connection and then post-norm
        mlp_input = mha_output + inputs
        mlp_input = self.post_norm(mlp_input)

        # run MLP and add skip-connection
        mlp_output = self.mlp(mlp_input) + mlp_input

        # apply optional dropout
        final_output = self.dropout(mlp_output)

        return final_output

class MultiHeadAttention(nn.Module) :

    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.attention_heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.c_proj = nn.Linear(self.n_head*self.n_embed, self.n_embed)
        
    def forward(self, inputs) :
        """ Concatenate the outputs from multiple attention heads. """
        outputs = []
        for single_head in self.attention_heads :
            outputs.append(single_head(inputs))
        multi_head_output = self.c_proj(torch.concat(outputs,dim=-1))
        return multi_head_output

class SingleHeadAttention(nn.Module) :

    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embed
        self.dropout = config.dropout

        self.k_proj = nn.Linear(in_features=self.n_embed, out_features=self.n_embed)
        self.q_proj = nn.Linear(in_features=self.n_embed, out_features=self.n_embed)
        self.v_proj = nn.Linear(in_features=self.n_embed, out_features=self.n_embed)

        self.dropout = nn.Dropout(self.dropout)

        # register a buffer for the causal mask - rather than creating it 
        # again and again in the forward pass
        causal_mask = torch.tril(torch.ones((config.block_size,config.block_size))).view(1,config.block_size,config.block_size)
        self.register_buffer("bias", causal_mask)

    def apply_causal_mask(self, attn_scores) :
        """ Make the attention scores '-infinity' where the causal mask is '0'. 
        
        Arguments
            attn_scores : [N, seq_len, seq_len]
        """
        seq_len = attn_scores.size()[1]
        attn_scores = torch.masked_fill(attn_scores, self.bias[:,:seq_len,:seq_len]==0, float('-inf'))
        return attn_scores

    def forward(self, inputs) :
        """
        Arguments
            inputs : [N, seq_len, n_embed]
        """

        keys = self.k_proj(inputs) # [N, seq_len, n_embed]
        queries = self.q_proj(inputs) # [N, seq_len, n_embed]
        values = self.v_proj(inputs) # [N, seq_len, n_embed]

        # attention scores
        attn_scores = queries @ keys.transpose(1,2) # [N, seq_len, seq_len]
        attn_scores = attn_scores/math.sqrt(self.n_embed) # [N, seq_len, seq_len]
        # Apply the following 
        # 1. Causal Mask
        # 2. Optional dropout to the attention matrix 
        # 3. Softmax
        attn_scores = self.apply_causal_mask(attn_scores)
        attn_scores = self.dropout(attn_scores)
        attn_scores = F.softmax(attn_scores,dim=2)

        # get final values
        final_vals = attn_scores @ values

        return final_vals

class GPT(nn.Module) :

    def __init__(self, config):
        super().__init__()

        self.config = config

        # collect all the paramters here
        self.transformer = nn.ModuleDict(
            dict(
                token_embed = nn.Embedding(config.vocab_size, config.n_embed),
                pos_embed = nn.Embedding(config.block_size, config.n_embed),
                blocks = nn.ModuleList( Block(config) for _ in range(config.n_layer)),
                ln = nn.LayerNorm(config.n_embed, bias=config.bias),
                dropout = nn.Dropout(config.dropout)
            )
        )

        # final vocab prediction
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.transformer.token_embed.weight

        ########## copied code from Andrej Karpathy ##########
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        #########################################################

    ########## copied code from Andrej Karpathy ##########
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.token_embed.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #########################################################

    def forward(self, inputs, targets=None) :
        """
        Arguments :
            'inputs' - Input size of [batch, seq_len]. The inputs are token IDs.
            'targets' - used only in training stage. [batch, seq_len]
        """

        bs, seq_len = inputs.size()

        token_embeds = self.transformer.token_embed(inputs)

        positions = torch.arange(seq_len).unsqueeze(0).repeat(bs,1)
        pos_embeds = self.transformer.pos_embed(positions)

        input_embeds = token_embeds + pos_embeds
        input_embeds = self.transformer.dropout(input_embeds)

        output = input_embeds
        for block in self.transformer.blocks :
            output = block(output)

        output = self.transformer.ln(output)

        # logits dimension = [bs, seq_len, vocab_size]
        logits = self.lm_head(output)

        if targets is not None :
            loss = F.cross_entropy(output.view(bs*seq_len, -1), 
                            targets.view(bs*seq_len),
                            ignore_index=-1) # whatever the ignore index is
            return logits, loss
        else :
            return logits, None
        

if __name__ == '__main__' :

    gpt_config = GPTConfig()
    gpt_model = GPT(gpt_config)

    inputs = torch.randint(low=0, high=10, size=(1,10))

    gpt_model(inputs)



