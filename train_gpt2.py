import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
import math

# Set this to True to enable debugging prints
debugF = False
debugB = False

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1): # split grouped QKV parameters
                    g = torch.cat([zeropower_backend(g1, steps=group['backend_steps']) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'])
                    scale = max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                p.data.add_(g, alpha=-lr * scale)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().to(dtype=x.dtype)
            self.sin_cached = freqs.sin().to(dtype=x.dtype)
            if debugF:
                print(f"[Rotary] Cached cos and sin with shapes: {self.cos_cached.shape}, {self.sin_cached.shape}")
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class DifferentialCausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head // 2
        assert self.n_embd % (self.n_head * 2) == 0, "Embedding dimension must be divisible by 2 * n_head"
        self.depth = layer_idx

        # Linear Projections
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Output Projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

        # Rotary embedding
        self.rotary = Rotary(self.head_dim)

        # Differential attention initialization
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * self.depth)

        # Differential attention parameters
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)

        # Normalization layers
        self.q_rmsnorm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.k_rmsnorm = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        self.subln = nn.LayerNorm(2 * self.head_dim, elementwise_affine=False)

    def forward(self, x):
        B, T, C = x.size()
        if debugF:
            print(f"[DifferentialCausalSelfAttention] Layer {self.depth}, Input x shape: {x.shape}")

        # Project x to queries, keys, values
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] q, k, v shapes: {q.shape}, {k.shape}, {v.shape}")

        # Reshape for differential attention
        q = q.view(B, T, 2 * self.n_head, self.head_dim)
        k = k.view(B, T, 2 * self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, 2 * self.head_dim)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] Reshaped q, k, v shapes: {q.shape}, {k.shape}, {v.shape}")

        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q = self.q_rmsnorm(q)
        k = self.k_rmsnorm(k)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, 2*n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, 2*n_head, T, head_dim)
        v = v.transpose(1, 2)  # (B, n_head, T, 2*head_dim)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] Transposed q, k, v shapes: {q.shape}, {k.shape}, {v.shape}")

        # Scale queries
        q = q * (1.0 / math.sqrt(self.head_dim))

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1))  # (B, 2*n_head, T, T)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] attn_weights shape: {attn_weights.shape}")

        # Apply causal mask
        attn_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(attn_weights.device)
        attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)

        # Reshape attention weights to separate components
        attn_weights = attn_weights.view(B, self.n_head, 2, T, T)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] Reshaped attn_weights shape: {attn_weights.shape}")

        # Compute lambda values
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        if debugF:
            print(f"[DifferentialCausalSelfAttention] lambda_1: {lambda_1.item()}, lambda_2: {lambda_2.item()}, lambda_full: {lambda_full.item()}")

        # Differential attention weights
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        if debugF:
            print(f"[DifferentialCausalSelfAttention] Differential attn_weights shape: {attn_weights.shape}")

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] attn_output shape before normalization: {attn_output.shape}")

        # Apply normalization per head
        attn_output = attn_output.view(B * self.n_head, T, 2 * self.head_dim)
        attn_output = self.subln(attn_output)
        attn_output = attn_output.view(B, self.n_head, T, 2 * self.head_dim)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] attn_output shape after normalization: {attn_output.shape}")

        # Scale output
        attn_output = attn_output * (1 - self.lambda_init)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn = self.c_proj(attn_output)
        if debugF:
            print(f"[DifferentialCausalSelfAttention] Output attn shape: {attn.shape}")
        return attn

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ~1-2% better than GELU
        x = self.c_proj(x)
        if debugF:
            print(f"[MLP] Output x shape: {x.shape}")
        return x

class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = DifferentialCausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.ln1 = nn.RMSNorm(config.n_embd, elementwise_affine=False)
        self.ln2 = nn.RMSNorm(config.n_embd, elementwise_affine=False)

    def forward(self, x):
        if debugF:
            print(f"[Block] Layer {self.attn.depth}, Input x shape: {x.shape}")
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if debugF:
            print(f"[Block] Layer {self.attn.depth}, Output x shape: {x.shape}")
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # head dim 128 suggested by @Grad62304977
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h   = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.ln_f = nn.RMSNorm(config.n_embd, elementwise_affine=False)

    def forward(self, idx, targets=None, return_logits=True):

        # Forward the GPT model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        if debugF:
            print(f"[GPT] Input embeddings x shape: {x.shape}")
        for block in self.transformer.h:
            x = block(x)
        x = self.ln_f(x)
        if debugF:
            print(f"[GPT] Final x shape after transformer: {x.shape}")

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float()  # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            if debugF:
                print(f"[GPT] Calculated loss: {loss.item()}")
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # preserve the time dim
            logits = logits.float()  # use tf32/fp32 for logits
            loss = None

        # There are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # Only reads the header, returns header data
    with open(filename, "rb") as f:
        # First read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # First read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # The rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # Glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # Load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # Kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        end_position = self.current_position + B * T + 1
        if end_position > len(self.tokens):
            self.advance()
            return self.next_batch()
        buf = self.tokens[self.current_position:end_position]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # Advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # Data hyperparameters
    input_bin: str = 'data/fineweb10B/fineweb_train_*.bin'  # input .bin to train on
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'  # input .bin to eval validation loss on
    # Optimization hyperparameters
    batch_size: int = 8 * 64  # batch size, in sequences, across all devices
    device_batch_size: int = 64  # batch size, in sequences, per device
    sequence_length: int = 1024  # sequence length, in tokens
    num_iterations: int = 5100  # number of iterations to run
    learning_rate: float = 0.0036
    warmup_iters: int = 0
    warmdown_iters: int = 1450  # number of iterations of linear warmup/warmdown
    weight_decay: float = 0
    # Evaluation and logging hyperparameters
    val_loss_every: int = 125  # every how many steps to evaluate val loss?
    val_tokens: int = 10485760  # how many tokens of validation data?
    save_every: int = 0  # every how many steps to save the checkpoint?
args = Hyperparameters()

# Set up DDP (distributed data parallel). torchrun sets this env variable
assert torch.cuda.is_available()
torch.autograd.set_detect_anomaly(True) # May delete this later

dist.init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
print(f"using device: {device}")
master_process = (ddp_rank == 0)  # this process will do logging, checkpointing etc.

# Convenience variables
B, T = args.device_batch_size, args.sequence_length
# Calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# Calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# Load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# There are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
num_vocab = 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=6, n_embd=768))
model = model.cuda()
if hasattr(config, "coordinate_descent_tuning"):
    config.coordinate_descent_tuning = True  # suggested by @Chillee
model = torch.compile(model)

print(f"RANK: {os.environ.get('RANK')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

device = torch.device(f'cuda:{ddp_local_rank}')

# Wrap model into DDP container
print("Setting device")
torch.cuda.set_device(device)
print("Wrapping model in DDP")
model = DDP(model, device_ids=[ddp_local_rank])

print("Called DDP on model")
raw_model = model.module  # always contains the "raw" unwrapped model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# Initialize the optimizer(s)

# Collect 'lambda' parameters
lambda_params = []
for block in raw_model.transformer.h:
    attn = block.attn
    lambda_params.extend([attn.lambda_q1, attn.lambda_k1, attn.lambda_q2, attn.lambda_k2])

# Create a set of IDs for 'lambda' parameters
lambda_param_ids = set(id(p) for p in lambda_params)

# Collect other parameters for Muon optimizer by excluding 'lambda' parameters
all_h_params = list(raw_model.transformer.h.parameters())
params_for_muon = [p for p in all_h_params if id(p) not in lambda_param_ids]

# Optimizer for lm_head.parameters()
optimizer1 = torch.optim.AdamW(
    raw_model.lm_head.parameters(),
    lr=args.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=args.weight_decay,
    fused=True
)

# Optimizer for transformer.h.parameters() excluding 'lambda' parameters
optimizer2 = Muon(
    params_for_muon,
    lr=0.1 * args.learning_rate,
    momentum=0.95,
    rank=ddp_rank,
    world_size=ddp_world_size
)

# Optimizer for 'lambda' parameters
optimizer3 = torch.optim.AdamW(
    lambda_params,
    lr=args.learning_rate,
    betas=(0.9, 0.95),
    weight_decay=args.weight_decay,
    fused=True
)

optimizers = [optimizer1, optimizer2, optimizer3]
# Learning rate schedulers
def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio

# Schedulers for each optimizer
schedulers = [
    torch.optim.lr_scheduler.LambdaLR(optimizer1, get_lr),
    torch.optim.lr_scheduler.LambdaLR(optimizer2, get_lr),
    torch.optim.lr_scheduler.LambdaLR(optimizer3, get_lr)
]

# Learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) Linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    # 2) Constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) Linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # Create the log file
    with open(logfile, "w") as f:
        # Begin the log by printing this file (the Python code)
        f.write('=' * 100 + '\n')
        f.write(code)
        f.write('=' * 100 + '\n')
        # Log information about the hardware/software environment
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('=' * 100 + '\n')

training_time_ms = 0
# Start the clock
torch.cuda.synchronize()
t0 = time.time()
# Begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # Ignore timing first 10 steps
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # Evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # Stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # Run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # Log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms '
                  f'step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms '
                        f'step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
        # Start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # Stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # Save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(),
                   optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # Start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        # Forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # Advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # Backward pass
        if i < train_accumulation_steps:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
            # Inside the training loop, after loss.backward()

    # Check for NaNs in gradients and print gradient statistics
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                print(f"[Training Loop] NaN detected in gradients of parameter '{name}' with shape {p.shape}")
            if debugB:
                grad_min = p.grad.min().item()
                grad_max = p.grad.max().item()
                grad_mean = p.grad.mean().item()
                print(f"[Training Loop] Grad stats for param '{name}': min={grad_min}, max={grad_max}, mean={grad_mean}")
        else:
            if debugB:
                print(f"[Training Loop] Gradient is None for parameter '{name}' with shape {p.shape}")

    for p in model.parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"NaN detected in gradients of parameter {p.shape}")
        p.grad /= train_accumulation_steps



    # Step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # Null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------
# everything that follows now is just diagnostics, prints, logging, etc.

#dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
if master_process:
    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
    with open(logfile, "a") as f:
        f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

# -------------------------------------------------------------------------
# clean up nice
dist.destroy_process_group()