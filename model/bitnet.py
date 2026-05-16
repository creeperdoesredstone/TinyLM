import torch
import torch.nn as nn
import torch.nn.functional as F


class BitRound(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		return x.round().clamp(-1, 1)

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output


class BitSelfAttention(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.q_proj = BitLinear(dim, dim)
		self.k_proj = BitLinear(dim, dim)
		self.v_proj = BitLinear(dim, dim)
		self.out_proj = BitLinear(dim, dim)
		self.scale = dim**-0.5
		self._mask_cache = {}

	def _get_mask(self, T, device):
		if T not in self._mask_cache:
			self._mask_cache[T] = torch.tril(
				torch.ones(T, T, device=device, dtype=torch.bool)
			)
		return self._mask_cache[T]

	def forward(self, x):
		B, T, C = x.shape

		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)

		attn = (q @ k.transpose(-2, -1)) * self.scale
		mask = self._get_mask(T, x.device)
		attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
		attn = torch.softmax(attn, dim=-1)

		return self.out_proj(attn @ v)

class BitSWAttention(nn.Module):
    def __init__(self, dim, window_size=1024):
        super().__init__()
        self.window_size = window_size
        self.q_proj = BitLinear(dim, dim)
        self.k_proj = BitLinear(dim, dim)
        self.v_proj = BitLinear(dim, dim)
        self.out_proj = BitLinear(dim, dim)
        self.scale = dim**-0.5
        self._mask_cache = {}

    def _get_swa_mask(self, T, device):
        if T not in self._mask_cache:
            # Create standard causal mask
            mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            # Create sliding window constraint
            # Only keep elements where (row - col) < window_size
            row_idx = torch.arange(T, device=device).view(-1, 1)
            col_idx = torch.arange(T, device=device).view(1, -1)
            window_mask = (row_idx - col_idx) < self.window_size
            
            self._mask_cache[T] = mask & window_mask
        return self._mask_cache[T]

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = self._get_swa_mask(T, x.device)
        
        attn = attn.masked_fill(~mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)

        return self.out_proj(attn @ v)

class BitLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		self.weight = nn.Parameter(
			torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
		)
		self.bias = nn.Parameter(torch.zeros(out_features))
		self.alpha = nn.Parameter(torch.ones(out_features))


	def forward(self, x):
		rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
		x = x / rms
		w = self.weight
		w_bin = torch.tanh(w)
		w_bin = (w + (w_bin - w).detach()).clone()
		
		w_bin = w_bin * self.alpha.view(-1, 1)

		return F.linear(x.contiguous(), w_bin.contiguous(), self.bias)

	def quantize(self):
		with torch.no_grad():
			w_bin = self.weight.sign()
			w = w_bin + (self.weight - self.weight.detach())  # STE

			# per-row quantization
			gamma = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
			w_q = (w / gamma).round().clamp(-1, 1) * gamma

			self.weight.copy_(w_q)
			self.is_quantized = torch.tensor(True)


class BitNetBlock(nn.Module):
	def __init__(self, dim, dropout=0.1):
		super().__init__()

		self.norm1 = nn.LayerNorm(dim)
		self.attn = BitSWAttention(dim, 128)

		self.norm2 = nn.LayerNorm(dim)

		hidden = dim * 3
		self.up_proj = BitLinear(dim, hidden * 2)
		self.down_proj = BitLinear(hidden, dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		res = x
		x = self.norm1(x)
		x = res + self.attn(x)

		res = x
		x = self.norm2(x)

		x_proj = self.up_proj(x)
		gate, val = x_proj.chunk(2, dim=-1)
		x = F.silu(gate) * val

		x = self.down_proj(x)
		x = self.dropout(x)

		return res + x


class DeepScratchBitNet(nn.Module):
	def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.1):
		super().__init__()

		self.embed = nn.Embedding(vocab_size, embed_dim)
		self.in_proj = BitLinear(embed_dim, hidden_dim)
		self.layers = nn.ModuleList(
			[BitNetBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
		)

		self.norm_final = nn.LayerNorm(hidden_dim)

		self.out_proj = BitLinear(hidden_dim, vocab_size)

	def forward(self, x):
		x = self.embed(x) * (self.embed.embedding_dim**0.5)

		x = self.in_proj(x)

		for layer in self.layers:
			x = layer(x)

		x = self.norm_final(x)

		return self.out_proj(x)
