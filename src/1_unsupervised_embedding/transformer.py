import torch
import torch.nn as nn

class Attention(nn.Module):
	def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
		super().__init__()
		self.n_heads = n_heads
		self.dim = dim
		self.head_dim = dim // n_heads
		self.scale = self.head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_p)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_p)

	def forward(self, x, mask):
		n_samples, n_tokens, dim = x.shape

		if dim != self.dim:
			raise ValueError

		qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
		qkv = qkv.reshape(
				n_samples, n_tokens, 3, self.n_heads, self.head_dim
		)  # (n_samples, n_patches + 1, 3, n_heads, head_dim)
		qkv = qkv.permute(
				2, 0, 3, 1, 4
		)  # (3, n_samples, n_heads, n_patches + 1, head_dim)

		q, k, v = qkv[0], qkv[1], qkv[2]
		k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
		dp = (
		   q @ k_t
		) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)

		# Perform masked attention
		dp 	= dp.masked_fill(mask == 0, -1e4);

		attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
		attn = self.attn_drop(attn)

		weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
		weighted_avg = weighted_avg.transpose(
				1, 2
		)  # (n_samples, n_patches + 1, n_heads, head_dim)
		weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

		x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
		x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

		return x

class MLP(nn.Module):
	def __init__(self, in_features, hidden_features, out_features, p=0.):
		super().__init__()
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = nn.GELU()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(p)

	def forward(self, x):
		x = self.fc1(
				x
		) # (n_samples, n_patches + 1, hidden_features)
		x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
		x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
		x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
		x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

		return x

class AttentionBlock(nn.Module):
	def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim, eps=1e-6)
		self.attn = Attention(
				dim,
				n_heads=n_heads,
				qkv_bias=qkv_bias,
				attn_p=attn_p,
				proj_p=p
		)
		self.norm2 = nn.LayerNorm(dim, eps=1e-6)
		hidden_features = int(dim * mlp_ratio)
		self.mlp = MLP(
				in_features=dim,
				hidden_features=hidden_features,
				out_features=dim,
		)

	def forward(self, x, mask):
		x = x + self.attn(self.norm1(x), mask)
		x = x + self.mlp(self.norm2(x))

		return x

class Transformer(nn.Module):
	def __init__(
			self,
			embed_dim=128,
			depth=12, 			n_heads=12, mlp_ratio=4.0,
			qkv_bias=True, 		p=0.0, 		attn_p=0.0,
	):
		super().__init__()

		self.blocks = nn.ModuleList(
			[
				AttentionBlock(
					dim=embed_dim,
					n_heads=n_heads,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					p=p,
					attn_p=attn_p,
				)
				for _ in range(depth)
			]
		)


	def forward(self, x, mask):

		for block in self.blocks:
			x = block(x, mask)

		return x;

class AB_Transformer(nn.Module):
	def __init__(
		self,
		num_tokens,
		num_outputs=1000,
		input_padding_idx=-1,
		embed_dim=128,
		depth=12, 			n_heads=12, mlp_ratio=4.0,
		qkv_bias=True, 		p=0.0, 		attn_p=0.0,
	):
		super().__init__()

		self.transformer			= Transformer(embed_dim=embed_dim, depth=depth, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p);

		self.input_padding_idx		= input_padding_idx;
		self.ab_padding_idx			= 2;

		self.input_embeddings		= torch.nn.Embedding( num_tokens + 1, 	embedding_dim=embed_dim, padding_idx=input_padding_idx );
		self.ab_embeddings			= torch.nn.Embedding( 3, 				embedding_dim=embed_dim, padding_idx=self.ab_padding_idx );

		self.cls_token 				= nn.Parameter( torch.zeros(1, 1, embed_dim) );

		self.norm					= nn.LayerNorm(embed_dim, eps=1e-6)
		self.cls_head 				= nn.Linear(embed_dim, num_outputs);

	def forward(self, inputs_idxs, ab_idxs, ab_padding_idx_tensor):

		# Compute attention mask
		n_samples 			= inputs_idxs.shape[0]
		mask 				= (torch.cat((torch.zeros_like(inputs_idxs)[:, 0:1], inputs_idxs), dim=1) != self.input_padding_idx).unsqueeze(1).repeat(1, inputs_idxs.size(1) + 1, 1).unsqueeze(1);

		# a.) Extract the embeddings based on the input indices
		# b.) Extract the AB embeddings
		input_tokens 		= self.input_embeddings(inputs_idxs)
		ab_tokens			= self.ab_embeddings(ab_idxs);

		# a.) Insert one CLS token at the start of the inputs
		# b.) Insert one segment-padding token at the start of the inputs
		cls_token 			= self.cls_token.expand( n_samples, -1, -1 );
		padding_ab_token	= self.ab_embeddings(ab_padding_idx_tensor).reshape(1, 1, -1).expand( n_samples, -1, -1 );

		input_tokens		= torch.cat( (cls_token, input_tokens), dim=1 );
		ab_tokens			= torch.cat( (padding_ab_token, ab_tokens), dim=1 );

		input_tokens		= input_tokens + ab_tokens;

		# Feed embeddings through the transformer
		x 					= self.transformer(input_tokens, mask);

		# Normalize the final embedding outputs
		x 					= self.norm(x)

		# Predict from the final token
		cls_token_final		= x[:, 0];
		cls_output			= self.cls_head(cls_token_final)

		return cls_output;