import torch
import torch.nn as nn

class ReverseLayerF(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

class Attention(nn.Module):
	"""Attention mechanism.
	Parameters
	----------
	dim : int
		The input and out dimension of per token features.
	n_heads : int
		Number of attention heads.
	qkv_bias : bool
		If True then we include bias to the query, key and value projections.
	attn_p : float
		Dropout probability applied to the query, key and value tensors.
	proj_p : float
		Dropout probability applied to the output tensor.
	Attributes
	----------
	scale : float
		Normalizing consant for the dot product.
	qkv : nn.Linear
		Linear projection for the query, key and value.
	proj : nn.Linear
		Linear mapping that takes in the concatenated output of all attention
		heads and maps it into a new space.
	attn_drop, proj_drop : nn.Dropout
		Dropout layers.
	"""
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
		"""Run forward pass.
		Parameters
		----------
		x : torch.Tensor
			Shape `(n_samples, n_patches + 1, dim)`.
		Returns
		-------
		torch.Tensor
			Shape `(n_samples, n_patches + 1, dim)`.
		"""
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
	"""Multilayer perceptron.
	Parameters
	----------
	in_features : int
		Number of input features.
	hidden_features : int
		Number of nodes in the hidden layer.
	out_features : int
		Number of output features.
	p : float
		Dropout probability.
	Attributes
	----------
	fc : nn.Linear
		The First linear layer.
	act : nn.GELU
		GELU activation function.
	fc2 : nn.Linear
		The second linear layer.
	drop : nn.Dropout
		Dropout layer.
	"""
	def __init__(self, in_features, hidden_features, out_features, p=0.):
		super().__init__()
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = nn.GELU()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(p)

	def forward(self, x):
		"""Run forward pass.
		Parameters
		----------
		x : torch.Tensor
			Shape `(n_samples, n_patches + 1, in_features)`.
		Returns
		-------
		torch.Tensor
			Shape `(n_samples, n_patches +1, out_features)`
		"""
		x = self.fc1(
				x
		) # (n_samples, n_patches + 1, hidden_features)
		x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
		x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
		x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
		x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

		return x

class AttentionBlock(nn.Module):
	"""Transformer block.
	Parameters
	----------
	dim : int
		Embeddinig dimension.
	n_heads : int
		Number of attention heads.
	mlp_ratio : float
		Determines the hidden dimension size of the `MLP` module with respect
		to `dim`.
	qkv_bias : bool
		If True then we include bias to the query, key and value projections.
	p, attn_p : float
		Dropout probability.
	Attributes
	----------
	norm1, norm2 : LayerNorm
		Layer normalization.
	attn : Attention
		Attention module.
	mlp : MLP
		MLP module.
	"""
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
		"""Run forward pass.
		Parameters
		----------
		x : torch.Tensor
			Shape `(n_samples, n_patches + 1, dim)`.
		Returns
		-------
		torch.Tensor
			Shape `(n_samples, n_patches + 1, dim)`.
		"""
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

		# self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

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

class QAB_Transformer(nn.Module):
	def __init__(
		self,
		num_tokens,
		num_embeddings_outputs=1000,
		input_pad_idx=-1,
		qab_pad_idx=-1,
		embed_dim=128,
		depth=12, 			n_heads=12, mlp_ratio=4.0,
		qkv_bias=True, 		p=0.0, 		attn_p=0.0,
	):
		super().__init__()

		self.transformer			= Transformer(embed_dim=embed_dim, depth=depth, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, attn_p=attn_p);

		self.input_pad_idx			= input_pad_idx;

		self.input_embeddings		= torch.nn.Embedding( num_tokens + 1, 	embedding_dim=embed_dim, padding_idx=input_pad_idx );
		self.qab_embeddings			= torch.nn.Embedding( 4, 				embedding_dim=embed_dim, padding_idx=qab_pad_idx );

		self.norm					= nn.LayerNorm(embed_dim, eps=1e-6)
		self.cls_embeddings_outputs	= nn.Linear(embed_dim, num_embeddings_outputs);

	def forward(self, inputs_idxs, qab_idxs):

		# Compute attention mask
		mask 				= (inputs_idxs != self.input_pad_idx).unsqueeze(1).repeat(1, inputs_idxs.size(1), 1).unsqueeze(1)

		# a.) Extract the embeddings based on the input indices
		# b.) Extract the QAB embeddings
		input_tokens		= self.input_embeddings(inputs_idxs);
		qab_tokens			= self.qab_embeddings(qab_idxs);

		input_tokens 		= input_tokens + qab_tokens;

		# Feed embeddings through the transformer
		x 					= self.transformer(input_tokens, mask);

		# Normalize the final embedding outputs
		x 					= self.norm(x)

		# Predict from each embedding
		cls_embed_output	= self.cls_embeddings_outputs(x);

		return cls_embed_output;
