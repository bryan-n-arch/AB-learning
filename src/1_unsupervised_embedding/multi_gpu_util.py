import numpy as np

import torch
torch.backends.cudnn.benchmark = True
import torch.optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def ReduceTensor(tensor, worldSize, average=False):
	rt = tensor.clone();
	dist.all_reduce(rt, op=dist.ReduceOp.SUM);

	if average:
		rt /= worldSize;

	return rt;

def GatherTensor(tensor, worldSize):

	tensorList = [torch.ones_like(tensor).cuda() for _ in range(dist.get_world_size())];
	dist.all_gather(tensorList, tensor);

	return tensorList;

class Lamb(torch.optim.Optimizer):
	r"""Implements Lamb algorithm.
	It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
	Arguments:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		lr (float, optional): learning rate (default: 1e-3)
		betas (Tuple[float, float], optional): coefficients used for computing
			running averages of gradient and its square (default: (0.9, 0.999))
		eps (float, optional): term added to the denominator to improve
			numerical stability (default: 1e-8)
		weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
		adam (bool, optional): always use trust ratio = 1, which turns this into
			Adam. Useful for comparison purposes.
	.. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
		https://arxiv.org/abs/1904.00962
	"""

	def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
				 weight_decay=0, adam=False):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
		defaults = dict(lr=lr, betas=betas, eps=eps,
						weight_decay=weight_decay)
		self.adam = adam
		super(Lamb, self).__init__(params, defaults)

	def step(self, closure=None):
		"""Performs a single optimization step.
		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad.data
				if grad.is_sparse:
					raise RuntimeError('Lamb does not support sparse gradients.')

				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state['step'] = 0
					# Exponential moving average of gradient values
					state['exp_avg'] = torch.zeros_like(p.data)
					# Exponential moving average of squared gradient values
					state['exp_avg_sq'] = torch.zeros_like(p.data)

				exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
				beta1, beta2 = group['betas']

				state['step'] += 1

				# Decay the first and second moment running average coefficient
				# m_t
				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				# v_t
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

				# Paper v3 does not use debiasing.
				# bias_correction1 = 1 - beta1 ** state['step']
				# bias_correction2 = 1 - beta2 ** state['step']
				# Apply bias to lr to avoid broadcast.
				step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

				weight_norm = p.data.norm(p=2).clamp_(0, 10)

				adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
				if group['weight_decay'] != 0:
					adam_step.add_(group['weight_decay'], p.data)

				adam_norm = adam_step.norm(p=2)

				if weight_norm == 0.0 or adam_norm == 0.0:
					trust_ratio = 1
				else:
					trust_ratio = weight_norm / (adam_norm + group['eps'])

				state['weight_norm'] = weight_norm
				state['adam_norm'] = adam_norm
				state['trust_ratio'] = trust_ratio
				if self.adam:
					trust_ratio = 1

				p.data.add_(-step_size * trust_ratio, adam_step)

		return loss

def InitEnvironment(model, weight_decay, local_rank):

	gpu = local_rank;
	torch.cuda.set_device(f'cuda:{gpu}');
	torch.distributed.init_process_group(backend='nccl', init_method='env://');

	assert torch.backends.cudnn.enabled, 'Requires cudnn backend to be enabled.';

	# Move model to GPU
	model		= model.cuda();
	embeddings	= model.input_embeddings;
	optimizer	= torch.optim.AdamW(model.parameters(), lr=3e-6, betas=(0.9, 0.999), weight_decay=weight_decay); # Used for everything

	# optimizer	= Lamb(model.parameters(), lr=3e-6, weight_decay=weight_decay);

	# Wrap with distributed parallelization
	dpp_model	= DDP(model, device_ids=[local_rank])

	return dpp_model, optimizer, embeddings;
