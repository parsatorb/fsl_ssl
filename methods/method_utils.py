import torch

def closest_ortho_regularizer(model, writer, global_count, test=False):
	old_device = next(model.parameters()).device
	model.to("cpu") #SVD is much faster on CPU
	total_diff = 0
	with torch.no_grad():
		for name, param in model.named_parameters():
			if "conv" in name:
				u,s,v = torch.svd(param)
				pparam = torch.matmul(u, v)
				param.data = pparam
# 				print(param-pparam)
				total_diff += abs(param-pparam).sum()
	if not test:
		writer.add_scalar('train/loss_ortho', float(total_diff), global_count)
	else:
		print(total_diff)
	model.to(old_device)

def loss_ortho_regularizer(loss_type, loss_factor, model, writer, global_count):
	ortho_loss = 0

	if loss_type == "weights_normal":
		for name, wparam in model.named_parameters():
			if 'weight' in name and wparam.requires_grad and len(wparam.shape)==4:
				N, C, H, W = wparam.shape
				weight = wparam.view(N * C, H, W)
				weight_squared = torch.bmm(weight, weight.permute(0, 2, 1)) # (N * C) * W * H
				ones_mask = torch.ones(N * C, W, H, dtype=torch.float32).cuda() # (N * C) * W * H
				diag = torch.eye(H, dtype=torch.float32).cuda() # to be broadcast per channel
				ortho_loss += torch.abs(weight_squared*(ones_mask-diag)).sum()
					
		ortho_loss = ortho_loss*ortho_factor
		writer.add_scalar('train/loss_ortho', float(ortho_loss), global_count)
	elif loss_type == "weights":
		for name, wparam in model.named_parameters():
			if 'weight' in name and wparam.requires_grad and len(wparam.shape)==4:
				N, C, H, W = wparam.shape
				weight = wparam.view(N * C, H, W)
				weight_squared = torch.bmm(weight, weight.permute(0, 2, 1)) # (N * C) * W * H
				diag = torch.eye(H, dtype=torch.float32).cuda() # to be broadcast per channel
				ortho_loss += torch.abs(weight_squared - diag).sum()
				
		ortho_loss = ortho_loss*ortho_factor
		writer.add_scalar('train/loss_ortho', float(ortho_loss), global_count)
	elif loss_type == "srip":
		l2_reg = None
		for W in model.parameters():
			if W.ndimension() < 2:
					continue
			else:
				cols = W[0].numel()
				rows = W.shape[0]
				w1 = W.view(-1,cols)
				wt = torch.transpose(w1,0,1)
				if (rows > cols):
					m  = torch.matmul(wt,w1)
					ident = Variable(torch.eye(cols,cols),requires_grad=True)
				else:
					m = torch.matmul(w1,wt)
					ident = Variable(torch.eye(rows,rows), requires_grad=True)

				ident = ident.cuda()
				w_tmp = (m - ident)
				b_k = Variable(torch.rand(w_tmp.shape[1],1))
				b_k = b_k.cuda()

				v1 = torch.matmul(w_tmp, b_k)
				norm1 = torch.norm(v1,2)
				v2 = torch.div(v1,norm1)
				v3 = torch.matmul(w_tmp,v2)

				if l2_reg is None:
					l2_reg = (torch.norm(v3,2))**2
				else:
					l2_reg = l2_reg + (torch.norm(v3,2))**2
		
		ortho_loss = l2_reg*ortho_factor
	
	return ortho_loss
