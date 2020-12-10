import torch

def closest_ortho_regularizer(model, writer, global_count):
    old_device = next(model.parameters()).device
    model.to("cpu") #SVD is much faster on CPU
    total_diff = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "conv" in name:
                u,s,v = torch.svd(param)
                pparam = torch.matmul(u, v)
                param.data = pparam
                total_diff += abs(param-pparam).sum()

    writer.add_scalar('train/loss_ortho', float(total_diff), global_count)
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
    
    return ortho_loss
