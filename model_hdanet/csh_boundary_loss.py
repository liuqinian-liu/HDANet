"""DCP-SFR Eq.12-inspired output edge consistency, adapted to two-logit CE."""
import torch
import torch.nn.functional as F

def soft_sobel(probability):
    kernel = probability.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]) / 8
    padded = F.pad(probability,(1,1,1,1),mode='replicate')
    gx = F.conv2d(padded,kernel[None,None])
    gy = F.conv2d(padded,kernel.t()[None,None])
    return (gx.square()+gy.square()+1e-6).sqrt()-0.001

def squared_edge_dice(predicted_edge, target_edge, valid, zeta=1e-6):
    axes=(1,2,3)
    intersection=(valid*predicted_edge*target_edge).sum(axes)
    denominator=(valid*(predicted_edge.square()+target_edge.square())).sum(axes)
    per_image=1-(2*intersection+zeta)/(denominator+zeta)
    active=(valid.sum(axes)>0).to(per_image.dtype)
    return (per_image*active).sum()/active.sum().clamp_min(1)

def csh_boundary_loss(logits, labels, ignore_label=255, zeta=1e-6):
    if logits.shape[1]!=2 or logits.shape[-2:]!=labels.shape[-2:]:
        raise ValueError('Expected two-channel logits already aligned to target resolution')
    # Float32 loss even if an optional caller uses mixed precision.
    probability=logits.float().softmax(1)[:,1:2]
    target=(labels==1).to(probability.dtype).unsqueeze(1)
    valid=(labels!=ignore_label).to(probability.dtype).unsqueeze(1)
    valid=-F.max_pool2d(-F.pad(valid,(1,1,1,1),mode='replicate'),3,1)
    return squared_edge_dice(soft_sobel(probability),soft_sobel(target),valid,zeta)
