"""Re-evaluate each selected checkpoint on the same source-seeded episodes."""
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
import argparse
import hashlib
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from util import config, transform, dataset_defect as dataset
from model_hdanet.HDANet import HDANet
from model_hdanet.HDANet_移除aspp import HDANet as HDANetNoASPP
from model_hdanet.edge_prior import inner_boundary
from b4_g1_experiment_utils import tensor_digest_update

@torch.no_grad()
def main():
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--disable-spatial-gate', action='store_true')
    cli = parser.parse_args()
    args = config.load_cfg_from_cfg_file(cli.config)
    root = Path(args.save_path).parent
    best = json.loads((root/'best.json').read_text())
    model = (HDANetNoASPP if getattr(args, 'remove_aspp', False) else HDANet)(args).cuda().eval()
    state = torch.load(best['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    if cli.disable_spatial_gate:
        for module_name in ('maf_l', 'maf_m', 'maf_h'):
            getattr(model, module_name).spatial_gate.forward = lambda feature, prior: feature
    mean, std = [255*v for v in [.485,.456,.406]], [255*v for v in [.229,.224,.225]]
    trans = transform.Compose([transform.Resize(h=args.val_size,w=args.val_size),
                               transform.ToTensor(), transform.Normalize(mean=mean,std=std)])
    ds = dataset.SemData(split=args.split,shot=args.shot,data_root=args.data_root,
                         data_list=args.val_list,transform=trans,mode='val',data_set=args.data_set)
    loader = torch.utils.data.DataLoader(ds,batch_size=1,shuffle=False,num_workers=args.workers,pin_memory=True)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    digest = hashlib.sha256()
    confusion = np.zeros((2,2),dtype=np.int64)
    ci, cu = np.zeros(4), np.zeros(4)
    counts = np.zeros(4)  # matched prediction, all prediction, matched truth, all truth
    records, times = [], []
    preview = root/('previews_gate_off' if cli.disable_spatial_gate else 'previews')
    preview.mkdir(exist_ok=True)
    for index,(x,y,sx,sy,subcls,original) in enumerate(loader):
        tensor_digest_update(digest,x,y,sx,sy,*subcls)
        x, sx, sy = x.cuda(), sx.cuda(), sy.cuda()
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(x,sx,sy)
        end.record()
        end.synchronize()
        if index >= 5:
            times.append(start.elapsed_time(end))
        target = original.cuda().long()
        side = max(target.shape[-2:])
        truth = torch.full((1,side,side),255,device='cuda',dtype=torch.long)
        truth[:,:target.shape[-2],:target.shape[-1]] = target
        logits = F.interpolate(logits,size=truth.shape[-2:],mode='bilinear',align_corners=True)
        pred = logits.argmax(1)
        valid = truth != 255
        bins = torch.bincount((2*truth[valid]+pred[valid]).flatten(),minlength=4).reshape(2,2).cpu().numpy()
        confusion += bins
        c = int(subcls[0][0])
        intersection = int(bins[1,1])
        union = int(bins[1,:].sum()+bins[:,1].sum()-bins[1,1])
        ci[c] += intersection
        cu[c] += union
        pe,pv = inner_boundary(pred.masked_fill(~valid,255))
        te,tv = inner_boundary(truth)
        vp = pv*tv
        pe,te = pe*vp,te*vp
        match_p = (pe*F.max_pool2d(te,5,1,2)*vp).sum().item()
        match_t = (te*F.max_pool2d(pe,5,1,2)*vp).sum().item()
        counts += np.array([match_p,pe.sum().item(),match_t,te.sum().item()])
        records.append({'index':index,'query':ds.img_gt_class_list[index][0],
                        'class_id':ds.val_class[c],'intersection':intersection,'union':union,
                        'foreground_iou':intersection/(union+1e-10), 'boundary_counts':[match_p,pe.sum().item(),match_t,te.sum().item()]})
        if index in [0,25,50,75,100,125,150,175]:
            rgb = cv2.imread(ds.img_gt_class_list[index][0])
            pm = pred[0,:target.shape[-2],:target.shape[-1]].cpu().numpy().astype(np.uint8)*255
            gt = (target[0].cpu().numpy()==1).astype(np.uint8)*255
            panel = np.concatenate([rgb,cv2.cvtColor(gt,cv2.COLOR_GRAY2BGR),cv2.cvtColor(pm,cv2.COLOR_GRAY2BGR)],axis=1)
            cv2.imwrite(str(preview/f'query_{index:03d}.png'),panel)
    class_iou = ci/(cu+1e-10)
    iu = np.diag(confusion)/(confusion.sum(0)+confusion.sum(1)-np.diag(confusion)+1e-10)
    precision,recall = counts[0]/max(counts[1],1),counts[2]/max(counts[3],1)
    result = {'experiment_id':args.experiment_id,'checkpoint':best['checkpoint'],'epoch':state['epoch'],
              'class_miou':float(class_iou.mean()),'fbiou':float(iu.mean()),
              'class_iou':dict(zip(map(str,ds.val_class),class_iou.tolist())),
              'boundary_f1_tolerance_2px':float(2*precision*recall/(precision+recall+1e-10)),
              'foreground_recall':float(confusion[1,1]/max(confusion[1,:].sum(),1)),
              'background_false_positive_rate':float(confusion[0,1]/max(confusion[0,:].sum(),1)),
              'val_inputs_sha256':digest.hexdigest(),'episodes':len(records),
              'mean_inference_ms_shared_gpu':float(np.mean(times)),
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
              'protocol':'best checkpoint selected on this same validation list; not independent test',
              'records':records}
    if 'val_inputs_sha256' in best:
        assert result['val_inputs_sha256']==best['val_inputs_sha256'], 'Validation episodes differ from checkpoint selection'
    result['historical_selection_miou'] = best['class_miou']
    result['selection_reeval_delta'] = result['class_miou']-best['class_miou']
    if 'val_inputs_sha256' in best and not cli.disable_spatial_gate:
        assert abs(result['selection_reeval_delta'])<1e-6, 'Checkpoint score mismatch'
    result['spatial_gate_disabled'] = bool(cli.disable_spatial_gate)
    output_name = 'evaluation_gate_off.json' if cli.disable_spatial_gate else 'evaluation.json'
    (root/output_name).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)

if __name__ == '__main__':
    main()
