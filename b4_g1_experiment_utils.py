import hashlib
import json
import time
from pathlib import Path
import torch

def tensor_digest_update(digest, *tensors):
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())

def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith('edge_prior.') or '.frequency_filter.' in name or '.scale_gate.' in name or '.spatial_gate.' in name:
            continue
        digest.update(name.encode())
        tensor_digest_update(digest, value)
    return digest.hexdigest()

def save_epoch(args, model, epoch, train_loss, val_loss, miou, fbiou, best_epoch, best_iou, filename):
    if not all(torch.isfinite(torch.tensor(v)) for v in [train_loss, val_loss, miou, fbiou]):
        raise FloatingPointError('Non-finite epoch metric')
    row = dict(epoch=epoch, epoch_one_based=epoch+1, time=time.time(),
               train_loss=train_loss, val_loss=val_loss, class_miou=miou, fbiou=fbiou,
               train_inputs_sha256=model.train_data_digest, val_inputs_sha256=model.val_data_digest,
               losses=model.epoch_losses, best_epoch=best_epoch, best_class_miou=best_iou,
               checkpoint=filename)
    root = Path(args.save_path).parent
    with (root / 'epochs.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps(row) + '\n')
    atomic_json(root / 'latest.json', row)
    if epoch == best_epoch:
        atomic_json(root / 'best.json', row)


def atomic_json(path, row):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(row, indent=2), encoding='utf-8')
    tmp.replace(path)
