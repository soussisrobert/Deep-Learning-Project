"""Night fine-tuning of HiFT (copy of tools/train.py, single GPU, per-epoch checkpoints and validation).

Differences from tools/train.py (which is not changed):
  * starts from tools/snapshot/first.pth (a dict with epoch / state_dict / optimizer; only state_dict is used, the optimizer is new);
  * the backbone is frozen explicitly and then layers 3-5 are unfrozen (tools/train.py's resume path would also leave layers 1-2 trainable);
  * data from pysot/datasets/night_dataset.py (mix, night augmentation, real negatives), numpy RNG seeded per worker (the original
    gave every worker the same augmentation random numbers);
  * after every epoch: checkpoint {epoch, state_dict, optimizer, val}, and the loss on a fixed set of UAVDark135 val pairs, once
    without enhancement and once with gamma 0.6 on both crops (the best test-time setting); one json line per epoch;
  * resume per epoch only (--resume <checkpoint>), --time_budget_s stops after an epoch if the next one might not fit in the job;
  * --max_iters N: smoke test, prints loss, GPU memory, speed and the estimated epoch / run time, no checkpoint.

usage (from the repository root): python -I tools/train_night.py --cfg experiments/night_ft/config.yaml experiments/night_ft/run_B.yaml --run B
"""
import os
import sys
import json
import time
import argparse
import logging
import math
import random

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.datasets.night_dataset import NightTrkDataset
from pysot.utils.lr_scheduler import build_lr_scheduler

logger = logging.getLogger('global')
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', nargs='+', required=True)
parser.add_argument('--run', default='default')
parser.add_argument('--init', default='tools/snapshot/first.pth')
parser.add_argument('--resume', default='')
parser.add_argument('--epochs_this_job', type=int, default=0, help='0 = all remaining epochs')
parser.add_argument('--time_budget_s', type=float, default=0, help='stop after an epoch if the next one might exceed this wall time')
parser.add_argument('--max_iters', type=int, default=0, help='smoke test: stop after this many iterations')
parser.add_argument('--val_batches', type=int, default=0, help='0 = all val batches')
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--opts', nargs='*', default=[])
args = parser.parse_args()


def winit(worker_id):
    s = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


def set_modes(model):
    """frozen backbone layers (and their BatchNorm) stay in eval mode, layers 3-5 and the heads train"""
    model.train()
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for layer in cfg.BACKBONE.TRAIN_LAYERS:
        for m in getattr(model.backbone, layer).modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()


def build_model():
    model = ModelBuilder().train()
    ck = torch.load(args.init, map_location='cpu', weights_only=False)
    sd = {k.split('module.', 1)[-1] if k.startswith('module.') else k: v for k, v in ck['state_dict'].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    logger.info('init from {} (epoch {}): missing keys {}, unexpected keys {}'.format(args.init, ck.get('epoch'), len(miss), len(unexp)))
    assert len(miss) == 0 and len(unexp) == 0, (miss[:3], unexp[:3])
    for p in model.backbone.parameters():
        p.requires_grad = False
    for layer in cfg.BACKBONE.TRAIN_LAYERS:
        for p in getattr(model.backbone, layer).parameters():
            p.requires_grad = True
    return model


def build_opt(model):
    groups = [{'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR},
              {'params': model.grader.parameters(), 'lr': cfg.TRAIN.BASE_LR}]
    opt = torch.optim.SGD(groups, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    sched = build_lr_scheduler(opt, epochs=cfg.TRAIN.EPOCH)
    return opt, sched


@torch.no_grad()
def validate(model, gamma):
    ds = NightTrkDataset('val', n_epochs=1, start_epoch=0, seed=0, val_gamma=gamma)
    dl = DataLoader(ds, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True, worker_init_fn=winit)
    model.eval()
    tot = {'total_loss': 0.0, 'cls_loss': 0.0, 'loc_loss': 0.0}
    n = 0
    for i, data in enumerate(dl):
        if args.val_batches and i >= args.val_batches:
            break
        out = model(data)
        for k in tot:
            tot[k] += float(out[k].mean())
        n += 1
    set_modes(model)
    return {k: round(v / max(n, 1), 4) for k, v in tot.items()}, n * cfg.TRAIN.BATCH_SIZE


def main():
    cfg.merge_from_file(args.cfg[0])
    for f in args.cfg[1:]:
        cfg.merge_from_file(f)
    cfg.TRAIN.SNAPSHOT_DIR = 'experiments/night_ft/runs/{}/snapshot'.format(args.run)
    cfg.TRAIN.LOG_DIR = 'experiments/night_ft/runs/{}/logs'.format(args.run)
    if args.opts:
        cfg.merge_from_list(args.opts)
    os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(cfg.TRAIN.LOG_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)
    logger.info('cfg {}'.format(json.dumps(cfg, indent=1, default=str)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    bs = cfg.TRAIN.BATCH_SIZE
    ppe = cfg.DATASET.VIDEOS_PER_EPOCH
    assert cfg.TRAIN.NUM_GPU == 1 and ppe % bs == 0, 'NUM_GPU must be 1 and pairs per epoch a multiple of the batch size'
    ipe = ppe // bs
    model = build_model()
    opt, sched = build_opt(model)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['state_dict'])
        opt.load_state_dict(ck['optimizer'])
        start = int(ck['epoch'])
        logger.info('resumed from {} at epoch {}'.format(args.resume, start))
    n_ep = (args.epochs_this_job or cfg.TRAIN.EPOCH - start)
    n_ep = min(n_ep, cfg.TRAIN.EPOCH - start)
    if args.max_iters:
        n_ep = 1
    sched.step(start)
    set_modes(model)
    ds = NightTrkDataset('train', n_epochs=n_ep, start_epoch=start, seed=args.seed)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS, pin_memory=True, drop_last=True,
                    worker_init_fn=winit, persistent_workers=True, prefetch_factor=4)
    logger.info('iterations per epoch {}, epochs this job {} (epochs {}..{})'.format(ipe, n_ep, start, start + n_ep - 1))
    t_job = time.time()
    t_ep = time.time()
    t_prev = time.time()
    m = {'total_loss': 0.0, 'cls_loss': 0.0, 'loc_loss': 0.0}
    cnt, wait, step_t = 0, 0.0, 0.0
    first_losses, last_losses = [], []
    max_ep_time = 0.0
    epoch = start
    for idx, data in enumerate(dl):
        wait += time.time() - t_prev
        t0 = time.time()
        out = model(data)
        loss = out['total_loss'].mean()
        if math.isfinite(loss.item()) and loss.item() < 1e4:
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            opt.step()
        torch.cuda.synchronize()
        step_t += time.time() - t0
        for k in m:
            m[k] += float(out[k].mean())
        cnt += 1
        if args.max_iters:
            (first_losses if idx < 20 else last_losses).append(float(loss))
            if idx >= 20:
                last_losses = last_losses[-20:]
        if (idx + 1) % 20 == 0:
            logger.info('epoch {} it {}/{} lr {:.2e}/{:.2e} loss {:.4f} cls {:.4f} loc {:.4f}  data wait {:.2f}s step {:.2f}s per it'.format(
                epoch + 1, (idx % ipe) + 1, ipe, opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], float(loss), float(out['cls_loss'].mean()),
                float(out['loc_loss'].mean()), wait / cnt, step_t / cnt))
        t_prev = time.time()
        if args.max_iters and idx + 1 == 20:
            t_20 = time.time()
        if args.max_iters and idx + 1 >= args.max_iters:
            it_s = (idx + 1 - 20) / max(time.time() - t_20, 1e-6)
            break
        if (idx + 1) % ipe == 0:
            epoch += 1
            ep_time = time.time() - t_ep
            max_ep_time = max(max_ep_time, ep_time)
            val_none, nv = validate(model, None)
            val_g06, _ = validate(model, 0.6)
            row = {'epoch': epoch, 'train': {k: round(v / cnt, 4) for k, v in m.items()}, 'val_none': val_none, 'val_gamma0.6': val_g06,
                   'val_pairs': nv, 'epoch_time_s': round(ep_time, 1), 'lr_heads': opt.param_groups[1]['lr']}
            logger.info('EPOCH {} {}'.format(epoch, json.dumps(row)))
            with open(os.path.join(cfg.TRAIN.LOG_DIR, 'epochs.jsonl'), 'a') as f:
                f.write(json.dumps(row) + '\n')
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'val': row},
                       os.path.join(cfg.TRAIN.SNAPSHOT_DIR, 'checkpoint_e{}.pth'.format(epoch)))
            if epoch < cfg.TRAIN.EPOCH:
                sched.step(epoch)
            m = {k: 0.0 for k in m}
            cnt, wait, step_t = 0, 0.0, 0.0
            t_ep = time.time()
            t_prev = time.time()
            if args.time_budget_s and time.time() - t_job + 1.15 * max_ep_time > args.time_budget_s and epoch < cfg.TRAIN.EPOCH:
                logger.info('time budget: stopping after epoch {}; resume with --resume checkpoint_e{}.pth'.format(epoch, epoch))
                break
    if args.max_iters:
        vn, nv = validate(model, None) if args.val_batches else (None, 0)
        n = args.max_iters
        logger.info('SMOKE iterations {}'.format(n))
        logger.info('SMOKE loss first 20 its mean {:.4f}, last 20 its mean {:.4f}'.format(np.mean(first_losses), np.mean(last_losses)))
        logger.info('SMOKE speed {:.2f} it/s (after the first 20), {:.0f} pairs/s; data wait {:.2f}s, step {:.2f}s per it (last 20 reported)'.format(
            it_s, it_s * bs, wait / max(cnt, 1), step_t / max(cnt, 1)))
        logger.info('SMOKE gpu memory max allocated {:.2f} GB, max reserved {:.2f} GB, device {}'.format(
            torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9, torch.cuda.get_device_name(0)))
        logger.info('SMOKE estimate: epoch of {} pairs ({} its) = {:.1f} min; {} epochs = {:.2f} h (training only, without validation)'.format(
            ppe, ipe, ipe / it_s / 60, cfg.TRAIN.EPOCH, ipe * cfg.TRAIN.EPOCH / it_s / 3600))
        if vn is not None:
            logger.info('SMOKE val loss (first {} batches, no enhancement): {}'.format(args.val_batches, vn))


if __name__ == '__main__':
    main()
