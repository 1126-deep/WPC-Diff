# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import datetime
import json
import math
import os
import time

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ddpm.gaussian_diffusion_cond import (
    GaussianDiffusionSamplerCond,
    GaussianDiffusionTrainerCond,
    sobel_edge_01,
    ssim_map_01,
)
from data.outpaint_txt_dataset import OutpaintTxtDataset
from model.unet_heat_struct_pre_heat_ssim import UNet


dataset_name = "brain"
out_name = "unet"
model_type = "unet"

batch_size = 2
grad_accum_steps = 4
T = 1000
ch = 128
ch_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.3

guide_gain = 0.5

struct_cond_last_pct = 0.0
struct_cond_source = "edge"
struct_cond_extra_channels = 1
struct_cond_carry = False

base_in_channels = 6
need_struct_ch = bool(struct_cond_carry) or (float(struct_cond_last_pct) > 0.0)
in_channels = int(base_in_channels + (struct_cond_extra_channels if need_struct_ch else 0))

net_model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout, guide_gain=float(guide_gain), in_channels=int(in_channels))

beta_1 = 1e-4
beta_T = 0.02
grad_clip = 1

save_weight_dir = "./outputs/unet_ad"
img_size = 192
ema_decay = 0.9999
eval_seed_base = 12345

resume_epoch = 0
ckpt_resume_path = ""

aux_t_max = 300
# Keep only noise + heat_pred + weighted x0
lambda_edge = 0.0
lambda_cons = 0.0
lambda_x0 = 0.1
lambda_heat_pred = 1.0

use_heat_ssim = False
lambda_heat_ssim = 0.0
use_heat_edge_ssim = False
lambda_heat_edge_ssim = 0.0
heat_ssim_win = 7

use_heat_mse = False
lambda_heat_mse = 0.0
use_heat_edge_mse = False
lambda_heat_edge_mse = 0.0

# Far-region weighting (anchor-outside & heat-outside)
far_lambda = 0.5
far_anchor_thr = 0.1
far_heat_p = 2.0

warmup_epochs = 200
n_epochs = 2000

ramp1_end = 600
ramp2_end = 1200

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)
device = accelerator.device

os.makedirs("%s" % save_weight_dir, exist_ok=True)
train_dataloader = DataLoader(
    OutpaintTxtDataset("./%s" % dataset_name, split="train", image_size=img_size),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_dataloader = DataLoader(
    OutpaintTxtDataset("./%s" % dataset_name, split="val", image_size=img_size),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

optimizer = torch.optim.AdamW(net_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

net_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(net_model, optimizer, train_dataloader, val_dataloader)

ema_model = UNet(T, ch, ch_mult, attn, num_res_blocks, dropout, guide_gain=float(guide_gain), in_channels=int(in_channels)).to(device)
ema_model.load_state_dict(accelerator.unwrap_model(net_model).state_dict(), strict=True)
ema_model.eval()
for p in ema_model.parameters():
    p.requires_grad_(False)

start_epoch = 0
if int(resume_epoch) > 0 and os.path.exists(ckpt_resume_path):
    ckpt = torch.load(ckpt_resume_path, map_location="cpu")
    accelerator.unwrap_model(net_model).load_state_dict(ckpt, strict=True)
    ema_model.load_state_dict(accelerator.unwrap_model(net_model).state_dict(), strict=True)
    start_epoch = int(resume_epoch)
    if accelerator.is_main_process:
        print(f"Resumed net_model from: {ckpt_resume_path} (start_epoch={start_epoch})")

trainer = GaussianDiffusionTrainerCond(
    net_model,
    beta_1,
    beta_T,
    T,
    t_sampling="stratified",
    t_num_bins=1000,
    aux_t_max=aux_t_max,
    lambda_edge=float(lambda_edge),
    lambda_cons=float(lambda_cons),
    lambda_x0=float(lambda_x0),
    lambda_heat_pred=float(lambda_heat_pred),
    use_heat_ssim=bool(use_heat_ssim),
    lambda_heat_ssim=float(lambda_heat_ssim),
    use_heat_edge_ssim=bool(use_heat_edge_ssim),
    lambda_heat_edge_ssim=float(lambda_heat_edge_ssim),
    use_heat_mse=bool(use_heat_mse),
    lambda_heat_mse=float(lambda_heat_mse),
    use_heat_edge_mse=bool(use_heat_edge_mse),
    lambda_heat_edge_mse=float(lambda_heat_edge_mse),
    heat_ssim_win=int(heat_ssim_win),
    far_lambda=float(far_lambda),
    far_anchor_thr=float(far_anchor_thr),
    far_heat_p=float(far_heat_p),
).to(device)


def _lambda_scale(epoch: int) -> float:
    e = int(epoch)
    if e <= int(warmup_epochs):
        return 0.0
    if e <= int(ramp1_end):
        return 0.5 * float(e - int(warmup_epochs)) / float(int(ramp1_end) - int(warmup_epochs))
    if e <= int(ramp2_end):
        return 0.5 + 0.5 * float(e - int(ramp1_end)) / float(int(ramp2_end) - int(ramp1_end))
    return 1.0


def get_alpha(epoch, total_epochs):
    if epoch <= 100:
        return 0.0
    elif epoch <= 400:
        return (epoch - 100) / 300.0
    else:
        return 1.0


def get_heat_drop_prob(epoch, total_epochs):
    if epoch <= 100:
        return 0.0
    if epoch <= 400:
        return (epoch - 100) / 300.0 * 0.5
    return 0.4


def _run_eval(split_name: str, *, epoch: int):
    if not accelerator.is_main_process:
        return

    net_model.eval()
    sampler = GaussianDiffusionSamplerCond(
        ema_model,
        beta_1,
        beta_T,
        T,
        struct_cond_last_pct=float(struct_cond_last_pct),
        struct_cond_source=str(struct_cond_source),
        struct_cond_carry=bool(struct_cond_carry),
    ).to(device)
    out_dir = os.path.join(save_weight_dir, f"{split_name}_epoch_{epoch}")
    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "epoch": int(epoch),
        "split": str(split_name),
        "eval_seed_base": int(eval_seed_base),
        "samples": [],
    }

    with torch.no_grad():
        for ii, batch in enumerate(val_dataloader):
            if ii >= 8:
                break
            gt = batch["a"].to(device)
            cond = batch["b"].to(device)

            try:
                g = torch.Generator(device=device)
            except Exception:
                g = torch.Generator()
            g.manual_seed(int(eval_seed_base) + int(ii))
            noisy = torch.randn(size=[1, 1, int(gt.shape[-2]), int(gt.shape[-1])], generator=g, device=device, dtype=gt.dtype)
            x_in = torch.cat((noisy, cond), 1)
            x_out = sampler(x_in)
            pred = x_out[:, 0:1, :, :]

            mse = torch.mean((pred - gt) ** 2).item()
            if mse > 0:
                psnr = 10.0 * math.log10((2.0 * 2.0) / float(mse))
            else:
                psnr = 100.0
            pred01 = ((pred + 1.0) * 0.5).clamp(0.0, 1.0)
            gt01 = ((gt + 1.0) * 0.5).clamp(0.0, 1.0)
            ssim = ssim_map_01(pred01, gt01, win=7).mean().item()
            metrics["samples"].append({"idx": int(ii), "mse": float(mse), "psnr": float(psnr), "ssim": float(ssim)})

            anchor = cond[:, 0:1, :, :]
            heatmap = cond[:, 1:2, :, :]

            t0 = pred.new_zeros([1], dtype=torch.long)
            out_eval = ema_model(torch.cat([pred, cond], dim=1), t0, alpha=1.0)
            if isinstance(out_eval, (tuple, list)) and len(out_eval) == 3:
                _, edge_logits, heat_logits = out_eval
            else:
                _, edge_logits = out_eval
                heat_logits = None

            edge_pred = torch.sigmoid(edge_logits) if edge_logits is not None else sobel_edge_01(pred)
            heat_pred = torch.sigmoid(heat_logits) if heat_logits is not None else heatmap

            pred_sobel = sobel_edge_01(pred)
            gt_sobel = sobel_edge_01(gt)

            vis = torch.cat([anchor, heatmap, heat_pred, pred, gt, edge_pred, pred_sobel, gt_sobel], dim=0)
            from torchvision.utils import save_image

            save_image((vis + 1) * 0.5, os.path.join(out_dir, f"sample_{ii:04d}.png"), nrow=8)

    if len(metrics["samples"]) > 0:
        metrics["mean_mse"] = float(sum(s["mse"] for s in metrics["samples"]) / len(metrics["samples"]))
        metrics["mean_psnr"] = float(sum(s["psnr"] for s in metrics["samples"]) / len(metrics["samples"]))
        metrics["mean_ssim"] = float(sum(s["ssim"] for s in metrics["samples"]) / len(metrics["samples"]))
    else:
        metrics["mean_mse"] = float("nan")
        metrics["mean_psnr"] = float("nan")
        metrics["mean_ssim"] = float("nan")

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    net_model.train()


if accelerator.is_main_process:
    print(f"Start training {model_type} on {dataset_name}, save_dir={save_weight_dir}")

_run_eval("val", epoch=start_epoch)

for epoch in range(int(start_epoch) + 1, int(n_epochs) + 1):
    alpha = get_alpha(epoch, n_epochs)
    trainer.heat_drop_prob = float(get_heat_drop_prob(epoch, n_epochs))

    s = _lambda_scale(epoch)
    trainer.lambda_heat_pred = float(lambda_heat_pred)

    # Keep only weighted x0 (plus noise/heat_pred). Others stay 0.
    trainer.lambda_edge = float(lambda_edge) * float(s)
    trainer.lambda_cons = float(lambda_cons) * float(s)
    trainer.lambda_x0 = float(lambda_x0) * float(s)

    net_model.train()

    t_start = time.time()
    for i, batch in enumerate(train_dataloader):
        with accelerator.accumulate(net_model):
            target = batch["a"].to(device)
            cond = batch["b"].to(device)
            x_0 = torch.cat((target, cond), 1)
            loss = trainer(x_0, alpha=alpha)
            accelerator.backward(loss)
            if grad_clip > 0:
                accelerator.clip_grad_norm_(net_model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                src = accelerator.unwrap_model(net_model)
                for p_ema, p in zip(ema_model.parameters(), src.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=(1.0 - ema_decay))

    if accelerator.is_main_process:
        dt = time.time() - t_start
        print(f"Epoch {epoch}/{n_epochs} done in {dt:.1f}s")

    if epoch % 100 == 0:
        if accelerator.is_main_process:
            ckpt_path = os.path.join(save_weight_dir, f"ckpt_{epoch}_.pt")
            torch.save(accelerator.unwrap_model(net_model).state_dict(), ckpt_path)
            print(f"Saved ckpt: {ckpt_path}")
        _run_eval("val", epoch=epoch)
