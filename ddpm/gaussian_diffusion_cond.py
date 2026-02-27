# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def sobel_edge_01(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4 and int(x.shape[1]) == 1
    x = x.clamp(-1.0, 1.0)
    x01 = (x + 1.0) * 0.5

    kx = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)

    gx = F.conv2d(x01, kx, padding=1)
    gy = F.conv2d(x01, ky, padding=1)
    mag = gx.abs() + gy.abs()

    denom = mag.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    mag = mag / denom
    return mag.clamp(0.0, 1.0)


def ssim_map_01(x01: torch.Tensor, y01: torch.Tensor, *, win: int = 7) -> torch.Tensor:
    assert x01.shape == y01.shape
    assert x01.ndim == 4 and int(x01.shape[1]) == 1
    x01 = x01.clamp(0.0, 1.0)
    y01 = y01.clamp(0.0, 1.0)
    w = int(win)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w = w + 1

    mu_x = F.avg_pool2d(x01, kernel_size=w, stride=1, padding=w // 2)
    mu_y = F.avg_pool2d(y01, kernel_size=w, stride=1, padding=w // 2)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.avg_pool2d(x01 * x01, kernel_size=w, stride=1, padding=w // 2) - mu_x2
    sigma_y2 = F.avg_pool2d(y01 * y01, kernel_size=w, stride=1, padding=w // 2) - mu_y2
    sigma_xy = F.avg_pool2d(x01 * y01, kernel_size=w, stride=1, padding=w // 2) - mu_xy

    sigma_x2 = sigma_x2.clamp(min=0.0)
    sigma_y2 = sigma_y2.clamp(min=0.0)

    c1 = 0.01**2
    c2 = 0.03**2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (num / den.clamp(min=1e-6)).clamp(-1.0, 1.0)


class GaussianDiffusionTrainerCond(nn.Module):
    def __init__(
        self,
        model,
        beta_1,
        beta_T,
        T,
        t_sampling: str = "uniform",
        t_num_bins: int = 100,
        *,
        aux_t_max: int = 300,
        lambda_edge: float = 0.01,
        lambda_cons: float = 0.005,
        lambda_x0: float = 0.003,
        lambda_heat_pred: float = 0.1,
        use_heat_ssim: bool = False,
        lambda_heat_ssim: float = 0.0,
        use_heat_edge_ssim: bool = False,
        lambda_heat_edge_ssim: float = 0.0,
        use_heat_mse: bool = False,
        lambda_heat_mse: float = 0.0,
        use_heat_edge_mse: bool = False,
        lambda_heat_edge_mse: float = 0.0,
        heat_ssim_win: int = 7,
        far_lambda: float = 0.0,
        far_anchor_thr: float = 0.0,
        far_heat_p: float = 1.0,
    ):
        super().__init__()

        self.model = model
        self.T = int(T)
        self.t_sampling = str(t_sampling)
        self.t_num_bins = int(t_num_bins)

        self.aux_t_max = int(aux_t_max)
        self.lambda_edge = float(lambda_edge)
        self.lambda_cons = float(lambda_cons)
        self.lambda_x0 = float(lambda_x0)
        self.lambda_heat_pred = float(lambda_heat_pred)
        self.use_heat_ssim = bool(use_heat_ssim)
        self.lambda_heat_ssim = float(lambda_heat_ssim)
        self.use_heat_edge_ssim = bool(use_heat_edge_ssim)
        self.lambda_heat_edge_ssim = float(lambda_heat_edge_ssim)
        self.use_heat_mse = bool(use_heat_mse)
        self.lambda_heat_mse = float(lambda_heat_mse)
        self.use_heat_edge_mse = bool(use_heat_edge_mse)
        self.lambda_heat_edge_mse = float(lambda_heat_edge_mse)
        self.heat_ssim_win = int(heat_ssim_win)

        self.far_lambda = float(far_lambda)
        self.far_anchor_thr = float(far_anchor_thr)
        self.far_heat_p = float(far_heat_p)

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    @staticmethod
    def _per_sample_mse_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).pow(2).view(pred.shape[0], -1).sum(dim=1)

    @staticmethod
    def _per_sample_l1_sum(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().view(pred.shape[0], -1).sum(dim=1)

    @staticmethod
    def _per_sample_charbonnier_sum(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return torch.sqrt(delta * delta + float(eps)).view(delta.shape[0], -1).sum(dim=1)

    @staticmethod
    def _far_region_weight(anchor01: torch.Tensor, heat01: torch.Tensor, *, anchor_thr: float, heat_p: float) -> torch.Tensor:
        # anchor01, heat01: [B,1,H,W] in [0,1]
        # far region means anchor is ~0 and heat is ~0
        far_anchor = (anchor01 <= float(anchor_thr)).to(dtype=anchor01.dtype)
        far_heat = (1.0 - heat01).clamp(0.0, 1.0)
        w = far_anchor * (far_heat ** float(heat_p))
        return w

    def forward(self, x_0, alpha=1.0):
        b = int(x_0.shape[0])
        if str(self.t_sampling).lower() in ("stratified", "stratified_uniform") and int(self.t_num_bins) > 0:
            bins = int(self.t_num_bins)
            bin_id = torch.randint(bins, size=(b,), device=x_0.device)
            u = torch.rand(size=(b,), device=x_0.device)
            t = (((bin_id.to(dtype=torch.float32) + u) / float(bins)) * float(self.T)).to(dtype=torch.long)
            t = torch.clamp(t, min=0, max=int(self.T) - 1)
        else:
            t = torch.randint(self.T, size=(b,), device=x_0.device)

        ct = x_0[:, 0:1, :, :]
        cond = x_0[:, 1:, :, :]
        gt_heat = (cond[:, 1:2, :, :] + 1.0) * 0.5

        noise = torch.randn_like(ct)
        x_t_ct = extract(self.sqrt_alphas_bar, t, ct.shape) * ct + extract(self.sqrt_one_minus_alphas_bar, t, ct.shape) * noise

        x_in = torch.cat((x_t_ct, cond), 1)
        out = self.model(x_in, t, gt_heatmap=gt_heat, alpha=alpha)
        if isinstance(out, (tuple, list)):
            if len(out) == 3:
                eps_pred, edge_logits, heat_logits = out
            else:
                eps_pred, edge_logits = out
                heat_logits = None
        else:
            eps_pred, edge_logits, heat_logits = out, None, None

        loss_noise = self._per_sample_mse_sum(eps_pred, noise)
        total = loss_noise

        if heat_logits is not None:
            pred_heat = torch.sigmoid(heat_logits)
            loss_heat_pred = (pred_heat - gt_heat).pow(2).view(b, -1).sum(dim=1)
            total = total + float(self.lambda_heat_pred) * loss_heat_pred

        if edge_logits is None:
            return total.sum()

        sqrt_ab = extract(self.sqrt_alphas_bar, t, ct.shape)
        sqrt_1m_ab = extract(self.sqrt_one_minus_alphas_bar, t, ct.shape)
        x0_hat = (x_t_ct - sqrt_1m_ab * eps_pred) / sqrt_ab
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        edge_pred = torch.sigmoid(edge_logits)
        gt_edge = sobel_edge_01(ct)
        pred_edge_from_img = sobel_edge_01(x0_hat)

        heat = cond[:, 1:2, :, :]
        w_heat = ((heat + 1.0) * 0.5).clamp(0.0, 1.0)

        loss_edge = self._per_sample_l1_sum(edge_pred, gt_edge)
        loss_cons = self._per_sample_l1_sum(edge_pred, pred_edge_from_img.detach())
        loss_x0 = self._per_sample_charbonnier_sum(x0_hat - ct)

        aux_mask = (t <= int(self.aux_t_max)).to(dtype=loss_edge.dtype)

        heat_active = aux_mask > 0
        heat_any = bool(torch.any(heat_active))

        loss_heat_ssim = loss_x0.new_zeros([b])
        loss_heat_edge_ssim = loss_x0.new_zeros([b])
        loss_heat_mse = loss_x0.new_zeros([b])
        loss_heat_edge_mse = loss_x0.new_zeros([b])

        heat_needed = (
            (bool(self.use_heat_ssim) and float(self.lambda_heat_ssim) != 0.0)
            or (bool(self.use_heat_edge_ssim) and float(self.lambda_heat_edge_ssim) != 0.0)
            or (bool(self.use_heat_mse) and float(self.lambda_heat_mse) != 0.0)
            or (bool(self.use_heat_edge_mse) and float(self.lambda_heat_edge_mse) != 0.0)
        )

        if bool(heat_needed) and bool(heat_any):
            idx = torch.nonzero(heat_active, as_tuple=False).squeeze(1)
            ct_s = ct.index_select(0, idx)
            x0_s = x0_hat.index_select(0, idx)
            w_s = w_heat.index_select(0, idx)

            w_sum = w_s.view(w_s.shape[0], -1).sum(dim=1).clamp(min=1e-6)

            if bool(self.use_heat_mse) and float(self.lambda_heat_mse) != 0.0:
                d2 = (x0_s - ct_s).pow(2)
                mse_w = (w_s * d2).view(w_s.shape[0], -1).sum(dim=1)
                loss_heat_mse.index_copy_(0, idx, mse_w)

            if bool(self.use_heat_edge_mse) and float(self.lambda_heat_edge_mse) != 0.0:
                e_pred_s = pred_edge_from_img.index_select(0, idx)
                e_gt_s = gt_edge.index_select(0, idx)
                d2 = (e_pred_s - e_gt_s).pow(2)
                mse_w = (w_s * d2).view(w_s.shape[0], -1).sum(dim=1)
                loss_heat_edge_mse.index_copy_(0, idx, mse_w)

            if bool(self.use_heat_ssim) and float(self.lambda_heat_ssim) != 0.0:
                x01 = ((x0_s + 1.0) * 0.5).clamp(0.0, 1.0)
                y01 = ((ct_s + 1.0) * 0.5).clamp(0.0, 1.0)
                ssim_map = ssim_map_01(x01, y01, win=int(self.heat_ssim_win))
                ssim_w = (w_s * ssim_map).view(w_s.shape[0], -1).sum(dim=1) / w_sum
                loss_heat_ssim.index_copy_(0, idx, (1.0 - ssim_w))

            if bool(self.use_heat_edge_ssim) and float(self.lambda_heat_edge_ssim) != 0.0:
                e_pred_s = pred_edge_from_img.index_select(0, idx)
                e_gt_s = gt_edge.index_select(0, idx)
                e_ssim_map = ssim_map_01(e_pred_s, e_gt_s, win=int(self.heat_ssim_win))
                ssim_w = (w_s * e_ssim_map).view(w_s.shape[0], -1).sum(dim=1) / w_sum
                loss_heat_edge_ssim.index_copy_(0, idx, (1.0 - ssim_w))

        total = total + aux_mask * (float(self.lambda_edge) * loss_edge)
        total = total + aux_mask * (float(self.lambda_cons) * loss_cons)
        total = total + aux_mask * (float(self.lambda_x0) * loss_x0)
        total = total + aux_mask * (float(self.lambda_heat_ssim) * loss_heat_ssim)
        total = total + aux_mask * (float(self.lambda_heat_edge_ssim) * loss_heat_edge_ssim)
        total = total + aux_mask * (float(self.lambda_heat_mse) * loss_heat_mse)
        total = total + aux_mask * (float(self.lambda_heat_edge_mse) * loss_heat_edge_mse)

        if float(self.far_lambda) != 0.0:
            anchor01 = ((cond[:, 0:1, :, :] + 1.0) * 0.5).clamp(0.0, 1.0)
            heat01 = ((cond[:, 1:2, :, :] + 1.0) * 0.5).clamp(0.0, 1.0)
            far_w = self._far_region_weight(anchor01, heat01, anchor_thr=float(self.far_anchor_thr), heat_p=float(self.far_heat_p))
            far_loss = (far_w * (x0_hat - ct).abs()).view(b, -1).sum(dim=1)
            total = total + float(self.far_lambda) * far_loss

        return total.sum()


class GaussianDiffusionSamplerCond(nn.Module):
    def __init__(
        self,
        model,
        beta_1,
        beta_T,
        T,
        *,
        struct_cond_last_pct: float = 0.0,
        struct_cond_source: str = "edge",
        struct_cond_carry: bool = False,
    ):
        super().__init__()

        self.model = model
        self.T = int(T)

        self.struct_cond_last_pct = float(struct_cond_last_pct)
        self.struct_cond_source = str(struct_cond_source)
        self.struct_cond_carry = bool(struct_cond_carry)

        self.register_buffer("betas", torch.linspace(beta_1, beta_T, int(T)).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[: int(T)]

        self.register_buffer("coeff1", torch.sqrt(1.0 / alphas))
        self.register_buffer("coeff2", self.coeff1 * (1.0 - alphas) / torch.sqrt(1.0 - alphas_bar))

        self.register_buffer("posterior_var", self.betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        ct = x_t[:, 0:1, :, :]
        assert ct.shape == eps.shape
        return extract(self.coeff1, t, ct.shape) * ct - extract(self.coeff2, t, ct.shape) * eps

    def p_mean_variance(self, x_t, t):
        ct = x_t[:, 0:1, :, :]
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, ct.shape)

        out = self.model(x_t, t)
        if isinstance(out, (tuple, list)):
            eps = out[0]
        else:
            eps = out

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        ct = x_t[:, 0:1, :, :]
        cond = x_t[:, 1:, :, :]

        struct_prev_m11 = None
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0]], dtype=torch.long) * int(time_step)

            use_struct = False
            if bool(self.struct_cond_carry):
                use_struct = struct_prev_m11 is not None
            elif float(self.struct_cond_last_pct) > 0.0:
                last_k = int(round(float(self.T) * float(self.struct_cond_last_pct)))
                last_k = max(0, min(int(self.T), last_k))
                if int(time_step) < int(last_k):
                    use_struct = True

            if use_struct:
                x_t_cond = torch.cat((ct, cond, struct_prev_m11.to(dtype=x_t.dtype)), 1)
                out_s = self.model(x_t_cond, t)
                if isinstance(out_s, (tuple, list)):
                    eps = out_s[0]
                    if len(out_s) >= 2:
                        edge_logits = out_s[1]
                    else:
                        edge_logits = None
                    if len(out_s) >= 3:
                        heat_logits = out_s[2]
                    else:
                        heat_logits = None
                else:
                    eps = out_s
                    edge_logits = None
                    heat_logits = None

                if str(self.struct_cond_source).lower() in ("heat", "heatmap"):
                    if heat_logits is None:
                        struct01 = None
                    else:
                        struct01 = torch.sigmoid(heat_logits)
                else:
                    if edge_logits is None:
                        struct01 = None
                    else:
                        struct01 = torch.sigmoid(edge_logits)

                if struct01 is not None:
                    struct_m11 = (struct01 * 2.0 - 1.0).to(dtype=x_t.dtype)
                    # NOTE: carry mode uses struct_prev_m11 to condition current eps;
                    # struct_m11 is saved for NEXT step.
                else:
                    struct_m11 = None

                mean = self.predict_xt_prev_mean_from_eps(x_t_cond, t, eps=eps)
                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var, t, ct.shape)
            else:
                out_base = self.model(x_t, t)
                if isinstance(out_base, (tuple, list)):
                    eps = out_base[0]
                else:
                    eps = out_base

                mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
                var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
                var = extract(var, t, ct.shape)

                # compute struct prediction for carry mode (for next step)
                if bool(self.struct_cond_carry):
                    if isinstance(out_base, (tuple, list)):
                        edge_logits = out_base[1] if len(out_base) >= 2 else None
                        heat_logits = out_base[2] if len(out_base) >= 3 else None
                    else:
                        edge_logits = None
                        heat_logits = None

                    if str(self.struct_cond_source).lower() in ("heat", "heatmap"):
                        struct01 = torch.sigmoid(heat_logits) if heat_logits is not None else None
                    else:
                        struct01 = torch.sigmoid(edge_logits) if edge_logits is not None else None
                    if struct01 is not None:
                        struct_prev_m11 = (struct01 * 2.0 - 1.0).detach()

            if int(time_step) > 0:
                noise = torch.randn_like(ct)
            else:
                noise = 0
            ct = mean + torch.sqrt(var) * noise

            if bool(self.struct_cond_carry):
                # update carry struct for next step using current model prediction
                if use_struct:
                    if struct_m11 is not None:
                        struct_prev_m11 = struct_m11.detach()
                if struct_prev_m11 is not None:
                    x_t = torch.cat((ct, cond, struct_prev_m11.to(dtype=x_t.dtype)), 1)
                else:
                    x_t = torch.cat((ct, cond), 1)
            else:
                if use_struct:
                    if "x_t_s" in locals() and int(x_t_s.shape[1]) != int(x_t.shape[1]):
                        x_t = torch.cat((ct, x_t_s[:, 1:, :, :]), 1)
                    else:
                        x_t = torch.cat((ct, cond), 1)
                else:
                    x_t = torch.cat((ct, cond), 1)
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1.0, 1.0)
