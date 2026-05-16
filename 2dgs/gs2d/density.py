import torch


def prune_gaussians(mus, sigmas, thetas, opacities, rgbs, min_opacity=0.01, max_sigma=None):
    keep_mask = opacities >= min_opacity

    if max_sigma is not None:
        largest_sigma = sigmas.max(dim=1).values
        size_mask = largest_sigma <= max_sigma
        keep_mask = keep_mask & size_mask

    num_pruned = (~keep_mask).sum().item()
    if num_pruned > 0:
        print(f"Pruning: {num_pruned} Gaussians removed (kept {keep_mask.sum().item()})")

    if not torch.any(keep_mask):
        print("Warning: All Gaussians would be pruned! Keeping all instead.")
        return mus, sigmas, thetas, opacities, rgbs, torch.ones_like(opacities, dtype=torch.bool)

    return (
        mus[keep_mask],
        sigmas[keep_mask],
        thetas[keep_mask],
        opacities[keep_mask],
        rgbs[keep_mask],
        keep_mask,
    )


def remove_from_optimizer(opt, old_params, keep_mask):
    mus_old, sigmas_old, thetas_old, opacities_old, rgbs_old = old_params

    def _filter_param_and_state(opt, old_param, keep_mask):
        group = opt.param_groups[0]
        idx = None
        for i, p in enumerate(group["params"]):
            if p is old_param:
                idx = i
                break
        if idx is None:
            raise ValueError("Parameter not found in optimizer")

        filtered_data = old_param.data[keep_mask]
        new_param = torch.nn.Parameter(filtered_data)
        group["params"][idx] = new_param

        state = opt.state.pop(old_param, {})
        if state:
            step = state.get("step", 0)
            exp_avg = state.get("exp_avg", torch.zeros_like(old_param))
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(old_param))
            opt.state[new_param] = {
                "step": step,
                "exp_avg": exp_avg[keep_mask],
                "exp_avg_sq": exp_avg_sq[keep_mask],
            }

        return new_param

    mus_param = _filter_param_and_state(opt, mus_old, keep_mask)
    sigmas_param = _filter_param_and_state(opt, sigmas_old, keep_mask)
    thetas_param = _filter_param_and_state(opt, thetas_old, keep_mask)
    opacities_param = _filter_param_and_state(opt, opacities_old, keep_mask)
    rgbs_param = _filter_param_and_state(opt, rgbs_old, keep_mask)

    return mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param


def clone_gaussians_if_high_grad(mus, sigmas, thetas, opacities, rgbs, mus_grad, grad_threshold=0.1, max_sigma=None):
    if mus_grad is None:
        return mus, sigmas, thetas, opacities, rgbs

    grad_norms = torch.norm(mus_grad, dim=-1)
    clone_mask = grad_norms > grad_threshold

    if max_sigma is not None:
        largest_sigma = sigmas.max(dim=1).values
        large_mask = largest_sigma > max_sigma
        clone_mask = clone_mask & (~large_mask)

    if not torch.any(clone_mask):
        return mus, sigmas, thetas, opacities, rgbs

    clone_indices = torch.nonzero(clone_mask).flatten()
    num_clones = len(clone_indices)
    if num_clones > 0:
        print(f"Cloning: {num_clones} Gaussians -> {num_clones} new points added")

    added_mus = mus[clone_indices].clone()
    added_sigmas = sigmas[clone_indices].clone()
    added_thetas = thetas[clone_indices].clone()
    added_opacities = opacities[clone_indices].clone()
    added_rgbs = rgbs[clone_indices].clone()

    return (
        torch.cat([mus, added_mus], dim=0),
        torch.cat([sigmas, added_sigmas], dim=0),
        torch.cat([thetas, added_thetas], dim=0),
        torch.cat([opacities, added_opacities], dim=0),
        torch.cat([rgbs, added_rgbs], dim=0),
    )


def split_gaussians_if_large(mus, sigmas, thetas, opacities, rgbs, max_sigma=0.35):
    largest = sigmas.max(dim=1).values
    large_mask = largest > max_sigma
    if not torch.any(large_mask):
        return mus, sigmas, thetas, opacities, rgbs

    new_mus = mus.clone()
    new_sigmas = sigmas.clone()
    new_thetas = thetas.clone()
    new_opacities = opacities.clone()
    new_rgbs = rgbs.clone()

    added_mus, added_sigmas, added_thetas, added_opacities, added_rgbs = [], [], [], [], []

    split_indices = torch.nonzero(large_mask).flatten()
    num_splits = len(split_indices)
    if num_splits > 0:
        print(f"Splitting: {num_splits} Gaussians -> {num_splits} new points added")

    for i in split_indices:
        cos_t, sin_t = torch.cos(thetas[i]), torch.sin(thetas[i])
        R = torch.stack([torch.stack([cos_t, -sin_t]), torch.stack([sin_t, cos_t])])
        major_first = sigmas[i, 0] >= sigmas[i, 1]
        major_dir = R[:, 0] if major_first else R[:, 1]

        offset = major_dir * (largest[i] * 0.35)
        child_sigma = torch.clamp(sigmas[i] * 0.7, min=1e-3)
        child_opacity = opacities[i] * 0.5

        new_mus[i] = mus[i] + offset
        new_sigmas[i] = child_sigma
        new_opacities[i] = child_opacity

        added_mus.append(mus[i] - offset)
        added_sigmas.append(child_sigma)
        added_thetas.append(thetas[i])
        added_opacities.append(child_opacity)
        added_rgbs.append(rgbs[i])

    return (
        torch.cat([new_mus, torch.stack(added_mus)], dim=0),
        torch.cat([new_sigmas, torch.stack(added_sigmas)], dim=0),
        torch.cat([new_thetas, torch.stack(added_thetas)], dim=0),
        torch.cat([new_opacities, torch.stack(added_opacities)], dim=0),
        torch.cat([new_rgbs, torch.stack(added_rgbs)], dim=0),
    )


def _cat_param_and_state(opt, old_param, new_rows):
    if new_rows.numel() == 0:
        return old_param
    group = opt.param_groups[0]
    idx = None
    for i, p in enumerate(group["params"]):
        if p is old_param:
            idx = i
            break
    if idx is None:
        raise ValueError("Parameter not found in optimizer")
    device, dtype = old_param.device, old_param.dtype
    with torch.no_grad():
        combined = torch.cat([old_param.detach(), new_rows.to(device=device, dtype=dtype)], dim=0)
    new_param = torch.nn.Parameter(combined)
    group["params"][idx] = new_param
    state = opt.state.pop(old_param, {})
    step = state.get("step", 0)
    exp_avg = state.get("exp_avg", torch.zeros_like(old_param))
    exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(old_param))
    opt.state[new_param] = {
        "step": step,
        "exp_avg": torch.cat([exp_avg, torch.zeros_like(new_rows)], dim=0),
        "exp_avg_sq": torch.cat([exp_avg_sq, torch.zeros_like(new_rows)], dim=0),
    }
    return new_param


def extend_optimizer_with_new_points(opt, old_params, full_params, prev_count):
    mus_old, sigmas_old, thetas_old, opacities_old, rgbs_old = old_params
    mus_full, sigmas_full, thetas_full, opacities_full, rgbs_full = full_params

    mus_old.data[:] = mus_full[:prev_count]
    sigmas_old.data[:] = sigmas_full[:prev_count]
    thetas_old.data[:] = thetas_full[:prev_count]
    opacities_old.data[:] = opacities_full[:prev_count]
    rgbs_old.data[:] = rgbs_full[:prev_count]

    mus_new_rows = mus_full[prev_count:]
    sigmas_new_rows = sigmas_full[prev_count:]
    thetas_new_rows = thetas_full[prev_count:]
    opacities_new_rows = opacities_full[prev_count:]
    rgbs_new_rows = rgbs_full[prev_count:]

    mus_param = _cat_param_and_state(opt, mus_old, mus_new_rows)
    sigmas_param = _cat_param_and_state(opt, sigmas_old, sigmas_new_rows)
    thetas_param = _cat_param_and_state(opt, thetas_old, thetas_new_rows)
    opacities_param = _cat_param_and_state(opt, opacities_old, opacities_new_rows)
    rgbs_param = _cat_param_and_state(opt, rgbs_old, rgbs_new_rows)
    return mus_param, sigmas_param, thetas_param, opacities_param, rgbs_param
