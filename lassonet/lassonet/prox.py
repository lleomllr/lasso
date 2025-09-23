import torch 
import torch.nn as nn 
import torch.nn.functional as F


def soft_threshold(x, lamb):
    return torch.sign(x) * torch.clamp(torch.abs(x) - lamb, min=0.0)


def hier_prox(theta, W1, lamb, M):
    d, K = W1.shape

    theta_new = theta.clone()
    W1_new = W1.clone()

    for j in range(d):
        wj = W1[j]
        abs_wj = torch.abs(wj)

        sorted_w, _ = torch.sort(abs_wj, descending=True)
        csum = torch.cumsum(sorted_w, dim=0)

        abs_theta_j = torch.abs(theta[j])
        wm = torch.zeros(K+1, device=theta.device)

        wm[0] = (M / (1.0 + 0.0 * (M ** 2))) * soft_threshold(abs_theta_j, lamb)

        for m in range(1, K+1):
            base = abs_theta_j + M * csum[m-1]
            wm[m] = (M / (1.0 + m * (M ** 2))) * soft_threshold(base, lamb)

        w_right = torch.cat([sorted_w, torch.tensor([0.0], device=theta.device)])
        w_left = torch.cat([torch.tensor([float('inf')], device=theta.device), sorted_w])

        m_tilde = None
        for m in range(K+1):
            if w_right[m] <= wm[m] <= w_left[m]:
                m_tilde = m
                break
        if m_tilde is None:
            m_tilde = K

        wm_tilde = wm[m_tilde]

        theta_new[j] = (1 / M) * torch.sign(theta[j]) * wm_tilde
        W1_new[j] = torch.sign(wj) * torch.minimum(wm_tilde, abs_wj)
    
    return theta_new, W1_new


def group_hp(theta, W1, lamb, M):
    d, K = W1.shape

    theta_new = theta.clone()
    W1_new = W1.clone()

    if M is None or M <= 0:
        norm = torch.linalg.norm(theta, dim=1)
        scale = torch.clamp(1.0 - lamb / (norm + 1e-12), min=0.0)
        theta_new = scale * theta
        W1_new.zero_()
        return theta_new, W1_new


    for j in range(d):
        wj = W1[j]
        abs_wj = torch.abs(wj)

        sorted_wj, _ = torch.sort(abs_wj, descending=True)
        csum = torch.cumsum(sorted_wj, dim=0)

        theta_j = theta[j]
        norm = torch.linalg.norm(theta_j)

        wm = torch.zeros(K+1, device=theta.device)
        wm[0] = (M / (1.0 + 0.0 * (M ** 2))) * soft_threshold(norm, lamb)

        for m in range(1, K+1):
            base = norm + M * csum[m-1]
            wm[m] = (M / (1.0 + m * (M ** 2))) * soft_threshold(base, lamb)

        w_right = torch.cat([sorted_wj, torch.tensor([0.0], device=theta.device, dtype=theta.dtype)])
        w_left = torch.cat([torch.tensor([float('inf')], device=theta.device, dtype=theta.dtype), sorted_wj])
        
        m_tilde = None
        for m in range(K+1):
            if w_right[m] <= wm[m] <= w_left[m]:
                m_tilde = m
                break
        if m_tilde is None: 
            m_tilde = K

        wm_tilde = wm[m_tilde]

        if norm.item() > 0:
            theta_new[j] = (1 / M) * wm_tilde * (theta_j / norm)
        else: 
            theta_new[j].zero_()

        W1_new[j] = torch.sign(wj) * torch.minimum(wm_tilde, abs_wj)
    
    return theta_new, W1_new