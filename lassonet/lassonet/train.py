import torch 
import torch.nn as nn 
import torch.nn.functional as F
from prox import hier_prox, group_hp


def train(model, X, y, lamb_init=1e-3, eps=0.03, lr=1e-3, epochs=100, max_steps=50, use_group=False):
    d = model.theta.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    lamb = lamb_init
    path = []

    for step in range(max_steps):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            out = model(X).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if use_group: 
                    theta_new, W1_new = group_hp(model.theta.data, model.fc1.weight.data.T, lamb, model.M)
                else:
                    theta_new, W1_new = hier_prox(model.theta.data, model.fc1.weight.data.T, lamb, model.M)

                model.theta.data.copy_(theta_new)
                model.fc1.weight.data.copy_(W1_new.T)

        sparsity = (model.theta.abs() < 1e-6).float().mean().item()
        path.append({
            "lambda" : lamb, 
            "theta" : model.theta.detach().clone(), 
            "sparsity" : sparsity, 
            "loss" : loss.item()
        })
        print(f"[step {step}] lambda={lamb:.4e} | Loss={loss.item():.4f} | Sparsity={sparsity*100:.1f}%")

        if model.theta.abs().max().item() < 1e-8:
            print("Toutes les features sont éteintes → arrêt.")
            break

        lamb = (1 + eps) * lamb
    return path