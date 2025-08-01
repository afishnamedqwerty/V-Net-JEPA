# VICRegL refinement (two views, E + VICRegL)
# trainers/ssl_refine.py
import torch
from losses.vicregl import VICRegLLoss
from losses.energy import EnergyLoss
from models.hnet import HNetViT
from data.ssv2 import SSv2Dataset
from utils.augmentations import TwoViewAug

def ssl_refine(cfg):
    device = torch.device('cuda')
    model = HNetViT().to(device)  # From pretrain
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    dataset = SSv2Dataset(mode='train')
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    energy = EnergyLoss()
    vicregl = VICRegLLoss(lambda_var=cfg.lambda_var, lambda_inv=cfg.lambda_inv, lambda_cov=cfg.lambda_cov)

    aug = TwoViewAug()  # Different masks/augs

    for epoch in range(cfg.epochs):
        for batch in loader:
            x = batch['video'].to(device)
            view1, view2 = aug(x), aug(x)

            z1 = model(view1)
            z2 = model(view2)

            masked1 = mask_chunks(z1)
            pred1 = model.predictor(masked1)
            target1 = model.target_encoder(z1)

            loss_e = energy(pred1, target1)
            loss_v = vicregl(z1, z2)  # On chunk embeds

            loss = loss_e + cfg.lambda_vic * loss_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
