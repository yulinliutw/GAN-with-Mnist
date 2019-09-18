import torch

def Discriminator_loss(prob_realimg, prob_gan):
    loss = - torch.mean(torch.log(prob_realimg+1e-8)) - torch.mean(torch.log((1. - prob_gan)+1e-8))
    return loss

def Generator_loss(prob_gan):
    loss = - torch.mean((torch.log(prob_gan+1e-8)))
    return loss