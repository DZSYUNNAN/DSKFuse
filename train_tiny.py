import argparse
import torch
import os
from models import swinformer
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder
import time
import numpy as np
import torch.nn.functional as F
from input_data import ImageDataset_1
from torchvision.utils import save_image
from pytorch_ssim import SSIM,ssim,kd_loss
from pytorch_mssim import ms_ssim, MS_SSIM,L_Grad

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
torch.set_num_threads(6)

parser = argparse.ArgumentParser()
parser.add_argument("--infrared_dataroot", default="/media/dzs/新加卷/xuexi_xiangmu/dataset/MSRS/Infrared/train/", type=str)
parser.add_argument("--visible_dataroot", default="/media/dzs/新加卷/xuexi_xiangmu/dataset/MSRS/Visible/train/", type=str)
parser.add_argument("--label_dataroot", default="/media/dzs/新加卷/xuexi_xiangmu/dataset/MSRS/Label/train/", type=str)
parser.add_argument("--fusion_dataroot", default="/media/dzs/新加卷/xuexi_xiangmu/dataset/MSRS/Swin_Fusion/train/", type=str)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--image_size", type=int, default=[128, 128])
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/vif/")

def tv_loss(x, batch_size=1):
    batch_size = x.shape[0]
    c_x = x.shape[1]
    h_x = x.shape[2]
    w_x = x.shape[3]
    count_h = x[:, :, 1:, :].size(1) * x[:, :, 1:, :].size(2) * x[:, :, 1:, :].size(3)
    count_w = x[:, :, :, 1:].size(1) * x[:, :, :, 1:].size(2) * x[:, :, :, 1:].size(3)
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return h_tv / count_h + w_tv / count_w

if __name__ == "__main__":
    opt = parser.parse_args()
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    #device = torch.device('cuda:0')
    #writer = SummaryWriter(log_dir= 'iv')
    net = swinformer.SwinFusion(upscale=1, in_chans=1, img_size=opt.image_size, Ex_depths=[2], window_size=8,
                img_range=1., Fusion_depths=[2], Re_depths=[2], embed_dim=20, Ex_num_heads=[4], Fusion_num_heads=[4,4], Re_num_heads=[4,4],
                mlp_ratio=2, upsampler=None, resi_connection='1conv').cuda()
    # net = fusion_model.Fusion_fastkanconv().cuda()
    # net=RMT.RMT_T(opt).cuda()
    # net = BIFormer.biformer_tiny(pretrained=False, pretrained_cfg=None,
    #               pretrained_cfg_overlay=None).cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),lr=opt.lr)
    train_datasets = ImageDataset_1(opt.infrared_dataroot, opt.visible_dataroot,opt.label_dataroot, opt.fusion_dataroot, opt.image_size)
    lens = len(train_datasets)
    log_file = './log_dir'
    dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=opt.batch_size,num_workers = 4, shuffle=True, pin_memory=True)
    runloss = 0
    runlosses = []
    MSE = torch.nn.MSELoss()
    L1_Loss = torch.nn.L1Loss()
    L_Grad =L_Grad()
    total_params = sum(p.numel() for p in net.parameters())
    print('total parameters:', total_params)
    for epoch in range(opt.epoch):
        for index, data in enumerate(dataloader):
            infrared = data[0].cuda()
            infrared_noise=data[1].cuda()
            visible = data[2].cuda()
            visible_dwt = data[3].cuda()
            label = data[4].cuda()
            fusion = data[5].cuda()
            # save_image(infrared.cpu(), os.path.join("./outputs/VIFiv/",  "infrared.png"))
            # save_image(visible.cpu(), os.path.join("./outputs/VIFiv/", "visible.png"))
            # save_image(infrared_noise.cpu(), os.path.join("./outputs/VIFiv/",  "infrared_noise.png"))
            # save_image(visible_dwt.cpu(), os.path.join("./outputs/VIFiv/", "visible_dwt.png"))
            if (epoch+1) <=  50:
                fused_img,_ = net(infrared_noise,visible_dwt)
                ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=1)
                # LOSS_MUSSIM = 1 - ssim(fused_img, infrared, visible)
                LOSS_SSIM_KD = 1 - SSIM(fusion, fused_img)
                LOSS_MSE_KD= L1_Loss(fusion, fused_img)
                # LOSS_GRAD = L_Grad(infrared, visible, fused_img)
                # fusion_label = label*infrared + (1-label)*visible
                # LOSS_label = MSE(fusion_label, fused_img)
                loss = 2*LOSS_MSE_KD+5*LOSS_SSIM_KD   #+ 5*LOSS_MUSSIM#+ LOSS_MSE_KD+ 5*LOSS_GRAD+ 3*LOSS_label
                runloss += loss.item()
                runlosses.append(runloss / lens)
                print('epoch [{}/{}], images [{}/{}], LOSS_MSE_KD loss is {:.5}, LOSS_SSIM_KD loss is {:.5}, total loss is  {:.5}, lr: {}'.
                  format(epoch + 1, opt.epoch, (index + 1) * opt.batch_size, lens+1, LOSS_MSE_KD.item(),LOSS_SSIM_KD.item(), loss.item(), opt.lr))
                optim.zero_grad()
                loss.backward()
                optim.step()
                torch.save(net.state_dict(), opt.checkpoint_dir + 'vif_swin_stu_tiny_20250417.pth'.format(opt.lr, log_file[2:]))
            else:
                freezemodel = swinformer.SwinFusion(upscale=1, in_chans=1, img_size=opt.image_size, Ex_depths=[2], window_size=8,
                img_range=1., Fusion_depths=[2], Re_depths=[2], embed_dim=20, Ex_num_heads=[4], Fusion_num_heads=[4,4], Re_num_heads=[4,4],
                mlp_ratio=2, upsampler=None, resi_connection='1conv').cuda()
                save_pth = os.path.join(opt.checkpoint_dir, 'vif_swin_stu_tiny_20250417.pth')
                freezemodel.load_state_dict(torch.load(save_pth))
                freezemodel.cuda()
                freezemodel.eval()
                for p in freezemodel.parameters():
                    p.requires_grad = False
                fusion_freeze,conv_fuse_1 = freezemodel(infrared,visible)
                fused_unfreeze,conv_fuse_2 = net(infrared_noise, visible_dwt)
                LOSS_GRAD = L_Grad(infrared, visible, fused_unfreeze)
                # fusion_label = label*infrared + (1-label)*visible
                # LOSS_label = MSE(fusion_label, fused_unfreeze)
                LOSS_MSE = kd_loss(conv_fuse_1,conv_fuse_2,1)
                L1_LOSS = L1_Loss(fusion_freeze, fused_unfreeze)
                loss = 2*LOSS_GRAD + 5*LOSS_MSE + L1_LOSS
                runloss += loss.item()
                runlosses.append(runloss / lens)
                print('epoch [{}/{}], images [{}/{}], LOSS_GRAD loss is {:.5}, LOSS_MSE loss is {:.5},L1_LOSS loss is {:.5},total loss is  {:.5}, lr: {}'.
                  format(epoch + 1, opt.epoch, (index + 1) * opt.batch_size, lens+1, LOSS_GRAD.item(),LOSS_MSE.item(),L1_LOSS.item(), loss.item(), opt.lr))
                optim.zero_grad()
                loss.backward()
                optim.step()
                torch.save(net.state_dict(), opt.checkpoint_dir + 'vif_swin_self_tiny_20250417.pth'.format(opt.lr, log_file[2:]))