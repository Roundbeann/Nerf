import os
import cv2
import json
import torch
import imageio
import torch.optim as optim
import numpy as np

import torch.optim as optimizer
import torch.nn as nn
from utils import DatasetProvider, Embedder, ViewDependentHead, NoViewDirHead, NeRF, NeRFDataset
from utils import render_rays

from tqdm import tqdm

def train():
    pbar = tqdm(range(1,maxiters))
    # 对每轮迭代展示进度条
    for global_step in tqdm(range(1,maxiters)):
        # 随机选训练数据集中的一张图片
        idx = np.random.randint(0,len(trainset))
        # 从数据集中取出这张图片随机1024个像素点对应的光线方向、原点和像素值
        ray_dirs,ray_oris,img_pixels=trainset[idx]

        # rgb1, rgb2分别是粗网络和细网络渲染得到的RGB值
        # rgb1【1024，3】
        # rgb2【1024，3】
        rgb1,rgb2 = render_rays(coarseModel,fineModel,ray_dirs,ray_oris,sample_z_vals,num_samples2,white_background)

        # img_pixels【1024，3】
        # 均方误差
        loss1 = ((rgb1 - img_pixels)**2).mean()
        loss2 = ((rgb2 - img_pixels)**2).mean()

        psnr = -10. * torch.log(loss2.detach()) / np.log(10.)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        # 优化器更新模型参数
        optimizer.step()
        # 对进度条的说明
        pbar.set_description(
            f"{global_step} / {maxiters}, Loss: {loss.item()}, PSNR:{psnr.item():.6f}"
        )

        decay_rate = 0.1
        # 随着训练的次数增大,学习率要减小
        # 学习率从 1 * learning_rate 变成 0.1 * learning_rate
        new_lrate = learning_rate * (decay_rate ** (global_step / learning_rate_decay))

        for param_group in optimizer.param_groups:
            # 更新学习率
            param_group['lr'] = new_lrate

        # 每训练5000轮保存一次图像和模型参数
        if global_step % 5000 == 0 or global_step == 500:
            imgpath = f"imgs/{global_step:02f}.png"
            pthpath = f"ckpt/{global_step:02d}.pth"

            # 将模型设为评估状态.也就是关闭训练时用到的一些技巧(dropout batchNormal等)
            coarseModel.eval()
            # 避免计算图的构建和方向传播
            with torch.no_grad():
                rgbs,imgpixels = [], []
                # 每训练5000轮之后,进入到测试过程
                for ray_dirs,ray_oris,imagepixels in trainset.get_test_item():
                    # 获取测试图片的光线 原点 以及 像素值
                    # 然后仅仅根据一个像素点的观察方向 对应的相机原点 以及采样点 使用render_rays方法就能够把这个点的预测RGB值渲染出来
                    # 对测试图片的推理流程
                    rgb1, rgb2 = render_rays(
                        coarseModel,fineModel,ray_dirs,ray_oris,sample_z_vals,num_samples2,white_background
                    )
                    # 如此往复,就能够把一张测试图片的640000个像素值的真值和预测值按每批次1024个全部存储下来
                    # 真值存储在 imgpixels 里面
                    #
                    rgbs.append(rgb2)
                    imgpixels.append(imagepixels)
                rgb = torch.cat(rgbs,dim = 0)
                imgpixels = torch.cat(imgpixels, dim = 0)
                # 计算对测试图像的均方误差
                loss = ((rgb - imgpixels)**2).mean()
                # 计算对测试图像的PSNR的值
                psnr = -10. * torch.log(loss) / np.log(10)
                # 输出测试图像的存储路径,输出测试图像的Loss和PSNR值
                print(f"Save image{imgpath},Loss:{loss.item()},PSNR:{psnr.item():.6f}")
            # 关闭评估模式,恢复训练模式
            coarseModel.train()
            # 将图片的像素值从【0，1】映射回【0，255】
            temp_image = (rgb.view(provider.height,provider.width,3).cpu().numpy() *255).astype(np.float32)
            # 把颜色通道转换成BGR的形式
            cv2.imwrite(imgpath,temp_image[...,::-1])
            torch.save([coarseModel.state_dict(),fineModel.state_dict()],pthpath)


def make_video360():
    # map_location='cpu'将模型加载到cpu上
    cstate,fstate = torch.load(ckpt,map_location='cpu')
    coarseModel.load_state_dict(cstate)
    fineModel.load_state_dict(fstate)
    coarseModel.eval()
    fineModel.eval()

    imageList = []
    # 遍历全景的图片，渲染并保存每个视角的照片
    for i, gfn in tqdm(enumerate(trainset.get_rotate_360_rays()),desc="Rendering"):
        # 设置不需要梯度更新
        with torch.no_grad():
            rgbs = []
            for ray_dirs,ray_oris in gfn():
                rgb1,rgb2 = render_rays(
                    coarseModel, fineModel, ray_dirs,ray_oris,sample_z_vals,num_samples2,white_background
                )
                rgbs.append(rgb2)
            rgb = torch.cat(rgbs,dim=0)
        rgb = (rgb.view(provider.height, provider.width, 3 ).cpu().numpy() *255).astype(np.uint8)
        file = f"rotate360/{i:03d}.png"
        print(f"Rendering to {file}")
        cv2.iwrite(file,rgb[...,::-1])
        imageList.append(rgb)

    video_file = f"videos/rotate360"
    print(f"Write imagelist to video file {video_file}")
    imageio.mimwrite(video_file,imageList,fps = 30,quality = 10 )

if __name__ == '__main__':
    root = '/data2/yuanshou/tmp/nerf-main-handai/data/nerf_synthetic/lego'
    transforms_file = 'transforms_train.json'
    half_resolution = False
    batch_size = 1024
    device = 'cuda'
    # provider 读取了100张训练图片的像素值和相机位姿
    provider = DatasetProvider(root, transforms_file, half_resolution)

    # trainset 根据100张训练图片的像素值和相机位姿构造每张训练图片每个像素对应的三维世界的光线
    trainset = NeRFDataset(provider, batch_size, device)

    # 对 x 的位置编码的维度：10
    # 对视角 view 的位置编码的维度：4
    x_PEdim = 10
    view_PEdim = 4

    # 之前的代码
    # 1.读取了全部的训练图片（100张）的像素值
    # 2.对每张训练图片的每个像素值构造世界坐标系下的光线

    # 定义 coarse、fine 两个模型
    # coarse 和 fine是两个一模一样的模型
    coarseModel = NeRF(x_PEdim=x_PEdim, view_PEdim=view_PEdim).to(device)
    fineModel = NeRF(x_PEdim=x_PEdim, view_PEdim=view_PEdim).to(device)

    # 把coarse和fine两个模型的参数保存一下
    # params保存了以下参数：
    # 1.所有线性层的权重参数 θ+bias
    # 2.输出头的权重参数 θ+bias
    params = list(coarseModel.parameters())
    params.extend(list(fineModel.parameters()))

    # 定义学习率
    learning_rate_decay = 500 * 1000
    learning_rate = 5e-4

    # 导入优化器
    optimizer = optim.Adam(params, learning_rate)

    # trainset = NeRFDataset保存了100张训练图片每个像素点对应的像素值、对应光线的相机光心以及光线的投射方向
    # 可以通过trainset[0]取第一张训练图片的所有像素的像素值、对应光线的相机光心以及光线的投射方向
    #

    # 取第一张图片当中的1024个像素值，拿到这些像素对应的：
    # trainset[0] 这里主要是调用了 NeRFDataset() 类的魔法方法 __getitem__
    # 1.img_pixels  [1024, 3] 像素值
    # 2.ray_dirs    [1024, 3] 像素对应的三维世界的光线
    # 3.ray_oris    [1024, 3] 像素对应的三维世界的相机原点
    ray_dirs, ray_oris, img_pixels = trainset[0]

    # 上面这行代码就是对数据迭代器进行测试

    # 一条光线上采样点的个数是64
    num_samples1 = 64
    num_samples2 = 128

    # 配置项: white_background
    white_background = True

    # 对一条光线的near(2) to far(6)，均匀地取光线上的64个点
    # sample_z_vals [1, 64]
    # sample_z_vals 是一条光线上的64个均匀采样点

    # 构造从近平面（2）到远平面(6) 的 64个光线采样点
    sample_z_vals = torch.linspace(
        2.0, 6.0, num_samples1, device=device
    ).view(1, num_samples1)


    # 光线渲染器利用coarse模型、fine模型、像素点的光线方向、像素点的相机光心、粗采样的64个采样点、精细采样采样点的个数、是否是白背景
    # 光线渲染器利用这些信息渲染得到每个点的像素值
    # rgb1, rgb2 = render_rays(coarseModel, fineModel, ray_dirs, ray_oris, sample_z_vals, num_samples2, white_background)


    maxiters = 100000 + 1

    train()

    make_video360()
    ckpt = '300000.pth'

