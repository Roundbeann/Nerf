import os
import cv2
import json
import torch
import imageio

import numpy as np

import torch.optim as optim
import torch.nn as nn
from utils import DatasetProvider,Embedder,ViewDependentHead,NoViewDirHead,NeRF,NeRFDataset

# 传入：
# 一张图片中1024个采样点对应的光线方向、相机原点和光线上的64个z值(sample_z_vals)
# 返回
def sample_rays(ray_directions, ray_origins, sample_z_vals):
    # ray_directions 【1024,3】
    # ray_origins    【1024,3】
    # sample_z_vals  【1,  64】 => 【1024，64】

    # 一共传入1024条光线
    nrays = len(ray_origins)

    # 每条光线上的64个采样点也复制1024份
    # 传入的 sample_z_vals 原本是 [1,64] 把它复制成 [1024,64] 份
    sample_z_vals = sample_z_vals.repeat(nrays,1)

    # r = o + td 构造射线
    # r : rays => result
    # o : ray_origins => 相机原点     [1024 * 3]  ==> [1024,1,3]  一张图片中1024个采样像素的相机原点
    # t : sample_z_vals => 64个采样点 [1024 * 64] ==> [1024,64,1] 一张图片中1024个采样像素对应光线上的64个采样点
    # d : ray_directions => 光线方向  [1024 * 3]  ==> [1024,1,3]  一张图片中1024个采样像素的光线方向

    # t * d: [1024,64,1] * [1024,1,3] = [1024,64,3]
    # o:     [1024,1, 3]
    # o + td:[1024,64,3]

    # t:sample_z_vals [64*1]
    # tensor([[2.0000], [2.0635], [2.1270], [2.1905], [2.2540], [2.3175], [2.3810], [2.4444],
    #         [2.5079], [2.5714], [2.6349], [2.6984], [2.7619], [2.8254], [2.8889], [2.9524],
    #         [3.0159], [3.0794], [3.1429], [3.2063], [3.2698], [3.3333], [3.3968], [3.4603],
    #         [3.5238], [3.5873], [3.6508], [3.7143], [3.7778], [3.8413], [3.9048], [3.9683],
    #         [4.0317], [4.0952], [4.1587], [4.2222], [4.2857], [4.3492], [4.4127], [4.4762],
    #         [4.5397], [4.6032], [4.6667], [4.7302], [4.7937], [4.8571], [4.9206], [4.9841],
    #         [5.0476], [5.1111], [5.1746], [5.2381], [5.3016], [5.3651], [5.4286], [5.4921],
    #         [5.5556], [5.6190], [5.6825], [5.7460], [5.8095], [5.8730], [5.9365], [6.0000]], device='cuda:0')
    # d:ray_directions [1*3]
    # tensor([[ 0.1025, -0.9554, -0.2911]], device='cuda:0')

    # t 和 d 相乘意味着什么?
    # d 是一个单位观察方向， d = [ 0.1025, -0.9554, -0.2911]
    # t * d 表示这个对单位观察方向在光线区间[2,6]上的放大，最终获得64条这样的观察方向向量，也就是[64,3]

    # o + t*d 意味着什么？
    # 意味着观察视角向量t*d 找到了起点，从而就有这个观察视角的终点 形成了一条位置固定的线段 仍然是 [64,3]

    # tensor([[ 0.2049, -1.9108, -0.5822],[ 0.2115, -1.9714, -0.6007],[ 0.2180, -2.0321, -0.6192],
    #         [ 0.2245, -2.0928, -0.6377],[ 0.2310, -2.1534, -0.6561],[ 0.2375, -2.2141, -0.6746],
    #         [ 0.2440, -2.2747, -0.6931],[ 0.2505, -2.3354, -0.7116],[ 0.2570, -2.3961, -0.7301],
    #         [ 0.2635, -2.4567, -0.7485],[ 0.2700, -2.5174, -0.7670],[ 0.2765, -2.5780, -0.7855],
    #         [ 0.2830, -2.6387, -0.8040],[ 0.2895, -2.6994, -0.8225],[ 0.2960, -2.7600, -0.8410],
    #         [ 0.3025, -2.8207, -0.8594],[ 0.3091, -2.8813, -0.8779],[ 0.3156, -2.9420, -0.8964],
    #         [ 0.3221, -3.0027, -0.9149],[ 0.3286, -3.0633, -0.9334],[ 0.3351, -3.1240, -0.9519],
    #         [ 0.3416, -3.1846, -0.9703],[ 0.3481, -3.2453, -0.9888],[ 0.3546, -3.3060, -1.0073],
    #         [ 0.3611, -3.3666, -1.0258],[ 0.3676, -3.4273, -1.0443],[ 0.3741, -3.4879, -1.0628],
    #         [ 0.3806, -3.5486, -1.0812],[ 0.3871, -3.6093, -1.0997],[ 0.3936, -3.6699, -1.1182],
    #         [ 0.4001, -3.7306, -1.1367],[ 0.4066, -3.7912, -1.1552],[ 0.4132, -3.8519, -1.1736],
    #         [ 0.4197, -3.9126, -1.1921],[ 0.4262, -3.9732, -1.2106],[ 0.4327, -4.0339, -1.2291],
    #         [ 0.4392, -4.0945, -1.2476],[ 0.4457, -4.1552, -1.2661],[ 0.4522, -4.2159, -1.2845],
    #         [ 0.4587, -4.2765, -1.3030],[ 0.4652, -4.3372, -1.3215],[ 0.4717, -4.3978, -1.3400],
    #         [ 0.4782, -4.4585, -1.3585],[ 0.4847, -4.5192, -1.3770],[ 0.4912, -4.5798, -1.3954],
    #         [ 0.4977, -4.6405, -1.4139],[ 0.5042, -4.7011, -1.4324],[ 0.5107, -4.7618, -1.4509],
    #         [ 0.5173, -4.8225, -1.4694],[ 0.5238, -4.8831, -1.4879],[ 0.5303, -4.9438, -1.5063],
    #         [ 0.5368, -5.0044, -1.5248],[ 0.5433, -5.0651, -1.5433],[ 0.5498, -5.1258, -1.5618],
    #         [ 0.5563, -5.1864, -1.5803],[ 0.5628, -5.2471, -1.5988],[ 0.5693, -5.3077, -1.6172],
    #         [ 0.5758, -5.3684, -1.6357],[ 0.5823, -5.4291, -1.6542],[ 0.5888, -5.4897, -1.6727],
    #         [ 0.5953, -5.5504, -1.6912],[ 0.6018, -5.6110, -1.7096],[ 0.6083, -5.6717, -1.7281],
    #         [ 0.6148, -5.7324, -1.7466]], device='cuda:0')

    # rays [1024, 64, 3]

    # ray_directions 【1024,3】
    # ray_origins    【1024,3】
    # sample_z_vals  【1,  64】 => 【1024，64】

    # ray_origins    【1024,1,3】
    # sample_z_vals  【1024,64,1】
    # ray_directions 【1024,1,3】

    # rays 【1024,64,3】
    rays = ray_origins[:,None,:] + sample_z_vals[...,None] * ray_directions[:,None,:]

    return rays,sample_z_vals


def sample_viewdirs(ray_directions):
    return ray_directions/torch.norm(ray_directions,dim=-1,keepdim=True)


# 接收的输入：
# 1.粗网络输出的sigma 和 rgb
# 1.sigma【1024 64 1】, rgb【1024 64 3】 是粗网络预测输出的,它们作为精细网络的输入,预测得到最终的颜色值
# 2.z_vals【1024, 64】是 1024 份 从【near:2】到【far:6】的64个光线深度(z_value)
# 3.ray_dirs【1024,3】是未进行归一化的光线方向
# 4.white_background 是背景配置项
def predict_to_rgb(sigma,rgb,z_vals,raydirs,white_background = False):
    device = sigma.device

    # delta_prefix_z 每个采样点之间的距离
    # z_vals[...,1:]  光线上第2个点到第64个点的位置
    # z_vals[...,:-1] 光线上第1个点到第63个点的位置
    # 二者相减得到每两个点之间的距离 也就是 δi = ti+1 - ti
    # => 由于光线上的采样点在粗采样的时候是均匀采样的，因此此时的 δ(ti+1-ti) 是定值 0.0635
    delta_sample_point_distance = z_vals[...,1:] - z_vals[...,:-1]
    # 假设最后两个点的距离无限大
    delta_addition = torch.full((z_vals.size(0),1),1e10,device=device)
    # 拼接 delta_sample_point_distance 和 delta_addition
    delta = torch.cat([delta_sample_point_distance,delta_addition],dim = -1)
    # delta                 [1024,64]
    # raydirs[...,None,:]   [1024,1]
    # 通过样本点之间的光线长度来缩放距离
    delta = delta * torch.norm(raydirs[...,None,:],dim = -1)

    # 最终得到delta[1024 * 64]:
    # 每个点不透明度的计算公式：αi = 1 - exp(-σi * δi)
    # 通过矩阵运算可以得到一张图片中1024个采样点的不透明度
    # sigma : 1024 * 64 * 1
    # delta : 1024 * 64

    # 疑问，这个sigma是从哪里传进来的？
    # alpha 是 一条光线上每个点的不透明度
    # alpha_i = 1 - exp(-σi * δi)
    # 根据 alpha 的表达式，σ为0,alpha也为0
    # σ 和 alpha 正相关, alpha的值最大为1
    alpha = 1.0 - torch.exp(- sigma.reshape(len(sigma), -1) * delta)

    exp_term = 1.0 - alpha
    epsilon = 1e-10
    exp_addition = torch.ones(exp_term.size(0), 1 ,device = device)

    exp_term = torch.cat([exp_addition,exp_term+epsilon], dim = -1)
    # 疑问:连乘的使用
    # transmittance = torch.cumprod(torch.tensor([1, 2, 3, 4, 5, 6]), axis=-1)
    # tensor([1, 2, 6, 24, 120, 720])
    # 这里对exp_term的扩展十分巧妙，解决了 Ti = exp(-σi-1 * δi-1)! 连乘上下标不对应的情况。
    # 注意，不取最后一项！ 因为 Ti = exp(0) * exp(-σ1 * δ1) * exp(-σ2 * δ2) * ... * exp(-σi-1 * δi-1)
    # 而 exp_term 的 -1 维度有 65 个元素 分别是[1, exp(-σ1 * δ1) , exp(-σ2 * δ2) , ... , exp(-σi * δi)]
    # 因此，对 exp_term进行连乘后，只用取前64维，就可以得到 Ti 了
    transmittance = torch.cumprod(exp_term, axis = -1)[...,:-1]

    # transmittance:        [1024,64]
    # alpha:                [1024,64]
    # transmittance * alpha [1024,64]
    weights = alpha * transmittance

    # weights:              [1024,64]
    # weights[..., None]:   [1024,64,1]
    # rgb:                  [1024,64,3]
    rgb = torch.sum(weights[...,None] * rgb, dim = -2)
    depth = torch.sum(weights * z_vals ,dim = -1)
    acc_map = torch.sum(weights, -1)

    if white_background:
        rgb = rgb + (1.0 - acc_map[...,None])
    return rgb, depth, acc_map, weights

# bins【1024,63】 就是 z_vals_mid [2.0317 ... 5.9683]
# weights【1024,62】
# N_samples = 128  精细化采样时，一共128个采样点
# det = True : 是均匀采样还是随机生成

# sample_pdf 根据权重的概率分布，对光线进行二次采样，对1024条光线，选取128个新的采样点
def sample_pdf(bins, weights, N_samples, det = True):
    # det = False # 测试用

    # weights[1024,62] 舍去了第一列和最后一列
    device = weights.device
    # 避免出现全零张量
    weights = weights + 1e-5

    # fineNetWork: Cc(r) = ∑ wi * ci , wi = Ti(1 - exp(-σi*δi))
    # normalizing these weights as wi_hat = wi/∑(Nc j=1) wj
    # 将权重归一化，把权重变成了一个概率分布:得到权重的概率分布( weights / sum(weights) )
    pdf = weights / torch.sum(weights, -1 ,keepdim =True)
    cdf = torch.cumsum(pdf,-1)
    # pdf = torch.tensor([0.1, 0.3, 0.2, 0.1, 0.3])
    # cdf = torch.cumsum(pdf, -1)
    # tensor([0.1000, 0.4000, 0.6000, 0.7000, 1.0000])

    # 对权重的概率分布函数左加一列0
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]),cdf],-1)

    # 采用均匀采样 det = True
    if det:
        # u:[0 ... 1] 0-1之间 等分128份 【128】
        u = torch.linspace(0.,1.,steps= N_samples,device = device)
        # u:【1024 128】 1024份 128等分采样点
        u = u.expand(list(cdf.shape[:-1])+[N_samples])
    # det = False
    else:
        # torch.rand(*size)
        # 返回一个张量，包含了从区间[0,1)的均匀分布中抽取的一组随机数。张量的形状由参数size定义。
        u = torch.rand(list(cdf.shape[:-1])+[N_samples]).to(device)

    # 对 u 在内存中重新排列，避免内存重叠
    u = u.contiguous()

    # cdf = torch.tensor([[0.1, 0.2]])
    # u = torch.tensor([[0.09, 0.1, 0.11]])
    # # searchsorted就是返回一个和values一样大小的tensor,
    # # 其中的元素是在cdf中大于等于u中值的索引
    # print(torch.searchsorted(cdf, u))                 tensor([[0, 0, 1]])
    # print(torch.searchsorted(cdf, u, right=False))    tensor([[0, 0, 1]])
    # # 其中的元素是在cdf中大于u中值的索引
    # print(torch.searchsorted(cdf, u, right=True))     tensor([[0, 1, 1]])

    #   在cdf中查找小于等于u的最大索引的位置
#   cdf 【1024, 63】 cdf是weight的概率分布
#   u   【1024,128】 u是采样点的
#   inds【1024,128】
    inds = torch.searchsorted(cdf,u,right=True)
    # 上限索引的位置和下限索引的位置
    below = torch.max(torch.zeros_like(inds-1),inds-1)
    above = torch.min(cdf.shape[-1]-1 * torch.ones_like(inds),inds)
    inds_g = torch.stack([below, above],-1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1],cdf.shape[-1]]

    # inds_g【1024，128，2】
    # cdf_g 【1024，128，2】
    # bins_g【1024，128，2】
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1] - cdf_g[...,0])
    denom = torch.where(denom<1e-5,torch.ones_like(denom),denom)

    # 计算插值系数 t
    t = (u - cdf_g[...,0]) / denom

    # 计算插值
    # samples 【1024,128】
    samples = bins_g[...,0] + t * (bins_g[...,1] - bins_g[...,0])

    # 完成二次采样的操作
    return samples




if __name__ == '__main__':
    root = '/data2/yuanshou/tmp/nerf-main-handai/data/nerf_synthetic/lego'
    transforms_file = 'transforms_train.json'
    half_resolution = False
    provider = DatasetProvider(root,transforms_file,half_resolution)

    batch_size = 1024
    device = 'cuda'
    trainset = NeRFDataset(provider, batch_size, device)



    x_PEdim = 10
    view_PEdim = 4
    coarseModel = NeRF(x_PEdim=x_PEdim,view_PEdim=view_PEdim).to(device)
    fineModel = NeRF(x_PEdim = x_PEdim,view_PEdim = view_PEdim).to(device)
    # params = list(coarseModel.parameters())
    # params.extend(list(fineModel.parameters()))

    # 定义学习率
    learning_rate_decay = 500 * 1000
    learning_rate = 5e-4

    # 导入优化器
    # optimizer = optim.Adam(params,learning_rate)

    # 取第一张图片当中的1024个像素值，拿到这些像素对应的：
    # trainset[0] 这里主要是调用了 NeRFDataset() 类的魔法方法 __getitem__
    # 1.img_pixels  [1024, 3] 像素值
    # 2.ray_dirs    [1024, 3] 像素对应的三维世界的光线
    # 3.ray_oris    [1024, 3] 像素对应的三维世界的相机原点
    ray_dirs, ray_oris, img_pixels = trainset[0]

    # 一条光线上采样点的个数是64
    num_samples = 64

    # 对一条光线的near(2) to far(6)，均匀地取光线上的64个点
    # sample_z_vals [1, 64]
    # sample_z_vals 是一条光线上的64个均匀采样点
    sample_z_vals = torch.linspace(
        2.0,6.0,num_samples,device = device
    ).view(1,num_samples)

    # ray_dirs      【1024,3】  仅仅是光线的3D方向
    # ray_oris      【1024,3】  相机原点
    # sample_z_vals 【1,64】
    rays,z_vals = sample_rays(ray_dirs,ray_oris,sample_z_vals)
    # rays  【1024,64,3】 rays是构造出来的光线，离散表示成64个光线采样点
    # z_vals【1024,64】

    # 归一化 view_dirs [1024, 3]
    # 对一张图像抽取1024个采样点，构造这些采样点对应的现实世界中的光线
    # input:    ray_dirs 【1024, 3】
    # output:   view_dirs【1024, 3】
    view_dirs = sample_viewdirs(ray_dirs)

    # 神经网络传回来的 sigma 和 c
    # 注意这里进行的已经是模型的前向过程了不是初始化过程


    # MLP的输入
    # ① rays        空间中点的三维位置         1024条采样光线上的64个采样点的世界坐标
    # ② view_dirs   观看视角d(归一化视角向量)   对一个像素的观看视角d

    # MLP的输出
    # ① σ:空间中某个位置的体密度值
    # ② c:空间中某个位置的颜色值
    # forward接收的参数(x,vew_dirs)
    # rays【1024, 64, 3】 作为 x
    # view_dirs【1024,3】 作为 vew_dirs

    # forward过程 前向调用MPL的时候，传入的是
    # 1.世界坐标系下，场景中空间位置的三维编码
    # rays 【1024, 64, 3】 即1024条光线上，
    # 每条光线上的64个点的空间位置
    # 2.1024条光线，每条光线对应的方向，作为目标像素的观看视角
    # vew_dirs 【1024，3】

    # NeRF的大型MLP接受的内容就是【x:(rays),d:(view_dirs)】
    # 输出的内容就是：
    # 1.sigma 【1024 64 1】 三维空间中采样点的sigma值,体密度
    # 2.rgb   【1024 64 3】 三维空间中采样点的rgb值,颜色

    sigma, rgb = coarseModel(rays, view_dirs)

    # 配置项: white_background
    white_background = True


    # 1.传入的 sigma【1024 64 1】, rgb【1024 64 3】 是粗网络预测输出的,
    # 这些颜色和体密度值都是空间中1024 * 64个采样点所对应的，而不是预测的每个像素对应的体密度和颜色值
    # 2.传入的 z_vals【1024, 64】是 1024 份 从【near:2】到【far:6】的64个光线深度(z_value)
    # 3.传入的 ray_dirs【1024,3】是未进行归一化的光线方向
    # 4.传入的 white_background 是背景配置项

    # 使用加权求和的方式(体渲染)计算最终呈现在屏幕上的颜色（粗网络阶段的体渲染）
    # 提渲染的过程,返回了4个参数: rgb1, depth1, acc_map, weights
    # 后续会使用的有 rgb1, weights
    rgb1, depth1, acc_map, weights = predict_to_rgb(sigma,rgb,z_vals,ray_dirs,white_background)


    # 计算z_val的中间值

    # z_vals = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    # z_vals_mid = 0.5 * (z_vals[1:] + z_vals[:-1])
    # print(z_vals_mid)
    # tensor([ 1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,  7.5000,  8.5000,
    #          9.5000, 10.5000, 11.5000, 12.5000, 13.5000, 14.5000, 15.5000, 16.5000,
    #         17.5000])
    # 对采样点进行重采样 z_vals[...,1:]舍弃第一个元素； z_vals[...,:-1]舍弃最后一个元素
    z_vals_mid = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
    # 定义精细采样的采样点的个数
    new_samples_num = 128

    # a = [1, 2, 3, 4, 5, 6]
    # print((a[1:-1]))
    # [2, 3, 4, 5]

    # 【sample_pdf】重采样函数
    # 【z_vals_mid】1024，63
    # 【weights[...,1:-1]】
    # z_samples存储128个采样点(采样结果) det = True:是均匀采样还是随机生成
    # sample_pdf 根据权重的概率分布，对光线进行二次采样，对1024条光线，选取128个新的采样点
    z_samples = sample_pdf(z_vals_mid,weights[...,1:-1], new_samples_num,det = True)
    print("123")
n = NeRF()