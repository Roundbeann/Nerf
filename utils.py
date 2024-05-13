import torch.optim as optim
from tqdm import tqdm
import cv2
import os
import json
import argparse
import imageio
import torch
import numpy as np
import torch.nn as nn
from scipy.interpolate import griddata
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
class Embedder(nn.Module):
    def __init__(self,PE_dim):
        super().__init__()
        self.PEdim = PE_dim
    def forward(self,x):
        res = [x]
        for i in range(self.PEdim):
            for triFunction in [torch.sin,torch.cos]:
                res.append(triFunction((2.**i)*x))
        return torch.cat(res, dim=-1)


class ViewDependentHead(nn.Module):
    # self.head = ViewDependentHead(256, 27)
    def __init__(self,input_num,view_dim):
        super().__init__()
        # 第9层（从1开始数）
        self.feature = nn.Linear(input_num,input_num)
        # 输出1维的σ，体密度与观察视角无关，因此输入神经元的个数是256
        # self.alpha = nn.Linear(256,1)
        self.alpha = nn.Linear(input_num,1)
        # 加入观察视角的全连接层 输入是256+27，输出是128
        # self.view_fullConnection = nn.Linear(256+27,128)
        self.view_fullConnection = nn.Linear(input_num + view_dim,input_num//2)
        # 定义最终输出rgb的层
        # self.rgb = nn.Linear(128, 3)
        self.rgb = nn.Linear(input_num//2,3)
    def forward(self,x,view_dirs):
        feature = self.feature(x)
        sigma = self.alpha(x).relu()
        feature = torch.cat((feature,view_dirs),dim = -1)
        feature = self.view_fullConnection(feature.relu())
        rgb = self.rgb(feature).sigmoid()
        return sigma,rgb

class NoViewDirHead(nn.Module):
    # self.head = NoViewDirHead(256,4)
    def __init__(self,input_num,output_num):
        super().__init__()
        self.head = nn.Linear(input_num,output_num)

    def forward(self,x,view_dirs):
        x = self.head(x)
        # 有疑问,是一下子直接输出σ和c了
        # len(x) = 4
        # 取x的前三维为rgb
        rgb = x[..., :3].sigmoid()
        # 取x的最后一维为体密度
        sigma = x[..., 3].relu()
        return sigma, rgb

class DatasetProvider:
    def __init__(self,root,transforms_file, half_resolution = False):
        self.meta = json.load(open(os.path.join(root,transforms_file),"r"))
        self.root = root
        #  self.frames 包含100张训练图像的RT矩阵
        self.frames = self.meta['frames']
        # 所有图片的 camera_angle_x 是相同值（可能和相机固有的属性有关）
        self.camera_angle_x = self.meta['camera_angle_x']

        # self.poses 和 self.images用于存储每张图片的RT矩阵和图片像素值
        # self.poses 【list: 100】
        # self.images【list: 100】【eachPic: 800,800,4】 【4 代表每个像素点的RGBA值】
        self.poses = []
        self.images = []

        # 读取每张图片的像素值和RT矩阵，存储到self.poses = [] 和 self.images = [] 当中
        for frame in self.frames:
            image_file = os.path.join(self.root,frame['file_path']+'.png')
            image = imageio.imread(image_file)
            if half_resolution:
                image = cv2.resize(image, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            self.images.append(image)
            self.poses.append(frame['transform_matrix'])

        # 把图片位姿和图片像素都转换为numpy格式
        # np.stack可以把多层list直接转成numpy
        # self.poses 【100，4，4】 RT矩阵是4*4的
        self.poses = np.stack(self.poses)

        # self.images的RGBA的值域为[0,255] 要把像素值归一化到[0,1]
        self.images = (np.stack(self.images)/255).astype(np.float32)
        # self.images中存储了所有的图像数据 其形状为(100,800,800,4)
        # 即 100 张图片，每张图片的长为800，
        # self.images.shape = (100,800,800,4)

        # 为什么要存宽高？ 后面要构建射线，需要拿到每个像素的坐标，要将像素坐标转换为相机坐标，再从相机坐标转换为世界坐标
        self.width = self.images.shape[2]
        self.height = self.images.shape[1]

        # 焦距可以将像素坐标转化为归一化平面上的坐标
        # 疑问：焦距为什么这样计算？ 计算焦距
        self.focal = 0.5 * self.width /  np.tan(0.5 * self.camera_angle_x)

        # 获得透明度 图像的像素是RGBA四个维度 最后一个维度是不透明度alpha
        alpha = self.images[:,:,:,[3]]
        rgb = self.images[:, :, :, :3]
        # 对于空白区域 rgb 是 0，alpha 是 0, 1 - alpha 是 1
        # 在这种情况下，中间有像素的区域变成了（0，1），完全无像素的区域变成了
        self.images = rgb * alpha + (1-alpha)
        # 测试以下转换后的self.images长什么样子
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(self.images[2,:,:,[0]][0])
        # plt.show()
        # plt.imshow(self.images[2, :, :, [1]][0])
        # plt.show()
        # plt.imshow(self.images[2, :, :, [2]][0])
        # plt.show()

        pass

class NeRF(nn.Module):
    # 构造函数定义好处理batch数据的模型（权重矩阵）
    def __init__(self, x_PEdim = 10, nnWidth = 256, nnDeepth = 8 , view_PEdim = 4):
        super().__init__()
        # (10*2+1)*3 = 63 = 3 + 60
        # x y z + sin1(x y z) + cos1(x y z)
        #       + sin2(x y z) + cos2(x y z)
        #       + sin...(x y z) + cos3(x y z)
        #       + sin10(x y z) + cos3(x y z)
        # x_dim = (x_PEdim * 2 + 1) * 3
        # 根据位置编码的维度确定 MLP输入层的维度大小和输出层的维度大小
        x_dim = 3 * x_PEdim * 2 + 3
        layers = []
        layers_in = [nnWidth] * nnDeepth
        # layers_in = [256,256,256,256,256,256,256,256]
        layers_in[0] = x_dim
        layers_in[5] = nnWidth + x_dim
        # layers_in = [63,256,256,256,256,319,256,256]


        # layers_in 定义了Nerf MLP 每层的输入大小
        # nnWidth 每一层的输出大小均定义为256
        for i in range(nnDeepth):
            # nn.Linear(输入神经元的个数,输出神经元的个数)
            layers.append(nn.Linear(layers_in[i],nnWidth))

        # layers
        # [
        # Linear(in_features=63, out_features=256, bias=True),
        # Linear(in_features=256, out_features=256, bias=True),
        # Linear(in_features=256, out_features=256, bias=True),
        # Linear(in_features=256, out_features=256, bias=True),
        # Linear(in_features=256, out_features=256, bias=True),

        # 这一层的输入维度突然增加是因为 256 + 63 = 319 重新加了一次 63 维度的位置编码
        # Linear(in_features=319, out_features=256, bias=True),
        # Linear(in_features=256, out_features=256, bias=True),
        # Linear(in_features=256, out_features=256, bias=True)
        # ]

        # 以下根据是否添加观察视角 决定使用哪个输出头
        # 添加了观察视角的输出头：ViewDependentHead()
        # 不添加观察视角的输出头：NoViewDirHead()
        # 在 MLP 输出体密度时添加观看视角的位置编码
        if view_PEdim>0:
            # (4*2+1) * 3 = 27
            # x y z + sin1(x y z) + cos1(x y z)
            #       + sin2(x y z) + cos2(x y z)
            #       + sin3(x y z) + cos3(x y z)
            #       + sin4(x y z) + cos4(x y z)
            view_dim = view_PEdim * 3 * 2 +  3
            self.view_embed = Embedder(view_PEdim)
            # self.head = ViewDependentHead(256,27)
            self.head = ViewDependentHead(nnWidth,view_dim)
        else:
            # self.head = NoViewDirHead(256,4)
            self.head = NoViewDirHead(nnWidth,4)

        # 【定义】三维点和光线的【位置编码函数】
        self.xembed = Embedder(x_PEdim)
        # 把输出头之前的线性层打包到【nn.Sequential】当中
        self.layers = nn.Sequential(*layers)
        pass
    # MLP接收的是 1.空间中三维点的位置坐标 x 2.对位置 x 的观察视角
    # 执行forward前向操作（用于接收batch数据的）
    def forward(self,x,view_dirs):
        # 输入的 x 是【1024】条光线上的【64】个位置
        xshape = x.shape
        # 这里不应该调用forward吗？
        # 为什么上面的代没有看到 __init__()、forword()函数的出现就完成了上述代码的调用呢？
        # 初始化一个类时，则自动调用了该类的__init__()方法【net = SimpleNet(10, 20, 5)】
        # 调用一个类的实例时，会自动调用该类的forward()方法【output_tensor = net(input_tensor)】

        # 对空间当中三维点的位置坐标进行位置编码
        # 对一条光线上64个采样点的位置进行PositionalEncoding
        x = self.xembed(x)

        # 用于消融实验：不加对三维点的观察方向向量
        if self.view_embed is not None:
            # 原来：1024 * 3
            # 现在扩展成：1024 * 64 * 3 (为什么要进行这样的拓展？=>
            # 给每一个空间当中的采样点都分配一个观看方向，即使一条光线上采样点的观看方向是相同的，
            # 也就是会有64个重复的观看方向)
            view_dirs = view_dirs[:,None].expand(xshape)
            view_dirs = self.view_embed(view_dirs)
        raw_x = x

        # 下面的代码 把三维点的坐标作为神经网络的输入
        # 开始进行一层又一层的非线性(由linear+relu函数实现)变换
        for i ,layers in enumerate(self.layers):
            x = torch.relu(layers(x))
            # 为了防止原始的三维点的位置信息在神经网络中被遗忘，
            # 在神经网络的第五层重新把三维点的位置信息concat加入到网络当中：
            if i == 4:
                x = torch.cat([x,raw_x],axis = -1)

        # 三维空间中的位置坐标经历完几层MLP之后，在经历输出头
        # 1.输出每个点的体密度值
        # 2.根据是否参考观察视角，输出每个点的RGB值

        return self.head(x,view_dirs)


# NeRFdataset不是普通的Dataset,不断吐出一个批量大小的图片就好了
# 而是：
class NeRFDataset:
    def __init__(self,provider:DatasetProvider,batch_size = 1024, device = "cuda"):
        # provider包含定义好的成员变量 即self.images,self.poses,self.focal,self.width,self.height
        self.images = provider.images
        self.poses = provider.poses

        self.focal = provider.focal
        self.width = provider.width
        self.height = provider.height

        self.batch_size = batch_size
        self.num_image = len(self.images)
        # 让模型的前500轮只取图像的中间部分(有实际像素的部分去训练)
        self.precrop_iters = 500
        self.precrop_frac = 0.5

        self.niter = 0
        self.device = device
        self.initialize()

    # NeRFDataset 在 initialize() 中最重要的事情就是存储了:
    # 1.所有训练图像的像素值            【100，640000，3】self.images
    # 2.所有训练图像每个像素对应的光线    【100，640000，3】self.all_ray_dirs
    # 3.所有训练图像每个像素对应的光线原点 【100，640000，3】self.all_ray_origins


    # initialize得到了100张训练图片每张图片的
    # 1.640000（800*800）每个像素的像素值
    # 2.640000（800*800）每个像素对应的三维世界的相机光心
    # 3.640000（800*800）每个像素对应的三维世界的光线方向
    def initialize(self):
        # warange = [0,1,2,...,799]
        warange = torch.arange(self.width,dtype = torch.float32,device = self.device)
        # harange = [0,1,2,...,799]
        harange = torch.arange(self.height, dtype=torch.float32, device=self.device)

        # x、y 代表每个像素点的横纵坐标
        y,x =torch.meshgrid(harange,warange)

        # 把像素坐标转换到相机坐标 疑问：明确一下这个计算
        # 根据相似三角形，计算相机坐标系下的点的x和y
        # 完成从图片像素坐标到相机坐标的转换

        # self.transformed_x: 相机坐标系下的 x
        # self.transformed_y: 相机坐标系下的 y
        self.transformed_x = (x - self.width*0.5) / self.focal
        self.transformed_y = (y - self.height*0.5) / self.focal

        # self.precrop_index:
        # tensor([[     0,      1,      2,  ...,    797,    798,    799],
        #         [   800,    801,    802,  ...,   1597,   1598,   1599],
        #         [  1600,   1601,   1602,  ...,   2397,   2398,   2399],
        #         ...,
        #         [637600, 637601, 637602,  ..., 638397, 638398, 638399],
        #         [638400, 638401, 638402,  ..., 639197, 639198, 639199],
        #         [639200, 639201, 639202,  ..., 639997, 639998, 639999]])
        # 宽和高相乘 self.precrop_index 用于索引出precrop之后的坐标

        # 639999这个数字不奇怪的，图片的大小是 800 * 800，那么就代表有640000个像素点
        self.precrop_index = torch.arange(self.width*self.height).view(self.height,self.width)
        dH = int(self.height // 2 * self.precrop_frac)
        dW = int(self.width // 2 * self.precrop_frac)
        # tensor([[160200, 160201, 160202,  ..., 160597, 160598, 160599],
        #         [161000, 161001, 161002,  ..., 161397, 161398, 161399],
        #         [161800, 161801, 161802,  ..., 162197, 162198, 162199],
        #         ...,
        #         [477800, 477801, 477802,  ..., 478197, 478198, 478199],
        #         [478600, 478601, 478602,  ..., 478997, 478998, 478999],
        #         [479400, 479401, 479402,  ..., 479797, 479798, 479799]])
        #  self.precrop_index = self.precrop_index[
        #             800 // 2 - 200 : 800 // 2 + 200,  纵向中心点往下一半，往上一半
        #             800 // 2 - 200 : 800 // 2 + 200,  横向中心点往左一半，往右一半
        #         ].reshape(-1)

        # 刚开始训练（仅对前500张图片做这样的操作），为了快速收敛
        # 对训练的图像进行剪裁，只取图像中间主体部分，不要图像周围的空白部分
        # 构建出中间区域的像素索引
        self.precrop_index = self.precrop_index[
            self.height// 2 - dH : self.height// 2 + dH,
            self.width // 2 - dW : self.width // 2 + dH,
        ].reshape(-1)
        # self.precrop_index = tensor([160200, 160201, 160202,  ..., 479797, 479798, 479799])

        # 得到 100 张训练图像的相机位姿，每个相机位姿是 4 * 4 的 RT 矩阵
        # 相机位姿的作用：构造像素对应的光线
        # poses 和 self.poses的区别
        # poses 不是类的成员变量，poses被放在了GPU上
        poses = torch.tensor(self.poses,dtype=torch.float32, device = self.device)

        # 定义射线的原点和方向
        all_ray_dirs, all_ray_origins = [],[]

        # 对每一张图片对应的相机原点和相机光线进行处理
        # 对每张输入图片执行以下操作（循环）
        for i in range(len(self.images)):
            # self.transformed_x : 相机坐标系下的x => 相机坐标系下的横坐标 类似于800行 [0，1，2，3...，799]
            # self.transformed_y : 相机坐标系下的y => 相机坐标系下的纵坐标 类似于800列 [0，1，2，3...，799]^T
            # 对于一张图片，要获得每个像素在世界坐标系下的光线和相机原点
            # 这样的光线和相机原点分别存储在ray_dirs和ray_origins当中
            # 如何获得光线和相机原点？
            # 需要先self.make_rays()函数当中传入：self.transformed_x,self.transformed_y,poses[i]，这三个东西是像素对应在相机坐标系的位置，以及相机的RT矩阵
            # 也就是图片中每一个位置的像素，都对应相机坐标系下的一个位置（归一化平面）
            # 而相机坐标系下的一个位置self.transformed_x,self.transformed_y,根据poses，也就是相机的旋转平移矩阵的变换，可以得到图片中一个像素对应的在世界坐标系下的那条射线
            # 输入数据的维度：
            # self.make_rays([800,800]=>800*800个点的x,[800,800]=>800*800个点的y)
            # 因此就有800*800 = 640000个像素点，因此对应640000条世界坐标系下的光线

            # 对于相机坐标系下的每一个坐标即[self.transformed_x[i],self.transformed_y[j]]
            # 对这个相机坐标系下的每个坐标位置，都能根据【图像对应的相机位姿】，
            # ①【构造出一条射线】这条射线对应世界坐标系下的一个点到相机坐标系下一个点【坐标】的投影方向
            # ray_dirs
            # ②【构造出相机原点】
            # ray_origins

            # 根据相机坐标系下的点 (self.transformed_x,self.transformed_y) 以及对应的相机位姿
            # 构建得到那一点 对应的 相机光心和光线方向


            # 注意，这里是遍历过程
            # 每次得到一张图片的800*800=640000个像素 每个像素对应的现实世界的光线及光心
            ray_dirs,ray_origins = self.make_rays(self.transformed_x,self.transformed_y,poses[i])
            # 保存这张图片的全部光线及光心
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)
        # all_ray_dirs:     result = {list:100}, 每个list: Tensor(640000,3)
        # self.all_ray_dirs:result = {Tensor:(100, 640000, 3)}

        # 把全部训练数据（100张图片）的光线及光心保存到成员变量当中 (转成tensor)
        self.all_ray_dirs = torch.stack(all_ray_dirs,dim = 0)
        self.all_ray_origins = torch.stack(all_ray_origins, dim=0)
        # 把全部训练数据（100张图片）的像素值保存到成员变量
        self.images = torch.tensor(self.images,dtype=torch.float32).view(self.num_image,-1,3)

        pass

    # 构造光线,返回的是每张照片里的像素在世界坐标系下的坐标点
    # 还有每张照片在相机坐标系下的原点在世界坐标系下的表示

    # make_rays()是在迭代中被调用的
    # 输入:
    # ① x,y 是相机坐标系下点的坐标    【800 * 800】，【800 * 800】
    # ② pose是单张图片的相机位姿      【4 * 4】

    # x     ([[-0.3600, -0.3591, -0.3582,  ...,  0.3573,  0.3582,  0.3591],
    #         [-0.3600, -0.3591, -0.3582,  ...,  0.3573,  0.3582,  0.3591],
    #         [-0.3600, -0.3591, -0.3582,  ...,  0.3573,  0.3582,  0.3591],
    #         ...,
    #         [-0.3600, -0.3591, -0.3582,  ...,  0.3573,  0.3582,  0.3591],
    #         [-0.3600, -0.3591, -0.3582,  ...,  0.3573,  0.3582,  0.3591],
    #         [-0.3600, -0.3591, -0.3582,  ...,  0.3573,  0.3582,  0.3591]],
    #        device='cuda:0')

    # y     ([[-0.3600, -0.3600, -0.3600,  ..., -0.3600, -0.3600, -0.3600],
    #         [-0.3591, -0.3591, -0.3591,  ..., -0.3591, -0.3591, -0.3591],
    #         [-0.3582, -0.3582, -0.3582,  ..., -0.3582, -0.3582, -0.3582],
    #         ...,
    #         [ 0.3573,  0.3573,  0.3573,  ...,  0.3573,  0.3573,  0.3573],
    #         [ 0.3582,  0.3582,  0.3582,  ...,  0.3582,  0.3582,  0.3582],
    #         [ 0.3591,  0.3591,  0.3591,  ...,  0.3591,  0.3591,  0.3591]],
    #        device='cuda:0')

    # pose  ([[-9.9990e-01,  4.1922e-03, -1.3346e-02, -5.3798e-02],
    #         [-1.3989e-02, -2.9966e-01,  9.5394e-01,  3.8455e+00],
    #         [-4.6566e-10,  9.5404e-01,  2.9969e-01,  1.2081e+00],
    #         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device='cuda:0')


    # 如何构造一条光线？
    # 1.传入图片中一个像素对应相机坐标系下的坐标 以及这张图片的RT矩阵
    # 2.就可以把相机坐标系下的坐标(x,y)利用 RT(pose)投射到 三维世界的一条光线上
    # 3.这条光线由 r = o + td进行表示
    # 4.o代表相机光心，t代表 t_near 到 t_far 的 64/128 个采样点，d表示归一化的光线方向
    def make_rays(self,x,y,pose):
        # pose [4, 4] 一张图片对应的位姿
        # 注意，这里输入的x不是像素值，而是每张图片的像素在相机坐标系下的坐标     （x中每个位置相差的值为0.0009
        # 同样，输入的y也不是像素值，而是图片中每个像素在相机坐标系下对应的坐标   （相邻的两个位置之间相差0.0009

        # 相机坐标系：[x,y,z]
        # COLMAP:[x,-y,-z]
        # 转换为colmap坐标系
        # 这里假设光线的方向垂直于相机平面 z = -1
        directions = torch.stack([x,-y,-torch.ones_like(x)], dim=-1)

        # pose是什么？ 是4*4的RT变换矩阵，也是相机坐标系到世界坐标系的转换矩阵
        # 因此一个点乘以这个矩阵的话可以完成一个点在相机坐标系下的表示到世界坐标系的表示
        # 另外一种理解方法：是相机坐标系在世界坐标系下的位姿 transform_matrix在世界坐标系下是可以来回运动的

        # 提取旋转矩阵 R
        # 相机坐标系到世界坐标系的旋转矩阵 R (4*4当中的3*3)
        camera_matrix = pose[:3,:3]

        # 将光线的方向旋转到世界坐标系下
        ray_dirs = directions.reshape(-1,3) @ camera_matrix.T

        # 把平移量当成相机光心
        # 根据(平移向量)获得相机中心在世界坐标系下对应的位置
        ray_origin = pose[:3,3].view(1,3).repeat(len(ray_dirs),1)

        # 返回世界坐标系下的光心和光线方向
        return ray_dirs,ray_origin

    def __len__(self):
        return self.num_image

    # 在索引trainset[5]时，调用魔法方法__getitem__()
    # 假设传来的索引为5


    # NeRFDataset这个数据迭代器要保证
    # 1.对于吐出的前500张图像的像素和光线，要保证这些像素来自图像的中心区域，来加速模型的收敛速度
    # 2.每张图片要么有800*800=640000 要么有400*400=160000(对于前500轮而言)
    #   不可能把这么多像素点全部交给MLP进行训练，要选取 光线总条数(640000/160000) 当中的nrays = 1024条进行训练:
    #   np.random.choice(ray_dirs.shape[0],size = [nrays],replace = False)
    def __getitem__(self,index):
        self.niter += 1
        # 得到 第5张图片 640000个像素点 每个像素点对应的光线方向(世界坐标系下)
        ray_dirs = self.all_ray_dirs[index]
        # 得到 第5张图片 640000个像素点 每个像素点对应的相机原点(世界坐标系下)
        ray_oris = self.all_ray_origins[index]
        # 得到 第5张图片的像素值
        img_pixels = self.images[index].to(self.device)

        if self.niter < self.precrop_iters:
            # 对前500(self.precrop_iters)张图片,只取800*800中间的400*400像素进行采样
            # 获取一张图片中间部分像素对应的光线、世界坐标系下相机中心、像素值

            ray_dirs = ray_dirs[self.precrop_index]
            ray_oris = ray_oris[self.precrop_index]
            img_pixels = img_pixels[self.precrop_index]

        # 一张图片要么有160000个像素点，要么有640000个像素点  要取这些像素点中的1024个进行批处理
        nrays = self.batch_size

        select_ints = np.random.choice(ray_dirs.shape[0],size = [nrays],replace = False)
        ray_dirs = ray_dirs[select_ints]
        ray_oris = ray_oris[select_ints]
        img_pixels = img_pixels[select_ints]

        return ray_dirs,ray_oris,img_pixels

    # 获取测试数据的函数
    def get_test_item(self, index = 0 ):
        ray_dirs = self.all_ray_dirs[index]
        ray_oris = self.all_ray_origins[index]
        image_pixels = self.images[index].to(self.device)

        # for i in range(0, 640000, 1024)
        # i: [0, 1024, 2048, 3072, 4096, 5120 ... 640000]
        for i in range(0,len(ray_dirs), self.batch_size):
            # 一批一批地返回一张图片(包含640000个像素)的1024个像素点的光线方向,相机原点,像素值
            # 返回的ray_dirs,ray_oris,image_pixels用于测试,他们的大小都是【1024，3】
            yield ray_dirs[i:i+self.batch_size],ray_oris[i:i+self.batch_size],image_pixels[i:i+self.batch_size]

    # 生成360°的渲染的视频
    def get_rotate_360_rays(self):
        # 平移矩阵
        def trans_t(t):
            return np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, t],
                [0, 0, 0, 1],
            ], dtype=np.float32)

        # 绕 x 轴旋转的函数
        def rot_phi(phi):
            return np.array([
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

        # 绕 y 轴旋转的函数
        def rot_theta(th):
            return np.array([
                [np.cos(th), 0, -np.sin(th), 0],
                [0, 1, 0, 0],
                [np.sin(th), 0, np.cos(th), 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)


        # 生成球面坐标系下，相机位姿矩阵
        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            # 绕 x 轴旋转
            c2w = rot_phi(phi / 180. * np.pi) @ c2w
            # 绕 y 轴旋转
            c2w = rot_theta(theta / 180. * np.pi) @ c2w
            # 坐标系的转换
            c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
            return c2w

        # 在-180°到+180°均匀生成41个角度
        for th in np.linspace(-180., 180., 41, endpoint=False):
            # 对每个角度生成相应的旋转矩阵pose
            pose = torch.FloatTensor(pose_spherical(th, -30., 4.), device=self.device)

            def genfunc():
                ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, pose)
                for i in range(0, len(ray_dirs), 1024):
                    yield ray_dirs[i:i + 1024], ray_origins[i:i + 1024]

            yield genfunc












# 传入：
# 一张图片中1024个采样点对应的光线方向、相机原点和光线上的64个z值(sample_z_vals)
# 返回：
# rays          【1024,64,3】
# sample_z_vals 【1024,64,1】
def sample_rays(ray_directions, ray_origins, sample_z_vals):
    # ray_directions 【1024,3】
    # ray_origins    【1024,3】
    # sample_z_vals  【1,  64】 => 【1024，64】

    # 一共传入一张图片中1024个像素对应的光线
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

    # test stuff
    numpy_rays = rays.detach().cpu().numpy()
    test_list = []
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for ray in numpy_rays[:100]:
        vec_norm = np.linalg.norm(ray)

        unit_vec = ray / vec_norm
        test_list.append(unit_vec)

        x = ray[:, 0]  # [ 0  3  6  9 12 15 18 21]
        y = ray[:, 1]  # [ 1  4  7 10 13 16 19 22]
        z = ray[:, 2]



        # s：marker标记的大小
        # c: 颜色  可为单个，可为序列
        # depthshade: 是否为散点标记着色以呈现深度外观。对 scatter() 的每次调用都将独立执行其深度着色。
        # marker：样式
        ax.scatter(xs=x, ys=y, zs=z, s=1, depthshade=True, cmap="jet", marker="^")
    ax.scatter(xs=float(ray_origins[0][0]), ys=float(ray_origins[0][1]), zs=float(ray_origins[0][2]), s=1, depthshade=True, cmap="jet", marker="^")
    end_point = numpy_rays[:100][:,-16,:]
    X = end_point[:,0]
    Y = end_point[:,1]
    Z = end_point[:,2]
    # 创建网格点，用于插值
    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]

    # 将散点数据插值到网格上
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')


    end_point = numpy_rays[:100][:,-32,:]
    X = end_point[:,0]
    Y = end_point[:,1]
    Z = end_point[:,2]
    # 创建网格点，用于插值
    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]

    # 将散点数据插值到网格上
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')

    end_point = numpy_rays[:100][:, -48, :]
    X = end_point[:, 0]
    Y = end_point[:, 1]
    Z = end_point[:, 2]
    # 创建网格点，用于插值
    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]

    # 将散点数据插值到网格上
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')


    fig.colorbar(surf)

    # 设置标签
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    end_point = numpy_rays[:100][:, 0, :]
    X = end_point[:, 0]
    Y = end_point[:, 1]
    Z = end_point[:, 2]
    # 创建网格点，用于插值
    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]

    # 将散点数据插值到网格上
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')

    end_point = numpy_rays[:100][:, -1, :]
    X = end_point[:, 0]
    Y = end_point[:, 1]
    Z = end_point[:, 2]
    # 创建网格点，用于插值
    grid_x, grid_y = np.mgrid[X.min():X.max():100j, Y.min():Y.max():100j]

    # 将散点数据插值到网格上
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')

    plt.show()

    # 返回当前训练图片的1024个像素对应的空间光线采样点，以及2-6的64个点
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

    # 根据体渲染的公式，首先获得每两个采样点之间的间隔（采样点是均匀分布的，故间隔都相等）
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
    # bins光线采样的均匀分箱

    # weights[1024,62] 舍去了第一列和最后一列
    device = weights.device
    # 避免出现全零张量

    # weights:一张图片 1024条光线上 62个采样点每个采样点的权重值/亮度值
    weights = weights + 1e-5

    # fineNetWork: Cc(r) = ∑ wi * ci , wi = Ti(1 - exp(-σi*δi))
    # normalizing these weights as wi_hat = wi/∑(Nc j=1) wj
    # 将权重归一化，把权重变成了一个概率分布:得到权重的概率分布( weights / sum(weights) )

    # 计算权重的概率密度函数
    pdf = weights / torch.sum(weights, -1 ,keepdim =True)
    # 计算权重的概率分布函数
    cdf = torch.cumsum(pdf,-1)
    # pdf = torch.tensor([0.1, 0.3, 0.2, 0.1, 0.3])
    # cdf = torch.cumsum(pdf, -1)
    # tensor([0.1000, 0.4000, 0.6000, 0.7000, 1.0000])

    # pdf存储了一条光线上每个空间点的权重的概率
    # cdf存储了一条光线上全体空间点权重的概率分布

    plt.scatter(np.arange(62),pdf[0].detach().cpu().numpy())
    plt.scatter(np.arange(62), cdf[0].detach().cpu().numpy())
    plt.show()
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
    # 查看新的采样点的分布状况

    plt.scatter(x=samples[0].detach().cpu().numpy(), y=np.ones(128))
    plt.show()
    # 完成二次采样的操作
    return samples


def render_rays(coarseModel, fineModel, ray_dirs, ray_oris, sample_z_vals, num_samples2, white_background=False):
    # 1.【sample_rays】 对一张图像抽取1024个采样点，构造这些采样点对应的现实世界中的光线
    # 【input】
    # ray_dirs      【1024,3】  仅仅是光线的3D方向
    # ray_oris      【1024,3】  相机原点
    # sample_z_vals 【1,64】    near to far [2:6]
    # 【output】
    # rays  【1024,64,3】 rays是构造出来的光线，离散表示成64个光线采样点
    # z_vals【1024,64】

    # 在光线方向上从近平面到远平面取64个点
    # rays其实不是光线，而是光线上64个均匀分布的采样点，可以看作是光线的离散表达
    # z_vals代表的就是每个采样点在近平面到远平面的尺度放缩
    rays, z_vals = sample_rays(ray_dirs, ray_oris, sample_z_vals)

    # 2.【sample_viewdirs】
    # 归一化 view_dirs [1024, 3]
    # 【input】    ray_dirs 【1024, 3】
    # 【output】   view_dirs【1024, 3】

    # 得到归一化的光线方向
    view_dirs = sample_viewdirs(ray_dirs)

    # 3.【coarseModel】 传入x和d,调用Coarse模型

    # 神经网络传回来的 sigma 和 c
    # 注意这里进行的已经是模型的前向过程了不是初始化过程
    # forward接收的参数(x,vew_dirs)

    # MLP的输入
    # ① rays : x       空间中点的三维位置         1024条采样光线上的64个采样点的世界坐标
    # ② view_dirs : d  观看视角d(归一化视角向量)   对一个像素的观看视角d
    # rays【1024, 64, 3】 作为 x
    # view_dirs【1024,3】 作为 d

    # MLP的输出
    # ① σ:空间中某个位置的体密度值  sigma  【1024 64 1】 三维空间中采样点的sigma值,体密度
    # ② c:空间中某个位置的颜色值    rgb    【1024 64 3】 三维空间中采样点的rgb值,颜色

    # 训练所需的数据处理好了，下一步开始利用MLP对一个点的颜色值和体密度值进行预测
    # rays：1024个像素对应现实世界光线上的64个采样点

    #  MLP的输入：
    # 光线采样点rays
    # 对这个采样点的观察视角
    sigma, rgb = coarseModel(rays, view_dirs)

    sigma = sigma.squeeze(dim=-1)

    # 【4.predict_to_rgb】
    # 1.传入的 sigma【1024 64 1】, rgb【1024 64 3】 是粗网络预测输出的,
    # 这些颜色和体密度值都是空间中1024 * 64个采样点所对应的，而不是预测的每个像素对应的体密度和颜色值
    # 2.传入的 z_vals【1024, 64】是 1024 份 从【near:2】到【far:6】的64个光线深度(z_value)
    # 3.传入的 ray_dirs【1024,3】是未进行归一化的光线方向
    # 4.传入的 white_background 是背景配置项

    # 使用加权求和的方式(体渲染)计算最终呈现在屏幕上的颜色（粗网络阶段的体渲染）
    # 提渲染的过程,返回了4个参数: rgb1, depth1, acc_map, weights
    # 后续会使用的有 rgb1, weights


    # 经过第一层MLP得到了三维空间中每个点的颜色值和体密度值
    # 接下来就需要使用体渲染公式对一条光线上64个点的颜色值和体密度值进行加权计算
    # 得到这条光线对应像素的颜色值，由predict_to_rgb实现
    rgb1, _, _, weights1 = predict_to_rgb(sigma, rgb, z_vals, ray_dirs, white_background)

    # 【5.sample_pdf】 计算z_val的中间值，对光线进行重采样

    # z_vals = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
    # z_vals_mid = 0.5 * (z_vals[1:] + z_vals[:-1])
    # print(z_vals_mid)
    # tensor([ 1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000,  7.5000,  8.5000,
    #          9.5000, 10.5000, 11.5000, 12.5000, 13.5000, 14.5000, 15.5000, 16.5000,
    #         17.5000])
    # 对采样点进行重采样 z_vals[...,1:]舍弃第一个元素； z_vals[...,:-1]舍弃最后一个元素
    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

    # a = [1, 2, 3, 4, 5, 6]
    # print((a[1:-1]))
    # [2, 3, 4, 5]

    # 【sample_pdf】重采样函数
    # 【z_vals_mid】1024，63
    # 【weights[...,1:-1]】
    # z_samples存储128个采样点(采样结果) det = True:是均匀采样还是随机生成
    # sample_pdf 根据权重的概率分布，对光线进行二次采样，对1024条光线，选取128个新的采样点
    z_samples = sample_pdf(z_vals_mid, weights1[..., 1:-1],num_samples2 , det=True)
    z_samples = z_samples.detach()



#     将原始的深度和重采样的深度值拼接起来，并且按照深度从小到大排序
#     z_vals【1024,192】
#     rays  【1024,192,3】
    z_vals,_ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    rays = ray_oris[...,None,:] + ray_dirs[...,None,:] * z_vals[...,:,None]

    sigma,rgb = fineModel(rays,view_dirs)
    sigma = sigma.squeeze(dim=-1)

    rgb2,_,_,_ = predict_to_rgb(sigma,rgb,z_vals,ray_dirs,white_background)
    return rgb1,rgb2




