# import torch
# from math import  sqrt
# def forward(x):
#     res = [x]
#     for i in range(10):
#         for triFunction in [torch.sin, torch.cos]:
#             res.append(triFunction((2. ** i) * x))
#     return torch.cat(res,dim=-1)
# x = torch.tensor([1.0,1.5,2.0])
# print(forward(x))
#
# apple = [5]
# applebox = apple * 5
# print(applebox)
#
# import numpy as np
#
# a = np.array([0,1,2,3,4,5])
# print('第2列元素', a[..., 3])
# import torch
# import torch.nn as nn
#
#
#
#
#
# class SimpleNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(SimpleNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out
#
#     # # 创建一个 SimpleNet 对象，设置输入维度为 10，隐藏层维度为 20，输出维度为 5
#     # net = SimpleNet(10, 20, 5)
#     #
#     # # 构造一个随机的输入张量，大小为 [batch_size, input_dim]，这里令 batch_size=1
#     # input_tensor = torch.randn(1, 10)
#     #
#     # # 将输入张量传入网络中，得到输出张量
#     # output_tensor = net(input_tensor)
#     #
#     # # 打印输出张量的形状
#     # print(output_tensor.shape)
#
#
# aa=np.array([[[[1],[2],[3]],
#     [[1], [2], [3]],
#     [[1], [2], [3]]],
# [[[1],[2],[3]],
#     [[1], [2], [3]],
#     [[1], [2], [3]]]])
# bb =np.array([[[[1,2],[2,2],[3,2]],
#     [[1,2],[2,2],[3,2]],
#     [[1,2],[2,2],[3,2]]],
# [[[1,2],[2,2],[3,2]],
#  [[1, 2], [2, 2], [3, 2]],
#     [[1,2],[2,2],[3,2]]]])
# print(aa.shape,bb.shape)
# cc = bb*aa
# dd = aa * bb
# print("cc",cc,"dd",dd,cc == dd)
#
#
# aaa = [ 0.6148, -5.7324, -1.7466]
# sum = 0
# for i in aaa:
#     sum = sum + i**2
# print(sqrt(sum))
#
#
# aaa = [1,2,3,4,5]
# print(aaa[1:])
# print(aaa[:-1])
#
# aaaa = torch.tensor([[1],[2],[3]])
# bbbb = torch.tensor([[1,2],[3,4],[5,6]])
# print("123",aaaa*bbbb)
#
# transmittance = torch.cumprod(torch.tensor([1,2,3,4,5,6]), axis=-1)
# print(transmittance)
#
# z_vals = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
# z_vals_mid = 0.5 * (z_vals[1:] + z_vals[:-1])
# print(z_vals_mid)
#
# pdf = torch.tensor([0.1,0.3,0.2,0.1,0.3])
# cdf = torch.cumsum(pdf,-1)
# print(cdf)
#
#
#
# tensor3_4 = torch.tensor(
#     [[1,2,3,4],[2,3,4,5],[3,4,5,6]]
# )
# print(tensor3_4.shape)
# print(tensor3_4[...,None],tensor3_4[...,None].shape)
# print(tensor3_4[None,...],tensor3_4[None,...].shape)
# print(tensor3_4[...,None,:],tensor3_4[...,None,:].shape)
# print(tensor3_4[None,:,None,:],tensor3_4[None,:,None,:].shape)
# tensor_1314 = tensor3_4[None,:,None,:]
# print("------")
# print(tensor_1314[...,None,:].shape)
# print(tensor_1314[:,None,...].shape)
# print(tensor_1314[:,None,:].shape)
# print(tensor_1314[...,None,...].shape)
# print("------")
# print(tensor_1314[:,None].shape)
# print(tensor_1314[...,None].shape)
# print("------")
# print(tensor_1314[:,None].shape)
# print(tensor_1314[:,:,None].shape)
# print(tensor_1314[:,:,:,None].shape)
# print(tensor_1314[:,:,:,:,None].shape)
# print("------")
# print(tensor_1314[...,None,:].shape)
# print(tensor_1314[...,None,:,:].shape)
# print(tensor_1314[...,None,:,:,:].shape)
# print(tensor_1314[...,None,:,:,:,:].shape)
#
#
# a = torch.tensor([0,1,2,3])
# b = torch.tensor([0,1,2,3])
# x,y = torch.meshgrid(a,b)
# print(f"{x}\n{y}")
#
# # 待验证
# # ray_origins    【1024,1,3】
# # sample_z_vals  【1024,64,1】
# # ray_directions 【1024,1,3】
#
# # rays 【1024,64,3】
# # rays = ray_origins[:, None, :] + sample_z_vals[..., None] * ray_directions[:, None, :]
#
#
#
# a = [1,2,3,4,5,6]
# print((a[1:-1]))
#
#
#

import torch
cdf = torch.tensor([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]])
u = torch.tensor([[0.09,0.1,0.11]])
# searchsorted就是返回一个和values一样大小的tensor,
# 其中的元素是在cdf中大于等于u中值的索引
print( torch.searchsorted(cdf, u))
print( torch.searchsorted(cdf, u,right=False))
# 其中的元素是在cdf中大于u中值的索引
print( torch.searchsorted(cdf, u,right=True))
#
#
# t = torch.tensor([[1,2],[3,4]])
# print(f"torch.gather:{torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))}")

# for i in range(0, 640000, 1024):
#     print(i)
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# a = np.array([[[1,2,3],
# [1,2,3],
# [1,2,3],
# [1,2,3]],
# [[1,2,3],
# [1,2,3],
# [1,2,3],
# [1,2,3]],
# [[1,2,3],
# [1,2,3],
# [1,2,3],
# [1,2,3]]])
# print(a.shape)
# b= a[...,::-1]
# print(a[...,::-1], b.shape)
from math import  sqrt

# def dist(x,y,z):
#     return sqrt(x**2 + y**2 +z**2)
# pass
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
# data = np.array([[-1.7981313,-0.8341881,0.50109303],
# [-1.7487019,-0.79550755,0.48897007],
# [-1.6992724,-0.7568269,0.47684714],
# [-1.649843,-0.7181463,0.46472424],
# [-1.6004134,-0.6794658,0.4526013],
# [-1.550984,-0.6407851,0.44047838],
# [-1.5015546,-0.60210454,0.42835546],
# [-1.4521251,-0.563424,0.41623256],
# [-1.4026957,-0.5247433,0.40410963],
# [-1.3532662,-0.48606277,0.3919867],
# [-1.3038368,-0.4473822,0.3798638],
# [-1.2544072,-0.40870154,0.36774087],
# [-1.2049778,-0.370021,0.35561794],
# [-1.1555483,-0.33134043,0.343495],
# [-1.1061189,-0.29265976,0.33137208],
# [-1.0566895,-0.2539792,0.31924915],
# [-1.0072598,-0.21529865,0.30712622],
# [-0.9578302,-0.17661786,0.2950033],
# [-0.9084008,-0.1379373,0.28288037],
# [-0.85897136,-0.099256754,0.27075744],
# [-0.80954194,-0.06057608,0.2586345],
# [-0.7601125,-0.021895409,0.24651158],
# [-0.71068287,0.016785145,0.23438865],
# [-0.66125345,0.0554657,0.22226572],
# [-0.61182404,0.09414625,0.21014279],
# [-0.5623946,0.1328268,0.19801986],
# [-0.5129652,0.17150736,0.185897],
# [-0.46353555,0.21018815,0.17377406],
# [-0.41410613,0.2488687,0.16165113],
# [-0.3646767,0.28754926,0.1495282],
# [-0.3152473,0.3262298,0.13740528],
# [-0.26581788,0.36491036,0.12528235],
# [-0.21638846,0.40359092,0.11315948],
# [-0.16695881,0.4422717,0.10103649],
# [-0.11752963,0.48095202,0.08891362],
# [-0.068099976,0.5196328,0.07679063],
# [-0.018670797,0.55831313,0.06466776],
# [0.030758858,0.5969939,0.052544832],
# [0.080188274,0.6356745,0.040421963],
# [0.12961793,0.67435527,0.028298974],
# [0.17904711,0.7130356,0.016176105],
# [0.22847676,0.7517164,0.004053116],
# [0.27790618,0.7903967,-0.008069754],
# [0.32733583,0.8290775,-0.020192742],
# [0.376765,0.86775804,-0.03231561],
# [0.42619467,0.9064388,-0.04443854],
# [0.47562385,0.94511914,-0.05656147],
# [0.5250535,0.98379993,-0.0686844],
# [0.5744829,1.0224802,-0.08080727],
# [0.6239126,1.061161,-0.09293026],
# [0.673342,1.0998416,-0.10505313],
# [0.7227714,1.1385224,-0.117176056],
# [0.7722008,1.1772027,-0.12929893],
# [0.82163024,1.2158835,-0.14142191],
# [0.87105966,1.2545638,-0.15354478],
# [0.9204891,1.2932446,-0.16566777],
# [0.9699185,1.3319252,-0.17779064],
# [1.0193484,1.370606,-0.18991363],
# [1.0687773,1.4092863,-0.2020365],
# [1.1182072,1.447967,-0.21415949],
# [1.1676366,1.4866474,-0.22628236],
# [1.217066,1.5253282,-0.23840535],
# [1.2664955,1.5640087,-0.25052822],
# [1.3159249,1.6026895,-0.2626512]])
#
# x = data[:, 0]  # [ 0  3  6  9 12 15 18 21]
# y = data[:, 1]  # [ 1  4  7 10 13 16 19 22]
# z = data[:, 2]  # [ 2  5  8 11 14 17 20 23]
#
#

# # 绘制散点图
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x, y, z)
#
# # 添加坐标轴(顺序是Z, Y, X)
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# print("done1!\n")
# plt.show()
# print("done2!\n")

# 散点图
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
#
# # s：marker标记的大小
# # c: 颜色  可为单个，可为序列
# # depthshade: 是否为散点标记着色以呈现深度外观。对 scatter() 的每次调用都将独立执行其深度着色。
# # marker：样式
# ax.scatter(xs=x, ys=y, zs=z,  s=1, c="g", depthshade=True, cmap="jet", marker="^")
#
# plt.show()

# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = Axes3D(fig)
#
# x = np.arange(-4, 4, 0.25)
# y = np.arange(-4, 4, 0.25)
# x, y = np.meshgrid(x, y)
# r = np.sqrt(x ** 2 + y ** 2)
# z = np.sin(r)
#
# ax.plot_surface(x, y, z, rstride=1,  # row 行步长
#                 cstride=2,  # colum 列步长
#                 cmap=plt.cm.hot)  # 渐变颜色
# ax.contourf(x, y, z,
#             zdir='z',  # 使用数据方向
#             offset=-2,  # 填充投影轮廓位置
#             cmap=plt.cm.hot)
# ax.set_zlim(-2, 2)
#
# plt.show()
# print("done")


"""
    绘制3D曲面图
"""
# import numpy as np
# import matplotlib.pyplot as mp
# from mpl_toolkits.mplot3d import Axes3D
#
# # 准备数据
# n = 1000
# x, y = np.meshgrid(np.linspace(-3, 3, n),
#                    np.linspace(-3, 3, n))
#
# z = (1 - x / 2 + x ** 5 + y ** 3) * \
#     np.exp(-x ** 2 - y ** 2)
#
# # 绘制图片
# fig = mp.figure("3D Surface", facecolor="lightgray")
# mp.title("3D Surface", fontsize=18)
#
# # 设置为3D图片类型
# ax3d = Axes3D(fig)
# # ax3d = mp.gca(projection="3d")    # 同样可以实现
#
# ax3d.set_xlabel("X")
# ax3d.set_ylabel("Y")
# ax3d.set_zlabel("Z")
# mp.tick_params(labelsize=10)
#
# ax3d.plot_surface(x, y, z, cstride=20, rstride=20, cmap="jet")
#
# mp.show()
#

# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax3 = plt.axes(projection='3d')
#
# plt.rcParams['font.sans-serif']=['FangSong'] # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
#
# #定义三维数据
# xx = np.arange(-5,5,0.5)
# yy = np.arange(-5,5,0.5)
# X, Y = np.meshgrid(xx, yy)
# Z = np.sin(X)+np.cos(Y)
#
# #作图
# ax3.plot_surface(X,Y,Z,cmap='rainbow')
# # 改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的
#
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import griddata
#
# # 示例数据，随机生成一些三维坐标点
# np.random.seed(0)
# x = np.random.standard_normal(100)
# y = np.random.standard_normal(100)
# z = x**2 + y**2 + np.random.standard_normal(100)*0.1  # 生成一些依赖于x和y的z值
#
# # 创建网格点，用于插值
# grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
#
# # 将散点数据插值到网格上
# grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
#
# # 使用matplotlib创建3D图表
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制曲面图
# surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
#
# # 添​​加颜色条
# fig.colorbar(surf)
#
# # 设置标签
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Z Coordinate')
#
# # 显示图表
# plt.show()