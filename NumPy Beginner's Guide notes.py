#coding:utf-8
"""
作者：zhaoxingfeng	日期：2016.8
说明：看《Python数据分析基础教程：NumPy学习指南（第2版）》所做的笔记
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d

print('----------------Numpy数组---------------------')
a = np.array([np.arange(2),np.arange(2)])
print(a.dtype)                     # dtype -> 获取数据类型：int32
print(a.shape)
print(a[0,1])
b = np.arange(7,dtype = float)
print(b)
print(b.dtype.itemsize)
c = np.arange(9)
print(c[3:7])
print(c[:7:2])                      # 以2为步长选取元素
print(c[::-1])                      # -1 起翻转数组作用
d = np.arange(24).reshape(2,3,4)       # 改变维度
print(d)
print(d[1,2,1])
print(d[...,1])
print(d.ravel())                    # 将数组展平
print(d.flatten())                  # 也可以将数组展平，不同的是 ravel只是返回数组的一个视图，而 flatten会请求分配内存来保存结果
d.shape = (6,4)                     # 用 元组 来设置数组的维度
print(d)
print(d.transpose())                # transpose 转置
print('------------------数组组合-------------------')
a = np.arange(9).reshape(3,3)
b = 2 * a
c = np.hstack((a,b))
print(c)                            # hstack 数组水平组合
c = np.concatenate((a,b),axis = 1)     # concatenate 实现水平组合
print(c)
d = np.vstack((a,b))
print(d)                            # vstack 数组垂直组合
d = np.concatenate((a,b),axis = 0)     # concatenate 实现垂直组合
print(d)
print('------------------数组分割-------------------')
a= np.arange(9).reshape(3,3)
print( np.hsplit(a,3))                 # hsplit实现水平分割
print( np.split(a,3,axis = 1))         # split实现水平分割
print( np.vsplit(a,3))                 # vsplit实现垂直分割
print( np.split(a,3,axis = 0))         # split实现垂直分割

# 数组属性，shape,dtype
a = np.arange(9).reshape(3,3)
print(a.ndim)                       # ndim ->  维度
print(a.size)                       # size ->  数组元素的个数
print(a.T)                          # .T -> 转置
print( a.tolist())                  # tolist() -> 装换成列表 list

print('------------------第三章：常用函数-------------------')
i2 = np.eye(2)
print(i2)
os.chdir('C:\\Users\\user\\Desktop\\Python')              # os.chdir(path) 改变工作目录
print(os.getcwd())                                           # os.getcwd() 获取当前工作目录
np.savetxt('eye.txt',i2,delimiter = ',',fmt = '%g')        # savetxt() -> 保存为txt
c, v = np.loadtxt('data.csv',delimiter = ',',usecols = (6,7), unpack = True)
vwap = np.average(c,weights = v)                             # 计算加权平均价格
print('vwap = ',vwap)
mean = np.mean(c)                                            # 计算平均值
print('mean = ',mean)
a = np.loadtxt('data.csv',delimiter = ',',usecols = (5,))  # 获取单独一列数据时：usecols = (5,)
max,min = np.loadtxt('data.csv',delimiter = ',',usecols = (4,5),unpack = True)  # 载入当日最高价和最低价
print('highet = ',np.max(max))                              # max 最高价
print('lowest = ',np.min(min))                              # min 最低价
print('Spread high price',np.ptp(max))                     # ptp 计算数组的取值范围,即为 极差：max(array) - min(array)
print('Spread low price',np.ptp(min))

print('------------------统计分析：股票收盘价-------------------')
c = np.loadtxt('data.csv',delimiter = ',',usecols = (6,),unpack =True)
print(type(c))
print('median = ',np.median(c))                             # median：求中位数
sorted_c = np.msort(c)                                        # msort -> 对数组进行排序 -> 结果还是数组
print('sorted_c = ',sorted_c)
sorted1_c = sorted(c)                                         # sorted -> 排序 -> 结果是list
print('sorted1_c = ',sorted1_c)
print(type(sorted_c))
print(type(sorted1_c))

N = len(sorted_c)
#print('Middle = ',(sorted_c[(N - 1) / 2] + sorted_c[N / 2])/2)    #求中位数
print('variance = ',np.var(c))                                    # var() -> 求方差
print('variance from definition = ',np.mean((c - c.mean()) ** 2))    # c.mean() -> 求平均值，ndarray对象

print('------------------统计分析：股票收盘价-------------------')
arr = np.loadtxt('data.csv',delimiter = ',',usecols = (-2,))
print(arr)
returns = np.diff(arr) / arr[:-1]                             # diff: 返回一个由相邻数组元素的差值构成的数组,类似于微积分中的微分
print(arr[:-1])
print('standard deviation = ',np.std(returns))
logreturns = np.diff(np.log(c))
positive = np.where(returns > 0)
print('positive = ',positive)
a = np.array([1,2,5,3,9,1,2,5])
print('argmax = ',np.argmax(a))                              # argmax : 求数组中的最大元素的索引值
print('argmin = ',np.argmin(a))                              # argmin： 求数组中的最小元素的索引值
print(np.ones(9))                                              # np.ones() 创建元素为1的数组
# 绘图
x,y = np.loadtxt('data.csv',delimiter = ',',usecols = (5,-2),unpack = True)
#plt.plot(x,y,lw = 1.0)
#plt.show()
x = np.arange(9)
print('exp = ',np.exp(x))                                    # exp：指数
print('linspace = ',np.linspace(0,9,10))                    # linspace(起始值，终止值，可选元素的个数）
print('x_sum = ',x.sum())                                    # sum()

c = np.loadtxt('data.csv',delimiter = ',',usecols = (6,),unpack = True)
print('c = ',c)
b = c[-5:]
b = b[::-1]
print('b = ',b)
A = np.zeros((5,5),float)
for i in range(5):
    A[i,] = c[-6 - i:-1 - i]
print('A = ',A)
import sys
(x,residuals,rank,s) = np.linalg.lstsq(A,b)                    # 最小二乘
print ('x = ',x)
print('预测值 = ',np.dot(b[::-1],x))
def Mylinalg():                                                # 最小二乘，举例
    plt.figure(figsize = (10,6))
    x = np.array([0,1,2,3])
    y = np.array([-1,0.2,0.9,2.1])
    A = np.vstack([x,np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A,y)[0]
    plt.plot(x,y,'o',label = 'Original data',markersize = 10)
    plt.plot(x,m*x+c,'r',label = 'Fitted line')
    plt.legend()
    plt.show()

print('------------------统计分析：股票收盘价-------------------')
print('修剪：Clipped = ',np.arange(9).clip(2,5))              # clipe（min,max) -> 把比给定值还小的元素设为给定值，把比给定值还大的元素设为给定值
a = np.arange(9)
print('筛选：Compress = ',a.compress(a > 5))                  # compress -> 返回一个根据给定条件筛选后的数组

print('------------------第四章：便捷函数-------------------')
a,b = np.loadtxt('data.csv',delimiter = ',',usecols = (3,4),unpack = True)
covariance = np.cov(a,b)
print('协方差：covariance = \n',covariance)
print('对角线元素：diagonal = \n',covariance.diagonal())
print('迹：trace = ',covariance.trace())
print('相关系数定义：coefficient = \n',covariance / (a.std() * b.std()))
print('相关系数： cofficient = \n',np.corrcoef(a,b))
# 多项式拟合
bhp = np.loadtxt('BHP.csv',delimiter = ',',usecols = (6,),unpack = True)
vale = np.loadtxt('VALE.csv',delimiter = ',',usecols = (6,),unpack = True)
t = np.arange(len(bhp))
poly = np.polyfit(t,bhp - vale,5)
print('fit coefficient = ',poly)
print('下一个值预测 = ',np.polyval(poly,t[-1]+1))             # polyval -> 求值

der = np.polyder(poly)
print('多项式求导 = ',der)
print('导函数的根即为极值点 = ',np.roots(der))

def zhao():
    plt.plot(t,bhp - vale)                              # 原始数据，锯齿状
    vals = np.polyval(poly,t)
    plt.plot(t,vals)                                    # 生成的拟合后的光滑折线图
    plt.show()

x = np.roots((1,0,1))                                   # 求多项式函数的根
print(x)
y = np.hstack((x,0,12,0,1+1j,5))
print(y)
RealNum = np.isreal(y)                                  # isreal -> 判断元素是否为实数
print('Real number ?',RealNum)
xpoints = np.select([RealNum],[y])                      # select -> 选择实数
xpoints = xpoints.real
print('Real number: ',xpoints)
print('Sans 0s: ',np.trim_zeros(xpoints))             # trim_zeros -> 去掉一维数组中开头和末尾为 0 的元素

print('------------------第五章：矩阵和通用函数-------------------')
a = np.mat('1 2 3;4 5 6;7 8 9')                            # 创建矩阵
print(a)
print('矩阵转置:transpose a = ',a.T)                # .T -> 矩阵转置
print('逆矩阵：inverse a = ',a.I)                   # .I -> 矩阵求逆,时间复杂度为 o(n3),广义逆矩阵（不一定是方阵）
print('用数组创建矩阵：creation from array = ',np.mat(np.arange(9).reshape(3,3)))
a = np.eye(2)
b = 2 * a
c = np.bmat('a b;a b;a b')                           # bmat -> block matrix 利用已有的较小的矩阵创建一个新的大矩阵
print(c)
# 定义一个回答宇宙、生命及万物的终极问题的python函数， 使用 frompyfunc 函数
def utimate_answer(a):
    result = np.zeros_like(a)
    result.flat = 42
    return result
ufunc = np.frompyfunc(utimate_answer,1,1)               # fromfunc -> 指定输入参数的个数为1，随后的1为输出参数的个数
print('The answer = ',ufunc(np.arange(4)))
print('The answer = ',ufunc(np.arange(4).reshape(2,2)))

# 数组的除法运算: divide, true_divide, floor_division
a = np.array([2,6,5])
b = np.array([1,2,3])
print('Divide = ',np.divide(a,b),np.divide(b,a))     # divide -> 运算结果的小数部分被截断
print('True Divide = ',np.true_divide(a,b),np.true_divide(b,a))     # true_divide -> 相当于数学中的除法，返回浮点数结果而不做截断
# 如果在python 开头 -> from __future__ import division -> 则改为调用 true_divide 函数

# 计算模数或余数：mod,remainder,%,fmod -> fmod处理负数与其他三个不同
a = np.arange(-4,4)
print('Remainder = ',np.remainder(a,2))             # remainder() -> 逐个返回两个数组中元素相除后的余数，如果第二个数字为 0 ，则直接返回 0
print('Mod = ',np.mod(a,2))                          # mod() -> 功能和 remainder 完全一样
print('% = ',a % 2)                                   # % -> 仅仅是remainder 的简写
print('Fmod = ',np.fmod(a,2))                        # fomod -> 所得余数的正负号由被除数决定，与除数的正负号无关
print('Rint = ',np.rint(9.490005))                   # rint -> 对浮点数取整 ，四舍五入，但不改变浮点数类型,结果还是浮点数

# 绘制利萨如曲线: x = A sin(at+n/2)   y = B sin(bt)
def lissajius():
    a = 9
    b = 8
    t = np.linspace(-np.pi,np.pi,201)
    x = 2 * np.sin(a * t + np.pi/2)
    y = 3 * np.sin(b * t)
    plt.plot(x,y)
    plt.show()
# lissajius()

# 绘制方波：方波可以近似表示为多个正弦波的叠加，任何一个方波信号都可以用无穷傅立叶级数来表示
def squarewave(n):
    t = np.linspace(-np.pi,np.pi,201)
    k = np.arange(1,n)
    k = 2 * k - 1
    f = np.zeros_like(t)
    for i in range(len(t)):
        f[i] = np.sum(np.sin(k * t[i]) / k)
        f[i] = f[i] * 4 / np.pi
    plt.plot(t,f)
    plt.show()
#squarewave(100)

print('------------------第六章：深入学习 numpy 模块-------------------')
a = np.mat('1 2 3;2 2 1;3 4 3')
det = np.linalg.det(a)                                    # det -> 求行列式值
print('det = ',det)
inverse = np.linalg.inv(a)                                # inv -> 计算逆矩阵，如果矩阵是奇异或非方阵，则会抛出 LinAlgError 异常
print('inverse of a = ',np.linalg.inv(a))              # 可逆矩阵一定是方阵
print(a * inverse)

# 求解线性方程组
a = np.mat('2 3 11 5;1 1 5 2;2 1 3 2;1 1 3 3')
b = np.mat('2;1;-3;-3')
x = np.linalg.solve(a,b)                                 # solve -> 解线性方程组
print(x)
print(np.dot(a,x))

# 求解特征值和特征向量
a = np.mat('3 -2;1 0')
eigvals = np.linalg.eigvals(a)                          # eigvals -> 求解特征值
print('特征值：eigvals = ',eigvals)
eigvals,eigvector = np.linalg.eig(a)                    # eig -> 求解特征值和特征向量，返回一个元组，第一列为特征值，第二列为特征向量
print('eigvals = ',eigvals)
print('eigvector = ',eigvector)
for i in range(len(eigvals)):                          # 验证结果
    print('Left = ',np.dot(a,eigvector[:,i]))
    print('Right = ',eigvals[i] * eigvector[:,i])

# 奇异值分解 SVD，返回三个参数： U,Sigma，V，其中U和V是正交矩阵（U*U转置 = 单位矩阵），Sigma包含输入矩阵的奇异值(对角阵的对角线元素）
a = np.mat('4 11 14;8 7 -2')
U,Sigma,V = np.linalg.svd(a,full_matrices = False)     # svd -> 奇异值分解
print('U = ',U)
print('Sigma = ',Sigma)
print('V = ',V)
print('a = ',U * np.diag(Sigma) * V)                   # diag -> 知道对角线元素，化成矩阵

# 计算广义逆矩阵
a = np.mat('4 11 14;8 7 -2')
pinv = np.linalg.pinv(a)                                # pinv -> 广义逆矩阵
print('pinv = ',pinv)
print(' I = ',a.I)
print('eye = ',a * pinv)                               # 验证：相乘为单位阵

print('------------------随机数-------------------')
# 二项分布： binomial
# 假设你来到赌场，每一轮抛9枚硬币，如果至少5枚正面朝上则赢一份赌注，否则输掉一份赌注
# 初始资本为 1000，玩 10000 轮
cash = np.zeros(10000)                                   # 玩 10000 轮
cash[0] = 1000                                           # 初始资金
outcome = np.random.binomial(9, 0.5, size = len(cash))   # binomial -> 二项分布
for i in range(1, len(cash)):
    if outcome[i] < 5:
        cash[i] = cash[i-1] - 1
    elif outcome[i] < 10:
        cash[i] = cash[i-1] + 1
    else:
        raise AssertionError('Unexpected outcom ' + outcome)
print(outcome.min(),outcome.max())
def erxiangfenbu():
    plt.plot(np.arange(len(cash)),cash)
    plt.show()
#erxiangfenbu()

# 超几何分布: hypergeometric
# 袋子里有一个‘倒霉球‘，有25个正常球；摸到‘倒霉球’扣6分，摸到正常球加一分，每次抽3个球；总共抽100次
points = np.zeros(100)
outcomes = np.random.hypergeometric(25, 1, 3, size = len(points))               # hypergeometric: 第一个参数是普通球数量，第二个参数是倒霉球数量，第三个参数是每次采样的数量
for i in range(len(points)):
    if outcomes[i] == 3:
        points[i] = points[i-1] + 1
    elif outcomes[i] == 2:
        points[i] = points[i-1] - 6
    else:
        print(outcomes)
def chaojihefenbu():
    plt.plot(np.arange(len(points)),points)
    plt.show()
#chaojihefenbu()

# 正态分布
N = 10000
normal_values = np.random.normal(size = N)
def zhengtaifenbu():
    dummy,bins,dummy = plt.hist(normal_values, 100, normed = True, facecolor = 'green',lw = 1,alpha = 0.65)      # 返回值中 bins 最重要,normed 默认为1
    mu = 0
    sigma = 1
    plt.plot(bins,1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu) ** 2 / (2 * sigma ** 2)),color = 'red',lw = 2)
    plt.show()
#zhengtaifenbu()

np.random.random(10)                                      # random.random -> 产生 (0, 1) 的随机数
# 去掉文件隐藏属性：attrib 文件名-s -h -r

# 搜索
a = np.array([2,4,8])
print(np.argmax(a))                                       # argmax -> 返回数组中最大值对应的下标
a = np.array([np.nan,2,4,8])
print(np.nanargmax(a))                                    # nanargmax -> 和 argmax 功能一样，但忽略 NaN 值
                                                          # argmin, nanargmin -> 返回最小值下标
a = np.array([0,2,40,8,10,15,55])
print('where = ',np.argwhere(a < 20))                   # argwhere -> 根据条件搜索非零元素，并返回对应的下标
a = np.mat('1 2 3;2 8 5;2 6 4')
index = np.argwhere(a <= 5)
print('index = ',index[:3])
print('where = ',np.argwhere(a <= 5))

a = np.arange(5)
index = np.searchsorted(a, [-2,7])                       # searchsorted -> 为指定的插入值寻找可以维持数组排序的索引位置
print('index = ',index)
b = np.insert(a, index, [-2,7])                          # insert -> 构建完整的数组
print('now = ',b)

a = np.arange(40)
print('extract = ',np.extract(a < 20,a))               # extract -> 返回满足指定条件的数组元素
condition = (a % 2) == 0
print('Even num = ',np.extract(condition,a))           # extract
print('non zero = ',np.nonzero(a))                     # nonzero -> 抽取数组中的非零元素

print('------------------第八章：质量控制-------------------')
# 断言近似相等：assert_almost_equal， assert_approx_equal, assert_array_equal, assert_allclose
print('dicimal 7 = ',np.testing.assert_almost_equal(0.12345678,0.123456789,decimal=7))      # 这里没有抛出异常，小数点后第七位相同，精度
#print('decimal 8 = ',np.testing.assert_almost_equal(0.12345678,0.123456786,decimal = 8))      # 会抛出异常,0.12345678-(0~5) 则不会抛出异常

print('significance 8 = ',np.testing.assert_approx_equal(0.123456789,0.123456780,significant = 8))      # 这里没有抛出异常
#print('significance 9 = ',np.testing.assert_approx_equal(0.123456789,0.123456780,significant = 9))         # 会抛出异常，第9位不同

# 断言数组近似相等：首先检查两个数组形状是否一致，然后逐一比较两个数组中的元素
print('dicimal 8 = ',np.testing.assert_array_almost_equal([0,1,0.123456789],[0,1,0.123456780],decimal = 8))         # 不会抛出异常
#print('dicimal 9 = ',np.testing.assert_array_almost_equal([0,1,0.123456789],[0,1,0.123456780],decimal = 9))          # 会抛出异常

#print('fail = ',np.testing.assert_array_equal([0,0.123456789,np.nan],[0,0.123456780,np.nan]))           # 数组相等必须形状和元素严格相等，允许数组中存在 NaN 元素，此处不相等，抛出异常
print('pass = ',np.testing.assert_allclose([0,0.123456789,np.nan],[0,0.123456780,np.nan],rtol = 1e-7,atol = 0))
                                                                                              # atol = absolute tolerance(绝对容差),rtol = relative tolerance(相对容差)
                                                                                              # |a - b| <= (atol + rtol * |b|)
print('pass = ',np.testing.assert_array_less([0,0.12345670,np.nan],[1,0.123456789,np.nan]))     # 检查一个数组是否严格大于另一个数组,形状一致 + 第一个数组元素严格小于第二个数组元素
# print('fail = ',np.testing.assert_array_less([1,0.12345670,np.nan],[1,0.123456789,np.nan]))     # 检查一个数组是否严格大于另一个数组

# print('equal ?',np.testing.assert_equal((1,2),(1,3)))      # 如果两个对象不相同，抛出异常，这里的对象不一定是 numpy 对象，可以使 python 中的列表、元组、字典
                                                            # 这里比较两个元组
# print('equal ?',np.testing.assert_equal([1,2],[1,3]))       # 会抛出异常
print('pass = ',np.testing.assert_string_equal('Numpy','Numpy'))                    # 比较字符串，区分大小写
# print('fail = ',np.testing.assert_string_equal('Numpy','numpy'))

print('------------------第九章：使用 matplotlib 绘图-------------------')
def Plot1():                                                        # 直方图
    data = np.random.normal(5.0, 3.0, 1000)
    plt.hist(data,10)
    plt.xlabel('data')
    plt.show()
# Plot1()
def Plot2():                                                        # 散点图
    x = np.array([np.random.normal(5.0, 3.0, 1000),np.random.normal(2.0, 5.0, 1000)])
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.scatter(x[0],x[1],c = 'red')
    plt.show()
# Plot2()
def Plot3():                                                        # 散点图
    x = np.array([np.random.normal(5.0, 3.0, 1000),np.random.normal(2.0, 5.0, 1000)])
    plt.figure(figsize = (6,6)).add_subplot(211)
    plt.grid(True)                                                  # 网格线
    plt.scatter(x[0],x[1],s = 30,c = 'red',marker='^')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(-10,20)
    plt.title('Scatter')
    plt.legend('sin')
    plt.show()
# Plot3()
def Plot4():
    plt.figure(figsize = (10,6))
    x = np.array([0,1,2,3])
    y = np.array([-1,0.2,0.9,2.1])
    A = np.vstack([x,np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A,y)[0]
    plt.plot(x,y,'o',label = 'Original data',markersize = 10)
    plt.plot(x,m*x+c,'r',label = 'Fitted line')
    plt.legend()
    plt.fill_between(x,y.min(),y,where = y < y.mean(),facecolor = 'green',alpha = 0.4)
    plt.fill_between(x,y.min(),y,where = y> y.mean(),facecolor = 'yellow',alpha = 0.6)
    plt.show()
# Plot4()
def Plot5():                                                         # 绘制箭头等，全
    plt.figure(figsize = (10,6))
    x = np.arange(-100,100)
    z = np.random.uniform(0,50,200)                                  # uniform -> 随机生成一个实数
    y = x ** 2 + 10
    m = x ** 2 - 6 * x +500
    plt.plot(x,y,label = 'y = x ** 2 + 10',markersize = 5)
    plt.legend()
    plt.annotate('y = x ** 2 + 10',fontsize = 20,xy = (50,2700),xytext = (0,10000),\
                 arrowprops = dict(arrowstyle = '->',facecolor = 'blue',connectionstyle = 'arc3',linestyle = 'solid'))
    plt.fill_between(x,y.min(),y,where = y < y.mean(),facecolor = 'green',alpha = 1)
    plt.fill_between(x,y.min(),y,where = y> y.mean(),facecolor = 'yellow',alpha = 0.6)
    plt.plot(x,m,color = 'red',label = 'm = x**2 - 6*x + 500',markersize = '2')
    plt.grid(True)
    plt.annotate('y = x ** 2 + 10',fontsize = 16,xy = (-50,2500),xytext = (0,6000),\
                 arrowprops=dict(arrowstyle='->',facecolor='red',connectionstyle='arc3,rad=.5',linestyle='dashed'))
                                                                      # 作箭头，参数xy为箭头坐标，xytext为箭尾坐标
                                                                      #  linestyle : solid/dashed/dotted/
    plt.legend()
    plt.show()
def Plot3d():                                                        # 3D 绘图
    x,y = np.mgrid[-100:100:1,-60:60:8]
    z = x ** 2 + y ** 2

    ax = plt.subplot(111,projection = '3d')
    ax.plot_surface(x,y,z,rstride = 2,cstride = 1,cmap = cm.coolwarm,alpha = 0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('z = x ** 2 + y ** 2')
    plt.show()

def Plot3dz():                                                      # 3D 绘图
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    u = np.linspace(-1,1,100)
    x, y = np.meshgrid(u,u)
    z = x**2 + y**2
    ax.plot_surface(x, y, z,cmap=cm.YlGnBu_r)
    ax.contourf(x,y,z)
    plt.show()
def PlotContour():                                                      # 绘制等高线
    fig = plt.figure()
    ax = fig.add_subplot(111)

    u = np.linspace(-1,1,100)
    x, y = np.meshgrid(u,u)
    x,y = np.mgrid[-100:100:1,-60:10:8]
    z = x**2 + y**2

    ax.contourf(x,y,z)
    plt.show()
# PlotContour()

print('------------------第十章： Numpy 扩展： Scipy-------------------')
from scipy import io
a = np.arange(9)
io.savemat('a.mat',{'Myarray':a})                       # savemat -> 保存为 mat 文件
from scipy import stats
generated = stats.norm.rvs(size = 900)                     # 使用 scipy.stats 包按正态分布生成随机数
print('Mean and Std : ',stats.norm.fit(generated))      # 均值和标准差
print(generated.mean(),   generated.std())
print(stats.skewtest(generated))                           # skewtest -> p-value，观察到的数据集服从正态分布的概率，取值范围 0~1
print(stats.normaltest(generated))                         # normaltest -> 检查数据集服从正态分布的程度
print('95% percentile = ',stats.scoreatpercentile(generated, 95))           # 95% 处的数值
print('percentile at 1 = ',stats.percentileofscore(generated,1))            # 从数值1出发找到对应的百分比
# plt.hist(generated);plt.show()
a,b = np.loadtxt('data.csv',delimiter = ',',usecols = (3,4),unpack = True)
print(stats.ttest_ind(a,b))                                 # stats.ttest_int() -> 均值检验，检查两组不同的样本是否有相同的均值，返回值的第二个为 p_value ，有多大的概率均值相同
print(stats.ks_2samp(a,b))                                  # stats.ks_2samp() -> 判断两组样本同分布的可能性

def chazhi():                                                   # 插值函数，线性插值和三次插值
    from scipy import interpolate
    x = np.linspace(-18,18,36)
    noise = 0.1 * np.random.random(len(x))
    signal = np.sinc(x) + noise                                  # sinc = sin(pi * x) / (pi * x)
    interpreted = interpolate.interp1d(x,signal)                 # 创建线性插值函数

    x1 = np.linspace(-18,18,180)
    y1 = interpreted(x1)
    cubic = interpolate.interp1d(x,signal,kind = 3)              # 三次插值
    y2 = cubic(x1)
    plt.plot(x,signal,'o',label = 'data')
    plt.plot(x1,y1,'-',label = 'liner')
    plt.plot(x1,y2,'-',lw = 2,label = 'cubic')
    plt.legend()
    plt.show()
# chazhi()
# 使用 scipy.io.wavfile.read() 将 wav 文件转换为一个 numpy 数组，使用 wavfile.write() 写入一个新的wav文件，使用tile函数重复播放音频片段
def Mywav():
    from scipy.io import wavfile
    WAV_FILE = 'match4.wav'
    sample_rate, data = wavfile.read(WAV_FILE)                       # 返回采样率和音频数据
    print(type(data))
    print('Data type',data.dtype,'Shape',data.shape)
    plt.subplot(211)
    plt.title('Original')
    plt.plot(data)

    plt.subplot(212)
    repeated = np.tile(data,5)                                       # 重复音频片段
    plt.title('Repeated')
    plt.plot(repeated)
    wavfile.write('repeated_air.wav',sample_rate,repeated)        # 绘制音频数据
    plt.show()
# Mywav()

# 书名《Python程序设计》 董付国，清华大学出版社
print('------------------字典-------------------')
keys = ['a','b','c','d']
values = [1,2,3,4]
dictionary = dict(zip(keys,values))
print(dictionary)
d = dict(name = 'Dong',age = 37)
print (d)
print(d.get('age'))	#get()方法可以获取指定键对应的值，并且在指定键不存在的时候返回制定值，如果不指定则迷人返回None
print(d.get('zhao','no zhao'))
for item in d.items():
	print (item)
for key in d:
	print (key)
for key,value in d.items():
	print(key,value)
print(d.keys())
print(d.values())
d['age'] = 38		#当以指定键为下标为字典元素赋值时，若该键存在，则表示修改该键的值，如果不存在，则表示添加一个新的键-值对，也就是一个新元素
d['address'] = 'SDIBT'

s = 0
for i in range(1,101):
	s += i
else:				#如果循环因为条件表达式不成立而自然结束，则执行else结构中的语句。如果是因为执行了break语句而提前结束则不执行else
	print(i)
# 9*9 乘法表
for i in range(1,10):
    for j in range(1,i+1):
        print j,'*',i,'=',i*j,'\t',
    print '\n'
s1 = "apple,peach,banana,pear"
s1_s = s1.split(',')
print(s1_s)
s2 = "2016-09-19"
s2_s = s2.split('-')
print(s2_s)
#可变长度参数
def demo(*p):		# *p -> 无论调用该函数时传递了多少实参，一律将其放入元组中
	print(p)
demo(1,2,3)
def demo(**p):		# **p -> 无论调用该函数时传递了多少实参，一律将其放入字典中
	for item in p.items():
		print(item)
demo(x = 1,y = 2,z = 3)
#绘制散点图
a = np.arange(0,2 * np.pi,0.1)
b = np.cos(a)
plt.scatter(a,b)
# plt.show()

print('------------------杂记-------------------')
from sklearn import  cross_validation
x = np.array([int(i) for i in range(0,10)])
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, x, test_size=0.4, random_state=None)#交叉验证

class Duck(object):
      def eat(self):#类实例方法,第一个参数为self
          print("I like eat fish")
      #@staticmethod是函数修饰符，表示接下来是一个静态方法，静态方法不需要定义实例即可使用。另外，多个实例共享此静态方法
      @staticmethod
      def eat1():
          print("I like eat samllfish")
      @classmethod#类方法,第一个参数为cls
      def eat2(cls):
          print("I like eat bigfish")
#类实例方法仅可以被类实例调用
duck = Duck()
duck.eat()
#类方法和静态方法都可以被类和类实例调用，这两种方法都可以直接调用而不用创建实例
#区别在于：类方法的隐含调用参数是类，而静态方法没有隐含调用参数；类实例方法的隐含调用参数是类的实例
#静态方法相当于是个全局的方法，整个程序里面都起作用，
duck1 = Duck()
duck1.eat1()
Duck.eat1()
duck2 = Duck()
duck2.eat2()
Duck.eat2()
