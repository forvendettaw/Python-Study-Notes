#coding:utf-8
"""
作者：zhaoxingfeng	日期：2016.8
说明：看《小甲鱼零基础入门学习Python》所做的笔记
"""
member = ['赵','钱','孙']  #数组，列表
print(member)
member.append('李')         #append
print(member)
member.extend(['周','武']) #extend
print(len(member))
print(member)
member.insert(1,'拱健婷')  #insert，从0开始
print(member)
member.remove('赵')        #remove
print(member)
del member[0]              #del
print(member)
member.pop()               #pop
print(member)
member.pop(0)
print(member)
member1 = member[1:3]
print(member1)
member2 = member[:2]
print(member2)

print('-------------------------------------')

list1 = [123, 456]
list2 = [234, 789]
list3 = list1 + list2
print(list3)
list4 = [1,2,[3,4],5,4,3,2,1]
print(list4[2][1])      #二维数组
print(list4.count(1))
print(list4.index(1))
list4.reverse()
print(list4)

print('-------------------------------------')#列表

list5 = [5,6,2,8,11,9,55,88,7]
list5.sort(reverse=True)  #True，False
print(list5)
list11 = list5[:]          #这样复制才是对的,分片拷贝
list22 = list5             #不正确，会跟着改变，只他的标签是多了一个指向
list5.sort()
print(list11)
print(list22)

print('-------------------------------------')#元组，不能改变

tumple1 = (1,2,3,4,5,6,7,8,9)#元组用小括号,小括号不是必须的，逗号是必须的
print(tumple1[5:])
tumple1 = tumple1[:2] + ('45',) + tumple1[2:]      #拼接
print(tumple1)
print(2 * tumple1)

print('-------------------------------------')#字符串

str1 = 'i love fishC'
print(str1[3])
print(str1[:6]+'插入'+str1[6:])    #插入字符串
print(str1.capitalize())            #大写
print(str1.lower())              #小写
print(str1.center(10))
print(str1.find('ff'))             #检测是否包含字符串，是则返回1，否则返回-1
print(str1.find('love'))
print('-------------------------------------')#格式化字符串
print("{0} love {1}.{2}".format("I","fishC","com"))#｛｝是位置参数
print("{a} love {b}.{c}".format(a="I",b="fishC",c="com"))#｛a｝是关键字参数
print("{0} love {b}.{c}".format("I",b="fishC",c="com"))#｛｝位置参数应该位于关键字参数之前，否则会报错
print('%c %c %c' % (97,98,99))
print('%d + %d = %d' % (4,5.3,9))   # %c:格式化字符及其ASC码；%s：格式化字符串；
                                      # %x：格式化无符号16进制；%f：格式化定点数，默认6位小数
                                      # %e：用定点数科学计数法格式化；%g：根据值的大小决定使用%f或%e
print('%e'% 566.3333)
print('%f' % 27.658)
print('%e' % 27.658)
print('%9.5g' % 27.888888)
print('%9.1f' % 27.658)             # m.n：m表示总共占多少位，前面补0；n表示精确位数

print('-------------------------------------')

a = ' I love fishC.com'
print(list(a))                       #list通过循环将字符串转化为列表
c = (1,1,2,3,5,8,13,21.036,34)
print(list(c))                       #list通过循环将元组转化为列表
a = ' I love fishC.com'
print(tuple(a))                      #tuple通过循环将字符串转化为元组
print(len(a))                        #len()返回长度
print(max(a))                        #max()返回最大值
numbers = [1,18,13,0,-98,36,85.3699,25e-9]
print(max(numbers))
print(sum(numbers))
print(sorted(numbers))               #从小到大排序
print(list(reversed(numbers)))       #左右互换
print(list(enumerate(numbers)))      #第一个是元素位置，第一个是元素值
a = [1,2,3,4,5,6]
b = [4,5,6,7]
print(list(zip(a,b)))                #zip打包，返回由各个参数的序列组成的元组

print('-------------------------------------')

def MyFirstFunction():              # 函数用法
    print('这是我创建的第一个函数！')
MyFirstFunction()
def MySecondFunction(name):         # name为形参
    '这里是函数文档,和注释不一样'
    print(name + ' LOVE 拱健婷')
MySecondFunction('赵兴锋')          # 传递进来的“赵兴锋”为实参
print(MySecondFunction.__doc__)      # 默认属性
def add(num1,num2):
    return  (num1 + num2)           # return用法
print(add(1, 5))
def SaySome(name,words):
    print(name + '->' + words)
SaySome('小甲鱼','让编程改变世界')
SaySome(words='让编程改变世界',name='小甲鱼')
def SaySome(name='小甲鱼',words='让编程改变世界'):# 默认参数，如果调用时忘了传递参数，可以找到初值
    print(name + '->' + words)
SaySome('苍井空')
SaySome('苍井空','是日本人')
def test(*params):                    # 收集参数，参数长度可变
    print('参数的长度是:',len(params))
    print('第二个参数是:',params[1])
test(1,'小甲鱼',3.14,5,6,7,8)
# def test(*params, exp = 9):                    # 收集参数，参数长度可变
#     print('参数的长度是:',len(params),exp)
#     print('第二个参数是:',params[1])
# test(1,'小甲鱼',3.14,5,6,7,8)

print('-------------------------------------')

"""
def discounts(price, rate):                    # price, rate, final_price都是局部变量
    final_price = price * rate
    return final_price
old_price = float(input('请输入原价：'))
rate = float(input('请输入折扣:'))
new_price = discounts(old_price, rate)
print('打折后的价格是:',new_price)
#print('这里试图打印局部变量final_price的值:',final_price)
"""
def fun1():                                     # 内嵌函数
    print('fun1()正在调用')
    def fun2():
        print('fun2()正在调用')
    fun2()
fun1()
#fun2()
def FunX(x):
    def FunY(y):
        print(y)
        return x * y
    return FunY
print( FunX(8)(5) )

print('-------------------------------------')

def ds(x):
    return 2 * x + 1
print(ds(5))
g = lambda x : 2 * x + 1                  # lambda函数,x是参数，后面是返回值，不需要专门定义一个函数
                                           # 可以使代码更加精简
print(g(8))
gg = lambda x,y : x + y
print(gg(8,90))
print(list(filter(None, [1, 0, False, True, 5])))   # filter用法
def odd(x):
    return x % 2
temp = range(10)
#print(list(temp))
show = filter(odd,temp)
print(list(show))
print(list(filter(lambda x : x % 2, range(10))))         # filter
print(list(map(lambda x: x * 2, range(8))))              # map:将值带入，求出若干结果

print('-------------------------------------')

# 递归：汉诺塔，树，谢尔宾斯基三角形
"""
def factorial(n):                                        # 非递归求阶乘
    result = n
    for i in range(1, n):
        result *= i
    return result
number = int(input('请输入一个正整数：'))
print('%d 的阶乘是：%d' % (number,factorial(number)))     # 递归求阶乘,其实没有必要用递归求，每次调用自己
                                                         # 都会压栈，弹栈，保存，恢复寄存器的栈操作，非常消耗时间和空间
def factorial(n):
    if n == 1:
        return 1                                         # 必须有一个终止条件
    else:
        return n * factorial(n-1)                        # 必须调用自身
number = int(input('请输入一个正整数：'))
print('%d 的阶乘是：%e' % (number,factorial(number)))
"""

print('-------------------------------------')

# 斐波那契数列：1，1，2，3，5，8，13，21，34，55，89，144，越往后两项之比越接近黄金分割比
"""
def rabbit1(n):
    '用迭代实现斐波那契数列'
    n1 = 1
    n2 = 1
    n3 = 1
    while(n-2) > 0:
        n3 = n2 + n1
        n1 = n2
        n2 = n3
        n -= 1
    return n3
n = int(input('请输入月份数：'))
print('%d 月后兔子总数是：%d' % (n, rabbit1(n)))

def rabbit2(n):
    '用递归实现斐波那契数列,效率非常低：试一下35，100'
    if n == 1 or n == 2:
        return  1
    else:
        return  rabbit2(n-1) + rabbit2(n - 2)
n = int(input('请输入月份数：'))
print('%d 月后兔子总数是：%d' % (n, rabbit2(n)))
"""
"""
def hanoi(n, x, y, z):
    if n == 1:
        print(x, '-->', z)
    else:
        hanoi(n-1, x, z, y)              # 将前n-1个盘子从x移动到y上
        print(x,'-->',z)                 # 将最底下的最后一个盘子从x移动到z上
        hanoi(n-1, y, x, z)              # 将y上的n-1个盘子移动到z上
n = int(input('请输入汉诺塔的层数：'))
hanoi(n, 'x', 'y', 'z')
"""

print('-------------------------------------')

brand = ['李宁','耐克','阿迪达斯','咸鱼']    #[]代表列表
slogan = ['一切皆有可能','just do it','impossible is nothing','咸鱼改变世界']
print('咸鱼的口号是：',slogan[brand.index('咸鱼')])
dict1 = {'李宁':'一切皆有可能','耐克':'just do it','阿迪达斯':'impossible is nothing','咸鱼':'咸鱼改变世界'}    #{}代表字典，映射类型
print(dict1)
print('李宁的口号是：',dict1['李宁'])         #李宁是键key，一切皆有可能是值
dict1['爱迪生'] = '天才是99%的汗水+1%的灵感'
print(dict1)
dict2 = {}
dict2 = dict2.fromkeys((1,2,3),'number')       #创建并返回一个新的字典,第一个参数不能为空，第二个可为空
print(dict2)

dict2 = dict2.fromkeys(range(5),'赞')
print(dict2)
for eachKey in dict2.keys():
    print(eachKey)
for eachValue in dict2.values():
    print(eachValue)
for eachItem in dict2.items():
    print(eachItem)
print(dict2.get(2))                             # get,如果没有则不返回不报错
print(dict2[2])
print(6 in dict2)                               #成员资格
dict2.clear()                                   # clear,不要用dict1 = {}
print(dict2)
a = {1:'one',2:'two',3:'three'}
b = a.copy                                      # 前拷贝，对对象表层的拷贝
c = a                                           # 贴了一个不同的标签
print(id(a),id(b),id(c))
a[9] = 'eight'
print(a)
a.pop(3)                                        # pop
print(a)
b = {'小白':'一只狗'}
a.update(b)                                     # 按照b来更新a
print(a)
num2 = {1,2,3,4,5,4,3,2,1}                      # 集合
num3 = set([1,2,3,4,3])                         # set
print(type(num2))
print(type(num3))
print(num2)
print(num3)

num3 = [1,2,3,4,5,3,5,1,0]                      # 去掉列表里边重复的数字
temp = []
for each in num3:
    if each not in temp:
        temp.append(each)                       # 追加append
print(temp)
num3 = list(set(num3))                          # set()->集合,无序，list->列表
print(num3)
#访问集合中的值
num3 = set([1,2,3,4,5,3,5,1,0])
print(type(num3))
num3.add(6)                                     # set, add
print(num3)
num3.remove(5)
print(num3)
#不可变集合
num3 = frozenset([1,2,3])

print('-------------------------------------')

f = open('C:\\Users\\user\\Desktop\\Python\\a1.txt','r')   # 打开文件
#print(f.read())                                                # 读取文件,读取之后文件内容为空
print(f.tell())                                                 # 返回当前在文件中的位置
#print(list(f))                                                 # 直接将文件内容转化为列表
lines = list(f)
for each_line in lines:
    print(each_line)                                            # 一行一行打印出来
f = open('C:\\Users\\user\\Desktop\\Python\\a2.txt','w')   # 打开文件
print(f.write('i love gong'))                                 # 文件写入
f.close()

print('-------------------------------------')

# 实现功能：将小甲鱼和小客服的 三段对话 分别保存为三个文件，每个人的对话单独保存
def save_file(boy,girl,count):                                      # 封装 保存文件函数
    file_name_boy = 'boy_' + str(count) + '.txt'
    file_name_girl = 'girl_' + str(count) + '.txt'

    boy_file = open(file_name_boy,'w')
    girl_file = open(file_name_girl,'w')

    boy_file.writelines(boy)                                        # wirtelines：向文件写入字符串序列seq，seq应该是一个返回字符串的可迭代对象
    girl_file.writelines(girl)

    boy_file.close()
    girl_file.close()
def split_file(file_name):                                          # 封装 分割文件，传入 文件名
    f = open('a3.txt')                                             # 打开文件

    boy = []
    girl = []
    count = 1

    for each_line in f:
        if each_line[:6] != '======':                              #这里进行字符串分割
            (role, line_spoken) = each_line.split('：', 1)          #这里进行字符串分割
            if role == '小甲鱼':
                boy.append(line_spoken)
            if role == '小客服':
                girl.append(line_spoken)
        else:                                                       # 文件的分别保存
            """file_name_boy = 'boy_' + str(count) + '.txt'
            file_name_girl = 'girl_' + str(count) + '.txt'

            boy_file = open(file_name_boy,'w')
            girl_file = open(file_name_girl,'w')

            boy_file.writelines(boy)                       # wirtelines：向文件写入字符串序列seq，seq应该是一个返回字符串的可迭代对象
            girl_file.writelines(girl)

            boy_file.close()
            girl_file.close()"""
            save_file(boy,girl,count)

            boy = []
            girl = []
            count += 1
    save_file(boy,girl,count)
    f.close()
# split_file('a3.txt')

print('-------------------------------------')  # 模块,后缀也是 .py

# os 模块，python内置，一般足够
import os
print(os.getcwd())                                     # os.getcwd() 获取当前工作目录
print(os.chdir('C:\\Users\\user\\Desktop\\Python\\UCI'))                # os.chdir(path) 改变工作目录
print(os.getcwd())
print(os.listdir('.'))                                    # os.listdir(path='.' )列举指定目录中的文件名
                                                       # '.'表示当前目录，'..'表示上一级目录
"""
os.mkdir('c:\\a')                                     #  os.mkdir(path) 创建单层目录，如果该目录已经存在则抛出异常
os.makedirs('c:\\a\\b')                               # os.makedirs(path) 递归创建多层目录，如果目录已经存在则抛出异常
os.remove('c:\\a\\b\\a3.txt')                        # 删除文件
os.rmdir('c:\\a\\b')                                 # 删除单层目录，如果该目录非空则抛出异常
os.removedirs('c:\\a\\b')                            # 递归删除目录，从子目录到父目录逐层尝试删除，遇到目录非空则抛出异常
os.rename('c:\\a\\b\\a3.txt','c:\\a\\b\\a36.txt')
"""
print(os.listdir(os.curdir))                                # os.curdir 指代当前目录,相当于'.'
print(os.listdir(os.pardir))                                # os.pardir 指代上一级目录，相当于'..'
print(os.name)                                              # os.name 指代当前使用的操作系统，包括：posix,nt,mac,os2,ce,java
# os.path 模块
print(os.path.basename('c:\\a\\b\\c\\a6.txt'))          # basename（path） 去掉目录路径，单独返回文件名
print(os.path.dirname('c:\\a\\b\\c\\a6.txt'))           # sirname (path) 去掉文件名，单独返回目录路径
print(os.path.join('c:\\','a','b','c'))                  # join(path1,path2...) 将path1，path2各部分组合成一个路径名
print(os.path.split('c:\\ss\\ff\\dd\\k.avi'))           # split(path) 分割文件名和路径，返回(f_path,f_name)元组，不会判断文件或目录是否存在
print(os.path.splitext('c:\\ss\\ff\\dd\\k.avi'))        # splitext(path) 分离文件名和扩展名，返回(f_name,f_extension)元组
print(os.path.getsize(r'C:\Users\user\Desktop\Python\temp.txt'))                   # getsize(file) 返回指定文件的尺寸，单位是字节
import time
print(time.localtime(os.path.getctime(r'C:\Users\user\Desktop\Python\temp.txt')))                  # getctime : 创建时间
print(time.localtime(os.path.getatime(r'C:\Users\user\Desktop\Python\temp.txt')))                  # getatime ：最近访问时间
print(time.localtime(os.path.getmtime(r'C:\Users\user\Desktop\Python\temp.txt')))                  # getmtime ： 最近修改时间
print(os.path.exists('C:\\Python.txt'))                    # 判断指定路径（目录或文件）是否存在

print('-------------------------------------')
# 31讲，中国天气网城市天气数据查询
# 泡菜技术, 字典->二进制
# pickling 对象转化为二进制文件
import pickle
my_list = [123,3.14,'小甲鱼',['another list']]           # 希望将my_list永久保存，即保存为一个文件
pickle_file = open('my_list.pkl','wb')                   # wb：write binary;pkl 后缀无所谓，可以是任意字符
pickle.dump(my_list,pickle_file)                            # dump 倒进去
pickle_file.close()
pickle_file = open('my_list.pkl','rb')
my_list2 = pickle.load(pickle_file)
print(my_list2)

# SyntaxErro():是语法错误，比如 pyhton2.7的语法用在 python3
try:                                                        # 捕获异常
    int('abc')                                              # ValueError'
    sum = 1 + '1'                                           # TypeError
    f = open('我为什么是一个文件.txt')                    #  OSError
    print(f.read())
    f.close()
except OSError as reason:
    print('文件出错啦T_T\n错误的原因是：' + str(reason))
except TypeError as reason:
    print('文件出错啦T_T\n错误的原因是：' + str(reason))
except:
    print('出错了！')                                       # 检测范围一旦出现异常，剩下的语句将不会执行

try:
    f = open(r'C:\Users\user\Desktop\Python\a1.txt')
    print(f.read())
    f.close()
except (OSError,TypeError):                                  # 统一处理
    print('出错了！')

try:
    f = open(r'C:\Users\user\Desktop\Python\temp.txt','w')
    print(f.write('我存在了！'))
    sum = 1 + '1'                                            # 并没有保存
    f.close()
except (OSError,TypeError):                                  # 统一处理
    print('出错了！')
finally:                                                     # 无论是否有异常，都将执行finally
    f.close()

try:
    int('aaa')
except ValueError as reason:
    print('出错了：' + str(reason))
else:
    print('没有任何错误!')

try :
    f = open(r'C:\Users\user\Desktop\Python\a1.txt','r')                # with open('data.txt','w') as f:下面的两行缩进，这样当后面忘记close时可以关闭
    for each_line in f:
        print(each_line)
except OSError as reason:
    print('出错了：' + str(reason))
finally:
    f.close()

print('-------------------------------------')
"""
# GUI编程：图形用户界面
import easygui as gui
import sys
while 1:
    gui.msgbox('欢迎进入第一个界面小游戏^_^')

    msg = '请问你希望在鱼C工作室学习到什么知识呢？'
    title = '小游戏互动'
    choices = ['谈恋爱','编程','OOXX','琴棋书画']

    choice = gui.choicebox(msg,title,choices)

    gui.msgbox('你的选择是：' + str(choice),'结果')

    msg = '你希望重新开始小游戏吗？'
    title = '请选择'

    if gui.ccbox(msg,title):           # ccbox：continue & cancle
        pass
    else:
        sys.exit(0)
"""

print('-------------------------------------')
# 对象
# 乡愁不老
class Turtle:                           # python中的类名约定以大写字母开头
    # =====关于类的一个简单例子=====,  类对象
    # 属性
    color = 'green'
    weight = 10
    legs = 4
    shell = True
    mouth = '大嘴'

    # 方法
    def climb(selfself):
        print('我很努力的向前爬......')
    def run(selfself):
        print('我正在飞快的向前跑...')
    def eat(self):
        print('有的吃，真满足^_^')
tt = Turtle()                           # Turtle类的实例对象
tt.climb()
tt.eat()

list1 = [2,1,7,5,3]
list1.sort()
list1.append(9)
print(list1)

class Mylist(list):                     # Mylist 继承 list
    pass
list2 = Mylist()
list2.append(8)
list2.append(3)
list2.append(9)
print(list2)

class A:
    def fun(self):
        print('我是小A')
class B:
    def fun(self):
        print('我是小B')
a = A()
b = B()
a.fun()                                     # 多态
b.fun()                                     # 面向对象：封装 + 继承 + 多态

# self 是什么
class Ball:
    def setName(self, name):                # def 必须写
        self.name = name
    def kick(self):
        print('我叫%s,该死的，谁踢我...' % self.name)
a = Ball()
a.setName('球A')
b = Ball()
b.setName('球B')
c = Ball()
c.setName('土豆')
a.kick()
c.kick()

class Ball:
    def __init__(self,name):
        self.name = name
    def kick(self):
        print('我叫%s,该死的，谁踢我...' % self.name)
b = Ball('土豆')
b.kick()

class Person:
    name = '小甲鱼1'
p = Person()
print(p.name)
class Person:
    __name = '小甲鱼2'          # __私有变量
p = Person()
#print(p.name)

print('-------------------------------------')      # matplot 绘图
"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 1000)
y = np.sin(x)
z = np.cos(x ** 2)

plt.figure(figsize = (8,4))
plt.plot(x,y,label='$sin(x)$', color = 'red', linewidth = 2)
plt.plot(x,z,'b--',label = '$cos(x^2)$', linewidth = 1)
plt.xlabel('Time(s)')
plt.ylabel('Volt')
plt.title('PyPlot first example')
plt.ylim(-1.2, 1.2)
plt.xlim(0,10)
plt.legend()
plt.show()

# x = random.uniform(0,6.28,100)
# y = x * sin(2 * x + 0.5 * pi)
# y =  sin(2 * x + 0.5 * pi)
x = np.linspace(0, 6.28, 1000)
y = sin(2 * x )
plt.plot(x, y, color = 'red',linewidth = 5)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,6.28)
plt.ylim(-1.2,1.2)
plt.title('sin( x )')
plt.show()
"""
print('-------------------------------------')
# 组合：类和实例化放在一个新类里面，将没有继承关系，有横向关系的几个类组合
class Turtle:
    def __init__(self,x):               # self 是实例的名字
        self.num = x
class Fish:
    def __init__(self,x):
        self.num = x
class Pool:
    def __init__(self,x,y):
        self.turtle = Turtle(x)
        self.fish = Fish(y)
    def print_num(self):
        print('水池里面有乌龟 %d 只，小鱼 %d 条！' % (self.turtle.num, self.fish.num))
pool = Pool(1, 10)
pool.print_num()

class C:            # 类 -> 对象
    count = 33
a = C()              # 实例化
print(C.count)

class C:
    def __init__(self,size = 10):
        self.size = size
    def getSize(self):
        return self.size
    def setSize(self,value):
        self.size = value
    def delSize(self):
        del self.size
    x = property(getSize, setSize,delSize)    # property(1,2,3)用法;设置定义好的属性，传入写好的方法；获得、设置、删除
                                              # 当程序大改，需要将 getSize 改动，接口会改变。用property提供给用户的是 x，用x间接获取和修改
c1 = C()
print(c1.getSize())
print(c1.x)
c1.x = 18
print(c1.x)
print(c1.size)
del c1.x
#print(c1.size)

print('-------------------------------------')
# 构造和析构
# 魔法方法，总是用双下划线包围，例如__init__71 40外事学院利群
# __init__(self[,...])
class Rectangle:
    def __init__(self,x,y):                     # 希望传入两个参数：长，宽;这里不能有返回 -> return
        self.x = x
        self.y = y
    def getPeri(self):
        return (self.x + self.y) * 2
    def getArea(self):
        return self.x * self.y
rect = Rectangle(3,4)
print(rect.getPeri())
print(rect.getArea())
# __new__(cls[,...]),极少重写,当继承不可变类型但需要修改是要重写
class CapStr(str):                             #继承 str,字符串str是不可改变的
    def __new__(cls,string):                    # cls可叫其他名字
        string = string.upper()                 # 变成全部大写
        return str.__new__(cls,string)         #必须有返回,返回实例化的对象
a = CapStr('i love fishc.com')
print(a)

# 魔法方法__add__
class New_int(int):
    def __add__(self, other):
        return int.__sub__(self,other)         #加法 调用 减法
    def __sub__(self, other):
        return int.__add__(self,other)
a = New_int(3)
b = New_int(5)
print(a + b)                                    # 其实是相减
print(a - b)                                    # 其实是相加

class C:
    def __init__(self):
        self.x = 'X-mam'
c = C()
print(c.x)
print(getattr(c, 'x', '没有这个属性'))        # __getattr__:定义当用户试图获取一个不存在的属性时的行为
print(getattr(c, 'xy', '没有这个属性'))
# 属性访问
# 写一个矩形类，默认 长宽 两个属性
# 如果为一个叫square的属性赋值，那么说明这是一个正方形，值及时正方形的边长，
# 此时宽和高都应该等于边长
class Rectangle:
    def __init__(self,width=0,height=0):
        self.width = width
        self.height = height
    def __setattr__(self,name,value):
        if name == 'square':
            self.width = value
            self.height = value
        else:
            #self.name = value                      # 无限递归，死循环
            # super().__setattr__(name,value)         # 用系统写好的
            pass
    def getArea(self):
        return self.width * self.height
# r1 = Rectangle(4, 5)
# print(r1.getArea())
# r1.square = 10
# print(r1.width)
# print(r1.height)
# print(r1.getArea())

#描述符
class MyDecriptor:
    def __get__(self, instance, owner):
        print('getting...',self,instance,owner)
    def __set__(self, instance, value):
        print('setting...',self,instance,value)
    def __delete__(self, instance):
        print('deleting...',self,instance)
class Test:
    x = MyDecriptor()
test = Test()
test.x
print(test)
print(Test)
test.x = 'x-man'
del test.x
# 实现 property 函数,可在 IDL 中运行
"""
class MyProperty:
    def __init__(self,fget=None,fset=None,fdel=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
    def __get__(self, instance, owner):
        return self.fget(instance)
    def __set__(self, instance, value):
        self.fset(instance,value)
    def __del__(self,instance):
        self.fdel(instance)
class C:
    def __init__(self):
        self._x = None
    def getx(self):
        return self._x
    def setx(self,value):
        self._x = value
    def delx(self):
        del self._x
    x = MyProperty(getx,setx,delx)
c = C()
c.x = 'x-man'
print(c.x)
"""

# 先定义一个温度类，然后定义两个描述符类用于描述摄氏度和华氏度两个属性
# 要求两个属性会自动进行转换，给定摄氏度 -> 打印华氏度
class Celsius:                  # 摄氏度
    def __init__(self,value=26.0):
        self.value = float(value)
    def __get__(self, instance, owner):
        return self.value
    def __set__(self, instance, value):
        self.value = float(value)
class Fahrenheit:               # 华氏度
    def __get__(self, instance, owner):
        return instance.cel * 1.8 + 32
    def __set__(self, instance, value):
        instance.cel = (float(value) - 32) / 1.8
class Temprature:
    cel = Celsius()
    fah = Fahrenheit()
temp = Temprature()
print(temp.cel)
temp.cel = 30
print(temp.fah)
temp.fah = 100
print(temp.cel)

from numpy import *
arrayA = array([[1],[2]])

# matrix 和 array 区别
import numpy as np
a = np.mat('4 3;2 1')  #在numpy中matrix-矩阵 的主要优势是：相对简单的乘法运算符号。例如，a和b是两个matrices，那么a*b，就是矩阵积
b = np.mat('1 2;3 4')
print(a)
print(b.I)			   # b.I:逆矩阵 b.H:矩阵转置
print(a*b)			   # 矩阵相乘
c = np.array([[4,3],[2,1]])		#相反的是在numpy里面arrays遵从逐个元素的运算，所以array：c 和d的c*d运算相当于matlab里面的c.*d运算
d = np.array([[1,2],[3,4]])		
print(d.T)						# d.T 是转置
print(c*d)						# 元素逐个相乘
print(np.dot(c,d))				# 相当于矩阵相乘

# 矩阵转化
s = [[1,2]]
A = mat(s)			# array -> matrix
a = A.getA()			# matrix -> array
list(A)					# matrix -> list


