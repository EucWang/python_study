import numpy as np
import scipy.spatial.distance as dist
from numpy.linalg import linalg

class DistanceCalculate(object):
    def __init__(self):
        super()

    @staticmethod
    def euclidean_distance(vector1, vector2):
        """欧氏距离（L2 范数） 是最易于理解的一种距离计算方法，源自欧氏空间中两点间的距离公式"""
        return np.sqrt((vector1 - vector2) * ((vector1 - vector2).T))

    @staticmethod
    def manhattan_distance(vector1, vector2):
        """曼哈顿距离(Manhattan Distance)
            实际驾驶距离就是这个“曼哈顿距离” (L1 范数)。而这也是曼哈顿距离名称的
            来源，曼哈顿距离也称为城市街区距离(City Block distance)"""
        abs1 = abs(vector1 - vector2)  # 计算两个矩阵相减的绝对值的结果
        result = np.sum(abs1)          # 对结果矩阵的各个元素想加
        return result

    @staticmethod
    def chebyshev_distance(vector1, vector2):
        """切比雪夫距离(Chebyshev Distance)
            国际象棋玩过么？国王走一步能够移动到相邻的 8 个方格中的任意一个（如图
            1.11） 。那么国王从格子(x1,y1)走到格子(x2,y2)最少需要多少步？自己走走试试。你会
            发现最少步数总是 max(| x2-x1| , |y2-y1| ) 步。有一种类似的一种距离度量方法叫切比雪
            夫距离(L∞范数)"""
        abs_vector = abs(vector1 - vector2)  # 两个矩阵相减, 取绝对值
        max_value = abs_vector.max()         # 结果矩阵中所有元素,取其中的最大值
        return max_value

    @staticmethod
    def cosine_distance(vector1, vector2):
        """夹角余弦(Cosine)
            几何中夹角余弦可用来衡量两个向量方向的差异，机器学习中借用这一概念来衡量样本向量之间的差异
            夹角余弦取值范围为[-1,1]
            夹角余弦越大表示两个向量的夹角越小，夹角余弦越小表示两向量的夹角越大。
            当两个向量的方向重合时夹角余弦取最大值 1，当两个向量的方向完全相反夹角余弦取最小值-1。
            """
        # numpy.multiply(v1, v2)  v1矩阵和v2矩阵各个元素乘积之后的矩阵
        # numpy.dot(v1, v2)   v1矩阵和v2矩阵的 矩阵乘积  等同于  v1 * v2  当v1和v2都是numpy.matrixlib.defmatrix.matrix类型时
        dot = np.multiply(vector1, vector2)
        # print("dot", dot)

        # linalg.norm(v) # 计算矩阵距离原点的距离, 也就是范数
        norm1 = linalg.norm(vector1)
        # print("norm1", str(norm1))
        norm2 = linalg.norm(vector2)
        # print("norm2", str(norm2))

        # 两个矩阵的范数的乘积
        norm_vector_ = norm1 * norm2
        # print("norm_vector_", str(norm_vector_))

        result = (np.sum(dot) / norm_vector_)
        # print("result", str(result))
        return result

    @staticmethod
    def hamming_distance(vector1, vector2):
        """汉明距离(Hamming distance)
            两个等长字符串 s1 与 s2 之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数。
           例如字符串―1111‖与―1001‖之间的汉明距离为 2。
           应用：信息编码（为了增强容错性，应使得编码间的最小汉明距离尽可能大）"""
        vector_ = vector1 - vector2  # 两个矩阵各个元素相减,获得一个新矩阵
        # print("vector_", vector_)
        #  nonzero(a)返回数组a中值不为零的元素的下标，
        # 它的返回值是一个长度为a.ndim(数组a的轴数)的元组，
        # 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
        smstr = np.nonzero(vector_)

        # print("smstr", smstr)
        # shape() 函数 功能是查看矩阵或者数组的维数。
        shape = np.shape(smstr[0])
        # print('shape', shape)
        return shape[0]

    @staticmethod
    def jaccard_distance(vector1, vector2):
        """杰卡德相似系数(Jaccard similarity coefficient)
           两个集合 A 和 B 的交集元素在 A， B 的并集中所占的比例，称为两个集合的杰卡德相似系数

           杰卡德距离(Jaccard distance)
           杰卡德距离用两个集合中不同元素占所有元素的比例来衡量两个集合的区分度。
           """
        # mat = np.mat(vector1, vector2)
        concatenate = np.concatenate((vector1, vector2), axis=0)
        print("concatenate", concatenate)
        return dist.pdist(concatenate, 'jaccard')