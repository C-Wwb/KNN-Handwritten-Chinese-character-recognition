import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

"""
工程说明：
* 数据集来自github：https://github.com/Asia-Lee/KNN
包含从0-9的二进制数字数据集，并且按照首字母对应数字编号
* 训练集本地地址：'Y:/NLP/KNN/KNN-master/trainingDigits'
* 测试集本地地址：'Y:/NLP/KNN/KNN-master/testDigits'
"""


"""
将32*32的二进制图像转换为1*1024向量
"""
def img2vector(fileName):
    returnVect = np.zeros((1, 1024)) #创建1*1024零向量
    fr = open(fileName) #打开文件
    for i in range(32): #按行读取文件
        lineStr = fr.readline() #读取每一行数据
        for j in range(32): #每一行的前32个元素依次添加到returnVect中
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect#返回转换后的1*1024向量

"""
手写数字分类测试
"""
def handWritingClassTest():
    hwLabels = [] #训练集的labels
    trainingFileList = listdir('Y:/NLP/KNN/KNN-master/trainingDigits') #返回训练集下的文件名
    m = len(trainingFileList) #返回文件夹下文件的个数
    trainingMat = np.zeros((m, 1024)) #初始化训练集的Mat矩阵，训练集
    for i in range(m): #从文件夹中解析出训练集的类别
        fileNameStr = trainingFileList[i] #获得文件的名字
        classNumber = int(fileNameStr.split('_')[0]) #获得分类的数字
        hwLabels.append(classNumber) #将获得的类别添加到hwLabels中
        trainingMat[i,:] = img2vector('Y:/NLP/KNN/KNN-master/trainingDigits/%s' % (fileNameStr)) #将每一个文件的1*1024数据存储到trainingMat矩阵中
    neigh = KNN(n_neighbors = 3, algorithm = 'auto') #构建KNN分类器
    neigh.fit(trainingMat, hwLabels) #拟合模型，trainingMat为训练矩阵，hwlabels为对应的标签
    testFileList = listdir('Y:/NLP/KNN/KNN-master/testDigits') #返回训练数据集目录下的文件列表
    errorCount = 0.0 #错误检测计数
    mTest = len(testFileList) #测试数据集的数量
    for i in range(mTest):  # 从文件中解析出测试数据记得类别并进行分类测试
        fileNameStr = testFileList[i]  # 获取文件的名字
        classNumber = int(fileNameStr.split('_')[0])  # 获得分类的数字
        vectorUnderTest = img2vector('Y:/NLP/KNN/KNN-master/testDigits/%s' % (fileNameStr))  # 获得测试集的1*1024向量，用于训练
        classifierResult = neigh.predict(vectorUnderTest)  # 获得预测结果
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("总共错了%d个数据\n数据的错误率为%f%%" % (errorCount, errorCount / mTest * 100))

"""
main函数
"""
if __name__ == '__main__':
    handWritingClassTest()

