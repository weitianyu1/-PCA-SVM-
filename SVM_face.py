# -*- coding:utf-8 -*-
from __future__ import print_function

from time import time
import logging,cv2
import matplotlib.pyplot as plt
from numpy import *
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


PICTURE_PATH = "./pgm"

def get_Image():
    for i in range(1, 40):
        for j in range(1, 11):
            path = PICTURE_PATH + "\\s" + str(i) + "\\" + str(j) + ".pgm"
            img = cv2.imread(path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img_gray.shape
            img_col = img_gray.reshape(h * w)
            all_data_set.append(img_col)
            all_data_label.append(i)
    return h, w

all_data_set = []
all_data_label = []
h, w = get_Image()

X = array(all_data_set)
y = array(all_data_label)
n_samples, n_features = X.shape
n_classes = len(unique(y)) #numpy.unique()
target_names = []
for i in range(1, 40):
    names = "person" + str(i)
    target_names.append(names)

print("Total dataset size:")
print("n_samples: %d" % n_samples)#400
print("n_features: %d" % n_features)#10304
print("n_classes: %d" % n_classes)#40

# split into a training and testing set
'''
random_state：随机数种子——其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的
随机数。比如每次都为1，其他参数一样的情况下你得到的随机数组是一样的。当为None时，产生的随机数组
也会是随机的。

随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；
种子相同，即使实例不同也产生相同的随机数。
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
n_components = 10
print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))

t0 = time()
#选择一种svd方式,whiten是一种数据预处理方式，会损失一些数据信息，但可获得更好的预测结果
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))#特征脸

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)#得到训练集投影系数
X_test_pca = pca.transform(X_test)#得到测试集投影系数
print("done in %0.3fs" % (time() - t0))

print("Fitting the classifier to the training set")
t0 = time()
'''C为惩罚因子，越大模型对训练数据拟合程度越高，gama是高斯核参数'''
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
'''
class_weight='balanced'
表示调整各类别权重，权重与该类中样本数成反比，防止模型过于拟合某个样本数量过大的类
'''
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


def plot_gallery(images, titles, h, w, n_row=3, n_col=3):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        #cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i] - 1]
    true_name = target_names[y_test[i] - 1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
