from struct import unpack
import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import minimize

def load_dataset(images_filename, labels_filename):
    with open(images_filename, 'rb') as im:
        images = im.read()
    with open(labels_filename, 'rb') as lab:
        labels = lab.read()
    images_magic, *images_shape = unpack('>IIII', images[:16])
    labels_magic, labels_shape = unpack('>II', labels[:8])
    assert images_magic == 2051
    assert labels_magic == 2049
    
    images_raw = np.frombuffer(images[16:], dtype=np.uint8).reshape(images_shape)
    labels_raw = np.frombuffer(labels[8:], dtype=np.uint8).reshape(labels_shape)
    return images_raw, labels_raw

def generate_pca_basis(images, labels, k=40):
    numbers = 1. * images
    avg = np.mean(numbers, axis=(1, 2))
    numbers -= avg.reshape(-1, 1, 1)
    u, s, v = svds(numbers.reshape(-1, 28**2), k=k)
    return v.reshape(-1, 28, 28)

def generate_basis_moments(images, bas):
    imsz = 28
    moments = []
    for k in range(len(bas)):
        val = np.einsum('ijk,jk', images, bas[k]) / imsz**2
        moments.append(val)
    for i in range(len(bas)):
        for j in range(i+1):
            moments.append(moments[i] * moments[j])
    return np.transpose(moments)

def compute_norm(X):
    return np.amax(X, axis=0)

def add_bias_and_normalize(X, norm):
    X = X / norm
    bias = np.ones((X.shape[0], 1), dtype=np.double)
    return np.hstack((bias, X))

"""
разбиваем данные на две части (train, valid), 
seed задаёт начальное состояние случайного генератора, 
служит для воспроизводимости результатов
"""
def splitting(X, y, seed=20):
    n = X.shape[0]
    permut = np.random.RandomState(seed).permutation(n)
    X = X[permut]
    y = y[permut]
    ntrain = int(0.75 * n)
    Xtrain = X[:ntrain]
    ytrain = y[:ntrain]
    Xvalid = X[ntrain:]
    yvalid = y[ntrain:]
    return Xtrain, ytrain, Xvalid, yvalid

class Logit:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    # вычисляем предсказание модели
    def model(self, w, X=None):
        """
        X - матрица размера количество наблюдений*количество признаков
        W - вектор весов размера количество признаков*количество вариантов весов
        """
        if X is None:
            X = self.X
    #    return 1/(1 + np.exp(-X.dot(w)))
        z = X.dot(w)
        return .5 + .5 * np.tanh(.5 * z)
    
    def predict(self, X, w):
        p = self.model(w, X)
        return np.int_(p > 0.5)
    
    # вычисляем функцию потерь (при lam = 0) или l2-регуляризованный функционал (при lam > 0)
    def logloss(self, w, lam=0):
        X = self.X
        y = self.y
        f = self.model(w)
        eps = 1e-15
        fc = eps + (1-2*eps) * f
        return -(np.dot(y, np.log(fc)) + np.dot(1 - y, np.log1p(-fc))) / len(y) + lam * np.dot(w, w)
    
    # считаем аналитически градиент функции потерь (при lam > 0) или l2-регуляризованного функционала (при lam > 0)
    def analgrad(self, w, lam=0):
        f = self.model(w)
        X = self.X
        y = self.y
        return X.T.dot(f - y) / len(y) + lam*2*np.array(w)
    
    def minloss(self, lam, wapprox=None, method='BFGS'):
        X = self.X
        def wrapped_loss(w):
            return self.logloss(w, lam)
        def wrapped_jac(w):
            return self.analgrad(w, lam)
        if wapprox is None:
            wapprox = np.zeros(X.shape[1])
        wopt = minimize(wrapped_loss, x0=wapprox, jac=wrapped_jac, method=method).x
        return wopt

def evalf1(y, ypred):
    tp = np.dot(y, ypred)
    tn = np.dot(1 - y, 1 - ypred)
    fn = np.dot(y, 1 - ypred)
    fp = np.dot(1 - y, ypred)
    return 2 * tp/(2 * tp + fp +fn)

