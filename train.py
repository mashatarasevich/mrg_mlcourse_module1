import numpy as np
from common import (load_dataset, splitting, generate_pca_basis, generate_basis_moments, 
                    compute_norm, add_bias_and_normalize, Logit, evalf1, predict_all)
from sklearn.metrics import classification_report
import pickle
import argparse
from os.path import join

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--x_train_dir', default='.', help='Directory with train images')
parser.add_argument('--y_train_dir', default='.', help='Directory with train labels')
parser.add_argument('--model_output_dir', default='.', help='Directory with trained model')
parser.add_argument('--skip', default='no', help='Load existing model')

args = parser.parse_args()

images, y = load_dataset(join(args.x_train_dir, 'train-images.idx3-ubyte'),
                         join(args.y_train_dir, 'train-labels.idx1-ubyte'))

if args.skip == 'no': # Train model

    imtrain, ytrain, imvalid, yvalid = splitting(images, y)

    print('PCA')
    bas = generate_pca_basis(imtrain, ytrain, k=30)

    print('Making features')
    Xtrainm = generate_basis_moments(imtrain, bas)
    Xvalidm = generate_basis_moments(imvalid, bas)

    norms = compute_norm(Xtrainm)

    Xtrain = add_bias_and_normalize(Xtrainm, norms)
    Xvalid = add_bias_and_normalize(Xvalidm, norms)

    wall = []
    for i in range(10):
        print('===========================================')
        print('Training one-vs-all classifier for label =', i, '\n')
        Xtrain_i, ytrain_i = Xtrain, np.int_(ytrain == i)
        Xvalid_i, yvalid_i = Xvalid, np.int_(yvalid == i)
         
        digit = Logit(Xtrain_i, ytrain_i)
        f1best = -1
        wapprox = 0.01 * np.random.rand(Xtrain_i.shape[1])
        for lam in reversed(np.logspace(-9, -4, 12)):
            w = digit.minloss(lam, wapprox=wapprox)
            wapprox = w
            ypred = digit.predict(Xvalid_i, w)
            f1 = 100*evalf1(yvalid_i, ypred)
            ypred = digit.predict(Xtrain_i, w)
            f1train = 100*evalf1(ytrain_i, ypred)
            print('lam = %e, f1 = %6.2f%%, f1train = %6.2f%%' % (lam, f1, f1train))
            if f1 > f1best:
                wopt_i = w
        wall.append(wopt_i)

else: # Load existing

    with open(join(args.model_output_dir, 'model.pickle'), 'rb') as f:
        ans = pickle.load(f)
    wall = ans['wall']
    bas = ans['bas']
    norms = ans['norms']

Xm = generate_basis_moments(images, bas)
X = add_bias_and_normalize(Xm, norms)

ypred = predict_all(X, wall)

print(classification_report(y, ypred, digits=3))

ans = {
    'wall': wall,
    'bas': bas,
    'norms': norms    
}

with open(join(args.model_output_dir, 'model.pickle'), 'wb') as f:
    pickle.dump(ans, f, pickle.HIGHEST_PROTOCOL)
