import numpy as np
from common import (load_dataset, generate_basis_moments, 
                    add_bias_and_normalize, predict_all)
from sklearn.metrics import classification_report
import pickle
import argparse
from os.path import join

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--x_test_dir', default='.', help='Directory with test images')
parser.add_argument('--y_test_dir', default='.', help='Directory with test labels')
parser.add_argument('--model_input_dir', default='.', help='Directory with trained model')

args = parser.parse_args()

images, y = load_dataset(join(args.x_test_dir, 't10k-images.idx3-ubyte'),
                         join(args.y_test_dir, 't10k-labels.idx1-ubyte'))

with open(join(args.model_input_dir, 'model.pickle'), 'rb') as f:
    ans = pickle.load(f)

bas = ans['bas']
norms = ans['norms']
wall = ans['wall']

Xtestm = generate_basis_moments(images, bas)
Xtest = add_bias_and_normalize(Xtestm, norms)

ypred = predict_all(Xtest, wall)

print(classification_report(y, ypred, digits=3))
