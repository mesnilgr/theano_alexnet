import sys
sys.path.append('./lib')
import os
from layers import DropoutLayer
import theano
from alex_net import AlexNet
import numpy as np
import pdb
import time
from tools import load_weights
import hickle as hkl
import cPickle

model_epoch = 23
model_folder = "./models/"
img_mean_path = "/mnt/bigdisk/data/datasets/gender-1M/train/image_mean.npy"
data_folder = "/mnt/bigdisk/data/datasets/gender-1M/valid/"
y = np.load("/mnt/bigdisk/data/datasets/gender-1M/labels/valid_labels.npy")

def valid():
    rng = np.random.RandomState(23455)

    config = {'batch_size':256,
              'use_data_layer':True,
              'lib_conv':'cudnn',
              'rand_crop':False,
              'rng':rng
              }

    model = AlexNet(config)
    DropoutLayer.SetDropoutOff()
    load_weights(model.layers, model_folder, model_epoch)
    img_mean = np.load(img_mean_path)[:, :, :, np.newaxis]

    rep = np.zeros((195 * 256, 4096))
    batch_size = 256
    x = theano.shared(np.random.normal(0, 1 , (3, 227, 227, 256)).astype(theano.config.floatX))
    f = theano.function([], [model.proba, model.layers[-2].output], givens={model.x:x})
    accuracy = []
    tic = time.time()

    for i, filename in enumerate(sorted(os.listdir(data_folder))):
        val_img = hkl.load(os.path.join(data_folder, filename)) - img_mean
        x.set_value(val_img)
        probas, last_layer = f()
        pred = np.argmax(probas, axis=1)

        accuracy += [(pred == y[i * batch_size: (i + 1) * batch_size]).mean() * 100.]
        toc = time.time()
        print "filename %s in %.2f sec \taccuracy :: %.2f\r" % (filename, toc - tic, np.mean(accuracy)),
        sys.stdout.flush()
        tic = toc

        rep[i * batch_size:(i + 1) * batch_size, :] = last_layer
    print "\naccuracy :: ", np.array(accuracy).mean()
    pdb.set_trace()
    cPickle.dump(rep, open("representation.pkl", "w"))

def test_speed_cpu():
    rng = np.random.RandomState(23455)

    config = {'batch_size':1,
              'use_data_layer':False,
              'lib_conv':'cpu',
              'rand_crop':False,
              'rng':rng
              }

    model = AlexNet(config)
    DropoutLayer.SetDropoutOff()
    load_weights(model.layers, model_folder, model_epoch)
    img_mean = np.load(img_mean_path)[:, :, :, np.newaxis]

    # testing time to predict per image
    tic = time.time()
    niters = 50
    for i in range(niters):
        x = np.random.normal(0, 1 , (3, 227, 227, 1)).astype(theano.config.floatX)   
        f = theano.function([model.x], model.outputs[-1])
        #print f(x).shape
    print "%.2f secs per images (averaged over %i iterations)" % ((time.time() - tic) / niters, niters)

if __name__ == "__main__":
    test_speed_cpu()
