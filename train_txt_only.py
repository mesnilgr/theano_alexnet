import hickle as hkl
import pickle as pkl
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import os
import ipdb
from os.path import join
import numpy as np
import time, sys

path = "/mnt/bigdisk/data/datasets/gender-375k-hkl+txt" 

def score(probas, y):
    return (np.argmax(probas, 1) == y).mean()

if __name__ == "__main__":
    use_cnn = True
    n_epochs = 10
    batch_size = 256

    gender_model = SGDClassifier(penalty='l2', n_jobs=-1, warm_start=True, verbose=0, loss="modified_huber")
    train_y = np.load(join(path, "labels", "train_labels.npy"))
    valid_y = np.load(join(path, "labels", "valid_labels.npy"))
    test_y = np.load(join(path, "labels", "test_labels.npy"))
    
    if use_cnn:
        cnn_test = np.vstack(pkl.load(open("test_probas.pkl")))
        cnn_valid = np.vstack(pkl.load(open("validation_probas.pkl")))

    for n_train_batches, _ in enumerate(sorted(os.listdir(join(path, "train-txt")))):
        pass
    for n_valid_batches, _ in enumerate(sorted(os.listdir(join(path, "valid-txt")))):
        pass
    for n_test_batches, _ in enumerate(sorted(os.listdir(join(path, "test-txt")))):
        pass

    for epoch in range(n_epochs):
        for i, f in enumerate(sorted(os.listdir(join(path, "train-txt")))):
            tic = time.time()
            fname = join(path, "train-txt", f)
            X = hkl.load(fname, safe=False)
            y = train_y[i * batch_size: (i + 1) * batch_size]
            gender_model.partial_fit(X, y, classes=[0, 1])
            toc = time.time()

            print "[training] split train %i/%i in %.2f sec\r" % (i + 1, n_train_batches, toc - tic),
            sys.stdout.flush()

        # evaluation
        valid_acc, txt_valid = [], []
        for i, f in enumerate(sorted(os.listdir(join(path, "valid-txt")))):
            tic = time.time()
            fname = join(path, "valid-txt", f)
            X = hkl.load(fname, safe=False)
            y = valid_y[i * batch_size: (i + 1) * batch_size]
            valid_acc += [gender_model.score(X, y)]
            txt_valid += [gender_model.predict_proba(X)]
            toc = time.time()
            print "[evaluation] split valid %i/%i in %.2f sec\r" % (i + 1, n_valid_batches, toc - tic),
            sys.stdout.flush()
        test_acc, txt_test = [], []
        for i, f in enumerate(sorted(os.listdir(join(path, "test-txt")))):
            tic = time.time()
            fname = join(path, "test-txt", f)
            X = hkl.load(fname, safe=False)
            y = test_y[i * batch_size: (i + 1) * batch_size]
            test_acc += [gender_model.score(X, y)]
            txt_test += [gender_model.predict_proba(X)]
            toc = time.time()
            print "[evaluation] split test %i/%i in %.2f sec\r" % (i + 1, n_test_batches, toc - tic),
            sys.stdout.flush()
 
        # ensemble
        txt_valid, txt_test = np.vstack(txt_valid), np.vstack(txt_test)
        best = -np.inf
        for alpha in np.linspace(0, 1, 25):
            ens_valid = alpha * txt_valid + (1 - alpha) * cnn_valid
            perf_valid = score(ens_valid, valid_y)
            if perf_valid > best:
                best = perf_valid
                best_alpha = alpha
        ens_valid = best_alpha * txt_valid + (1 - best_alpha) * cnn_valid
        ens_test = best_alpha * txt_test + (1 - best_alpha) * cnn_test
        
        print '\n\t ** epoch %i, text only valid %.3f' % (epoch, np.mean(valid_acc))
        print '\t ** epoch %i, text only test %.3f\n\n' % (epoch, np.mean(test_acc))
        print '\t ** epoch %i, image only valid %.3f' % (epoch, score(cnn_valid, valid_y))
        print '\t ** epoch %i, image only test %.3f\n\n' % (epoch, score(cnn_test, test_y))
        print '\t ** epoch %i, text + image valid %.3f' % (epoch, score(ens_valid, valid_y))
        print '\t ** epoch %i, text + image test %.3f (alpha=%.3f)\n' % (epoch, score(ens_test, test_y), best_alpha)
