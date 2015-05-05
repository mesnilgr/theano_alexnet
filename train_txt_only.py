import hickle as hkl
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import os
import ipdb
from os.path import join
import numpy as np
import time, sys

path = "/mnt/bigdisk/data/datasets/gender-375k-hkl+txt" 

if __name__ == "__main__":
    n_epochs = 10
    batch_size = 256

    gender_model = SGDClassifier(penalty='l2', n_jobs=-1, warm_start=True, verbose=0)
    train_y = np.load(join(path, "labels", "train_labels.npy"))
    valid_y = np.load(join(path, "labels", "valid_labels.npy"))
    
    for n_train_batches, _ in enumerate(sorted(os.listdir(join(path, "train-txt")))):
        pass
    for n_valid_batches, _ in enumerate(sorted(os.listdir(join(path, "valid-txt")))):
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
        valid_acc = []
        for i, f in enumerate(sorted(os.listdir(join(path, "valid-txt")))):
            tic = time.time()
            fname = join(path, "valid-txt", f)
            X = hkl.load(fname, safe=False)
            y = valid_y[i * batch_size: (i + 1) * batch_size]
            valid_acc += [gender_model.score(X, y)]
            toc = time.time()
            print "[evaluation] split valid %i/%i in %.2f sec\r" % (i + 1, n_valid_batches, toc - tic),
            sys.stdout.flush()
        print '\n\t ** epoch %i, valid %.3f\n' % (epoch, np.mean(valid_acc))

    if False:
        for i, batch in enumerate(load_csv_batch('train.csv')):
            X = vectorizer.transform(batch_X(batch))
            gender_y, type_y = batch_y(batch)

            gender_model.partial_fit(X, gender_y, classes=["M", "F"])

            #print 'batch %d (%d)' % (i + 1, (i + 1) * BATCH_SIZE)

        if False:
            train_acc = []
            for i, batch in enumerate(load_csv_batch('train.csv')):
                X = vectorizer.transform(batch_X(batch))
                gender_y, type_y = batch_y(batch)
                #pdb.set_trace()
                train_acc += [gender_model.score(X, gender_y)]
            print 'epoch %i, gender train %.3f' % (epoch, np.mean(train_acc))

        validation = load_csv('valid.csv')
        X = vectorizer.transform(batch_X(validation))
        gender_y, type_y = batch_y(validation)

        print 'epoch %i, gender validation %.3f' % (epoch, gender_model.score(X, gender_y))

        validation = load_csv('valid-balanced.csv')
        X = vectorizer.transform(batch_X(validation))
        gender_y, type_y = batch_y(validation)

        print 'epoch %i, gender validation (balanced) %.3f' % (epoch, gender_model.score(X, gender_y))

