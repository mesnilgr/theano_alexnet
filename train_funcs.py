import glob
import time
import os
import ipdb

import numpy as np

import pickle
import hickle as hkl

def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Creat folder: " + config['weights_dir']

    return config

def unpack_configs(config, ext_data='.pkl', ext_label='.npy'):
    flag_para_load = config['para_load']
    flag_datalayer = config['use_data_layer']

    # Load Training/Validation Filenames and Labels
    train_folder = config['train_folder']
    val_folder = config['val_folder']
    label_folder = config['label_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data))
    val_filenames = sorted(glob.glob(val_folder + '/*' + ext_data))
    labels = range(2)
    train_labels = np.load(label_folder + 'train_labels' + ext_label)
    val_labels = np.load(label_folder + 'valid_labels' + ext_label)
    img_mean = np.load(config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
    return (flag_para_load, flag_datalayer,
            train_filenames, val_filenames, train_labels, val_labels, img_mean)

def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch == config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx

def get_val_error_pkl(rand_arr, shared_x, shared_y, img_mean,
                       val_filenames, val_labels,
                       flag_datalayer, flag_para_load,
                       batch_size, validate_model,
                       name,
                       send_queue=None, recv_queue=None,
                       flag_top_5=False):

    if flag_datalayer:
        rand_arr.set_value(np.float32([0.5, 0.5, 0]))

    validation_losses = []
    validation_errors = []
    validation_probas = []
    if flag_top_5:
        validation_errors_top_5 = []

    n_val_batches = len(val_filenames)

    if flag_para_load:
        # send the initial message to load data, before each epoch
        send_queue.put(str(val_filenames[0]))
        if not flag_datalayer:
            send_queue.put(np.float32([0.5, 0.5, 0]))

        send_queue.put('calc_finished')

    for val_index in range(n_val_batches):
        print val_index, n_val_batches

        if flag_para_load:
            # load by self or the other process

            # wait for the copying to finish
            msg = recv_queue.get()
            assert msg == 'copy_finished'

            if val_index + 1 < n_val_batches:
                name_to_read = str(val_filenames[val_index + 1])
                send_queue.put(name_to_read)
                if not flag_datalayer:
                    send_queue.put(np.float32([0.5, 0.5, 0]))
        else:
            #val_img = pickle.load(open(str(val_filenames[val_index]))) #- img_mean
            val_img = hkl.load(str(val_filenames[val_index]))
            shared_x.set_value(val_img)

        shared_y.set_value(val_labels[val_index * batch_size:
                                      (val_index + 1) * batch_size])

        if flag_top_5:
            loss, error, error_top_5 = validate_model()
        else:
            loss, error, proba = validate_model()
            #loss, error = validate_model()


        # print loss, error
        validation_losses.append(loss)
        validation_errors.append(error)
        validation_probas.append(proba)
        if flag_top_5:
            validation_errors_top_5.append(error_top_5)

        if flag_para_load and (val_index + 1 < n_val_batches):
            send_queue.put('calc_finished')
    
    print "done! now pickling.."
    pickle.dump(validation_probas, open(name+"_probas.pkl", "w"))
    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)
    print this_validation_error

    if flag_top_5:
        this_validation_error_top_5 = np.mean(validation_errors_top_5)
        return this_validation_error, this_validation_error_top_5, this_validation_loss
    else:
        return this_validation_error, this_validation_loss


def get_val_error_loss(rand_arr, shared_x, shared_y, img_mean,
                       val_filenames, val_labels,
                       flag_datalayer, flag_para_load,
                       batch_size, validate_model,
                       send_queue=None, recv_queue=None,
                       flag_top_5=False):

    if flag_datalayer:
        rand_arr.set_value(np.float32([0.5, 0.5, 0]))

    validation_losses = []
    validation_errors = []
    validation_probas = []
    if flag_top_5:
        validation_errors_top_5 = []

    n_val_batches = len(val_filenames)

    if flag_para_load:
        # send the initial message to load data, before each epoch
        send_queue.put(str(val_filenames[0]))
        if not flag_datalayer:
            send_queue.put(np.float32([0.5, 0.5, 0]))

        send_queue.put('calc_finished')

    for val_index in range(n_val_batches):
        #print val_index, n_val_batches

        if flag_para_load:
            # load by self or the other process

            # wait for the copying to finish
            msg = recv_queue.get()
            assert msg == 'copy_finished'

            if val_index + 1 < n_val_batches:
                name_to_read = str(val_filenames[val_index + 1])
                send_queue.put(name_to_read)
                if not flag_datalayer:
                    send_queue.put(np.float32([0.5, 0.5, 0]))
        else:
            #val_img = pickle.load(open(str(val_filenames[val_index]))) #- img_mean
            val_img = hkl.load(str(val_filenames[val_index])) - img_mean
            shared_x.set_value(val_img)

        shared_y.set_value(val_labels[val_index * batch_size:
                                      (val_index + 1) * batch_size])

        if flag_top_5:
            loss, error, error_top_5 = validate_model()
        else:
            #loss, error, proba = validate_model()
            loss, error = validate_model()


        # print loss, error
        validation_losses.append(loss)
        validation_errors.append(error)
        #validation_probas.append(proba)
        if flag_top_5:
            validation_errors_top_5.append(error_top_5)

        if flag_para_load and (val_index + 1 < n_val_batches):
            send_queue.put('calc_finished')
        if val_index > 10:
            break 
    #print "done! now pickling.."
    #pickle.dump(validation_probas, open("test_probas.pkl", "w"))
    this_validation_loss = np.mean(validation_losses)
    this_validation_error = np.mean(validation_errors)

    if flag_top_5:
        this_validation_error_top_5 = np.mean(validation_errors_top_5)
        return this_validation_error, this_validation_error_top_5, this_validation_loss
    else:
        return this_validation_error, this_validation_loss

def get_rand3d():
    tmp_rand = np.float32(np.random.rand(3))
    tmp_rand[2] = round(tmp_rand[2])
    return tmp_rand

def train_model_wrap(train_model, shared_x, shared_y, rand_arr, img_mean,
                     count, minibatch_index, minibatch_range, batch_size,
                     train_filenames, train_labels,
                     flag_para_load, flag_datalayer,
                     send_queue=None, recv_queue=None):

    if flag_para_load:
        # load by self or the other process

        # wait for the copying to finish
        msg = recv_queue.get()
        assert msg == 'copy_finished'

        if count < len(minibatch_range):
            ind_to_read = minibatch_range[count]
            name_to_read = str(train_filenames[ind_to_read])
            send_queue.put(name_to_read)
            if not flag_datalayer:
                send_queue.put(get_rand3d())

    else:
        #batch_img = np.random.normal(0, 1, (3, 227, 227, 256)).astype(np.float32) 
        batch_img = hkl.load(str(train_filenames[minibatch_index])) - img_mean
        #batch_img = pickle.load(open(str(train_filenames[minibatch_index]))) - img_mean
        shared_x.set_value(batch_img)

    batch_label = train_labels[minibatch_index * batch_size:
                               (minibatch_index + 1) * batch_size]
    shared_y.set_value(batch_label)

    if flag_datalayer:
        rand_arr.set_value(get_rand3d())

    cost_ij = train_model()


    return cost_ij
