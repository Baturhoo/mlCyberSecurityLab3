from logging import exception
import keras
import sys
import h5py
from keras.utils.generic_utils import validate_config
from tensorflow.keras import models
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
weights_filename = "models/bd_weights.h5"
model_filename = "models/bd_net.h5"
clean_data_filename = "data/cl/valid.h5"
clean_test_data_filename = "data/cl/test.h5"
save_model_filename = "models/repaired_bd_net.h5"
save_weights_filename = "models/repaired_bd_weights.h5"
save_model_filename2 = "models/repaired_bd_net2.h5"
save_weights_filename2 = "models/repaired_bd_weights2.h5"
save_model_filename4 = "models/repaired_bd_net4.h5"
save_weights_filename4 = "models/repaired_bd_weights4.h5"
save_model_filename10 = "models/repaired_bd_net10.h5"
save_weights_filename10 = "models/repaired_bd_weights10.h5"
bd_test_data_filename = "data/bd/bd_test.h5"
gacc =0

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    return x_data/255, y_data

def repair(model, pruneNum,epoch):
    cl_x_valid,cl_y_valid = data_loader(clean_data_filename)
    cl_x_test,cl_y_test = data_loader(clean_test_data_filename)
    bd_x_test, bd_y_test = data_loader(bd_test_data_filename)
    weights_dict = {}
    bias_dict = {}
    acc_dict = {}
    x = keras.Input(shape=(55, 47, 3), name='input')
    conv_1 = keras.layers.Conv2D(20, (4, 4), activation='relu', name='conv_1')(x)
    pool_1 = keras.layers.MaxPooling2D((2, 2), name='pool_1')(conv_1)
    conv_2 = keras.layers.Conv2D(40, (3, 3), activation='relu', name='conv_2')(pool_1)
    pool_2 = keras.layers.MaxPooling2D((2, 2), name='pool_2')(conv_2)
    conv_3 = keras.layers.Conv2D(60, (3, 3), activation='relu', name='conv_3')(pool_2)
    reModel = keras.Model(inputs=x, outputs=conv_3)
    model.load_weights(weights_filename)
    for i in range(1, 4):
        layer = "conv_" + str(i)
        weights, bias = model.get_layer(layer).get_weights()
        reModel.get_layer(layer).set_weights((weights, bias))
        weights_dict[layer] = weights
        bias_dict[layer] = bias
    yhat = reModel.predict(cl_x_valid)
    out = np.mean(yhat, axis=0)
    score = np.argsort(np.sum(out, axis=(0, 1)))
    weights, bias = weights_dict['conv_3'], bias_dict['conv_3']
    for i in range (1, pruneNum):
        weights[:, :, :, score[i]] = np.zeros(np.shape(weights[:, :, :, score[i]]))
        yrepair = np.argmax(model.predict(cl_x_valid), axis=1)
        acc = np.mean(np.equal(yrepair, cl_y_valid)) * 100
        acc_dict[i] = acc
        bias[score[i]] = 0
        model.get_layer('conv_3').set_weights((weights, bias))
    model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(cl_x_valid, cl_y_valid, epochs=epoch)
    clean_label_p = np.argmax(model.predict(cl_x_test), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, cl_y_test))*100
    print('Classification accuracy:', class_accu)
    print('Prune:', pruneNum)
#    gacc = class_accu
    return model,class_accu

if __name__ == "__main__":
    model = keras.models.load_model(model_filename)
    reModel,gacc = repair(model, 20,3)
    gacc = round(gacc)
    reModel.save(save_model_filename)
    reModel.save_weights(save_weights_filename)
#    newModel10, nacc = repair(model, 51, 2)
#    newModel10.save(save_model_filename10)
#    newModel10.save_weights(save_weights_filename10)
#    newModel4, nacc = repair(model, 42, 2)
#    newModel4.save(save_model_filename4)
#    newModel4.save_weights(save_weights_filename4)
#    newModel2, nacc = repair(model, 28, 2)
#    newModel2.save(save_model_filename2)
#    newModel2.save_weights(save_weights_filename2)
 #   for i in range(15,25):
 #       prune = i*2
 #       newModel,nacc = repair(model,prune,2)
 #       if round(nacc) == gacc-2:
 #           newModel.save(save_model_filename2)
 #           newModel.save_weights(save_weights_filename2)
 #       if round(nacc) == gacc-4:
 #           newModel.save(save_model_filename4)
 #           newModel.save_weights(save_weights_filename4)
 #       if round(nacc) == gacc-10:
 #           newModel.save(save_model_filename10)
 #           newModel.save_weights(save_weights_filename10)
