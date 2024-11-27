from pyclbr import readmodule
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, AveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Add, GlobalAveragePooling2D, Activation, ZeroPadding2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import keras
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.decomposition import PCA
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs
from tensorflow.keras.datasets import mnist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import scipy
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
import matplotlib
from keras_flops import get_flops
import time
from HE11e import HE11e
from HE12e import HE12e
from HE21e import HE21e
from TE02 import TE02

def to_one(x):
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
    return y

def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a, b):
    a = a - mean2(a)
    b = b - mean2(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def concat_np7(x1,x2,x3,x4,x5,x6,x7):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    return np.array(result)

def concat_np11(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    result.append(x8)
    result.append(x9)
    result.append(x10)
    result.append(x11)
    return np.array(result)

def concat_np15(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15):
    result = []
    result.append(x1)
    result.append(x2)
    result.append(x3)
    result.append(x4)
    result.append(x5)
    result.append(x6)
    result.append(x7)
    result.append(x8)
    result.append(x9)
    result.append(x10)
    result.append(x11)
    result.append(x12)
    result.append(x13)
    result.append(x14)
    result.append(x15)
    return np.array(result)

# 不考虑径向偏移和相位偏移下计算相关系数
def correction_2(m, r):
    # 根据U、V计算模场分布
    Npoint = 128  # 设置二维作图的分辨率
    Rx = 2        # 设置横坐标方向的归一化半径取值范围
    Ry = 2        # 设置纵坐标方向的归一化半径取值范围
    # 创建x和y轴的线性空间
    x = np.linspace(-Rx, Rx, Npoint)
    y = np.linspace(-Ry, Ry, Npoint)
    # 创建X和Y的网格
    X, Y = np.meshgrid(x, y)
    max_cor = []
    best_value = []
    orignal_data = []
    restore_data = []
    for i in range(r.shape[0]):
        data_value1 = HE11e(X, Y, r[i][2], r[i][4], 0)  #HE11e
        data_value2 = HE21e(X, Y, r[i][3], r[i][5], r[i][6])
        measure_data1 = HE11e(X, Y, m[i][2], m[i][4], 0)
        measure_data2 = HE21e(X, Y, m[i][3], m[i][5], m[i][6])
        result1 = r[i][0]*data_value1+r[i][1]*data_value2
        result2 = m[i][0]*measure_data1+m[i][1]*measure_data2
        result = corr2(np.abs(result1)**2, np.abs(result2)**2)      
        max_cor.append(result)
        orignal_data.append(result2)
        restore_data.append(result1)
        best_value.append(concat_np7(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6]))
    return np.array(orignal_data), np.mean(max_cor), np.array(restore_data), np.array(best_value)

def correction_3(m, r):
    # 根据U、V计算模场分布
    Npoint = 128  # 设置二维作图的分辨率
    Rx = 2        # 设置横坐标方向的归一化半径取值范围
    Ry = 2        # 设置纵坐标方向的归一化半径取值范围
    # 创建x和y轴的线性空间
    x = np.linspace(-Rx, Rx, Npoint)
    y = np.linspace(-Ry, Ry, Npoint)
    # 创建X和Y的网格
    X, Y = np.meshgrid(x, y)
    max_cor = []
    best_value = []
    orignal_data = []
    restore_data = []
    for i in range(r.shape[0]):
        data_value1 = HE11e(X, Y, r[i][3], r[i][6], 0)  #HE11e
        data_value2 = HE21e(X, Y, r[i][4], r[i][7], r[i][9])
        data_value3 = HE12e(X, Y, r[i][5], r[i][8], r[i][10])
        measure_data1 = HE11e(X, Y, m[i][3], m[i][6], 0)
        measure_data2 = HE21e(X, Y, m[i][4], m[i][7], m[i][9])
        measure_data3 = HE12e(X, Y, m[i][5], m[i][8], m[i][10])
        result1 = r[i][0]*data_value1+r[i][1]*data_value2+r[i][2]*data_value3
        result2 = m[i][0]*measure_data1+m[i][1]*measure_data2+m[i][2]*measure_data3
        result = corr2(np.abs(result1)**2, np.abs(result2)**2)      
        max_cor.append(result)
        orignal_data.append(result2)
        restore_data.append(result1)
        best_value.append(concat_np11(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7], r[i][8], r[i][9], r[i][10]))
    return np.array(orignal_data), np.mean(max_cor), np.array(restore_data), np.array(best_value)

def correction_4(m, r):
    # 根据U、V计算模场分布
    Npoint = 128  # 设置二维作图的分辨率
    Rx = 2        # 设置横坐标方向的归一化半径取值范围
    Ry = 2        # 设置纵坐标方向的归一化半径取值范围
    # 创建x和y轴的线性空间
    x = np.linspace(-Rx, Rx, Npoint)
    y = np.linspace(-Ry, Ry, Npoint)
    # 创建X和Y的网格
    X, Y = np.meshgrid(x, y)
    max_cor = []
    best_value = []
    orignal_data = []
    restore_data = []
    for i in range(r.shape[0]):
        data_value1 = HE11e(X, Y, r[i][4], r[i][8], 0)  #HE11e
        data_value2 = HE21e(X, Y, r[i][5], r[i][9], r[i][12])
        data_value3 = HE12e(X, Y, r[i][6], r[i][10], r[i][13])
        data_value4 = TE02(X, Y, r[i][7], r[i][11], r[i][14])
        measure_data1 = HE11e(X, Y, m[i][4], m[i][8], 0)
        measure_data2 = HE21e(X, Y, m[i][5], m[i][9], m[i][12])
        measure_data3 = HE12e(X, Y, m[i][6], m[i][10], m[i][13])
        measure_data4 = TE02(X, Y, m[i][7], m[i][11], m[i][14])
        result1 = r[i][0]*data_value1+r[i][1]*data_value2+r[i][2]*data_value3+r[i][3]*data_value4
        result2 = m[i][0]*measure_data1+m[i][1]*measure_data2+m[i][2]*measure_data3+m[i][3]*measure_data4
        result = corr2(np.abs(result1)**2, np.abs(result2)**2)      
        max_cor.append(result)
        orignal_data.append(result2)
        restore_data.append(result1)
        best_value.append(concat_np15(r[i][0], r[i][1], r[i][2], r[i][3], r[i][4], r[i][5], r[i][6], r[i][7], r[i][8], r[i][9], r[i][10], r[i][11], r[i][12], r[i][13], r[i][14]))
    return np.array(orignal_data), np.mean(max_cor), np.array(restore_data), np.array(best_value)

class ReservoirLayer(Layer):
    def __init__(self, num_reservoir_nodes=5000, sparse_rate=0.5, input_scaling=0.8, spectral_radius=1.2, **kwargs):
        super(ReservoirLayer, self).__init__(**kwargs)
        self.num_reservoir_nodes = num_reservoir_nodes
        self.sparse_rate = sparse_rate
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        # self.normalizer = tf.keras.layers.Normalization()

    def build(self, input_shape):
        # Initialize input weights (W_in)
        self.W_in = self.add_weight(shape=(self.num_reservoir_nodes, input_shape[-1]),
                                    initializer='random_uniform',
                                    trainable=False) * self.input_scaling
        # Create sparse random weight matrix (W_reservoir)
        W_reservoir = sparse_random(self.num_reservoir_nodes, self.num_reservoir_nodes, density=self.sparse_rate).toarray() - 0.5
        # Adjust spectral radius
        original_spectral_radius = abs(eigs(W_reservoir, k=1, which='LM', return_eigenvectors=False)[0])
        self.W_reservoir = tf.convert_to_tensor(W_reservoir * (self.spectral_radius / original_spectral_radius), dtype=tf.float32)
        self.b = self.add_weight(shape=(self.num_reservoir_nodes,),
                             initializer='random_uniform',
                             trainable=True)  # 加入偏置项

    def call(self, inputs):
        # inputs: shape (batch_size, num_features)
        # inputs = self.normalizer(inputs)
        batch_size = tf.shape(inputs)[0]
        reservoir_state = tf.zeros((batch_size, self.num_reservoir_nodes), dtype=tf.float32)  # Adjusted shape
        u = inputs  
        # Calculate new reservoir state
        reservoir_state = tf.nn.relu(tf.matmul(reservoir_state, self.W_reservoir) + 
                                    tf.matmul(u, tf.transpose(self.W_in)) + self.b)
        # Stack states into a 2D tensor of shape (batch_size, num_reservoir_nodes)
        return reservoir_state

class MultiReservoirLayer(Layer):
    def __init__(self, num_layers=3, nodes_per_layer=1000, **kwargs):
        super().__init__(**kwargs)
        self.layers = [ReservoirLayer(num_reservoir_nodes=nodes_per_layer) for _ in range(num_layers)]

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class RidgeRegressionLayer(Layer):
    def __init__(self, output_dim, reg=1e-6, **kwargs):
        super(RidgeRegressionLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.reg = reg

    def build(self, input_shape):
        self.W_out = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                     initializer='zeros',
                                     regularizer=tf.keras.regularizers.L2(self.reg),
                                     trainable=True)

    def call(self, inputs):
        # Ridge regression output calculation
        return tf.nn.sigmoid(tf.matmul(inputs, self.W_out))
        #  return tf.matmul(inputs, self.W_out)

class Normalizer(tf.keras.layers.Layer):
    def __init__(self, method='standard', **kwargs):
        super(Normalizer, self).__init__(**kwargs)
        self.method = method

    def call(self, inputs):
        if self.method == 'standard':
            # 标准化：均值为0，标准差为1
            mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=True)
            normalized_data = (inputs - mean) / tf.sqrt(variance + 1e-6)
        elif self.method == 'minmax':
            # 归一化：压缩到 [0,1] 范围
            min_val = tf.reduce_min(inputs, axis=[0, 1, 2], keepdims=True)
            max_val = tf.reduce_max(inputs, axis=[0, 1, 2], keepdims=True)
            normalized_data = (inputs - min_val) / (max_val - min_val + 1e-6)
        else:
            raise ValueError("Method must be either 'standard' or 'minmax'")     
        return normalized_data

# 适用Resnet的basic_block
def basic_block_identity(inp, filters, kernel_size, block, layer):
    conv_name = 'basic_conv_b' + block + '_l' + layer
    batch_name = 'basic_batch_b' + block + '_l' + layer
    z = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = conv_name + '_a')(inp)
    z = BatchNormalization(name = batch_name + '_a')(z)
    # if((block=='2' and layer=='2') or (block=='3' and layer=='2')):
    # z = Activation('relu')(z)
    z = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer = 'he_normal', name = conv_name + '_b')(z)
    z = BatchNormalization(name = batch_name + '_b')(z)
    # if((block=='2' and layer=='2') or (block=='3' and layer=='2')):
    # z = Activation('relu')(z)
    add = Add()([inp, z])
    z = add
    # if((block=='2' and layer=='2') or (block=='3' and layer=='2')):
    # z = Activation('relu')(add) 
    return z

def basic_block_convolutional(inp, filters, kernel_size, block, layer, strides = 2):
    conv_name = 'basic_conv_b' + block + '_l' + layer
    batch_name = 'basic_batch_b' + block + '_l' + layer
    w = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', strides = 1, kernel_initializer = 'he_normal', name = conv_name + '_a')(inp)
    w = BatchNormalization(name = batch_name + '_a')(w)
    # if((block=='2' and layer=='1') or (block=='3' and layer=='1')):
    # w = Activation('relu')(w)
    w = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', strides = strides, kernel_initializer = 'he_normal', name = conv_name + '_b')(w)
    w = BatchNormalization(name = batch_name + '_b')(w)
    # if((block=='2' and layer=='1') or (block=='3' and layer=='1')):
    # w = Activation('relu')(w)
    shortcut = Conv2D(filters = filters, kernel_size = 1, strides = strides, kernel_initializer = 'he_normal', name = conv_name + '_shortcut')(inp)  # 若这个卷积块涉及到size变换, 那么要将计算shortcut的conv的strides设置为2
    shortcut = BatchNormalization(name = batch_name + '_shortcut')(shortcut)
    add = Add()([shortcut, w])
    w = add
    # if((block=='2' and layer=='1') or (block=='3' and layer=='1')):
    # w = Activation('relu')(add)
    return w

img = Input(shape=(128, 128, 2), name='input')
padd = ZeroPadding2D(3)(img)

conv1 = Conv2D(64, 7, strides = 2, padding = 'valid', name = 'conv1')(padd)  # (32, 14, 14, 64)
conv1 = BatchNormalization(name = 'batch2')(conv1)
# conv1 = Activation('relu')(conv1)
conv1 = ZeroPadding2D(1)(conv1)
conv1 = MaxPooling2D(3, 2)(conv1)  # (32, 7, 7, 64)

#Resnet18
conv2 = basic_block_convolutional(conv1, 64, 3, '2', '1', strides = 1)
# conv2 = Activation('relu')(conv2)
conv2 = basic_block_identity(conv2, 64, 3, '2', '2')
# conv2 = Activation('relu')(conv2)
conv3 = basic_block_convolutional(conv2, 128, 3, '3', '1')
# conv3 = Activation('relu')(conv3)
conv3 = basic_block_identity(conv3, 128, 3, '3', '2')
# conv3 = Activation('relu')(conv3)
conv4 = basic_block_convolutional(conv3, 256, 3, '4', '1') 
# conv4 = Activation('relu')(conv4)
conv4 = basic_block_identity(conv4, 256, 3, '4', '2')
# conv4 = Activation('relu')(conv4)

conv5 = basic_block_convolutional(conv4, 512, 3, '5',  '1')
conv5 = basic_block_identity(conv5, 512, 3, '5', '2')
avg_pool = GlobalAveragePooling2D()(conv5)
# dense = Dense(7, activation='sigmoid')(avg_pool)  # 最后一个fc层, 10根据图片的类别数而改变
dense = Dense(11)(avg_pool)  # 最后一个fc层,10根据图片的类别数而改变

def conv_preprocess(input_shape):
    input_img = Input(shape=input_shape)
    padd = ZeroPadding2D(3)(input_img)
    conv1 = Conv2D(64, 7, strides = 2, padding = 'valid', name = 'conv1')(padd)  # (32, 14, 14, 64)
    conv1 = BatchNormalization(name = 'batch2')(conv1)
    # conv1 = Activation('relu')(conv1)
    conv1 = ZeroPadding2D(1)(conv1)
    conv1 = MaxPooling2D(3, 2)(conv1)  # (32, 7, 7, 64)
    conv2 = basic_block_convolutional(conv1, 64, 3, '2', '1', strides = 1)
    # conv2 = Activation('relu')(conv2)
    conv2 = basic_block_identity(conv2, 64, 3, '2', '2')
    # conv2 = Activation('relu')(conv2)
    conv3 = basic_block_convolutional(conv2, 128, 3, '3', '1')
    # conv3 = Activation('relu')(conv3)
    conv3 = basic_block_identity(conv3, 128, 3, '3', '2')
    # conv3 = Activation('relu')(conv3)
    conv4 = basic_block_convolutional(conv3, 256, 3, '4', '1') 
    # conv4 = Activation('relu')(conv4)
    conv4 = basic_block_identity(conv4, 256, 3, '4', '2')
    # conv4 = Activation('relu')(conv4)
    conv5 = basic_block_convolutional(conv4, 512, 3, '5', '1')
    # conv5 = Activation('relu')(conv5)
    conv5 = basic_block_identity(conv5, 512, 3, '5', '2')
    # conv5 = Activation('relu')(conv5)
    # avg_pool = GlobalAveragePooling2D()(conv5)
    x = Flatten()(conv5) 
    model = Model(inputs=input_img, outputs=x)
    return model

# 建立整体模型
def build_full_model_with_rc(input_shape, num_reservoir_nodes=5000, num_classes=7):
    # resnet_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape, pooling='avg')
    inputs = tf.keras.Input(shape=input_shape)
    # normalized_layer = Normalizer(method='standard')(inputs)
    # 卷积预处理
    conv_model = conv_preprocess(input_shape)
    x_conv = conv_model(inputs)
    # 添加 Reservoir Layer
    reservoir_layer = ReservoirLayer(num_reservoir_nodes=num_reservoir_nodes)
    reservoir_output = reservoir_layer(x_conv)
    # mutil_RC = MultiReservoirLayer()
    # reservoir_output = mutil_RC(x_conv)
    # 添加 Dense 层进行最终预测
    # output = Dense(num_classes)(reservoir_output)
    ridge_layer = RidgeRegressionLayer(output_dim=num_classes)
    output = ridge_layer(reservoir_output)
    # output = Dense(num_classes, activation='sigmoid')(reservoir_output)
    model = Model(inputs=inputs, outputs=output)
    return model

def get_data(folder):  # folder是根目录文件夹
    result_list = []
    label_list = []
    for file in sorted(os.listdir(folder)):# 按文件字母顺序读取
        img = scipy.io.loadmat(folder + file)
        data_values1 = img['E'].reshape(128, 128, 1)  # 将读取的权重和旋转角度转换为一维数组
        data_label = np.array(img['label'].reshape(7, ))
        abs_temp = np.abs(data_values1)**2
        angle_temp = np.angle(data_values1)
        data = tf.concat([abs_temp, angle_temp], axis = 2)
        # anglelabel_temp = data_label[8:15]
        # anglelabel_temp = data_label[6:11]
        anglelabel_temp = data_label[4:7]
        anglelabel_temp2 = np.divide(anglelabel_temp, np.pi*2)
        label = [data_label[0], data_label[1], data_label[2], data_label[3], anglelabel_temp2[0], anglelabel_temp2[1], anglelabel_temp2[2]]
        # label = [data_label[0], data_label[1], data_label[2], data_label[3], data_label[4], data_label[5], anglelabel_temp2[0], anglelabel_temp2[1], anglelabel_temp2[2], anglelabel_temp2[3], anglelabel_temp2[4]]
        # label = [data_label[0], data_label[1], data_label[2], data_label[3], data_label[4], data_label[5], data_label[6], data_label[7], anglelabel_temp2[0], anglelabel_temp2[1], anglelabel_temp2[2], anglelabel_temp2[3], anglelabel_temp2[4], anglelabel_temp2[5], anglelabel_temp2[6]]
        label_list.append(label)
        result_list.append(data)
    return result_list, label_list

def load_data():
    data_image_train, data_label_train = get_data('/tf/桌面/FiOLS/temp_modal/mode2/train_image/')
    data_image_train = np.array(data_image_train)
    data_label_train = np.array(data_label_train)
    data_image_val, data_label_val = get_data('/tf/桌面/FiOLS/temp_modal/mode2/val_image/')
    data_image_val = np.array(data_image_val)
    data_label_val = np.array(data_label_val)
    data_image_test, data_label_test = get_data('/tf/桌面/FiOLS/temp_modal/mode2/test_image/')
    data_image_test = np.array(data_image_test)
    data_label_test = np.array(data_label_test)
    return data_image_train, data_label_train, data_image_val, data_label_val, data_image_test, data_label_test
train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data()
# 建立并编译模型
model = build_full_model_with_rc(input_shape=(128, 128, 2), num_reservoir_nodes=1000, num_classes=7) # RC
# model = Model(img, dense) # Resnet18
print(model.summary())
flops = get_flops(model, batch_size = 32) # 计算计算算力
print("计算算力",flops/10**6)
# 编译模型
def scheduler(epoch):
    if epoch < 40:
        return 0.001
    else:
        return 0.0001
rms = RMSprop(0.01)
loss_func = tf.keras.losses.MeanSquaredError()
mse_result1 = []
mse_result2 = []
mse_result3 = []
mse_result4 = []
cor_result = []
learning_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
class Printcor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        output_result = self.model.predict(test_images)
        output_temp = []
        output_temp2 = []
        test_labels_temp = []
        test_labels_temp2 = []
        for i in range(output_result.shape[0]):
            output_temp.append(output_result[i][4:7]*2*math.pi)
            output_temp2.append(output_result[i][0:4])
            test_labels_temp.append(test_labels[i][4:7]*2*math.pi)
            test_labels_temp2.append(test_labels[i][0:4])
            # output_temp.append(output_result[i][6:11]*2*math.pi)
            # output_temp2.append(output_result[i][0:6])
            # test_labels_temp.append(test_labels[i][6:11]*2*math.pi)
            # test_labels_temp2.append(test_labels[i][0:6])
            # output_temp.append(output_result[i][8:15]*2*math.pi)
            # output_temp2.append(output_result[i][0:8])
            # test_labels_temp.append(test_labels[i][8:15]*2*math.pi)
            # test_labels_temp2.append(test_labels[i][0:8])
        output_data = tf.concat([output_temp2, output_temp], axis = 1)
        output_data = np.array(output_data)
        test_label_data = tf.concat([test_labels_temp2, test_labels_temp], axis = 1)
        test_label_data = np.array(test_label_data)
        print(test_label_data[0], output_data[0])
        mse_modal_weight = np.mean(np.abs(np.array(test_labels_temp2)[:,0:2] - np.array(output_temp2)[:,0:2]))
        mse_rotation_offset = np.mean(np.abs(np.array(test_labels_temp2)[:,2:4] - np.array(output_temp2)[:,2:4]))
        mse_phase_offset = np.mean(np.abs(np.array(test_labels_temp)[:,0:2] - np.array(output_temp)[:,0:2])/(2*math.pi))
        mse_modal_phase = np.mean(np.abs(np.array(test_labels_temp)[:,2:3] - np.array(output_temp)[:,2:3])/(2*math.pi))
        # mse_modal_weight = np.mean(np.abs(np.array(test_labels_temp2)[:,0:3] - np.array(output_temp2)[:,0:3]))
        # mse_rotation_offset = np.mean(np.abs(np.array(test_labels_temp2)[:,3:6] - np.array(output_temp2)[:,3:6]))
        # mse_phase_offset = np.mean(np.abs(np.array(test_labels_temp)[:,0:3] - np.array(output_temp)[:,0:3])/(2*math.pi))
        # mse_modal_phase = np.mean(np.abs(np.array(test_labels_temp)[:,3:5] - np.array(output_temp)[:,3:5])/(2*math.pi))
        # mse_modal_weight = np.mean(np.abs(np.array(test_labels_temp2)[:,0:4] - np.array(output_temp2)[:,0:4]))
        # mse_rotation_offset = np.mean(np.abs(np.array(test_labels_temp2)[:,4:8] - np.array(output_temp2)[:,4:8]))
        # mse_phase_offset = np.mean(np.abs(np.array(test_labels_temp)[:,0:4] - np.array(output_temp)[:,0:4])/(2*math.pi))
        # mse_modal_phase = np.mean(np.abs(np.array(test_labels_temp)[:,4:7] - np.array(output_temp)[:,4:7])/(2*math.pi))
        mse_result1.append(mse_modal_weight)
        mse_result2.append(mse_modal_phase)
        mse_result3.append(mse_rotation_offset)
        mse_result4.append(mse_phase_offset)
        print(mse_modal_weight)
        print(mse_modal_phase)
        print(mse_rotation_offset)
        print(mse_phase_offset)
        _, cor, _, _ = correction_2(test_label_data, output_data)
        # _, cor, _, _ = correction_3(test_label_data, output_data)
        # _, cor, _, _ = correction_4(test_label_data, output_data)
        print("\n")
        print(cor)
        cor_result.append(cor)
model.compile(optimizer=rms, loss=loss_func, metrics='mse')
# 训练模型
start_time_train = time.time()
model.fit(train_images, train_labels, validation_data = (val_images, val_labels), epochs=80, batch_size=32, callbacks = [learning_scheduler, Printcor()]) 
end_time_train = time.time()
print("训练时间",(end_time_train-start_time_train))
start_time_test = time.time()
output_result = model.predict(test_images)
end_time_test = time.time()
print("总的预测时间",(end_time_test-start_time_test))
output_temp = []
output_temp2 = []
test_labels_temp = []
test_labels_temp2 = []
for i in range(output_result.shape[0]):
    output_temp.append(output_result[i][4:7]*2*math.pi)
    output_temp2.append(output_result[i][0:4])
    test_labels_temp.append(test_labels[i][4:7]*2*math.pi)
    test_labels_temp2.append(test_labels[i][0:4])
    # output_temp.append(output_result[i][6:11]*2*math.pi)
    # output_temp2.append(output_result[i][0:6])
    # test_labels_temp.append(test_labels[i][6:11]*2*math.pi)
    # test_labels_temp2.append(test_labels[i][0:6])
    # output_temp.append(output_result[i][8:15]*2*math.pi)
    # output_temp2.append(output_result[i][0:8])
    # test_labels_temp.append(test_labels[i][8:15]*2*math.pi)
    # test_labels_temp2.append(test_labels[i][0:8])
output_data = tf.concat([output_temp2, output_temp], axis = 1)
output_data = np.array(output_data)
test_label_data = tf.concat([test_labels_temp2, test_labels_temp], axis = 1)
test_label_data = np.array(test_label_data)
# output_data = test_label_data # 测试使用
orignal_data, cor, restore_data, Y2 = correction_2(test_label_data, output_data)
# orignal_data, cor, restore_data, Y2 = correction_3(test_label_data, output_data)
# orignal_data, cor, restore_data, Y2 = correction_4(test_label_data, output_data)
print(mse_result1)
print(mse_result2)
print(mse_result3)
print(mse_result4)
# 画4条mse曲线  测试集
# X = np.arange(80)
# plt.figure(1)
# print(mse_result1)
# print(mse_result2)
# print(mse_result3)
# print(mse_result4)
# 绘制四条MSE曲线
# plt.plot(X, np.array(mse_result1), label='Mode weight mse', marker='o', markersize=4)
# plt.plot(X, np.array(mse_result2), label='Mode phase mse', marker='x', markersize=4)
# plt.plot(X, np.array(mse_result3), label='Radial offset mse', marker='s', markersize=4)
# plt.plot(X, np.array(mse_result4), label='Phase offset mse', marker='d', markersize=4)
# plt.legend()
# plt.title('MSE values')
# plt.xlabel('Epochs')
# plt.ylabel('MSE')
# plt.grid(linewidth = 0.1)
# plt.savefig('/tf/桌面/FiOLS/temp_modal/mode66/result/mse_result.svg')

# # 画相关系数
# X = np.arange(80)
# plt.figure(2)
# print(cor_result)
# plt.plot(X, np.array(cor_result), color = 'red', linestyle = '-', marker = '.', markersize=4)
# plt.xlabel('Epochs',fontsize=13)
# plt.ylabel('Correlation',fontsize=13)
# # plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80],fontsize=13)
# # plt.yticks([0.80, 0.84, 0.88,0.92,0.96,1.00],fontsize=13)
# plt.grid(linewidth = 0.1)
# plt.savefig('/tf/桌面/FiOLS/temp_modal/mode66/result/cor_result.svg')

# # 画模式权重、模式相位柱状图 
# temppp = []      
# for i in range(1000):
#     phase_temp2 = test_labels[i]
#     phase2_temp2 = output_result[i]
#     # print(phase,phase2)
#     if(phase_temp2[10]<0.8 and phase_temp2[6]<0.8 and phase_temp2[8]<0.8):
#         if(phase2_temp2[10]<0.8 and phase2_temp2[6]<0.8 and phase2_temp2[8]<0.8):
#     # if(phase_temp2[8]<0.8 and phase_temp2[11]<0.8 and phase_temp2[14]<0.8):
#     #     if(phase2_temp2[8]<0.8 and phase2_temp2[11]<0.8 and phase2_temp2[14]<0.8):
#     # if(phase_temp2[6]<0.8 and phase_temp2[5]<0.8 and phase_temp2[4]<0.8):
#     #     if(phase2_temp2[6]<0.8 and phase2_temp2[5]<0.8 and phase2_temp2[4]<0.8):
#             temppp.append(i)
#             # print(i)
# # temppp = [0,1,2,3,4]
# fig = plt.figure(figsize=(38,10))
# gs = gridspec.GridSpec(1,3)
# ax1 = plt.subplot(1,3,1)
# ax11 = ax1.twinx()
# ax2 = plt.subplot(1,3,2)
# ax22 = ax2.twinx()
# ax3 = plt.subplot(1,3,3)
# ax33 = ax3.twinx()
# # Y2 = test_label_data # 测试使用
# # actual_1 = test_label_data[temppp[0]][0:2]
# # actual_2 = test_label_data[temppp[1]][0:2]
# # actual_3 = test_label_data[temppp[2]][0:2]
# # actual_4 = test_label_data[temppp[0]][6]
# # actual_5 = test_label_data[temppp[1]][6]
# # actual_6 = test_label_data[temppp[2]][6]
# actual_1 = test_label_data[temppp[0]][0:3]
# actual_2 = test_label_data[temppp[1]][0:3]
# actual_3 = test_label_data[temppp[2]][0:3]
# actual_4 = test_label_data[temppp[0]][9:11]
# actual_5 = test_label_data[temppp[1]][9:11]
# actual_6 = test_label_data[temppp[2]][9:11]
# # actual_1 = test_label_data[temppp[0]][0:4]
# # actual_2 = test_label_data[temppp[1]][0:4]
# # actual_3 = test_label_data[temppp[2]][0:4]
# # actual_4 = test_label_data[temppp[0]][12:15]
# # actual_5 = test_label_data[temppp[1]][12:15]
# # actual_6 = test_label_data[temppp[2]][12:15]
# actual_value = np.array([actual_1, actual_2, actual_3, actual_4, actual_5, actual_6], dtype='object')
# # predicted_1 = Y2[temppp[0]][0:2]
# # predicted_2 = Y2[temppp[1]][0:2]
# # predicted_3 = Y2[temppp[2]][0:2]
# # predicted_4 = Y2[temppp[0]][6]
# # predicted_5 = Y2[temppp[1]][6]
# # predicted_6  = Y2[temppp[2]][6]
# predicted_1 = Y2[temppp[0]][0:3]
# predicted_2 = Y2[temppp[1]][0:3]
# predicted_3 = Y2[temppp[2]][0:3]
# predicted_4 = Y2[temppp[0]][9:11]
# predicted_5 = Y2[temppp[1]][9:11]
# predicted_6  = Y2[temppp[2]][9:11]
# # predicted_1 = Y2[temppp[0]][0:4]
# # predicted_2 = Y2[temppp[1]][0:4]
# # predicted_3 = Y2[temppp[2]][0:4]
# # predicted_4 = Y2[temppp[0]][12:15]
# # predicted_5 = Y2[temppp[1]][12:15]
# # predicted_6  = Y2[temppp[2]][12:15]
# predicted_value = np.array([predicted_1, predicted_2, predicted_3, predicted_4, predicted_5, predicted_6], dtype='object')
# X1 = np.arange(3)
# # X2 = np.arange(1,2)
# X2 = np.arange(1,3)
# # X2 = np.arange(1,4)
# bar_width = 0.35
# # tick_label_amp = ["HE$_{11}^e$","HE$_{21}^e$"]
# tick_label_amp = ["HE$_{11}^e$","HE$_{21}^e$","HE$_{12}^e$"]
# # tick_label_amp = ["HE$_{11}^e$","HE$_{21}^e$","HE$_{12}^e$","$TE_{02}$"]
# axx = np.array([ax1,ax2,ax3])
# axxx = np.array([ax11,ax22,ax33])
# for i in range(3):
#     axx[i].bar(X1,actual_value[i],bar_width,color="black",align="center",label="Actual weight")
#     axx[i].bar(X1+bar_width,predicted_value[i],bar_width,color="red",align="center",label="Preditced weight")
#     axx[i].set_xticks(X1+bar_width/2,tick_label_amp)
#     axx[i].set_xticklabels(tick_label_amp,fontsize=28)
#     axx[i].set_yticks([0.0,0.4,0.8,1.2,1.6,2.0])
#     axx[i].set_yticklabels([0.0,0.4,0.8,1.2,1.6,2.0],fontsize=28)
#     axx[i].legend(fontsize=12)
#     axxx[i].bar(X2,actual_value[i+3],bar_width,color="green",align="center",label="Actual phase")
#     axxx[i].bar(X2+bar_width,predicted_value[i+3],bar_width,color="purple",align="center",label="Preditced phase")
#     axxx[i].spines['bottom'].set_position(('data', 0))
#     # axxx[i].set_yticks([-1.5*math.pi,-math.pi,-0.5*math.pi,0,0.5*math.pi,math.pi,1.5*math.pi])
#     axxx[i].set_yticks([-2.0*math.pi,-math.pi,0,math.pi,2.0*math.pi])
#     # axxx[i].set_yticklabels([r'$-1.5\pi$',r'$-\pi$',r'$-0.5\pi$',0,r'$0.5\pi$',r'$\pi$',r'$1.5\pi$'],fontsize=28)
#     axxx[i].set_yticklabels([r'$-2.0\pi$',r'$-\pi$',0,r'$\pi$',r'$2.0\pi$'],fontsize=28)
#     axxx[i].legend(loc='upper left',fontsize=12)
# axx[0].set_ylabel("Modal weigths", fontsize=30)
# axxx[0].set_ylabel("Modal phase", fontsize=30)
# axx[1].set_ylabel("Modal weigths", fontsize=30)
# axxx[1].set_ylabel("Modal phase", fontsize=30)
# axx[2].set_ylabel("Modal weigths", fontsize=30)
# axxx[2].set_ylabel("Modal phase", fontsize=30)
# axx[0].set_title("Group1", fontsize = 32)
# axx[1].set_title("Group2", fontsize = 32)
# axx[2].set_title("Group3", fontsize = 32)
# plt.tight_layout()
# plt.savefig('/tf/桌面/FiOLS/temp_modal/mode66/result/histogram_amp_pha.svg',bbox_inches = 'tight')

# # 画径向偏移、相位偏移柱状图   
# fig = plt.figure(figsize=(38,10))
# gs = gridspec.GridSpec(1,3)
# ax1 = plt.subplot(1,3,1)
# ax11 = ax1.twinx()
# ax2 = plt.subplot(1,3,2)
# ax22 = ax2.twinx()
# ax3 = plt.subplot(1,3,3)
# ax33 = ax3.twinx()
# # actual_1 = test_label_data[temppp[0]][2:4]
# # actual_2 = test_label_data[temppp[1]][2:4]
# # actual_3 = test_label_data[temppp[2]][2:4]
# # actual_4 = test_label_data[temppp[0]][4:6]
# # actual_5 = test_label_data[temppp[1]][4:6]
# # actual_6 = test_label_data[temppp[2]][4:6]
# actual_1 = test_label_data[temppp[0]][3:6]
# actual_2 = test_label_data[temppp[1]][3:6]
# actual_3 = test_label_data[temppp[2]][3:6]
# actual_4 = test_label_data[temppp[0]][6:9]
# actual_5 = test_label_data[temppp[1]][6:9]
# actual_6 = test_label_data[temppp[2]][6:9]
# # actual_1 = test_label_data[temppp[0]][4:8]
# # actual_2 = test_label_data[temppp[1]][4:8]
# # actual_3 = test_label_data[temppp[2]][4:8]
# # actual_4 = test_label_data[temppp[0]][8:12]
# # actual_5 = test_label_data[temppp[1]][8:12]
# # actual_6 = test_label_data[temppp[2]][8:12]
# actual_value = np.array([actual_1, actual_2, actual_3, actual_4, actual_5, actual_6], dtype='object')
# # predicted_1 = Y2[temppp[0]][2:4]
# # predicted_2 = Y2[temppp[1]][2:4]
# # predicted_3 = Y2[temppp[2]][2:4]
# # predicted_4 = Y2[temppp[0]][4:6]
# # predicted_5 = Y2[temppp[1]][4:6]
# # predicted_6  = Y2[temppp[2]][4:6]
# predicted_1 = Y2[temppp[0]][3:6]
# predicted_2 = Y2[temppp[1]][3:6]
# predicted_3 = Y2[temppp[2]][3:6]
# predicted_4 = Y2[temppp[0]][6:9]
# predicted_5 = Y2[temppp[1]][6:9]
# predicted_6  = Y2[temppp[2]][6:9]
# # predicted_1 = Y2[temppp[0]][4:8]
# # predicted_2 = Y2[temppp[1]][4:8]
# # predicted_3 = Y2[temppp[2]][4:8]
# # predicted_4 = Y2[temppp[0]][8:12]
# # predicted_5 = Y2[temppp[1]][8:12]
# # predicted_6  = Y2[temppp[2]][8:12]
# predicted_value = np.array([predicted_1, predicted_2, predicted_3, predicted_4, predicted_5, predicted_6], dtype='object')
# X1 = np.arange(3)
# X2 = np.arange(3)
# bar_width = 0.35
# # tick_label_radio_offset = ["HE$_{11}^e$","HE$_{21}^e$"]
# tick_label_radio_offset = ["HE$_{11}^e$","HE$_{21}^e$","HE$_{12}^e$"]
# # tick_label_radio_offset = ["HE$_{11}^e$","HE$_{21}^e$","HE$_{12}^e$","$TE_{02}$"]
# axx = np.array([ax1,ax2,ax3])
# axxx = np.array([ax11,ax22,ax33])
# for i in range(3):
#     axx[i].bar(X1,actual_value[i],bar_width,color="black",align="center",label="Actual radio offset")
#     axx[i].bar(X1+bar_width,predicted_value[i],bar_width,color="red",align="center",label="Preditced radio offset")
#     axx[i].set_xticks(X1+bar_width/2,tick_label_radio_offset)
#     axx[i].set_xticklabels(tick_label_radio_offset,fontsize=28)
#     axx[i].set_yticks([0.0,0.4,0.8,1.2,1.6,2.0])
#     axx[i].set_yticklabels([0.0,0.4,0.8,1.2,1.6,2.0],fontsize=28)
#     axx[i].legend(fontsize=12)
#     axxx[i].bar(X2,actual_value[i+3],bar_width,color="green",align="center",label="Actual phase offset")
#     axxx[i].bar(X2+bar_width,predicted_value[i+3],bar_width,color="purple",align="center",label="Preditced phase offset")
#     axxx[i].spines['bottom'].set_position(('data', 0))
#     # axxx[i].set_yticks([-1.5*math.pi,-math.pi,-0.5*math.pi,0,0.5*math.pi,math.pi,1.5*math.pi])
#     # axxx[i].set_yticklabels([r'$-1.5\pi$',r'$-\pi$',r'$-0.5\pi$',0,r'$0.5\pi$',r'$\pi$',r'$1.5\pi$'],fontsize=28)
#     axxx[i].set_yticks([-2.0*math.pi,-math.pi,0,math.pi,2.0*math.pi])
#     axxx[i].set_yticklabels([r'$-2.0\pi$',r'$-\pi$',0,r'$\pi$',r'$2.0\pi$'],fontsize=28)
#     axxx[i].legend(loc='upper left',fontsize=12)
# axx[0].set_ylabel("Modal weigths", fontsize=30)
# axxx[0].set_ylabel("Modal phase", fontsize=30)
# axx[1].set_ylabel("Modal weigths", fontsize=30)
# axxx[1].set_ylabel("Modal phase", fontsize=30)
# axx[2].set_ylabel("Modal weigths", fontsize=30)
# axxx[2].set_ylabel("Modal phase", fontsize=30)
# axx[0].set_title("Group1", fontsize = 32)
# axx[1].set_title("Group2", fontsize = 32)
# axx[2].set_title("Group3", fontsize = 32)
# plt.tight_layout()
# plt.savefig('/tf/桌面/FiOLS/temp_modal/mode66/result/histogram_offset.svg',bbox_inches = 'tight')

# # 画光斑图
# corr_temp = []
# for i in range(1000):
#     # print(i, np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4))
#     if((np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4)<0.90) and (np.round(corr2(np.abs(restore_data[i])**2, np.abs(orignal_data[i])**2), 4)>0.80)):
#       corr_temp.append(i)
# fig, ax = plt.subplots(3, 5, figsize = (8,3))
# ax = ax.flatten()
# x1 = to_one(np.abs(orignal_data[corr_temp[0]]))
# x2 = to_one(np.abs(restore_data[corr_temp[0]]))
# x3 = to_one(np.abs(orignal_data[corr_temp[1]]))
# x4 = to_one(np.abs(restore_data[corr_temp[1]]))
# x5 = to_one(np.abs(orignal_data[corr_temp[2]]))
# x6 = to_one(np.abs(restore_data[corr_temp[2]]))
# x7 = to_one(np.abs(orignal_data[corr_temp[3]]))
# x8 = to_one(np.abs(restore_data[corr_temp[3]]))
# x9 = to_one(np.abs(orignal_data[corr_temp[4]]))
# x10 = to_one(np.abs(restore_data[corr_temp[4]]))
# temp = np.array([x1, x3, x5, x7, x9, x2, x4, x6, x8, x10, np.abs(x1 - x2), np.abs(x3 - x4), np.abs(x5 - x6), np.abs(x7 - x8), np.abs(x9 - x10)])
# norm = matplotlib.colors.Normalize(vmin = 0.0, vmax = 1.0)
# print(np.mean(np.abs(x1 - x2)), np.max(np.abs(x1 - x2)))
# print(np.mean(np.abs(x3 - x4)), np.max(np.abs(x3 - x4)))
# print(np.mean(np.abs(x5 - x6)), np.max(np.abs(x5 - x6)))
# print(np.mean(np.abs(x7 - x8)), np.max(np.abs(x7 - x8)))
# print(np.mean(np.abs(x9 - x10)), np.max(np.abs(x9 - x10)))
# cmap = plt.cm.jet
# for i in range(15):
#     img = temp[i]
#     im = ax[i].imshow(img, norm = norm, cmap = cmap)
#     ax[i].set_xticks([])
#     ax[i].set_yticks([])
# ax[0].text(-140,80,'Actual',fontsize=13)
# ax[5].text(-250,80,'Reconstructed',fontsize=13)
# ax[10].text(-175,80,'Residual',fontsize=13)
# ax[10].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[0]])**2, np.abs(orignal_data[corr_temp[0]])**2), 4),fontsize=10)
# ax[11].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[1]])**2, np.abs(orignal_data[corr_temp[1]])**2), 4),fontsize=10)
# ax[12].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[2]])**2, np.abs(orignal_data[corr_temp[2]])**2), 4),fontsize=10)
# ax[13].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[3]])**2, np.abs(orignal_data[corr_temp[3]])**2), 4),fontsize=10)
# ax[14].set_xlabel(np.round(corr2(np.abs(restore_data[corr_temp[4]])**2, np.abs(orignal_data[corr_temp[4]])**2), 4),fontsize=10)
# plt.colorbar(im, ax =[ax[i] for i in range(15)], ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# plt.savefig('/tf/桌面/FiOLS/temp_modal/mode66/result/photomap.svg',bbox_inches = 'tight')

