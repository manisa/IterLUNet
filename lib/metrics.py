import os
from pathlib import Path
from csv import writer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def jaccard(y_true, y_pred):
    def f(y_true, y_pred):
        smooth = 1e-15
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def jacard_dice(yp, Y_test):
    jacard = 0
    dice = 0
    smooth = 0.0
    for i in range(len(Y_test)):
        flat_pred = K.flatten(Y_test[i])
        flat_label = K.flatten(yp[i])
        
        intersection_i = K.sum(flat_label * flat_pred)
        union_i = K.sum( flat_label + flat_pred - flat_label * flat_pred)
        
        dice_i = (2. * intersection_i + smooth) / (K.sum(flat_label) + K.sum(flat_pred) + smooth)
        jacard_i = intersection_i / union_i
        
        jacard += jacard_i
        dice += dice_i

    jacard /= len(Y_test)
    dice /= len(Y_test)
    print(jacard.numpy())
    print(dice.numpy())
    
    return jacard.numpy(), dice.numpy()

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    smootth = smooth = 1e-15
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    smooth = 1e-15
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    smooth = 1e-15
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


def iou_metric(y_true_in, y_pred_in):
    labels = y_true_in
    y_pred = y_pred_in

    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))

    intersection = temp1[0]

    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
  
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    iou = intersection / union
    iou = str(iou)[2:-2]
    iou = float(iou)
    return iou