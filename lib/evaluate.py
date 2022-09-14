import os
from pathlib import Path
from csv import writer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

import matplotlib.font_manager

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_csv_file(path_to_file):
    path = Path(path_to_file)
    if not path.is_file():
        list_names = ['experiment_name','accuracy', 'binary_accuracy', 'mean_iou', 'jaccard', 'dice_coeff','precision', 'recall', 'f1-score']
        with open(path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(list_names)
            f_object.close()

def f1(y_true, y_pred):
    '''
    Calculates the F1 by using keras.backend
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    print(precision.numpy(), recall.numpy(), f1_score.numpy())
    return precision.numpy(), recall.numpy(), f1_score.numpy()

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
    return iou

def iou_metric_batch(y_true_in, y_pred_in):
    y_pred_in = y_pred_in
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def evaluateModel(yp, Y_test):
    flat_pred = K.flatten(Y_test)
    flat_label = K.flatten(yp)

    binaryacc = tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.5)
    acc = tf.keras.metrics.Accuracy()
    #auc = tf.keras.metrics.AUC()
    miou = tf.keras.metrics.MeanIoU(num_classes=2)
    
    r1 = binaryacc.update_state(flat_label,flat_pred)
    r1 = binaryacc.result().numpy()
    
    r2 = acc.update_state(flat_label,flat_pred)
    r2 = acc.result().numpy()
    
    r3 = miou.update_state(flat_label,flat_pred)
    r3 = miou.result().numpy()
    
    return r2, r1, r3


def testModel(model, X_test, Y_test, batchSize, experiment_name):
    result_folder = "../results/" + experiment_name
    create_dir(result_folder)
    
    csv_filename = "../results/" + "test_results.csv"
    create_csv_file(csv_filename)
    
    jacard = 0
    dice = 0
    results = []
    record_results = pd.DataFrame()
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    #yp = np.round(yp,0)
    #yp = np.int32(yp > threshold)
    yp = np.round(yp,0)
    smooth = 1e-15
    for i in range(10):
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        fig.subplots_adjust(top=0.95)
        axs[0].imshow(X_test[i])
        axs[1].imshow(Y_test[i])
        axs[2].imshow(yp[i])
        
        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jaccard = ((np.sum(intersection) + smooth)/(np.sum(union) + smooth))
        iou = iou_metric(Y_test[i], yp[i])
        iou = str(iou)[2:-2]
        
        #axs[0].title.set_text('Image')
        axs[0].set_title("Image",fontsize=14)
        axs[0].set_axis_off()    
        #axs[1].title.set_text('Ground Truth')
        axs[1].set_title("Ground Truth", fontsize=14)
        axs[1].set_axis_off()    
        #axs[2].title.set_text('Prediction')
        axs[2].set_title("Prediction", fontsize=14)
        axs[2].set_axis_off()


        fig.tight_layout()
        plt.suptitle('F1 Score computed as: ' + str(jaccard) + ' or ' + str(iou), fontweight='regular', fontsize=16)
        plt.savefig(result_folder + "/" + str(i)+'.png',format='png')
        plt.close()
        
    
    jaccard = 0.0
    dice = 0.0
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jaccard += ((np.sum(intersection) + smooth)/(np.sum(union) + smooth))  
        
        dice += (2. * np.sum(intersection)+smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)

    jaccard /= len(Y_test)
    dice /= len(Y_test)
    iou_all = iou_metric_batch(Y_test, yp)
    precision, recall, f1_score = f1(Y_test, yp)
    acc, binary_acc, mean_iou = evaluateModel(yp,Y_test)
    results.append(experiment_name)
    results.append("{:.2%}".format(round(acc,4)))
    results.append("{:.2%}".format(round(binary_acc,4)))
    results.append("{:.2%}".format(round(mean_iou,4)))
    results.append("{:.2%}".format(round(jaccard,4)))
    results.append("{:.2%}".format(round(dice,4)))
    results.append("{:.2%}".format(round(precision,4)))
    results.append("{:.2%}".format(round(recall,4)))
    results.append("{:.2%}".format(round(f1_score,4)))
    
    with open(csv_filename, 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(results)
        f_object.close()
    
    print('Test Jacard Index : '+str(jaccard))
    print('Test Dice Coefficient : '+str(dice))
    print('Test iou_all :' + str(iou_all))
    return results


def draw_get_best_threshold(ious, thresholds):
    """
    Returns threshold_best, iou_best
    """
    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    return threshold_best, iou_best

    
def saveResultsOnly(model, X_test, batchSize, experiment_name):
    result_folder = "../results/" + experiment_name
    create_dir(result_folder)
    
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)
    
    for i in range(len(X_test)):
        fig, axs = plt.subplots(1, 3, figsize=(20, 10))
        fig.subplots_adjust(top=0.95)
        axs[0].imshow(X_test[i])
        axs[1].imshow(Y_test[i])
        axs[2].imshow(yp[i])
        
        #axs[0].title.set_text('Image')
        axs[0].set_title("Image",fontsize=14)
        axs[0].set_axis_off()    
        #axs[1].title.set_text('Ground Truth')
        axs[1].set_title("Ground Truth", fontsize=14)
        axs[1].set_axis_off()    
        #axs[2].title.set_text('Prediction')
        axs[2].set_title("Prediction", fontsize=14)
        axs[2].set_axis_off()


        fig.tight_layout()
        plt.suptitle('Original image, ground-truth and predicted mask by : ' + str(experiment_name), fontweight='regular', fontsize=16)
        plt.savefig(result_folder + "/" + str(i)+'.png',format='png')
        plt.close()

        