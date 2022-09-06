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


def create_csv_file(path_to_file):
    path = Path(path_to_file)
    if not path.is_file():
        list_names = ['experiment_name','accuracy', 'binary_accuracy', 'mean_iou', 'jaccard', 'dice_coef','iou_metric']
        with open(path, 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(list_names)
            f_object.close()

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


def testModel(model, X_test, Y_test, batchSize, experiment_name, threshold):
    result_folder = "../results/" + experiment_name + "_" + str(threshold)
    create_dir(result_folder)
    
    csv_filename = "../results/" + "test_results.csv"
    create_csv_file(csv_filename)
    
    jacard = 0
    dice = 0
    results = []
    record_results = pd.DataFrame()
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    #yp = np.round(yp,0)
    yp = np.int32(yp > threshold)
    yp = np.round(yp,0)
    smooth = 1e-15
    for i in range(10):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        #intersection = yp[i].ravel() * Y_test[i].ravel()
        #union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jaccard = iou_metric(Y_test[i], yp[i])  
        plt.suptitle('F1 Score computed as:  '+str(jaccard))

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
    
    acc, binary_acc, mean_iou = evaluateModel(yp,Y_test)
    results.append(experiment_name)
    results.append(acc)
    results.append(binary_acc)
    results.append(mean_iou)
    results.append(jaccard)
    results.append(dice)
    results.append(iou_all)
    
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