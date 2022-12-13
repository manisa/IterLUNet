import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys; sys.path.insert(0, '..')
from lib.metrics import jaccard, tversky, dice_coef, dice_loss, bce_dice_loss, tversky, focal_tversky_loss, tversky_loss, create_dir
from lib.load_data import get_data, get_train_test_augmented
from lib.plot import plot_loss_dice_history, plot_dice_jacc_history
from lib.evaluate import testModel

from archs.unet import UNet
from archs.multiresunet import MultiResUnet
from archs.attentionunet import AttUNet
from archs.nestedunet import NestedUNet
from archs.iterlunet import IterLUNet

import cv2
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger


print('TensorFlow version: {version}'.format(version=tf.__version__))
print('Keras version: {version}'.format(version=tf.keras.__version__))
print('Eager mode enabled: {mode}'.format(mode=tf.executing_eagerly()))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))


def compile_and_train_model(config, train_images):

	if config.model_type == "iterlunet":
		model = IterLUNet(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

	elif config.model_type == "unet":
		model = UNet(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

	elif config.model_type == "multiresunet":
		model = MultiResUnet(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

	elif config.model_type == "attentionunet":
		model = AttUNet(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)

	elif config.model_type == "nestedunet":
		model = NestedUNet(input_filters=config.input_filters, height=config.img_height, width=config.img_width, n_channels=config.img_ch)
		

	# Defining optimizer
	if config.optimizer == "Adam":
		optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr, beta_1=config.beta1, beta_2=config.beta2)
	elif confg.optimizer == "Nadam":
		optimizer = tf.keras.optimizers.Nadam(learning_rate=config.lr, beta_1=config.beta1, beta_2=config.beta2)
	elif config.optimizer == "RMSProp":
		optimizer = tf.keras.optimizers.RMSprop(learning_rate=config.lr, momentum=config.beta1)


		# Defining loss functions
	if config.loss_function == "bce":
		model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	elif config.loss_function == "dice_loss":
		model_loss = dice_loss
	elif config.loss_function == "bce_dice_loss":
		model_loss = bce_dice_loss
	elif config.loss_function == "tversky_loss":
		model_loss = tversky_loss
	elif config.loss_function == "focal_tversky_loss":
		model_loss = focal_tversky_loss

	# define metrics
	metrics =[dice_coef, jaccard, tversky, 'accuracy']
	model.compile(loss=model_loss, optimizer=optimizer, metrics=metrics)
	print(f'model created and compiled for model {config.model_type}')
	print(model.summary())


	csv_path = config.model_path + "/metrics_" + config.model_type + ".csv"

	# print summary of model
	print('Loading dataset...')
	X, y = get_data(train_images, config.train_valid_path, config.img_height, config.img_width, train=True)
	
	train_generator, valid_generator = get_train_test_augmented(X, y, 
																validation_split=config.valid_perc, 
																batch_size=config.batch_size, seed=config.seed)

	steps_per_epoch = (len(train_images) *(1-config.valid_perc)) // config.batch_size
	val_steps_per_epoch =  (len(train_images) * config.valid_perc) // config.batch_size


	# run training
	callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=8, min_lr=15e-8, verbose=1, mode='min'),
				CSVLogger(csv_path),
				ModelCheckpoint(filepath=os.path.join(config.model_path, 'best_model.h5'), monitor='val_tversky', mode='max', save_best_only=True, save_weights_only=False, verbose=1)]

	
	print('Fitting model...')

	history = model.fit(train_generator, validation_data=valid_generator,
									steps_per_epoch=steps_per_epoch, 
									validation_steps=val_steps_per_epoch,
									epochs=config.num_epochs, callbacks=callbacks, verbose=1)
	tf.keras.backend.clear_session()
	plot_loss_dice_history(history, config.model_type, config.graph_path)
	plot_dice_jacc_history(history, config.model_type, config.graph_path)

	print(f'==================Model {config.model_type} training completed====================')
	#atexit.register(strategy._extended._collective_ops._pool.close)



def main(config):
	if config.model_type not in ['unet','multiresunet','attentionunet','nestedunet', 'iterlunet']:
		print('ERROR!! model_type should be selected in unet/multiresunet/attentionunet/nestedunet/iterlunet')
		print('Your input for model_type was %s'%config.model_type)
		return

	# Create all the directories if not existed
	# Create directories if not exist
	create_dir(config.model_path)
	create_dir(config.result_path)
	create_dir(config.graph_path)
	model_path =  config.model_type + "_" + config.loss_function 
	config.model_path = os.path.join(config.model_path, model_path)
	create_dir(config.model_path)
	print(config)

	
	# load data X and Y 
	train_images = sorted(glob(os.path.join(config.train_valid_path, "images/*")))
	compile_and_train_model(config, train_images)


if __name__ == '__main__':
	# model hyper-parameters

	parser = argparse.ArgumentParser()
	parser.add_argument('--img_width', type=int, default=256)
	parser.add_argument('--img_height', type=int, default=256)

	# training hyper-parameters
	parser.add_argument('--img_ch', type=int, default=3)
	parser.add_argument('--output_ch', type=int, default=1)
	parser.add_argument('--input_filters', type=int, default=64)

	parser.add_argument('--num_epochs', type=int, default=150)
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--lr', type=float, default=2e-3)
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--loss_function', type=str, default='focal_tversky_loss')
	parser.add_argument('--beta1', type=float, default=0.9)        # momentum1 in Adam
	parser.add_argument('--beta2', type=float, default=0.9)      # momentum2 in Adam    

	parser.add_argument('--model_type', type=str, default='iterlunet', help='unet/multiresunet/attentionunet/nestedunet/iterlunet')
	parser.add_argument('--model_path', type=str, default='./models/')
	parser.add_argument('--graph_path', type=str, default='./models/focal_tversky_loss_metric_graphs')
	parser.add_argument('--result_path', type=str, default='./results/')

	parser.add_argument('--train_valid_path', type=str, default='./datasets/experiment_3/train/')
	parser.add_argument('--test_path', type=str, default='./datasets/experiment_3/test/')
	parser.add_argument('--valid_perc', type=float, default=0.2)
	

	parser.add_argument('--seed', type=int, default=2021)

	config = parser.parse_args()
	main(config)
