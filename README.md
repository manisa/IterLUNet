## IterLUnet: Deep Learning Architecture for Pixel-Wise Crack Detection in Levee Systems

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

You would need to install the following software before replicating this framework in your local or server machine.

```
Python version 3.7+
Aanaconda version 3+
TensorFlow version 2.6.0
Keras version 2.6.0

```

## Download and install code
- Retrieve the code
```
git clone https://github.com/manisa/IterLUNet.git
cd IterLUNet
```

- Create and activate the virtual environment with python dependendencies. 
```
conda create -n gpu-tf tensorflow-gpu
conda activate gpu-tf
source installPackages.sh

```

## Folder Structure
```
IterLUNet/
	archs/
	lib/
	dataset/
		experiment_3/
	models/
		experiment_3/
```

## Download datasets
- Experiment 1 [train data](https://drive.google.com/file/d/1Jk6VLWVBTBfVPI0jbxuftNDFLEVfqsXs/view?usp=sharing) and [test data](https://cs.uno.edu/~mpanta1/IterLUNet/datasets/exp_1_test.zip)
- Experiment 2 [train and validation data](https://cs.uno.edu/~mpanta1/IterLUNet/datasets/train_valid_data.zip) and [test_data](https://cs.uno.edu/~mpanta1/IterLUNet/datasets/exp_2_test.zip)
- Experiment 3 [train and validation data](https://cs.uno.edu/~mpanta1/IterLUNet/datasets/levee_augmented_IEEEAccessData.zip) and [test_data](https://cs.uno.edu/~mpanta1/IterLUNet/datasets/exp_3_test.zip)
- Unzip and copy dataset from the respecitve experiment into the folder **dataset** inside the root folder **IterLUNet**.
- Your directory structure should look like this:

```
IterLUNet/
	dataset/
		experiment_3/
			train/
				images/
				masks/
			test/
				images/
				masks/
```

## Download trained models
- [Best performing trained models from 10-Fold CV Experiment](https://cs.uno.edu/~mpanta1/IterLUNet/models/exp_1_models.zip)
- [Experiment 2 best performing trained models over 80 epochs](https://cs.uno.edu/~mpanta1/IterLUNet/models/exp_2_models.zip)
- [Experiment 3 best performing trained models over 150 epochs](https://cs.uno.edu/~mpanta1/IterLUNet/models/exp_3_models.zip)
- Unzip and copy models from respective experiment to **models** inside the root folder **IterLUNet**.
- Your directory structure should look like this:

```
IterLUNet/
	models/
		experiment_1/
		experiment_2/
		experiment_3/
```

## Training
- To replicate the training procedure, follow following command line.
```
cd src
python train.py --model_type=iterlunet --input_filters=64 --lr=2e-3 --loss_function='focal_tversky_loss' --model_path='./models/iterlunet'  --train_valid_path='./datasets/experiment_3/train/'

```

## Authors
Manisha Panta, Md Tamjidul Hoque, Mahdi Abdelguerfi, Maik C. Flanagin
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
