## IterLUnet: Deep Learning Architecture FOR PIXEL-WISE CRACK DETECTION IN LEVEE SYSTEMS 

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

## Download datasets
- Go to [this link](https://drive.google.com/file/d/1Jk6VLWVBTBfVPI0jbxuftNDFLEVfqsXs/view?usp=sharing).
- Click on **LeveeCrack_dataset.zip**. This will automatically download the datasets used to to perform 10FCV.
- Unzip and copy all the datasets from **LeveeCrack_dataset** directory into the folder **LeveeCrack_dataset** inside the root folder **IterLUNet**.
- Your directory structure should look like this:

```
IterLUNet/
    LeveeCrack_dataset/
    	train/
        	images/
        	masks/
        test/
        	images/
        	masks/
```
## Folder Structure
```
IterLUNet/
    archs/
    lib/
    src/
    LeveeCrack_dataset/

```

## Training
- To replicate the training procedure, follow following command line.
```
cd src
python 10FCV_train_iterlunet.py

```

## Authors
Manisha Panta, Md Tamjidul Hoque, Mahdi Abdelguerfi, Maik C. Flanagin
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
