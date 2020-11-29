# coralml

This is the repository for the ImageCLEF team at the [NOAA-NVIDIA](https://www.gpuhackathons.org/event/noaa-gpu-hackathon) hackathon

## Data

In order to obtain the training data (for the 2020 ImageCLEF competition): 
```bash
wget https://annotator.uk/data/imageCLEFcoral2020_training_v4.zip
```
And unzip it using the password provided by @aCampello, for example with 

```bash
unzip -j -P <password> imageCLEFcoral2020_training_v4.zip -d data/images
```


## Clef2019 code

### Installation (pytorch):

Clone this repository and `cd imageclef-2019-code`

Clone git repository for deeplabv3+:

```bash	
mkdir src
cd src
git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
cd ..
```

Create python environment (e.g. with conda) and activate it

```bash
virtualenv -p python3.7 env/
. env/bin/activate
```

```bash
pip install torch torchvision cudatoolkit==9.0 #For gpu
```

Install requirements and package
```bash
pip install -r requirements.txt
pip install -e .
```

### Installation (tensorflow)

In addition, the tensorflow installation requires protobuf and:

```bash
pip install -U --pre tensorflow=="2.2.0"
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .
```

### Usage

First move all the CLEF images to `data/images` and the csv with clef annotations in `data/annotations.csv`. Then create masks with

```bash
python coralml/data/create_masks.py
``` 

And subsequently split into train and validation

```bash
python coralml/data/data_split.py
```

Modify the file `data/instruction.json` to change hyperparameters of the network and train with:

```bash
python -m coralml
```

This will generate `data_train.json` and `data_valid.json` which will be used to train.

	
	~~~~
	conda create --name coralml python==3.6.7
	source activate coralml
	~~~~
	
## install pytorch
	---- Cuda available, Cuda version 9.0: ----
	~~~~	
	conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
	~~~~
    
    OR
    
	---- Cuda not available: ---
	~~~~	
	conda install pytorch-cpu torchvision-cpu -c pytorch
	~~~~

# install requirements
```bash
	pip install --upgrade pip
	pip install -r requirements.txt
```
