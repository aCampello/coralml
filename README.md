# coralml

This is the repository for the ImageCLEF team at the [NOAA-NVIDIA](https://www.gpuhackathons.org/event/noaa-gpu-hackathon) hackathon

## Status

|Model|Status|Evaluation CLEF DATA|Evaluation NOAA DATA|
|---|---|---| --- |
|[Deep Segmentation'19](http://www.dei.unipd.it/~ferro/CLEF-WN-Drafts/CLEF2019/paper_151.pdf)| ✅ Implemented| ✅ Test data <br> 🔧 Val data: Implemented (numbers TBC) | 🔧 In progress  |
|Mask-RCNN | ✅ Implemented |  🔧 In progress | ❌ Not Implemented |
|[UNET](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical)| ❌ Not Implemented| ❌ Not Implemented | ❌ Not Implemented |
|DETR| ❌ Not Implemented| ❌ Not Implemented | ❌ Not Implemented |



## Data

In order to obtain the training data (for the 2020 ImageCLEF competition): 
```bash
wget https://annotator.uk/data/imageCLEFcoral2020_training_v4.zip
wget https://annotator.uk/data/annotations-train-NVIDIA-NOAA-2020.zip
wget https://annotator.uk/data/imageCLEFCORAL2020_GT.zip
wget https://annotator.uk/data/imageCLEFcoral2020_test_v4.zip
```
And unzip it using the password provided by @aCampello, for example with 

```bash
unzip -j -P <password> imageCLEFcoral2020_training_v4.zip -d data/images
unzip -j -P <password> annotations-train-NVIDIA-NOAA-2020.zip -d data/images
```

```bash
unzip -j -P <password> imageCLEFCORAL2020_GT.zip -d data
unzip -j -P <password> imageCLEFcoral2020_test_v4.zip -d data/images_val
```


## Clef2019 code

### Installation:

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
pip install torch torchvision cudatoolkit #For gpu
```

Install requirements and package
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Train
First move all the CLEF images to `data/images` and the csv with clef annotations in `data/annotations.csv`. Then create masks with

```bash
python coralml/data/create_masks.py
``` 

And subsequently split into train and validation

```bash
python coralml/data/data_split.py
```

Modify the file `data/instructions.json` to change hyperparameters of the network and train with:

```bash
python -m coralml --instructions data/instructions.json
```

### Evaluate against CLEF 2020

```bash
python coralml/data/create_masks.py --data_folder_path data --image_folder images_val --mask_folder masks_val --annotations_file imageCLEFcoral2020_GT.csv
```

```bash
python coralml/ml/evaluate_clef.py --data_folder_path data --image_folder images_val --mask_folder masks_val --model_path models/test_model/model_best.pth
```

### (To be deprecated): old conda instructions
	
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
