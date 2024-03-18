# KoBERT Based Emotion Classification
This project was carried out in vscode.

If you want to proceed with this project as a whole, proceed with the preprocessing process in the Dataset folder first.

In this project, seven emotions are classified through text.

- angry
- sadness
- fear
- disgust
- neutral
- happiness
- surprise

## Modeling
The entire modeling process was carried out in the Modeling.py file.

For modeling to proceed, the following files must exist in the root folder.

- train_data.pkl
- test_data.pkl

This file is a file that has undergone preprocessing in the Dataset folder.

#### When using a virtual environment
Currently, Python is version 3.8.18

Also, cuda runs on version 11.6.

Follow the below method in anaconda terminal.

Clone this repository:

```
git clone https://github.com/Oneul-hyeon/Emotion_Classification_Using_KoBERT.git
cd {location of that folder}
```

Install Python and other dependencies:

```
conda create -n {env_name} python=3.8
conda activate {env_name}
pip install -r requirements.txt
```

And we need to resolve the compatibility issue between PyTorch and CUDA.

```
conda install pytorch==1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

This file is the overall library and package installation file of the AI part of this project.

This process allows you to run the Modeling.py
