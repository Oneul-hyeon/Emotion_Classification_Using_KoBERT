This process is a data preprocessing process for modeling.

The file was processed in Colab.

## Dataset
Datasets for the project can be downloaded from the following link.

In order to use the dataset, it is necessary to apply for the use of the data after signing up as a member.
```
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263
```
The following files were used as datasets used for modeling.

- 5차년도_2차.csv

Save the file to this folder.

## Environment Setting
To run preprocessing.ipynb, library and package installation process is required.

### When using in colab
The code for installing libraries and packages is at the top of preprocessing.ipynb.

You can proceed further by running the code.

### When using a virtual environment
Currently, Colab's Python is version 3.10.

So follow the below method in anaconda terminal.

Clone this repository:

```
git clone https://github.com/toy-f-rebellion/toy_ai.git
cd {location of that folder}
```

Install Python and other dependencies:

```
conda create -n {env_name} python=3.10
conda activate {env_name}
pip install -r requirements.txt
```

This file is the overall library and package installation file of the AI part of this project.

This process allows you to run the preprocessing.ipynb file.

Also, in this case, delete the library and package installation block at the top of the file.

Since this file was created in Colab, it is necessary to reset the file path to load or save in order to proceed in the virtual environment.

### Preprocessed Data
The file should be stored in the root folder.