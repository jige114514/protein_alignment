# protein_alignment
hitsz project for advancing protein sequences alignment using supervised contrastive learning

## Install
To install the model, please clone the repo first:
```
git clone https://github.com/jige114514/protein_alignment.git
```
Then in the folder, you should download the trained dedal model from the following link:
```
https://tfhub.dev/google/dedal/3
```
After that, there will be a folder called dedal_3 in this folder.
Then you may install the necessary requirements by executing:
```
pip install -r requirements.txt
```

## Train
Then you can execute the code by running the following command, you can choose supervised or unsupervised dataset(supervised default):
```
python main.py --dataset supervised
python main.py --dataset unsupervised
```
In the process of executing, the program will output train loss and F1 scores of sequence pairs with distinct PID