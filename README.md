# Melanoma Detection
Detect early-stage of melanoma skin cancer

## Dataset
I used a dataset on Kaggle which you can find [here](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

## Training
I created 2 versions for the training. Use can use Kaggle to train this model or download and use your own PC (but I think you should you Kaggle because of its durability and convenience).

## Using
Download the `model1` folder and run `main.py` with the following struct:
- Have the path to the model folder
- Have name of image files need to be checked

And at the end, we have something like this
```
python main.py C:\Path\model1 melanoma1.jpg melanoma2.jpg
```

If it shows `malignant`, you should go to the hospital and do more tests for suitable treatments.

## Note!!!
This is just early-stage detection, you shouldn't misuse this model. 

Your image should be bright enough and you should crop the image to show the skin part that you suspect has melanoma cancer. If you put other objects in the image, the result won't be correct!
