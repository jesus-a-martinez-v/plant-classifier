# Plant Classifier

Plant classifier implemented using features based on color histograms, in combination with a random forest machine
learning model. 


## Installation

I recommend you have `virtualenv` installed. After that, create a new environment and activate it:

```bash
$> virtualenv -p python3 venv
$> source venv/bin/activate
```

After that, just install the requirements:

```bash
pip install -r requirements.txt
```

## Try it

At the root of the project, execute the following command to get a taste of how it works:

```bash
python classify.py -i ./dataset/images -m ./dataset/masks
```

You'll get a result similar to this:

![Sunflower](https://github.com/jesus-a-martinez-v/plant-classifier/blob/master/dataset/images/image_sunflower_0096.png)

```
              precision    recall  f1-score   support

      crocus       0.92      1.00      0.96        12
       daisy       0.88      0.93      0.90        15
       pansy       1.00      0.85      0.92        20
   sunflower       0.96      1.00      0.98        24

   micro avg       0.94      0.94      0.94        71
   macro avg       0.94      0.95      0.94        71
weighted avg       0.95      0.94      0.94        71

./dataset/images/image_sunflower_0096.png
I think this flower is a SUNFLOWER
```
