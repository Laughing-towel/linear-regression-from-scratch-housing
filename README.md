# Linear Regression From Scratch: Housing Prices

CRUDE and BARBARIC linear regression model built from scratch because I'm learning linear regression from scratch.

This project predicts house prices using a manually implemented linear regression model. No scikit-learn model fitting, no PyTorch, no `model.fit()`. The point was to understand the actual training loop myself.

Dataset: https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data

## What this project does

* Loads a housing CSV dataset
* Converts yes/no columns into `1` and `0`
* One-hot encodes the `furnishingstatus` column
* Scales numerical features so gradient descent behaves better
* Splits the data into 80% training and 20% testing
* Trains a linear regression model from scratch using gradient descent
* Compares the model against a simple baseline that always predicts the average house price

## Model

The model uses multiple linear regression:

```text
price = w1x1 + w2x2 + ... + b
```

Each feature gets its own weight, and the bias handles the base prediction when feature values are zero.

## Results

The handmade model got roughly:

```text
Model RMSE: £1.116m
Baseline RMSE: £1.877m
```

So it actually learned something useful compared to just guessing the average price every time.

## Limitations

This is very much a learning project. The implementation is intentionally manual and not optimised. The model is also only linear, so it cannot capture more complex relationships in house prices.
