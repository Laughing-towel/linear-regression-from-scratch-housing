import pandas as pd

# Load the housing dataset
housing = pd.read_csv("Housing.csv")

# These columns are yes/no in the CSV, so we convert them into numbers.
# The model cannot learn directly from text like "yes" and "no".
yes_no_columns = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]

for column in yes_no_columns:
    housing[column] = housing[column].map({"yes": 1, "no": 0})

# Convert furnishingstatus into separate binary columns.
# This avoids giving fake numeric meaning to categories like furnished/semi-furnished/unfurnished.
housing = pd.get_dummies(housing, columns=["furnishingstatus"], drop_first=True)

# Scale numerical features so gradient descent does not get dominated by large values like area.
# Min-max scaling puts each feature roughly between 0 and 1.
numeric_columns = ["area", "bedrooms", "bathrooms", "stories", "parking"]

for column in numeric_columns:
    housing[column] = (housing[column] - housing[column].min()) / (
        housing[column].max() - housing[column].min()
    )

# Scale price into millions.
# This keeps the target smaller and makes training less unstable.
housing["price"] = housing["price"] / 1_000_000

# Shuffle the dataset before splitting so the training/test split is random.
# random_state makes the shuffle repeatable (it's a seed)
housing_shuffled = housing.sample(frac=1, random_state=42)

# 80/20 train-test split
split_index = int(0.8 * len(housing_shuffled))

train_data = housing_shuffled[:split_index]
test_data = housing_shuffled[split_index:]

# Every column except price is used as an input feature.
feature_columns = [column for column in housing.columns if column != "price"]

# Initialise one weight per feature.
# Starting from zero is fine for this simple linear regression model.
weights = {}

for column in feature_columns:
    weights[column] = 0

# Bias is the base prediction before feature effects are added.
b = 0

learning_rate = 0.001
epochs = 1500 # that's around the area the model converges, this variable used to be 5000.

# Baseline model: always predict the average training price.
# Useful later to check if our model is actually better than a dumb guess.
average_price = train_data["price"].mean()


def predict(row, weights, b):
    prediction = b

    # Add each feature's contribution: weight * feature value
    for column in feature_columns:
        prediction += weights[column] * row[column]

    return prediction


def mse_loss(prediction, actual):
    # Squared error punishes larger mistakes more heavily.
    return (prediction - actual) ** 2


if __name__ == "__main__":
    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for _, row in train_data.iterrows():
            prediction = predict(row, weights, b)
            actual = row["price"]

            error = prediction - actual
            total_loss += mse_loss(prediction, actual)

            # Update every feature weight using the current error.
            # This is the manual gradient descent part.
            for column in feature_columns:
                weights[column] = weights[column] - learning_rate * error * row[column]

            # Update the bias separately.
            b = b - learning_rate * error

        if epoch % 500 == 0:
            print(f"epoch: {epoch}")
            print(f"training loss: {total_loss:.4f}")
            print()

    # Testing
    # At this point the model is frozen. No more weight updates. We just testing now.
    test_loss = 0

    for _, row in test_data.iterrows():
        prediction = predict(row, weights, b)
        actual = row["price"]

        test_loss += mse_loss(prediction, actual)

        print(f"prediction: {prediction:.2f}, actual: {actual:.2f}, error: {prediction - actual:.2f}")

    # RMSE converts the squared loss back into the same unit as the target.
    # Since price was scaled to millions, we multiply by 1,000,000 to get pounds.
    rmse = (test_loss / len(test_data)) ** 0.5
    rmse_pounds = rmse * 1_000_000

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"RMSE: £{rmse_pounds:.2f}")