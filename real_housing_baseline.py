from real_housing import test_data, average_price

baseline_loss = 0

for _, row in test_data.iterrows():
    actual = row["price"]

    baseline_loss += (
        average_price - actual
    ) ** 2

baseline_rmse = (
    baseline_loss / len(test_data)
) ** 0.5

print(baseline_rmse)