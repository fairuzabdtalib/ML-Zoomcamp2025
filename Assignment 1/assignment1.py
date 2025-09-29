import polars as pl
import numpy as np

data = pl.read_csv(r"Assignment 1\car_fuel_efficiency.csv")
data.head()

## Q2. Records count
# How many records are in the dataset?
unique_fuel_type = data.select('fuel_type').unique().count()
print(f"There are {unique_fuel_type} unique fuel types in the dataset")

# Q3. Fuel types
# How many fuel types are presented in the dataset?
unique_fuel_type = data.select('fuel_type').unique().count()

# Q4. Missing values
# How many columns in the dataset have missing values?
missing_values_per_column = data.null_count()
print(f"There are {missing_values_per_column} missing values per column")

# Q5. Max fuel efficiency
# What's the maximum fuel efficiency of cars from Asia?
max_fuel_efficiency_Asia = (
    data
    .filter(pl.col('origin') == 'Asia')
    .select('fuel_efficiency_mpg').max())
print(f"The maximum fuel efficiency of cars from Asia is {max_fuel_efficiency_Asia}")

## Q6. Median value of horsepower
# Find the median value of the horsepower column in the dataset.
# Next, calculate the most frequent value of the same horsepower column.
# Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
# Now, calculate the median value of horsepower once again.

# Step 1: Find the median of horsepower
median_before = data.select(pl.col("horsepower").median())

# Step 2: Find the most frequent (mode) value of horsepower
mode_val = (
    data
    .filter(pl.col("horsepower").is_not_null())
    .group_by("horsepower")
    .len()
    .sort("len", descending=True)
    .select("horsepower")
    .head(1)             # take top row
    .item()              # extract scalar
)

# Step 3: Fill missing horsepower with the mode value (wrap with pl.lit)
df_filled = data.with_columns(
    pl.col("horsepower").fill_null(pl.lit(mode_val)).alias("horsepower")
)

# Step 4: Find the median again after filling missing values
median_after = df_filled.select(pl.col("horsepower").median())

print("Median horsepower before filling:", median_before.item())
print("Most frequent horsepower value:", mode_val)
print("Median horsepower after filling:", median_after.item())

# Q7
# Select all the cars from Asia
# Select only columns vehicle_weight and model_year
# Select the first 7 values
# Get the underlying NumPy array. Let's call it X.
# Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# Invert XTX.
# Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# What's the sum of all the elements of the result?

# Step 1: Select cars from Asia
df_asia = data.filter(pl.col("origin") == "Asia")

# Step 2: Select only vehicle_weight and model_year columns
df_asia_small = df_asia.select(["vehicle_weight", "model_year"])

# Step 3: Take the first 7 values
df_top7 = df_asia_small.head(7)

# Step 4: Convert to NumPy array
X = df_top7.to_numpy()

# Step 5: Compute XTX = X.T @ X
XTX = X.T @ X

# Step 6: Invert XTX
XTX_inv = np.linalg.inv(XTX)

# Step 7: Create array y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# Step 8: Compute w = (XTX_inv @ X.T) @ y
w = XTX_inv @ X.T @ y

# Step 9: Sum of all elements in w
result_sum = w.sum()

print("w:", w)
print("Sum of elements in w:", result_sum)