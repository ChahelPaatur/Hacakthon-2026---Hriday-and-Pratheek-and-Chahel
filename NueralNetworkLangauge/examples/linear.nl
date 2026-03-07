# Simple linear regression — no hidden layers
# Great for quick baselines or small datasets

task regression
predict price
inputs size bedrooms
loss mse
optimizer sgd
learning_rate 0.01
epochs 100
learn linear
dataset housing.csv
