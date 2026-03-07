# Housing Price Regression
# Predicts price from property features

task regression
predict price
inputs size bedrooms bathrooms age zipcode
loss mse
optimizer adam
epochs 60
learn nonlinear
layers 128 64 32
batch_size 16
normalize true
split 0.8
dataset housing.csv
