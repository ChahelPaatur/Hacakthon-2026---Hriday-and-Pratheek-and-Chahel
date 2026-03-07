# Regression with Huber loss (robust to outliers)

task regression
predict price
inputs size bedrooms bathrooms age zipcode
loss huber
optimizer adamw
learning_rate 0.001
epochs 80
layers 256 128 64
dropout 0.2
batch_norm true
batch_size 32
seed 123
early_stop 12
validate 0.2
dataset housing.csv
