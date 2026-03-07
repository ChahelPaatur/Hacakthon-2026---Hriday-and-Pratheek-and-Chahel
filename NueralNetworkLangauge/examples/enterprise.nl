# Enterprise-grade NeuroLang program
# Demonstrates all language features

task classification
predict species
inputs petal_length petal_width sepal_length sepal_width

dataset iris.csv
loss cross_entropy
optimizer adam
learning_rate 0.0008
epochs 100

learn deep
batch_norm true
dropout 0.25
activation relu

batch_size 16
normalize true
split 0.8
seed 42

early_stop 8
lr_schedule cosine
validate 0.15

export "./trained-model"
