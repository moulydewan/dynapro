# always use 
# conda activate dynapro
# which pip
import pandas as pd
import os

# always use conda activate dynapro
from data import datasets_info

dataset_cls = datasets_info["medium"]["class"]
dataset = dataset_cls().to_hf_dataset()

print(dataset)

# create output folder
os.makedirs("output", exist_ok=True)

# save train and test as separate csv files
train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

train_df.to_csv("output/medium_train.csv", index=False)
test_df.to_csv("output/medium_test.csv", index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("---")
print(train_df.head(2))