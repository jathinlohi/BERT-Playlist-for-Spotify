from datasets import load_dataset
import pandas as pd

# Load from HuggingFace
dataset = load_dataset("go_emotions")

def save_clean_csv(split, path):
    df = pd.DataFrame(dataset[split])
    # Convert labels list to proper string like '[0, 18]'
    df["labels"] = df["labels"].apply(lambda x: str(x))
    df.to_csv(path, index=False)

save_clean_csv("train", "data/goemotions_train.csv")
save_clean_csv("validation", "data/goemotions_val.csv")
save_clean_csv("test", "data/goemotions_test.csv")

