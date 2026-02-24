from datasets import load_dataset
import os

# Download SQuAD v1.1
print("Downloading SQuAD v1.1 dataset...")
dataset = load_dataset("squad")

print("\nDataset structure:")
print(dataset)

print("\nSample record from training set:")
sample = dataset["train"][0]
print(f"Context: {sample['context'][:200]}...")
print(f"Question: {sample['question']}")
print(f"Answers: {sample['answers']}")

# Save a small sample to data/ for inspection if needed
import json
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

with open(os.path.join(data_dir, "squad_sample.json"), "w") as f:
    json.dump(sample, f, indent=4)

print(f"\nSample saved to {os.path.join(data_dir, 'squad_sample.json')}")
