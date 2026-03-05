import json
from datasets import load_dataset

def read_hf_dataset(
    dataset_name: str = "arron666/seq_monkey",
    num_samples: int = 2000,
    output_file: str = "seq_monkey_data.jsonl"
) -> None:
    """
    Read samples from HuggingFace dataset and save to JSONL file.
    Uses streaming mode to avoid loading all data into memory.
    
    Args:
        dataset_name: HuggingFace dataset path
        num_samples: Number of samples to read (default: 2000)
        output_file: Output JSONL file path
    """
    print(f"Loading dataset: {dataset_name} (streaming mode)")
    dataset = load_dataset(dataset_name, streaming=True)
    
    print(f"Available splits: {list(dataset.keys())}")
    
    if "train" in dataset:
        split = "train"
    else:
        split = list(dataset.keys())[0]
    
    print(f"Using split: {split}")
    print(f"Reading first {num_samples} samples (streaming)...")
    
    samples = list(dataset[split].take(num_samples))
    
    if samples:
        print(f"Fields: {list(samples[0].keys())}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(samples):
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1} samples...")
    
    print(f"Successfully saved {num_samples} samples to {output_file}")

if __name__ == "__main__":
    read_hf_dataset()
