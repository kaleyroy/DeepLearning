from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, path, block_size=512):
        # 读取前 1000 行
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        import json

        self.encoded_data = []

        self.max_lines = 1000
        raw_data = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.block_size):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i+self.block_size+1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)

if __name__ == '__main__':
    train_dataset = MyDataset('D:/GitRepos/Tests/DeepLearning/test_codes/seq_monkey_data.jsonl')
    # split traindataset to train and val
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

    # 查看一个 batch 的数据
    for batch in train_loader:
        x, y = batch
        print("Input:", x[0])
        print("Target:", y[0])
        break