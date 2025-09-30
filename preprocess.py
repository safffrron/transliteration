import json
import torch
from torch.utils.data import Dataset, DataLoader

class TransliterationDataset(Dataset):
    def __init__(self, X, Y, input_stoi, target_stoi):
        self.X, self.Y = X, Y
        self.input_stoi = input_stoi
        self.target_stoi = target_stoi

    def encode(self, text, stoi, add_sos_eos=True):
        tokens = [stoi.get(ch, 0) for ch in text]
        if add_sos_eos:
            tokens = [stoi["<sos>"]] + tokens + [stoi["<eos>"]]
        return tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.encode(self.X[idx], self.input_stoi, add_sos_eos=True)
        y = self.encode(self.Y[idx], self.target_stoi, add_sos_eos=True)
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    max_src_len = max(len(x) for x in src_batch)
    max_trg_len = max(len(y) for y in trg_batch)

    src_padded = torch.stack([torch.cat([x, torch.zeros(max_src_len - len(x), dtype=torch.long)]) for x in src_batch])
    trg_padded = torch.stack([torch.cat([y, torch.zeros(max_trg_len - len(y), dtype=torch.long)]) for y in trg_batch])

    return src_padded, trg_padded

def get_loaders(batch_size=128, train_file="hin_train.json", val_file="hin_valid.json", test_file="hin_test.json"):
    # Load data
    def load_json(file_path):
        X, Y = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    X.append(obj["english word"])
                    Y.append(obj["native word"])
        return X, Y

    X_train, Y_train = load_json(train_file)
    X_val, Y_val = load_json(val_file)
    X_test, Y_test = load_json(test_file)

    # Build vocab
    input_chars = sorted(set("".join(X_train)))
    target_chars = sorted(set("".join(Y_train)))

    input_chars = ["<pad>", "<sos>", "<eos>"] + input_chars
    target_chars = ["<pad>", "<sos>", "<eos>"] + target_chars

    input_stoi = {ch: i for i, ch in enumerate(input_chars)}
    input_itos = {i: ch for ch, i in input_stoi.items()}

    target_stoi = {ch: i for i, ch in enumerate(target_chars)}
    target_itos = {i: ch for ch, i in target_stoi.items()}

    # Create datasets
    train_dataset = TransliterationDataset(X_train, Y_train, input_stoi, target_stoi)
    val_dataset = TransliterationDataset(X_val, Y_val, input_stoi, target_stoi)
    test_dataset = TransliterationDataset(X_test, Y_test, input_stoi, target_stoi)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, input_stoi, target_stoi, input_itos, target_itos
