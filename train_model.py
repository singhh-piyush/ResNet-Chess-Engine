import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import numpy as np
import tqdm
import os

# --- Configuration ---
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5
DROPOUT2D = 0.2
VALUE_LAMBDA = 5.0
EARLY_STOP_PATIENCE = 5
NUM_RES_BLOCKS = 15
CHANNELS = 192
INPUT_PLANES = 19
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUTS_FILE = 'inputs.npz'
TARGETS_FILE = 'targets.npz'
EVALS_FILE = 'evals.npz'
MODEL_SAVE_BEST = 'chess_clone.pth'
MODEL_SAVE_FINAL = 'chess_clone_final.pth'
NUM_WORKERS = 4

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop2d = nn.Dropout2d(DROPOUT2D)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop2d(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, num_res_blocks=NUM_RES_BLOCKS, channels=CHANNELS):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.res_tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(32 * 8 * 8, 4096)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class ChessDataset(Dataset):
    def __init__(self):
        print("Loading data...")
        with np.load(INPUTS_FILE) as data:
            self.inputs = data['arr_0'].astype(np.float32)
        with np.load(TARGETS_FILE) as data:
            self.targets = data['arr_0'].astype(np.int64)
        with np.load(EVALS_FILE) as data:
            self.evals = data['arr_0'].astype(np.float32)
        print(f"Loaded {len(self.inputs)} samples. Tensor shape: {self.inputs.shape[1:]}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.evals[idx]

def main():
    if not os.path.exists(INPUTS_FILE) or not os.path.exists(TARGETS_FILE) or not os.path.exists(EVALS_FILE):
        print("Data files not found! Run data_miner.py first.")
        return

    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    full_dataset = ChessDataset()

    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=NUM_WORKERS, persistent_workers=True
    )

    model = ChessModel().to(DEVICE)
    criterion_policy = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_value = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE,
                           steps_per_epoch=len(train_loader), epochs=EPOCHS)

    scaler = GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    best_val_loss = float('inf')
    early_stop_counter = 0
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"Batch size: {BATCH_SIZE} | Epochs: {EPOCHS} | AMP: {DEVICE.type == 'cuda'}")
    print(f"ResBlocks: {NUM_RES_BLOCKS} | Dropout: {DROPOUT} | Dropout2d: {DROPOUT2D}")
    print(f"Label Smoothing: {LABEL_SMOOTHING} | Value λ: {VALUE_LAMBDA}")
    print(f"Optimizer: AdamW (peak_lr={LEARNING_RATE}, wd={WEIGHT_DECAY}) + OneCycleLR")
    print(f"Early Stopping: patience={EARLY_STOP_PATIENCE}")
    print("-" * 70)

    for epoch in range(EPOCHS):
        model.train()
        running_policy_loss = 0.0
        running_value_loss = 0.0
        running_total_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

        for inputs, targets, evals in loop:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            evals = evals.to(DEVICE, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                policy_out, value_out = model(inputs)
                policy_loss = criterion_policy(policy_out, targets)
                value_loss = criterion_value(value_out, evals)
                total_loss = policy_loss + VALUE_LAMBDA * value_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()
            running_total_loss += total_loss.item()
            _, predicted = torch.max(policy_out, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            loop.set_postfix(p_loss=f"{policy_loss.item():.3f}", v_loss=f"{value_loss.item():.3f}")

        train_acc = 100 * train_correct / train_total
        n_batches = len(train_loader)
        avg_policy = running_policy_loss / n_batches
        avg_value = running_value_loss / n_batches
        avg_total = running_total_loss / n_batches

        current_lr = scheduler.get_last_lr()[0]

        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets, evals in val_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                evals = evals.to(DEVICE, non_blocking=True).view(-1, 1)
                with autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    policy_out, value_out = model(inputs)
                    p_loss = criterion_policy(policy_out, targets)
                    v_loss = criterion_value(value_out, evals)
                    t_loss = p_loss + VALUE_LAMBDA * v_loss
                val_policy_loss += p_loss.item()
                val_value_loss += v_loss.item()
                val_total_loss += t_loss.item()
                _, predicted = torch.max(policy_out, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        n_val = len(val_loader)
        avg_val_policy = val_policy_loss / n_val
        avg_val_value = val_value_loss / n_val
        avg_val_total = val_total_loss / n_val
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}: "
              f"Train [P={avg_policy:.4f} V={avg_value:.4f} T={avg_total:.4f} Acc={train_acc:.2f}%] | "
              f"Val [P={avg_val_policy:.4f} V={avg_val_value:.4f} T={avg_val_total:.4f} Acc={val_acc:.2f}%] | "
              f"LR={current_lr:.6f}")

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_BEST)
            print(f"-> Saved new best model: {avg_val_total:.4f}")
        else:
            early_stop_counter += 1
            print(f"   No improvement ({early_stop_counter}/{EARLY_STOP_PATIENCE})")
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    torch.save(model.state_dict(), MODEL_SAVE_FINAL)
    print(f"Training Complete. Final model saved to {MODEL_SAVE_FINAL}")

if __name__ == "__main__":
    main()
