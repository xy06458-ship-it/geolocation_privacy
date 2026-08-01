import os, csv, time, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

SPLIT_ROOT = PROJECT_ROOT / 'splits' / 'cityscapes'
DATA_ROOT = PROJECT_ROOT / 'data' / 'cityscapes'
CLASSIFIER_DIR = DATA_ROOT / 'city_classifier'

CITY2IDX_PATH = SPLIT_ROOT / 'city2idx.txt'
TRAIN_CSV = SPLIT_ROOT / 'train_list.csv'
TEST_CSV = SPLIT_ROOT / 'test_list.csv'

CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with CITY2IDX_PATH.open('r', encoding='utf-8-sig') as f:
    NUM_CLASSES = sum(1 for line in f if line.strip())

print(f"Device: {DEVICE}")
print(f"Number of cities: {NUM_CLASSES}")


class CityDataset(Dataset):
    def __init__(self, csv_path, data_root, transform):
        self.data = []
        self.transform = transform

        with Path(csv_path).open(
            'r', encoding='utf-8-sig', newline=''
        ) as f:
            reader = csv.DictReader(f)

            for row in reader:
                image_path = Path(row['path'])

                if not image_path.is_absolute():
                    image_path = Path(data_root) / image_path

                self.data.append(
                    (image_path, int(row['label']))
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), label

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_dataset = CityDataset(
    TRAIN_CSV,
    DATA_ROOT,
    transform_train
)

test_dataset = CityDataset(
    TEST_CSV,
    DATA_ROOT,
    transform_test
)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ── General training function ──────────────────────────────────────────
def train_and_eval(model, model_name, epochs, optimizer, train_ldr=None, test_ldr=None):
    print(f"\n{'='*55}")
    print(f"Training: {model_name}  Epochs={epochs}")
    print(f"{'='*55}")

    trl = train_ldr if train_ldr is not None else train_loader
    tel = test_ldr  if test_ldr  is not None else test_loader

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in trl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        scheduler.step()
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tel:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                correct += (out.argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, f'{model_name}_best.pth'))

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Loss: {total_loss/len(trl):.4f} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f}"
              + (" ★" if val_acc == best_acc else ""))

    elapsed = time.time() - t0
    print(f"\n{model_name} Best Test Accuracy: {best_acc:.4f}  "
          f"Training Time: {elapsed/60:.1f}min")
    return best_acc

# ── model 1：ResNet50-ImageNet  40 LR=5e-4 ───────────────
resnet_imagenet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet_imagenet.fc = nn.Linear(2048, NUM_CLASSES)
opt1 = torch.optim.Adam(resnet_imagenet.parameters(), lr=5e-4, weight_decay=1e-4)
acc_imagenet = train_and_eval(resnet_imagenet, 'resnet50_imagenet', epochs=40, optimizer=opt1)

# ── model 2：CLIP Full-parameter fine-tuning, different learning rates ────────────────────
import clip

clip_model, _ = clip.load('ViT-B/16', device='cpu')
clip_model = clip_model.float()

class CLIPFinetune(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.encoder = clip_model.visual  
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(feat)

clip_transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275,  0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])
clip_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275,  0.40821073],
                         [0.26862954, 0.26130258, 0.27577711])
])
clip_train_loader = DataLoader(
    CityDataset(os.path.join(OUT_DIR,'train_list.csv'), clip_transform_train),
    batch_size=32, shuffle=True,  num_workers=4)
clip_test_loader  = DataLoader(
    CityDataset(os.path.join(OUT_DIR,'test_list.csv'),  clip_transform_test),
    batch_size=32, shuffle=False, num_workers=4)

clip_ft = CLIPFinetune(clip_model, NUM_CLASSES)

# Differential Learning Rates: 1e-5 for the encoder, 1e-3 for the classification head
opt2 = torch.optim.Adam([
    {'params': clip_ft.encoder.parameters(), 'lr': 1e-5},
    {'params': clip_ft.fc.parameters(),      'lr': 1e-3}
], weight_decay=1e-4)

acc_clip = train_and_eval(
    clip_ft, 'clip_finetune', epochs=30, optimizer=opt2,
    train_ldr=clip_train_loader, test_ldr=clip_test_loader
)

# ── Final summary ──────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'Model':<28} {'Test Accuracy':>10}")
print(f"{'='*55}")
print(f"{'ResNet50 (ImageNet)  40ep':<28} {acc_imagenet:>10.4f}")
print(f"{'CLIP ViT-B/16 Full-parameter Fine-tuning 30ep':<28} {acc_clip:>10.4f}")
print(f"{'='*55}")
