import os
import json
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchvision import datasets, transforms
from prettytable import PrettyTable
from thop import profile
from models.airmamba import airmamba_s

class ConfusionMatrixPercent:
    def __init__(self, num_classes: int, labels: list, model_name: str):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.model_name = model_name

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        acc = np.trace(self.matrix) / np.sum(self.matrix)
        print(f"Accuracy: {acc:.4%}")
        return acc

    def plot(self):
        fig, ax = plt.subplots(figsize=(9, 8))
        norm_matrix = self.matrix / self.matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(norm_matrix, cmap='greens', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(self.labels, rotation=45, fontsize=12)
        ax.set_yticklabels(self.labels, fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_title('Normalized Confusion Matrix', fontsize=14)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, f'{norm_matrix[i, j]:.2f}',
                        ha="center", va="center",
                        color="white" if norm_matrix[i, j] > 0.5 else "black", fontsize=12)
        plt.tight_layout()
        os.makedirs("Results", exist_ok=True)
        plt.savefig(f"Results/confusion_{self.model_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"Results/confusion_{self.model_name}.pdf", bbox_inches='tight')
        plt.close()

def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def measure_inference_time(model, input_size=(1, 3, 256, 256), device='cuda', repetitions=100):
    model.eval()
    inputs = torch.randn(input_size).to(device)
    for _ in range(10):
        _ = model(inputs)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repetitions):
        _ = model(inputs)
    torch.cuda.synchronize()
    return (time.time() - start_time) * 1000 / repetitions

def evaluate_model():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5108, 0.9670, 0.4890), (0.1176, 0.0978, 0.1228))
    ])
    dataset = datasets.ImageFolder(root="../data/val", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    label_path = '../class_indices.json'
    with open(label_path, 'r') as f:
        class_map = json.load(f)
    labels = ['DJI Phantom4', 'Fixed-Wing', 'Helicopter', 'Hexacopter', 'Lacent', 'Shahed']

    model = airmamba_s()
    weight_path = "../checkpoints/airmamba_s_best.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    model_size = get_model_size(model)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))
    inf_time = measure_inference_time(model)

    confusion = ConfusionMatrixPercent(num_classes=6, labels=labels, model_name="source_source")
    with torch.no_grad():
        for imgs, targets in tqdm(loader):
            imgs = imgs.to(device)
            preds = model(imgs)
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            confusion.update(preds.cpu().numpy(), targets.cpu().numpy())

    acc = confusion.summary()
    confusion.plot()

    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Accuracy", f"{acc:.4%}"])
    table.add_row(["Parameters", f"{params / 1e6:.2f}M"])
    table.add_row(["Model Size", f"{model_size:.2f} MB"])
    table.add_row(["FLOPs", f"{flops / 1e9:.2f} G"])
    table.add_row(["Inference Time", f"{inf_time:.2f} ms"])
    print("\nModel Performance Summary:")
    print(table)

if __name__ == '__main__':
    evaluate_model()
