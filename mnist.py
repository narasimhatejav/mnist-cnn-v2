import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Set device for MacBook Pro (prioritize MPS for Apple Silicon)
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# Hyperparameters optimized for MacBook Pro
batch_size = 128  # Larger batch size for better MPS utilization
learning_rate = 0.1  # Keep learning rate at 0.1
num_epochs = 10  # Reduced epochs for faster training

# Data preprocessing - separate transforms for train and test
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=15),  # ±15° rotation
    transforms.RandomAffine(degrees=0, translate=(2/28, 2/28)),  # ±2 pixels translation (2/28 = fraction of image size)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std - no augmentation
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=train_transform,  # Use augmented transform for training
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=test_transform,  # Use clean transform for testing
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0 if device.type == 'mps' else 4,  # MPS works better with num_workers=0
    pin_memory=True if device.type != 'cpu' else False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0 if device.type == 'mps' else 4,  # MPS works better with num_workers=0
    pin_memory=True if device.type != 'cpu' else False
)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block: conv -> conv -> max
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)                 # 28x28x1 -> 26x26x8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)                # 26x26x8 -> 24x24x16
        self.pool1 = nn.MaxPool2d(2, 2)                             # 24x24x16 -> 12x12x16

        # Second block: conv -> conv -> max
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)               # 12x12x16 -> 10x10x16
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)               # 10x10x16 -> 8x8x16
        self.pool2 = nn.MaxPool2d(2, 2)                             # 8x8x16 -> 4x4x16

        # Third block: conv -> conv (adjusted kernels to fit)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)               # 4x4x16 -> 2x2x16
        self.conv6 = nn.Conv2d(16, 16, kernel_size=2)               # 2x2x16 -> 1x1x16

        # Add batch normalization
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(16)

        # Fully connected layer
        self.fc = nn.Linear(16 * 1 * 1, 10)  # 16→10

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # First block: conv -> conv -> max
        x = F.relu(self.bn1(self.conv1(x)))     # 28x28x1 -> 26x26x8
        x = F.relu(self.bn2(self.conv2(x)))     # 26x26x8 -> 24x24x16
        x = self.pool1(x)                       # 24x24x16 -> 12x12x16

        # Second block: conv -> conv -> max
        x = F.relu(self.bn3(self.conv3(x)))     # 12x12x16 -> 10x10x16
        x = F.relu(self.bn4(self.conv4(x)))     # 10x10x16 -> 8x8x16
        x = self.pool2(x)                       # 8x8x16 -> 4x4x16

        # Third block: conv -> conv
        x = F.relu(self.bn5(self.conv5(x)))     # 4x4x16 -> 2x2x16
        x = F.relu(self.bn6(self.conv6(x)))     # 2x2x16 -> 1x1x16

        # Flatten and fully connected
        x = torch.flatten(x, 1)                 # 1x1x16 -> 16
        x = self.dropout(x)
        x = self.fc(x)                          # 16 -> 10

        return F.log_softmax(x, dim=1)

# Initialize model, loss function, and optimizer
model = Net().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
# Add One Cycle LR scheduler (disable momentum cycling for Adagrad)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.75,  # Peak LR (0.75)
    epochs=num_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos',
    div_factor=7.5,  # Initial LR = 0.75 / 7.5 = 0.1
    final_div_factor=7.5,  # Final LR = 0.75 / 7.5 = 0.1
    cycle_momentum=False  # Disable momentum cycling for Adagrad
)

# Print model summary and parameter count
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(model)
    print("\n" + "-"*50)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print("-"*50 + "\n")

print_model_summary(model)

# Training function
def train_epoch(model, train_loader, optimizer, criterion, device, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, targets in tqdm(train_loader, desc="Training"):
        data, targets = data.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step scheduler after each batch for OneCycleLR
        scheduler.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc

# Training loop
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
learning_rates = []

print("Starting training...")
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Test
    test_loss, test_acc = test(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print(f'Learning Rate: {current_lr:.6f}')

    # OneCycleLR steps after each batch, not each epoch

print("Training completed!")

# Create and display training metrics dataframe
training_metrics = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Train_Loss': train_losses,
    'Test_Loss': test_losses,
    'Train_Accuracy': train_accuracies,
    'Test_Accuracy': test_accuracies,
    'Learning_Rate': learning_rates
})

print("\n" + "="*60)
print("TRAINING METRICS SUMMARY")
print("="*60)
print(training_metrics.to_string(index=False))
print("="*60 + "\n")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', marker='o', color='green')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved as 'mnist_model.pth'")

# Function to visualize some predictions
def visualize_predictions(model, test_loader, device, num_samples=8):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    
    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f'True: {labels[i]}, Pred: {predicted[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=300, bbox_inches='tight')
    print("Predictions visualization saved as 'predictions_visualization.png'")

# Visualize some predictions
visualize_predictions(model, test_loader, device)