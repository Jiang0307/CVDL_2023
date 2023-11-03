import torch
import matplotlib
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from torchvision.models import vgg19_bn

DOWNLOAD = False
save_path = str(Path(__file__).joinpath("model.pth"))
matplotlib.use('TkAgg')

# Step 1: Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=DOWNLOAD, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=DOWNLOAD, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Step 2: Define the model (VGG19 with Batch Normalization)
class VGG19BN(nn.Module):
    def __init__(self):
        super(VGG19BN, self).__init__()
        self.vgg19_bn = torchvision.models.vgg19_bn(num_classes=10)
        self.features = self.vgg19_bn.features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # 减少了第一个全连接层的单元数
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, 10)  # 减少了第二个全连接层的单元数
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    save_path = str( Path(__file__).parent.joinpath("model.pth") )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGG19BN().to(device)

    # Step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Step 4: Training loop
    num_epochs = 100
    best_accuracy = 0.0

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(enumerate(trainloader), total=len(trainloader))

        for i, data in pbar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            pbar.set_postfix({'Loss': running_loss / (i+1), 'Accuracy': 100 * correct_train / total_train})

        train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_loss /= len(testloader)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
            f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)

        # Save values for plotting
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_accuracy)
        val_acc_list.append(val_accuracy)

    # Step 5: Plot and save figure
    epochs = range(1, num_epochs+1)

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss_list, label='Training Loss')
    plt.plot(epochs, val_loss_list, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training Accuracy and Validation Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc_list, label='Training Accuracy')
    plt.plot(epochs, val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()  # 保证子图不会重叠
    plt.savefig('accuracy and loss.png')
    plt.show()