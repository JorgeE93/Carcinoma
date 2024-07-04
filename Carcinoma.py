
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from tqdm.notebook import tqdm
import pydicom as di
import os
import cv2
from sklearn.model_selection import train_test_split as split
from PIL import Image
import csv
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

s3 = boto3.resource('s3')

class Data_frame:
    def __init__(self):
        bucket = 'myvermabucket123'
        data_key = 'Adrenal-ACC-Ki67-Seg_SupportingData_20230522.xlsx'
        self.data_location = f's3://{bucket}/{data_key}'
        self.df = pd.read_excel(self.data_location)

    def get_data(self, df):
        mappings = {
            'SEX': {'M': 0, 'F': 1},
            'Race': {'White': [0, 0], 'Hispanic or Latino': [0, 1], 'Black': [1, 0], 'default': [1, 1]},
            'PreContrast': {'Y': 1, 'N': 0},
            'Venous': {'Y': 1, 'N': 0},
            'Delayed': {'Y': 1, 'N': 0},
            'Laterality': {'Left': 0, 'Right': 1},
            'CLINIC PRESENT': {
                'Hormonal hypersecretion': [0, 0, 0],
                'Abdominal / Back pain': [0, 0, 1],
                'Incidental finding': [0, 1, 0],
                'Hypertension': [0, 1, 1],
                'Hyperandrogenism': [1, 0, 0],
                'default': [1, 0, 1]
            },
            'TStaging': {'T1': [0, 0], 'T2': [0, 1], 'T3': [1, 0], 'default': [1, 1]},
            'NStaging': {'N0': [0, 0], 'N1': [0, 1], 'default': [1, 0]},
            'MStaging': {'M0': 0, 'M1': 1},
            'ResectionMargin': {'R0': [0, 0], 'R1': [0, 1], 'default': [1, 0]},
            'Lymph node/ PATHOLOGY ': {'N0': [0, 0], 'N1': [0, 1], 'default': [1, 0]},
            'MetsFutureSite': {'N': [0, 0], 'Y': [0, 1], 'default': [1, 0]},
            'FutureMets - Liver': {'N': 0, 'Y': 1},
            'FutureMets - Lymph nodes': {'N': 0, 'Y': 1},
            'FutureMets - Bone': {'N': 0, 'Y': 1},
            'FutureMets-Lung': {'N': 0, 'Y': 1},
            'FutureMets-Others': {'N': [0, 0], 'Peritoneal': [0, 1], 'default': [1, 0]},
            'MetsAtDX': {'N': 0, 'Y': 1},
        }

        vector = [df['Age'].values[0]]
        target = [df['Patient_ID'].values[0]]

        for key, map_dict in mappings.items():
            value = df[key].values[0]
            vector.extend(map_dict.get(value, map_dict.get('default', [])))

        vector.append(df['PathTumorSize'].values[0])
        vector.append(df['TTPCensor'].values[0])
        vector.append(df['Ki67'].values[0])
        vector.append(df['DaysToDiagnosis'].values[0])
        target_number = df['TNM_Stage'].values[0] - 1
        target.append(target_number)

        return vector, target

    def get_vector(self):
        dataframe = self.df
        person, target = [], []
        for i, row in dataframe.iterrows():
            a, b = self.get_data(row.to_frame().T)
            person.append(a)
            target.append(b)
        return person, target

    def show_grid(self, image, title=None):
        image = image.permute(1, 2, 0)  # Rearranging the image tensor dimensions
        mean = torch.FloatTensor([0.485, 0.456, 0.406])  # Mean for denormalization
        std = torch.FloatTensor([0.229, 0.224, 0.225])  # Standard deviation for denormalization

        image = image * std + mean  # Apply denormalization
        image = np.clip(image, 0, 1)  # Clip values to be between 0 and 1

        plt.figure(figsize=[15, 15])  # Set the size of the figure
        plt.imshow(image)  # Display the image grid
        if title is not None:
            plt.title(title)  # Set the title if provided

    def accuracy(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)  # Apply softmax to get probability distributions
        top_p, top_class = y_pred.topk(1, dim=1)  # Get the top class predictions
        equals = top_class == y_true.view(*top_class.shape)  # Compare with true labels
        return torch.mean(equals.type(torch.FloatTensor))  # Calculate the mean accuracy

df = Data_frame()

class DCMReader:
    def __init__(self):
        bucket = 'myvermabucket123/Carcinoma/'
        data_key = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']
        self.data_location = [f's3://{bucket}/{dk}/' for dk in data_key]
        _, self.target = df.get_vector()

    def load_images(self):
        images = []
        mybucket = s3.Bucket('myvermabucket123')
        for k in tqdm(mybucket.objects.all()):
            s = k.key
            t = s.split('/')[-1]
            if t.endswith('.dcm'):
                mybucket.download_file(s, t)
                images1 = di.dcmread(t, force=True)
                image_name = images1.PatientName
                target_image = self.find_image_target(image_name)
                images1 = cv2.resize((images1.pixel_array - images1.RescaleIntercept) / (images1.RescaleSlope * 1000), (512, 512))
                images1 = (images1 * 255).astype(np.uint8)
                images.append([images1, target_image])
                os.remove(t)
        print(f"Number of (.dcm) files = {len(images)}")
        return images

    def find_image_target(self, image_name):
        for t in range(len(self.target)):
            if image_name == self.target[t][0]:
                return self.target[t][1]

dcm = DCMReader()

class Config:
    def __init__(self):
        self.epochs = 30
        self.max_lr = 0.1
        self.base_lr = 0.001
        self.batch_size = 128
        self.img_size = 512

config = Config()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"On which device we are on: {device}")

# Load Images
images = dcm.load_images()

PIL_images = []
for img, target in tqdm(images):
    im = Image.fromarray(np.uint16(img)).convert('L')
    PIL_images.append([im, target])

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
for ax, (img, target) in zip(axs.flatten(), PIL_images[:10]):
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Target: {target}")
    ax.axis("off")
plt.tight_layout()
plt.show()

def get_weights(dataset):
    class_counts = Counter([y for _, y in dataset])
    weights = [1.0 / class_counts[y] for _, y in dataset]
    return weights

# Shuffle Files
train_dataset, val_dataset = split(PIL_images, train_size=0.8, shuffle=True)

# Transform the image
train_transform = T.Compose([
    T.Resize(size=config.img_size),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=(-30, +30)),
    T.ToTensor(),
    T.Lambda(lambda x: x.broadcast_to(3, x.shape[1], x.shape[2])),
    T.Normalize(mean=[0.485], std=[0.229])
])

# Define transformations for the validation dataset
valid_transform = T.Compose([
    T.Resize(size=config.img_size),
    T.ToTensor(),
    T.Lambda(lambda x: x.broadcast_to(3, x.shape[1], x.shape[2])),
    T.Normalize(mean=[0.485], std=[0.229])
])

class MyDataset(Dataset):
    def __init__(self, img_list, augmentations):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx][0]
        img = self.augmentations(img)
        label = self.img_list[idx][1]
        return img, label

trainset = MyDataset(train_dataset, train_transform)
validset = MyDataset(val_dataset, valid_transform)
print(f"Trainset Size: {len(trainset)}")
print(f"Validset Size: {len(validset)}")

train_weights = get_weights(trainset)
train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

trainloader = DataLoader(trainset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4)
validloader = DataLoader(validset, batch_size=config.batch_size, shuffle=True)
print(f"No. of batches in trainloader: {len(trainloader)}")
print(f"No. of Total examples: {len(trainloader.dataset)}")

weights = EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=625),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=625, out_features=256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=256, out_features=4)
)

model.to(device)

def printChart(train_data, valid_data, epoch):
    train_loss = [loss for loss, _ in train_data]
    train_acc = [acc for _, acc in train_data]
    valid_loss = [loss for loss, _ in valid_data]
    valid_acc = [acc for _, acc in valid_data]
    epoch_count = range(1, epoch + 2)

    plt.plot(epoch_count, train_loss, '-c', label='Train Loss')
    plt.plot(epoch_count, valid_loss, '-g', label='Validation Loss')
    plt.title("Loss per Epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(epoch_count, train_acc, '-c', label='Train Accuracy')
    plt.plot(epoch_count, valid_acc, '-g', label='Validation Accuracy')
    plt.title("Accuracy per Epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc='upper left')
    plt.show()

def create_csv(train_data, valid_data, epoch):
    with open('ModelStats.csv', mode='a') as csvfile:
        fieldnames = ['Epoch', 'Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat('ModelStats.csv').st_size == 0:
            writer.writeheader()
        writer.writerow({
            'Epoch': epoch + 1,
            'Train Loss': train_data[0],
            'Validation Loss': valid_data[0],
            'Train Accuracy': train_data[1].item(),
            'Validation Accuracy': valid_data[1].item()
        })

def print_confusion_matrix(actual, predicted, epoch):
    cm = confusion_matrix(actual, predicted)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], yticklabels=['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.show()

class CarcinomaTrainer:
    def __init__(self, criterion=None, optimizer=None, scheduler=None):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_stats = []
        self.valid_stats = []

    def train_batch_loop(self, model, trainloader):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_acc += df.accuracy(logits, labels)
        self.train_stats.append((train_loss / len(trainloader), train_acc / len(trainloader)))
        return train_loss / len(trainloader), train_acc / len(trainloader)

    def valid_batch_loop(self, model, validloader, epoch):
        model.eval()
        valid_loss, valid_acc = 0.0, 0.0
        y_actual, y_predicted = [], []
        with torch.no_grad():
            for images, labels in tqdm(validloader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = self.criterion(logits, labels)
                valid_loss += loss.item()
                valid_acc += df.accuracy(logits, labels)
                _, predicted = torch.max(logits, 1)
                y_actual.extend(labels.cpu().numpy())
                y_predicted.extend(predicted.cpu().numpy())
        self.valid_stats.append((valid_loss / len(validloader), valid_acc / len(validloader)))
        print_confusion_matrix(y_actual, y_predicted, epoch)
        return valid_loss / len(validloader), valid_acc / len(validloader)

    def fit(self, model, trainloader, validloader, epochs):
        valid_min_loss = np.Inf
        for i in range(epochs):
            avg_train_loss, avg_train_acc = self.train_batch_loop(model, trainloader)
            avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model, validloader, i)
            printChart(self.train_stats, self.valid_stats, i)
            create_csv((avg_train_loss, avg_train_acc), (avg_valid_loss, avg_valid_acc), i)
            if avg_valid_loss <= valid_min_loss:
                print(f"Valid_loss decreased {valid_min_loss} --> {avg_valid_loss}")
                torch.save(model.state_dict(), 'EfficientNetCarcinomaModel.pth')
                valid_min_loss = avg_valid_loss
            print(f"Epoch: {i+1} Train Loss: {avg_train_loss:.6f} Train Acc: {avg_train_acc:.6f}")
            print(f"Epoch: {i+1} Valid Loss: {avg_valid_loss:.6f} Valid Acc: {avg_valid_acc:.6f}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
scheduler = CyclicLR(optimizer, base_lr=config.base_lr, max_lr=config.max_lr, step_size_up=5*len(trainloader), mode='exp_range', cycle_momentum=False)
trainer = CarcinomaTrainer(criterion, optimizer, scheduler)
trainer.fit(model, trainloader, validloader, epochs=config.epochs)
torch.save(model.state_dict(), 'EfficientNetCarcinomaModel.pth')
torch.cuda.empty_cache()


if __name__ == "__main__":
    # Call the optimization script
    from Optimizer import objective


    # Define a function to train the final model with the best hyperparameters
    def train_final_model():
        # Read best hyperparameters from the file
        with open('best_hyperparameters.txt', 'r') as f:
            best_params = eval(f.read())

        # Update config with best hyperparameters
        config.base_lr = best_params['base_lr']
        config.max_lr = best_params['max_lr']
        config.batch_size = best_params['batch_size']

        # Update DataLoader with the best batch size
        trainloader.batch_size = config.batch_size
        validloader.batch_size = config.batch_size

        # Initialize model and optimizer with the best hyperparameters
        optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
        scheduler = CyclicLR(optimizer, base_lr=config.base_lr, max_lr=config.max_lr, step_size_up=5 * len(trainloader),
                             mode='exp_range', cycle_momentum=False)

        trainer = CarcinomaTrainer(criterion, optimizer, scheduler)
        trainer.fit(model, trainloader, validloader, epochs=config.epochs)

        torch.save(model.state_dict(), 'EfficientNetCarcinomaModel_best.pth')
        torch.cuda.empty_cache()


    # Train the final model
    train_final_model()