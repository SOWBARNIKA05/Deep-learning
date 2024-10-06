import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
import torch.optim as optim 
df = pd.read_csv('Train.csv') 
print(df.head()) 
categorical_columns = ["protocol_type", "service", "flag", "attack"] 
label_encoders = {} 
for col in categorical_columns: 
    le = LabelEncoder() 
df[col] = le.fit_transform(df[col]) 
label_encoders[col] = le 
numerical_columns = df.columns.difference(categorical_columns) 
scaler = StandardScaler() 
df[numerical_columns] = scaler.fit_transform(df[numerical_columns]) 
X = df.drop('attack', axis=1).values 
y = df['attack'].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
class NetworkTrafficDataset(Dataset): 
    def __init__(self, features, labels): 
        self.features = torch.tensor(features, dtype=torch.float32) 
        self.labels = torch.tensor(labels, dtype=torch.long) 
     
    def __len__(self): 
        return len(self.features) 
     
    def __getitem__(self, idx): 
        return self.features[idx], self.labels[idx] 
# Create PyTorch datasets 
train_dataset = NetworkTrafficDataset(X_train, y_train) 
test_dataset = NetworkTrafficDataset(X_test, y_test) 
# Create DataLoaders 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) 
class SimpleNN(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size): 
        super(SimpleNN, self).__init__() 
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_size, output_size) 
    def forward(self, x): 
        x = self.fc1(x) 
        x = self.relu(x) 
        x = self.fc2(x) 
        return x 
# Define model, loss function, and optimizer 
input_size = X_train.shape[1] 
hidden_size = 64 
output_size = len(label_encoders['attack'].classes_) 
model = SimpleNN(input_size, hidden_size, output_size) 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) 
def train(model, train_loader, criterion, optimizer, epochs=30): 
    model.train() 
    for epoch in range(epochs): 
        running_loss = 0.0 
        for inputs, labels in train_loader: 
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item() * inputs.size(0) 
        epoch_loss = running_loss / len(train_loader.dataset) 
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}') 
# Train the model 
train(model, train_loader, criterion, optimizer) 
def evaluate(model, test_loader): 
    model.eval() 
    correct = 0 
    total = 0 
    with torch.no_grad(): 
        for inputs, labels in test_loader: 
            outputs = model(inputs) 
            _, predicted = torch.max(outputs, 1) 
            total += labels.size(0) 
            correct += (predicted == labels).sum().item() 
    accuracy = correct / total 
    print(f'Accuracy: {accuracy*100:.2f}') 
evaluate(model, test_loader) 