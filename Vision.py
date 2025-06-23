from Render import VIEW_X, VIEW_Y
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Assuming DatasetExtractor is available from Dataset.py
# You might need to adjust the import path if Dataset.py is not in the same directory
from Dataset import DatasetExtractor

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.tanh = nn.Tanh()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.75),
        )

        # Calculate flattened features size dynamically
        # We'll pass a dummy tensor to determine the size after conv layers
        self._to_linear = None
        self.fc_input_size = self._get_conv_output_size()

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 6) # 4 for logits, 2 for continuous values
        )

    def _get_conv_output_size(self):
        # Pass a dummy input to calculate the output size of the conv layers
        dummy_input = torch.zeros(1, 3, VIEW_X, VIEW_Y)
        output = self.conv_layers(dummy_input)
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten the output for the fully connected layers
        x = self.fc_layers(x)
        return torch.concatenate((x[:, :4], self.tanh(x[:, 4:])), dim=1)

class CarDataset(Dataset):
    def __init__(self, data):
        self.frames = []
        self.targets = []
        for frame, target in data:
            # PyTorch expects NCHW format, so permute from HWC (400x300x3) to CHW (3x400x300)
            self.frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0) # Normalize to [0, 1]
            # Clip continuous values to (-1, 1) as specified
            target_clipped = np.copy(target).astype(np.float32)
            target_clipped[4:] = np.clip(target_clipped[4:], -1.0, 1.0)
            self.targets.append(torch.from_numpy(target_clipped).float())

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.targets[idx]

class StreamingCarDataset(Dataset):
    def __init__(self, data_extractor: DatasetExtractor):
        self.data_extractor = data_extractor

    def __len__(self):
        return len(self.data_extractor)

    def __getitem__(self, idx):
        frame, target = self.data_extractor[idx]
        # PyTorch expects NCHW format, so permute from HWC (400x300x3) to CHW (3x400x300)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 # Normalize to [0, 1]
        # Clip continuous values to (-1, 1) as specified
        target_clipped = np.copy(target).astype(np.float32)
        target_clipped[4:] = np.clip(target_clipped[4:], -1.0, 1.0)
        target = torch.from_numpy(target_clipped).float()
        return frame, target

class ModelTrainer:
    def __init__(self, model, dataloader, learning_rate=0.001):
        self.model = model
        self.dataloader = dataloader
        self.criterion_logits = nn.BCEWithLogitsLoss()
        self.criterion_mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device: {self.device}")
        self.model.to(self.device)

    def run_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_loss_logits = 0.0
        running_loss_mse = 0.0
        
        # Initialize correct predictions for each logit
        correct_predictions_logit = [0, 0, 0, 0]
        total_predictions_logit = [0, 0, 0, 0]

        for i, (frames, targets) in enumerate(self.dataloader):
            frames = frames.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(frames)

            outputs_logits = outputs[:, :4]
            targets_logits = targets[:, :4]
            outputs_mse = outputs[:, 4:]
            targets_mse = targets[:, 4:]

            loss_logits = self.criterion_logits(outputs_logits, targets_logits)
            loss_mse = self.criterion_mse(outputs_mse, targets_mse)

            loss = loss_logits + loss_mse
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_loss_logits += loss_logits.item()
            running_loss_mse += loss_mse.item()

            # Calculate accuracy for each logit individually
            predicted_logits = (torch.sigmoid(outputs_logits) > 0.5).float()
            for j in range(4):
                correct_predictions_logit[j] += (predicted_logits[:, j] == targets_logits[:, j]).sum().item()
                total_predictions_logit[j] += targets_logits[:, j].numel()

        epoch_loss = running_loss / len(self.dataloader)
        epoch_loss_logits = running_loss_logits / len(self.dataloader)
        epoch_loss_mse = running_loss_mse / len(self.dataloader)

        logit_names = ["Right Obstacle", "Front Obstacle", "Left Obstacle", "Right/Left Lane"]
        individual_logit_accuracies = []
        for j in range(4):
            accuracy = correct_predictions_logit[j] / total_predictions_logit[j] if total_predictions_logit[j] > 0 else 0
            individual_logit_accuracies.append(f"{logit_names[j]}: {accuracy:.4f}")
        
        print(f"Epoch {epoch+1}, Total Loss: {epoch_loss:.4f}, Logits Loss: {epoch_loss_logits:.4f}, MSE Loss: {epoch_loss_mse:.4f}")
        print(f"  Individual Logit Accuracies: {', '.join(individual_logit_accuracies)}")

def test_code():
    print("Collecting data from Dataset.py...")
    num_samples = 1000
    extractor = DatasetExtractor(num_samples, sim_time_delta=0.05, collection_interval=0.2)
    # Collect a small number of samples for demonstration purposes
    # In a real scenario, you'd collect much more data
    collected_data = extractor.run()
    print(f"Collected {len(collected_data)} samples.")

    if collected_data:
        print("Creating dataset and dataloader...")
        dataset = CarDataset(collected_data)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        print("Dataset and dataloader created.")

        print("Initializing model...")
        model = SimpleCNN()
        print("Model initialized.")

        print("Starting training...")
        trainer = ModelTrainer(model, dataloader, learning_rate=0.001)
        for epoch in range(15):
            trainer.run_epoch(epoch)
        print("Training complete.")
    else:
        print("No data collected. Cannot proceed with training.")

if __name__ == "__main__":
    test_code()