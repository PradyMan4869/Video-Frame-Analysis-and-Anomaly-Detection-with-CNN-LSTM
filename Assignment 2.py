import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoDataset(Dataset):
    def __init__(self, video_path: str, transform=None, max_frames: int = None):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.transform = transform
        self.max_frames = max_frames
        self.frames = self._extract_frames()

    def _extract_frames(self) -> List[np.ndarray]:
        frames = []
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.video_path}")

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or (self.max_frames and frame_count >= self.max_frames):
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                frame_count += 1

        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
        finally:
            cap.release()

        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return frame

class DiffusionModel(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(DiffusionModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def get_encoding_size(self, input_height, input_width):
        x = torch.zeros(1, 3, input_height, input_width)
        x = x.to(next(self.parameters()).device)
        encoded, _ = self.forward(x)
        return encoded.size(1), encoded.size(2), encoded.size(3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        
        # Adjust reconstructed tensor dimensions to match x if necessary
        if reconstructed.shape != x.shape:
            reconstructed = torch.nn.functional.interpolate(reconstructed, size=(x.size(2), x.size(3)))

        # Calculate reconstruction error
        reconstruction_error = torch.abs(x - reconstructed).mean(dim=[1, 2, 3])
        return encoded, reconstruction_error

class CNNLSTM(nn.Module):
    def __init__(self, input_channels: int = 256, input_height: int = 23, input_width: int = 40, hidden_size: int = 64, num_classes: int = 1):
        super(CNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )

        # Calculate the output size of CNN
        with torch.no_grad():
            x = torch.zeros(1, input_channels, input_height, input_width)
            x = self.cnn(x)
            self.cnn_output_size = x.numel()

        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add sequence dimension if missing
        if len(x.size()) == 4:  # If x is of shape (batch_size, channels, height, width)
            x = x.unsqueeze(1)  # Add sequence dimension to make it (batch_size, seq_length, channels, height, width)

        batch_size, seq_length, c, h, w = x.size()
        x = x.view(batch_size * seq_length, c, h, w)  # Merge batch and sequence dimensions
        x = self.cnn(x)
        x = x.view(batch_size, seq_length, -1)  # Reshape to (batch_size, seq_length, cnn_output_size)
        
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use the last LSTM output
        return self.sigmoid(x).squeeze(-1)



def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Training function with proper logging and error handling
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        try:
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device).float()  # Convert labels to float

                # Ensure inputs have a sequence length dimension
                if len(inputs.size()) == 4:
                    inputs = inputs.unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                              f'Step [{i+1}/{len(train_loader)}], '
                              f'Loss: {running_loss/100:.4f}')
                    running_loss = 0.0

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

def evaluate_model(model, test_loader, device):
    """
    Evaluation function with metrics calculation
    """
    model.eval()
    all_preds = []
    all_labels = []

    try:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions = (outputs > 0.5).float()

                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        # Handle the case where all predictions are of the same class
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = 0.5  # Default AUC when all predictions are the same

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc
        }

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def main():
    # Configuration
    VIDEO_PATH = '/home/pradyumna/Videos/onepiece.mp4'
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_FRAMES = 5000  # Limit number of frames to process

    try:
        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((180, 320)),  # Resize to half of 360x640
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Initialize dataset and dataloader
        logger.info("Loading dataset...")
        dataset = VideoDataset(
            video_path=VIDEO_PATH,
            transform=transform,
            max_frames=MAX_FRAMES
        )

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Initialize models
        logger.info("Initializing models...")
        diffusion_model = DiffusionModel().to(device)
        
        # Calculate the encoding size
        encoding_channels, encoding_height, encoding_width = diffusion_model.get_encoding_size(180, 320)
        
        cnn_lstm_model = CNNLSTM(input_channels=encoding_channels, 
                                 input_height=encoding_height, 
                                 input_width=encoding_width).to(device)

        # Extract features using diffusion model
        logger.info("Extracting features...")
        features = []
        reconstruction_errors = []

        diffusion_model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                encoded, error = diffusion_model(batch)
                features.append(encoded.cpu())
                reconstruction_errors.append(error.cpu())

        features = torch.cat(features, dim=0)
        reconstruction_errors = torch.cat(reconstruction_errors, dim=0)

        # Prepare data for CNN-LSTM
        # Create dummy labels for demonstration (replace with actual labels)
        labels = torch.randint(0, 2, (len(features),))

        # Create dataset for CNN-LSTM
        feature_dataset = torch.utils.data.TensorDataset(features, labels)
        train_size = int(0.8 * len(feature_dataset))
        test_size = len(feature_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            feature_dataset, [train_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )

        # Model training
        logger.info("Starting training...")
        best_f1 = 0.0
        for epoch in range(NUM_EPOCHS):
            train_model(cnn_lstm_model, train_loader, criterion, optimizer, 1, device)

            # Evaluate model
            metrics = evaluate_model
            # Model training
        logger.info("Starting training...")
        best_f1 = 0.0
        for epoch in range(NUM_EPOCHS):
            train_model(cnn_lstm_model, train_loader, criterion, optimizer, 1, device)

            # Evaluate model
            metrics = evaluate_model(cnn_lstm_model, test_loader, device)
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            logger.info(f"Metrics: {metrics}")

            # Learning rate scheduling
            scheduler.step(metrics['f1_score'])

            # Save best model
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cnn_lstm_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, 'best_model.pth')
                logger.info("Saved new best model")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if 'dataloader' in locals():
            del dataloader
        if 'train_loader' in locals():
            del train_loader
        if 'test_loader' in locals():
            del test_loader
        torch.cuda.empty_cache()

class ModelInference:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((180, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_model(self, model_path: str) -> nn.Module:
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = CNNLSTM().to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, video_path: str) -> np.ndarray:
        """
        Predict the class for a given video
        """
        try:
            # Load and preprocess the video
            dataset = VideoDataset(video_path, transform=self.transform)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

            # Extract features using the diffusion model
            diffusion_model = DiffusionModel().to(self.device)
            diffusion_model.eval()
            features = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    encoded, _ = diffusion_model(batch)
                    features.append(encoded.cpu())

            features = torch.cat(features, dim=0)

            # Make predictions using the CNN-LSTM model
            self.model.eval()
            predictions = []
            with torch.no_grad():
                for i in range(0, len(features), 16):  # Process in batches of 16
                    batch = features[i:i+16].unsqueeze(1).to(self.device)  # Add sequence dimension
                    output = self.model(batch)
                    predictions.extend(output.cpu().numpy())

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise