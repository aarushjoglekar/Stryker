import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from video_transformer import VideoTransformer
from tqdm import tqdm
from torch.utils.data import Dataset
import json
from PIL import Image
from sklearn.metrics import f1_score
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import os

NUM_CLASSES = 17  # Number of classes in the dataset
NUM_FRAMES_PER_CLIP = 180  # Number of frames per video clip
FRAME_STEPS = 3

# positiveCountsList = [22453, 5434, 3303, 1666, 8135, 1118, 7246, 1702, 1404, 122, 35, 3549, 3937, 1980, 13502, 1361, 31]
# positiveCounts = np.array(positiveCountsList)

# totalSamplesList = [31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997, 31997]
# totalSamples = np.array(totalSamplesList)

# pos_weights = torch.from_numpy((totalSamples - positiveCounts) / positiveCounts)

# alpha = 0.1

# pos_weights = pos_weights * alpha

class SoccerDataset(Dataset):
    """
    This class should be more compatible with PyTorch's Dataset
    """
    def __init__(self, db:list):
        # self.db = json.load(open(db, 'r'))
        self.db = db
        self.frame_per_clip = NUM_FRAMES_PER_CLIP
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
    
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        entry = self.db[idx]

        frames = []
        for i in range(1, self.frame_per_clip + 1, FRAME_STEPS):
            imageDir = entry['image_dir']
            imageDir = imageDir.removeprefix("SoccerNet/")
            frame_path = f"{imageDir}/{i}.jpg"
            frame = self.transform(Image.open(frame_path).convert("RGB"))
            frames.append(frame)
        frames = torch.stack(frames)

        # Convert event indices to one-hot label
        labels_idx = entry['labels']  # e.g., [7, 0, 14]
        numLabels = len(labels_idx)
        label_tensor = torch.zeros(NUM_CLASSES, dtype=torch.float)
        for idx in labels_idx:
            label_tensor[idx] = 1.0  # set corresponding positions to 1

        return frames, label_tensor, numLabels
        

def load_dataset(batch_size):

    train_db = json.load(open('labels/train_label_cleaned.json', 'r'))
    valid_db = json.load(open('labels/valid_label_cleaned.json', 'r'))[:500] # limit to 500 for validation
    test_db = json.load(open('labels/test_label_cleaned.json', 'r'))

    trainData = SoccerDataset(train_db)
    validData = SoccerDataset(valid_db)
    testData = SoccerDataset(test_db)

    train_sampler = DistributedSampler(trainData, shuffle=True) #Changed
    valid_sampler = DistributedSampler(validData, shuffle=False) #Changed
    test_sampler = DistributedSampler(testData, shuffle=False) #Changed

    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=False) #Changed 
    valid_loader = torch.utils.data.DataLoader(validData, batch_size=batch_size, sampler=valid_sampler, shuffle=False, num_workers=4, pin_memory=False) #Changed
    test_loader = torch.utils.data.DataLoader(testData, batch_size=batch_size, sampler=test_sampler, shuffle=False, num_workers=4, pin_memory=False) #Changed

    return train_loader, valid_loader, test_loader

def rank0_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

def plot_loss(train_losses, val_losses, train_f1s, val_f1s):
    """
    Generate a separate plot for loss and F1 score.
    """
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Loss per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot F1 Score
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Train F1 Score', color='green')
    plt.plot(val_f1s, label='Validation F1 Score', color='red')
    plt.title('F1 Score per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # Save the plot
    if dist.get_rank() == 0:
        plt.savefig('figures/loss_f1_plot.png')

    plt.show()

def getIndicesOfOne(tensor):
    rows, cols = (tensor == 1).nonzero(as_tuple=True)
    result = [ cols[rows == i].tolist() for i in range(tensor.size(0)) ]
    return result

# Add gradient clipping
def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def train(train_loader, valid_loader, lr, epochs, local_rank, frame_batch_size=30, tf_num_layers=6, skipTraining = False):

    # ========== Model Initialization ==========
    model_args = {
        'embed_dim': 768,
        'num_layers': tf_num_layers, #originally 12
        'num_heads': 12,
        'mlp_dim': 3072,
        'num_classes': NUM_CLASSES,
        'dropout': 0.0,
        'num_frames_per_clip': NUM_FRAMES_PER_CLIP // FRAME_STEPS, # NOTE: Too many frames in one clip so we reduce it by FRAME_STEPS (180/3 = 60 frames per clip)
        'frame_batch_size': frame_batch_size,
    }
    # ==========================================

    model = VideoTransformer(**model_args)

    # set trainable parameters
    for parameters in model.parameters():
        parameters.requires_grad = True
    
    # freeze pretrained DINOv2
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    
    #REPLACED WITH THIS
    # 1) pick the right CUDA device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2) move model to that device
    model.to(device)

    # 3) now wrap in DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)  # Reduce learning rate by 10% every epoch

    best_eval_f1 = -1
    
    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []
    
    if not skipTraining:
    
        for epoch in range(epochs):
            model.train()
            train_loader.sampler.set_epoch(epoch)  # Shuffle data at each epoch
            valid_loader.sampler.set_epoch(epoch) # ADDED
            train_loss = 0.0
            train_f1 = 0.0
            
            batches_seen = 0
            
            train_epoch_labels = []
            train_epoch_preds  = []

            for batch_idx, (inputs, labels, num_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                
                batches_seen = batches_seen + 1
                
                # inputs: [B, num_frames, C, H, W]
                # labels: [B, num_classes]
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs) # logits: [B, num_classes]
                loss = criterion(outputs, labels)
                loss.backward()
                
                optimizer.step()

                rank0_print(outputs.sigmoid())
                pred = (outputs.sigmoid() > 0.5).float()
                
                predsIndex = getIndicesOfOne(pred)
                labelsIndex = getIndicesOfOne(labels)
                predsAndLabels = [[x, y] for x, y in zip(predsIndex, labelsIndex)]
                rank0_print(f"{predsAndLabels}")

                # Calculate F1 score
                batch_f1  = f1_score(labels.cpu().numpy(), pred.cpu().numpy(), average='micro', zero_division=0)
                
                subset_accuracy = ((pred == labels).all(dim=1)).float().mean().item()
                mean_label_accuracy = (pred == labels).float().mean(dim=1).mean().item()
                train_loss += loss.item()
                train_f1 += batch_f1
                avg_num_labels = num_labels.float().mean().item()

                rank0_print(f"- Train Loss: {loss.item():.4f}, F1Micro: {batch_f1:.4f}, Subset Acc: {subset_accuracy:.4f}, Mean Label Acc: {mean_label_accuracy:.4f}, Average NumLabels in Batch: {avg_num_labels}\n")
                
                train_epoch_labels.append(labels.detach().cpu().numpy())
                train_epoch_preds .append(pred.detach().cpu().numpy())
                
                # do evaluation every 100 batches
                if (batch_idx+1) % 100 == 0:
                    
                    model.eval()
                    with torch.no_grad():
                        val_loss = 0
                        all_labels = []
                        all_preds = []
                        for valid_idx, (inputs, labels, num_labels) in enumerate(tqdm(valid_loader, desc="Validating")):
                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            
                            pred = (outputs.sigmoid() > 0.5).float()
                            
                            all_labels.append(labels.cpu().numpy())
                            all_preds.append(pred.cpu().numpy())
                    
                    #ADDED        
                    model.train()
                        
                    val_loss /= len(valid_loader)
                    # Calculate F1 score
                    val_f1 = f1_score(np.vstack(all_labels), np.vstack(all_preds), average='micro', zero_division=0)

                    # Calculate average trainig loss for 100 batches and add to train_losses and train_f1s
                    train_loss /= (batches_seen) #CHANGED
                    train_f1 /= (batches_seen) #CHANGED
                    train_losses.append(train_loss)
                    train_f1s.append(train_f1)
                    val_losses.append(val_loss)
                    val_f1s.append(val_f1)
                    train_loss = 0.0
                    train_f1 = 0.0
                    
                    batches_seen = 0

                    rank0_print(f"- Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

                    if val_f1 > best_eval_f1:
                        best_eval_f1 = val_f1

                        if dist.get_rank() == 0:
                            torch.save(model.state_dict(), f'checkpoints/best_video_transformer.pth')
                            rank0_print("* Best model saved with F1 score:", best_eval_f1)

            lr_scheduler.step()  # Step the learning rate scheduler

            if batches_seen > 0:
                train_loss /= (batches_seen) #CHANGED
                train_f1 /= (batches_seen) #CHANGED
                train_losses.append(train_loss)
                train_f1s.append(train_f1)
                train_loss = 0.0
                train_f1 = 0.0
                batches_seen = 0

            # train_loss /= len(train_loader)
            
            all_labels = np.vstack(train_epoch_labels)
            all_preds  = np.vstack(train_epoch_preds)
            epoch_train_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            rank0_print(f"Epoch {epoch+1} Train F1Micro: {epoch_train_f1:.4f}")

            # Save the model checkpoint every epoch
            if dist.get_rank() == 0:
                torch.save(model.state_dict(), f'checkpoints/video_transformer_epoch_{epoch+1}.pth')
                rank0_print("* Model checkpoint saved for epoch", epoch + 1)

        # load the best model
        rank0_print("Loading the best model with F1 score:", best_eval_f1)
    
    folder = 'checkpoints'
    if skipTraining:
        folder = 'BestCheckpoints'
    state_dict = torch.load(folder + '/best_video_transformer.pth', map_location=device)
    # model.module.load_state_dict(state_dict)
    model.load_state_dict(state_dict) # CHANGED (used to be the line above)
    plot_loss(train_losses, val_losses, train_f1s, val_f1s)

    return model

def test(test_loader, model):
    
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():

        all_labels = []
        all_preds = []
        for batch_idx, (inputs, labels, num_labels) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            preds = (outputs.sigmoid() > 0.5).float()
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
        
    # Calculate F1 score
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    rank0_print(f"- Test F1Micro Score: {test_f1}")


if __name__ == "__main__":
    # ========== Hyperparameters ==========
    lr = 1e-4  
    epochs = 20 # 3
    batch_size = 64 # 32 # 64
    frame_batch_size = 180 * 4 
    local_rank = int(os.environ['LOCAL_RANK'])
    tf_num_layers = 6
    # =====================================
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    print({"lr": lr, "epochs": epochs, "batch_size": batch_size, "frame_batch_size": frame_batch_size, "local_rank": local_rank, "tf_num_layers": tf_num_layers})

    # Initialize DDP process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    train_loader, valid_loader, test_loader = load_dataset(batch_size=batch_size)
    best_model = train(train_loader, valid_loader, lr=lr, epochs=epochs, local_rank=local_rank, frame_batch_size=frame_batch_size, tf_num_layers=tf_num_layers, skipTraining = True)

    test(test_loader, best_model)

    rank0_print("Testing complete.")
    
    dist.barrier()
    dist.destroy_process_group()