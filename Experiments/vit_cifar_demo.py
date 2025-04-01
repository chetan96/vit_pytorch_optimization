import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse
from tqdm import tqdm

from dataset.cifar import get_cifar_dataloaders
from vit_pytorch.vit import ViT

def parse_args():
    parser = argparse.ArgumentParser(description='ViT CIFAR-10 Training with Profiling')
    parser.add_argument('--data-root', type=str, default='../data',
                      help='Path to dataset directory (outside project)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--profile-batches', type=int, default=10,
                      help='Number of batches to profile for each phase')
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA even if available')
    return parser.parse_args()

def get_device(use_cuda=True):
    """Safe device handling for current and future CUDA versions"""
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        # Basic device verification
        try:
            _ = torch.zeros(1).to(device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        except RuntimeError as e:
            print(f"CUDA error, falling back to CPU: {str(e)}")
    return torch.device('cpu')

def train_model():
    args = parse_args()
    device = get_device(not args.no_cuda)
    
    # Initialize TensorBoard
    log_dir = os.path.join("runs", f"vit_profile_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)
    
    # Get dataloaders
    train_loader, _, test_loader = get_cifar_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=0
    )
    
    # Initialize model and move to device
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=256
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Training Phase Profiling
    def run_phase(phase_name, dataloader, train_mode=True, profile_steps=args.profile_batches):
        model.train() if train_mode else model.eval()
        phase_desc = f"{phase_name.capitalize()} ({'train' if train_mode else 'eval'})"
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA] if device.type == 'cuda' 
                       else [torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=profile_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                log_dir,
                worker_name=phase_name
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as profiler:
            
            with torch.set_grad_enabled(train_mode):
                for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=phase_desc)):
                    if batch_idx >= profile_steps + 2:
                        break
                        
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    if train_mode:
                        loss = criterion(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    profiler.step()
    
    print("\n=== Starting Profiling ===")
    run_phase("train", train_loader, train_mode=True)
    run_phase("inference", test_loader, train_mode=False)
    
    print(f"\nProfiling complete. View results with:")
    print(f"tensorboard --logdir={log_dir}")

if __name__ == '__main__':
    train_model()