import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
import os
import argparse
from tqdm import tqdm

from dataset.cifar import get_cifar_dataloaders
from vit_pytorch.simple_flash_attn_vit import SimpleViT


def parse_args():
    parser = argparse.ArgumentParser(description='ViT CIFAR-10 flash attention Training with Profiling')
    
    parser.add_argument('--data-root', type=str, default='../data', help='Path to dataset directory (outside project)')
    
    parser.add_argument('--batch-size', type=int, default=64)
    
    parser.add_argument('--image-size', type=int, default=64, help='Size to resize images to (default: 64)')
    
    parser.add_argument('--patch-size', type=int, default=16, help='Size to resize images to (default: 16)')
    
    parser.add_argument('--epochs', type=int, default=3)
    
    parser.add_argument('--no-profiler', action='store_false', dest='run_profiler', help='Disable GPU profiling (enabled by default)')
    
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')
    
    return parser.parse_args()

def setup_profiler_output_dir(args):
    # Create timestamped directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "profiler_results_flash_vit"
    
    # Create parameter-based subfolder name
    experiment_name = f"ps{args.patch_size}_img{args.image_size}"
    output_dir = os.path.join(base_dir, experiment_name, timestamp)
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

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

def train_test_model():
    args = parse_args()
    device = get_device(not args.no_cuda)
    
    # Get dataloaders
    train_loader, _, test_loader = get_cifar_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=0.0,
        img_size=args.image_size
    )
    
    # Initialize model
    model = SimpleViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=10,
        dim=128,
        depth=6,
        heads=8,
        mlp_dim=256,
        channels = 3,
        use_flash = True
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Batch processing function
    def train_batch(data, target):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Return metrics
        _, predicted = output.max(1)
        correct = predicted.eq(target).sum().item()
        total = target.size(0)
        return loss.item(), correct, total

    # Test function
    def test_model():
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Timing variables with torch.cuda.Event for precise measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        total_inference_time_ms = 0.0  # Using milliseconds for better precision
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # Synchronize and start timing
                torch.cuda.synchronize()
                start_event.record()
                
                output = model(data)
                
                # End timing and synchronize
                end_event.record()
                torch.cuda.synchronize()
                
                # Accumulate time in milliseconds
                batch_time_ms = start_event.elapsed_time(end_event)
                total_inference_time_ms += batch_time_ms
                
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        # Safely calculate metrics with zero division protection
        avg_loss = test_loss / max(1, len(test_loader))  # Avoid division by zero
        
        accuracy = 0.0
        if test_total > 0:
            accuracy = 100. * test_correct / test_total
        
        # Convert to seconds for final reporting
        total_inference_time_sec = total_inference_time_ms / 1000.0
        
        # Calculate timing statistics with safeguards
        avg_batch_time_ms = total_inference_time_ms / max(1, len(test_loader))
        avg_batch_time_sec = avg_batch_time_ms / 1000.0
        
        images_per_second = 0.0
        if total_inference_time_sec > 1e-6:  # Only calculate if time > 1 microsecond
            images_per_second = test_total / total_inference_time_sec
        
        print(f'\nTest Results:')
        print(f'Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}%')
        print(f'Timing:')
        print(f'- Total inference time: {total_inference_time_sec:.6f} seconds')
        print(f'- Average per batch: {avg_batch_time_sec:.6f} seconds')
        if images_per_second > 0:
            print(f'- Throughput: {images_per_second:.2f} images/second')
        else:
            print('- Throughput: Not measurable (time too short)')
        
        return {
            'avg_loss': avg_loss,
            'accuracy': accuracy,
            'total_time_sec': total_inference_time_sec,
            'throughput_ips': images_per_second,
            'total_samples': test_total
        }
    
    
    # Main training loop
    profile_completed = False
    
    profiler_dir = setup_profiler_output_dir(args)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Profiling context (only runs once)
        if args.run_profiler and not profile_completed:
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Run training step
                    loss, correct, total = train_batch(data, target)
                    
                    # Update metrics
                    epoch_loss += loss
                    epoch_correct += correct
                    epoch_total += total
                    
                    # Profiler step
                    prof.step()
                    
                    # Check if profiling should stop
                    if batch_idx >= 1 + 1 + 3:  # wait + warmup + active
                        profile_completed = True
                        break
                    
                    
            
            print("Profiling completed. Continuing normal training...")
        
        # Normal training loop (after profiling or if profiling not enabled)
        for batch_idx, (data, target) in enumerate(train_loader):
            # Skip already processed batches from profiling
            if args.run_profiler and epoch == 0 and batch_idx <= 1 + 1 + 3:
                continue
                
            # Normal training step
            loss, correct, total = train_batch(data, target)
            
            # Update metrics
            epoch_loss += loss
            epoch_correct += correct
            epoch_total += total
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} Batch: {batch_idx} Loss: {loss:.4f}')

        # Epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * epoch_correct / epoch_total
        print(f'Epoch {epoch} Complete: Avg Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%')

    # Final testing
    test_model()
    return model

if __name__ == '__main__':
    train_test_model()