import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from optimizers.Tadam_v2 import Tadam
from models.CAE import CAE
import matplotlib.pyplot as plt
import wandb
from utils import set_seed
import copy
import argparse
from optimizers.sophia import SophiaG

# Load MNIST dataset
def load_mnist(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='/data/data', train=True, transform=transform)
    test_dataset = datasets.MNIST(root='/data/data', train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
# MSE Loss Computation
def compute_mse(train_model, x, training=True):
    if training:
        train_model.train()
    else:
        train_model.eval()
        
    x_reconstructed = train_model(x)
    loss_fn = nn.MSELoss()
    loss = loss_fn(x_reconstructed, x)
    
    return loss

# Training Step with optional closure
def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    # Standard training step without closure
    loss = compute_mse(model, data, True)
    loss.backward()
    def closure():
        optimizer.zero_grad()  # Zero the gradients
        loss = compute_mse(model, data, True)  # Compute loss
        loss.backward()        # Backward pass (calculate gradients)
        return loss
    if isinstance(optimizer, Tadam):
        optimizer.step(closure)
    else:
        optimizer.step()

    return loss.item()

# Main training loop
def train(model, device, train_loader, optimizer, epochs=100):
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        start_time = time.time()

        for batch_idx, (train_x, _) in enumerate(train_loader):
            train_x = train_x.to(device)  # Move data to the appropriate device
            loss = train_step(model, optimizer, train_x)
            train_loss += loss

        train_loss /= len(train_loader)
        losses.append(train_loss)
        end_time = time.time()

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4e}, Time: {end_time - start_time:.2f}s')

    return model, losses

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:  # We ignore the target since it's an autoencoder
            data = data.to(device)
            output = model(data)
            test_loss += criterion(output, data).item()  # Compare reconstruction with the original input
    
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

# Run experiments with Tadam and Adam
def run_experiment(model, optimizer_name='Tadam', epochs=5, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_mnist()

    # Initialize model, criterion, and optimizer
    criterion = nn.MSELoss()

    # Choose optimizer
    if optimizer_name == 'Tadam':
        optimizer = Tadam(model.parameters(), total_steps=len(train_loader) * epochs, lr=lr)
    elif optimizer_name == 'Sophia':
        optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError('Invalid optimizer name')

    # Train and test the model
    print(f'Running {optimizer_name} optimizer:')
    _, train_loss = train(model, device, train_loader, optimizer, epochs=epochs)
    
    test(model, device, test_loader, criterion)
    
    return train_loss

# Run the experiments for both optimizers
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default="Tadam", help='Number of epochs to train')
    
    args = parser.parse_args()
    optimizer_name = args.optimizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    wandb.init(project='optimizer_name')
    print(f"Experiment with {optimizer_name}:")
    in_shape = (1, 28, 28)
    filters = [16, 32, 64]
    code_dim = 16
    model = CAE(in_shape, filters, code_dim)
    optim_loss = run_experiment(copy.deepcopy(model).to(device), optimizer_name=optimizer_name, epochs=5, lr=1e-3)
    
    print("\nExperiment with Adam:")
    Adam_loss = run_experiment(copy.deepcopy(model).to(device), optimizer_name='Adam', epochs=5, lr=1e-3)
    
    # Plot the training losses and save
    plt.figure()
    plt.plot(optim_loss, label='Tadam')
    plt.plot(Adam_loss, label='Adam')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('training_losses.png')
