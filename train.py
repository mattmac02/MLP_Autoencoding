import argparse
import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from model import autoencoderMLP4Layer
import matplotlib.pyplot as plt

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device):
    print("Training...")
    model.train()
    losses_train = []

    for epoch in range(1, n_epochs + 1):# Use n_epochs instead of args.epochs
        print("Epoch", epoch)
        loss_train = 0.0

        for batch_imgs, _ in train_loader:# Assuming you have labels, use "_"
            # Reshape and preprocess the batch_imgs to match the model's input shape
            batch_imgs = batch_imgs.view(batch_imgs.size(0), -1).to(device) # Flatten input
            outputs = model(batch_imgs)
            loss = loss_fn(outputs, batch_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        scheduler.step(loss_train)

        losses_train.append(loss_train / len(train_loader))

        print("{} Epoch {}, Training Loss {}".format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))
        plt.figure()
        plt.plot(range(1, epoch + 1), losses_train, marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss vs. Epoch')
        plt.grid(True)
        plt.savefig(args.plot_path)
        plt.close()

    torch.save(model.state_dict(), args.save_path)
    print("Saved model to", args.save_path)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Autoencoder Training")

    # Define command line arguments
    parser.add_argument("-z", "--bottleneck-size", type=int, default=8, help="Size of the bottleneck layer")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("-s", "--save-path", type=str, default="MLP.8.pth", help="Path to save the trained model")
    parser.add_argument("-p", "--plot-path", type=str, default="loss.MLP.8.png", help="Path to save loss plot")

    # Parse the command line arguments
    args = parser.parse_args()

    # Set up other components for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: Load the MNIST dataset (you can replace this with your dataset loading code)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # Display model summary
    model = autoencoderMLP4Layer() # Move the model to the appropriate device
    input_shape = (1, 28 * 28) # Flatten input shape
    sample_input = torch.randn(1, *input_shape).to(device=device)
    summary(model, input_size=sample_input.size())

    # Define your optimizer, loss function, and scheduler here
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

    # Call the train function with the parsed arguments and defined components
    train(args.epochs, optimizer, model, loss_fn, train_loader, scheduler, device)
