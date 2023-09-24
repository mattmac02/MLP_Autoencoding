import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import autoencoderMLP4Layer  # Import your autoencoder model

# Define a transform to convert images to tensors
train_transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
train_set = datasets.MNIST('./data/mnist', train=True, download=True, transform=train_transform)

# Load the trained autoencoder model
model = autoencoderMLP4Layer(N_input=784, N_bottleneck=8, N_output=784)  # Adjust model configuration if needed
model.load_state_dict(torch.load("MLP.8.pth"))
model.eval()  # Set the model to evaluation mode
n_steps = 8

# Prompt the user for an index
try:
    idx1 = int(input("Enter an integer index between 0 and 59999: "))  # Index of the first image
    idx2 = int(input("Enter an integer index between 0 and 59999: "))  # Index of the second image

    # Check if the index is within the valid range
    if 0 <= idx1 < len(train_set):
        # Get the original image
        image1, label = train_set[idx1]
        image2, label = train_set[idx2]

        noise = torch.rand_like(image1)  # Create random noise with the same shape as the image
        noisy_image = image1 + noise
        with torch.no_grad():
            reconstructed_image_noisy = model(noisy_image.view(-1, 28 * 28))
            reconstructed_image = model(image1.view(-1, 28 * 28))
            bottleneck1 = model.encode(image1.view(-1, 28 * 28))
            bottleneck2 = model.encode(image2.view(-1, 28 * 28))
        noisy_image = noisy_image.squeeze().numpy()

        interpolated_bottlenecks = []
        for alpha in torch.linspace(0, 1, n_steps):
            interpolated_bottleneck = alpha * bottleneck1 + (1 - alpha) * bottleneck2
            interpolated_bottlenecks.append(interpolated_bottleneck)

        interpolated_images = []
        for interpolated_bottleneck in interpolated_bottlenecks:
            with torch.no_grad():
                interpolated_image = model.decode(interpolated_bottleneck)
                interpolated_images.append(interpolated_image.view(28, 28).squeeze().numpy())

        # Plot the original images and the interpolated images
        plt.figure(figsize=(16, 4))
        plt.subplot(1, n_steps + 2, 1)
        plt.title("Starting Image")
        plt.imshow(image2.squeeze(), cmap='gray')
        plt.axis('off')

        for i, interpolated_image in enumerate(interpolated_images):
            plt.subplot(1, n_steps + 2, i + 2)
            plt.title(f"Interpolation {i + 1}")
            plt.imshow(interpolated_image, cmap='gray')
            plt.axis('off')
            plt.tight_layout(w_pad=4.0)

        plt.subplot(1, n_steps, n_steps)
        plt.title("Final Image")
        plt.imshow(image1.squeeze(), cmap='gray')
        plt.axis('off')

        plt.tight_layout(w_pad=4.0)

        # Convert tensors to numpy arrays for visualization
        image = image1.squeeze().numpy()
        reconstructed_image_noisy = reconstructed_image_noisy.view(28, 28).squeeze().numpy()
        reconstructed_image = reconstructed_image.view(28, 28).squeeze().numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Reconstructed Image")
        plt.imshow(reconstructed_image, cmap='gray')
        plt.axis('off')

        plt.tight_layout(w_pad=4.0)

        # Display the input and reconstructed output images side by side
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Reconstructed Image")
        plt.imshow(reconstructed_image_noisy, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Noisy Image")
        plt.imshow(noisy_image, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Index is out of range.")
except ValueError:
    print("Invalid input. Please enter a valid integer index.")