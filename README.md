# MNIST MLP Autoencoder

This repository contains code for training an MLP-based autoencoder. Follow the instructions below to run the training:

1. Open your terminal.

2. Navigate to the project directory.

3. Run the following command to start training:

python train.py -z 8 -e 50 -b 2048 -s MLP.8.pth -p loss.MLP.8.png

- `-z 8`: Set the latent dimension to 8.
- `-e 50`: Train for 50 epochs.
- `-b 2048`: Use a batch size of 2048.
- `-s MLP.8.pth`: Save the trained model as "MLP.8.pth".
- `-p loss.MLP.8.png`: Save the loss plot as "loss.MLP.8.png".

Feel free to adjust the command line arguments to suit your specific training needs.
