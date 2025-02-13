# Diff_model

# SkipUNet Diffusion Model

This neural network is an example of a simple diffusion neural network that generates images of handwritten numbers. It perfectly fits for educational purposes sinse it:
1. Simple and easy to understand for a person with little experience in neural networks.
2. It uses a well-known free MNIST datatset of handwritten numbers for training.
3. It trains on a single GPU within 20 minutes.
4. Using skip-connection layers it models the behavior of large Diffusion models.

## Example of generated image:


## Model Architecture

The `SkipUNet` model is an extension of the traditional UNet architecture, incorporating additional layers and dropout for better feature extraction and robustness. The model consists of the following components:

- **Encoder**: A series of convolutional layers with ReLU activation and dropout.
- **Middle**: An extended block with multiple convolutional layers, ReLU activation, and dropout.
- **Decoder**: A series of transposed convolutional layers with skip connections from the encoder, ReLU activation, and dropout.



## How to create environment and run the  environment:

In your `bash` terminal run:
```
conda env create -f diff_env.yaml
conda activate diff_env
```
