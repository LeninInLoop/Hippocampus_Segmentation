import torch
from torchviz import make_dot
import torch.nn as nn
from graphviz import Digraph


class UNet3DVisualizer:
    def __init__(self, model):
        self.model = model

    def generate_diagram(self, input_shape=(16, 1, 48, 64, 48), filename="UNet3D_architecture", format="png"):
        """
        Generate and save a diagram of the UNet3D model architecture.

        Args:
            input_shape (tuple): Shape of the input tensor (batch_size, channels, depth, height, width)
            filename (str): Name of the output file (without extension)
            format (str): Output format (e.g., "png", "svg", "pdf")

        Returns:
            str: Path to the saved diagram file
        """
        # Create a dummy input
        x = torch.randn(input_shape)

        # Generate the dot graph
        y = self.model(x)
        dot = make_dot(y, params=dict(self.model.named_parameters()))

        # Save the graph to a file
        output_path = dot.render(filename, format=format, cleanup=True)

        print(f"Diagram saved as {output_path}")
        return output_path

    def print_model_summary(self):
        """
        Print a summary of the model architecture.
        """
        print(self.model)
