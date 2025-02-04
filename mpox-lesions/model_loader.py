from swin import SwinTransformer
from vnet import VNet


def get_model(model_name, num_classes=None, input_channels=1):
    """
    Returns the specified model architecture.

    Args:
        model_name (str): The name of the model to initialize. Options: "swin", "vnet".
        num_classes (int): Number of output classes (for classification tasks). Required for Swin.
        input_channels (int): Number of input channels (for segmentation tasks). Required for V-Net.

    Returns:
        nn.Module: The initialized model.
    """
    if model_name.lower() == "swin":
        if num_classes is None:
            raise ValueError("For Swin Transformer, 'num_classes' must be specified.")
        return SwinTransformer(input_channels=3, num_classes=num_classes)

    elif model_name.lower() == "vnet":
        return VNet(input_channels=input_channels, num_classes=num_classes)

    else:
        raise ValueError(
            f"Unsupported model name: {model_name}. Choose 'swin' or 'vnet'."
        )
