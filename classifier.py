def classifier(image_path, model_arch):
    """
    Mock classifier function.
    In practice, this function would use a pre-trained model to classify the image.
    
    Args:
    - image_path (str): Path to the image file.
    - model_arch (str): Model architecture ('resnet', 'alexnet', 'vgg').
    
    Returns:
    - str: Predicted label of the image.
    """
    # Mock predictions
    mock_predictions = {
        'resnet': 'beagle',
        'alexnet': 'german shepherd',
        'vgg': 'kuvasz'
    }
    return mock_predictions.get(model_arch, 'unknown')
