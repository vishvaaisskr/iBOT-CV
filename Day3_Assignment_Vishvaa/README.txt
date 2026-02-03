Day3 Assignment

Vishvaa

This project implements a binary image classifier that distinguishes between cats and dogs using Transfer Learning with a pretrained ResNet18 model in PyTorch. The goal was to achieve greater than 90% accuracy on the test dataset using a structured deep learning pipeline.

DATA AUGMENTATION TECHNIQUES USED:
1. Random Resized Crop  
2. Random Horizontal Flip  
3. Random Rotation (±15 degrees)  
4. Color Jitter (brightness, contrast, saturation, hue)  
5. Random Affine Transformations  


LEARNING RATE SCHEDULING:

The learning rate scheduler used in this project was:

ReduceLROnPlateau Scheduler

Configuration:
 Mode: "min"
 Patience: 3 epochs 
 Factor: 0.5

MODEL DETAILS

 Base Model: ResNet18 (pretrained on ImageNet)
 All layers frozen except final fully connected layer
 Final layer modified to output 2 classes (Cat / Dog)
 Loss Function: CrossEntropyLoss
 Optimizer: Adam Optimizer
 Training Epochs: 10
 Best model saved based on highest validation accuracy

RESULTS

 Training Accuracy reached above 90%
 Validation Accuracy reached approximately 97%
 Test Accuracy exceeded the required 90% threshold
 Confusion matrix and training curves were generated successfully