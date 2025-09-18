# MNIST Classification Model

## **Summary**
This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch, achieving **99.52%** test accuracy in just 10 epochs through optimized architecture, data augmentation, and learning rate scheduling.

### **Key Metrics**
- **🔗 Google Colab URL**: [Google colab URL](https://colab.research.google.com/drive/1DR63wMR3Dxer-H4fzi1HdDCeBLH6meW_#scrollTo=66OCGkHhRWov)
- **📊 Number of Parameters**: **9,594**
- **🎯 Final Test Accuracy**: **99.52%**
- **⏱️ Training Epochs**: **10**

## **Model Performance Visualization**
![Accuracy and Loss Curves](predictions_visualization.png)

## **Architecture**

The model uses a lightweight CNN architecture with three convolutional blocks designed for efficient feature extraction:

### **Block 1 (Feature Detection)**
```
Input: 28×28×1 (grayscale MNIST image)
├── Conv2d(1→8, kernel=3×3) → BatchNorm → ReLU → 26×26×8
├── Conv2d(8→16, kernel=3×3) → BatchNorm → ReLU → 24×24×16
└── MaxPool2d(2×2) → 12×12×16
```

### **Block 2 (Feature Refinement)**
```
Input: 12×12×16
├── Conv2d(16→16, kernel=3×3) → BatchNorm → ReLU → 10×10×16
├── Conv2d(16→16, kernel=3×3) → BatchNorm → ReLU → 8×8×16
└── MaxPool2d(2×2) → 4×4×16
```

### **Block 3 (High-level Features)**
```
Input: 4×4×16
├── Conv2d(16→16, kernel=3×3) → BatchNorm → ReLU → 2×2×16
└── Conv2d(16→16, kernel=2×2) → BatchNorm → ReLU → 1×1×16
```

### **Classification Head**
```
Input: 1×1×16 (flattened to 16)
├── Dropout(0.1)
├── Linear(16→10)
└── LogSoftmax → 10 class probabilities
```

### **Architecture Highlights**
- **Parameter Efficiency**: Only 9,594 parameters through strategic channel sizing
- **Regularization**: Batch normalization after each conv layer + dropout before classification
- **Activation**: ReLU throughout for computational efficiency

## **Data Augmentation**
Training data augmentation includes:
- **Random Rotation**: ±15 degrees to handle rotated digits
- **Random Translation**: ±2 pixels in both directions for positional invariance
- **Normalization**: Mean=0.1307, Std=0.3081 (MNIST dataset statistics)

Test data uses only normalization (no augmentation for consistent evaluation).

## **Learning Rate Schedule**
**OneCycleLR Scheduler** with Adagrad optimizer:
- **Base LR**: 0.1 (starting and ending learning rate)
- **Max LR**: 0.75 (peak learning rate at 30% of training)
- **Div Factor**: 7.5 (base_lr = max_lr / div_factor)
- **Strategy**: Cosine annealing for smooth transitions
- **Warmup**: 30% of total steps for gradual ramp-up
- **Momentum Cycling**: Disabled for Adagrad compatibility
- **Steps per Epoch**: Scheduler updates after each batch (not epoch)

## **Training Results**

```
Starting training...

Epoch 1/10
Training: 100%|██████████| 469/469 [01:21<00:00,  5.78it/s]
Testing: 100%|██████████| 79/79 [00:04<00:00, 18.79it/s]
Train Loss: 0.2876, Train Acc: 91.12%
Test Loss: 0.0772, Test Acc: 97.55%
Learning Rate: 0.100000

Epoch 2/10
Training: 100%|██████████| 469/469 [01:19<00:00,  5.91it/s]
Testing: 100%|██████████| 79/79 [00:03<00:00, 24.13it/s]
Train Loss: 0.1437, Train Acc: 95.78%
Test Loss: 0.0626, Test Acc: 97.90%
Learning Rate: 0.262710

Epoch 3/10
Training: 100%|██████████| 469/469 [01:21<00:00,  5.76it/s]
Testing: 100%|██████████| 79/79 [00:03<00:00, 23.94it/s]
Train Loss: 0.1140, Train Acc: 96.75%
Test Loss: 0.0477, Test Acc: 98.50%
Learning Rate: 0.587919

Epoch 4/10
Training: 100%|██████████| 469/469 [01:22<00:00,  5.71it/s]
Testing: 100%|██████████| 79/79 [00:03<00:00, 23.29it/s]
Train Loss: 0.0947, Train Acc: 97.26%
Test Loss: 0.0331, Test Acc: 99.00%
Learning Rate: 0.750000

Epoch 5/10
Training: 100%|██████████| 469/469 [01:20<00:00,  5.83it/s]
Testing: 100%|██████████| 79/79 [00:03<00:00, 23.89it/s]
Train Loss: 0.0787, Train Acc: 97.76%
Test Loss: 0.0315, Test Acc: 99.01%
Learning Rate: 0.713370

Epoch 6/10
Training: 100%|██████████| 469/469 [01:20<00:00,  5.83it/s]
Testing: 100%|██████████| 79/79 [00:04<00:00, 16.58it/s]
Train Loss: 0.0666, Train Acc: 98.07%
Test Loss: 0.0259, Test Acc: 99.21%
Learning Rate: 0.611043

Epoch 7/10
Training: 100%|██████████| 469/469 [01:20<00:00,  5.82it/s]
Testing: 100%|██████████| 79/79 [00:03<00:00, 23.59it/s]
Train Loss: 0.0606, Train Acc: 98.27%
Test Loss: 0.0182, Test Acc: 99.41%
Learning Rate: 0.463285

Epoch 8/10
Training: 100%|██████████| 469/469 [01:20<00:00,  5.85it/s]
Testing: 100%|██████████| 79/79 [00:03<00:00, 23.78it/s]
Train Loss: 0.0503, Train Acc: 98.58%
Test Loss: 0.0175, Test Acc: 99.49%
Learning Rate: 0.299361

Epoch 9/10
Training: 100%|██████████| 469/469 [01:20<00:00,  5.83it/s]
Testing: 100%|██████████| 79/79 [00:04<00:00, 19.23it/s]
Train Loss: 0.0480, Train Acc: 98.64%
Test Loss: 0.0166, Test Acc: 99.52%
Learning Rate: 0.151739

Epoch 10/10
Training: 100%|██████████| 469/469 [01:20<00:00,  5.84it/s]
Testing: 100%|██████████| 79/79 [00:03<00:00, 23.93it/s]
Train Loss: 0.0435, Train Acc: 98.70%
Test Loss: 0.0164, Test Acc: 99.52%
Learning Rate: 0.049657
Training completed!

============================================================
TRAINING METRICS SUMMARY
============================================================
 Epoch  Train_Loss  Test_Loss  Train_Accuracy  Test_Accuracy  Learning_Rate
     1    0.287639   0.077175       91.118333          97.55       0.100000
     2    0.143723   0.062647       95.775000          97.90       0.262710
     3    0.114044   0.047663       96.755000          98.50       0.587919
     4    0.094663   0.033140       97.256667          99.00       0.750000
     5    0.078692   0.031542       97.760000          99.01       0.713370
     6    0.066570   0.025928       98.073333          99.21       0.611043
     7    0.060624   0.018198       98.273333          99.41       0.463285
     8    0.050252   0.017530       98.580000          99.49       0.299361
     9    0.047962   0.016631       98.645000          99.52       0.151739
    10    0.043471   0.016378       98.700000          99.52       0.049657
============================================================
```



*Training curves and sample predictions will be generated when running the model*
