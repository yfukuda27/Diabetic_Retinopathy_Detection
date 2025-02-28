# Diabetic Retinopathy Stage Classification Model

## Project Overview
This project aims to develop a deep learning model that automatically classifies stages of Diabetic Retinopathy (DR) using fundus images. The model classifies DR into 5 stages (0-4), providing a crucial tool for early detection and monitoring of DR progression.

<details>
<summary><b>DR Stages Description</b></summary>

| Stage | Description | Visual Characteristics |
|-------|-------------|------------------------|
| 0 | No DR | Normal retina without abnormalities |
| 1 | Mild DR | Presence of microaneurysms |
| 2 | Moderate DR | Microaneurysms, hard exudates, cotton wool spots |
| 3 | Severe DR | Multiple hemorrhages, venous beading |
| 4 | Proliferative DR | Neovascularization, vitreous hemorrhage |

</details>

## 🚀 Model Architecture and Performance

### Key Features
- **Base Model**: ResNet152V2 (with ImageNet pre-trained weights)
- **Input Image Size**: 380 x 380 x 3
- **Transfer Learning Applied**:
  - CNN feature extractor frozen
  - Custom top classification layers (Dense layers)
- **Loss Function**: Focal Loss for class imbalance handling
- **Data Preprocessing**: 
  - Pixel normalization (0-1 range)
  - Contrast enhancement
  - Gaussian blur for noise reduction

### 📊 Performance Metrics
python
{
'accuracy': 0.84,
'precision': 0.83,
'recall': 0.82,
'f1_score': 0.82
}

## 📂 Dataset Structure
dataset/
├── train_images/
│   ├── stage0/
│   ├── stage1/
│   ├── stage2/
│   ├── stage3/
│   └── stage4/
├── val_images/
└── test_images/


### Dataset Distribution
- **Total Images**: 5,000
  - Training: 3,500
  - Validation: 500
  - Test: 1,000

<details>
<summary><b>Class Distribution Details</b></summary>

| Stage | Count | Percentage |
|-------|--------|------------|
| 0 | 1,500 | 42.8% |
| 1 | 800 | 22.9% |
| 2 | 600 | 17.1% |
| 3 | 400 | 11.4% |
| 4 | 200 | 5.7% |

</details>

## 🛠 Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA compatible GPU (recommended)
- Google Colab (A100 GPU) for training

### Installation Steps
1. Clone the repository:

```bash
git clone https://github.com/yfukuda27/Diabetic_Retinopathy_Detection.git
cd dr-classification
```
2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔄 Training Process

### Model Training Configuration

```python
config = {
'batch_size': 32,
'epochs': 100,
'learning_rate': 1e-4,
'optimizer': 'Adam',
'image_size': (380, 380),
'augmentation': True
}
```

### Data Augmentation Techniques
- Random rotation (±20°)
- Random zoom (0.9-1.1)
- Horizontal/Vertical flip
- Random brightness/contrast adjustment

## 📈 Results and Visualization

<div align="center">
  <img src="assets/training_curves.png" alt="Training Curves" width="600"/>
  <p><i>Training and Validation Curves</i></p>
</div>

## 📈 Future Improvements
1. **Data Collection**:
   - Increase samples for Stage 4 (Proliferative DR)
   - Enhance data quality through better preprocessing
2. **Model Enhancement**:
   - Experiment with ensemble methods
   - Implement attention mechanisms
3. **Performance Goals**:
   - Improve Stage 4 classification accuracy
   - Target 90%+ overall accuracy based on Kaggle benchmarks

## 📚 References
1. [Kaggle DR Detection](https://www.kaggle.com/code/sovitrath/diabetic-retinopathy-fastai)
2. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
3. Lin, T.Y., et al. (2017). "Focal Loss for Dense Object Detection"

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors
- [Yuji Fukuda](https://github.com/yfukuda27)

## 📮 Contact
- Email: yuji050808@gmail.com
- Project Link: [GitHub](https://github.com/yfukuda27/Diabetic_Retinopathy_Detection)

---
<div align="center">
  <p>If you find this project helpful, please consider giving it a ⭐</p>
</div>

## 🔬 Technical Implementation Details

### Preprocessing Pipeline
```python
preprocessing_steps = {
    'image_enhancement': [
        'Pixel Normalization (0-1 range)',
        'Contrast Enhancement',
        'Gaussian Blur for Noise Reduction'
    ]
}
```

### Model Architecture Details
- **Base Model**: ResNet152V2 with ImageNet weights (frozen)
- **Custom Layers**:
  - Global Average Pooling
  - Dense Layers for Classification
  - Dropout for Regularization
- **Final Layer**: Dense(5, activation='softmax')

## 📊 Model Performance Analysis
| Stage | Precision | Recall | F1-Score |
|-------|-----------|--------|-----------|
| 0 (No DR) | 0.86 | 0.88 | 0.87 |
| 1 (Mild) | 0.84 | 0.83 | 0.83 |
| 2 (Moderate) | 0.82 | 0.81 | 0.81 |
| 3 (Severe) | 0.83 | 0.82 | 0.82 |
| 4 (Proliferative) | 0.78 | 0.76 | 0.77 |

## 💡 Key Features
1. **Transfer Learning**: Utilizing pre-trained ResNet152V2
2. **Class Imbalance Handling**: Focal Loss implementation
3. **Data Preprocessing**: Standardized image processing pipeline

## 🔄 Training Process
- Training Environment: Google Colab (A100 GPU)
- Batch Size: 32
- Epochs: 100
- Optimizer: Adam (lr=1e-4)
- Best Validation Accuracy: 84%
