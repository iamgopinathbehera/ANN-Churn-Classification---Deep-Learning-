# ANN Churn Classification - Deep Learning 🤖

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Demo](https://img.shields.io/badge/Live-Demo-brightgreen.svg)](https://ann-churn-classification-4snbyflxdzpy6wdpmf5oru.streamlit.app)

## 🌐 Live Demo
Try out the live demo of the Churn Classification model:
[ANN Churn Classifier Demo](https://ann-churn-classification-4snbyflxdzpy6wdpmf5oru.streamlit.app)

### Demo Features
- Real-time prediction interface
- Interactive input controls
- Instant churn probability calculation
- Visual results presentation
- Mobile-responsive design

## 📋 Table of Contents
- [Live Demo](#live-demo)
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🎯 Overview
This project implements a sophisticated customer churn prediction system using Artificial Neural Networks (ANN). By analyzing various customer attributes, the model predicts the likelihood of customer churn, providing valuable insights for business decision-making. The solution features a user-friendly Streamlit interface for real-time predictions.

## ✨ Features
- **Advanced Churn Prediction**
  - Real-time customer churn probability assessment
  - High-accuracy deep learning model
  - Interactive prediction interface

- **Robust Data Processing Pipeline**
  - Automated feature encoding (LabelEncoder, OneHotEncoder)
  - Intelligent data normalization (StandardScaler)
  - Comprehensive data validation

- **Production-Ready Deployment**
  - Streamlit Cloud hosting
  - Interactive web interface
  - Real-time predictions
  - Responsive design for all devices

## 📊 Dataset
The model is trained on a comprehensive customer dataset including:

| Feature | Type | Description |
|---------|------|-------------|
| Geography | Categorical | Customer's location |
| Gender | Categorical | Customer's gender |
| Age | Numerical | Customer's age |
| Balance | Numerical | Account balance |
| Credit Score | Numerical | Customer's credit rating |
| Estimated Salary | Numerical | Approximate annual salary |
| Number of Products | Numerical | Number of bank products used |
| Is Active Member | Boolean | Active membership status |
| Has Credit Card | Boolean | Credit card ownership |
| Tenure | Numerical | Years as a customer |

## 🛠️ Technologies Used
### Core Technologies
- **Python** (≥3.8)
- **TensorFlow & Keras** (Deep Learning)
- **Streamlit** (Web Interface)
- **Streamlit Cloud** (Deployment)

### Key Libraries
```python
dependencies = {
    'data_processing': ['numpy', 'pandas'],
    'machine_learning': ['tensorflow', 'scikit-learn'],
    'visualization': ['matplotlib', 'seaborn'],
    'deployment': ['streamlit'],
}
```

## 📁 Project Structure
```
ANN-Churn-Classification/
├── app.py                      # Streamlit application
├── model/
│   ├── model.h5               # Trained model
│   └── preprocessing/
│       ├── label_encoder_gender.pkl
│       ├── onehot_encoder_geo.pkl
│       └── scaler.pkl
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── model_development.ipynb
├── requirements.txt
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── README.md
```

## 🚀 Installation
1. **Clone Repository**
```bash
git clone https://github.com/iamgopinathbehera/ANN-Churn-Classification---Deep-Learning-
cd ANN-Churn-Classification---Deep-Learning-
```

2. **Create Virtual Environment** (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage
### Local Development
1. **Start Application**
```bash
streamlit run app.py
```

2. **Access Interface**
- Open browser: `http://localhost:8501`
- Input customer data
- Get instant churn predictions

### Using Live Demo
1. Visit [ANN Churn Classifier Demo](https://ann-churn-classification-4snbyflxdzpy6wdpmf5oru.streamlit.app)
2. Enter customer information:
   - Credit Score
   - Geography
   - Gender
   - Age
   - Tenure
   - Balance
   - Products
   - Credit Card status
   - Active Member status
   - Estimated Salary
3. Click "Predict" to get results

## 🌩️ Deployment
The application is deployed on Streamlit Cloud. Here's how to deploy your own instance:

1. **Fork the Repository**
2. **Create Streamlit Cloud Account**
   - Visit [streamlit.io](https://streamlit.io)
   - Sign up/Login with GitHub
3. **Deploy Application**
   - Select your forked repository
   - Choose main file: `app.py`
   - Configure deployment settings
4. **Access Your Deployed App**
   - Use the provided Streamlit URL
   - Share with others!

## 🧠 Model Architecture
```
Model Summary:
- Input Layer: [Customer Features]
- Hidden Layer 1: Dense(128, ReLU)
- Hidden Layer 2: Dense(64, ReLU)
- Hidden Layer 3: Dense(32, ReLU)
- Output Layer: Dense(1, Sigmoid)

Optimization:
- Loss: Binary Crossentropy
- Optimizer: Adam
- Metrics: Accuracy, AUC
```

[Previous sections remain the same...]

## 👤 Author
**Gopinath Behera**
- GitHub: [@iamgopinathbehera](https://github.com/iamgopinathbehera)
- Project Demo: [Churn Classification App](https://ann-churn-classification-4snbyflxdzpy6wdpmf5oru.streamlit.app)

[Rest of the sections remain the same...]
