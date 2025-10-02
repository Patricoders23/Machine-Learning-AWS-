# 🤖 Machine Learning with AWS

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![AWS](https://img.shields.io/badge/AWS-Machine_Learning-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)

> Collection of Machine Learning projects developed as part of the AWS Machine Learning course, focused on solving real-world business problems using classical datasets and industry best practices.

<div align="center">

**🎯 92% Recall in Fraud Detection | 84% Accuracy in Flight Prediction | Production-Ready Code**

</div>

---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [Projects Included](#-projects-included)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Roadmap](#️-roadmap)
- [Contributing](#-contributing)
- [Contact](#-contact)
- [License](#-license)

---

## 🎯 About the Project

This repository contains practical implementations of Machine Learning algorithms applied to real-world business use cases. Each project follows industry standards with comprehensive exploratory data analysis, data preprocessing, feature engineering, model training, and thorough performance evaluation.

### 🎓 Learning Objectives

- ✅ Implement complete end-to-end Machine Learning pipelines
- ✅ Apply advanced preprocessing and feature engineering techniques
- ✅ Evaluate and compare multiple algorithms systematically
- ✅ Interpret performance metrics in business contexts
- ✅ Write clean, documented, and reproducible code
- ✅ Deploy ML solutions following AWS best practices

### 💡 Why This Project Matters

These projects demonstrate practical ML skills that translate directly to industry needs:
- Real datasets with real challenges (imbalanced data, missing values, etc.)
- Business-focused problem statements
- Production-ready code structure
- Clear documentation and reproducibility

---

## 📁 Projects Included

### 1. 🔐 Credit Card Fraud Detection

**Business Problem:** Financial institutions lose billions annually to credit card fraud. This project builds a real-time fraud detection system to minimize financial losses while maintaining customer experience.

**Technical Challenge:** Highly imbalanced dataset (fraud represents <0.2% of transactions)

**Applied Techniques:**
- Handling imbalanced datasets (SMOTE, Random Undersampling)
- Random Forest and XGBoost classifiers
- Hyperparameter tuning with GridSearchCV
- Recall optimization to minimize false negatives
- Feature importance analysis
- ROC-AUC and Precision-Recall curves

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions
- 492 frauds (0.172% of all transactions)
- 30 features (anonymized via PCA)

**Key Metrics:**
- **Recall:** 92% (catches 92% of actual fraud cases)
- **Precision:** 88% (low false positive rate)
- **F1-Score:** 0.90
- **AUC-ROC:** 0.96

**Business Impact:**
- Potential 40% reduction in fraud-related losses
- Improved customer trust through reduced false declines
- Real-time scoring capability

**Technologies:** Python, Scikit-learn, Pandas, NumPy, Imbalanced-learn, XGBoost, Matplotlib

---

### 2. ✈️ Flight Delay Prediction

**Business Problem:** Flight delays cost airlines billions in operational costs and damage customer satisfaction. This predictive model enables proactive decision-making and improves passenger experience.

**Technical Challenge:** Complex temporal patterns, seasonal effects, and multiple categorical variables

**Applied Techniques:**
- Feature engineering with temporal variables (hour, day, month, season)
- One-hot encoding for categorical features
- Seasonal pattern analysis
- Logistic Regression baseline
- Decision Trees and Random Forest comparison
- Cross-validation to prevent overfitting
- Feature importance visualization

**Dataset:** U.S. Department of Transportation flight data
- Historical flight records
- Weather conditions
- Airline performance metrics
- Airport congestion data

**Key Metrics:**
- **Accuracy:** 84% on test set
- **Precision:** 81%
- **Recall:** 79%
- **Top Predictive Features:** 
  - Time of day (departure hour)
  - Airline carrier
  - Route (origin-destination pair)
  - Day of week
  - Historical airport performance

**Business Impact:**
- Early warning system for passengers (2-4 hours in advance)
- Improved resource allocation for airlines
- Better crew scheduling and gate management
- Enhanced customer satisfaction through proactive communication

**Technologies:** Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

### 3. 📊 Performance Metrics Analysis in Machine Learning

**Business Problem:** Choosing the wrong evaluation metric can lead to models that perform well on paper but fail in production. This project provides a comprehensive guide to selecting appropriate metrics.

**Educational Focus:** Understanding the trade-offs between different metrics and when to use each

**Topics Covered:**
- **Classification Metrics:**
  - Accuracy vs. Balanced Accuracy
  - Precision vs. Recall trade-off
  - F1-Score and F-beta Score
  - ROC-AUC and PR-AUC
  - Confusion Matrix interpretation
- **When to use what:**
  - Imbalanced datasets → Precision-Recall
  - Cost-sensitive problems → Custom thresholds
  - Multi-class problems → Macro vs. Micro averaging
- **Threshold Optimization:**
  - ROC curve analysis
  - Precision-Recall curves
  - Business-driven threshold selection

**Practical Examples:**
- Medical diagnosis (minimize false negatives)
- Spam detection (minimize false positives)
- Fraud detection (balance precision and recall)

**Key Learnings:**
- Accuracy is NOT always the best metric
- Context determines metric selection
- Visual analysis aids understanding
- Business costs should drive threshold decisions

**Technologies:** Python, Scikit-learn, Matplotlib, Seaborn

---

## 🛠️ Technologies Used

### Core Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.8+ | Primary programming language |
| **Pandas** | ≥1.5.0 | Data manipulation and analysis |
| **NumPy** | ≥1.23.0 | Numerical computing |
| **Scikit-learn** | ≥1.2.0 | ML algorithms and metrics |
| **Matplotlib** | ≥3.6.0 | Static visualizations |
| **Seaborn** | ≥0.12.0 | Statistical data visualization |
| **Jupyter** | ≥1.0.0 | Interactive development environment |

### Advanced Libraries

| Library | Purpose |
|---------|---------|
| **XGBoost** | Gradient boosting for high-performance models |
| **Imbalanced-learn** | SMOTE and resampling techniques |
| **Plotly** | Interactive visualizations (optional) |

### Development Tools

- **Git/GitHub** - Version control and collaboration
- **VS Code / PyCharm** - IDE
- **Conda / venv** - Environment management

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git (for cloning)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Patricoders23/Machine-Learning-AWS-.git
cd Machine-Learning-AWS-

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook
```

### Alternative: Using Conda

```bash
# Create conda environment
conda create -n ml-aws python=3.8
conda activate ml-aws

# Install dependencies
pip install -r requirements.txt
```

### Requirements File

Create a `requirements.txt` file with:

```txt
# Core Data Science Libraries
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0

# Machine Learning
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0

# Jupyter
jupyter>=1.0.0
notebook>=6.5.0

# Utilities
tqdm>=4.65.0
joblib>=1.2.0

# Optional: Advanced visualization
plotly>=5.13.0
```

---

## 💻 Usage

### Running the Projects

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Navigate to desired project folder**

3. **Open the notebook** (`.ipynb` file)

4. **Run cells sequentially** (Shift + Enter)

### Notebook Structure

Each notebook follows this standard structure:

```
📝 Project Title
│
├── 1️⃣ Introduction
│   ├── Problem statement
│   ├── Business context
│   └── Success criteria
│
├── 2️⃣ Data Loading & Overview
│   ├── Import libraries
│   ├── Load dataset
│   └── Initial exploration
│
├── 3️⃣ Exploratory Data Analysis (EDA)
│   ├── Statistical summary
│   ├── Distribution analysis
│   ├── Correlation analysis
│   └── Visualization
│
├── 4️⃣ Data Preprocessing
│   ├── Handle missing values
│   ├── Feature scaling/encoding
│   ├── Handle imbalanced data (if applicable)
│   └── Train-test split
│
├── 5️⃣ Feature Engineering
│   ├── Create new features
│   ├── Feature selection
│   └── Feature importance
│
├── 6️⃣ Model Training
│   ├── Baseline model
│   ├── Algorithm comparison
│   ├── Hyperparameter tuning
│   └── Final model selection
│
├── 7️⃣ Model Evaluation
│   ├── Performance metrics
│   ├── Confusion matrix
│   ├── ROC/PR curves
│   └── Cross-validation results
│
└── 8️⃣ Conclusions & Next Steps
    ├── Key findings
    ├── Business recommendations
    ├── Limitations
    └── Future improvements
```

---

## 📈 Key Results

### Summary of Achievements

| Project | Main Metric | Score | Business Impact |
|---------|-------------|-------|-----------------|
| **Fraud Detection** | Recall | 92% | 40% reduction in fraud losses |
| **Flight Delays** | Accuracy | 84% | 2-4h early warnings |
| **Metrics Analysis** | Educational | N/A | Improved metric selection |

### Detailed Results

#### 🔐 Fraud Detection
```python
Classification Report:
                precision    recall  f1-score
    No Fraud       0.99      0.98      0.99
    Fraud          0.88      0.92      0.90
    
ROC-AUC Score: 0.96
```

#### ✈️ Flight Prediction
```python
Model Performance:
    Accuracy:  84%
    Precision: 81%
    Recall:    79%
    F1-Score:  80%
```

---

## 📁 Project Structure

```
Machine-Learning-AWS/
│
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
│
├── notebooks/
│   ├── 01_fraud_detection/
│   │   ├── fraud_detection.ipynb    # Main notebook
│   │   ├── utils.py                 # Helper functions
│   │   └── README.md                # Project-specific docs
│   │
│   ├── 02_flight_delays/
│   │   ├── flight_prediction.ipynb
│   │   ├── data_preprocessing.py
│   │   └── README.md
│   │
│   └── 03_metrics_analysis/
│       ├── metrics_analysis.ipynb
│       ├── visualization_utils.py
│       └── README.md
│
├── data/                             # Data folder (gitignored)
│   ├── raw/                         # Original datasets
│   ├── processed/                   # Cleaned datasets
│   └── README.md                    # Data documentation
│
├── models/                           # Saved models (gitignored)
│   └── .gitkeep
│
├── reports/                          # Generated reports
│   ├── figures/                     # Saved visualizations
│   └── results/                     # Performance metrics
│
└── docs/                             # Additional documentation
    ├── setup_guide.md
    └── troubleshooting.md
```

---

## 🗺️ Roadmap

### Short-term (Next Month)
- [ ] Add unit tests for preprocessing functions
- [ ] Create Docker container for easy deployment
- [ ] Add data validation scripts
- [ ] Improve documentation with more examples

### Medium-term (2-3 Months)
- [ ] Implement Deep Learning models (TensorFlow/PyTorch)
- [ ] Create REST API with FastAPI
- [ ] Add model interpretability (SHAP, LIME)
- [ ] Deploy on AWS SageMaker
- [ ] Create Streamlit dashboard

### Long-term (6+ Months)
- [ ] Implement MLOps pipeline with MLflow
- [ ] Add A/B testing framework
- [ ] Create comprehensive model monitoring
- [ ] Automated retraining pipeline
- [ ] Production deployment on AWS

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Add docstrings to functions
- Update README if needed
- Add tests for new features

---

## 👩‍💻 About the Author

**Patricia Lugo**  
*Petroleum Engineer → Data Scientist | ML Engineer*

I'm a Petroleum Engineer successfully transitioning into Data Science and Machine Learning. I combine engineering rigor with data-driven innovation to solve complex business problems.

**Background:**
- 🎓 Petroleum Engineering
- 📊 Specialization in Data Science & Machine Learning
- ☁️ AWS & GCP Cloud Computing
- 📍 Currently based in **Berlin, Germany** 🇩🇪

**Core Competencies:**
- Machine Learning & Deep Learning
- Python (Pandas, Scikit-learn, TensorFlow, PyTorch)
- Cloud Computing (AWS: EC2, S3, SageMaker | GCP)
- Big Data (Apache Spark)
- Data Visualization (Power BI, Tableau)
- MLOps & Model Deployment

---

## 📫 Contact & Connect

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/patricia-lugo)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Patricoders23)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-000000?style=for-the-badge&logo=About.me&logoColor=white)](your-portfolio-url)

**📍 Location:** Berlin, Germany | **💼 Status:** Open to opportunities

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

```
Copyright (c) 2024 Patricia Lugo

Permission is hereby granted, free of charge, to use, copy, modify, merge, 
publish, distribute, sublicense, and/or sell copies of the Software.
```

---

## 🙏 Acknowledgments

- **AWS** for the comprehensive Machine Learning course and resources
- **Kaggle** for providing high-quality public datasets
- **Scikit-learn** community for excellent documentation
- **Machine Learning community** for open-source contributions
- **Stack Overflow** for problem-solving support

---

## 📚 Additional Resources

### Learning Materials
- [AWS Machine Learning Training](https://aws.amazon.com/training/learn-about/machine-learning/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Related Projects
- [My Data Science Portfolio](your-portfolio-link)
- [Other ML Projects](your-other-repos)

### Recommended Reading
- "Hands-On Machine Learning" by Aurélien Géron
- "The Elements of Statistical Learning"
- "Python Data Science Handbook" by Jake VanderPlas

---

<div align="center">

### ⭐ If you find this project useful, please give it a star!

**Questions? Suggestions? Opportunities?**  
Feel free to [open an issue](https://github.com/Patricoders23/Machine-Learning-AWS-/issues) or contact me directly.

---

**Built with ❤️ in Berlin** 🇩🇪

![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)

**Last Updated:** October 2024

</div>
