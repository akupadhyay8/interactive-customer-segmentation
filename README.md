# 🧠 Interactive Customer Segmentation Dashboard

This project is an **interactive Streamlit dashboard** that allows users to explore and understand customer segments using **K-Means Clustering**. It’s designed to help businesses better understand their customer base by grouping them based on purchasing behavior and demographics.

---

## 🚀 Demo

👉 **Live Streamlit App:** [Click here to explore](https://interactive-customer-segmentation-yqbhn9gajleouy95smecyz.streamlit.app/) 

---

## 📊 Project Overview

**Key Features:**
- Visualizes clusters using 2D scatter plots.
- Allows users to interactively select the number of clusters.
- Explains cluster characteristics clearly.
- Segmentation Explorer to analyze customer demographics and purchasing patterns.

**Technologies Used:**
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Joblib

---

## 📁 Dataset

The dataset used is from Kaggle and contains 200 customer records including:
- Customer ID
- Age
- Annual Income (k$)
- Spending Score (1–100)

📥 **[Download Dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)**

---

## 📂 Project Structure

```
├── Dataset/
│ └── Mall_Customers.csv
├── app.py
├── kmeans_model.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 How It Works

1. **Data Preprocessing**: Cleans the dataset and scales the relevant features.
2. **Model Training**: Applies K-Means clustering and stores the model.
3. **Dashboard**:
   - Users can explore how customers are segmented into clusters.
   - Visuals and metrics help interpret results.
   - Segment Explorer provides insights into age, income, and spending habits.

---

## 🛠️ Installation & Usage

### ⚙️ Local Setup

# Clone the repo
```
git clone https://github.com/akupadhyay8/interactive-customer-segmentation.git
cd interactive-customer-segmentation
```
# Install dependencies
```
pip install -r requirements.txt
```
# Run the app
```
streamlit run app.py
```

