# 6G IoT Beamforming Classification with Explainable AI

This project presents a complete machine learning pipeline to classify beamforming optimization outcomes in 6G Internet of Things (IoT) environments. Beamforming is critical in 6G networks to direct signal strength and improve communication quality for edge devices. The dataset used includes various environmental and device-specific features such as frequency band, signal strength, transmission power, and device type, with the goal of predicting whether beamforming was successfully optimized.

We employ a wide range of classical machine learning classifiers, including Random Forest, XGBoost, KNN, AdaBoost, and Gradient Boosting, each tuned through GridSearchCV for best performance. Performance metrics such as Accuracy, F1 Score, Precision, Recall, Confusion Matrix, and ROC-AUC are evaluated to benchmark results.

Beyond performance, we focus on **model explainability** using **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**. These tools allow us to understand the impact of each feature on the model's predictions, offering transparency and trust in model outcomes—essential for research, telecom applications, and edge deployment decisions.

The repository is designed for researchers, telecom engineers, and ML practitioners interested in robust classification, explainable AI, and 6G network optimization. All code is modular and ready to be extended for future use cases such as regression modeling or real-time deployment.

## 📦 Dataset Description

This repository uses the [6G IoT Intelligent Management Dataset](https://www.kaggle.com/datasets/ziya07/6g-iot-intelligent-management-dataset) from Kaggle, designed for analyzing and optimizing beamforming in next-generation **6G IoT networks**. The dataset integrates **network configurations, environmental settings, device profiles, computer vision outputs**, and **real-world performance indicators** to enable intelligent, AI-assisted communication system design.

### 📥 Input Features

All features in the dataset, including traditional signal parameters and real-time performance metrics, are used as input for training the classification model. This approach enables a more realistic simulation of real-world decision-making, where systems use available performance feedback to determine optimization success.

- **Network Parameters**:
  - `Frequency (GHz)`, `Transmit Power (dBm)`, `Bandwidth (MHz)`, `Codebook Size`
- **Environmental Factors**:
  - `Obstacle Density`, `Mobility (m/s)`, `Indoor/Outdoor` (binary flags)
- **Device Characteristics**:
  - `Number of Antennas`, `Device Type` (Smartphone, IoT Sensor, Drone - one-hot encoded)
- **Computer Vision Features**:
  - Extracted SIFT keypoints that aid in vision-based beamforming intelligence
- **Real-Time Performance Metrics** (used as input features, not targets):
  - `Beamforming Gain (dB)`
  - `Latency (ms)`
  - `Energy Consumption (kWh/Gb)`
  - `Throughput (Mbps)`
  - `Beam Training Time (s)`
  - `SNR Improvement (dB)`
  - `Processing Time (ms)`
  - `Memory Usage (MB)`

These metrics give the model a nuanced understanding of system performance during training, enhancing its ability to learn optimization outcomes based on holistic inputs.

### 🎯 Classification Target

- **Optimized**: A binary variable indicating whether the beamforming process succeeded (`1`) or failed (`0`). This is the primary target variable for the classification models in this project.

### 🔍 Use Cases

- Classification of beamforming optimization outcomes based on real-time metrics
- Training interpretable ML models for decision-making in 6G wireless systems
- Investigating the impact of environmental and CV-based features on network intelligence
- Applying Explainable AI techniques (SHAP, LIME) to uncover feature contributions

> This dataset serves as a powerful tool for telecom researchers, ML engineers, and AI practitioners exploring optimization techniques for intelligent 6G-IoT networks.


## 🔄 Workflow Pipeline

1. **Data Loading & Exploration**  
   Load the Kaggle dataset and explore input features, target distribution, and missing values.

2. **Preprocessing**  
   - Numerical features are scaled using **MinMaxScaler**  
   - Categorical features are **one-hot encoded**  
   - Outputs are structured for classification (`Optimized` column)

3. **Model Training & Tuning**  
   - Trains 7 different classifiers including Random Forest, XGBoost, AdaBoost, and more  
   - Performs **GridSearchCV** with cross-validation  
   - Selects best models based on test accuracy and F1-score

4. **Evaluation**  
   - Reports accuracy, F1-score, classification report  
   - Includes **Confusion Matrix** and **ROC-AUC Curve** for all models

5. **Explainable AI**  
   - Applies **SHAP** (TreeExplainer) for global and local interpretability  
   - Uses **LIME** for instance-level explanation  
   - Displays feature importance bar plots from tree-based models

## 📊 Results & Visualizations

This project evaluated multiple machine learning classifiers to predict whether beamforming in 6G-IoT networks can be optimized based on technical, environmental, and device-level features.

### ✅ Model Performance Summary

Seven different classification models were trained using GridSearchCV and 10-fold cross-validation. Key metrics evaluated include **Accuracy**, **F1-Score**, **Classification Report**, **Confusion Matrix**, and **ROC-AUC Curve**.

| Model             | Test Accuracy | F1 Score | CV Accuracy (Mean ± Std) |
|------------------|---------------|----------|---------------------------|
| Random Forest     | ✔️ High       | ✔️ Strong| Robust CV performance     |
| XGBoost           | ✔️ High       | ✔️ Strong| Best on ROC-AUC           |
| AdaBoost          | Good          | Balanced | Lightweight and fast      |
| Extra Trees       | ✔️ High       | ✔️ Strong| Stable across folds       |
| Gradient Boosting | Moderate      | Moderate | Good for structured data  |
| KNN               | Varies        | Lower    | Sensitive to scaling      |
| Decision Tree     | Varies        | Varies   | Easily interpretable       |

> Random Forest and XGBoost emerged as top performers, with high test accuracy and interpretability.

### 📈 Visualizations

- **Confusion Matrix:** Highlights correct vs. incorrect predictions for each class. This image represents the Confusion Matrix for the best performing model. The confusion matrices for all other models are present in the code file. 
  

  ![image](https://github.com/user-attachments/assets/d4c860d4-09cf-4402-80b0-f17ee32b4463)


  
- **ROC-AUC Curve:** Shows model's ability to distinguish between optimized (1) and non-optimized (0) samples. This image represents the ROC-AUC Curve for the best performing model. The confusion matrices for all other models are present in the code       file.


  ![image](https://github.com/user-attachments/assets/6881ed36-70ce-44e6-a442-7622af86b9f2)


  
- **Feature Importance Plot:** Identifies which features (e.g., Frequency, Mobility, Obstacle Density) influence the decision most.

### 🧠 Explainability with XAI

- **SHAP (SHapley Additive Explanations):**  
  - 🔹 *Summary Plot:* Shows global feature importance.

          ![image](https://github.com/user-attachments/assets/c9a0025b-e759-4e3f-a692-8d36e5f89547)

    
  - 🔹 *Force Plot:* Visualizes feature contributions for individual predictions.
 
          ![image](https://github.com/user-attachments/assets/98aafd4c-7bff-401b-bca7-29efcb4edbf5)

    
- **LIME (Local Interpretable Model-Agnostic Explanations):**  
  - 🔸 *Sample-level Interpretation:* Explains why a particular test sample was classified as optimized or not.
 
          ![image](https://github.com/user-attachments/assets/779d2844-93bf-42b3-91f0-f667761aef3f)

    
  - 🔸 *Tree Based Feature Importance:* Tree-based feature importance measures how much each feature contributes to reducing impurity (e.g., Gini or entropy) across all decision trees in the model.

          ![image](https://github.com/user-attachments/assets/177cecd7-689e-4c12-acec-d0d86ec2a86f)


These explainability tools provide deep insight into **why** the model made certain predictions—critical for trust in AI systems used in wireless network optimization.

---

---
