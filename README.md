Multiclass Fish Image Classification
A deep learning project that classifies fish images into multiple species using CNN and transfer learning techniques. The best-performing model is deployed as a Streamlit web application for real-time predictions.

Project Structure
.
├── Multiclass_Fish_Image_Classification.ipynb   # Model training & evaluation
├── app.py                                       # Streamlit app for deployment
├── MobileNet_final.h5                           # Trained model file
├── class_names.txt                              # List of fish species
├── requirements.txt                             # Dependencies
└── README.md                                    # Project documentation


Features
Train a CNN from scratch and use pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).
Data preprocessing & augmentation for better generalization.
Model evaluation with accuracy, precision, recall, F1-score, confusion matrix.
Streamlit app for uploading images & getting predictions with confidence scores.
Fully documented and ready-to-deploy code.

🧠 Skills Used
Deep Learning
Python
TensorFlow/Keras
Data Preprocessing
Transfer Learning
Model Evaluation
Streamlit Deployment
Visualization

📊 Dataset
Images of fish categorized by species in separate folders.
Loaded and processed using ImageDataGenerator.
Dataset: Provided as a ZIP file in the project repository.

🚀 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/yourusername/multiclass-fish-classification.git
cd multiclass-fish-classification
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py

💻 Usage
Open the Streamlit app in your browser (default: http://localhost:8501).
Upload an image (.jpg, .jpeg, .png).
Get the predicted fish species and confidence scores instantly.

📈 Model Evaluation
Multiple models tested (CNN from scratch + 5 pre-trained architectures).
MobileNet performed best and was selected for deployment.
Evaluation metrics and training curves included in the notebook.

📦 Deliverables
.ipynb notebook with training & evaluation.
Trained .h5 model.
Streamlit deployment script.
Documentation & GitHub repo.

Deeveloped by Ishan Mishra
https://www.linkedin.com/in/ihrm-ishan/






