# Image Captioning System

## 1. Project Overview
This project implements an **automatic image captioning system** that generates meaningful natural language descriptions for images.  
It combines **Computer Vision** for extracting visual features and **Natural Language Processing** for sequential modeling, producing captions incrementally on a word-by-word basis.

## 2. Key Features
- **Visual Feature Extraction:** Utilizes the VGG16 architecture pre-trained on ImageNet.  
- **Caption Generation:** Employs an LSTM-based sequence model for text generation.  
- **Decoding Strategies:** Supports Greedy Search and Beam Search methods.  
- **Interactive Web Interface:** Provides a Streamlit-based application for real-time demonstrations.

## 3. Dataset
- **Dataset Used:** Flickr8k Dataset  
- **Content:** ~8,000 images, each with 5 descriptive captions.  
- **Data Split:** 90% training / 10% testing.

**Required Directory Structure:**
```
project/
│
├─ Images/             # All Flickr8k images
├─ captions.txt        # Caption annotations file
```

## 4. Project Workflow

### 4.1 Image Feature Extraction
- Uses the pre-trained VGG16 model to extract 4,096-dimensional feature vectors from each image.

### 4.2 Caption Preprocessing
- Converts captions to lowercase.  
- Removes punctuation.  
- Tokenizes captions.  
- Adds special tokens `startseq` and `endseq` to indicate sentence boundaries.

### 4.3 Sequence Model Training
- Trains an LSTM-based model to predict the next word in the caption given image features and preceding words.

### 4.4 Caption Generation
- Greedy Search for straightforward decoding.  
- Beam Search for improved diversity and accuracy.

### 4.5 Deployment
- Interactive **Streamlit** app for uploading images and generating captions in real time.

## 5. Model Architecture

**Image Branch:**  
- Dense layers applied to VGG16 features with dropout regularization.

**Text Branch:**  
- Embedding layer for word representation.  
- Stacked LSTM layers for sequence modeling.

**Fusion & Output:**  
- Concatenates image and text features → Dense layers → Softmax output for word prediction.

## 6. Training Configuration
- **Embedding Dimension:** 256  
- **LSTM Units:** 512  
- **Batch Size:** 32  
- **Learning Rate:** 0.0005  
- **Epochs:** 25

## 7. How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/HossamCSE/NTI_Projects/tree/main/Image%20Caption%20Generator.git
cd "NTI_Projects/Image Caption Generator"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Dataset
- Download the Flickr8k dataset.  
- Place images in the `Images/` directory and captions in `captions.txt`.

### Step 4: Train the Model
```bash
python train_model.py
```
- Extracts features, preprocesses captions, and trains the LSTM model.  
- Saves `final_caption_model.h5` and `tokenizer.pkl`.

### Step 5: Run the Web App
```bash
streamlit run app.py
```
- Upload an image to generate captions in real time.

## 8. Project Structure
```
├─ app.py                  # Streamlit web app
├─ train_model.py          # Training and model building script
├─ features.pkl            # Extracted image features
├─ tokenizer.pkl           # Tokenizer for caption sequences
├─ best_caption_model.h5   # Best performing model
├─ final_caption_model.h5  # Final trained model
├─ requirements.txt        # Dependencies
├─ README.md               # Documentation
└─ Images/                 # Flickr8k dataset images
```

## 9. License
This project is licensed under the **MIT License**.  
You may use, modify, and distribute it with proper attribution.
