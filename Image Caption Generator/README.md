Image Captioning System
1. Project Overview
This project implements an automatic image captioning system capable of generating meaningful, natural language descriptions for given images.
It integrates Computer Vision for visual feature extraction and Natural Language Processing for sequential language modeling, producing captions incrementally on a word-by-word basis.

2. Key Features
Visual Feature Extraction: Utilizes the VGG16 architecture pre-trained on the ImageNet dataset.

Caption Generation: Employs an LSTM-based sequence model for text generation.

Decoding Strategies: Supports both Greedy Search and Beam Search methods.

Interactive Web Interface: Includes a Streamlit-based application for real-time demonstrations.

3. Dataset
Dataset Used: Flickr8k Dataset

Content: Approximately 8,000 images, each paired with 5 descriptive captions.

Data Split: 90% for training and 10% for testing.

Required Directory Structure:
project/
│
├─ Images/             # All Flickr8k images
├─ captions.txt        # Caption annotations file

4. Project Workflow
4.1 Image Feature Extraction
Utilizes the pre-trained VGG16 network to extract 4,096-dimensional feature vectors for each image.

4.2 Caption Preprocessing
Converts all text to lowercase.

Removes punctuation.

Tokenizes captions.

Adds special tokens startseq and endseq to indicate sentence boundaries.

4.3 Sequence Model Training
Trains an LSTM-based sequence model to predict the next word in the caption, given the image features and the preceding words.

4.4 Caption Generation
Implements Greedy Search for straightforward decoding.

Implements Beam Search for improved caption diversity and accuracy.

4.5 Deployment
Provides a Streamlit web application for real-time caption generation from user-uploaded images.

5. Model Architecture
Image Processing Branch
Dense layers applied to VGG16-derived features.

Dropout layers used for regularization.

Text Processing Branch
Embedding layer to map words to dense vector representations.
Stacked LSTM layers for sequence generation.
Fusion and Output
Concatenates image and text feature representations.
Applies Dense layers followed by a Softmax output layer for word prediction.

6. Training Configuration
Embedding Dimension: 256
LSTM Units: 512
Batch Size: 32
Learning Rate: 0.0005
Epochs: 25

7. Running the Project
Clone the Repository:
git clone https://github.com/HossamCSE/NTI_Projects/tree/main/Image%20Caption%20Generator.git
cd NTI_Projects/Image Caption Generator

Install Dependencies:
pip install -r requirements.txt

Prepare the Dataset:
Download the Flickr8k dataset.
Place all images in the Images/ directory and captions in captions.txt.

python train_model.py

Extracts image features.
Preprocesses captions.
Trains the LSTM model.
Saves final_caption_model.h5 and tokenizer.pkl.

Run the Web Application:
streamlit run app.py

Upload an image.
View generated captions in real time.

8. Project Structure:
├─ app.py                  # Streamlit web app
├─ train_model.py          # Model training and building script
├─ features.pkl            # Extracted image features
├─ tokenizer.pkl           # Tokenizer for caption sequences
├─ best_caption_model.h5   # Best performing model checkpoint
├─ final_caption_model.h5  # Final trained model
├─ requirements.txt        # Project dependencies
├─ README.md               # Documentation
└─ Images/                 # Flickr8k images

9. License
This project is distributed under the MIT License.
You are free to use, modify, and share this work, provided that appropriate attribution is given.
