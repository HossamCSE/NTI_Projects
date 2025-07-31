Image Captioning System 

Project Overview
This project implements an automatic image captioning system, which generates meaningful natural language descriptions for images.
It combines Computer Vision for image feature extraction and Natural Language Processing for sequence modeling to produce captions word by word.

Key Features:

Extracts visual features using VGG16 (pre-trained on ImageNet).

Generates captions using LSTM-based sequence model.

Supports Greedy Search and Beam Search decoding methods.

Provides an interactive Streamlit web app for live demo.

Dataset
Dataset Used: Flickr8k Dataset

Content: ~8,000 images, 5 captions per image

Split: 90% training / 10% testing

Ensure that you download and place the dataset in the following structure:
project/
│
├─ Images/             # All Flickr8k images
├─ captions.txt        # Captions file

Project Workflow
Image Feature Extraction
Uses pre-trained VGG16 to extract 4096-dimensional feature vectors for each image.

Caption Preprocessing
Lowercasing, punctuation removal, and tokenization.
Added startseq and endseq tokens to mark sentence boundaries.

Sequence Model Training
LSTM-based model trained to generate captions word by word.

Caption Generation
Supports Greedy Search and Beam Search decoding.

Deployment
Interactive Streamlit app to generate captions for uploaded images.

Model Architecture
Image Branch:
Dense layers applied on VGG16 extracted features with dropout for regularization.

Text Branch:
Embedding layer followed by stacked LSTM layers for sequence generation.

Fusion & Output:
Concatenates image and text features → Dense layers → Softmax output for word prediction.

Training Details:
Embedding Dimension: 256
LSTM Units: 512
Batch Size: 32
Learning Rate: 0.0005
Epochs: 25

How to Run the Project
1. Clone the repository
git clone https://github.com/YourUsername/YourRepoName.git
cd YourRepoName

2. Install Dependencies
pip install -r requirements.txt

3. Prepare Dataset
Download the Flickr8k dataset and place it under Images/ with captions.txt.

4. Train the Model
python train_model.py

Extracts features, cleans captions, trains the LSTM model.
Saves final_caption_model.h5 and tokenizer.pkl.

5. Run the Streamlit App
streamlit run app.py

Upload an image and the model will generate captions in real-time.

Project Structure:
├─ app.py                  # Streamlit web app
├─ train_model.py          # Full training and model building script
├─ features.pkl            # Extracted image features (generated after first run)
├─ tokenizer.pkl           # Saved tokenizer for caption sequences
├─ best_caption_model.h5   # Best model checkpoint
├─ final_caption_model.h5  # Final trained model
├─ requirements.txt        # Project dependencies
├─ README.md               # Project documentation
└─ Images/                 # Flickr8k images

License
This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.
