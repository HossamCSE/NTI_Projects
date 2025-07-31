import streamlit as st
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    .caption-result {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .model-info {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🖼️ Image Caption Generator</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📋 Model Information")
    st.markdown("""
    <div class="model-info">
    <h4>Model Details:</h4>
    <ul>
        <li><strong>Architecture:</strong> VGG16 + LSTM</li>
        <li><strong>Dataset:</strong> Flickr8k</li>
        <li><strong>Embedding Dim:</strong> 256</li>
        <li><strong>LSTM Units:</strong> 512</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Settings")
    generation_method = st.selectbox(
        "Caption Generation Method:",
        ["Greedy Search", "Beam Search"],
        help="Choose the method for generating captions"
    )
    
    if generation_method == "Beam Search":
        beam_width = st.slider("Beam Width:", 2, 5, 3)
    
    max_caption_length = st.slider("Max Caption Length:", 10, 50, 20)

@st.cache_resource
def load_models_and_tokenizer():
    """Load the trained model, tokenizer, and VGG16 feature extractor"""
    try:
        # Load the trained caption model
        model_path = 'best_caption_model.h5'  # or 'final_caption_model.h5'
        if not os.path.exists(model_path):
            model_path = 'final_caption_model.h5'
            if not os.path.exists(model_path):
                st.error(f"Model file not found. Please ensure you have either 'best_caption_model.h5' or 'final_caption_model.h5'")
                return None, None, None
        
        caption_model = load_model(model_path)
        
        # Load tokenizer
        tokenizer_path = 'tokenizer.pkl'
        if not os.path.exists(tokenizer_path):
            st.error(f"Tokenizer file not found: {tokenizer_path}")
            return None, None, None
        
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Note: VGG16 will be loaded when needed to save memory
        return caption_model, tokenizer, None
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def extract_image_features(image, vgg_model):
    """Extract features from uploaded image"""
    try:
        # Resize image to 224x224
        image = image.resize((224, 224))
        
        # Convert to array
        image_array = img_to_array(image)
        image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        
        # Preprocess for VGG16
        image_array = preprocess_input(image_array)
        
        # Extract features
        features = vgg_model.predict(image_array, verbose=0)
        
        return features
    
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def generate_caption_greedy(model, tokenizer, image_feature, max_length):
    """Generate caption using greedy search"""
    in_text = 'startseq'
    
    for i in range(max_length):
        # Encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length-1, padding='post')
        
        # Predict next word
        y_pred = model.predict([image_feature.reshape(1, -1), sequence], verbose=0)
        
        # Get word with highest probability
        seq_len = len(tokenizer.texts_to_sequences([in_text])[0]) - 1
        if seq_len >= max_length - 1:
            break
        
        y_pred = np.argmax(y_pred[0, seq_len, :])
        
        # Convert index to word
        word = None
        for word_text, index in tokenizer.word_index.items():
            if index == y_pred:
                word = word_text
                break
        
        # Stop if no word found or end sequence
        if word is None or word == 'endseq':
            break
        
        # Add word to sequence
        in_text += ' ' + word
    
    # Remove start sequence
    caption = in_text.replace('startseq ', '')
    return caption

def generate_caption_beam_search(model, tokenizer, image_feature, max_length, beam_width=3):
    """Generate caption using beam search"""
    # Initialize beam
    sequences = [(['startseq'], 0.0)]
    
    for _ in range(max_length):
        all_candidates = []
        
        for seq, score in sequences:
            if seq[-1] == 'endseq':
                all_candidates.append((seq, score))
                continue
            
            # Encode sequence
            text = ' '.join(seq)
            encoded = tokenizer.texts_to_sequences([text])[0]
            padded = pad_sequences([encoded], maxlen=max_length-1, padding='post')
            
            # Get predictions
            preds = model.predict([image_feature.reshape(1, -1), padded], verbose=0)
            
            # Get top predictions
            position = min(len(encoded)-1, max_length-2)
            if position < 0:
                position = 0
            
            top_indices = np.argsort(preds[0, position, :])[-beam_width:]
            
            for idx in top_indices:
                # Find word for index
                word = None
                for w, i in tokenizer.word_index.items():
                    if i == idx:
                        word = w
                        break
                
                if word:
                    new_seq = seq + [word]
                    new_score = score + np.log(preds[0, position, idx] + 1e-8)
                    all_candidates.append((new_seq, new_score))
        
        # Keep top sequences
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Check if all ended
        if all(seq[-1] == 'endseq' for seq, _ in sequences):
            break
    
    # Return best sequence
    best_seq = sequences[0][0]
    caption = ' '.join(best_seq[1:]).replace(' endseq', '')
    return caption

def generate_caption(model, tokenizer, image_feature, max_length, method='greedy', beam_width=3):
    """Generate caption based on selected method"""
    if method.lower() == 'beam search':
        return generate_caption_beam_search(model, tokenizer, image_feature, max_length, beam_width)
    else:
        return generate_caption_greedy(model, tokenizer, image_feature, max_length)

def predict_caption_for_image(image_path, model, tokenizer, max_length, method='greedy', beam_width=3):
    """Complete pipeline for new image - original function from your code"""
    # Load VGG16 model
    vgg_model = VGG16(weights='imagenet')
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    
    # Extract features
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    
    # Generate caption
    caption = generate_caption(model, tokenizer, feature, max_length, method, beam_width)
    return caption

def predict_caption_for_uploaded_image(uploaded_image, model, tokenizer, max_length, method='greedy', beam_width=3):
    """Complete pipeline for uploaded image (Streamlit version)"""
    # Load VGG16 model
    vgg_model = VGG16(weights='imagenet')
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    
    # Process uploaded image
    image = uploaded_image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    # Extract features
    feature = vgg_model.predict(image, verbose=0)
    
    # Generate caption
    caption = generate_caption(model, tokenizer, feature, max_length, method, beam_width)
    return caption

# Main application
def main():
    # Load models
    caption_model, tokenizer, vgg_model = load_models_and_tokenizer()
    
    if caption_model is None or tokenizer is None:
        st.error("Failed to load required models. Please ensure model files are available.")
        st.info("Required files: 'best_caption_model.h5' (or 'final_caption_model.h5') and 'tokenizer.pkl'")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload an Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a JPG, JPEG, or PNG image"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display the image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        
        with col2:
            st.subheader("🔮 Generated Caption")
            
            # Generate caption button
            if st.button("🚀 Generate Caption", type="primary"):
                with st.spinner("Extracting features and generating caption..."):
                    try:
                        # Method 1: Using the complete pipeline (like your original code)
                        if generation_method == "Beam Search":
                            caption = predict_caption_for_uploaded_image(
                                image, caption_model, tokenizer, 
                                max_caption_length, method='beam search', 
                                beam_width=beam_width
                            )
                        else:
                            caption = predict_caption_for_uploaded_image(
                                image, caption_model, tokenizer, 
                                max_caption_length, method='greedy'
                            )
                        
                        # Display result
                        st.markdown(f"""
                        <div class="caption-result">
                        <h4>Generated Caption:</h4>
                        <p style="font-size: 1.2rem; font-weight: bold; color: #1f77b4;">
                        "{caption.title()}"
                        </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional info
                        st.info(f"Method used: {generation_method}")
                        if generation_method == "Beam Search":
                            st.info(f"Beam width: {beam_width}")
                            
                    except Exception as e:
                        st.error(f"Error generating caption: {str(e)}")
                        st.error("Please check that your model files are compatible.")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
        <h3>👆 Please upload an image to get started</h3>
        <p>The model will analyze your image and generate a descriptive caption.</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>Built with Streamlit | Image Caption Generation using VGG16 + LSTM</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()