import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Symptom data and specialty mapping
symptoms = [
    "I have chest pain and shortness of breath.",
    "My skin is itchy and I have a rash.",
    "I'm feeling tired and have high blood sugar.",
    "My knee hurts when I walk.",
    "I have a sore throat and trouble hearing.",
    "I'm pregnant and need a checkup for women.",
    "My baby is crying and has a fever."
]

symptom_to_specialty = {
    symptoms[0]: "Cardiology",
    symptoms[1]: "Dermatology",
    symptoms[2]: "Internal Medicine",
    symptoms[3]: "Orthopedics",
    symptoms[4]: "ENT (Ear, Nose, Throat)",
    symptoms[5]: "Gynecology",
    symptoms[6]: "Pediatrics"
}

# Prepare FAISS index
symptom_embeddings = embedder.encode(symptoms).astype("float32")
dim = symptom_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(symptom_embeddings)

# System prompt (replace with your full prompt)

system_prompt = """
You are MedBot, an automated assistant for a medical reservation system. Start by greeting the patient in a friendly, caring tone. Then guide them step-by-step through booking an appointment.

First, ask the patient about their symptoms or what they are suffering from. Based on their answer, suggest the most suitable medical specialty.

Next, show the patient a list of available doctors who specialize in that field. Ask the patient to choose a doctor from the list.

After the patient selects a doctor, inform them of the available days and times for that doctor. Let the patient pick the day and time that suits them best.

If the available days or times are not convenient for the patient, offer to suggest another doctor within the same specialty.

Once the patient confirms their choice, summarize the booking details clearly, including:
- Doctor’s Name
- Specialty
- Appointment Day and Time
- Consultation Fee

Always confirm the patient's choices after each step to avoid confusion.

Finally, after confirming the booking, Ask the patient about his opinion of the service. Then evaluate his opinion. Is it: Positive, Negative, or Neutral.

Be friendly, supportive, and clear throughout the conversation.

---

Available Specialties and Doctors:

Cardiology:
  - Dr. Ahmed Hassan — Available: Sunday 5 PM - 8 PM, Wednesday 3 PM - 6 PM, Fee: $50
  - Dr. Sara Ali — Available: Monday 2 PM - 5 PM, Thursday 4 PM - 7 PM, Fee: $55

Dermatology:
  - Dr. Omar Tarek — Available: Tuesday 1 PM - 4 PM, Friday 3 PM - 6 PM, Fee: $40
  - Dr. Mona Youssef — Available: Saturday 10 AM - 1 PM, Tuesday 2 PM - 5 PM, Fee: $45

Internal Medicine:
  - Dr. Khaled Nabil — Available: Sunday 9 AM - 12 PM, Wednesday 5 PM - 8 PM, Fee: $35
  - Dr. Layla Samir — Available: Monday 1 PM - 4 PM, Thursday 10 AM - 1 PM, Fee: $38

Orthopedics:
  - Dr. Mostafa Adel — Available: Tuesday 3 PM - 6 PM, Friday 11 AM - 2 PM, Fee: $60
  - Dr. Heba Fathy — Available: Sunday 4 PM - 7 PM, Thursday 5 PM - 8 PM, Fee: $62

ENT (Ear, Nose, and Throat):
  - Dr. Youssef Hani — Available: Monday 5 PM - 8 PM, Wednesday 2 PM - 5 PM, Fee: $45
  - Dr. Salma Reda — Available: Tuesday 10 AM - 1 PM, Saturday 3 PM - 6 PM, Fee: $47

Gynecology:
  - Dr. Reem Saad — Available: Sunday 11 AM - 2 PM, Thursday 6 PM - 9 PM, Fee: $55
  - Dr. Nourhan Adel — Available: Monday 12 PM - 3 PM, Friday 4 PM - 7 PM, Fee: $58

Pediatrics:
  - Dr. Hany Farouk — Available: Saturday 9 AM - 12 PM, Tuesday 3 PM - 6 PM, Fee: $30
  - Dr. Mariam Atef — Available: Sunday 2 PM - 5 PM, Wednesday 10 AM - 1 PM, Fee: $32

---

Notes:
- Always confirm each selected option.
- Always be patient and supportive.
- Summarize clearly before final confirmation.

"""


# Load the LLM model
@st.cache_resource
def load_model():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    return pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.9
    )

pipe = load_model()

def get_specialty(user_input):
    """Identify medical specialty based on symptom similarity"""
    query_embedding = embedder.encode([user_input]).astype("float32")
    distances, indices = index.search(query_embedding, 1)
    best_match_idx = indices[0][0]
    best_symptom = symptoms[best_match_idx]
    return symptom_to_specialty[best_symptom], best_symptom

def get_response(messages):
    """Generate response from the LLM"""
    # Format messages for the model
    formatted_messages = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in messages if m["role"] != "system"]
    )
    full_prompt = f"System: {system_prompt}\n\n{formatted_messages}\nAssistant:"
    
    # Generate response
    outputs = pipe(
        full_prompt,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return outputs[0]['generated_text']

# Streamlit interface
st.set_page_config(page_title="MedBot - Medical Assistant", page_icon="🤖")
st.title(" MedBot - Medical Reservation Assistant")

# Initialize session state
if "context" not in st.session_state:
    st.session_state.context = [{"role": "system", "content": system_prompt}]

# User input
user_input = st.chat_input(" Describe your symptoms or request:")

if user_input:
    # Add user message to context
    st.session_state.context.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get specialty for new symptoms
    if "specialty" not in st.session_state:
        try:
            specialty, matched_symptom = get_specialty(user_input)
            st.session_state.specialty = specialty
            st.session_state.context.append({
                "role": "system", 
                "content": f"Identified specialty: {specialty} (based on: '{matched_symptom}')"
            })
        except Exception as e:
            st.error(f"Error identifying specialty: {str(e)}")
            st.session_state.specialty = "General Medicine"
    
    # Generate and display response
    with st.spinner("Thinking..."):
        try:
            response = get_response(st.session_state.context)
            
            # Add assistant response to context
            st.session_state.context.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

# Display conversation history
st.divider()
st.subheader("Conversation History")
for msg in st.session_state.context:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message("assistant").write(msg["content"])
    elif msg["role"] == "system" and "specialty" in msg["content"]:
        st.info(f" {msg['content']}")

# Add reset button
if st.button("Reset Conversation"):
    st.session_state.clear()
    st.session_state.context = [{"role": "system", "content": system_prompt}]
    st.experimental_rerun()