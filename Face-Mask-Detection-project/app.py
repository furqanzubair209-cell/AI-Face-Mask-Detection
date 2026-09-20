import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
# Heavy imports moved inside functions to prevent blank screen on startup

# Page configuration
st.set_page_config(
    page_title='AI Face Mask Detector',
    page_icon='😷',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e94560;
    }
    .stApp {
        background: #0f3460;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(90deg, #e94560, #ff2e63);
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(233, 69, 96, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.5);
        color: #ffffff;
    }
    h1 {
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #e94560, #f08a5d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a2e;
    }
    .stImage {
        border-radius: 15px;
        border: 2px solid #e94560;
        overflow: hidden;
    }
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        color: #4CAF50;
        border: 1px solid #4CAF50;
        border-radius: 10px;
    }
    .stInfo {
        background-color: rgba(33, 150, 243, 0.1);
        color: #2196F3;
        border: 1px solid #2196F3;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load face and mask detection models."""
    from tensorflow.keras.models import load_model
    prototxtPath = os.path.join("face_detector", "deploy.prototxt")
    weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    
    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        st.error(f"Face detector files missing! Expected at: {prototxtPath}")
        return None, None
        
    face_net = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    model_path = "mask_detector.h5"
    if not os.path.exists(model_path):
        st.error(f"Mask detector model missing! Expected at: {model_path}")
        return face_net, None
        
    mask_model = load_model(model_path)
    return face_net, mask_model

def process_image(image, face_net, mask_model):
    """Detect faces and predict masks."""
    alert_needed = False
    # Convert PIL to OpenCV BGR
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    (h, w) = img.shape[:2]

    # Construct blob
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from tensorflow.keras.preprocessing.image import img_to_array
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = img[startY:endY, startX:endX]
            if face.size == 0: continue
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = mask_model.predict(face)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            if "No Mask" in label:
                alert_needed = True
            
            label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
            cv2.putText(img, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 3)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alert_needed

def main():
    st.markdown("<h1>😷 AI Face Mask Detector</h1>", unsafe_allow_html=True)
    
    with st.spinner("Initializing AI Models..."):
        face_net, mask_model = load_models()
        
    if face_net is None or mask_model is None:
        st.stop()

    st.sidebar.title("🛠️ Control Panel")
    st.sidebar.markdown("---")
    st.sidebar.info("Upload an image or use your webcam to detect face masks in real-time.")
    
    confidence_threshold = st.sidebar.slider("Face Detection Confidence", 0.0, 1.0, 0.5, 0.05)
    
    mode = st.sidebar.radio("Detection Mode", ["📸 Image Upload", "📹 Live Webcam"])

    if mode == "📸 Image Upload":
        st.write("### 📤 Upload Image for Analysis")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Input Image")
                st.image(image, use_container_width=True)
            
            if st.button("🚀 Analyze Mask"):
                with st.spinner("AI is thinking..."):
                    result_img, alert = process_image_with_thresh(image, face_net, mask_model, confidence_threshold)
                    with col2:
                        st.markdown("#### Detected Output")
                        st.image(result_img, use_container_width=True)
                        if alert:
                            st.error("🚨 ALERT: No Mask Detected!")
                            # Use a component with a unique key to force re-run
                            import time
                            st.components.v1.html(f"""
                                <script>
                                    window.speechSynthesis.cancel();
                                    var msg = new SpeechSynthesisUtterance('No mask detected');
                                    window.speechSynthesis.speak(msg);
                                    console.log('Voice alert triggered');
                                </script>
                            """, height=0)
                        else:
                            st.success("Analysis Complete!")
        else:
            st.info("Awaiting image upload...")

    elif mode == "📹 Live Webcam":
        st.write("### 🎥 Live Detection Feed")
        img_file_buffer = st.camera_input("Capture a moment")
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            with st.spinner("Analyzing snapshot..."):
                result_img, alert = process_image_with_thresh(image, face_net, mask_model, confidence_threshold)
                st.image(result_img, caption="AI Analysis Result", use_container_width=True)
                
                if alert:
                    st.error("🚨 ALERT: No Mask Detected!")
                    st.components.v1.html(f"""
                        <script>
                            window.speechSynthesis.cancel();
                            var msg = new SpeechSynthesisUtterance('No mask detected');
                            window.speechSynthesis.speak(msg);
                            console.log('Voice alert triggered');
                        </script>
                    """, height=0)
                else:
                    st.success("Face Mask Status Detected!")

def process_image_with_thresh(image, face_net, mask_model, threshold):
    """Detect faces and predict masks with a custom threshold."""
    alert_needed = False
    # Convert PIL to OpenCV BGR
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    (h, w) = img.shape[:2]

    # Construct blob for face detection
    # Standard mean for Res10-SSD is (104, 117, 123). The previous '177' may have been a typo.
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 117.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # If no faces detected with standard scale, try a smaller scale (sometimes helps with masks/angles)
    if detections.shape[2] == 0 or np.max(detections[0, 0, :, 2]) < threshold:
        blob_alt = cv2.dnn.blobFromImage(img, 1.0, (400, 400), (104.0, 117.0, 123.0))
        face_net.setInput(blob_alt)
        detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from tensorflow.keras.preprocessing.image import img_to_array
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = img[startY:endY, startX:endX]
            if face.size == 0: continue
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = mask_model.predict(face, verbose=0)[0]
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
            prob = max(mask, withoutMask) * 100
            label_text = f"{label}: {prob:.1f}%"
            
            if "No Mask" in label:
                alert_needed = True
            
            # Draw on image
            cv2.putText(img, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 3)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alert_needed

if __name__ == "__main__":
    main()
