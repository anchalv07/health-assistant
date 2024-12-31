import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import cv2
import numpy as np
import streamlit as st
import pyttsx3
import threading
import warnings
import time
import re
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Suppress the specific warning
warnings.filterwarnings("ignore", message="missing ScriptRunContext! This warning can be ignored when running in bare mode.")

# Set page configuration
st.set_page_config(page_title='Health Assistant', layout='wide', page_icon="🩺")

if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

# Function to load models safely
@st.cache_resource
def load_model_safely(model_path):
    try:
        model = load_model(model_path, compile=False)  # Fixed the recursive call
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None6

# Load the CNN models for chest X-rays and malaria cell images
malaria_model = load_model_safely("C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/malaria_cells.keras")
chest_xray_model = load_model_safely("C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/chest x-ray.keras")
Covid_19_model = load_model_safely("C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/Covid 19.keras")

# Load the other disease models using pickle
@st.cache_resource
def load_pickle_models():
    model_paths = {
        'Cancer_model': "C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/cancer dataset.pkl",
        'Diabetes_model': "C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/diabetes dataset.pkl",
        'Heart_model': "C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/heart disease dataset.pkl",
        'Parkinson_model': "C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/parkinson dataset.pkl",
        'Kidney_model': "C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/kidney dataset.pkl",
        'Liver_model': "C:/Users/Anchal Verma/OneDrive/Desktop/disease prediction app/liver patient dataset.pkl",
    }
    model = {}
    for disease, path in model_paths.items():
        try:
            with open(path, 'rb') as file:
                model[disease] = pickle.load(file)
        except Exception as e:
            st.error(f"Error loading {disease}: {e}")
    return model

models = load_pickle_models()

# Loading disease-related images
@st.cache_resource
def load_images():
    image_paths = {
        'Cancer': "C:/Users/Anchal Verma/OneDrive/Pictures/Screenshots/Screenshot 2024-12-10 132145.png",
        'Diabetes': "C:/Users/Anchal Verma/OneDrive/Pictures/Screenshots/Screenshot 2024-12-10 132827.png",
        'Heart': "C:/Users/Anchal Verma/OneDrive/Pictures/Screenshots/Screenshot 2024-12-10 133526.png",
        'Parkinson': "C:/Users/Anchal Verma/OneDrive/Pictures/Screenshots/Screenshot 2024-12-10 135107.png",
        'Kidney': "C:/Users/Anchal Verma/OneDrive/Pictures/Screenshots/Screenshot 2024-12-10 135637.png",
        'Liver': "C:/Users/Anchal Verma/OneDrive/Pictures/Screenshots/Screenshot 2024-12-10 140351.png",
    }
    image = {}
    for disease, path in image_paths.items():
        try:
            image[disease] = Image.open(path)
        except Exception as e:
            st.error(f"Error loading image for {disease}: {e}")
    return image

images = load_images()

# Function to preprocess image using OpenCV
def preprocess_with_opencv(captured_image, target_size=(128, 128)):
    try:
        # Read the image with OpenCV
        image = cv2.imread(captured_image)
        # Resize the image
        image_resized = cv2.resize(image, target_size)
        # Normalize the pixel values
        image_normalized = image_resized / 255.0
        # Convert to RGB format
        image_rgb = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(image_rgb, axis=0)
    except Exception as e:
        st.error(f"Error processing image with OpenCV: {e}")
        return None

# Function to preprocess image for models
def preprocess_image(image, target_size=(128, 128)):
    try:
        image = image.convert("RGB")
        image = image.resize(target_size)
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"Error in image preprocessing: {e}")
        return None

# Function to speak text
def speak_text(text, delay=0):
    engine = pyttsx3.init()
    if delay > 0:
        time.sleep(delay)  # Delay before speaking
    engine.say(text)
    engine.runAndWait()

# Function to run the speak_text function in a separate thread
def speak_in_thread(text, delay=0):
    thread = threading.Thread(target=speak_text, args=(text, delay))
    thread.daemon = True
    thread.start()

# Welcome page
def welcome_page():
    st.title("Welcome to the Disease Prediction System")
    st.write("Hey User 😊!")
    st.image("C:/Users/Anchal Verma/OneDrive/Pictures/Screenshots/Screenshot 2024-12-10 141419.png", width=1000)
    st.write("""
    This application is designed to assist in predicting potential diseases based on user inputs for various health parameters. 
    The Health Assistant supports predictions for multiple diseases, including:

    - Cancer
    - Diabetes
    - Heart Disease
    - Parkinson's Disease
    - Kidney Disease
    - Liver Disease

    By entering specific health metrics and indicators, you can receive a prediction of whether you may be at risk for any of these conditions.
    You can also upload Chest X-rays for pneumonia detection, Malaria cell images for infection classification and detection for Covid 19.
    This tool is intended as an informational resource and should not replace professional medical advice.
    """)

    # Initialize session state for speech_done if not already set
    if "speech_done" not in st.session_state:
        st.session_state.speech_done = False

    # Trigger speech only once
    if not st.session_state.speech_done:
        speak_in_thread("Welcome to the Disease Prediction System I am your Health Assistant and will further help you with some diseases predictions. Click continue to move further")
        st.session_state.speech_done = True

    if st.button("Continue"):
        st.session_state.page = "form"

# Information form page
def info_form_page():
    st.header("Please Enter Your Information 😊")

    # Name, Age, Gender inputs
    st.session_state.name = st.text_input("Name")
    st.session_state.age = st.number_input("Age", min_value=0)
    st.session_state.gender = st.selectbox("Gender", ("Male", "Female", "Other"))

    # New email input
    email = st.text_input("Email Address")

    if 'spoken' not in st.session_state:
        st.session_state.spoken = False

    if not st.session_state.spoken:
        speak_text("Please Enter Your Information")
        st.session_state.spoken = True

    # Ensure proper indentation of the block below
    if st.button("Submit"):
        if email and st.session_state.name:  # Ensure email and name are provided
            # Validate email before submitting
            if is_valid_email(email):  # Ensure email and name are provided
             st.session_state.page = "image_prediction"
             # Send confirmation email to the user
             send_confirmation_email(email)
             speak_in_thread("Submission successful. We have sent a confirmation mail. Please check your email.")
            else:
             st.error("Invalid email address! Please enter a correct email.")
             speak_in_thread("Invalid email address! Please enter a correct email.")
        else:
            st.error("Please fill in both your name and email address!")

# Validate email address
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def send_confirmation_email(user_email):
    sender_email = "av013310@gmail.com"  # Your email address
    sender_password = "rity kpqa ttws ovjf"  # Your email password (or app password if using Gmail)

    subject = "Health Assistant - Form Submission Confirmation"
    body = f"Hello {st.session_state.name},\n\nThank you for submitting your information. We will process it and provide disease predictions soon!\n\nBest Regards,\nHealth Assistant Team"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = user_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    # Validate email before sending
    if not is_valid_email(user_email):
        st.error("Invalid email address! Please enter a correct email.")
        return

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, msg.as_string())
            st.success("Confirmation email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {e}")
        speak_in_thread("There was an error sending the email. Please try again.")

# Function to capture image from webcam
def capture_image_from_webcam():
    st.text("Press 'c' to capture the image and 'q' to quit webcam preview.")
    try:
        cap = cv2.VideoCapture(0)  # Open the webcam (use 1 if multiple webcams)
        if not cap.isOpened():
            st.error("Could not open webcam. Make sure the webcam is connected and accessible.")
            return None

        captured_image = None
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame from webcam.")
                break

            cv2.imshow("Webcam - Press 'enter' to Capture", frame)

            # Wait for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Capture image on pressing 'c'
                captured_image = frame
                break
            elif key == ord('q'):  # Quit on pressing 'q'
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_image is not None:
            # Save captured image temporarily
            captured_path = "captured_image.jpg"
            cv2.imwrite(captured_path, captured_image)
            return captured_path
        return None
    except Exception as e:
        st.error(f"Error capturing image from webcam: {e}")
        return None


# Health Assistant Page
def health_assistant_page():
    st.sidebar.title('Health Assistant 🩺')
    if "subheader_selected" not in st.session_state:
        st.session_state.subheader_selected = False

    if not st.session_state.subheader_selected:
        speak_in_thread("Select an option from the sidebar whether you want to use image prediction or Disease prediction.", delay = 2)

    selected_option = st.sidebar.selectbox('Select an option', ['Image Prediction using Deep Learning', 'Disease Prediction using Machine Learning'])

    # Image Prediction
    if selected_option == 'Image Prediction using Deep Learning':
        if not st.session_state.subheader_selected:
            speak_in_thread("Select further Image predictions options from the sidebar. You can either browse your file or can use webcam.", delay=4)

        st.sidebar.subheader('Select Image Prediction Type')
        image_type = st.sidebar.selectbox(
            'Choose prediction type',
            ['Chest X-ray (Pneumonia Detection)', 'Malaria Cell Image Classification', 'Covid 19 Detection']
        )

        if image_type:
            st.session_state.subheader_selected = True

        # Chest X-ray (Pneumonia Detection)
        if image_type == 'Chest X-ray (Pneumonia Detection)':
            st.subheader("Chest X-ray Pneumonia Detection")

            # Speak only for this option
            if not st.session_state.get("spoken_intro_xray", False):
                speak_in_thread("Chest X-ray Pneumonia Detection", delay=8)
                st.session_state.spoken_intro_xray = True

            xray_image = st.file_uploader("Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])
            st.subheader('OR')
            captured_image = st.camera_input("Capture an image using your camera")

            if xray_image or captured_image:
                st.session_state.spoken_xray_result = False

            if xray_image or captured_image:
                try:
                    if xray_image:
                        image = Image.open(xray_image)
                    elif captured_image:
                        image = Image.open(captured_image)

                    image_array = preprocess_image(image, target_size=(128, 128))

                    if image_array is None:
                        st.error("Failed to preprocess the image.")
                        return

                    st.image(image, caption='Uploaded or Captured Chest X-ray image.', width=300)
                    prediction = chest_xray_model.predict(image_array)
                    result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal X-ray"
                    st.success(result)

                    if not st.session_state.spoken_xray_result:
                        speak_in_thread(result)
                        st.session_state.spoken_xray_result = True

                except Exception as e:
                    st.error(f"Error predicting Chest X-ray: {e}")

        # Malaria Cell Classification
        elif image_type == 'Malaria Cell Image Classification':
            st.subheader("Malaria Cell Classification")

            # Speak only for this option
            if not st.session_state.get("spoken_intro_malaria", False):
                speak_in_thread("Malaria Cell Classification", delay=1)
                st.session_state.spoken_intro_malaria = True

            malaria_image = st.file_uploader("Upload a Malaria cell image", type=["jpg", "jpeg", "png"])
            st.subheader('OR')
            captured_image = st.camera_input("Capture an image using your camera")

            if malaria_image or captured_image:
                st.session_state.spoken_malaria_result = False

            if malaria_image or captured_image:
                try:
                    if malaria_image:
                        image = Image.open(malaria_image)
                    elif captured_image:
                        image = Image.open(captured_image)

                    image_array = preprocess_image(image, target_size=(128, 128))

                    if image_array is None:
                        st.error("Failed to preprocess the image.")
                        return

                    st.image(image, caption='Uploaded or Captured Malaria cell image.', width=300)
                    prediction = malaria_model.predict(image_array)
                    result = "Malaria Detected" if prediction[0][0] > 0.5 else "Uninfected"
                    st.success(result)

                    if not st.session_state.spoken_malaria_result:
                        speak_in_thread(result)
                        st.session_state.spoken_malaria_result = True

                except Exception as e:
                    st.error(f"Error predicting Malaria cell: {e}")

        # Covid 19 Detection
        elif image_type == 'Covid 19 Detection':
            st.subheader("Covid 19 Detection")

            # Speak only for this option
            if not st.session_state.get("spoken_intro_covid", False):
                speak_in_thread("Covid 19 Detection", delay=1)
                st.session_state.spoken_intro_covid = True

            covid_image = st.file_uploader("Upload a Covid image", type=["jpg", "jpeg", "png"])
            st.subheader('OR')
            captured_image = st.camera_input("Capture an image using your camera")

            if covid_image or captured_image:
                st.session_state.spoken_covid_result = False

            if covid_image or captured_image:
                try:
                    if covid_image:
                        image = Image.open(covid_image)
                    elif captured_image:
                        image = Image.open(captured_image)
                    image_array = preprocess_image(image, target_size=(128, 128))

                    if image_array is None:
                        st.error("Failed to preprocess the image.")
                        return

                    st.image(image, caption='Uploaded or Captured Covid 19 image.', width=300)

                    prediction = Covid_19_model.predict(image_array)
                    result = "Covid" if prediction[0][0] > 0.5 else "Non Covid"
                    st.success(result)

                    if not st.session_state.spoken_covid_result:
                        speak_in_thread(result)
                        st.session_state.spoken_covid_result = True

                except Exception as e:
                    st.error(f"Error predicting Covid 19: {e}")


    elif selected_option == 'Disease Prediction using Machine Learning':
        if not st.session_state.get('subheader_selected', False):
            speak_in_thread("You can either choose more disease prediction options from the sidebar.", delay=1)
        st.sidebar.subheader('Select Disease Type')
        disease_type = st.sidebar.selectbox('Choose Disease Prediction Type',
                                           ['Cancer Prediction', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Prediction', 'Kidney Disease Prediction', 'Liver Prediction'])

        if disease_type:
            st.session_state.subheader_selected = True

        # Cancer Prediction Page
        if disease_type == 'Cancer Prediction':
            st.subheader('Cancer Prediction')

            if not st.session_state.get("spoken_cancer_intro", False):
                speak_in_thread("Cancer Prediction. Please enter the required details.")
                st.session_state.spoken_cancer_intro = True

            st.image(images['Cancer'], width=1000)
            st.text("Texture → Cell Texture\n"
                    "Fractal Dimension → Cell Complexity\n"
                    "Smoothness → Cell Smoothness\n"
                    "Area → Tumor Area\n"
                    "Compactness → Tumor Shape Compactness\n"
                    "Symmetry → Tumor Symmetry\n"
                    "Perimeter → Tumor Boundary\n"
                    "Concavity → Tumor Indentation Depth\n"
                    "Concave Points → Indentation Points on Tumor")

            if "spoken_cancer_result" not in st.session_state:
                st.session_state.spoken_cancer_result = False

            texture_mean = st.number_input("texture_mean", placeholder="Enter the value between 9 to 40")
            fractal_dimension_mean = st.number_input("fractal_dimension_mean",  placeholder="Enter the value between 0.048 to 0.098")
            smoothness_mean = st.number_input("smoothness_mean",  placeholder="Enter the value between 0.051 to 0.17")
            area_se = st.number_input("area_se",  placeholder="Enter the value between 5 to 543")
            compactness_se = st.number_input("compactness_se", placeholder="Enter the value between 0.0022 to 0.14")
            symmetry_se = st.number_input("symmetry_se", placeholder="Enter the value between 0.007882 to 0.079")
            perimeter_worst = st.number_input("perimeter_worst", placeholder="Enter the value between 50 to 251")
            concavity_worst = st.number_input("concavity_worst", placeholder="Enter the value between 0 to 2")
            concave_points_worst = st.number_input('concave_points_worst', placeholder="Enter the value between 0.007882 to 0.079")
            diagnosis = st.selectbox("diagnosis", options=['B', 'M'])
            if st.button('Cancer Test Result'):
                diagnosis_encoded = 0 if diagnosis == 'B' else 1
                features = np.array([[texture_mean, fractal_dimension_mean, smoothness_mean, area_se, compactness_se,
                                      symmetry_se, perimeter_worst, concavity_worst,
                                      concave_points_worst + diagnosis_encoded]])

                try:
                    cancer_prediction = models['Cancer_model'].predict(features)
                    result = 'The person is suffering from Cancer.' if cancer_prediction[
                                                                           0] == 1 else 'The person is not suffering from Cancer and is fine.'
                    st.success(result)

                    # Speak the test result once
                    if not st.session_state.get("spoken_cancer_result", False):
                        speak_in_thread(result)
                        st.session_state.spoken_cancer_result = True
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        # Diabetes Prediction Page
        elif disease_type == 'Diabetes Prediction':
            st.subheader('Diabetes Prediction')

            if not st.session_state.get("spoken_diabetes_intro", False):
                speak_in_thread("Diabetes Prediction. Please enter the required details.")
                st.session_state.spoken_diabetes_intro = True

            st.image(images['Diabetes'], width=1000)
            st.text("BMI → Body Mass Index\n"
                    "Insulin → Insulin Level\n"
                    "SkinThickness → Skin fold Thickness\n"
                    "DiabetesPedigreeFunction → Family History of Diabetes")

            if "spoken_diabetes_result" not in st.session_state:
                st.session_state.spoken_diabetes_result = False

            pregnancies = st.number_input("Pregnancies", placeholder="Enter the value between 0 to 1")
            glucose = st.number_input("Glucose", placeholder="Enter the value between 40 to 120")
            blood_pressure = st.number_input("BloodPressure", placeholder="Enter the value between 20 to 80")
            bmi = st.number_input("BMI", placeholder="Enter the value between 17 to 25")
            insulin = st.number_input("Insulin", placeholder="Enter the value between 10 to 50")
            skin_thickness = st.number_input("SkinThickness", placeholder="Enter the value between 5 to 20")
            diabetes_pedigree_function = st.number_input("DiabetesPedigreeFunction", placeholder="Enter the value between 0.076 to 0.5")

            if st.button('Diabetes Test Result'):
                features = np.array(
                    [[pregnancies, glucose, blood_pressure, bmi, insulin, skin_thickness, diabetes_pedigree_function]])

                try:
                    diabetes_prediction = models['Diabetes_model'].predict(features)
                    result = 'The person is Diabetic.' if diabetes_prediction[0] == 1 else 'The person is not Diabetic.'
                    st.success(result)

                    # Speak the test result once
                    if not st.session_state.get("spoken_diabetes_result", False):
                        speak_in_thread(result)
                        st.session_state.spoken_diabetes_result = True
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        # Heart Disease Prediction Page
        elif disease_type == 'Heart Disease Prediction':
            st.subheader('Heart Disease Prediction')

            if not st.session_state.get("spoken_heart_intro", False):
                speak_in_thread("Heart Disease Prediction. Please enter the required details.")
                st.session_state.spoken_heart_intro = True

            st.image(images['Heart'], width=1000)
            st.text("Old Peak → Previous Exercise Stress Level\n"
                    "Thalassemia (thal) → Blood Disorder (Thalassemia)")

            if "spoken_heart_disease_result" not in st.session_state:
                st.session_state.spoken_heart_disease_result = False

            cp = st.number_input("Chest Pain Type (cp)", placeholder="Enter the value between 0 to 1")
            chol = st.number_input("Cholesterol (chol)", placeholder="Enter the value between 120 to 250")
            fbs = st.number_input("Fasting Blood Sugar (fbs)", placeholder="Enter the value between 0 to 1")
            thalach = st.number_input("Maximum Heart Rate (thalach)", placeholder="Enter the value between 70 to 150")
            oldpeak = st.number_input("Old Peak", placeholder="Enter the value between 0 to 1")
            thal = st.number_input("Thalassemia (thal)", placeholder="Enter the value between 0 to 2")

            if st.button('Heart Disease Test Result'):
                features = np.array([[cp, chol, fbs, thalach, oldpeak, thal]])

                try:
                    heart_disease_prediction = models['Heart_model'].predict(features)
                    result = 'The person is suffering from Heart Disease.' if heart_disease_prediction[
                                                                              0] == 1 else 'The person is not suffering from Heart Disease.'
                    st.success(result)

                    # Speak the test result once
                    if not st.session_state.get("spoken_heart_disease_result", False):
                        speak_in_thread(result)
                        st.session_state.spoken_heart_disease_result = True
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        # Parkinson Prediction Page
        elif disease_type == 'Parkinson Prediction':
            st.subheader('Parkinson Prediction')

            if not st.session_state.get("spoken_parkinson_intro", False):
                speak_in_thread("Parkinson Prediction. Please enter the required details.")
                st.session_state.spoken_parkinson_intro = True

            st.image(images['Parkinson'], width=1000)
            st.text("MDVP:Fo(Hz) → Pitch Frequency\n"
                    "MDVP:Flo(Hz) → Minimum Frequency\n"
                    "DFA → Fractal Dimension\n"
                    "RPDE → Non-Linear Measures of Voice\n"
                    "spread2 → Frequency Spread\n"
                    "HNR → Harmonics-to-Noise Ratio")

            if "spoken_Parkinson_result" not in st.session_state:
                st.session_state.spoken_Parkinson_result = False

            mdvp_fo_hz = st.number_input("MDVP:Fo(Hz)", placeholder="Enter the value between 85 to 120")
            mdvp_flo_hz = st.number_input("MDVP:Flo(Hz)", placeholder="Enter the value between 65 to 110")
            dfa = st.number_input("DFA", placeholder="Enter the value between 0.50 to 0.75")
            rpde = st.number_input("RPDE", placeholder="Enter the value between 0.20 to 0.5")
            spread2 = st.number_input("spread2", placeholder="Enter the value between 0 to 0.3")
            hnr = st.number_input("HNR", placeholder="Enter the value between 8 to 20")

            if st.button('Parkinson Test Result'):
                features = np.array([[mdvp_fo_hz, mdvp_flo_hz, dfa, rpde, spread2, hnr]])

                try:
                    parkinson_prediction = models['Parkinson_model'].predict(features)
                    result = 'The person is suffering from Parkinson.' if parkinson_prediction[
                                                                           0] == 1 else 'The person is not suffering from Parkinson.'
                    st.success(result)

                    # Speak the test result once
                    if not st.session_state.get("spoken_Parkinson_result", False):
                        speak_in_thread(result)
                        st.session_state.spoken_Parkinson_result = True
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        # Kidney Disease Prediction Page
        elif disease_type == 'Kidney Disease Prediction':
            st.subheader('Kidney Disease Prediction')

            if not st.session_state.get("spoken_kidney_intro", False):
                speak_in_thread("Kidney Disease Prediction. Please enter the required details.")
                st.session_state.spoken_kidney_intro = True

            st.image(images['Kidney'], width=1000)
            st.text("Specific Gravity (sg) → Urine Density\n"
                    "Albumin (al) → Protein Level in Urine\n"
                    "Hemoglobin (hemo) → Blood Hemoglobin Level\n"
                    "Pus Cell (pc) → Pus Cells in Urine\n"
                    "Appetite (appet) → Appetite Level")

            if "spoken_Kidney_disease_result" not in st.session_state:
                st.session_state.spoken_Kidney_disease_result = False

            bp = st.number_input("Blood Pressure (bp)", placeholder="Enter the value between 50 to 80")
            sg = st.number_input("Specific Gravity (sg)", placeholder="Enter the value between 1 to 1.05")
            al = st.number_input("Albumin (al)", placeholder="Enter the value between 0 to 1")
            hemo = st.number_input("Hemoglobin (hemo)", placeholder="Enter the value between 3 to 13")
            pc = st.selectbox("Pus Cell (pc)", options=['normal', 'abnormal'])
            ba = st.selectbox("Bacteria (ba)", options=['present', 'not present'])
            appet = st.selectbox("Appetite (appet)", options=['good', 'poor'])

            pc_encoded = [1 if pc == 'abnormal' else 0]
            ba_encoded = [1 if ba == 'present' else 0]
            appet_encoded = [1 if appet == 'poor' else 0]

            if st.button('Kidney Disease Test Result'):
                features = np.array([[bp, sg, al, hemo, pc_encoded[0], ba_encoded[0], appet_encoded[0]]])

                try:
                    Kidney_Disease_prediction = models['Kidney_model'].predict(features)
                    result = 'The person is suffering from Kidney Disease.' if Kidney_Disease_prediction[
                                                                               0] == 1 else 'The person is not suffering from Kidney Disease.'
                    st.success(result)

                    # Speak the test result once
                    if not st.session_state.get("spoken_Kidney_disease_result", False):
                        speak_in_thread(result)
                        st.session_state.spoken_Kidney_disease_result = True
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

        # Liver Prediction Page
        elif disease_type == 'Liver Prediction':
            st.subheader('Liver Prediction')

            if not st.session_state.get("spoken_liver_intro", False):
                speak_in_thread("Liver Prediction. Please enter the required details.")
                st.session_state.spoken_liver_intro = True

            st.image(images['Liver'], width=1000)
            st.text("Total Bilirubin → Total Bilirubin Level\n"
                    "Direct Bilirubin → Direct Bilirubin Level\n"
                    "Alkaline Phosphatase → Liver Enzyme Level\n"
                    "Alamine Aminotransferase → Liver Enzyme (ALT)\n"
                    "Aspartate Aminotransferase → Liver Enzyme (AST)\n"
                    "Albumin → Protein Level in Blood\n"
                    "Albumin and Globulin Ratio → Protein Ratio (Albumin/Globulin)")

            if "spoken_Liver_disease_result" not in st.session_state:
                st.session_state.spoken_Liver_disease_result = False

            Total_Bilirubin = st.number_input("Total Bilirubin", placeholder="Enter the value between 0 to 1")
            Direct_Bilirubin = st.number_input("Direct Bilirubin", placeholder="Enter the value between 0 to 0.3")
            Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", placeholder="Enter the value between 100 to 250")
            Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", placeholder="Enter the value between 10 to 30")
            Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", placeholder="Enter the value between 10 to 40")
            Albumin = st.number_input("Albumin", placeholder="Enter the value between 2 to 4")
            Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", placeholder="Enter the value between 0.3 to 1")

            if st.button('Liver Test Result'):
                features = np.array([[Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase,
                                      Aspartate_Aminotransferase, Albumin, Albumin_and_Globulin_Ratio]])
                try:
                    Liver_prediction = models['Liver_model'].predict(features)
                    result = 'The person is suffering from Liver Disease.' if Liver_prediction[
                                                                              0] == 1 else 'The person is not suffering from Liver Disease.'
                    st.success(result)

                    # Speak the test result once
                    if not st.session_state.get("spoken_Liver_disease_result", False):
                        speak_in_thread(result)
                        st.session_state.spoken_Liver_disease_result = True
                except Exception as e:
                    st.error(f"Error in prediction: {e}")

def page_navigation():
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "form":
        info_form_page()
    elif st.session_state.page == "health_assistant":
        health_assistant_page()
    elif st.session_state.page == "image_prediction":
        health_assistant_page()

# Start the page navigation
page_navigation()