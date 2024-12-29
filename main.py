import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np

# Cache the model loading process
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/trained_model.h5')
    return model

# TensorFlow model prediction
def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                }
        </style>
        """, unsafe_allow_html=True)



# Sidebar Navigation
app_mode = st.sidebar.radio(
    "Choose a Page",
    ["Home", "About Us", "Diagnosis"],
    index=2  # Default to "Diagnosis"
)



# Homepage
if app_mode == "Home":
    st.title("Welcome to Diabetic Retinopathy Detection")
    
    # Display an image of diabetic retinopathy
    # st.image("path_to_your_image/diabetic_retinopathy.jpg", use_column_width=True)  # Update the image path

    st.markdown("""
    Diabetic retinopathy is a serious eye condition that affects individuals with diabetes, potentially leading to vision loss. Our application leverages cutting-edge machine learning techniques to **detect diabetic retinopathy early**, enabling timely intervention and effective management.
    """)

    # Display the image
    st.image("img/stageDR.jpg", caption="Stages of Diabetic Retinopathy", use_column_width=True)


    st.markdown("""
    ## How It Works
    1. **Image Upload**: Use the file uploader to upload a retinal image.
    2. **Prediction**: Click the 'Predict' button to start the analysis.
    3. **Results**: Receive instant feedback on the presence and severity of diabetic retinopathy.

    ## Why Early Detection Matters
    Early diagnosis of diabetic retinopathy can significantly reduce the risk of severe vision impairment. Regular screenings and prompt interventions can help manage the disease effectively and protect your vision.
    Start your journey towards better eye health today! 

    ### Ready to Get Started?
    - **Upload your retinal image now** and see how our tool can help you!
    """)

    # Additional Call to Action with a friendly reminder
    st.markdown("""

    🥳 **Let’s begin your journey to healthier eyes!** \n
    If you're ready to analyze your retinal image, head over to the **`Diagnosis`** page for predictions!
    """)

    

# About Us Page 
elif app_mode == "About Us":
    st.header("About Us")
    st.markdown(
        """
        We are a dedicated team focused on developing an innovative machine learning solution for Diabetic Retinopathy Detection, aimed at early diagnosis to improve patient care and enhance eye healthcare.

        ### Guidance:
        This project has been created under the expert guidance of **Dr. Savita Gupta**, whose invaluable mentorship has helped shape our approach and ensure the accuracy and relevance of our solution in the field of healthcare.

        ### Our Team:

        - **Shivanshu Sawan**  
          I am final-year Computer Science student at UIET, Panjab University, I am responsible for the overall development and implementation of the project. My expertise in artificial intelligence and machine learning drives the core of the model, ensuring it is both innovative and impactful.
          - 🔗 [GitHub](https://github.com/shivanshusawan)
          - 🔗 [LinkedIn](https://www.linkedin.com/in/shivanshu-sawan)

        - **Zul Quarnain Azam**  
          Zul contributes primarily to documentation and research, playing an essential role in organizing project information, improving technical documentation, and supporting the team in dataset preparation. His dedication ensures that the project maintains its focus on quality and usability.

        - **Kshitij Negi**  
          Kshitij plays a crucial role in researching and sourcing relevant datasets for the project. His attention to detail and commitment to documentation ensure the smooth integration of data and the overall transparency of the development process.

        ### Our Vision:
        We aim to transform healthcare by empowering clinicians with fast, accurate tools for diabetic retinopathy detection. Combining machine learning with intuitive design, our platform will improve diagnosis efficiency and accessibility for healthcare professionals worldwide.

        ### Connect With Us!
        We're excited to share our progress as we continue innovating in medical AI. Your feedback is crucial to our success—stay tuned for updates, and feel free to reach out! Your support helps shape the future of this project.

        Thank you for exploring our project! 😊
        """
    )




# Prediction Page
elif app_mode == "Diagnosis":
    st.header("Diabetic Retinopathy Detection")
    
    # Section for downloading the test dataset
    st.markdown("### Download the Test Dataset")
    st.write("""
    Enhance your experience with the Diabetic Retinopathy Detection tool by downloading our comprehensive test dataset. 
    This dataset contains a variety of retinal images, useful for practicing and testing the diagnostic capabilities 
    of our model. Click the button below to download the dataset in `.rar` format.
    """)
    
    # Download button for the test dataset
    with open("test_dataset.rar", "rb") as f:
        bytes_data = f.read()
    
    st.download_button(label="Download Test Dataset",
                       data=bytes_data,
                       file_name="test_dataset.rar",
                       mime="application/octet-stream",
                       help="Click here to download the test dataset")

    st.markdown("---")  # Add a horizontal line for better separation

    # Image uploader with size limit (in bytes)
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"], 
                                   help="Upload an image file (max size: 2MB)", 
                                   label_visibility="collapsed")


    # Automatically display the uploaded image
    if test_image is not None:
        # Check the file size
        if test_image.size > 2 * 1024 * 1024:  # 2MB limit
            st.error("File size exceeds 2MB. Please upload a smaller image.")
            test_image = None
        else:
            # Display the image 
            st.image(test_image, caption='Original Image', use_column_width=0.5)


            # Make prediction using the AI model
            if st.button("Predict", help="Make a prediction on the processed image"):
                with st.spinner("Processing... Please wait..."):                 
                    st.write("Our Prediction:")
                    result_index = model_prediction(test_image)

                    # Define class names
                    class_name = ['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']
                    st.success(f"The model predicts it's a: **{class_name[result_index]}**")

                    

    else:
        st.info("✨ **Please upload your retinal image for prediction.**  \n"
                 "To get started, simply select an image from your device.")