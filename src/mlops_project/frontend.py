import requests
import streamlit as st

API_ENDPOINT = "http://127.0.0.1:8000/predict_pneumonia/"


def main() -> None:
    """Main function of the Streamlit frontend."""
    st.title("Pneumonia Detection")

    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.write("")

        if st.button("Predict"):
            try:
                files = {"file": uploaded_file.getvalue()}
                headers = {"Content-Type": "multipart/form-data"}

                response = requests.post(API_ENDPOINT, files={"file": uploaded_file})

                if response.status_code == 200:
                    st.success(f"Prediction: {response.json()['label']}")
                else:
                    st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
