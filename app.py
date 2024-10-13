import os
import io
import re
import glob
import streamlit as st
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
import paddle
from PIL import Image
import pytesseract
import pandas as pd
import numpy as np
from streamlit_cropper import st_cropper

# Set up folders
if not os.path.exists("Circuit"):
    os.makedirs("Circuit")
if not os.path.exists("Label"):
    os.makedirs("Label")

# Set environment variable to force PaddlePaddle to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Check if PaddlePaddle is using GPU or CPU
gpu_available = paddle.device.is_compiled_with_cuda()

# Initialize session state for view mode and selected page
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'PDF Viewer'
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = None
if 'uploaded_pdf' not in st.session_state:
    st.session_state.uploaded_pdf = None

# Sidebar: Tool selection and folder image display
st.sidebar.title("Select Tool")
tool_option = st.sidebar.radio(
    "Choose a tool", ["PDF Cropping Tool", "Comparison Page"])

# st.sidebar.subheader("Circuit Folder")
# circuit_images = sorted(glob.glob("Circuit/*.png"))
# for img_path in circuit_images:
#     st.sidebar.image(img_path, caption=os.path.basename(img_path), use_column_width=True)

# st.sidebar.subheader("Label Folder")
# label_images = sorted(glob.glob("Label/*.png"))
# for img_path in label_images:
#     st.sidebar.image(img_path, caption=os.path.basename(img_path), use_column_width=True)

st.sidebar.subheader("Circuit Folder")
circuit_images = sorted(glob.glob("Circuit/*.png"))

if "selected_circuit_image" not in st.session_state:
    st.session_state["selected_circuit_image"] = None

for img_path in circuit_images:
    if st.sidebar.button(f"Select {os.path.basename(img_path)}", key=f"circuit_{img_path}"):
        st.session_state["selected_circuit_image"] = img_path

st.sidebar.subheader("Label Folder")
label_images = sorted(glob.glob("Label/*.png"))

if "selected_label_image" not in st.session_state:
    st.session_state["selected_label_image"] = None

for img_path in label_images:
    if st.sidebar.button(f"Select {os.path.basename(img_path)}", key=f"label_{img_path}"):
        st.session_state["selected_label_image"] = img_path


def extract_text_from_image(image_path):
    # Open the image file directly using the path
    image = Image.open(image_path)
    image_np = np.array(image)

    ocr = PaddleOCR(
        use_angle_cls=True,  # Enable angle classification to detect rotated text
        det_db_thresh=0.5,  # Adjusted threshold for detection
        det_db_box_thresh=0.6,  # Adjusted box detection threshold
        det_limit_side_len=2048  # Larger limit for detection
    )

    # Perform OCR on the image
    result = ocr.ocr(image_np, cls=True)

    # Extract and return text
    extracted_text = []
    if result:
        for line in result:
            for word_info in line:
                text = word_info[1][0]
                extracted_text.append(text)
    return extracted_text


def extract_and_sort_components(image_path):
    # Load the image
    img = Image.open(image_path)

    # # Set the path to the Tesseract executable
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Users\User.ET-A0078\Downloads\tesseract.exe"

    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(img)

    # # Define a regex pattern to match component codes and descriptions
    # pattern = re.compile(r'([A-Z$]\d+(?:-\d+)?)\s*[-—]\s*(.+?)(?=\s*[A-Z$]\d+[-—]|\s*$)', re.DOTALL)

    # This gives the 1 2 3
    pattern = re.compile(
        r'(\b[123]\b|[A-Z$]\d+(?:-\d+)?)\s*[-—]\s*(.+?)(?=\s*(?:\b[123]\b|[A-Z$]\d+[-—]|\s*$))',
        re.DOTALL
    )

    # Find all matches
    matches = pattern.findall(text)

    # Clean up and correct the matches
    cleaned_matches = []
    for code, description in matches:
        # Replace '$' with 'S' in component codes
        code = code.replace('$', 'S')
        # Remove newlines and extra spaces from descriptions
        description = ' '.join(description.split())
        cleaned_matches.append((code, description))

    # Sort the matches
    sorted_matches = sorted(cleaned_matches, key=lambda x: (
        x[0][0], int(re.findall(r'\d+', x[0])[0])))

    return sorted_matches


def compare_components(circuit_texts, sorted_components):
    missing_components = []
    for code, _ in sorted_components:
        if code not in circuit_texts:
            missing_components.append(code)
    return missing_components


def save_to_excel(circuit_texts, sorted_components, missing_components):
    # Your existing save_to_excel function, modified to return a BytesIO object
    max_len = max(len(circuit_texts), len(
        sorted_components), len(missing_components))
    circuit_texts += [''] * (max_len - len(circuit_texts))
    sorted_components += [('', '')] * (max_len - len(sorted_components))
    missing_components += [''] * (max_len - len(missing_components))

    circuit_labels = pd.Series(circuit_texts, name='Circuit Labels')
    component_labels = pd.Series(
        [f"{code} - {desc}" for code, desc in sorted_components], name='Component Labels')
    missing_labels = pd.Series(missing_components, name='Missing Labels')

    df = pd.DataFrame({
        'Circuit Labels': circuit_labels,
        'Component Labels': component_labels,
        'Missing Labels': missing_labels
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


# PDF Cropping Tool Page
if tool_option == "PDF Cropping Tool":
    st.title("PDF Cropping Tool")

    # Upload PDF
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        # Save uploaded PDF in session state
        st.session_state.uploaded_pdf = uploaded_pdf

        # Convert PDF pages to images
        pages = convert_from_bytes(uploaded_pdf.read())

        # PDF Viewer Mode
        if st.session_state.view_mode == 'PDF Viewer':
            st.subheader("Scrollable PDF Viewer")
            st.write("Scroll to view pages and select one for cropping:")

            # Display all pages with select buttons
            for i, page_image in enumerate(pages):
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(f"Select Page {i + 1}", key=f"page_button_{i}"):
                        st.session_state.selected_page = i
                        st.session_state.view_mode = 'Cropping Tool'  # Switch to cropping mode
                with col2:
                    st.image(page_image, caption=f'''Page {
                             i + 1}''', use_column_width=True)

        # Cropping Tool Mode
        elif st.session_state.view_mode == 'Cropping Tool' and st.session_state.selected_page is not None:
            selected_page = st.session_state.selected_page
            st.subheader(f"Cropping Tool - Page {selected_page + 1}")
            st.write("Select area to crop:")
            cropped_image = st_cropper(
                pages[selected_page], realtime_update=True, box_color="blue", aspect_ratio=None)

            # Display cropped image preview and save options
            if cropped_image:
                st.image(cropped_image, caption="Cropped Image Preview",
                         use_column_width=True)

                # Save options
                save_as = st.selectbox("Save as", ["Circuit", "Label"])
                save_button = st.button("Save Screenshot")

                if save_button:
                    # Choose folder based on selection
                    folder = "Circuit" if save_as == "Circuit" else "Label"

                    # Get the current count of images in the folder to increment filename
                    count = len(glob.glob(f"{folder}/*.png")) + 1
                    filename = f"{folder.lower()}{count:02d}.png"
                    filepath = os.path.join(folder, filename)

                    # Save the cropped image
                    cropped_image.save(filepath)
                    st.success(f"Saved as {filename}")

                    # After saving, return to PDF Viewer
                    st.session_state.view_mode = 'PDF Viewer'

                    # Re-initialize the uploaded PDF in session state to retain the previous upload
                    st.session_state.uploaded_pdf = uploaded_pdf

# Comparison Page
elif tool_option == "Comparison Page":
    st.title("Circuit-Label Analyzer")

    # Display selected images
    st.subheader("Selected Images for Comparison")

    # Display the selected Circuit image
    if st.session_state["selected_circuit_image"]:
        st.write("Circuit Image:")
        st.image(
            st.session_state["selected_circuit_image"], use_column_width=True)
    else:
        st.write("No Circuit image selected.")

    # Display the selected Label image
    if st.session_state["selected_label_image"]:
        st.write("Label Image:")
        st.image(
            st.session_state["selected_label_image"], use_column_width=True)
    else:
        st.write("No Label image selected.")

    # Image comparison and analysis logic
    if st.session_state["selected_circuit_image"] and st.session_state["selected_label_image"]:
        if st.button("Analyze Images"):
            with st.spinner("Processing images..."):
                # Extract text from the circuit image
                circuit_texts = extract_text_from_image(
                    st.session_state["selected_circuit_image"])

                # Extract and sort components from the component image
                sorted_components = extract_and_sort_components(
                    st.session_state["selected_label_image"])

                # Compare components
                missing_components = compare_components(
                    circuit_texts, sorted_components)

                # Display results
                st.subheader("Results")
                st.write("Circuit Components:", circuit_texts)
                st.write("Label Components:", [
                         f"{code} - {desc}" for code, desc in sorted_components])

                if missing_components:
                    st.write("Missing Components:", missing_components)
                else:
                    st.write("All components are present in the circuit image.")

                # Generate Excel file
                excel_file = save_to_excel(
                    circuit_texts, sorted_components, missing_components)
                excel_filename = f'''{os.path.basename(st.session_state['selected_circuit_image'])}__{
                    os.path.basename(st.session_state['selected_label_image'])}.xlsx'''

                # Provide download link
                st.download_button(
                    label="Download Excel Report",
                    data=excel_file,
                    file_name=excel_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
