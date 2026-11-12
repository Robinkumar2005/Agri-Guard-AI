
import streamlit as st
import requests
import json

from pathlib import Path
from PIL import Image
from io import BytesIO


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgriGuard AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000"


# =========================================================
# 2. SESSION STATE
# =========================================================

defaults = {
    "navigation": "Dashboard",
    "selected_prediction": None,
    "single_result": None,
    "single_result_filename": None,
    "batch_results": [],
    "prediction_history": [],
    "chat_messages": [],
    "chat_prediction_id": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main application background */

.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #F4F7F4 !important;
}

/* Main content text */

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] li,
[data-testid="stMain"] span {
    color: #17251B !important;
}

/* Main headings */

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #153D2B !important;
    font-weight: 700 !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #153D2B !important;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Sidebar navigation selection */

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF !important;
}

/* Sidebar status */

section[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: #E1F2E6 !important;
}

section[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #153D2B !important;
}

/* Metric cards */

.metric-card {
    background-color: #FFFFFF;
    padding: 22px;
    border: 1px solid #DCE6DE;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    min-height: 105px;
}

.metric-value {
    font-size: 27px;
    font-weight: 700;
    color: #1D6840 !important;
}

.metric-label {
    font-size: 14px;
    color: #17251B !important;
}

/* General information cards */

.info-card {
    background-color: #FFFFFF;
    border: 1px solid #DCE6DE;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 12px;
}

.info-card h3 {
    color: #153D2B !important;
}

.info-card p {
    color: #17251B !important;
}

/* Buttons */

.stButton > button {
    background-color: #FFFFFF !important;
    color: #153D2B !important;
    border: 1px solid #1D6840 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background-color: #E1F2E6 !important;
    color: #153D2B !important;
}

/* Primary buttons */

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #1D6840 !important;
    color: #FFFFFF !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-color: #153D2B !important;
    color: #FFFFFF !important;
}

/* Text inputs */

.stTextInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #17251B !important;
    border: 1px solid #AABBAE !important;
}

/* Input placeholder */

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #64736A !important;
}

/* Select boxes */

[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
}

[data-baseweb="select"] * {
    color: #17251B !important;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px dashed #8DA895 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploader"] * {
    color: #17251B !important;
}

/* Alerts */

[data-testid="stAlert"] {
    background-color: #E8F2EB !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #17251B !important;
}

/* Expanders */

[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE6DE !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #17251B !important;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
}

[data-testid="stChatMessage"] p {
    color: #17251B !important;
}

/* Tables */

[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def check_backend():

    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def get_prediction_details(result):

    prediction = result.get("prediction", {})

    disease = prediction.get("disease", "Unknown")

    confidence = prediction.get("confidence", 0)

    return disease, confidence


def get_ai_explanation(result):

    if result.get("ai_explanation"):
        return result["ai_explanation"]

    ai_agent = result.get("ai_agent", {})

    if isinstance(ai_agent, dict):
        return ai_agent.get(
            "response",
            "No AI explanation available."
        )

    return "No AI explanation available."


def save_prediction(result):

    st.session_state.selected_prediction = result

    st.session_state.prediction_history.append(result)

    st.session_state.chat_messages = []

    st.session_state.chat_prediction_id = None


def show_prediction_result(result):

    disease, confidence = get_prediction_details(result)

    explanation = get_ai_explanation(result)

    st.markdown("## Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Detected Class",
            disease
        )

    with col2:
        st.metric(
            "CNN Confidence Score",
            f"{confidence * 100:.2f}%"
        )

    st.progress(
        min(max(float(confidence), 0.0), 1.0)
    )

    st.caption(
        "The CNN confidence score is not a calibrated probability "
        "or a confirmed agricultural diagnosis."
    )

    st.divider()

    st.markdown("### AI-Generated Explanation")

    st.markdown(explanation)

    st.caption(
        "Verify agricultural treatment recommendations with "
        "reliable agricultural sources and local experts."
    )

    prediction_id = result.get("database_id")

    if prediction_id:
        st.caption(f"Prediction ID: {prediction_id}")


def format_class_name(class_name):

    return (
        class_name
        .replace("___", " - ")
        .replace("_", " ")
        .replace("  ", " ")
        .title()
    )


def load_disease_classes():

    class_file = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "models"
        / "class_names.json"
    )

    try:

        with open(class_file, "r", encoding="utf-8") as file:
            classes = json.load(file)

        if isinstance(classes, dict):
            classes = list(classes.keys())

        return [format_class_name(name) for name in classes]

    except (FileNotFoundError, json.JSONDecodeError, TypeError):

        return [
            "Apple - Apple Scab",
            "Apple - Black Rot",
            "Apple - Cedar Apple Rust",
            "Apple - Healthy",
            "Blueberry - Healthy",
            "Cherry - Healthy",
            "Cherry - Powdery Mildew",
            "Corn - Cercospora Leaf Spot / Gray Leaf Spot",
            "Corn - Common Rust",
            "Corn - Healthy",
            "Corn - Northern Leaf Blight",
            "Grape - Black Rot",
            "Grape - Esca (Black Measles)",
            "Grape - Healthy",
            "Grape - Leaf Blight (Isariopsis Leaf Spot)",
            "Orange - Huanglongbing (Citrus Greening)",
            "Peach - Bacterial Spot",
            "Peach - Healthy",
            "Pepper - Bell Bacterial Spot",
            "Pepper - Bell Healthy",
            "Potato - Early Blight",
            "Potato - Late Blight",
            "Potato - Healthy",
            "Raspberry - Healthy",
            "Soybean - Healthy",
            "Squash - Powdery Mildew",
            "Strawberry - Leaf Scorch",
            "Strawberry - Healthy",
            "Tomato - Bacterial Spot",
            "Tomato - Early Blight",
            "Tomato - Late Blight",
            "Tomato - Leaf Mold",
            "Tomato - Septoria Leaf Spot",
            "Tomato - Spider Mites (Two-Spotted Spider Mite)",
            "Tomato - Target Spot",
            "Tomato - Yellow Leaf Curl Virus",
            "Tomato - Tomato Mosaic Virus",
            "Tomato - Healthy"
        ]


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🌱 AgriGuard AI")

    st.caption("Intelligent Plant Health System")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Disease Detection",
            "AI Assistant",
            "Prediction History",
            "About Model"
        ],
        key="navigation"
    )

    st.divider()

    st.markdown("### System Status")

    if check_backend():
        st.success("Backend Connected")
    else:
        st.error("Backend Offline")

    st.divider()

    st.caption("Powered by CNN + AI")


# =========================================================
# 6. DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        '<h1 style="color:#153D2B;">Plant Health Dashboard</h1>',
        unsafe_allow_html=True
    )

    st.markdown(
        "Detect plant diseases and receive AI-assisted "
        "agricultural guidance."
    )

    st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">38</div>
            <div class="metric-label">Disease Classes</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">CNN</div>
            <div class="metric-label">Classification Model</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">AI</div>
            <div class="metric-label">Agricultural Assistant</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.markdown("## Welcome to AgriGuard AI 🌿")

    st.write(
        "AgriGuard AI combines deep learning and AI-assisted "
        "agricultural guidance to analyze plant leaf images."
    )

    st.markdown("## Explore AgriGuard")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h3>🌿 Disease Detection</h3>
            <p>
                Upload plant images and identify possible
                disease classes using a CNN model.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("Open the Disease Detection page from the sidebar.")

    with col2:

        st.markdown("""
        <div class="info-card">
            <h3>🤖 AI Assistant</h3>
            <p>
                Ask follow-up questions about plant predictions
                and receive AI-assisted guidance.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("Open the AI Assistant page from the sidebar.")

    st.divider()

    st.markdown("## How AgriGuard Works")

    st.write("""
    1. Upload a plant leaf image.
    2. The CNN model analyzes the image.
    3. The model predicts one of its 38 trained classes.
    4. The AI agent generates an explanation and management guidance.
    5. Ask additional questions using the AI Assistant.
    """)


# =========================================================
# 7. DISEASE DETECTION
# =========================================================

elif page == "Disease Detection":

    st.markdown("# 🌿 Disease Detection")

    st.write(
        "Upload plant leaf images to identify possible disease classes."
    )

    detection_mode = st.radio(
        "Select Detection Mode",
        ["Single Image", "Batch Images"],
        horizontal=True
    )

    st.divider()

    # SINGLE IMAGE DETECTION

    if detection_mode == "Single Image":

        uploaded_file = st.file_uploader(
            "Upload Plant Leaf Image",
            type=["jpg", "jpeg", "png"],
            key="single_upload"
        )

        if uploaded_file:

            image = Image.open(
                BytesIO(uploaded_file.getvalue())
            ).convert("RGB")

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Uploaded Plant Image",
                    use_container_width=True
                )

            with col2:

                st.markdown("### Image Information")

                st.write(f"Filename: {uploaded_file.name}")

                st.write(f"Image Size: {image.size}")

                st.write(
                    f"File Size: {uploaded_file.size / 1024:.2f} KB"
                )

                if st.button(
                    "Analyze Plant Image",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner("Analyzing plant image..."):

                        try:

                            files = {
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type
                                )
                            }

                            response = requests.post(
                                f"{API_URL}/single_predict",
                                files=files,
                                timeout=300
                            )

                            if response.status_code == 200:

                                result = response.json()

                                st.session_state.single_result = result

                                st.session_state.single_result_filename = (
                                    uploaded_file.name
                                )

                                save_prediction(result)

                                st.success(
                                    "Plant analysis completed successfully!"
                                )

                            else:

                                st.error(
                                    f"API Error: {response.status_code}"
                                )

                                st.code(response.text)

                        except requests.RequestException as error:

                            st.error(f"Connection Error: {error}")

            if (
                st.session_state.single_result is not None
                and st.session_state.single_result_filename
                == uploaded_file.name
            ):

                st.divider()

                show_prediction_result(
                    st.session_state.single_result
                )

    # BATCH IMAGE DETECTION

    else:

        uploaded_files = st.file_uploader(
            "Upload Multiple Plant Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload"
        )

        if uploaded_files:

            st.write(
                f"Total Images Selected: {len(uploaded_files)}"
            )

            cols = st.columns(3)

            for index, file in enumerate(uploaded_files):

                image = Image.open(
                    BytesIO(file.getvalue())
                ).convert("RGB")

                with cols[index % 3]:

                    st.image(
                        image,
                        caption=file.name,
                        use_container_width=True
                    )

            if st.button(
                "Analyze All Images",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("Processing all images..."):

                    try:

                        files = [
                            (
                                "files",
                                (
                                    file.name,
                                    file.getvalue(),
                                    file.type
                                )
                            )
                            for file in uploaded_files
                        ]

                        response = requests.post(
                            f"{API_URL}/batch_predict",
                            files=files,
                            timeout=600
                        )

                        if response.status_code == 200:

                            results = response.json()

                            st.session_state.batch_results = results

                            for result in results:

                                if result.get("prediction"):
                                    st.session_state.prediction_history.append(
                                        result
                                    )

                            st.success(
                                "Batch analysis completed!"
                            )

                        else:

                            st.error(
                                f"API Error: {response.status_code}"
                            )

                            st.code(response.text)

                    except requests.RequestException as error:

                        st.error(f"Connection Error: {error}")

            if st.session_state.batch_results:

                st.divider()

                st.markdown("## Batch Prediction Results")

                for index, result in enumerate(
                    st.session_state.batch_results
                ):

                    filename = result.get(
                        "filename",
                        f"Image {index + 1}"
                    )

                    with st.expander(filename):

                        if result.get("error"):

                            st.error(result["error"])

                        elif result.get("prediction"):

                            disease, confidence = (
                                get_prediction_details(result)
                            )

                            st.write(f"**Prediction:** {disease}")

                            st.write(
                                f"**CNN Confidence:** "
                                f"{confidence * 100:.2f}%"
                            )

                            st.progress(
                                min(max(float(confidence), 0.0), 1.0)
                            )

                            st.markdown("### AI Explanation")

                            st.markdown(
                                get_ai_explanation(result)
                            )

                            if st.button(
                                "Continue Chat About This Prediction",
                                key=f"batch_chat_{index}"
                            ):

                                st.session_state.selected_prediction = result

                                st.session_state.chat_messages = []

                                st.session_state.chat_prediction_id = None

                                st.session_state.navigation = "AI Assistant"

                                st.rerun()

                        else:

                            st.warning(
                                "Unexpected prediction response format."
                            )


# =========================================================
# 8. AI ASSISTANT
# =========================================================

elif page == "AI Assistant":

    st.markdown("# 🤖 AgriGuard AI Assistant")

    st.write(
        "Ask questions about your plant prediction and its management."
    )

    result = st.session_state.selected_prediction

    if not result:

        st.warning(
            "Please analyze a plant image before starting a conversation."
        )

    else:

        disease, confidence = get_prediction_details(result)

        prediction_id = result.get("database_id")

        st.info(
            f"Current Prediction: {disease} | "
            f"CNN Score: {confidence * 100:.2f}%"
        )

        if not prediction_id:

            st.error(
                "Prediction ID is missing. AI chat is unavailable."
            )

        else:

            if (
                st.session_state.chat_prediction_id
                != prediction_id
            ):

                st.session_state.chat_messages = []

                st.session_state.chat_prediction_id = prediction_id

            for message in st.session_state.chat_messages:

                with st.chat_message(message["role"]):

                    st.markdown(message["content"])

            user_message = st.chat_input(
                "Ask something about this plant prediction..."
            )

            if user_message:

                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": user_message
                })

                with st.chat_message("user"):

                    st.markdown(user_message)

                with st.chat_message("assistant"):

                    with st.spinner("AgriGuard AI is thinking..."):

                        try:

                            response = requests.post(
                                f"{API_URL}/chat",
                                json={
                                    "prediction_id": prediction_id,
                                    "message": user_message
                                },
                                timeout=180
                            )

                            if response.status_code == 200:

                                data = response.json()

                                ai_response = data.get(
                                    "ai_response",
                                    "No response received."
                                )

                            else:

                                ai_response = (
                                    f"API Error: {response.status_code}\n\n"
                                    f"{response.text}"
                                )

                        except requests.RequestException as error:

                            ai_response = f"Connection Error: {error}"

                        st.markdown(ai_response)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": ai_response
                })


# =========================================================
# 9. PREDICTION HISTORY
# =========================================================

elif page == "Prediction History":

    st.markdown("# 📋 Prediction History")

    st.write(
        "Review plant analyses performed during this session."
    )

    results = st.session_state.prediction_history

    if results:

        st.metric(
            "Total Predictions",
            len(results)
        )

        for index, result in enumerate(reversed(results)):

            filename = result.get(
                "filename",
                f"Image {index + 1}"
            )

            with st.expander(filename):

                if result.get("prediction"):

                    disease, confidence = (
                        get_prediction_details(result)
                    )

                    st.write(f"**Disease:** {disease}")

                    st.write(
                        f"**CNN Confidence:** "
                        f"{confidence * 100:.2f}%"
                    )

                    st.markdown("### AI Explanation")

                    st.markdown(
                        get_ai_explanation(result)
                    )

                    if st.button(
                        "Open in AI Assistant",
                        key=f"history_chat_{index}"
                    ):

                        st.session_state.selected_prediction = result

                        st.session_state.chat_messages = []

                        st.session_state.chat_prediction_id = None

                        st.session_state.navigation = "AI Assistant"

                        st.rerun()

                elif result.get("error"):

                    st.error(result["error"])

    else:

        st.info(
            "No predictions available. "
            "Analyze a plant image to get started."
        )

    st.caption(
        "History is currently session-based. "
        "Your backend stores predictions in MongoDB, but permanent "
        "history retrieval requires a GET endpoint."
    )


# =========================================================
# 10. ABOUT MODEL
# =========================================================

elif page == "About Model":

    st.markdown("# 🧠 About AgriGuard AI")

    st.write(
        "Learn about the machine learning model and technologies "
        "used in AgriGuard AI."
    )

    st.divider()

    st.markdown("## CNN Classification Model")

    st.write("""
    AgriGuard AI uses a custom Convolutional Neural Network (CNN)
    to classify plant leaf images into 38 categories using the
    PlantVillage dataset.
    """)

    col1, col2 = st.columns(2)

    with col1:

        st.metric("Input Image Size", "224 × 224")

        st.metric("Output Classes", "38")

        st.metric("Optimizer", "Adam")

    with col2:

        st.metric("Architecture", "Custom CNN")

        st.metric("Activation", "ReLU")

        st.metric("Loss Function", "Cross Entropy")

    st.divider()

    st.markdown("## Model Evaluation")

    st.write(
        "Recorded evaluation results from the model's test dataset:"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "94.15%")

    with col2:
        st.metric("Precision", "94.52%")

    with col3:
        st.metric("Recall", "94.15%")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("F1 Score", "94.15%")

    with col2:
        st.metric("AUC", "99.92%")

    st.divider()

    st.markdown("## All Supported Plant Classes")

    disease_classes = load_disease_classes()

    st.write(
        f"Your CNN model supports **{len(disease_classes)} classes**."
    )

    class_columns = st.columns(3)

    for index, disease_class in enumerate(disease_classes):

        with class_columns[index % 3]:

            st.markdown(
                f"""
                <div style="
                    background-color: #FFFFFF;
                    color: #17251B;
                    padding: 12px;
                    margin-bottom: 9px;
                    border: 1px solid #DCE6DE;
                    border-radius: 8px;
                    font-size: 13px;
                ">
                    <strong style="color: #1D6840;">
                        {index + 1}.
                    </strong>
                    <span style="color: #17251B;">
                        {disease_class}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.markdown("## AI Agricultural Assistant")

    st.write("""
    The AI assistant uses a large language model and external
    agricultural information retrieval to explain CNN predictions
    and provide management guidance.

    The AI assistant supports the classifier but does not replace
    professional agricultural diagnosis.
    """)

    st.divider()

    st.markdown("## Technologies Used")

    technologies = [
        "Python",
        "PyTorch",
        "Custom CNN",
        "FastAPI",
        "Streamlit",
        "MongoDB",
        "LangChain",
        "Mistral AI",
        "Tavily Search"
    ]

    tech_columns = st.columns(3)

    for index, technology in enumerate(technologies):

        with tech_columns[index % 3]:

            st.markdown(
                f"""
                <div style="
                    background-color: #FFFFFF;
                    color: #17251B;
                    padding: 12px;
                    margin-bottom: 9px;
                    border: 1px solid #DCE6DE;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: 600;
                ">
                    {technology}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.caption(
        "AgriGuard AI | Plant Disease Classification "
        "and AI-Assisted Agricultural Guidance"
    )
