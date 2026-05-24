import os
import streamlit as st
import requests

# ======================================================
# API URL
# ======================================================

API_URL = os.getenv(
    "API_URL",
    "http://flask-api:5000/predict"
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(

    page_title="Academic Performance Analyzer",
    page_icon="🎓",
    layout="centered"

)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""

<style>

.main {
    padding-top: 2rem;
}

.block-container {
    padding-top: 1rem;
}

h1 {
    text-align: center;
}

.metric-card {

    background-color: #111827;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
    color: white;
}

.metric-title {

    font-size: 18px;
    opacity: 0.8;
}

.metric-value {

    font-size: 36px;
    font-weight: bold;
    margin-top: 10px;
}

.section-header {

    font-size: 22px;
    font-weight: bold;
    margin-top: 30px;
    margin-bottom: 10px;
}

.insight-box {

    background-color: #1f2937;
    padding: 18px;
    border-radius: 12px;
    margin-top: 15px;
    color: white;
}

.stButton > button {

    width: 100%;
    height: 3.2em;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

.stTextInput > div > div > input {

    border-radius: 10px;
}

</style>

""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================

st.title("🎓 Academic Performance Analyzer")

st.caption(
    "AI-powered student performance evaluation system"
)

# ======================================================
# ACADEMIC METRICS
# ======================================================

st.markdown(
    '<div class="section-header">📘 Academic Metrics</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    study_hours = float(st.text_input(
        "Study Hours",
        "6"
    ))

    attendance_pct = int(st.text_input(
        "Attendance Percentage",
        "85"
    ))

    prev_gpa = float(st.text_input(
        "Previous GPA",
        "3.0"
    ))

with col2:

    assignments_done = int(st.text_input(
        "Assignments Completed Percentage",
        "80"
    ))

    quiz_avg = int(st.text_input(
        "Quiz Average",
        "75"
    ))

    midterm_score = int(st.text_input(
        "Midterm Score",
        "78"
    ))

# ======================================================
# LIFESTYLE METRICS
# ======================================================

st.markdown(
    '<div class="section-header">🌙 Lifestyle Metrics</div>',
    unsafe_allow_html=True
)

col3, col4 = st.columns(2)

with col3:

    sleep_hours = float(st.text_input(
        "Sleep Hours",
        "7"
    ))

    internet_hours = float(st.text_input(
        "Internet Usage Hours",
        "3"
    ))

with col4:

    extracurricular_label = st.selectbox(
        "Participates in Extracurricular Activities",
        [
            "No",
            "Yes"
        ]
    )

    extracurricular = 1 if extracurricular_label == "Yes" else 0

    part_time_job_label = st.selectbox(
        "Has Part-Time Job",
        [
            "No",
            "Yes"
        ]
    )

    part_time_job = 1 if part_time_job_label == "Yes" else 0

# ======================================================
# ENVIRONMENT METRICS
# ======================================================

st.markdown(
    '<div class="section-header">🏠 Environment Metrics</div>',
    unsafe_allow_html=True
)

col5, col6 = st.columns(2)

with col5:

    family_support_label = st.selectbox(
        "Family Support Level",
        [
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
    )

    family_support_map = {

        "Very Low": 1,
        "Low": 2,
        "Moderate": 3,
        "High": 4,
        "Very High": 5

    }

    family_support = family_support_map[family_support_label]

    parent_edu_label = st.selectbox(
        "Parent Education Level",
        [
            "No Formal Education",
            "High School",
            "Bachelor's Degree",
            "Postgraduate Degree"
        ]
    )

    parent_edu_map = {

        "No Formal Education": 0,
        "High School": 1,
        "Bachelor's Degree": 2,
        "Postgraduate Degree": 3

    }

    parent_edu = parent_edu_map[parent_edu_label]

with col6:

    school_type_label = st.selectbox(
        "School Type",
        [
            "Public School",
            "Private School"
        ]
    )

    school_type = 1 if school_type_label == "Private School" else 0

    distance_km = int(st.text_input(
        "Distance From School (km)",
        "5"
    ))

# ======================================================
# PREDICT BUTTON
# ======================================================

st.markdown("")

if st.button("Generate Academic Analysis"):

    try:

        payload = {

            "study_hours": study_hours,
            "attendance_pct": attendance_pct,
            "prev_gpa": prev_gpa,
            "assignments_done": assignments_done,
            "sleep_hours": sleep_hours,
            "internet_hours": internet_hours,
            "family_support": family_support,
            "part_time_job": part_time_job,
            "extracurricular": extracurricular,
            "parent_edu": parent_edu,
            "gender": 0,
            "school_type": school_type,
            "distance_km": distance_km,
            "quiz_avg": quiz_avg,
            "midterm_score": midterm_score

        }

        response = requests.post(

            API_URL,

            json=payload

        )

        result = response.json()

        prediction = result["prediction"]

        probabilities = result["probabilities"]

        confidence = round(
            max(probabilities.values()) * 100,
            2
        )

        # ==================================================
        # MAIN RESULT
        # ==================================================

        st.markdown(f"""

        <div class="metric-card">

            <div class="metric-title">
                Expected Academic Performance
            </div>

            <div class="metric-value">
                {prediction}
            </div>

        </div>

        """, unsafe_allow_html=True)

        st.success(f"Confidence Score: {confidence}%")

        # ==================================================
        # INSIGHTS
        # ==================================================

        strengths = []

        concerns = []

        if attendance_pct >= 85:
            strengths.append("Excellent attendance consistency")

        if study_hours >= 7:
            strengths.append("Strong study discipline")

        if assignments_done >= 85:
            strengths.append("High assignment completion rate")

        if prev_gpa >= 3.5:
            strengths.append("Strong historical academic performance")

        if sleep_hours < 5:
            concerns.append("Low sleep duration may affect focus")

        if internet_hours >= 6:
            concerns.append("High internet usage detected")

        if attendance_pct < 60:
            concerns.append("Low attendance may impact outcomes")

        if study_hours < 3:
            concerns.append("Insufficient study duration")

        st.markdown(
            '<div class="section-header">📈 Performance Insights</div>',
            unsafe_allow_html=True
        )

        if strengths:

            st.markdown("""

            <div class="insight-box">
            <b>Strengths</b><br><br>

            """ + "<br>".join(
                [f"✅ {s}" for s in strengths]
            ) + """

            </div>

            """, unsafe_allow_html=True)

        if concerns:

            st.markdown("""

            <div class="insight-box">
            <b>Risk Factors</b><br><br>

            """ + "<br>".join(
                [f"⚠️ {c}" for c in concerns]
            ) + """

            </div>

            """, unsafe_allow_html=True)

        recommendation = ""

        if prediction == "Distinction":

            recommendation = (
                "Maintain current consistency and continue balancing "
                "academic discipline with healthy routines."
            )

        elif prediction == "Pass":

            recommendation = (
                "Improving study consistency and attendance could "
                "significantly improve performance."
            )

        else:

            recommendation = (
                "Immediate focus on attendance, study habits, and "
                "structured academic support is recommended."
            )

        st.markdown(
            '<div class="section-header">🧠 Recommendation</div>',
            unsafe_allow_html=True
        )

        st.info(recommendation)

        with st.expander("View Detailed Prediction Probabilities"):

            st.json(probabilities)

    except ValueError:

        st.error(
            "Please enter valid numeric values in all input fields."
        )

    except Exception as e:

        st.error(f"Error connecting to API: {e}")