import streamlit as st
import numpy as np
import pandas as pd
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.metrics import roc_curve, auc

AUDIT_FILE = "audit_log.csv"
FILE_NAME = "employee_data.csv"
st.title("Login")
role = st.selectbox("Login as", ["Employee", "HR"])
password = st.text_input("Password", type="password")

if role == "HR" and password != "hr123":
    st.error("Unauthorized HR access")
    st.stop()
def backup_file(file_path):
    if os.path.exists(file_path):
        os.makedirs("backups", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backups/{timestamp}_{os.path.basename(file_path)}"
        shutil.copy(file_path, backup_name)

def generate_pdf_report(df):
    file_name = "Work_Life_Compliance_Report.pdf"
    c = canvas.Canvas(file_name, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Work-Life Balance Compliance Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 90, f"Total Employees: {len(df)}")
    c.drawString(50, height - 110, f"Avg Work Hours: {round(df['WorkHours'].mean(),1)}")
    c.drawString(50, height - 130, f"Avg Stress Level: {round(df['Stress'].mean(),1)}")

    violations = df[df["WorkHours"] > 48]
    c.drawString(50, height - 170, f"Legal Violations (>48 hrs/week): {len(violations)}")

    c.drawString(50, height - 210, "Generated for HR Compliance Audit")

    c.save()
    return file_name
def export_excel(df):
    file_name = "Employee_WorkLife_Report.csv"
    df.to_csv(file_name, index=False)
    return file_name

# ---------- LOAD KAGGLE DATA ----------
@st.cache_data
def load_kaggle():
    return pd.read_csv("synthetic_employee_burnout.csv")

kaggle_df = load_kaggle()
kaggle_df = kaggle_df[
    ["WorkHoursPerWeek", "StressLevel", "SatisfactionLevel", "Burnout"]
]

# -------- ADD REALISTIC NOISE (IMPORTANT) --------
# Feature noise
kaggle_df["StressLevel"] = kaggle_df["StressLevel"] + np.random.normal(
    0, 2.5, size=len(kaggle_df)
)

kaggle_df["SatisfactionLevel"] = kaggle_df["SatisfactionLevel"] + np.random.normal(
    0, 2.0, size=len(kaggle_df)
)

# Clip to valid human ranges
kaggle_df["StressLevel"] = kaggle_df["StressLevel"].clip(1, 10)
kaggle_df["SatisfactionLevel"] = kaggle_df["SatisfactionLevel"].clip(1, 10)

# -------- ADD LABEL NOISE (CRUCIAL) --------
flip_mask = np.random.rand(len(kaggle_df)) < 0.15   # 15% mislabels
kaggle_df.loc[flip_mask, "Burnout"] = 1 - kaggle_df.loc[flip_mask, "Burnout"]

# keep values in valid range
kaggle_df["StressLevel"] = kaggle_df["StressLevel"].clip(1, 10)
avg_hours = kaggle_df["WorkHoursPerWeek"].mean()

# ---------- TRAIN ML MODEL ----------
X = kaggle_df[["WorkHoursPerWeek", "StressLevel", "SatisfactionLevel"]]
y = kaggle_df["Burnout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,          # ↑ slightly harder task
    random_state=42,
    stratify=y               # IMPORTANT for imbalance
)

@st.cache_resource
def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.6                  # regularization (reduces overfitting)
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Predictions
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Metrics
model_accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)


# ---------- UI ----------
if role == "Employee":

    st.title("Work-Life Balance Data Collection")

    emp_id = st.text_input("Employee ID")
    name = st.text_input("Employee Name")

    department = st.selectbox(
        "Department",
        ["IT", "HR", "Finance", "Operations", "Marketing", "Other"]
    )
    job_role = st.text_input("Job Role")

    work_hours = st.number_input("Work hours per week", min_value=0, max_value=100)
    leaves = st.number_input("Leaves taken this month", min_value=0, max_value=31)
    stress = st.slider("Stress level (1 = low, 10 = high)", 1, 10)
    productivity = st.slider("Productivity level (1 = low, 10 = high)", 1, 10)
    health = st.selectbox("Health status", ["Good", "Okay", "Bad"])

    st.subheader("Employee Feedback")
    job_satisfaction = st.slider(
        "Job Satisfaction (1 = very dissatisfied, 10 = very satisfied)", 1, 10
    )

    burnout_feeling = st.selectbox(
        "Do you feel burned out?",
        ["No", "Sometimes", "Often"]
    )

    feedback_text = st.text_area(
        "Additional feedback about your work-life balance"
    )

    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # ---------- LOAD EXISTING DATA ----------
    if os.path.exists(FILE_NAME):
        df_existing = pd.read_csv(FILE_NAME)
        df_existing["EmployeeID"] = df_existing["EmployeeID"].astype(str)
    else:
        df_existing = pd.DataFrame(
            columns=[
                "EmployeeID","Name","Department","JobRole",
                "WorkHours","Leaves","Stress","Health",
                "Productivity","Timestamp",
                "JobSatisfaction","BurnoutFeeling","Feedback"
            ]
        )

    # ---------- SUBMIT ----------
    if st.button("Submit Data"):
        # ---------- BASIC INPUT VALIDATION ----------
        if work_hours < 10 or work_hours > 80:
            st.error("Work hours must be between 10 and 80 per week")
            st.stop()

        if stress <= 2 and burnout_feeling == "Often":
            st.error("Burnout feeling conflicts with very low stress level")
            st.stop()

        if productivity >= 9 and work_hours >= 70:
            st.error("High productivity with extreme work hours is unrealistic")
            st.stop()


        if emp_id == "" or name == "":
            st.error("Employee ID and Name are required")
        else:
            new_row = {
                "EmployeeID": emp_id,
                "Name": name,
                "Department": department,
                "JobRole": job_role,
                "WorkHours": work_hours,
                "Leaves": leaves,
                "Stress": stress,
                "Health": health,
                "Productivity": productivity,
                "Timestamp": date_time,
                "JobSatisfaction": job_satisfaction,
                "BurnoutFeeling": burnout_feeling,
                "Feedback": feedback_text,
            }

            df_new = pd.DataFrame([new_row])

            if emp_id in df_existing["EmployeeID"].values:
                df_existing.loc[
                    df_existing["EmployeeID"] == emp_id, df_new.columns
                ] = df_new.iloc[0].values
                df = df_existing
                action = "UPDATE"
                st.warning("Employee ID exists. Data updated.")
            else:
                df = pd.concat([df_existing, df_new], ignore_index=True)
                action = "CREATE"


            df.to_csv(FILE_NAME, index=False)
            audit_entry = pd.DataFrame([{
                "EmployeeID": emp_id,
                "Role": role,
                "Action": action,
                "Timestamp": date_time
            }])

            audit_entry.to_csv(
                AUDIT_FILE,
                mode="a",
                header=not os.path.exists(AUDIT_FILE),
                index=False
            )
            backup_file(FILE_NAME)
            backup_file(AUDIT_FILE)


            st.success("Employee data saved successfully")
            st.dataframe(df.tail(5))

            # ---------- ML BURNOUT ----------
            st.subheader("ML Burnout Prediction")
            st.info(
                """
                 **Important Disclaimer**
                
                This burnout prediction is generated using a machine learning model 
                trained on synthetic employee data.  
                The output is for **decision support only** and **not a medical or psychological diagnosis**.
                
                Final decisions should involve HR review and professional judgment.
                """
            )

            input_df = pd.DataFrame(
                [[work_hours, stress, job_satisfaction]],
                columns=["WorkHoursPerWeek", "StressLevel", "SatisfactionLevel"]
            )

            ml_burnout_prob = model.predict_proba(input_df)[0][1] * 100

            if ml_burnout_prob >= 70:
                st.error(f"High Burnout Risk (ML): {ml_burnout_prob:.1f}%")
            else:
                st.success(f"Low Burnout Risk (ML): {ml_burnout_prob:.1f}%")

            st.caption(f"Model Accuracy: {model_accuracy*100:.2f}%")
            # ---------- ML EXPLAINABILITY ----------
            st.subheader("Why this burnout risk?")

            reasons = []

            if work_hours > avg_hours:
                reasons.append("Work hours are higher than average")
            if stress >= 7:
                reasons.append("High stress level reported")
            if job_satisfaction <= 4:
                reasons.append("Low job satisfaction")
            if burnout_feeling == "Often":
                reasons.append("Employee frequently feels burned out")

            if reasons:
                for r in reasons:
                    st.write("•", r)
            else:
                st.write("No major risk factors detected based on current inputs.")

            # ---------- HEURISTIC BURNOUT SCORE ----------
            heuristic_burnout_prob = 0

            if work_hours > avg_hours:
                heuristic_burnout_prob += 30
            if stress >= 7:
                heuristic_burnout_prob += 40
            if job_satisfaction <= 4:
                heuristic_burnout_prob += 20
            if burnout_feeling == "Often":
                heuristic_burnout_prob += 30

            heuristic_burnout_prob = min(heuristic_burnout_prob, 100)

            st.metric("Burnout Probability (Heuristic)", heuristic_burnout_prob)

            # ---------- MENTAL HEALTH RESOURCES & ESCALATION ----------
            st.subheader(" Mental Health Support & Escalation")

            # CRITICAL CONDITION CHECK
            critical_risk = False

            if heuristic_burnout_prob >= 75 or stress >= 9:
                critical_risk = True


            if critical_risk:
                st.error("CRITICAL MENTAL HEALTH RISK DETECTED")

                st.markdown(
                    """
                    **Immediate Actions Required:**
                    - Notify HR & Direct Manager
                    - Schedule mandatory mental health consultation
                    - Temporary workload reduction
                    - Encourage mental health leave
                    """
                )

                st.info(
                    """
                    **Mental Health Resources:**
                    - Company Mental Health Counselor
                    - Employee Assistance Program (EAP)
                    - Confidential Psychological Support
                    """
                )

            elif heuristic_burnout_prob >= 50 or ml_burnout_prob >= 50:
                st.warning("MODERATE MENTAL HEALTH RISK")

                st.markdown(
                    """
                    **Recommended Actions:**
                    - Optional counseling sessions
                    - Stress management workshops
                    - Regular check-ins with manager
                    """
                )

            else:
                st.success("Mental Health Risk Level: LOW")

                st.markdown(
                    """
                    **Preventive Resources:**
                    - Wellness webinars
                    - Mindfulness programs
                    - Work-life balance coaching
                    """
                )


            # ---------- REAL-TIME ALERTS ----------
            st.subheader("Real-Time Alerts")

            personal_time = 168 - work_hours
            alerts = False

            if work_hours > 60:
                st.error("Overwork detected (>60 hrs/week)")
                alerts = True
            if stress >= 8:
                st.error("Critical stress level")
                alerts = True
            if personal_time < 90:
                st.warning("Very low personal time")
                alerts = True
            if ml_burnout_prob >= 70:
                st.error("High ML burnout risk")
                alerts = True

            if job_satisfaction <= 3:
                st.warning("Low job satisfaction")
                alerts = True


            if not alerts:
                st.success("No critical alerts detected")
            # ---------- WELLNESS & SUPPORT RECOMMENDATIONS ----------
            st.subheader("Wellness Programs & Support")

            recommendations = []

            # Overwork
            if work_hours > 55:
                recommendations.append(
                    "Enroll in Time Management & Workload Optimization Program"
                )

            # High stress
            if stress >= 7:
                recommendations.append(
                    "Mandatory Stress Management & Mindfulness Sessions"
                )

            # Burnout risk
            if heuristic_burnout_prob >= 60 or ml_burnout_prob >= 60:
                recommendations.append(
                    "Burnout Recovery Program (Reduced workload + mental health support)"
                )

            # Low job satisfaction
            if job_satisfaction <= 4:
                recommendations.append(
                    "One-on-one Career Counseling & Role Alignment Session"
                )

            # Poor health
            if health == "Bad":
                recommendations.append(
                    "Company Health Checkup & Wellness Leave Recommendation"
                )

            # Low productivity
            if productivity <= 4:
                recommendations.append(
                    "Productivity Coaching & Skill Enhancement Program"
                )

            if recommendations:
                for r in recommendations:
                    st.info(r)
            else:
                st.success("No wellness interventions required at this time")

# ================= HR DASHBOARD =================
if role == "HR":
    # ---------- AUDIT LOG VIEWER ----------
    st.subheader("System Audit Log")

    if os.path.exists(AUDIT_FILE):
        audit_df = pd.read_csv(AUDIT_FILE)

        # Optional: sort latest first
        audit_df["Timestamp"] = pd.to_datetime(audit_df["Timestamp"])
        audit_df = audit_df.sort_values("Timestamp", ascending=False)

        st.dataframe(audit_df)

    else:
        st.info("No audit logs found yet.")


    st.divider()
    st.title(" HR Analytics Dashboard")

    if os.path.exists(FILE_NAME):
        hr_df = pd.read_csv(FILE_NAME)
    else:
        st.warning("No data available yet.")
        st.stop()

    selected_dept = st.selectbox(
        "Filter by Department",
        ["All"] + sorted(hr_df["Department"].dropna().unique())
    )

    if selected_dept != "All":
        hr_df = hr_df[hr_df["Department"] == selected_dept]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Employees", len(hr_df))
    col2.metric("Avg Work Hours", round(hr_df["WorkHours"].mean(), 1))
    col3.metric("Avg Stress", round(hr_df["Stress"].mean(), 1))
    col4.metric("Avg Productivity", round(hr_df["Productivity"].mean(), 1))

    st.subheader("Department-wise Metrics")
    st.bar_chart(
        hr_df.groupby("Department")[["WorkHours","Stress","Productivity"]].mean()
    )

    st.subheader("Burnout & Satisfaction Distribution")
    st.bar_chart(hr_df["BurnoutFeeling"].value_counts())
    st.bar_chart(hr_df["JobSatisfaction"].value_counts().sort_index())

    st.subheader("Historical Trends")
    hr_df["Timestamp"] = pd.to_datetime(hr_df["Timestamp"])
    st.line_chart(
        hr_df.sort_values("Timestamp")
        .set_index("Timestamp")[["WorkHours","Stress","Productivity"]]
    )
    if st.button("Export Compliance Report (PDF)"):
        pdf = generate_pdf_report(hr_df)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF Report", f, file_name=pdf)


    if st.button("Export CSV Report"):
        excel = export_excel(hr_df)
        with open(excel, "rb") as f:
            st.download_button("Download Excel File", f, file_name=excel)
    st.subheader("Compliance Summary")

    legal_limit = 48
    violations_df = hr_df[hr_df["WorkHours"] > legal_limit]

    st.metric("Legal Work Hour Violations", len(violations_df))
    st.metric(
        "Compliance Rate",
        f"{round((1 - len(violations_df)/len(hr_df)) * 100, 2)}%"
    )

    if len(violations_df) > 0:
        st.error(" Work-hour compliance violations detected")
        st.dataframe(violations_df[["EmployeeID","Name","WorkHours"]])
    else:
        st.success("✔ Organization is compliant with work-hour laws")
    st.subheader("HR Compliance Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Overworked Employees",
        len(hr_df[hr_df["WorkHours"] > 48])
    )

    col2.metric(
        "High Stress Employees",
        len(hr_df[hr_df["Stress"] >= 8])
    )

    col3.metric(
        "High Burnout Risk",
        len(hr_df[hr_df["BurnoutFeeling"] == "Often"])
    )
    st.subheader("Burnout Distribution (Pie Chart)")

    burnout_counts = hr_df["BurnoutFeeling"].value_counts()

    st.pyplot(
        burnout_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            ylabel=""
        ).figure
    )

    st.subheader("Confusion Matrix (Burnout Prediction)")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")

    corr_df = hr_df[
        ["WorkHours", "Stress", "Productivity", "JobSatisfaction"]
    ].corr()

    fig, ax = plt.subplots()
    sns.heatmap(
        corr_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )
    st.pyplot(fig)
    st.subheader("AUC-ROC Curve")

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Burnout Prediction")
    ax.legend()
    st.pyplot(fig)

