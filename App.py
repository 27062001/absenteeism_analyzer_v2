# app.py
# Absenteeism Anomaly Detector – uses fpdf2 for reliable plot embedding

from __future__ import annotations

import os
import io
import datetime as dt
from email.message import EmailMessage
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from fpdf import FPDF  # Requires fpdf2: pip install fpdf2
import smtplib

# =============================================================================
# Helpers
# =============================================================================

def to_latin1(s: object) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")

def find_target_column(df: pd.DataFrame, needle: str) -> str | None:
    needle_lower = needle.lower()
    for col in df.columns:
        if needle_lower in col.lower():
            return col
    return None

def run_model(model_name: str, df: pd.DataFrame, target_col: str, contamination: float) -> pd.DataFrame:
    df_clean = df.dropna()
    features = df_clean.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")

    if features.empty:
        raise ValueError("No numeric features found to train on (after dropping target).")

    if model_name == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(features)
        pred = model.predict(features)
        score = model.decision_function(features)
    elif model_name == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        pred = model.fit_predict(features)
        score = model.negative_outlier_factor_
    elif model_name == "One-Class SVM":
        model = OneClassSVM(nu=contamination, gamma="auto")
        model.fit(features)
        pred = model.predict(features)
        score = model.decision_function(features)
    else:
        raise ValueError("Invalid model selected")

    result = df_clean.copy()
    result["Anomaly"] = pred
    result["Score"] = score
    result["Label"] = np.where(pred == -1, "Anomaly", "Normal")
    return result

def save_plot(fig):
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        fig.savefig(tmp.name, format="jpg", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return tmp.name

def generate_pdf(df: pd.DataFrame, model_name: str) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=16)
        unicode_ok = True
    except Exception:
        pdf.set_font("Arial", "B", 16)
        unicode_ok = False

    def safe(txt: str) -> str:
        return txt if unicode_ok else to_latin1(txt)

    pdf.cell(0, 10, safe("Absenteeism Anomaly Report"), ln=True, align="C")
    pdf.set_font("DejaVu" if unicode_ok else "Arial", size=12)
    pdf.cell(0, 8, safe(f"Generated on: {dt.datetime.now():%Y-%m-%d %H:%M}"), ln=True)
    pdf.cell(0, 8, safe(f"Model used: {model_name}"), ln=True)
    pdf.cell(0, 8, safe(f"Total records: {len(df)}"), ln=True)
    pdf.cell(0, 8, safe(f"Detected anomalies: {int((df['Anomaly'] == -1).sum())}"), ln=True)
    pdf.ln(6)

    pdf.multi_cell(0, 10, safe(
        "This report highlights employees with abnormal absenteeism patterns using the selected AI model.\n"
        "Visualizations and explanations below assist non-technical users."
    ))
    pdf.ln(4)

    plots = []
    target_col = find_target_column(df, "Absenteeism time in hours")

    fig, ax = plt.subplots()
    sns.histplot(df[target_col], bins=30, kde=True, ax=ax)
    ax.set_title("Absenteeism Hours Distribution")
    plots.append((fig, "Distribution of absenteeism hours. Helps identify normal vs abnormal behavior."))

    workload_col = [c for c in df.columns if "work load" in c.lower()]
    if workload_col:
        fig2, ax2 = plt.subplots()
        colors = df["Anomaly"].map({1: "green", -1: "red"})
        ax2.scatter(df[workload_col[0]], df[target_col], c=colors)
        ax2.set_xlabel(workload_col[0])
        ax2.set_ylabel(target_col)
        ax2.set_title("Workload vs Absenteeism")
        plots.append((fig2, "Relationship between workload and absenteeism. Red dots show outliers."))

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, ax=ax3, cmap="coolwarm", annot=False)
    ax3.set_title("Correlation Heatmap")
    plots.append((fig3, "Shows how different features relate. Helps understand what influences absenteeism."))

    if "Month of absence" in df.columns:
        fig4, ax4 = plt.subplots()
        sns.boxplot(x=df["Month of absence"], y=df[target_col], ax=ax4)
        ax4.set_xlabel("Month")
        ax4.set_ylabel(target_col)
        ax4.set_title("Monthly Absenteeism Distribution")
        plots.append((fig4, "Monthly trends in absenteeism. Helps spot seasonality or time-based spikes."))

    for fig, caption in plots:
        img_path = save_plot(fig)
        pdf.add_page()
        pdf.set_font("DejaVu" if unicode_ok else "Arial", size=12)
        pdf.multi_cell(0, 10, safe(caption))
        try:
            pdf.image(img_path, x=10, w=190)
        except Exception as e:
            pdf.set_text_color(255, 0, 0)
            pdf.multi_cell(0, 10, safe(f"[Image embed failed: {e} - {img_path}]."))
            pdf.set_text_color(0, 0, 0)
        finally:
            os.remove(img_path)

    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, safe("Top 10 Anomalies (by Score)"), ln=True)
    pdf.set_font("DejaVu" if unicode_ok else "Arial", size=10)
    pdf.cell(30, 8, safe("Index"), 1)
    pdf.cell(40, 8, safe("Score"), 1)
    pdf.cell(40, 8, safe("Label"), 1)
    pdf.ln()

    top = df[df["Anomaly"] == -1].head(10)
    for idx, row in top.iterrows():
        pdf.cell(30, 8, safe(str(idx)), 1)
        pdf.cell(40, 8, safe(f"{row.get('Score', 0.0):.4f}"), 1)
        pdf.cell(40, 8, safe(row.get("Label", "")), 1)
        pdf.ln()

    out = pdf.output(dest="S")
    return out.encode("latin-1", "ignore") if isinstance(out, str) else bytes(out)

def send_email(to_address: str, subject: str, message: str, attachment_data: bytes, filename: str) -> bool:
    try:
        email_user = st.session_state.get("email_user") or os.getenv("EMAIL_USER")
        email_pass = st.session_state.get("email_pass") or os.getenv("EMAIL_PASS")

        if not email_user or not email_pass:
            st.error("❌ Email credentials not configured. Please set up your email in the Email Configuration section.")
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = email_user
        msg["To"] = to_address
        msg.set_content(message)
        msg.add_attachment(attachment_data, maintype="application", subtype="pdf", filename=filename)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(email_user, email_pass)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"❌ Email failed: {e}")
        return False

# =============================================================================
# UI
# =============================================================================

st.set_page_config(page_title="Absenteeism Anomaly Detector", layout="wide")

st.title("📊 Absenteeism Anomaly Detector")
st.caption("Detect unusual absentee patterns with AI models. Upload → Analyze → Export")

tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Settings", "📈 Analysis", "⬇️ Download Report", "ℹ️ About"])

# Session defaults
st.session_state.setdefault("scored_df", None)
st.session_state.setdefault("uploaded_file", None)
st.session_state.setdefault("pdf_bytes", None)

# ---------------------------------- TAB 1 ------------------------------------
with tab1:
    st.header("📤 Upload Your Data")
    uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"], key="fileUploader")

    model_choice = st.selectbox("Choose Detection Model", ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"])
    contamination = st.slider("Contamination (% of outliers expected)", 0.01, 0.3, 0.1, 0.01)
    col1, col2 = st.columns([1, 1])
    run_button = col1.button("🚀 Run Detection")
    reset_button = col2.button("❌ Reset Upload")

    if reset_button:
        st.session_state.scored_df = None
        st.session_state.uploaded_file = None
        st.session_state.pdf_bytes = None
        st.rerun()

# ---------------------------------- TAB 2 ------------------------------------
with tab2:
    st.header("📊 Visualize Anomalies")

    if run_button and uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)

            # Ensure unique column names
            seen: dict[str, int] = {}
            new_cols: list[str] = []
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_cols.append(f"{col}.{seen[col]}")
                else:
                    seen[col] = 0
                    new_cols.append(col)
            df.columns = new_cols

            target_col = find_target_column(df, "Absenteeism time in hours")
            if target_col is None:
                st.error("❌ Could not find a column containing 'Absenteeism time in hours'.")
                st.stop()

            result_df = run_model(model_choice, df, target_col, contamination)
            st.session_state.scored_df = result_df

            st.success(f"✅ Detection completed using {model_choice}")

            st.subheader("📌 Summary Stats")
            st.dataframe(result_df.describe().T)

            st.subheader("📉 Absenteeism Distribution")
            fig, ax = plt.subplots()
            sns.histplot(result_df[target_col], bins=30, kde=True, ax=ax)
            ax.set_title("Absenteeism Hours")
            st.pyplot(fig)

            st.subheader("🚨 Top 20 Anomalies")
            anoms = result_df[result_df["Anomaly"] == -1].nlargest(20, target_col)
            st.dataframe(anoms)

            workload_col = [c for c in result_df.columns if "work load" in c.lower()]
            if workload_col:
                st.subheader("⚙️ Workload vs Absenteeism (Anomalies in Red)")
                fig2, ax2 = plt.subplots()
                colors = result_df["Anomaly"].map({1: "green", -1: "red"})
                ax2.scatter(result_df[workload_col[0]], result_df[target_col], c=colors)
                ax2.set_xlabel(workload_col[0])
                ax2.set_ylabel(target_col)
                ax2.set_title("Workload vs Absenteeism")
                st.pyplot(fig2)

            st.subheader("🔍 Feature Correlation Heatmap")
            corr = result_df.select_dtypes(include="number").corr(numeric_only=True)
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax3)
            ax3.set_title("Correlation Heatmap")
            st.pyplot(fig3)

            if "Month of absence" in result_df.columns:
                st.subheader("📆 Time Trend: Absenteeism Over Months")
                fig4, ax4 = plt.subplots()
                sns.boxplot(x=result_df["Month of absence"], y=result_df[target_col], ax=ax4)
                ax4.set_xlabel("Month")
                ax4.set_ylabel(target_col)
                ax4.set_title("Monthly Absenteeism Distribution")
                st.pyplot(fig4)

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    elif not uploaded_file:
        st.info("Please upload a file and run detection from the Upload tab.")

# ---------------------------------- TAB 3 ------------------------------------
with tab3:
    st.header("⬇️ Export Results")

    if st.session_state.scored_df is not None:
        to_download = st.session_state.scored_df.copy()

        # Excel
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                to_download.to_excel(writer, index=False, sheet_name="Anomaly Results")
        except Exception:
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                to_download.to_excel(writer, index=False, sheet_name="Anomaly Results")

        st.download_button(
            label="📥 Download Excel Report",
            data=output.getvalue(),
            file_name="anomaly_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Generate PDF
        if st.button("📄 Generate PDF Summary"):
            try:
                pdf_bytes = generate_pdf(to_download, model_choice)
                st.session_state.pdf_bytes = pdf_bytes
                st.success("✅ PDF generated successfully!")
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=f"anomaly_summary_{dt.date.today()}.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"❌ Error generating PDF: {e}")
        elif st.session_state.get("pdf_bytes"):
            st.download_button(
                label="📥 Download PDF",
                data=st.session_state.pdf_bytes,
                file_name=f"anomaly_summary_{dt.date.today()}.pdf",
                mime="application/pdf",
            )

        st.subheader("📧 Send Report via Email")

        # Email configuration
        with st.expander("🔧 Email Configuration"):
            st.markdown(
                """
                **Quick Email Setup:**

                Enter your Gmail credentials below to enable email functionality.\
                You need a Gmail *App Password* (not your normal password):

                1. Go to Google Account Settings → Security
                2. Enable 2-factor authentication
                3. Generate an App Password for this application
                4. Paste that 16-character password below
                """
            )

            email_user_in = st.text_input("Gmail Address", value=os.getenv("EMAIL_USER", ""))
            email_pass_in = st.text_input("Gmail App Password", value=os.getenv("EMAIL_PASS", ""), type="password")

            if st.button("💾 Save Email Settings"):
                st.session_state.email_user = email_user_in
                st.session_state.email_pass = email_pass_in
                st.success("✅ Email settings saved for this session!")

        current_email_user = st.session_state.get("email_user") or os.getenv("EMAIL_USER")
        current_email_pass = st.session_state.get("email_pass") or os.getenv("EMAIL_PASS")

        send_btn = False
        demo_btn = False
        to_addr = ""
        subject = ""
        body = ""

        if current_email_user and current_email_pass:
            with st.form("email_form"):
                to_addr = st.text_input("Recipient email")
                subject = st.text_input("Subject", "Absenteeism Anomaly Report")
                body = st.text_area("Message", "Please find attached the absenteeism anomaly report.")
                col1, col2 = st.columns([1, 1])
                send_btn = col1.form_submit_button("📤 Send Email")
                demo_btn = col2.form_submit_button("🎭 Demo Mode")
        else:
            st.info("⚠️ Configure email settings above to enable sending.")
            st.subheader("🎭 Demo Mode")
            st.info("Try the email feature with demo data:")
            with st.form("demo_email_form"):
                demo_to = st.text_input("Demo Recipient", "demo@example.com")
                demo_subject = st.text_input("Demo Subject", "Absenteeism Anomaly Report")
                demo_body = st.text_area("Demo Message", "This is a demo of the email functionality.")
                demo_send = st.form_submit_button("🎭 Send Demo Email")
                if demo_send:
                    st.success("🎉 Demo email would be sent!")
                    st.info("In a real scenario, this would send the report to the specified email address.")

        if send_btn:
            if not st.session_state.get("pdf_bytes"):
                st.warning("Generate the PDF first.")
            elif not to_addr:
                st.warning("Please enter a recipient email.")
            else:
                ok = send_email(
                    to_addr,
                    subject,
                    body,
                    st.session_state.pdf_bytes,
                    f"anomaly_summary_{dt.date.today()}.pdf",
                )
                if ok:
                    st.success("✅ Email sent!")

        if demo_btn:
            st.success("🎉 Demo email would be sent!")
            st.info("In a real scenario, this would send the report to the specified email address.")
            st.json(
                {
                    "to": to_addr,
                    "subject": subject,
                    "message": body,
                    "attachment": f"anomaly_summary_{dt.date.today()}.pdf",
                }
            )

    else:
        st.info("No results available. Please run detection first.")

# ---------------------------------- TAB 4 ------------------------------------
with tab4:
    st.header("ℹ️ About This App")
    st.markdown(
        """
This app helps detect anomalies in employee absenteeism data using AI models like:

- **Isolation Forest** (fast, good for general use)
- **Local Outlier Factor (LOF)** (better for local clusters)
- **One-Class SVM** (sensitive, good for small datasets)

### How it works:
1. Upload your Excel sheet with a column like `Absenteeism time in hours`
2. Choose a model
3. Get insights and download the results!

Built using Streamlit and scikit-learn.

---

👨‍💻 Developed by YOU  
🎓 Perfect for HR, data science beginners, or analytics teams
"""
    )
