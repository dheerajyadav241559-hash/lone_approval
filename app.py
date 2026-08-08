import os
import joblib
import pandas as pd
import gradio as gr

# ==========================================================
# 1. Load Dataset & Trained Model
# ==========================================================
DATASET_PATH = "loan_approval_dataset.csv"

# Load Dataset for UI analytics and column alignment
try:
    df_raw = pd.read_csv(DATASET_PATH)
    # Clean up column names (strip leading/trailing whitespaces if present)
    df_raw.columns = df_raw.columns.str.strip()
except Exception as e:
    print(f"Warning: Dataset not found or error loading. {e}")
    df_raw = None

# Load Model
try:
    deployed_rf = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_rf = None

# ==========================================================
# 2. Diagnostic Engine (Why was it rejected?)
# ==========================================================
def analyze_rejection(income, loan_amount, loan_term, cibil, assets):
    """Provides financial reasoning for why a loan might be rejected."""
    reasons = []
    
    # 1. Credit Check
    if cibil < 650:
        reasons.append("📉 **Low CIBIL Score:** Scores below 650 indicate higher credit risk.")
        
    # 2. Debt-to-Income (DTI) Check
    term_months = loan_term * 12 if loan_term < 50 else loan_term 
    if term_months > 0 and income > 0:
        monthly_income = income / 12
        estimated_emi = (loan_amount * 1.10) / term_months
        dti_ratio = (estimated_emi / monthly_income) * 100
        
        if dti_ratio > 40:
            reasons.append(
                f"⚖️ **High Debt-to-Income Ratio:** The estimated EMI takes up {dti_ratio:.1f}% "
                f"of monthly income (lenders typically prefer < 40%)."
            )
    elif income <= 0:
        reasons.append("💵 **Invalid Income:** Income must be greater than zero.")
            
    # 3. Collateral/Asset Check
    total_assets = sum(assets)
    if total_assets < (loan_amount * 0.3):
        reasons.append("🏠 **Insufficient Assets:** Total assets fall below 30% of the requested loan amount.")
        
    if not reasons:
        reasons.append("🤖 **Complex Risk Pattern:** The AI model identified patterns in historical data that correlate this profile with default risk.")
        
    return "\n".join(reasons)

# ==========================================================
# 3. Main Prediction Function
# ==========================================================
def predict_loan_status(
    dependents, education, self_employed, income, 
    loan_amount, loan_term, cibil, res_asset, 
    com_asset, lux_asset, bank_asset
):
    # Input Validation
    values = [dependents, education, self_employed, income, loan_amount, loan_term, cibil, res_asset, com_asset, lux_asset, bank_asset]
    
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill in all the input fields.", gr.update(visible=False)

    if cibil < 300 or cibil > 900:
        return "❌ CIBIL score must be between 300 and 900.", gr.update(visible=False)

    if deployed_rf is None:
        return "❌ Model failed to load. Please check your .pkl file.", gr.update(visible=False)

    try:
        # Standard column layout matching 'loan_approval_dataset.csv'
        columns = [
            'no_of_dependents', 'education', 'self_employed', 'income_annum',
            'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
        ]
        
        # Build DataFrame with exact feature structure
        input_data = pd.DataFrame([[
            int(dependents), int(education), int(self_employed), float(income),
            float(loan_amount), float(loan_term), float(cibil), float(res_asset),
            float(com_asset), float(lux_asset), float(bank_asset)
        ]], columns=columns)
        
        # Predict
        prediction = deployed_rf.predict(input_data)[0]

        if prediction == 1:
            return (
                "## 🟢 Status: APPROVED\n\n**Congratulations!** The applicant meets the criteria for this loan.", 
                gr.update(visible=False)
            )
        else:
            # Diagnostics
            assets = [float(res_asset), float(com_asset), float(lux_asset), float(bank_asset)]
            explanation = analyze_rejection(float(income), float(loan_amount), float(loan_term), float(cibil), assets)
            
            result_text = "## 🔴 Status: REJECTED\n\nUnfortunately, the applicant does not meet the criteria at this time."
            return result_text, gr.update(visible=True, value=explanation)

    except Exception as e:
        return f"❌ Prediction failed. Error: {str(e)}", gr.update(visible=False)

# ==========================================================
# 4. AI Advisor Agent Logic
# ==========================================================
def chat_with_agent(message, history):
    msg = message.lower()
    if "cibil" in msg or "score" in msg:
        return "To improve your CIBIL score quickly: pay outstanding credit card balances, never miss an EMI, and avoid taking on new debt for the next 6 months."
    elif "income" in msg or "dti" in msg or "debt" in msg:
        return "If your Debt-to-Income ratio is too high, try applying for a lower loan amount or increasing the loan term (duration) to reduce your monthly EMI."
    elif "asset" in msg or "collateral" in msg:
        return "Lenders look at your assets (residential, commercial, bank balances) as collateral. Having assets that total at least 30-50% of your loan amount significantly improves approval odds."
    elif "reject" in msg:
        return "Loans are typically rejected due to low CIBIL scores (under 650), a high Debt-to-Income ratio, or insufficient assets to back the loan amount."
    else:
        return "I am your AI Loan Advisor! Ask me how to improve your approval chances, how CIBIL works, or what factors lenders look at."

# ==========================================================
# 5. Build the UI with Gradio Blocks
# ==========================================================
custom_theme = gr.themes.Soft(
    primary_hue="blue", 
    secondary_hue="indigo",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(title="Loan Prediction System", theme=custom_theme) as app:
    # Header
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1>🏦 Intelligent Loan Approval Dashboard</h1>
            <p style="font-size: 1.1rem; color: #555;">Assess loan applications using Machine Learning and explore underlying dataset trends.</p>
        </div>
        """
    )
    
    with gr.Tabs():
        # --- TAB 1: PREDICTION DASHBOARD ---
        with gr.Tab("📊 Application Assessment"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 👤 Applicant Details")
                    with gr.Group():
                        dependents = gr.Number(label="Number of Dependents", value=0)
                        with gr.Row():
                            education = gr.Dropdown(choices=[("Graduate", 1), ("Not Graduate", 0)], label="Education", value=1)
                            self_employed = gr.Dropdown(choices=[("Yes", 1), ("No", 0)], label="Self Employed", value=0)
                    
                    gr.Markdown("### 💰 Loan & Financials")
                    with gr.Group():
                        income = gr.Number(label="Annual Income (₹/$)", value=500000)
                        loan_amount = gr.Number(label="Loan Amount Requested", value=1000000)
                        loan_term = gr.Slider(minimum=1, maximum=30, step=1, label="Loan Term (Years)", value=5)
                        cibil = gr.Slider(minimum=300, maximum=900, step=1, label="CIBIL Score", value=750)

                with gr.Column(scale=1):
                    gr.Markdown("### 🏢 Asset Valuation")
                    with gr.Group():
                        res_asset = gr.Number(label="Residential Assets Value", value=0)
                        com_asset = gr.Number(label="Commercial Assets Value", value=0)
                        lux_asset = gr.Number(label="Luxury Assets Value", value=0)
                        bank_asset = gr.Number(label="Bank Asset Value", value=0)
                    
                    gr.HTML("<br>")
                    submit_btn = gr.Button("Evaluate Application", variant="primary", size="lg")
            
            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    result_box = gr.Markdown("### 📋 Awaiting Submission...")
                    rejection_box = gr.Textbox(
                        label="Diagnostic Engine: Reason for Rejection", 
                        lines=4, 
                        visible=False, 
                        interactive=False,
                        elem_id="rejection-box"
                    )

            inputs = [
                dependents, education, self_employed, income, 
                loan_amount, loan_term, cibil, res_asset, 
                com_asset, lux_asset, bank_asset
            ]
            submit_btn.click(
                fn=predict_loan_status,
                inputs=inputs,
                outputs=[result_box, rejection_box]
            )

        # --- TAB 2: DATASET EXPLORER ---
        with gr.Tab("📁 Dataset Insights"):
            gr.Markdown("### 📈 Historical Dataset Overview (`loan_approval_dataset.csv`)")
            if df_raw is not None:
                gr.Markdown(f"**Total Records:** `{len(df_raw)}` rows | **Features:** `{len(df_raw.columns)}` columns")
                gr.DataFrame(df_raw.head(10), label="Preview (First 10 Rows)")
            else:
                gr.Markdown("⚠️ `loan_approval_dataset.csv` was not found in the project root directory.")

        # --- TAB 3: AI ADVISOR ---
        with gr.Tab("🤖 AI Loan Advisor"):
            gr.Markdown(
                """
                ### Chat with our AI Financial Assistant
                Have questions about a rejection or want to know how to improve your financial profile? Ask below!
                """
            )
            
            gr.ChatInterface(
                fn=chat_with_agent,
                examples=[
                    "How can I improve my CIBIL score?", 
                    "Why is my Debt-to-Income ratio important?", 
                    "What do lenders look for in assets?"
                ]
            )

    with gr.Accordion("🛠️ About the Developer & System", open=False):
        gr.Markdown(
            """
            **Created by:** Dheeraj  
            **Tech Stack:** Scikit-learn (Random Forest), Pandas, Gradio Blocks, Python  
            **Dataset:** `loan_approval_dataset.csv`  
            **Deployment:** Render
            """
        )

# ==========================================================
# Launch
# ==========================================================
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("PORT", 7860))
    )
