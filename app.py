import os
import joblib
import numpy as np
import gradio as gr


# Load Trained Model

model = joblib.load("loan_approval_model.pkl")


# Prediction Function

def predict_loan(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):
    try:

        # Label Encoding Mapping
        # Graduate -> 0
        # Not Graduate -> 1
        education = 0 if education == "Graduate" else 1

        # No -> 0
        # Yes -> 1
        self_employed = 1 if self_employed == "Yes" else 0

        features = np.array([[
            int(no_of_dependents),
            education,
            self_employed,
            float(income_annum),
            float(loan_amount),
            int(loan_term),
            int(cibil_score),
            float(residential_assets_value),
            float(commercial_assets_value),
            float(luxury_assets_value),
            float(bank_asset_value)
        ]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            return """
            <div style="
                background:#d1fae5;
                padding:25px;
                border-radius:15px;
                text-align:center;
                box-shadow:0 5px 20px rgba(0,0,0,.2);
            ">
                <h1 style="color:#065f46;"> LOAN APPROVED</h1>
                <h3 style="color:#065f46;">
                Congratulations! The applicant is eligible for the loan.
                </h3>
            </div>
            """

        else:
            return """
            <div style="
                background:#fee2e2;
                padding:25px;
                border-radius:15px;
                text-align:center;
                box-shadow:0 5px 20px rgba(0,0,0,.2);
            ">
                <h1 style="color:#991b1b;"> LOAN REJECTED</h1>
                <h3 style="color:#991b1b;">
                Sorry! The applicant is not eligible for the loan.
                </h3>
            </div>
            """

    except Exception as e:
        return f"<h3 style='color:red;'>Error: {e}</h3>"


# Custom CSS


custom_css = """
body{
background:linear-gradient(135deg,#0f172a,#1e40af);
}

.gradio-container{
max-width:1100px !important;
margin:auto;
}

footer{
display:none !important;
}

h1,h2,h3{
text-align:center;
}

.gr-button{
background:#2563eb !important;
color:white !important;
font-size:18px !important;
font-weight:bold !important;
border-radius:12px !important;
height:50px !important;
}

.gr-button:hover{
background:#1d4ed8 !important;
}

textarea,input{
border-radius:10px !important;
}

.block{
border-radius:15px !important;
}
"""


# Gradio Interface


demo = gr.Interface(
    fn=predict_loan,

    inputs=[

        gr.Number(
            label=" Number of Dependents",
            value=0
        ),

        gr.Dropdown(
            choices=["Graduate", "Not Graduate"],
            value="Graduate",
            label=" Education"
        ),

        gr.Dropdown(
            choices=["No", "Yes"],
            value="No",
            label=" Self Employed"
        ),

        gr.Number(
            label=" Annual Income"
        ),

        gr.Number(
            label=" Loan Amount"
        ),

        gr.Number(
            label=" Loan Term"
        ),

        gr.Slider(
            minimum=300,
            maximum=900,
            value=700,
            step=1,
            label=" CIBIL Score"
        ),

        gr.Number(
            label=" Residential Assets Value"
        ),

        gr.Number(
            label=" Commercial Assets Value"
        ),

        gr.Number(
            label=" Luxury Assets Value"
        ),

        gr.Number(
            label=" Bank Assets Value"
        ),
    ],

    outputs=gr.HTML(label="Prediction"),

    title=" Loan Approval Prediction System",

    description="""
#  AI Loan Approval Prediction

### Enter the applicant's financial details and click **Submit**.

---

###  Developed By
**Dheeraj**

**Roll No:** 241559

**College:** Panipat Institute of Engineering & Technology

**Course:** BCA - Data Science

---

**Machine Learning Model:** Random Forest Classifier
""",

    theme=gr.themes.Soft(),

    css=custom_css
)


# Launch App


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
