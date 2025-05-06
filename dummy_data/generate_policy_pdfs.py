from fpdf import FPDF
import os

# Output directory (update to your local path if needed)
output_dir = "data/rag_documents"
os.makedirs(output_dir, exist_ok=True)

# Sample policy titles
policy_types = [
    "Auto Insurance Policy - NSW",
    "Home Insurance - Flood Coverage",
    "Life Insurance Basic Plan",
    "Comprehensive Health Insurance",
    "Business Interruption Insurance",
    "Workers Compensation Terms",
    "Cyber Liability Insurance",
    "Travel Insurance Conditions",
    "Rental Property Insurance",
    "Personal Injury Protection Plan"
]

# Sections inside each policy document
sections = [
    "Eligibility and Coverage Terms",
    "Claim Filing Procedure",
    "Exclusions and Limitations",
    "Policyholder Responsibilities",
    "Premium Payment Structure",
    "Risk Assessment Methodology",
    "Cancellation and Refund Policy",
    "Legal Dispute Resolution",
    "Fraud Detection and Penalties",
    "Data Privacy and Protection"
]

# Function to create a PDF
def generate_policy_pdf(title, idx):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align="C")
    pdf.ln(10)

    for section in sections:
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(200, 10, txt=section, ln=True)
        pdf.set_font("Arial", size=10)
        for _ in range(5):
            text = (
                f"{section}: This clause explains how the {title.lower()} handles this aspect. "
                "It is governed by regional laws, underwriting models, risk scoring systems, and actuarial assumptions. "
                "Policyholders are advised to review specific clause details before submitting claims or adjustments."
            )
            pdf.multi_cell(0, 10, txt=text)
        pdf.ln(5)

    file_path = os.path.join(output_dir, f"policy_document_{idx+1}.pdf")
    pdf.output(file_path)
    print(f"✅ Saved: {file_path}")

# Generate 10 PDFs
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    for i, title in enumerate(policy_types):
        generate_policy_pdf(title, i)
