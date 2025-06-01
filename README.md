# Devesh_VLM
In this repo, I have fine-tuned a VLM(Qwen2-VL-2B-Instruct) on bridge-damage datasets, and then inferenced it, which generates inspection reports. 

Here is the demo video: https://drive.google.com/drive/folders/1BurEeqaMoMcYjXjS-sg4o27Rbbr658XX?usp=sharing

# 🧠 Automated Bridge Damage Inspection using Vision-Language Models
A fine-tuned Qwen-2VL-2B model for automated bridge damage classification and report generation, developed during my research internship at ICoM, RWTH Aachen.

## 🔍 Project Goals
Classify bridge damage types from images (single & multi-label)

Generate bullet-point bridge inspection reports using VLM

Automate end-to-end inference and PDF report generation

## 🚀 Features
✅ Fine-tuning of Qwen-2VL-2B Instruct using LoRA + SFT

✅ FlashAttention 2 for accelerated training and inference

✅ PDF report generation with text + visual overlays (via reportlab)

✅ Handles both single-label and multi-label classification

✅ Custom damage taxonomy used for few-shot learning

## 🧰 Tech Stack
Component	Tools / Frameworks
Model	Qwen-2VL-2B (VLM)
Fine-tuning	LoRA, PEFT, trl
Training	PyTorch, Hugging Face Transformers
Inference	FlashAttention 2
Reporting	ReportLab
Evaluation	Scikit-learn

## 📊 Dataset Structure (Example)
json
Copy
Edit
{
  "id": 32,
  "image": "/path/to/image.jpg",
  "label": [3, 17]
}
label: List of class IDs (supports both single- and multi-label classification)

Damage class mapping is defined in label_map.json

## 🧪 Example Output
🖼️ Input Image:
<img src="examples/sample_input.jpg" width="400"/>

📝 Generated Report:

diff
Copy
Edit
- Damage Type: Exposed rebars, Cracks  
- Impact: Medium  
- Area: 23.5 sq.cm  
- Direction: Horizontal  
- Possible Cause: Corrosion due to water seepage  
📄 Generated PDF sample → see /examples/sample_output.pdf

## ⚙️ How to Run (Inference Only)
Clone the repo

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run inference:

bash
Copy
Edit
python run_inference.py --image path/to/image.jpg
## 📁 Folder Structure
kotlin
Copy
Edit
Qwen2VL_BridgeInspection/
├── data/
├── training/
├── inference/
├── report_generation/
├── examples/
├── README.md
└── requirements.txt
## 📌 Future Work
Add GUI for uploading images and generating report

Integrate multi-image support

Explore multi-modal attention visualizations

## 🧑‍🔬 Acknowledgments
This project was developed during my internship at ICoM, RWTH Aachen.


