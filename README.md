# Devesh_VLM
In this repo, I have fine-tuned a VLM(Qwen2-VL-2B-Instruct) on bridge-damage datasets, and then inferenced it, which generates inspection reports. 

Here is the demo video: https://drive.google.com/drive/folders/1BurEeqaMoMcYjXjS-sg4o27Rbbr658XX?usp=sharing

# 🧠 Development of Vision-Language Model for the intelligent inspection of existing bridge structures
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
{
  "id": 32,
  "image": "/path/to/image.jpg",
  "label": [3, 17]
}
label: List of class IDs (supports both single- and multi-label classification)

Damage class mapping is defined in label_map.json

## 🧪 Example Output
🖼️ Input Image:

<img src="examples/Report Example 03.jpg" width="400"/>

📝 Generated Report:


- Damage Type: Graffiti, Weathering, PEquipment (Protective Equipment)  
- Impact: The graffiti and weathering have caused damage to the concrete wall, including the railway track, which is a piece of equipment.
- Size: The damage is estimated to be around 10 cm² for the graffiti and 20 cm² for the weathering. 
- Direction: Horizontal (graffiti) and vertical (weathering).  
- Possible Reasons: The graffiti and weathering have exposed the underlying concrete, leading to damage to the railway track.  
📄 Generated PDF sample → see /sample_inspection_reports/Example Report 03.pdf

### ⚙️ How to Run (Context)

Due to infrastructure limitations, full inference execution is not available at this time.

However, the repository includes:
- `run_inference.py`: the original inference script used during the internship
- `generate_report.py`: PDF report generator based on model outputs
- Model weights used during evaluation (currently not publicly hosted due to size)

**Note:** This project was developed on a NVIDIA RTX A4000 GPU, and certain dependencies (e.g., FlashAttention 2) may not run on standard local hardware without modification.

## 📁 Folder Structure
Devesh_VLM/
├── Notebooks/
├── annotation_files/
├── python_files/
├── sample_inspection_reports/
├── examples/
├── README.md
└── requirements.txt

## 🧑‍🔬 Acknowledgments
This project was developed during my internship at ICoM, RWTH Aachen.


