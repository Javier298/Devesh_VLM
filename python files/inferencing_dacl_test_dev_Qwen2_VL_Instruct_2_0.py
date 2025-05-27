#######TO AVOID DWONLOADING STUFF (SHARDS) AGAIN#######################################################################
import os
#os.environ['CUDA_LAUNCH_BLOCKING']= "1"
#os.environ['TRANSFORMERS_CACHE'] = 'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/hugging_face/hub'
os.environ['TRANSFORMERS_CACHE'] = 'C:/Users/Ddimble/.cache/huggingface/hub' #if the checkpoint shards are not downloaded, then comment this line 

#######################IMPORT LIBRARIES###################################
#os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig, Trainer, TrainingArguments, get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from peft.optimizers import create_loraplus_optimizer
import bitsandbytes as bnb
import wandb
import warnings
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph
from reportlab.platypus import Image as replIm
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from datetime import date
from qwen_vl_utils import process_vision_info
warnings.filterwarnings("ignore")
import ast
import gc
import time

##############################CLEAR MEMORY############################################

def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    if "trainer" in globals():
        del globals()["trainer"]
    if "peft_model" in globals():
        del globals()["peft_model"]
    if "bnb_config" in globals():
        del globals()["bnb_config"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

#############################RESIZING THE IMAGE###########################################################################
def resize_image(image: Image.Image):
    min_size = 56  # Ensure minimum 56X56
    max_size = 1260#1008 #Ensure max 1008X1008 #1120
    width, height = image.size
    
    if width < min_size or height < min_size:
        image = image.resize((max(width,min_size), max(height,min_size)), Image.BILINEAR)
    elif width > max_size or height > max_size:
        image = image.resize((min(width,max_size), min(height,max_size)), Image.BILINEAR)
    return image

#########################################MAPPINGS######################################################
label_mapping = {
                                    0: "Crack",
                                    1: "ACrack",
                                    2: "Wetspot",
                                    3: "Efflorescence",
                                    4: "Rust",
                                    5: "Rockpocket",
                                    6: "Hollowareas",
                                    7: "Cavity",
                                    8: "Spalling",
                                    9: "Graffiti",
                                    10: "Weathering",
                                    11: "Restformwork",
                                    12: "ExposedRebars",
                                    13: "Bearing",
                                    14: "EJoint (Expansion Joint)",
                                    15: "Drainage",
                                    16: "PEquipment (Protective Equipment)",
                                    17: "JTape (Joint Tape)",
                                    18: "Concrete Corrosion (ConcreteC)",
                                    19: "Corrosion, no rust staining",
                                    20: "NO Exposed Reinforcement",
                                    21: "Scaling",
                                    22: "General Defects",
                                    23: "No defect"
                                    }
######################################IMAGE FOLDER#########################################################
image_folder = "./datasets/dacl10k_v2_devphase/images/testdev/" #do not change!!
image_name = "Report Example 05.jpg" #"dacl10k_v2_testdev_0580.jpg" #"Report Example 05.jpg" #just change the last 3 digits
image_path = os.path.join(image_folder, image_name)

######################################CONVERSATION PART 1#######################################################
system_message = """You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, and interpreting visual data. 
You have currently learned about several bridge-damage types. Your task is to generate a short inspection report on seeing the image."""
conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resize_image(Image.open(image_path)) #Image.open(image_path) #resize_image(Image.open(image_path)),
                },
                {
                    "type": "text",
                    "text": f"""Here is the label-mapping of numbers to damage types {label_mapping}. Numbers 12 to 17(both included) are object types. Using the numbers, state the damage type(s) and object type(s) present in the image:"""
                 },
            ],
        },
    ]
######################################LOAD FINE-TUNED MODEL#######################################################
clear_memory()
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2-VL-2B-Instruct",
     torch_dtype= torch.bfloat16, #torch.bfloat16,
     attn_implementation="flash_attention_2",
     #load_in_4bit=True,
     low_cpu_mem_usage=True,
     #quantization_config=bnb_config,
     device_map="auto",
     use_cache=False,
 )

#device = "cuda"
#model.to("cpu")

min_pixels = 4 * 28 * 28
max_pixels = 2025 * 28 * 28 #1296 * 28 * 28
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

#print(f"Before adapter parameters: {model.num_parameters()}")


adapter_path_1 = "./fine_tuned_weights/codebrim_with_metrics" #"./output"
adapter_path_2 = "./fine_tuned_weights/2601506_with_metrics"
adapter_path_3 = "./fine_tuned_weights/dacl_with_metrics"

'''
weighted_adapter_name="codebrim-2601506-dacl" #adapter_path_1
peft_model = PeftModel.from_pretrained(model,adapter_path_1, adapter_name="codebrim")
#peft_model.load_adapter(adapter_path_2, adapter_name="2601506")
peft_model.load_adapter(adapter_path_3, adapter_name="dacl")

peft_model.add_weighted_adapter(adapters=["codebrim", "dacl"], weights=[0.28,0.72], adapter_name=weighted_adapter_name,combination_type="linear") #"codebrim"
peft_model.set_adapter(weighted_adapter_name)
'''

print(f"Before adapter parameters: {model.num_parameters()}")
#'''
adapter_name ="dacl"
peft_model = PeftModel.from_pretrained(model,adapter_path_3, adapter_name=adapter_name)
#model.load_adapter(adapter_path_3)
#'''

print(f"After adapter parameters: {model.num_parameters()}")

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
#print(f"Before adapter parameters: {model.num_parameters()}")

image_inputs,_ = process_vision_info(conversation)

inputs = processor(
    text=[text_prompt], images=image_inputs, padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda") #"cuda"

"""
with torch.no_grad():
    output = model(**inputs)
    #print(output)
"""

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=2048) #.to("cpu")
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

new_output = ast.literal_eval(output_text[0])
output_labels = [label_mapping.get(int(op)) for op in new_output]


damage_str = ", ".join(output_labels)

print(f"Current number of parameters: {model.num_parameters()}")

############################CONVERSATION PART 2##########################################
conversation2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": resize_image(Image.open(image_path)) #Image.open(image_path) #resize_image(Image.open(image_path)), #val_dataset3[69][1]["content"][0]["image"]
            },
            {"type": "text", 
             "text": f"""Here is the label-mapping of numbers to damaage types {label_mapping}. Based on identified damage type(s): {output_text[0]}, give a concise report containing the following details:\n
                    - Damages and object type(s): {damage_str}\n
                    - Impact: (Brief description of the effect on the structure)\n
                    - Size: (Estimated size in cm² if possible)\n
                    - Direction: (Horizontal, vertical, diagonal, etc.)\n
                    - Possible Reasons: (What could have caused this damage?)\n:"""
                    }, #Numbers 12 to 17(both included) are object types.
                    #Give a short inspection report consisting of :  1. damage impact, 2. damage size (in cm²), 3. damage direction, 4. possible causes #, ""object type"", and ""its functonalities""
        ],
    }
]

# Preprocess the inputs
text_prompt2 = processor.apply_chat_template(conversation2, tokenize=False, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
#print(f"Before adapter parameters: {model.num_parameters()}")

image_inputs2,_ = process_vision_info(conversation2)
#print(image_inputs)
inputs2 = processor(
    text=[text_prompt2], images=image_inputs2, padding=True, return_tensors="pt"
)
inputs2 = inputs2.to("cuda") #"cuda"

# Inference: Generation of the output
output_ids2 = model.generate(**inputs2, max_new_tokens=2048) #8192
generated_ids2 = [
    output_ids2[len(input_ids2) :]
    for input_ids2, output_ids2 in zip(inputs2.input_ids, output_ids2)
]
output_text2 = processor.batch_decode(
    generated_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
#print(f"Inspection Report: \n{output_text2[0]}")

peft_model.unload()
peft_model.delete_adapter(adapter_name) #weighted_adapter_name #adapter_name
print(f"Current number of parameters: {model.num_parameters()}")

###########################STROING THE CONTENT IN A DICTIONARY##############################################
damage_info = {}
current_key = None  # To track the current dictionary key

for line in output_text2[0].strip().split("\n"):
    line = line.strip()
    
    if line.startswith("- "):  
        # New key-value pair
        key, value = line[2:].split(": ", 1)  # Remove "- " and split at ": "
        current_key = key.strip()
        damage_info[current_key] = value.strip()
    elif current_key:
        # Continuation of the previous value
        damage_info[current_key] += " " + line.strip()


#print(damage_info)
################################CREATING A PDF#########################################
def create_pdf(image_name, details):
    base_name = os.path.splitext(image_name)[0]
    output_pdf = f"{base_name}.pdf"

    # Create the PDF document
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    #elements = []  # Holds all the components (image + table)

    # Add the image
    img = replIm(image_path, width=300, height=200) #Image(image_path, width=300, height=200)  
    #elements.append(img)

    styles = getSampleStyleSheet()
    text_style = styles["BodyText"] 
    # Define table data
    data = [
        ["Category", "Details"],  # Table headers
        ["Project Name", "x"],
        ["Project ID", "x"],
        ["Project Location", "x"],
        ["Company Name", "ICoM GmbH"],
        ["Inspector Name", "Max Mastermann"],
        ["Date of Inspection", f"{date.today().strftime("%d/%m/%Y")}"],
        ["Damage & Object Type(s)", Paragraph(damage_str)],
        ["Impact", Paragraph(details["Impact"])],
        ["Size", Paragraph(details["Size"])],
        ["Direction", Paragraph(details["Direction"])],
        ["Possible Reasons", Paragraph(details["Possible Reasons"])],
    ]

    # Create the table
    table = Table(data, colWidths=[2.5 * inch, 4 * inch]) #[150, 350]

    # Table styling
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),  # Header background
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),  # Header text color
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),  # Align all text to left
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),  # Header font
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),  # Header padding
        ("GRID", (0, 0), (-1, -1), 1, colors.black),  # Add grid lines
    ])

    table.setStyle(style)

    #elements.append(table)  # Add table to document

    # Build the PDF
    #doc.build(elements)
    doc.build([img, Spacer(1,20), table])
    print(f"PDF saved as {output_pdf}")


#base_name = os.path.splitext(image_name)[0]
#output_pdf = f"{base_name}.pdf"
text = f"Bridge Damage Report\n\n output_text2[0]"
create_pdf(image_name, damage_info)
#print(f"PDF saved as {output_pdf}")

################################################################################################