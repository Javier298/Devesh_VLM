#######TO AVOID DWONLOADING STUFF AGAIN#######################################################################
import os
os.environ['TRANSFORMERS_CACHE'] = 'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/hugging_face/hub'
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
import ast
import warnings
warnings.filterwarnings("ignore")
##############################CLEAR MEMORY############################################
import gc
import time


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

#############################resize image###########################################################################
def resize_image(image: Image.Image):
    min_size = 56  # Ensure minimum 56X56
    max_size = 1008 #Ensure max 1008X1008
    width, height = image.size
    
    if width < min_size or height < min_size:
        image = image.resize((min_size, min_size), Image.BILINEAR)
    elif width > max_size or height > max_size:
        image = image.resize((max_size, max_size), Image.BILINEAR)   
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

label_mapping_damage_type = {
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
                                    18: "Concrete Corrosion (ConcreteC)",
                                    19: "Corrosion, no rust staining",
                                    20: "NO Exposed Reinforcement",
                                    21: "Scaling",
                                    22: "General Defects",
                                    23: "No defect"
                                    } 
                                    
label_mapping_objects = {
                                    12: "ExposedRebars",
                                    13: "Bearing",
                                    14: "EJoint (Expansion Joint)",
                                    15: "Drainage",
                                    16: "PEquipment (Protective Equipment)",
                                    17: "JTape (Joint Tape)",
                                    
                    }

damage_type_to_appearance_mapping = {
    "Crack": """** Elongated and narrow zigzag line**\n ** Clearly darker compared to the surrounding area or black**\n""",
    "ACrack": """** Many branched cracks**\n ** Mostly arbitrarily orientated**\n ** Usually with a small crack width (compared to Crack)**\n""",
    "Wetspot": """**Wet/darker mirroring area**\n""",
    "Efflorescence": """**Mostly roundish areas of white to yellowish or reddish color**\n **Strong efflorescence can look similar to stalactites.**\n **Often appears in weathered (Weathering) or wet areas (WetSpot) of the building and in combination with Crack and/or Rust**\n""",
    "Rust": """**Reddish to brownish area**\n **Often appears on concrete surfaces and metallic objects**\n""",
    "Rockpocket": """**Visible coarse aggregate**\n **Often in tilts of the formwork and the bottom of building parts (opposite side from which the concrete is poured into the formwork)**\n""",
    "Hollowareas": """**Hollowareas are not visually recognizable but their markings made with crayons (mostly yellow, red or blue) during close-up/hands-on inspections.**\n **Note: The outer edge of the marking is considered as the boundary of the according area. We annotate every chalk marking that approximately forms a closed geometric figure. Single lines are not labeled as Hollowarea as they are often used for the marking of Cracks.**\n """,
    "Cavity": """**Small air voids**\n **Mostly on vertical surfaces**\n""",
    "Spalling": """**Spalled concrete area revealing the coarse aggregate**\n **Significantly rougher surface (texture) inside the Spalling than in the surrounding surface**\n""",
    "Graffiti": """**All kinds of paintings on concrete and objects apart from defect markings**\n""",
    "Weathering": """**Summarizes all kinds of weathering on the structure (e.g. smut, dirt, debris) and Vegetation (e.g. plait, algae, moss, grass, plants).**\n **Weathering leads to a darker or greenish concrete surface compared to the rest of the surface.**\n""",
    "Restformwork": """**Visual Appearance:**\n **Left pieces of formwork in joints or on the structure's surface**\n **Restformwork can be made of wood and polystyrene (PS).**\n **PS is often used as a placeholder in joints during concreting.**\n""",
    "Concrete Corrosion (ConcreteC)": """**Includes the visually similar defects:**\n Washouts, Concrete corrosion and generally all kinds of planar corrosion/erosion/abrasion of concrete.**\n **Note: We summarize all these "planar corrosion defects“ in this class because they are visually hard to differ. According to inspection standards they have to subdivided which requires strong expertise in building defects.**\n""",
}

object_type_to_appearance_mapping = {
    "ExposedRebars": """**Exposed Reinforcement (non-prestressed and prestressed) and cladding tubes of tendons**\n **Often appears in combination with Spalling or Rockpocket, and Rust**\n""",
    "Bearing": """** All kinds of bearings, such as rocker-, elastomer- or spherical bearings**\n""",
    "EJoint (Expansion Joint)": """**Located at the beginning and end of the bridge**\n **Assembled cross to the longitudinal bridge axis**\n""",
    "Drainage": """**All kinds of pipes and outlets made of Polyvinylchlorid or metal mounted on the bridge.**\n""",
    "PEquipment (Protective Equipment)": """ """,
    "JTape (Joint Tape)": """**All joints that are filled with elastomer or silicon**\n **Note: Originally, Joint Tape means an elastomer strap at the end and beginning of relatively small bridges.**\n""",
}
######################################CONVERT TO TEXT TEMPLATE#######################################################
system_message = """You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, and interpreting visual data. 
You have currently learned about several bridge-damage types. Your task is to generate a short inspection report on seeing the image."""
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resize_image(Image.open(sample["image"])),
                },
                {
                    "type": "text",
                    "text": f"""Here is the mapping from damage type to its visual appearance {damage_type_to_appearance_mapping}. Here is the mapping from object type to its visual appearance {object_type_to_appearance_mapping}. Here is the label-mapping of numbers to damage types {label_mapping_damage_type}, and label mapping from number to object types {label_mapping_objects}. Using the numbers, state the damage type(s) and object type(s) present in the image:"""
                 },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text":sample["new_label"] }], #'The supervisor will check' #sample["label"]
        },
    ]

###############################################################dataset 3#########################################################################################################
#'''
dataset3 = load_dataset("json", data_files = {'train':'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/labels_for_two_datasets/Train/dacl/dacl_labels_train.json',
                                                  'val': 'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/labels_for_two_datasets/Val/dacl/dacl_labels_val.json'})

#use image_cast from hf library

train_dataset3 = dataset3['train']
#train_label3 = train_dataset3["label"]

val_dataset3 = dataset3['val']

train_dataset3 = [format_data(sample) for sample in train_dataset3]
val_dataset3 = [format_data(sample) for sample in val_dataset3]
#'''
print("Formatted!!\n")
##############################
###################################GENERATE SAMPLE FUNCTION##########################################################
from qwen_vl_utils import process_vision_info


def generate_text_from_sample(model, processor, sample, max_new_tokens=2048, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        [sample[1]], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    del model_inputs
    actual_answer = sample[2]["content"][0]["text"]
    return output_text[0], actual_answer  # Return the first decoded output text
######################################LOAD FINE-TUNED MODEL#######################################################
clear_memory()
model_id = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

min_pixels = 4 * 28 * 28
max_pixels = 1296 * 28 * 28
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

#print(f"Before adapter parameters: {model.num_parameters()}")
'''
adapter_path_1 = "./fine_tuned_weights/codebrim" #"./fine_tuned_weights/260156" #"./output"
adapter_path_2 = "./fine_tuned_weights/2601506"
peft_model = PeftModel.from_pretrained(model,adapter_path_1, adapter_name="codebrim")
weighted_adapter_name="codebrim-2601506" #adapter_path_1
peft_model.load_adapter(adapter_path_2, adapter_name="2601506")
peft_model.add_weighted_adapter(adapters=["codebrim","2601506"], weights=[0.5,0.5], adapter_name=weighted_adapter_name,combination_type="linear") #"codebrim"
peft_model.set_adapter(weighted_adapter_name) #"codebrim"
'''

adapter_path_1 = "./fine_tuned_weights/codebrim_with_metrics" #"./output"
adapter_path_2 = "./fine_tuned_weights/2601506_with_metrics"
adapter_path_3 = "./fine_tuned_weights/dacl_with_metrics_final" #"./fine_tuned_weights/dacl_with_metrics"

dataset = "dacl" #"2601506" #"codebrim"
peft_model = PeftModel.from_pretrained(model,adapter_path_3, adapter_name=dataset)
model.load_adapter(adapter_path_3)

####################################################################################################
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0) #USE average="samples" FOR dacl-10k DATASET
    accuracy = accuracy_score(labels, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_model(dataset):
    mlb = MultiLabelBinarizer()
    preds = []
    labels = []
    i=1

    for sample_data in dataset:
        generated_answer, actual_answer = generate_text_from_sample(model, processor,sample_data)

        # Convert text labels to numeric values
        
        predicted_label = ast.literal_eval(generated_answer) #label_mapping.get(generated_answer.strip(),-1)  # Default to -1 if unknown
        #print(f"Data Type of generated answer: {type(predicted_label)}\n")

        preds.append(predicted_label)
        labels.append(actual_answer)

        print(i)
        i+=1

    mlb.fit(preds + labels)
    preds_bin = mlb.transform(preds)
    labels_bin = mlb.transform(labels)
    #print(f"length of preds: {len(preds)}")
    return compute_metrics(preds_bin, labels_bin)





metrics = evaluate_model(val_dataset3)
print(f"The evaluation metrics for the {dataset} dataset are: \n {metrics}")

peft_model.unload()
peft_model.delete_adapter(dataset)
print(f"Current number of parameters: {model.num_parameters()}")

print("Done and finished")
######################################################################