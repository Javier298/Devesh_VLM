
#######TO AVOID DWONLOADING STUFF AGAIN#######################################################################
import os
os.environ['TRANSFORMERS_CACHE'] = 'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/hugging_face/hub'
#######################IMPORT LIBRARIES###################################
#os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, interleave_datasets
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig, Trainer, TrainingArguments, get_scheduler
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from peft.optimizers import create_loraplus_optimizer
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
import bitsandbytes as bnb
import wandb
import warnings
warnings.filterwarnings("ignore")
####################################RESIZE IMAGE################################################################
def resize_image(image: Image.Image):
    min_size = 56  # Ensure minimum 56X56
    max_size = 1120 #Ensure max 1008X1008 #1120
    width, height = image.size
    
    if width < min_size or height < min_size:
        image = image.resize((max(width,min_size), max(height,min_size)), Image.BILINEAR)
    elif width > max_size or height > max_size:
        image = image.resize((min(width,max_size), min(height,max_size)), Image.BILINEAR)   
    # Now apply your original rules

    '''
    if max(image.size) < 56:
        image = image.resize((56, 56), Image.BILINEAR)
    elif max(image.size) > 1008:
        image = image.resize((1008, 1008), Image.BILINEAR)
    '''
    return image

###########PARAMS FOR TRAINING_ARGS############
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
EPOCHS = 1#2
BATCH_SIZE = 1
GRADIENT_CHECKPOINTING = True  # Tradeoff between memory efficiency and computation time.
GRADIENT_ACCUMULATION_STEPS = 4 #4
USE_REENTRANT = False
OPTIM = "paged_adamw_32bit" #"paged_adamw_32bit"
LEARNING_RATE = 2e-5 #2e-4
LOGGING_STEPS = 180 #180 #173 #433 
EVAL_STEPS = 180 #180 #173 #433 
SAVE_STEPS = 360 #360 #346 #866 
EVAL_STRATEGY = "steps"
SAVE_STRATEGY = "steps" #"steps"
METRIC_FOR_BEST_MODEL="eval_loss" #"eval_loss"
LOAD_BEST_MODEL_AT_END=True
MAX_GRAD_NORM = 1
WARMUP_RATIO = 0.1 #delete
WARMUP_STEPS = 0 #delete
WEIGHT_DECAY = 0.01 #delete
DATASET_KWARGS={"skip_prepare_dataset": True} # We have to put for VLMs
REMOVE_UNUSED_COLUMNS = False # VLM thing
MAX_SEQ_LEN= 1024 #128
NUM_STEPS = (6935 // (GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE)) * EPOCHS #6935 #2882 #2777 
print(f"NUM_STEPS: {NUM_STEPS}")
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
            "content": [{"type": "text", "text": [sample["new_label"]] }], #'The supervisor will check' #sample["label"]
        },
    ]

###############LOAD CODEBRIM DATASET##########################
'''
dataset = load_dataset("json", data_files = {'train':'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/labels_for_two_datasets/Train/BInet/BInet_train.json',
                                                  'val': 'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/labels_for_two_datasets/Val/BInet/BInet_val.json'})

#use image_cast from hf library

train_dataset = dataset['train']
train_label = train_dataset["label"]

val_dataset = dataset['val']


train_dataset = [format_data(sample) for sample in train_dataset]
val_dataset = [format_data(sample) for sample in val_dataset]
'''
############################### DATASET 2601506 #######################################################
#print(train_dataset[0])
#print(val_dataset[0])
#'''
dataset2 = load_dataset("json", data_files = {'train':'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/labels_for_two_datasets/Train/2601506/2601506_train.json',
                                                  'val': 'D:/mdfBIM+ - VLM 4 Bridge Damages - Jäkel_Bitte nicht löschen!_/labels_for_two_datasets/Val/2601506/2601506_val.json'})

#use image_cast from hf library

train_dataset2 = dataset2['train']
train_label2 = train_dataset2["label"]

val_dataset2 = dataset2['val']


train_dataset2 = [format_data(sample) for sample in train_dataset2]
val_dataset2 = [format_data(sample) for sample in val_dataset2]
#'''
print("Formatted!!\n")

##################################COMBINE DATASETS############################################
'''
combined_train_dataset = interleave_datasets([train_dataset,train_dataset2], probabilities=[0.5,0.5])
combined_val_dataset = interleave_datasets([val_dataset,val_dataset2], probabilities=[0.5,0.5])
print("Combined successfully!")
combined_train_dataset = [format_data(sample) for sample in combined_train_dataset]
combined_val_dataset = [format_data(sample) for sample in combined_val_dataset]
print("Formatted Successfully")
'''
################################# checking ############################################
#sample_data = train_dataset[0]
#sample_question = train_dataset[0][1]["content"][1]["text"]
#sample_answer = train_dataset[0][2]["content"][0]["text"]
#sample_image = train_dataset[0][1]["content"][0]["image"]

#print(sample_question)
#print(sample_answer)
#sample_image

#################################### LOAD MODEL ##############################################
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        #bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

model = Qwen2VLForConditionalGeneration.from_pretrained(
     "Qwen/Qwen2-VL-2B-Instruct",
     torch_dtype=torch.bfloat16,
     attn_implementation="flash_attention_2",
     #load_in_4bit=True,
     low_cpu_mem_usage=True,
     #quantization_config=bnb_config,
     device_map="auto",
     use_cache=False,
 )

min_pixels = 4 * 28 * 28
max_pixels = 1600 * 28 * 28 #1849 #1296 #1600
processor = Qwen2VLProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels) #min_pixels=min_pixels
processor.tokenizer.padding_side = "right"

#print(model)

########################### text generator ########################
def text_generator(sample_data):
    text = processor.apply_chat_template(
        sample_data[0:2], tokenize=False, add_generation_prompt=True
    )

    #print(f"Prompt: {text}")
    #print("-"*30)

    image_inputs, _ = process_vision_info(sample_data) #sample_data[1]["content"][0]["image"]

    inputs = processor(
        text=[text],
        images = image_inputs,
        return_tensors="pt"
    )
    inputs = inputs.to(device=model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )
    del inputs
    actual_answer = sample_data[2]["content"][0]["text"]
    return output_text[0], actual_answer
    

#generated_text, actual_answer = text_generator(sample_data)
#print(f"Generated Answer: {generated_text}")
#print(f"Actual Answer: {actual_answer}")


#####################LORA CONFIG############################
peft_config = LoraConfig(
    use_dora=True,
    inference_mode=False,
    lora_alpha=64, #16 #64
    lora_dropout=0.1, #0.05
    r=16, #16 #8
    bias="none",
    target_modules=["q_proj", "v_proj"], #"k_proj", #"o_proj" #, "qkv", "proj"
    task_type="CAUSAL_LM",
    init_lora_weights= "eva", #"eva"
    use_rslora=True,
)

#print(f"Before adapter parameters: {model.num_parameters()}")
peft_model = get_peft_model(model, peft_config)

peft_model = torch.compile(peft_model)

peft_model.print_trainable_parameters() 

########################TRAINING ARGS################################################
training_args = SFTConfig(
    ##change this accordingly####
    output_dir="./fine_tuned_weights/260156_with_metrics_final", #260156 #codebrim #codebrim_and_2601506
    #label_names= train_label,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type= "cosine_with_restarts",
    lr_scheduler_kwargs= {"num_cycles": 2}, #{"power": 3} #"constant", #"linear", #"polynomial", #"cosine_with_restarts"
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    max_grad_norm=MAX_GRAD_NORM,
    #warmup_ratio=WARMUP_RATIO,
    #warmup_steps=WARMUP_STEPS,
    bf16=True,
    tf32=True,
    gradient_accumulation_steps=4, #16 #8 #4 #2
    dataset_kwargs=DATASET_KWARGS,
    max_seq_length=MAX_SEQ_LEN,
    remove_unused_columns = REMOVE_UNUSED_COLUMNS,
    optim=OPTIM,
    label_names=["labels"],
    report_to="wandb", #"wandb"
)

wandb.init(project="VQA or Image Captioning", name="21.03.2025-dataset 2601506_with_metrics-Qwen-2-VL-Instruct-2B", config=training_args,) #Qwen-2-VL-Instruct-2B-dataset 2601506-18.02.2025

#################################### COLLATE FUNCTION #############################################
#collate_sample = [train_dataset[0], train_dataset[1]] # for batch size 2.

def collate_fn(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    #image_inputs = [example[1]["content"][0]["image"] for example in examples] #[resize_image(example[1]["content"][0]["image"]) for example in examples]

    image_inputs = [process_vision_info(example)[0] for example in examples]
    #resize_image(image_inputs)
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    #batch["input_ids"]
    #batch["labels"] = batch["input_ids"]
    #batch=batch.to(device=model.device)
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels
    
    batch["labels"] = labels
    return batch

#collated_data = collate_fn(collate_sample)
#print(collated_data.keys())  # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'labels'])

#####################################METRIC_FUNCTION##############################################
# Load text evaluation metrics
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
# Load evaluation metrics
#bleu = evaluate.load("bleu")
#rouge = evaluate.load("rouge")
#meteor = evaluate.load("meteor")

#accuracy_metric = evaluate.load("accuracy")
#precision_metric = evaluate.load("precision")
#recall_metric = evaluate.load("recall")
#f1_metric = evaluate.load("f1")

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, preds)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_model(dataset):
    preds = []
    labels = []
    i=1

    for sample_data in dataset:
        generated_answer, actual_answer = text_generator(sample_data)

        # Convert text labels to numeric values
        predicted_label = int(generated_answer) # Default to -1 if unknown
        actual_label = actual_answer 

        if predicted_label != -1 and actual_label != -1:  # Ensure valid labels
            preds.append(predicted_label)
            labels.append(actual_label)
        print(i)
        i+=1

    print(f"length of preds: {len(preds)}")
    return compute_metrics(preds, labels)

# Define the label mapping
'''

label_mapping = {
    "Crack": 0,
    "ACrack": 1,
    "Wetspot": 2,
    "Efflorescence": 3,
    "Rust": 4,
    "Rockpocket": 5,
    "Hollowareas": 6,
    "Cavity": 7,
    "Spalling": 8,
    "Graffiti": 9,
    "Weathering": 10,
    "Restformwork": 11,
    "ExposedRebars": 12,
    "Bearing": 13,
    "EJoint": 14,
    "Drainage": 15,
    "PEquipment": 16,
    "JTape": 17,
    "WConccor": 18

}
'''
#inverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Reverse mapping
#####################################SFT TRAINER##################################################
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset= train_dataset2, #train_dataset2 #combined_train_dataset
    eval_dataset= val_dataset2, #val_dataset2 #combined_val_dataset
    data_collator=collate_fn,
    peft_config=peft_config,
    #optimizers = (optimizer,scheduler),
    #compute_metrics=compute_metrics,  # Updated
    tokenizer = processor.tokenizer,
)

################################################################################################

torch.cuda.empty_cache()
#print(f"Before adapter parameters: {model.num_parameters()}")
######################################EVAL######################################################
print("-"*30)
print("Initial Evaluation")
metric = trainer.evaluate()
print(metric)
print("-"*30)

print("Training")
trainer.train()
print("-"*30)
trainer.save_model(training_args.output_dir)

#dataset = "dacl"
#metrics = evaluate_model(val_dataset)
#print(f"The evaluation metrics for the {dataset} dataset are: \n {metrics}") #print(metrics)

print("Done and finished")
print("-"*30)