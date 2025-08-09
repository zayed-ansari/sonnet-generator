from flask import Flask, render_template, jsonify
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from huggingface_hub import snapshot_download
import os
import random

app = Flask(__name__)

MODEL_DIR = "./fine_tuned_sonnet_model"

# Loading the model 
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Predefined prompts
poem_prompts = [
    # Romantic and passionate
    "Your lips taste like forever",
    "I trace constellations on your skin",
    "Love letters written in heartbeat rhythms",
    "The way your hands memorize my curves",
    "Your name dances on my tongue like poetry",
    
    # Tender and intimate
    "Morning light tangled in your hair",
    "Whispered secrets between shared pillows",
    "The quiet magic of your sleeping breath",
    "Our shadows merge as one at sunset",
    "Fingertips tracing promises on bare skin",
    
    # Longing and desire
    "The ache of your absence between my ribs",
    "I collect your laughter like rare jewels",
    "Your voice echoes in the hollow of my throat",
    "The miles tremble between our longing",
    "I wear your memory like a second skin",
    
    # Eternal love
    "We wrote our vows in the book of eternity",
    "Your love blooms eternal in my soul's garden",
    "Two souls stitched with golden thread",
    "Our love outshines every constellation",
    "Forever tastes like your kiss at midnight",
    
    # Nature-inspired romance
    "Rose petals blush at our love story",
    "The ocean roars what my heart whispers",
    "Moonlight spills secrets about our passion",
    "Wildflowers bow to our devotion",
    "The stars envy how you look at me",
    
    # Classic romantic themes
    "Time slows in a lover's embrace",
    "The scent of roses fills the air",
    "Dancing flames in winter's heart", 
    "A love untold in silent words",
    "Golden fields stretch endlessly",
    
    # New sensual additions
    "Your gaze ignites wildfires in my veins",
    "The poetry of your pulse against my lips",
    "Our bodies speak the language of worship",
    "I drown willingly in your ocean eyes",
    "Every touch writes sonnets on my flesh"
]

def generate_poem():
    """
    Generate a poem using random prompt from above.
    """
    prompt = random.choice(poem_prompts)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            attention_mask=attention_mask,
            top_k=50,
            top_p=0.95,
            temperature=0.6,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )
    
    poem = tokenizer.decode(output[0], skip_special_tokens=True)
    poem = poem.replace("[POEM_START] ", "").replace("[POEM_END]", "").strip()
    return "\n".join(line.strip() for line in poem.split("\n") if line.strip())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        generated_poem = generate_poem()
        return jsonify({"poem": generated_poem})
    except Exception as e:
        return jsonify({"poem": "Oops! Something went wrong. Try again!"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
