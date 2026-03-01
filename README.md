# Sonnet Generator
A Flask-based web app that generates original sonnets using a fine-tuned GPT-2 model trained on Shakespeare's Sonnets dataset. It uses predefined romantic prompts to create poetic outputs. (Note: Hosting requires GPU for optimal performance; a video demo is available on my LinkedIn.)

## Dataset:
  The model was fine-tuned on Shakespeare's Sonnets from Project Gutenberg: [Download here](https://www.gutenberg.org/ebooks/1041).

### Features:
- Generates sonnets via a simple web interface.
- Uses random romantic prompts (e.g., "Your lips taste like forever") for inspiration.
- Built with PyTorch, Transformers, and Flask.

## Installation

**Clone the repo:**
```
git clone https://github.com/zayed-ansari/sonnet-generator.git
cd sonnet-generator
```
**Install dependencies:**
```
pip install -r requirements.txt
```
Download the fine-tuned model (place in ./fine_tuned_sonnet_model folder):

Use Hugging Face Hub or your saved model snapshot.

## Usage
**Run the app:**
```
python app.py
# Open http://localhost:5000 in your browser.
```

Click to generate a sonnet.

**Code Structure:**
```
app.py: Main Flask application with model loading and poem generation logic.

index.html: Front-end template for the web interface.

requirements.txt: List of dependencies (Flask, Torch, Transformers, etc.).
```
**Limitations:**
- Requires GPU for efficient generation (CPU fallback available but slower).
- Model outputs may vary; fine-tuning details not included here.

Demo
Video: [LinkedIn post link](https://www.linkedin.com/feed/update/urn:li:activity:7324122776359915520/)
