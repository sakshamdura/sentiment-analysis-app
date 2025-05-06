import torch
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertConfig, DistilBertForSequenceClassification

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load the fine-tuned encoder weights you saved earlier
encoder_state_dict = torch.load("./Fine-TunedModel/model_state.pth", map_location="cpu")  # adjust path if needed

# Step 1: Create a config for classification
config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 2: Create the classification model
model = DistilBertForSequenceClassification(config)

# Step 3: Load encoder weights into classification model
model.distilbert.load_state_dict(encoder_state_dict)

# ✅ Now you have a complete classification model
# Step 4: Save the model and tokenizer
model.save_pretrained("hf_distilbert_sentiment")
tokenizer.save_pretrained("hf_distilbert_sentiment")
