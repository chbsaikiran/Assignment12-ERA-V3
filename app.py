from transformer import Config, DecoderOnlyTransformer
import torch
from transformers import GPT2Tokenizer
import gradio as gr

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Model configuration
config = Config()

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer for compatibility

# Load trained model
model = DecoderOnlyTransformer(config)
model.load_state_dict(torch.load("decoder_only_transformer.pth", map_location=torch.device('cpu')))
model.eval()
model.to(device)

def predict_next_words(input_text, num_words=50):
    """
    Generate next words given input text.
    """
    tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated = tokens

    model.eval()
    with torch.no_grad():
        for _ in range(num_words):
            logits = model(generated)  # Model returns only logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1)  # Get the token with the highest probability
            generated = torch.cat((generated, next_token.unsqueeze(-1)), dim=1)

    output_text = tokenizer.decode(generated[0])
    return output_text



# Gradio interface
def gradio_interface(input_text):
    return predict_next_words(input_text)

# Build the app
interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your text here..."),
    outputs=gr.Textbox(lines=2, placeholder="Generated text will appear here..."),
    title="Next Word Prediction",
    description="Input some text, and the model will predict the next fifty words.",
)

# Launch the app
interface.launch()

