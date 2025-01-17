import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import gradio as gr

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Using GPT-2 tokenizer for compatibility

# Load model
from transformer import GPT, GPTConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model
config = GPTConfig()
model = GPT(config)
model.load_state_dict(torch.load("decoder_only_transformer.pth", map_location=torch.device(device)))
model.eval()
model.to(device)

# Prediction function
def generate_text(input_text, max_length=50, top_k=50, num_sequences=1):
    results = []
    with torch.no_grad():
        for _ in range(num_sequences):
            tokens = tokenizer.encode(input_text, return_tensors="pt").to(device)
            x = tokens
            while x.size(1) < max_length:
                logits = model(x)[0]
                logits = logits[:, -1, :]

                # Top-k sampling
                if top_k > 0:
                    logits_top_k, indices_top_k = torch.topk(logits, k=top_k, dim=-1)
                    probs = F.softmax(logits_top_k, dim=-1)
                    sampled_index = torch.multinomial(probs, 1)
                    next_token = torch.gather(indices_top_k, -1, sampled_index)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)

                x = torch.cat((x, next_token), dim=1)

            generated_text = tokenizer.decode(x[0].tolist(), skip_special_tokens=True)
            results.append(generated_text)
    return results

# Gradio Interface
def gradio_interface(input_text, max_length, num_sequences):
    sequences = generate_text(input_text, max_length=max_length, num_sequences=num_sequences)
    return "\n\n".join([f"Sequence {i+1}: {seq}" for i, seq in enumerate(sequences)])

interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your text here...", label="Input Text"),
        gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Max Length"),
        gr.Slider(minimum=1, maximum=5, step=1, value=2, label="Number of Sequences")
    ],
    outputs=gr.Textbox(lines=5, placeholder="Generated text will appear here...", label="Generated Text"),
    title="Next Word Generator Trained On Shakespeare's Text",
    description="Enter a text prompt. Adjust the max sequences of words. Adjust the max length",
)

# Launch the app
interface.launch()
