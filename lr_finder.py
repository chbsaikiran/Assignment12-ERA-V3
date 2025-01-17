from torch_lr_finder import LRFinder
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torch
from transformer import Config, DecoderOnlyTransformer


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y


batches, no_of_tokens = 16, 128
train_loader = DataLoaderLite(B=batches, T=no_of_tokens)
steps_per_epoch = len(train_loader.tokens) // (batches * no_of_tokens)

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

amp_config = {
    'device_type': 'cuda',
    'dtype': torch.float16,
}
criterion = CrossEntropyLoss()
grad_scaler = torch.cuda.amp.GradScaler()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Define a custom batch fetching wrapper
class CustomDataLoader:
    def __init__(self, next_batch_func, num_batches):
        self.next_batch_func = next_batch_func
        self.num_batches = num_batches
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch < self.num_batches:
            self.current_batch += 1
            return self.next_batch_func()
        else:
            raise StopIteration

# Create a custom data loader using next_batch
custom_train_loader = CustomDataLoader(train_loader.next_batch(), num_batches=steps_per_epoch)

# Use the custom data loader with LRFinder
lr_finder = LRFinder(
    model, optimizer, criterion, device='cuda',
    amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
)
lr_finder.range_test(custom_train_loader, end_lr=5, num_iter=1000, step_mode='exp')
lr_finder.plot()
lr_finder.reset()
