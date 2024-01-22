import sys
import subprocess
import pkg_resources
import warnings
warnings.filterwarnings('ignore')

# Source
# https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/transformers-marianmt.html


""""Install missing libraries."""
#libraries = {'torch', 'numpy', 'sentencepiece'}
#installed = {pkg.key for pkg in pkg_resources.working_set}
#missing = libraries - installed

# False if list is empty. Installs the missing python libraries.
#if missing:
#    python = sys.executable
#    if 'sentencepiece' in missing:
#        subprocess.check_call([sys.executable, "-m", "pip", "install", 'sentencepiece', 'transformers==4.26.1'], stdout = subprocess.DEVNULL)
#        missing.remove('sentencepiece')
#    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing], stdout = subprocess.DEVNULL)


model_name = "Helsinki-NLP/opus-mt-fi-en" # English -> German model
num_texts = 1                             # Number of input texts to decode
num_beams = 1                             # Number of beams per input text
max_encoder_length = 32                   # Maximum input token length
max_decoder_length = 32                   # Maximum output token length

data = open('./sanat3.txt', 'r').read()


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'Helsinki-NLP/opus-mt-fi-en'
model = MarianMTModel.from_pretrained(model_name).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


model_cpu = MarianMTModel.from_pretrained(model_name)
model_cpu.config.max_length = max_decoder_length
model_cpu.eval()

tokenizer = MarianTokenizer.from_pretrained(model_name)



import torch
from torch.nn import functional as F


class PaddedEncoder(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.main_input_name = 'input_ids'

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask=attention_mask, return_dict=False)


class PaddedDecoder(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.weight = model.model.shared.weight.clone().detach()
        self.bias = model.final_logits_bias.clone().detach()
        self.decoder = model.model.decoder

    def forward(self, input_ids, attention_mask, encoder_outputs, index):

        # Invoke the decoder
        hidden, = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            return_dict=False,
            use_cache=False,
        )

        _, n_length, _ = hidden.shape

        # Create selection mask
        mask = torch.arange(n_length, dtype=torch.float32) == index
        mask = mask.view(1, -1, 1)

        # Broadcast mask
        masked = torch.multiply(hidden, mask)

        # Reduce along 1st dimension
        hidden = torch.sum(masked, 1, keepdims=True)

        # Compute final linear layer for token probabilities
        logits = F.linear(
            hidden,
            self.weight,
            bias=self.bias
        )
        return logits



import os

from transformers import GenerationMixin, AutoConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.modeling_utils import PreTrainedModel


class PaddedGenerator(PreTrainedModel, GenerationMixin):

    @classmethod
    def from_model(cls, model):
        generator = cls(model.config)
        generator.encoder = PaddedEncoder(model)
        generator.decoder = PaddedDecoder(model)
        return generator

    def prepare_inputs_for_generation(
            self,
            input_ids,
            encoder_outputs=None,
            attention_mask=None,
            **kwargs,
    ):
        # Pad the inputs for Neuron
        current_length = input_ids.shape[1]
        pad_size = self.config.max_length - current_length
        return dict(
            input_ids=F.pad(input_ids, (0, pad_size)),
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs.last_hidden_state,
            current_length=torch.tensor(current_length - 1),
        )

    def get_encoder(self):
        def encode(input_ids, attention_mask, **kwargs):
            output, = self.encoder(input_ids, attention_mask)
            return BaseModelOutput(
                last_hidden_state=output,
            )
        return encode

    def forward(self, input_ids, attention_mask, encoder_outputs, current_length, **kwargs):
        logits = self.decoder(input_ids, attention_mask, encoder_outputs, current_length)
        return Seq2SeqLMOutput(logits=logits)

    @property
    def device(self):  # Attribute required by beam search
        return torch.device('cpu')

    def save_pretrained(self, directory):
        if os.path.isfile(directory):
            print(f"Provided path ({directory}) should be a directory, not a file")
            return
        os.makedirs(directory, exist_ok=True)
        torch.jit.save(self.encoder, os.path.join(directory, 'encoder.pt'))
        torch.jit.save(self.decoder, os.path.join(directory, 'decoder.pt'))
        self.config.save_pretrained(directory)

    @classmethod
    def from_pretrained(cls, directory):
        config = AutoConfig.from_pretrained(directory)
        obj = cls(config)
        obj.encoder = torch.jit.load(os.path.join(directory, 'encoder.pt'))
        obj.decoder = torch.jit.load(os.path.join(directory, 'decoder.pt'))
        setattr(obj.encoder, 'main_input_name', 'input_ids')  # Attribute required by beam search
        return obj



def infer(model, tokenizer, text):

    # Truncate and pad the max length to ensure that the token size is compatible with fixed-sized encoder (Not necessary for pure CPU execution)
    batch = tokenizer(text, max_length=max_decoder_length, truncation=True, padding='max_length', return_tensors="pt")
    output = model.generate(**batch, max_length=max_decoder_length, num_beams=num_beams, num_return_sequences=1)
    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]

    return results



padded_model_cpu = PaddedGenerator.from_model(model_cpu)
data_lista = data.split('€')


#kaannos = []
#for x in range(0, len(data_lista), 100):
for x in data_lista:
    #y = x + 100
    #tulos = infer(padded_model_cpu, tokenizer, data_lista[x:y])
    tulos = infer(padded_model_cpu, tokenizer, x)[0]
    f = open ('./kaannos2.txt', 'a')
    f.write(f'{tulos}\n')
    f.close()
#with open('./kaannos.txt', 'a') as f:
#    f.write(kaannos)
