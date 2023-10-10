import os
import sys
import math
import random
import pickle
import transformers

import torch.optim as optim
from termcolor import colored
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F

n_experts = 16 # 64
devices_ids = [i for i in range(100)]
devices = [torch.device(id) for id in devices_ids]


def get_tokens(file_name):
    checkpoint = torch.load(file_name + '.pt')
    return checkpoint['tokens_processed']


def color(t, color):
    return colored(t, color) if sys.stdout.isatty() else t


def directory():
    return os.path.basename(os.path.dirname(os.path.realpath(__file__)))


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_dataset(file_name, tokenizer):
    encoded_file_name = file_name + '.encoded'

    if not os.path.exists(encoded_file_name):
        with open(file_name) as f:
            text = f.read()
        encoded_text = tokenizer.encode(text, add_special_tokens=False)
        pickle.dump(encoded_text, open(encoded_file_name, 'wb'))

    return torch.LongTensor(pickle.load(open(encoded_file_name, 'rb')))


def load_dataset_ascii(file_name):
    with open(file_name) as f:
        text = f.read()
    text = list(text.encode('ascii', 'ignore'))
    return torch.LongTensor(text)

    
def default_optimizer(model, training_configuration):
    no_decay = ['bias', 'layer_norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
         'weight_decay': training_configuration['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]
    return optim.AdamW(optimizer_grouped_parameters, lr=training_configuration['learning_rate'], betas=(0.9, 0.999))


def prepare_optimizer(model, load, training_configuration, file_name):
    optimizer = default_optimizer(model, training_configuration)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=training_configuration['num_warmup_steps'],
                                                             num_training_steps=training_configuration['num_training_steps'])
    file_name += '.pt'
    if load and os.path.exists(file_name):
        checkpoint = torch.load(file_name)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        tokens_processed = checkpoint['tokens_processed']
        total_time = checkpoint['total_time']
    else:
        tokens_processed = 0
        total_time = 0.0
        
    return optimizer, scheduler, tokens_processed, total_time


def save_checkpoint(model, optimizer, scheduler, file_name, model_configuration, training_configuration, tokens_processed, total_time):
    file_name += '.pt'
    time = timer()
    print('saving the model to', file_name)
    torch.save({'model' : model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_configuration': model_configuration,
                'training_configuration': training_configuration,
                'tokens_processed': tokens_processed,
                'total_time': total_time}, file_name)
    print('finished saving the model (took {:.5f} seconds)'.format(timer() - time))


def initialize_weights(module):
    if type(module) == nn.Linear or type(module) == nn.Embedding:
        module.weight.data.normal_(std=0.02)
    if type(module) == nn.Linear and module.bias is not None:
        module.bias.data.zero_()


def prepare_model(load, file_name='', model_configuration=None, model_class=None, version_name=None, strict=True):
    file_name += '.pt' if version_name is None else '_' + version_name + '.pt'
    load = load and os.path.exists(file_name)

    if load:
        checkpoint = torch.load(file_name)
        model_configuration = checkpoint['model_configuration']
        training_configuration = checkpoint['training_configuration']
    else:
        training_configuration = None
        
    if model_class is not None:
        model_configuration['model_class'] = model_class
    model = model_configuration['model_class'](model_configuration['vocabulary_size'],
                                               model_configuration['number_of_layers'],
                                               model_configuration['n_tokens'],
                                               model_configuration['dimension'],
                                               model_configuration['dropout'])
    
    if load:
        model.load_state_dict(checkpoint['model'], strict=strict)    
    else:
        model.apply(initialize_weights)
            
    # tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'transfo-xl-wt103')
    return model, model_configuration, training_configuration, transformers.BertTokenizer.from_pretrained('bert-base-uncased')


def test(model, encoded_validation_text, n_tokens, test_length, tokens_processed):
    print('measuring perplexity and accuracy')

    prompt = encoded_validation_text[0 : n_tokens].to(devices[0])
    perplexity = 0.0
    n_correct = 0

    model.eval()
    with torch.no_grad():
        for position in range(test_length):
            prediction, outputs_ids = model(prompt.unsqueeze(0))

            correct_token_id = encoded_validation_text[len(prompt) + position].item()
            perplexity += prediction[0, -1, correct_token_id].item()

            if outputs_ids[0, -1].item() == correct_token_id:
                n_correct += 1

            prompt = prompt.roll(-1)
            prompt[-1] = correct_token_id

            if (position + 1) % 100 == 0:
                print(f'{directory()}, {tokens_processed / 10 ** 9:.5f} billion tokens, {position + 1} / {test_length} = {100 * (position + 1) / test_length:.1f} % done, perplexity: {math.exp(- perplexity / (position + 1)):.2f}, accuracy: {n_correct}/{position + 1} = {100 * n_correct / (position + 1):.2f} %')

    perplexity = math.exp(- perplexity / test_length) # F.log_softmax uses the natural logarithm, so we use math.exp(x), which is the same as math.e ** x
    accuracy = 100 * n_correct / test_length
    print(f'perplexity: {perplexity:.5f}, accuracy: {accuracy:.5f}')

    return perplexity, accuracy




class AttentionHead(nn.Module):
    def __init__(self, dimension, dimension_of_attention, dropout_value, layer_id, number_of_layers, n_tokens):
        super().__init__()
        self.query = nn.Linear(dimension, dimension_of_attention)
        self.key   = nn.Linear(dimension, dimension_of_attention)
        self.value = nn.Linear(dimension, dimension_of_attention)
        self.dimension_of_attention = dimension_of_attention
        self.register_buffer('mask', torch.tril(torch.ones(n_tokens, n_tokens)).unsqueeze(0))
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        inner_products = torch.bmm(queries, keys.transpose(-2, -1)) / (math.sqrt(self.dimension_of_attention) * 12)
        inner_products = inner_products.masked_fill(self.mask == 0, float('-inf'))
        attention = torch.bmm(F.softmax(inner_products, dim=-1), values)
        return attention


class AttentionLayer(nn.Module):
    def __init__(self, dimension, dimension_of_attention, number_of_heads, dropout_value, layer_id, number_of_layers, n_tokens):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            AttentionHead(dimension, dimension_of_attention, 0.0, layer_id, number_of_layers, n_tokens) for _ in range(number_of_heads)
        ])

        self.layer_norm_1 = nn.LayerNorm(dimension)
        #self.dropout = nn.Dropout(dropout_value)
        self.linear_1 = nn.Linear(dimension, 4 * dimension)
        self.linear_2 = nn.Linear(4 * dimension, dimension)
        self.layer_norm_2 = nn.LayerNorm(dimension)

        self.scalar_1 = nn.Parameter(torch.zeros(1))
        self.scalar_2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_1 = [attention_head(x) for attention_head in self.attention_heads]
        x_1 = torch.cat(x_1, dim=-1)
        x_1 = self.layer_norm_1(x + x_1 * self.scalar_1)

        x_2 = self.linear_2(F.gelu(self.linear_1(x_1)))
        x_2 = self.layer_norm_2(x_1 + x_2 * self.scalar_2)
        return x_2

class Model_Attention(nn.Module):
    def __init__(self, vocabulary_size, number_of_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.position = nn.Parameter(torch.randn(n_tokens, dimension).unsqueeze(0))
        self.layers = nn.ModuleList([AttentionLayer(dimension, dimension // 12, 12, 0.0, layer_id, number_of_layers, n_tokens) for layer_id in range(number_of_layers)])
        self.linear = nn.Linear(dimension, vocabulary_size)

    def forward(self, x):
        x = self.embedding(x) + self.position
        for layer in self.layers:
            x = layer(x)
        x = F.log_softmax(self.linear(x), dim=-1)
        return x




class Layer(nn.Module):
    def __init__(self, dimension, n_tokens):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_tokens).unsqueeze(0).unsqueeze(0))
        self.register_buffer('scale', torch.FloatTensor([math.sqrt(1 / (i + 1)) for i in range(n_tokens)]).unsqueeze(-1).unsqueeze(0))
        self.norm_1 = nn.LayerNorm(dimension)

        self.linear_1 = nn.Linear(dimension, 4 * dimension) #4
        self.linear_2 = nn.Linear(4 * dimension, dimension) #4
        self.scalar = nn.Parameter(torch.zeros(1))
        self.norm_2 = nn.LayerNorm(dimension)

    def forward(self, x):
        n_tokens = self.weights.shape[-1]

        x_1 = x.transpose(-2, -1)
        # the next line implements convolution using fft
        x_1 = torch.fft.ifft(torch.fft.fft(x_1, n=2*n_tokens) * torch.fft.fft(self.weights, n=2*n_tokens)).real[..., :n_tokens]
        x_1 = x_1.transpose(-2, -1)

        x_1 = self.norm_1(x + x_1 * self.scale)
        x_2 = self.norm_2(x_1 + self.scalar * self.linear_2(F.gelu(self.linear_1(x_1))))
        return x_2


class Model(nn.Module):
    def __init__(self, vocabulary_size, number_of_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.layers = nn.ModuleList([Layer(dimension, n_tokens) for _ in range(number_of_layers)])
        self.linear = nn.Linear(dimension, vocabulary_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(self.linear(x), dim=-1)










class Mixing_Sum(nn.Module):
    def __init__(self, dimension, n_tokens):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)
        self.norm = nn.LayerNorm(dimension)
        self.scalar = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_1 = self.linear(x).cumsum(dim=1)
        return x + self.norm(x_1) * self.scalar


class Mixing(nn.Module):
    def __init__(self, dimension, n_tokens):
        super().__init__()
        self.n_tokens = n_tokens
        self.weights = nn.Parameter(torch.zeros(self.n_tokens).unsqueeze(0).unsqueeze(0))
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x):
        x_1 = x.transpose(-2, -1)
        x_1 = torch.fft.ifft(torch.fft.fft(x_1, n=2*self.n_tokens) * torch.fft.fft(self.weights, n=2*self.n_tokens)).real[..., :self.n_tokens]
        x_1 = x_1.transpose(-2, -1)
        x_1 = x + self.norm(x_1)
        return x_1