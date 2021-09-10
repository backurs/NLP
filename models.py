import os
import sys
import math
import random
import pickle
import pathlib
import transformers

import numpy as np
import torch.optim as optim
import torch.nn.init as init
from termcolor import colored
import torch.utils.checkpoint as checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

n_experts = 64
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
    print('saving the model to', file_name)
    torch.save({'model' : model.state_dict(), 
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_configuration': model_configuration,
                'training_configuration': training_configuration,
                'tokens_processed': tokens_processed,
                'total_time': total_time}, file_name)


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


def perplexity(model, encoded_validation_text, n_tokens, test_length, tokens_processed):
    print('measuring perplexity')

    prompt = encoded_validation_text[0 : n_tokens].to(devices[0])
    perplexity = 0

    model.eval()
    with torch.no_grad():
        for position in range(test_length):
            prediction = model(prompt.unsqueeze(0))[0, -1]

            real_token_id = encoded_validation_text[len(prompt) + position].item()
            perplexity += prediction[real_token_id].item()

            prompt = prompt.roll(-1)
            prompt[-1] = real_token_id

            if (position + 1) % 100 == 0:
                print('{}, {:.5f} billion tokens, {} / {} = {:.1f} % done, perplexity: {:.2f}'.format(
                        directory(),
                        tokens_processed / 10 ** 9,
                        position + 1,
                        test_length,
                        100 * (position + 1) / test_length,
                        math.exp(- perplexity / (position + 1))
                    ))

    perplexity = math.exp(- perplexity / test_length) # F.log_softmax uses the natural logarithm, so we use math.exp(x), which is the same as math.e ** x
    print('perplexity:', perplexity)

    return perplexity




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








class Expert(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear_1 = nn.Linear(dimension, 4 * dimension)
        self.linear_2 = nn.Linear(4 * dimension, dimension)
        self.scalar = nn.Parameter(torch.zeros(1))
        self.layer_norm = nn.LayerNorm(dimension)

    def forward(self, x):
        x_1 = self.linear_2(F.gelu(self.linear_1(x)))
        x_1 = self.layer_norm(x + x_1 * self.scalar)
        return x_1


class Layer_Experts(nn.Module):
    def __init__(self, dimension, n_tokens):
        super().__init__()
        self.n_experts = 2

        self.weights = nn.Parameter(torch.zeros(n_tokens).unsqueeze(0).unsqueeze(0))
        self.register_buffer('scale', torch.FloatTensor([math.sqrt(1 / (i + 1)) for i in range(n_tokens)]).unsqueeze(-1).unsqueeze(0))
        self.norm = nn.LayerNorm(dimension)

        self.experts = nn.ModuleList([Expert(dimension) for i in range(self.n_experts)])

    def forward(self, x):
        n_tokens = x.shape[1]
        x_1 = x.transpose(-2, -1)
        # the next line implements convolution using fft
        x_1 = torch.fft.ifft(torch.fft.fft(x_1, n=2*n_tokens) * torch.fft.fft(self.weights, n=2*n_tokens)).real[..., :n_tokens]
        x_1 = x_1.transpose(-2, -1)
        x_1 = self.norm(x + x_1 * self.scale)

        inputs = torch.chunk(x_1, chunks=self.n_experts, dim=1)
        outputs = [self.experts[i](inputs[i]) for i in range(self.n_experts)]
        x_1 = torch.cat(outputs, dim=1)
        
        return x_1


class Model_Experts(nn.Module):
    def __init__(self, vocabulary_size, n_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.layers = nn.ModuleList([Layer_Experts(dimension, n_tokens) for _ in range(n_layers)])
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
        self.register_buffer('scale', torch.FloatTensor([math.sqrt(1 / (i + 1)) for i in range(self.n_tokens)]).unsqueeze(-1).unsqueeze(0))
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x):
        x_1 = x.transpose(-2, -1)
        x_1 = torch.fft.ifft(torch.fft.fft(x_1, n=2*self.n_tokens) * torch.fft.fft(self.weights, n=2*self.n_tokens)).real[..., :self.n_tokens]
        x_1 = x_1.transpose(-2, -1)
        x_1 = self.norm(x + x_1 * self.scale)
        return x_1


class Layer_Experts_Parallel(nn.Module):
    def __init__(self, dimension, n_tokens):
        super().__init__()
        self.n_devices = torch.cuda.device_count()

        self.mixing = Mixing(dimension, n_tokens).to(devices[0]) # Mixing
        self.mixing = nn.DataParallel(self.mixing)
        self.experts = nn.ModuleList([Expert(dimension).to(devices[i % self.n_devices]) for i in range(n_experts)])

    def forward(self, x):
        x = self.mixing(x)

        inputs = torch.chunk(x, chunks=n_experts, dim=1)
        inputs = [inputs[i].to(devices[i % self.n_devices]) for i in range(n_experts)]
        outputs = [self.experts[i](inputs[i]) for i in range(n_experts)]
        outputs = [output.to(devices[0]) for output in outputs]
        x = torch.cat(outputs, dim=1)

        return x

class Model_Experts_Parallel(nn.Module):
    def __init__(self, vocabulary_size, n_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension).to(devices[0])
        self.layers = nn.ModuleList([Layer_Experts_Parallel(dimension, n_tokens) for _ in range(n_layers)])
        self.linear = nn.Linear(dimension, vocabulary_size).to(devices[0])

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(self.linear(x), dim=-1)

class Compute_Loss(nn.Module):
    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.linear = nn.Linear(dimension, self.vocabulary_size)
    def forward(self, x, labels=None):
        x = F.log_softmax(self.linear(x), dim=-1)
        if labels == None:
            return x
        labels_loss = F.one_hot(labels, self.vocabulary_size).float()
        loss = - (x * labels_loss).sum(dim=0)
        return loss, x.argmax(dim=-1)

class Model_Experts_Parallel_Compute_Loss(nn.Module):
    def __init__(self, vocabulary_size, n_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension).to(devices[0])
        self.layers = nn.ModuleList([Layer_Experts_Parallel(dimension, n_tokens) for _ in range(n_layers)])
        self.loss = Compute_Loss(vocabulary_size, dimension).to(devices[0])
        self.loss = nn.DataParallel(self.loss)

    def forward(self, x, labels=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        if labels == None:
            return self.loss(x)
        loss, outputs_ids = self.loss(x, labels)
        return loss.sum(), outputs_ids






class AttentionHead_Experts(nn.Module):
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


class AttentionLayer_Experts(nn.Module):
    def __init__(self, dimension, dimension_of_attention, number_of_heads, dropout_value, layer_id, number_of_layers, n_tokens):
        super().__init__()
        self.n_experts = 4

        self.attention_heads = nn.ModuleList([
            AttentionHead_Experts(dimension, dimension_of_attention, 0.0, layer_id, number_of_layers, n_tokens) for _ in range(number_of_heads)
        ])
        self.scalar = nn.Parameter(torch.zeros(1))
        self.layer_norm = nn.LayerNorm(dimension)

        self.experts = nn.ModuleList([Expert(dimension) for i in range(self.n_experts)])

    def forward(self, x):
        x_1 = [attention_head(x) for attention_head in self.attention_heads]
        x_1 = torch.cat(x_1, dim=-1)
        x_1 = self.layer_norm(x + x_1 * self.scalar)

        inputs = torch.chunk(x_1, chunks=self.n_experts, dim=1)
        outputs = [self.experts[i](inputs[i]) for i in range(self.n_experts)]
        x_1 = torch.cat(outputs, dim=1)
        
        return x_1

class Model_Attention_Experts(nn.Module):
    def __init__(self, vocabulary_size, number_of_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.position = nn.Parameter(torch.randn(n_tokens, dimension).unsqueeze(0))
        self.layers = nn.ModuleList([AttentionLayer_Experts(dimension, dimension // 12, 12, 0.0, layer_id, number_of_layers, n_tokens) for layer_id in range(number_of_layers)])
        self.linear = nn.Linear(dimension, vocabulary_size)

    def forward(self, x):
        x = self.embedding(x) + self.position
        for layer in self.layers:
            x = layer(x)
        x = F.log_softmax(self.linear(x), dim=-1)
        return x








class MoE(nn.Module):
    def __init__(self, input_size, output_size, generate_expert_fnc, num_experts=16, k=4, gating_policy="plain"):
        super().__init__()
        self.input_size = input_size # dimension of word embeddings
        self.output_size = output_size # output dimension of each expert
        self.num_experts = num_experts
        self.k = k
        self.generate_expert_fnc = generate_expert_fnc
        self.gating_policy = gating_policy # currently plain or noisy

        self.experts = nn.ModuleList([generate_expert_fnc() for _ in range(self.num_experts)])

        self.w_gate = nn.Parameter(torch.randn(self.input_size, self.num_experts)*1./100, requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)


    def forward(self, x):
        # Gating
        if self.gating_policy == "plain":
            logits = torch.matmul(x, self.w_gate)

            logits = torch.zeros_like(logits)
            
            logits[..., 1] = 100.0

            #logits = logits + F.one_hot(torch.randint(0, self.num_experts, x.shape[:2]).cuda()) * 100.0

            #print(logits)
        elif self.gating_policy == "noisy":
            clean_logits = torch.matmul(x, self.w_gate)
            raw_noise_stddev = torch.matmul(x, self.w_noise)
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            #print("NOISE STDDV:::::", noise_stddev)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        elif self.gating_policy == "random":
            logits = torch.randn(x.size()[0], x.size()[1], self.num_experts).cuda()
        else:
            raise Exception("Unrecognized gating policy")

        top_logits, top_indices = logits.topk(min(self.k, self.num_experts), dim=2)
        top_k_logits = top_logits[:, :, :self.k]
        top_k_indices = top_indices[:, :, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(2, top_k_indices, top_k_gates)
        # expert_loads = list((gates > 0).sum(0).numpy())
        # importance_loss = gates.sum(0)

        # Define output tensor
        output = torch.zeros((self.num_experts, x.size()[0], x.size()[1], self.output_size), requires_grad=True).cuda()

        # Create batches for experts
        num_active_experts = 0 # Just for collecting data, not used in code

        to_print = random.randint(1,200) == 1

        for iex in range(self.num_experts):
            # Create batch for expert
            expert_loads = [] # Same, just collecting info
            channels = []
            channel_indices = {}
            channel_start = 0
            max_channel_length = 0
            for b in range(x.shape[0]):
                expert_indices = torch.nonzero(gates[b, :, iex])
                channel_length = len(expert_indices.squeeze(-1))
                expert_loads.append(channel_length)
                if channel_length > 0:
                    channels.append(x[b, expert_indices.squeeze(-1), :])
                    channel_indices[b] = (channel_start, channel_start + channel_length)
                    channel_start += channel_length
                    max_channel_length = max(max_channel_length, channel_length)
                #else:
                    #print("ONE CHANNEL MISSING")
                    #channels.append(torch.zeros(max_channel_length, self.output_size).cuda())
            if to_print:
                print("Expert " + str(iex) + " loads: " + str(expert_loads))
            if max_channel_length == 0 or len(channels)==0:
                continue
            num_active_experts += 1
            expert_batch = torch.cat(channels)
            # Send batch to expert
            expert_output = self.experts[iex].forward(expert_batch)
            # Combine expert output into output
            for b in range(x.shape[0]):
                if b not in channel_indices:
                    continue
                expert_indices = torch.nonzero(gates[b, :, iex])
                expanded_expert_output = torch.zeros((x.size()[1], self.output_size), requires_grad=True).cuda()
                expanded_indices = expert_indices.repeat(1, self.output_size)
                #print("~~~", expanded_expert_output.size(), expanded_indices.size(), expert_output[channel_indices[b][0]:channel_indices[b][1]].size())
                expanded_expert_output.scatter_(0, expanded_indices, expert_output[channel_indices[b][0]:channel_indices[b][1],:])
                expert_weights = gates[b, :, iex]
                output[iex,b,:,:] = expanded_expert_output * expert_weights.unsqueeze(-1)

        #print("NUMBER OF ACTIVE EXPERTS WAS " + str(num_active_experts) + " out of " + str(self.num_experts))
        return torch.sum(output, dim=0)


class Layer_Experts_Standard(nn.Module):
    def __init__(self, dimension, n_tokens):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)
        self.weights = nn.Parameter(torch.zeros(n_tokens).unsqueeze(0).unsqueeze(0))
        self.register_buffer('scale', torch.FloatTensor([math.sqrt(1 / (i + 1)) for i in range(n_tokens)]).unsqueeze(-1).unsqueeze(0))
        self.norm_1 = nn.LayerNorm(dimension)

        self.moe = MoE(dimension, dimension, lambda: Expert(dimension), 2, 1)

    def forward(self, x):
        n_tokens = x.shape[1]
        x_1 = x.transpose(-2, -1)
        # the next line implements convolution using fft
        x_1 = torch.fft.ifft(torch.fft.fft(x_1, n=2*n_tokens) * torch.fft.fft(self.weights, n=2*n_tokens)).real[..., :n_tokens]
        x_1 = x_1.transpose(-2, -1)
        x_1 = self.norm_1(x + x_1 * self.scale)

        return self.moe(x_1)


class Model_Experts_Standard(nn.Module):
    def __init__(self, vocabulary_size, n_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.layers = nn.ModuleList([Layer_Experts_Standard(dimension, n_tokens) for _ in range(n_layers)])
        self.linear = nn.Linear(dimension, vocabulary_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(self.linear(x), dim=-1)


class AttentionHead_Experts_Standard(nn.Module):
    def __init__(self, dimension, dimension_of_attention, layer_id, number_of_layers, n_tokens):
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


class AttentionLayer_Experts_Standard(nn.Module):
    def __init__(self, dimension, dimension_of_attention, number_of_heads, layer_id, number_of_layers, n_tokens):
        super().__init__()
        self.attention_heads = nn.ModuleList([
            AttentionHead_Experts_Standard(dimension, dimension_of_attention, layer_id, number_of_layers, n_tokens) for _ in range(number_of_heads)
        ])

        self.scalar = nn.Parameter(torch.zeros(1))
        self.layer_norm = nn.LayerNorm(dimension)

        self.moe = MoE(dimension, dimension, lambda: Expert(dimension), 2, 1)


    def forward(self, x):
        x_1 = [attention_head(x) for attention_head in self.attention_heads]
        x_1 = torch.cat(x_1, dim=-1)
        x_1 = self.layer_norm(x + x_1 * self.scalar)

        return self.moe(x_1)

class Model_Attention_Experts_Standard(nn.Module):
    def __init__(self, vocabulary_size, number_of_layers, n_tokens, dimension, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.position = nn.Parameter(torch.randn(n_tokens, dimension).unsqueeze(0))
        self.layers = nn.ModuleList([AttentionLayer_Experts_Standard(dimension, dimension // 12, 12, layer_id, number_of_layers, n_tokens) for layer_id in range(number_of_layers)])
        self.linear = nn.Linear(dimension, vocabulary_size)

    def forward(self, x):
        x = self.embedding(x) + self.position
        for layer in self.layers:
            x = layer(x)
        x = F.log_softmax(self.linear(x), dim=-1)
        return x