print('importing modules', flush=True)

import models

import os
import sys
import math
import torch
import pprint
import random
import logging
import pathlib
import collections
import transformers

import numpy as np
import torch.nn as nn
import plotext.plot as plx
import torch.optim as optim
from termcolor import colored
import torch.nn.functional as F
from timeit import default_timer as timer
import torch.distributions.categorical as categorical


configuration = {
    #'train_data': '/share/project/arturs/datasets/bookcorpus/encoded_text.p',

    'train_data': '/share/project/arturs/datasets/wiki/encoded_text.p',
    'validation_data': '/share/project/arturs/datasets/wiki/encoded_validation_text.p',

    'support_size': 100,
    'probability_threshold': 0.9,

    'file_name': 'model'
}


def generate(model, configuration, model_configuration, tokenizer):
    print('generation')
    device = torch.device('cuda')
    model.to(device)

    pprint.pprint({'configuration' : configuration, 'model_configuration' : model_configuration})
    print(f'number of all parameters: {models.count_all_parameters(model):,}   number of trainable parameters: {models.count_trainable_parameters(model):,}')
    print(torch.cuda.device_count(), 'GPU(s) available')

    print('loading training data')
    time = timer()
    encoded_text = models.load_dataset(configuration['train_data'])
    print('finished loading training data')
    print('took {:.5f} seconds'.format(timer() - time))

    if 'validation_data' in configuration:
        encoded_validation_text = models.load_dataset(configuration['validation_data'])
    else:
        encoded_validation_text = encoded_text[-240000 : ]
        encoded_text = encoded_text[ : -240000]

    prompt = encoded_validation_text[0 : model_configuration['length']]

    print('prompt:')
    print(tokenizer.decode(prompt[ : -1]))
    print('------------')

    samples = []

    model.eval()
    with torch.no_grad():
        for _ in range(1000):
            prompt = prompt.roll(-1)
            prompt[-1] = 1

            logits = model(prompt.unsqueeze(0))[0, -1]

            #indices_to_remove = logits < logits.topk(configuration['support_size']).values[-1]
            #logits[indices_to_remove] = - 1000
            #logits = logits.log_softmax(-1)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > configuration['probability_threshold']
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = - 1000

            sample = categorical.Categorical(logits=logits).sample()

            samples.append(sample)

            prompt[-1] = sample

    print(tokenizer.decode(samples))

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'GPU is not available'
    print('starting execution', flush=True)
    model, model_configuration, _, tokenizer, _ = models.prepare_model(load=True, file_name=configuration['file_name'])

    generate(model, configuration, model_configuration, tokenizer)