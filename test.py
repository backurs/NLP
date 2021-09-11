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


configuration = {
    'load': True,
    'full_test': True,

    'train_data': '/share/project/arturs/datasets/wiki/wiki.train.tokens',
    'validation_data': '/share/project/arturs/datasets/wiki/wiki.valid.tokens',

    'file_name': 'model'
}


model_configuration = {
    'model_class': models.Model,
    'vocabulary_size': 32000,
    'length': 64,
    'number_of_layers': 12,
    'dimension': 768,
    'dropout': 0.1
}


def test(model, configuration, model_configuration, tokenizer, tokens_processed):
    print('testing the model')
    model.to(models.devices[0])

    pprint.pprint({'configuration' : configuration, 'model_configuration' : model_configuration})
    print(f'number of parameters: {models.count_all_parameters(model):,}')
    print(torch.cuda.device_count(), 'GPU(s) available')

    print('loading training data')
    time = timer()
    encoded_text = models.load_dataset(configuration['train_data'], tokenizer)
    print('finished loading training data')
    print('took {:.5f} seconds'.format(timer() - time))

    if 'validation_data' in configuration:
        encoded_validation_text = models.load_dataset(configuration['validation_data'], tokenizer)
    else:
        encoded_validation_text = encoded_text[-240000 : ]
        encoded_text = encoded_text[ : -240000]

    if configuration['full_test']:
        test_length = len(encoded_validation_text) - model_configuration['n_tokens']
    else:
        test_length = 1000

    time = timer()
    models.test(model, encoded_validation_text, model_configuration['n_tokens'], test_length, tokens_processed)
    print('took {:.5f} seconds'.format(timer() - time))

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'GPU is not available'
    print('starting execution', flush=True)
    if configuration['load']:
        model, model_configuration, _, tokenizer = models.prepare_model(load=True, file_name=configuration['file_name'])
        tokens_processed = models.get_tokens(configuration['file_name'])
    else:
        model, _, _, tokenizer = models.prepare_model(load=False, model_configuration=model_configuration)
        tokens_processed = 0

    print('{:.5f} billion tokens processed'.format(tokens_processed / 10 ** 9))
    test(model, configuration, model_configuration, tokenizer, tokens_processed)
