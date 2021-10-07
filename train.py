print('importing modules', flush=True)

from timeit import default_timer as timer

experiment_start_time = timer()

import models

import os
import sys
import torch
import pprint
import random
import argparse

import numpy as np
import torch.nn as nn
import plotext as plt
import torch.nn.functional as F

configuration = {
    'load': False,

    'print_iteration': 20, # 20
    'save_iteration': 3000
}

model_configuration = {
    'model_class': models.Model_Experts_Parallel_Compute_Loss,
    #'model_class': models.Model_Experts_Parallel,
    #'model_class': models.Model_Experts,
    #'model_class': models.Model_Attention_Experts,
    #'model_class': models.Model_Attention_Experts_Standard,
    #'model_class': models.Model,
    #'model_class': models.Model_Experts_Standard,
    'vocabulary_size': 32000, # 32000
    'n_tokens': 64 * 2, # 64 * 2, # 64
    'number_of_layers': 24, # 36, # 12
    'dimension': 27 * 64, # 768
    'dropout': 0.0
}

training_configuration = {
    'batch_size': 64 * 4, # 64
    'number_of_splits': 1, # 1
    'weight_decay': 0.01,
    'learning_rate': 1e-3, # 1e-3
    'num_warmup_steps': 100 # 100
}
tokens_for_experiment = 10 ** 10
training_configuration['num_training_steps'] = tokens_for_experiment // (model_configuration['n_tokens'] * training_configuration['batch_size'])
# the total number of batches in training

parser = argparse.ArgumentParser()
parser.add_argument('--load', default=configuration['load'], action='store_true')
parser.add_argument('--number_of_splits', default=training_configuration['number_of_splits'], type=int)
parser.add_argument('--amlt', action='store_true')
args = parser.parse_args()

configuration['load'] = args.load
training_configuration['number_of_splits'] = args.number_of_splits
if args.amlt:
    configuration['train_data'] = os.path.join(os.environ['AMLT_DATA_DIR'],'wiki.train.tokens')
    configuration['validation_data'] = os.path.join(os.environ['AMLT_DATA_DIR'],'wiki.valid.tokens')
    configuration['file_name'] = os.path.join(os.environ['AMLT_OUTPUT_DIR'], 'model')
else:
    configuration['train_data'] = '/workspace/data/wikitext-103/wiki.train.tokens'
    configuration['validation_data'] = '/workspace/data/wikitext-103/wiki.valid.tokens'
    configuration['file_name'] = 'model'


def print_input_output(labels, outputs_ids, tokenizer):
    print('---')
    for token in range(len(labels)):
        true_token = tokenizer.decode([labels[token]])
        output_token = tokenizer.decode([outputs_ids[token]])

        if true_token == output_token:
            t = '[{}]'.format(true_token)
            t = models.color(t, 'green')
        else:
            t = '[{}->{}]'.format(true_token, output_token)
            t = models.color(t, 'red')
        print(t, end=' ')
    print('\n---')

def print_accuracy_plot(accuracy_file_name):
    if not sys.stdout.isatty():
        return
    print('accuracy plot:')
    tokens = []
    accuracy = []
    with open(accuracy_file_name) as file:
        for line in file:
            x, y = map(float, line.split('\t'))
            tokens += [x]
            accuracy += [y]
    plt.scatter(tokens, accuracy)
    plt.plotsize(90, 30)
    plt.colorless()
    plt.show()

def train():
    print(models.color('training (each token depends on itself and the previous tokens)', 'green'), flush=True)
    
    global model_configuration
    global training_configuration

    model, model_configuration, train_conf, tokenizer = models.prepare_model(load=configuration['load'], file_name=configuration['file_name'], model_configuration=model_configuration)
    if train_conf is not None:
        training_configuration = train_conf
    optimizer, scheduler, tokens_processed, total_time = models.prepare_optimizer(model, load=configuration['load'], training_configuration=training_configuration, file_name=configuration['file_name'])
            
    pprint.pprint({'configuration' : configuration, 'model_configuration' : model_configuration, 'training_configuration' : training_configuration})
    print(f'number of parameters: {models.count_all_parameters(model):,}')

    print('loading the training data')
    time = timer()
    encoded_text = models.load_dataset(configuration['train_data'], tokenizer)
    print(f'finished loading the training data (took {timer() - time:.5f} seconds)')

    if 'validation_data' in configuration:
        encoded_validation_text = models.load_dataset(configuration['validation_data'], tokenizer)
    else:
        encoded_validation_text = encoded_text[-240000 : ]
        encoded_text = encoded_text[ : -240000]

    accuracy_file_name = configuration['file_name'] + '.accuracy'
    if tokens_processed == 0 and os.path.exists(accuracy_file_name):
        os.remove(accuracy_file_name)

    perplexity_file_name = configuration['file_name'] + '.perplexity'
    if tokens_processed == 0 and os.path.exists(perplexity_file_name):
        os.remove(perplexity_file_name)

    running_loss = 0.0
    running_accuracy = 0
    number_of_ids = len(encoded_text)
    ids_per_batch = training_configuration['batch_size'] * model_configuration['n_tokens']
    number_of_iterations = number_of_ids // ids_per_batch - 1

    done = False
    for epoch in range(100):
        print('\n\nstarting epoch:', epoch + 1)

        permutation = np.random.permutation(number_of_iterations)

        for iteration in range(number_of_iterations):
            if tokens_processed >= tokens_for_experiment:
                done = True
                break
            
            start_time = timer()

            model.train()
            optimizer.zero_grad()

            random_shift = random.randint(0, model_configuration['n_tokens'] - 1)

            for split in range(training_configuration['number_of_splits']):
                ids_per_split = ids_per_batch // training_configuration['number_of_splits']
                start_position = ids_per_batch * permutation[iteration] + split * ids_per_split + random_shift
                end_position = start_position + ids_per_split

                inputs = encoded_text[start_position : end_position].to(models.devices[0])
                inputs = inputs.contiguous().view(-1, model_configuration['n_tokens'])

                labels = encoded_text[start_position + 1 : end_position + 1].to(models.devices[0])
                labels = labels.contiguous().view(-1, model_configuration['n_tokens'])

                loss, outputs_ids = model(inputs, labels)

                running_loss += loss.item()
                loss = loss / ids_per_split
                loss.backward()

                running_accuracy += (labels == outputs_ids).sum().item()

            optimizer.step()
            scheduler.step()

            tokens_processed += ids_per_batch
            if (iteration + 1) % configuration['print_iteration'] == 0:
                ids_per_print = configuration['print_iteration'] * ids_per_batch

                print('\n[{}, {}]'.format(epoch + 1, iteration + 1), flush=True)
                text = f'accuracy: {running_accuracy}/{ids_per_print} = '
                text += models.color(f'{100 * running_accuracy / ids_per_print:.5f} %', 'cyan')
                text += f', loss per token: {running_loss / ids_per_print:.5f}'
                print(text)
                memory = [f'{torch.cuda.max_memory_allocated(i) / 2 ** (10 * 3):.5f} GB' for i in range(torch.cuda.device_count())]
                memory = ', '.join(memory)
                if torch.cuda.device_count() >= 2:
                    memory = '(' + memory + ')'
                total_days = (total_time + timer() - experiment_start_time) / (60 * 60 * 24)
                print('time for the last batch: {:.5f}, memory: {}, lr: {:.7f}, tokens: {:,}, total time: {:.6f} days'.format(timer() - start_time, memory, scheduler.get_last_lr()[0], tokens_processed, total_days))
                with open(accuracy_file_name, 'a') as accuracy_file:
                    accuracy_file.write(f'{tokens_processed / 10 ** 6:.5f}\t{running_accuracy / ids_per_print:.5f}\n')
                print_input_output(labels[0], outputs_ids[0], tokenizer)
                running_loss = 0.0
                running_accuracy = 0
            if (iteration + 1) % configuration['save_iteration'] == 0:
                print_accuracy_plot(accuracy_file_name)
                perplexity, _ = models.test(model, encoded_validation_text, model_configuration['n_tokens'], 1000, tokens_processed)
                with open(perplexity_file_name, 'a') as perplexity_file:
                    perplexity_file.write('{:.5f}\t{:.5f}\n'.format(tokens_processed / 10 ** 6, perplexity))
                time = timer() - experiment_start_time
                models.save_checkpoint(model, optimizer, scheduler, configuration['file_name'], model_configuration, training_configuration, tokens_processed, total_time + time)
                if time / (60 * 60) > 10 ** 10:
                    done = True
                    break

        if done:
            break


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'GPU is not available'
    print(torch.cuda.device_count(), 'GPU(s) available')

    train()