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
import transformers

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from timeit import default_timer as timer

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from transformers import glue_tasks_num_labels


task2folder = {'sts-b': 'STS-B', 'cola': 'CoLA', 'mnli': 'MNLI', 'qnli': 'QNLI', 'qqp': 'QQP', 'rte': 'RTE', 'sst-2': 'SST-2', 'wnli': 'WNLI', 'mrpc': 'MRPC'}
task2metric = {'sts-b': 'pearson', 'cola': 'mcc', 'mnli': 'mnli/acc', 'qnli': 'acc', 'qqp': 'acc', 'rte': 'acc', 'sst-2': 'acc', 'wnli': 'acc', 'mrpc': 'acc'}
task2lr = {'sts-b': 1e-3, 'sst-2': 1e-3}


configuration = {
    'load': True,
    'file_name': 'model',
    'print_iteration': 20,
    'save_iteration': 200
}

model_configuration = {
    'model_class': models.Model_Attention_Experts_Standard,
    'vocabulary_size': 32000,
    'n_tokens': 64,
    'number_of_layers': 12,
    'dimension': 768,
    'dropout': 0.1
}

training_configuration = {
    'learning_rate': task2lr.get(sys.argv[1], 1e-3),
    'weight_decay': 0.01,
    'train_epoch': 3,
    'train_warmup_ratio': 0.1,
    'train_batch_size': 32,
    'number_of_splits': 1,
    'eval_batch_size': 100,

    'data_dir': '/share/project/arturs/datasets/glue_data'
}
if 'AMLT_DATA_DIR' in os.environ:
    training_configuration['data_dir'] = os.environ['AMLT_DATA_DIR']

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)


def get_tokens_class(file_name):
    checkpoint = torch.load(file_name + '.pt')
    return checkpoint['tokens_processed'], checkpoint['model_configuration']['model_class']

    
def load_and_cache_examples(task, data_dir, n_tokens, tokenizer, evaluate=False):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            list(filter(None, data_dir.split('/'))).pop(),
            str(n_tokens),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        print('Loading features from cached file', cached_features_file)
        features = torch.load(cached_features_file, map_location=torch.device('cpu'))
    else:
        print('Creating features from dataset file at', data_dir)
        label_list = processor.get_labels()
        examples = (processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=n_tokens,
            output_mode=output_mode,
            #pad_on_left=False,  # pad on the left for xlnet
            #pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            #pad_token_segment_id=0,
        )
        print('Saving features into cached file %s', cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == 'classification':
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == 'regression':
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def validation(model, model_configuration, training_configuration, data_loader, output_mode, device):
    model.eval()
    with torch.no_grad():
        print('starting validation')
        preds = None
        out_label_ids = None
        for batch in data_loader:
            batch = tuple(t.to(device) for t in batch)

            outputs, _ = model(x=batch[0], input_mask=batch[1])

            if preds is None:
                preds = outputs.detach().cpu().numpy()
                out_label_ids = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, batch[3].detach().cpu().numpy(), axis=0)

        if output_mode == 'classification':
            preds = np.argmax(preds, axis=1)
        elif output_mode == 'regression':
            preds = np.squeeze(preds)
        result = compute_metrics(training_configuration['task_name'], preds, out_label_ids)

    return result[task2metric[sys.argv[1]]]


def finetune(model, configuration, model_configuration, training_configuration, tokenizer, tokens_processed):
    print('finetuning')
    if not torch.cuda.is_available():
        return None
    
    device = torch.device('cuda')
    model.to(device)
    optimizer = models.default_optimizer(model, training_configuration)

    pprint.pprint({'configuration' : configuration, 'model_configuration' : model_configuration, 'training_configuration' : training_configuration})
    print(f'number of all parameters: {models.count_all_parameters(model):,}   number of trainable parameters: {models.count_trainable_parameters(model):,}')

    train_dataset = load_and_cache_examples(training_configuration['task_name'], 
                                            training_configuration['data_dir'], 
                                            model_configuration['n_tokens'], 
                                            tokenizer,
                                            evaluate=False)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=training_configuration['train_batch_size'])
    
    t_total = training_configuration['train_epoch'] * len(train_dataloader)
    warmup_steps = math.floor(t_total * training_configuration['train_warmup_ratio'])
    print('total number of training steps', t_total, 'warmup steps', warmup_steps)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    eval_dataset = load_and_cache_examples(training_configuration['task_name'], 
                                           training_configuration['data_dir'], 
                                           model_configuration['n_tokens'], 
                                           tokenizer,
                                           evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=training_configuration['eval_batch_size'])

    output_mode = output_modes[training_configuration['task_name']]
    running_loss = 0.0
    running_accuracy = 0
    running_number_of_instances = 0
    best_result = 0.0
    running_mse = 0.0

    for epoch in range(training_configuration['train_epoch']):
        print('\n\nstarting epoch:', epoch + 1)

        for iteration, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            start_time = timer()

            optimizer.zero_grad()

            inputs_per_batch = len(batch[0]) // training_configuration['number_of_splits']
            number_of_splits = len(batch[0]) // inputs_per_batch
            for split in range(number_of_splits):
                start = split * inputs_per_batch
                end = start + inputs_per_batch

                outputs, loss = model(x=batch[0][start : end], labels=batch[3][start : end], input_mask=batch[1][start : end])
                running_loss += loss.item() * inputs_per_batch
                loss = loss / number_of_splits
                loss.backward()

                outputs_ids = outputs.argmax(dim=-1)
                running_accuracy += (batch[3][start : end].long() == outputs_ids).sum().item()
                if sys.argv[1] == 'sts-b':
                    running_mse += ((batch[3][start : end].float() - outputs) ** 2).sum().item()

                running_number_of_instances += inputs_per_batch

            optimizer.step()
            scheduler.step()

            if (iteration + 1) % configuration['print_iteration'] == 0:
                print('\n[{}, {}], {}, {}, time per batch: {:.3f}, {:.5f} billion tokens'.format(epoch + 1, iteration + 1, models.directory(), sys.argv[1], timer() - start_time, tokens_processed / 10 ** 9), flush=True)
                print('loss: {:.5f}, lr: {:.7f} ({}), best dev {}: {:.5f}'.format(running_loss / running_number_of_instances, scheduler.get_last_lr()[0], training_configuration['learning_rate'], task2metric[sys.argv[1]], best_result))
                if sys.argv[1] == 'sts-b':
                    print('mse = {:.5f}'.format(running_mse / running_number_of_instances))
                else:
                    print('accuracy {}/{} = {:.5f}'.format(running_accuracy, running_number_of_instances, running_accuracy / running_number_of_instances))
                running_loss = running_mse = 0.0
                running_accuracy = running_number_of_instances = 0
            if (iteration + 1) % configuration['save_iteration'] == 0 or (iteration + 1) % len(train_dataloader) == 0:
                result = validation(model, model_configuration, training_configuration, eval_dataloader, output_mode, device)
                text = 'iteration {} current ' .format(iteration + 1)
                text += models.color('dev {} {:.5f}'.format(task2metric[sys.argv[1]], result), 'cyan')
                text += ' current best dev {} {:.5f}'.format(task2metric[sys.argv[1]], best_result)
                print(text)
                if result > best_result:
                    best_result = result

    print(models.color('best dev {}: {:.5f}'.format(task2metric[sys.argv[1]], best_result), 'green'))


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'GPU is not available'
    print('starting execution', sys.argv[1])
    training_configuration['task_name'] = sys.argv[1]
    training_configuration['data_dir'] = os.path.join(training_configuration['data_dir'], task2folder[sys.argv[1]])

    if configuration['load']:
        print(models.color('using a pretrained model', 'green'))
        tokens_processed, model_class = get_tokens_class(configuration['file_name'])
    else:
        print(models.color('not using a pretrained model', 'red'))
        tokens_processed = 0
        model_class = model_configuration['model_class']


    class FinetuneModel(model_class):
        def __init__(self, vocabulary_size, number_of_layers, n_tokens, dimension, dropout):
            super(FinetuneModel, self).__init__(vocabulary_size, number_of_layers, n_tokens, dimension, dropout)
            self.num_labels = glue_tasks_num_labels[training_configuration['task_name']]
            self.linear_1 = nn.Linear(dimension, dimension)
            self.linear_2 = nn.Linear(dimension, self.num_labels)

        def forward(self, x, labels=None, input_mask=None):
            #x = self.embedding(x)
            x = self.embedding(x) + self.position


            #x = x * input_mask.unsqueeze(-1)

            for layer in self.layers:
                x = layer(x)

            x = (x * input_mask.unsqueeze(-1)).sum(1) / input_mask.sum(1, keepdim=True)
            #x = x.mean(1)

            x = self.linear_1(x)
            logits = self.linear_2(x.tanh())

            if labels is not None:
                if self.num_labels == 1:
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = None

            return logits, loss

    if configuration['load']:
        model, model_configuration, _, tokenizer, _ = models.prepare_model(load=True, file_name=configuration['file_name'], model_class=FinetuneModel, strict=False)
    else:
        model, model_configuration, _, tokenizer, _ = models.prepare_model(load=False, model_configuration=model_configuration, model_class=FinetuneModel)

    print('{:.5f} billion tokens processed'.format(tokens_processed / 10 ** 9))
    finetune(model, configuration, model_configuration, training_configuration, tokenizer, tokens_processed)
    