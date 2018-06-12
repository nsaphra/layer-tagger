import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from os.path import join

import data

class TaggerHook:
    def __init__(self, key, module, output_vocab, save_prefix, hidden_size=None, dropout=0.5):
        print(key)
        self.module = module
        self.input_size = NetworkLayerInvestigator.module_output_size(module)
        self.output_size = len(output_vocab)
        if hidden_size is None:
            hidden_size = self.output_size  # as specified in belinkov et al.

        self.tagger = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size)
        )
        self.tagger.cuda()

        print(self.tagger)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.tagger.parameters(), lr=0.001)

        self.key = key
        self.handle = None

        self.eval_loss = 0
        self.num_correct = 0
        self.num_labeled = 0

        self.label_targets = None

        self.checkpoint = join('{}.{}.model'.format(save_prefix, self.key))

        #TODO inspect individual label performance

    def set_batch(self, label_targets):
        self.label_targets = label_targets

    def training_hook(self, layer, input, output):
        for activation in output:
            activation = self.process_activations(activation, self.label_targets)
            if activation is not None:
                break  # use the first output that has the correct dimensions
        if activation is None:
            return

        self.tagger.requires_grad = True
        self.optimizer.zero_grad()
        prediction = self.tagger(activation)
        prediction_flat = prediction.view(-1, self.output_size)
        loss = self.criterion(prediction_flat, self.label_targets)
        loss.backward()
        self.optimizer.step()

    def testing_hook(self, layer, input, output):
        for activation in output:
            activation = self.process_activations(activation, self.label_targets)
            if activation is not None:
                break  # use the first output that has the correct dimensions
        if activation is None:
            return

        self.tagger.requires_grad = False
        label_output = self.tagger(output)
        label_predictions = torch.max(label_output.data, 1)
        self.num_labeled += label_targets.size(0)
        self.cumulative_loss += label_targets.size(0) * self.loss_function(label_output, self.label_targets).data
        self.num_correct += (label_output == self.label_targets).sum()

    def process_activations(self, activations, sequence):
        if activations is None:
            return None

        if type(activations) is tuple:
            activations = torch.stack(activations, dim=0)

        if type(activations.data) is not torch.cuda.FloatTensor:
            return None

        if activations.dim() == 3 and (activations.size(0) * activations.size(1)) == (sequence.size(0)):
            # activations: sequence_length x batch_size x hidden_size
            activations = activations.view(sequence.size(0), -1)
        if activations.size(0) != sequence.size(0) or activations.size(1) != self.input_size:
            return None
        # activations: (sequence_length * batch_size) x hidden_size

        # rapid activations in a new Variable to block backprop
        return Variable(activations.data)

    def register_hook(self, module, evaluation=True):
        if self.handle is not None:
            self.handle.remove()
        if evaluation:
            self.tagger.eval()
            self.handle = module.register_forward_hook(self.testing_hook)
        else:
            self.tagger.train()
            self.handle = module.register_forward_hook(self.training_hook)

    def save_model(self, dir):
        with open(self.checkpoint, 'wb') as file:
            torch.save(self.tagger, file)

class NetworkLayerInvestigator:
    def __init__(self, model, output_vocab, batch_size, bptt, save_prefix):
        self.hooks = {}
        self.model = model
        self.output_vocab = output_vocab
        self.batch_size = batch_size
        self.bptt = bptt
        self.results = {}

        self.evaluation = False

        self.next_batch = 0
        self.batch_hook = None

        self.save_prefix = save_prefix

    @staticmethod
    def module_output_size(module):
        # return the size of the final parameters in the module,
        # or 0 if there are no parameters
        output_size = 0
        for key, parameter in module.named_parameters():
            if key.find('weight') < 0:
                continue
            output_size = parameter.size(-1)
        return output_size

    def get_batch(self, module, input, output):
        i = self.next_batch
        self.next_batch += 1
        seq_len = min(self.bptt, len(self.data_source) - 1 - i)
        data = Variable(self.data_source[i:i+seq_len], volatile=self.evaluation)
        target = Variable(self.data_source[i+1:i+1+seq_len].view(-1))

        for key, handle in self.hooks.items():
            handle.set_batch(target)

    def set_label_batch(self, labels):
        for key, hook in self.hooks.items():
            hook.set_batch(labels)

    def add_model_hooks(self, evaluation=False):
        self.evaluation = evaluation
        for module_key, module in self.model.named_modules():
            output_size = self.module_output_size(module)
            if output_size == 0:
                continue

            if module_key not in self.hooks:
                self.hooks[module_key] = TaggerHook(module_key, module, self.output_vocab, self.save_prefix)

            # TODO use module.apply()
            self.hooks[module_key].register_hook(module, evaluation=self.evaluation)

    def results_dict(self):
        for key, tagger in self.hooks.items():
            self.results[key] = {
                'loss': tagger.cumulative_eval_loss / tagger.num_labeled,
                'accuracy': 100 * tagger.num_correct / tagger.num_labeled
            }
        return self.results
