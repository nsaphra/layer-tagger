import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from os.path import join, isfile

import data

class TaggerHook:
    def __init__(self, analyzer, key, module, output_vocab, save_prefix, hidden_size=None, dropout=0.5):
        self.module = module
        self.output_size = len(output_vocab)
        self.hidden_size = hidden_size
        self.input_size = None
        self.dropout = dropout

        self.tagger = None

        if hidden_size is None:
             self.hidden_size = self.output_size  # as specified in belinkov et al.

        self.key = key
        self.handle = None

        self.cumulative_eval_loss = 0
        self.num_correct = 0
        self.num_labeled = 0

        self.label_targets = None

        self.checkpoint = join('{}.{}.model'.format(save_prefix, self.key))
        self.best_validation_loss = None

        self.analyzer = analyzer

        #TODO inspect individual label performance

    def construct_model(self):
        self.tagger = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.tagger.cuda()
        self.tagger.train()

        print(self.key)
        print(self.tagger)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.tagger.parameters(), lr=0.001)

    def set_batch(self, label_targets):
        self.label_targets = label_targets

    def training_hook(self, layer, input, output):
        activation = self.process_activations(output, self.label_targets)
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
        activation = self.process_activations(output, self.label_targets)
        if activation is None:
            return

        self.tagger.requires_grad = False
        prediction = self.tagger(activation)
        prediction_flat = prediction.view(-1, self.output_size)
        self.num_labeled += self.label_targets.size(0)
        self.cumulative_eval_loss += self.label_targets.size(0) * self.criterion(prediction_flat, self.label_targets).data
        label_predictions = torch.max(prediction_flat.data, 1)[1]
        self.num_correct += (label_predictions == self.label_targets.data).sum()

    def process_activations(self, activations, sequence):
        if activations is None:
            return None

        if type(activations) is tuple:
            if type(activations[0]) is torch.cuda.FloatTensor:
                activations = torch.stack(activations, dim=0)
            else:
                for output in activations:
                    activation = self.process_activations(output, sequence)
                    if activation is not None:
                        return activation # use the first output that has the correct dimensions
                return None
        elif type(activations.data) is not torch.cuda.FloatTensor:
            return None

        if activations.dim() > 3:
            return None

        if activations.dim() == 3 and (activations.size(0) * activations.size(1)) == (sequence.size(0)):
            # activations: sequence_length x batch_size x hidden_size
            activations = activations.view(sequence.size(0), -1)

        if activations.size(0) != sequence.size(0):
            return None
        # activations: (sequence_length * batch_size) x hidden_size

        if self.input_size is None:
            self.input_size = activations.size(1)
            self.construct_model()

        # wrap activations in a new Variable to block backprop
        return Variable(activations.data)

    def register_hook(self, module, evaluation=True):
        if self.handle is not None:
            self.handle.remove()

            self.cumulative_eval_loss = 0
            self.num_correct = 0
            self.num_labeled = 0

            self.label_targets = None
        if evaluation:
            self.tagger.eval()
            self.handle = module.register_forward_hook(self.testing_hook)
        else:
            if self.tagger is not None:
                self.tagger.train()
            self.handle = module.register_forward_hook(self.training_hook)

    def save_model(self):
        with open(self.checkpoint, 'wb') as file:
            torch.save(self.tagger, file)

    def load_model(self):
        if not isfile(self.checkpoint):
            return
        with open(self.checkpoint, 'rb') as file:
            self.tagger = torch.load(file)

    def save_best_model(self):
        loss = self.compute_loss()
        if loss is None:
            return
        if self.best_validation_loss is None or loss < self.best_validation_loss:
            self.best_validation_loss = loss
            self.save_model()

    def compute_loss(self):
        if self.num_labeled is 0:
            return None
        return self.cumulative_eval_loss[0] / self.num_labeled

    def compute_accuracy(self):
        if self.num_labeled is 0:
            return None
        return 100 * self.num_correct / self.num_labeled

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
                self.hooks[module_key] = TaggerHook(self, module_key, module, self.output_vocab, self.save_prefix)

            # TODO use module.apply()
            self.hooks[module_key].register_hook(module, evaluation=self.evaluation)

    def save_best_taggers(self):
        # assumes that we have recently run a validation set, which is in the current results
        for key, tagger in self.hooks.items():
            tagger.save_best_model()

    def load_best_taggers(self):
        # run before running on official test set
        for key, tagger in self.hooks.items():
            tagger.load_model()

    def results_dict(self):
        for key, tagger in self.hooks.items():
            self.results[key] = {
                'loss': tagger.compute_loss(),
                'accuracy': tagger.compute_accuracy()
            }
        return self.results
