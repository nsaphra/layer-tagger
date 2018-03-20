import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import data

class LayerTagger(nn.Module):
    def __init__(self, key, input_size, output_vocab, hidden_size=None, dropout=0.5):
        super(LayerTagger, self).__init__()
        if hidden_size is None:
            hidden_size = input_size  # as specified in belinkov et al.
        output_size = len(output_vocab.idx2word)
        self.encode = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.decode = nn.Softmax()
        self.loss_function = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        self.key = key
        self.handle = None

        self.eval_loss = 0
        self.num_correct = 0
        self.num_labeled = 0

        #TODO inspect individual label performance

    def forward(self, x):
        return self.decode(self.drop(self.relu(self.encode(x))))

    def training_hook(self, layer, input, output):
        self.zero_grad()
        loss = self.loss_function(self(output), label_targets)
        loss.backward()
        self.optimizer.step()

    def testing_hook(self, layer, input, output):
        label_output = self(output)
        label_predictions = torch.max(label_output.data, 1)
        self.num_labeled += label_targets.size(0)
        self.cumulative_loss += label_targets.size(0) * self.loss_function(label_output, label_targets).data
        self.num_correct += (label_output == label_targets).sum()


    def dummy_hook(self, layer, input, output):
        print(layer)
        print(NetworkLayerInvestigator.module_output_size(layer))

    def register_hook(self, module, evaluation=True):
        if evaluation:
            self.eval()
            self.handle.remove()
            # self.handle = module.register_forward_hook(self.testing_hook)
            self.handle = module.register_forward_hook(self.dummy_hook)
        else:
            self.train()
            # self.handle = module.register_forward_hook(self.training_hook)
            self.handle = module.register_forward_hook(self.dummy_hook)

class NetworkLayerInvestigator:
    def __init__(self, model, input_vocab, output_vocab, batch_size, bptt):
        self.hooks = {}
        self.model = model
        self.output_vocab = output_vocab
        self.input_vocab = input_vocab
        self.batch_size = batch_size
        self.bptt = bptt
        self.results = {}

        self.evaluation = False

        self.next_batch = 0
        self.batch_hook = None

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
        label_targets = target

    def add_hooks_recursively(self, parent_module: nn.Module, prefix=''):
        # add hooks to the modules in a network recursively
        for module_key, module in parent_module.named_children():
            module_key = prefix + module_key
            output_size = self.module_output_size(module)
            # print("output_size", output_size)
            if output_size == 0:
                continue
            # if self.evaluation:
            self.hooks[module_key] = LayerTagger(module_key, self.batch_size, self.output_vocab)
            self.hooks[module_key].cuda()

            self.hooks[module_key].register_hook(module, evaluation=self.evaluation)
            self.add_hooks_recursively(module, prefix=module_key)
        # alternatively we could have a special case for:
        #     if torch.nn.modules.rnn.RNNBase in type(module).__bases__:
        # in which case we would create a module for each rnn layer

    def add_model_hooks(self, data_source, evaluation=False):
        self.evaluation = evaluation
        self.model.eval()

        if self.batch_hook is not None:
            self.batch_hook.remove()
        self.next_batch = 0
        self.data_source = data_source
        self.batch_hook = self.model.register_forward_hook(self.get_batch)

        self.add_hooks_recursively(self.model)

    def results_dict(self):
        for key, tagger in self.hooks.items():
            self.results[key] = {
                'loss': tagger.cumulative_eval_loss / tagger.num_labeled,
                'accuracy': 100 * tagger.num_correct / tagger.num_labeled
            }
        return self.results
