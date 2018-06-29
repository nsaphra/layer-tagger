# layer-tagger
generic pytorch implementation of the layer wise representation evaluation from Belinkov et al. 2018

This will take a pytorch language model as input, along with a corpus and corresponding tag data. 
It will attach a hook to each module in the model, making it easy to inspect the performance of activations 
at each layer for the tagging tasks. 

```
python src/main.py --checkpoint saved_language_model.model --cuda --vocab-file new_line_separated_vocab_list.txt
    --train-file data/train --valid-file  data/dev --test-file data/test --original-src language_model_code/ --batch-size 1
    --save-prefix intermediate_model --epochs 20 --save-results tagged.json
```

This command would load in `saved_language_model.model`, as produced by classes stored in the directory `language_model_code/`.
It assumes there is a list of the vocabulary for the language model sorted in order of index
(so if a word is indexed by 10 in the embedding matrix, it is found on line 10 in `new_line_separated_vocab_list.txt`).
It assumes that the training input is tokenized and stored at `data/train.tok` and the labels corresponding to each token are stored at `data/train.tag`.
As it trains, the classifier associated with the module `encoder` is stored at the end of each epoch in `intermediate_model.encoder.model`.
The validation set is used to judge the model from the epoch with the highest performance at the end, and its output on the test set will be saved in `tagged.json`. 
