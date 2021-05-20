# Japanese bidding document information extraction
Extracted important information from Japanese bidding contracts with an average of 0.97 F1 scores by formulating the problem as a question answering task and utilizing **Transformer** to solve it.

### To train the models:

the most simple way to train the three models .

* bert japanese: `python train_bert_japanese.py`
* bert multilingual: `python train_bert_multilingual.py`
* albert: `python train_albert.py`

#### Training arguments includes : seed, gpu, epochs, batch_size, lr, only_positive... Please refers to one of the training file to see all arguments.

### Two steps training:
If you want to do two steps training, please follow the instruction. (take albert as an example)

1. train a model with only positive examples by setting only_positive to 1.

`python train_albert.py --only_positive 1`

2. load the pretrained model to do the second step training and set load_pretrained_model to 1.

`python train_albert.py --pretrained_model_path <path to the saved model> --load_pretrained_model 1`

### To test the models:

1. download models.
`bash download.sh`
2. run test.py with a args which specify test data directory.
e.g. `python test.py 'release/test/ca_data'`

### Experiment plotting:
![Image Not Found](/png/acc.png)

### Summary:
Please refer to "Report.pdf" or see the video summarization linked below.
<br/>
[![Image Not Found](http://img.youtube.com/vi/kl92dEIZnn8/1.jpg)](http://www.youtube.com/watch?v=kl92dEIZnn8 "Japanese Document Extraction")
