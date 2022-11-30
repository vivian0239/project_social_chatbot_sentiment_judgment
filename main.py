# import sys
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import pyqtSlot
#
# def window():
#    app = QApplication(sys.argv)
#    widget = QWidget()
#
#    textLabel = QLabel(widget)
#    textLabel.setText("Hello World!")
#    textLabel.move(110,85)
#
#    widget.setGeometry(50,50,320,200)
#    widget.setWindowTitle("PyQt5 Example")
#    widget.show()
#    sys.exit(app.exec_())
#
# if __name__ == '__main__':
#    window()



# import remi.gui as gui
# from remi import start, App
# from threading import Timer

# from transformers import pipeline
# classifier = pipeline("sentiment-analysis")
# res = classifier("We are very happy to see you!")
# print(res)

#-------------------------------------------------------------------------------

from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification
#
# model_name="ydshieh/tiny-random-gptj-for-sequence-classification" #"distilbert-base-uncased-finetuned-sst-2-english"
# model=AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer=AutoTokenizer.from_pretrained(model_name)
# classifier=pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
# res=classifier("we love the show")
# print(res)

# tokens=tokenizer.tokenize("we love the show")
# token_ids=tokenizer.convert_tokens_to_ids(tokens)
# input_ids=tokenizer("we love the show")
#
# print(f'    Tokens: {tokens}')
# print(f'Tokens IDs: {token_ids}')
# print(f' Input IDs: {input_ids}')

#-----------------------------------------------
#--------------fine tunning----------------------
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# import torch
# from torch.utils.data import Dataset
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
# from transformers import Trainer, TrainingArguments

# model_name="distilbert-base-uncased"
#-------------below is example--------------------------------

#
# def read_imdb_split(split_dir):
#     split_dir = Path(split_dir)
#     texts = []
#     labels = []
#     for label_dir in ["pos", "neg"]:
#         for text_file in (split_dir/label_dir).iterdir():
#             texts.append(text_file.read_text())
#             labels.append(0 if label_dir == "neg" else 1)
#     return texts, labels
#
# train_texts, train_labels = read_imdb_split('aclImdb/train')
# test_texts, test_labels = read_imdb_split('aclImdb/test')
#------------------example end----------------------------

# def read_split(filename):
#     f=open(filename)
#     line=f.readline()
#     texts = []
#     labels = []
#     while line:
#         print(line)
#         d1=line[9]
#         d2=line[12:]
#         texts.append(d2)
#         if d1=="2":
#             labels.append(1) #positive
#         if d1=="1":
#             labels.append(0) #negative
#         line = f.readline()
#     f.close()
#     return texts, labels
#
# train_texts, train_labels = read_split('test.txt')
# test_texts, test_labels = read_split('test.txt')
#
# train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
#
# class IMDbDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#
#     def __len__(self):
#         return len(self.labels)
#
# tokenizer=DistilBertTokenizerFast.from_pretrained(model_name)
#
# train_encodings=tokenizer(train_texts,truncation=True,padding=True)
# val_encodings=tokenizer(val_texts,truncation=True,padding=True)
# test_encodings=tokenizer(test_texts,truncation=True,padding=True)
#
# train_dataset = IMDbDataset(train_encodings, train_labels)
# val_dataset = IMDbDataset(val_encodings, val_labels)
# test_dataset = IMDbDataset(test_encodings, test_labels)
#
# training_args=TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=2,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=64,
#     warmup_steps=500,
#     learning_rate=5e-5,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
# )
#
# model=DistilBertForSequenceClassification.from_pretrained(model_name)
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset
# )
#
# trainer.train()
# trainer.evaluate()
#
# model_dir = '/PycharmProjects/pythonProject/trainedmodel/'
# trainer.save_model(model_dir + 'fine-tunned-model')

#------use the fine-tunned model-------------
#imports
import io
import os
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

# Set seed for reproducibility.
set_seed(123)

# Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
epochs = 4

# Number of batches - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batch_size = 32

# Pad or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 60

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.
model_name_or_path = 'gpt2'

# Dictionary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids = {'neg': 0, 'pos': 1}

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

#--------- add input variable--------------------------
input_text = ''

# #Helper functions-------------------------------------
class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask.

    It uses a given tokenizer and label encoder to convert any text and labels to numbers that
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels)})

        return inputs

#-------Input data--------------------------
class MovieReviewsDataset(Dataset):
    r"""PyTorch Dataset class for loading data.

    This is where the data parsing happens.

    This class is built with reusability in mind: it can be used as is as.

    Arguments:

      path (:obj:`str`):
          Path to the data partition.

    """

    def __init__(self, str, use_tokenizer):

        # Check if path exists.
        # if not os.path.isdir(path):
        #     # Raise error if path is invalid.
        #     raise ValueError('Invalid `path` variable! Needs to be a directory')
        self.texts = [str]
        self.labels = ['pos']
        # Since the labels are defined by folders with data we loop
        # through each label.
        # for label in ['pos', 'neg']:
        #     sentiment_path = os.path.join(path, label)
        #
        #     # Get all files from path.
        #     files_names = os.listdir(sentiment_path)  # [:10] # Sample for debugging.
        #     # Go through each file and read its content.
        #     for file_name in tqdm(files_names, desc=f'{label} files'):
        #         file_path = os.path.join(sentiment_path, file_name)
        #
        #         # Read content.
        #         content = io.open(file_path, mode='r', encoding='utf-8').read()
        #         # Fix any unicode issues.
        #         content = fix_text(content)
        #         # Save content.
        #         self.texts.append(content)
        #         # Save encode labels.
        #         self.labels.append(label)

        # Number of exmaples.
        self.n_examples = len(self.labels)

        return

    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return self.n_examples

    def __getitem__(self, item):
        r"""Given an index return an example from the position.

        Arguments:

          item (:obj:`int`):
              Index position to pick an example to return.

        Returns:
          :obj:`Dict[str, str]`: Dictionary of inputs that contain text and
          asociated labels.

        """

        return {'text': self.texts[item],
                'label': self.labels[item]}


def validation(dataloader, device_):
    r"""Validation function to evaluate model performance on a
    separate set of data.

    This function will return the true and predicted labels so we can use later
    to evaluate the model's performance.

    This function is built with reusability in mind: it can be used as is as long
      as the `dataloader` outputs a batch in dictionary format that can be passed
      straight into the model - `model(**batch)`.

    Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
            Parsed data into batches of tensors.

      device_ (:obj:`torch.device`):
            Device used to load tensors before feeding to model.

    Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
          Labels, Train Average Loss]
    """

    # Use global variable for model.
    global model

    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):
        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

            # update list
            predictions_labels += predict_content

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss
#
#
# # Load Model and Tokenizer-----------------------------
# # Get model configuration.
# print('Loading configuraiton...')
# model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
#
# # Get model's tokenizer.
# print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
# # default to left padding
# tokenizer.padding_side = "left"
# # Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
#
#
# # Get the actual model.
# print('Loading model...')
# model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
#
# # resize model embedding to match new tokenizer
# model.resize_token_embeddings(len(tokenizer))
#
# # fix model padding token id
# model.config.pad_token_id = model.config.eos_token_id
#
# # Load model to defined device.
# model.to(device)
# print('Model loaded to `%s`'%device)
# #dataset and collator--------------------------------
# # Create data collator to encode text and labels into numbers.
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                          labels_encoder=labels_ids,
                                                          max_sequence_len=max_length)
# # print('------pass2----------')
input_text = 'I have never meet such a '
valid_dataset =  MovieReviewsDataset(input_text,use_tokenizer=tokenizer)
# print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
#
# # Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
# print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

import torch.nn as nn

global model
model = torch.load("model.pth")
model.eval()

# valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
# print("valid_label is:")
# print(valid_labels)
# print("valid_predict is:")
# print(valid_predict)
# print("val_loss is:")/
# print(val_loss)

model_name="ydshieh/tiny-random-gptj-for-sequence-classification" #"distilbert-base-uncased-finetuned-sst-2-english"
model1=AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer1=AutoTokenizer.from_pretrained(model_name)
classifier=pipeline("sentiment-analysis",model=model1,tokenizer=tokenizer1)
# res=classifier("we love the show")

#assistant chatbot
from ShopAssistant import ShopAssistant
from transformers import pipeline

Ella = ShopAssistant("Ella")
val = input(Ella.intro() + "\n")
end_flag = False

while not end_flag:
    input_text = val
    valid_dataset = MovieReviewsDataset(input_text, use_tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  collate_fn=gpt2_classificaiton_collator)
    alid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
    print("valid_predict from gpt2 is: \n")
    print(valid_predict)
    #
    # res = classifier(val)
    # print("valid_predict from sentiment-analysis is: \n")
    # print(res)

    if valid_predict[0] == 0:
        Ella.set_complain()

    re = Ella.assistant(val)
    end_flag = Ella.get_end_flag()
    if end_flag:
        break
    val = input(re + "\n")
print(Ella.bye())










