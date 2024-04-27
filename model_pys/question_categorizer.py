import time
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import torch
import gzip
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import argparse

from torch import nn

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, vocab):
        self.model = super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
        self.vocab_size = vocab_size
        self.emsize = embed_dim
        self.num_class = num_class
        self.vocab = vocab
        self.text_pipeline = self.tokenizer

    def tokenizer(self, text):
        return self.vocab(tokenizer(text))

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

    def train_model(self, train_dataloader, valid_dataloader):

        total_accu = None
        for epoch in range(1, EPOCHS + 1):
          epoch_start_time = time.time()
          
          self.train()
          total_acc, total_count = 0, 0
          log_interval = 500
          start_time = time.time()

          for idx, (label, text, offsets) in enumerate(train_dataloader):
              optimizer.zero_grad()
              predicted_label = self(text, offsets)
              loss = criterion(predicted_label, label)
              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
              optimizer.step()
              total_acc += (predicted_label.argmax(1) == label).sum().item()
              total_count += label.size(0)
              if idx % log_interval == 0 and idx > 0:
                  elapsed = time.time() - start_time
                  print(
                      "| epoch {:3d} | {:5d}/{:5d} batches "
                      "| accuracy {:8.3f}".format(
                          epoch, idx, len(train_dataloader), total_acc / total_count
                      )
                  )
                  total_acc, total_count = 0, 0
                  start_time = time.time()


          accu_val = self.evaluate(valid_dataloader)
          if total_accu is not None and total_accu > accu_val:
              scheduler.step()
          else:
              total_accu = accu_val
          print("-" * 59)
          print(
              "| end of epoch {:3d} | time: {:5.2f}s | "
              "valid accuracy {:8.3f} ".format(
                  epoch, time.time() - epoch_start_time, accu_val
              )
          )
          print("-" * 59)


    #TODO: FIX THE LOADING MODEL
    def save_model(self, file_path):
        model_state = {
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.emsize,
            'num_class': self.num_class,
            'text_pipeline': self.text_pipeline
        }
        torch.save(model_state, file_path)
        print("Model saved successfully.")

    @classmethod
    def load_model(cls, file_path):
        model_state = torch.load(file_path)
        vocab_size = model_state['vocab_size']
        embed_dim = model_state['embed_dim']
        num_class = model_state['num_class']

        model = cls(vocab_size, embed_dim, num_class)
        model.load_state_dict(model_state['state_dict'])
        model.eval()
        print("Model loaded successfully.")
        return model

    def evaluate(self, dataloader):
      self.eval()
      total_acc, total_count = 0, 0

      with torch.no_grad():
          for idx, (label, text, offsets) in enumerate(dataloader):
              predicted_label = self(text, offsets)
              loss = criterion(predicted_label, label)
              total_acc += (predicted_label.argmax(1) == label).sum().item()
              total_count += label.size(0)
      return total_acc / total_count

    def predict(self, text):
      with torch.no_grad():
          text = torch.tensor(self.text_pipeline(text))
          output = model(text, torch.tensor([0]))
          return output
      
    @staticmethod
    def read_gz_json(file_path):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
            for obj in data:
                yield obj['text'], obj['category']

    @staticmethod
    def preprocess_text(text):
        sentences = sent_tokenize(text)
        return sentences

    @staticmethod
    def data_iter(file_paths, categories):

        categories = np.array(categories)

        for path in file_paths:
            for text, category in TextClassificationModel.read_gz_json(path):
                sentences = TextClassificationModel.preprocess_text(text)

                for sentence in sentences:
                    yield np.where(categories == category)[0][0], sentence
    @staticmethod
    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)
    
    def save_model(self, file_path):
        model_state = {
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_dim': self.emsize,
            'num_class': self.num_class
        }
        torch.save(model_state, file_path)
        print("Model saved successfully.")

    @classmethod
    def load_model(cls, file_path):
        model_state = torch.load(file_path)
        vocab_size = model_state['vocab_size']
        embed_dim = model_state['embed_dim']
        num_class = model_state['num_class']

        model = cls(vocab_size, embed_dim, num_class)
        model.load_state_dict(model_state['state_dict'])
        model.eval()
        print("Model loaded successfully.")
        return model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text Classification Model")
    parser.add_argument("--train_path", type=str, nargs='+', required=True, help="Path to the training data")
    parser.add_argument("--test_path", type=str, nargs='+', required=True, help="Path to the test data")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    categories = ['Geography', 'Religion', 'Philosophy', 'Trash', 'Mythology', 'Literature', 'Science', 'Social Science', 'History', 'Current Events', 'Fine Arts']

    test_path = args.test_path
    train_path = args.train_path

    tokenizer = get_tokenizer("basic_english")
    train_iter = iter(TextClassificationModel.data_iter(train_path, categories))

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)


    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataloader = DataLoader(
        train_iter, batch_size=8, shuffle=False, collate_fn=TextClassificationModel.collate_batch
    )

    train_iter = iter(TextClassificationModel.data_iter(train_path, categories))
    classes = set([label for (label, text) in train_iter])
    num_class = len(classes)
    print(num_class)
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    print(model)



    # Hyperparameters
    EPOCHS = args.epochs  # epoch
    LR = args.lr  # learning rate
    BATCH_SIZE = args.batch_size  # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_iter = iter(TextClassificationModel.data_iter(train_path, categories))
    test_iter = iter(TextClassificationModel.data_iter(test_path, categories))
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(
        train_dataset, [num_train, len(train_dataset) - num_train]
    )

    train_dataloader = DataLoader(
        split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TextClassificationModel.collate_batch
    )
    valid_dataloader = DataLoader(
        split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TextClassificationModel.collate_batch
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TextClassificationModel.collate_batch
    )

    model.train_model(train_dataloader,valid_dataloader)

    print("Checking the results of test dataset.")
    accu_test = model.evaluate(test_dataloader)
    print("test accuracy {:8.3f}".format(accu_test))

    model.save_model("text_classification_model.pth")


