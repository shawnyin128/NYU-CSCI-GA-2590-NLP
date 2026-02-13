# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from typing import List
from sentiment_data import *
from utils import *
from collections import Counter

import string


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        index_sentence = []
        for i in range(len(sentence)):
            # preprocess: lowercase
            index_sentence.append(sentence[i].lower())
            # indexer: update index if there are new tokens
            index_sentence[-1] = self.indexer.add_and_get_index(index_sentence[-1], add=add_to_indexer)
        # counter: construct a counter
        counter = Counter(index_sentence)
        return counter


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        bigram_sentence = []
        for i in range(len(sentence) - 1):
            bigram_sentence.append(sentence[i].lower() + '_' + sentence[i + 1].lower())
        index_sentence = []
        for i in range(len(bigram_sentence)):
            # preprocess: lowercase
            index_sentence.append(bigram_sentence[i].lower())
            # indexer: update index if there are new tokens
            index_sentence[-1] = self.indexer.add_and_get_index(index_sentence[-1], add=add_to_indexer)
        # counter: construct a counter
        counter = Counter(index_sentence)
        return counter


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        bigram_sentence = []
        translator = str.maketrans('', '', string.punctuation)
        for i in range(len(sentence) - 1):
            bigram_sentence.append(sentence[i].translate(translator).lower() + '_' + sentence[i + 1].translate(translator).lower())
        index_sentence = []
        for i in range(len(sentence)):
            # preprocess: lowercase
            index_sentence.append(sentence[i].translate(translator).lower())
            # indexer: update index if there are new tokens
            index_sentence[-1] = self.indexer.add_and_get_index(index_sentence[-1], add=add_to_indexer)
        for i in range(len(bigram_sentence)):
            # preprocess: lowercase
            index_sentence.append(bigram_sentence[i].lower())
            # indexer: update index if there are new tokens
            index_sentence[-1] = self.indexer.add_and_get_index(index_sentence[-1], add=add_to_indexer)
        # counter: construct a counter
        counter = Counter(index_sentence)
        return counter


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_dim: int, feat_extractor: FeatureExtractor, threshold: float = 0.5):
        self.feature_dim = feature_dim
        self.weight = np.zeros(self.feature_dim)
        self.feat_extractor = feat_extractor
        self.threshold = threshold
        self._train = True

        self._grad_cache = {}
        self._grad_cache['x'] = []
        self._grad_cache['prob'] = []

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        # get the feature vector
        feature = np.zeros(self.feature_dim)
        feature_counter = self.feat_extractor.extract_features(ex_words, add_to_indexer=False)
        for idx, freq in feature_counter.items():
            feature[idx] = freq
        # linear compute
        logits = np.dot(self.weight, feature)
        self._grad_cache['x'].append(feature)
        # logistic output
        prob = 1 / (1 + np.exp(-logits))
        self._grad_cache['prob'].append(prob)
        if self._train:
            return prob
        else:
            # get result
            y = prob >= self.threshold
            return y

    def gradient(self, idx: int, label: int):
        prob = self._grad_cache['prob'][idx]
        p_L_p_p = (prob - label) / (prob * (1 - prob))
        p_p_p_l = prob * (1 - prob)
        x = self._grad_cache['x'][idx]
        p_l_p_w = x
        p_L_p_w = p_L_p_p * p_p_p_l * p_l_p_w
        return p_L_p_w

    def optimize_all(self, label: List[int], lr: float, weight_decay: float = 0.0001):
        batch_size = len(label)
        batch_grad = []
        for idx in range(batch_size):
            batch_grad.append(self.gradient(idx, label[idx]))
        avg_grad = np.mean(np.array(batch_grad), axis=0)

        self.weight = self.weight * (1 - lr * weight_decay)
        self.weight -= lr * avg_grad
        self._grad_cache['x'] = []
        self._grad_cache['prob'] = []


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    # go through train samples to find feature dim
    for train_sample in train_exs:
        words = train_sample.words
        feat_extractor.extract_features(words, add_to_indexer=True)

    # init model
    feature_dim = len(feat_extractor.get_indexer())
    model = LogisticRegressionClassifier(feature_dim, feat_extractor)
    model.train()

    # define training configuration
    # epochs = args.num_epochs
    # batch_size = args.batch_size
    # max_lr = args.lr
    # overwrite training config
    epochs = 20
    batch_size = 32
    max_lr = 1.0
    min_lr = 0.00001
    random.seed(42)
    total_iter = (len(train_exs) // batch_size) * epochs
    cur_iter = 0

    # train loop
    for epoch in range(epochs):
        train_exs_shuffled = random.sample(train_exs, len(train_exs))
        for itr in range(len(train_exs) // batch_size): # drop last batch
            # get batch data
            batch = train_exs_shuffled[itr * batch_size: (itr + 1) * batch_size]
            x = [b.words for b in batch]
            y = [b.label for b in batch]

            # get model output
            probs = np.array(model.predict_all(x))

            # compute loss
            loss = -np.sum(np.log(probs) * y + np.log(1 - probs) * (1 - np.array([b.label for b in batch])))
            # wandb.log({'loss': loss, 'step': cur_iter})

            # compute gradient
            lr = min_lr + (1 - cur_iter / total_iter) * (max_lr - min_lr)
            cur_iter += 1
            model.optimize_all(y, lr=lr)
    model.eval()
    return model


def train_linear_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your linear model. You may modify this, but do not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    model = train_logistic_regression(train_exs, feat_extractor)
    return model


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, network: nn.Module, word_embeddings: WordEmbeddings, padding_length: int = 128):
        self.network = network
        self.word_embeddings = word_embeddings
        self.word_indexer = self.word_embeddings.word_indexer
        self.padding_length = padding_length

        self._train = True

    def train(self):
        self._train = True
        self.network.train()

    def eval(self):
        self._train = False
        self.network.eval()

    def tokenize(self, train_exs: List[List[str]]) -> torch.Tensor:
        pad_length = self.padding_length
        pad_token_id = self.word_indexer.index_of("PAD")
        input_ids = []
        for sample in train_exs:
            cur_input_ids = []
            for word in sample:
                idx = self.word_indexer.index_of(word)
                if idx == -1:
                    cur_input_ids.append(self.word_indexer.index_of("UNK"))
                else:
                    cur_input_ids.append(idx)
            if len(cur_input_ids) < pad_length:
                cur_input_ids.extend([pad_token_id] * (pad_length - len(cur_input_ids)))
            elif len(cur_input_ids) > pad_length:
                cur_input_ids = cur_input_ids[:pad_length]
            input_ids.append(cur_input_ids)
        return torch.LongTensor(input_ids)

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        # tokenize sample
        x = self.tokenize([ex_words])

        # run model
        logits = self.network(x)

        # output
        if self._train:
            return logits
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output = torch.argmax(probs, dim=-1)
            return output

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        # tokenize samples
        x = self.tokenize(all_ex_words)

        # run model
        logits = self.network(x)

        # output
        if self._train:
            return logits
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            output = torch.argmax(probs, dim=-1)
            return output


class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings):
        super().__init__()
        self.class_num = 2
        self.embeddings = word_embeddings.get_initialized_embedding_layer()
        self.feature_size = word_embeddings.get_embedding_length()
        self.linear1 = nn.Linear(self.feature_size, self.feature_size * 4, bias=True)
        self.linear2 = nn.Linear(self.feature_size * 4, self.class_num, bias=True)

        self._linear_init()

    def _linear_init(self):
        torch.nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="relu")
        torch.nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity="relu")

    def forward(self, x):
        embed = self.embeddings(x)
        avg_embed = embed.mean(dim=1)

        out = torch.nn.functional.relu(self.linear1(avg_embed))
        out = torch.nn.functional.dropout(out, training=self.training, p=0.6)
        out = self.linear2(out)

        return out


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    Main entry point for your deep averaging network model.
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # init model
    DAN = DeepAveragingNetwork(word_embeddings)
    model = NeuralSentimentClassifier(DAN, word_embeddings)
    model.train()

    # define training configuration
    # epochs = args.num_epochs
    # batch_size = args.batch_size
    # max_lr = args.lr
    # overwrite training config
    epochs = 20
    batch_size = 32
    lr = 1e-2
    random.seed(42)
    total_iter = (len(train_exs) // batch_size) * epochs

    # define optimizer
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter, eta_min=0.0001)

    # train loop
    for epoch in range(epochs):
        train_exs_shuffled = random.sample(train_exs, len(train_exs))
        for itr in range(len(train_exs) // batch_size):  # drop last batch
            # get batch data
            batch = train_exs_shuffled[itr * batch_size: (itr + 1) * batch_size]
            x = [b.words for b in batch]
            y = torch.LongTensor([b.label for b in batch])

            # get model output
            logits = model.predict_all(x)

            # compute loss
            loss = torch.nn.functional.cross_entropy(logits, y)
            print(f"[INFO] Epoch: {epoch}, iteration: {itr}, loss: {loss.item()}")

            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.eval()
    return model
