"""
The script shows how to train Augmented SBERT (Cross-Domain) strategy for STSb-QQP dataset.
 
1. Cross-Encoder aka BERT is trained over STSb dataset.
2. Cross-Encoder is used to label QQP training dataset (Assume no labels/no annotations are provided).
3. Bi-encoder aka SBERT is trained over the labeled QQP dataset.


For more details you can refer - cite paper

Usage:
python sts_qqp_crossdomain.py

OR
python sts_qqp_crossdomain.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
from simpletransformers.classification import ClassificationModel
from sentence_transformers import models, losses, util
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import logging
import scipy.spatial
import csv
import sys
import torch
import math
import gzip
import os

def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]

def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
batch_size = 16
num_epochs = 1
max_seq_length = 128
use_cuda = torch.cuda.is_available()

###### Read Datasets ######
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
qqp_dataset_path = 'quora-IR-dataset'


# Check if the STSb dataset exsist. If not, download and extract it
if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Check if the QQP dataset exists. If not, download and extract
if not os.path.exists(qqp_dataset_path):
    logging.info("Dataset not found. Download")
    zip_save_path = 'quora-IR-dataset.zip'
    util.http_get(url='https://sbert.net/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(qqp_dataset_path)


bert_model_path = 'output/bert/stsb_indomain_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
augsbert_model_path = 'output/augsbert/qqp_cross_domain_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

###### Cross-encoder (simpletransformers) ######

logging.info("Loading cross-encoder model: {}".format(model_name))

# Setting optional model configuration
train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': num_epochs,
    'max_seq_length': max_seq_length,
    'evaluate_during_training': True,
    'train_batch_size': batch_size,
    'best_model_dir': bert_model_path,
    'regression': True, # Enabling regression
}

bert_model = ClassificationModel(model_name.split("-")[0], model_name, \
    num_labels=1, use_cuda=use_cuda, cuda_device=0, args=train_args)

###### Bi-encoder (sentence-transformers) ######

logging.info("Loading bi-encoder model: {}".format(model_name))

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

augsbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


############################################################
#
# Step 1: Train cross-encoder (BERT) model with STSbenchmark
#
############################################################

logging.info("Step 1: Train cross-encoder: ({}) with STSbenchmark".format(model_name))

train_data = []
dev_data = []
test_data = []

with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            dev_data.append([row['sentence1'], row['sentence2'], score])
        elif row['split'] == 'test':
            test_data.append([row['sentence1'], row['sentence2'], score])
        else:
            train_data.append([row['sentence1'], row['sentence2'], score])

train_df = pd.DataFrame(train_data, columns=['text_a', 'text_b', 'labels'])
eval_df = pd.DataFrame(dev_data, columns=['text_a', 'text_b', 'labels'])

# Train the cross-encoder (BERT) model
bert_model.train_model(train_df=train_df, eval_df=eval_df, output_dir=bert_model_path, \
    pearson_corr=pearson_corr, spearman_corr=spearman_corr)

##################################################################
#
# Step 2: Label QQP train dataset using cross-encoder (BERT) model
#
##################################################################

logging.info("Step 2: Label QQP dataset with cross-encoder ({})".format(model_name))

bert_model = ClassificationModel(model_name.split("-")[0], bert_model_path, \
    num_labels=1, use_cuda=use_cuda, cuda_device=0, args=train_args)

silver_data = []

with open(os.path.join(qqp_dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['is_duplicate'] == '1':
            silver_data.append([row['question1'], row['question2']])

silver_scores, _ = bert_model.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

binary_silver_scores = [1 if score >= 0.5 else 0 for score in silver_scores]

###########################################################################
#
# Step 3: Train bi-encoder (SBERT) model with QQP dataset - Augmented SBERT
#
###########################################################################

logging.info("Step 3: Train bi-encoder ({}) with QQP labeled dataset".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Loading BERT labeled QQP dataset")
qqp_train_data = list(InputExample(texts=[data[0], data[1]], label=score) \
        for data, score in zip(silver_data, binary_silver_scores))

train_dataset = SentencesDataset(qqp_train_data, augsbert_model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

###### Classification ######
# Given (quesiton1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.
logging.info("Read QQP dev dataset")

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []

with open(os.path.join(dataset_path, "classification/dev_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        dev_sentences1.append(row['question1'])
        dev_sentences2.append(row['question2'])
        dev_labels.append(int(row['is_duplicate']))

evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels)

# Configure the training.
warmup_steps = math.ceil(len(train_dataset) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
augsbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=augsbert_model_path,
          output_path_ignore_not_empty=True
          )

###############################################################
#
# Evaluate Augmented SBERT performance on QQP benchmark dataset
#
###############################################################

# Loading the augmented sbert model 
augsbert_model = SentenceTransformer(augsbert_model_path)

logging.info("Read QQP test dataset")
test_sentences1 = []
test_sentences2 = []
test_labels = []

with open(os.path.join(dataset_path, "classification/test_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        test_sentences1.append(row['question1'])
        test_sentences2.append(row['question2'])
        test_labels.append(int(row['is_duplicate']))

evaluator = evaluation.BinaryClassificationEvaluator(test_sentences1, test_sentences2, test_labels)
augsbert_model.evaluate(evaluator)