"""
The script shows how to train Augmented SBERT (In-Domain) strategy for STSb dataset.

Sentence-pair combinations are sampled via the BM25 algorithm.

Cross-encoder aka BERT uses simpletransformers (pip install simpletransformers)
Bi-Encoder aka SBERT uses sentence-transformers (pip install sentence-transformers)
BM25 sampling uses elasticsearch (pip install elasticsearch)


Three consecutives steps to be followed for AugSBERT data-augmentation strategy - 

1. Fine-tune cross-encoder (BERT) on gold STSb dataset
2. Fine-tuned Cross-encoder is used to label on BM25 sampled unlabeled pairs (silver STSb dataset) 
3. Bi-encoder (SBERT) is finally fine-tuned on both gold + silver STSb dataset

For more details you can refer - cite paper

Usage:
python sts_indomain_bm25.py

OR
python sts_indomain_bm25.py pretrained_transformer_model_name top_k

python sts_indomain_bm25.py bert-base-uncased 3
"""
from torch.utils.data import DataLoader
from simpletransformers.classification import ClassificationModel
from sentence_transformers import models, losses, util
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from elasticsearch import Elasticsearch
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import logging
import scipy.spatial
import csv
import sys
import tqdm
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

# supressing INFO messages for elastic-search logger
tracer = logging.getLogger('elasticsearch') 
tracer.setLevel(logging.CRITICAL)
es = Elasticsearch()

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
batch_size = 16
num_epochs = 1
max_seq_length = 128
use_cuda = torch.cuda.is_available()

###### Read Datasets ######

#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

bert_model_path = 'output/bert/stsb_indomain_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
augsbert_model_path = 'output/aug-sbert-bm25/stsb_indomain_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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


#####################################################
#
# Step 1: Train cross-encoder model with STSbenchmark
#
#####################################################

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

# Train the cross-encoder model
bert_model.train_model(train_df=train_df, eval_df=eval_df, output_dir=bert_model_path, \
    pearson_corr=pearson_corr, spearman_corr=spearman_corr)

##########################################################################
#
# Step 2: Label BM25 sampled STSb silver dataset using cross-encoder model
#
##########################################################################

#### Top k similar sentences to be retrieved ####
#### Larger the k, bigger the silver dataset ####

index_name = "stsb" # index-name should be in lowercase

logging.info("Step 2.1: Generate STSb silver dataset using top-{} bm25 combinations".format(top_k))

# unique sentences present in STSb corpus
unique_sentences = set([data[0] for data in train_data] + [data[1] for data in train_data])
sent2idx = {sentence: idx for idx, sentence in enumerate(unique_sentences)}

# Ignore 400 cause by IndexAlreadyExistsException when creating an index
logging.info("Creating elastic-search index - {}".format(index_name))
es.indices.create(index=index_name, ignore=[400]) 

# indexing all sentences
logging.info("Starting to index....")
for sent in unique_sentences:
    response = es.index(
        index = index_name,
        id = sent2idx[sent],
        body = {"sent" : sent})

logging.info("Indexing complete for {} unique sentences".format(len(unique_sentences)))

silver_data = [] 
duplicates = [(sent2idx[data[0]], sent2idx[data[1]]) for data in train_data]
progress = tqdm.tqdm(unit="docs", total=len(sent2idx))

# retrieval of top-k sentences which forms the silver training data
for sent, idx in sent2idx.items():
    res = es.search(index = index_name, body={"query": {"match": {"sent": sent} } }, size = top_k)
    progress.update(1)
    for hit in res['hits']['hits']:
        if idx != int(hit["_id"]) and (idx, int(hit["_id"])) not in set(duplicates):
            silver_data.append([sent, hit['_source']["sent"]])
            duplicates.append((idx, int(hit["_id"])))

progress.reset()

logging.info("Step 2.2: Label STSb silver dataset with cross-encoder ({})".format(model_name))
bert_model = ClassificationModel(model_name.split("-")[0], bert_model_path, \
    num_labels=1, use_cuda=use_cuda, cuda_device=0, args=train_args)
silver_scores, _ = bert_model.predict(silver_data)

# All model predictions should be between [0,1]
assert all(0.0 <= score <= 1.0 for score in silver_scores)

############################################################################################
#
# Step 3: Train bi-encoder model with both STSbenchmark and labeled AllNlI - Augmented SBERT
#
############################################################################################

logging.info("Step 3: Train bi-encoder ({}) with both STSbenchmark Gold and Silver data".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark gold and silver train dataset")
gold_samples = list(InputExample(texts=[data[0], data[1]], label=data[2]) for data in train_data)
silver_samples = list(InputExample(texts=[data[0], data[1]], label=score) for data, score in zip(silver_data, silver_scores))

train_dataset = SentencesDataset(gold_samples + silver_samples, augsbert_model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=augsbert_model)

logging.info("Read STSbenchmark dev dataset")
dev_samples = list(InputExample(texts=[data[0], data[1]], label=data[2]) for data in dev_data)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training.
warmup_steps = math.ceil(len(train_dataset) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
augsbert_model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=augsbert_model_path
          )

#################################################################################
#
# Evaluate cross-encoder and Augmented SBERT performance on STS benchmark dataset
#
#################################################################################

# load the stored augmented-sbert model
augsbert_model = SentenceTransformer(augsbert_model_path)
test_samples = list(InputExample(texts=[data[0], data[1]], label=data[2]) for data in test_data)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(augsbert_model, output_path=augsbert_model_path)