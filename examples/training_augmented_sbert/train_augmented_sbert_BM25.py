"""
The code shows how to train Augmented SBERT on the STS + (AllNLI) dataset sampled using BM25 Algorithm
with cosine similarity loss function. At every 1000 training steps, 
the model is evaluated on the STS benchmark dataset

There are three important steps which are followed - 

1. Cross-encoder is trained upon the STS dataset
2. Cross-encoder is used to label BM25 sampled AllNLI dataset 
3. Bi-encoder is finally trained on both STS dataset + labeled AllNLI dataset

For more details you can refer -

For BM25 algorithm, we are using ElasticSearch (https://elasticsearch-py.readthedocs.io/)
Install the elasticsearch package with pip: pip install elasticsearch  

Usage:
python train_augmented_sbert_BM25.py

OR
python train_augmented_sbert_BM25.py pretrained_transformer_model_name
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import scipy.spatial
from elasticsearch import Elasticsearch
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

tracer = logging.getLogger('elasticsearch') 
tracer.setLevel(logging.CRITICAL) # supressing INFO messages for elastic-search

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

###### Read Datasets ######
batch_size = 16
num_epochs = 1
nli_reader = NLIDataReader('../datasets/AllNLI')
sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark', normalize_scores=True)
cross_encoder_model_save_path = 'output/BM25_cross_encoder_sts_allnli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
aug_sbert_model_save_path = 'output/BM25_augmented_sbert_sts_allnli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

cross_encoder_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
aug_sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
es = Elasticsearch()

#####################################################
#
# Step 1: Train cross-encoder model with STSbenchmark
#
#####################################################

logging.info("Step 1:   Train cross-encoder ({}) with STSbenchmark".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), cross_encoder_model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=cross_encoder_model)

# Convert the dataset to a DataLoader ready for dev
logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=cross_encoder_model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader, name='cross-encoder')


# Configure the training.
warmup_steps = math.ceil(len(train_data)*num_epochs/batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the cross-encoder model
cross_encoder_model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          steps_per_epoch=1,
          evaluation_steps=1000,
          warmup_steps=warmup_steps, 
          output_path=cross_encoder_model_save_path)


######################################################################################
#
# Step 2: filter our AllNlI top-k pairs using BM25 and label using cross-encoder model
#
######################################################################################

logging.info("Step 2: Label AllNLI silver dataset with cross-encoder ({})".format(model_name))

top_k = 3 # you can set parameter for k here: top k embeddings to be retrived
index_name = "allnli" # index-name should be in lowercase

# Embedding pair of sentences (s1,s2) present in AllNLI corpus
s1, s2 = map(list, zip(*[(input_ex.texts[0], input_ex.texts[1]) for input_ex in sts_reader.get_examples('sts-test.csv')]))
sentences = list(set(s1 + s2))
logging.info("Unique sentences found in AllNLI: {}".format(len(sentences)))

# Get embeddings for all sentences using cross-encoder
embeddings = dict(zip(sentences, cross_encoder_model.encode(sentences, batch_size=batch_size)))

# Ignore 400 cause by IndexAlreadyExistsException when creating an index
es.indices.create(index=index_name, ignore=[400]) 

# BM25 - Indexing sentences
for _id, sentence in enumerate(sentences):
    response = es.index(
        index = index_name,
        id = _id,
        body = {"sentence" : sentence })

logging.info("Indexing complete for {} sentences".format(len(sentences)))

s1 = []
s2 = []
duplicates = [] # remove duplicates i.e. to avoid combinations like (s2,s1) for (s1,s2)

# BM25 - Retrieval of top-k sentences 
for id, sentence in enumerate(sentences):
    if ((id < 10000 and id % 1000 == 0) or (id < 100000 and id % 10000 == 0) or id % 100000 == 0):
        logging.info("Completed retrieval of {} sentences".format(id))

    res = es.search(index=index_name, body={"query": {"match": {"sentence": sentence} } }, size=top_k)
    for hit in res['hits']['hits']:
        if str(id) != hit["_id"] and (hit['_id'], str(id)) not in set(duplicates):
            s1.append(sentence)
            s2.append(hit['_source']["sentence"])
            duplicates.append((str(id), hit["_id"]))

scores = 1 - scipy.spatial.distance.cdist([embeddings[s] for s in s1], [embeddings[s] for s in s2], "cosine")[0]

############################################################################################
#
# Step 3: Train bi-encoder model with both STSbenchmark and labeled AllNlI - Augmented SBERT
#
############################################################################################

logging.info("Step 3: Train bi-encoder ({}) with both STSbenchmark and BM25 sampled and labeled AllNlI".format(model_name))

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), aug_sbert_model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model=aug_sbert_model)

logging.info("Read ALLNLI silver train dataset")
examples = [InputExample(guid=str(id), texts=[s1_text, s2_text], label=score) for id, (s1_text, s2_text, score) in enumerate(zip(s1, s2, scores))]
silver_data = SentencesDataset(examples=examples, model=aug_sbert_model)
silver_dataloader = DataLoader(silver_data, shuffle=True, batch_size=batch_size)
silver_loss = losses.CosineSimilarityLoss(model=aug_sbert_model)

logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=aug_sbert_model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader, name='augmented-sbert')

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# You can pass as many (dataloader, loss) tuples as you like. They are iterated in a round-robin way.
train_objectives = [(train_dataloader, train_loss), (silver_dataloader, silver_loss)]

# Train the Aug-SBERT model
aug_sbert_model.fit(train_objectives=train_objectives,
          evaluator=evaluator,
          epochs=num_epochs,
          steps_per_epoch=1,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=aug_sbert_model_save_path
          )

#################################################################################
#
# Evaluate cross-encoder and Augmented SBERT performance on STS benchmark dataset
#
#################################################################################


# load the stored cross-encoder model
logging.info("Evaluation of cross-encoder model on STSbenchmark dataset")
model = SentenceTransformer(cross_encoder_model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader, name='cross-encoder')

model.evaluate(evaluator)

# load the stored augmented-sbert model
logging.info("Evaluation of Augmented SBERT model on STSbenchmark dataset")
model = SentenceTransformer(aug_sbert_model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader, name='augmented-sbert')

model.evaluate(evaluator)