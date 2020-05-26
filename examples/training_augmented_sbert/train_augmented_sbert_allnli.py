"""
The code shows how to train Augmented SBERT on the STS + (AllNLI) dataset
with cosine similarity loss function. At every 1000 training steps, 
the model is evaluated on the STS benchmark dataset

There are three important steps which are followed - 

1. Cross-encoder is trained upon the STS dataset
2. Cross-encoder is used to label AllNLI dataset
3. Augmented SBERT (Bi-encoder) is finally trained on both STS dataset + labeled AllNLI dataset

For more details you can refer - 

Usage:
python training_augmented_sbert_allnli.py

OR
python training_augmented_sbert_allnli.py pretrained_transformer_model_name
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
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'

###### Read Datasets ######
batch_size = 16
num_epochs = 1
nli_reader = NLIDataReader('../datasets/AllNLI')
sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark', normalize_scores=True)
cross_encoder_model_save_path = 'output/cross_encoder_sts_allnli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
aug_sbert_model_save_path = 'output/augmented_sbert_sts_allnli_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

cross_encoder_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
aug_sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

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
          evaluation_steps=1000,
          warmup_steps=warmup_steps, 
          output_path=cross_encoder_model_save_path)


########################################################
#
# Step 2: Label AllNlI dataset using cross-encoder model
#
########################################################

logging.info("Step 2:   Label AllNLI silver dataset with cross-encoder ({})".format(model_name))

# embedding pair of sentences (s1,s2) present in AllNLI corpus
s1, s2 = map(list, zip(*[(input_ex.texts[0], input_ex.texts[1]) for input_ex in nli_reader.get_examples('train.gz')]))
s1_embed = cross_encoder_model.encode(s1, batch_size=batch_size)
s2_embed = cross_encoder_model.encode(s2, batch_size=batch_size)
scores = 1 - scipy.spatial.distance.cdist(s1_embed, s2_embed, "cosine")[0]

############################################################################################
#
# Step 3: Train bi-encoder model with both STSbenchmark and labeled AllNlI - Augmented SBERT
#
############################################################################################


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