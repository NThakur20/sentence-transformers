# Augmented SBERT

## Motivation
We recently obseve an increase of people using bi-encoders aka sentence-bert (sentence-transformers) to train their own sentence embeddings, mostly with limited data and / or a domain shift (specialized data). However, bi-encoders require substantial training data and fine-tuning over the target task to achieve competitive performances. To solve this practical issue, we release an effective data-augmentation strategy known as <b>Augmented SBERT</b> where we utilize a high performing and slow cross-encoder (BERT) to label a larger set of input pairs to augment the training data for the bi-encoder (SBERT).

For more details, refer to our publication - [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks]() which is a joint effort by Nandan Thakur, Nils Reimers and Johannes Daxenberger of UKP Lab, TU Darmstadt.


## Usage
There are two major use-cases for the Augmented SBERT approach for pairwise-sentence regression or classification tasks - 

## Scenario 1: Limited or small annotated datasets (few labeled sentence-pairs)

We apply the Augmented SBERT (<b>In-domain</b>) strategy to solve the Issue, it involves majorly three steps - 

 - ``Step 1:  Train a cross-encoder (BERT) over the small gold or annotated dataset``

 - ``Step 2.1: Create sentence-pairs by recombination and reduce the pairs via BM25 or Semantic Search sampling strategies``

 - ``Step 2.2: Weakly label new sentence-pairs with cross-encoder (BERT). Call them as silver pairs or silver dataset``

 - ``Step 3:  Finally, train a bi-encoder (SBERT) on the extended training dataset including both gold and silver datasets``

### Visual Description of Augmented SBERT (In-Domain)

<img src="https://raw.githubusercontent.com/Nthakur20/sentence-transformers/master/docs/img/augsbert-indomain.png" width="400" height="500">



## Scenario 2: No annotated datasets (Only unlabeled sentence-pairs)

We apply the Augmented SBERT (<b>Domain-Transfer</b>) strategy to solve the Issue, it involves majorly three steps - 

 - ``Step 1: Train from scratch a cross-encoder (BERT) over a source dataset, for which we contain annotations``

 - ``Step 2: Use this cross-encoder (BERT) to label your target dataset i.e. unlabled sentence pairs``

 - ``Step 3: Finally, train a bi-encoder (SBERT) on the labeled target dataset``

### Visual Description of Augmented SBERT (Domain-Transfer)

<img src="https://raw.githubusercontent.com/Nthakur20/sentence-transformers/master/docs/img/augsbert-domain-transfer.png" width="500" height="300">


## Training
 
The [examples/training/augmented_sbert](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/augmented_sbert/) folder contains simple training examples for each scenario explained below:

- [train_sts_seed_optimization.py](train_sts_seed_optimization.py) 
    - This script trains a bi-encoder (SBERT) model from scratch for STS benchmark dataset with seed-optimization. 
    - Seed optimization technique is insiped from [(Dodge et al., 2020)](https://arxiv.org/abs/2002.06305). 
    - For Seed opt., we train our bi-encoder for various seeds and evaluate using an early stopping algorithm. 
    - Finally, measure dev performance across the seeds to get the highest performing seeds.

- [train_sts_indomain_eda.py](train_sts_indomain_eda.py)
    - This script trains a bi-encoder (SBERT) model from scratch for STS benchmark dataset using easy data augmentation. 
    - Data augmentation strategies are used from popular [nlpaug](https://github.com/makcedward/nlpaug) package.
    - Augment single sentences with synonyms using (word2vec, BERT or WordNet). Forms our silver dataset.
    - Train bi-encoder model on both original small training dataset and synonym based silver dataset. 

- [train_sts_indomain_bm25.py](train_sts_indomain_bm25.py)
    - Script intially trains a cross-encoder (BERT) model from scratch for small STS benchmark dataset.
    - Recombine sentences from our small training dataset and form lots of sentence-pairs.
    - Limit number of combinations with BM25 sampling using [ElasticSearch](https://www.elastic.co/).
    - Retrieve top-k sentences given a sentence and label these pairs using the cross-encoder (silver dataset).
    - Train a bi-encoder (SBERT) model on both gold + silver STSb dataset. (Augmented SBERT (In-domain) Strategy).

- [train_sts_indomain_semantic.py](train_sts_indomain_semantic.py)
    - This script intially trains a cross-encoder (BERT) model from scratch for small STS benchmark dataset.
    - We recombine sentences from our small training dataset and form lots of sentence-pairs.
    - Limit number of combinations with Semantic Search sampling using pretrained SBERT model.
    - Retrieve top-k sentences given a sentence and label these pairs using the cross-encoder (silver dataset).
    - Train a bi-encoder (SBERT) model on both gold + silver STSb dataset. (Augmented SBERT (In-domain) Strategy).

- [train_sts_qqp_crossdomain.py](train_sts_qqp_crossdomain.py)
    - This script intially trains a cross-encoder (BERT) model from scratch for STS benchmark dataset.
    - Label the Quora Questions Pair (QQP) training dataset (Assume no labels present) using the cross-encoder.
    - Train a bi-encoder (SBERT) model on the QQP dataset. (Augmented SBERT (Domain-Transfer) Strategy).


## Performance
The performance was evaluated on the [Semantic Textual Similarity (STS) 2017 dataset](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page). The task is to predict the semantic similarity (on a scale 0-5) of two given sentences. STS2017 has monolingual test data for English, Arabic, and Spanish, and cross-lingual test data for English-Arabic, -Spanish and -Turkish.

We extended the STS2017 and added cross-lingual test data for English-German, French-English, Italian-English, and Dutch-English ([STS2017-extended.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/STS2017-extended.zip)). The performance is measured using Spearman correlation between the predicted similarity score and the gold score.

<table class="docutils">
  <tr>
    <th>Model</th>
    <th>AR-AR</th>
    <th>AR-EN</th>
    <th>ES-ES</th>
    <th>ES-EN</th>
    <th>EN-EN</th>
    <th>TR-EN</th>
    <th>EN-DE</th>
    <th>FR-EN</th>
    <th>IT-EN</th>
    <th>NL-EN</th>
    <th>Average</th>
  </tr>
  <tr>
    <td>XLM-RoBERTa mean pooling </td>
    <td align="center">25.7</td>
    <td align="center">17.4</td>
    <td align="center">51.8</td>
    <td align="center">10.9</td>
    <td align="center">50.7</td>
    <td align="center">9.2</td>
    <td align="center">21.3</td>
    <td align="center">16.6</td>
    <td align="center">22.9</td>
    <td align="center">26.0</td>
    <td align="center">25.2</td>
  </tr>
  <tr>
    <td>mBERT mean pooling </td>
    <td align="center">50.9</td>
    <td align="center">16.7</td>
    <td align="center">56.7</td>
    <td align="center">21.5</td>
    <td align="center">54.4</td>
    <td align="center">16.0</td
    <td align="center">33.9</td>
    <td align="center">33.0</td>
    <td align="center">34.0</td>
    <td align="center">35.6</td>
    <td align="center">35.3</td>
  </tr>
  <tr>
    <td>LASER</td>
    <td align="center">68.9</td>
    <td align="center">66.5</td>
    <td align="center">79.7</td>
    <td align="center">57.9</td>
    <td align="center">77.6</td>
    <td align="center">72.0</td>
    <td align="center">64.2</td>
    <td align="center">69.1</td>
    <td align="center">70.8</td>
    <td align="center">68.5</td>
    <td align="center">69.5</td>
  </tr> 
  <tr>
    <td colspan="12"><b>Sentence Transformer Models</b></td>
  </tr>
  <tr>
  <td>distiluse-base-multilingual-cased</td>
    <td align="center">75.9</td>
    <td align="center">77.6</td>
    <td align="center">85.3</td>
    <td align="center">78.7</td>
    <td align="center">85.4</td>
    <td align="center">75.5</td>
    <td align="center">80.3</td>
    <td align="center">80.2</td>
    <td align="center">80.5</td>
    <td align="center">81.7</td>
    <td align="center">80.1</td>
    </tr>
</table>


## Extend your own datasets

The idea is based on a fixed (monolingual) **teacher model**, that produces sentence embeddings with our desired properties in one language. The **student model** is supposed to mimic the teacher model, i.e., the same English sentence should be mapped to the same vector by the teacher and by the student model. In order that the student model works for further languages, we train the student model on parallel (translated) sentences. The translation of each sentence should also be mapped to the same vector as the original sentence.

In the above figure, the student model should map *Hello World* and the German translation *Hallo Welt* to the vector of *teacher_model('Hello World')*. We achieve this by training the student model using mean squared error (MSE) loss.

In our experiments we initiliazed the student model with the multilingual XLM-RoBERTa model. 


## Citation
If you use the code for augmented sbert, feel free to cite our publication [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks]():
```
```