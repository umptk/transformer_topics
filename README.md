# Topic Modelling the 20 Newsgroups Dataset


## Introduction

In an effort to develop a sample multilingual topic model applicable to multilingual document sets, we revisit BERTopics, a pipeline that integrates sentence transformers and embeddings to accomplish this task. The easily integrable sentence transformer models introduces the ability to utilize a number of pretrained models capable of vectorizing documents in non-English languages.

For the sake of this exercise, we'll be utilizing the 20 Newsgroups Dataset (said to be available in different languages; otherwise I'll either translate it using `mbart-large` or find something else to use).


The goal is to be able to multilingual topic-cluster all documents in the source language.


## 20 Newsgroups as a Multilingual Dataset

In order to develop hierarchical topic modeling for multilingual documents, we first need a multilingual dataset. In order to achieve this, we can leverage the the Huggingface library's many machine translation models, such as the [mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt). Translation can be found in [`translation.py`](https://github.com/umptk/transformer_topics/blob/main/translation.py) and requires defining source and target languages, such as `EN_XX` for English and `ko_KR` for Korean. A full list of language options can be found [here](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt). 

Small batches of English to Korean translations showed impressive performance.

> Missing sample output, please take my word for it.

But the model itself is massive and slow when run on CPU, so this may be best left for a GPU provisioned in an EC2 instance.

Note that running this model in a Python environment requires the `protobuf` package. Make sure this requirement is met by running `pip install protobuf` in the command line.


## Setting up BERTopics and 20 Newsgroups Dataset

Load the dataset from `scikit-learn`'s `fetch_20newsgroups`. We can opt to remove unnecessary data.

```
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
docs_samp = random.sample(docs, docs_size) 
```

For the sake of this exercise, we'll be working with a `doc_size` sample of 10000 randomly selected documents. The documents don't repeat. 

This sample size will decrease to 10 documents for the multilingual portion (since my dog poo laptop can't handle more than that).

![Alt text](figures/hierarchical_topics_1.png?raw=true "A decent first pass at hierarchical clustering.")

## Multilingual Capabilities

These results were achieved with BERTopics default-linked English transformer model. While there is also a default multilingual option as well, the many-to-many model referenced above could be inserted directly and applied directly on a set of non-English documents.

## Next Steps

BERTopics has easily accessible controls for adjusting the output topics. There are standard topic model parameters such as the number and size of topics as well as a unique parameter, diversity, which limits the number of duplicate words across topics using Maximal Marginal Relevance (MMR). The full documentation is available [here](https://maartengr.github.io/BERTopic/api/bertopic.html).


