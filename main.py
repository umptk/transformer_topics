from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pickle
import plotly.express as px
import random
from translation import many_to_many


# SOME VARIABLES
docs_size = 10   # Lower this when things get sticky and slow, such as many_to_many
target_lang = 'ko_KR'

# LOAD 20 NEWSGROUPS DATASET, AND RANDOMLY SAMPLE 10K DOCS 
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
docs_samp = random.sample(docs, docs_size)   # No duplication


# INITIALIZE BERTopic MODEL, FIT DOCS AND VISUALIZE
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs_samp)

topic_model.get_topic_info()
topic_model.visualize_hierarchy().show()


# PRESERVE MODEL AND DATA
with open('topic_model', 'wb') as f:
    pickle.dump(topic_model, f)

with open('docs_samp', 'wb') as f:
    pickle.dump(docs_samp, f)


# REDUCE NUMBER OF TOPICS AND REFINE RESULTS
new_topics, new_probs = topic_model.reduce_topics(docs, topics, probs, nr_topics=50)
fig = topic_model.visualize_hierarchy()
fig.write_html("figures//hierarchical_topics_1.html")
fig.write_image("figures//hierarchical_topics_1.png")


# SO IT KINDA WORKS, NOW TO MAKE IT MULTILINGUAL
target_docs = many_to_many(docs_samp, target_lang=target_lang)

multi_model = BERTopic(language="multilingual")
topics, probs = multi_model.fit_transform(target_docs)








