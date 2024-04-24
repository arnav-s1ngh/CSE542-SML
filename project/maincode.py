import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing data
data1="theoffice.csv"
data2="seinfeld.csv"
dat=data1
df = pd.read_csv(dat)


def corpus_creator(name):
    st = ""
    for i in df["Dialogue"][df["Character"]==name]:
        st = st + i
    return st

corpus_df = pd.DataFrame()
corpus_df["Character"] = list(df["Character"].value_counts().head(12).index)

li = []
for i in corpus_df["Character"]:
    li.append(corpus_creator(i))

corpus_df["Dialogues"] = li

corpus_df


from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)

from nltk.tokenize import word_tokenize
def text_processor(dialogue):
    dialogue = word_tokenize(dialogue)
    nopunc=[word.lower() for word in dialogue if word not in stop_words]
    nopunc=' '.join(nopunc)
    return [word for word in nopunc.split()]

corpus_df["Dialogues"] = corpus_df["Dialogues"].apply(lambda x: text_processor(x))
corpus_df


corpus_df["Length"] = corpus_df["Dialogues"].apply(lambda x: len(x))
corpus_df

fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(ax=ax,y="Length",x="Character",data=corpus_df)

import gensim

# Creating a dictionary for mapping every word to a number
dictionary = gensim.corpora.Dictionary(corpus_df["Dialogues"])
print("Number of words in dictionary: ", len(dictionary))
print(dictionary)

# Now, we create a corpus which is a list of bags of words. A bag-of-words representation for a document just lists the number of times each word occurs in the document.
corpus = [dictionary.doc2bow(bw) for bw in corpus_df["Dialogues"]]

# Now, we use tf-idf model on our corpus
tf_idf = gensim.models.TfidfModel(corpus)

# Creating a Similarity objectr
sims = gensim.similarities.Similarity('', tf_idf[corpus], num_features=len(dictionary))

# Creating a dataframe out of similarities
sim_list = []
for i in range(12):
    query = dictionary.doc2bow(corpus_df["Dialogues"][i])
    query_tf_idf = tf_idf[query]
    sim_list.append(sims[query_tf_idf])

corr_df = pd.DataFrame()
j = 0
for i in corpus_df["Character"]:
    corr_df[i] = sim_list[j]
    j = j + 1


fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corr_df,ax=ax,annot=True)
ax.set_yticklabels(corpus_df.Character)
plt.show()

def text_process(dialogue):
    nopunc=[word.lower() for word in dialogue if word not in stop_words]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split()]


season_groups = df.groupby('season')

# Iterate over each season
for season, season_df in season_groups:
    print(f"Season {season}")


    # Create a corpus for the season
    def season_corpus_creator(name):
        st = ""
        for i in season_df["Dialogue"][season_df["Character"] == name]:
            st = st + i
        return st


    corpus_df_season = pd.DataFrame()
    corpus_df_season["Character"] = list(season_df["Character"].value_counts().head(12).index)

    li = []
    for i in corpus_df_season["Character"]:
        li.append(season_corpus_creator(i))

    corpus_df_season["Dialogues"] = li

    # Perform text processing
    corpus_df_season["Dialogues"] = corpus_df_season["Dialogues"].apply(lambda x: text_processor(x))
    # Create a dictionary and corpus for the season
    dictionary_season = gensim.corpora.Dictionary(corpus_df_season["Dialogues"])
    corpus_season = [dictionary_season.doc2bow(bw) for bw in corpus_df_season["Dialogues"]]

    # Create tf-idf model for the season
    tf_idf_season = gensim.models.TfidfModel(corpus_season)

    # Create a Similarity object for the season
    sims_season = gensim.similarities.Similarity('', tf_idf_season[corpus_season], num_features=len(dictionary_season))

    # Calculate similarities for the season
    sim_list_season = []
    for i in range(12):
        query_season = dictionary_season.doc2bow(corpus_df_season["Dialogues"][i])
        query_tf_idf_season = tf_idf_season[query_season]
        sim_list_season.append(sims_season[query_tf_idf_season])

    # Create a dataframe out of similarities for the season
    corr_df_season = pd.DataFrame()
    j = 0
    for i in corpus_df_season["Character"]:
        corr_df_season[i] = sim_list_season[j]
        j = j + 1
    # Visualize the similarity matrix for the season
    # fig, ax = plt.subplots(figsize=(12, 12))
    # sns.heatmap(corr_df_season, ax=ax, annot=True)
    # ax.set_yticklabels(corpus_df_season.Character)
    # plt.title(f"Similarity Matrix for Season {season}")
    # plt.show()
