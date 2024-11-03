# Ensure you have a "processed" directory containing the sentences and an "embeddings" and "index" directory for output
# I'm also providing the last trained models as models-2023-04-10-2229.pkl. You'll need cuML to use them.

import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer, util
from functools import reduce, partial
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch, copy, nltk, shutil
import faiss
from faiss.contrib.ondisk import merge_ondisk
import os, re, glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import namedtuple
import pickle as pkl
from datetime import timezone, datetime, timedelta
import pytz

from tqdm.auto import tqdm
from tqdm.contrib import tenumerate

from cuml.ensemble import RandomForestClassifier as curfc
from cuml.svm import SVC
from cuml.linear_model import LogisticRegression

from cuml.metrics import accuracy_score

from xgboost import XGBClassifier

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
def files(dir):
  return sorted_alphanumeric([dir + x for x in os.listdir(dir)])

model = SentenceTransformer('msmarco-distilbert-base-v4') # asymmetric

taxonomy = {
    "competence": {
        "competence_trust": ["intelligent", "reasoned", "informed", "evidence", "accurate", "truthful"],
        "competence_distrust": ["stupid", "incompetent", "ignorant", "idiot", "sheep", "insane", "moron",
                                "dumbass", "clown", "living in your bubble", "incoherent", "nonsense", "no proof",
                                "accept reality", "irrational", "retarded", "intellectually dishonest",
                                "misleading", "indoctrinated"] # new
    },
    "sincerity": {
        "sincerity_trust": ["trustworthy", "reputable", "sincere", "genuine"],
        "sincerity_distrust": ["liar", "dishonest", "untrustworthy", "corrupt", "inhuman", "immoral",
                               "disinformation", "misinformation", "propaganda", "fake news", "unreliable source", "paranoid", "shill",
                               "bias", "discredited", "manipulated", "scam"] # new
    }
}

def flatten_taxonomy(taxonomy, keys=[]):
  if type(taxonomy) is not dict:
    return [(x, keys) for x in taxonomy]
  return reduce(lambda x, y: x + y, [flatten_taxonomy(t, keys + [k]) for k, t in taxonomy.items()])

def t2i_i2t_embeddings(flat_taxonomy, model):
  t2i = {k: i for i, (k, v) in enumerate(flat_taxonomy)}
  i2t = {i: k for i, (k, v) in enumerate(flat_taxonomy)}
  embs = model.encode([i2t[i] for i in range(len(flat_taxonomy))], normalize_embeddings=True)
  return (t2i, i2t, embs)

flat_taxonomy = flatten_taxonomy(taxonomy)
t2i, i2t, taxonomy_embs = t2i_i2t_embeddings(flat_taxonomy, model)
taxonomy_embs.shape

## You don't need the below code -- I provide a trained index file. But here it is for reference!
## Embed a sample of sentences and train index
# embeddings = model.encode(sentences_sample, batch_size=128, normalize_embeddings=True)

# index = faiss.index_factory(768, "OPQ64_256,IVF262144_HNSW32,PQ64") # train on embeddings...

# index_ivf = faiss.extract_index_ivf(index)
# clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
# index_ivf.clustering_index = clustering_index

# index.train(embeddings) # training on GPU -- 42 mins vs > 5hrs!!
# index_ivf.clustering_index = faiss.index_gpu_to_cpu(index_ivf.clustering_index)
# faiss.write_index(index, "trained.index")

csvs = files("processed/")

offset_total = 0
for i, csv in enumerate(csvs):
  print(f"Processing {csv}")
  df = pd.read_csv(csv)
  sents = [str(x) for x in df["sentence"]]
  embs = model.encode(sents, normalize_embeddings=True)
  sims = util.cos_sim(taxonomy_embs, embs)
  with open(f"embeddings/embs-{os.path.basename(csv)}.pkl", "wb") as f:
    pkl.dump(sims, f)

  print(f"Loading pre-trained index...")
  index = faiss.read_index("trained.index")
  index.add_with_ids(embs, np.arange(int(i * len(df)), int((i + 1) * len(df))) + offset_total)
  print(f"Writing block_{os.path.basename(csv)}.index...")
  faiss.write_index(index, f"index/block_{os.path.basename(csv)}.index")

## Running models

def construct_data(prompts, sims, t2i, threshold=0.5, neg_to_pos_ratio=1):
  sent_emb_idx_positive = set()
  sent_emb_idx_negative = set()
  #sims = util.cos_sim(model.encode(prompts, normalize_embeddings=True), sentence_embeddings)
  sims = sims[[t2i[x] for x in prompts]] # keeps row logic working below
  for i, prompt in enumerate(prompts):
    #sorted_idxs = torch.argsort(-sims[i])
    pos_idxs = torch.argwhere(sims[i] >= threshold).flatten().tolist()
    neg_idxs = torch.argwhere(sims[i] < threshold).flatten().tolist()
    pos = set(pos_idxs)
    neg = set(np.random.choice(neg_idxs, round(len(pos) * neg_to_pos_ratio)))
    sent_emb_idx_positive = sent_emb_idx_positive.union(pos)
    sent_emb_idx_negative = sent_emb_idx_negative.union(neg)
  return {"positive": list(sent_emb_idx_positive), "negative": list(sent_emb_idx_negative)}

def expand_data(data, df, model, index, nprobe=2048, k=3, expand=True):
  if not expand:
    return []
  query_embs = model.encode(df.iloc[data["positive"]]["sentence"].tolist(), normalize_embeddings=True) #embeddings[data["positive"]]
  index.nprobe = nprobe
  D, I = index.search(query_embs, k)
  new_sents = I.flatten().tolist()
  print(f"Expanded {query_embs.shape[0]} examples by {len(new_sents)} to {query_embs.shape[0] + len(new_sents)}")
  return new_sents

# Use the pretrained index...
index = faiss.read_index("trained.index")
blocks = files("index/")
merge_ondisk(index, blocks, "merged_index.ivfdata")
faiss.write_index(index, "populated.index")
#index.ntotal

# Construct and expand data
eptrust_categories = ["competence_distrust", "competence_trust", "sincerity_distrust", "sincerity_trust"]
mdl_data = {category: {"pos": [], "neg": [], "pos_sims": [], "neg_sims": [], "pos_nn": []} for category in eptrust_categories}

csvs = files("processed/")

for csv in tqdm(csvs):
  #print(f"##### Processing {csv}...")
  df = pd.read_csv(csv)
  with open(f"embeddings/embs-{os.path.basename(csv)}.pkl", "rb") as f:
    sims = pkl.load(f)

  for category in eptrust_categories:
    #print(f"# Processing {category}...")
    qry = [x for x, y in flat_taxonomy if category in y]
    data = construct_data(qry, sims, t2i, neg_to_pos_ratio=20) # df.iloc[x["positive"]]
    pos_expand = expand_data(data, df, model, index, nprobe=2048, k=5, expand=True)

    mdl_data[category]["pos"].append(df.iloc[data["positive"]])
    mdl_data[category]["neg"].append(df.iloc[data["negative"]])
    mdl_data[category]["pos_sims"].append(sims[:, data["positive"]])
    mdl_data[category]["neg_sims"].append(sims[:, data["negative"]])
    mdl_data[category]["pos_nn"].append(pos_expand)


mdl_nn_data = {category: {"pos_nn_index": {}, "pos_nn_data": []} for category in eptrust_categories}
n_per_file = 1000000

for category in eptrust_categories:
  pos_nn = set(reduce(lambda x, y: x + y, mdl_data[category]["pos_nn"]))
  pos_nn_index = {}

  for x in pos_nn:
    if (x // n_per_file) not in pos_nn_index:
      pos_nn_index[x // n_per_file] = []
    pos_nn_index[x // n_per_file].append(x % n_per_file)
  mdl_nn_data[category]["pos_nn_index"] = pos_nn_index

for file_id, csv in tenumerate(csvs):
  for category in eptrust_categories:
    df = pd.read_csv(csv)
    if file_id not in pos_nn_index:
      print(f"Nothing in {csv}... weird.")
    else:
      pos_nn_index = mdl_nn_data[category]["pos_nn_index"]
      mdl_nn_data[category]["pos_nn_data"].append(df.iloc[pos_nn_index[file_id]])

# Now to collate the data and train models
def prepare_data(category, mdl_data, mdl_nn_data, model, taxonomy_embs, i2t, raw=False):
  print(f"Preparing {category}...")
  pos_mat = torch.concat(mdl_data[category]["pos_sims"], dim=1)
  pos_data = pd.DataFrame(pos_mat.numpy()).rename(i2t).transpose()
  
  neg_mat = torch.concat(mdl_data[category]["neg_sims"], dim=1)
  neg_data = pd.DataFrame(neg_mat.numpy()).rename(i2t).transpose()

  df_expanded = pd.concat(mdl_nn_data[category]["pos_nn_data"])
  pos_nn_embs = model.encode(df_expanded["sentence"].tolist(), normalize_embeddings=True)
  pos_nn_mat = util.cos_sim(taxonomy_embs, pos_nn_embs)
  pos_nn_data = pd.DataFrame(pos_nn_mat.numpy()).rename(i2t).transpose()
  print(f"Positive: {pos_mat.shape}, Negative: {neg_mat.shape}, Expanded: {pos_nn_mat.shape}")
  pos_data = pd.concat([pos_data, pos_nn_data])

  if raw:
    pos_mat = model.encode(pd.concat(mdl_data[category]["pos"])["sentence"].tolist(), normalize_embeddings=True)
    neg_mat = model.encode(pd.concat(mdl_data[category]["neg"])["sentence"].tolist(), normalize_embeddings=True)
    pos_data = pd.concat([pd.DataFrame(pos_mat), pd.DataFrame(pos_nn_embs)])
    neg_data = pd.DataFrame(neg_mat)
    print(f"Positive: {pos_mat.shape}, Negative: {neg_mat.shape}, Expanded: {pos_nn_embs.shape}")

  pos_data["label"] = 1
  neg_data["label"] = 0
  data = pd.concat([pos_data, neg_data], ignore_index=True)

  return data

def run_mdl(data, rf_max_depth=5, test=0.3):
  y_data = data["label"]
  x_data = data.drop("label", axis = 1)
  x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = test)

  #lm = LogisticRegression(penalty="none", solver="saga", max_iter=1000), MLPClassifier(random_state=1, hidden_layer_sizes=(384, 192, 96))
  mdl = curfc(max_depth=rf_max_depth) # this one standard
  #mdl = MLPClassifier(random_state=1, hidden_layer_sizes=(24, 12, 6))
  #mdl = XGBClassifier(n_estimators=4, max_depth=rf_max_depth, learning_rate=1, objective='binary:logistic')
  mdl.fit(x_training_data, y_training_data)
  predictions = mdl.predict(x_test_data)
  print(classification_report(y_test_data, predictions))
  return mdl

def run_mdl_full(data, estimator):
  y_data = data["label"]
  x_data = data.drop("label", axis = 1)

  mdl = estimator(None)
  mdl.fit(x_data, y_data)
  return mdl

estimators = {
    "RF": lambda x: curfc(max_depth=10),
    "MLP": lambda x: MLPClassifier(random_state=1, hidden_layer_sizes=(24, 12, 6)),
    "XGB": lambda x: XGBClassifier(n_estimators=4, max_depth=10, learning_rate=1, objective='binary:logistic'),
    "LR": lambda x: LogisticRegression(),
    #"SVM": SVC(kernel='rbf', degree=3, gamma='auto', C=1)
}

def write_sentence_data(mdl_data, mdl_nn_data, category, outfile):
  pos = pd.concat(mdl_data[category]["pos"] + mdl_nn_data[category]["pos_nn_data"])
  neg = pd.concat(mdl_data[category]["neg"])
  pos["label"] = 1
  neg["label"] = 0
  print(f"Positive: {pos.shape}, Negative: {neg.shape}, outfile: {outfile}")
  pd.concat([pos, neg]).to_csv(outfile, index=False)

write_sentence_data(mdl_data, mdl_nn_data, "competence_distrust", "competence_distrust_data.csv")
write_sentence_data(mdl_data, mdl_nn_data, "sincerity_distrust", "sincerity_distrust_data.csv")

# Finally!
prepped_data = {category: prepare_data(category, mdl_data, mdl_nn_data, model, taxonomy_embs, i2t, raw=False) for category in eptrust_categories}
mdls = {
    est_name: {
        cat: run_mdl_full(df, estimator) for cat, df in prepped_data.items()
    } for est_name, estimator in estimators.items()
}

def cls_(sents, mdls, model, taxonomy_embs, i2t, raw=False, sims=None):
  if sims is None:
    embs = model.encode(sents, normalize_embeddings=True)
    test_data = pd.DataFrame(util.cos_sim(taxonomy_embs, embs).numpy()).rename(i2t).transpose()
  else:
    test_data = pd.DataFrame(sims.numpy()).rename(i2t).transpose()
  if raw:
    test_data = pd.DataFrame(embs)
  df_res = pd.concat([mdl.predict_proba(test_data)[1].rename(cat) for cat, mdl in mdls.items()], axis=1) # RFs
  #df_res = pd.concat([pd.Series(mdl.predict_proba(test_data)[:,1]).rename(cat) for cat, mdl in mdls.items()], axis=1) # almost everything else lol
  return df_res.rename(index={i: s for i, s in enumerate(sents)})
cls = partial(cls_, mdls=mdls["RF"], model=model, taxonomy_embs=taxonomy_embs, i2t=i2t)

# Example table in the paper
sents = ["you are ignorant and a liar", "democrats are idiots", "democrats are liars", "republicans are dumb", "you're smart", "democrats are intelligent, trustworthy", "Alice is trustworthy, but Bob is stupid."]
res = cls(sents)
res.round(4)

threshold = 0.5
res.mask(res > threshold, "*").mask(res <= threshold, "")

# Readout models
fname = "models-" + datetime.today().astimezone(pytz.timezone('US/Central')).strftime('%Y-%m-%d-%H%M') + ".pkl"
with open(fname, "wb") as f:
  pkl.dump(mdls, f)

# Running the models on your own data, assuming it's a dataframe with a "sentence" column (feel free to precompute with the "sims" arg):
# cls(sents=df["sentence"].tolist())