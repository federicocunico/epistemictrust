import zstandard as zstd
import pandas as pd
import numpy as np
import json, csv, os, re, sqlite3, itertools
import multiprocessing as mp
import pickle as pkl
from urllib.parse import urlparse

### 1. Decompression

# For a notebook environment
# url = "https://files.pushshift.io/reddit/comments/RC_2020-01.zst"
# fname = os.path.basename(urlparse(url).path)
# dirname = os.path.splitext(fname)[0]

# !mkdir {dirname}
# !mkdir processed
# !wget {url}

# You'll need to change these and re-run with however many archives you want to unpack
dirname = "RC_2020-01"
fname = "RC_2020-01.zst"

# PushShift data import code

def decompress_zst(pushshift_zst_file, fields, pattern="csv-%d.csv", verbose=True, filter=None):
  data = []
  with open(pushshift_zst_file, 'rb') as fh:
      dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
      with dctx.stream_reader(fh) as reader:
          previous_line = ""
          chunk_count = 0
          file_count = 0
          while True:
              chunk = reader.read(2**24)  # 16mb chunks
              if not chunk:
                  break

              string_data = chunk.decode('utf-8')
              lines = string_data.split("\n")
              for i, line in enumerate(lines[:-1]):
                  if i == 0:
                      line = previous_line + line
                  object = json.loads(line)
                  if fields is not None:
                    if filter is not None:
                      if not filter(object):
                        continue
                    data.append(tuple(object[fld] for fld in fields))
                  else:
                    data.append(tuple(object))
              previous_line = lines[-1]
              chunk_count = chunk_count + 1
              if verbose and chunk_count % 10 == 0: print("Processed %dMB" % (chunk_count * 16))
              if chunk_count % 200 == 0:
                  with open(pattern % file_count, 'w') as out:
                      csv_out = csv.writer(out)
                      if fields is not None:
                        csv_out.writerow(fields)
                      for row in data:
                          csv_out.writerow(row)
                  file_count = file_count + 1
                  data = []
  with open(pattern % file_count,'w') as out: # Remainder
      csv_out = csv.writer(out)
      if fields is not None:
        csv_out.writerow(fields)
      for row in data:
          csv_out.writerow(row)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
def files(dir):
  return sorted_alphanumeric([dir + x for x in os.listdir(dir)])

def summarise(csv_file, column_a, column_b):
  print("Processing %s" % csv_file)
  df = pd.read_csv(csv_file)
  df = df.loc[df['author'] != "[deleted]"]
  return df.groupby(by=[column_a, column_b]).size().reset_index(name="n")

def process_comments_submissions(csv_file, cur):
    print("Processing %s" % csv_file)
    df = pd.read_csv(csv_file)[["author", "link_id", "created_utc", "id"]]
    cur.executemany("INSERT INTO comments_submissions (author, link_id, created_utc, id) VALUES (?, ?, ?, ?)", list(df.itertuples(index=False, name=None)))

decompress_zst(fname,
               ['author', 'author_created_utc',
                'created_utc', 'id', 'link_id',
                'parent_id', 'permalink', 'subreddit',
                'subreddit_id', 'score', 'controversiality', 'body'],
               f"{dirname}/comments-%d.csv")

### 2. Crossover graph

#!csvtool namedcol subreddit,author RC_2020-01/*.csv | sort | uniq -c > users.csv # | sed 's/ /,/g' > users.csv
#!cat users.csv | sed "s/^[ \t]*//" | sed 's/ /,/g' > users2.csv

users = pd.read_csv("users2.csv", names=["n", "subreddit", "author"])

ht = {} # 30 mins lmao
for index, row in users.iterrows():
  if row["subreddit"] not in ht:
    ht[row["subreddit"]] = {}
  ht[row["subreddit"]][row["author"]] = row["n"]

counts = {sub: sum([v for u, v in users.items() if u not in {"[deleted]", "[removed]"}]) for sub, users in ht.items()}
subs_filtered = [sub for sub, n in counts.items() if n >= 1000]

import requests
link = "https://raw.githubusercontent.com/valentinhofmann/politosphere/main/data/subreddits.txt"
f = requests.get(link)
subreddits = f.text.split("\n") + ["conspiracy"]

combs = reduce(lambda x, y: x + y, [[(x, y) for y in subs_filtered] for x in subreddits])

res = [] # 14 mins
for x, y in combs:
  if x not in ht or y not in ht:
    continue
  count = sum( [n for author, n in ht[x].items() if author in ht[y] and author not in {"[deleted]", "[removed]"}] )
  res.append((x, y, count))
graph = pd.DataFrame([(suba, subb, n, counts[suba], 100*n/counts[suba]) for suba, subb, n in res if counts[suba] != 0 and suba != subb],
                     columns=["sub_a", "sub_b", "crossover", "total_a", "pct"])

similar_subs = graph.sort_values(["sub_a", "pct"], ascending=[True, False])#.groupby("sub_a").head(20)
n_docs = len(similar_subs["sub_a"].unique())
idf = np.log( n_docs / similar_subs.groupby("sub_b").size() ).reset_index(name="idf")
similar_subs = similar_subs.merge(idf, on="sub_b")
similar_subs["tfidf"] = similar_subs["crossover"] * similar_subs["idf"]

# politiosphere + expanded + conspiracy = list of expanded subreddits
# I then manually applied judgement to remove several. subreddit_sample.txt is the final list
expanded_subs = sorted( [(x, counts[x]) for x in similar_subs["sub_b"].unique().tolist()], key=lambda x: -x[1])
conspiracy_subs = ["conspiracy",
                         "conspiracy_commons",
                         "conspiracytheories",
                         "nonewnormal",
                         "coronaviruscirclejerk",
                         "aliens",
                         "walkaway",
                         "lockdownskepticism",
                         "ufos",
                         "conservatives",
                         "anarcho_capitalism",
                         "ufo",
                         "louderwithcrowder",
                         "republican",
                         "asktrumpsupporters",
                         "collapse",
                         "shitpoliticssays",
                         "joerogan",
                         "conservative",
                         "socialjusticeinaction"]

### 3. Sentence splitting

# After you've decompressed everything, I found csvtool + sed to be faster to clean and get the columns you want

# I like to zero pad the processed files too
# %cd processed
# !rename 's/\d+/sprintf("%05d", $&)/e' *.csv # replace any numbers with zero-padded
# %cd /content

bots = ["AutoModerator", "PoliticsModeratorBot", "autotldr", "SnapshillBot", "CenturionBot", "WikiTextBot", "BotForceOne",
        "nwordcountbot", "electioninfobot", "jobautomator", "AmputatorBot", "MAGICEYEBOT", "userleansbot", "HCEReplacementBot",
        "SmileBot-2020", "RepostSleuthBot", "rConBot", "jre-mod", "DeltaBot", "GoodNewsBot", "VredditDownloader", "RemindMeBot",
        "sneakpeekbot", "CryptoMods", "POTUSArchivistBot", "RealTweetOrNotBot", "rTrumpTweetsBot", "groupbot", "stthomasbot", "voteleft-bot",
        "CivilServantBot", "imdadbot", "geopoliticsbanbot", "ClickableLinkBot", "caucusvote-bot", "smile-bot-2019", "HelperBot",
        "ContextualRobot", "rBitcoinMod", "randia-bot", "BotThatSaysBro", "TrumpBrickBot", "outlinelinkbot", "TrumpTrainBot", "SenatorsInfoBot",
        "tutestbot", "BM2018Bot", "ukpolbot"]

with open("subreddit_sample.txt", "r") as f:
  subreddit_sample = [x for x in f.read().split("\n") if x != ""]

for f in files("rawsents/"):
  print(f"Processing {f}...")
  df = pd.read_csv(f)
  df = df.query("subreddit == @subreddit_sample")[["author", "subreddit", "id", "body"]] # should probably filter AutoModerator here (and other known bots)
  df.to_csv(f"preprocessed/{os.path.basename(f)}")

# !csvtool namedcol id,author,subreddit,body preprocessed/* | \
#   sed 's/\!\{0,1\}\[[^]]*\]([^)]*)//g' | \
#   sed 's/[^[:alnum:][:punct:][:space:]]//g' | \
#   sed 's/\[deleted\]//g' | \
#   sed 's/\[removed\]//g' | \
#   sed "s/[^A-Za-z0-9 \.,\:\!\?\/'\"-]//g" | tr -s ' ' \
#   > subreddit_sample_comments.csv

df = pd.read_csv("subreddit_sample_comments.csv")

new_df = []
for index, row in df.query("author != @bots").iterrows():
  if index % 1000000 == 0:
    print(f"Processed {index}")
  body = str(row["body"])
  if body == "nan":
    continue
  sents = [x.strip() for x in re.split("\n|\.", body)]
  new_df.extend([(row["id"], sent) for sent in sents if sent != ""])
#new_df = pd.DataFrame(new_df, columns=["id", "sentence"])

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

new_df_chunks = chunks(new_df, 1000000)

for i, chunk in enumerate(new_df_chunks):
  pd.DataFrame(chunk, columns=["id", "sentence"]).dropna().to_csv(f"processed/sentences_{i}.csv", index=False)
