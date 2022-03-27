import pandas as pd

dataPath = "./data/covid19_partition_1.2020-06-01_00.jsonl"

twitterdf = pd.read_json(dataPath, lines=True)
print(twitterdf.info())
count = 0
if twitterdf["retweet_count"] > 0:
    count+=1
    
print(count)
# print(twitterdf.head(10))