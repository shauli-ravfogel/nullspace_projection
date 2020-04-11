import pandas as pd
import pickle

with open("words_before_and_after.pickle", "rb") as f:
        words = pickle.load(f)
columns = ["Word", "Neighbors before", "Neighbors after"]
data = []
for w in list(words.keys())[:80]:

        before, after = list(words[w]["before"]), list(words[w]["after"])
        data.append([w, ", ".join(before), ", ".join(after)])
                
df = pd.DataFrame(data, columns=columns)



with open('mytable.tex','w') as tf:
    tf.write(df.to_latex(index = False, longtable = False))
