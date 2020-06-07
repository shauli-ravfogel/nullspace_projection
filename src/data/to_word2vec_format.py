import sys
from gensim.scripts.glove2word2vec import glove2word2vec
import shutil
import os

path = sys.argv[1]
glove2word2vec(glove_input_file=path, word2vec_output_file=path+".tmp")
shutil.copyfile(path+".tmp", path)
os.remove(path+".tmp")
