mkdir data

# We took the biasbios data from: https://github.com/IBM/sensitive-subspace-robustness
# and performed the splits into train/dev/test to 65/10/25 percent accordingly with a random split per profession.
# This is the same proportions from the original paper: https://arxiv.org/abs/1901.09451

mkdir -p data/embeddings

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip -P data/embeddings/
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P data/embeddings/
unzip data/embeddings/crawl-300d-2M.vec.zip -d data/embeddings/                 
unzip data/embeddings/glove.42B.300d.zip -d data/embeddings/ 

mkdir -p data/biasbios
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle -P data/biasbios/

mkdir -p data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_pos.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_neg.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_pos.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_neg.npy -P data/deepmoji

mkdir -p data/bert_encode_biasbios
wget https://storage.googleapis.com/ai2i/nullspace/bert_encode_biasbios/train_avg.npy -P data/bert_encode_biasbios
wget https://storage.googleapis.com/ai2i/nullspace/bert_encode_biasbios/train_cls.npy -P data/bert_encode_biasbios
wget https://storage.googleapis.com/ai2i/nullspace/bert_encode_biasbios/dev_avg.npy -P data/bert_encode_biasbios
wget https://storage.googleapis.com/ai2i/nullspace/bert_encode_biasbios/dev_cls.npy -P data/bert_encode_biasbios
wget https://storage.googleapis.com/ai2i/nullspace/bert_encode_biasbios/test_avg.npy -P data/bert_encode_biasbios
wget https://storage.googleapis.com/ai2i/nullspace/bert_encode_biasbios/test_cls.npy -P data/bert_encode_biasbios