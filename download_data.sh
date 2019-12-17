mkdir data

# We took the biasbios data from: https://github.com/IBM/sensitive-subspace-robustness
# and performed the splits into train/dev/test to 65/10/25 percent accordingly with a random split per profession.
# This is the same proportions from the original paper: https://arxiv.org/abs/1901.09451
mkdir data/biasbios
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle -P data/biasbios/

mkdir data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_pos.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/pos_neg.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_pos.npy -P data/deepmoji
wget https://storage.googleapis.com/ai2i/nullspace/deepmoji/neg_neg.npy -P data/deepmoji