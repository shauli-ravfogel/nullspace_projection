// Configuration for a the baseline model with w2v embeddings used in tacl paper


local data_path = "../data/sent_race.jsonl";
local emb_dim = 300;
local emb_out = 200;
{
  "dataset_reader": {
    "type": "classification_reader",
    "token_indexers": {
            "tokens": { "type": "single_id" }
        }
  },
  "train_data_path": data_path,// + "train.jsonl",
  "validation_data_path": data_path,// + "dev.jsonl",
  //"test_data_path": "allennlp_models/data_converters/real_train_to_dobj.jsonl",
  "evaluate_on_test": false,

  "model": {
    "type": "model_base",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/word2vec/GoogleNews-vectors-negative300.txt.gz",
            //"pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
            "embedding_dim": 300,
            "trainable": false
        }
      }
    },
    "emb_size": 300,
    "mlp_dropout": 0.2
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 50,
    "grad_norm": 1.0,
    "patience" : 20,
    "cuda_device" : -1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 5
    },
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  }
}
