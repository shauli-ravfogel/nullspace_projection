local data_path = "../data/emoji_sent_race/";
{
  "dataset_reader": {
    "type": "deep_moji_reader",
  },
  "train_data_path": data_path + "train",
  "validation_data_path": data_path + "dev",
  //"test_data_path": data_path + "test",
  "evaluate_on_test": false,

  "model": {
    "type": "model_deep_moji",
    "emb_size": 2304,
    "mlp_dropout": 0.2
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 10,
    "grad_norm": 1.0,
    "patience" : 3,
    "cuda_device" : 0,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adam"
    }
  }
}
