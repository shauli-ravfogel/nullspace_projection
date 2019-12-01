local data_path = "../data/sent_race.jsonl";
{
  "dataset_reader": {
    "type": "classification_reader",
    "token_indexers": {
			"bert": {
		    	"type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "truncate_long_sequences": false,
    		}
	}
  },
  "train_data_path": data_path + "train.jsonl",
  "validation_data_path": data_path + "dev.jsonl",
  //"test_data_path": data_path + "test.jsonl",
  "evaluate_on_test": false,

  "model": {
    "type": "model_base",
    "text_field_embedder": {
	  	"allow_unmatched_keys": true,
      	"embedder_to_indexer_map": {
          "bert": ["bert", "bert-offsets"]
   		},

      	"bert": {
        	"type": "bert-pretrained",
        	"pretrained_model": "bert-base-uncased",
	        "requires_grad": true,
    	    "top_layer_only": false
        }
    },
    "emb_size": 768,
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
    "cuda_device" : 1,
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
