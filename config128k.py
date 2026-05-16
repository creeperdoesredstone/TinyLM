CONFIG = {
		"seq_len": 64,
		"batch_size": 4,

		"embed_dim": 192,
		"hidden_dim": 512,
		"num_layers": 3,

		"vocab_size": 1024,

		"max_lr": 1e-3,
		"min_lr": 8e-5,
		"warmup_steps": 2000,

		"label_smoothing": 0.0,
		"dropout": 0.1,
		"use_gradient_checkpointing": False
	}