from rootml.train.xgb_trainer import XGBTrainer


def run_training(data_path, config, out_dir):

    model_type = config["model"]

    if model_type == "xgboost":
        trainer = XGBTrainer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    trainer.train(data_path, config, out_dir)
