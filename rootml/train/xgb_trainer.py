import json
import os

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from rootml.train.base import BaseTrainer


class XGBTrainer(BaseTrainer):

    def train(self, data_path, config, out_dir):

        os.makedirs(out_dir, exist_ok=True)

        df = pd.read_parquet(data_path)
                # Flatten ROOT-exported columns (unwrap 1-element arrays)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: x[0] if hasattr(x, "__len__") else x
                )
        
        # Ensure numeric types
        for col in config["features"] + [config["target"], config["weight"]]:
            df[col] = pd.to_numeric(df[col])



        X = df[config["features"]]
        y = df[config["target"]]
        w = df.get(config.get("weight"))

        X_train, X_tmp, y_train, y_tmp, w_train, w_tmp = train_test_split(
            X, y, w,
            test_size=config["test_size"] + config["val_size"],
            random_state=config["seed"]
        )

        val_frac = config["val_size"] / (
            config["test_size"] + config["val_size"]
        )

        X_val, X_test, y_val, y_test, w_val, w_test = train_test_split(
            X_tmp, y_tmp, w_tmp,
            test_size=1 - val_frac,
            random_state=config["seed"]
        )

        model = xgb.XGBClassifier(**config["xgb_params"])

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, preds, sample_weight=w_test)

        model.save_model(os.path.join(out_dir, "model.json"))

        metrics = {"auc": float(auc)}

        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print("AUC:", auc)
        
        # Predict on full dataset
        probs = model.predict_proba(X)[:, 1]

        df_out = df.copy()
        df_out["ml_score"] = probs

        df_out.to_parquet(
            f"{out_dir}/scores.parquet",
            index=False
        )

