import os
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from utils.create_folders import create_dirs


def handle_with_config_data(config: dict):
    if config is None:
        config = {}
    if "paths" not in config:
        config["paths"] = {"base": "tmp"}

    if "saved_models" not in config["paths"]:
        config["paths"]["saved_models"] = "files"

    create_dirs(config)

    return config


class Automl:
    """Automl class for automating the process of training, tuning,
    evaluating and saving machine learning models.

    This class simplifies common machine learning tasks using
    `scikit-learn`:
    - Splitting a dataset into training and testing sets.
    - Automatically preprocessing numeric and categorical features.
    - Training and comparing multiple models using cross-validation.
    - Tuning the best models using randomized search.
    - Saving trained models to disk.
    - Evaluating model performance on a held-out test set.

    Attributes:
        filename (str): The filename used as a base for saving trained
            models.
        metric (str): The scoring metric used for evaluation
            (e.g. "accuracy").
        saved_models_path (str): Path where trained models are saved.
        X_train, X_test (pd.DataFrame): Feature subsets for training and
            testing.
        y_train, y_test (pd.DataFrame): Target subsets for training and
            testing.
        preprocessor (sklearn.compose.ColumnTransformer): Data
            preprocessing pipeline.

    Parameters:
        dataset (pd.DataFrame): The full dataset including features and
            target column.
        filename (str): Filename used as base name for saving models.
        metric (str, optional): Scoring metric to optimize during model
            selection and tuning. Defaults to "accuracy".
        target_position (int, optional): Index of the target column in
            the dataset. Defaults to -1 (last column).
        config (dict, optional): Configuration dictionary. If not
            provided, defaults are used.

    Methods:
        find_best_models(n: int) -> List[Pipeline]
        tuning_model(pipeline: Pipeline) -> Pipeline
        save_model(pipeline: Pipeline, model_name: str = "") -> str
        eval_model(pipeline: Pipeline) -> Tuple[...]

    Notes:
        - Models supported for training and tuning include:
            - RandomForestClassifier
            - GradientBoostingClassifier
            - LogisticRegression
            - SVC
        - Uses OneHotEncoder for categorical features and StandardScaler
          for numeric features.
        - Uses 3-fold cross-validation and a fixed random seed for
          reproducibility.
    """
    def __init__(
        self,
        dataset: pd.DataFrame,
        filename: str,
        metric: str = "accuracy",
        target_position: int = -1,
        config: dict = {},
    ) -> None:
        """
        Initializes the Automl object by preparing dataset splits and
        preprocessing pipeline.

        Parameters
        ----------
        dataset : pd.DataFrame
            Full dataset containing both features and the target column.
        filename : str
            The base filename to use for saving the trained model.
        metric : str, optional
            Scoring metric to use for model evaluation and tuning.
            Default is "accuracy".
        target_position : int, optional
            The index of the target column in the dataset. Default is -1
            (last column).
        config : dict, optional
            Configuration dictionary with optional keys:
            - "paths": {
                "base": str,  # Base path for saving outputs
                "saved_models": str  # Subfolder for saving models
            }
        """
        self.filename = filename
        self.metric = metric
        self._target_position = target_position
        self._dataset = dataset

        self.target = self._dataset.columns[self._target_position]
        self.X = self._dataset.drop(columns=[self.target])
        self.y = self._dataset[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=0.3,
                stratify=self.y,
                random_state=42,  # Answer for everything
            )
        )
        self.preprocessor = self._create_preprocessor()

        config = handle_with_config_data(config)
        base_path = config["paths"]["base"]
        saved_models_path = config["paths"]["saved_models"]
        self.saved_models_path = os.path.join(base_path, saved_models_path)

    def _create_preprocessor(self):
        """
        Creates a preprocessing pipeline for numerical and categorical
        features.

        Returns
        -------
        sklearn.compose.ColumnTransformer
            A transformer that applies:
            - Mean imputation and standard scaling to numerical features.
            - Mode imputation and one-hot encoding to categorical
              features.
        """
        num_features = self.X.select_dtypes(
            include=["int64", "float64"]
        ).columns
        cat_features = self.X.select_dtypes(
            include=["object", "category"]
        ).columns

        num_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            [
                ("num", num_pipeline, num_features),
                ("cat", cat_pipeline, cat_features),
            ]
        )

    def find_best_models(self, n: int) -> List[Pipeline]:
        """
        Trains multiple classification models and selects the top-n based
        on cross-validation score.

        Parameters
        ----------
        n : int
            Number of top-performing models to return.

        Returns
        -------
        List[Pipeline]
            List of fitted sklearn Pipelines sorted by performance.

        Raises
        ------
        ValueError
            If `n` is less than 1 or more than the number of available
            models.
        """
        models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(),
            "GradientBoosting": GradientBoostingClassifier(),
            "SVC": SVC(probability=True),
        }

        results = []
        for model in models.values():
            pipeline = Pipeline(
                [
                    ("preprocessing", self.preprocessor),
                    ("classifier", model),
                ]
            )
            score = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=3,
                scoring=self.metric,
            ).mean()
            results.append((pipeline, score))
        results.sort(key=lambda x: x[1], reverse=True)
        top_models = results[:n]

        return [
            pipe.fit(self.X_train, self.y_train) for pipe, _ in top_models
        ]

    def tuning_model(self, pipeline: Pipeline) -> Pipeline:
        """
        Performs hyperparameter tuning on a model pipeline using
        randomized search.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            A scikit-learn pipeline containing a classifier as its final
            step.

        Returns
        -------
        sklearn.pipeline.Pipeline
            The pipeline with the best hyperparameters found, or the
            original pipeline if tuning is unsupported.

        Notes
        -----
        Supported classifiers:
        - RandomForestClassifier
        - GradientBoostingClassifier
        - LogisticRegression
        - SVC
        """
        estimator = pipeline.named_steps["classifier"]

        param_grid = {
            RandomForestClassifier: {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [None, 10, 20],
            },
            GradientBoostingClassifier: {
                "classifier__learning_rate": [0.01, 0.1],
                "classifier__n_estimators": [100, 200],
            },
            LogisticRegression: {
                "classifier__C": [0.1, 1, 10],
                "classifier__l1_ratio": [0],
            },
            SVC: {
                "classifier__C": [0.01, 1, 10],
                "classifier__kernel": ["linear", "rbf"],
            },
        }

        model_type = type(estimator)
        if model_type not in param_grid:
            return pipeline

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid[model_type],
            n_iter=3,
            cv=3,
            scoring=self.metric,
            verbose=0,
            random_state=42,  # Answer for everything
        )
        search.fit(self.X_train, self.y_train)
        return search.best_estimator_

    def save_model(self, pipeline: Pipeline, model_name: str = "") -> str:
        """
        Saves a trained model pipeline to disk as a .pkl file.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            The trained model pipeline to save.
        model_name : str, optional
            Custom name for the model file. Defaults to classifier class name.

        Returns
        -------
        str
            Full file path to the saved model.

        Raises
        ------
        OSError
            If the model file cannot be written to the disk.
        """
        if not model_name:
            model_name = type(
                pipeline.named_steps["classifier"]
            ).__name__

        model_name = model_name.replace(" ", "_")
        base_path = self.saved_models_path

        file_name = self.filename.split("/")[-1]
        file_name = file_name.split(".")[0]

        model_file_name = f"{file_name}_{model_name}.pkl"
        pickle_file = os.path.join(base_path, model_file_name)

        if os.path.exists(pickle_file):
            pickle_file = pickle_file.replace(".pkl", "_new.pkl")

        joblib.dump(pipeline, pickle_file)
        return pickle_file

    def eval_model(
        self, pipeline: Pipeline
    ) -> Dict[str, Any]:
        """
        Evaluates a trained model pipeline on the test dataset.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Trained pipeline to evaluate.

        Returns
        -------
        Dict
            A dictionary containing:
            - accuracy (float): Accuracy score on the test set.
            - X_train (pd.DataFrame): Training feature set.
            - y_train (pd.Series): Training target values.
            - X_test (pd.DataFrame): Test feature set.
            - y_test (pd.Series): Test target values.
            - predictions (pd.DataFrame): Model predictions on the test
              set.
            - model_name (str): Name of the model.
        """
        y_pred = pipeline.predict(self.X_test)

        y_proba = None
        if hasattr(
            pipeline.named_steps["classifier"], "predict_proba"
        ):
            y_proba = pipeline.predict_proba(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)

        model_name = type(pipeline.named_steps["classifier"]).__name__

        return {
            "accuracy": accuracy,
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "predictions": pd.DataFrame(y_pred, columns=["predictions"]),
            "model_name": model_name,
        }
