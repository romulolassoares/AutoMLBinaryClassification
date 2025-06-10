import os
import joblib
import pandas as pd

from typing import List, Tuple

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


class Automl:
    def __init__(
        self,
        dataset:pd.DataFrame,
        filename:str,
        metric:str = "accuracy",
        target_position:int = -1
    ) -> None:
        self.filename = filename
        self.metric = metric
        self._target_position = target_position
        self._dataset = dataset
        
        self.target = self._dataset.columns[self._target_position]
        self.X = self._dataset.drop(columns=[self.target])
        self.y = self._dataset[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.3,
            stratify=self.y,
            random_state=42 # Answer for everyfing
        )
        self.preprocessor = self._create_preprocessor()
        
    def _create_preprocessor(self):
        num_features = self.X.select_dtypes(include=["int64", "float64"]).columns
        cat_features = self.X.select_dtypes(include=["object","category"]).columns
        
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        
        return ColumnTransformer([
            ("num", num_pipeline, num_features),
            ("cat", cat_pipeline, cat_features)
        ])

    def find_best_models(self, n:int) -> List[Pipeline]:
        models = {
            "RandonForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(),
            "GradientBoosting": GradientBoostingClassifier(),
            "SVC": SVC(probability=True)
        }
        
        results = []
        for name, model in models.items():
            pipeline = Pipeline([
                ("preprocessing", self.preprocessor),
                ("classifier", model)
            ])
            score = cross_val_score(
                pipeline,
                self.X_train,
                self.y_train,
                cv=3,
                scoring=self.metric
            ).mean()
            # results.append(score)
            results.append((pipeline, score))
        results.sort(key=lambda x: x[1], reverse=True)
        top_models = results[:n]
        

        return [pipe.fit(self.X_train, self.y_train) for pipe, _ in top_models]
        
    def tuning_model(self, pipeline: Pipeline) -> Pipeline:
        estimator = pipeline.named_steps["classifier"]
        
        param_grid = {
            RandomForestClassifier: {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [None, 10, 20]
            },
            GradientBoostingClassifier: {
                "classifier__learning_rate": [0.01, 0.1],
                "classifier__n_estimator": [100, 200]
            },
            LogisticRegression: {
                "classifier__C": [0.1, 1, 10],
                "classifier__penalty": ["l2"]
            },
            SVC: {
                "classifier__C": [0.01, 1, 10],
                "classifier__kernel": ["linear", "rbf"]
            }
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
            random_state=42 # Answer for everyfing
        )
        search.fit(self.X_train, self.y_train)
        return search.best_estimator_
    
    def save_model(self, pipeline: Pipeline, model_name:str = "") -> str:
        if not model_name:
            model_name = type(pipeline.named_steps["classifier"]).__name__
            
        model_name = model_name.replace(" ", "_")
        base_path = "./tmp/files"
        model_file_name = f"{self.filename.split('.')[0]}_{model_name}.pkl"
        pickle_file = os.path.join(base_path, model_file_name)
        
        if os.path.exists(pickle_file):
            pickle_file = pickle_file.replace(".pkl", "_new.pkl")
        
        joblib.dump(pipeline, pickle_file)
        return pickle_file
    
    def eval_model(self, pipeline:Pipeline) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        y_pred = pipeline.predict(self.X_test)
        
        y_proba = None
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_proba = pipeline.predict_proba(self.X)
            
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Teste Accurracy: {accuracy:.4f}")
        return self.X_train, self.y_train, self.X_test, self.y_test, pd.DataFrame(y_pred, columns=["predictions"])