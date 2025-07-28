import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'XGBRegressor': XGBRegressor(eval_metric='rmse')
            }

            param_grids = {
                'RandomForestRegressor': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20]
                },
                'GradientBoostingRegressor': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1]
                },
                'AdaBoostRegressor': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1, 1.0]
                },
                'LinearRegression': {},
                'KNeighborsRegressor': {
                    'n_neighbors': [3, 5, 7]
                },
                'Ridge': {
                    'alpha': [ 5.0, 10.0, 15.0]
                },
                'DecisionTreeRegressor': {
                    'max_depth': [None, 10, 20]
                },
                'CatBoostRegressor': {
                    'depth': [4, 6],
                    'learning_rate': [0.01, 0.1]
                },
                'XGBRegressor': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
            }

            model_report = {}
            best_params_report = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name} with hyperparameter tuning")
                params = param_grids.get(model_name, {})
                if params:
                    search = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_score = search.best_score_
                    best_params_report[model_name] = search.best_params_
                    logging.info(f"{model_name} best params: {search.best_params_}")
                else:
                    model.fit(X_train, y_train)
                    best_model = model
                    best_score = best_model.score(X_train, y_train)
                    best_params_report[model_name] = "Default"
                y_pred = best_model.predict(X_test)
                r2_square = r2_score(y_test, y_pred)
                model_report[model_name] = r2_square
                logging.info(f"{model_name} R2 Score: {r2_square}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model_params = best_params_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")
            logging.info(f"Best Params for {best_model_name}: {best_model_params}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )

            return best_model_name, best_model_score, best_model_params

        except Exception as e:
            raise CustomException(e, sys)