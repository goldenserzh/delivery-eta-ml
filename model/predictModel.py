from catboost import CatBoostRegressor, Pool
import logging
import pandas as pd



class PredictModel:

    def __init__(self, model_path='../model/prod_model.cbm'):
        self.model = None
        self.predictions = None
        self._load_model(model_path)

    def _load_model(self, model_path):
        try:
            self.model = CatBoostRegressor().load_model(model_path)

        except FileNotFoundError:
            logging.error(('Файл не найден'))
            raise

        except Exception:
            logging.error(f"Ошибка загрузки модели")
            raise

    def predict(self, data):

        if self.model is None:
            raise RuntimeError("Модель не найдена")


        if not isinstance(data, pd.DataFrame):
            logging.error("Входные данные должны быть pandas DataFrame")
            raise

        if data.empty:
            raise ValueError("Данных нет")

        try:
            cat_features = data.select_dtypes(['object']).columns
            if cat_features:
                dataPool = Pool(data, cat_features=cat_features)
                self.predictions = self.model.predict(dataPool)
            else:
                self.predictions = self.model.predict(data)

            return self.predictions
        except Exception as e:
            logging.error(f"Ошибка при предсказании: {e}")
            raise