from src.data.preprossed_data import PreprossedData
from catboost import CatBoostRegressor, Pool
from typing import Optional
import optuna
from sklearn.metrics import (mean_absolute_error)


class DataToModel:

    def __init__(self):
        self.trainPool = None
        self.valPool = None
        self.testPool = None


    def makeReadyDf(self):
        df = PreprossedData().makePreprocessingData()

        drop_cols = ['order_id', 'order_datetime', 'customer_id', 'courier_id', 'start_date', 'restaurant_id',
                     'open_date', 'registration_date']

        target = 'delivery_time_minutes'

        q60 = df['order_datetime'].quantile(0.6)
        q80 = df['order_datetime'].quantile(0.8)

        train_df = df[df['order_datetime'] < q60]
        val_df = df[(df['order_datetime'] >= q60) & (df['order_datetime'] < q80)]
        test_df = df[df['order_datetime'] >= q80]

        X_train, y_train = train_df.drop(target, axis=1), train_df[target]
        X_val, y_val = val_df.drop(target, axis=1), val_df[target]
        X_test, y_test = test_df.drop(target, axis=1), test_df[target]

        X_train = X_train.drop(drop_cols, axis=1)
        X_val = X_val.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)

        cat_features = X_train.select_dtypes(['object']).columns.to_list()

        self.trainPool = Pool(X_train, y_train, cat_features=cat_features)
        self.valPool = Pool(X_val, y_val, cat_features=cat_features)
        self.testPool = Pool(X_test, y_test, cat_features=cat_features)

    def getData(self):
        return self.trainPool, self.valPool, self.testPool




class HyperParamOptimizer:
    def __init__(self, dataloader: DataToModel):
        self.params = {}
        self.dataloader = dataloader

    def findParams(self):
        self.dataloader.makeReadyDf()
        trainPool, valPool, _ = self.dataloader.getData()

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1.0, 10.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 30),
            }

            model = CatBoostRegressor(**params)
            model.fit(trainPool, eval_set=valPool,
                      early_stopping_rounds=75,
                      use_best_model=True)

            y_pred = model.predict(valPool)
            mae = mean_absolute_error(valPool.get_label(), y_pred)
            return mae

        study = optuna.create_study(study_name='cb_model', direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=15)

        self.params = study.best_params

    def getParams(self):
        return self.params


class LearningModel:

    def __init__(self, dataloader: DataToModel, optimizer: HyperParamOptimizer):
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.model: Optional[CatBoostRegressor] = None

    def startTraining(self):


        trainPool, valPool, testPool = self.dataloader.getData()
        params = self.optimizer.getParams()
        cat_features = trainPool.select_dtypes(['object']).columns.to_list()



        self.model = CatBoostRegressor(
            **params,
            cat_features=cat_features,
            early_stopping_rounds=75
        )


        self.model.fit(
            trainPool,
            eval_set=valPool,
            use_best_model=True,
            verbose=100  # показываем прогресс каждые 100 итераций
        )

        self.model.save_model('../model/prod_model.cbm')


if __name__ == '__main__':
    dataloader = DataToModel()
    dataloader.makeReadyDf()
    X_train, _, _ = dataloader.getData()
    print(X_train.shape())
