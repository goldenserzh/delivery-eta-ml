import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
from loguru import logger

from model.learnig_model import HyperParamOptimizer, LearningModel
from model.predictModel import  PredictModel
from catboost import CatBoostRegressor

app = FastAPI()


class OrderFeatures(BaseModel):
    """Признаки заказа для предсказания времени доставки"""

    # === Числовые признаки (без пропусков) ===
    order_amount: float = Field(..., ge=0, description="Сумма заказа")
    items_count: int = Field(..., ge=0, description="Количество позиций")
    distance_km: float = Field(..., ge=0, description="Расстояние в км")
    is_express_delivery: int = Field(..., ge=0, le=1, description="Экспресс-доставка (0/1)")

    account_age_days: int = Field(..., ge=0, description="Возраст аккаунта в днях")
    customer_rating: float = Field(..., ge=0, le=5, description="Рейтинг клиента")

    experience_days: int = Field(..., ge=0, description="Опыт курьера в днях")
    base_skill: float = Field(..., ge=0, description="Базовый навык курьера")
    base_speed_kmh: float = Field(..., ge=0, description="Базовая скорость курьера")
    courier_rating: float = Field(..., ge=0, le=5, description="Рейтинг курьера")
    on_time_ratio: float = Field(..., ge=0, le=1, description="Доля вовремя доставленных")

    is_fast_food: int = Field(..., ge=0, le=1, description="Фастфуд (0/1)")
    restaurant_rating: float = Field(..., ge=0, le=5, description="Рейтинг ресторана")
    prep_time_avg: int = Field(..., ge=0, description="Среднее время готовки")

    is_rush_hour: int = Field(..., ge=0, le=1, description="Час пик (0/1)")

    # === Числовые с пропусками (Optional) ===
    temperature: Optional[float] = Field(None, description="Температура (°C)")
    precip_mm: Optional[float] = Field(None, ge=0, description="Осадки в мм")
    wind_speed: Optional[float] = Field(None, ge=0, description="Скорость ветра")
    traffic_level: Optional[float] = Field(None, ge=0, description="Уровень трафика")

    # === Категориальные признаки ===
    payment_method: str = Field(..., description="Способ оплаты")
    customer_city: str = Field(..., description="Город клиента")
    preferred_payment_method: str = Field(..., description="Предпочитаемый способ оплаты")
    mobile_os: str = Field(..., description="Мобильная ОС")
    vehicle_type: str = Field(..., description="Тип транспорта курьера")
    restaurant_city: str = Field(..., description="Город ресторана")
    cuisine_type: str = Field(..., description="Тип кухни")

class Prediction(BaseModel):
    minutes: int


logger.info("Загрузка модели")

MODEL_PATH = os.path.join(os.getcwd(), "model")
MODEL = PredictModel._load_model(MODEL_PATH)

logger.info("Модель успешно зпгружена")

@app.get('/')
def checkStatus():
    return {'status': 'ok'}


@app.post("/prediction", response_model=Prediction)
def makePrediction(features: OrderFeatures):
    try:
        df = pd.DataFrame([features.model_dump()])
        prediction = MODEL.predict(df)
        logger.info("Модель дала прогноз")

    except Exception as e:
        logger.error("Не удалось подключиться")
        raise HTTPException(status_code=500, detail="Не удалось получить прогноз")

    return Prediction(**prediction)


@app.get("/data", response_model=OrderFeatures)
def getTempData():
    df = pd.DataFrame([OrderFeatures.model_dump()])
    return OrderFeatures(**df)





