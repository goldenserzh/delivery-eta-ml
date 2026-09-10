import pandas as pd
from src.connection.connection_db import ConnectionBD
from dotenv import load_dotenv, find_dotenv
import os
from typing import Optional


class PreprossedData:

    def __init__(self):
        self.df : Optional[pd.DataFrame]= None
        self.start_df: Optional[pd.DataFrame] = None


    @staticmethod
    def checkPartOfday(time):
        if time < 12:
            return 'Morning'
        elif 12 <= time < 18:
            return 'Day'
        else:
            return 'Evening'


    def makePreprocessingData(self):
        load_dotenv(find_dotenv(usecwd=True))

        LOCAL_PATH = os.getenv('LOCAL_PATH')
        URL = os.getenv('URL')

        con = ConnectionBD(LOCAL_PATH, URL)
        engine = con.engine

        query = query = """
with full_data as (select *
from orders as o
inner join customers c on c.customer_id = o.customer_id
inner join couriers co on co.courier_id = o.courier_id
inner join restaurants r on r.restaurant_id = o.restaurant_id
left join weather w on w.city = c.customer_city AND w.date = date(o.order_datetime)
left join traffic t on t.date = date(o.order_datetime)),

without_id_data as (select order_id, order_datetime, order_amount,items_count,
 distance_km,
 is_express_delivery,
 delivery_time_minutes,
 payment_method,
 customer_id,
 customer_city,
 account_age_days,
 preferred_payment_method,
 mobile_os,
 registration_date,
 customer_rating,
 courier_id,
 vehicle_type,
 experience_days,
 base_skill,
 start_date,
 base_speed_kmh,
 courier_rating,
 on_time_ratio,
 restaurant_id,
 restaurant_city,
 cuisine_type,
 is_fast_food,
 open_date,
 restaurant_rating,
 prep_time_avg,
 temperature,
 precip_mm,
 wind_speed,
 traffic_level,
 is_rush_hour
from full_data)

select * from without_id_data

"""
        self.start_df = pd.read_sql(query, engine)
        self.df = pd.read_sql(query, engine)
        print("Данные загружены !")
        print(self.df.shape)
        self.df['order_datetime'] = pd.to_datetime(self.df['order_datetime'])
        self.df['part_of_day'] = self.df['order_datetime'].dt.hour.apply(lambda x: self.checkPartOfday(x))
        self.df['hour'] = self.df['order_datetime'].dt.hour
        self.df['day'] = self.df['order_datetime'].dt.dayofweek
        self.df['month'] = self.df['order_datetime'].dt.month

        self.df = self.df.dropna()

        return self.df, self.start_df


if __name__ == "__main__":
    _, df = PreprossedData().makePreprocessingData()

    print(df.info())