import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
import requests
from typing import Optional
import pandas as pd


class ConnectionBD:

    def __init__(self, local_path: Optional[str], url: Optional[str]) -> None:
        self.local_path = local_path
        self.url = url
        self._engine : Optional[Engine] = None


    @property
    def engine(self) -> Engine:

        if self._engine is not None:
            return self._engine

        if not self.local_path:
            raise ValueError("Нет local_path")
        if not self.url:
            raise ValueError("Нет URL")


        if not os.path.exists(self.local_path):
            try:
                r = requests.get(self.url)
                r.raise_for_status()

                with open(self.local_path, 'wb') as f:
                    f.write(r.content)
                print("БД успешно загружен")

            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Не удалось подключиться: {e}")


        self._engine = create_engine(f"sqlite:///{self.local_path}")
        return self._engine




if __name__ == "__main__":
    # Загружаем .env
    load_dotenv(dotenv_path='../../.env')

    # Получаем переменные с проверкой
    LOCAL_PATH = os.getenv("LOCAL_PATH")
    URL = os.getenv("URL")

    if not LOCAL_PATH or not URL:
        raise ValueError("LOCAL_PATH или URL не найдены в .env")

    print(f"LOCAL_PATH: {LOCAL_PATH}")
    print(f"URL: {URL}")

    test = ConnectionBD(LOCAL_PATH, URL)
    engine = test.engine

    df = pd.read_sql("select * from sqlite_master where type = 'table'", engine)
    print(df)






