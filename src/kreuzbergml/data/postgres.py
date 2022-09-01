import csv
import io
from typing import Any, Iterable, List, Optional

import gin
import pandas as pd
import pandas.io.sql
import sqlalchemy


@gin.configurable
class AbstractPostgresDbDAO:
    def __init__(
        self, username: str, password: str, host: str, port: str, database: str
    ):
        self.engine = self.create_engine(username, password, host, port, database)

    @staticmethod
    def create_engine(
        username: str, password: str, host: str, port: str, database: str
    ) -> sqlalchemy.engine.Engine:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        )
        return engine

    def configure_db(self, schema_sql: str) -> None:
        escaped_sql = sqlalchemy.text(schema_sql)
        with self.engine.connect() as connection:
            connection.execute(escaped_sql)

    def export_from_db(self, table_name: str, schema: str) -> pd.DataFrame:
        df = pd.read_sql_table(schema=schema, table_name=table_name, con=self.engine)
        return df

    def import_to_db(
        self,
        source_df: pd.DataFrame,
        target_table_name: str,
        target_schema_name: Optional[str] = None,
        drop_table_if_exists: bool = False,
    ) -> int:
        if drop_table_if_exists:
            method = self.__psql_insert_copy
            if_exists = "replace"
        else:
            method = self.__psql_truncate_insert_copy
            if_exists = "append"
        cnt = source_df.to_sql(
            schema=target_schema_name,
            name=target_table_name,
            con=self.engine,
            method=method,
            if_exists=if_exists,
            index=False,
        )
        if not cnt:
            cnt = source_df.shape[0]
        return cnt

    @staticmethod
    def __psql_truncate_insert_copy(
        table: pd.io.sql.SQLTable,
        conn: sqlalchemy.engine.base.Connection,
        keys: List[str],
        data_iter: Iterable[Any],
    ):
        """
        Based on https://pandas.pydata.org/docs/user_guide/io.html#io-sql-method
        It is a more performant insertion method using PostgreSQL COPY clause.
        """
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = io.StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ", ".join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = "{}.{}".format(table.schema, table.name)
            else:
                table_name = table.name

            cur.execute("TRUNCATE {}".format(table_name))
            sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf)

    @staticmethod
    def __psql_insert_copy(
        table: pd.io.sql.SQLTable,
        conn: sqlalchemy.engine.base.Connection,
        keys: List[str],
        data_iter: Iterable[Any],
    ):
        """
        Copied from https://pandas.pydata.org/docs/user_guide/io.html#io-sql-method
        It is a more performant insertion method using PostgreSQL COPY clause.
        """
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            s_buf = io.StringIO()
            writer = csv.writer(s_buf)
            writer.writerows(data_iter)
            s_buf.seek(0)

            columns = ", ".join('"{}"'.format(k) for k in keys)
            if table.schema:
                table_name = "{}.{}".format(table.schema, table.name)
            else:
                table_name = table.name

            sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
            cur.copy_expert(sql=sql, file=s_buf)
