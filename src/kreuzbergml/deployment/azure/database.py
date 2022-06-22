import logging

import gin
from azureml.core import Datastore
from azureml.data.azure_postgre_sql_datastore import AzurePostgreSqlDatastore
from azureml.exceptions import UserErrorException

from .app import AzureApp

logger = logging.getLogger(__name__)


@gin.configurable
class AzureDatabase:
    def __init__(
        self,
        azure_config: AzureApp,
        datastore_name: str,
        server_name: str,
        database_name: str,
        user_id: str,
        user_password: str,
    ):
        self.__azure_config = azure_config
        self.__datastore_name = datastore_name
        self.__server_name = server_name
        self.__database_name = database_name
        self.__user_id = user_id
        self.__user_password = user_password

    @property
    def azure_config(self) -> AzureApp:
        return self.__azure_config

    @property
    def datastore_name(self) -> str:
        return self.__datastore_name

    @property
    def server_name(self) -> str:
        return self.__server_name

    @property
    def database_name(self) -> str:
        return self.__database_name

    @property
    def user_id(self) -> str:
        return self.__user_id

    @property
    def user_password(self) -> str:
        return self.__user_password

    def get_or_register_postgres_db(self) -> AzurePostgreSqlDatastore:
        ws = self.azure_config.get_workspace()
        try:
            psql_datastore = Datastore.get(ws, self.datastore_name)
            logger.info(f"PostgreSQL database '{self.datastore_name}' already exists.")
        except UserErrorException:
            psql_datastore = Datastore.register_azure_postgre_sql(
                workspace=ws,
                datastore_name=self.datastore_name,
                server_name=self.server_name,
                database_name=self.database_name,
                user_id=self.user_id,
                user_password=self.user_password,
            )
            logger.info(
                f"PostgreSQL database '{self.datastore_name}' has been registered."
            )
        return psql_datastore
