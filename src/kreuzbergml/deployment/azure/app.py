import logging
from typing import Dict, Optional

import gin
from azureml.core import (
    ComputeTarget,
    Environment,
    Model,
    RunConfiguration,
    Webservice,
    Workspace,
)
from azureml.core.authentication import (
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication,
)
from azureml.core.compute import AmlCompute
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.core import Pipeline, PipelineEndpoint
from azureml.pipeline.steps import PythonScriptStep

logger = logging.getLogger(__name__)


@gin.configurable
class AzureApp:
    instance: Optional[object] = None

    def __init__(
        self,
        # code_path: str,
        tenant_id: str,
        subscription_id: str,
        service_principle_id: Optional[str] = None,
        service_principle_password: Optional[str] = None,
        resource_group: Optional[str] = None,
        workspace_name: Optional[str] = None,
        model_name: Optional[str] = None,
        code_path: Optional[str] = None,
        config_path: Optional[str] = None,
        compute_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        batch_service_name: Optional[str] = None,
        db_host: Optional[str] = None,
        db_port: Optional[str] = None,
        db_user: Optional[str] = None,
        db_password: Optional[str] = None,
        db_name: Optional[str] = None,
        blob_string: Optional[str] = None,
        blob_container: Optional[str] = None,
    ):
        self.__tenant_id = tenant_id
        self.__subscription_id = subscription_id
        self.__service_principle_id = service_principle_id
        self.__service_principle_password = service_principle_password
        self.__resource_group = resource_group
        self.__workspace_name = workspace_name
        self.__model_name = model_name
        self.__code_path = code_path
        self.__config_path = config_path
        self.__compute_name = compute_name
        self.__experiment_name = experiment_name
        self.__batch_service_name = batch_service_name
        self.__db_host = db_host
        self.__db_port = db_port
        self.__db_user = db_user
        self.__db_password = db_password
        self.__db_name = db_name
        self.__blob_string = blob_string
        self.__blob_container = blob_container
        # self.__realtime_service_name = realtime_service_name
        # self.__batch_service_name = batch_service_name

    @staticmethod
    def get_instance() -> "AzureApp":
        if not AzureApp.instance:
            AzureApp.instance = AzureApp()
        return AzureApp.instance

    @property
    def tenant_id(self) -> str:
        return self.__tenant_id

    @property
    def subscription_id(self) -> str:
        return self.__subscription_id

    @property
    def service_principle_id(self) -> str:
        return self.__service_principle_id

    @property
    def service_principle_password(self) -> str:
        return self.__service_principle_password

    @property
    def resource_group(self):
        return self.__resource_group

    @property
    def workspace_name(self) -> str:
        return self.__workspace_name

    @property
    def model_name(self) -> str:
        return self.__model_name

    @property
    def code_path(self) -> str:
        return self.__code_path

    @property
    def config_path(self) -> str:
        return self.__config_path

    @property
    def compute_name(self) -> str:
        return self.__compute_name

    @property
    def db_host(self) -> str:
        return self.__db_host

    @property
    def db_port(self) -> str:
        return self.__db_port

    @property
    def db_user(self) -> str:
        return self.__db_user

    @property
    def db_password(self) -> str:
        return self.__db_password

    @property
    def db_name(self) -> str:
        return self.__db_name

    @property
    def blob_string(self) -> str:
        return self.__blob_string

    @property
    def blob_container(self) -> str:
        return self.__blob_container

    # @property
    # def realtime_service_name(self) -> str:
    #     return self.__realtime_service_name
    #
    # @property
    # def batch_service_name(self) -> str:
    #     return self.__batch_service_name

    def get_workspace(self) -> Workspace:
        if self.service_principle_id and self.service_principle_password:
            auth = self.get_service_principle_authentication()
        else:
            auth = self.get_interactive_authentication()
        ws = Workspace(
            subscription_id=self.subscription_id,
            resource_group=self.resource_group,
            workspace_name=self.workspace_name,
            auth=auth,
        )
        return ws

    def get_interactive_authentication(self) -> InteractiveLoginAuthentication:
        auth = InteractiveLoginAuthentication(tenant_id=self.tenant_id)
        return auth

    def get_service_principle_authentication(self) -> ServicePrincipalAuthentication:
        auth = ServicePrincipalAuthentication(
            tenant_id=self.tenant_id,
            service_principal_id=self.service_principle_id,
            service_principal_password=self.service_principle_password,
        )
        return auth

    def get_or_create_compute_target(
        self,
        compute_name: Optional[str] = None,
        vm_size: str = "STANDARD_DS11_V2",
        vm_priority: str = "lowpriority",
        max_nodes: int = 1,
    ) -> ComputeTarget:
        """
        Creates a compute cluster.
        See: "Azure Machine Learning Studio" -> "Compute" -> "Compute clusters"
        :param vm_size: by default, it's a General-purpose processing unit which costs €0.04/h per node (assuming vm_priority is low, otherwise it costs €0.17/h)
        :param vm_priority: by default, the priority is low, which means the cluster will not execute given jobs immediately. Execution can start, for example, 10 mintes later.
        :param max_nodes: by default, maximum number of nodes is 1. Increasing the number of nodes will increase the costs proportionally.
        :return: a compute target object representing a cluster of one or more computers
        """
        if not compute_name:
            compute_name = self.compute_name
        ws = self.get_workspace()
        try:
            compute_target = ComputeTarget(workspace=ws, name=compute_name)
            logger.info(f"Compute instance '{compute_name}' already exists.")
        except ComputeTargetException:
            logger.info(f"Creating compute instance '{compute_name}'.")
            config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                vm_priority=vm_priority,
                max_nodes=max_nodes,
            )
            compute_target = ComputeTarget.create(
                workspace=ws,
                name=self.compute_name,
                provisioning_configuration=config,
            )
            compute_target.wait_for_completion(show_output=True)
            logger.info(f"Compute instance '{compute_name}' has been created.")
        return compute_target

    def delete_compute_instance(self) -> bool:
        ws = self.get_workspace()
        compute_target = None
        try:
            compute_target = ComputeTarget(workspace=ws, name=self.compute_name)
        except ComputeTargetException:
            logger.warning(f"Compute instance '{self.compute_name}' does not exist.")
        if not compute_target:
            return False
        compute_target.delete()
        logger.info(f"Compute instance '{self.compute_name}' has been deleted.")
        return True

    def deploy_model(self) -> Model:
        ws = self.get_workspace()
        model = Model.register(
            workspace=ws,
            model_name=self.model_name,
            model_path=self.code_path,
        )
        return model

    def get_env(
        self,
        endpoint_azure_name: str,
        conda_file: str,
        principal_access: bool = False,
        db_access: bool = False,
        blob_access: bool = False,
        custom_environment_variables: Optional[Dict[str, str]] = None,
    ) -> Environment:
        env = Environment.from_conda_specification(
            name=f"{endpoint_azure_name}-env", file_path=conda_file
        )
        env.environment_variables = {
            "tenant_id": self.__tenant_id,
            "subscription_id": self.__subscription_id,
            "resource_group": self.__resource_group,
            "workspace_id": self.__workspace_name,
            "model_name": self.__model_name,
            "service_principal_id": self.__service_principle_id,
            "service_principal_password": self.__service_principle_password,
            "experiment_name": self.__experiment_name,
            "batch_service_name": self.__batch_service_name,
        }
        if principal_access:
            env.environment_variables.update(
                {
                    "service_principal_id": self.__service_principal_id,
                    "service_principal_password": self.__service_principal_password,
                }
            )
        if db_access:
            env.environment_variables.update(
                {
                    "db_host": self.__db_host,
                    "db_port": self.__db_port,
                    "db_user": self.__db_user,
                    "db_password": self.__db_password,
                    "db_name": self.__db_name,
                }
            )
        if blob_access:
            env.environment_variables.update(
                {
                    "blob_string": self.__blob_string,
                    "blob_container": self.__blob_container,
                }
            )
        if custom_environment_variables:
            env.environment_variables.update(custom_environment_variables)
        return env

    def create_real_time_endpoint(
        self,
        conda_file: str = "conda.yml",
        entry_script_file: str = "score.py",
        endpoint_azure_name: str = "default-service",
        endpoint_dns_name: Optional[str] = None,
        aci_cpu_cores: int = 1,
        aci_memory_gb: int = 0.5,
        ssl_cert_pem_file: Optional[str] = None,
        ssl_key_pem_file: Optional[str] = None,
        ssl_cname: Optional[str] = None,
        location: str = "northeurope",
        tags: Optional[dict] = None,
        deployment_target_compute_name: Optional[str] = None,
        deployment_target_vm_size: Optional[str] = None,
        principal_access: bool = False,
        db_access: bool = False,
        blob_access: bool = False,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> Webservice:
        ws = self.get_workspace()
        ssl_enabled = (
            True if ssl_cert_pem_file and ssl_key_pem_file and ssl_cname else False
        )
        env = self.get_env(
            endpoint_azure_name,
            conda_file,
            principal_access,
            db_access,
            blob_access,
            environment_variables,
        )
        inference_config = InferenceConfig(
            environment=env,
            entry_script=entry_script_file,
            source_directory=self.code_path,
        )
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=aci_cpu_cores,
            memory_gb=aci_memory_gb,
            dns_name_label=endpoint_dns_name,
            ssl_enabled=ssl_enabled,
            ssl_cert_pem_file=ssl_cert_pem_file,
            ssl_key_pem_file=ssl_key_pem_file,
            ssl_cname=ssl_cname,
            location=location,
            tags=tags,
        )
        deployment_target = None
        if deployment_target_compute_name and deployment_target_vm_size:
            deployment_target = self.get_or_create_compute_target(
                compute_name=deployment_target_compute_name,
                vm_size=deployment_target_vm_size,
                vm_priority="dedicated",
            )
        service = Model.deploy(
            workspace=ws,
            name=endpoint_azure_name,
            models=[],
            inference_config=inference_config,
            deployment_config=deployment_config,
            deployment_target=deployment_target,
            overwrite=True,
            show_output=True,
        )
        service.wait_for_deployment(show_output=True)
        return service

    def create_pipeline_endpoint(
        self,
        entry_script_file: str = "score.py",
        service_name: str = "test-service",
        pipeline_name: str = "batch-pipeline",
        pipeline_endpoint_name: str = "batch-endpoint",
        conda_file: str = "./conda.yml",
        vm_size: str = "STANDARD_DS11_V2",
        vm_priority: str = "lowpriority",
        max_nodes: int = 1,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> PipelineEndpoint:
        ws = self.get_workspace()
        env = self.get_env(
            service_name,
            conda_file,
            db_access=True,
            blob_access=True,
            custom_environment_variables=environment_variables,
        )
        compute_target = self.get_or_create_compute_target(
            vm_size=vm_size,
            vm_priority=vm_priority,
            max_nodes=max_nodes,
        )
        runconfig = RunConfiguration()
        runconfig.environment = env

        batch_execution_step = PythonScriptStep(
            source_directory=self.code_path,
            script_name=entry_script_file,
            arguments=["--model_name", self.model_name],
            runconfig=runconfig,
            compute_target=compute_target,
            allow_reuse=False,
        )
        published_pipeline = Pipeline(
            workspace=ws,
            steps=[batch_execution_step],
            default_source_directory=self.code_path,
            description="Batch pipeline",
        ).publish(
            name=pipeline_name,
            description="Published batch pipeline.",
        )
        existing_pipelines_list = PipelineEndpoint.list(ws, active_only=False)
        existing_pipelines_names_set = {p.name for p in existing_pipelines_list}
        if pipeline_endpoint_name in existing_pipelines_names_set:
            pipeline_endpoint = PipelineEndpoint.get(
                workspace=ws, name=pipeline_endpoint_name
            )
            pipeline_endpoint.add_default(published_pipeline)
            if pipeline_endpoint.status == "Disabled":
                pipeline_endpoint.enable()
            logging.info(
                f"Existing pipeline has been updated. URL={pipeline_endpoint.endpoint}"
            )
        else:
            pipeline_endpoint = PipelineEndpoint.publish(
                workspace=ws,
                name=pipeline_endpoint_name,
                pipeline=published_pipeline,
                description="Batch Pipeline Endpoint",
            )
            logger.info(
                f"New pipeline has been published. URL={pipeline_endpoint.endpoint}"
            )
        return pipeline_endpoint
