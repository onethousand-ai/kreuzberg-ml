import logging

import gin
from azureml.core import ComputeTarget, Workspace, Model, Environment, Webservice
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import AmlCompute
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.exceptions import ComputeTargetException

logger = logging.getLogger(__name__)


@gin.configurable
class AzureApp:
    def __init__(
            self,
            code_path: str,
            tenant_id: str,
            subscription_id: str,
            resource_group: str,
            workspace_name: str,
            compute_name: str,
            realtime_service_name: str,
            batch_service_name: str,
            app_name: str = "app",
    ):
        # self.__code_path = code_path
        self.__tenant_id = tenant_id
        self.__subscription_id = subscription_id
        self.__resource_group = resource_group
        self.__workspace_name = workspace_name
        self.__compute_name = compute_name
        self.__realtime_service_name = realtime_service_name
        self.__batch_service_name = batch_service_name

    @property
    def tenant_id(self) -> str:
        return self.__tenant_id

    @property
    def subscription_id(self) -> str:
        return self.__tenant_id

    @property
    def resource_group(self):
        return self.__tenant_id

    @property
    def workspace_name(self) -> str:
        return self.__tenant_id

    @property
    def compute_name(self) -> str:
        return self.__compute_name

    @property
    def realtime_service_name(self) -> str:
        return self.__realtime_service_name

    @property
    def batch_service_name(self) -> str:
        return self.__batch_service_name

    def get_workspace(self) -> Workspace:
        auth = self.get_interactive_authentication()
        ws = Workspace(
            subscription_id=self.subscription_id,
            resource_group=self.resource_group,
            workspace_name=self.workspace_name,
            auth=auth
        )
        return ws

    @staticmethod
    def get_conda_env(
            name: str = "conda-env",
            file_path: str = "conda.yml",
    ) -> Environment:
        return Environment.from_conda_specification(name=name, file_path=file_path)

    def get_interactive_authentication(self) -> InteractiveLoginAuthentication:
        return InteractiveLoginAuthentication(tenant_id=self.tenant_id)

    def get_or_create_compute_target(
            self,
            vm_size: str = 'STANDARD_DS11_V2',
            vm_priority: str = 'lowpriority',
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
        ws = self.get_workspace()
        try:
            compute_target = ComputeTarget(workspace=ws, name=self.compute_name)
            logger.info(f"Compute instance '{self.compute_name}' already exists.")
        except ComputeTargetException:
            logger.info(f"Creating compute instance '{self.compute_name}'.")
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
            logger.info(f"Compute instance '{self.compute_name}' has been created.")
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

    def register_model(
            self,
            model_name: str = "default_model",
            model_path: str = "project_dir",
    ) -> Model:
        ws = self.get_workspace()
        model = Model.register(
            workspace=ws,
            model_name=model_name,
            model_path=model_path,
        )
        return model

    def create_real_time_endpoint(
            self,
            source_dir: str,
            entry_script_file: str = "score.py",
            conda_file: str = "./conda.yml",
            aci_cpu_cores: int = 1,
            aci_memory_gb: int = 4,
    ) -> Webservice:
        ws = self.get_workspace()
        env = Environment.from_conda_specification(
            name=f"{self.realtime_service_name}-env",
            file_path=conda_file
        )
        inference_config = InferenceConfig(
            environment=env,
            entry_script=entry_script_file,
            source_directory=source_dir,
        )
        deployment_config = AciWebservice.deploy_configuration(
            cpu_cores=aci_cpu_cores,
            memory_gb=aci_memory_gb,
        )
        service = Model.deploy(
            workspace=ws,
            name=self.realtime_service_name,
            models=[],
            inference_config=inference_config,
            deployment_config=deployment_config,
            overwrite=True,
            show_output=True,
        )
        service.wait_for_deployment(show_output=True)
        return service
