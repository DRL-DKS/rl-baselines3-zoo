from os import getenv
from azure.common.client_factory import get_client_from_auth_file
from azure.mgmt.containerinstance import ContainerInstanceManagementClient
from azure.mgmt.resource import ResourceManagementClient


# Authenticate the management clients with Azure.
# Set the AZURE_AUTH_LOCATION environment variable to the full path to an
# auth file. Generate an auth file with the Azure CLI or Cloud Shell:
# az ad sp create-for-rbac --sdk-auth > my.azureauth
auth_file_path = getenv("AZURE_AUTH_LOCATION", None)
if auth_file_path is not None:
    print(
        "Authenticating with Azure using credentials in file at {0}".format(
            auth_file_path
        )
    )

    aciclient = get_client_from_auth_file(ContainerInstanceManagementClient)
    resclient = get_client_from_auth_file(ResourceManagementClient)
else:
    print("\nFailed to authenticate to Azure. Have you set the")
