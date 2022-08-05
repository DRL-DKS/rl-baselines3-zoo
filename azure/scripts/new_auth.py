import os
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient

tenant_id = os.environ["AZURE_TENANT_ID"]
client_id = os.environ["AZURE_CLIENT_ID"]
client_secret = os.environ["AZURE_CLIENT_SECRET"]

credential = ClientSecretCredential(
    tenant_id=tenant_id,
    client_id=client_id,
    client_secret=client_secret,
)

client = ResourceManagementClient(credential, os.environ["AZURE_SUBSCRIPTION_ID"])

print(client.resource_groups.list())
for item in client.resource_groups.list():
    print(item)
