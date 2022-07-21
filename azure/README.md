<!-- https://docs.microsoft.com/en-us/python/api/overview/azure/container-instance?view=azure-python#create-task-based-container-group

https://portal.azure.com/#home -->
# Installation
```
pip install -r requirements.txt
```

# Setup
0) Login on Azure CLI and also check your ID
```
az login
```
1) Set your subscription from Azure CLI after doing az login (check your ID)
```
az account set --subscription <name or id>
```
2) Create your own auth file to automate python SDK login through scripts
``` 
az ad sp create-for-rbac --sdk-auth > my.azureauth
```
3) Save the resulting file somewhere and set it as an OS environment variable on `~\.bashrc` to be read by python scripts
4) 
```
export AZURE_AUTH_LOCATION=/home/yourusername/my.azureauth

```
5) To create resource groups automatically under any azure subscription, you need to start by [elevating your acess](https://docs.microsoft.com/en-us/azure/role-based-access-control/elevate-access-global-admin)

