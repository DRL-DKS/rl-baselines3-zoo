# Ray Azure SB3 extension
This branch implements logic to run multiple azure instances for different seeds for statistical significance.
Features:
1) Run and control multiple instances from a single head-node.
2) Upload all folder experiments to a custom url.
3) Monotoring is done via an azure vscode extension, and the weights and biases app.

# Requirements
1) Install [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/) and create credentials, with the aim of accessing [python's azure SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/container-instance?view=azure-python#create-task-based-container-group)

# How to use
Change credentials on keys/credentials.yml


