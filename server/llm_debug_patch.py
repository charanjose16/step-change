# Temporary debug patch to inspect AzureOpenAI.complete method
from llama_index.llms.azure_openai import AzureOpenAI
import inspect

print('AzureOpenAI.complete:', inspect.getsource(AzureOpenAI.complete))
