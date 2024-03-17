#Note: The openai-python library support for Azure OpenAI is in preview.
#Note: This code sample requires OpenAI Python library version 1.0.0 or higher.
import os
from openai import AzureOpenAI


client = AzureOpenAI(
  azure_endpoint = "https://openai-ecolution.openai.azure.com/openai/deployments/sid_test1/chat/completions?api-version=2024-02-15-preview",
  api_key="681bfdd0e876498c80c28e8d7b31871c",
  api_version="2024-02-15-preview"
)


message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."}]

# completion = client.chat.completions.create(
#   model="sid_test1", # model = "deployment_name"
#   messages = message_text,
#   temperature=0.7,
#   max_tokens=800,
#   top_p=0.95,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stop=None
# )

while True:
  message = input("User : ")
  if message:
    message_text.append({"role":"user","content":message})
    reply = client.chat.completions.create(model="sid_test1",messages=message_text).choices[0].message.content
    print(reply)
    message_text.append({"role":"assistant","content":reply})