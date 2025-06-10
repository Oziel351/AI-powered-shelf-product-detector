""" # import the inference-sdk
from inference_sdk import InferenceHTTPClient

image_path="productos-estante2.jpeg"
# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="3jQAu52F9uiAV5Wsw0GJ"
)

# infer on a local image
result = CLIENT.infer(image_path, model_id="dataset-counter-products/3") """

import inference
dir(inference)
