from typing_extensions import override
from openai import AzureOpenAI, AssistantEventHandler
import os
import logging
import glob
from dotenv import load_dotenv

class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)

    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    @override
    def on_message_done(self, message) -> None:
        # print a citation to the file searched
        message_content = message.content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f"[{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        print(message_content.value)
        print("\n".join(citations))


load_dotenv()

# setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

azure_openai_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
client = AzureOpenAI(
    api_key=azure_openai_key,  
    api_version="2024-05-01-preview",
    azure_endpoint = azure_openai_endpoint
    )


# Check if assistant_id.txt exists
if os.path.exists("assistant_id.txt"):
    # Read the assistant id from the file
    with open("assistant_id.txt", "r") as f:
        assistant_id = f.read().strip()
    logger.info(f"Using existing assistant with  ID: {assistant_id}")
else:
    # Create a new assistant
    logger.info("Creating a new assistant")
    assistant = client.beta.assistants.create(
        name="Microsoft Build Session Assistant",
        instructions="You know all about Microsoft Build 2024. You always use the file_search tool to answer questions.",
        model="gpt-4-turbo",
        tools=[{"type": "file_search"}],
    )
    # Save the assistant id in a text file
    with open("assistant_id.txt", "w") as f:
        f.write(assistant.id)
        assistant_id = assistant.id
        
logger.info(f"Assistant ID: {assistant_id}")

# try to get existing vector store
vector_store_id = None
if os.path.exists("vector_store_id.txt"):
    # Read the vector store id from the file
    logger.info("Using existing vector store")
    with open("vector_store_id.txt", "r") as f:
        vector_store_id = f.read().strip()
    logger.info(f"Using existing vector store with ID: {vector_store_id}")

vector_store = None
if vector_store_id:
    # Retrieve the vector store based on the id
    vector_store = client.beta.vector_stores.retrieve(vector_store_id=vector_store_id)
    logger.info(f"Retrieved vector store: {vector_store}")

if not vector_store:
    # Create a vector store called "MSBuild"
    logger.info("Creating a new vector store")
    vector_store = client.beta.vector_stores.create(name="MSBuild")
    
    # Store vector store id in a text file
    with open("vector_store_id.txt", "w") as f:
        logger.info(f"Saving vector store ID: {vector_store.id}")
        f.write(vector_store.id)
        
    vector_store_id = vector_store.id
    logger.info(f"Vector Store ID: {vector_store_id}")
    
    # Ready the files for upload to OpenAI
    file_paths = glob.glob("videos/*.md")
    file_streams = [open(path, "rb") for path in file_paths]
    
    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    logger.info("Uploading files to vector store")
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store_id, files=file_streams
    )
    
    # You can print the status and the file counts of the batch to see the result of this operation.
    logger.info(file_batch.status)
    logger.info(file_batch.file_counts)

logger.info(f"Updating assistant with id {assistant_id} to use vector store {vector_store_id}")
assistant = client.beta.assistants.update(
  assistant_id=assistant_id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
)

logger.info("Assistant updated successfully. Creating a new thread.")
thread = client.beta.threads.create()
logger.info(f"Thread created with ID: {thread.id}. Running the thread.")

while True:
    user_input = input("You > ")
    if user_input == "exit":
        break
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input
    )

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant_id,
        event_handler=EventHandler()
    ) as stream:
        stream.until_done()
