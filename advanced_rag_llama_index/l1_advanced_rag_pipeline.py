from dotenv import load_dotenv
from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader, OpenAIEmbedding

load_dotenv()

documents = SimpleDirectoryReader(
    input_files=["docs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

print("type of the document : ", type(documents), "\n")
print("length of the document :", len(documents), "\n")
print("type of the first element : ", type(documents[0]))
print("first element : ", documents[0])

document = Document(text="\n\n".join([doc.text for doc in documents]))

llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)

embed_model = OpenAIEmbedding(embed_batch_size=10)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

index = VectorStoreIndex.from_documents([document], service_context=service_context)

query_engine = index.as_query_engine()

response = query_engine.query(
    "What are steps to take when finding projects to build your experience?"
)
print(str(response))

# evals
eval_questions = []
with open('eval_questions.txt', 'r') as file:
    for line in file:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)

new_question = "What is the right AI job for me?"
eval_questions.append(new_question)
print(eval_questions)

from trulens_eval import Tru

tru = Tru()

tru.reset_database()

# from utils import get_prebuilt_trulens_recorder
from utils_temp import get_prebuilt_trulens_recorder

tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                             app_id="Direct Query Engine")

with tru_recorder as recording:
    for question in eval_questions:
        response = query_engine.query(question)

records, feedback = tru.get_records_and_feedback(app_ids=[])

"""loop through the records and print the content of each record"""
for record in records:
    print(record)

"""print the feedback"""
print(feedback)

# launches on http://localhost:8501/
tru.run_dashboard()
