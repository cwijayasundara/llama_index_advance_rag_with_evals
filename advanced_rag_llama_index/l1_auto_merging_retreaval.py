from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, OpenAIEmbedding
from llama_index import Document
from llama_index.llms import OpenAI

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

from utils_temp import build_automerging_index

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)

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

from utils_temp import get_automerging_query_engine, get_prebuilt_trulens_recorder

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)

auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))

tru.reset_database()

tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")

for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

tru.get_leaderboard(app_ids=[])

# launches on http://localhost:8501/
tru.run_dashboard()
