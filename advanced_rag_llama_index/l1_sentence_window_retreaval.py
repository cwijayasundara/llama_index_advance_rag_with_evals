from dotenv import load_dotenv
from llama_index.llms import OpenAI
from llama_index import Document
from llama_index import SimpleDirectoryReader, OpenAIEmbedding
from utils_temp import get_sentence_window_query_engine, get_prebuilt_trulens_recorder, build_sentence_window_index

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-1106",
             temperature=0.1)

embed_model = OpenAIEmbedding(embed_batch_size=10)

documents = SimpleDirectoryReader(
    input_files=["docs/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

document = Document(text="\n\n".join([doc.text for doc in documents]))

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model=embed_model,
    save_dir="sentence_index"
)

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


sentence_window_engine = get_sentence_window_query_engine(sentence_index)

window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))

tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id="Sentence Window Query Engine"
)

for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))

tru.get_leaderboard(app_ids=[])

# launches on http://localhost:8501/
tru.run_dashboard()
