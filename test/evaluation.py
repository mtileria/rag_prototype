import os
import json

from dotenv import load_dotenv

# Load environment variables from .env file 
load_dotenv()

# --- Core LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Evaluation Imports ---
from langchain.evaluation.qa import QAEvalChain
from langchain_openai import OpenAIEmbeddings
from langchain.evaluation.criteria import CriteriaEvalChain # Added for placeholder


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def evaluate_with_custom_criteria(query, prediction, reference, context, criteria_llm):
    """
    This function evaluates the prediction against custom criteria.
    """
    print(f"\n--- Running Criteria Evaluation Placeholder for Query: '{query}' ---")

    # Define custom criteria - these instruct the LLM on what to evaluate
    criteria = {
        "faithfulness": "Does the prediction accurately reflect ONLY the information provided in the context? It should not add outside information or contradict the context.",
        "conciseness": "Is the prediction concise and to the point, avoiding unnecessary verbose language?",
        "relevance_to_query": "Is the prediction directly relevant to the input query?",
        "matches_reference": "Does the prediction convey the same essential information as the reference answer, even if worded differently?"
    }

    # Instantiate the evaluation chain for the defined criteria
    eval_chain = CriteriaEvalChain.from_llm(
        llm=criteria_llm,
        criteria=criteria,
        input_key="input",
        prediction_key="prediction")
        # Example: If criteria prompts need context/reference explicitly:
        # requires_context=True, 
        # requires_reference=True,
    

    try:
        # Let's try evaluate on a list containing the single item
        result = eval_chain.evaluate_strings(input = query,
            prediction = prediction,      
            reference = reference,
            context = context) # if required 
        
        # The result is usually a list of dictionaries, one per input item
        print(f"  Criteria Evaluation Result:")

        if isinstance(result, str): # Handle simple string output
            print(f"Raw Output: {result.strip()}")
        elif isinstance(result, dict): # Handle structured dict output
            for key in result.keys():
                print(f"    - {key}: {result[key]}")
        else:
            print(f"Unexpected result format: {result}")

    except Exception as e:
        print(f"  Criteria Evaluation failed: {e}")
        print("  Note: Check input keys ('input', 'prediction', 'context', 'reference') required by the CriteriaEvalChain's prompt and criteria definitions.")
    print("-" * 30)
    return True

def generate_testing_dataset():
    # --- 1. Define Knowledge Base ---
    knowledge_base_text = """
    Brighton is a vibrant seaside resort located on the south coast of England, in the county of East Sussex.
    It forms part of the Brighton and Hove unitary authority. Historically part of Sussex, Brighton emerged as a
    health resort featuring sea bathing during the 18th century. It became a popular destination for Londoners,
    especially after the railway arrived in 1841.

    One of Brighton's most famous landmarks is the Royal Pavilion. Built in three stages beginning in 1787,
    it served as a seaside retreat for George, Prince of Wales, who became the Prince Regent in 1811, and King George IV in 1820.
    The Pavilion is notable for its unique Indo-Saracenic architecture and Oriental interior design.
    Other attractions include the Brighton Pier, the Lanes shopping area, and the i360 observation tower.
    The city is also known for its large LGBT population and hosts a major Pride festival annually.
    Brighton beach is primarily a pebble beach.
    """

    # --- 2. Define Evaluation Data ---
    # List of dictionaries, each with the input query and a reference ground truth answer.
    eval_dataset = [
        {
            "input": "Where is Brighton?",
            "reference": "Brighton is a seaside resort on the south coast of England in East Sussex."
        },
        {
            "input": "Describe the Royal Pavilion's architecture.",
            "reference": "The Royal Pavilion has Indo-Saracenic architecture and an Oriental interior."
        },
        {
            "input": "What material is Brighton beach made of?",
            "reference": "Brighton beach is mainly made of pebbles."
        },
        {
            "input": "What is the population of Brighton?", # Answer not in context
            "reference": "The provided context does not state the population of Brighton."
        }
    ]

    return knowledge_base_text, eval_dataset

if __name__ == "__main__":
    # --- 0. Install Required Packages ---
    # pip install langchain langchain-openai langchain-chroma langchain-core python-dotenv

    # --- Load Environment Variables ---
    load_dotenv()  
        
    # --- Configuration ---
    # read OPENAI_API_KEY from .env file 
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")


    FULL_WORKFLOW = False  # Set to True to run all steps, including dataset generation
    SAVE_DATASET = False  # set to True to save predictions and contexts

    # LLM for the RAG chain and Evaluation (Using OpenAI)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    # Can use the same or different model for eval. ideally use the best model available
    eval_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key) 

    # Embedding model to use for vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    
    if FULL_WORKFLOW:

        # Simple Knowledge Base (Context relevant to Brighton, UK for this example)
        # These are simple string and list, but in practice, you will load this from a file or database.
        knowledge_base_text, eval_dataset = generate_testing_dataset()

        # --- Setup Basic RAG Chain ---

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([knowledge_base_text])

        # Create a vector store from the documents 
        # Chroma runs in-memory by default here. For persistence, specify a directory in from_documents().
        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2}) # top 2 context chunks

        # Define RAG Prompt
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}

        Answer: """
        prompt = ChatPromptTemplate.from_template(template)

        # Define the RAG pipeline
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm 
            | StrOutputParser()
        )

        # --- Run QA Evaluation ---

        print("Running RAG Chain and QA Evaluation (using OpenAI)...")

        predictions_for_qa = []
        contexts_for_criteria = {} # Store context for criteria evaluation later

        for example in eval_dataset:
            question = example["input"]
            reference = example["reference"]

            # Get prediction from RAG chain
            prediction = rag_chain.invoke(question)
            predictions_for_qa.append(
                {"query": question, "answer": reference, "result": prediction}
            )

            # Also retrieve context separately if needed for criteria eval (optional)
            retrieved_docs = retriever.invoke(question)
            formatted_context = format_docs(retrieved_docs)
            contexts_for_criteria[question] = formatted_context # Store context by question

        if SAVE_DATASET:
            with open("predictions_for_qa.json", "w") as f:
                json.dump(predictions_for_qa, f, indent=4)
            with open("contexts_for_criteria.json", "w") as f:
                json.dump(contexts_for_criteria, f, indent=4)
                    
        # Instantiate the QA evaluation chain using the OpenAI LLM
        qa_eval_chain = QAEvalChain.from_llm(llm=eval_llm)

        # Run the evaluation. Compares 'result' (prediction) against 'answer' (reference) for the given 'query'.
        qa_eval_results = qa_eval_chain.evaluate(
            eval_dataset,
            predictions_for_qa,
            question_key="input",
            answer_key="reference",
            prediction_key="result"
        )

        # --- Display QA Results and Run Criteria Placeholder ---
        print("\n" + "=" * 50)
        print("QA Evaluation Results (using OpenAI):")
        print("=" * 50)

        for i, example in enumerate(eval_dataset):
            print(f"Example {i+1}:")
            print(f"  Query: {example['input']}")
            print(f"  Reference: {example['reference']}")
            print(f"  Prediction: {predictions_for_qa[i]['result']}")
            # QAEvalChain adds its result under the 'results' key
            print(f"  QA Evaluation (Correctness): {qa_eval_results[i]['results'].strip()}")
            print("-" * 20)
    
        print("\nEvaluation Complete.")

    else:
        # --- 2. Run Custom Criteria Evaluation ---
        print("\nRunning Custom Criteria Evaluation with saved dataset ...")

        try:
            with open("predictions_for_qa.json", "r") as f:
                predictions_for_qa = json.load(f)
            with open("contexts_for_criteria.json", "r") as f:
                contexts_for_criteria = json.load(f)
            print("Loaded predictions and contexts from files.")
        except FileNotFoundError:
            print("Prediction files not found. Proceeding with new evaluation.")
            raise Exception("Prediction files not found.")

        for i, pred_data in enumerate(predictions_for_qa):
            query = pred_data['query']
            prediction = pred_data['result']
            reference = pred_data['answer']
            context = contexts_for_criteria.get(query, "")
            evaluate_with_custom_criteria(query, prediction, reference, context, eval_llm)
    
    # Cleanup in-memory Chroma collection 
    try:
        vectorstore.delete_collection()
        print("\nChroma collection deleted.")
    except Exception as e:
        print(f"\nCould not delete Chroma collection: {e}")
    
