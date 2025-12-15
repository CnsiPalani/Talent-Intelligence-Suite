# --- HR Policy Retrieval ---
import pandas as pd
import re

def retrieve_policy_text(query: str, csv_path: str = "data/Job_description_data.csv") -> str:
	"""
	Retrieve relevant HR policy or benefits text from job descriptions.
	"""
	try:
		df = pd.read_csv(csv_path, dtype=str, encoding_errors="ignore")
	except Exception:
		return ""
	# Combine all text columns
	text_data = " ".join([str(x) for x in df.values.flatten() if pd.notnull(x)])
	# Find sentences with keywords
	keywords = ["leave", "policy", "vacation", "holiday", "pto", "time off", "benefit"]
	pattern = re.compile(r"([^.]*?(?:" + "|".join(keywords) + ")[^.]*\.)", re.IGNORECASE)
	matches = pattern.findall(text_data)
	# Further filter by query words
	query_words = query.lower().split()
	relevant = [m for m in matches if any(qw in m.lower() for qw in query_words)]
	# Return up to 3 most relevant sentences
	return " ".join(relevant[:3]) if relevant else " ".join(matches[:3])
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


_tokenizer = None
_model = None

def _get_gpt2():
	global _tokenizer, _model
	if _tokenizer is None or _model is None:
		_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		_model = GPT2LMHeadModel.from_pretrained('gpt2')
	return _tokenizer, _model


def gpt_summarize(text: str, max_tokens: int = 150) -> str:
	"""
	Summarize input text using GPT2 (simulated, as GPT2 is not trained for summarization).
	"""
	tokenizer, model = _get_gpt2()
	prompt = f"Summarize the following text:\n{text}\nSummary:"
	inputs = tokenizer.encode(prompt, return_tensors="pt")
	outputs = model.generate(inputs, max_length=inputs.shape[1] + max_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
	result = tokenizer.decode(outputs[0], skip_special_tokens=True)
	# Extract summary after 'Summary:'
	if 'Summary:' in result:
		return result.split('Summary:')[-1].strip()
	return result.strip()


def gpt_chatbot(user_prompt: str, max_tokens: int = 150) -> str:
    """
    Chatbot function using GPT2 (simulated, as GPT2 is not trained for chat). Logs messages to DB.
    """
    raise Exception("session_id required for DB-integrated chatbot. Use gpt_chatbot_db(session_id, user_prompt, max_tokens)")

# --- DB chat history integration ---
from .db_utils import read_sql_df

def load_chat_history(session_id: int):
	sql = '''
	SELECT role, message_text, created_at
	FROM hrtech.llm_chat_message
	WHERE session_id = :session_id
	ORDER BY created_at ASC, message_id ASC
	'''
	return read_sql_df(sql, params={"session_id": session_id})

def load_session_info(session_id: int):
	sql = '''
	SELECT * FROM hrtech.llm_chat_session WHERE session_id = :session_id
	'''
	return read_sql_df(sql, params={"session_id": session_id})

def gpt_chatbot_with_history(session_id: int, user_prompt: str, max_tokens: int = 150) -> str:
	"""
	Chatbot with context from previous messages in the session.
	"""
	history_df = load_chat_history(session_id)
	context = ""
	if not history_df.empty:
		for _, row in history_df.tail(5).iterrows():
			context += f"{row['role'].capitalize()}: {row['message_text']}\n"
	context += f"User: {user_prompt}\nAI:"
	tokenizer, model = _get_gpt2()
	inputs = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=1024)
	outputs = model.generate(inputs, max_length=inputs.shape[1] + max_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
	result = tokenizer.decode(outputs[0], skip_special_tokens=True)
	if 'AI:' in result:
		return result.split('AI:')[-1].strip()
	return result.strip()

# --- DB logging for chatbot ---
import datetime
from .db_utils import execute

def log_message_to_db(session_id: int, role: str, message_text: str, tokens_in: int = 0, tokens_out: int = 0, helpfulness_score: int = None, hallucination_flag: int = None):
    sql = '''
    INSERT INTO hrtech.llm_chat_message (session_id, created_at, role, tokens_in, tokens_out, message_text, helpfulness_score, hallucination_flag)
    VALUES (:session_id, :created_at, :role, :tokens_in, :tokens_out, :message_text, :helpfulness_score, :hallucination_flag)
    '''
    params = {
        "session_id": session_id,
        "created_at": datetime.datetime.now(),
        "role": role,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "message_text": message_text,
        "helpfulness_score": helpfulness_score,
        "hallucination_flag": hallucination_flag
    }
    execute(sql, params)

def gpt_chatbot_db(session_id: int, user_prompt: str, max_tokens: int = 150) -> str:
	"""
	Chatbot function using GPT2, logs user and assistant messages to DB. Uses retrieval-augmented prompt.
	"""
	# Retrieve relevant policy text
	policy_context = retrieve_policy_text(user_prompt)
	tokenizer, model = _get_gpt2()
	if policy_context:
		prompt = f"Relevant HR Policy Info: {policy_context}\n\nUser: {user_prompt}\nAI:"
	else:
		prompt = f"User: {user_prompt}\nAI:"
	inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
	outputs = model.generate(inputs, max_length=inputs.shape[1] + max_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
	result = tokenizer.decode(outputs[0], skip_special_tokens=True)
	if 'AI:' in result:
		assistant_reply = result.split('AI:')[-1].strip()
	else:
		assistant_reply = result.strip()
	# Log user and assistant messages
	log_message_to_db(session_id, 'user', user_prompt, tokens_in=len(inputs[0]), tokens_out=0)
	log_message_to_db(session_id, 'assistant', assistant_reply, tokens_in=0, tokens_out=len(tokenizer.encode(assistant_reply)))
	return assistant_reply
