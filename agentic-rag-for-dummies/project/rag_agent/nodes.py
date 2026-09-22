from typing import Literal, Set
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.types import Command
from .graph_state import State, AgentState
from .schemas import QueryAnalysis
from .prompts import *
from utils import estimate_context_tokens
from config import BASE_TOKEN_THRESHOLD, CHILD_CHUNK_SEPARATOR, MAIN_HISTORY_MESSAGES_TO_KEEP, TOKEN_GROWTH_FACTOR

if MAIN_HISTORY_MESSAGES_TO_KEEP < 2:
    raise ValueError("MAIN_HISTORY_MESSAGES_TO_KEEP must be at least 2.")

PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP = max(MAIN_HISTORY_MESSAGES_TO_KEEP - 1, 0)

def _is_plain_conversation_message(msg) -> bool:
    return (
        isinstance(msg, (HumanMessage, AIMessage))
        and not getattr(msg, "tool_calls", None)
        and not getattr(msg, "name", None)
    )

def _name_internal_message(message, name):
    """Tag a subgraph-only message so it is not treated as chat history."""
    return message.model_copy(update={"name": name})

def _retrieval_contexts(messages) -> list[str]:
    contexts = []
    ignored_prefixes = (
        "NO_RELEVANT_CHUNKS",
        "NO_PARENT_DOCUMENT",
        "RETRIEVAL_ERROR:",
        "PARENT_RETRIEVAL_ERROR:",
    )
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content).strip()
        if content and not content.startswith(ignored_prefixes):
            parts = content.split(CHILD_CHUNK_SEPARATOR) if message.name == "search_child_chunks" else [content]
            contexts.extend(part for part in parts if part)
    return list(dict.fromkeys(contexts))

def _format_conversation(messages) -> str:
    lines = []
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)

def _remove_messages_not_in(messages, keep_ids):
    removals = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if isinstance(msg, SystemMessage) or not msg_id:
            continue
        if msg_id not in keep_ids:
            removals.append(RemoveMessage(id=msg_id))
    return removals

def _recent_conversation(messages, pending_query="") -> list:
    """Return recent context before the current user message.

    During clarification, exclude the unresolved query and the assistant's
    clarification request because they are represented explicitly.
    """
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    recent_messages = plain_messages[:-1]

    if pending_query:
        for index in range(len(recent_messages) - 1, -1, -1):
            msg = recent_messages[index]
            if isinstance(msg, HumanMessage) and str(msg.content).strip() == pending_query:
                return recent_messages[:index]

    return recent_messages


def _invoke_with_retry(llm, messages):
    """Invoke the LLM with automatic retries.

    DashScope's OpenAI-compatible endpoint intermittently aborts qwen
    streaming JSON generation with a 400 InvalidParameter error. Retrying the
    call usually succeeds on the next attempt.
    """
    return llm.with_retry(stop_after_attempt=3).invoke(messages)

def summarize_history(state: State, llm):
    messages = state.get("messages", [])
    updates = {"agent_answers": [{"__reset__": True}]}

    if not messages:
        return updates

    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    keep_count = PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP
    messages_to_summarize = plain_messages[:-keep_count] if len(plain_messages) > keep_count else []
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-keep_count:]}
    keep_ids.discard(None)

    removals = _remove_messages_not_in(messages, keep_ids)
    if removals:
        updates["messages"] = removals

    if not messages_to_summarize:
        return updates

    existing_summary = state.get("conversation_summary", "").strip()
    conversation = "Existing summary:\n"
    conversation += f"{existing_summary or '(none)'}\n\n"
    conversation += "New messages to merge into the summary:\n"
    conversation += _format_conversation(messages_to_summarize)

    summary_response = _invoke_with_retry(llm, [
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation),
    ])
    updates["conversation_summary"] = summary_response.content.strip()
    return updates

def rewrite_query(state: State, llm):
    last_message = state["messages"][-1]
    current_query = str(last_message.content).strip()
    conversation_summary = state.get("conversation_summary", "").strip()
    pending_query = state.get("pendingQuery", "").strip()
    pending_clarifications = state.get("pendingClarifications", [])
    recent_messages = _recent_conversation(state["messages"], pending_query)

    context_parts = []
    if conversation_summary:
        context_parts.append(f"Conversation Summary:\n{conversation_summary}")
    if recent_messages:
        context_parts.append(f"Recent Conversation:\n{_format_conversation(recent_messages)}")

    if pending_query:
        clarifications = [*pending_clarifications, current_query]
        clarification_text = "\n".join(
            f"{index}. {value}" for index, value in enumerate(clarifications, start=1)
        )
        context_parts.append(
            f"Unresolved User Query:\n{pending_query}\n\n"
            f"User Clarifications:\n{clarification_text}"
        )
        original_query = f"{pending_query}\nClarifications:\n{clarification_text}"
    else:
        clarifications = []
        context_parts.append(f"User Query:\n{current_query}")
        original_query = current_query

    context_section = "\n\n".join(context_parts)
    llm_with_structure = llm.with_structured_output(QueryAnalysis).with_retry(
        stop_after_attempt=3
    )
    response = llm_with_structure.invoke([SystemMessage(content=get_rewrite_query_prompt()), HumanMessage(content=context_section)])
    clarification_message_update = (
        [_name_internal_message(last_message, "clarification_response")]
        if pending_query else []
    )

    if response.questions and response.is_clear:
        return {
            "questionIsClear": True,
            "originalQuery": original_query,
            "pendingQuery": "",
            "pendingClarifications": [],
            "rewrittenQuestions": response.questions,
            "messages": clarification_message_update,
        }

    clarification = response.clarification_needed if response.clarification_needed and len(response.clarification_needed.strip()) > 10 else "I need more information to understand your question."
    return {
        "questionIsClear": False,
        "originalQuery": "",
        "pendingQuery": pending_query or current_query,
        "pendingClarifications": clarifications,
        "rewrittenQuestions": [],
        "messages": clarification_message_update + [
            AIMessage(content=clarification, name="clarification")
        ],
    }

def request_clarification(state: State):
    return {}

# --- Agent Nodes ---
def orchestrator(state: AgentState, llm_with_tools):
    context_summary = state.get("context_summary", "").strip()
    sys_msg = SystemMessage(content=get_orchestrator_prompt())
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )
    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"], name="agent_question")
        force_search = HumanMessage(content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION.")
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
        response = _name_internal_message(response, "agent_response")
        return {"messages": [human_msg, response], "tool_call_count": len(response.tool_calls or []), "iteration_count": 1}

    response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
    response = _name_internal_message(response, "agent_response")
    tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
    return {"messages": [response], "tool_call_count": len(tool_calls) if tool_calls else 0, "iteration_count": 1}

def fallback_response(state: AgentState, llm):
    seen = set()
    unique_contents = []
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.content not in seen:
            unique_contents.append(m.content)
            seen.add(m.content)

    context_summary = state.get("context_summary", "").strip()

    context_parts = []
    if context_summary:
        context_parts.append(f"## Compressed Research Context (from prior iterations)\n\n{context_summary}")
    if unique_contents:
        context_parts.append(
            "## Retrieved Data (current iteration)\n\n" +
            "\n\n".join(f"--- DATA SOURCE {i} ---\n{content}" for i, content in enumerate(unique_contents, 1))
        )

    context_text = "\n\n".join(context_parts) if context_parts else "No data was retrieved from the documents."

    prompt_content = (
        f"USER QUERY: {state.get('question')}\n\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\nProvide the best possible answer using only the data above."
    )
    response = _invoke_with_retry(llm, [
        SystemMessage(content=get_fallback_response_prompt()),
        HumanMessage(content=prompt_content),
    ])
    response = _name_internal_message(response, "agent_response")
    return {"messages": [response]}

def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
    messages = state["messages"]

    new_ids: Set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "retrieve_parent_chunks":
                    raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                    if isinstance(raw, str):
                        new_ids.add(f"parent::{raw}")
                    else:
                        new_ids.update(f"parent::{r}" for r in raw)

                elif tc["name"] == "search_child_chunks":
                    query = tc["args"].get("query", "")
                    if query:
                        new_ids.add(f"search::{query}")
            break

    updated_ids = state.get("retrieval_keys", set()) | new_ids

    current_token_messages = estimate_context_tokens(messages)
    current_token_summary = estimate_context_tokens([HumanMessage(content=state.get("context_summary", ""))])
    current_tokens = current_token_messages + current_token_summary

    max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)

    goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
    return Command(
        update={
            "retrieval_keys": updated_ids,
            "retrieved_contexts": _retrieval_contexts(messages),
        },
        goto=goto,
    )

def compress_context(state: AgentState, llm):
    messages = state["messages"]
    existing_summary = state.get("context_summary", "").strip()

    if not messages:
        return {}

    conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
    if existing_summary:
        conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"

    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            tool_calls_info = ""
            if getattr(msg, "tool_calls", None):
                calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                tool_calls_info = f" | Tool calls: {calls}"
            conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool")
            conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"

    summary_response = _invoke_with_retry(llm, [
        SystemMessage(content=get_context_compression_prompt()),
        HumanMessage(content=conversation_text),
    ])
    new_summary = summary_response.content

    retrieved_ids: Set[str] = state.get("retrieval_keys", set())
    if retrieved_ids:
        parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
        search_queries = sorted(r.replace("search::", "") for r in retrieved_ids if r.startswith("search::"))

        block = "\n\n---\n**Already executed (do NOT repeat):**\n"
        if parent_ids:
            block += "Parent chunks retrieved:\n" + "\n".join(f"- {p.replace('parent::', '')}" for p in parent_ids) + "\n"
        if search_queries:
            block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
        new_summary += block

    return {"context_summary": new_summary, "messages": [RemoveMessage(id=m.id) for m in messages[1:]]}

def collect_answer(state: AgentState):
    last_message = state["messages"][-1]
    is_valid = isinstance(last_message, AIMessage) and last_message.content and not last_message.tool_calls
    answer = last_message.content if is_valid else "Unable to generate an answer."
    return {
        "final_answer": answer,
        "agent_answers": [{
            "index": state["question_index"],
            "question": state["question"],
            "answer": answer,
            "contexts": state.get("retrieved_contexts", []),
        }]
    }
# --- End of Agent Nodes---

def aggregate_answers(state: State, llm):
    messages = state.get("messages", [])
    plain_messages = [msg for msg in messages if _is_plain_conversation_message(msg)]
    keep_ids = {getattr(msg, "id", None) for msg in plain_messages[-PRE_ANSWER_HISTORY_MESSAGES_TO_KEEP:]}
    keep_ids.discard(None)
    removals = _remove_messages_not_in(messages, keep_ids)

    if not state.get("agent_answers"):
        return {"messages": removals + [AIMessage(content="No answers were generated.")]}

    sorted_answers = sorted(state["agent_answers"], key=lambda x: x["index"])

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (f"\nRetrieved response {i}:\n"f"{ans['answer']}\n")

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}\nRetrieved answers:{formatted_answers}""")
    synthesis_response = _invoke_with_retry(llm, [
        SystemMessage(content=get_aggregation_prompt()),
        user_message,
    ])
    return {"messages": removals + [AIMessage(content=synthesis_response.content)]}
