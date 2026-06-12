import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from rag.analytics.tracker import tracker
from rag.utils.exceptions import LLMError

logger = logging.getLogger("rag_platform")

class TokenTrackerWrapper:
    """Wraps a LangChain LLM to track token usage details after each invocation."""
    def __init__(self, llm: ChatOpenAI):
        self._llm = llm

    def invoke(self, *args, **kwargs):
        try:
            response = self._llm.invoke(*args, **kwargs)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise LLMError(f"LLM call failed: {e}") from e

        try:
            # Extract token usage
            if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
                usage = response.response_metadata["token_usage"]
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                
                cached_prompt_tokens = 0
                if "prompt_tokens_details" in usage:
                    cached_prompt_tokens = usage["prompt_tokens_details"].get("cached", 0)
                elif "cached_tokens" in usage:
                    cached_prompt_tokens = usage.get("cached_tokens", 0)
                
                tracker.add_llm_tokens(prompt_tokens, completion_tokens, cached_prompt_tokens)
            else:
                # Fallback to character-based estimates (len // 4)
                input_chars = 0
                if args:
                    messages = args[0]
                    if isinstance(messages, list):
                        for m in messages:
                            if isinstance(m, dict) and "content" in m:
                                input_chars += len(m["content"])
                            elif hasattr(m, "content"):
                                input_chars += len(m.content)
                    elif isinstance(messages, str):
                        input_chars += len(messages)
                
                output_chars = len(response.content) if hasattr(response, "content") else 0
                est_input = max(1, input_chars // 4)
                est_output = max(1, output_chars // 4)
                tracker.add_llm_tokens(est_input, est_output, 0)
        except Exception as e:
            logger.warning(f"Error tracking LLM tokens: {e}")
            
        return response

    def __getattr__(self, name):
        return getattr(self._llm, name)


def create_llm_client(model_name: str, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com", temperature: float = 0.2) -> TokenTrackerWrapper:
    """Factory function to build and wrap a LangChain ChatOpenAI instance with TokenTrackerWrapper."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise LLMError("API Key is missing. Please set the DEEPSEEK_API_KEY environment variable.")
        
    raw_llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=key,
        base_url=base_url,
    )
    return TokenTrackerWrapper(raw_llm)
