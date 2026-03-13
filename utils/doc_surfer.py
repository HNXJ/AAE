import os
import sys
from typing import List, Optional

def surf_documentation(urls: List[str], topic: str, context: Optional[str] = None):
    """
    Structured interface for systematic documentation analysis.
    This function is designed to be called by Gemini CLI to extract high-signal
    technical content for skill enrichment.
    """
    prompt = f"Topic: {topic}\n"
    if context:
        prompt += f"Context: {context}\n"
    prompt += "Please analyze the following documentation URLs and extract: \n"
    prompt += "1. Core API definitions and signatures.\n"
    prompt += "2. Best practices and 'Programming Grammar' unique to this library.\n"
    prompt += "3. Advanced features relevant to biophysical modeling and JAX optimization.\n"
    prompt += "4. Code snippets following modular and PyTree-native patterns.\n\n"
    prompt += "URLs:\n" + "\n".join(urls)
    
    # In practice, Gemini CLI uses the web_fetch tool directly.
    # This script serves as the 'grammar' for how to prompt that tool.
    print("--- SURF PROMPT GENERATED ---")
    print(prompt)
    print("------------------------------")
    return prompt

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python doc_surfer.py <topic> <url1> <url2> ...")
    else:
        surf_documentation(sys.argv[2:], sys.argv[1])
