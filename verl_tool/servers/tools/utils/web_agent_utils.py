import sys
sys.path.append("..")

import re
import json
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict
from openai import OpenAI
from typing import Optional, List, Dict, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .deepsearch_utils import extract_snippet_with_context

def extract_answer(output, mode='gen'):
    extracted_text = ''
    if output is None:
        output = "None"
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen': # 提取模型基于网页内容生成的推理
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        # pattern_info = "\n**Final Information**"
        # pattern_step = "\n**Modified Reasoning Steps**"
        pattern_info = "**Final Information**"
        pattern_step = "**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            # extracted_text = "No helpful information found."
            extracted_text = output
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text

def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""Role
You are a strict information extractor for a reasoning agent.

# Task 
Analyze the <searched_web_pages> to find specific evidence that answers the <current_search_query>.

# Constraints
1. Read the web pages and formulate a clear, concise, and direct answer to the <current_search_query>.
2. Your answer must be based *only* on the <searched_web_pages>. Do not use outside knowledge.
3. If the pages contain no relevant info, output "No helpful information found."

# Output Format
You must output the answer inside the following block:
**Final Information**: Your synthesized answer here

---
**Input Data:**

<Previous Reasoning Steps> (For context only):
{prev_reasoning}

<Current Search Query> (The question you need to answer):
{search_query}

<Searched Web Pages> (The source material):
{document}
"""


def webpage_analysis_single(summ_model_url, summ_model_path, prompt) -> str:
    client_summ_model = OpenAI(
        base_url=summ_model_url,
        api_key="EMPTY"
    )
    for i in range(10): # max retry 10 times
        try:
            completion = client_summ_model.chat.completions.create(
                model=summ_model_path,
                max_tokens=8192,
                temperature=0.6,
                top_p=0.95,
                messages=[prompt],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            time.sleep(1)
            continue
    return "None"

def get_prev_reasoning_chain(all_reasoning_steps: Union[str, List[str]], begin_search_tag:str="<search>", begin_search_result_tag:str="<result>") -> str:
    if isinstance(all_reasoning_steps, str):
        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")
    else:
        all_reasoning_steps = [step for step in all_reasoning_steps if step]

    prev_steps = [f"Step {i + 1}: {step}" for i, step in enumerate(all_reasoning_steps)]

    if len(prev_steps) <= 5:
        truncated_prev_reasoning = '\n\n'.join(prev_steps)
    else:
        truncated_prev_reasoning = ''
        for i, step in enumerate(prev_steps):
            if i == 0 or i >= len(prev_steps) - 4 or begin_search_tag in step or begin_search_result_tag in step:
                truncated_prev_reasoning += step + '\n\n'
            else:
                if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                    truncated_prev_reasoning += '...\n\n'
    truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')
    return truncated_prev_reasoning

def generate_webpage_to_reasonchain_batch(
    prev_reasonings: List[str],
    search_queries: List[str],
    documents: List[str],
    summ_model_url: OpenAI,
    summ_model_path: str,
) -> List[str]:

    user_prompts = [
        get_webpage_to_reasonchain_instruction(r, sq, doc)
        for r, sq, doc in zip(prev_reasonings, search_queries, documents)
    ]


    prompts = [{"role": "user", "content": up} for up in user_prompts]
    print("webpage ana prompts[0]")
    print(prompts[0])

    with ThreadPoolExecutor(max_workers=10) as executor:
        raw_outputs = list(tqdm(
            executor.map(lambda p: webpage_analysis_single(summ_model_url, summ_model_path, p), prompts),
            total=len(prompts), desc="generate webpage analyses")
        )

    # Count the number of summarization errors
    sum_error = 0
    for output in raw_outputs:
        if output is None or output == "None" or output == "":
            sum_error += 1
    print(f"summarization_error: {sum_error}, ratios: {sum_error / len(raw_outputs)}")
    
    extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]

    return extracted_infos


def generate_webpage_to_reasonchain(
    prev_reasoning: str,
    search_query: str,
    document: str,
    summ_model_url: str,
    summ_model_path: str,
) -> str:
    user_prompt = get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document)
    prompt = {"role": "user", "content": user_prompt}
    raw_output = webpage_analysis_single(summ_model_url, summ_model_path, prompt)
    analyzed_info = extract_answer(raw_output, mode='infogen')
    return analyzed_info

