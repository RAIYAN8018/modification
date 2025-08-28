"""
chatbot_setup.py
Core validation & correction utilities for Q/A JSON records.
- No translation: outputs preserve original languages.
- Robust handling of MCQ mapping and explanation consistency.
"""

import os
import json
import json5
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import openai
import concurrent.futures
import re

# ===============================
# Environment & OpenAI Client
# ===============================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Use the new style OpenAI v1 client
client = openai.OpenAI()

# ===============================
# Exceptions
# ===============================
class QuotaExceededException(Exception):
    """Raised when OpenAI returns an insufficient quota / rate-limit condition."""
    def __init__(self, message: str = "OpenAI API quota exceeded"):
        super().__init__(message)

# ===============================
# API Key Check
# ===============================
def check_api_key_validity() -> bool:
    """
    Quick sanity check that the API key works.
    Prints a message and returns True/False.
    """
    try:
        response = client.models.list()
        if response:
            print("API key is valid. Proceeding with the task...")
            return True
    except openai.AuthenticationError:
        print("Error: The API key is invalid. Please check your API key and try again.")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False
    return False

# ===============================
# Key Maps (English canonical keys)
# ===============================
KEY_MAPS = {
    "English": {
        "question": "Question",
        "discipline": "Discipline",
        "language": "Language",
        "grade": "Grade",
        "explanation": "Explanation",
        "competition_name": "Competition Name",
        "competition_year": "Competition Year",
        "question_number": "Question Number",
        "difficulty": "Difficulty",
        "question_type": "Question Type",
        "answer": "Answer",
    }
}

def normalize_key(key: str) -> str:
    return unicodedata.normalize("NFKC", key.strip().lower())

def get_key_map_for_language(_language: str) -> Dict[str, str]:
    """We always map using English canonical keys for field extraction."""
    return KEY_MAPS["English"]

def map_record_fields(record: Dict[str, Any], key_map: Dict[str, str]) -> Dict[str, Any]:
    """
    Normalize keys and map to canonical field names (question/answer/etc.)
    without altering values.
    """
    normalized_record = {normalize_key(k): v for k, v in record.items()}
    mapped = {}
    for eng_key, local_key in key_map.items():
        val = normalized_record.get(normalize_key(local_key), "")
        mapped[eng_key] = val
    return mapped

# ===============================
# Validation (LLM grading)
# ===============================
def get_system_prompt(question_type: str, target_lang: str) -> str:
    """
    Builds a system prompt that:
      - Evaluates correctness.
      - Produces corrected answer/explanation (in the same language as input).
      - Returns strict JSON with fields: valid, reason, corrected_answer, corrected_explanation.
    """
    base = (
        "You are an expert educator, grader, and language expert specializing in mathematics, physics, and theoretical subjects. "
        "Your task is to evaluate the correctness of a given question and its provided answer in any language. "
        "Follow these steps precisely:\n"
        "1. Assess the provided answer for accuracy based on the given question. Check logical consistency and correctness.\n"
        "2. If incorrect or incomplete, provide a corrected final **Answer** and a step-by-step **Explanation** (clear, conversational, rigorous). Use LaTeX for math when appropriate.\n"
        "3. Ensure the **Explanation** is internally consistent with the final **Answer**.\n"
        "Return valid JSON ONLY in this format:\n"
        "{\"valid\": bool, \"reason\": str, \"corrected_answer\": str or null, \"corrected_explanation\": str or null}\n"
        "- valid: True if the answer is correct, else False.\n"
        "- reason: Why correct/incorrect.\n"
        "- corrected_answer: Provide only if original answer is wrong.\n"
        "- corrected_explanation: Provide only if the original explanation is wrong.\n"
        f"IMPORTANT: Write all your fields (reason, corrected_answer, corrected_explanation) strictly in {target_lang}. "
        "If the input is in a different language, translate your outputs into the target language."
    )
    qt = (question_type or "").strip().lower()
    if qt in ("explanation", "exp"):
        return base + " For EXP problems, put the full solution ONLY in corrected_explanation and leave corrected_answer null."
    elif qt in ("short answer", "sa"):
        return base + " The answer should be concise and precise."
    elif qt in ("multiple choice", "multi"):
        return base + " The answer must be one of the provided options and the reasoning should be clear."
    elif qt in ("true/false", "tf"):
        return base + " The answer must be either 'True' or 'False', explained simply."
    else:
        return base + " Provide a clear, step-by-step explanation."

def extract_json_substring(s: str) -> str:
    start = s.find("{")
    end = s.rfind("}")
    return s[start:end+1] if start != -1 and end != -1 and end > start else s

def call_openai(question: str, answer: str, question_type: str, target_lang: str) -> Dict[str, Any]:
    """
    Calls OpenAI to evaluate and (if needed) correct a single record.
    Returns the parsed JSON dict from the model.
    """
    system_prompt = get_system_prompt(question_type, target_lang)
    prompt = f"Question:\n{question}\n\nAnswer:\n{answer}"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.8,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000
        )
        raw = response.choices[0].message.content
        json_text = extract_json_substring(raw)
        try:
            result = json.loads(json_text)
        except json.JSONDecodeError:
            result = json5.loads(json_text)
        return result
    except openai.RateLimitError as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            raise QuotaExceededException(f"OpenAI API quota exceeded: {error_msg}")
        return {"valid": False, "reason": f"Rate limit error: {error_msg}", "corrected_answer": None, "corrected_explanation": None}
    except openai.AuthenticationError as e:
        return {"valid": False, "reason": f"Authentication error: {str(e)}", "corrected_answer": None, "corrected_explanation": None}
    except Exception as e:
        return {"valid": False, "reason": f"OpenAI API error: {str(e)}", "corrected_answer": None, "corrected_explanation": None}

# ===============================
# MC helpers (no extra fields in output)
# ===============================
_letter_text_re = re.compile(r"\s*[\(\[]?([A-Za-z])[\)\].]?\s*(.*)")

def _extract_letter_and_text(s: str) -> Tuple[Optional[str], str]:
    if not isinstance(s, str):
        return None, str(s)
    m = _letter_text_re.match(s)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return None, s.strip()

def _build_options_map(options: Any) -> Dict[str, str]:
    """
    Returns a map like {"A": "10", "B": "20", ...}
    Supports list[str] (e.g., ["A) 10", "B) 20"]) or dict values containing such strings.
    """
    out: Dict[str, str] = {}
    if isinstance(options, list):
        for item in options:
            letter, text = _extract_letter_and_text(item)
            if letter:
                out[letter] = text
    elif isinstance(options, dict):
        for _, v in options.items():
            letter, text = _extract_letter_and_text(v)
            if letter:
                out[letter] = text
    return out

def _coerce_answer_to_option_text(record: Dict[str, Any], candidate: Optional[str]) -> Optional[str]:
    """
    Force the candidate answer to match one of the option TEXTS.
    If candidate is 'B' or '(B)', returns the text behind B (e.g., '20').
    If candidate equals the text itself (e.g., '20'), returns it unchanged if present.
    If unmappable, return None.
    """
    if candidate is None:
        return None

    options = record.get("options", record.get("Options"))
    if not options:
        return None

    omap = _build_options_map(options)  # {"A": "10", "B": "20", ...}
    if not omap:
        return None

    cand = str(candidate).strip()

    # If candidate looks like a labeled option, map to its text
    c_letter, c_text = _extract_letter_and_text(cand)
    if c_letter and c_letter in omap:
        return omap[c_letter]

    # Direct text match?
    if c_text and c_text in omap.values():
        return c_text
    if cand in omap.values():
        return cand

    # Sometimes candidate may be like "B) 20" exactly
    parts = re.split(r"\s+", cand, maxsplit=1)
    if parts and len(parts) == 2:
        _, maybe_text = _extract_letter_and_text(cand)
        if maybe_text in omap.values():
            return maybe_text

    return None

# ===============================
# MC explanation consistency helpers
# ===============================
_num_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _extract_numbers(s: str) -> list:
    if not isinstance(s, str):
        return []
    return [n for n in _num_re.findall(s)]

def _needs_explanation_fix(final_answer_text: str, explanation: str) -> bool:
    """
    Heuristic: if the final answer contains a number and the explanation never mentions that number,
    we likely need to fix the explanation. If the answer is non-numeric, skip fixing.
    """
    ans_nums = _extract_numbers(final_answer_text)
    if not ans_nums:
        return False
    exp_nums = _extract_numbers(explanation or "")
    return not any(a in exp_nums for a in ans_nums)

def _format_options_for_prompt(options: Any) -> str:
    if isinstance(options, list):
        return "\n".join(str(x) for x in options)
    if isinstance(options, dict):
        return "\n".join(str(v) for v in options.values())
    return ""

def _fix_mc_explanation_in_language(
    question: str,
    options: Any,
    final_answer_text: str,
    explanation: str,
    language: str
) -> str:
    """
    Rewrite the explanation so it logically leads to the final_answer_text.
    Output must be in the given language. Returns the revised explanation text.
    """
    options_block = _format_options_for_prompt(options)
    system = (
        f"You are a careful educator. Rewrite the explanation so that it is internally consistent and correctly justifies "
        f"the final answer: {final_answer_text}. Use the target language: {language}. "
        "Preserve LaTeX if present. Do not mention contradictions or previous wrong values. "
        "Return only the final explanation text, no preface."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Options:\n{options_block}\n\n"
        f"Final answer (must be justified): {final_answer_text}\n\n"
        f"Current explanation (may be inconsistent):\n{explanation}"
    )
    try:
        rsp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=1200
        )
        fixed = rsp.choices[0].message.content.strip()
        return fixed or explanation
    except openai.RateLimitError as e:
        msg = str(e)
        if "insufficient_quota" in msg.lower() or "quota" in msg.lower():
            raise QuotaExceededException(f"Explanation-fix quota exceeded: {msg}")
        return explanation
    except openai.AuthenticationError as e:
        raise QuotaExceededException(f"Authentication error during explanation-fix: {str(e)}")
    except Exception:
        return explanation

# ===============================
# Main record validator
# ===============================
def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates/corrects a single record while preserving the original language.
    - EXP: put the full solution in 'answer' and REMOVE 'explanation'.
    - MC: ensure final 'answer' is the option TEXT; fix explanation if inconsistent.
    - Other: adopt corrected_answer/explanation when provided.
    """
    key_map = get_key_map_for_language("English")
    mapped = map_record_fields(record, key_map)
    question = mapped.get("question", "")
    answer = mapped.get("answer", "")
    question_type = mapped.get("question_type", "unspecified")

    # Keep the record's own language; no translation.
    record_lang = str(record.get("Language", mapped.get("language", "English"))) or "English"

    result = call_openai(question, answer, question_type, record_lang)
    new_rec = record.copy()
    qt_norm = (question_type or "").strip().lower()

    # ---------- EXP behavior ----------
    if qt_norm in {"exp", "explanation"}:
        corrected_expl = result.get("corrected_explanation")
        corrected_ans = result.get("corrected_answer")
        final_text = corrected_expl or corrected_ans or new_rec.get("explanation") or new_rec.get("answer", "")
        new_rec["answer"] = final_text
        if "explanation" in new_rec:
            del new_rec["explanation"]

    # ---------- MC behavior ----------
    elif qt_norm in {"multi", "multiple choice", "multiple-choice"}:
        corrected_expl = result.get("corrected_explanation")
        if corrected_expl:
            new_rec["explanation"] = corrected_expl

        corrected_ans = result.get("corrected_answer")
        candidate = corrected_ans if corrected_ans else new_rec.get("answer", "")
        coerced = _coerce_answer_to_option_text(new_rec, candidate)
        if coerced is not None:
            new_rec["answer"] = coerced
        else:
            coerced_orig = _coerce_answer_to_option_text(new_rec, new_rec.get("answer", ""))
            if coerced_orig is not None:
                new_rec["answer"] = coerced_orig

        final_answer_text = str(new_rec.get("answer", ""))
        current_expl = str(new_rec.get("explanation", "") or "")
        if _needs_explanation_fix(final_answer_text, current_expl):
            options = new_rec.get("options", new_rec.get("Options"))
            fixed_expl = _fix_mc_explanation_in_language(
                question=question,
                options=options,
                final_answer_text=final_answer_text,
                explanation=current_expl,
                language=record_lang
            )
            if fixed_expl:
                new_rec["explanation"] = fixed_expl

    # ---------- Other types ----------
    else:
        corrected_expl = result.get("corrected_explanation")
        if corrected_expl:
            new_rec["explanation"] = corrected_expl
        corrected_ans = result.get("corrected_answer")
        if corrected_ans:
            new_rec["answer"] = corrected_ans

    # Defensive cleanup: never ship helper keys
    for k in ["_validation", "explanation_status", "_mc_warning", "mc"]:
        if k in new_rec:
            del new_rec[k]

    return new_rec

# ===============================
# Parallel Processing
# ===============================
def process_records_parallel(records: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Validate/correct multiple records concurrently.
    Raises QuotaExceededException to caller for UI handling.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(validate_record, rec): i for i, rec in enumerate(records)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except QuotaExceededException:
                # Cancel remaining tasks and bubble up
                for f in futures:
                    f.cancel()
                raise
    return [r for r in results if r is not None]

# ===============================
# Entry Check (optional use)
# ===============================
if __name__ == "__main__":
    # Only runs when executing this file directly.
    if not check_api_key_validity():
        raise SystemExit(1)
