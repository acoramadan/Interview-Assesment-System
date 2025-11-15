import os
import json
import yaml
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError
load_dotenv()

CONFIG_PATH = "../conf.yaml"
JSON_PATH = "../result/asr_output/interview_question_5_asr_output.json"  #


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def assess_answer(qid: str, candidate_answer: str):
    cfg = load_config()

    model_cfg = cfg["model"]
    rubric = cfg["rubric"][qid]
    prompt_cfg = cfg["prompts"]["assesor_single"]

    user_prompt = prompt_cfg["template"].format(
        question=rubric["question"],
        lvl4=rubric["scale"]["4"],
        lvl3=rubric["scale"]["3"],
        lvl2=rubric["scale"]["2"],
        lvl1=rubric["scale"]["1"],
        lvl0=rubric["scale"]["0"],
        answer=candidate_answer.strip(),
    )

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        )
    ]

    tools = []
    if model_cfg.get("use_google_search", False):
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    gen_config = types.GenerateContentConfig(
        system_instruction=prompt_cfg["system"],
        temperature=model_cfg.get("temperature", 0.0),
        max_output_tokens=model_cfg.get("max_output_tokens", 2048),
        response_mime_type=model_cfg.get("response_mime_type", "application/json"),
        thinking_config=types.ThinkingConfig(
            thinking_budget=model_cfg.get("thinking_budget", -1)
        ),
        tools=tools or None,
    )

    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=model_cfg["model_name"],
                contents=contents,
                config=gen_config,
            )

            break
        
        except ServerError as e:
            if e.status_code == 503 and attempt < 2:
                wait_time = (2 ** attempt) * 2
                print(f"ServerError 503 encountered. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise e

    if resp.text is None:
        texts = []
        for cand in resp.candidates:
            for part in cand.content.parts:
                if hasattr(part, 'thought') and part.thought:
                    continue  
                if getattr(part, "text", None):
                    texts.append(part.text)
        joined = "\n".join(texts)
        if not joined:
            raise RuntimeError("No text in response. Check prompt or safety blocks.")
        result = json.loads(joined)
        return result

    result = json.loads(resp.text)
    return result


if __name__ == "__main__":
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        asr_output = json.load(f)

    candidate_answer_q1 = asr_output["full_scripts"]

    result = assess_answer("q3", candidate_answer_q1)

    print("\nParsed result:")
    print("Score:", result["score"])
    print("Reason:", result["reason"])
    print("Matched level:", result["matched_level"])

    with open("../result/reasoning_output/assessment_result_5.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
