"""複数のモデル・質問で応答時間とトークン数を比較する。"""

import argparse
import time

from google import genai
from google.genai import errors

from app.ask import DEFAULT_QUESTION
from app.config import MODEL, api_error_message, get_api_key

DEFAULT_MODELS = [MODEL, "gemini-3.5-flash"]
DEFAULT_QUESTIONS = ["LoRAとは何ですか？", DEFAULT_QUESTION]


def run_benchmarks(
    client, models: list[str], questions: list[str], runs: int = 1
) -> None:
    for model in models:
        for question in questions:
            for run in range(1, runs + 1):
                started = time.perf_counter()
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=question,
                    )
                except errors.APIError as error:
                    elapsed = time.perf_counter() - started
                    print(f"[error] model={model} run={run} elapsed={elapsed:.2f}s")
                    print(api_error_message(error.code, error.message))
                    if error.code == 429:
                        print("レート制限のため、残りの計測を中止します。")
                        return
                    continue

                elapsed = time.perf_counter() - started
                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count if usage else "N/A"
                output_tokens = usage.candidates_token_count if usage else "N/A"
                print(
                    f"[result] model={model} run={run}/{runs} "
                    f"elapsed={elapsed:.2f}s input_tokens={input_tokens} "
                    f"output_tokens={output_tokens}"
                )
                print(f"[question] {question}")
                print(f"[answer]\n{response.text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="複数のモデル・質問で応答時間とトークン数を比較する"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="比較するモデル名（既定: %(default)s）",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        default=DEFAULT_QUESTIONS,
        help="比較する質問（既定: 一般的な質問と講座固有の質問）",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="各モデル・質問の実行回数（既定: %(default)s）",
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs は1以上を指定してください")

    print(f"[models] {', '.join(args.models)}")
    client = genai.Client(api_key=get_api_key())
    run_benchmarks(client, args.models, args.questions, args.runs)


if __name__ == "__main__":
    main()