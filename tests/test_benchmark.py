from types import SimpleNamespace

from app.benchmark import run_benchmarks


def test_benchmark_runs_each_model_question_pair(capsys):
    calls = []

    def generate_content(*, model, contents):
        calls.append((model, contents))
        return SimpleNamespace(
            text=f"answer: {contents}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=20,
            ),
        )

    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )

    run_benchmarks(
        client,
        ["model-a", "model-b"],
        ["question-1", "question-2"],
        runs=2,
    )

    assert calls == [
        ("model-a", "question-1"),
        ("model-a", "question-1"),
        ("model-a", "question-2"),
        ("model-a", "question-2"),
        ("model-b", "question-1"),
        ("model-b", "question-1"),
        ("model-b", "question-2"),
        ("model-b", "question-2"),
    ]
    output = capsys.readouterr().out
    assert output.count("[result]") == 8
    assert "input_tokens=10 output_tokens=20" in output