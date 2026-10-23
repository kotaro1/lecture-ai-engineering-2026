from types import SimpleNamespace

from app import ask


def test_follow_up_questions_share_chat_history(monkeypatch, capsys):
    sent_questions = []

    def send_message(question):
        sent_questions.append(question)
        return SimpleNamespace(text=f"reply to {question}", usage_metadata=None)

    chat = SimpleNamespace(send_message=send_message)
    client = SimpleNamespace(chats=SimpleNamespace(create=lambda model: chat))
    monkeypatch.setattr(ask.genai, "Client", lambda api_key: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "first question"])
    questions = iter(["follow-up", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(questions))

    ask.main()

    assert sent_questions == ["first question", "follow-up"]
    assert "reply to first question" in capsys.readouterr().out