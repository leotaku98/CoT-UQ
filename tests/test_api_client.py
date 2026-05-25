import pytest
from unittest.mock import MagicMock, patch


def test_chat_complete_featherless(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "fake-key")
    monkeypatch.setenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Step 1: test\nFinal Answer: 42"

    with patch("src.model.api_client.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        from src.model.api_client import chat_complete
        result = chat_complete(
            "some prompt",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            provider="featherless",
            temperature=1.0,
            max_new_tokens=128,
            top_p=0.9,
        )

    assert result == "Step 1: test\nFinal Answer: 42"
    mock_openai_cls.assert_called_once_with(
        api_key="fake-key", base_url="https://api.featherless.ai/v1"
    )


def test_chat_complete_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Final Answer: Paris"

    with patch("src.model.api_client.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        from src.model.api_client import chat_complete
        result = chat_complete(
            "capital of France?",
            model_id="gpt-4o-mini",
            provider="openai",
        )

    assert result == "Final Answer: Paris"
    mock_openai_cls.assert_called_once_with(
        api_key="sk-fake", base_url="https://api.openai.com/v1"
    )


def test_chat_complete_unknown_provider():
    from src.model.api_client import chat_complete
    with pytest.raises(ValueError, match="Unknown provider"):
        chat_complete("prompt", model_id="model", provider="mystery")


def test_chat_complete_retries_on_error(monkeypatch):
    monkeypatch.setenv("FEATHERLESS_API_KEY", "fake-key")
    monkeypatch.setenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"

    call_count = 0

    def flaky_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("temporary error")
        return mock_response

    with patch("src.model.api_client.OpenAI") as mock_openai_cls:
        with patch("src.model.api_client.time.sleep"):
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = flaky_create

            from src.model.api_client import chat_complete
            result = chat_complete("prompt", model_id="m", provider="featherless")

    assert result == "ok"
    assert call_count == 3
