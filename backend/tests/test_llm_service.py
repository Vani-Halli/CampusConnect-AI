from unittest.mock import patch

from app.services.llm_service import generate_answer_with_llm


def test_llm_service_rejects_unsupported_provider_without_request():
    with patch("app.services.llm_service.request.urlopen") as urlopen:
        answer, status = generate_answer_with_llm(
            "Question",
            [],
            provider="unsupported",
            api_key="test-key",
        )

    assert answer is None
    assert status == "unsupported_provider"
    urlopen.assert_not_called()


def test_llm_service_sends_api_key_as_bearer_auth():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"Grounded answer"}}]}'

    with patch("app.services.llm_service.request.urlopen", return_value=FakeResponse()) as urlopen:
        answer, status = generate_answer_with_llm(
            "Question",
            [{"title": "Source", "answer": "Evidence"}],
            provider="openai-compatible",
            api_key="test-key",
            model="test-model",
        )

    sent_request = urlopen.call_args.args[0]
    assert sent_request.get_header("Authorization") == "Bearer test-key"
    assert answer == "Grounded answer"
    assert status == "llm_generated"
