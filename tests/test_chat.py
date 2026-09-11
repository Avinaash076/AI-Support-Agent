import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.llm_client import LLMClient
from src.reply_generator import GroundedReplyGenerator


def response(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content))])


class FinalAnswerTests(unittest.TestCase):
    def client(self, responses):
        client = LLMClient.__new__(LLMClient)
        client.provider = "groq"
        client.fast_model = client.reasoning_model = "openai/gpt-oss-20b"
        client.client = Mock()
        client.client.chat.completions.create.side_effect = responses
        return client

    def test_retries_truncation_and_returns_only_final(self):
        client = self.client([response("thinking", "length"),
                              response("<think>private analysis</think>Final answer")])
        self.assertEqual(client.completion("question"), "Final answer")
        calls = client.client.chat.completions.create.call_args_list
        self.assertEqual(calls[1].kwargs["max_tokens"], 2000)
        self.assertFalse(calls[0].kwargs["include_reasoning"])

    def test_empty_or_unfinished_reasoning_never_displayed(self):
        client = self.client([response(""), response("<think>unfinished")])
        with self.assertRaises(RuntimeError):
            client.completion("question")
        self.assertEqual(client.client.chat.completions.create.call_count, 2)

    def test_grounding_and_history(self):
        index, llm = Mock(), Mock()
        index.retrieve.return_value = [
            {"similarity_score": 0.8, "historical_query": "@person phone",
             "historical_response": "@person DM us https://t.co/old"},
            {"similarity_score": 0.0, "historical_query": "irrelevant",
             "historical_response": "unrelated"}]
        llm.completion.return_value = "Which iPhone model?"
        generator = GroundedReplyGenerator(index, llm)
        generator.generate("It still happens", history=[
            {"role": "user", "content": "My iPhone battery drains"}],
            retrieval_query="iPhone battery drains It still happens")
        prompt = llm.completion.call_args.args[0]
        self.assertIn("My iPhone battery drains", prompt)
        self.assertNotIn("https://t.co", prompt)
        self.assertNotIn("@person", prompt)
        self.assertNotIn("unrelated", prompt)

    def test_account_security_uses_fixed_referral(self):
        index, llm = Mock(), Mock()
        index.retrieve.return_value = []
        answer = GroundedReplyGenerator(index, llm).generate(
            "My account is locked", intent="apple_id_security")["drafted_reply"]
        llm.completion.assert_not_called()
        self.assertIn("https://support.apple.com/", answer)
        self.assertIn("cannot access your account", answer)

    def test_unverified_links_are_replaced(self):
        index, llm = Mock(), Mock()
        index.retrieve.return_value = []
        llm.completion.return_value = "See [help](https://t.co/old)."
        answer = GroundedReplyGenerator(index, llm).generate("Help")["drafted_reply"]
        self.assertEqual(answer, "See [help](https://support.apple.com/).")


class ChatInterfaceTests(unittest.TestCase):
    def test_send_followup_and_clear(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        st.cache_resource.clear()
        with patch("src.agent_pipeline.SupportAgentPipeline") as pipeline:
            pipeline.return_value.process_query.return_value = {
                "drafted_reply": "Which iPhone model do you have?", "action": "AUTO_HANDLE"}
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("My battery drains").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.messages), 2)
            app.chat_input[0].set_value("iPhone 13").run()
            self.assertEqual(len(app.session_state.messages), 4)
            self.assertEqual(pipeline.call_count, 1)
            app.sidebar.button[0].click().run()
            self.assertEqual(app.session_state.messages, [])
        st.cache_resource.clear()

    def test_provider_error_keeps_chat_usable(self):
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        st.cache_resource.clear()
        with patch("src.agent_pipeline.SupportAgentPipeline") as pipeline:
            pipeline.return_value.process_query.side_effect = RuntimeError("private provider detail")
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()
            app.chat_input[0].set_value("Help with iCloud").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertNotIn("private provider detail", app.error[0].value)
            self.assertEqual(app.session_state.messages, [])
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
