import asyncio
import base64
import json
import threading

import httpx
import pytest
from PIL import Image
from vrms import png, tiny_vrm

from tomo import ai, characters, sysinfo
from tomo.ai import (ACT_NOW, CHECK_RESULTS, AiClient, Api, ModelInfo, Msg, SharedCatalog, ToolCall, action_failed,
                     billions, calls_in_text, claims_success, conversation, ollama_request, only_promises,
                     openai_request, parse_ollama, parse_openai, pick_model, server_candidates, tidy, tool_names,
                     tool_specs, without_thinking)
from tomo.commands import Executor
from tomo.config import Config, LlmProvider
from tomo.db import Store
from tomo.events import Animate, Attachment, Chat, ChatLine, Role
from tomo.language import Language


def line(role, text):
    return ChatLine(role, text, 0)


def test_a_conversation_opens_with_the_user_and_alternates():
    history = [line(Role.ASSISTANT, "hi, I'm Tomo"), line(Role.USER, "hey"), line(Role.SYSTEM, "status"),
               line(Role.ASSISTANT, "what's up?"), line(Role.ASSISTANT, "anything?")]
    msgs = conversation(history, "open firefox")
    assert [(m.role, m.text) for m in msgs] == [("user", "hey"), ("assistant", "what's up?\n\nanything?"),
                                                ("user", "open firefox")]


def test_a_new_line_after_an_unanswered_one_joins_it():
    msgs = conversation([line(Role.USER, "hello?")], "are you there")
    assert [(m.role, m.text) for m in msgs] == [("user", "hello?\n\nare you there")]


def test_it_looks_for_ollama_then_lm_studio():
    assert server_candidates(LlmProvider.AUTO, "") == [(Api.OLLAMA, ai.OLLAMA_URL), (Api.OPENAI, ai.LM_STUDIO_URL)]
    assert server_candidates(LlmProvider.AUTO, "http://pc:11434/") == [(Api.OLLAMA, "http://pc:11434")]
    assert server_candidates(LlmProvider.LM_STUDIO, "http://pc:1234") == [(Api.OPENAI, "http://pc:1234/v1")]
    assert server_candidates(LlmProvider.OLLAMA, "http://pc:11434/v1") == [(Api.OLLAMA, "http://pc:11434")]
    with pytest.raises(LookupError):
        server_candidates(LlmProvider.OPENAI_COMPATIBLE, "")


def test_it_picks_a_local_chat_model():
    models = [ModelInfo("qwen2.5:0.5b", True, 0.49), ModelInfo("gemma4:31b-cloud", False, None),
              ModelInfo("nomic-embed-text", True, 0.13), ModelInfo("qwen2.5:7b", True, 7.6)]
    assert pick_model(models).name == "qwen2.5:7b"
    only_tiny = pick_model(models[:1])
    assert only_tiny.name == "qwen2.5:0.5b" and only_tiny.small


def test_model_sizes_are_read_as_ollama_writes_them():
    assert abs(billions("494.03M") - 0.49403) < 1e-6
    assert billions("7.6B") == 7.6 and billions(" 1.2T ") == 1200.0 and billions("8000000000") == 8.0
    assert billions("") is None and billions("big") is None


def test_ollama_requests_carry_the_context_images_and_tool_calls():
    call = ToolCall("call_0", "animate", {"clip": "wave"})
    messages = [Msg("system", "be nice"), Msg("user", "look", images=["QUJD"]), Msg("assistant", "", calls=[call]),
                Msg("tool", "waving", tool_id="call_0", tool_name="animate")]
    body = ollama_request("qwen2.5:7b", messages, tool_specs(False), 8192, False, "1h")
    assert body["options"]["num_ctx"] == 8192 and body["stream"] is False
    assert body["think"] is False and body["keep_alive"] == "1h"
    assert body["messages"][1]["images"] == ["QUJD"]
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"]["clip"] == "wave"
    assert body["messages"][3] == {"role": "tool", "content": "waving", "tool_name": "animate"}
    assert "tools" not in ollama_request("m", messages, None, 4096, False, "1h")
    assert ollama_request("m", messages, None, 4096, True, "-1")["keep_alive"] == -1


def test_openai_requests_use_data_urls_and_json_text_arguments():
    call = ToolCall("call_7", "walk_to", {"position": 0.3})
    messages = [Msg("user", "look", images=["QUJD"]), Msg("assistant", "", calls=[call]),
                Msg("tool", "walking", tool_id="call_7", tool_name="walk_to")]
    body = openai_request("m", messages, tool_specs(False))
    assert body["messages"][0]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"position": 0.3}
    assert body["messages"][1]["content"] is None
    assert body["messages"][2]["tool_call_id"] == "call_7"
    assert body["tool_choice"] == "auto"


def test_replies_are_read_in_both_formats():
    text, calls = parse_ollama({"message": {"content": "hi", "tool_calls": [
        {"function": {"name": "express", "arguments": {"emotion": "happy"}}}]}})
    assert text == "hi" and calls[0].name == "express" and calls[0].arguments == {"emotion": "happy"}
    text, calls = parse_openai({"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "x", "function": {"name": "walk_to", "arguments": "{\"position\": 1}"}}]}}]})
    assert text == "" and calls[0].id == "x" and calls[0].arguments == {"position": 1}
    with pytest.raises(RuntimeError):
        parse_ollama({"error": "model not found"})


def test_thinking_stays_out_of_the_reply():
    assert without_thinking("<think>hmm</think>Hello!") == "Hello!"
    assert without_thinking("planning…</think>Hi") == "Hi"
    assert tidy("<think>x</think>  Hey  ", [], [])[0] == "Hey"


def test_tool_calls_written_into_the_text_are_understood():
    known = tool_names(tool_specs(True))
    rest, calls = calls_in_text('Sure. <tool_call>{"name": "animate", "arguments": {"clip": "wave"}}</tool_call>', known)
    assert rest.strip() == "Sure." and calls[0].name == "animate"
    rest, calls = calls_in_text('{"name": "walk_to", "parameters": {"position": 0.2}}', known)
    assert rest == "" and calls[0].arguments == {"position": 0.2}
    assert calls_in_text('{"name": "rm_rf", "arguments": {}}', known)[1] == []


def test_screen_tools_are_offered_only_when_allowed():
    assert "look_at_screen" in tool_names(tool_specs(True))
    assert "look_at_screen" not in tool_names(tool_specs(False))
    for tool in tool_specs(True):
        assert tool["function"]["parameters"]["type"] == "object"


def test_promises_are_told_from_answers():
    assert only_promises("I'll open Word, Excel, and PowerPoint for you now. Let me get started!")
    assert only_promises("Веднага ще отворя браузъра!")
    assert not only_promises("Done, the volume's at 50%.")
    assert not only_promises("В момента е 7 часа.")
    assert not only_promises("Notepad opened. Let me know if you'd like anything else!")


def test_failures_and_claims_are_recognised():
    assert action_failed("REFUSED: command execution is disabled in config. Nothing was done: …")
    assert action_failed("exit=1\nstderr:\nnot found")
    assert action_failed("no installed app is called 'wrod'; the nearest names: Word")
    assert not action_failed("exit=0")
    assert not action_failed("exit=0\nstdout:\nvolume=40 muted=false")
    assert claims_success("I've opened Word, Excel, and PowerPoint, but I couldn't close DeepCool.")
    assert claims_success("I was able to open Word, Excel, and PowerPoint, but I couldn't close DeepCool.")
    assert claims_success("Готово, отворих браузъра.")
    assert not claims_success("I tried to open the apps, but I don't have permission to run commands.")


# ---- whole turns against a fake model server --------------------------------------------------


class FakeServer:
    """Answers by path, in order: of several replies for a path, each is used
    once (the last one stays); a single one answers every time."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.content:
            self.seen.append(json.loads(request.content))
        path = request.url.path
        matching = [i for i, (p, _) in enumerate(self.replies) if path.endswith(p)]
        if not matching:
            return httpx.Response(404, json={"error": "not found"})
        index = matching[0]
        body = self.replies.pop(index)[1] if len(matching) > 1 else self.replies[index][1]
        return httpx.Response(200, json=body)


def client(tmp_path, server, provider=LlmProvider.OLLAMA, model=""):
    cfg = Config.for_tests(tmp_path)
    cfg.llm_provider = provider
    cfg.llm_url = "http://fake:11434" if provider == LlmProvider.OLLAMA else "http://fake:1234/v1"
    cfg.llm_model = model
    store = Store(tmp_path / "chroma")
    http = httpx.AsyncClient(transport=httpx.MockTransport(server))
    return AiClient(cfg, store, Executor(False, tmp_path / "audit.log"), SharedCatalog(), threading.Event(), http)


def run_turn(ai_client, text):
    events = []
    reply = asyncio.run(ai_client.respond(text, events.append))
    return reply, events


def test_a_turn_with_ollama_runs_the_tools_then_answers(tmp_path):
    server = FakeServer([
        ("/api/tags", {"models": [{"name": "gemma4:31b-cloud", "remote_host": "https://ollama.com"},
                                  {"name": "qwen2.5:0.5b", "details": {"parameter_size": "494.03M"}}]}),
        ("/api/chat", {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "animate", "arguments": {"clip": "wave"}}}]}, "done": True}),
        ("/api/chat", {"message": {"role": "assistant", "content": "<think>ok</think>Hi! *waves*"}, "done": True}),
    ])
    reply, events = run_turn(client(tmp_path, server), "wave at me")
    assert reply == "Hi! *waves*"
    assert Animate("wave") in events
    assert any(isinstance(e, Chat) and "too small to use tools" in e.line.text for e in events)
    assert len(server.seen) == 2
    assert server.seen[0]["model"] == "qwen2.5:0.5b", "the local model, not the cloud one"
    assert server.seen[1]["messages"][-1]["role"] == "tool"


def test_a_turn_with_lm_studio_uses_the_openai_api(tmp_path):
    server = FakeServer([
        ("/v1/models", {"data": [{"id": "qwen2.5-7b-instruct"}]}),
        ("/v1/chat/completions", {"choices": [{"message": {"role": "assistant", "content": "Hello from LM Studio"}}]}),
    ])
    reply, _ = run_turn(client(tmp_path, server, LlmProvider.LM_STUDIO), "hi")
    assert reply == "Hello from LM Studio"
    assert server.seen[0]["model"] == "qwen2.5-7b-instruct"


def test_with_no_server_tomo_says_how_to_get_one(tmp_path):
    def unreachable(request):
        raise httpx.ConnectError("refused", request=request)

    reply, _ = run_turn(client(tmp_path, unreachable, LlmProvider.AUTO), "hi")
    assert "ollama pull" in reply


def test_a_promise_without_a_tool_call_gets_one_reminder_to_act(tmp_path):
    server = FakeServer([
        ("/api/tags", {"models": [{"name": "qwen2.5:7b", "details": {"parameter_size": "7.6B"}}]}),
        ("/api/chat", {"message": {"role": "assistant", "content": "I'll wave right now!"}}),
        ("/api/chat", {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "animate", "arguments": {"clip": "wave"}}}]}}),
        ("/api/chat", {"message": {"role": "assistant", "content": "There, a wave!"}}),
    ])
    reply, events = run_turn(client(tmp_path, server), "wave at me")
    assert reply == "There, a wave!"
    assert Animate("wave") in events
    assert server.seen[1]["messages"][-1]["content"] == ACT_NOW


def test_a_claim_that_nothing_backs_is_sent_back_once(tmp_path):
    # The executor in `client` refuses every command.
    server = FakeServer([
        ("/api/tags", {"models": [{"name": "qwen2.5:7b", "details": {"parameter_size": "7.6B"}}]}),
        ("/api/chat", {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "execute_command", "arguments": {"command": "Start-Process notepad"}}}]}}),
        ("/api/chat", {"message": {"role": "assistant", "content": "Done, I've opened Notepad."}}),
        ("/api/chat", {"message": {"role": "assistant", "content": "Notepad didn't open: commands are switched off."}}),
    ])
    reply, _ = run_turn(client(tmp_path, server), "open notepad")
    assert reply == "Notepad didn't open: commands are switched off."
    assert server.seen[2]["messages"][-1]["content"] == CHECK_RESULTS


def test_only_gestures_with_an_answer_end_the_turn_at_once(tmp_path):
    server = FakeServer([
        ("/api/tags", {"models": [{"name": "qwen2.5:7b", "details": {"parameter_size": "7.6B"}}]}),
        ("/api/chat", {"message": {"role": "assistant", "content": "Hello!", "tool_calls": [
            {"function": {"name": "animate", "arguments": {"clip": "wave"}}}]}}),
    ])
    reply, events = run_turn(client(tmp_path, server), "hi")
    assert reply == "Hello!" and Animate("wave") in events
    assert len(server.seen) == 1, "no second round for a wave"


def test_the_brain_keeps_its_one_model_after_an_error(tmp_path):
    server = FakeServer([
        # The first look finds model A; a second look would find only B.
        ("/api/tags", {"models": [{"name": "a:7b", "details": {"parameter_size": "7B"}}]}),
        ("/api/tags", {"models": [{"name": "b:7b", "details": {"parameter_size": "7B"}}]}),
        ("/api/chat", {"error": "model ran out of memory"}),
        ("/api/chat", {"message": {"role": "assistant", "content": "Back again."}}),
    ])
    ai_client = client(tmp_path, server)
    with pytest.raises(RuntimeError):
        run_turn(ai_client, "hello")
    assert run_turn(ai_client, "hello again")[0] == "Back again."
    assert [r["model"] for r in server.seen] == ["a:7b", "a:7b"], "the same model both times, never another"


def test_the_reply_is_asked_for_in_the_users_language(tmp_path):
    ai_client = client(tmp_path, FakeServer([]))
    assert ai_client.system_prompt(Language.of("Колко е часът?")).endswith("answer in Bulgarian.")
    assert ai_client.system_prompt(Language.of("What time is it?")).endswith("answer in English.")


def test_it_prefers_tomos_own_model():
    models = [ModelInfo("gemma4:26b", True, 25.8), ModelInfo("qwen3.5:9b", True, 9.7), ModelInfo("qwen3.5:35b", True, 36)]
    assert ai.SUGGESTED_MODEL == "qwen3.5:9b" and pick_model(models).name == "qwen3.5:9b"
    assert pick_model(models[:1]).name == "gemma4:26b"


def test_the_persona_knows_the_computer_the_character_and_its_voice(tmp_path):
    ai_client = client(tmp_path, FakeServer([]))
    prompt = ai_client.system_prompt(Language.ENGLISH)
    assert sysinfo.describe() in prompt
    assert "send you images" in prompt and "You appear as" not in prompt
    ai_client.db.add_character("Kiyotaka", str(tmp_path / "k.vrm"))
    ai_client.db.set_active_character("Kiyotaka")
    assert "“Kiyotaka”; your English voice is female (en_GB-cori-medium)" in ai_client.system_prompt(Language.ENGLISH)
    characters.set_voice(ai_client.db, "male")
    assert "your English voice is male (en_GB-alan-medium)" in ai_client.system_prompt(Language.ENGLISH)


def test_set_voice_offers_the_piper_voices_and_gives_the_character_one(tmp_path):
    spec = next(t for t in tool_specs(False) if t["function"]["name"] == "set_voice")["function"]
    assert spec["parameters"]["properties"]["voice"]["enum"] == ["male", "female"]
    assert all(v in spec["description"] for v in ("en_GB-alan-medium", "en_GB-cori-medium", "bg_BG-dimitar-medium"))
    ai_client = client(tmp_path, FakeServer([]))
    set_voice = lambda voice: asyncio.run(ai_client.dispatch_tool("set_voice", {"voice": voice}, []))  # noqa: E731
    assert "no character" in set_voice("male")
    ai_client.db.add_character("Ayako", str(tmp_path / "a.vrm"))
    ai_client.db.set_active_character("Ayako")
    assert set_voice("male") == ("Ayako now speaks English in the male voice (en_GB-alan-medium); "
                                 "Bulgarian stays bg_BG-dimitar-medium")
    assert characters.voice_of(ai_client.db) == "male"
    assert set_voice("robot") == "the voice must be male or female"


def picture(tmp_path, name, size=(320, 240)):
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", size, (90, 160, 90)).save(path, "JPEG")
    return Attachment(str(path), f"C:\\Pictures\\{name}.jpg", f"picture: JPEG {size[0]}×{size[1]}")


def test_pictures_go_with_the_latest_message_that_has_any(tmp_path):
    cat, dog = picture(tmp_path, "cat"), picture(tmp_path, "dog")
    history = [ChatLine(Role.USER, "what's this?", 1, (cat,)), line(Role.ASSISTANT, "A cat."),
               ChatLine(Role.USER, "and this?", 3, (dog,)), line(Role.ASSISTANT, "A dog.")]
    msgs = conversation(history, "which is bigger?")
    assert [len(m.images) for m in msgs] == [0, 0, 1, 0, 0], "only the dog, the latest, is shown again"
    assert "[Image 1: C:\\Pictures\\cat.jpg, sent earlier and not shown again]" in msgs[0].text
    assert msgs[2].text.startswith("and this?\n\n[Image 2: C:\\Pictures\\dog.jpg]\npicture: JPEG 320×240")
    assert Image.open(__import__("io").BytesIO(base64.b64decode(msgs[2].images[0]))).size == (320, 240)
    # A new picture takes over; one sent alone is just its notes.
    msgs = conversation(history, "", (cat,))
    assert [len(m.images) for m in msgs][-1] == 1 and sum(len(m.images) for m in msgs) == 1
    assert msgs[-1].text == "[Image 3: C:\\Pictures\\cat.jpg]\npicture: JPEG 320×240"


def test_a_turn_with_a_picture_sends_it_and_its_notes(tmp_path):
    server = FakeServer([
        ("/api/tags", {"models": [{"name": "qwen3.5:9b", "details": {"parameter_size": "9.7B"}}]}),
        ("/api/chat", {"message": {"role": "assistant", "content": "A green square, taken in July."}}),
    ])
    ai_client = client(tmp_path, server)
    events = []
    reply = asyncio.run(ai_client.respond("what is it?", events.append, (picture(tmp_path, "square"),)))
    assert reply == "A green square, taken in July."
    sent = server.seen[0]["messages"][-1]
    assert sent["role"] == "user" and len(sent["images"]) == 1
    assert sent["content"].startswith("what is it?\n\n[Image 1: C:\\Pictures\\square.jpg]")


def voice_server(*replies):
    return FakeServer([("/api/tags", {"models": [{"name": "qwen3.5:9b", "details": {"parameter_size": "9.7B"}}]})]
                      + [("/api/chat", r) for r in replies])


def set_voice_call(voice):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "set_voice", "arguments": {"voice": voice}}}]}}


def test_the_model_chooses_a_new_characters_voice_from_its_picture(tmp_path):
    server = voice_server(set_voice_call("male"))
    ai_client = client(tmp_path, server)
    vrm = tiny_vrm(tmp_path / "k.vrm", "Kiyotaka Ayanokōji", ("Kalata",), png())
    ai_client.db.add_character("Kiyotaka", str(vrm))
    shown = []
    assert asyncio.run(ai_client.ensure_voice("Kiyotaka", vrm, shown.append)) == "male"
    assert {c.name: c.voice for c in ai_client.db.list_characters()} == {"Kiyotaka": "male"}
    asked = server.seen[0]
    assert [t["function"]["name"] for t in asked["tools"]] == ["set_voice"]
    assert len(asked["messages"][-1]["images"]) == 1, "it's shown the character's picture"
    assert "“Kiyotaka”" in asked["messages"][-1]["content"] and "Kalata" in asked["messages"][-1]["content"]
    assert shown[0].line.text == "Kiyotaka speaks English in a male voice (en_GB-alan-medium)"
    # Known now: not asked again.
    assert asyncio.run(ai_client.ensure_voice("Kiyotaka", vrm, shown.append)) == "male"
    assert len(server.seen) == 1


def test_a_voice_answered_in_words_counts_and_no_answer_keeps_the_default(tmp_path):
    vrm = tiny_vrm(tmp_path / "a.vrm", "Ayako")
    ai_client = client(tmp_path, voice_server({"message": {"role": "assistant", "content": "Female, I'd say."}}))
    assert asyncio.run(ai_client.choose_voice("Ayako", vrm)) == "female"
    assert {c.name: c.voice for c in ai_client.db.list_characters()} == {"Ayako": "female"}, "registered on the way"
    ai_client = client(tmp_path / "2", voice_server({"message": {"role": "assistant", "content": "Hard to say."}}))
    assert asyncio.run(ai_client.choose_voice("Ayako", vrm)) is None
    assert characters.voice_of(ai_client.db) == ""
