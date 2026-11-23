"""Replies as they're written: the model's answer streamed into the chat,
and spoken a piece at a time."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from test_ai import client

from tomo import speech
from tomo.ai import streamable
from tomo.body.chat import ChatState
from tomo.brain import LiveReply
from tomo.config import LlmProvider
from tomo.events import Chat, ChatDraft, ChatLine, Role, Transcript


class StreamServer:
    """Answers by path, in order, like test_ai.FakeServer — but a reply that's
    a list is sent streamed: Ollama's lines of JSON, or OpenAI-style events."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.content:
            self.seen.append(json.loads(request.content))
        path = request.url.path
        index = next(i for i, (p, _) in enumerate(self.replies) if path.endswith(p))
        body = self.replies.pop(index)[1] if sum(p == path or path.endswith(p) for p, _ in self.replies) > 1 \
            else self.replies[index][1]
        if not isinstance(body, list):
            return httpx.Response(200, json=body)
        if path.endswith("/api/chat"):
            return httpx.Response(200, content="\n".join(json.dumps(c) for c in body).encode(),
                                  headers={"content-type": "application/x-ndjson"})
        events = "".join(f"data: {json.dumps(c)}\n\n" for c in body) + "data: [DONE]\n\n"
        return httpx.Response(200, content=events.encode(), headers={"content-type": "text/event-stream"})


def written(ai_client, text):
    seen, events = [], []
    reply = asyncio.run(ai_client.respond(text, events.append, on_text=lambda t, speak: seen.append((t, speak))))
    return reply, seen, events


TAGS = ("/api/tags", {"models": [{"name": "qwen2.5:7b", "details": {"parameter_size": "7.6B"}}]})


def test_an_ollama_reply_streams_and_a_try_that_turns_to_tools_is_withdrawn(tmp_path):
    server = StreamServer([
        TAGS,
        ("/api/chat", [{"message": {"content": "Let"}}, {"message": {"content": " me check"}},
                       {"message": {"tool_calls": [{"function": {"name": "remember",
                                                                  "arguments": {"content": "likes tea"}}}]},
                        "done": True}]),
        ("/api/chat", [{"message": {"content": "Hello! "}}, {"message": {"content": "You like tea."}},
                       {"message": {"content": " Bye"}, "done": True}]),
    ])
    reply, seen, _ = written(client(tmp_path, server), "remember I like tea")
    assert reply == "Hello! You like tea. Bye"
    assert all(body["stream"] for body in server.seen)
    withdrawn = seen.index(("", False))
    assert seen[withdrawn - 1] == ("Let me check", False), "a promise isn't spoken while it's written"
    assert seen[withdrawn + 1:] == [("Hello!", True), ("Hello! You like tea.", True), ("Hello! You like tea. Bye", True)]


def test_lm_studio_events_stream_and_its_tool_calls_come_in_pieces(tmp_path):
    server = StreamServer([
        ("/v1/models", {"data": [{"id": "qwen2.5-7b-instruct"}]}),
        ("/v1/chat/completions", [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "remem"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "ber", "arguments": "{\"con"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "tent\": \"tea\"}"}}]}}]},
        ]),
        ("/v1/chat/completions", [{"choices": [{"delta": {"content": "Got it"}}]},
                                  {"choices": [{"delta": {"content": ", tea it is."}}]}]),
    ])
    ai_client = client(tmp_path, server, LlmProvider.LM_STUDIO)
    reply, seen, _ = written(ai_client, "remember tea")
    assert reply == "Got it, tea it is."
    assert any("tea" in m.content for m in ai_client.db.top_memories(5)), "the call, put together, ran"
    assert seen[-1] == ("Got it, tea it is.", True)


def test_while_every_action_has_failed_the_reply_shows_but_waits_to_be_spoken(tmp_path):
    server = StreamServer([
        TAGS,
        ("/api/chat", [{"message": {"tool_calls": [{"function": {"name": "execute_command",
                                                                  "arguments": {"command": "echo hi"}}}]},
                        "done": True}]),
        ("/api/chat", [{"message": {"content": "That didn't work, sorry."}, "done": True}]),
    ])
    reply, seen, _ = written(client(tmp_path, server), "run echo hi")  # commands are off in the tests
    assert reply == "That didn't work, sorry."
    assert seen[-1] == ("That didn't work, sorry.", False)


def test_a_server_that_answers_whole_is_read_whole(tmp_path):
    server = StreamServer([TAGS, ("/api/chat", {"message": {"content": "All at once."}, "done": True})])
    reply, seen, _ = written(client(tmp_path, server), "hi")
    assert reply == "All at once." and seen == [("All at once.", True)]


def test_what_might_be_a_tool_call_or_thinking_isnt_shown_while_written():
    assert streamable("  Hello there ") == "Hello there"
    assert streamable("") is None and streamable("   ") is None
    for held in ("<tool_call>{\"name\"", "{\"name\": \"open_app\"", "<think>hm", "Here:\n```bash\nls"):
        assert streamable(held) is None


@pytest.mark.parametrize("text, start, first, pieces", [
    ("Hello! How are you today? I am", 0, False, ["Hello!", " How are you today?"]),
    ("It's 3.5 degrees today. And", 0, False, ["It's 3.5 degrees today."]),
    ("Short one, ok", 0, True, []),
    ("Line one\nLine two\n", 0, False, ["Line one\n", "Line two\n"]),
    ("He said \"Hi.\" Then left. ", 0, False, ["He said \"Hi.\"", " Then left."]),
    ("Sure, I can do that for you, and I will open the calculator right after the volume, then", 0, True,
     ["Sure, I can do that for you,"]),
    ("Готово, звукът е на 40%. Вече седиш от 2 часа", 0, False, ["Готово, звукът е на 40%."]),
    ("Done. Next one. Third", 5, False, [" Next one."]),
])
def test_a_reply_is_spoken_in_pieces_that_end_where_sentences_do(text, start, first, pieces):
    ends = speech.speech_ends(text, start, first)
    got, at = [], start
    for end in ends:
        got.append(text[at:end])
        at = end
    assert got == pieces


class FakeStream:
    def __init__(self, speaking=None, heard=None):
        self.pieces, self.tags, self.closed, self.failed = [], [], False, None
        self.speaking, self.heard = speaking, heard

    def add(self, text, tag=None):
        self.pieces.append(text)
        self.tags.append(tag)

    def close(self):
        self.closed = True

    async def done(self):
        pass


class FakeSpeech:
    def __init__(self):
        self.streams = []

    def stream(self, speaking, heard=None):
        self.streams.append(FakeStream(speaking, heard))
        return self.streams[-1]

    def first_failure_note(self, error):
        return None


def test_a_live_reply_shows_each_piece_once_its_done_and_speaks_it_once():
    events, voice = [], FakeSpeech()

    async def run():
        live = LiveReply(events.append, voice, None)
        live.text("Let me", False)  # held, and no piece complete yet
        live.text("", False)  # that try wasn't the answer
        live.text("Hello! How", True)
        await asyncio.sleep(0)
        live.text("Hello! How are you? Fine", True)
        await asyncio.sleep(0)
        kept = await live.finish("Hello! How are you? Fine thanks.")
        live.end_voice()
        return kept

    kept = asyncio.run(run())
    stream, = voice.streams
    assert kept == "Hello! How are you? Fine thanks."
    assert stream.pieces == ["Hello!", " How are you?", " Fine thanks."] and stream.closed
    assert [e.text for e in events if isinstance(e, ChatDraft)] == [
        "", "Hello!", "Hello! How are you?", "Hello! How are you? Fine thanks."]
    assert stream.tags == [(1, 6), (1, 19), (1, 32)], "each piece: its try, and where it ends"


def test_a_reply_that_was_held_is_spoken_once_its_the_answer():
    voice = FakeSpeech()

    async def run():
        live = LiveReply(lambda e: None, voice, None)
        live.text("That didn't work. Sorry", False)
        await asyncio.sleep(0)
        assert voice.streams == [], "held: not a word yet"
        await live.finish("That didn't work. Sorry.")
        live.end_voice()

    asyncio.run(run())
    assert voice.streams[0].pieces == ["That didn't work.", " Sorry."]


def test_with_the_voice_off_the_draft_still_shows():
    events = []

    async def run():
        live = LiveReply(events.append, None, None)
        live.text("Hi there.", True)
        await live.finish("Hi there.")
        live.end_voice()

    asyncio.run(run())
    assert events == [ChatDraft("Hi there.")]


class FakeGrammar:
    """Proofreading: "грешка" becomes "поправка", slowly."""

    def __init__(self):
        self.seen = []

    async def fix(self, piece):
        self.seen.append(piece)
        await asyncio.sleep(0.01)
        return piece.replace("грешка", "поправка")


def test_each_piece_is_proofread_before_its_shown_or_spoken_and_the_reply_kept_is_proofread():
    events, voice, grammar = [], FakeSpeech(), FakeGrammar()

    async def run():
        live = LiveReply(events.append, voice, None, grammar)
        live.text("Тук има грешка. Още", True)
        await asyncio.sleep(0)
        assert [e for e in events if isinstance(e, ChatDraft)] == [], "not shown before it's proofread"
        await asyncio.sleep(0.05)
        kept = await live.finish("Тук има грешка. Още една грешка.")
        live.end_voice()
        return kept

    kept = asyncio.run(run())
    assert kept == "Тук има поправка. Още една поправка."
    assert grammar.seen == ["Тук има грешка.", " Още една грешка."]
    assert voice.streams[0].pieces == ["Тук има поправка.", " Още една поправка."]
    assert [e.text for e in events if isinstance(e, ChatDraft)] == ["Тук има поправка.", kept]


def test_a_piece_of_a_dropped_try_isnt_shown_even_if_its_proofreading_ends_later():
    events, grammar = [], FakeGrammar()

    async def run():
        live = LiveReply(events.append, None, None, grammar)
        live.text("Ще отворя грешка. Да", False)
        live.text("", False)  # dropped while that piece is being proofread
        await asyncio.sleep(0.05)
        return await live.finish("Готово.")

    assert asyncio.run(run()) == "Готово."
    assert [e.text for e in events if isinstance(e, ChatDraft)] == ["", "Готово."]


class FakeVoice:
    """Speech for SpeechStream: a file per piece, made slowly."""

    def __init__(self, folder):
        self.folder, self.made = folder, []

    def voice_for(self, text):
        return "bg" if any("а" <= c <= "я" for c in text.lower()) else "en"

    async def synthesize(self, text, voice=None):
        await asyncio.sleep(0.01)
        if "boom" in text:
            raise RuntimeError("it went boom")
        self.made.append((text, voice))
        path = Path(self.folder) / f"{len(self.made)}.wav"
        path.write_text(text, encoding="utf-8")
        return path


class FakePlayer:
    played: list = []

    def __init__(self):
        self.closed = self.aborted = False

    def play(self, path):
        FakePlayer.played.append(Path(path).read_text(encoding="utf-8"))

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


def test_a_speech_stream_makes_pieces_in_order_in_one_voice_and_plays_them_back_to_back(tmp_path):
    FakePlayer.played = []
    said = []

    async def run():
        voice = FakeVoice(tmp_path)
        stream = speech.SpeechStream(voice, said.append, FakePlayer)
        stream.add("Hello there.")
        stream.add(" Как си?")  # the reply's voice stays the first piece's
        stream.add("**boom**")
        stream.add(" Bye.")
        stream.close()
        await stream.done()
        return voice, stream

    voice, stream = asyncio.run(run())
    assert voice.made == [("Hello there.", "en"), ("Как си?", "en"), ("Bye.", "en")]
    assert FakePlayer.played == ["Hello there.", "Как си?", "Bye."]
    assert said == [True, False] and stream.player.closed
    assert str(stream.failed) == "it went boom", "a piece that fails is skipped, and remembered"
    assert not list(tmp_path.glob("*.wav")), "each file goes once it's played"


def test_a_new_replys_speech_stops_the_one_still_talking(tmp_path):
    from tomo.config import Config

    async def run():
        s = speech.Speech(Config.for_tests(tmp_path))
        first = s.stream(lambda on: None)
        second = s.stream(lambda on: None)
        await asyncio.sleep(0)
        return first, second

    first, second = asyncio.run(run())
    assert first.player.stopped and all(t.cancelled() or t.done() for t in first.tasks)
    assert not second.player.stopped


def test_the_chat_shows_the_draft_until_the_reply_takes_its_place():
    chat = ChatState()
    chat.receive(ChatDraft("Hello"), None)
    assert chat.draft == "Hello" and chat.scroll_to_end
    chat.receive(Chat(ChatLine(Role.SYSTEM, "a note")), None)
    assert chat.draft == "Hello", "a note doesn't end it"
    chat.receive(Chat(ChatLine(Role.ASSISTANT, "Hello there.")), None)
    assert chat.draft == "" and chat.transcript[-1].text == "Hello there."
    chat.receive(ChatDraft("Another"), None)
    chat.receive(Transcript(()), None)
    assert chat.draft == ""


def test_every_kind_of_news_from_the_brain_reaches_the_body():
    """The app hands each BrainToUi message on by its type: one left out of
    dispatch() is dropped without a word (as ChatDraft and ChatSize were)."""
    import inspect
    import typing

    from tomo import events
    from tomo.body import app

    source = inspect.getsource(app.App.dispatch)
    missing = [kind.__name__ for kind in typing.get_args(events.BrainToUi) if f"events.{kind.__name__}" not in source]
    assert missing == []


def test_tomos_reply_appears_a_word_at_a_time_and_catches_up_with_whats_written():
    from tomo.body.chat import REVEAL_RATE

    chat = ChatState()
    chat.receive(ChatDraft("Cats can make"), None)
    assert chat.visible(chat.draft) == "", "nothing until the first word is reached"
    assert chat.advance(0.1, None), "still appearing: the window keeps drawing"
    assert chat.visible(chat.draft) == "Cats", "seven characters out: the words reached, whole"
    chat.advance(0.03, None)
    assert chat.visible(chat.draft) == "Cats can"
    chat.receive(ChatDraft("Cats can make over a hundred sounds."), None)
    for _ in range(10):
        chat.advance(0.05, None)
    shown = chat.visible(chat.draft)
    assert shown.startswith("Cats can make over") and not shown.endswith(" "), "whole words only"
    # The reply carries on from where the draft got to.
    before = chat.revealed
    chat.receive(Chat(ChatLine(Role.ASSISTANT, "Cats can make over a hundred sounds.")), None)
    assert chat.draft == "" and chat.revealing == 0 and chat.revealed == before
    while chat.advance(0.05, None):
        pass
    assert chat.revealing is None, "all of it is out, and it can be selected again"
    # A lot waiting comes out faster: 1,400 characters at once take seconds, not the 20 the steady pace would.
    long = ChatLine(Role.ASSISTANT, "word " * 280)
    chat.receive(Chat(long), None)
    steps = 0
    while chat.advance(0.05, None):
        steps += 1
    assert steps * 0.05 < 6.0 < len(long.text) / REVEAL_RATE


def test_a_dropped_try_starts_the_next_draft_from_nothing_and_other_lines_appear_at_once():
    chat = ChatState()
    chat.receive(ChatDraft("Let me check that"), None)
    for _ in range(5):
        chat.advance(0.05, None)
    chat.receive(ChatDraft(""), None)
    assert chat.revealed == 0.0
    chat.receive(Chat(ChatLine(Role.USER, "hello")), None)
    chat.receive(Chat(ChatLine(Role.SYSTEM, "a note")), None)
    assert chat.revealing is None, "only Tomo's replies appear word by word"
    chat.receive(Chat(ChatLine(Role.ASSISTANT, "Different words entirely.")), None)
    assert chat.revealing == 2 and chat.revealed == 0.0


def test_with_the_voice_on_the_words_come_out_as_they_are_spoken():
    from tomo.events import RevealTo

    events, voice = [], FakeSpeech()

    async def run():
        live = LiveReply(events.append, voice, None)
        live.text("Hello there. How", True)  # held from the first word…
        await asyncio.sleep(0)
        stream = voice.streams[0]
        stream.heard((0, 12), 0.8)  # …"Hello there." starts to sound
        await live.finish("Hello there. How are you?")
        live.end_voice()
        stream.heard((0, 25), 1.2)
        stream.speaking(True)
        stream.speaking(False)  # done: whatever's left comes out

    asyncio.run(run())
    reveals = [e for e in events if isinstance(e, RevealTo)]
    assert reveals == [RevealTo(0), RevealTo(12, 0.8), RevealTo(25, 1.2), RevealTo(None)]


def test_a_reply_with_nothing_to_say_isnt_held_and_an_old_one_cant_let_go_of_a_new_one():
    from tomo.events import RevealTo

    events, voice = [], FakeSpeech()

    async def run():
        old = LiveReply(events.append, voice, None)
        old.text("First reply. More", True)
        await asyncio.sleep(0)
        old_stream = voice.streams[0]
        new = LiveReply(events.append, voice, None)  # the next turn
        new.text("Ok 👍", True)
        old_stream.heard((0, 12), 1.0)  # the old one, still talking: not the new one's words
        old_stream.speaking(False)  # the old one stops: it mustn't let the new one's words out
        await new.finish("Ok 👍")
        new.end_voice()

    asyncio.run(run())
    reveals = [e for e in events if isinstance(e, RevealTo)]
    assert reveals[:2] == [RevealTo(0), RevealTo(0)], "neither the old voice nor its end touch the new reply"
    assert RevealTo(12, 1.0) not in reveals


def test_a_reply_with_no_voice_at_all_lets_its_words_out_at_once():
    from tomo.events import RevealTo

    events, voice = [], FakeSpeech()

    async def run():
        live = LiveReply(events.append, voice, None)
        live.text("👍", True)
        await live.finish("👍")
        live.end_voice()

    asyncio.run(run())
    assert [e for e in events if isinstance(e, RevealTo)] == [RevealTo(0), RevealTo(None)]
    assert voice.streams == []


def test_the_chat_follows_the_voice_word_for_word():
    from tomo.events import RevealTo

    chat = ChatState()
    chat.receive(Chat(ChatLine(Role.USER, "hi")), None)
    chat.receive(RevealTo(0), None)
    chat.receive(ChatDraft("Hello there. How are you?"), None)
    chat.advance(0.5, None)
    assert chat.visible(chat.draft) == "", "nothing before the voice starts"
    chat.receive(RevealTo(12, 1.0), None)  # "Hello there." takes a second to say
    chat.advance(0.5, None)
    assert chat.visible(chat.draft) == "Hello", "halfway through the second, halfway through the words"
    chat.advance(0.6, None)
    assert chat.visible(chat.draft) == "Hello there.", "and no further than the voice"
    chat.receive(RevealTo(None), None)
    while chat.advance(0.05, None):
        pass
    assert chat.visible(chat.draft) == "Hello there. How are you?"
    # A new turn doesn't wait for the last one's voice.
    chat.receive(Chat(ChatLine(Role.USER, "again")), None)
    assert chat.allowed is None


def test_a_speech_stream_says_when_each_piece_starts_and_how_long_it_lasts(tmp_path):
    import wave

    heard = []

    class WavVoice(FakeVoice):
        async def synthesize(self, text, voice=None):
            if "boom" in text:
                raise RuntimeError("boom")
            self.made.append((text, voice))
            path = Path(self.folder) / f"{len(self.made)}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(1000)
                w.writeframes(b"\x00\x00" * 500 * len(self.made))
            return path

    class SilentPlayer(FakePlayer):
        def play(self, path):
            pass

    async def run():
        stream = speech.SpeechStream(WavVoice(tmp_path), lambda on: None, SilentPlayer,
                                     heard=lambda tag, seconds: heard.append((tag, seconds)))
        stream.add("One.", "a")
        stream.add("boom", "b")
        stream.add("Two.", "c")
        stream.close()
        await stream.done()

    asyncio.run(run())
    assert heard == [("a", 0.5), ("b", 0.0), ("c", 1.0)]
