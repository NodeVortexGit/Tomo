import sqlite3
from pathlib import Path

import pytest
from vrms import png, tiny_vrm

from tomo import characters
from tomo.config import Config
from tomo.db import Store, ngram_embedding
from tomo.events import Attachment, ChatLine, Role


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "chroma")


def test_prefs_roundtrip_and_upsert(store):
    store.set_pref("name", "Alex")
    store.set_pref("name", "Maria")
    store.set_pref("city", "Sofia")
    assert store.get_pref("name") == "Maria"
    assert store.get_pref("missing") is None
    assert store.all_prefs() == [("city", "Sofia"), ("name", "Maria")]


def test_memories_rank_by_importance(store):
    store.add_memory("fact", "likes tea", 1)
    store.add_memory("fact", "is called Maria", 5)
    store.add_memory("project", "builds Tomo", 3)
    assert [m.content for m in store.top_memories(2)] == ["is called Maria", "builds Tomo"]


def test_memories_are_found_by_words_and_by_near_words(store):
    store.add_memory("fact", "The user's name is Maria", 3)
    store.add_memory("like", "Обича зелен чай", 2)
    store.add_memory("fact", "Drives a blue car", 1)
    assert [m.content for m in store.search_memories("Maria", 5)] == ["The user's name is Maria"]
    # Another word form: "зеления" isn't in the text, but shares most of its
    # n-grams with "зелен". (Very short words — "чая"/"чай" — share too few.)
    assert "Обича зелен чай" in [m.content for m in store.search_memories("зеления", 5)]
    assert store.search_memories("quantum physics", 5) == []


def test_messages_return_in_order(store):
    for i in range(30):
        store.add_message(ChatLine(Role.USER if i % 2 == 0 else Role.ASSISTANT, f"line {i}", 1000 + i))
    recent = store.recent_messages(5)
    assert [m.text for m in recent] == [f"line {i}" for i in range(25, 30)]
    assert recent[0].role == Role.ASSISTANT


def test_only_one_character_is_active(store):
    store.add_character("Tomo", "/c/tomo.vrm")
    store.add_character("Kiyotaka Ayanokōji", "/c/k.vrm")
    store.set_active_character("Tomo")
    store.set_active_character("Kiyotaka Ayanokōji")
    assert store.active_character().name == "Kiyotaka Ayanokōji"
    assert [c.name for c in store.list_characters() if c.is_active] == ["Kiyotaka Ayanokōji"]


def test_messages_keep_their_pictures(store):
    photo = Attachment("C:/data/attachments/1-beach.jpg", "C:/Pictures/beach.jpg", "picture: JPEG 1600×1200")
    pasted = Attachment("C:/data/attachments/2-pasted.jpg", "", "picture: PNG 800×600")
    store.add_message(ChatLine(Role.USER, "what's this?", 1, (photo,)))
    store.add_message(ChatLine(Role.USER, "", 2, (pasted, photo)))  # pictures alone, no words
    store.add_message(ChatLine(Role.ASSISTANT, "A beach.", 3))
    first, second, reply = store.recent_messages(3)
    assert first.attachments == (photo,) and first.text == "what's this?"
    assert second.attachments == (pasted, photo) and second.text == ""
    assert reply.attachments == ()


def test_a_characters_voice_is_remembered(store):
    store.add_character("Ayako", "/c/ayako.vrm")
    store.add_character("Kiyotaka", "/c/k.vrm")
    store.set_active_character("Ayako")
    assert characters.voice_of(store) == "", "not chosen yet"
    assert characters.set_voice(store, "female") == "Ayako", "the showing one, by default"
    assert characters.set_voice(store, "male", "Kiyotaka") == "Kiyotaka"
    assert characters.set_voice(store, "robot") is None and characters.set_voice(store, "male", "Nobody") is None
    # Registered again (moved, say): still active, same voice.
    store.add_character("Ayako", "/d/ayako.vrm")
    assert store.active_character().voice == "female" and characters.voice_of(store) == "female"
    store.set_active_character("Kiyotaka")
    assert characters.voice_of(store) == "male"
    assert {c.name: c.voice for c in store.list_characters()} == {"Ayako": "female", "Kiyotaka": "male"}


def test_a_character_goes_by_its_remembered_name(store, tmp_path):
    path = tmp_path / "k.vrm"
    path.write_bytes(b"glTF")
    store.add_character("Kiyotaka Ayanokōji", str(path))
    assert characters.name_of(store, path) == "Kiyotaka Ayanokōji"
    assert characters.name_of(store, tmp_path / "other one.vrm") == "other one"


def test_a_vrm_tells_its_name_authors_and_picture(tmp_path):
    picture = png()
    facts, thumbnail = characters.vrm_facts(tiny_vrm(tmp_path / "a.vrm", "Ayako", ("Emir",), picture))
    assert facts == {"name": "Ayako", "authors": ["Emir"], "version": ""} and thumbnail == picture
    # VRM 0.x names a texture instead of an image.
    facts, thumbnail = characters.vrm_facts(tiny_vrm(tmp_path / "b.vrm", "Old", ("Kalata",), picture, version=0))
    assert facts["name"] == "Old" and facts["authors"] == ["Kalata"] and thumbnail == picture
    assert characters.vrm_facts(tiny_vrm(tmp_path / "c.vrm", "Plain"))[1] is None
    (tmp_path / "d.vrm").write_bytes(b"not a model at all")
    with pytest.raises(ValueError):
        characters.vrm_facts(tmp_path / "d.vrm")


def test_the_bundled_character_has_a_picture_to_choose_a_voice_by():
    bundled = Path(__file__).resolve().parent.parent / "assets" / "characters" / "female model.vrm"
    facts, thumbnail = characters.vrm_facts(bundled)
    assert facts["name"] == "female model" and thumbnail[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_store_survives_a_reopen(tmp_path):
    first = Store(tmp_path / "chroma")
    first.set_pref("name", "Maria")
    first.add_message(ChatLine(Role.USER, "hello", 1))
    del first
    again = Store(tmp_path / "chroma")
    assert again.get_pref("name") == "Maria"
    assert [m.text for m in again.recent_messages(5)] == ["hello"]


def test_embeddings_are_normalised_and_language_independent():
    a, b, c = ngram_embedding("зелен чай"), ngram_embedding("зеления чай"), ngram_embedding("blue car")
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))  # noqa: E731
    assert abs(dot(a, a) - 1.0) < 1e-6
    assert dot(a, b) > dot(a, c)


def old_sqlite(path):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE preferences (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at INTEGER NOT NULL);
        CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL DEFAULT 'fact',
            content TEXT NOT NULL, importance INTEGER NOT NULL DEFAULT 1, created_at INTEGER NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT NOT NULL, content TEXT NOT NULL,
            at_ms INTEGER NOT NULL);
        CREATE TABLE characters (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
            added_at INTEGER NOT NULL, is_active INTEGER NOT NULL DEFAULT 0);
        INSERT INTO preferences VALUES ('name', 'Maria', 1);
        INSERT INTO memories (kind, content, importance, created_at) VALUES ('fact', 'likes tea', 2, 5);
        INSERT INTO messages (role, content, at_ms) VALUES ('user', 'Hi!', 10), ('assistant', 'Hey there!', 11);
        INSERT INTO characters (name, path, added_at, is_active)
            VALUES ('Kiyotaka Ayanokōji', '\\\\?\\C:\\chars\\k.vrm', 1, 1);
    """)
    con.commit()
    con.close()


def test_the_old_sqlite_memory_moves_over_once(tmp_path, store):
    path = tmp_path / "memory.sqlite3"
    old_sqlite(path)
    assert store.migrate_from_sqlite(path)
    assert store.get_pref("name") == "Maria"
    assert [m.content for m in store.top_memories(5)] == ["likes tea"]
    assert [(m.role, m.text) for m in store.recent_messages(5)] == [(Role.USER, "Hi!"), (Role.ASSISTANT, "Hey there!")]
    active = store.active_character()
    assert active.name == "Kiyotaka Ayanokōji" and active.path == "C:\\chars\\k.vrm"
    # Once only: a second start copies nothing again.
    assert not store.migrate_from_sqlite(path)
    assert len(store.recent_messages(10)) == 2


# ---- characters --------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    bundled = tmp_path / "assets" / "characters"
    bundled.mkdir(parents=True)
    for name in ["Tomo.vrm", "ayako.VRM", "notes.txt"]:
        (bundled / name).write_bytes(b"glTF")
    cfg = Config.for_tests(tmp_path)
    cfg.character_path = bundled / "Tomo.vrm"
    return cfg, Store(tmp_path / "chroma")


def test_it_offers_the_models_in_the_character_folders(world):
    cfg, store = world
    assert [c.name for c in characters.available(store, cfg)] == ["ayako", "Tomo"]


def test_a_choice_is_remembered_and_shown_at_start(world):
    cfg, store = world
    ayako = cfg.assets_dir / "characters" / "ayako.VRM"
    characters.activate(store, cfg, ayako, "ayako")
    assert characters.startup(store, cfg) == ayako.resolve()


def test_a_file_from_elsewhere_is_copied_in(world, tmp_path):
    cfg, store = world
    outside = tmp_path / "Downloads" / "hero.vrm"
    outside.parent.mkdir()
    outside.write_bytes(b"glTF")
    path = characters.activate(store, cfg, outside, "hero")
    assert path.parent == (cfg.data_dir / "characters").resolve()
    assert "hero" in [c.name for c in characters.available(store, cfg)]
