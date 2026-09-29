"""Which language a line is in — English or Bulgarian, the two Tomo speaks.

Bulgarian is written in Cyrillic and English in Latin letters, so the letters
tell them apart, for typed text and for what the speech recogniser heard (it
writes each language in its own script). The AI answers in the user's
language (ai.py) and the voice is picked by the reply's (speech.py).
"""

from __future__ import annotations

from enum import Enum


class Language(Enum):
    ENGLISH = "English"
    BULGARIAN = "Bulgarian"

    @classmethod
    def of(cls, text: str) -> "Language":
        """The language most of ``text``'s letters are written in; English
        when there are none (numbers, emoji)."""
        cyrillic = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        return cls.BULGARIAN if cyrillic > latin else cls.ENGLISH
