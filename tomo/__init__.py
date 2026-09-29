"""Tomo — a local AI VRM desktop companion for Windows and Linux.

The package is split the way the program is:

* the "brain" (this package's top-level modules): settings, memory
  (ChromaDB), the local model's tool-use loop, the safe command executor,
  speech, voice input and the health programme. No graphics — it is tested
  on its own.
* the "body" (:mod:`tomo.body`): the window, the VRM character's rendering,
  physics, animation and the chat window. It talks to the brain only through
  the message queues in :mod:`tomo.events`.
"""

__version__ = "0.2.0"
