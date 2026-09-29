"""Tomo's body: the window on the desktop, the 3D character in it, her
physics and animation, the chat window and the health programme's lock
screen.

The body runs on the main thread (GLFW needs it) and never waits on the
brain: once a frame it drains the brain's messages (:mod:`tomo.events`) and
turns them into motion and pixels. The modules, in the order a frame uses
them:

* :mod:`.window` — the transparent, always-on-top window over the desktop.
* :mod:`.vrm` — reading a ``.vrm`` file: meshes, skeleton, materials, rig.
* :mod:`.skeleton` — the posed model: where every bone is this frame.
* :mod:`.movement` — her physics: walking, falling, being thrown about.
* :mod:`.animation` — bones and face from what she's doing.
* :mod:`.springs` — hair and clothes swinging with the motion.
* :mod:`.mtoon` and :mod:`.renderer` — drawing her with VRM's toon shading.
* :mod:`.ui`, :mod:`.chat`, :mod:`.workout`, :mod:`.control` — the chat
  window, the lock screen, the mouse/keyboard control badge.
* :mod:`.app` — the loop that ties them together.
"""
