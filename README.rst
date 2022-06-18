=====================================================
K-AIKO: A sound-controlled terminal-based rhythm game
=====================================================

.. image:: https://github.com/worldmaker18349276/K-AIKO/raw/master/screenshot.png

Instruction
===========

K-AIKO runs on linux with python 3.9, make sure you have it.

You can check your python version with the following command:

::

    python --version

K-AIKO requires PyAudio package, which has external dependencies that pip cannot handle.

You can install PyAudio via apt:

::

    sudo apt-get install python3-pyaudio

If it fails, install PortAudio first:

::

    sudo apt-get install python-dev portaudio19-dev
    python -m pip install pyaudio

Then install K-AIKO from PyPi:

::

    python -m pip install K-AIKO

Now you can play K-AIKO, just run:

::

    kaiko

Have fun!

License
=======

MIT
