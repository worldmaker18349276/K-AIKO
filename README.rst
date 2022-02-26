.. image:: https://github.com/worldmaker18349276/K-AIKO/raw/master/logo.png

K-AIK▣ is a sound-controlled terminal-based rhythm game.

::

     ⣿⣴⣧⣰⣤⣄ [00000/00032] □   □⛶  □   ■       ■   □   □   ■   ■   □   [  0.9%|00:01]

Install
-------

K-AIKO runs on linux with python 3.9, make sure you have it

::

    python --version

K-AIKO requires PyAudio package, which has external dependencies that pip cannot handle.

You can install PyAudio via apt

::

    sudo apt-get install python3-pyaudio

Or install PortAudio first

::

    sudo apt-get install python-dev portaudio19-dev
    python -m pip install pyaudio

You can install K-AIKO from PyPi

::

    python -m pip install K-AIKO

Or fetch from source directly

::

    git clone git@github.com:worldmaker18349276/K-AIKO
    cd K-AIKO
    python -m pip install .

Now you can play K-AIKO

::

    kaiko

Or

::

    python -m kaiko

License
-------

MIT
