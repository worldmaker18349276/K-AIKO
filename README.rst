.. image:: https://github.com/worldmaker18349276/K-AIKO/raw/master/logo.png

K-AIK▣ is a sound-controlled terminal-based rhythm game.

::

     ⣿⣴⣧⣰⣤⣄ [00000/00032] □   □⛶  □   ■       ■   □   □   ■   ■   □   [  0.9%|00:01]

Getting Started
---------------

Install PyAudio
~~~~~~~~~~~~~~~

Installing PyAudio via pip may encounter some problems, which is caused by the external dependency of PyAudio.
Our recommendation is to use `Anaconda <https://www.anaconda.com/products/individual>`__, it can solve all problems.
After installing Anaconda, just run

::

    conda install -c anaconda pyaudio

If you have issue (no sound or error while playing), see `next section <#install-pyaudio-without-using-conda>`__.

Install K-AIKO
~~~~~~~~~~~~~~

You can install it from PyPi

::

    python -m pip install K-AIKO

Or fetch from source directly

::

    git clone git@github.com:worldmaker18349276/K-AIKO
    cd K-AIKO
    python -m pip install .

Play
~~~~

Now you can play K-AIKO

::

    kaiko

Or

::

    python -m kaiko

Install PyAudio Without Using Conda
-----------------------------------

The PortAudio provided by Annaconda conda-forge channel seems have some issues, see `<https://github.com/ContinuumIO/anaconda-issues/issues/4139>`__.
The solution is simple: don't use Annaconda.
Before install PyAudio, you should uninstall them in conda

::

    conda remove pyaudio

Linux
~~~~~

Install PyAudio via apt

::

    sudo apt-get install python3-pyaudio

Or install PortAudio first

::

    sudo apt-get install python-dev portaudio19-dev
    python -m pip install pyaudio

(see https://stackoverflow.com/a/61993070/3996613)

Mac
~~~

Install portaudio using homebrew (or method of your choice)

::

    brew install portaudio

Create ``$HOME/.pydistutils.cfg`` using the include and lib directories of your portaudio install

::

    [build_ext]
    include_dirs=/Users/jrobert1271/homebrew/Cellar/portaudio/19.20140130/include/
    library_dirs=/Users/jrobert1271/homebrew/Cellar/portaudio/19.20140130/lib/

Then in your virtualenv

::

    pip install --allow-external pyaudio --allow-unverified pyaudio pyaudio

(see https://stackoverflow.com/a/62091426/3996613)

Windows
~~~~~~~

Download the wheel on this site https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio.

Choose ``PyAudio‑*‑win32.whl`` (the newest one) if you use 32 bit, or ``PyAudio‑*‑win_amd64.whl`` for 64 bit. Then go to your download folder

::

    cd <your donwload path>

Then, install by wheel

::

    python -m pip install <PyAudio's wheel file>

(see https://stackoverflow.com/a/54999645/3996613)

Tests
-----

You can prepare environment for testing

::

    git clone git@github.com:worldmaker18349276/K-AIKO
    cd K-AIKO
    conda env create -f environment.yml --prefix ./envs

Remember to activate environment before testing

::

    conda activate ./envs
    python -m kaiko

Publish

::

    python -m build --sdist --wheel .
    twine upload dist/*
    git push origin master --tags

Compatibilities
---------------

It is compatible to linux system with VT100 terminal emulator.

Tested terminals:

-  GNOME terminal (Linux)

License
-------

MIT
