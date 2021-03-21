from setuptools import setup

exec(open("kaiko/_version.py").read())

setup(
    name="K-AIKO",
    version=__version__,
    description="A voice-controlled terminal-based rhythm game",
    long_description=open("README.rst").read(),
    url="https://github.com/worldmaker18349276/K-AIKO",
    author="worldmaker18349276",
    author_email="worldmaker18349276@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment",
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords=["voice-controlled", "terminal-based", "rhythm game"],
    packages=["kaiko"],
    python_requires=">=3.6, <4",
    install_requires=[
        'dataclasses; python_version < "3.7"',
        "lark",
        "numpy",
        "scipy",
        "audioread",
        "pyaudio",
        "wcwidth",
        "psutil",
        "appdirs"
    ],
    entry_points={
        "console_scripts": [
            "kaiko = kaiko.__main__:main"
        ]
    },
)
