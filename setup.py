from setuptools import setup

long_description = open("README.md", "r").read()

setup(
    name="K-AIKO",
    version="0.1.0",
    description="A voice-controlled terminal-based rhythm game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/worldmaker18349276/K-AIKO",
    author="worldmaker18349276",
    author_email="worldmaker18349276@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    keywords="voice-controlled, terminal-based, rhythm game",
    packages=".",
    python_requires=">=3.6, <4",
    install_requires=["dataclasses", "lark", "numpy", "scipy", "audioread", "pyaudio", "wcwidth"],
)
