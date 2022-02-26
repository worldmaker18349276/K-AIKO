from setuptools import setup, find_packages

exec(open("kaiko/__init__.py").read())

setup(
    name="K-AIKO",
    version=__version__,
    description="A sound-controlled terminal-based rhythm game",
    long_description=open("README.rst", "r").read(),
    long_description_content_type="text/x-rst",
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
        "Programming Language :: Python :: 3.9",
    ],
    keywords=[
        "sound-controlled",
        "terminal-based",
        "rhythm game",
    ],
    packages=find_packages(),
    python_requires=">=3.9, <4",
    install_requires=[
        "numpy",
        "scipy",
        "audioread",
        "pyaudio",
        "wcwidth",
        "appdirs",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "kaiko = kaiko.__main__:main",
        ],
    },
    project_urls={
        "Issue Tracker": "https://github.com/worldmaker18349276/K-AIKO/issues",
    },
)
