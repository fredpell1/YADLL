from setuptools import setup


setup(
    name="yadll",
    description="Yet Another Deep Learning Library",
    version="0.1",
    author="Frederic Pelletier",
    license="MIT",
    packages=["yadll", "yadll.nn"],
    install_requires=["numpy", "scikit-image"],
    python_requires=">=3.9",
    extras_require={
        'testing': ['torch', 'pytest'],
        "formatting": ["ruff", "pre-commit"]
    },
    include_package_data=True
)
