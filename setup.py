"""Package setup for wildfire-governance-agentic-ai."""
from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")


def read_requirements(fname: str) -> list:
    lines = (HERE / fname).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith(("#", "-r"))]


setup(
    name="wildfire-governance",
    version="1.0.0",
    author="Ali Akarma, Toqeer Ali Syed, Salman Jan, Hammad Muneer, Abdul Khadar Jilani",
    author_email="443059463@stu.iu.edu.sa",
    description=(
        "Governance-Invariant MDP with Blockchain-Enforced Human Oversight "
        "for Safety-Critical Wildfire Monitoring"
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/akarma-iu/wildfire-governance-agentic-ai",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        "console_scripts": [
            "wildfire-train=wildfire_governance.rl.trainer:main",
            "wildfire-eval=wildfire_governance.rl.evaluator:main",
        ]
    },
)
