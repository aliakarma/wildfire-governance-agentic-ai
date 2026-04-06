from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

def read_requirements(fname):
    lines = (HERE / fname).read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith(("#", "-r"))]

setup(name="wildfire-governance", version="1.0.0",
    author="Ali Akarma, Toqeer Ali Syed, Salman Jan, Hammad Muneer, Abdul Khadar Jilani",
    author_email="443059463@stu.iu.edu.sa",
    description="Governance-Invariant MDP with Blockchain-Enforced Human Oversight",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/aliakarma/wildfire-governance-agentic-ai",
    license="MIT", package_dir={"": "src"}, packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": read_requirements("requirements-dev.txt")})
