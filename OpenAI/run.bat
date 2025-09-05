@echo off
if not exist .\OpenAI_Tests_09.py (
    jupyter nbconvert --to script .\OpenAI.ipynb
)
python .\OpenAI_Tests_09.py