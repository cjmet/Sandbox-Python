#!/usr/bin/env python
# coding: utf-8

# # In-Class Thurs 25/09/04 

from OpenAI import generate_response

model = "gpt-5-nano"
while True:
    prompt = input(f"{model}: ")
    prompt = prompt.strip()
    if prompt.lower() in ['exit', 'quit', '']:
        break
    elif prompt.startswith("model:"):
        model = prompt.split("model:")[1].strip()
        print(f"\tSwitched to model: {model}\n")
    else:
        response = generate_response(
                    model=model, 
                    prompt=prompt, 
                    )
        text = response['text'].strip()
        ## add a tab before each line of the response
        text = "\n".join(f"\t{line}" for line in text.split("\n"))
        print(f"{text}\n")
list
