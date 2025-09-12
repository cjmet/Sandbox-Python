#!/usr/bin/env python
# coding: utf-8

# In[35]:


import os
import time
from openai import OpenAI

## Global Options
default_model = "gpt-5-nano" 
minimum_tokens = 2500   
rate_limit_seconds = 3  
default_debug = False   

# def generate_response(prompt, model=None, temperature=None, max_tokens=None, top_p=None, 
#                       seed=None, effort="minimal", use_completions=False, service_tier="flex", 
#                       previous_response_id=None, prompt_instructions=None, debug=None):
#    return {"text": response.output_text, "id": response.id, "model": model, "service_tier": service_tier, "usage": response.usage}


# In[36]:


# -------------------------------------------------------
rate_limit_timeout = 0  # When the rate limit will expire

def get_global_settings_string(): 
     if not os.environ.get("OPENAI_API_KEY"):
          raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to your API key.")
     return f"{default_model} {minimum_tokens}tk {rate_limit_seconds}s { 'debug' if default_debug else '' }"

def divider(): 
    print("-" * 50)


# In[37]:


import math
def display_cost_human(cost): # GenAI
    """
    Display cost in human-readable format:
    - If cost >= $1, show as $0.00
    - If cost >= $0.01 and < $1, show as 0.00¢
    - If cost < $0.01, show as fractional cents: 1/100¢, 1/1,000¢, etc.
    """
    if cost >= 0.994:
        return f"${cost:,.2f}"
    elif cost > 0.001:
        cost = math.ceil(cost * 100) / 100  # Round up to nearest cent
        return f"{cost * 100:,.0f}¢"
    else:
        # Find denominator for fractional cent
        denominator = 1000
        while cost < 1 / denominator:
            denominator *= 10
        denominator //= 100  # Adjust to the last valid denominator
        return f"1/{denominator:,}¢"


# In[38]:


def cost_of_openai_api_call(model, input_tokens, output_tokens, flex=False):
    # Token cost rates per million
    token_cost = {
        "gpt-5": (1.25, 10),
        "gpt-5-chat-latest": (1.25, 10),
        "gpt-5-mini": (0.25, 2),
        "gpt-5-nano": (0.05, 0.4),
        "gpt-4.1": (2.00, 8),
        "gpt-4.1-mini": (0.4, 1.6),
        "gpt-4.1-nano": (0.1, 0.4),
        "o3-deep-research-2025-06-26": (10, 40),
        "o4-mini-deep-research-2025-06-26": (2, 8)
    }

    if model not in token_cost:
        raise ValueError(f"Unknown model: {model}")

    # Calculate cost
    input_cost = input_tokens * token_cost[model][0]  # Input token cost
    output_cost = output_tokens * token_cost[model][1]  # Output token cost
    cost = input_cost + output_cost
    if flex: cost *= 0.5  # flex discount
    cost /= 1000000 # prices are per million

    return display_cost_human(cost)


# In[39]:


def cost_of_responses_api(response_dict):
    if response_dict == None or response_dict['usage'] == None: return ValueError
    model = response_dict['model']
    service_tier = response_dict['service_tier']
    return (
        f"{model}: {response_dict['usage'].input_tokens} + {response_dict['usage'].output_tokens} = {response_dict['usage'].total_tokens}\t"
        + cost_of_openai_api_call(model, response_dict['usage'].input_tokens, response_dict['usage'].output_tokens, service_tier)
    )


# In[40]:


def cost_of_completions_api(response_dict):
    if response_dict == None or response_dict.usage == None: return ValueError
    model = response_dict.model
    service_tier = response_dict.service_tier
    return (
        f"Usage: {response_dict.usage.prompt_tokens} + {response_dict.usage.completion_tokens} = {response_dict.usage.total_tokens}\t"
        + cost_of_OPENAI_API_KEY_call(model, response_dict.usage.prompt_tokens, response_dict.usage.completion_tokens, service_tier)
    )


# In[41]:


if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to your API key.")

# -------------------------------------------------
# Function to generate a response 

## Models to test with gpt-5, gpt-5-chat-latest (chatty), gpt-5-nano (cheap reasoning, no temperature)
# seed=seed  # Not supported with client.responses.create

def generate_response(prompt, model=None, temperature=None, max_tokens=None, top_p=None, 
                      seed=None, effort="minimal", use_completions=False, service_tier="flex", 
                      previous_response_id=None, prompt_instructions=None, verbose=None, debug=None):
    # Configurable Defaults
    if model == None: model = default_model  # Use default model if not specified
    if model not in ["gpt-5", "gpt-5-chat-latest", "gpt-5-mini", "gpt-5-nano", "gpt-4.1", "gpt-4.1-mini" ,"gpt-4.1-nano"]:
        raise ValueError(f"Model {model} is not supported.") 
    if max_tokens == None: max_tokens = minimum_tokens  # Use minimum tokens if not specified  
    if max_tokens <= 0 : max_tokens = None # Use None for unlimited tokens
    if verbose == None: verbose = default_debug  # Use default debug setting if not specified
    if debug : verbose = True

    # NOT supported on gpt-5, convert gpt-4.1
    if temperature != None or top_p != None or seed != None:
        model = model.replace("5", "4.1")  # Use gpt-4.1 for completions
        model = model.replace("-chat-latest", "") 
    # NOT Supported on Responses API
    if seed != None:
        use_completions = True
    # NOT Supported on Completions API
    if use_completions: 
        effort = None 
    # NOT Supported on gpt-4.1
    if model.startswith("gpt-4.1"):
        effort = None
        service_tier = "default"
    # NOT Supported on nano - 
    if model.endswith("nano"):
        service_tier = "default" # This is supposed to be supported on 5-nano, but does not appear to be functional today.

    reasoning={"effort": effort} if effort != None else None


    # DEBUG Message
    if verbose:
        message = f"{model} " + f"{'C' if use_completions else 'R'} " + f"{max_tokens}tk "
        if temperature != None: message += f"{temperature}° "
        if top_p != None: message += f"{top_p}p "
        if seed != None: message += f"{seed}r "
        if effort != None: message += f"{effort} "
        if service_tier: message += f"{service_tier} "
        print(message)
    # /message


    # Rate limit
    global rate_limit_seconds
    global rate_limit_timeout
    now = time.time()
    dif = rate_limit_timeout - now   

    # Spinner 
    if (dif > 0):
        # Spinner Countdown   
        while dif > 0:
            print(f"\rRate limit {dif:5.3f}s", end='', flush=True)
            time.sleep(0.25)  # Update spinner every 0.25 seconds
            dif = rate_limit_timeout - time.time()
        print("\r                         \r", end='', flush=True)
        # /Spinner 

    # Set the Rate Limit    
    rate_limit_timeout = time.time() + rate_limit_seconds


    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    )

    try:
        if use_completions:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                seed=seed,
                service_tier=service_tier,
                safety_identifier="test_user",
                prompt_cache_key="test_user",
                # reasoning=reasoning,
                previous_response_id=previous_response_id,
                instructions=prompt_instructions,
            )
        else:
            response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,  
            # seed=seed,
            service_tier=service_tier,  
            safety_identifier="test_user",
            prompt_cache_key="test_user",
            reasoning=reasoning,
            previous_response_id=previous_response_id,
            instructions=prompt_instructions,
            )
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

    if verbose and debug:
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

    if use_completions:
        if verbose: print(f"Usage: {response.usage.prompt_tokens} + {response.usage.completion_tokens} = {response.usage.total_tokens}\t", 
                        cost_of_openai_api_call(model, response.usage.prompt_tokens, response.usage.completion_tokens, service_tier))
        return {"text": response.choices[0].message.content, "id": response.id, "model": model, "service_tier": service_tier, "usage": response.usage}
    else:
        if verbose: print(f"Usage: {response.usage.input_tokens} + {response.usage.output_tokens} = {response.usage.total_tokens}\t", 
                        cost_of_openai_api_call(model, response.usage.input_tokens, response.usage.output_tokens, service_tier))
        return {"text": response.output_text, "id": response.id, "model": model, "service_tier": service_tier, "usage": response.usage}

# /------------------------------------------------


# In[ ]:


if False:                        # True or False to TEST
    test_prompt = "Hello."
    response = generate_response(test_prompt, verbose=True)
    print(f"Prompt: {test_prompt}\nResponse: {response}\n")

