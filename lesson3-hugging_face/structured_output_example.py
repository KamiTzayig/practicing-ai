import json
import torch
import re
import ast # Added for sanitizing LLM's JSON-like output
from datetime import date, timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_REPO_ID = "KamiTzayig/llama-3.2-1b-hermes-fc-adapter-colab"
# This should be the path to the directory where your adapter was saved by SFTTrainer
# Set to None or an empty string to load only the base model.
# Example: ADAPTER_PATH = "llama-3.2-1b-yoda-adapter-cpu" 
# Example: ADAPTER_PATH = None
ADAPTER_PATH = None #"llama-3.2-1b-yoda-adapter-cpu" # Default to your previous path, can be changed to None
DEVICE = "cpu" # or "cuda" if you have a GPU

# --- 1. Load Model and Tokenizer ---
def load_model_and_tokenizer(base_model_id, adapter_path_or_none, device):
    print(f"Loading base model: {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # bf16 for GPU, fp32 for CPU
        device_map="auto" if device == "cuda" else {"": device} # "auto" for GPU, explicit for CPU
    )
    print(f"Base model memory footprint: {base_model.get_memory_footprint()/1e6:.2f} MB")

    print(f"Loading tokenizer for {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Set pad token if it was set during fine-tuning
    # From your fine_tune.py: tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        print("Setting pad_token to unk_token.")
        tokenizer.pad_token = tokenizer.unk_token 
        # Note: For some models, eos_token might be a better choice if unk_token is not appropriate.
        # Llama-3.2-Instruct tokenizer might already have pad_token set or handle it.
        # Check tokenizer.special_tokens_map and tokenizer.pad_token_id after loading.
        # If fine-tuning modified vocab or special tokens, ensure consistency.
        
    # Attempt to load adapter if path is provided
    model = base_model
    if adapter_path_or_none and isinstance(adapter_path_or_none, str) and adapter_path_or_none.strip():
        print(f"Loading adapter from: {adapter_path_or_none}...")
        try:
            # Dynamically assign the loaded PEFT model to 'model'
            model = PeftModel.from_pretrained(base_model, adapter_path_or_none)
            model = model.merge_and_unload() # Optional: merge adapter for faster inference if not training further
            print("Adapter loaded and merged successfully.")
        except Exception as e:
            print(f"Could not load or merge adapter from '{adapter_path_or_none}': {e}. Using base model only.")
            model = base_model # Ensure model is explicitly the base_model if adapter loading fails
    else:
        print("No adapter path provided or path is empty. Using base model only.")
        
    model.eval() # Set model to evaluation mode
    print(f"Final model memory footprint: {model.get_memory_footprint()/1e6:.2f} MB")
    return model, tokenizer

# --- 2. Text Generation Helper ---
def generate_response(model, tokenizer, messages, max_new_tokens=150):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(model.device) # Let apply_chat_template handle special tokens logic

    # Ensure eos_token_id is set for generation
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list): # Some tokenizers might have multiple EOS tokens
        eos_token_id = eos_token_id[0]

    print(f"\n--- Generating Response for Prompt ---\n{prompt}\n------------------------------------")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id, # Common practice
            do_sample=True, # Enable sampling for more diverse outputs
            temperature=0.6, # Control randomness; lower is more deterministic
            top_p=0.9,       # Nucleus sampling
        )
    
    # Decoding the response
    # We need to decode only the generated part, not the input prompt.
    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0, input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"Raw model output: {response_text}")
    return response_text.strip()

# --- 3. Dummy Functions (for Function Calling Example) ---
def get_weather(location: str, date: str, **kwargs):
    """Gets the weather for a given location and date.
    Date should be YYYY-MM-DD, 'today', or 'tomorrow'.
    """
    print(f"get_weather called with location: {location}, date: {date}, extra_args: {kwargs}")
    today_date = date.today()
    processed_date_str = ""

    if date == "today":
        processed_date_str = today_date.isoformat()
    elif date == "tomorrow":
        processed_date_str = (today_date + timedelta(days=1)).isoformat()
    else:
        try:
            # Try to parse as YYYY-MM-DD
            parsed_dt = date.fromisoformat(date)
            processed_date_str = parsed_dt.isoformat()
        except ValueError:
            # If not YYYY-MM-DD, and not today/tomorrow, it's ambiguous or unsupported
            print(f"Unsupported or ambiguous date format for get_weather: {date}")
            return {"error": True, "message": f"The date '{date}' is not in YYYY-MM-DD format or 'today'/'tomorrow'. Please specify a valid date."}

    if "coruscant" in location.lower():
        return {"location": location, "date": processed_date_str, "forecast": "Always bustling with some chance of senate debates."}
    return {"location": location, "date": processed_date_str, "forecast": "Sunny with a chance of meteor showers."}

def get_character_info(character_name: str, **kwargs):
    """Gets information about a Star Wars character."""
    print(f"get_character_info called with character_name: {character_name}, extra_args: {kwargs}")
    if "yoda" in character_name.lower():
        return {"name": character_name, "species": "Unknown", "affiliation": "Jedi Order", "quote": "Do or do not. There is no try."}
    if "luke skywalker" in character_name.lower():
        return {"name": character_name, "species": "Human", "affiliation": "Rebel Alliance", "quote": "I am a Jedi, like my father before me."}
    if "r2-d2" in character_name.lower():
        return {"name": character_name, "species": "Astromech Droid", "affiliation": "Rebel Alliance", "primary_function": "Starship repair and co-pilot"}
    return {"name": character_name, "info": f"Information not found for {character_name} in this simple demo."}

AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "get_character_info": get_character_info,
}

# --- 4. Function Calling Example (Multi-Turn Hermes Style) ---
def run_function_calling_example(model, tokenizer, user_query, available_functions):
    print(f"\n--- Running Function Calling Example for Query: '{user_query}' ---")

    # Revised System Prompt: Clearer, more direct, and focused on common failure points.
    system_prompt = f"""You are a helpful assistant. Your task is to answer user questions.
You can use functions to get information.

**Available Functions:**
1.  `get_weather(location: str, date: str)`
    *   Description: Gets the weather forecast.
    *   Arguments:
        *   `location` (string): The city or place.
        *   `date` (string): Use 'YYYY-MM-DD', 'today', or 'tomorrow'. For vague dates like 'next week', ask the user for a specific date or use 'today' if sensible.
2.  `get_character_info(character_name: str)`
    *   Description: Gets information about a Star Wars character.
    *   Arguments:
        *   `character_name` (string): The name of the character.

**Your Behavior:**

1.  **Analyze the User's Query:** Understand what the user is asking.
2.  **Decide to Call a Function (or not):**
    *   **If a function IS NEEDED:**
        *   You MUST respond with EXACTLY ONE tool call.
        *   The format is: `<tool_call>SINGLE_JSON_OBJECT</tool_call>`
        *   Replace `SINGLE_JSON_OBJECT` with a VALID JSON object using DOUBLE QUOTES for all keys and string values.
        *   Example of `SINGLE_JSON_OBJECT`: {{ "name": "get_weather", "arguments": {{ "location": "Naboo", "date": "today" }} }}
        *   Use the EXACT argument names specified for the function (e.g., `location`, `date`, `character_name`).
        *   Do NOT invent functions. Only use the functions listed above.
    *   **If NO function is needed (or after all calls are done):**
        *   Respond with a plain text answer. Do NOT use `<tool_call>` or any other tags.
        *   Example: "Hello there!" or "The weather on Coruscant for 2023-10-26 is sunny."

3.  **After I Execute a Function:**
    *   I will provide the result in this format: `<tool_response>JSON_FUNCTION_RESULT</tool_response>`
    *   Your NEXT response MUST be one of these two:
        1.  Another SINGLE tool call (if more information is needed and another function call is appropriate): `<tool_call>SINGLE_JSON_OBJECT</tool_call>`
        2.  A final plain text answer to the user (if you have all the information).
    *   **IMPORTANT:** Do NOT output `<tool_response>` yourself. That is my role.

**Summary of Critical Rules:**
*   Only ONE tool call at a time if a function is needed.
*   Tool call content MUST be a single, valid JSON object with double quotes.
*   Use EXACT function and argument names.
*   If no function is needed, or after function results, answer in plain text.
*   DO NOT generate `<tool_response>` tags in your output.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    max_turns = 5
    for turn in range(max_turns):
        print(f"\nTurn {turn + 1}")

        llm_response_text = generate_response(model, tokenizer, messages, max_new_tokens=200)
        messages.append({"role": "assistant", "content": llm_response_text})

        # Check if LLM incorrectly generated a tool_response
        if "<tool_response>" in llm_response_text:
            print("LLM Error: Model incorrectly generated a <tool_response> tag.")
            # Add a corrective message to send back to the LLM
            correction_message_content = "You incorrectly generated a `<tool_response>`. Your role is to either call a function using `<tool_call>SINGLE_JSON_OBJECT</tool_call>` or provide a final text answer to the user. Please try again."
            messages.append({"role": "user", "content": correction_message_content}) # Send as user to prompt for new assistant response
            continue # Go to the next turn to let the LLM correct itself

        tool_call_match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", llm_response_text, re.DOTALL)

        if tool_call_match:
            raw_tool_call_content = tool_call_match.group(1).strip()
            print(f"Extracted raw tool call content: {raw_tool_call_content}")
            
            # Handle if the model outputs the literal placeholder or empty content
            if not raw_tool_call_content or raw_tool_call_content == "SINGLE_JSON_OBJECT" or raw_tool_call_content == "JSON_OBJECT":
                print(f"LLM Error: Model outputted placeholder or empty content: '{raw_tool_call_content}'")
                correction_message_content = f"You provided placeholder or empty content ('{raw_tool_call_content}') inside `<tool_call>`. Please provide a_real_ JSON object like {{ \"name\": \"function_name\", \"arguments\": {{ \"arg\": \"value\" }} }}."
                messages.append({"role": "user", "content": correction_message_content})
                continue

            sanitized_json_str = ""
            try:
                python_obj = ast.literal_eval(raw_tool_call_content)
                sanitized_json_str = json.dumps(python_obj) # Convert to strict JSON
                print(f"Sanitized JSON string for parsing: {sanitized_json_str}")
                
                tool_call_data = json.loads(sanitized_json_str)
                function_name = tool_call_data.get("name")
                function_args = tool_call_data.get("arguments", {})

                if not function_name:
                    print("LLM produced a tool_call without a function name. Treating as text response.")
                    break
                
                if function_name not in available_functions:
                    print(f"LLM tried to call an unavailable function: '{function_name}'. Instructing to use available functions or answer directly.")
                    # Provide feedback to the LLM that the function is not available
                    error_message = f"Function '{function_name}' is not available. Please use one of the available functions or answer directly if possible."
                    error_object_for_llm = {"error": error_message}
                    tool_data_for_response = {"name": function_name, "content": error_object_for_llm}
                    tool_response_for_llm = f"<tool_response>\n{json.dumps(tool_data_for_response)}\n</tool_response>"
                    messages.append({"role": "tool", "content": tool_response_for_llm, "name": function_name})
                    continue # Allow LLM to try again

                print(f"LLM wants to call function: {function_name} with args: {function_args}")
                function_to_call = available_functions[function_name]
                try:
                    function_response_content = function_to_call(**function_args)
                    print(f"Function {function_name} executed. Response: {function_response_content}")
                    tool_data_for_response = {"name": function_name, "content": function_response_content}
                    tool_response_for_llm = f"<tool_response>\n{json.dumps(tool_data_for_response)}\n</tool_response>"
                    messages.append({"role": "tool", "content": tool_response_for_llm, "name": function_name})
                except Exception as e:
                    print(f"Error calling function {function_name}: {e}")
                    error_object = {"error": f"Error executing function {function_name}: {str(e)}"}
                    tool_data_for_response = {"name": function_name, "content": error_object}
                    tool_response_for_llm = f"<tool_response>\n{json.dumps(tool_data_for_response)}\n</tool_response>"
                    messages.append({"role": "tool", "content": tool_response_for_llm, "name": function_name})
            
            except (SyntaxError, ValueError) as e:
                print(f"Failed to sanitize or decode JSON from tool_call: {e}. Raw content: {raw_tool_call_content}")
                correction_message_content = f"The content you provided in `<tool_call>` ('{raw_tool_call_content}') was not a valid JSON structure or Python literal. Please ensure it's a single, well-formed JSON object with double quotes for keys/strings, like: {{ \"name\": \"func_name\", \"arguments\": {{ \"arg\": \"value\" }} }}."
                messages.append({"role": "user", "content": correction_message_content})
                continue # Allow LLM to try again
        else:
            print(f"LLM final response (no tool call detected):\n{llm_response_text}")
            break

    if turn == max_turns - 1 and tool_call_match: # Still in a tool call loop at max_turns
        print("\nMax turns reached during function calling. Ending conversation.")
        # Potentially add a message to say max turns reached before final answer.

    final_llm_answer = messages[-1]["content"] if messages[-1]["role"] == "assistant" else llm_response_text
    # Ensure the final answer is not a tool_call itself if the loop broke early due to bad parsing
    if re.search(r"<tool_call>", final_llm_answer, re.DOTALL):
        final_llm_answer = "I encountered an issue with formatting my function call. Please try rephrasing your request."
        print(f"Overriding final answer due to lingering tool_call structure.")

    print(f"\nFinal conversational answer from LLM: {final_llm_answer}")
    return final_llm_answer

# --- 5. Structured JSON Output Example ---
def run_structured_json_example(model, tokenizer):
    print("\n--- Running Structured JSON Output Example ---")
    
    text_to_extract = "Luke Skywalker, a hero from Tatooine, piloted an X-Wing and later owned a T-16 Skyhopper. He faced Darth Vader."
    
    # Note: The persona of your fine-tuned model (Yoda-speak) might interfere with strict JSON output.
    # A more general instruction model might be better for pure JSON tasks unless the fine-tuning
    # also included JSON output examples.
    system_prompt = """You are an information extraction assistant.
From the provided text, extract the character's name, their home planet, and any starships they are associated with.
Respond ONLY with a single JSON object containing these fields: "name", "planet", "starships" (which should be a list of strings).
Example: {"name": "Han Solo", "planet": "Corellia", "starships": ["Millennium Falcon"]}
Do not add any other text before or after the JSON object.
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the text:\n{text_to_extract}"}
    ]
    
    response_text = generate_response(model, tokenizer, messages)
    
    try:
        # Try to find JSON within the response if the model adds extra text.
        # This is a common workaround.
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1 and json_start < json_end:
            json_str = response_text[json_start:json_end]
            extracted_data = json.loads(json_str)
            print("\nSuccessfully parsed JSON output:")
            print(json.dumps(extracted_data, indent=2))
        else:
            raise json.JSONDecodeError("No JSON object found in response", response_text, 0)
            
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse JSON from model response. Error: {e}")
        print(f"Raw response was:\n{response_text}")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(BASE_MODEL_REPO_ID, ADAPTER_PATH, DEVICE)
    
    if loaded_model and loaded_tokenizer:
        # Test Cases for Multi-Turn Function Calling
        test_queries = [
            "What's the weather like on Coruscant tomorrow?", # Test 1: Single function call
            "Tell me about Yoda.", # Test 2: Another single function call
            "What's the weather on Hoth next week and also, can you tell me about Luke Skywalker?", # Test 3: Two function calls needed
            "Give me info on R2-D2 and then find out the weather on Naboo for today.", # Test 4: Two function calls, different order
            "I don't need any functions, just say hello.", # Test 5: No function call needed
        ]

        for i, query in enumerate(test_queries):
            print(f"\n--- Test Case {i+1} ---")
            run_function_calling_example(loaded_model, loaded_tokenizer, query, AVAILABLE_FUNCTIONS)
            print("\n" + "="*50 + "\n") # Separator

        # Keep the structured JSON example if you still want to test it separately
        # print("\n" + "="*50 + "\n")
        # run_structured_json_example(loaded_model, loaded_tokenizer) # You might want to comment this out if focusing on function calling
    else:
        print("Failed to load model and tokenizer. Exiting.") 