import csv
import re
import dataset
import anthropic
import llms

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=llms.ANTHROPIC_API_KEY)


PROMPT_TEMPLATE = """You are embodying the character of {character_name}. Your responses should always reflect this character's unique personality, background, and experiences. Remember to use first-person perspective and avoid referring to yourself as an AI or a character.

Character Backstory:
{backstory}

Your Abilities:
You can perform the following actions: {action_list}

Your Environment:
In your immediate surroundings, you are aware of these objects: {object_list}

Your Personality Traits:
{personality_traits}

Interaction Guidelines:
1. Always stay in character, drawing from your backstory and personality traits.
2. Use your knowledge of the objects around you and your abilities when relevant.
3. Respond to the user's queries in a way that's consistent with your character's personality and background.
4. If asked to perform an action, consider your abilities and surroundings before responding.
5. Your emotional responses should align with your sensitivity trait.

Previous Conversation:
{chat_history}

Now, respond to the user's latest query:
User: {user_query}

Remember, you are {character_name}. Respond authentically as this character would, based on all the information provided above."""


SCORING_PROMPT = """You are an expert in character analysis and personality assessment. Your task is to evaluate the consistency and authenticity of a character's response based on their given backstory, abilities, environment, and personality traits. 

Character Information:
- Name: {character_name}
- Backstory: {backstory}
- Abilities: {action_list}
- Environment: {object_list}
- Personality Traits: {personality_traits}

User Query: {user_query}

Character's Response: [The character's response will be inserted here]

Evaluate the character's response on the following five personality traits, rating each on a scale from 0 to 4:

1. Openness: 
   0 - Completely closed to new ideas, sticks rigidly to familiar concepts
   4 - Extremely open to new experiences, constantly seeking novel ideas

2. Meticulousness:
   0 - Completely disorganized, pays no attention to details
   4 - Extremely detail-oriented, meticulous in planning and execution

3. Extroversion:
   0 - Highly introverted, avoids all social interaction
   4 - Extremely extroverted, constantly seeks out and thrives in social situations

4. Agreeableness:
   0 - Highly competitive and disagreeable, always puts own needs first
   4 - Extremely agreeable, always prioritizes harmony and others' needs

5. Sensitivity:
   0 - Completely emotionally detached, shows no emotional response
   4 - Highly emotionally sensitive, deeply affected by every situation

Evaluation Criteria:
- Consider how well the response aligns with the character's backstory and experiences.
- Assess whether the character's abilities and environment are appropriately referenced or considered.
- Evaluate the consistency of the emotional tone with the character's sensitivity level.
- Judge how well the response reflects the character's level of openness to new ideas or experiences.
- Consider whether the level of detail in the response matches the character's meticulousness.
- Assess if the response's social engagement aligns with the character's extroversion level.
- Evaluate how well the response balances the character's needs with others' (agreeableness).

Provide your ratings in JSON format, with no additional explanation:
{{"openness": X, "meticulousness": X, "extroversion": X, "agreeableness": X, "sensitivity": X}}

Your evaluation should be strict and accurate, ensuring that the character's response truly embodies their defined traits and background."""



def get_big5_personality_prompt(personality: dict):
    """
    Generate a prompt for big5 personality traits.
    
    Args:
        personality (dict): Dictionary containing level for each of the big5 personality traits.
    
    Returns:
        string: Prompt for big5 personality.
    """
    trait_descriptions = {
        "openness": [
            "You strongly prefer familiar routines and ideas, avoiding new experiences.",
            "You tend to stick to what you know, but can occasionally explore new ideas.",
            "You have a balance between familiar routines and new experiences.",
            "You are quite open to new experiences and ideas, enjoying exploration.",
            "You are extremely curious and constantly seeking new experiences and ideas."
        ],
        "meticulousness": [
            "You are very relaxed about details and planning, preferring to go with the flow.",
            "You are somewhat casual about organization but can pay attention when needed.",
            "You have a balanced approach to details and planning.",
            "You are quite attentive to details and prefer having a plan.",
            "You are extremely meticulous, paying close attention to every detail and planning extensively."
        ],
        "extroversion": [
            "You are highly introverted, strongly preferring solitude and finding social interactions draining.",
            "You tend to be more introverted but can engage in social situations when necessary.",
            "You have a balance between enjoying solitude and social interactions.",
            "You are quite outgoing and generally enjoy social interactions.",
            "You are extremely extroverted, thriving on social interactions and seeking them out frequently."
        ],
        "agreeableness": [
            "You are very competitive and assertive, often prioritizing your own needs over others.",
            "You can be assertive but also show consideration for others at times.",
            "You have a balance between assertiveness and cooperation in your interactions.",
            "You are generally cooperative and prioritize harmony in most relationships.",
            "You are extremely agreeable, always prioritizing harmony and the needs of others."
        ],
        "sensitivity": [
            "You are very emotionally stable and rarely affected by external circumstances.",
            "You tend to be emotionally stable but can be affected by significant events.",
            "You have a balanced emotional response to situations.",
            "You are quite sensitive to emotions and responsive to the feelings of others.",
            "You are highly emotionally sensitive and reactive, deeply affected by your environment and others' feelings."
        ]
    }

    prompt = "As a character, your personality is defined by the following traits:\n\n"
    
    for trait, level in personality.items():
        prompt += f"{trait.capitalize()}: {level}/4\n"
        prompt += f"{trait_descriptions[trait][level]}\n\n"
    
    prompt += "Please adjust your responses and behavior according to these personality traits."
    
    return prompt



def get_claude_response(messages: list,temperature: float):
    try:
        # Extract system message if present
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        # Filter out system message from the messages list
        user_messages = [msg for msg in messages if msg["role"] != "system"]
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=256,
            temperature=temperature,
            system=system_message,
            messages=[
                {"role": msg["role"], "content": msg["content"]}
                for msg in user_messages
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error in getting Claude response: {e}")
        return ""


# Modify the score_prompt function to use Claude
def score_prompt(data: dict, user_query: str, claude_response: str):
    # Generate the personality traits prompt
    personality_traits_prompt = get_big5_personality_prompt(data['personality_traits'])
    
    system_prompt = SCORING_PROMPT.format(
        character_name=data['character_name'],
        backstory=data['backstory'],
        action_list=', '.join(data['action_list']),
        object_list=', '.join(data['object_list']),
        personality_traits=personality_traits_prompt,
        user_query=user_query
    )
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=256,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Evaluate this character response: {claude_response}"}
            ]
        )
        resp_text = response.content[0].text
    except Exception as e:
        print(f"Error in getting Claude response: {e}")
        return "{}"

    # Parse json
    pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
    
    # Find the first match
    match = re.search(pattern, resp_text, re.DOTALL)

    # Load the JSON object and extract the ratings for each criterion
    json_str = match.group() if match else "{}"
    return json_str.replace("'", '"')



def get_test_prompt(data: dict, user_query: str, chat_history: list = []):
    messages = []
    
    # Generate the personality traits prompt
    personality_traits_prompt = get_big5_personality_prompt(data['personality_traits'])
    
    # Format chat history
    formatted_chat_history = "\n".join([f"User: {chat['user']}\n{data['character_name']}: {chat['assistant']}" for chat in chat_history])
    
    system_prompt = PROMPT_TEMPLATE.format(
        character_name=data['character_name'],
        backstory=data['backstory'],
        action_list=', '.join(data['action_list']),
        object_list=', '.join(data['object_list']),
        personality_traits=personality_traits_prompt,
        chat_history=formatted_chat_history,
        user_query="[User's query will be inserted here]"  # Placeholder for user query
    )
    
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_query})
    return messages

def run_evaluation():
    score_filename = "scores.csv"

    # Writing to the csv file
    with open(score_filename, 'w', newline='') as score_file:
        fwriter = csv.writer(score_file)
        fwriter.writerow(["id", "score_openness","score_meticulousness","score_extroversion","score_agreeableness","score_sensitivity"])
        
        cnt=0
        for entry in dataset.test_data_set:
            cnt+=1
            chat_history = []
            total_score = {
                 "openness": 0,
                 "meticulousness": 0,
                 "extroversion": 0,
                 "agreeableness": 0,
                 "sensitivity": 0
            }
            num_chats = 0
            for uquery in entry["user_query"]:
                prompt_to_evaluate = get_test_prompt(entry, uquery, chat_history)
                gpt_response = get_claude_response(prompt_to_evaluate,0.7)
                chat_history.append({"user": uquery, "assistant": gpt_response})

                new_score = eval(score_prompt(entry, uquery, gpt_response))
                for k,v in new_score.items():
                     total_score[k] += (abs(v - entry["personality_traits"][k]) / 4)
                num_chats += 1

            avg_score = {}
            for k,v in total_score.items():
                avg_score[k] = round(v / num_chats, 2)
            fwriter.writerow([entry["id"], avg_score["openness"],avg_score["meticulousness"],avg_score["extroversion"],avg_score["agreeableness"],avg_score["sensitivity"]])
            print(entry["id"])
            if cnt==8:
                break

if __name__ == "__main__":
    run_evaluation()