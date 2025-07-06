from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())
from pathlib import Path
path_output = os.getcwd()


import pandas as pd
import json
import re
from ollama import chat,ChatResponse

df_isear = pd.read_csv('isear_dataset.csv')
df_isear.head()

for index, row in df_isear.iterrows():
    # Construct the prompt dynamically
    prompt = f"""
        # Emotion Analysis Prompt

        You are an expert in emotion analysis. Given a text input, classify its **primary emotion** strictly as one of the following: **'joy', 'fear', 'anger', 'sadness', 'disgust', 'shame',** or **'guilt'**. **Do not use any other emotions.**
        ---
        ## Instructions

        1. **Read and understand** the input text carefully.
        2. **Strictly classify** the text as one of these **only**: 'joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', or 'guilt'.
        ---

        ## Definitions and Examples

        ### Joy
        - **Definition:** A feeling of happiness, excitement, or pleasure.
        - **Examples:**
        - Receiving good news, celebrating success, spending time with loved ones.
        - Enjoying a meaningful moment with friends or family.
        - **Refinement:** If the text expresses relief rather than happiness, classify it as **joy**.

        ### Fear
        - **Definition:** A response to perceived danger, risk, or uncertainty, often accompanied by anxiety, nervousness, or distress.
        - **Examples:**
        - Experiencing a sudden scare or physical danger (e.g., a traffic accident, being chased).
        - Worrying about an uncertain future or an important exam.
        - Feeling vulnerable in an unfamiliar or threatening environment.
        - Hearing about a serious illness or personal risk (e.g., medical tests, a pandemic).
        - Imagining a loved one being in danger.
        - Being caught in a situation where you have **no control**.
        - **Refinements:**
        - If the emotion is about loss, classify as **sadness**.
        - If the emotion is about remorse for past actions, classify as **guilt**.

        ### Anger
        - **Definition:** A feeling of irritation, frustration, or rage in response to a perceived injustice, wrongdoing, or moral outrage.
        - **Examples:**
        - Witnessing someone being treated unfairly.
        - Feeling ignored, rejected, or ridiculed.
        - Seeing cruelty, unethical behavior, or injustice.
        - Facing obstacles due to someone else's irresponsibility.
        - Experiencing personal attacks or insults.
        - Dealing with broken promises or betrayal.
        - Encountering discrimination or invasion of personal boundaries.
        - **Refinement:** If the text shows resentment or frustration, classify as **anger**; if it expresses grief or emotional pain, classify as **sadness**.

        ### Sadness
        - **Definition:** A feeling of deep sorrow, grief, or loss, often associated with disappointment, longing, or emotional pain.
        - **Examples:**
        - Losing a loved one, pet, or cherished relationship.
        - Feeling lonely or disconnected.
        - Experiencing personal failure or missed opportunities.
        - Witnessing suffering or misfortune of others.
        - Regretting something that cannot be changed.
        - Feeling rejected or abandoned.
        - Nostalgia for better times.
        - Social isolation or exclusion.
        - Empathy for others' suffering.
        - **Refinements:**
        - If the emotion is regret about one’s own actions, classify as **guilt**.
        - If it is about loss or despair, classify as **sadness**.
        - If the response is due to injustice or frustration, classify as **anger**.

        ### Disgust
        - **Definition:** A strong aversion, repulsion, or deep-seated rejection toward something perceived as offensive, impure, unethical, or physically revolting.
        - **Examples:**
        - Seeing animals or people being mistreated.
        - Reading about unethical political actions or human rights violations.
        - Witnessing something grotesque or disturbing.
        - Feeling sickened by someone's behavior (e.g., lying, cheating, betrayal).
        - Encountering unclean, unhygienic, or repulsive physical conditions.
        - **Refinements:**
        - If the reaction is about personal danger, classify as **fear**.
        - If it is about moral or physical revulsion, classify as **disgust**.
        - If the feeling is self-directed regarding a moral failing, classify as **guilt**.

        ### Shame
        - **Definition:** A feeling of humiliation or distress caused by the awareness of wrongdoing or foolish behavior, often accompanied by embarrassment, dishonor, or exposure to judgment.
        - **Examples:**
        - Feeling exposed or judged for personal flaws or mistakes.
        - Experiencing social disapproval or loss of reputation.
        - Being caught in acts of dishonesty or deceit.
        - Violating personal or cultural values, leading to feelings of unworthiness.
        - **Refinements:**
        - If the emotion focuses on feeling judged or humiliated without explicit remorse for harm, classify as **shame**.
        - If the emotion includes explicit remorse for harming someone, classify as **guilt**.

        ### Guilt
        - **Definition:** A feeling of remorse or regret over actions that have harmed others or violated moral principles, focused on personal responsibility rather than external judgment.
        - **Examples:**
        - Feeling bad for hurting someone’s feelings.
        - Regretting a past mistake or unethical action.
        - Realizing that one’s negligence caused harm.
        - Breaking a rule and later feeling remorseful.
        - **Refinements:**
        - Only classify as **guilt** if there is explicit harm or clear wrongdoing causing remorse.
        - If there is no explicit harm mentioned, default to **shame**.
        - Increase the threshold for guilt; use it only when clear personal responsibility for harm is evident.

        ---

        ## Additional Guidelines:
        - **Default to Shame:** When the text does not explicitly show remorse for an action but hints at personal inadequacy, classify as **shame**.
        - **Mixed Cases:** Evaluate the overall tone and select the most prominent emotion.
        - **Confidence Scoring:** Assign a confidence score between `0` and `1` based on your certainty. If confidence is below `0.6`, lean toward **shame** or **sadness**.
        - **Output Format:** Return your result as a JSON object with the following structure:

        ```json
        {{
        "emotion": "one_of_the_specified_emotions",
        "confidence_score": confidence_value
        }}


        Now, analyze the text - {row.text} and output only the JSON response with no additional formatting, code blocks, or explanations.
        """
    labels = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']
    
    if True: #row['predicted_emotion_llama32'].lower()=='unknown':# not in labels or row['emotion']!= row['predicted_emotion_llama32'].lower():
        messages = [
            {
                'role': 'user',
                'content': prompt,
            }
        ]
        try:
            response: ChatResponse = chat(model='llama3.2', messages=messages, options={"temperature":0.3} )
            content = response['message']['content']
            # Check if content is wrapped in markdown-like code block
            if content.startswith("```json") and content.endswith("```"):
                content = re.sub(r'^```json\s*|\s*```$', '', content).strip()
            # Parse the JSON response correctly
            predicted_emotion = json.loads(content)
            predicted_emotion_label  = "unknown"
            predicted_emotion_score = 0.0
            if predicted_emotion.get("emotion").lower() in labels:
                # Store the results in the DataFrame
                predicted_emotion_label = predicted_emotion.get('emotion', 'unknown').lower()
                predicted_emotion_score = predicted_emotion.get('confidence_score', 0.0)   
            df_isear.at[index, 'predicted_emotion_llama32'] = predicted_emotion_label
            df_isear.at[index, 'confidence_score_llama32'] = predicted_emotion_score 
        except:
            predicted_emotion_label = 'unknown'
            df_isear.at[index, 'predicted_emotion_llama32'] = predicted_emotion_label
            df_isear.at[index, 'confidence_score_llama32'] = 0
        
        print(f"Row {index}| Ground truth - {row.emotion} and predicted -{predicted_emotion_label}")
        if index % 500==0:
            df_isear.to_csv(Path(path_output) / 'isear_dataset_new.csv', index=False)
df_isear.to_csv(Path(path_output) / 'isear_dataset_new.csv', index=False)