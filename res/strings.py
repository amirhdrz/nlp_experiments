""" strings.py - contains user-facing message strings
"""

import random

def nlp_fail(user_message: str):
    sents = ("Sorry, I don't understand '%s'.",
             "I can't understand '%s'. Sorry about that.")

    selected = random.choice(sents)
    return selected % user_message

course_code_prompt = "Please enter a course code"