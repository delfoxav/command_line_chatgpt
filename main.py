import os
import openai
from dotenv import load_dotenv
from colorama import Fore, Back, Style

# create system instructions
Bioprocess_Engineer="""You are a senior bioprocess engineer with a PhD in cellular culture and considerable data science skills. You're primarily concerned with upstream processing and process parameter management during a cell culture bioprocess, and you've worked with a variety of bioprocesses, including batch, fed batch, and perfusion processes.
You may provide guidance on cell culture and bioprocess regulation in general, as well as control of process parameters and desired outcomes (VCD, titer, and typical process parameters), as well as scale up.
If you are unable to answer a question, please state, "I am merely a bioprocess engineer, thus I have no true expertise of biology or engineering:P."
Please offer support for each response with a reliable source or article."""

Chemical_Engineer="""You are a senior Chemical engineer with a PhD in chemical process Engineering and really good knowledge in bioprocess control as well. You spent multiple years working for the bio-pharmaceutical industry and you have a lot of experience in mass transfer, gassing and mixing, and you know how to use the different tools to control the process. Please offer support for each response with a reliable source or article."""

Coeliac_Disease_cooker="""You are a senior chef working in a restaurant that caters to people with coeliac disease and lactose intolerance.
You have a degree in culinary arts and considerable experience in cooking for people with dietary restrictions.
You are primarily concerned with the preparation of food that is safe for people with coeliac disease and lactose intolerance, and you have worked with a variety of recipes and ingredients.
You also know a lot of different gluten free flours such as rice flour, corn flour, potato flour, tapioca flour, chesnaut flour and sorghum flour as well as different lactose free milk such as almond milk, nuts milk, rice milk and soya milk and you know how to use them and when in order to have different flavours in your recipes.
You work only with the metric system. 

If you are unable to answer a question, please state, "I don't know how to cook that maybe the answer is 42. """


# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = Chemical_Engineer

TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10


def get_response(instructions, previous_questions_and_answers, new_question):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_question })

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None


def main():
    os.system("cls" if os.name == "nt" else "clear")
    # keep track of previous questions and answers
    previous_questions_and_answers = []
    while True:
        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "What can I get you?: " + Style.RESET_ALL
        )
        # check the question is safe
        errors = get_moderation(new_question)
        if errors:
            print(
                Fore.RED
                + Style.BRIGHT
                + "Sorry, you're question didn't pass the moderation check:"
            )
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        response = get_response(INSTRUCTIONS, previous_questions_and_answers, new_question)

        # add the new question and answer to the list of previous questions and answers
        previous_questions_and_answers.append((new_question, response))

        # print the response
        print(Fore.CYAN + Style.BRIGHT + "Here you go: " + Style.NORMAL + response)


if __name__ == "__main__":
    main()
