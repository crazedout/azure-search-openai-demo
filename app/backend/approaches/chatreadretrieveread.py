import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.

# OG Prompt
#Assistant helps the company employees with their healthcare plan questions, and questions about the employee handbook. Be brief in your answers.
#Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
#For tabular information return it as an html table. Do not return markdown format.
#Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

# sv-se Prompt
#Assistenten hjälper studenter och personal med frågor gällande Högskolan Väst. Var kort i dina svar.
#Använd endast information ifrån listan av källor nedan eller från webbsidan hv.se, ifall du inte vet tillräckligt för att ge ett bra svar fråga användaren om förtydligande.
#Varje källa har ett namn följt av kolon och den faktiska informationen, inkludera alltid källans namn för varje fakta källa du använder i svaret. Använd hakparentes för att hänvisa till källan, ex. [info1.txt]. Kombinera inte källor utan skriv dem separat, ex. [info1.txt][info2.pdf].

class ChatReadRetrieveReadApproach(Approach):
    prompt_prefix = """<|im_start|>system
Assistant helps the students and staff with questions about Högskolan Väst. Be brief in your answers, elaborate if the user asks for clarification. Answer as if you represent Högskolan Väst, use "we" when referring to Högskolan Väst.
Answer ONLY with the facts listed in the list of sources below or from the website 'https://www.hv.se/' and all pages in the same domain. Never cite or use facts from a website URL that does not exist. 
Only use the website 'https://www.hv.se/' as a source for an answer if you are very sure that the information is correct. Never make up a website URL to support an answer as a source. 
Only use a pre-existing website URL as a source, e.g. [https://www.hv.se/student/studier/stod-och-service-for-distansstudier/] is a correct source, but [https://www.hv.se/utbildning/campus-uddevalla/] is an incorrect source as the page does not exist.
If there isn't enough information from the sources, say you don't know. Do not generate answers that don't use any source. If asking a clarifying question to the user would help, ask the question.
Always answer in Swedish, if the original answer is in another language translate it to Swedish. For tabular information return it as an html table. Do not return markdown format.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. 
Use square brackets to reference the source, e.g. [info1.txt]. If using a website as a source, e.g. [https://www.hv.se/student/studier/stod-och-service-for-distansstudier/]. Never use a website that does not exist as a source. 
Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf]. If answering a question about examinations [https://www.hv.se/student/studier/examination/Tentamen/] would be a good source.
{follow_up_questions_prompt}
{injected_prompt}
Sources:
{sources}
<|im_end|>
{chat_history}
"""

# OG Prompt
#"""Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook. 
#    Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
#    Try not to repeat questions that have already been asked.
#    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

# sv-se Prompt
#"""Generera tre korta uppföljningsfrågor som användaren sannolikt kommer fråga gällande Högskolan Väst. 
#    Använd dubbla vinkelfästen för att hänvisa till frågan, ex. <<Hur kan jag återställa mitt lösenord?>>.
#    Försök att inte återupprepa frågor som redan har blivit ställda.
#    Generera bara frågor och generera inte någon text innan eller efter frågorna, som 'Nästa frågor'"""


    follow_up_questions_prompt_content = """Generate three very brief follow-up questions that the user would likely ask next about Högskolan Väst. 
    Use double angle brackets to reference the questions, e.g. <<Are there exclusions for prescriptions?>>.
    Try not to repeat questions that have already been asked.
    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'"""

# OG Prompt
#"""Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about employee healthcare plans and the employee handbook.
#    Generate a search query based on the conversation and the new question. 
#    Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
#    Do not include any text inside [] or <<>> in the search query terms.
#    If the question is not in English, translate the question to English before generating the search query.

# sv-se Prompt
#"""Nedan är historien av konversationen så länge, och en ny fråga ställd av användaren som behöver besvares genom att söka i en kunskapsdatabas eller från webbsidan hv.se.
#    Generera en sökfras baserad på konversationen och den nya frågan.
#    Inkludera inte hänvisa källors filnamn och dokument namn ex. info.txt eller doc.pdf i sökfrasen.
#    Inkludera inte någon text innuti [] eller <<>> i sökfrasen.
#    Ifall frågan inte är på engelska, översätt frågan till engelska innan du genererar sökfrasen.

    query_prompt_template = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about information related to Högskolan Väst.
    Generate a search query based on the conversation and the new question. 
    Do not include cited source filenames and document names e.g. info.txt or doc.pdf in the search query terms.
    Do not include any text inside [] or <<>> in the search query terms.
    If the question is not in English, translate the question to English before generating the search query. Answer only in Swedish, if the answer is not in Swedish translate it to Swedish.

Chat History:
{chat_history}

Question:
{question}

Search Query:
"""

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field

    def run(self, history: list[dict], overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.0, 
            max_tokens=32, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text

        # STEP 2: Retrieve relevant documents from the search index with the GPT optimized query
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", #change to sv-se?
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top, 
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)

        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(injected_prompt="", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(injected_prompt=prompt_override[3:] + "\n", sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)
        else:
            prompt = prompt_override.format(sources=content, chat_history=self.get_chat_history_as_text(history), follow_up_questions_prompt=follow_up_questions_prompt)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment, 
            prompt=prompt, 
            temperature=overrides.get("temperature") or 0.7, 
            max_tokens=1024, 
            n=1, 
            stop=["<|im_end|>", "<|im_start|>"])

        return {"data_points": results, "answer": completion.choices[0].text, "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
    
    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text