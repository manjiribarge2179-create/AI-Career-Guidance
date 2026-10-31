"""
Career Guidance Chatbot - Final Version
Strategy: exact match → keyword sub-topic routing → TF-IDF fallback
"""

import os, json, pickle, math, re
from collections import Counter

DATASET_PATH = "data/career_dataset.json"

_STOP = {"i","a","an","the","is","are","was","were","be","been","do","does",
         "did","will","would","can","could","should","have","has","had","to",
         "of","in","on","at","for","and","or","but","with","from","that",
         "this","my","me","what","how","which","who","where","when","why",
         "about","after","before","if","not","it","its","into","by","up","so",
         "we","he","she","they","their","you","your","am","any","some","all",
         "also","no","than","then","there"}

def _norm(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _tok(t):
    return [w for w in _norm(t).split() if w not in _STOP]

def _bigrams(t):
    tk = _tok(t)
    return tk + [f"{tk[i]}_{tk[i+1]}" for i in range(len(tk)-1)]


# ── Sub-topic routing rules ────────────────────────────────────────────────
# (keywords_any_must_match, dataset_input_substring_to_return)
# MORE SPECIFIC rules come FIRST

RULES = [
    # Engineering sub-topics
    (["which branch","best branch","branch is best","cse vs","vs mechanical","branch scope","branch should","branch engineering"],
     "Which branch is best in engineering"),
    (["cse mechanical","computer mechanical","cse or mechanical"],
     "CSE vs mechanical which is better"),
    (["scope civil engineering","civil engineering scope","civil career"],
     "What is the scope of civil engineering"),
    (["jee advanced","what is jee advanced"],
     "What is JEE Advanced"),
    (["jee pattern","jee syllabus","jee format","jee exam pattern"],
     "What is the JEE exam pattern"),
    (["prepare jee","jee preparation","how to prepare for jee","crack jee","jee study"],
     "How to prepare for JEE"),
    (["jee attempts","how many attempts jee","attempt in jee"],
     "How many attempts are allowed in JEE"),
    (["become engineer","how to become engineer","engineer kaise bane","btech after 12",
      "engineering after 12","requirements for engineering","subjects for engineering"],
     "How can I become an engineer"),
    (["what is jee","jee main","jee exam","jee kya","engineering entrance"],
     "What is JEE"),
    (["scope engineering","engineering scope","engineering career","engineering jobs"],
     "Which branch is best in engineering"),

    # Medical sub-topics
    (["neet pattern","neet syllabus","neet format","neet exam pattern","neet marking","neet questions"],
     "What is the NEET exam pattern"),
    (["prepare neet","neet preparation","how to prepare for neet","crack neet","neet study","neet coaching"],
     "How to prepare for NEET"),
    (["neet attempts","how many attempts neet","neet age limit","neet eligibility"],
     "How many attempts in NEET"),
    (["without neet","courses without neet","neet not required","medical without neet","no neet"],
     "Which medical courses do not need NEET"),
    (["specialization","best specialization","doctor specialization","which specialization","md ms"],
     "Which doctor specialization is best"),
    (["scope mbbs","mbbs scope","mbbs career","mbbs future","doctor scope","doctor career"],
     "What is the scope of MBBS in India"),
    (["after mbbs","what after mbbs","md after mbbs","pg after mbbs","post mbbs"],
     "What can I do after MBBS"),
    (["become doctor","how to become doctor","doctor kaise","mbbs after 12",
      "enter healthcare","healthcare after 12","medical after 12"],
     "How can I enter healthcare after 12th"),
    (["what is neet","neet kya","neet overview","medical entrance exam","neet ug"],
     "What is NEET"),
    (["biology students","medical courses available","neet required for doctor"],
     "How can I enter healthcare after 12th"),

    # CA / Commerce sub-topics
    (["how long ca","ca duration","ca time","ca years","ca kitne saal"],
     "How long does CA take"),
    (["ca difficult","ca tough","ca hard","ca pass rate"],
     "Is CA difficult"),
    (["salary after ca","ca salary","ca income","ca package","ca earn"],
     "What is the salary after CA"),
    (["become ca","how to become ca","ca kaise","ca steps","ca pathway","start ca","ca after 12"],
     "How do I become a CA"),
    (["what is ca","ca course","chartered accountancy kya","explain ca","ca overview"],
     "What is Chartered Accountancy"),
    (["what is bba","bba course","bba kya","bba degree"],
     "What is BBA course"),
    (["what is bcom","bcom course","bcom degree","b com kya"],
     "What is BCom course"),
    (["scope bba","bba scope","bba career","bba jobs","bba future"],
     "What is the scope of BBA"),
    (["commerce after 12","career commerce","commerce options","commerce careers","commerce degree","bcom bba"],
     "What can I do after 12th Commerce"),

    # Law sub-topics
    (["scope law","law scope","law career","lawyer scope","law future","law india"],
     "What is the scope of law in India"),
    (["what is clat","clat exam","clat kya","law entrance exam","clat overview"],
     "What is CLAT exam"),
    (["best law college","top law college","nlu","law university india"],
     "Which law college is best in India"),
    (["become lawyer","lawyer kaise","ba llb","llb after 12","study law","law after 12","advocate kaise"],
     "How do I become a lawyer"),

    # Design sub-topics
    (["scope design","design scope","design career","design field","design jobs"],
     "What is the scope of design career"),
    (["nid admission","nift admission","get into nid","nift entrance","nid dat","design entrance"],
     "How to get into NID or NIFT"),
    (["salary design","design salary","designer salary","graphic designer earn","ux salary"],
     "What is the salary in design field"),
    (["like drawing","i like art","animation career","graphic design career","ui ux career",
      "which design courses","design course","drawing career"],
     "I like drawing, what can I study"),

    # Computer Science sub-topics
    (["scope computer science","cse scope","cs scope","it scope","software field scope","computer career scope"],
     "What is the scope of computer science after 12th"),
    (["what is bca","bca course","bca kya","bca degree","bca overview"],
     "What is BCA course"),
    (["bca vs btech","btech or bca","bca or btech","should i do bca"],
     "Should I do BTech CSE or BCA"),
    (["programming language","which language learn","learn coding","best language","python java","which programming"],
     "What programming languages should I learn"),
    (["it job","software job","coding job","work in software","degree for it"],
     "Which degree is good for IT jobs"),
    (["like computers","interested in computers","computer lover","enter it after 10","it after 10","i like computer"],
     "I like computers, what should I choose"),

    # Arts sub-topics
    (["scope arts","arts scope","arts career","humanities career","arts after 12","arts jobs"],
     "What is the scope of arts after 12th"),
    (["scope psychology","psychology career","psychologist scope","psychology jobs","counsellor career"],
     "What is the scope of psychology"),
    (["become journalist","journalism career","journalist kaise","journalism after 12","mass communication career"],
     "How to become a journalist"),
    (["become teacher","teacher kaise","teaching career","b ed","ctet","teacher after 12"],
     "How to become a teacher after 12th"),

    # Government jobs sub-topics
    (["become ias","ias officer kaise","how to become ias","civil services kaise","crack upsc"],
     "How to become an IAS officer"),
    (["ias subjects","subjects for ias","upsc syllabus","upsc subjects"],
     "What subjects are needed for IAS"),
    (["join army","army after 12","indian army kaise","nda exam","army recruit","defence after 12"],
     "How to join the Indian Army after 12th"),
    (["become pilot","pilot kaise","aviation career","pilot after 12","commercial pilot"],
     "How to become a pilot after 12th"),
    (["prepare ssc","ssc preparation","crack ssc","ssc study","ssc exam tips"],
     "How to prepare for SSC exam"),
    (["salary government","govt salary","ssc salary","railway salary","government pay"],
     "What is the salary in government jobs"),
    (["government job after 12","govt job 12th","sarkari job 12th","public job 12th","ssc after 12"],
     "What government jobs can I get after 12th"),

    # Hotel / Pharmacy / Abroad
    (["scope hotel management","hotel management career","hospitality scope","hospitality career","chef career"],
     "What is the scope of hotel management"),
    (["hotel management after 12","nchmct","bhm course","hotel management admission","get into hotel"],
     "How to get into hotel management"),
    (["scope pharmacy","pharmacy career","pharmacist scope","pharmacy jobs","b pharma scope"],
     "What is the scope of pharmacy"),
    (["scope studying abroad","study abroad scope","why study abroad","studying outside india","foreign university benefits"],
     "What is the scope of studying abroad"),
    (["apply abroad","foreign college admission","usa college apply","uk college apply","ielts toefl","how to apply abroad"],
     "How to apply for colleges abroad"),

    # General 10th options
    (["after 10th","options after 10","10th ke baad","ssc ke baad","class 10 options","tenth options","passed 10th","10vi nanthar"],
     "What can I do after 10th"),
    (["science stream","take science after","science after 10","is science right","pcm pcb stream"],
     "Should I take Science after 10th"),
    (["commerce stream after 10","what is commerce 10","commerce 10th","commerce stream kya"],
     "What is Commerce after 10th"),
    (["arts stream after 10","what is arts 10","humanities 10th","arts 10th"],
     "What is Arts after 10th"),
    (["diploma after 10","polytechnic course","diploma engineering 10th","technical course after 10"],
     "Can I do a diploma after 10th"),
    (["what is iti","iti course","iti after 10","vocational course","skill course after 10","trade course"],
     "What is ITI"),
    (["low marks","less marks","average marks","fewer marks","marks kam","scored low","poor result 10th"],
     "I got low marks in 10th, what now"),
    (["choose stream","which stream is best","confused 10th","stream selection","science commerce arts"],
     "How do I choose my stream"),
    (["study plan","study routine","how to study 11th","prepare class 11","study tips"],
     "How should I study in 11th"),

    # General 12th options
    (["after 12th","options after 12","12th ke baad","hsc ke baad","class 12 options","passed 12th","12vi nanthar","post 12th"],
     "What can I do after 12th"),

    # Scholarships / College / Confusion
    (["scholarship","financial help","low budget study","fee assistance","student scholarship"],
     "How can I find a scholarship"),
    (["choose college","which college","college admission","compare college","check college before"],
     "How do I choose a college"),
    (["career confusion","dont know career","confused about career","which career suits","discover interest",
      "no idea what to do","confused about future","what career should i"],
     "I don't know what career I want"),
    (["skills to learn","job ready","employability skills","extra skills","improve employability"],
     "Which skills should I learn"),
]


_CAREER_KEYWORDS = {
    "career","stream","course","study","college","exam","degree","job","salary",
    "scope","engineering","medical","commerce","arts","science","neet","jee",
    "10th","12th","ssc","hsc","btech","mbbs","ca","law","design","iti","diploma",
    "doctor","lawyer","engineer","teacher","pilot","army","ias","upsc","nda",
    "bba","bca","bcom","nift","nid","clat","scholarship","admission","eligib",
    "after 10","after 12","pcm","pcb","polytechnic","vocational","government",
    "software","computer","coding","programmer","data","animation","fashion",
    "journalism","psychology","hotel","pharmacy","abroad","university","marks",
    "preparation","prepare","attempt","branch","specialization","subject"
}

def _is_career_related(query):
    q = _norm(query)
    return any(kw in q for kw in _CAREER_KEYWORDS)


def _keyword_route(query, dataset):
    q = _norm(query)
    for kws, target in RULES:
        if any(kw in q for kw in kws):
            tgt = target.lower()
            for entry in dataset:
                if tgt in entry["input"].lower():
                    return entry["response"], True
    return None, False
    q = _norm(query)
    for kws, target in RULES:
        if any(kw in q for kw in kws):
            tgt = target.lower()
            for entry in dataset:
                if tgt in entry["input"].lower():
                    return entry["response"], True
    return None, False


# ── TF-IDF fallback ────────────────────────────────────────────────────────

class TFIDFMatcher:
    def __init__(self, dataset):
        self.dataset = dataset
        N = len(dataset)
        df = Counter()
        self.tokenized = []
        for d in dataset:
            toks = _bigrams(d["input"] + " " + d.get("topic",""))
            self.tokenized.append(toks)
            for t in set(toks): df[t] += 1
        self.idf = {t: math.log((N+1)/(df[t]+1))+1.0 for t in df}

    def _vec(self, tokens):
        tf = Counter(tokens)
        n = max(len(tokens), 1)
        return {t: (tf[t]/n)*self.idf.get(t,1.0) for t in tokens}

    def _cos(self, v1, v2):
        c = set(v1)&set(v2)
        if not c: return 0.0
        dot = sum(v1[t]*v2[t] for t in c)
        return dot/(math.sqrt(sum(x**2 for x in v1.values()))*math.sqrt(sum(x**2 for x in v2.values()))+1e-9)

    def match(self, query):
        qv = self._vec(_bigrams(query))
        best, idx = -1, 0
        for i, toks in enumerate(self.tokenized):
            s = self._cos(qv, self._vec(toks))
            # overlap boost
            orig = set(_tok(self.dataset[i]["input"]))
            s += 0.1 * len(orig & set(_tok(query)))
            if s > best: best, idx = s, i
        return self.dataset[idx]["response"], best


# ── Chatbot class ──────────────────────────────────────────────────────────

class CareerChatbot:
    def __init__(self):
        self.ready = False
        self.dataset = []
        self.matcher = None
        self._load()

    def _load(self):
        if not os.path.exists(DATASET_PATH):
            print("[Chatbot] Dataset not found.")
            return
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        self.matcher = TFIDFMatcher(self.dataset)
        self.ready = True
        print(f"[Chatbot] Ready. {len(self.dataset)} entries loaded.")

    def respond(self, user_input: str) -> str:
        if not self.ready:
            return "Dataset not found. Please check data/career_dataset.json"
        user_input = user_input.strip()
        if not user_input:
            return "Please ask me about career guidance for 10th or 12th students."

        # Check if question is career-related at all
        if not _is_career_related(user_input):
            return ("Sorry, I can only answer questions related to career guidance "
                    "for 10th and 12th students. Please ask about streams, courses, "
                    "entrance exams, or career options.")

        # Layer 1: Keyword sub-topic routing (specific rules)
        response, matched = _keyword_route(user_input, self.dataset)
        if matched:
            return response

        # Layer 2: TF-IDF semantic match
        response, score = self.matcher.match(user_input)

        if score < 0.1:
            return ("Sorry, I can only answer questions related to career guidance "
                    "for 10th and 12th students. Please ask about streams, courses, "
                    "entrance exams, or career options.")

        return response


_instance = None

def get_chatbot():
    global _instance
    if _instance is None:
        _instance = CareerChatbot()
    return _instance


if __name__ == "__main__":
    bot = get_chatbot()
    print("Career Guidance Chatbot (type 'quit' to exit)")
    print("-"*50)
    while True:
        u = input("You: ").strip()
        if u.lower() in ("quit","exit","bye"):
            break
        print(f"Bot: {bot.respond(u)}\n")
