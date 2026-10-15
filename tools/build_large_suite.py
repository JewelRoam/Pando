"""Generate a large labeled choice suite with unambiguous by-construction labels.

Round 117 showed that 27 labeled tasks cannot separate a 52% combiner from a
63% one, so the ensemble question needs a bigger suite. The existing pool
builder deliberately emits *unlabeled* rows (labels come from the live
majority for RLCD), so evaluation needs its own generator.

Label validity is the hard constraint, so difficulty comes from three things
that cannot make a row ambiguous:

  - eight deliberately confusable intents (refund / chargeback / exchange /
    complaint / praise / greeting / login_issue / invoice);
  - heavy paraphrase, including colloquial and typo'd surface forms;
  - neutral filler sentences that carry no class vocabulary at all - asserted
    programmatically, not trusted.

Every row has exactly one class cue, so `expected` is known by construction.
The generator never sees a model.

Usage:
    python tools/build_large_suite.py --count 240 --seed 1 --out assets/big_dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTENT_CUES = {
    "refund": [
        "I want my money back", "please refund my card", "give me a refund now",
        "I would like to be reimbursed", "return my payment please",
        "pls refund my money asap", "I want the money returned to me",
        "credit my account back please", "send my payment back",
    ],
    "chargeback": [
        "I am filing a dispute with my bank", "I will dispute this charge",
        "my bank will reverse this transaction", "initiate a chargeback",
        "I told my bank to contest the payment",
        "the card issuer is reversing it",
    ],
    "exchange": [
        "can I swap this for another size", "I want a replacement item instead",
        "exchange it for the other colour", "send me a different unit",
        "I would rather have the newer model", "swap the defective one out",
    ],
    "complaint": [
        "this is completely unacceptable", "terrible service, honestly",
        "I am very disappointed", "worst experience I have had",
        "your support has been awful", "this whole thing is a mess",
        "I am frustrated with how this was handled",
    ],
    "praise": [
        "your team did a fantastic job", "thanks, excellent work",
        "the new release is great", "support was superb today",
        "I am really impressed by the service", "keep it up, brilliant",
    ],
    "greeting": [
        "hello there", "good morning", "hi, how are you",
        "greetings from Berlin", "hey, hope you are well",
        "just saying hi",
    ],
    "login_issue": [
        "I cannot sign in", "my password keeps getting rejected",
        "the login page just loops", "my account is locked out",
        "two-factor codes never arrive", "sign in fails every time",
        "I am stuck at the password screen",
    ],
    "invoice": [
        "I need a copy of the invoice", "where is my receipt",
        "please send the billing statement", "I need the VAT document",
        "send the tax receipt for last month", "I need the paperwork for accounting",
    ],
}

FILLERS = [
    "It has been two days.", "This is the second time I am writing.",
    "Please advise.", "I have been a customer for years.",
    "I am not sure this is the right place to ask.", "Thank you.",
    "Let me know what you need from me.", "I am on a deadline.",
    "Sorry for the trouble.", "It is quite urgent.",
    "I attached a photo.", "I can provide details if useful.",
    "No rush, whenever you can.", "I appreciate your help.",
    "This is my first message about it.", "I tried the website already.",
    "I am writing from my phone.", "The reference number is in the email.",
]

# Round 122 found that the two seeds shared their phrasing inventory: 100 of
# the test split's 102 distinct first sentences appeared verbatim in dev, so a
# learner fitted on dev could recognise wording instead of generalising. This
# inventory is the second half of the fix - a set of phrasings that appears in
# no earlier split, so a suite built from it measures new wording.
HELDOUT_CUES = {
    "refund": [
        "kindly put the money back on my card", 
        "I paid twice and want one of the payments undone",
        "could you undo what I was charged for",
        "the charge should never have gone through, take it off",
        "please release the funds back to me",
        "I would like my outlay returned",
    ],
    "chargeback": [
        "I have escalated this with my card provider",
        "the transaction is being contested through my bank",
        "my credit card company is looking into the payment",
        "I started a reversal with my issuer",
        "this is going through the dispute process now",
        "the payment network has been asked to undo it",
    ],
    "exchange": [
        "may I trade this for the larger model",
        "can you send an alternative item",
        "I got the wrong size and want to switch it",
        "let me trade this in for a different one",
        "I would prefer a substitute product",
        "the one I received is the wrong variant, I want the other",
    ],
    "complaint": [
        "this has been a dreadful experience",
        "honestly, the whole thing was handled poorly",
        "I am angry that this took so long",
        "nobody helped me and it shows",
        "the standard of service was not acceptable",
        "I expected much better from you",
    ],
    "praise": [
        "you people are wonderful",
        "that was a superb experience",
        "the crew went above and beyond",
        "five stars, honestly",
        "I could not be happier with the result",
        "whoever handled it deserves a raise",
    ],
    "greeting": [
        "hey there, just popping in",
        "top of the morning to you",
        "howdy, hope all is well",
        "this is just a hello from Lisbon",
        "evening, hope you had a good day",
        "waving hello to everyone",
    ],
    "login_issue": [
        "I am unable to access my account",
        "the security code does not work",
        "authentication keeps failing",
        "I am locked out of my profile",
        "the app rejects my credentials",
        "I never get the verification code",
    ],
    "invoice": [
        "could you send me the bill for last month",
        "I require the VAT paperwork",
        "please issue an itemised bill",
        "where can I find my payment documents",
        "I need the accounts document for my records",
        "can you send over the statement of charges",
    ],
}

CUE_SETS = {"a": INTENT_CUES, "b": HELDOUT_CUES}

# A third inventory, so that a model can be trained on one wording set, a
# combiner fitted on a second, and the whole thing evaluated on a third that
# neither has seen (round 141). With two sets the two roles collide: a text
# model trained on set B is out-of-sample on A but contaminates B.
THIRD_CUES = {
    "refund": [
        "put the charge back on my card",
        "I want a full reversal of the payment",
        "please return what I paid",
        "the money should come back to me",
        "undo the transaction and give me my funds",
        "I am asking for my cash again",
    ],
    "chargeback": [
        "I have taken this up with my card company",
        "the bank is contesting it for me",
        "I filed a claim with my issuer",
        "the payment is being clawed back by the scheme",
        "my provider is disputing the amount",
        "this follows the card scheme's rules",
    ],
    "exchange": [
        "can I get a different model instead",
        "swap it for a bigger one",
        "I want the other colourway",
        "send a replacement unit",
        "trade mine for another",
        "I would like a substitute",
    ],
    "complaint": [
        "this is really poor",
        "the service was shocking",
        "I am unhappy with all of it",
        "nothing has worked at all",
        "the handling was a disgrace",
        "I have lost patience",
    ],
    "praise": [
        "you did a marvellous job",
        "the team was excellent",
        "very pleased with this",
        "outstanding support",
        "that was a delight",
        "absolutely spot on",
    ],
    "greeting": [
        "hiya, just checking in",
        "good afternoon",
        "hello from Madrid",
        "just dropping by to say hi",
        "hey, nice to meet you",
        "salutations",
    ],
    "login_issue": [
        "I cannot get into my account",
        "my credentials are rejected",
        "the sign-in keeps looping",
        "my profile is inaccessible",
        "the code never arrives",
        "I am blocked at the entrance screen",
    ],
    "invoice": [
        "please send the bill again",
        "I need the tax paperwork",
        "where do I find my billing summary",
        "send the tax documentation",
        "I want the itemised account",
        "please issue the statement for last month",
    ],
}

CUE_SETS["c"] = THIRD_CUES

QUESTION = "intent"


def cue_words(table: dict) -> set:
    owners: dict[str, set] = {}
    for label, group in table.items():
        for cue in group:
            for raw in cue.split():
                word = raw.strip(".,!?").lower()
                if len(word) > 4:
                    owners.setdefault(word, set()).add(label)
    return {word for word, labels in owners.items() if len(labels) == 1}


def assert_fillers_are_neutral() -> None:
    """A filler must not contain vocabulary that identifies one class.

    A word shared by several classes ("please", "send") cannot disambiguate a
    row, so only words appearing in exactly one class count as class
    vocabulary. Every cue set is checked: a filler that is neutral for one
    inventory and distinctive for another would leak into whichever suite
    used the second one.
    """
    for name, table in CUE_SETS.items():
        distinctive = cue_words(table)
        for filler in FILLERS:
            tokens = {raw.strip(".,!?").lower() for raw in filler.split()}
            overlap = tokens & distinctive
            if overlap:
                raise SystemExit(
                    f"filler {filler!r} contains class vocabulary {overlap} "
                    f"from cue set {name!r}")


def overlap_report() -> None:
    """How much wording cue set b still shares with cue set a, per class.

    Paraphrasing a concept reuses some of its words, so this is a diagnostic
    rather than a guarantee: the number printed is the longest run of words
    that appears in a cue from both sets. A suite built from set b is only as
    "held out" as this makes it.
    """
    for label in sorted(INTENT_CUES):
        longest = 0
        for left in INTENT_CUES[label]:
            for right in HELDOUT_CUES[label]:
                words_left = left.lower().split()
                words_right = right.lower().split()
                for size in range(1, min(len(words_left), len(words_right)) + 1):
                    runs = {" ".join(words_left[start:start + size])
                            for start in range(len(words_left) - size + 1)}
                    if any(" ".join(words_right[start:start + size]) in runs
                           for start in
                           range(len(words_right) - size + 1)):
                        longest = max(longest, size)
        print(f"  {label:14s} longest shared run: {longest} words")


def build(count: int, seed: int, exclude: set, table: dict) -> list[dict]:
    rng = random.Random(seed)
    labels = sorted(table)
    cues = {label: list(table[label]) for label in labels}
    for group in cues.values():
        rng.shuffle(group)
    tasks = []
    index = 0
    while len(tasks) < count:
        label = labels[index % len(labels)]
        index += 1
        if not cues[label]:
            cues[label] = list(table[label])
            rng.shuffle(cues[label])
        cue = cues[label].pop()
        filler = rng.choice(FILLERS)
        state = (f"User says: {cue}." if rng.random() < 0.5
                 else f"User says: {cue}. {filler}")
        if rng.random() < 0.3:
            state = state.replace("User says: ", "Message: ")
        key = (state, label)
        if key in exclude:
            continue
        exclude.add(key)
        tasks.append({
            "task_id": f"big{seed}-{len(tasks) + 1:04d}",
            "checkpoint": "english",
            "qtype": "choice",
            "state": state,
            "question": QUESTION,
            "candidates": labels,
            "expected": label,
        })
    rng.shuffle(tasks)
    for position, task in enumerate(tasks, start=1):
        task["task_id"] = f"big{seed}-{position:04d}"
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=str(ROOT / "assets" / "big_dev.jsonl"))
    parser.add_argument("--cues", choices=sorted(CUE_SETS), default="a",
                        help="a: the original inventory; b: phrasings that "
                             "appear in no earlier split (round 122)")
    args = parser.parse_args()

    assert_fillers_are_neutral()
    tasks = build(args.count, args.seed, set(), CUE_SETS[args.cues])
    target = Path(args.out)
    target.write_text("\n".join(json.dumps(task, ensure_ascii=False)
                                for task in tasks) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["expected"]] = counts.get(task["expected"], 0) + 1
    print(f"wrote {target.name}: {len(tasks)} tasks, seed {args.seed}, "
          f"cue set {args.cues}")
    print(f"class balance: {dict(sorted(counts.items()))}")
    print(f"unique states: {len({task['state'] for task in tasks})}")
    if args.cues == "b":
        overlap_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
