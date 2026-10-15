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

QUESTION = "intent"


def assert_fillers_are_neutral() -> None:
    """A filler must not contain vocabulary that identifies one class.

    A word shared by several classes ("please", "send") cannot disambiguate a
    row, so only words appearing in exactly one class count as class
    vocabulary.
    """
    owners: dict[str, set] = {}
    for label, group in INTENT_CUES.items():
        for cue in group:
            for raw in cue.split():
                word = raw.strip(".,!?").lower()
                if len(word) > 4:
                    owners.setdefault(word, set()).add(label)
    distinctive = {word for word, labels in owners.items() if len(labels) == 1}
    for filler in FILLERS:
        tokens = {raw.strip(".,!?").lower() for raw in filler.split()}
        overlap = tokens & distinctive
        if overlap:
            raise SystemExit(
                f"filler {filler!r} contains class vocabulary {overlap}")


def build(count: int, seed: int, exclude: set) -> list[dict]:
    rng = random.Random(seed)
    labels = sorted(INTENT_CUES)
    cues = {label: list(INTENT_CUES[label]) for label in labels}
    for group in cues.values():
        rng.shuffle(group)
    tasks = []
    index = 0
    while len(tasks) < count:
        label = labels[index % len(labels)]
        index += 1
        if not cues[label]:
            cues[label] = list(INTENT_CUES[label])
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
    args = parser.parse_args()

    assert_fillers_are_neutral()
    tasks = build(args.count, args.seed, set())
    target = Path(args.out)
    target.write_text("\n".join(json.dumps(task, ensure_ascii=False)
                                for task in tasks) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["expected"]] = counts.get(task["expected"], 0) + 1
    print(f"wrote {target.name}: {len(tasks)} tasks, seed {args.seed}")
    print(f"class balance: {dict(sorted(counts.items()))}")
    print(f"unique states: {len({task['state'] for task in tasks})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
