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

# A fourth inventory. Sets A-C are now the training corpus for the fine-tuned
# member of rounds 150-151; this one is what its next generation is judged on
# (round 152), so that the loop's own data-scaling curve can be measured
# without reusing an evaluation set.
FOURTH_CUES = {
    "refund": [
        "give back what I was charged",
        "I want the amount credited back",
        "please undo my purchase",
        "the money has to be returned",
        "I am requesting my funds back",
        "I was overcharged and want it corrected",
    ],
    "chargeback": [
        "I have raised a case with my bank",
        "the card company is taking this up",
        "I am contesting this with my provider",
        "the payment is being reversed by the network",
        "my issuer has been asked to claw it back",
        "this has gone to the dispute team at my bank",
    ],
    "exchange": [
        "can you send the larger variant",
        "I want to change it for a different one",
        "please replace it with another",
        "I would like to trade this",
        "send me the alternative instead",
        "I want to switch to the other model",
    ],
    "complaint": [
        "this is a shambles",
        "the whole thing has been dreadful",
        "I am furious",
        "I have been let down badly",
        "the experience has been miserable",
        "your handling has been disgraceful",
    ],
    "praise": [
        "the whole thing was flawless",
        "you have been marvellous",
        "what a lovely experience",
        "I am so grateful for this",
        "the quality is top notch",
        "my compliments to the team",
    ],
    "greeting": [
        "hello hello",
        "hi there, how are things",
        "morning, hope you are well",
        "just popping in to say hello",
        "greetings and good wishes",
        "hellooo",
    ],
    "login_issue": [
        "I am unable to log in",
        "my password is refused",
        "the site keeps me out",
        "I cannot reach my profile",
        "the verification step fails",
        "my access has been blocked",
    ],
    "invoice": [
        "please send the bill for March",
        "I need the tax document",
        "where is my billing paperwork",
        "send the accounts paperwork over",
        "I want the itemised charges",
        "please issue the outstanding bill",
    ],
}

CUE_SETS["d"] = FOURTH_CUES


# A fifth inventory, for the split-E evaluation (task 007): 48 fresh
# phrasings, none of which appear verbatim in A, B, C or D, so a model or
# combiner fitted on any earlier inventory is measured on genuinely new
# wording again.
FIFTH_CUES = {
    "refund": [
        "I need the transaction refunded to my bank account",
        "would you return the purchase price",
        "please compensate me for this charge",
        "issue me a credit for the amount paid",
        "return the money taken from my card",
        "I am looking for a repayment of the fee",
    ],
    "chargeback": [
        "I am disputing this payment through my bank",
        "my card provider has opened a dispute",
        "the issuer has begun a reversal case",
        "my bank has started a challenge against the payment",
        "this charge is now in the card network's dispute process",
        "my financial institution is reversing the amount",
    ],
    "exchange": [
        "could this purchase be exchanged for another style",
        "can I exchange the order for another model",
        "please let me choose a replacement variant",
        "I need to swap the item I received",
        "could you trade this item for the one I wanted",
        "send a different version instead of this one",
    ],
    "complaint": [
        "I am upset with the way this was dealt with",
        "the support I received was disappointing",
        "what a frustrating ordeal this turned into",
        "I have never had such poor help",
        "your service fell far below expectations",
        "nothing in this process worked well",
    ],
    "praise": [
        "the assistance I got was outstanding",
        "I am delighted with how this turned out",
        "your representative handled everything perfectly",
        "this was an excellent piece of support",
        "many thanks for the great help",
        "the service could not have been better",
    ],
    "greeting": [
        "good evening, I hope you are doing well",
        "hello, I just wanted to check in",
        "greetings from Seoul",
        "hey there, pleased to make your acquaintance",
        "a quick hello from my side",
        "good day to you",
    ],
    "login_issue": [
        "I cannot get past the sign-in screen",
        "my password is not being accepted",
        "the account recovery code is missing",
        "authentication is not letting me through",
        "the platform keeps rejecting my login",
        "my access credentials are not working",
    ],
    "invoice": [
        "please send the invoice for my order",
        "please supply a record of the charges",
        "can you send the payment summary",
        "kindly forward the receipt for this transaction",
        "I would like the billing information",
        "where can I download the tax invoice",
    ],
}

CUE_SETS["e"] = FIFTH_CUES


# A sixth inventory (set G), for the distillation round (task 020): the eight
# original intents again, with 48 phrasings that appear in no earlier
# inventory.  The cluster's strongest zero-shot member (8B) labels rows drawn
# from this set and the 4B filters them by agreement, so the phrasings have to
# be new to the fine-tuned lineage or the teacher would be rewarded for
# memorised wording.
SIXTH_CUES = {
    "refund": [
        "I would like the amount put back",
        "please reverse the payment I made",
        "the fee needs to be returned to me",
        "give me back the money I spent",
        "I want to be made whole for this charge",
        "credit the payment back to my account",
    ],
    "chargeback": [
        "I have opened a dispute with my bank over this",
        "my card issuer is taking the money back",
        "the payment is being reversed through the dispute process",
        "I asked my bank to challenge the charge",
        "this is now a formal dispute with my provider",
        "the network is forcing a reversal of this transaction",
    ],
    "exchange": [
        "I would like to swap it for the newer one",
        "please send me the other model instead",
        "can this be traded for a different variant",
        "I want to replace this with another version",
        "let me exchange it for the model I wanted",
        "send out a substitute unit please",
    ],
    "complaint": [
        "this has been a truly awful experience",
        "I am extremely unhappy with the service",
        "the way this was handled was poor",
        "I have had a terrible time with this",
        "your team let me down badly",
        "this is far below the standard I expect",
    ],
    "praise": [
        "your support was absolutely brilliant",
        "I am thrilled with how this went",
        "the team did a wonderful job",
        "this is easily the best service I have had",
        "many thanks, you were superb",
        "everything in this was excellent",
    ],
    "greeting": [
        "hello, I hope your day is going well",
        "good morning from my side",
        "just dropping a line to say hello",
        "hi there, nice to meet you",
        "greetings, hope all is well with you",
        "hey, just a quick hello",
    ],
    "login_issue": [
        "I cannot get into my account at all",
        "the password I set is not accepted",
        "I keep getting thrown back to the sign-in page",
        "my account appears to be locked",
        "the one-time code never turns up",
        "signing in fails no matter what I try",
    ],
    "invoice": [
        "I need the bill for my last order",
        "please send me the tax paperwork",
        "where can I find the invoice for this",
        "could I get an itemised statement",
        "I would like a copy of the charges",
        "send the billing document when you can",
    ],
}

CUE_SETS["g"] = SIXTH_CUES


# A seventh inventory (set H), for the fresh evaluator of task 024.  The
# distillation loop has been scored against D/E/F for several generations, so
# its selection pressure is compounding; H is a new 48-phrasing inventory over
# the same eight original intents, again overlapping no earlier split
# verbatim, so that admission is measured on wording no member has seen.
SEVENTH_CUES = {
    "refund": [
        "I want the money I paid returned",
        "give me my cash back for this order",
        "send the amount I was charged back to me",
        "I would like my outlay refunded in full",
        "put the sum back onto my card",
        "kindly restore the funds you took",
    ],
    "chargeback": [
        "my bank is pulling the payment back",
        "I have asked my card company to reverse this",
        "this charge is being disputed by my issuer",
        "the card scheme is clawing the money back",
        "I filed a dispute over this charge with my bank",
        "my issuer has opened a case against this transaction",
    ],
    "exchange": [
        "can I get this swapped for the larger one",
        "I would rather have a different item",
        "send a replacement for what arrived",
        "I want to trade this for the newer version",
        "could you substitute the one I got",
        "let me pick a different variant of this",
    ],
    "complaint": [
        "this has been a rotten experience",
        "your people handled this dreadfully",
        "I am livid at how long this took",
        "the service was appalling from start to finish",
        "what a disappointing outcome",
        "nobody appeared to care for my case",
    ],
    "praise": [
        "the agent did a splendid job",
        "I am over the moon with this",
        "what a fantastic result",
        "your team deserves a lot of credit",
        "this went brilliantly from start to finish",
        "thanks a million for the superb work",
    ],
    "greeting": [
        "howdy from Texas",
        "good evening over there",
        "hi, just saying a quick hello",
        "hey, I hope this finds you well",
        "greetings from my corner of the world",
        "just a friendly hello to start",
    ],
    "login_issue": [
        "I am shut out of my account",
        "the sign-in screen will not let me through",
        "my password is being denied",
        "I keep failing the two-factor step",
        "the app says my session is invalid",
        "I cannot reach my profile at all",
    ],
    "invoice": [
        "send me the bill for this order",
        "I need the tax paperwork for my accounts",
        "where is the receipt for the purchase",
        "could I get an itemised list of charges",
        "send the billing summary over",
        "I require the VAT record",
    ],
}

CUE_SETS["h"] = SEVENTH_CUES

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


def longest_shared_run(left_cues: list[str], right_cues: list[str]) -> int:
    """Longest run of words that appears in a cue from both inventories."""
    longest = 0
    for left in left_cues:
        for right in right_cues:
            words_left = left.lower().split()
            words_right = right.lower().split()
            for size in range(1, min(len(words_left), len(words_right)) + 1):
                runs = {" ".join(words_left[start:start + size])
                        for start in range(len(words_left) - size + 1)}
                if any(" ".join(words_right[start:start + size]) in runs
                       for start in range(len(words_right) - size + 1)):
                    longest = max(longest, size)
    return longest


def cross_inventory_report(target_name: str) -> None:
    """Exact-string overlap and longest shared run of one cue set vs the rest.

    A2 is not a separate inventory: ``big2`` (seed 2, dev/test) reuses cue set
    A, so the five named inventories A-E are the full comparison space.
    """
    target = CUE_SETS[target_name]
    target_strings = {cue.lower() for group in target.values()
                      for cue in group}
    target_total = sum(len(group) for group in target.values())
    print(f"cross-inventory check for cue set {target_name!r}: "
          f"{target_total} cues, {len(target_strings)} unique")
    grand_total = 0
    for other_name in sorted(CUE_SETS):
        if other_name == target_name:
            continue
        other = CUE_SETS[other_name]
        other_strings = {cue.lower() for group in other.values()
                         for cue in group}
        exact = target_strings & other_strings
        grand_total += len(exact)
        print(f"  vs {other_name}: exact overlap {len(exact)}")
    print(f"  total exact overlap across all other sets: {grand_total}")
    for label in sorted(target):
        longest = 0
        for other_name in sorted(CUE_SETS):
            if other_name == target_name:
                continue
            for right in CUE_SETS[other_name].get(label, []):
                longest = max(longest,
                              longest_shared_run(target[label], [right]))
        print(f"  {label:14s} longest shared run: {longest} words")


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
                             "appear in no earlier split (round 122); "
                             "c/d: later held-out inventories; "
                             "e: fresh inventory for split E (task 007); "
                             "g: distillation inventory (task 020); "
                             "h: fresh evaluator for split H (task 024)")
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
    if args.cues == "e":
        cross_inventory_report("e")
    if args.cues == "g":
        cross_inventory_report("g")
    if args.cues == "h":
        cross_inventory_report("h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
