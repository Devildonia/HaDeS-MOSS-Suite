"""Training prompts for turbo v1.0: the 1024 prompts of turbo v0.9 (BeTa) with 230 of their no-person prompts replaced by people
whose ethnicity is not given, so about a quarter of the prompts, and of the regression pairs, ask for a person and leave the
appearance to the teacher.

    python tools/moss_turbo_v10_prompts.py --source <v0.9 run>/prompts.json --out <file>.json

Why: in turbo v0.9 only 29 of 1024 prompts showed people with no ethnicity (86 of 115 people prompts named one), and the turbo
drifted to East Asian faces on the neutral café prompt (about 12 / 16, Base about 2-4, tools/RESULTS.md). DMD follows the
teacher's dominant mode; regression pairs of the teacher's own varied people on neutral prompts are what anchors that variety.
"""
import argparse
import json
import random
import re

REPLACE = 230
PERSON = re.compile(r"\b(woman|man|women|men|girl|boy|person|people|child|children|surfer|florist|chef|farmer|portrait|dancer|"
                    r"musician|worker|fisherman|grandmother|grandfather|student|couple|family|baker|rider|climber|artist)\b", re.I)
ETHNIC = re.compile(r"\b(Asian|African|European|Black|white|Latina|Latino|Hispanic|Arab|Indian|Japanese|Chinese|Korean|"
                    r"Scandinavian|Nordic|Caucasian)\b", re.I)
# photographic styles first: the bias was counted on photographs
STYLES = ["Candid street photograph", "35mm film photograph with grain", "Editorial portrait photograph, natural light",
          "Documentary photograph", "Cinematic film still, shallow depth of field", "Black and white street photograph",
          "Smartphone snapshot", "Studio portrait photograph, softbox light", "Watercolor painting with loose washes",
          "Oil painting with visible brushstrokes", "Anime illustration with clean line art and cel shading", "Comic book panel with halftone dots"]
STYLE_WEIGHTS = [4, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1]
SUBJECTS = ["a woman", "a man", "a young woman", "a young man", "a middle-aged woman", "a middle-aged man", "an elderly woman",
            "an elderly man", "two friends", "a couple", "a group of coworkers", "a woman and a man", "three people",
            "a mother and her adult son", "a father and his adult daughter"]
ACTIONS = ["sitting in a café with a cup of coffee", "reading a book on a park bench", "waiting at a bus stop", "cooking in a small kitchen",
           "shopping at an outdoor market", "riding a bicycle through the city", "working at a laptop in an office",
           "walking a dog on a beach", "laughing at a dinner table", "standing on a train platform", "browsing in a bookshop",
           "carrying groceries up a staircase", "painting at an easel", "jogging along a river path", "sitting on the steps of a building",
           "playing the guitar on a sofa", "hiking on a mountain trail", "looking out of a rainy window", "repairing a bicycle in a garage",
           "watering plants on a balcony"]
LIGHTS = ["in soft morning light", "at golden hour", "under overcast daylight", "in bright midday sun", "at night under street lights",
          "in warm indoor lamplight", "in blue hour twilight", "in light rain"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=10)
    args = p.parse_args()
    prompts = json.load(open(args.source, encoding="utf-8"))
    rng = random.Random(args.seed)
    no_person = [i for i, t in enumerate(prompts) if not PERSON.search(t)]
    drop = set(rng.sample(no_person, REPLACE))
    new = set()
    while len(new) < REPLACE:
        style = rng.choices(STYLES, STYLE_WEIGHTS)[0]
        new.add(f"{style}: {rng.choice(SUBJECTS)} {rng.choice(ACTIONS)}, {rng.choice(LIGHTS)}.")
    out = sorted([t for i, t in enumerate(prompts) if i not in drop] + sorted(new))
    assert len(out) == len(prompts) and not any(ETHNIC.search(t.split(": ", 1)[1]) for t in new)  # the subject, not the style
    neutral = sum(1 for t in out if PERSON.search(t) and not ETHNIC.search(t.split(": ", 1)[-1])
                  and not re.search(r"\b(Brazilian|French|Maori|Vietnamese|Inuit|Italian|Peruvian|Filipino|Mexican|Nigerian|Greek)\b", t))
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{len(out)} prompts, {neutral} with people and no ethnicity or nationality", flush=True)


if __name__ == "__main__":
    main()
