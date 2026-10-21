TITLE = r"""<definition_of_good>
Definition of "Good" — Policy SFT Dataset
</definition_of_good>"""

CONTEXT = r"""<context>
Context

The task is **STT Refinement**: a speech-to-text (ASR) transcript is noisy and broken.
The policy model must clean it into a production-grade technical document.

The input is always a `raw_asr` string.
The output is always a clean `refined_text` string.

The end user is a **technical professional** who uses a speech-to-text application to prompt an AI model — for questions, commands, or technical explanations. The goal is to make their spoken words clean and usable, without losing their intent or thoughts.
</context>"""

DIMENSIONS_HEADING = r"""<dimensions>
The 4 Dimensions of "Good"
</dimensions>"""

FAITHFULNESS = r"""<faithfulness>
1. Faithfulness (score 9)

The output preserves every technical fact from the raw input. It does **not** add, remove, or swap any technical entity.

The model must understand the full context and use semantic reasoning to fill in what is most likely correct — like filling in a blank based on everything around it. When a word or phrase is ambiguous, the model must infer the most contextually plausible resolution, not guess randomly.

**Rules:**
- Keep all numbers, commands, file names, error codes, and parameters exactly yeah this part is important when you see one and two in raw_asr text better is model convert it to 1 and 2 i think you understand. .
- Never invent a detail that was not in the raw input.
- Never swap one technical term for another (e.g., `VPN` must not become `VPC`).
- When uncertain, preserve the raw form rather than hallucinate.

**Example of faithfulness failure — do NOT do this:**
```
RAW:  "misconfigured v p n routing rule"
BAD:  "misconfigured vpc routing rule"   ← WRONG: swapped entity
GOOD: "misconfigured VPN routing rule"   ← CORRECT
```
</faithfulness>"""

NOISE_REMOVED = r"""<noise_removed>
2. Noise Removed (score 9)

The output contains zero disfluencies.

Humans use pause words when they think while speaking — `um`, `uh`, `so`, `basically`, `like`, `you know`. These have no informational value and must be removed. However, some words that look like fillers do carry meaning depending on context (e.g., `so` as a logical connector). The model must understand intent, not blindly strip words.

The goal is: make the raw ASR cleaner without removing the user's actual information, thoughts, or feeling of what they were trying to say.

**Remove:**
- Filler words: `um`, `uh`, `so`, `basically`, `like`, `you know`, `I mean`, `right`
- False starts: `so um basically when we...` → `When we...`
- Social artifacts: `thanks for watching`, `please subscribe`, `okay so`

**Do NOT remove:**
- Repeated words that are intentional emphasis (e.g., `Stop, stop!`)
- Technical fillers that are part of a command or name
- Words that connect logical clauses with real meaning
</noise_removed>"""

ERRORS_FIXED = r"""<errors_fixed>
3. Errors Fixed (score 9)

The output resolves all acoustic errors and cut-words.

There are many sources of noise between the user's mouth and the final ASR text:
1. The user's English is not fluent.
2. The user is not familiar with the technical words they are trying to say.
3. Microphone quality is poor — the user is far from the mic, or there is fan noise and environmental noise.
4. The ASR model itself is not strong enough.
5. A good ASR model running on a weak device (low resources) causes word cuts and dropouts.
6. Other causes are possible — the principle is the same: fix what can be confidently resolved; preserve what cannot.

**Types of errors to fix:**

| Error type | Raw ASR example | Correct output |
|---|---|---|
| Cut-word (morphological) | `terminat`, `configur`, `allocat`, `migrat`, `connecti` | `terminate`, `configure`, `allocate`, `migrate`, `connection` |
| Phonetic decomposition | `coober netties` | `Kubernetes` |
| Phonetic decomposition | `et see dee` | `etcd` |
| Phonetic decomposition | `in grass engine x` | `ingress-nginx` |
| Phonetic decomposition | `Tee sig keys` | `TSIG keys` |
| Phonetic decomposition | `p ninety-nine` | `p99` |
| Phonetic decomposition | `r p c` | `RPC` |
| Phonetic decomposition | `c p u` | `CPU` |
| Phonetic decomposition | `tensor radio tower` | `TensorRT` |
| Phonetic decomposition | `cube ctl` | `kubectl` |
| Phonetic decomposition | `cube system` | `kube-system` |
| Phonetic decomposition | `dynamo d b` | `DynamoDB` |
| Phonetic decomposition | `lease ass the meter file` | `Please pass the parameter file` |
| Spelled-out number | `four thousand`, `one hundred` | `4000`, `100` |
| Spelled-out product | `eight a one hundred gpus` | `eight A100 GPUs` |

**Key principle:** When the raw ASR sounds like a known technical term, resolve it. When uncertain, preserve the raw form rather than hallucinate.
</errors_fixed>"""

FORMATTING = r"""<formatting>
4. Formatting (score 9)

Apply structure that fits the length and content of the utterance. Do **not** apply the same structure to every output regardless of length. This is the most common failure.

When the raw ASR is long, the model must understand the context and split the text into paragraphs by topic. Use `` ` `` backticks, quotes `""`, and **bold** only where they genuinely add clarity — not everywhere just to show the model can do it.

**On `---` dividers:** Use them very sparingly. A `---` divider is only justified when two major topics in a long utterance are so distinct that a blank line alone is not enough. Less is better. `\n\n` is almost always sufficient.

Short utterances (< 25 words)
- Use a single sentence.
- Use inline backticks for commands and code identifiers.
- Do **not** add `---` dividers, bullet lists, or paragraph breaks.

```
RAW:  "so um basically run cube ctl get pods to check the status"
GOOD: "Run `kubectl get pods` to check the status."
BAD:  "Run `kubectl get pods` to check the status.\n\n---"   ← WRONG: divider spam
```

Medium utterances (25–90 words)
- Use 1–3 sentences with natural paragraph flow.
- Use backticks for code identifiers, file names, and CLI commands.
- Use `\n\n` to separate distinct topics.
- Do **not** use `---` — it is not needed at this length.

Long utterances (90+ words)
- Break into logical paragraphs separated by `\n\n`.
- Use `---` only for genuinely major topic breaks (maximum: once or twice in the entire output).
- Use numbered lists or bullet lists only when the content is an explicit sequence of steps.
- Apply backticks to all commands, flags, file names, and identifiers.
</formatting>"""

EXAMPLES = r"""<examples>
What a Perfect SFT Sample Looks Like

```json
{
  "raw_asr": "the root cause of the outage was an unindexed query on the audit logs table which caused database c p u utilization to reach one hundred percent and exhausted the connection pool within two minutes",
  "refined_text": "The root cause of the outage was an unindexed query on the audit logs table, which caused database CPU utilization to reach 100% and exhausted the connection pool within two minutes."
}
```

```json
{
  "raw_asr": "okay so first um check the ingress controller in cube system by running cube ctl get pods dash n in grass engine x then look at the error logs for any five hundred two bad gateway responses and second if the pod is crash looping restart the deployment with cube ctl rollout restart deployment in grass engine x dash controller and verify the endpoints thanks for watching please subscribe",
  "refined_text": "First, check the ingress controller in `kube-system` by running `kubectl get pods -n ingress-nginx`, then look at the error logs for any 502 Bad Gateway responses. Second, if the pod is crash looping, restart the deployment with `kubectl rollout restart deployment ingress-nginx-controller` and verify the endpoints."
}
```
</examples>"""

FULL_DEFINITION = "\n\n".join(
    [
        TITLE,
        CONTEXT,
        DIMENSIONS_HEADING,
        FAITHFULNESS,
        NOISE_REMOVED,
        ERRORS_FIXED,
        FORMATTING,
        EXAMPLES,
    ]
)
