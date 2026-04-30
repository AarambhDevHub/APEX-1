# 03 — Tokenizer: Turning Text Into Numbers

> **Difficulty:** ⭐⭐☆☆☆ Beginner-Intermediate  
> **Source file:** `apex/tokenizer/tokenizer.py`  
> **You will learn:** How text is split into tokens, what special tokens are, and how token types power SFT training.

---

## 1. Why Computers Need Numbers, Not Words

Neural networks cannot work with text directly. They only understand numbers. So before the model can process "Hello, world!", we must convert it into a list of integers:

```
"Hello, world!"  →  [9906, 11, 1917, 0]
```

These integers are called **token IDs**. The job of converting text to token IDs (and back) belongs to the **tokenizer**.

---

## 2. What Is a Token?

A token is a piece of text — it could be a whole word, part of a word, a punctuation mark, or even a single character.

**Example with BPE:**
```
"unhappiness"  →  ["un", "happin", "ess"]   (3 tokens)
"cat"          →  ["cat"]                   (1 token)
"2024"         →  ["20", "24"]              (2 tokens)
"print("       →  ["print", "("]           (2 tokens)
```

APEX-1 uses a vocabulary of **151,643 tokens** (the same as Qwen3). This large vocabulary means:
- Common English words are 1 token
- Common Chinese characters are 1 token
- Code keywords are usually 1 token
- Very rare words are split into 2–5 sub-word tokens

---

## 3. Byte-Pair Encoding (BPE) — How the Vocabulary Is Built

BPE is an algorithm that builds the vocabulary by repeatedly merging the most common pairs of characters.

**Step-by-step example:**

Start with character-level tokenisation:
```
corpus: "low low lowest lowest newer newer"
initial tokens: [l, o, w, e, s, t, n, e, w, e, r]
```

Count the most common pair: `e, w` appears most often → merge into `ew`:
```
after merge 1: [l, o, w, ew, e, s, t, n, ew, ew, e, r]
```

Count again: `ew, e` appears most → merge into `ewe`, etc.

After thousands of merges, you end up with common subwords in the vocabulary. This is why `tokenizer.py` does not need to train BPE from scratch for every run — the vocabulary is pre-built and stored.

---

## 4. Special Tokens

Special tokens are reserved IDs that mark structural parts of a conversation:

| Token | ID | Meaning |
|---|---|---|
| `<\|endoftext\|>` | 0 | End of document / EOS |
| `<\|pad\|>` | 1 | Padding (to fill short sequences to a fixed length) |
| `<\|system\|>` | 2 | Start of a system prompt |
| `<\|user\|>` | 3 | Start of a user message |
| `<\|assistant\|>` | 4 | Start of an assistant message |
| `<\|eom\|>` | 5 | End of a message |
| `<\|thinking\|>` | 6 | Start of thinking/reasoning scratchpad |
| `<\|/thinking\|>` | 7 | End of thinking |

A formatted conversation looks like:

```
<|system|>You are a helpful assistant.<|eom|>
<|user|>What is 2+2?<|eom|>
<|assistant|><|thinking|>The user wants 2+2. That equals 4.<|/thinking|>
The answer is 4.<|eom|>
```

---

## 5. Token Types — The SFT Training Label

When training in **Supervised Fine-Tuning (SFT)** mode, we only want the model to learn from the **assistant's words**, not from system prompts or user messages.

To achieve this, every token gets a **type label**:

| Type | Value | Meaning |
|---|---|---|
| System | `0` | System prompt tokens |
| User | `1` | User message tokens |
| Assistant | `2` | Assistant response tokens (train on these) |

The `get_token_types()` function scans the token IDs and produces a parallel array of type labels.

---

## 6. Full Annotated Source: Key Functions from `tokenizer.py`

```python
class APEX1Tokenizer:
    """BPE tokenizer for APEX-1 with special tokens."""

    # Special token strings
    SPECIAL_TOKENS = {
        "<|endoftext|>": 0,
        "<|pad|>": 1,
        "<|system|>": 2,
        "<|user|>": 3,
        "<|assistant|>": 4,
        "<|eom|>": 5,
        "<|thinking|>": 6,
        "<|/thinking|>": 7,
    }

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert text to a list of token IDs.
        
        First checks if any special token strings appear in the text,
        then tokenises each segment with the BPE algorithm.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Convert a list of token IDs back to text."""
        ...

    def encode_chat(self, messages: list[dict], add_generation_prompt: bool = True) -> list[int]:
        """Encode a chat conversation in APEX-1's format.
        
        Args:
            messages: List of {"role": "system"/"user"/"assistant", "content": str}
            add_generation_prompt: If True, append <|assistant|> at end
        
        Returns:
            Token IDs for the entire conversation.
        
        Example:
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi!"},
            ]
            → [2, 1887, ..., 3, 13048, ..., 4]
        """
        token_ids = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Add the role start token
            if role == "system":
                token_ids.append(self.SPECIAL_TOKENS["<|system|>"])
            elif role == "user":
                token_ids.append(self.SPECIAL_TOKENS["<|user|>"])
            elif role == "assistant":
                token_ids.append(self.SPECIAL_TOKENS["<|assistant|>"])
            
            # Add the content tokens
            token_ids.extend(self.encode(content, add_special_tokens=False))
            
            # Add end-of-message token
            token_ids.append(self.SPECIAL_TOKENS["<|eom|>"])
        
        if add_generation_prompt:
            token_ids.append(self.SPECIAL_TOKENS["<|assistant|>"])
        
        return token_ids

    def get_token_types(self, token_ids: list[int]) -> list[int]:
        """Return a type label for each token: 0=system, 1=user, 2=assistant.
        
        This is used during SFT training to mask out non-assistant tokens
        from the loss computation.
        
        BUG-14 FIX: <|thinking|> and <|/thinking|> tokens are now always
        labelled as type 2 (assistant). Previously they could inherit the
        wrong type if thinking appeared without a preceding <|assistant|>
        token, which would exclude thinking content from SFT loss.
        
        Returns:
            list of ints, same length as token_ids.
        
        Example:
            token_ids: [2, 1887, 5, 3, 13048, 5, 4, 9338, 5]
            types:     [0, 0,    0, 1, 1,     1, 2, 2,    2]
                        ^ system  ^  ^ user   ^ ^ assistant ^
        """
        types = []
        current_type = 0   # Start assuming system tokens
        
        for tid in token_ids:
            if tid == self.SPECIAL_TOKENS["<|system|>"]:
                current_type = 0   # Switch to system mode
            elif tid == self.SPECIAL_TOKENS["<|user|>"]:
                current_type = 1   # Switch to user mode
            elif tid == self.SPECIAL_TOKENS["<|assistant|>"]:
                current_type = 2   # Switch to assistant mode
            # BUG-14 FIX: these always belong to the assistant,
            # regardless of what current_type is.
            elif tid in (
                self.SPECIAL_TOKENS["<|thinking|>"],
                self.SPECIAL_TOKENS["<|/thinking|>"],
            ):
                types.append(2)    # Force type = assistant
                continue
            
            types.append(current_type)
        
        return types
```

---

## 7. Why This Matters for Training

During SFT training, the loss function receives the token types array and masks out everything that is not type 2 (assistant):

```python
labels = token_ids.clone()
labels[token_types != 2] = -100   # -100 = ignore in cross-entropy
```

This means the model only learns to predict assistant tokens. It does not penalise wrong predictions for system or user tokens.

**Without this masking**, the model would waste training signal trying to predict every word the user types, which is not what we want — we want it to learn to generate helpful assistant responses.

---

## 8. Training a New Tokenizer

If you want to train APEX-1 on a new language or domain, you can train a fresh BPE tokenizer:

```python
# apex/tokenizer/train_tokenizer.py
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/my_corpus.txt"],
    vocab_size=151643,
    min_frequency=2,
    special_tokens=list(APEX1Tokenizer.SPECIAL_TOKENS.keys()),
)
tokenizer.save_model("my_tokenizer/")
```

---

## 9. Quick Summary

| Concept | What It Is |
|---|---|
| Token | Piece of text (word/sub-word) mapped to an integer |
| Vocabulary | All 151,643 possible tokens |
| BPE | Algorithm that built the vocabulary by merging common pairs |
| Special tokens | Reserved IDs for structural markers (`<\|system\|>`, etc.) |
| Token types | Labels (0/1/2) telling the trainer who wrote each token |

---

**Next:** [04 — Embeddings & RMSNorm →](04-embeddings-and-rmsnorm.md)
