# 00 — Introduction: What Is a Language Model?

> **Difficulty:** ⭐☆☆☆☆ Absolute Beginner  
> **Time to read:** ~15 minutes  
> **You will learn:** What AI models are, why we build one from scratch, and how to use this guide.

---

## 1. What Is Artificial Intelligence?

**Artificial Intelligence (AI)** is a computer program that can perform tasks that normally require human intelligence — like understanding text, answering questions, writing code, or translating languages.

Think of AI like a **very sophisticated calculator** — but instead of calculating numbers, it calculates which word comes next in a sentence.

---

## 2. What Is a Language Model?

A **language model** is a specific type of AI that works with text.

### The Auto-Complete Analogy

You have probably seen auto-complete on your phone keyboard. When you type "I am going to the", the phone suggests "store", "gym", or "beach". That is the basic idea of a language model.

A language model answers one simple question, over and over:

> **"Given what I have read so far, what word (or token) comes next?"**

A modern large language model (LLM) like ChatGPT, Claude, or APEX-1 does this with billions of parameters (numbers the model has learned) and can generate entire essays, solve math problems, and write code.

---

## 3. What Is a "Token"?

A **token** is the basic unit that a language model reads and writes. It is usually a word or part of a word.

For example:
- The word `"unhappiness"` might be split into 3 tokens: `"un"`, `"happiness"` wait, actually: `["un", "happin", "ess"]`
- The word `"cat"` is usually 1 token: `["cat"]`
- The word `"ChatGPT"` might be 3 tokens: `["Chat", "G", "PT"]`

APEX-1 uses **151,643 tokens** — meaning it knows 151,643 different "pieces" of text.

---

## 4. How Does the Model Learn?

The model learns by reading enormous amounts of text (trillions of words from the internet, books, code, etc.) and trying to predict the next token over and over.

Each time it gets the prediction wrong, it adjusts its internal numbers slightly to do better next time. After millions of adjustments, the model learns grammar, facts, reasoning, code syntax, and much more.

This process is called **training**.

The mathematical measure of "how wrong" the model is at any moment is called the **loss**. Lower loss = better model.

---

## 5. Key Vocabulary

Here are the terms you will see throughout this documentation. Do not worry if they sound complex now — each one will be explained in its own document.

| Term | Simple Definition |
|---|---|
| **Token** | A piece of text (word or sub-word) |
| **Vocabulary** | The full set of tokens the model knows |
| **Embedding** | A list of numbers that represents a token |
| **Attention** | How the model decides which previous tokens to focus on |
| **Layer** | One processing step in the model (APEX-1 has 12–72 layers) |
| **Parameter** | A number the model learned during training |
| **Loss** | A measure of how wrong the model's predictions are |
| **Gradient** | The direction to adjust parameters to reduce loss |
| **Batch** | A group of training examples processed together |
| **Epoch** | One complete pass through the training data |
| **Checkpoint** | A saved copy of the model's parameters |
| **Inference** | Using the model to generate text (not training) |
| **Fine-tuning** | Training a pre-trained model on specific data |

---

## 6. What Makes APEX-1 Special?

APEX-1 is not just another copy of an existing model. It is a **research architecture** that picks the single best innovation from each major AI lab and combines them:

| Innovation | Where It Comes From | What It Does |
|---|---|---|
| Multi-Head Latent Attention | DeepSeek-V3 | Reduces memory by 93% |
| Mixture of Experts | DeepSeek-V3 | 256 specialists, only 2 active per token |
| RoPE + YaRN | KIMI / DeepSeek | Handles very long documents |
| Sliding Window Attention | Mistral / Llama 3 | Efficient local focus |
| Constitutional AI | Anthropic | Safety built in |
| GRPO Alignment | DeepSeek-R1 | Stable RL training |
| Thinking Mode | Claude / DeepSeek-R1 | Built-in reasoning scratchpad |

This makes APEX-1 an excellent **learning project** because you can see exactly how each idea is implemented in real, working Python code.

---

## 7. Why Build From Scratch?

Most AI courses teach theory, but students never see how the formulas map to actual code. This documentation is different:

- Every formula is connected to the **exact line of Python** that implements it.
- Every design decision is explained — not just *what* but *why*.
- Bugs that were found and fixed are explained as lessons.

By the end of this guide, you will understand not just APEX-1, but the principles behind **any modern LLM**.

---

## 8. What You Need to Know Already

To get the most from this guide, you should be comfortable with:

- **Python** — functions, classes, loops, lists
- **Basic math** — what a matrix is, what multiplication means
- **NumPy or PyTorch basics** — what a tensor is (it is just an array of numbers)

If you are shaky on any of these, do not worry. We will explain everything we use.

---

## 9. How to Use This Guide

Read the files in order:

```
00-introduction.md          ← You are here
01-project-structure.md
02-configuration.md
03-tokenizer.md
04-embeddings-and-rmsnorm.md
...
31-end-to-end-walkthrough.md
```

Each file:
1. Starts with a **plain-English definition**
2. Gives a **real-world analogy**
3. Shows the **math formula** (with each piece explained)
4. Shows the **full source code** with annotations
5. Explains **why APEX-1 uses this design**

---

## 10. The Big Picture

Here is the complete journey of a sentence through APEX-1:

```
You type:   "The capital of France is"
                        ↓
Step 1:  Tokenizer splits into tokens: ["The", "capital", "of", "France", "is"]
                        ↓
Step 2:  Each token becomes a vector of numbers (embedding)
                        ↓
Step 3:  Positional encoding added (RoPE)
                        ↓
Step 4:  Pass through 12–72 transformer layers, each doing:
         • Attention  (which tokens relate to which)
         • FFN        (apply learned knowledge)
                        ↓
Step 5:  Final projection → scores over all 151,643 tokens
                        ↓
Step 6:  Sample: pick "Paris" (highest probability)
                        ↓
Output:  "Paris"
```

Every single one of these steps has its own documentation file. Let's go!

---

**Next:** [01 — Project Structure →](01-project-structure.md)
