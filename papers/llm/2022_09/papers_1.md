# llm - 2022_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2209.11000v1">Selecting Better Samples from Pre-trained LLMs: A Case Study on Question Generation</a></div>
    <div class="paper-meta">
      📅 2022-09-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have in recent years demonstrated impressive prowess in natural language generation. A common practice to improve generation diversity is to sample multiple outputs from the model. However, there lacks a simple and robust way of selecting the best output from these stochastic samples. As a case study framed in the context of question generation, we propose two prompt-based approaches to selecting high-quality questions from a set of LLM-generated candidates. Our method works under the constraints of 1) a black-box (non-modifiable) question generation model and 2) lack of access to human-annotated references -- both of which are realistic limitations for real-world deployment of LLMs. With automatic as well as human evaluations, we empirically demonstrate that our approach can effectively select questions of higher qualities than greedy generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2207.04237v2">Few-shot training LLMs for project-specific code-summarization</a></div>
    <div class="paper-meta">
      📅 2022-09-08
      | 💬 Accepted at ASE-NIER (2022) track
    </div>
    <details class="paper-abstract">
      Very large language models (LLMs), such as GPT-3 and Codex have achieved state-of-the-art performance on several natural-language tasks, and show great promise also for code. A particularly exciting aspect of LLMs is their knack for few-shot and zero-shot learning: they can learn to perform a task with very few examples. Few-shotting has particular synergies in software engineering, where there are a lot of phenomena (identifier names, APIs, terminology, coding patterns) that are known to be highly project-specific. However, project-specific data can be quite limited, especially early in the history of a project; thus the few-shot learning capacity of LLMs might be very relevant. In this paper, we investigate the use few-shot training with the very large GPT (Generative Pre-trained Transformer) Codex model, and find evidence suggesting that one can significantly surpass state-of-the-art models for code-summarization, leveraging project-specific training.
    </details>
</div>
