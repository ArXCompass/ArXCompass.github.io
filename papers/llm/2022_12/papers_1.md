# llm - 2022_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2212.11311v1">What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2022-12-21
    </div>
    <details class="paper-abstract">
      Market sentiment analysis on social media content requires knowledge of both financial markets and social media jargon, which makes it a challenging task for human raters. The resulting lack of high-quality labeled data stands in the way of conventional supervised learning methods. Instead, we approach this problem using semi-supervised learning with a large language model (LLM). Our pipeline generates weak financial sentiment labels for Reddit posts with an LLM and then uses that data to train a small model that can be served in production. We find that prompting the LLM to produce Chain-of-Thought summaries and forcing it through several reasoning paths helps generate more stable and accurate labels, while using a regression loss further improves distillation quality. With only a handful of prompts, the final model performs on par with existing supervised models. Though production applications of our model are limited by ethical considerations, the model's competitive performance points to the great potential of using LLMs for tasks that otherwise require skill-intensive annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2212.06295v1">Despite "super-human" performance, current LLMs are unsuited for decisions about ethics and safety</a></div>
    <div class="paper-meta">
      📅 2022-12-13
      | 💬 ML Safety Workshop, NeurIPS 2022
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exploded in popularity in the past few years and have achieved undeniably impressive results on benchmarks as varied as question answering and text summarization. We provide a simple new prompting strategy that leads to yet another supposedly "super-human" result, this time outperforming humans at common sense ethical reasoning (as measured by accuracy on a subset of the ETHICS dataset). Unfortunately, we find that relying on average performance to judge capabilities can be highly misleading. LLM errors differ systematically from human errors in ways that make it easy to craft adversarial examples, or even perturb existing examples to flip the output label. We also observe signs of inverse scaling with model size on some examples, and show that prompting models to "explain their reasoning" often leads to alarming justifications of unethical actions. Our results highlight how human-like performance does not necessarily imply human-like understanding or reasoning.
    </details>
</div>
