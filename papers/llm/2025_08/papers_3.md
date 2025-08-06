# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00998v1">Are LLM-Powered Social Media Bots Realistic?</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted into SBP-BRiMS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become more sophisticated, there is a possibility to harness LLMs to power social media bots. This work investigates the realism of generating LLM-Powered social media bot networks. Through a combination of manual effort, network science and LLMs, we create synthetic bot agent personas, their tweets and their interactions, thereby simulating social media networks. We compare the generated networks against empirical bot/human data, observing that both network and linguistic properties of LLM-Powered Bots differ from Wild Bots/Humans. This has implications towards the detection and effectiveness of LLM-Powered Bots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00965v1">VAULT: Vigilant Adversarial Updates via LLM-Driven Retrieval-Augmented Generation for NLI</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      We introduce VAULT, a fully automated adversarial RAG pipeline that systematically uncovers and remedies weaknesses in NLI models through three stages: retrieval, adversarial generation, and iterative retraining. First, we perform balanced few-shot retrieval by embedding premises with both semantic (BGE) and lexical (BM25) similarity. Next, we assemble these contexts into LLM prompts to generate adversarial hypotheses, which are then validated by an LLM ensemble for label fidelity. Finally, the validated adversarial examples are injected back into the training set at increasing mixing ratios, progressively fortifying a zero-shot RoBERTa-base model.On standard benchmarks, VAULT elevates RoBERTa-base accuracy from 88.48% to 92.60% on SNLI +4.12%, from 75.04% to 80.95% on ANLI +5.91%, and from 54.67% to 71.99% on MultiNLI +17.32%. It also consistently outperforms prior in-context adversarial methods by up to 2.0% across datasets. By automating high-quality adversarial data curation at scale, VAULT enables rapid, human-independent robustness improvements in NLI inference tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02732v1">A Note on Code Quality Score: LLMs for Maintainable Large Codebases</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 24 pages, ICLR format
    </div>
    <details class="paper-abstract">
      Maintaining code quality in large-scale software systems presents significant challenges, particularly in settings where a large numbers of engineers work concurrently on a codebase. This paper introduces Code Quality Score (CQS) system to automatically detect issues with a set of code changes and provide actionable insights. At its core, the CQS system is powered by two Llama3 models, fine-tuned (with SFT and offline RL approaches), to a) detect common code quality issues related to coding best practices and b) to provide good ``critiques'' for LLM-generated code review respectively. To maintain good user experience, we layer the system with hand-crafted rules to filter out incorrect responses/hallucinations. Offline evaluations show that our CQS system is able to achieve an impressive precision rate for identifying valid issues. This system has already been rolled out to developers in an industrial scale setting and has consistently achieved 60\% week over week user helpfulness rate, demonstrating its effectiveness in a real-world environment. In this paper, we present details of the CQS system along with some learnings on curating developer feedback to create training data for LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02721v1">Blueprint First, Model Second: A Framework for Deterministic LLM Workflow</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 8 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      While powerful, the inherent non-determinism of large language model (LLM) agents limits their application in structured operational environments where procedural fidelity and predictable execution are strict requirements. This limitation stems from current architectures that conflate probabilistic, high-level planning with low-level action execution within a single generative process. To address this, we introduce the Source Code Agent framework, a new paradigm built on the "Blueprint First, Model Second" philosophy. Our framework decouples the workflow logic from the generative model. An expert-defined operational procedure is first codified into a source code-based Execution Blueprint, which is then executed by a deterministic engine. The LLM is strategically invoked as a specialized tool to handle bounded, complex sub-tasks within the workflow, but never to decide the workflow's path. We conduct a comprehensive evaluation on the challenging tau-bench benchmark, designed for complex user-tool-rule scenarios. Our results demonstrate that the Source Code Agent establishes a new state-of-the-art, outperforming the strongest baseline by 10.1 percentage points on the average Pass^1 score while dramatically improving execution efficiency. Our work enables the verifiable and reliable deployment of autonomous agents in applications governed by strict procedural logic.
    </details>
</div>
