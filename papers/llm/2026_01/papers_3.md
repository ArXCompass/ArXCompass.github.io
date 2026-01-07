# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00213v1">Overlooked Safety Vulnerability in LLMs: Malicious Intelligent Optimization Algorithm Request and its Jailbreak</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      The widespread deployment of large language models (LLMs) has raised growing concerns about their misuse risks and associated safety issues. While prior studies have examined the safety of LLMs in general usage, code generation, and agent-based applications, their vulnerabilities in automated algorithm design remain underexplored. To fill this gap, this study investigates this overlooked safety vulnerability, with a particular focus on intelligent optimization algorithm design, given its prevalent use in complex decision-making scenarios. We introduce MalOptBench, a benchmark consisting of 60 malicious optimization algorithm requests, and propose MOBjailbreak, a jailbreak method tailored for this scenario. Through extensive evaluation of 13 mainstream LLMs including the latest GPT-5 and DeepSeek-V3.1, we reveal that most models remain highly susceptible to such attacks, with an average attack success rate of 83.59% and an average harmfulness score of 4.28 out of 5 on original harmful prompts, and near-complete failure under MOBjailbreak. Furthermore, we assess state-of-the-art plug-and-play defenses that can be applied to closed-source models, and find that they are only marginally effective against MOBjailbreak and prone to exaggerated safety behaviors. These findings highlight the urgent need for stronger alignment techniques to safeguard LLMs against misuse in algorithm design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00926v1">MACA: A Framework for Distilling Trustworthy LLMs into Efficient Retrievers</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Modern enterprise retrieval systems must handle short, underspecified queries such as ``foreign transaction fee refund'' and ``recent check status''. In these cases, semantic nuance and metadata matter but per-query large language model (LLM) re-ranking and manual labeling are costly. We present Metadata-Aware Cross-Model Alignment (MACA), which distills a calibrated metadata aware LLM re-ranker into a compact student retriever, avoiding online LLM calls. A metadata-aware prompt verifies the teacher's trustworthiness by checking consistency under permutations and robustness to paraphrases, then supplies listwise scores, hard negatives, and calibrated relevance margins. The student trains with MACA's MetaFusion objective, which combines a metadata conditioned ranking loss with a cross model margin loss so it learns to push the correct answer above semantically similar candidates with mismatched topic, sub-topic, or entity. On a proprietary consumer banking FAQ corpus and BankFAQs, the MACA teacher surpasses a MAFA baseline at Accuracy@1 by five points on the proprietary set and three points on BankFAQs. MACA students substantially outperform pretrained encoders; e.g., on the proprietary corpus MiniLM Accuracy@1 improves from 0.23 to 0.48, while keeping inference free of LLM calls and supporting retrieval-augmented generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00912v1">The Discovery Gap: How Product Hunt Startups Vanish in LLM Organic Discovery Queries</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 20 pages, 7 figures. Based on M.Tech thesis research, Indian Institute of Technology Patna, 2025
    </div>
    <details class="paper-abstract">
      When someone asks ChatGPT to recommend a project management tool, which products show up in the response? And more importantly for startup founders: will their newly launched product ever appear? This research set out to answer these questions. I randomly selected 112 startups from the top 500 products featured on the 2025 Product Hunt leaderboard and tested each one across 2,240 queries to two different large language models: ChatGPT (gpt-4o-mini) and Perplexity (sonar with web search). The results were striking. When users asked about products by name, both LLMs recognized them almost perfectly: 99.4% for ChatGPT and 94.3% for Perplexity. But when users asked discovery-style questions like "What are the best AI tools launched this year?" the success rates collapsed to 3.32% and 8.29% respectively. That's a gap of 30-to-1 for ChatGPT. Perhaps the most surprising finding was that Generative Engine Optimization (GEO), the practice of optimizing website content for AI visibility, showed no correlation with actual discovery rates. Products with high GEO scores were no more likely to appear in organic queries than products with low scores. What did matter? For Perplexity, traditional SEO signals like referring domains (r = +0.319, p < 0.001) and Product Hunt ranking (r = -0.286, p = 0.002) predicted visibility. After cleaning the Reddit data for false positives, community presence also emerged as significant (r = +0.395, p = 0.002). The practical takeaway is counterintuitive: don't optimize for AI discovery directly. Instead, build the SEO foundation first and LLM visibility will follow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02407v1">Evolving Personalities in Chaos: An LLM-Augmented Framework for Character Discovery in the Iterated Prisoners Dilemma under Environmental Stress</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 10 pages, 5 figures. Project assignment; exploratory study on LLM-based adaptive agents
    </div>
    <details class="paper-abstract">
      Standard simulations of the Iterated Prisoners Dilemma (IPD) operate in deterministic, noise-free environments, producing strategies that may be theoretically optimal but fragile when confronted with real-world uncertainty. This paper addresses two critical gaps in evolutionary game theory research: (1) the absence of realistic environmental stressors during strategy evolution, and (2) the Interpretability Gap, where evolved genetic strategies remain opaque binary sequences devoid of semantic meaning. We introduce a novel framework combining stochastic environmental perturbations (God Mode) with Large Language Model (LLM)-based behavioral profiling to transform evolved genotypes into interpretable character archetypes. Our experiments demonstrate that strategies evolved under chaotic conditions exhibit superior resilience and present distinct behavioral phenotypes, ranging from Ruthless Capitalists to Diplomatic Enforcers. These phenotypes are readily classified by LLMs but remain nearly impossible to interpret through manual genome inspection alone. This work bridges evolutionary computation with explainable AI and provides a template for automated agent characterization in multi-agent systems.
    </details>
</div>
