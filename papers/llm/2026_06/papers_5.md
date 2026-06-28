# llm - 2026_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13397v1">Mod-Guide: An LLM-based Content Moderation Feedback System to Address Insensitive Speech toward Indigenous Ethnic and Religious Minority Communities</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Language operates as a mechanism of both marginalization and resistance, especially for minority communities navigating insensitive and harmful speech online. As content moderation increasingly depends on large language models (LLMs), concerns arise about whether these systems can recognize culturally insensitive speech-language that disregards or marginalizes the cultural and religious perspectives of historically underrepresented communities, often through implicit erasure, misrepresentation, or normative framing, rather than overt hostility. Focusing on Bangladesh's Hindu and Chakma communities -- the country's largest religious and Indigenous ethnic minorities, respectively -- this paper investigates the epistemic limits of LLM-based moderation systems and explores methods for incorporating minority perspectives. We co-created a culturally grounded corpus of insensitive speech with community members and integrated their narratives into moderation pipelines using retrieval augmented generation (RAG). Our tool, Mod-Guide, improves LLM sensitivity to minority viewpoints by leveraging contextual cues derived from lived experience. Through mixed-method evaluations involving both minority and majority participants, we demonstrate that RAG-enhanced moderation responses are more contextually accurate and perceived differently across ethnic lines. This work advances research in human-computer interaction, AI ethics, and social computing by foregrounding restorative justice and hermeneutical inclusion in the design of content moderation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13380v1">An LLM System for Autonomous Variational Quantum Circuit Design</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 63 pages, 19 figures, 3 tables
    </div>
    <details class="paper-abstract">
      The design of high performing quantum circuits remains largely dependent on human expertise. We introduce an autonomous agentic framework that employs large language models (LLMs) to conduct iterative quantum circuit designs under explicit design constraints. Our system integrates seven components: Exploration, Generation, Discussion, Validation, Storage, Evaluation, and Review. These components form a closed-loop workflow that combines web-based knowledge acquisition, literature-grounded critique, executable code generation, and experimental feedback. We evaluate the framework on two tasks: quantum feature map construction for quantum machine learning and ansatz generation for variational quantum eigensolver applications in quantum chemistry. In image classification benchmarks, the best generated feature map outperforms representative quantum feature maps and, when scaled to larger qubit counts, surpasses the classical radial basis function kernel. In molecular ground state estimation across seven molecules, the generated ansatz attains competitive accuracy with widely used chemically inspired and hardware-efficient constructions while satisfying the imposed scaling constraints. These results establish LLM driven agentic system as a viable paradigm for automated quantum circuit design and illustrate how AI systems can participate in iterative scientific optimization workflows across scientific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12160v2">A Controlled Study of Decoding-Time Truthfulness Methods on Instruction-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Decoding-time truthfulness methods -- layer-contrast decoding, inference-time intervention, and learned logit adapters -- have demonstrated 10-30 point gains on TruthfulQA when applied to base language models. However, modern instruction-tuned LLMs already achieve substantially higher baselines (61-76%), raising the question of whether these methods remain effective in practice. We design a six-control evaluation framework -- out-of-distribution training, multi-judge validation, simple decoding baselines, confound controls, bootstrap confidence intervals, and seed variance -- and apply it across 5 models (1B-70B), 3 benchmarks, and 15 methods. We find that previously reported gains shrink substantially under strict controls: on the full TruthfulQA benchmark (N=817), no token-level method achieves statistically significant improvement, and the best learned adapter scores -2.0 points below greedy (p=.23). We identify five evaluation sensitivities -- contamination, judge choice, missing baselines, confounds, and statistical noise -- that individually or jointly account for these discrepancies. Cross-benchmark validation on HaluEval QA and TriviaQA confirms that these patterns extend beyond TruthfulQA. Deliberative prompting methods (chain-of-thought, self-critique) appear more robust in the evaluated regime, with CoT achieving +5.6-19pp across benchmarks as a training-free, single-pass method. We release a seven-point evaluation checklist and discuss implications for future truthfulness research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13317v1">SkillCAT: Contrastive Assessment and Topology-Aware Skill Self-Evolution for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 9 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Skill self-evolution methods for LLM agents aim to turn execution trajectories into reusable skill documents, but current pipelines typically learn from one trajectory per task, merge candidate skill patches before checking them, and load the full skill corpus before inference. We propose SkillCAT, a training-free framework that separates this process into three stages. Contrastive Causal Extraction (CCE) samples multiple trajectories for each task and compares same-task success/failure pairs to identify evidence that explains outcome differences. Assessment-Augmented Evolution (AAE) replays each candidate patch on source-task clones and keeps only patches that improve or preserve task outcomes before hierarchical skill patch merging. Topology-Aware Task Execution (TTE) compiles the evolved skills into a routable sub-skill topology, so inference loads only the capability nodes relevant to the task. We evaluate SkillCAT on common agent benchmarks, including SpreadsheetBench, WikiTableQuestions, and DocVQA, and further test cross-model and out-of-distribution generalization. Across these settings, SkillCAT raises the average score over baselines by up to 40.40%, demonstrating reliable skill evolution without model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13316v1">ReSum: Synergizing LLM Reasoning and Summarization with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 24 pages, including 13 pages of main text and 11 pages of appendix
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is a central technique for improving long-horizon reasoning in Large Language Models (LLMs). However, existing RLVR methods often encourage unnecessarily long reasoning rollouts, which can degrade reasoning coherence and exhaust the available context budget. Existing approaches to long-context organization often depend on external mechanisms to organize rollouts, rather than enabling the model to manage its own reasoning trajectory. To address this limitation, we propose ReSum, a novel RLVR framework that enables LLMs to compress and organize their reasoning trajectories through self-summarization. Our pilot studies show that self-summarization stabilizes generation by lowering token-level entropy, and that introducing a ``summarization'' phrase can substantially mitigate errors propagated from an incorrect rollout prefix. Motivated by these findings, ReSum adopts a summarization-aware adaptive rollout mechanism that contrastively evaluates whether self-summarization benefits the ongoing reasoning process. Specifically, when the model spontaneously triggers self-summarization, ReSum masks the summarization phrase to create a contrastive branch; for non-summarization positions, it instead randomly injects the phrase to create a matched branch. We further design a summarization-aware advantage to enable finer-grained comparison between contrastive rollout trajectories. Extensive experiments show that ReSum improves performance at an average of 4\% while reducing rollout length by 18.6\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13254v1">Evaluating Pluralism in LLMs through Latent Perspectives</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Pluralistic Alignment Workshop @ ICML 2026
    </div>
    <details class="paper-abstract">
      The growing need to represent diverse perspectives has increased interest in pluralistic LLM generation. Although difficult to operationalize, identifying perspectives expressed in text would provide clear guidance on pluralistic alignment and more clearly articulate the pluralistic gap in LLM generation. While models have been shown to reduce the diversity of training data and generate homogeneously, this has been demonstrated primarily on multiple-choice questionnaires or using high-level characteristics of free-form text. In this paper, we introduce and implement a domain-agnostic multi-layered framework for unsupervised extraction of perspectives suitable for identifying the pluralistic gap in LLM-generated text. We evaluate our framework on book reviews, a highly opinionated dataset representing diverse perspectives, and compare various prompts and models. Our results show that while some models and prompting techniques come close to covering a broad spectrum of perspectives, rarer perspectives remain disproportionately underrepresented, resulting in distributions that diverge from human text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.09500v3">Deterministic Integrity Gates for LLM-Assisted Clinical Manuscript Preparation: An Auditable Biomedical Informatics Architecture</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 28 pages, 3 figures, 4 tables; includes supplementary material (deterministic-detector inventory, per-class defect breakdown, worked example). Software (MIT): https://github.com/Aperivue/medsci-skills . Archived on Zenodo: concept DOI https://doi.org/10.5281/zenodo.20155321 and version DOI (v3.8.0) https://doi.org/10.5281/zenodo.20582972
    </div>
    <details class="paper-abstract">
      As autonomous research agents and AI co-scientist systems push large language models (LLMs) from drafting toward end-to-end manuscript production, the bottleneck shifts from generation to verification. Fluent LLM output can hide fabricated citations, numbers that drift from source tables, and unmet reporting-guideline items; existing tools generate without verifying, and self-critique inherits the blind spots that produce confident fabrication. We describe an architecture pairing generation with verification, resting on three principles: decompose the workflow into self-contained skills, gate every stage transition with halt-on-failure, and resolve each integrity question with the cheapest sufficient mechanism, a deterministic, re-executable check where one suffices and a prose-level probe only where interpretation is unavoidable. This determinism-where-possible split, organized as an integrity-gate taxonomy, is the core contribution. It is realized as MedSci Skills, an open-source toolkit of 43 skills with a 21-detector deterministic tier, evaluated on three public-dataset pipelines (STARD, PRISMA, STROBE) and a seeded-defect ablation. Across the three pipelines every content-hash manifest verified clean and the gates surfaced real defects; on 27 identical injected defects the deterministic gates detected all 27 with no false positives on the matched clean fixtures, whereas a single-prompt LLM reviewer detected 11, its misses in code, bibliography, and style defects the prose hides. Determinism-where-possible verification yields an auditable, re-executable trail that exposes the evidence a human needs to check an LLM-assisted manuscript: feasibility and reproducibility evidence, not a claim of human-competitive quality, which a separate blinded study addresses. MedSci Skills is MIT-licensed and archived (v3.8.0).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.17062v2">The Range Shrinks, the Threat Remains: Re-evaluating LLM Package Hallucinations on the 2026 Frontier-Model Cohort</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 13 pages, 3 figures, 4 tables. v2: incorporates coordinated-disclosure feedback from PyPI Security and Socket.dev; registrable attack surface refined to 53 names (41 PyPI, 12 npm). Headline rates unchanged. Replication of Spracklen et al. (USENIX Security 2025). Data and code: https://github.com/churik5/slopsquatting-replication-2026 and https://doi.org/10.5281/zenodo.19859120
    </div>
    <details class="paper-abstract">
      Spracklen et al. (USENIX Security '25) showed that code-generating large language models hallucinate package names that do not exist on PyPI or npm at rates ranging from 5.2% on commercial models to 21.7% on open-source models, creating an attack surface for slopsquatting -- the registration of malicious packages under hallucinated names. We replicate their methodology on five frontier code-capable LLMs released between October 2025 and March 2026: Claude Sonnet 4.6, Claude Haiku 4.5, GPT-5.4-mini, Gemini 2.5 Pro, and DeepSeek V3.2. Across 199,845 paired Python and JavaScript prompts validated against PyPI and npm master lists, we measure overall hallucination rates between 4.62% (Claude Haiku 4.5) and 6.10% (GPT-5.4-mini) -- an order-of-magnitude compression of the inter-model spread observed by Spracklen, but not a retirement of the threat. Beyond replication, we identify a set of 127 package names (109 on PyPI, 18 on npm) that all five evaluated models invent identically; following coordinated disclosure with PyPI Security and Socket.dev, 53 of these (41 on PyPI, 12 on npm) remain registrable by an attacker after each registry's existing defenses, constituting a model-agnostic supply-chain attack surface that no single-model study can reveal. We further document a Python-over-JavaScript hallucination asymmetry that inverts Spracklen's 2024 finding, identify a Haiku-below-Sonnet inversion within the Anthropic family, and observe a Jaccard-similarity peak between DeepSeek V3.2 and GPT-5.4-mini (J = 0.343) suggestive of shared training-data origins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13221v1">From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Evaluating new large language models typically requires costly human annotation campaigns at scale. LLM-as-a-judge offers a cheaper alternative, but judge scores carry systematic errors - such as position bias, self-preference, or intransitivity - that can strongly miscalibrate the resulting rankings. We quantify the resulting judge-human disagreement at two complementary levels. At the local level, we estimate per-battle uncertainty from the judge's own score differences by propagating calibrated win probabilities rather than hard labels into the Bradley-Terry procedure. This alone provides a drastic improvement to Elo estimation accuracy, bringing LLM-derived ratings within 17.9 Elo MAE of human-derived ones when averaged over 55 held-out models on LMArena. At the global level, we apply split conformal prediction to the residual gap between LLM-derived and human-derived Elo ratings across held-out models, producing prediction intervals with distribution-free marginal coverage guarantees that account for irreducible LLM-human disagreement. Together, these two layers yield a low-cost evaluation tool that provides developers with calibrated Elo estimates and honest uncertainty bounds, without access to large-scale human annotations.To facilitate reproducibility, we release our code at https://github.com/kargibora/SoftElo .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13220v1">LLM-as-an-Investigator: Evidence-First Reasoning for Robust Interactive Problem Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as interactive assistants for technical problem solving. However, when users provide incomplete descriptions or plausible but unverified explanations, LLMs may prematurely align with these assumptions and propose solutions before collecting sufficient evidence. We refer to this behavior as user-driven sycophancy: the tendency of an LLM to reinforce a user-provided hypothesis instead of testing alternative explanations. This paper introduces LLM-as-an-Investigator, an evidence-first agentic AI methodology for robust problem diagnosis. The approach is implemented through a Solution Investigator Agent, which estimates the ambiguity of an initial problem description, generates candidate hypotheses, asks targeted clarification questions, and updates hypothesis probabilities after each answer. Rather than producing an immediate response, the agent continues the investigation until the evidence makes one candidate explanation stronger than the alternatives. To evaluate the approach, we build a benchmark from solved technical forum threads in mechanical, electrical, and hydraulic domains. We use a three-agent evaluation pipeline in which a Problem-Solution Extractor Agent converts solved threads into structured cases, a Ground-Truth Evaluator Agent simulates the user while hiding the known solution, and the tested assistant attempts to recover the solution through dialogue. The experiments compare standard assistants, reasoning-oriented LLMs, and the proposed investigator-based model across LLM backbones. In addition to diagnostic accuracy, we analyze how standard assistants follow misleading user hypotheses in diagnostic cases. The results show that the proposed approach identifies the problem more accurately than direct prompting and reasoning-only baselines, while its evidence-first protocol helps reduce user-induced conversational bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13218v1">When Similar Means Different: Evaluating LLMs on Arabic--Hebrew Cognates</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Arabic and Hebrew, as closely related Semitic languages, share a substantial lexicon of true cognates, misleading false friends, and modern loanwords. This overlap poses a challenge for cross-lingual semantic understanding in large language models (LLMs). To evaluate this capability, we introduce SemCog Bench, a curated benchmark of 1,858 Arabic--Hebrew word pairs with sentence-level annotations for cognate identification and semantic disambiguation. We evaluate open-source and commercial LLMs across multiple input representations (raw, diacritized, Romanized, and phonetic) and reveal a critical gap in cross-lingual reasoning. While models achieve high accuracy on true cognates, performance drops sharply on false friends and loanwords, reflecting a strong reliance on surface-form similarity. Furthermore, sentence-level context yields only modest improvements, suggesting that contextual cues alone are insufficient to overcome misleading form-based signals. These findings reveal a fundamental limitation of current LLMs in resolving cross-lingual form--meaning conflicts and establish SemCog Bench as a rigorous benchmark for multilingual semantic reasoning. Our code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.08098v2">When Does Delegation Beat Majority? A Delegation-Based Aggregator for Multi-Sample LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Preprint. 16 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Majority voting over sampled answers is the dominant unsupervised aggregator for multi-sample LLM inference. In this paper, we show a delegation-based aggregator (Propagational Proxy Voting, PPV; Sakai et al., 2025) yields an unsupervised consensus rule that beats majority on MMLU-Pro by +1.5 pp overall and +2.24 pp on the non-trivial subset (paired McNemar p ~ 1.0e-14, n = 8,099). Majority discards two signals that every sample carries: within-group letter entropy and between-group reasoning geometry. PPV exposes per-voter levers that consume exactly these two signals: When (how much weight a voter keeps on its own pick) and Whom (how it splits the remainder across peers). We drive When with letter entropy and Whom with per-question-centered embedding cosine. Our method needs no gold labels and no auxiliary training: per-question, we partition 128 sampled generations into 16 groups, compute each group's letter-level semantic entropy and reasoning embedding centroid, and feed both into a stochastic delegation matrix whose stationary distribution selects the consensus answer. We walk through an example in which PPV overturns a clear 10-6 majority for the wrong letter: the 10-voter majority cluster is geometrically incoherent (mean within-cluster cosine -0.02) while the 6-voter minority is tight (+0.26), so propagated delegation mass concentrates on the minority's answer even though entropy alone would keep the majority ahead. We further report delegation strategies with negative results that constrain the design space for unsupervised LLM aggregation. No within-question ensemble of confidence modes closes the oracle gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13189v1">SICI: A Semantic-Pragmatic Complexity Index Reveals Regime Shifts in LLM Stance Detection</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Prompt-based LLMs are increasingly used for stance detection, but harder examples are not always repaired by clearer instructions, reasoning prompts, retrieval, or debate. We introduce SICI (Stance Inference Complexity Index), a seven-dimensional diagnostic measure of the semantic-pragmatic burden imposed by a target--text pair. Across SemEval-2016 and VAST, SICI predicts LLM accuracy better than surface proxies and shows substantial cross-scorer reliability ($α=0.771$). More importantly, LLM errors change regime as SICI increases: low-complexity examples invite over-attribution, especially Against predictions; intermediate examples form an unstable boundary; and high-complexity examples rapidly concentrate on None. This phase-transition-like structure persists across GPT-3.5, GPT-4o-mini, DeepSeek-V3, and GPT-4o, although stronger models move the boundaries. A 15-method intervention study further shows that prompting, retrieval, and debate often shift models along the attribution--abstention axis rather than removing the high-complexity bottleneck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13177v1">MemRefine: LLM-Guided Compression for Long-Term Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are increasingly expected to operate over long-term interactions, where information from past dialogues must be preserved and recalled to support future tasks. However, as interactions accumulate, the memory store grows without bound and fills with redundant entries that inflate storage cost and degrade retrieval by crowding out the most useful evidence. Furthermore, this is especially limiting on resource-constrained platforms with hard memory budgets, motivating us to formulate storage-budgeted memory management, the task of keeping an already constructed memory store within a fixed budget while preserving information useful for future interactions. To this end, we then propose MemRefine, an LLM-guided framework that, since surface similarity poorly reflects factual value, uses similarity only to propose candidate pairs and defers delete, merge, and preserve decisions to an LLM judge based on factual content, iterating until the budget is met. Across multiple memory frameworks and long-term conversation benchmarks, MemRefine consistently meets target budgets while preserving downstream performance and outperforming rule-based baselines under tight budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13171v1">NTS-CoT: Mitigating Hallucinations in LLM-based News Timeline Summarization with Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      The rapid updates of online news make tracking event developments challenging, highlighting the need for timeline summarization (TLS). Hallucinations, where LLM-generated content deviates from source news, still remain a critical issue in LLM-based TLS and are not well studied in existing works. To bridge this gap, we identify two primary types of hallucinations: unfaithful content during news summarization and information omission in date-event summarization. Then, we propose NTS-CoT, a novel framework that leverages Chain-of-Thought (CoT) reasoning to mitigate hallucinations in TLS. The framework consists of three key modules: i) Element-CoT to capture essential news elements for faithful summarization, ii) Date Selection to combine temporal saliency and event prominence for timestamp selection, and iii) Causal-CoT to infer causal relationships and reduce omissions in date-event summarization. Extensive experiments, including quantitative analysis on three TLS benchmarks and human evaluation, demonstrate that NTS-CoT outperforms state-of-the-art baselines, effectively mitigating hallucinations and improving LLM-based TLS performance. Our source code is available at https://anonymous.4open.science/r/NTS-CoT .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13140v1">MIDSim: Simulating Multi-Channel Information Diffusion in Social Media with LLM-Powered Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Information diffusion in social media shapes public opinion and collective behavior, making its modeling and simulation an important research problem. Existing studies have investigated information diffusion through epidemic-based, cascade-based, and point process models. However, they predominantly focus on diffusion through social links, overlooking other diffusion channels enabled by platform algorithms (e.g., recommender systems) and failing to capture user behavioral complexity. To address these limitations, we propose an LLM-powered multi-agent system for simulating multi-channel information diffusion, where large language models instantiate personalized user agents and the diffusion process jointly models social and algorithmic exposure streams. We further construct three real-world diffusion dataset spanning Sina Weibo, RedNote, and Twitter, containing diffusion records, user profiles, historical posts, and social relationships. Experimental results on real diffusion events show that our proposed framework realistically simulate macro diffusion phenomenon and generate diverse comment content, significantly outperforming baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13111v1">MÖVE: A Holistic LLM Benchmark for the German Public Sector</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      We present MÖVE (Modelle für die Öffentliche Verwaltung Evaluieren), a holistic benchmark for evaluating large language models (LLMs) in the context of the German public sector. While LLMs are increasingly adopted in public administration, model selection remains largely ad hoc, and existing benchmarks offer limited guidance: they are predominantly English-centric, US-centric in content, and focus exclusively on task performance. MÖVE addresses these gaps by evaluating 39 models across two complementary dimensions. Performance criteria cover summarization, question answering, and topic extraction. Governance criteria assess hallucination tendencies, energy consumption, provider transparency, and alignment with German constitutional values and knowledge about positions by German political parties. In total, we utilize ten German-language datasets, including gold- and silverstandard datasets that we constructed to reflect public-administration domains. We employ a multi-metric evaluation strategy combining classical NLP metrics, embedding-based methods, and LLM-as-a-judge approaches. Our results show that no single model dominates across all criteria: top performers differ between tasks, and model size alone is a poor predictor of quality. We further evaluate the benchmark itself, analyzing its statistical precision, LLM judge reliability, the impact of our private datasets on model rankings, the sensitivity of our results to prompt formulation, and the validity of our energy consumption estimates. MÖVE is designed as a living benchmark under active development; results are publicly available at https://moeve.bundesdruckerei.de/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13095v1">Balancing ASR and diarization in end-to-end LLMs for multi-talker speech recognition</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted in Interspeech 2026
    </div>
    <details class="paper-abstract">
      Multi-talker speech recognition is often addressed by combining automatic speech recognition (ASR) and speaker diarization in a pipeline system. Recently, LLM-based approaches have shown promise by jointly modeling semantic and speaker information, but they typically require large-scale multi-talker corpora that are costly to annotate. In this paper, we investigate how to efficiently train an LLM-based system with limited real-recorded data while maintaining high accuracy in speaker attribution. We propose several strategies: (1) a dual-encoder architecture to extract semantic and speaker features, (2) a feature interleaving format to merge these features as the inputs to the LLM, (3) a length-aware speaker ID loss to enhance diarization capability, and (4) an adaptive threshold strategy for ASR loss computation to mitigate hallucinations caused by speech overlaps. These strategies balance training between ASR and diarization tasks. Our system outperforms open-source baseline approaches, achieving relative improvements of 18% on the AliMeeting corpus and 24% on the Aishell4 corpus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13082v1">sebis at CRF Filling 2026: A Two-Stage Local LLM Pipeline for Medical CRF Filling</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Published in Proceedings of the Third Workshop on Patient-Oriented Language Processing (CL4Health), LREC 2026
    </div>
    <details class="paper-abstract">
      The extraction of structured clinical information from unstructured EHR notes is a persistent bottleneck in healthcare informatics. While large language models (LLMs) offer high performance, their deployment in clinical settings is hindered by privacy risks, inference costs, and the tendency to hallucinate beyond textual evidence. We address these challenges for the CL4Health 2026 Case Report Form (CRF) filling task by proposing a fully local, domain-adapted pipeline using the MedGemma-27B model. Our two-stage architecture, which separates binary presence classification from value extraction, enforces strict adherence to textual evidence and ensures deterministic outputs for negated, uncertain, or unknown states. By leveraging item-specific, few-shot in-context learning without external API calls or fine-tuning, our approach achieves a macro-F1 score of 0.55 on the official English test track. This result secures second place among all locally-hosted, open-source submissions. Our work demonstrates that privacy-preserving, on-premise LLM pipelines can achieve near-competitive performance with proprietary frontier models, providing a practical, data-sovereign framework for clinical NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04885v2">CuMA: Aligning LLMs with Sparse Cultural Values via Demographic-Aware Mixture of Adapters</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 ACL 2026 Main
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) serve a global audience, alignment must transition from enforcing universal consensus to respecting cultural pluralism. We demonstrate that dense models, when forced to fit conflicting value distributions, suffer from \textbf{Mean Collapse}, converging to a generic average that fails to represent diverse groups. We attribute this to \textbf{Cultural Sparsity}, where gradient interference prevents dense parameters from spanning distinct cultural modes. To resolve this, we propose \textbf{\textsc{CuMA}} (\textbf{Cu}ltural \textbf{M}ixture of \textbf{A}dapters), a framework that frames alignment as a \textbf{conditional capacity separation} problem. By incorporating demographic-aware routing, \textsc{CuMA} internalizes a \textit{Latent Cultural Topology} to explicitly disentangle conflicting gradients into specialized expert subspaces. Extensive evaluations on WorldValuesBench, Community Alignment, and PRISM demonstrate that \textsc{CuMA} achieves state-of-the-art performance, significantly outperforming both dense baselines and semantic-only MoEs. Crucially, our analysis confirms that \textsc{CuMA} effectively mitigates mean collapse, preserving cultural diversity. Our code is available at https://github.com/Throll/CuMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13054v1">TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted by ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit exceptional general language processing capabilities, but their memory and compute costs hinder deployment. Ternarization has emerged as a promising compression technique, offering significant reductions in model size and inference complexity. However, existing methods struggle with heavy-tailed activation distributions and therefore keep activations in high precision, fundamentally limiting end-to-end inference acceleration. To overcome this limitation, we propose TWLA, a post-training quantization (PTQ) framework that achieves 1.58-bit weight compression and 4-bit activation quantization while maintaining high accuracy. TWLA comprises three components: (1) Euclidean-to-Manifold Asymmetric Ternary Quantizer (E2M-ATQ) minimizes layer-output error under weight ternarization via a two-stage optimization from Euclidean initialization to manifold relocation; (2) Kronecker Orthogonal Tri-Modal Shaping (KOTMS) applies a Kronecker-structured orthogonal rotation to reshape weights into ternary-friendly tri-modal distributions, while the shared rotation statistically suppresses activation outliers; and (3) Inter-Layer Aware Activation Mixed Precision (ILA-AMP) explicitly introduces adjacent-layer second-order interaction costs in bit allocation and jointly optimizes for the layer-wise disparity of activation quantization gains induced by the shared orthogonal transform, preventing cascades triggered by a few weak layers. Extensive experiments demonstrate that TWLA maintains high accuracy under W1.58A4, while delivering significant inference acceleration. The code is available at <https://github.com/Kishon-zzx/TWLA>.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.14568v2">Given, When, Then, Again: Mining Subscenario Refactoring Candidates in Behaviour-Driven Test Suites with ML Classifiers and LLM-Judge Baselines</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 31 pages, 10 figures, 6 tables, 56 references. v2: retitled; reference list fully corrected and verified; decision-threshold sensitivity analysis and imbalance-robust baseline metrics added; figures restyled. Reproduction package at https://github.com/amughalbscs16/cukereuse_subscenarios_release (Apache-2.0). Upstream cukereuse corpus at https://doi.org/10.5281/zenodo.19754359
    </div>
    <details class="paper-abstract">
      Context. Behaviour-Driven Development (BDD) test suites accumulate duplicated step subsequences. Three published refactoring patterns are available (within-file Background, within-repo reusable-scenario invocation, cross-organisational shared higher-level step), but no prior work automates which recurring subsequences are worth extracting or which mechanism applies. Objective. Rank recurring step subsequences ("slices") by refactoring suitability (extraction-worthy), pre-map each to one of the three patterns, and quantify prevalence across the public BDD ecosystem. Method. Every contiguous L-step window (L in [2, 18]) in a 339-repository / 276-upstream-owner Gherkin corpus is keyed by paraphrase-robust cluster identifiers and counted under three scopes. SBERT / UMAP / HDBSCAN clustering recovers paraphrase-equivalent slices. Three authors label a stratified 200-slice pool against a written rubric. An XGBoost extraction-worthy classifier trained under 5-fold cross-validation is compared with a tuned rule baseline and two open-weight Large Language Model (LLM) judges. Results. The miner produces 5,382,249 slices collapsing to 692,020 recurring patterns. Three-author Fleiss' kappa = 0.56 (extraction-worthy) and 0.79 (mechanism). The classifier reaches out-of-fold F1 = 0.891 (95% CI [0.852, 0.927]), outperforming both the rule baseline (F1 = 0.836, p = 0.017) and the better LLM judge (F1 = 0.728, p = 1.5e-4). 75.0%, 59.5%, and 11.7% of scenarios carry a within-file Background, within-repo reusable-scenario, and cross-organisational shared-step candidate, respectively; the figures are stable under a sweep of the classifier decision threshold. Conclusion. Paraphrase-robust subscenario discovery yields a corpus-wide census of BDD refactoring candidates; pipeline, classifier predictions, labelled pool, and rubric are released under Apache-2.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13020v1">SciR: A Controllable Benchmark for Scientific Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Three paradigmatic forms of inference recur across scientific reasoning: deduction, induction, and causal abduction. Reliably evaluating LLMs on these in scientific settings is currently out of reach: scientific benchmarks built on human annotations are costly and lack mechanistic ground truth, while synthetic logical-reasoning benchmarks do not resemble real scientific documents. We introduce SciR, a benchmark that combines multi-paradigm reasoning with controllable scientific rendering, anchored on three paradigmatic scientific problems. Tasks are generated from formal objects (deduction tree, inductive rule hypothesis, causal graph) to guarantee verifiable answers, then rendered into multi-document scientific discourse via per-track domain-tuned genres. The construction lets us independently vary two difficulty axes: how hard it is to extract the key information needed for inference, and how hard the principled inference itself is. We test six models. Both axes hurt every model, and their effects compound. The rendering even hurts neurosymbolic pipelines, which hand inference to a verified solver. The two axes yield a per-model extraction-vs-inference profile: for instance, reasoning models like deepseek-r1 mostly surpass non-reasoning instruct models on the inference axis. To our knowledge, SciR is the first multi-paradigm scientific-reasoning benchmark with parametric control on both extraction and inference difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13007v1">scLLM-DSC: LLM-Knowledge Enhanced Cross-Modal Deep Structural Clustering for Single-Cell RNA Sequencing</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Clustering is fundamental to scRNA-seq analysis, serving as a cornerstone for identifying cell populations and resolving tissue heterogeneity. However, existing methods focus on mining numerical statistical patterns, suffering from semantic agnosticism by neglecting the intrinsic biological functions encoded by genes. While Large Language Models (LLMs) offer promising semantic capabilities, their direct adaptation to cell clustering is hindered by the structural mismatch between generative pre-training objectives and discriminative downstream tasks. To bridge this gap, we propose scLLM-DSC, a novel LLM-Knowledge Enhanced Cross-Modal Deep Structural Clustering framework. Diverging from data-driven paradigms, scLLM-DSC establishes a semantically-grounded representation by synergizing two views: a Knowledge-Driven Semantic View derived from NCBI gene priors and contextualized Cell2Sentence embeddings, and a Structure-Aware Topological View extracted via a graph-guided encoder. Crucially, we introduce a cross-modal contrastive alignment mechanism to enforce consistency between biological semantics and transcriptomic features within a unified latent space. Extensive benchmarks demonstrate that scLLM-DSC significantly outperforms eleven state-of-the-art baselines in clustering accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.13006v1">Emo-LiPO: Listwise Preference Optimization for Fine-Grained Emotion Intensity Control in LLM-based Text-to-Speech</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted by IJCAI 2026. Emotional TTS, Preference Optimization, Emotion Intensity Control
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based text-to-speech (TTS) systems enable prompt-conditioned emotional control but struggle with fine-grained emotion intensity due to the semantic -- acoustic gap between text and speech. To address this challenge, we formulate emotion intensity control in LLM-based TTS as a learning-to-rank problem and propose Emo-LiPO, a listwise preference optimization framework that aligns prompt-conditioned speech generation with relative emotion intensity expressed in text. Emo-LiPO explicitly models global intensity ordering within each emotion under fixed transcripts, enabling more faithful and continuous emotional expression. We further construct ESD-plus, a multi-speaker dataset with explicit emotion intensity variations, to support fine-grained emotion modeling and evaluation. Experiments on ESD-plus demonstrate that Emo-LiPO significantly improves emotion accuracy and intensity controllability over both supervised- and DPO-based LLM TTS baselines, with particularly pronounced gains at high intensity levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.01572v2">LLM-based Embeddings: Attention Values Encode Sentence Semantics Better Than Hidden States</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Sentence representations are foundational to many Natural Language Processing (NLP) applications. While recent methods leverage Large Language Models (LLMs) to derive sentence representations, most rely on final-layer hidden states, which are optimized for next-token prediction and thus often fail to capture global, sentence-level semantics. This paper introduces a novel perspective, demonstrating that attention value vectors capture sentence semantics more effectively than hidden states. We propose Value Aggregation (VA), a simple method that pools token values across multiple layers and token indices. In a training-free setting, VA outperforms other LLM-based embeddings, even matches or surpasses the ensemble-based MetaEOL. Furthermore, we demonstrate that when paired with suitable prompts, the layer attention outputs can be interpreted as aligned weighted value vectors. Specifically, the attention scores of the last token function as the weights, while the output projection matrix ($W_O$) aligns these weighted value vectors with the common space of the LLM residual stream. This refined method, termed Aligned Weighted VA (AlignedWVA), achieves state-of-the-art performance among training-free LLM-based embeddings, outperforming the high-cost MetaEOL by a substantial margin. Finally, we highlight the potential of obtaining strong LLM embedding models through fine-tuning Value Aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27960v2">LLMs as ASP Programmers: Self-Correction Enables Task-Agnostic Nonmonotonic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have achieved impressive reasoning milestones but continue to struggle with high computational costs, logical inconsistencies, and sharp performance degradation on high-complexity problems. While neuro-symbolic methods attempt to mitigate these issues by coupling LLMs with symbolic reasoners, existing approaches typically rely on monotonic logics (e.g., SMT) that cannot represent defeasible reasoning -- essential components of human cognition. We present "LLM+ASP," a framework that translates natural language into Answer Set Programming (ASP), a nonmonotonic formalism based on stable model semantics. Unlike prior "LLM+ASP" approaches that require manually authored knowledge modules, domain-specific prompts, or evaluation restricted to single problem classes, our framework operates without any per-task engineering and applies uniformly across diverse reasoning tasks. Our system utilizes an automated self-correction loop where structured feedback from the ASP solver enables iterative refinement. Evaluating across six diverse benchmarks, we demonstrate that: (1) stable model semantics allow LLMs to naturally express default rules and exceptions, outperforming SMT-based alternatives by significant margins on nonmonotonic tasks; (2) iterative self-correction is the primary driver of performance, effectively replacing the need for handcrafted domain knowledge; (3) compact in-context reference guides substantially outperform verbose documentation, revealing a "context rot" phenomenon where excessive context hinders constraint adherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12983v1">Structured Testbench Generation for LLM-Driven HDL Design and Verification-Oriented Data Curation</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 9 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Automated testbench generation has become a critical bottleneck in large language model (LLM)-driven Register Transfer Level (RTL) workflows, where large numbers of candidate designs must be verified rapidly and reliably. Existing prompt-based approaches treat testbench generation as unconstrained code synthesis, yielding stochastic outputs with high token cost, low reproducibility, and insufficient coverage. To address this gap, we present STG, a Structured Testbench Generation framework that exploits the inherent structure of hardware designs to generate deterministic testbenches. As a direct verification tool, STG runs 720x faster than an iterative LLM-based testbench generation flow and higher rate of successful compilation, achieves higher coverage, and reduces false-pass verdicts on incorrect DUTs. STG also helps identify errors in RTL generation benchmarks by exposing faulty benchmark testbenches. As a data curation engine, it is 11x faster than LLM-based filtering on a single CPU core with 127x less energy, and the resulting distilled models provide state-of-the-art performance in our multi-benchmark evaluation. As a test-time scaling oracle, it reduces node count by 14-47\%. Our models are available at https://huggingface.co/collections/AS-SiliconMind/siliconmind-v12.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12950v1">Maestro: Workload-Aware Cross-Cluster Scheduling for LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted to the 46th IEEE International Conference on Distributed Computing Systems (ICDCS 2026). 11 pages
    </div>
    <details class="paper-abstract">
      Large Language Model based Multi-Agent Systems (LLM-MAS) have emerged as a powerful paradigm for tackling complex tasks by breaking them into collaborative workflows of specialized LLM-powered agents. However, deploying such multi-agent workloads at scale poses significant system challenges. Each user query spawns an iterative pipeline of LLM calls, greatly amplifying resource consumption compared to single-turn queries. In resource-constrained cloud settings, these workflows face non-deterministic and input-dependent costs at decode stage, heavy-tailed multi-model requirements with memory fragmentation and over-provisioning, and cross-cluster scheduling trade-offs. We present Maestro, a workload-aware scheduling system designed for LLM-MAS serving under strict GPU budgets. Maestro explicitly leverages agent semantics and roles: it predicts the output length and memory usage of each stage and uses this prediction to drive a hierarchical scheduler. At the node level, Maestro enables dynamic multi-model co-location via hierarchical weight caching and elastic memory provisioning. At the cluster level, it performs latency-aware routing to avoid cold-start delays and memory overloads. At the global level, it enforces workflow-aware prioritization to minimize head-of-line blocking for interactive tasks. Across prototype experiments and trace-driven simulations, Maestro reduces KV-reservation HBM by 67.2% and improves high-contention SLO attainment over EDF by 23.6 percentage points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2605.16430v3">A Theory of Training Profit-Optimal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Minor edits for preprint
    </div>
    <details class="paper-abstract">
      Scaling LLMs requires tremendous computational resources, and recent advances in AI have gone hand in hand with massive amounts of capital expenditure. While it is established that scaling up LLMs reliably increases model quality (quantified in terms of loss or downstream evaluations), it is unclear how these quality improvements translate to potential revenue, and whether revenue increases would offset costs of larger-scale training and inference. In this work, we develop an economic model for characterizing the rational behavior of an LLM training firm by combining scaling laws with microeconomic theory. Under our model of firm behavior, LLM quality can be increased with more parameters and training tokens, leading to more potential adoption by consumers, who each have a quality threshold for using the LLM. On the other hand, additional parameters and training tokens both incur additional costs. We analyze the profit maximization problem for this model under compute-bound and data-bound regimes. In the compute-bound regime, optimal model size and token budget track hardware efficiency $E$ (FLOPs/\$) at a near-linear rate; total training cost then scales sub-quadratically in $E$. Data efficiency improvements incentivize larger models and training expenditure. When we are limited to $D$ data, profit-optimal training expenditure scales as $D^2/E$, i.e, increase with data and decreases with hardware efficiency (as well as data efficiency). Finally, we analyze practical trends in training expenditure: current trends are consistent with our most permissive model variants in the compute-bound regime, but are not profit-optimal in the data-bound regime or assuming hardware advances will stall. Overall, our results provide a theory of profit-optimal LLM training, providing a foundation for engaging critically with industry statements and supporting long-term economic decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12935v1">MARS: Margin-Adversarial Risk-controlled Stopping for Parallel LLM Test-time Scaling</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Parallel test-time scaling samples many reasoning traces and majority-votes their answers, improving LLM accuracy but requiring traces to run to completion, incurring substantial computational overhead. We observe that probing partial traces at intermediate checkpoints can extract current answers without disrupting generation, revealing an evolving aggregate vote. Based on this observation, we introduce MARS, a margin-adversarial stopping rule that estimates which active traces are likely to change their answers and stops once the leader remains safe under a conservative bound on future vote movement. The rule separates two sources of uncertainty. It learns the trace-level switch probabilities that determine how much of the current margin is likely to be retained, while handling the harder question of where switching traces land through an adversarial bound calibrated from warmup traces. With true switch probabilities, MARS guarantees with high probability that the early-stopped answer matches the full-budget vote. In practice, a five-feature logistic model closely matches oracle switching behavior. Across three reasoning models and three competition-math benchmarks, MARS saves 25-47% of self-consistency tokens and 14-29% on top of DeepConf Online, a strong confidence-weighted baseline that already filters and truncates weak traces, while matching the accuracy of the corresponding full-budget baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22695v2">LLM-ODDR: A Large Language Model Framework for Joint Order Dispatching and Driver Repositioning</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Published in IEEE Transactions on Intelligent Transportation Systems (TITS)
    </div>
    <details class="paper-abstract">
      Ride-hailing platforms face significant challenges in optimizing order dispatching and driver repositioning operations in dynamic urban environments. Traditional approaches based on combinatorial optimization, rule-based heuristics, and reinforcement learning often overlook driver income fairness, interpretability, and adaptability to real-world dynamics. To address these gaps, we propose LLM-ODDR, a novel framework leveraging Large Language Models (LLMs) for joint Order Dispatching and Driver Repositioning (ODDR) in ride-hailing services. LLM-ODDR framework comprises three key components: (1) Multi-objective-guided Order Value Refinement, which evaluates orders by considering multiple objectives to determine their overall value; (2) Fairness-aware Order Dispatching, which balances platform revenue with driver income fairness; and (3) Spatiotemporal Demand-Aware Driver Repositioning, which optimizes idle vehicle placement based on historical patterns and projected supply. We also develop JointDR-GPT, a fine-tuned model optimized for ODDR tasks with domain knowledge. Extensive experiments on real-world datasets from Manhattan taxi operations demonstrate that our framework significantly outperforms traditional methods in terms of effectiveness, adaptability to anomalous conditions, and decision interpretability. To our knowledge, this is the first exploration of LLMs as decision-making agents in ride-hailing ODDR tasks, establishing foundational insights for integrating advanced language models within intelligent transportation systems. While the current framework incurs higher computational costs than traditional methods, we show that parallel decomposition and model distillation can reduce latency to production-viable levels for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.20208v2">From Benchmarks to Skills: Low-Rank Factors for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Current evaluations of large language models (LLMs) rely heavily on a growing collection of benchmarks and on aggregate benchmark scores, yet it remains unclear what this comparison actually captures, and what these scores reveal about models' underlying capabilities. Here, we propose a new paradigm for LLM evaluation, by asking whether benchmark performance reflects many independent abilities, or rather relies on a small number of shared dimensions. To answer this, we apply Factor Analysis (FA) to a massive performance matrix of LLMs versus benchmarks \((60\times44)\) revealing an \emph{intrinsically low-rank} structure of that matrix. That is, a small number of latent factors captures most of the structure in the full task space. This low-rank geometry reveals substantial redundancy across existing tasks and explains why many benchmarks appear to be measuring overlapping abilities. We further show that these latent factors correspond to coherent, skill-like, dimensions of LLM behavior. Leveraging this latent skill-space, we deliver three practical tools for LLM evaluation and downstream users: (i)~identifying redundant tasks, (ii)~profiling new models using a small subset of tasks, and (iii)~selecting models aligned with desired skill profiles. Our method provides a solid alternative to the de-facto standard of a single aggregate score, and establishes an interpretable and practical framework for understanding and benchmarking LLM core capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12922v1">Polar: A Benchmark for Evaluating Political Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Submitted to ARR 2026 May cycle
    </div>
    <details class="paper-abstract">
      Political bias in large language models (LLMs) is increasingly significant, but difficult to measure reproducibly across political and linguistic contexts. We introduce Polar, a 4,026-instance multiple-choice benchmark that measures political bias through option-level likelihoods rather than prompt-based generation. Polar covers two ideological axes and eight issue categories derived from the Manifesto Project, and evaluates models in parallel across U.S. and South Korean political contexts. Across 38 LLMs, measured bias varies systematically with political context, issue category, model group, and presentation language. All models lean left-progressive on U.S. political content, but show more centered and mixed patterns on South Korean content. Translation experiments further show that presentation language alone can shift measured bias. These findings highlight the need for multilingual and cross-contextual evaluation of political bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12900v1">Zero-source LLM Hallucination Detection with Human-like Criteria Probing</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Accepted at ICML 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often hallucinate by generating factually incorrect or unfaithful content, posing significant risks to their safe use. Detecting such hallucinations is particularly challenging under the zero-source constraint, where no model internals or external references are available, and detection must rely solely on the textual query-answer pair. In this paper, we propose Human-like Criteria Probing for Hallucination Detection (HCPD), a paradigm that emulates the multi-faceted reasoning of human evaluators. Its core is a Human-like Criteria Probing (HCP) mechanism, in which a LLM agent adaptively decomposes its judgment into a weighted set of interpretable criteria and aggregates criterion-specific scores into a final truthfulness measure. To achieve this adaptive capability, we introduce a reward-based alignment scheme using only weak supervision from semantic consistency. At inference, we employ a multi-sampling aggregation strategy to ensure robust decisions while preserving full interpretability. We further provide theoretical analysis supporting the reliability of our approach. Extensive experiments show that HCPD consistently outperforms state-of-the-art baselines, offering an effective and explainable solution for zero-source hallucination detection. Code is available at https://github.com/TRISKEL10N/HCPD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09379v3">LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at https://github.com/Lingxi-mental-health/LingxiDiagBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.08794v3">One Token to Fool LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trusted as automated judges, assisting evaluation and providing reward signals for training other models, particularly in reference-based settings like Reinforcement Learning with Verifiable Rewards (RLVR). However, we uncover a critical vulnerability even in this reference-based paradigm: generative reward models are systematically susceptible to reward hacking. We find that superficial inputs, which we term ''master keys'' such as non-word symbols (e.g., '':'' or ''.'') or generic reasoning openers (e.g., ''Thought process:'' or ''Let's solve this problem step by step.''), can consistently elicit false positive rewards without any substantive reasoning. Our systematic evaluation demonstrates this is a widespread failure affecting a diverse range of models, including leading proprietary systems such as GPT-o1 and Claude-4. These results challenge the assumed robustness of LLM judges and pose a significant threat to their reliability. To address this, we propose a simple yet effective data augmentation strategy using truncated model outputs as adversarial negative examples. The resulting Master Reward Models (Master-RMs) demonstrate state-of-the-art robustness against these ''master key'' attacks while maintaining high performance in standard evaluation settings. We supplement these findings with a comprehensive analysis of the vulnerability across model scales, prompt variations, and common inference-time strategies, offering insights to guide future research on robust LLM evaluation. We release our robust, general-domain reward models and the synthetic training data at https://huggingface.co/sarosavo/Master-RM and https://huggingface.co/datasets/sarosavo/Master-RM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12876v1">Multi-Bitwidth Quantization for LLMs Using Additive Codebooks</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 37 pages, 12 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed across heterogeneous hardware with varying resource constraints, the ability to adaptively manage the trade-off between performance and efficiency without retraining is critical. We propose Drop-by-Drop, a novel multi-bitwidth post-training quantization framework that enables inference-time precision control over LLM weights from a single trained model. Our method is theoretically grounded in information theory and successive refinement. We establish that LLM weights, which commonly follow a Gaussian distribution, can be optimally reconstructed with increasing fidelity as additional bits are incorporated, under a weighted mean squared error distortion motivated by LLM loss functions. To realize this in practice, Drop-by-Drop incorporates Matryoshka-style supervision into the loss function, exploiting the structure of additive codebooks. Drop-by-Drop produces a single model where ordered subsets of codebooks yield accurate partial reconstructions at each precision level. This approach significantly reduces storage and memory overhead by allowing a single checkpoint to serve multiple bitwidths, while maintaining competitive perplexity and accuracy across major architectures, such as Qwen, LLaMA, Gemma, and Mistral.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.16548v2">A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      The emergence of writable, cross-session persistent memory in LLM agents introduces a qualitatively different threat landscape from conventional input-centric security concerns, characterized by three properties: persistence, statefulness, and propagation. To systematically characterize this landscape, we propose a Memory Lifecycle Framework that organizes attacks, defenses, and their cross-phase dependencies along two axes: six lifecycle phases (Write, Store, Retrieve, Execute, Share & Propagate, Forget & Rollback) and four security objectives (Integrity, Confidentiality, Availability, Governance). This analysis in turn exposes the need for formal security guarantees at the system level, motivating Verifiable Memory Governance(VMG), a framework of five architectural primitives that specifies what verifiable mechanisms a long-term-memory system must provide to maintain auditable, recoverable control over its memory state. Our analysis indicates that robust Long-Term Memory (LTM) security cannot be retrofitted at retrieval or execution time alone, but must be anchored in storage-time provenance, versioning, and policy-aware retention from the outset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12854v1">Small LLMs for Biomedical Claim Verification: Cost-Effective Fine-Tuning, Structural Dataset Shortcuts, and Cross-Domain Generalization</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 8 pages, 2 figures, 12 tables. To appear at BioNLP Workshop, ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models such as GPT-4o and GPT-5 achieve strong zero-shot performance on biomedical claim verification, but cost and opacity limit scalable use. We fine-tune three small LLMs: Phi-3-mini (3.8B), Qwen2.5-3B, and Mistral-7B, via QLoRA on SciFact and HealthVer, providing the first study of QLoRA models against GPT-4o and fine-tuned BioLinkBERT encoders. Mistral-7B QLoRA surpasses both GPT-4o and GPT-5 (up to 12% F1 gain) at a fractional cost using just 1,008 training examples. We conduct extensive in-domain and cross-domain evaluation: models trained on SciFact tested on HealthVer and vice versa, at matched sizes to isolate dataset structure from data quantity. We identify a previously unreported structural artifact in SciFact that inflates in-domain scores, and show through bidirectional out-of-domain evaluation that training on structurally sound data enables robust cross-domain transfer. We plan to release all code and adapter checkpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11898v2">GraspLLM: Towards Zero-Shot Generalization on Text-Attributed Graphs with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Research on Text-Attributed Graphs (TAGs) has gained significant attention recently due to its broad applications across various real-world data scenarios, such as citation networks, e-commerce platforms, social media, and web pages. Inspired by the remarkable semantic understanding ability of Large Language Models (LLMs), there have been numerous attempts to integrate LLMs into TAGs. However, existing methods still struggle to generalize across diverse graphs and tasks, and their ability to capture transferable graph structural patterns remains limited. To address this, we introduce the GraspLLM, a framework that combines Graph structural comprehension with semantic understanding prowess of LLMs to enhance the cross-dataset and cross-task generalizability. Specifically, we represent node texts from different graphs in a unified semantic space with a frozen general embedding model, on top of which we perform motif-aware contrastive learning across multiple motif-induced adjacency matrices to extract dataset-agnostic structural information. Then, with our proposed optimal contextual subgraph, we extract the most contextually relevant subgraph for each target node and align these subgraphs to the token space of LLM via an alignment projector. Extensive experiments on TAG benchmark datasets spanning diverse domains reveal that GraspLLM consistently outperforms previous LLM-based methods for TAGs, especially in zero-shot scenarios, highlighting its strong generalizability across different datasets and tasks. Our code is available at https://github.com/Heinz217/GraspLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10231v2">LLM can Read Spectrogram: Encoder-free Speech-Language Modeling</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Recent speech-aware large language models (Speech-LLMs) rely on a pre-trained speech encoder to convert audio into semantic-rich representations consumable by LLM. In this work, instead, we explore: can an LLM learn to read Mel spectrogram directly without a dedicated speech encoder? We propose Mel-LLM, an encoder-free Speech-LLM that feeds lightly pre-processed Mel spectrogram patches directly into the LLM through a linear projection, allowing the LLM to learn speech-text alignment purely through its own parameters. We conduct extensive experiments on both automatic speech recognition (ASR) and text-to-speech (TTS) tasks. For ASR, we evaluate on the OpenASR leaderboard public sets and production-level scaling experiments, demonstrating that the encoder-free solution achieves competitive performance with only limited degradation compared to encoder-initialized counterparts. We find that when data is limited, initialization from a multimodal checkpoint (Phi-4-MM) is crucial for maintaining performance. We also present ablation studies revealing which LLM layers are less relevant to speech encoding. For TTS, we show preliminary results with a next-token VAE approach. While TTS performance is not yet optimal, these results establish the feasibility of a fully unified encoder-free architecture for autoregressive speech-text modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24079v2">The Pragmatic Persona: Discovering LLM Persona through Bridging Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 15 pages, 4 figures, accepted to ICPR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) reveal inherent and distinctive personas through dialogue. However, most existing persona discovery approaches rely on surface-level lexical or stylistic cues, treating dialogue as a flat sequence of tokens and failing to capture the deeper discourse-level structures that sustain persona consistency. To address this limitation, we propose a novel analytical framework that interprets LLM dialogue through bridging inference -- implicit conceptual relations that connect utterances via shared world knowledge and discourse coherence. By modeling these relations as structured knowledge graphs, our approach captures latent semantic links that govern how LLMs organize meaning across turns, enabling persona discovery at the level of discourse coherence rather than surface realizations. Experimental results across multiple reasoning backbones and target LLMs, ranging from small-scale models to 80B-parameter systems, demonstrate that bridging-inference graphs yield significantly stronger semantic coherence and more stable persona identification than frequency or style-based baselines. These results show that persona traits are consistently encoded in the structural organization of discourse rather than isolated lexical patterns. This work presents a systematic framework for probing, extracting, and visualizing latent LLM personas through the lens of Cognitive Discourse Theory, bridging computational linguistics, cognitive semantics, and persona reasoning in large language models. Codes are available at https://github.com/JiSoo-Yang/Persona_Bridging.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12821v1">GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis Across Frontier and Open-Weight Foundation Models</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Preprint. 10 pages, 8 figures. Submitted to ACM SIGSPATIAL 2026
    </div>
    <details class="paper-abstract">
      Environmental scientists spend disproportionate effort on data wrangling rather than analysis, and AI agents that automate geospatial workflows remain unvalidated: no benchmark evaluates agents operating through structured tool calling against real APIs. We introduce the GeoNatureAgent Benchmark, the first benchmark for environmental analysis agents that operate via structured tool calls to a production-style geospatial API. It comprises 93 tasks across 18 categories, covering municipality analysis, multi-turn conversation, spatial reasoning, cross-indicator synthesis, error handling and recovery, ranking, comparison, multilingual understanding, habitat analysis, and task rejection. Tasks are evaluated against an open, self-hostable API serving three environmental indicators across Spain and Portugal via sixteen tools. We evaluate seven LLMs (Claude Sonnet 4, DeepSeek V3.2, GLM-5, Gemini 2.5 Pro, Qwen3-235B, GPT-OSS-120B, Llama 4 Scout) under three temperature-1.0 seeds, reporting capability and per-case cost as orthogonal axes. We find: (1) Claude Sonnet 4 leads at 60.8% +/- 0.8%, followed by DeepSeek V3.2 at 56.3% +/- 3.1%, with no other model above 51%; (2) the cost-accuracy Pareto frontier is occupied mostly by open-weight models, with DeepSeek V3.2 offering 93% of Claude's capability at 11x lower cost ($0.011/case); (3) comparison tasks remain universally unsolved (0% on close-value comparisons), exposing systematic reasoning limits; and (4) structured tool calling against a real API is more discriminative than general-purpose GIS benchmarks, with accuracies 25-35 points lower. We further show extensibility by integrating BigEarthNet V2 land cover for Portugal alongside Spanish CO2 and erosion indicators. The benchmark, harness, and self-hostable API are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12801v1">AiAWE: An Open-Source LLM Automated Writing Evaluation System Using LoRA-Adapted Instruction-Tuned Models</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 21 pages with 7 tables and 1 figure and appendices
    </div>
    <details class="paper-abstract">
      This study presents AiAWE, an open-source automated writing evaluation system that scores argumentative essays using a LoRA-adapted instruction-tuned large language model (Gemma-3-27B-it). Using a proprietary Educational Testing Service (ETS) dataset of 480 TOEFL Independent Writing essays, we fine-tune Gemma-3-27B and LLaMA-3.3-70B under identical LoRA configurations on a 120-essay training subset and evaluate on the remaining 360 essays under identical inference quantization. The fine-tuned Gemma model achieves a root mean square error of 0.474, a quadratic weighted kappa of 0.828, and an agreement rate of 90.56% within +/- 0.5 of the human score, outperforming both the larger LLaMA-3.3-70B model and the fine-tuned GPT-3.5 baseline reported in prior work on the same dataset. Three findings are of broader interest: open-weight LLMs can match or exceed proprietary fine-tuning for rubric-aligned scoring; model scale is not a reliable predictor of downstream performance under LoRA adaptation; and identical LoRA hyperparameters produce qualitatively different adaptation behaviors across architectures. The production system runs on a consumer-grade server and is publicly accessible at https://app.awade.gec.waseda.ac.jp. LoRA adapters, application code, and fine-tuning YAMLs are publicly available through their respective repositories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12780v1">ProPlay: Procedural World Models for Self-Evolving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-11
    </div>
    <details class="paper-abstract">
      Self-evolving agents are expected to improve through interaction without external supervision, but this remains difficult in partially observable environments where agents must explore actively, learn from limited feedback, and decide when to trust prior experience. Existing LLM-agent methods often rely on memory or planning modules, yet they rarely close the loop between them to continually refine an internal understanding of environment dynamics. We introduce ProPlay, a procedural world model that supports procedure-level preplay, where agents can rehearse future procedural paths using the learned world knowledge. Rather than representing experience as isolated rules or low-level action constraints, ProPlay abstracts successful trajectories into procedures and organizes them in a procedure graph that captures causal transitions among task stages. Each transition is associated with a reliability record embedding to estimate its task-specific contribution from past outcomes. Before each episode, ProPlay simulates future procedural trajectories over known graph structures as structured soft guidance; after execution, it refines the graph using environment feedback. Experiments on public benchmarks show that ProPlay consistently improves environment understanding and self-evolution capability over strong baselines. Our code has been released in https://github.com/antman9914/proplay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18085v4">Structuring The Future: Diffusion LLM Speculative Decoding via Calibrated Draft Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-11
      | 💬 Original version uploaded on Sep 22, 2025. (v2): Extended Table 2 with additional analysis and referenced it in Sec 5.2. (v3): Added note to Sec 4.2 and Appendix A.2 specifying conditions for losslessness. (v4): Updated with the version accepted to ICML 2026 workshops
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token-generation rates. To unlock this potential, we present Spiffy, a speculative decoding algorithm to accelerate dLLM inference while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to dLLMs. Spiffy performs auto-speculation to eliminate the overheads of an independent draft model, structuring draft states in the form of a novel directed draft graph to take advantage of the bidirectional, blockwise nature of dLLM generation. These draft graphs are calibrated offline to maximize acceptance rates and are dynamically pruned during inference for improved computational efficiency. We present a detailed formulation of Spiffy and demonstrate its ability to accelerate LLaDA, Dream, and SDAR models in combination with KV caching and threshold-based dynamic unmasking leading to up to $8.6\times$ reduction in model inferences and $6.3\times$ acceleration in token rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11755v1">Acoda: Adversarial Code Obfuscation for Defending against LLM-based Analysis</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs) in software engineering (SE) tasks such as code understanding, debugging, and vulnerability detection, their powerful semantic reasoning ability has also introduced new security and privacy risks. LLMs can analyze, reconstruct, or even reverse-engineer source code logic, potentially leading to the leakage of intellectual property. To address this issue, we propose Acoda, a genetic algorithm-based adversarial code obfuscation framework that defends against LLM-based code analysis. Acoda leverages two key mechanisms of LLMs, namely safety alignment and token-based information processing, to design 8 semantics-preserving obfuscation methods. It iteratively optimizes obfuscation strategies through a genetic algorithm to generate adversarial samples that maximize defensive effectiveness. In addition, we propose a quantitative evaluation framework based on LLM responses, which combines an auxiliary LLM and four evaluation metrics to assess how target LLMs analyze obfuscated code comprehensively. Experimental results show that Acoda can effectively induce LLMs to refuse or misinterpret code analysis. On 7 state-of-the-art LLMs, including GPT-4o, DeepSeek, Qwen, Llama, and Gemma, Acoda achieves an attack success rate (ASR) of up to 70%, with strong cross-model transferability and minimal runtime overhead, while ensuring that the semantics of the original code remain unchanged. Overall, this study provides a new perspective for code protection and LLM security defense in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12479v1">ReCal: Reward Calibration for RL-based LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) routing has emerged as an effective paradigm for leveraging the complementary strengths of multiple LLMs through dynamic model and reasoning-strategy selection. Recent reinforcement learning (RL)-based routing methods further improve routing quality by optimizing routing policies from interaction feedback. However, they still struggle to provide informative and comparable learning signals under heterogeneous tasks with varying difficulty. In practice, multiple objectives (e.g., correctness, format behavior) are aggregated into a single scalar reward, leading to ambiguous credit assignment and conflicting optimization signals. Moreover, reward signals exhibit significant variability across instances, where some instances produce higher or more variable rewards, introducing optimization bias that favors trivial samples over informative ones. To address these issues, we propose \textbf{ReCal}, a \textbf{\underline{Re}}ward \textbf{\underline{Cal}}ibration framework for RL-based LLM routing. We first introduce a hierarchical reward decomposition mechanism with component-wise advantage estimation. We further propose a distribution-aware optimization strategy that calibrates optimization variability through variance-aware reweighting and per-dataset normalization. Experiments on seven datasets demonstrate that ReCal consistently improves routing performance, and training stability over baselines. Code is available at https://anonymous.4open.science/r/ReCal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18636v2">LaQual: An Automated Framework for LLM App Quality Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Representing a new paradigm in software distribution, LLM app stores are rapidly emerging, offering users diverse choices for content generation, coding assistance, education, and more. However, current ranking and recommendation mechanisms in LLM app stores predominantly rely on static metrics, such as user interactions and favorites, making it challenging for users to efficiently identify high-quality apps. At the same time, current academic research focuses on specific vertical fields and lacks a general, automated evaluation framework applicable to the diverse LLM app ecosystem. To address the above challenges, we present LaQual, an automated framework for LLM app quality evaluation. LaQual integrates three key stages: (1) LLM app labeling and hierarchical classification for precise scenario mapping; (2) static indicator evaluation using time-weighted user engagement and functional capability indicators to filter low-quality apps; and (3) dynamic scenario-adapted evaluation, where an LLM generates scenario-specific evaluation metrics, scoring criteria, and tasks for comprehensive quality evaluation. Experiments on a mainstream LLM app store demonstrate the effectiveness of LaQual. Its automated scores show high consistency with human judgments. Through effective screening, LaQual can reduce the candidate LLM app pool by 66.7% to 81.3%. User studies further validate its significant outperformance over baseline systems, particularly in comparison efficiency (mean 5.45 vs. 3.30) and value of explanatory information (4.75 vs. 2.25). These results demonstrate that LaQual provides a scalable, objective, and user-centric solution for high-quality discovery and recommendation of LLM apps in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11700v1">CompRank: Efficient LLM Reranking via Token-Level Compression and Decoding-Free Scoring</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) rerankers have become an important component of modern retrieval and retrieval-augmented generation pipelines, but their high computational cost limits their applicability to long candidate lists. In this paper, we propose \textbf{CompRank}, a token-efficient reranking framework that reduces redundant computation by aligning reranker design with the sparsity of ranking signals. CompRank decouples document representations from candidate order and query context, enabling reusable document-side states; applies segment-wise token compression to reduce query--document interaction cost; and introduces a CopyNet-style objective that directly aligns attention-based document scoring with training supervision. Experiments on seven BEIR datasets show that CompRank achieves strong reranking performance while retaining only 10.2\% of document tokens, reaching an average NDCG@10 of 39.2 compared with 39.7 under full-token attention. Further scaling experiments on TREC-COVID show that CompRank remains stable when evaluated on candidate lists of up to 500 documents after training on 30-document lists, while achieving $4.9\times$--$9.5\times$ end-to-end speedup over generation-based listwise reranking and approximately $1.3\times$ speedup over the full-token CompRank variant. These results suggest that token-level compression and decoding-free attention scoring provide an effective path toward scalable LLM-based reranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10749v2">AttriGuard: Defeating Indirect Prompt Injection in LLM Agents via Causal Attribution of Tool Invocations</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Accepted by USENIX Security 2026
    </div>
    <details class="paper-abstract">
      LLM agents are highly vulnerable to Indirect Prompt Injection (IPI), where adversaries embed malicious directives in untrusted tool outputs to hijack execution. Most existing defenses treat IPI as an input-level semantic discrimination problem, which often fails to generalize to unseen payloads. We propose a new paradigm, action-level causal attribution, which secures agents by asking why a particular tool call is produced. The central goal is to distinguish tool calls supported by the user's intent from those causally driven by untrusted observations. We instantiate this paradigm with AttriGuard, a runtime defense based on parallel counterfactual tests. For each proposed tool call, AttriGuard verifies its necessity by re-executing the agent under a control-attenuated view of external observations. Technically, AttriGuard combines teacher-forced shadow replay to prevent attribution confounding, hierarchical control attenuation to suppress diverse control channels while preserving task-relevant information, and a fuzzy survival criterion that is robust to LLM stochasticity. Across four LLMs and two agent benchmarks, AttriGuard achieves 0% ASR under static attacks with negligible utility loss and moderate overhead. Importantly, it remains resilient under adaptive optimization-based attacks in settings where leading defenses degrade significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11690v1">Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 26 pages, 9 figures. Code: https://github.com/pChitral/vllm-cost-meter
    </div>
    <details class="paper-abstract">
      Every public LLM cost calculator we surveyed treats GPU utilization as a fixed input -- entered by the user, baked in as a preset, or silently assumed at 100% -- never measured against the operator's actual load. We show that this assumption is the dominant source of error: on identical H100 hardware, effective cost spans \$0.21 to \$15.25 per million output tokens, an underutilization penalty of 2.5-24x across low-to-moderate enterprise loads (1-10 rps) and up to 36.3x near idle -- driven by one operator-controlled variable, offered request rate lambda, which sets in-flight concurrency via Little's Law and which no open-source calculator exposes. Because calculators take utilization as a user-supplied input, any utilization-naive estimate understates true cost by exactly 1/U, systematically mispricing self-hosting -- most severely over-selling it for low-traffic workloads. We propose a measurement methodology that parameterizes the relationship as C_eff = f(H, M, Q, lambda, L), validate it with 42 benchmarks across dense, ultra-sparse MoE, and sparse MoE models, and release vllm-cost-meter, an open-source cost meter that attaches to a live vLLM server and reports real \$/M-tokens against the operator's own traffic. We further show that FP8 quantization benefits the MoE architectures we tested roughly 2.2-2.4x more than the dense model (+69 to +74% vs. +31% peak throughput; n=3, broader validation needed), and our data are consistent with active parameter count, not total model size, being a primary predictor of saturation economics. To rule out single-hardware confounding we repeat the core sweep on A100 80GB PCIe (56 runs): the load-driven spread reproduces at 7.0-11.4x, the active-parameters ordering survives at FP8, and the dense-FP8 advantage inverts on silicon without native FP8 tensor cores -- a hardware-conditional caveat the framework already accommodates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11686v1">Layer-Isolated Evaluation: Gating the Deterministic Scaffold of a Production LLM Agent with a No-LLM, Regression-Locked Test Harness</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 12 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      End-to-end task-success is the dominant way to evaluate LLM agents, but one aggregate number tells you that an agent regressed, not where. We present layer-isolated evaluation: a deployed ordering agent is decomposed into a fixed taxonomy of layers (ontology, intent, routing, decomposition, escalation, safety, memory, and cross-cutting envelope/defense), each exercised by its own assertion slice in a deterministic, no-LLM "pure" mode. The pure suite (238 cases across 23 slices; 225 run in 2.39 s, ~10 ms/case) runs in CI on every change against a locked per-slice baseline. We validate by controlled regression injection, degrading one layer at a time across seven non-safety layers. The effect we did not design in is masking: the aggregate pass-rate barely moves (-1.7 to -5.9 pp for six local regressions), while the matching slice craters (-25 to -91 pp). A layer's slice reacting to its own fault is partly by construction; the measured results are (i) the aggregate masking and (ii) that damage stays off the other slices: the injected layer's slice is the single worst-hit in 5 of 7 cases and top-3 in 7 of 7 (mean rank 1.29 of 19). Localization replicates on a second, structurally different tenant (Starbucks SG): all seven matching slices crater, so it is not a single-catalog artifact. We position it as a concrete, deterministic instantiation of the component-level evaluation EDDOps prescribes but leaves unimplemented, with CheckList as ancestor and as the deterministic mirror image of whole-workflow stochastic mutation testing. Our contributions: (a) a fully decomposed, sub-second, no-LLM per-layer harness for a production agent, (b) a coverage-honesty test-adequacy criterion that refuses to score an unexercised layer, and (c) the regression-injection demonstration that per-slice baseline-locked gates localize regressions an aggregate metric masks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00945v2">Neural FOXP2 -- Language Specific Neuron Steering for Targeted Language Improvement in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      LLMs are multilingual by training, yet their lingua franca is often English, reflecting English language dominance in pretraining. Other languages remain in parametric memory but are systematically suppressed. We argue that language defaultness is governed by a sparse, low-rank control circuit, language neurons, that can be mechanistically isolated and safely steered. We introduce Neural FOXP2, that makes a chosen language (Hindi or Spanish) primary in a model by steering language-specific neurons. Neural FOXP2 proceeds in three stages: (i) Localize: We train per-layer SAEs so each activation decomposes into a small set of active feature components. For every feature, we quantify English vs. Hindi/Spanish selectivity overall logit-mass lift toward the target-language token set. Tracing the top-ranked features back to their strongest contributing units yields a compact language-neuron set. (ii) Steering directions: We localize controllable language-shift geometry via a spectral low-rank analysis. For each layer, we build English to target activation-difference matrices and perform layerwise SVD to extract the dominant singular directions governing language change. The eigengap and effective-rank spectra identify a compact steering subspace and an empirically chosen intervention window (where these directions are strongest and most stable). (iii) Steer: We apply a signed, sparse activation shift targeted to the language neurons. Concretely, within low to mid layers we add a positive steering along the target-language dominant directions and a compensating negative shift toward the null space for the English neurons, yielding controllable target-language defaultness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11675v1">Lung-R1: A Knowledge Graph-Guided LLM for Pulmonary Diagnostic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Diagnosing pulmonary diseases requires integrating heterogeneous evidence amid phenotypic variability and cross-disease overlap. Although large language models (LLMs) have shown progress on pulmonary knowledge question answering (QA) and information-processing tasks, reliable pulmonary diagnosis requires patient-specific, relation-aware reasoning over electronic medical record (EMR) evidence rather than isolated knowledge recall. We define this gap between pulmonary knowledge and case-level diagnostic reasoning as the Pulmonary Knowledge-to-Diagnosis Gap. To address it, we introduce LungKG, the first structured pulmonary knowledge graph for diagnostic knowledge organization and record-grounded reasoning. LungKG contains 59,038 nodes and 164,308 edges across 15 entity types and 112 relation types, serving as both a reusable pulmonary knowledge resource and the foundation for LungKG-guided model adaptation. Built on LungKG, we propose Lung-R1, a LungKG-guided pulmonary LLM trained through KG-constrained reasoning-chain construction and KG-guided reinforcement learning. In a 20-system evaluation, Lung-R1-14B achieves state-of-the-art performance across Choice, Pulmonary-QA, and EMR Diagnosis, reaching an EMR Diagnosis score of 4.3583 and surpassing the strongest non-Lung-R1 baseline by 0.1476 points. These results demonstrate the value of LungKG-guided training for EMR-based pulmonary diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12474v1">SAIGuard: Communication-State Simulation for Proactive Defense of LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems (MAS) solve complex tasks through inter-agent collaboration, but their communication-driven nature also allows security risks to spread across agents and trigger system-wide failures. Existing MAS defenses mainly follow a reactive paradigm after execution by detecting and isolating harmful agents, which may cause irreversible damage and degrade collaborative utility. To address this, we propose a proactive defense framework for MAS security, namely a Simulation-aware Interception Guard (SAIGuard). SAIGuard performs communication-state simulation over the MAS interaction graph, estimates the impact of incoming messages on local agent states and the global MAS state, and detects risky messages via reconstruction deviations from benign communication patterns. Instead of isolating agents, SAIGuard sanitizes or regenerates suspicious messages before it propagation into system. Experiments across diverse topologies and attack scenarios show that SAIGuard reduces attack success rates while maintaining MAS utility, outperforming reactive defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11672v1">Can Open-Source LLM Agents Replace Static Application Security Testing Tools? An Empirical Assessment</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Keywords: Agentic AI, Cybersecurity, Large Language Models, Static Application Security Testing, Model performance evaluation
    </div>
    <details class="paper-abstract">
      This paper explores the value of agentic AI tools for cybersecurity purposes. We evaluate the efficacy of a general-purpose GenAI Large Language Model- (GenAI-) based agent when powered by three different Ollama-hosted general-purpose open source models. We assess each agent's performance using precision, recall, false positive count, and a calculated composite score based upon the interplay of the captured metrics, against the baseline performance of an existing, vetted Static Application Security Testing (SAST) tool, Bandit. Our findings refute the notion that a modern open-source GenAI LLM-based agent is currently suitable for the specialized task of SAST scanning under realistic conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.07506v2">Judging Against the Reference: Uncovering Knowledge-Driven Failures in LLM-Judges on QA Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Under review, 21 pgs, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used as automatic judges for question answering (QA) and other reference-conditioned evaluation tasks, little is known about their ability to adhere to a provided reference. We identify a critical failure mode of such reference-based LLM QA evaluation: when the provided reference conflicts with the judge model's parametric knowledge, the resulting scores become unreliable, substantially degrading evaluation fidelity. To study this phenomenon systematically, we introduce a controlled swapped-reference QA framework that induces reference-belief conflicts. Specifically, we replace the reference answer with an incorrect entity and construct diverse pairings of original and swapped references with correspondingly aligned candidate answers. Surprisingly, grading reliability drops sharply under swapped references across a broad set of judge models. We empirically show that this vulnerability is driven by judges' over-reliance on parametric knowledge, leading judges to disregard the given reference under conflict. Finally, we find that this failure persists under common prompt-based mitigation strategies, highlighting a fundamental limitation of LLM-as-a-judge evaluation and motivating reference-based protocols that enforce stronger adherence to the provided reference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11648v1">Dummy Backdoor as a Defense: Removing Unknown Backdoors via Shared Internal Mechanisms for Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Backdoor attacks pose a serious threat to the safety and reliability of Large Language Models (LLMs), as they cause models to behave normally on clean inputs while producing attacker-specified responses when hidden triggers are present. Removing such unknown backdoors is particularly challenging when the defender does not know the backdoor attack types or the internal mechanisms formed through backdoor training. In this work, we propose a simple but effective backdoor removal method based on shared internal mechanisms across different backdoors. First, we show that different backdoors with the same task (attack objective) induce similar trigger-activated changes in the internal activations. Motivated by this observation, our method intentionally embeds a backdoor with a known trigger (\emph{dummy backdoor}) and then removes it through further fine-tuning on dummy-triggered inputs paired with clean responses. Since the dummy backdoor and the unknown backdoor can rely on shared internal mechanisms, removing the dummy backdoor also reduces the effect of the unknown backdoor. We evaluate our method on three backdoor attack types across multiple model families. Experimental results show that our method substantially reduces the attack success rate of the unknown backdoor while preserving model utility, outperforming representative existing defense methods in both backdoor removal effectiveness and utility preservation. These findings suggest that a defender-controllable backdoor can serve as a helpful proxy for mitigating unknown backdoors in generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03855v2">A Sensitivity Analysis of Multi-Event Audio Grounding in Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 6 pages, Accepted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Audio LLMs have shown a strong ability to understand audio samples, yet their reliability in complex acoustic scenes remains under-explored. Unlike prior work limited to small scale or less controlled query construction, we present a large-scale evaluation of event grounding and false alarms as auditory scene complexity increases. Using 71K AudioCapsV2 clips, we extract normalized (source, attribute) events and build two query types: present-event queries for ground-truth detection and absent-event queries to probe hallucinations, using similarity-filtered negative sampling in an audio-aligned text embedding space. We evaluate four SOTA Audio LLMs with 12 prompt variants over 500K yes/no queries per model. Across models, increasing event count consistently lowers true-positive rate and raises false-positive rate, while prompts induce a strong trade-off between the two. Our confidence analysis shows that models become more uncertain on multi-event audio, revealing room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11640v1">TAROT: Task-Adaptive Refinement of LLM-prior Graphs for Few-shot Tabular Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Few-shot tabular learning provides a cost-effective approach for real-world applications where annotation is costly and collecting sufficient samples for new tasks is difficult. Existing Traditional and LLM-based methods have demonstrated effectiveness in few-shot scenarios. However, traditional methods need additional training on unlabeled or generated data, which incur significant computational overhead. In addition, LLM-based methods that directly feed raw tabular data into LLMs raise privacy and compliance concerns. More importantly, both paradigms largely overlook the semantic relationships between features, which provide structural and semantic prior for constructing a semantic graph. Semantic graph is essential for modeling meaningful feature interactions in few-shot scenarios. In this paper, we propose TAROT, a GNN-based framework that encodes the structural and semantic prior by constructing and refining a task-adaptive semantic graph from this prior, thereby improving predictive performance in few-shot tabular learning. TAROT first encodes heterogeneous tabular data into unified node semantic representations via a Unified Semantic Tabular Node Encoder (USTNE). Then, it prompts LLMs to infer the semantic relationship between features based on the task description and feature names to construct a semantic graph. To mitigate structural noise introduced by the hallucination of LLMs, TAROT introduces Task-adaptive Semantic Graph Refinement that prunes spurious or task-unrelated edges and adds missing task-related ones, aligning the graph structure with the downstream objective. Finally, a GNN performs message passing over the refined graph to capture task-related semantic dependencies for prediction. Extensive experiments on various few-shot tabular learning benchmarks demonstrate the superior performance of TAROT, establishing it as a state-of-the-art approach in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11635v1">Are LLMs Bad at Moral Reasoning?</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      For highly capable AI systems to operate safely in dynamic, open-ended environments, they must be able to identify, understand, and respond to moral reasons for action, and constrain their behaviour accordingly. A growing body of research aims to evaluate this capacity -- moral competence -- in today's most capable AI systems, recently reaching broadly pessimistic conclusions. One of the most ambitious such papers collects gold-standard human-authored rubrics for evaluating moral reasoning in 1,000 cases, and benchmarks frontier AI models against those rubrics, with underwhelming results. In this paper, we argue that the MoReBench dataset can be redeployed to give a much more optimistic picture of LLMs' moral reasoning (an essential part of moral competence). We show that if, instead of scoring LLMs' responses to these cases against these rubrics, we instead give the LLMs the same task given to humans -- to generate scoring rubrics for the moral analysis of particular cases -- the rubrics they generate are both better calibrated to the human rubrics than their open-ended responses, and, where they differ, plausibly reflect nothing more than the vast dimensionality of most moral problems, as well as highlighting some human departures from the "rubric for creating rubrics". Taking these points into consideration, the MoReBench dataset suggests that LLMs are significantly more capable at moral reasoning than was previously believed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04710v2">Steering the Noise: Turning Random Perturbations into Effective Descent for Memory-Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 12pages, 6figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) achieves strong performance but is often limited by the memory overhead of backpropagation. Zeroth-order (ZO) optimization avoids this overhead by estimating gradients through forward passes alone, yet it typically converges slowly because random Gaussian perturbations yield high-variance gradient estimates in high-dimensional parameter spaces. In this paper, we propose a plug-and-play framework that turns random perturbations into more effective descent directions. The key idea is to draw a small pool of candidate perturbations, evaluate their loss values, and then select or combine those that are best aligned with the optimization objective. We develop two instantiations of this idea: MeZO-GV, which forms a guiding vector from the contrast between low-loss and high-loss perturbation groups, and MeZO-Greedy, which keeps the single best perturbation within a fixed evaluation budget. We theoretically show that both strategies yield a larger per-step reduction in the objective than standard ZO estimation, leading to improved convergence rates. Experiments on LLMs of different scales and architectures confirm that the proposed methods integrate naturally with existing ZO optimizers and consistently improve convergence speed and task accuracy. On OPT-13B, our approach outperforms all ZO baselines across 11 benchmarks and exceeds gradient-based methods on 9 of them, while retaining the memory efficiency of forward-only optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10120v2">MetaPlate: Counterfactual-Guided RAG-LLM Tool for Personalized Food Recommendation and Hyperglycemia Prevention</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Postprandial hyperglycemia is a key risk factor for metabolic disorders; however, existing dietary guidance is often static, impractical, and insufficiently personalized, providing recommendations that are difficult to follow or not impactful. While recent advances leverage continuous glucose monitoring (CGM) and machine learning to predict glycemic responses, these approaches are largely predictive and lack actionable guidance. Moreover, recommendation systems are often misaligned with user goals and require extensive input. We present MetaPlate, a counterfactual explanation (CF) guided, context-aware decision-support framework that generates personalized meal recommendations to mitigate postprandial glucose excursions in healthy adults. MetaPlate integrates multimodal data, including CGM readings, wearable-derived physiological signals, and user-provided meal inputs from $25$ individuals to model pre-meal context. A machine learning model predicts glucose response, while a CF optimization module adjusts meal composition modifying macronutrient amounts to maintain glucose levels within a target range ($\leq 140$ mg/dL). An LLM-based retrieval-augmented generation (RAG) layer enhances interpretability by producing human-readable recommendations using constrained search of the USDA food database. We evaluate MetaPlate via a structured expert-in-the-loop assessment with registered dietitians (RDs), comparing performance before and after prompt refinement. Results show improvements in meal realism, portion suitability, and recommendation likelihood, with expert feedback indicating a shift from clinically implausible outputs to actionable, contextually appropriate recommendations. Our findings emphasize the importance of domain knowledge and structured constraints in LLM-driven systems and highlight the potential of MetaPlate as a real-time personalized dietary decision-support tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11592v1">Defense Against Prompt Inversion Attacks: An Information-Theoretic Approach for LLM Collaborative Inference</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Preprint. 33 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Collaborative edge-cloud inference enables resource-constrained devices to leverage large language models (LLMs) by offloading partial computation to cloud servers. However, transmitting intermediate activations exposes sensitive user prompts to prompt inversion attacks, where an adversary reconstructs the original input from shared representations. Existing defenses rely largely on heuristic perturbations or empirical tuning, offering limited theoretical understanding of privacy leakage and its interaction with utility and latency constraints. We propose an information-theoretic defense framework for prompt inversion in collaborative LLM inference. Our approach learns privacy-preserving representations by explicitly minimizing the mutual information between intermediate activations and the input prompt while maintaining task utility under computational constraints. We derive theoretical guarantees on prompt reconstruction error, characterize fundamental privacy-utility tradeoffs, and establish token-level accuracy bounds for downstream inference. We then propose a novel defense based on privacy adapters implemented via low-dimensional information bottlenecks. Extensive experiments across multiple settings demonstrate that our method achieves superior privacy-utility-latency tradeoffs compared to existing defenses (up to 35% reduction in attack success), providing a principled foundation for private and efficient collaborative LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11583v1">Beyond the Golden Teacher: Enhancing Graph Learning through LLM-GNN Co-teaching</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Code: https://github.com/llmgnncoteaching/LLM-GNN-Coteaching
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) underlie real-world applications such as citation networks, social media, and e-commerce. Few-shot graph learning on TAGs is hard: with only a handful of labels per class and the rest of the graph unannotated, neither GNNs nor LLMs can learn well on their own. GNNs read topology and fail on cold nodes; LLMs read text and fail on text-ambiguous nodes. Existing LLM-GNN methods all follow the same recipe: designate one model as the golden teacher and use its outputs (e.g., features or pseudo-labels) to supervise the other. We argue this golden-teacher assumption breaks under sparse supervision: neither model is golden, and treating either as such transfers its blind spots into the student. We therefore ask: can we avoid designating either model as the golden teacher, and still perform effective graph learning? We answer with LLM-GNN Co-Teaching, a bidirectional co-teaching framework in which neither model is fixed as teacher. The GNN and LLM exchange their most confident pseudo-labels under an architecture-specific small-loss criterion, and both update every round. Supervision is then mined from the trajectory: whenever a node moves from cross-model contradiction at round t to cross-model agreement at round t+1, the LLM's two answers on the same input form a preference pair (old contradicting self < new peer-endorsed self) for DPO training. We call this Round-based Pseudo-Label Preference Optimization (RPL-PO). On six benchmarks, LLM-GNN Co-Teaching consistently outperforms GNN-as-Judge and all prior methods, with absolute 3-shot gains of 7.86% on Cora and 7.73% on ogbn-arxiv; improvements carry over to 5-shot and to zero-shot cross-dataset transfer. Error-structure analysis further shows that abandoning the golden-teacher assumption substantially improves the LLM's graph learning capability on challenging samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11562v1">GraphInfer-Bench: Benchmarking LLM's Inference Capability on Graphs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Code: https://github.com/graphinfer/GraphInfer-Bench ; Dataset: https://huggingface.co/datasets/graphinfer/graphinfer
    </div>
    <details class="paper-abstract">
      Graph analysis underlies many applications whose answers cannot be looked up in a single record or retrieved along a path: laundering rings, drug repurposing, user preference, and scientific theme are all inferred from a node together with its neighbourhood. We introduce GraphInfer-Bench, a benchmark for whether LLMs can perform this graph inference: producing an open-ended answer that no single node supports and no path retrieves. Existing graph-QA protocols cannot test this capability: algorithm simulation, node classification, single-node description, KG-QA, and GraphRAG all admit answers retrievable from one node or along a path. GraphInfer-Bench defines five tasks along Description (what a region is) and Comparison (how regions differ), each constructed so the ground truth lives in no single node. The release contains 42,000 samples across six real-world graphs, produced automatically and screened by a four-layer quality-control protocol. We evaluate four method families against the same tasks: graph-token alignment models, zero-shot frontier closed-source LLMs, Graph2Text supervised fine-tuning, and plain GNNs as a structural reference. No method family closes the gap. Graph-token alignment partially handles description tasks (relational, theme) but collapses on comparison tasks. Frontier LLMs lead on outlier detection and community partition among LLM-based methods but lag on masked-node prediction. Graph2Text SFT is the strongest LLM-based method on the description side yet falls behind frontier LLMs on comparison. Across every task, plain GNNs match or beat the strongest LLM-based row, with the largest margin on community detection. GraphInfer-Bench surfaces graph inference as an open capability gap rather than a property of any one architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.11560v1">LLMs+Graphs: Toward Graph-Native, Synergistic AI Systems</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 10 pages, Accepted at PAKDD 2066 Tutorial
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced rapidly, but their limitations in structured and multi-hop reasoning underscore the need for graph-native, synergistic artificial intelligence (AI) systems. Graph-structured data underpins critical applications across social, biological, financial, transportation, web, and knowledge domains, making it essential to understand how LLMs can leverage graph computation for grounded, context-rich inference. Three complementary synergies are emerging: LLMs augmented with graph computation for retrieval and reasoning; bidirectional integration between LLMs and knowledge graphs (KGs), where LLMs support KG construction and curation while KGs enforce semantic constraints and factual consistency; and AI agents strengthened by graph algorithms for planning, decision making, and multi-step reasoning. In parallel, LLMs introduce new capabilities for graph data management and graph machine learning (ML) through natural language interfaces and hybrid LLM-graph neural network (GNN) pipelines. This tutorial synthesizes the algorithms, systems, and design principles driving these converging directions, offering data science and data mining researchers a unified perspective on integrating LLMs, graph data management, graph mining, graph ML, and agentic computation into next-generation graph-native AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10198v2">Density Ridge Selective Prediction for LLM and VLM Hallucination Detection under Calibration Label Scarcity</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Hallucination detection in large language and vision-language models is increasingly framed as selective prediction, where a detector assigns a confidence score and abstains when confidence is low. Unsupervised sampling detectors (Semantic Entropy) avoid labels but plateau in quality, while supervised probes attain stronger in-distribution scores yet degrade sharply when calibration labels are scarce. We recover the response manifold of an LLM as the density ridge of a kernel density estimate built on a six-dimensional kinematic feature map of hidden state generation trajectories. A test generation is scored by the negated Euclidean distance from its projected feature point to the nearest ridge vertex, yielding a low-dimensional geometric skeleton of the stochastic output distribution. We evaluate against Semantic Entropy, topological methods, and log-probability on six QA benchmarks (HaluEval-QA, TriviaQA, GSM8K, POPE, ScienceQA, A-OKVQA) using eight text and vision LLMs in a deliberately label-scarce protocol ($n_{\text{cal}}{=}200$ queries, $N{=}5$ generations). Our ridge-based score beats on AUROC with 5-20 points gain, while demonstrating tempered degradation under calibration-label scarcity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16456v3">GPO: Learning from Critical Steps to Improve LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in various domains, showing impressive potential on different tasks. Recently, reasoning LLMs have been proposed to improve the \textit{reasoning} or \textit{thinking} capabilities of LLMs to solve complex problems. Despite the promising results of reasoning LLMs, enhancing the multi-step reasoning capabilities of LLMs still remains a significant challenge. While existing optimization methods have advanced the LLM reasoning capabilities, they often treat reasoning trajectories as a whole, without considering the underlying critical steps within the trajectory. In this paper, we introduce \textbf{G}uided \textbf{P}ivotal \textbf{O}ptimization (GPO), a novel fine-tuning strategy that dives into the reasoning process to enable more effective improvements. GPO first identifies the `critical step' within a reasoning trajectory - a point that the model must carefully proceed to succeed at the problem. We locate the critical step by estimating the advantage function. GPO then resets the policy to the critical step, samples the new rollout and prioritizes the learning process on those rollouts. This focus allows the model to learn more effectively from pivotal moments within the reasoning process to improve the reasoning performance. We demonstrate that GPO is a general strategy that can be integrated with various optimization methods to improve reasoning performance. Besides theoretical analysis, our experiments across challenging reasoning benchmarks show that GPO can consistently and significantly enhance the performance of existing optimization methods, showcasing its effectiveness and generalizability in improving LLM reasoning by concentrating on pivotal moments within the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12754v1">LLMs Can Better Capture Human Judgments--With the Right Prompts</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Are large language models (LLMs) bad at capturing human judgment? Two commonly stated limitations are that LLMs fail to capture full distributions of responses, and that their judgments are unstable across wording variations. We demonstrate simple prompting strategies that mitigate these limitations. Across two datasets--a U.S.-representative set of 144 moral scenarios and 38 moral beliefs from the International Social Survey Programme's Family and Changing Gender Roles module covering 32 countries--we show how simple elicitation techniques help improve AI-human alignment. First, prompting models to report standard deviations and response proportions recovers the full range of human responses better than common strategies. Second, ensuring scenarios are clear to human participants--as reflected in human confusion ratings--boosts model alignment, and LLMs can track human confusion ratings. At the same time, we find that LLMs' estimates of their own error are poorly calibrated, though they can predict human variability relatively well. These results suggest that asking better questions to LLMs can yield better answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.04021v3">Prism: Cost-Efficient Multi-LLM Serving via GPU Memory Ballooning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 OSDI'26
    </div>
    <details class="paper-abstract">
      Inference providers must maintain availability for many LLMs, including low-volume but essential models, making resource efficiency increasingly important as token prices fall. Analysis of production traces reveals a dynamic bursty-group pattern in which sets of models become active together and shift over time; existing space- and time-sharing approaches lack principled mechanisms to adapt to this variability, forcing trade-offs between SLO adherence and efficiency. We observe that elastic memory allocation can unify spatial and temporal sharing. Based on this insight, we have developed Prism, a memory-centric LLM co-serving framework that applies memory ballooning to reclaim memory across models and support both forms of sharing under a single scheme. Prism's balloon driver, referred to as kvcached, has been open-sourced at https://github.com/ovg-project/kvcached, and deployed in production environments across 10K+ GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12731v1">Normative Robustness as a Frontier for Non-Verifiable Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      As LLMs increasingly serve in advisory and deliberative roles, users rely on them for non-verifiable reasoning in domains lacking objective ground truths. However, traditional evaluations of LLM reasoning focus almost exclusively on fact-based domains, such as mathematics and science, leaving uncertainty over whether and to what degree models can handle ambiguous, subjective, or value-laden problems over time. To address this concern, we propose moral reasoning as a paradigmatic subdomain of non-verifiable reasoning. We define moral robustness as a model's capacity to exhibit sound moral reasoning across time and contexts, and we introduce a scalable, adversarial, multi-turn evaluation framework to empirically measure this capability. We simulate 48,000 user-agent moral deliberations across four frontier LLMs, varying premise relevance, premise order, conversation duration, and the user's stated moral view. We find that models successfully ignore morally-irrelevant distractors, but shift their reasoning by up to 6.5%, on average, towards the user's stated preferred moral view, and varying their reasoning depending on factors such as order (altering moral judgments by order in 13-22% of the cases) and duration (altering moral judgments between single-turn and multi-turn in 10-24% of the cases). Our analysis indicates that models tailor not just their final verdicts but their underlying justifications to align with a user's moral viewpoint - a failure mode we characterize as moral deliberative sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12730v1">Rethinking Psychometric Evaluation of LLMs: When and Why Self-Reports Predict Behavior</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Accepted as an Oral (Contributed Talk) at the ICML 2026 Workshop on Combining Theory and Benchmarks (CTB)
    </div>
    <details class="paper-abstract">
      Anticipating LLM behavioral tendencies from low-cost psychometric probes is critical for safe deployment, but only if self-reports (SR) reliably predict behavior. Recent work documented substantial SR-behavior dissociation in LLMs, but relied on broad personality traits (Big 5) that predict specific behaviors weakly, even in humans. Furthermore, the isolation of conversational sessions combined with weak context matching left open whether LLMs truly lack coherence or whether the conditions needed to detect such coherence were not met. We contrast Big 5 with the Theory of Planned Behavior (TPB), which measures intention targeted to a specific behavior and predicts human behavior substantially better than broad traits. We run experiments across four behavioral tasks and 11 frontier LLMs, while also varying session context and identity induction. We find that SR-behavior coherence exists but is selective. 1) Within a shared conversation, the Theory of Planned Behavior reaches human-level coherence; Big 5 does not. 2) Across separate conversations, coherence survives only for behaviors anchored outside the immediate prompt, such as implicit bias shaped by training, and collapses when behavior is strongly primed by context, as with sycophancy. 3) Persona prompting makes self-reports more consistent across conversations, but does not bring behavior into alignment. These findings suggest that coarse personality frameworks, such as Big 5 may not be the best tools for testing deployment behavior. More task- and behavior-specific instruments are needed, and even these must be evaluated across tasks and contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12702v1">Deployment-Centered Evaluation: Predicting Query-Level Rejection Risk in a Clinical LLM System</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into clinical systems, making it essential to evaluate the real-world utility of these systems. However, static benchmarks tend to measure correctness rather than user acceptance, aggregate performance across queries, and require densely annotated datasets -- leading to major blind spots for evaluating clinical systems. In this work, we perform a deployment-centered evaluation of an LLM system embedded within electronic health records at an academic medical center, where user feedback is sparse but closely reflects the deployment conditions. Specifically, we train a pre-response classifier that estimates the risk that a future interaction will result in the user rejecting the LLM response, based on query content and deployment-specific context available before generation. We conduct a prospective analysis of our model over 4.5 months of user feedback, finding that our prediction model achieves an AUROC of 0.719. Further, we estimate the benefit of such predictions in two downstream use cases (guardrail triggering and abstention). Our key conceptual insight is that making use of deployment-specific context (i.e., the provider type, department name, language model used for response), as opposed to only query content, improves the ability to predict whether the user will reject the system output. Altogether, our empirical case study demonstrates the feasibility of predicting user rejection using deployment-specific context, opening the door to targeted guardrails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12699v1">LLM-Powered Personalized Glycemic Assessment in Type 2 Diabetes with Wearable Sensor Data</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 The 14th IEEE International Conference on Healthcare Informatics, 2026
    </div>
    <details class="paper-abstract">
      Type 2 Diabetes (T2D) poses an increasing global health threat, demanding effective glycemic assessment to support personalized and improved diabetes care. Wearable sensors such as continuous glucose monitors (CGM) and fitness trackers offer many valuable insights for glycemic assessment. However, effectively analyzing these data requires integration with essential individual-level context. Existing methods are often based on traditional machine learning (ML) and rely primarily on historical blood glucose measurements and overlook personalized information, which limits their performance across diverse diabetes populations. Recent advances in large language models (LLMs) have demonstrated their ability to integrate diverse data modalities while modeling sequential dependencies, motivating the exploration of their potential for personalized glycemic assessment. In this paper, we propose GlyLLM, an LLM-powered framework for modeling CGM-based glycemic dynamics through the integration of wearable sensor data and structured metadata. GlyLLM can leverage the extensive prior knowledge of pre-trained LLMs and achieve sensor-text semantic abstraction at decision time. Experiments on two related tasks on the AI-READI dataset demonstrate that our model outperforms traditional ML methods by an average of 13.66\% in Root Mean Squared Error (RMSE) for glucose forecasting and 13.08\% in Area Under the Receiver Operating Characteristic (AUROC) for diabetes categorization. Additionally, our ablation study shows that diabetes surveys and biometric tests are more critical than other health information for glycemic assessment. Our work presents a promising step toward harnessing the power of LLMs to advance personalized glycemic assessment in T2D care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12657v1">TrajGenAgent: A Hierarchical LLM Agent for Human Mobility Trajectory Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 14 pages, 2 figures, 8 tables. Accepted by the 27th IEEE International Conference on Mobile Data Management (MDM 2026)
    </div>
    <details class="paper-abstract">
      Human mobility data is important for transportation, urban planning, and epidemic control, but large-scale trajectory collection is often costly and privacy-constrained, motivating realistic synthetic trajectory generation. Existing LLM-based generators typically rely on either prompt engineering, which preserves zero-shot reasoning but lacks fine-grained spatiotemporal grounding, or trajectory-level fine-tuning, which improves statistical precision but incurs substantial computational cost and may weaken general reasoning. We propose TrajGenAgent, a semantic-aware hierarchical LLM-agent framework for human mobility trajectory generation without model fine-tuning. TrajGenAgent uses a two-stage orchestrator-worker design: an LLM first synthesizes an individual- and weekday-conditioned activity chain from historical evidence via in-context learning, and a deterministic workflow then grounds each activity into a complete visit using personalized POI retrieval, distance-aware location selection, kinematics-aware travel-time propagation, and LLM-based duration estimation. To evaluate realism beyond aggregate spatiotemporal statistics, we introduce an anomaly-detection-based evaluation framework using two complementary detectors to assess behavioral and semantic plausibility. Experiments on benchmark and large-scale simulation datasets show that TrajGenAgent improves spatiotemporal fidelity, semantic coherence, and individual-specific behavioral realism over representative neural and LLM-based baselines, while avoiding parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00462v4">LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 ICML 2026 (Camera Ready)
    </div>
    <details class="paper-abstract">
      Transforming a large language model (LLM) into a vision-language model (VLM) can be achieved by mapping the visual tokens from a vision encoder into the embedding space of an LLM. Intriguingly, this mapping can be as simple as a shallow MLP transformation. To understand why LLMs can so readily process visual tokens, we need interpretability methods that reveal what is encoded in the visual token representations at every layer of LLM processing. In this work, we introduce LatentLens, a novel approach for mapping latent representations to descriptions in natural language. LatentLens encodes a large text corpus and stores contextualized token representations for each token in that corpus. Visual token representations are then compared to these contextualized representations and the top-nearest neighbor representations serve as descriptions of the visual token. We evaluate this method on 15 different VLMs, showing that commonly used methods, such as LogitLens, substantially underestimate the interpretability of visual tokens. With LatentLens instead, the majority of visual tokens are interpretable across all studied models and all layers. Qualitatively, we show that the descriptions produced by LatentLens are semantically meaningful and provide more fine-grained interpretations for humans compared to individual tokens. More broadly, our findings contribute new evidence on the alignment between vision and language representations and open up new directions for analyzing the latent representations of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07294v4">Fin-RATE: A Real-world Financial Analytics and Tracking Evaluation Benchmark for LLMs on SEC Filings</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      With the increasing deployment of Large Language Models (LLMs) in the finance domain, LLMs are increasingly expected to parse complex regulatory disclosures. However, existing benchmarks often focus on isolated details, failing to reflect the complexity of professional analysis that requires synthesizing information across multiple documents, reporting periods, and corporate entities. Furthermore, these benchmarks do not disentangle whether errors arise from retrieval failures, generation inaccuracies, domain-specific reasoning mistakes, or misinterpretation of the query or context, making it difficult to precisely diagnose performance bottlenecks. To bridge these gaps, we introduce Fin-RATE, a benchmark built on U.S. Securities and Exchange Commission (SEC) filings and mirroring financial analyst workflows through three pathways: detail-oriented reasoning within individual disclosures, cross-entity comparison under shared topics, and longitudinal tracking of the same firm across reporting periods. We benchmark 17 leading LLMs, spanning open-source, closed-source, and finance-specialized models, under both ground-truth context and retrieval-augmented settings. Results show substantial performance degradation, with accuracy dropping by 18.60% and 14.35% as tasks shift from single-document reasoning to longitudinal and cross-entity analysis. This degradation is associated with increased comparison hallucinations, temporal and entity mismatches, and is further reflected in declines in reasoning quality and factual consistency--limitations that existing benchmarks have yet to formally categorize or quantify.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12599v1">Constrained Semantic Decompression in LLMs through Persian Proverb-Conditioned Story Generation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Transforming a dense, abstract proverb into an engaging and morally faithful narrative requires deep cultural understanding and robust semantic grounding. We frame this problem as a \emph{constrained semantic decompression} task and study proverb-conditioned story generation as a testbed for abstraction-to-realization in large language models (LLMs). Focusing on Persian, we introduce the Proverb Aligned Narrative Dataset (PAND), pairing proverbs with human-written stories and explicit meanings. By a hybrid evaluation framework that combines human-calibrated LLM-as-a-Judge with structural metrics, we analyze model behavior across multiple prompting regimes. Our findings reveal a persistent \emph{decompression gap}: current LLMs often achieve strong surface-level fluency while failing to faithfully instantiate the underlying moral and causal structure encoded in proverbs. We further show that explicit reasoning and iterative refinement can partially mitigate these failures, suggesting that many decompression errors arise from difficulties in translating abstract meaning into narrative form rather than a complete lack of relevant knowledge. Our proposed task naturally extends to other forms of compressed cultural knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12407v1">How Seemingly Inconsequential Design Choices Dictate Performance of LLMs in Pathology</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      General-purpose large language models (LLMs) are routinely used as baselines when evaluating specialized pathology models on whole-slide images (WSIs). Because WSIs exceed contemporary model context limits, LLM baselines routinely use small, high-magnification patches processed independently via majority voting, without systematic evaluation of seemingly inconsequential design choices such as patch size, patch count, and magnification. Generalist LLMs have consistently underperformed specialized systems, reinforcing the perception that domain-specific training or architectural adaptation is necessary for pathology tasks involving WSIs. Here, we conduct a systematic factorial analysis of four input design factors: inference mode, patch size, magnification, and patch count. We demonstrate that prior studies have overstated the gap between specialized models and general-purpose LLMs by choosing non-optimized input configurations. On the MultiPathQA benchmark, switching to a single balanced configuration (large patches at lower magnification, processed jointly) raises GPT-5 from 15.1% to 39.5% on cancer-type classification (TCGA) and from 38.1% to 62.9% on organ classification (GTEx). Per-task optimization yields further gains up to 43.9% (TCGA) and 71.6% (GTEx). The same configuration generalizes to two other models and to a fully held-out CPTAC cohort, where it improves Gemini 3 Flash by 23.4 percentage points without any task-specific tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12385v1">Which Models Are Our Models Built On? Auditing Invisible Dependencies in Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Modern LLM training pipelines increasingly rely on other models to generate data, filter corpora, judge outputs, and guide development decisions. These dependencies are recursive: a model may depend on an upstream artifact whose own dependencies are documented only in separate releases and artifacts. As a result, the full dependency structure is fragmented across heterogeneous public artifacts, with complexity and recursive depth far outpacing humans' ability to trace. We introduce ModSleuth, an agentic system that recursively reconstructs LLM dependency graphs from public artifacts with source-grounded evidence. We find that the primary challenge is no longer information extraction, but defining what constitutes a dependency and reconciling artifact references across inconsistent documentation. We address these challenges through a formalization that distinguishes direct and indirect dependencies, represents heterogeneous pipeline roles through operation-centered relationships, and resolves artifact identities across names, versions, and repositories. Applying ModSleuth to four public-artifact-rich LLM releases, we recover 1,060 source-verified dependencies and construct large-scale dependency graphs of modern LLM development. These graphs reveal multi-hop license obligations, train-evaluation coupling, discrepancies between released and training-time artifacts, and documentation inconsistencies that would otherwise be difficult to uncover. We release ModSleuth and the resulting dependency graphs to support transparent analysis of the increasingly complex ecosystems underlying modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12369v1">Should LLM Agents Decide in Social Simulations? Comparing Finite-State and LLM-Based Decision Policies</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as decision-making components in social simulations. This introduces a methodological risk: the simulation may deviate from the explicit behavioral policy defined by the researcher. In online social network (OSN) simulations, action choices shape system dynamics, interaction patterns, and model interpretability. This paper evaluates whether LLM action selectors preserve an interpretable reference policy in an OSN simulation. The reference is a finite state machine implemented as a first-order Markov model, with transition probabilities depending on the user type. The evaluation uses a synthetic network with 1,000 agents and 10,000 action decisions. Three open-weight LLMs are tested: LLaMA 3.1, GPT-OSS, and Mistral 24B. Each model is evaluated under three prompting strategies: base, guided, and probabilistic. Alignment is measured using Jensen-Shannon Divergence with Laplace smoothing, and execution time is reported. Results show that LLMs can approximate the reference policy in some configurations, but do not preserve it reliably. Alignment varies across models and prompts, and additional guidance can introduce systematic action biases. Even the best-aligned LLM configurations are several hundred times slower than direct Markov chain sampling. These findings indicate that LLM-based action selection is not a direct replacement for explicit decision policies: it can alter the intended behavior while increasing computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.10968v2">Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 Project Page: https://hunyuan-cppo.github.io/
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become standard for improving LLM reasoning. However, existing PPO-style trust-region mechanisms remain position-agnostic by enforcing uniform thresholds across all tokens independently. This pointwise treatment conflicts with autoregressive generation in two critical ways. First, uniform thresholds ignore autoregressive asymmetry. Early-stage deviations produce compounding sequence-level drift, causing static thresholds to under-regulate early divergence and excessively constrain late-stage exploration. Second, evaluating token-level divergence in isolation overlooks cumulative prefix drift, granting the same divergence allowance regardless of how far the conditioning history has already deviated from the rollout policy. To address this limitation, we propose CPPO (Cumulative Prefix-divergence Policy Optimization), a token-level masking rule that aligns updates with a finite-horizon policy-improvement bound via two coupled mechanisms. First, a position-weighted threshold imposes stricter limits at early positions whose effects persist longer, relaxing constraints for late-stage tokens. Second, a cumulative prefix budget tracks historical deviations, dynamically restricting further token-level deviation to prevent compounding errors along the prefix. Empirically, CPPO enhances training stability and significantly improves reasoning accuracy across various model scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03520v2">Certifiable Safe RLHF: Semantic Grounding and Fixed Penalty Constraint Optimization for Safer LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Ensuring safety is a foundational requirement for large language models (LLMs). Achieving an appropriate balance between enhancing the utility of model outputs and mitigating their potential for harm is a complex and persistent challenge. Contemporary approaches frequently formalize this problem within the framework of Constrained Markov Decision Processes (CMDPs) and employ established CMDP optimization techniques. However, these methods exhibit two notable limitations. First, their reliance on reward and cost functions renders performance highly sensitive to the underlying scoring mechanism, which must capture semantic meaning rather than being triggered by superficial keywords. Second, CMDP-based training entails tuning dual-variable, a process that is both computationally expensive and does not provide any provable safety guarantee for a fixed dual variable that can be exploitable through adversarial jailbreaks. To overcome these limitations, we introduce Certifiable Safe-RLHF (CS-RLHF) that introduces a cost model trained on a large-scale corpus to assign semantically grounded safety scores. In contrast to the lagrangian-based approach, CS-RLHF adopts a rectified penalty-based formulation. This design draws on the theory of exact penalty functions in constrained optimization, wherein constraint satisfaction is enforced directly through a suitably chosen penalty term. With an appropriately scaled penalty, feasibility of the safety constraints can be guaranteed at the optimizer, eliminating the need for dual-variable updates. Empirical evaluation demonstrates that CS-RLHF outperforms state-of-the-art LLM model responses rendering at-least 5 times efficient against nominal and jail-breaking prompts
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12341v1">OCELOT: Inference-Leakage Budgets for Privacy-Preserving LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents increasingly act on a user's behalf -- reading personal files, calling tools, transacting with external services -- possibly leaking personally identifiable information (PII) across trust boundaries at every step. Privacy here is a property not of a single output but of an entire trajectory, and three properties make it hard: leakage is cumulative, as individually innocuous releases accumulate across honest-but-curious or colluding sinks into inferences about a protected secret; bidirectional, as a malicious observation can inject instructions that turn the agent's own reasoning model against the user; and task-dependent, as the same field is necessary for one recipient yet gratuitous for another. Per-release contextual-integrity filters, information-flow controls, and posterior-leakage monitors each address part of this but none controls cumulative, inference-based leakage at runtime. We recast agent privacy as \emph{posterior-risk control} and present OCELOT, a runtime mediator that budgets how much an adversary's belief about a secret may improve across a trajectory, rather than filtering outputs. Its mechanism, \emph{Witness-Verified Declassification}, separates judgment from trust: an untrusted, locally fine-tuned defender model inspects each candidate release and emits structured evidence -- labeled atoms and proposed declassification operators -- which a deterministic verifier audits, charging a certified min-entropy cost for the chosen variant and authorizing the least-disclosing useful release under a sink-trust-weighted budget recorded on a tamper-evident ledger. Across diverse agent benchmarks and recent defenses, OCELOT attains significantly lower leakage at higher task utility, resists adaptive injection, jailbreak, cumulative inference, and sink collusion, and adds only modest overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12291v1">Measuring Epistemic Resilience of LLMs Under Misleading Medical Context</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now reach expert-level scores on medical licensing exams, encouraging the assumption that high scores imply safe medical judgment while patients increasingly use them for health advice. We show this assumption is fragile: when misleading context is injected into questions that LLMs originally answer correctly, they abandon the correct answer. We call the ability to maintain correct judgment under adversarial context epistemic resilience, and introduce MedMisBench to measure it. MedMisBench contains 10,932 medical question items and 48,889 misleading context-option pairs spanning medical reasoning, agentic capability, and patient-journey evaluation. Across 11 model configurations, mean accuracy falls from 71.1% on original questions to 38.0% under focused misleading context, with 51.5% attack success. The most damaging injections are formal, rule-like fabrications: authority-framed falsehoods reach 69.5% attack success and exception-poisoning claims reach 64.1%. A 14-member clinical panel from 7 countries identified serious potential harm in 38.2% of reviewed cases. MedMisBench exposes a structural blind spot in LLM evaluation in medical settings: existing benchmarks measure what models know, but not whether they preserve correct medical judgment under misleading context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08473v4">AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) improves performance but introduces critical safety vulnerabilities: even minimal harmful data can severely compromise safety measures. We observe that perturbations orthogonal to the alignment direction - defined by weight differences between aligned (safe) and unaligned models - rapidly compromise model safety. In contrast, updates along the alignment direction largely preserve it, revealing the parameter space as a "narrow safety basin". To address this, we propose AsFT (Anchoring Safety in Fine-Tuning) to maintain safety by explicitly constraining update directions during fine-tuning. By penalizing updates orthogonal to the alignment direction, AsFT effectively constrains the model within the "narrow safety basin," thus preserving its inherent safety. Extensive experiments on multiple datasets and models show that AsFT reduces harmful behaviors by up to 7.60%, improves task performance by 3.44%, and consistently outperforms existing methods across multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12250v1">Reassessing High-Performing LLMs on Polish Medical Exams: True Competence or Bias-Driven Performance?</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 26 pages total with references and appendix, preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) in medicine are mainly evaluated using multiple-choice question answering (MCQA), which can overestimate real clinical ability due to guessing strategies and answer biases. To address these limitations, we introduce an expanded and more challenging benchmark based on Polish medical exams, adding over 15,000 questions, two new domains, and four structural modifications that reduce MCQA-specific artifacts and better test reasoning. We evaluate 21 LLMs and show that evaluation design strongly affects results. Under our harder setup, the best model (Qwen3.5-122B) drops by 28.4 and 31 pp on English and Polish exams, respectively. Despite low evidence of data contamination, standard MCQA scores do not reliably reflect true medical competence. To facilitate further research, we make our benchmark publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12247v1">Beyond Third-Person Audits: Situated Interaction Auditing for User-Centered LLM Bias Research</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Research on bias in large language models (LLMs) has predominantly focused on third-person audits, which study how models represent or evaluate demographic groups as external subjects. However, this paradigm overlooks a structural blind spot because the user is absent from the audit. In practice, LLMs are used in open-ended, personal interactions, during which the model implicitly represents the user and adjusts its responses accordingly. When identical requests yield different responses depending on who is asking, bias manifests not in how the model describes others but in how it treats its interlocutor. We propose Situated Interaction Auditing (SIA), a user-centered framework for studying how user profile signals -- implicit sociodemographic markers, writing style, and stated identity -- systematically shape LLM response quality, content, and tone. We demonstrate the framework through a case study that intersects gender and socioeconomic status signals across multiple task domains and outline a research agenda for SIA as a new mission for natural language processing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12234v1">On The Effectiveness-Fluency Trade-Off In LLM Conditioning: A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 8 pages, 2 figure
    </div>
    <details class="paper-abstract">
      Controlling the output of Large Language Models (LLMs) is a central challenge for their reliable deployment, yet a clear understanding of the involved trade-offs remains elusive. Current approaches to conditioning are often evaluated with a narrow focus on their effectiveness at injecting or removing a target concept, neglecting generation quality. We systematically investigate a range of conditioning methods in both injection and removal scenarios. We find that efficient steering methods frequently achieve conditioning at a steep cost to fluency. Furthermore, we identify a critical yet previously overlooked interaction with the training paradigm: activation steering methods are far less effective on instruction-tuned models than on their base counterparts. Simple prompting and full-fledged supervised fine-tuning, on the other hand, are viable options for concept injection, but are not as good at concept removal. Finally, cheaply computed textual metrics highly correlate to costly LLM-as-judge scores, and provide insights on the behavior of conditioning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.19225v5">FinTradeBench: A Financial Reasoning Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 9 pages main text, 31 pages total (including references and appendix). 5 figures, 16 tables. Preprint under review. Code and data will be made available upon publication
    </div>
    <details class="paper-abstract">
      Real-world financial decision-making is a challenging problem that requires reasoning over heterogeneous signals, including company fundamentals derived from regulatory filings and trading signals computed from price dynamics. Recently, with advances in Large Language Models (LLMs), financial analysts have begun to use them for financial decision-making tasks. However, existing financial question-answering benchmarks for testing these models primarily focus on company balance sheet data and rarely evaluate reasoning about how company stocks trade in the market or their interactions with fundamentals. To leverage the strengths of both approaches, we introduce FinTradeBench, a benchmark for evaluating financial reasoning that integrates company fundamentals and trading signals. FinTradeBench contains 1,400 questions grounded in NASDAQ-100 companies over a ten-year historical window. The benchmark is organized into three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions requiring cross-signal reasoning. To ensure reliability at scale, we adopt a calibration-then-scaling framework that combines expert seed questions, multi-model response generation, intra-model self-filtering, numerical auditing, and human-LLM judge alignment. We evaluate 14 LLMs under zero-shot prompting and retrieval-augmented settings and witness a clear performance gap. Retrieval substantially improves reasoning over textual fundamentals, but provides limited benefit for trading-signal reasoning. These findings highlight fundamental challenges in the numerical and time-series reasoning for current LLMs and motivate future research in financial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12212v1">Mind your key: An Empirical Study of LLM API Credential Leakage in iOS Apps</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 12 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The rapid integration of large language models (LLMs) into mobile applications has introduced a new class of credential security risk: leaked credentials that grant unauthorized access to LLM inference services, causing financial damage to developers. Prior work on credential leakage has focused primarily on Android apps; to date, no empirical study has systematically investigated LLM API key leakage in iOS applications. We present the first in-depth empirical study of API key leakage in LLM-integrated apps. We construct a high-quality dataset of 444 iOS applications, filtered from 1092 candidates through a standardized process, and develop LLMKeyLens, a dynamic analysis framework that detects LLM API key leakage via traffic interception, provider-specific key extraction, and active validity confirmation, requiring neither source code access nor binary decryption. Our analysis reveals that 282 applications expose exploitable LLM API credentials in network traffic, spanning at least ten providers. We identify three leakage patterns: JWT-based token leakage (48%), unauthenticated backend proxy access (33%), and plaintext API key transmission (19%). To assess remediation, we re-analyzed the same 282 vulnerable applications three months after responsible disclosure; only 28% had remediated the reported vulnerability, while 72% remained exploitable, with persistent issues stemming from unauthenticated backends and broken JWT implementations. Our findings show that LLM API key leakage is both prevalent and persistent in the iOS ecosystem, exposing a systemic gap between developer practice and secure integration principles, and suggest that secure LLM integration requires not only developer awareness but also explicit security guidance from providers and platform-level enforcement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12198v1">LLM-Based User Personas for Recommendations at Scale</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer unprecedented potential for enhancing recommendation systems through their world knowledge and reasoning capabilities. However, existing approaches often rely on structured IDs or offline processing, limiting semantic richness, real-time adaptability, and user-facing interpretability. In this paper, we introduce a novel framework that enables real-time generation of LLM-based user interest personas for a large-scale commercial video recommendation platform. Our method generates natural-language user interest personas that address the exploitation-exploration trade-off by combining the summarization of existing interests with novel topics, directly during serving. To overcome the computational challenges of online LLM inference at a billion-user scale, we design a cost-efficient architecture leveraging knowledge distillation, asynchronous inference, and input optimization via semantically clustered video representations. Extensive offline evaluations, user studies, and live A/B tests demonstrate significant improvements in viewer value. This work bridges the gap between high-level semantic understanding and industrial-scale recommendation, paving the way for more dynamic, explainable, and satisfying personalized experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21594v3">Visualizing LLM Latent Space Geometry Through Dimensionality Reduction</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 25 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve state-of-the-art results across many natural language tasks, but their internal mechanisms remain difficult to interpret. In this work, we extract, process, and visualize latent state geometries in Transformer-based language models through dimensionality reduction. We capture layerwise activations at multiple points within Transformer blocks and enable systematic analysis through Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP). We demonstrate experiments on GPT-2 and LLaMa models, where we uncover interesting geometric patterns in latent space. Notably, we identify a clear separation between attention and MLP component outputs across intermediate layers, a pattern not documented in prior work to our knowledge. We also characterize the high norm of latent states at the initial sequence position and visualize the layerwise evolution of latent states. Additionally, we demonstrate the high-dimensional helical structure of GPT-2's positional embeddings and the sequence-wise geometric patterns in LLaMa. We make our code available at https://github.com/Vainateya/Feature_Geometry_Visualization. A better formatted blog-post with identical content is available at https://iclr-blogposts.github.io/2026/blog/2026/vis-llm-latent-geometry/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12142v1">AerialClaw: An Open-Source Framework for LLM-Driven Autonomous Aerial Agents</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Unmanned aerial vehicles (UAVs) are increasingly used in inspection, search and rescue, environmental monitoring, and emergency response. However, most UAV applications still rely on pre-defined command sequences or task-specific pipelines, where developers manually connect perception, planning, flight control, simulation, logging, and safety modules. This limits the flexibility, reproducibility, and extensibility of autonomous aerial systems. This paper presents AerialClaw, an open-source software framework that enables UAVs to operate as decision-making aerial agents rather than merely command-following platforms. Given a natural-language mission, AerialClaw allows an LLM-based agent to understand the task, maintain context, invoke executable aerial skills, observe perception and runtime feedback, and iteratively update its decisions in a closed loop. The framework adopts a modular brain-skill-runtime architecture, combining hard skills for atomic UAV operations, Markdown-based soft skills for reusable task strategies, document-driven agent state and capability boundaries, memory-driven reflection, safety-oriented runtime validation, and platform-agnostic execution adapters. AerialClaw supports lightweight mock execution, PX4 SITL with Gazebo, and AirSim-based simulation, together with a web console, pluggable model backends, example missions, simulation assets, and staged deployment scripts. By combining standardized aerial skills, document-driven agent state, memory, and closed-loop LLM decision-making, AerialClaw provides a reproducible and extensible open-source framework for building UAV systems that can interpret missions, make decisions, execute skills, and adapt their behavior from feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.13628v3">Compact LLM Deployment and World Model Assisted Offloading in Mobile Edge Computing</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper investigates compact large language model (LLM) deployment and world-model-assisted inference offloading in mobile edge computing (MEC) networks. We first propose an edge compact LLM deployment (ECLD) framework that jointly applies structured pruning, low-bit quantization, and knowledge distillation to construct edge-deployable LLM variants, and we evaluate these models using four complementary metrics: accessibility, energy consumption, hallucination rate, and generalization accuracy. Building on the resulting compact models, we formulate an MEC offloading optimization problem that minimizes the long-term average inference latency subject to per-device energy budgets and LLM-specific quality-of-service constraints on effective accuracy and hallucination. To solve this problem under unknown and time-varying network dynamics, we develop a world model-proximal policy optimization (PPO) algorithm, which augments an on-policy PPO algorithm with a learned recurrent world model that provides improved value targets and short imagination rollouts. Extensive experiments on Llama-3.1-8B, Qwen3-8B, and Mistral-12B show that ECLD compresses base models by about 70-80% in storage (i.e., from 15.3 GB to 3.3 GB for Llama-3.1-8B) and reduces per-query energy consumption by up to 50%, while largely preserving accuracy and often lowering hallucination compared with quantization-only or pruning-only baselines. Moreover, they also show that world model-PPO speeds up convergence by about 50%, improves the final reward by 15.8% over vanilla PPO, and reduces average inference latency by 12-30% across different user populations, while satisfying the accuracy and hallucination constraints and approaching the generation quality of always-offloading with much of the efficiency of local execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12117v1">Soft-Prompt Tuning for Fair and Efficient LLM Benchmark Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-06-10
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Benchmark scores often misrepresent a large language model's (LLM's) knowledge, because they rely, e.g., on the model's ability to follow specific formatting requirements. This especially penalizes base models that may know the correct answers but lack the ability -- typically introduced in post-training -- to structure them as instructed. To overcome this, we propose soft-prompt tuning, an efficient, fair, and architecture-agnostic model evaluation. By optimizing only 10 soft-prompt vectors (roughly 0.0006% parameters for a 7B model) over a short tuning period, we adapt models to specific benchmark formats, closing gaps in format-following and ensuring that underlying knowledge is accurately reflected in benchmark scores. This allows one to fairly compare different base models -- trained with various pre-training recipes -- on benchmarks without the need for full post-training. We evaluated soft-prompt tuning across 7 models and 7 datasets. The results show that (a) soft-prompt tuning saturates format-following within 80 steps (~640 samples) making it highly efficient, (b) soft-prompt tuning significantly outperforms zero- and few-shot prompting, surfacing base model knowledge that standard prompting misses, that (c) even post-trained models can benefit from soft-prompts to maximize format compliance, and that (d) soft-prompted base model performance predicts post-trained model rankings more reliably than zero- and few-shot baselines, offering a low-cost proxy for downstream model quality. Our contributions include (1) metrics which disentangle format-following and knowledge accuracy, (2) a fairer benchmarking protocol of LLM knowledge, and (3) a cost- and memory-effective recipe to identify optimal pre-training strategies early in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2606.12073v1">"That's AI Slop, You Bot!" Studying Accusations, Evidence, and Credibility in Online Discourse Towards LLM-Generated Comments</a></div>
    <div class="paper-meta">
      📅 2026-06-10
    </div>
    <details class="paper-abstract">
      Generative AI has made fluent prose cheap to produce, breaking the old promise to readers that good writing meant real thinking. How have readers responded, and what can this tell us about changing anti-AI attitudes? We analyzed 25 million comments from Hacker News and Reddit (2023-2026), combining LLM judgment on 7,500 sampled accusations of AI use, sentiment trajectories, speech-act coding of 300 confirmed accusations of AI use, and a matched-control test of accused versus non-accused parent comments. We found that the pejorative-label share of accusations rose more than tenfold on both platforms while a placebo vocabulary of pre-2022 inauthenticity terms (shill, astroturf) did not. This shift reflected a fast-growing trend of branding any suspicious or seemingly inauthentic prose as "AI slop". The slop frame now constitutes 94 percent of pejorative mentions, with the dominant comments shifting in tone from mockery toward gatekeeping and structural protest. The key surprise comes from a matched-control test which found that prose features that statistically distinguish AI from human text do not predict which human text gets accused as AI. The new accusations work as social gatekeeping of perceived authenticity without actually screening for AI. This research extends signaling theory by showing that substitute signals used socially can grow even when inaccurate if the underlying detection problem cannot be solved at the non-expert level. It shows that AI's effects on writing from the reader side are distinct from those on the production (writer) side. Detection technology cannot resolve this dynamic because the social function of accusations is increasingly to perform social gatekeeping and in-group signaling as opposed to identifying AI-generated writing.
    </details>
</div>
