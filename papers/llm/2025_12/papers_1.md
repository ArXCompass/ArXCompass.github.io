# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23684v1">Multilingual Hidden Prompt Injection Attacks on LLM-Based Academic Reviewing</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly considered for use in high-impact workflows, including academic peer review. However, LLMs are vulnerable to document-level hidden prompt injection attacks. In this work, we construct a dataset of approximately 500 real academic papers accepted to ICML and evaluate the effect of embedding hidden adversarial prompts within these documents. Each paper is injected with semantically equivalent instructions in four different languages and reviewed using an LLM. We find that prompt injection induces substantial changes in review scores and accept/reject decisions for English, Japanese, and Chinese injections, while Arabic injections produce little to no effect. These results highlight the susceptibility of LLM-based reviewing systems to document-level prompt injection and reveal notable differences in vulnerability across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23609v1">The Big Three in Marriage Talk: LLM-Assisted Analysis of Moral Ethics and Sentiment on Weibo and Xiaohongshu</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      China's marriage registrations have declined dramatically, dropping from 13.47 million couples in 2013 to 6.1 million in 2024. Understanding public attitudes toward marriage requires examining not only emotional sentiment but also the moral reasoning underlying these evaluations. This study analyzed 219,358 marriage-related posts from two major Chinese social media platforms (Sina Weibo and Xiaohongshu) using large language model (LLM)-assisted content analysis. Drawing on Shweder's Big Three moral ethics framework, posts were coded for sentiment (positive, negative, neutral) and moral dimensions (Autonomy, Community, Divinity). Results revealed platform differences: Weibo discourse skewed positive, while Xiaohongshu was predominantly neutral. Most posts across both platforms lacked explicit moral framing. However, when moral ethics were invoked, significant associations with sentiment emerged. Posts invoking Autonomy ethics and Community ethics were predominantly negative, whereas Divinity-framed posts tended toward neutral or positive sentiment. These findings suggest that concerns about both personal autonomy constraints and communal obligations drive negative marriage attitudes in contemporary China. The study demonstrates LLMs' utility for scaling qualitative analysis and offers insights for developing culturally informed policies addressing marriage decline in Chinese contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02976v3">How Safe Are AI-Generated Patches? A Large-scale Study on Security Risks in LLM and Agentic Automated Program Repair on SWE-bench</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their agentic frameworks are increasingly adopted to perform development tasks such as automated program repair (APR). While prior work has identified security risks in LLM-generated code, most have focused on synthetic, simplified, or isolated tasks that lack the complexity of real-world program repair. In this study, we present the first large-scale security analysis of LLM-generated patches using 20,000+ GitHub issues. We evaluate patches proposed by developers, a standalone LLM (Llama 3.3 Instruct-70B), and three top-performing agentic frameworks (OpenHands, AutoCodeRover, HoneyComb). Finally, we analyze a wide range of code, issue, and project-level factors to understand the conditions under which generating insecure patches is more likely. Our findings reveal that Llama introduces many new vulnerabilities, exhibiting unique patterns not found in developers' code. Agentic workflows also generate a number of vulnerabilities, particularly when given more autonomy. We find that vulnerabilities in LLM-generated patches are associated with distinctive code characteristics and are commonly observed in issues missing specific types of information. These results suggest that contextual factors play a critical role in the security of the generated patches and point toward the need for proactive risk assessment methods that account for both issue and code-level information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23587v1">Can AI Recognize Its Own Reflection? Self-Detection Performance of LLMs in Computing Education</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 10 pages, 5 tables. Accepted for publication at the 59th Hawaii International Conference on System Sciences
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) presents a significant challenge to academic integrity within computing education. As educators seek reliable detection methods, this paper evaluates the capacity of three prominent LLMs (GPT-4, Claude, and Gemini) to identify AI-generated text in computing-specific contexts. We test their performance under both standard and 'deceptive' prompt conditions, where the models were instructed to evade detection. Our findings reveal a significant instability: while default AI-generated text was easily identified, all models struggled to correctly classify human-written work (with error rates up to 32%). Furthermore, the models were highly susceptible to deceptive prompts, with Gemini's output completely fooling GPT-4. Given that simple prompt alterations significantly degrade detection efficacy, our results demonstrate that these LLMs are currently too unreliable for making high-stakes academic misconduct judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23547v1">Lie to Me: Knowledge Graphs for Robust Hallucination Self-Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 Accepted to ICPRAM 2026 in Marbella, Spain
    </div>
    <details class="paper-abstract">
      Hallucinations, the generation of apparently convincing yet false statements, remain a major barrier to the safe deployment of LLMs. Building on the strong performance of self-detection methods, we examine the use of structured knowledge representations, namely knowledge graphs, to improve hallucination self-detection. Specifically, we propose a simple yet powerful approach that enriches hallucination self-detection by (i) converting LLM responses into knowledge graphs of entities and relations, and (ii) using these graphs to estimate the likelihood that a response contains hallucinations. We evaluate the proposed approach using two widely used LLMs, GPT-4o and Gemini-2.5-Flash, across two hallucination detection datasets. To support more reliable future benchmarking, one of these datasets has been manually curated and enhanced and is released as a secondary outcome of this work. Compared to standard self-detection methods and SelfCheckGPT, a state-of-the-art approach, our method achieves up to 16% relative improvement in accuracy and 20% in F1-score. Our results show that LLMs can better analyse atomic facts when they are structured as knowledge graphs, even when initial outputs contain inaccuracies. This low-cost, model-agnostic approach paves the way toward safer and more trustworthy language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23518v1">Single LLM Debate, MoLaCE: Mixture of Latent Concept Experts Against Confirmation Bias</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are highly vulnerable to input confirmation bias. When a prompt implies a preferred answer, models often reinforce that bias rather than explore alternatives. This phenomenon remains underexplored, yet it is already harmful in base models and poses an even greater risk in multi-agent debate, where echo chambers reinforce bias instead of correction. We introduce Mixture of Latent Concept Experts (MoLaCE), a lightweight inference-time framework that addresses confirmation bias by mixing experts instantiated as different activation strengths over latent concepts that shape model responses. Our key insight is that, due to the compositional nature of language, differently phrased prompts reweight latent concepts in prompt-specific ways that affect factual correctness, so no single fixed intervention can be applied universally across inputs. This design enables a single LLM to emulate the benefits of debate internally while remaining computationally efficient and scalable. It can also be integrated into multi-agent debate frameworks to diversify perspectives and reduce correlated errors. We empirically show that it consistently reduces confirmation bias, improves robustness, and matches or surpasses multi-agent debate while requiring only a fraction of the computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23515v1">Alpha-R1: Alpha Screening with LLM Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Signal decay and regime shifts pose recurring challenges for data-driven investment strategies in non-stationary markets. Conventional time-series and machine learning approaches, which rely primarily on historical correlations, often struggle to generalize when the economic environment changes. While large language models (LLMs) offer strong capabilities for processing unstructured information, their potential to support quantitative factor screening through explicit economic reasoning remains underexplored. Existing factor-based methods typically reduce alphas to numerical time series, overlooking the semantic rationale that determines when a factor is economically relevant. We propose Alpha-R1, an 8B-parameter reasoning model trained via reinforcement learning for context-aware alpha screening. Alpha-R1 reasons over factor logic and real-time news to evaluate alpha relevance under changing market conditions, selectively activating or deactivating factors based on contextual consistency. Empirical results across multiple asset pools show that Alpha-R1 consistently outperforms benchmark strategies and exhibits improved robustness to alpha decay. The full implementation and resources are available at https://github.com/FinStep-AI/Alpha-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23511v1">Beyond Correctness: Exposing LLM-generated Logical Flaws in Reasoning via Multi-step Automated Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, leading to their adoption in high-stakes domains such as healthcare, law, and scientific research. However, their reasoning often contains subtle logical errors masked by fluent language, posing significant risks for critical applications. While existing approaches like fact-checking, self-consistency methods, and rule-based validation provide partial solutions, they fail to detect complex logical flaws in multi-step reasoning. To overcome these challenges, we present MATP, an evaluation framework for systematically verifying LLM reasoning via Multi-step Automatic Theorem Proving. MATP translates natural language reasoning into First-Order Logic (FOL) and applies automated theorem provers to assess step-by-step logical validity. This approach identifies hidden logical errors and provides fine-grained classifications of reasoning correctness. Evaluations on a benchmark comprising 10,830 reasoning instances generated by 10 LLMs across tasks from PrOntoQA-OOD, ProofWriter, and FOLIO show that MATP surpasses prompting-based baselines by over 42 percentage points in reasoning step verification. It further reveals model-level disparities, with reasoning models generating more logically coherent outputs than general models. These results demonstrate MATP's potential to enhance the trustworthiness of LLM-generated reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23489v1">The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: https://anonymous.4open.science/r/MIRAGE-VC-323F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23430v1">C2PO: Diagnosing and Disentangling Bias Shortcuts in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Bias in Large Language Models (LLMs) poses significant risks to trustworthiness, manifesting primarily as stereotypical biases (e.g., gender or racial stereotypes) and structural biases (e.g., lexical overlap or position preferences). However, prior paradigms typically address these in isolation, often mitigating one at the expense of exacerbating the other. To address this, we conduct a systematic exploration of these reasoning failures and identify a primary inducement: the latent spurious feature correlations within the input that drive these erroneous reasoning shortcuts. Driven by these findings, we introduce Causal-Contrastive Preference Optimization (C2PO), a unified alignment framework designed to tackle these specific failures by simultaneously discovering and suppressing these correlations directly within the optimization process. Specifically, C2PO leverages causal counterfactual signals to isolate bias-inducing features from valid reasoning paths, and employs a fairness-sensitive preference update mechanism to dynamically evaluate logit-level contributions and suppress shortcut features. Extensive experiments across multiple benchmarks covering stereotypical bias (BBQ, Unqover), structural bias (MNLI, HANS, Chatbot, MT-Bench), out-of-domain fairness (StereoSet, WinoBias), and general utility (MMLU, GSM8K) demonstrate that C2PO effectively mitigates stereotypical and structural biases while preserving robust general reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21580v2">Gamayun's Path to Multilingual Mastery: Cost-Efficient Training of a 1.5B-Parameter LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      We present Gamayun, a 1.5B-parameter multilingual language model trained entirely from scratch on 2.5T tokens. Designed for efficiency and deployment in resource-constrained environments, Gamayun addresses the lack of research on small non-English-centric LLMs by adopting a novel two-stage pre-training strategy: balanced multilingual training for cross-lingual alignment, followed by high-quality English enrichment to transfer performance gains across languages. Our model supports 12 languages, with special focus on Russian. Despite a significantly smaller training budget than comparable models, Gamayun outperforms LLaMA3.2-1B (9T tokens) on all considered benchmarks, and surpasses Qwen2.5-1.5B (18T tokens) on a wide range of English and multilingual tasks. It matches or exceeds Qwen3 (36T tokens) on most tasks outside advanced STEM, achieving state-of-the-art results in Russian, including the MERA benchmark, among the models of comparable size (1-2B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12620v3">Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 9 pages, 4 figures, 5 tables. Accepted at AAAI 2026 Bridge Program on Logic & AI. Code available at https://github.com/XAheli/Logic-in-LLMs
    </div>
    <details class="paper-abstract">
      We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25987v2">R-Log: Incentivizing Log Analysis Capability in LLMs via Reasoning-based Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 Accepted by ICSE 2026 (SEIP Track)
    </div>
    <details class="paper-abstract">
      The growing complexity of log data in modern software systems has prompted the use of Large Language Models (LLMs) for automated log analysis. Current approaches typically rely on direct supervised fine-tuning (SFT) on log-label pairs. However, this exacerbates the domain discrepancy between general-purpose LLMs and specialized log data, causing overfitting. Furthermore, SFT's imbalanced loss computation often allows lengthy contexts to overwhelm critical, concise details in model answers, leading to hallucinations. To address these limitations, we propose R-Log, a novel reasoning-based paradigm that mirrors the structured, step-by-step analytical process of human engineers. This approach enhances generalizability by learning the underlying rules behind conclusions. We further employ Reinforcement Learning (RL) to optimize the model within a simulated O&M environment, thereby reducing hallucinations by directly rewarding correct outcomes. R-Log is first cold-started on a curated dataset of 2k+ reasoning trajectories, guided by 13 strategies from manual O&M practices, to establish an initial reasoning capability. This ability is then refined via RL using a joint reward function. Empirical evaluations on real-world logs show that R-Log outperforms existing methods across five log analysis tasks, particularly in unseen scenarios (by 228.05%). We also designed R-Log-fast with 5x speedup while keeping 93% of the efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05822v3">Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Current video understanding models excel at recognizing "what" is happening but fall short in high-level cognitive tasks like causal reasoning and future prediction, a limitation rooted in their lack of commonsense world knowledge. To bridge this cognitive gap, we propose a novel framework that synergistically fuses a powerful Vision Foundation Model (VFM) for deep visual perception with a Large Language Model (LLM) serving as a knowledge-driven reasoning core. Our key technical innovation is a sophisticated fusion module, inspired by the Q-Former architecture, which distills complex spatiotemporal and object-centric visual features into a concise, language-aligned representation. This enables the LLM to effectively ground its inferential processes in direct visual evidence. The model is trained via a two-stage strategy, beginning with large-scale alignment pre-training on video-text data, followed by targeted instruction fine-tuning on a curated dataset designed to elicit advanced reasoning and prediction skills. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple challenging benchmarks. Notably, it exhibits remarkable zero-shot generalization to unseen reasoning tasks, and our in-depth ablation studies validate the critical contribution of each architectural component. This work pushes the boundary of machine perception from simple recognition towards genuine cognitive understanding, paving the way for more intelligent and capable AI systems in robotics, human-computer interaction, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01412v2">Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23310v1">Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 11 pages, 9 figures. Accepted by ACM for presentation at UCC '25 (18th International Conference on Utility and Cloud Computing), December 1-4, 2025, France. Proceedings publication pending
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge devices is challenging due to their limited memory and power resources. Cloud-only inference reduces device burden but introduces high latency and cost. Static edge-cloud partitions optimize a single metric and struggle when bandwidth fluctuates. We propose Splitwise, a novel Lyapunov-assisted deep reinforcement learning (DRL) framework for fine-grained, adaptive partitioning of LLMs across edge and cloud environments. Splitwise decomposes transformer layers into attention heads and feed-forward sub-blocks, exposing more partition choices than layer-wise schemes. A hierarchical DRL policy, guided by Lyapunov optimization, jointly minimizes latency, energy consumption, and accuracy degradation while guaranteeing queue stability under stochastic workloads and variable network bandwidth. Splitwise also guarantees robustness via partition checkpoints with exponential backoff recovery in case of communication failures. Experiments on Jetson Orin NX, Galaxy S23, and Raspberry Pi 5 with GPT-2 (1.5B), LLaMA-7B, and LLaMA-13B show that Splitwise reduces end-to-end latency by 1.4x-2.8x and cuts energy consumption by up to 41% compared with existing partitioners. It lowers the 95th-percentile latency by 53-61% relative to cloud-only execution, while maintaining accuracy and modest memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06435v2">Learning the Topic, Not the Language: How LLMs Classify Online Immigration Discourse Across Languages</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer new opportunities for scalable analysis of online discourse. Yet their use in multilingual social science research remains constrained by model size, cost and linguistic bias. We develop a lightweight, open-source LLM framework using fine-tuned LLaMA 3.2-3B models to classify immigration-related tweets across 13 languages. Unlike prior work relying on BERT style models or translation pipelines, we combine topic classification with stance detection and demonstrate that LLMs fine-tuned in just one or two languages can generalize topic understanding to unseen languages. Capturing ideological nuance, however, benefits from multilingual fine-tuning. Our approach corrects pretraining biases with minimal data from under-represented languages and avoids reliance on proprietary systems. With 26-168x faster inference and over 1000x cost savings compared to commercial LLMs, our method supports real-time analysis of billions of tweets. This scale-first framework enables inclusive, reproducible research on public attitudes across linguistic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09427v2">ODMA: On-Demand Memory Allocation Framework for LLM Serving on LPDDR-Class Accelerators</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Device-memory management is a key bottleneck for serving large language models (LLMs) on accelerators whose memory has poor small-granularity random-access bandwidth (e.g., LPDDR5-class). Existing approaches either statically pre-allocate worst-case KV-cache per request, wasting substantial device memory, or rely on fine-grained paging that assumes high random-access tolerance and is therefore ill-suited to LPDDR-style systems. We present ODMA, an on-demand memory allocation framework for LLM serving on random-access-constrained device memory (RACM) platforms such as LPDDR5-based Cambricon MLUs. ODMA builds on generation-length prediction while addressing distribution drift and heavy-tailed request lengths via dynamic bucket partitioning and a large-bucket safeguard: bucket boundaries are periodically re-learned from online histograms, and high-uncertainty or overflowed requests fall back to a reserved large bucket for robustness. On Alpaca and Google-NQ, ODMA improves S3's predictor accuracy from 98.60% to 99.55% and from 82.68% to 93.36%, respectively. Serving DeepSeek-R1-Distill-Qwen-7B on four Cambricon MLU370-X4 accelerators, ODMA increases device-memory utilization from 55.05% to 72.45% on Alpaca and from 42.54% to 61.79% on Google-NQ, and boosts throughput by 23% and 27% over a static pre-allocation baseline. These results show that predictor-driven, hardware-aware allocation can unlock efficient LLM serving on RACM accelerators without hardware changes, complementing paging-centric designs tailored to HBM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23214v1">Anka: A Domain-Specific Language for Reliable LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-29
      | 💬 11 pages, 1 figure, 4 tables. Code and benchmarks available at https://github.com/BleBlo/Anka
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet they exhibit systematic errors on complex, multi-step programming tasks. We hypothesize that these errors stem from the flexibility of general-purpose languages, which permits multiple valid approaches and requires implicit state management. To test this hypothesis, we introduce Anka, a domain-specific language (DSL) for data transformation pipelines designed with explicit, constrained syntax that reduces ambiguity in code generation. Despite having zero prior training exposure to Anka, Claude 3.5 Haiku achieves 99.9% parse success and 95.8% overall task accuracy across 100 benchmark problems. Critically, Anka demonstrates a 40 percentage point accuracy advantage over Python on multi-step pipeline tasks (100% vs. 60%), where Python's flexible syntax leads to frequent errors in operation sequencing and variable management. Cross-model validation with GPT-4o-mini confirms this advantage (+26.7 percentage points on multi-step tasks). Our results demonstrate that: (1) LLMs can learn novel DSLs entirely from in-context prompts, achieving near-native accuracy; (2) constrained syntax significantly reduces errors on complex tasks; and (3) domain-specific languages purposefully designed for LLM generation can outperform general-purpose languages on which the LLM has extensive training. We release the complete language implementation, benchmark suite, and evaluation framework to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23206v1">Not too long do read: Evaluating LLM-generated extreme scientific summaries</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      High-quality scientific extreme summary (TLDR) facilitates effective science communication. How do large language models (LLMs) perform in generating them? How are LLM-generated summaries different from those written by human experts? However, the lack of a comprehensive, high-quality scientific TLDR dataset hinders both the development and evaluation of LLMs' summarization ability. To address these, we propose a novel dataset, BiomedTLDR, containing a large sample of researcher-authored summaries from scientific papers, which leverages the common practice of including authors' comments alongside bibliography items. We then test popular open-weight LLMs for generating TLDRs based on abstracts. Our analysis reveals that, although some of them successfully produce humanoid summaries, LLMs generally exhibit a greater affinity for the original text's lexical choices and rhetorical structures, hence tend to be more extractive rather than abstractive in general, compared to humans. Our code and datasets are available at https://github.com/netknowledge/LLM_summarization (Lyu and Ke, 2025).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23184v1">From Model Choice to Model Belief: Establishing a New Measure for LLM-Based Research</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate human behavior, but common practices to use LLM-generated data are inefficient. Treating an LLM's output ("model choice") as a single data point underutilizes the information inherent to the probabilistic nature of LLMs. This paper introduces and formalizes "model belief," a measure derived from an LLM's token-level probabilities that captures the model's belief distribution over choice alternatives in a single generation run. The authors prove that model belief is asymptotically equivalent to the mean of model choices (a non-trivial property) but forms a more statistically efficient estimator, with lower variance and a faster convergence rate. Analogous properties are shown to hold for smooth functions of model belief and model choice often used in downstream applications. The authors demonstrate the performance of model belief through a demand estimation study, where an LLM simulates consumer responses to different prices. In practical settings with limited numbers of runs, model belief explains and predicts ground-truth model choice better than model choice itself, and reduces the computation needed to reach sufficiently accurate estimates by roughly a factor of 20. The findings support using model belief as the default measure to extract more information from LLM-generated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23167v1">SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often falter at complex planning tasks that require exploration and self-correction, as their linear reasoning process struggles to recover from early mistakes. While search algorithms like Monte Carlo Tree Search (MCTS) can explore alternatives, they are often ineffective when guided by sparse rewards and fail to leverage the rich semantic capabilities of LLMs. We introduce SPIRAL (Symbolic LLM Planning via Grounded and Reflective Search), a novel framework that embeds a cognitive architecture of three specialized LLM agents into an MCTS loop. SPIRAL's key contribution is its integrated planning pipeline where a Planner proposes creative next steps, a Simulator grounds the search by predicting realistic outcomes, and a Critic provides dense reward signals through reflection. This synergy transforms MCTS from a brute-force search into a guided, self-correcting reasoning process. On the DailyLifeAPIs and HuggingFace datasets, SPIRAL consistently outperforms the default Chain-of-Thought planning method and other state-of-the-art agents. More importantly, it substantially surpasses other state-of-the-art agents; for example, SPIRAL achieves 83.6% overall accuracy on DailyLifeAPIs, an improvement of over 16 percentage points against the next-best search framework, while also demonstrating superior token efficiency. Our work demonstrates that structuring LLM reasoning as a guided, reflective, and grounded search process yields more robust and efficient autonomous planners. The source code, full appendices, and all experimental data are available for reproducibility at the official project repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25743v2">Rotation Control Unlearning: Quantifying and Controlling Continuous Unlearning for LLM with The Cognitive Rotation Space</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent, their security vulnerabilities have already drawn attention. Machine unlearning is introduced to seek to mitigate these risks by removing the influence of undesirable data. However, existing methods not only rely on the retained dataset to preserve model utility, but also suffer from cumulative catastrophic utility loss under continuous unlearning requests. To solve this dilemma, we propose a novel method, called Rotation Control Unlearning (RCU), which leverages the rotational salience weight of RCU to quantify and control the unlearning degree in the continuous unlearning process. The skew symmetric loss is designed to construct the existence of the cognitive rotation space, where the changes of rotational angle can simulate the continuous unlearning process. Furthermore, we design an orthogonal rotation axes regularization to enforce mutually perpendicular rotation directions for continuous unlearning requests, effectively minimizing interference and addressing cumulative catastrophic utility loss. Experiments on multiple datasets confirm that our method without retained dataset achieves SOTA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2306.05499v3">Prompt Injection attack against LLM-integrated Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23136v1">Understanding EFL Learners' Code-Switching and Teachers' Pedagogical Approaches in LLM-Supported Speaking Practice</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      For English as a Foreign Language (EFL) learners, code-switching (CSW), or alternating between their native language and the target language (English), can lower anxiety and ease communication barriers. Large language models (LLMs), with their multilingual abilities, offer new opportunities to support CSW in speaking practice. Yet, the pedagogical design of LLM-based tutors remains underexplored. To this end, we conducted a six-week study of LLM-mediated speaking practice with 20 Korean EFL learners, alongside a qualitative study with nine English teachers who designed and refined responses to learner CSW. Findings show that learners used CSW not only to bridge lexical gaps but also to express cultural and emotional nuance, prompting teachers to employ selective interventions and dynamic scaffolding strategies. We conclude with design implications for bilingual LLM-powered tutors that leverage teachers' expertise to transform CSW into meaningful learning opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23126v1">InSPO: Unlocking Intrinsic Self-Reflection for LLM Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-29
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) and its variants have become standard for aligning Large Language Models due to their simplicity and offline stability. However, we identify two fundamental limitations. First, the optimal policy depends on arbitrary modeling choices (scalarization function, reference policy), yielding behavior reflecting parameterization artifacts rather than true preferences. Second, treating response generation in isolation fails to leverage comparative information in pairwise data, leaving the model's capacity for intrinsic self-reflection untapped. To address it, we propose Intrinsic Self-reflective Preference Optimization (\q), deriving a globally optimal policy conditioning on both context and alternative responses. We prove this formulation superior to DPO/RLHF while guaranteeing invariance to scalarization and reference choices. \q~serves as a plug-and-play enhancement without architectural changes or inference overhead. Experiments demonstrate consistent improvements in win rates and length-controlled metrics, validating that unlocking self-reflection yields more robust, human-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23097v1">A Note on Hybrid Online Reinforcement and Imitation Learning for LLMs: Formulations and Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      We present a unified framework for Large Language Model (LLM) fine-tuning that integrates Imitation Learning and Reinforcement Learning. By analyzing the gradient of a composite objective combining trajectory-level KL divergence with task rewards, we derive a natural decomposition into two components: (1) an analytically computable Dense Gradient for token-level imitation, and (2) a Monte Carlo estimated Sparse Gradient for long-horizon reward optimization. The Dense Gradient admits a closed-form logit-level formula, enabling efficient GPU implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23087v1">Taming the Tail: Stable LLM Reinforcement Learning via Dynamic Vocabulary Pruning</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Reinforcement learning for large language models (LLMs) faces a fundamental tension: high-throughput inference engines and numerically-precise training systems produce different probability distributions from the same parameters, creating a training-inference mismatch. We prove this mismatch has an asymmetric effect: the bound on log-probability mismatch scales as $(1-p)$ where $p$ is the token probability. For high-probability tokens, this bound vanishes, contributing negligibly to sequence-level mismatch. For low-probability tokens in the tail, the bound remains large, and moreover, when sampled, these tokens exhibit systematically biased mismatches that accumulate over sequences, destabilizing gradient estimation. Rather than applying post-hoc corrections, we propose constraining the RL objective to a dynamically-pruned ``safe'' vocabulary that excludes the extreme tail. By pruning such tokens, we trade large, systematically biased mismatches for a small, bounded optimization bias. Empirically, our method achieves stable training; theoretically, we bound the optimization bias introduced by vocabulary pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23075v1">Trust Region Masking for Long-Horizon LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Policy gradient methods for large language models optimize a surrogate objective computed from samples of a rollout policy $π_{\text{roll}}$. When $π_{\text{roll}} \ne π_θ$, there is approximation error between the surrogate and the true objective. Prior work has shown that this off-policy mismatch is unavoidable in modern LLM-RL due to implementation divergence, mixture-of-experts routing discontinuities, and distributed training staleness. Classical trust region bounds on the resulting error scale as $O(T^2)$ with sequence length $T$, rendering them vacuous for long-horizon tasks. We derive two tighter bounds: a Pinsker-Marginal bound scaling as $O(T^{3/2})$ and a Mixed bound scaling as $O(T)$. Crucially, both bounds depend on $D_{kl}^{tok,max}$ -- the maximum token-level KL divergence across all positions in a sequence. This is inherently a sequence-level quantity: it requires examining the entire trajectory to compute, and therefore cannot be controlled by token-independent methods like PPO clipping. We propose Trust Region Masking (TRM), which excludes entire sequences from gradient computation if any token violates the trust region, providing the first non-vacuous monotonic improvement guarantees for long-horizon LLM-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23036v1">Problems With Large Language Models for Learner Modelling: Why LLMs Alone Fall Short for Responsible Tutoring in K--12 Education</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      The rapid rise of large language model (LLM)-based tutors in K--12 education has fostered a misconception that generative models can replace traditional learner modelling for adaptive instruction. This is especially problematic in K--12 settings, which the EU AI Act classifies as high-risk domain requiring responsible design. Motivated by these concerns, this study synthesises evidence on limitations of LLM-based tutors and empirically investigates one critical issue: the accuracy, reliability, and temporal coherence of assessing learners' evolving knowledge over time. We compare a deep knowledge tracing (DKT) model with a widely used LLM, evaluated zero-shot and fine-tuned, using a large open-access dataset. Results show that DKT achieves the highest discrimination performance (AUC = 0.83) on next-step correctness prediction and consistently outperforms the LLM across settings. Although fine-tuning improves the LLM's AUC by approximately 8\% over the zero-shot baseline, it remains 6\% below DKT and produces higher early-sequence errors, where incorrect predictions are most harmful for adaptive support. Temporal analyses further reveal that DKT maintains stable, directionally correct mastery updates, whereas LLM variants exhibit substantial temporal weaknesses, including inconsistent and wrong-direction updates. These limitations persist despite the fine-tuned LLM requiring nearly 198 hours of high-compute training, far exceeding the computational demands of DKT. Our qualitative analysis of multi-skill mastery estimation further shows that, even after fine-tuning, the LLM produced inconsistent mastery trajectories, while DKT maintained smooth and coherent updates. Overall, the findings suggest that LLMs alone are unlikely to match the effectiveness of established intelligent tutoring systems, and that responsible tutoring requires hybrid frameworks that incorporate learner modelling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23029v1">Viability and Performance of a Private LLM Server for SMBs: A Benchmark Analysis of Qwen3-30B on Consumer-Grade Hardware</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has been accompanied by a reliance on cloud-based, proprietary systems, raising significant concerns regarding data privacy, operational sovereignty, and escalating costs. This paper investigates the feasibility of deploying a high-performance, private LLM inference server at a cost accessible to Small and Medium Businesses (SMBs). We present a comprehensive benchmarking analysis of a locally hosted, quantized 30-billion parameter Mixture-of-Experts (MoE) model based on Qwen3, running on a consumer-grade server equipped with a next-generation NVIDIA GPU. Unlike cloud-based offerings, which are expensive and complex to integrate, our approach provides an affordable and private solution for SMBs. We evaluate two dimensions: the model's intrinsic capabilities and the server's performance under load. Model performance is benchmarked against academic and industry standards to quantify reasoning and knowledge relative to cloud services. Concurrently, we measure server efficiency through latency, tokens per second, and time to first token, analyzing scalability under increasing concurrent users. Our findings demonstrate that a carefully configured on-premises setup with emerging consumer hardware and a quantized open-source model can achieve performance comparable to cloud-based services, offering SMBs a viable pathway to deploy powerful LLMs without prohibitive costs or privacy compromises.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23025v1">LENS: LLM-Enabled Narrative Synthesis for Mental Health by Aligning Multimodal Sensing with Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 22 pages, 9 figures, under review
    </div>
    <details class="paper-abstract">
      Multimodal health sensing offers rich behavioral signals for assessing mental health, yet translating these numerical time-series measurements into natural language remains challenging. Current LLMs cannot natively ingest long-duration sensor streams, and paired sensor-text datasets are scarce. To address these challenges, we introduce LENS, a framework that aligns multimodal sensing data with language models to generate clinically grounded mental-health narratives. LENS first constructs a large-scale dataset by transforming Ecological Momentary Assessment (EMA) responses related to depression and anxiety symptoms into natural-language descriptions, yielding over 100,000 sensor-text QA pairs from 258 participants. To enable native time-series integration, we train a patch-level encoder that projects raw sensor signals directly into an LLM's representation space. Our results show that LENS outperforms strong baselines on standard NLP metrics and task-specific measures of symptom-severity accuracy. A user study with 13 mental-health professionals further indicates that LENS-produced narratives are comprehensive and clinically meaningful. Ultimately, our approach advances LLMs as interfaces for health sensing, providing a scalable path toward models that can reason over raw behavioral signals and support downstream clinical decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14973v2">Attention Is All You Need for KV Cache in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 Code at: https://github.com/VILA-Lab/Elastic-Cache
    </div>
    <details class="paper-abstract">
      This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), and $45.1\times$ on longer sequences, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23014v1">Improving Generalization in LLM Structured Pruning via Function-Aware Neuron Grouping</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive performance across natural language tasks but incur substantial computational and storage costs due to their scale. Post-training structured pruning offers an efficient solution. However, when few-shot calibration sets fail to adequately reflect the pretraining data distribution, existing methods exhibit limited generalization to downstream tasks. To address this issue, we propose Function-Aware Neuron Grouping (FANG), a post-training pruning framework that alleviates calibration bias by identifying and preserving neurons critical to specific function. FANG groups neurons with similar function based on the type of semantic context they process and prunes each group independently. During importance estimation within each group, tokens that strongly correlate with the functional role of the neuron group are given higher weighting. Additionally, FANG also preserves neurons that contribute across multiple context types. To achieve a better trade-off between sparsity and performance, it allocates sparsity to each block adaptively based on its functional complexity. Experiments show that FANG improves downstream accuracy while preserving language modeling performance. It achieves the state-of-the-art (SOTA) results when combined with FLAP and OBC, two representative pruning methods. Specifically, FANG outperforms FLAP and OBC by 1.5%--8.5% in average accuracy under 30% and 40% sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22925v1">Argus: Token Aware Distributed LLM Inference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly being integrated into real-world applications, yet their autoregressive architectures introduce significant inference time variability, especially when deployed across heterogeneous edge-cloud systems. Existing solutions largely neglect the dynamic, stochastic, and heterogeneous nature of such environments, often ignoring the impact of variable output token lengths and device diversity. In this work, we present Argus, the first token-aware distributed edge-cloud LLM inference framework that conducts efficient task offloading. Argus features a Length-Aware Semantics (LAS) module, which predicts output token lengths for incoming prompts using a fine-tuned language model with token-length-sensitive feature modulation, enabling precise estimation. Building on this, our Lyapunov-guided Offloading Optimization (LOO) module formulates long-term Quality-of-Experience optimization that explicitly considers both LLM prefilling and decoding costs. We introduce a novel Iterative Offloading Algorithm with Damping and Congestion Control (IODCC) to effectively solve the resulting integer nonlinear programming problem under time-varying constraints. Extensive theoretical and empirical evaluations demonstrate that Argus achieves robust performance and superior efficiency in highly dynamic, heterogeneous settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22905v1">JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 Accepted by NeurIPS as a Spotlight paper. Code: https://github.com/JavisVerse/JavisGPT
    </div>
    <details class="paper-abstract">
      This paper presents JavisGPT, the first unified multimodal large language model (MLLM) for Joint Audio-Video (JAV) comprehension and generation. JavisGPT adopts a concise encoder-LLM-decoder architecture, featuring a SyncFusion module for spatio-temporal audio-video fusion and synchrony-aware learnable queries to bridge a pretrained JAV-DiT generator. This design enables temporally coherent video-audio understanding and generation from multimodal instructions. We design an effective three-stage training pipeline consisting of multimodal pretraining, audio-video fine-tuning, and large-scale instruction-tuning, to progressively build multimodal comprehension and generation from existing vision-language models. To support this, we further construct JavisInst-Omni, a high-quality instruction dataset with over 200K GPT-4o-curated audio-video-text dialogues that span diverse and multi-level comprehension and generation scenarios. Extensive experiments on JAV comprehension and generation benchmarks show that JavisGPT outperforms existing MLLMs, particularly in complex and temporally synchronized settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22827v1">FasterPy: An LLM-based Code Execution Efficiency Optimization Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 32 pages, 5 images, 7 tables, Manuscript submitted to a Journal (2025)
    </div>
    <details class="paper-abstract">
      Code often suffers from performance bugs. These bugs necessitate the research and practice of code optimization. Traditional rule-based methods rely on manually designing and maintaining rules for specific performance bugs (e.g., redundant loops, repeated computations), making them labor-intensive and limited in applicability. In recent years, machine learning and deep learning-based methods have emerged as promising alternatives by learning optimization heuristics from annotated code corpora and performance measurements. However, these approaches usually depend on specific program representations and meticulously crafted training datasets, making them costly to develop and difficult to scale. With the booming of Large Language Models (LLMs), their remarkable capabilities in code generation have opened new avenues for automated code optimization. In this work, we proposed FasterPy, a low-cost and efficient framework that adapts LLMs to optimize the execution efficiency of Python code. FasterPy combines Retrieval-Augmented Generation (RAG), supported by a knowledge base constructed from existing performance-improving code pairs and corresponding performance measurements, with Low-Rank Adaptation (LoRA) to enhance code optimization performance. Our experimental results on the Performance Improving Code Edits (PIE) benchmark demonstrate that our method outperforms existing models on multiple metrics. The FasterPy tool and the experimental results are available at https://github.com/WuYue22/fasterpy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.15759v2">Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 21 pages, 7 figures; Accepted by IEEE TIP
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual understanding and reasoning, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance the overall capabilities of LLMs, which could be regarded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce Modular Visual Memory (MVM), a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixture of Multimodal Experts (MoMEs) architecture in LLMs to invoke multimodal knowledge collaboration during text generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on image-text understanding multimodal benchmarks. The codes will be available at: https://github.com/HITsz-TMG/MKS2-Multimodal-Knowledge-Storage-and-Sharing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21218v3">Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22790v1">ChatGraPhT: A Visual Conversation Interface for Multi-Path Reflection with Agentic LLM Support</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 Early-stage research prototype; exploratory study
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in complex knowledge work, yet linear transcript interfaces limit support for reflection. Schon's Reflective Practice distinguishes between reflection-in-action (during a task) and reflection-on-action (after a task), both benefiting from non-linear, revisitable representations of dialogue. ChatGraPhT is an interactive tool that shows dialogue as a visual map, allowing users to branch and merge ideas, edit past messages, and receive guidance that prompts deeper reflection. It supports non-linear, multi-path dialogue, while two agentic LLM assistants provide moment-to-moment and higher-level guidance. Our inquiry suggests that keeping the conversation structure visible, allowing branching and merging, and suggesting patterns or ways to combine ideas deepened user reflective engagement. Contributions are: (1) the design of a node-link, agentic LLM interface for reflective dialogue, and (2) transferable design knowledge on balancing structure and AI support to sustain reflection in complex, open-ended tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22753v1">From Rookie to Expert: Manipulating LLMs for Automated Vulnerability Exploitation in Enterprise Software</a></div>
    <div class="paper-meta">
      📅 2025-12-28
    </div>
    <details class="paper-abstract">
      LLMs democratize software engineering by enabling non-programmers to create applications, but this same accessibility fundamentally undermines security assumptions that have guided software engineering for decades. We show in this work how publicly available LLMs can be socially engineered to transform novices into capable attackers, challenging the foundational principle that exploitation requires technical expertise. To that end, we propose RSA (Role-assignment, Scenario-pretexting, and Action-solicitation), a pretexting strategy that manipulates LLMs into generating functional exploits despite their safety mechanisms. Testing against Odoo -- a widely used ERP platform, we evaluated five mainstream LLMs (GPT-4o, Gemini, Claude, Microsoft Copilot, and DeepSeek) and achieved a 100% success rate: tested CVE yielded at least one working exploit within 3-4 prompting rounds. While prior work [13] found LLM-assisted attacks difficult and requiring manual effort, we demonstrate that this overhead can be eliminated entirely. Our findings invalidate core software engineering security principles: the distinction between technical and non-technical actors no longer provides valid threat models; technical complexity of vulnerability descriptions offers no protection when LLMs can abstract it away; and traditional security boundaries dissolve when the same tools that build software can be manipulated to break it. This represents a paradigm shift in software engineering -- we must redesign security practices for an era where exploitation requires only the ability to craft prompts, not understand code. Artifacts available at: https://anonymous.4open.science/r/From-Rookie-to-Attacker-D8B3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22742v1">Robust LLM-based Column Type Annotation via Prompt Augmentation with LoRA Tuning</a></div>
    <div class="paper-meta">
      📅 2025-12-28
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Column Type Annotation (CTA) is a fundamental step towards enabling schema alignment and semantic understanding of tabular data. Existing encoder-only language models achieve high accuracy when fine-tuned on labeled columns, but their applicability is limited to in-domain settings, as distribution shifts in tables or label spaces require costly re-training from scratch. Recent work has explored prompting generative large language models (LLMs) by framing CTA as a multiple-choice task, but these approaches face two key challenges: (1) model performance is highly sensitive to subtle changes in prompt wording and structure, and (2) annotation F1 scores remain modest. A natural extension is to fine-tune large language models. However, fully fine-tuning these models incurs prohibitive computational costs due to their scale, and the sensitivity to prompts is not eliminated. In this paper, we present a parameter-efficient framework for CTA that trains models over prompt-augmented data via Low-Rank Adaptation (LoRA). Our approach mitigates sensitivity to prompt variations while drastically reducing the number of necessary trainable parameters, achieving robust performance across datasets and templates. Experimental results on recent benchmarks demonstrate that models fine-tuned with our prompt augmentation strategy maintain stable performance across diverse prompt patterns during inference and yield higher weighted F1 scores than those fine-tuned on a single prompt template. These results highlight the effectiveness of parameter-efficient training and augmentation strategies in developing practical and adaptable CTA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10013v2">DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have risen in prominence over the past few years, there has been concern over the potential biases in LLMs inherited from the training data. Previous studies have examined how LLMs exhibit implicit bias, such as when response generation changes when different social contexts are introduced. We argue that this implicit bias is not only an ethical, but also a technical issue, as it reveals an inability of LLMs to accommodate extraneous information. However, unlike other measures of LLM intelligence, there are no standard methods to benchmark this specific subset of LLM bias. To bridge this gap, we developed a method for calculating an easily interpretable benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM logic and math problem datasets with sociodemographic personas, which is combined with a statistical robustness check using a null model. We demonstrate that this method can validate the presence of implicit bias in LLM behavior and find an novel inverse trend between question answering accuracy and implicit bias, supporting our argument.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21880v2">No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via Entropy-Guided Advantage Shaping</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Under review. Project page: https://bltnynk.github.io/publications/rl-zvp/
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) is a powerful framework for improving the reasoning abilities of Large Language Models (LLMs). However, current methods such as GRPO rely only on problems where the model responses to the same input differ in correctness, while ignoring those where all responses receive the same reward -- so-called zero-variance prompts. In this work, we argue that such prompts are not useless but can, in fact, provide meaningful feedback for policy optimization. To this end, we introduce RL with Zero-Variance Prompts (RL-ZVP), a novel algorithm that extract learning signals from zero-variance prompts. RL-ZVP directly rewards correctness and penalizes errors even without contrasting responses, modulating feedback with token-level characteristics to preserve informative, nuanced signals. Across six math reasoning benchmarks, RL-ZVP achieves significant improvements of up to 8.61 points in accuracy and 7.77 points in pass rate over GRPO, while consistently outperforming other baselines that filter out zero-variance prompts. These results highlight the untapped potential of learning from zero-variance prompts in RLVR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22631v1">Evaluating GRPO and DPO for Faithful Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) reasoning has emerged as a powerful technique for improving the problem-solving capabilities of large language models (LLMs), particularly for tasks requiring multi-step reasoning. However, recent studies show that CoT explanations often fail to reflect the model's actual reasoning process, as models may produce coherent yet misleading justifications or modify answers without acknowledging external cues. Such discrepancies undermine the reliability of CoT-based methods for safety supervision and alignment monitoring, as models can generate plausible but deceptive rationales for incorrect answers. To better understand this limitation, we evaluate two optimization methods, Group Relative Policy Optimization (GRPO) and Direct Preference Optimization (DPO), in their ability to improve CoT faithfulness. Our experiments show that GRPO achieves higher performance than DPO in larger models, with the Qwen2.5-14B-Instruct model attaining the best results across all evaluation metrics. Both approaches exhibit positive correlations between model size and performance, but GRPO shows greater potential for improving faithfulness metrics, albeit with less stable behavior at smaller scales. These results suggest that GRPO offers a promising direction for developing more transparent and trustworthy reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17923v2">Inferring Latent Market Forces: Evaluating LLM Detection of Gamma Exposure Patterns via Obfuscation Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 10 pages, 8 figures. Accepted at IEEE Big Data 2025. Extended journal version in preparation. ISBN: 979-8-3315-9447-3/25. Page numbers: 7226-7235
    </div>
    <details class="paper-abstract">
      We introduce obfuscation testing, a novel methodology for validating whether large language models detect structural market patterns through causal reasoning rather than temporal association. Testing three dealer hedging constraint patterns (gamma positioning, stock pinning, 0DTE hedging) on 242 trading days (95.6% coverage) of S&P 500 options data, we find LLMs achieve 71.5% detection rate using unbiased prompts that provide only raw gamma exposure values without regime labels or temporal context. The WHO-WHOM-WHAT causal framework forces models to identify the economic actors (dealers), affected parties (directional traders), and structural mechanisms (forced hedging) underlying observed market dynamics. Critically, detection accuracy (91.2%) remains stable even as economic profitability varies quarterly, demonstrating that models identify structural constraints rather than profitable patterns. When prompted with regime labels, detection increases to 100%, but the 71.5% unbiased rate validates genuine pattern recognition. Our findings suggest LLMs possess emergent capabilities for detecting complex financial mechanisms through pure structural reasoning, with implications for systematic strategy development, risk management, and our understanding of how transformer architectures process financial market dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22625v1">The Wisdom of Deliberating AI Crowds: Does Deliberation Improve LLM-Based Forecasting?</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 13 pages, 2 figures, 5 tables, for source code and data see https://github.com/priorb-source/delib-ai-wisdom
    </div>
    <details class="paper-abstract">
      Structured deliberation has been found to improve the performance of human forecasters. This study investigates whether a similar intervention, i.e. allowing LLMs to review each other's forecasts before updating, can improve accuracy in large language models (GPT-5, Claude Sonnet 4.5, Gemini Pro 2.5). Using 202 resolved binary questions from the Metaculus Q2 2025 AI Forecasting Tournament, accuracy was assessed across four scenarios: (1) diverse models with distributed information, (2) diverse models with shared information, (3) homogeneous models with distributed information, and (4) homogeneous models with shared information. Results show that the intervention significantly improves accuracy in scenario (2), reducing Log Loss by 0.020 or about 4 percent in relative terms (p = 0.017). However, when homogeneous groups (three instances of the same model) engaged in the same process, no benefit was observed. Unexpectedly, providing LLMs with additional contextual information did not improve forecast accuracy, limiting our ability to study information pooling as a mechanism. Our findings suggest that deliberation may be a viable strategy for improving LLM forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22603v1">Structured Prompting and LLM Ensembling for Multimodal Conversational Aspect-based Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Understanding sentiment in multimodal conversations is a complex yet crucial challenge toward building emotionally intelligent AI systems. The Multimodal Conversational Aspect-based Sentiment Analysis (MCABSA) Challenge invited participants to tackle two demanding subtasks: (1) extracting a comprehensive sentiment sextuple, including holder, target, aspect, opinion, sentiment, and rationale from multi-speaker dialogues, and (2) detecting sentiment flipping, which detects dynamic sentiment shifts and their underlying triggers. For Subtask-I, in the present paper, we designed a structured prompting pipeline that guided large language models (LLMs) to sequentially extract sentiment components with refined contextual understanding. For Subtask-II, we further leveraged the complementary strengths of three LLMs through ensembling to robustly identify sentiment transitions and their triggers. Our system achieved a 47.38% average score on Subtask-I and a 74.12% exact match F1 on Subtask-II, showing the effectiveness of step-wise refinement and ensemble strategies in rich, multimodal sentiment analysis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02910v2">In Silico Development of Psychometric Scales: Feasibility of Representative Population Data Simulation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Developing and validating psychometric scales requires large samples, multiple testing phases, and substantial resources. Recent advances in Large Language Models (LLMs) enable the generation of synthetic participant data by prompting models to answer items while impersonating individuals of specific demographic profiles, potentially allowing in silico piloting before real data collection. Across four preregistered studies (N = circa 300 each), we tested whether LLM-simulated datasets can reproduce the latent structures and measurement properties of human responses. In Studies 1-2, we compared LLM-generated data with real datasets for two validated scales; in Studies 3-4, we created new scales using EFA on simulated data and then examined whether these structures generalized to newly collected human samples. Simulated datasets replicated the intended factor structures in three of four studies and showed consistent configural and metric invariance, with scalar invariance achieved for the two newly developed scales. However, correlation-based tests revealed substantial differences between real and synthetic datasets, and notable discrepancies appeared in score distributions and variances. Thus, while LLMs capture group-level latent structures, they do not approximate individual-level data properties. Simulated datasets also showed full internal invariance across gender. Overall, LLM-generated data appear useful for early-stage, group-level psychometric prototyping, but not as substitutes for individual-level validation. We discuss methodological limitations, risks of bias and data pollution, and ethical considerations related to in silico psychometric simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00592v2">Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Accepted at the 34th International Conference on Parallel Architectures and Compilation Techniques (PACT 2025). 12 pages, plus appendix
    </div>
    <details class="paper-abstract">
      Automatic code optimization remains a difficult challenge, particularly for complex loop nests on modern hardware. This paper investigates a novel approach to code optimization where Large Language Models (LLMs) guide the process through a closed-loop interaction with a compiler. We present ComPilot, an experimental framework that leverages off-the-shelf LLMs, without any task-specific fine-tuning, as interactive optimization agents. ComPilot establishes a feedback loop where an LLM proposes transformations for a given loop nest to a compiler. The compiler attempts the transformations, reporting back legality status and measured speedup or slowdown. The LLM utilizes this concrete feedback to iteratively refine its optimization strategy. Our extensive evaluation across the PolyBench benchmark suite demonstrates the effectiveness of this zero-shot approach. ComPilot achieves geometric mean speedups of 2.66x (single run) and 3.54x (best-of-5 runs) over the original code. Furthermore, ComPilot demonstrates competitive performance against the state-of-the-art Pluto polyhedral optimizer, outperforming it in many cases. This experimental study demonstrates that general-purpose LLMs can effectively guide the code optimization process when grounded by compiler feedback, opening promising research directions for agentic AI in code optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12018v4">Atom of Thoughts for Markov LLM Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant performance gains through test-time scaling methods. However, existing approaches often incur redundant computations due to the accumulation of historical dependency information during inference. To address this challenge, we leverage the memoryless property of Markov processes to minimize reliance on historical context and propose a Markovian reasoning process. This foundational Markov chain structure enables seamless integration with various test-time scaling methods, thereby improving their scaling efficiency. By further scaling up the Markovian reasoning chain through integration with techniques such as tree search and reflective refinement, we uncover an emergent atomic reasoning structure, where reasoning trajectories are decomposed into a series of self-contained, low-complexity atomic units. We name this design Atom of Thoughts (\our). Extensive experiments demonstrate that \our consistently outperforms existing baselines as computational budgets increase. Importantly, \our integrates seamlessly with existing reasoning frameworks and different LLMs (both reasoning and non-reasoning), facilitating scalable, high-performance inference.We submit our code alongside this paper and will make it publicly available to facilitate reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.21015v3">Don't Retrieve, Generate: Prompting LLMs for Synthetic Training Data in Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Training effective dense retrieval models typically relies on hard negative (HN) examples mined from large document corpora using methods such as BM25 or cross-encoders, which require full corpus access and expensive index construction. We propose generating synthetic hard negatives directly from a provided query and positive passage, using Large Language Models(LLMs). We fine-tune DistilBERT using synthetic negatives generated by four state-of-the-art LLMs ranging from 4B to 30B parameters (Qwen3, LLaMA3, Phi4) and evaluate performance across 10 BEIR benchmark datasets. Contrary to the prevailing assumption that stronger generative models yield better synthetic data, find that our generative pipeline consistently underperforms traditional corpus-based mining strategies (BM25 and Cross-Encoder). Furthermore, we observe that scaling the generator model does not monotonically improve retrieval performance and find that the 14B parameter model outperforms the 30B model and in some settings it is the worst performing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22508v1">Predicting LLM Correctness in Prosthodontics Using Metadata and Hallucination Signals</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 Accepted as a Short Paper at HEALTHINF2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adopted in high-stakes domains such as healthcare and medical education, where the risk of generating factually incorrect (i.e., hallucinated) information is a major concern. While significant efforts have been made to detect and mitigate such hallucinations, predicting whether an LLM's response is correct remains a critical yet underexplored problem. This study investigates the feasibility of predicting correctness by analyzing a general-purpose model (GPT-4o) and a reasoning-centric model (OSS-120B) on a multiple-choice prosthodontics exam. We utilize metadata and hallucination signals across three distinct prompting strategies to build a correctness predictor for each (model, prompting) pair. Our findings demonstrate that this metadata-based approach can improve accuracy by up to +7.14% and achieve a precision of 83.12% over a baseline that assumes all answers are correct. We further show that while actual hallucination is a strong indicator of incorrectness, metadata signals alone are not reliable predictors of hallucination. Finally, we reveal that prompting strategies, despite not affecting overall accuracy, significantly alter the models' internal behaviors and the predictive utility of their metadata. These results present a promising direction for developing reliability signals in LLMs but also highlight that the methods explored in this paper are not yet robust enough for critical, high-stakes deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.10321v2">AdvPrefix: An Objective for Nuanced LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Many jailbreak attacks on large language models (LLMs) rely on a common objective: making the model respond with the prefix ``Sure, here is (harmful request)''. While straightforward, this objective has two limitations: limited control over model behaviors, yielding incomplete or unrealistic jailbroken responses, and a rigid format that hinders optimization. We introduce AdvPrefix, a plug-and-play prefix-forcing objective that selects one or more model-dependent prefixes by combining two criteria: high prefilling attack success rates and low negative log-likelihood. AdvPrefix integrates seamlessly into existing jailbreak attacks to mitigate the previous limitations for free. For example, replacing GCG's default prefixes on Llama-3 improves nuanced attack success rates from 14% to 80%, revealing that current safety alignment fails to generalize to new prefixes. Code and selected prefixes are released at github.com/facebookresearch/jailbreak-objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22492v1">Role-Based Fault Tolerance System for LLM RL Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 16 pages, 19 figures
    </div>
    <details class="paper-abstract">
      RL post-training for LLMs has been widely scaled to enhance reasoning and tool-using capabilities. However, RL post-training interleaves training and inference workloads, exposing the system to faults from both sides. Existing fault tolerance frameworks for LLMs target either training or inference, leaving the optimization potential in the asynchronous execution unexplored for RL. Our key insight is role-based fault isolation so the failure in one machine does not affect the others. We treat trainer, rollout, and other management roles in RL training as distinct distributed sub-tasks. Instead of restarting the entire RL task in ByteRobust, we recover only the failed role and reconnect it to living ones, thereby eliminating the full-restart overhead including rollout replay and initialization delay. We present RobustRL, the first comprehensive robust system to handle GPU machine errors for RL post-training Effective Training Time Ratio improvement. (1) \textit{Detect}. We implement role-aware monitoring to distinguish actual failures from role-specific behaviors to avoid the false positive and delayed detection. (2) \textit{Restart}. For trainers, we implement a non-disruptive recovery where rollouts persist state and continue trajectory generation, while the trainer is rapidly restored via rollout warm standbys. For rollout, we perform isolated machine replacement without interrupting the RL task. (3) \textit{Reconnect}. We replace static collective communication with dynamic, UCX-based (Unified Communication X) point-to-point communication, enabling immediate weight synchronization between recovered roles. In an RL training task on a 256-GPU cluster with Qwen3-8B-Math workload under 10\% failure injection frequency, RobustRL can achieve an ETTR of over 80\% compared with the 60\% in ByteRobust and achieves 8.4\%-17.4\% faster in end-to-end training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00256v2">LLM4FP: LLM-Based Program Generation for Triggering Floating-Point Inconsistencies Across Compilers</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      Floating-point inconsistencies across compilers can undermine the reliability of numerical software. We present LLM4FP, the first framework that uses Large Language Models (LLMs) to generate floating-point programs specifically designed to trigger such inconsistencies. LLM4FP combines Grammar-Based Generation and Feedback-Based Mutation to produce diverse and valid programs. We evaluate LLM4FP across multiple compilers and optimization levels, measuring inconsistency rate, time cost, and program diversity. LLM4FP detects nearly 2.5x the number of inconsistencies as the state-of-the-art tool Varity. Notably, most of the inconsistencies involve real-valued differences, rather than extreme values like NaN or infinities. LLM4FP also uncovers inconsistencies across a wider range of optimization levels, and finds the most mismatches between host and device compilers. These results show that LLM-guided program generation improves the detection of numerical inconsistencies. In practice, numerical software and HPC developers can use LLM4FP to compare compilers and select those that provide more accurate and consistent floating-point behavior, while compiler developers can use it to identify and address subtle consistency issues in their implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22470v1">DarkPatterns-LLM: A Multi-Layer Benchmark for Detecting Manipulative and Harmful AI Behavior</a></div>
    <div class="paper-meta">
      📅 2025-12-27
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has intensified concerns about manipulative or deceptive behaviors that can undermine user autonomy, trust, and well-being. Existing safety benchmarks predominantly rely on coarse binary labels and fail to capture the nuanced psychological and social mechanisms constituting manipulation. We introduce \textbf{DarkPatterns-LLM}, a comprehensive benchmark dataset and diagnostic framework for fine-grained assessment of manipulative content in LLM outputs across seven harm categories: Legal/Power, Psychological, Emotional, Physical, Autonomy, Economic, and Societal Harm. Our framework implements a four-layer analytical pipeline comprising Multi-Granular Detection (MGD), Multi-Scale Intent Analysis (MSIAN), Threat Harmonization Protocol (THP), and Deep Contextual Risk Alignment (DCRA). The dataset contains 401 meticulously curated examples with instruction-response pairs and expert annotations. Through evaluation of state-of-the-art models including GPT-4, Claude 3.5, and LLaMA-3-70B, we observe significant performance disparities (65.2\%--89.7\%) and consistent weaknesses in detecting autonomy-undermining patterns. DarkPatterns-LLM establishes the first standardized, multi-dimensional benchmark for manipulation detection in LLMs, offering actionable diagnostics toward more trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22462v1">Relational Mediators: LLM Chatbots as Boundary Objects in Psychotherapy</a></div>
    <div class="paper-meta">
      📅 2025-12-27
      | 💬 To be published in CSCW 2026, Issue 2 of PACMHCI, 35 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are embedded into mental health technologies, they are often framed either as tools assisting therapists or autonomous therapeutic systems. Such perspectives overlook their potential to mediate relational complexities in therapy, particularly for systemically marginalized clients. Drawing on in-depth interviews with 12 therapists and 12 marginalized clients in China, including LGBTQ+ individuals or those from other marginalized backgrounds, we identify enduring relational challenges: difficulties building trust amid institutional barriers, the burden clients carry in educating therapists about marginalized identities, and challenges sustaining authentic self-disclosure across therapy and daily life. We argue that addressing these challenges requires AI systems capable of actively mediating underlying knowledge gaps, power asymmetries, and contextual disconnects. To this end, we propose the Dynamic Boundary Mediation Framework, which reconceptualizes LLM-enhanced systems as adaptive boundary objects that shift mediating roles across therapeutic stages. The framework delineates three forms of mediation: Epistemic (reducing knowledge asymmetries), Relational (rebalancing power dynamics), and Contextual (bridging therapy-life discontinuities). This framework offers a pathway toward designing relationally accountable AI systems that center the lived realities of marginalized users and more effectively support therapeutic relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.03814v6">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 Accepted to EMNLP 2025 (Oral), Source Code: https://github.com/flowersteam/ce_llms
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12717v2">ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate significant reasoning capabilities, particularly through long chain-of-thought (CoT) processes, which can be elicited by reinforcement learning (RL). However, prolonged CoT reasoning presents limitations, primarily verbose outputs due to excessive introspection. The reasoning process in these LLMs often appears to follow a trial-and-error methodology rather than a systematic, logical deduction. In contrast, tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling reasoning as an exploration within a tree structure. This reasoning structure facilitates the parallel generation and evaluation of multiple reasoning branches, allowing for the active identification, assessment, and pruning of unproductive paths. This process can potentially lead to improved performance and reduced token costs. Building upon the long CoT capability of LLMs, we introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a rule-based reward. ToTRL is designed to guide LLMs in developing the parallel ToT strategy based on the sequential CoT strategy. Furthermore, we employ LLMs as players in a puzzle game during the ToTRL training process. Solving puzzle games inherently necessitates exploring interdependent choices and managing multiple constraints, which requires the construction and exploration of a thought tree, providing challenging tasks for cultivating the ToT reasoning capability. Our empirical evaluations demonstrate that our ToTQwen3-8B model, trained with our ToTRL, achieves significant improvement in performance and reasoning efficiency on complex reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00056v2">MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      The substantial memory demands of pre-training and fine-tuning large language models (LLMs) require memory-efficient optimization algorithms. One promising approach is layer-wise optimization, which treats each transformer block as a single layer and optimizes it sequentially, while freezing the other layers to save optimizer states and activations. Although effective, these methods ignore the varying importance of the modules within each layer, leading to suboptimal performance. Moreover, layer-wise sampling provides only limited memory savings, as at least one full layer must remain active during optimization. To overcome these limitations, we propose Module-wise Importance SAmpling (MISA), a novel method that divides each layer into smaller modules and assigns importance scores to each module. MISA uses a weighted random sampling mechanism to activate modules, provably reducing gradient variance compared to layer-wise sampling. Additionally, we establish an \(\mathcal{O}(1/\sqrt{K})\) convergence rate under non-convex and stochastic conditions, where $K$ is the total number of block updates, and provide a detailed memory analysis showcasing MISA's superiority over existing baseline methods. Experiments on diverse learning tasks validate the effectiveness of MISA. Source code is available at https://github.com/pkumelon/MISA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21933v1">Broken Words, Broken Performance: Effect of Tokenization on Performance of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL 2025)
    </div>
    <details class="paper-abstract">
      Tokenization is the first step in training any Large Language Model (LLM), where the text is split into a sequence of tokens as per the model's fixed vocabulary. This tokenization in LLMs is different from the traditional tokenization in NLP where the text is split into a sequence of "natural" words. In LLMs, a natural word may also be broken into multiple tokens due to limited vocabulary size of the LLMs (e.g., Mistral's tokenizer splits "martial" into "mart" and "ial"). In this paper, we hypothesize that such breaking of natural words negatively impacts LLM performance on various NLP tasks. To quantify this effect, we propose a set of penalty functions that compute a tokenization penalty for a given text for a specific LLM, indicating how "bad" the tokenization is. We establish statistical significance of our hypothesis on multiple NLP tasks for a set of different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21915v1">Exploring the Heterogeneity of Tabular Data: A Diversity-aware Data Generator via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 This manuscript has been submitted to IEEE Transactions on Knowledge and Data Engineering (TKDE) for peer review
    </div>
    <details class="paper-abstract">
      Tabular data generation has become increasingly essential for enabling robust machine learning applications, which require large-scale, high-quality data. Existing solutions leverage generative models to learn original data distributions. However, real-world data are naturally heterogeneous with diverse distributions, making it challenging to obtain a universally good model for diverse data generation. To address this limitation, we introduce Diversity-Aware Tabular data gEnerator (DATE), a framework that (i) prepares high-quality and distributionally distinct examples for in-context learning by effectively partitioning the original heterogeneous data into multiple diverse subsets; (ii) harnesses Large Language Models (LLMs) to explore the diversity of the partitioned distribution with decision tree reasoning as feedback, generating high-quality labeled data for each subset. However, the massive generated data inherently involves a trade-off between diversity and quality. To integrate this issue, existing solutions greedily select the validation-best data. However, we prove that the selection in heterogeneous settings does not possess the greedy-choice property, and design a Multi-Arm Bandit-based sampling algorithm that balances the diversity and quality of generated data. Extensive experiments on tabular classification and regression benchmarks demonstrate that DATE consistently outperforms state-of-the-art GAN-based and LLM-based methods. On average, DATE achieves a 23.75% reduction in error rate with just 100 generated data. Empirically, we demonstrate that data generated by DATE can improve the accuracy of Direct Preference Optimization (DPO) and enhance the reasoning capability of LLMs on the target data. Code is available at https://github.com/windblow32/DATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09990v3">X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Despite the rapid development of safety alignment techniques for LLMs, defending against multi-turn jailbreaks is still a challenging task. In this paper, we conduct a comprehensive comparison, revealing that some existing defense methods can improve the robustness of LLMs against multi-turn jailbreaks but compromise usability, i.e., reducing general capabilities or causing the over-refusal problem. From the perspective of mechanism interpretability of LLMs, we discover that these methods fail to establish a boundary that exactly distinguishes safe and harmful feature representations. Therefore, boundary-safe representations close to harmful representations are inevitably disrupted, leading to a decline in usability. To address this issue, we propose X-Boundary to push harmful representations away from boundary-safe representations and obtain an exact distinction boundary. In this way, harmful representations can be precisely erased without disrupting safe ones. Experimental results show that X-Boundary achieves state-of-the-art defense performance against multi-turn jailbreaks, while reducing the over-refusal rate by about 20% and maintaining nearly complete general capability. Furthermore, we theoretically prove and empirically verify that X-Boundary can accelerate the convergence process during training. Please see our code at: https://github.com/AI45Lab/X-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21902v1">Explainable Statute Prediction via Attention-based Model and LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      In this paper, we explore the problem of automatic statute prediction where for a given case description, a subset of relevant statutes are to be predicted. Here, the term "statute" refers to a section, a sub-section, or an article of any specific Act. Addressing this problem would be useful in several applications such as AI-assistant for lawyers and legal question answering system. For better user acceptance of such Legal AI systems, we believe the predictions should also be accompanied by human understandable explanations. We propose two techniques for addressing this problem of statute prediction with explanations -- (i) AoS (Attention-over-Sentences) which uses attention over sentences in a case description to predict statutes relevant for it and (ii) LLMPrompt which prompts an LLM to predict as well as explain relevance of a certain statute. AoS uses smaller language models, specifically sentence transformers and is trained in a supervised manner whereas LLMPrompt uses larger language models in a zero-shot manner and explores both standard as well as Chain-of-Thought (CoT) prompting techniques. Both these models produce explanations for their predictions in human understandable forms. We compare statute prediction performance of both the proposed techniques with each other as well as with a set of competent baselines, across two popular datasets. Also, we evaluate the quality of the generated explanations through an automated counter-factual manner as well as through human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21877v1">CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Cricket is the second most popular sport globally, commanding a massive following of over 2.5 billion fans globally. Enthusiasts and analysts frequently seek advanced statistical insights, such as long-term historical performance trends or complex player comparisons, that are often unavailable through standard web searches. While Large Language Models (LLMs) have advanced significantly in Text-to-SQL tasks, their capability to handle the domain-specific nuances, complex schema variations, and multilingual requirements inherent to sports analytics remains under-explored. To investigate this potential capability gap, we present CricBench, a comprehensive benchmark suite for evaluating LLMs on specialized cricket data. To curate a "Gold Standard" dataset, we collaborate with domain experts in cricket and SQL to manually author complex queries, ensuring logical correctness. Recognizing linguistic diversity, we construct the benchmark in both English and Hindi, establishing a framework that is open for further extension to other regional languages. We evaluate six state-of-the-art models, including GPT-4o, Claude 3.7 Sonnet, and open-source models, using a strict evaluation protocol. Our results reveal that high performance on general benchmarks does not guarantee success in specialized domains. While the open-weights reasoning model DeepSeek R1 achieves state-of-the-art performance (50.6%), surpassing proprietary giants like Claude 3.7 Sonnet (47.7%) and GPT-4o (33.7%), it still exhibits a significant accuracy drop when moving from general benchmarks (BIRD) to CricBench. Furthermore, we observe that code-mixed Hindi queries frequently yield parity or higher accuracy compared to English, challenging the assumption that English is the optimal prompt language for specialized SQL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.19855v4">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05822v2">Video Event Reasoning and Prediction by Fusing World Knowledge from LLMs with Vision Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Current video understanding models excel at recognizing "what" is happening but fall short in high-level cognitive tasks like causal reasoning and future prediction, a limitation rooted in their lack of commonsense world knowledge. To bridge this cognitive gap, we propose a novel framework that synergistically fuses a powerful Vision Foundation Model (VFM) for deep visual perception with a Large Language Model (LLM) serving as a knowledge-driven reasoning core. Our key technical innovation is a sophisticated fusion module, inspired by the Q-Former architecture, which distills complex spatiotemporal and object-centric visual features into a concise, language-aligned representation. This enables the LLM to effectively ground its inferential processes in direct visual evidence. The model is trained via a two-stage strategy, beginning with large-scale alignment pre-training on video-text data, followed by targeted instruction fine-tuning on a curated dataset designed to elicit advanced reasoning and prediction skills. Extensive experiments demonstrate that our model achieves state-of-the-art performance on multiple challenging benchmarks. Notably, it exhibits remarkable zero-shot generalization to unseen reasoning tasks, and our in-depth ablation studies validate the critical contribution of each architectural component. This work pushes the boundary of machine perception from simple recognition towards genuine cognitive understanding, paving the way for more intelligent and capable AI systems in robotics, human-computer interaction, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21852v1">A Comedy of Estimators: On KL Regularization in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      The reasoning performance of large language models (LLMs) can be substantially improved by training them with reinforcement learning (RL). The RL objective for LLM training involves a regularization term, which is the reverse Kullback-Leibler (KL) divergence between the trained policy and the reference policy. Since computing the KL divergence exactly is intractable, various estimators are used in practice to estimate it from on-policy samples. Despite its wide adoption, including in several open-source libraries, there is no systematic study analyzing the numerous ways of incorporating KL estimators in the objective and their effect on the downstream performance of RL-trained models. Recent works show that prevailing practices for incorporating KL regularization do not provide correct gradients for stated objectives, creating a discrepancy between the objective and its implementation. In this paper, we further analyze these practices and study the gradients of several estimators configurations, revealing how design choices shape gradient bias. We substantiate these findings with empirical observations by RL fine-tuning \texttt{Qwen2.5-7B}, \texttt{Llama-3.1-8B-Instruct} and \texttt{Qwen3-4B-Instruct-2507} with different configurations and evaluating their performance on both in- and out-of-distribution tasks. Through our analysis, we observe that, in on-policy settings: (1) estimator configurations with biased gradients can result in training instabilities; and (2) using estimator configurations resulting in unbiased gradients leads to better performance on in-domain as well as out-of-domain tasks. We also investigate the performance resulting from different KL configurations in off-policy settings and observe that KL regularization can help stabilize off-policy RL training resulting from asynchronous setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21849v1">HeartBench: Probing Core Dimensions of Anthropomorphic Intelligence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have achieved remarkable success in cognitive and reasoning benchmarks, they exhibit a persistent deficit in anthropomorphic intelligence-the capacity to navigate complex social, emotional, and ethical nuances. This gap is particularly acute in the Chinese linguistic and cultural context, where a lack of specialized evaluation frameworks and high-quality socio-emotional data impedes progress. To address these limitations, we present HeartBench, a framework designed to evaluate the integrated emotional, cultural, and ethical dimensions of Chinese LLMs. Grounded in authentic psychological counseling scenarios and developed in collaboration with clinical experts, the benchmark is structured around a theory-driven taxonomy comprising five primary dimensions and 15 secondary capabilities. We implement a case-specific, rubric-based methodology that translates abstract human-like traits into granular, measurable criteria through a ``reasoning-before-scoring'' evaluation protocol. Our assessment of 13 state-of-the-art LLMs indicates a substantial performance ceiling: even leading models achieve only 60% of the expert-defined ideal score. Furthermore, analysis using a difficulty-stratified ``Hard Set'' reveals a significant performance decay in scenarios involving subtle emotional subtexts and complex ethical trade-offs. HeartBench establishes a standardized metric for anthropomorphic AI evaluation and provides a methodological blueprint for constructing high-quality, human-aligned training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04440v3">StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 AAAI 2026 Oral. Extended version with full appendix, 25 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21835v1">LIME:Accelerating Collaborative Lossless LLM Inference on Memory-Constrained Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 15 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a powerful foundation for intelligent reasoning and decision-making, demonstrating substantial impact across a wide range of domains and applications. However, their massive parameter scales and substantial resource demands pose critical challenges for efficient inference on edge devices. These devices are inherently constrained by limited computational power and memory capacity, while bandwidth bottlenecks at the network edge further restrict distributed deployment and real-time responsiveness. Although existing research has explored lightweight optimization techniques to mitigate memory limitations, such approaches often incur significant degradation in model accuracy and performance. To address these challenges, we propose LIME, a collaborative system that enables lossless inference for large models across multiple memory-constrained edge devices under limited network bandwidth. LIME employs an interleaved pipeline parallelism in conjunction with model offloading to dynamically balance computation and communication. Furthermore, a fine-grained offline allocation scheduler and online memory adaptation strategy are introduced to enhance the device's computing and storage resources while minimizing inference latency. Extensive experiments demonstrate that LIME, deployed on four heterogeneous Nvidia Jetson edge devices for LLaMA3.3-70B-Instruct model inference, achieves 1.7$\times$ and 3.7$\times$ speedups over state-of-the-art baselines under sporadic and bursty request patterns respectively, without compromising model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16917v2">Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 V2: Added links to the code-generation results and additional details in the appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21817v1">Method Decoration (DeMe): A Framework for LLM-Driven Adaptive Method Generation in Dynamic IoT Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Intelligent IoT systems increasingly rely on large language models (LLMs) to generate task-execution methods for dynamic environments. However, existing approaches lack the ability to systematically produce new methods when facing previously unseen situations, and they often depend on fixed, device-specific logic that cannot adapt to changing environmental conditions.In this paper, we propose Method Decoration (DeMe), a general framework that modifies the method-generation path of an LLM using explicit decorations derived from hidden goals, accumulated learned methods, and environmental feedback. Unlike traditional rule augmentation, decorations in DeMe are not hardcoded; instead, they are extracted from universal behavioral principles, experience, and observed environmental differences. DeMe enables the agent to reshuffle the structure of its method path-through pre-decoration, post-decoration, intermediate-step modification, and step insertion-thereby producing context-aware, safety-aligned, and environment-adaptive methods. Experimental results show that method decoration allows IoT devices to derive ore appropriate methods when confronting unknown or faulty operating conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22396v1">HalluMat: Detecting Hallucinations in LLM-Generated Materials Science Content Through Multi-Stage Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI), particularly Large Language Models (LLMs), is transforming scientific discovery, enabling rapid knowledge generation and hypothesis formulation. However, a critical challenge is hallucination, where LLMs generate factually incorrect or misleading information, compromising research integrity. To address this, we introduce HalluMatData, a benchmark dataset for evaluating hallucination detection methods, factual consistency, and response robustness in AI-generated materials science content. Alongside this, we propose HalluMatDetector, a multi-stage hallucination detection framework that integrates intrinsic verification, multi-source retrieval, contradiction graph analysis, and metric-based assessment to detect and mitigate LLM hallucinations. Our findings reveal that hallucination levels vary significantly across materials science subdomains, with high-entropy queries exhibiting greater factual inconsistencies. By utilizing HalluMatDetector verification pipeline, we reduce hallucination rates by 30% compared to standard LLM outputs. Furthermore, we introduce the Paraphrased Hallucination Consistency Score (PHCS) to quantify inconsistencies in LLM responses across semantically equivalent queries, offering deeper insights into model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22387v1">AI-Generated Code Is Not Reproducible (Yet): An Empirical Study of Dependency Gaps in LLM-Based Coding Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) as coding agents promises to accelerate software development, but their impact on generated code reproducibility remains largely unexplored. This paper presents an empirical study investigating whether LLM-generated code can be executed successfully in a clean environment with only OS packages and using only the dependencies that the model specifies. We evaluate three state-of-the-art LLM coding agents (Claude Code, OpenAI Codex, and Gemini) across 300 projects generated from 100 standardized prompts in Python, JavaScript, and Java. We introduce a three-layer dependency framework (distinguishing between claimed, working, and runtime dependencies) to quantify execution reproducibility. Our results show that only 68.3% of projects execute out-of-the-box, with substantial variation across languages (Python 89.2%, Java 44.0%). We also find a 13.5 times average expansion from declared to actual runtime dependencies, revealing significant hidden dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22385v1">LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-26
      | 💬 This paper has been accepted for presentation at ABC 2026. The manuscript is under revision prior to camera-ready submission
    </div>
    <details class="paper-abstract">
      In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar weara-ble sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and $k$-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22364v1">Cost-Aware Text-to-SQL: An Empirical Study of Cloud Compute Costs for LLM-Generated Queries</a></div>
    <div class="paper-meta">
      📅 2025-12-26
    </div>
    <details class="paper-abstract">
      Text-to-SQL systems powered by Large Language Models (LLMs) achieve high accuracy on standard benchmarks, yet existing efficiency metrics such as the Valid Efficiency Score (VES) measure execution time rather than the consumption-based costs of cloud data warehouses. This paper presents the first systematic evaluation of cloud compute costs for LLM-generated SQL queries. We evaluate six state-of-the-art LLMs across 180 query executions on Google BigQuery using the StackOverflow dataset (230GB), measuring bytes processed, slot utilization, and estimated cost. Our analysis yields three key findings: (1) reasoning models process 44.5% fewer bytes than standard models while maintaining equivalent correctness (96.7%-100%); (2) execution time correlates weakly with query cost (r=0.16), indicating that speed optimization does not imply cost optimization; and (3) models exhibit up to 3.4x cost variance, with standard models producing outliers exceeding 36GB per query. We identify prevalent inefficiency patterns including missing partition filters and unnecessary full-table scans, and provide deployment guidelines for cost-sensitive enterprise environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.03688v3">A Comparison of DeepSeek and Other LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 30 pages, 7 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Recently, DeepSeek has been the focus of attention in and beyond the AI community. An interesting problem is how DeepSeek compares to other large language models (LLMs). There are many tasks an LLM can do, and in this paper, we use the task of "predicting an outcome using a short text" for comparison. We consider two settings, an authorship classification setting and a citation classification setting. In the first one, the goal is to determine whether a short text is written by human or AI. In the second one, the goal is to classify a citation to one of four types using the textual content. For each experiment, we compare DeepSeek with $4$ popular LLMs: Claude, Gemini, GPT, and Llama. We find that, in terms of classification accuracy, DeepSeek outperforms Gemini, GPT, and Llama in most cases, but underperforms Claude. We also find that DeepSeek is comparably slower than others but with a low cost to use, while Claude is much more expensive than all the others. Finally, we find that in terms of similarity, the output of DeepSeek is most similar to those of Gemini and Claude (and among all $5$ LLMs, Claude and Gemini have the most similar outputs). In this paper, we also present a fully-labeled dataset collected by ourselves, and propose a recipe where we can use the LLMs and a recent data set, MADStat, to generate new data sets. The datasets in our paper can be used as benchmarks for future study on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10436v2">RefactorCoderQA: Benchmarking LLMs for Multi-Domain Coding Question Solutions in Cloud and Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 12 pages, 5 figures, Submitted to IEEE Transactions on Services Computing
    </div>
    <details class="paper-abstract">
      To optimize the reasoning and problem-solving capabilities of Large Language Models (LLMs), we propose a novel cloud-edge collaborative architecture that enables a structured multi-agent prompting framework. This framework comprises three specialized components: GuideLLM, a lightweight model deployed at the edge to provide methodological guidance; SolverLLM, a more powerful model hosted in the cloud and responsible for generating code solutions; and JudgeLLM, an automated evaluator for assessing solution correctness and quality. To evaluate and demonstrate the effectiveness of this architecture in realistic settings, we introduce RefactorCoderQA, a comprehensive benchmark designed to evaluate and enhance the performance of LLMs across multi-domain coding tasks. Motivated by the limitations of existing benchmarks, RefactorCoderQA systematically covers multiple technical domains, including Software Engineering, Data Science, Machine Learning, and Natural Language Processing, using authentic coding challenges sourced from Stack Overflow. We propose RefactorCoder-MoE, a fine-tuned mixture-of-experts (MoE) code language model based on DeepSeek-Coder-7B-Instruct, adapted to the RefactorCoderQA benchmark using QLoRA for domain-specific coding question answering. Extensive experiments demonstrate that RefactorCoder-MoE achieves strong and competitive performance, significantly outperforming all evaluated open-source and commercial baselines, with an overall accuracy of 76.84%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18564v2">Vox Deorum: A Hybrid LLM Architecture for 4X / Grand Strategy Game AI -- Lessons from Civilization V</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models' capacity to reason in natural language makes them uniquely promising for 4X and grand strategy games, enabling more natural human-AI gameplay interactions such as collaboration and negotiation. However, these games present unique challenges due to their complexity and long-horizon nature, while latency and cost factors may hinder LLMs' real-world deployment. Working on a classic 4X strategy game, Sid Meier's Civilization V with the Vox Populi mod, we introduce Vox Deorum, a hybrid LLM+X architecture. Our layered technical design empowers LLMs to handle macro-strategic reasoning, delegating tactical execution to subsystems (e.g., algorithmic AI or reinforcement learning AI in the future). We validate our approach through 2,327 complete games, comparing two open-source LLMs with a simple prompt against Vox Populi's enhanced AI. Results show that LLMs achieve competitive end-to-end gameplay while exhibiting play styles that diverge substantially from algorithmic AI and from each other. Our work establishes a viable architecture for integrating LLMs in commercial 4X games, opening new opportunities for game design and agentic AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20926v2">Uncovering Hierarchical Structure in LLM Embeddings with $δ$-Hyperbolicity, Ultrametricity, and Neighbor Joining</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled significant strides in various fields. This paper introduces a novel approach to evaluate the effectiveness of LLM embeddings in the context of inherent geometric properties. We investigate the structural properties of these embeddings through three complementary metrics $δ$-hyperbolicity, Ultrametricity, and Neighbor Joining. $δ$-hyperbolicity, a measure derived from geometric group theory, quantifies how much a metric space deviates from being a tree-like structure. In contrast, ultrametricity characterizes strictly hierarchical structures where distances obey a strong triangle inequality. While Neighbor Joining quantifies how tree-like the distance relationships are, it does so specifically with respect to the tree reconstructed by the Neighbor Joining algorithm. By analyzing the embeddings generated by LLMs using these metrics, we uncover to what extent the embedding space reflects an underlying hierarchical or tree-like organization. Our findings reveal that LLM embeddings exhibit varying degrees of hyperbolicity and ultrametricity, which correlate with their performance in the underlying machine learning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.05347v2">A Queueing Theoretic Perspective on Low-Latency LLM Inference with Variable Token Length</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) propel the prosperity of interactive AI applications showcased by ChatGPT that demand timely response of inference services. However, LLM inference is computation intensive and memory intensive, and improper parameter configuration at LLM platforms may exacerbate the inference time. In this paper, we analyze the impact of LLM output token distribution on the inference queueing delay, where the max-token clipping and the batched inference are considered. By formulating an M/G/1 model, we observe that enforcing a maximum output token limit on a very small fraction of inference requests can significantly reduce the queueing delay, and our model facilitates the selection of the optimal limit. For the batch inference, we model the service process as a bulk queue in which the batch processing time is affected by the batch size and the maximum token size inside this batch jointly. The queueing delays of the batching of all buffered requests (dynamic batching), the batching of constant number of requests (fixed batching), and the batching without intra-batch waiting (elastic batching) are derived. Experimental results show that our mathematical models coincide with the event-driven simulations well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.20166v3">PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 21 pages, 20 figures, Accepted to 2026 IEEE International Symposium on High-Performance Computer Architecture
    </div>
    <details class="paper-abstract">
      The expansion of long-context Large Language Models (LLMs) creates significant memory system challenges. While Processing-in-Memory (PIM) is a promising accelerator, we identify that it suffers from critical inefficiencies when scaled to long contexts: severe channel underutilization, performance-limiting I/O bottlenecks, and massive memory waste from static KV cache management. In this work, we propose PIMphony, a PIM orchestrator that systematically resolves these issues with three co-designed techniques. First, Token-Centric PIM Partitioning (TCP) ensures high channel utilization regardless of batch size. Second, Dynamic PIM Command Scheduling (DCS) mitigates the I/O bottleneck by overlapping data movement and computation. Finally, a Dynamic PIM Access (DPA) controller enables dynamic memory management to eliminate static memory waste. Implemented via an MLIR-based compiler and evaluated on a cycle-accurate simulator, PIMphony significantly improves throughput for long-context LLM inference (up to 72B parameters and 1M context length). Our evaluations show performance boosts of up to 11.3x on PIM-only systems and 8.4x on xPU+PIM systems, enabling more efficient deployment of LLMs in real-world long-context applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.02404v3">AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 44 pages, 12 figures, 24 tables
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the medical field has garnered significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. To bridge this gap, we introduce AnesSuite, the first comprehensive dataset suite specifically designed for anesthesiology reasoning in LLMs. The suite features AnesBench, an evaluation benchmark tailored to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2). Alongside this benchmark, the suite includes three training datasets that provide an infrastructure for continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning with verifiable rewards (RLVR). Leveraging this suite, we develop Morpheus, the first baseline model collection for anesthesiology reasoning. Despite undergoing limited training with SFT and group relative policy optimization (GRPO), Morpheus demonstrates substantial performance improvements, rivaling the performance of larger-scale models. Furthermore, through comprehensive evaluations and experiments, we analyze the key factors influencing anesthesiology reasoning performance, including model characteristics, training strategies and training data. Both AnesSuite and Morpheus will be open-sourced at https://github.com/MiliLab/AnesSuite.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21635v1">Heaven-Sent or Hell-Bent? Benchmarking the Intelligence and Defectiveness of LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Published as a conference paper at KDD 2026
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) are commonly regarded as errors to be minimized. However, recent perspectives suggest that some hallucinations may encode creative or epistemically valuable content, a dimension that remains underquantified in current literature. Existing hallucination detection methods primarily focus on factual consistency, struggling to handle heterogeneous scientific tasks and balance creativity with accuracy. To address these challenges, we propose HIC-Bench, a novel evaluation framework that categorizes hallucinations into Intelligent Hallucinations (IH) and Defective Hallucinations (DH), enabling systematic investigation of their interplay in LLM creativity. HIC-Bench features three core characteristics: (1) Structured IH/DH Assessment. using a multi-dimensional metric matrix integrating Torrance Tests of Creative Thinking (TTCT) metrics (Originality, Feasibility, Value) with hallucination-specific dimensions (scientific plausibility, factual deviation); (2) Cross-Domain Applicability. spanning ten scientific domains with open-ended innovation tasks; and (3) Dynamic Prompt Optimization. leveraging the Dynamic Hallucination Prompt (DHP) to guide models toward creative and reliable outputs. The evaluation process employs multiple LLM judges, averaging scores to mitigate bias, with human annotators verifying IH/DH classifications. Experimental results reveal a nonlinear relationship between IH and DH, demonstrating that creativity and correctness can be jointly optimized. These insights position IH as a catalyst for creativity and reveal the ability of LLM hallucinations to drive scientific innovation.Additionally, the HIC-Bench offers a valuable platform for advancing research into the creative intelligence of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21595v1">LLM-I2I: Boost Your Small Item2Item Recommendation Model with Large Language Model</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Item-to-Item (I2I) recommendation models are widely used in real-world systems due to their scalability, real-time capabilities, and high recommendation quality. Research to enhance I2I performance focuses on two directions: 1) model-centric approaches, which adopt deeper architectures but risk increased computational costs and deployment complexity, and 2) data-centric methods, which refine training data without altering models, offering cost-effectiveness but struggling with data sparsity and noise. To address these challenges, we propose LLM-I2I, a data-centric framework leveraging Large Language Models (LLMs) to mitigate data quality issues. LLM-I2I includes (1) an LLM-based generator that synthesizes user-item interactions for long-tail items, alleviating data sparsity, and (2) an LLM-based discriminator that filters noisy interactions from real and synthetic data. The refined data is then fused to train I2I models. Evaluated on industry (AEDS) and academic (ARD) datasets, LLM-I2I consistently improves recommendation accuracy, particularly for long-tail items. Deployed on a large-scale cross-border e-commerce platform, it boosts recall number (RN) by 6.02% and gross merchandise value (GMV) by 1.22% over existing I2I models. This work highlights the potential of LLMs in enhancing data-centric recommendation systems without modifying model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21582v1">LLM-Free Image Captioning Evaluation in Reference-Flexible Settings</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Accepted for presentation at AAAI2026
    </div>
    <details class="paper-abstract">
      We focus on the automatic evaluation of image captions in both reference-based and reference-free settings. Existing metrics based on large language models (LLMs) favor their own generations; therefore, the neutrality is in question. Most LLM-free metrics do not suffer from such an issue, whereas they do not always demonstrate high performance. To address these issues, we propose Pearl, an LLM-free supervised metric for image captioning, which is applicable to both reference-based and reference-free settings. We introduce a novel mechanism that learns the representations of image--caption and caption--caption similarities. Furthermore, we construct a human-annotated dataset for image captioning metrics, that comprises approximately 333k human judgments collected from 2,360 annotators across over 75k images. Pearl outperformed other existing LLM-free metrics on the Composite, Flickr8K-Expert, Flickr8K-CF, Nebula, and FOIL datasets in both reference-based and reference-free settings. Our project page is available at https://pearl.kinsta.page/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21580v1">Gamayun's Path to Multilingual Mastery: Cost-Efficient Training of a 1.5B-Parameter LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      We present Gamayun, a 1.5B-parameter multilingual language model trained entirely from scratch on 2.5T tokens. Designed for efficiency and deployment in resource-constrained environments, Gamayun addresses the lack of research on small non-English-centric LLMs by adopting a novel two-stage pre-training strategy: balanced multilingual training for cross-lingual alignment, followed by high-quality English enrichment to transfer performance gains across languages. Our model supports 12 languages, with special focus on Russian. Despite a significantly smaller training budget than comparable models, Gamayun outperforms LLaMA3.2-1B (9T tokens) on all considered benchmarks, and surpasses Qwen2.5-1.5B (18T tokens) on a wide range of English and multilingual tasks. It matches or exceeds Qwen3 (36T tokens) on most tasks outside advanced STEM, achieving state-of-the-art results in Russian, including the MERA benchmark, among the models of comparable size (1-2B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21571v1">nncase: An End-to-End Compiler for Efficient LLM Deployment on Heterogeneous Storage Architectures</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      The efficient deployment of large language models (LLMs) is hindered by memory architecture heterogeneity, where traditional compilers suffer from fragmented workflows and high adaptation costs. We present nncase, an open-source, end-to-end compilation framework designed to unify optimization across diverse targets. Central to nncase is an e-graph-based term rewriting engine that mitigates the phase ordering problem, enabling global exploration of computation and data movement strategies. The framework integrates three key modules: Auto Vectorize for adapting to heterogeneous computing units, Auto Distribution for searching parallel strategies with cost-aware communication optimization, and Auto Schedule for maximizing on-chip cache locality. Furthermore, a buffer-aware Codegen phase ensures efficient kernel instantiation. Evaluations show that nncase outperforms mainstream frameworks like MLC LLM and Intel IPEX on Qwen3 series models and achieves performance comparable to the hand-optimized llama.cpp on CPUs, demonstrating the viability of automated compilation for high-performance LLM deployment. The source code is available at https://github.com/kendryte/nncase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21526v1">Selective LLM-Guided Regularization for Enhancing Recommendation Models</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Large language models provide rich semantic priors and strong reasoning capabilities, making them promising auxiliary signals for recommendation. However, prevailing approaches either deploy LLMs as standalone recommender or apply global knowledge distillation, both of which suffer from inherent drawbacks. Standalone LLM recommender are costly, biased, and unreliable across large regions of the user item space, while global distillation forces the downstream model to imitate LLM predictions even when such guidance is inaccurate. Meanwhile, recent studies show that LLMs excel particularly in re-ranking and challenging scenarios, rather than uniformly across all contexts.We introduce Selective LLM Guided Regularization, a model-agnostic and computation efficient framework that activates LLM based pairwise ranking supervision only when a trainable gating mechanism informing by user history length, item popularity, and model uncertainty predicts the LLM to be reliable. All LLM scoring is performed offline, transferring knowledge without increasing inference cost. Experiments across multiple datasets show that this selective strategy consistently improves overall accuracy and yields substantial gains in cold start and long tail regimes, outperforming global distillation baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21008v2">GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 Accepted by USENIX Security'26
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness. In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v2">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21309v2">A Plan Reuse Mechanism for LLM-Driven Agent</a></div>
    <div class="paper-meta">
      📅 2025-12-25
      | 💬 This paper is an English version of A Plan Reuse Mechanism for LLM-Driven Agent published in 2024 in the Journal of Computer Research and Development
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into personal assistants, like Xiao Ai and Blue Heart V, effectively enhances their ability to interact with humans, solve complex tasks, and manage IoT devices. Such assistants are also termed LLM-driven agents. Upon receiving user requests, the LLM-driven agent generates plans using an LLM, executes these plans through various tools, and then returns the response to the user. During this process, the latency for generating a plan with an LLM can reach tens of seconds, significantly degrading user experience. Real-world dataset analysis shows that about 30% of the requests received by LLM-driven agents are identical or similar, which allows the reuse of previously generated plans to reduce latency. However, it is difficult to accurately define the similarity between the request texts received by the LLM-driven agent through directly evaluating the original request texts. Moreover, the diverse expressions of natural language and the unstructured format of plan texts make implementing plan reuse challenging. To address these issues, we present and implement a plan reuse mechanism for LLM-driven agents called AgentReuse. AgentReuse leverages the similarities and differences among requests' semantics and uses intent classification to evaluate the similarities between requests and enable the reuse of plans. Experimental results based on a real-world dataset demonstrate that AgentReuse achieves a 93% effective plan reuse rate, an F1 score of 0.9718, and an accuracy of 0.9459 in evaluating request similarities, reducing latency by 93.12% compared with baselines without using the reuse mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19920v2">Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21506v1">MotionTeller: Multi-modal Integration of Wearable Time-Series with LLMs for Health and Behavioral Understanding</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      As wearable sensing becomes increasingly pervasive, a key challenge remains: how can we generate natural language summaries from raw physiological signals such as actigraphy - minute-level movement data collected via accelerometers? In this work, we introduce MotionTeller, a generative framework that natively integrates minute-level wearable activity data with large language models (LLMs). MotionTeller combines a pretrained actigraphy encoder with a lightweight projection module that maps behavioral embeddings into the token space of a frozen decoder-only LLM, enabling free-text, autoregressive generation of daily behavioral summaries. We construct a novel dataset of 54383 (actigraphy, text) pairs derived from real-world NHANES recordings, and train the model using cross-entropy loss with supervision only on the language tokens. MotionTeller achieves high semantic fidelity (BERTScore-F1 = 0.924) and lexical accuracy (ROUGE-1 = 0.722), outperforming prompt-based baselines by 7 percent in ROUGE-1. The average training loss converges to 0.38 by epoch 15, indicating stable optimization. Qualitative analysis confirms that MotionTeller captures circadian structure and behavioral transitions, while PCA plots reveal enhanced cluster alignment in embedding space post-training. Together, these results position MotionTeller as a scalable, interpretable system for transforming wearable sensor data into fluent, human-centered descriptions, introducing new pathways for behavioral monitoring, clinical review, and personalized health interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06303v3">Reward Is Enough: LLMs Are In-Context Reinforcement Learners</a></div>
    <div class="paper-meta">
      📅 2025-12-25
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21326v1">Measuring all the noises of LLM Evals</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      Separating signal from noise is central to experimental science. Applying well-established statistical method effectively to LLM evals requires consideration of their unique noise characteristics. We clearly define and measure three types of noise: prediction noise from generating different answers on a given question, data noise from sampling questions, and their combined total noise following the law of total variance. To emphasize relative comparisons and gain statistical power, we propose the all-pairs paired method, which applies the paired analysis to all pairs of LLMs and measures all the noise components based on millions of question-level predictions across many evals and settings. These measurements revealed clear patterns. First, each eval exhibits a characteristic and highly predictable total noise level across all model pairs. Second, paired prediction noise typically exceeds paired data noise, which means reducing prediction noise by averaging can significantly increase statistical power. These findings enable practitioners to assess significance without custom testing and to detect much smaller effects in controlled experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21316v1">Scaling Laws for Economic Productivity: Experimental Evidence in LLM-Assisted Consulting, Data Analyst, and Management Tasks</a></div>
    <div class="paper-meta">
      📅 2025-12-24
    </div>
    <details class="paper-abstract">
      This paper derives `Scaling Laws for Economic Impacts' -- empirical relationships between the training compute of Large Language Models (LLMs) and professional productivity. In a preregistered experiment, over 500 consultants, data analysts, and managers completed professional tasks using one of 13 LLMs. We find that each year of AI model progress reduced task time by 8%, with 56% of gains driven by increased compute and 44% by algorithmic progress. However, productivity gains were significantly larger for non-agentic analytical tasks compared to agentic workflows requiring tool use. These findings suggest continued model scaling could boost U.S. productivity by approximately 20% over the next decade.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21309v1">A Plan Reuse Mechanism for LLM-Driven Agent</a></div>
    <div class="paper-meta">
      📅 2025-12-24
      | 💬 This paper is an English version of A Plan Reuse Mechanism for LLM-Driven Agent published in 2024 in the Journal of Computer Research and Development
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into personal assistants, like Xiao Ai and Blue Heart V, effectively enhances their ability to interact with humans, solve complex tasks, and manage IoT devices. Such assistants are also termed LLM-driven agents. Upon receiving user requests, the LLM-driven agent generates plans using an LLM, executes these plans through various tools, and then returns the response to the user. During this process, the latency for generating a plan with an LLM can reach tens of seconds, significantly degrading user experience. Real-world dataset analysis shows that about 30% of the requests received by LLM-driven agents are identical or similar, which allows the reuse of previously generated plans to reduce latency. However, it is difficult to accurately define the similarity between the request texts received by the LLM-driven agent through directly evaluating the original request texts. Moreover, the diverse expressions of natural language and the unstructured format of plan texts make implementing plan reuse challenging. To address these issues, we present and implement a plan reuse mechanism for LLM-driven agents called AgentReuse. AgentReuse leverages the similarities and differences among requests' semantics and uses intent classification to evaluate the similarities between requests and enable the reuse of plans. Experimental results based on a real-world dataset demonstrate that AgentReuse achieves a 93% effective plan reuse rate, an F1 score of 0.9718, and an accuracy of 0.9459 in evaluating request similarities, reducing latency by 93.12% compared with baselines without using the reuse mechanism.
    </details>
</div>
