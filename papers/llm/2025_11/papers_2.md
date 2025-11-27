# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11255v2">Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted by AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate significant advantages in leveraging structured world knowledge and multi-step reasoning capabilities. However, fundamental challenges arise when transforming LLMs into real-world recommender systems due to semantic and behavioral misalignment. To bridge this gap, we propose Align$^3$GR, a novel framework that unifies token-level, behavior modeling-level, and preference-level alignment. Our approach introduces: Dual tokenization fusing user-item semantic and collaborative signals. Enhanced behavior modeling with bidirectional semantic alignment. Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO) for dynamic preference adaptation. Experiments show Align$^3$GR outperforms the SOTA baseline by +17.8% in Recall@10 and +20.2% in NDCG@10 on the public dataset, with significant gains in online A/B tests and full-scale deployment on an industrial large-scale recommendation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19023v1">OrdMoE: Preference Alignment via Hierarchical Expert Group Ranking in Multimodal Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Preference learning has recently emerged as a pivotal strategy for post-training alignment of Multimodal Large Language Models (MLLMs). However, existing approaches predominantly rely on external human-annotated preference data, which is costly and labor-intensive to collect. In this work, we propose OrdMoE, a novel preference alignment framework that bypasses the reliance on external human preferences entirely by leveraging intrinsic signals within Mixture-of-Experts (MoE) architectures. Specifically, we observe that the router's expert selection scores implicitly encode a quality-aware ranking of responses (i.e. higher-scoring experts consistently generate higher-quality outputs). Building on this insight, OrdMoE constructs an internal preference hierarchy by grouping experts into ranked tiers based on their per-token routing scores and activating each tier separately to produce a sequence of responses with increasing quality. This yields a zero-cost, self-supervised preference ordering over generated responses, which can be directly optimized using standard preference learning objectives. Extensive experiments across multiple multimodal benchmarks demnstrate that OrdMoE significantly enhances both alignment and overall performance of multimodal Mixture-of-Experts LLMs, achieving competitive results without requiring any human-annotated preference data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06852v4">Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 AAAI-26-AIA
    </div>
    <details class="paper-abstract">
      Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18985v1">LLM Chatbots in High School Programming: Exploring Behaviors and Interventions</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      This study uses a Design-Based Research (DBR) cycle to refine the integration of Large Language Models (LLMs) in high school programming education. The initial problem was identified in an Intervention Group where, in an unguided setting, a higher proportion of executive, solution-seeking queries correlated strongly and negatively with exam performance. A contemporaneous Comparison Group demonstrated that without guidance, these unproductive help-seeking patterns do not self-correct, with engagement fluctuating and eventually declining. This insight prompted a mid-course pedagogical intervention in the first group, designed to teach instrumental help-seeking. The subsequent evaluation confirmed the intervention's success, revealing a decrease in executive queries, as well as a shift toward more productive learning workflows. However, this behavioral change did not translate into a statistically significant improvement in exam grades, suggesting that altering tool-use strategies alone may be insufficient to overcome foundational knowledge gaps. The DBR process thus yields a more nuanced principle: the educational value of an LLM depends on a pedagogy that scaffolds help-seeking, but this is only one part of the complex process of learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18977v1">FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 5 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Pruning is an effective method for compressing Large Language Models, but finding an optimal, non-uniform layer-wise sparsity allocation remains a key challenge. While heuristic methods are fast but yield suboptimal performance, more powerful search-based approaches like Reinforcement Learning are often hindered by prohibitive computational costs on large-scale models. To overcome this efficiency barrier, we propose FastForward Pruning. Its core is a decoupled, single-step RL framework that separates policy optimization from the complex budget satisfaction problem. Such a decoupling is crucial for efficiently searching the vast policy space of LLMs. This curriculum-based strategy begins with low-cost, simple tasks and gradually increases in complexity, significantly reducing the search's computational overhead. Evaluated on the LLaMA, Mistral, and OPT model families, our framework discovers pruning policies that achieve superior performance over strong heuristic baselines. Crucially, when compared to other search-based algorithms, our method achieves competitive or superior results at a fraction of the computational cost, demonstrating a clear advantage in search efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18966v1">LLM-CSEC: Empirical Evaluation of Security in C/C++ Code Generated by Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      The security of code generated by large language models (LLMs) is a significant concern, as studies indicate that such code often contains vulnerabilities and lacks essential defensive programming constructs. This work focuses on examining and evaluating the security of LLM-generated code, particularly in the context of C/C++. We categorized known vulnerabilities using the Common Weakness Enumeration (CWE) and, to study their criticality, mapped them to CVEs. We used ten different LLMs for code generation and analyzed the outputs through static analysis. The amount of CWEs present in AI-generated code is concerning. Our findings highlight the need for developers to be cautious when using LLM-generated code. This study provides valuable insights to advance automated code generation and encourage further research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v21">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18924v1">LLM-Driven Kernel Evolution: Automating Driver Updates in Linux</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Linux kernel evolution breaks drivers through API/ABI changes, semantic shifts, and security-hardening updates. We introduce DRIVEBENCH, an executable corpus of kernel$\rightarrow$driver co-evolution cases, and AUTODRIVER, a closed-loop, LLM-driven system for automating driver maintenance. The system integrates prompt engineering, multi-agent collaboration, static analysis, and iterative validation to ensure that generated patches are not only syntactically correct but also functionally and semantically consistent with kernel conventions. The corpus spans v5.10-v6.10 with 235 validated cases drawn from 612 candidates. In evaluation across 55 cases, AUTODRIVER achieves 56.4% compilation success; QEMU-based boot verification indicates that compiled patches preserve driver initialization in most instances. By releasing DRIVEBENCH and tooling, we enable reproducible research and a practical route to continuous, safe co-evolution of drivers with the Linux kernel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18903v1">How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Due to the scarcity of high-quality data, large language models (LLMs) are often trained on mixtures of data with varying quality levels, even after sophisticated data curation. A natural approach to better leverage high-quality data is curriculum-based pretraining, where the model is trained on data sorted in ascending order of quality as determined by a quality metric. However, prior studies have reported limited improvements from such curriculum-based pretraining strategies. This work identifies a critical factor constraining these methods: the incompatibility between the ascending data quality order and the decaying learning rate (LR) schedule. We find that while curriculum-based training substantially outperforms random shuffling when using a constant LR, its advantage diminishes under standard LR decay schedules. Our experiments show this incompatibility can be mitigated by two simple strategies: (1) employing a more moderate LR decay schedule, where the final LR is only moderately smaller than the peak LR, and (2) replacing LR decay with model averaging, i.e., computing a weighted average of the final few checkpoints. By combining these strategies, we improve the average score on a suite of standard benchmarks by 1.64% over random shuffling, without additional data refinement. Validated on 1.5B-parameter models trained over 30B tokens with various data-quality metrics, our findings call for a re-evaluation of curriculum-based LLM pretraining and underscore the potential of co-designing data curricula with optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18889v1">CoreEval: Automatically Building Contamination-Resilient Datasets with Real-World Knowledge toward Reliable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 ACL'25
    </div>
    <details class="paper-abstract">
      Data contamination poses a significant challenge to the fairness of LLM evaluations in natural language processing tasks by inadvertently exposing models to test data during training. Current studies attempt to mitigate this issue by modifying existing datasets or generating new ones from freshly collected information. However, these methods fall short of ensuring contamination-resilient evaluation, as they fail to fully eliminate pre-existing knowledge from models or preserve the semantic complexity of the original datasets. To address these limitations, we propose \textbf{CoreEval}, a \textbf{Co}ntamination-\textbf{re}silient \textbf{Eval}uation strategy for automatically updating data with real-world knowledge. This approach begins by extracting entity relationships from the original data and leveraging the GDELT database to retrieve relevant, up-to-date knowledge. The retrieved knowledge is then recontextualized and integrated with the original data, which is refined and restructured to ensure semantic coherence and enhanced task relevance. Ultimately, a robust data reflection mechanism is employed to iteratively verify and refine labels, ensuring consistency between the updated and original datasets. Extensive experiments on updated datasets validate the robustness of CoreEval, demonstrating its effectiveness in mitigating performance overestimation caused by data contamination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18868v1">KernelBand: Boosting LLM-based Kernel Optimization with a Hierarchical and Hardware-aware Multi-armed Bandit</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      High quality kernels are critical for reducing training and inference costs of Large Language Models (LLMs), yet they traditionally require significant expertise in hardware architecture and software optimization. While recent advances in LLM-based code generation show promise for complex optimization, existing methods struggle with the vast optimization space due to insufficient hardware domain knowledge, failing to effectively balance exploration and exploitation. We present KernelBand, a novel framework that formulates kernel optimization as a hierarchical multi-armed bandit problem, enabling LLM agents to strategically navigate the optimization space by treating kernel selection and optimization strategy application as sequential decision-making processes. Our approach leverages hardware profiling information to identify promising optimization strategies and employs runtime behavior clustering to reduce exploration overhead across kernel candidates. Extensive experiments on TritonBench demonstrate that KernelBand significantly outperforms state-of-the-art methods, achieving superior performance with fewer tokens while exhibiting consistent improvement without saturation as computational resources increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18854v1">Time Travel: LLM-Assisted Semantic Behavior Localization with Git Bisect</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 submitted to Git Bisect SCALCOM 2025 Calgary (to be published)
    </div>
    <details class="paper-abstract">
      We present a novel framework that integrates Large Language Models (LLMs) into the Git bisect process for semantic fault localization. Traditional bisect assumes deterministic predicates and binary failure states assumptions often violated in modern software development due to flaky tests, nonmonotonic regressions, and semantic divergence from upstream repositories. Our system augments bisect traversal with structured chain of thought reasoning, enabling commit by commit analysis under noisy conditions. We evaluate multiple open source and proprietary LLMs for their suitability and fine tune DeepSeekCoderV2 using QLoRA on a curated dataset of semantically labeled diffs. We adopt a weak supervision workflow to reduce annotation overhead, incorporating human in the loop corrections and self consistency filtering. Experiments across multiple open source projects show a 6.4 point absolute gain in success rate from 74.2 to 80.6 percent, leading to significantly fewer failed traversals and by experiment up to 2x reduction in average bisect time. We conclude with discussions on temporal reasoning, prompt design, and finetuning strategies tailored for commit level behavior analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18850v1">Cognitive Alpha Mining via LLM-Driven Code-Based Evolution</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Discovering effective predictive signals, or ``alphas,'' from financial data with high dimensionality and extremely low signal-to-noise ratio remains a difficult open problem. Despite progress in deep learning, genetic programming, and, more recently, large language model (LLM)--based factor generation, existing approaches still explore only a narrow region of the vast alpha search space. Neural models tend to produce opaque and fragile patterns, while symbolic or formula-based methods often yield redundant or economically ungrounded expressions that generalize poorly. Although different in form, these paradigms share a key limitation: none can conduct broad, structured, and human-like exploration that balances logical consistency with creative leaps. To address this gap, we introduce the Cognitive Alpha Mining Framework (CogAlpha), which combines code-level alpha representation with LLM-driven reasoning and evolutionary search. Treating LLMs as adaptive cognitive agents, our framework iteratively refines, mutates, and recombines alpha candidates through multi-stage prompts and financial feedback. This synergistic design enables deeper thinking, richer structural diversity, and economically interpretable alpha discovery, while greatly expanding the effective search space. Experiments on A-share equities demonstrate that CogAlpha consistently discovers alphas with superior predictive accuracy, robustness, and generalization over existing methods. Our results highlight the promise of aligning evolutionary optimization with LLM-based reasoning for automated and explainable alpha discovery. All source code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08426v8">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to IEEE TKDE2025
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from users' natural language questions (text-to-SQL) remains a long-standing challenge due to the complexities involved in user question understanding, database schema comprehension, and SQL generation. Traditional text-to-SQL systems, which combine human engineering and deep neural networks, have made significant progress. Subsequently, pre-trained language models (PLMs) have been developed for text-to-SQL tasks, achieving promising results. However, as modern databases and user questions grow more complex, PLMs with a limited parameter size often produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which restricts the application of PLM-based systems. Recently, large language models (LLMs) have shown significant capabilities in natural language understanding as model scale increases. Thus, integrating LLM-based solutions can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we provide a comprehensive review of existing LLM-based text-to-SQL studies. Specifically, we offer a brief overview of the technical challenges and evolutionary process of text-to-SQL. Next, we introduce the datasets and metrics designed to evaluate text-to-SQL systems. Subsequently, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we make a summarization and discuss the remaining challenges in this field and suggest expectations for future research directions. All the related resources of LLM-based, including research papers, benchmarks, and open-source projects, are collected for the community in our repository: https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18849v1">Pre-Filtering Code Suggestions using Developer Behavioral Telemetry to Optimize LLM-Assisted Programming</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into code editors to provide AI-powered code suggestions. Yet many of these suggestions are ignored, resulting in wasted computation, increased latency, and unnecessary interruptions. We introduce a lightweight pre-filtering model that predicts the likelihood of suggestion acceptance before invoking the LLM, using only real-time developer telemetry such as typing speed, file navigation, and editing activity. Deployed in a production-grade Visual Studio Code plugin over four months of naturalistic use, our approach nearly doubled acceptance rates (18.4% -> 34.2%) while suppressing 35% of low-value LLM calls. These findings demonstrate that behavioral signals alone can meaningfully improve both user experience and system efficiency in LLM-assisted programming, highlighting the value of timing-aware, privacy-preserving adaptation mechanisms. The filter operates solely on pre-invocation editor telemetry and never inspects code or prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18842v1">Optimizing LLM Code Suggestions: Feedback-Driven Timing with Lightweight State Bounds</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed code auto-completion by generating context-aware suggestions. Yet, deciding when to present these suggestions remains underexplored, often leading to interruptions or wasted inference calls. We propose an adaptive timing mechanism that dynamically adjusts the delay before offering a suggestion based on real-time developer feedback. Our suggested method combines a logistic transform of recent acceptance rates with a bounded delay range, anchored by a high-level binary prediction of the developer's cognitive state. In a two-month deployment with professional developers, our system improved suggestion acceptance from 4.9% with no delay to 15.4% with static delays, and to 18.6% with adaptive timing-while reducing blind rejections (rejections without being read) from 8.3% to 0.36%. Together, these improvements increase acceptance and substantially reduce wasted inference calls by 75%, making LLM-based code assistants more efficient and cost-effective in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.18436v5">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06447v2">SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Long-context inference for Large Language Models (LLMs) is heavily limited by high computational demands. While several existing methods optimize attention computation, they still process the full set of hidden states at each layer, limiting overall efficiency. In this work, we propose SlimInfer, an innovative framework that aims to accelerate inference by directly pruning less critical prompt tokens during the forward pass. Our key insight is an information diffusion phenomenon: As information from critical tokens propagates through layers, it becomes distributed across the entire sequence. This diffusion process suggests that LLMs can maintain their semantic integrity when excessive tokens, even including these critical ones, are pruned in hidden states. Motivated by this, SlimInfer introduces a dynamic fine-grained pruning mechanism that accurately removes redundant tokens of hidden state at intermediate layers. This layer-wise pruning naturally enables an asynchronous KV cache manager that prefetches required token blocks without complex predictors, reducing both memory usage and I/O costs. Extensive experiments show that SlimInfer can achieve up to $\mathbf{2.53\times}$ time-to-first-token (TTFT) speedup and $\mathbf{1.88\times}$ end-to-end latency reduction for LLaMA3.1-8B-Instruct on a single RTX 4090, without sacrificing performance on LongBench. Our code is available at https://github.com/Longxmas/SlimInfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.13837v5">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 31 pages, 27 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18782v1">Summary-Mediated Repair: Can LLMs use code summarisation as a tool for program repair?</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 6 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce code with subtle implementation-level bugs despite strong benchmark performance. These errors are hard for LLMs to spot and can have large behavioural effects; yet when asked to summarise code, LLMs can frequently surface high-level intent and sometimes overlook this low-level noise. Motivated by this, we propose summary-mediated repair, a prompt-only pipeline for program repair that leverages natural-language code summarisation as an explicit intermediate step, extending previous work that has already shown code summarisation to be a useful intermediary for downstream tasks. We evaluate our method across eight production-grade LLMs on two function level benchmarks (HumanEvalPack and MBPP), comparing several summary styles against a direct repair baseline. Error-aware diagnostic summaries consistently yield the largest gains - repairing up to 65% of unseen errors, on average of 5% more than the baseline - though overall improvements are modest and LLM-dependent. Our results position summaries as a cheap, human-interpretable diagnostic artefact that can be integrated into program-repair pipelines rather than a stand-alone fix-all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26213v2">OmniDocLayout: Towards Diverse Document Layout Generation via Coarse-to-Fine LLM Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 TL;DR: With the proposed OmniDocLayout-1M dataset and the LLM-based coarse-to-fine learning strategy, we enable diverse and complex document layout generation that achieves both strong condition consistency and adherence to fundamental aesthetic principles
    </div>
    <details class="paper-abstract">
      Document AI has advanced rapidly and is attracting increasing attention. Yet, while most efforts have focused on document layout analysis (DLA), its generative counterpart, layout generation, remains underexplored. Distinct from traditional graphic layout design and room layout planning, document layout generation typically involves a larger number of elements per page and exhibits greater structural diversity and complexity. Currently, a major obstacle lies in the scarcity of diverse document layouts: academic papers with Manhattan-style structures dominate existing studies, while open-world genres such as newspapers and magazines remain severely underrepresented. To address this gap, we curate OmniDocLayout-1M, the first million-scale dataset of diverse document layouts, covering six common document types and comprising contemporary layouts collected from multiple sources. Moreover, since existing methods struggle in complex domains and often fail to arrange long sequences coherently, we introduce OmniDocLayout-LLM, a 0.5B model with designed two-stage Coarse-to-Fine learning paradigm:1) learning universal layout principles from our dataset with coarse category definitions, and 2) transferring the knowledge to a specific domain with few fine-grained annotated samples. Extensive experiments demonstrate that our approach achieves strong performance on multiple domains in M$^6$Doc dataset, substantially surpassing both existing layout generation experts and several latest general-purpose LLMs. Our code, dataset, and models will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18760v1">HERMES: Towards Efficient and Verifiable Mathematical Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Informal mathematics has been central to modern large language model (LLM) reasoning, offering flexibility and enabling efficient construction of arguments. However, purely informal reasoning is prone to logical gaps and subtle errors that are difficult to detect and correct. In contrast, formal theorem proving provides rigorous, verifiable mathematical reasoning, where each inference step is checked by a trusted compiler in systems such as Lean, but lacks the exploratory freedom of informal problem solving. This mismatch leaves current LLM-based math agents without a principled way to combine the strengths of both paradigms. In this work, we introduce Hermes, the first tool-assisted agent that explicitly interleaves informal reasoning with formally verified proof steps in Lean. The framework performs intermediate formal checking to prevent reasoning drift and employs a memory module that maintains proof continuity across long, multi-step reasoning chains, enabling both exploration and verification within a single workflow. We evaluate Hermes on four challenging mathematical reasoning benchmarks using LLMs of varying parameter scales, from small models to state-of-the-art systems. Across all settings, Hermes reliably improves the reasoning accuracy of base models while substantially reducing token usage and computational cost compared to reward-based approaches. On difficult datasets such as AIME'25, Hermes achieves up to a 67% accuracy improvement while using 80% fewer total inference FLOPs. The implementation and codebase are publicly available at https://github.com/aziksh-ospanov/HERMES.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15289v5">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 ACL Findings 2025. Welcome to employ SATA as a baseline
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18727v1">LogSyn: A Few-Shot LLM Framework for Structured Insight Extraction from Unstructured General Aviation Maintenance Logs</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted in Proceedings of the 3rd INCOM 2026
    </div>
    <details class="paper-abstract">
      Aircraft maintenance logs hold valuable safety data but remain underused due to their unstructured text format. This paper introduces LogSyn, a framework that uses Large Language Models (LLMs) to convert these logs into structured, machine-readable data. Using few-shot in-context learning on 6,169 records, LogSyn performs Controlled Abstraction Generation (CAG) to summarize problem-resolution narratives and classify events within a detailed hierarchical ontology. The framework identifies key failure patterns, offering a scalable method for semantic structuring and actionable insight extraction from maintenance logs. This work provides a practical path to improve maintenance workflows and predictive analytics in aviation and related industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.16216v4">GMoE: Empowering LLMs Fine-Tuning via MoE Graph Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 9 pages, 25 figures
    </div>
    <details class="paper-abstract">
      The sparse Mixture-of-Experts (MoE) architecture of large language models (LLMs) confronts an inherent issue of load imbalance arising from the simplistic linear router strategy, which ultimately causes the instability and inefficient learning of LLMs. To address this challenge, we introduce a novel MoE graph-based framework $\textbf{GMoE}$, aimed at enhancing the collaboration among multiple experts. In GMoE, a graph router function is designed to capture the collaboration signals among experts. This enables all experts to dynamically allocate information derived from input data by sharing information with their neighboring experts. Moreover, we put forward two coordination strategies in GMoE: the $\textit{Poisson distribution-based distinction strategy}$ and the $\textit{Normal distribution-based balance strategy}$, to further release the capacity of each expert and increase the model stability in the fine-tuning of LLMs. Specifically, we leverage a parameter-efficient fine-tuning technique, i.e., Low-Rank Adaptation (LoRA), to implement the graph MoE architecture. Extensive experiments on four real-world benchmark datasets demonstrate the effectiveness of GMoE, showing the benefits of facilitating collaborations of multiple experts in LLM fine-tuning. The code of experimental implementation is available at https://github.com/BAI-LAB/GMoE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18305v3">Evolving Triple Knowledge-Augmented LLMs for Code Translation in Repository Context</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have behaved well in function-level code translation without repository-level context. However, the performance of LLMs in repository-level context code translation remains suboptimal due to complex dependencies and context, hindering their adoption in industrial settings. In this work, we propose a novel LLM-based code translation technique K-Trans, which leverages triple knowledge augmentation to enhance LLM's translation quality under repository context in real-world software development. First, K-Trans constructs a evolving translation knowledge base by extracting relevant information from target-language codebases, the repository being translated, and prior translation results. Second, for each function to be translated, K-Trans retrieves relevant triple knowledge, including target-language code samples, dependency usage examples, and successful translation function pairs, serving as references to enhance LLM for translation. Third, K-Trans constructs a knowledge-augmented translation prompt using the retrieved triple knowledge and employs LLMs to generate the translated code while preserving repository context. It further leverages LLMs for self-debugging, enhancing translation correctness. Lastly, K-Trans continuously evolves the translation knowledge base. The experiments show that K-Trans substantially outperforms the baseline adapted from previous work by 19.4%/40.2% relative improvement in pass@1 and 0.138 in CodeBLEU. It is important to note that the results also demonstrate that each knowledge significantly contributes to K-Trans's effectiveness in handling repository-level context code translation, with dependency usage examples making the most notable contribution. Moreover, as the self-evolution process progresses, the knowledge base continuously enhances the LLM's performance across various aspects of the repository-level code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09598v6">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      This paper introduces an infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models in commercial datacenters. The framework combines public API performance data with company-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost and provide a dynamically updated dashboard that visualizes model-level energy, water, and carbon metrics. Results show the most energy-intensive models exceed 29 Wh per long prompt, over 65 times the most efficient systems. Even a 0.42 Wh short query, when scaled to 700M queries/day, aggregates to annual electricity comparable to 35{,}000 U.S. homes, evaporative freshwater equal to the annual drinking needs of 1.2M people, and carbon emissions requiring a Chicago-sized forest to offset. These findings highlight a growing paradox: as AI becomes cheaper and faster, global adoption drives disproportionate resource consumption. Our methodology offers a standardized, empirically grounded basis for sustainability benchmarking and accountability in AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07061v3">Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets -- IEMOCAP and MELD -- show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12109v3">Personalized LLM Decoding via Contrasting Personal Preference</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02025v2">SAFE: Harnessing LLM for Scenario-Driven ADS Testing from Multimodal Crash Data</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 The paper has been accepted for publication in the proceedings of the IEEE/ACM 48th International Conference on Software Engineering to be held 12-18 April 2026 (ICSE2026)
    </div>
    <details class="paper-abstract">
      Ensuring the safety of Autonomous Driving Systems (ADS) requires realistic and reproducible test scenarios, yet extracting such scenarios from multimodal crash reports remains a major challenge. Large Language Models (LLMs) often hallucinate and lose map structure, resulting in unrealistic road layouts and vehicle behaviors. To address this, we introduce SAFE, a novel Scenario-based ADS testing Framework via multimodal Extraction, which leverages Retrieval-Augmented Generation (RAG), knowledge-grounded prompting, Chain-of-Thought (CoT) reasoning, and self-validation to improve scenario reconstruction from multimodal crash data. SAFE achieves 93.8% accuracy in extracting road network details, 80.0% for actor information, and 100% for environmental context. In human studies, SAFE outperforms LCTGen and AC3R in reconstructing consistent road networks and vehicle behaviors. Under identical ADS and simulator settings, SAFE detects 39 and 71 more safety violations than LCTGen and AC3R, respectively, and reproduces 12 more real-world crash cases than LCTGen. On 19 cases supported by AC3R, SAFE reproduces one additional crash case with statistically significant gains across five runs. It generates scenarios within 25 seconds and triggers violations after just 1 case (IDM) and 3 cases (PPO) in MetaDrive, as well as 1 case (Auto) in BeamNG. Code: https://github.com/Siwei-Luo-MQ/SAFE-ADS-Testing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.14909v3">Mixture of Attention Spans: Optimizing LLM Inference Efficiency with Heterogeneous Sliding-Window Lengths</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Published at CoLM'25
    </div>
    <details class="paper-abstract">
      Sliding-window attention offers a hardware-efficient solution to the memory and throughput challenges of Large Language Models (LLMs) in long-context scenarios. Existing methods typically employ a single window length across all attention heads and input sizes. However, this uniform approach fails to capture the heterogeneous attention patterns inherent in LLMs, ignoring their distinct accuracy-latency trade-offs. To address this challenge, we propose *Mixture of Attention Spans* (MoA), which automatically tailors distinct sliding-window length configurations to different heads and layers. MoA constructs and navigates a search space of various window lengths and their scaling rules relative to input sizes. It profiles the model, evaluates potential configurations, and pinpoints the optimal length configurations for each head. MoA adapts to varying input sizes, revealing that some attention heads expand their focus to accommodate longer inputs, while other heads consistently concentrate on fixed-length local contexts. Experiments show that MoA increases the effective context length by 3.9x with the same average sliding-window length, boosting retrieval accuracy by 1.5-7.1x over the uniform-window baseline across Vicuna-{7B, 13B} and Llama3-{8B, 70B} models. Moreover, MoA narrows the performance gap with full attention, reducing the maximum relative performance drop from 9%-36% to within 5% across three long-context understanding benchmarks. MoA achieves a 1.2-1.4x GPU memory reduction, boosting decode throughput by 6.6-8.2x and 1.7-1.9x over FlashAttention2 and vLLM, with minimal performance impact. Our code is available at: https://github.com/thu-nics/MoA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19719v1">Can LLMs Faithfully Explain Themselves in Low-Resource Languages? A Case Study on Emotion Detection in Persian</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate self-explanations alongside their predictions, a practice that raises concerns about the faithfulness of these explanations, especially in low-resource languages. This study evaluates the faithfulness of LLM-generated explanations in the context of emotion classification in Persian, a low-resource language, by comparing the influential words identified by the model against those identified by human annotators. We assess faithfulness using confidence scores derived from token-level log-probabilities. Two prompting strategies, differing in the order of explanation and prediction (Predict-then-Explain and Explain-then-Predict), are tested for their impact on explanation faithfulness. Our results reveal that while LLMs achieve strong classification performance, their generated explanations often diverge from faithful reasoning, showing greater agreement with each other than with human judgments. These results highlight the limitations of current explanation methods and metrics, emphasizing the need for more robust approaches to ensure LLM reliability in multilingual and low-resource contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v5">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at https://github.com/OpenMLRL/CoMLRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19654v1">Accuracy and Efficiency Trade-Offs in LLM-Based Malware Detection and Explanation: A Comparative Study of Parameter Tuning vs. Full Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 Accepted in IEEE Big Data 2025
    </div>
    <details class="paper-abstract">
      This study examines whether Low-Rank Adaptation (LoRA) fine-tuned Large Language Models (LLMs) can approximate the performance of fully fine-tuned models in generating human-interpretable decisions and explanations for malware classification. Achieving trustworthy malware detection, particularly when LLMs are involved, remains a significant challenge. We developed an evaluation framework using Bilingual Evaluation Understudy (BLEU), Recall-Oriented Understudy for Gisting Evaluation (ROUGE), and Semantic Similarity Metrics to benchmark explanation quality across five LoRA configurations and a fully fine-tuned baseline. Results indicate that full fine-tuning achieves the highest overall scores, with BLEU and ROUGE improvements of up to 10% over LoRA variants. However, mid-range LoRA models deliver competitive performance exceeding full fine-tuning on two metrics while reducing model size by approximately 81% and training time by over 80% on a LoRA model with 15.5% trainable parameters. These findings demonstrate that LoRA offers a practical balance of interpretability and resource efficiency, enabling deployment in resource-constrained environments without sacrificing explanation quality. By providing feature-driven natural language explanations for malware classifications, this approach enhances transparency, analyst confidence, and operational scalability in malware detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19648v1">Efficient Multi-Hop Question Answering over Knowledge Graphs via LLM Planning and Embedding-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Multi-hop question answering over knowledge graphs remains computationally challenging due to the combinatorial explosion of possible reasoning paths. Recent approaches rely on expensive Large Language Model (LLM) inference for both entity linking and path ranking, limiting their practical deployment. Additionally, LLM-generated answers often lack verifiable grounding in structured knowledge. We present two complementary hybrid algorithms that address both efficiency and verifiability: (1) LLM-Guided Planning that uses a single LLM call to predict relation sequences executed via breadth-first search, achieving near-perfect accuracy (micro-F1 > 0.90) while ensuring all answers are grounded in the knowledge graph, and (2) Embedding-Guided Neural Search that eliminates LLM calls entirely by fusing text and graph embeddings through a lightweight 6.7M-parameter edge scorer, achieving over 100 times speedup with competitive accuracy. Through knowledge distillation, we compress planning capability into a 4B-parameter model that matches large-model performance at zero API cost. Evaluation on MetaQA demonstrates that grounded reasoning consistently outperforms ungrounded generation, with structured planning proving more transferable than direct answer generation. Our results show that verifiable multi-hop reasoning does not require massive models at inference time, but rather the right architectural inductive biases combining symbolic structure with learned representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19550v1">The Semiotic Channel Principle: Measuring the Capacity for Meaning in LLM Communication</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      This paper proposes a novel semiotic framework for analyzing Large Language Models (LLMs), conceptualizing them as stochastic semiotic engines whose outputs demand active, asymmetric human interpretation. We formalize the trade-off between expressive richness (semiotic breadth) and interpretive stability (decipherability) using information-theoretic tools. Breadth is quantified as source entropy, and decipherability as the mutual information between messages and human interpretations. We introduce a generative complexity parameter (lambda) that governs this trade-off, as both breadth and decipherability are functions of lambda. The core trade-off is modeled as an emergent property of their distinct responses to $λ$. We define a semiotic channel, parameterized by audience and context, and posit a capacity constraint on meaning transmission, operationally defined as the maximum decipherability by optimizing lambda. This reframing shifts analysis from opaque model internals to observable textual artifacts, enabling empirical measurement of breadth and decipherability. We demonstrate the framework's utility across four key applications: (i) model profiling; (ii) optimizing prompt/context design; (iii) risk analysis based on ambiguity; and (iv) adaptive semiotic systems. We conclude that this capacity-based semiotic approach offers a rigorous, actionable toolkit for understanding, evaluating, and designing LLM-mediated communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19537v1">Cross-Domain Generalization of Multimodal LLMs for Global Photovoltaic Assessment</a></div>
    <div class="paper-meta">
      📅 2025-11-24
      | 💬 5 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The rapid expansion of distributed photovoltaic (PV) systems poses challenges for power grid management, as many installations remain undocumented. While satellite imagery provides global coverage, traditional computer vision (CV) models such as CNNs and U-Nets require extensive labeled data and fail to generalize across regions. This study investigates the cross-domain generalization of a multimodal large language model (LLM) for global PV assessment. By leveraging structured prompts and fine-tuning, the model integrates detection, localization, and quantification within a unified schema. Cross-regional evaluation using the $Δ$F1 metric demonstrates that the proposed model achieves the smallest performance degradation across unseen regions, outperforming conventional CV and transformer baselines. These results highlight the robustness of multimodal LLMs under domain shift and their potential for scalable, transferable, and interpretable global PV mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19536v1">AttackPilot: Autonomous Inference Attacks Against ML Services With LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Inference attacks have been widely studied and offer a systematic risk assessment of ML services; however, their implementation and the attack parameters for optimal estimation are challenging for non-experts. The emergence of advanced large language models presents a promising yet largely unexplored opportunity to develop autonomous agents as inference attack experts, helping address this challenge. In this paper, we propose AttackPilot, an autonomous agent capable of independently conducting inference attacks without human intervention. We evaluate it on 20 target services. The evaluation shows that our agent, using GPT-4o, achieves a 100.0% task completion rate and near-expert attack performance, with an average token cost of only $0.627 per run. The agent can also be powered by many other representative LLMs and can adaptively optimize its strategy under service constraints. We further perform trace analysis, demonstrating that design choices, such as a multi-agent framework and task-specific action spaces, effectively mitigate errors such as bad plans, inability to follow instructions, task context loss, and hallucinations. We anticipate that such agents could empower non-expert ML service providers, auditors, or regulators to systematically assess the risks of ML services without requiring deep domain expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19523v1">EAGER: Edge-Aligned LLM Defense for Robust, Efficient, and Accurate Cybersecurity Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are highly effective for cybersecurity question answering (QA) but are difficult to deploy on edge devices due to their size. Quantization reduces memory and compute requirements but often degrades accuracy and increases vulnerability to adversarial attacks. We present EAGER, an edge-aligned defense framework that integrates parameter-efficient quantization with domain-specific preference alignment to jointly optimize efficiency, robustness, and accuracy. Unlike prior methods that address these aspects separately, EAGER leverages Quantized Low-Rank Adaptation (QLoRA) for low-cost fine-tuning and Direct Preference Optimization (DPO) on a self-constructed cybersecurity preference dataset, eliminating the need for human labels. Experiments show that EAGER reduces adversarial attack success rates by up to 7.3x and improves QA accuracy by up to 55% over state-of-the-art defenses, while achieving the lowest response latency on a Jetson Orin, demonstrating its practical edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19517v1">Automating Deception: Scalable Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-11-24
    </div>
    <details class="paper-abstract">
      Multi-turn conversational attacks, which leverage psychological principles like Foot-in-the-Door (FITD), where a small initial request paves the way for a more significant one, to bypass safety alignments, pose a persistent threat to Large Language Models (LLMs). Progress in defending against these attacks is hindered by a reliance on manual, hard-to-scale dataset creation. This paper introduces a novel, automated pipeline for generating large-scale, psychologically-grounded multi-turn jailbreak datasets. We systematically operationalize FITD techniques into reproducible templates, creating a benchmark of 1,500 scenarios across illegal activities and offensive content. We evaluate seven models from three major LLM families under both multi-turn (with history) and single-turn (without history) conditions. Our results reveal stark differences in contextual robustness: models in the GPT family demonstrate a significant vulnerability to conversational history, with Attack Success Rates (ASR) increasing by as much as 32 percentage points. In contrast, Google's Gemini 2.5 Flash exhibits exceptional resilience, proving nearly immune to these attacks, while Anthropic's Claude 3 Haiku shows strong but imperfect resistance. These findings highlight a critical divergence in how current safety architectures handle conversational context and underscore the need for defenses that can resist narrative-based manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18635v1">No Free Lunch in Language Model Bias Mitigation? Targeted Bias Reduction Can Exacerbate Unmitigated LLM Biases</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inherit societal biases from their training data, potentially leading to harmful or unfair outputs. While various techniques aim to mitigate these biases, their effects are often evaluated only along the dimension of the bias being targeted. This work investigates the cross-category consequences of targeted bias mitigation. We study four bias mitigation techniques applied across ten models from seven model families, and we explore racial, religious, profession- and gender-related biases. We measure the impact of debiasing on model coherence and stereotypical preference using the StereoSet benchmark. Our results consistently show that while targeted mitigation can sometimes reduce bias in the intended dimension, it frequently leads to unintended and often negative consequences in others, such as increasing model bias and decreasing general coherence. These findings underscore the critical need for robust, multi-dimensional evaluation tools when examining and developing bias mitigation strategies to avoid inadvertently shifting or worsening bias along untargeted axes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18632v1">The Locally Deployable Virtual Doctor: LLM Based Human Interface for Automated Anamnesis and Database Conversion</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Recent advances in large language models made it possible to achieve high conversational performance with substantially reduced computational demands, enabling practical on-site deployment in clinical environments. Such progress allows for local integration of AI systems that uphold strict data protection and patient privacy requirements, yet their secure implementation in medicine necessitates careful consideration of ethical, regulatory, and technical constraints. In this study, we introduce MedChat, a locally deployable virtual physician framework that integrates an LLM-based medical chatbot with a diffusion-driven avatar for automated and structured anamnesis. The chatbot was fine-tuned using a hybrid corpus of real and synthetically generated medical dialogues, while model efficiency was optimized via Low-Rank Adaptation. A secure and isolated database interface was implemented to ensure complete separation between patient data and the inference process. The avatar component was realized through a conditional diffusion model operating in latent space, trained on researcher video datasets and synchronized with mel-frequency audio features for realistic speech and facial animation. Unlike existing cloud-based systems, this work demonstrates the feasibility of a fully offline, locally deployable LLM-diffusion framework for clinical anamnesis. The autoencoder and diffusion networks exhibited smooth convergence, and MedChat achieved stable fine-tuning with strong generalization to unseen data. The proposed system thus provides a privacy-preserving, resource-efficient foundation for AI-assisted clinical anamnesis, also in low-cost settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18608v1">From Reviewers' Lens: Understanding Bug Bounty Report Invalid Reasons with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Bug bounty platforms (e.g., HackerOne, BugCrowd) leverage crowd-sourced vulnerability discovery to improve continuous coverage, reduce the cost of discovery, and serve as an integral complement to internal red teams. With the rise of AI-generated bug reports, little work exists to help bug hunters understand why these reports are labeled as invalid. To improve report quality and reduce reviewers' burden, it is critical to predict invalid reports and interpret invalid reasons. In this work, we conduct an empirical study with the purpose of helping bug hunters understand the validity of reports. We collect a dataset of 9,942 disclosed bug bounty reports, including 1,400 invalid reports, and evaluate whether state-of-the-art large language models can identify invalid reports. While models such as GPT-5, DeepSeek, and a fine-tuned RoBERTa achieve strong overall accuracy, they consistently struggle to detect invalid cases, showing a tendency to over-accept reports. To improve invalidity detection, we build a taxonomy of rejection reasons for Information Disclosure vulnerabilities and incorporate it into a retrieval-augmented generation (RAG) framework. This approach substantially improves classification consistency and reduces bias. We also examine whether reviewer decisions may be influenced by factors beyond the content of the report. Our analysis shows that reporters with higher reputations tend to receive more favorable outcomes in borderline cases, suggesting that perceived expertise can influence review judgments. Overall, our findings highlight the challenges of invalid report identification and show that combining LLMs with structured reviewer knowledge can support more transparent and consistent vulnerability report review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08663v2">A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Psychological assessments are dominated by rating scales, which cannot capture the nuance in natural language. Efforts to supplement them with qualitative text have relied on labelled datasets or expert rubrics, limiting scalability. We introduce a framework that avoids this reliance: large language models (LLMs) score free-text responses with simple prompts to produce candidate LLM items, from which we retain those that yield the most test information when co-calibrated with a baseline scale. Using depression as a case study, we developed and tested the method in upper-secondary students (n=693) and a matched synthetic dataset (n=3,000). Results on held-out test sets showed that augmenting a 19-item scale with LLM items improved its precision, accuracy, and convergent validity. Further, the test information gain matched that of adding as many as 16 rating-scale items. This framework leverages the increasing availability of transcribed language to enhance psychometric measures, with applications in clinical health and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15938v2">Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      LLMs are commonly trained with a learning rate (LR) warmup, followed by cosine decay to 10% of the maximum (10x decay). In a large-scale empirical study, we show that under an optimal peak LR, a simple linear decay-to-zero (D2Z) schedule consistently outperforms other schedules when training at compute-optimal dataset sizes. D2Z is superior across a range of model sizes, batch sizes, datasets, and vocabularies. Benefits increase as dataset size increases. Leveraging a novel interpretation of AdamW as an exponential moving average of weight updates, we show how linear D2Z optimally balances the demands of early training (moving away from initial conditions) and late training (averaging over more updates in order to mitigate gradient noise). In experiments, a 610M-parameter model trained for 80 tokens-per-parameter (TPP) using D2Z achieves lower loss than when trained for 200 TPP using 10x decay, corresponding to an astonishing 60% compute savings. Models such as Llama2-7B, trained for 286 TPP with 10x decay, could likely have saved a majority of compute by training with D2Z.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13738v2">Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Efficient LLM pre-training requires well-tuned hyperparameters (HPs), including learning rate $η$ and weight decay $λ$. We study scaling laws for HPs: formulas for how to scale HPs as we scale model size N, dataset size D, and batch size B. Recent work suggests the AdamW timescale, $τ= B/(ηλD)$, should remain constant across training settings, and we verify the implication that optimal $λ$ scales linearly with B, for a fixed N and D. However, as N and D scale, we show optimal $τ$ obeys a precise power law in the tokens-per-parameter ratio, D/N. This law thus provides a method to accurately predict $λ$opt in advance of large-scale training. We also study scaling laws for optimal batch size Bopt (the B enabling lowest loss at a given N,D) and critical batch size Bcrit (the B beyond which further data parallelism becomes ineffective). In contrast to prior work, we find both Bopt and Bcrit scale as power laws in D, independent of model size, N. Finally, we analyze how these findings inform the real-world selection of Pareto-optimal N and D under dual training time and compute objectives. All experiments were run on Cerebras CS-3 systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18589v1">Strategic Decision Framework for Enterprise LLM Adoption</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 14 pages, 1 key figure
    </div>
    <details class="paper-abstract">
      Organizations are rapidly adopting Large Language Models (LLMs) to transform their operations, yet they lack clear guidance on key decisions for adoption and implementation. While LLMs offer powerful capabilities in content generation, assisted coding, and process automation, businesses face critical challenges in data security, LLM solution development approach, infrastructure requirements, and deployment strategies. Healthcare providers must protect patient data while leveraging LLMs for medical analysis, financial institutions need to balance automated customer service with regulatory compliance, and software companies seek to enhance development productivity while maintaining code security. This article presents a systematic six-step decision framework for LLM adoption, helping organizations navigate from initial application selection to final deployment. Based on extensive interviews and analysis of successful and failed implementations, our framework provides practical guidance for business leaders to align technological capabilities with business objectives. Through key decision points and real-world examples from both B2B and B2C contexts, organizations can make informed decisions about LLM adoption while ensuring secure and efficient integration across various use cases, from customer service automation to content creation and advanced analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18581v1">TASO: Jailbreak LLMs via Alternative Template and Suffix Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Many recent studies showed that LLMs are vulnerable to jailbreak attacks, where an attacker can perturb the input of an LLM to induce it to generate an output for a harmful question. In general, existing jailbreak techniques either optimize a semantic template intended to induce the LLM to produce harmful outputs or optimize a suffix that leads the LLM to initiate its response with specific tokens (e.g., "Sure"). In this work, we introduce TASO (Template and Suffix Optimization), a novel jailbreak method that optimizes both a template and a suffix in an alternating manner. Our insight is that suffix optimization and template optimization are complementary to each other: suffix optimization can effectively control the first few output tokens but cannot control the overall quality of the output, while template optimization provides guidance for the entire output but cannot effectively control the initial tokens, which significantly impact subsequent responses. Thus, they can be combined to improve the attack's effectiveness. We evaluate the effectiveness of TASO on benchmark datasets (including HarmBench and AdvBench) on 24 leading LLMs (including models from the Llama family, OpenAI, and DeepSeek). The results demonstrate that TASO can effectively jailbreak existing LLMs. We hope our work can inspire future studies in exploring this direction. We will make code and data publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18531v1">LockForge: Automating Paper-to-Code for Logic Locking with Multi-Agent Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Despite rapid progress in logic locking (LL), reproducibility remains a challenge as codes are rarely made public. We present LockForge, a first-of-its-kind, multi-agent large language model (LLM) framework that turns LL descriptions in papers into executable and tested code. LockForge provides a carefully crafted pipeline realizing forethought, implementation, iterative refinement, and a multi-stage validation, all to systematically bridge the gap between prose and practice for complex LL schemes. For validation, we devise (i) an LLM-as-Judge stage with a scoring system considering behavioral checks, conceptual mechanisms, structural elements, and reproducibility on benchmarks, and (ii) an independent LLM-as-Examiner stage for ground-truth assessment. We apply LockForge to 10 seminal LL schemes, many of which lack reference implementations. Our evaluation on multiple SOTA LLMs, including ablation studies, reveals the significant complexity of the task. We show that an advanced reasoning model and a sophisticated, multi-stage framework like LockForge are required. We release all implementations and benchmarks, providing a reproducible and fair foundation for evaluation of further LL research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.09711v2">PsychiatryBench: A Multi-Task Benchmark for LLMs in Psychiatry</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential in enhancing psychiatric practice, from improving diagnostic accuracy to streamlining clinical documentation and therapeutic support. However, existing evaluation resources heavily rely on small clinical interview corpora, social media posts, or synthetic dialogues, which limits their clinical validity and fails to capture the full complexity of diagnostic reasoning. In this work, we introduce PsychiatryBench, a rigorously curated benchmark grounded exclusively in authoritative, expert-validated psychiatric textbooks and casebooks. PsychiatryBench comprises eleven distinct question-answering tasks ranging from diagnostic reasoning and treatment planning to longitudinal follow-up, management planning, clinical approach, sequential case analysis, and multiple-choice/extended matching formats totaling 5,188 expert-annotated items. {\color{red}We evaluate a diverse set of frontier LLMs (including Google Gemini, DeepSeek, Sonnet 4.5, and GPT 5) alongside leading open-source medical models such as MedGemma using both conventional metrics and an "LLM-as-judge" similarity scoring framework. Our results reveal substantial gaps in clinical consistency and safety, particularly in multi-turn follow-up and management tasks, underscoring the need for specialized model tuning and more robust evaluation paradigms. PsychiatryBench offers a modular, extensible platform for benchmarking and improving LLM performance in mental health applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18467v1">Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted by AAAI 2026 Alignment Track
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Model (LLM)-driven multi-agent systems has significantly streamlined software developing tasks, enabling users with little technical expertise to develop executable applications. While these systems democratize software creation through natural language requirements, they introduce significant security risks that remain largely unexplored. We identify two risky scenarios: Malicious User with Benign Agents (MU-BA) and Benign User with Malicious Agents (BU-MA). We introduce the Implicit Malicious Behavior Injection Attack (IMBIA), demonstrating how multi-agent systems can be manipulated to generate software with concealed malicious capabilities beneath seemingly benign applications, and propose Adv-IMBIA as a defense mechanism. Evaluations across ChatDev, MetaGPT, and AgentVerse frameworks reveal varying vulnerability patterns, with IMBIA achieving attack success rates of 93%, 45%, and 71% in MU-BA scenarios, and 71%, 84%, and 45% in BU-MA scenarios. Our defense mechanism reduced attack success rates significantly, particularly in the MU-BA scenario. Further analysis reveals that compromised agents in the coding and testing phases pose significantly greater security risks, while also identifying critical agents that require protection against malicious user exploitation. Our findings highlight the urgent need for robust security measures in multi-agent software development systems and provide practical guidelines for implementing targeted, resource-efficient defensive strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11088v2">One SPACE to Rule Them All: Jointly Mitigating Factuality and Faithfulness Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted as NIPS 2025 poster
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated unprecedented capabilities in natural language processing, yet their practical deployment remains hindered by persistent factuality and faithfulness hallucinations. While existing methods address these hallucination types independently, they inadvertently induce performance trade-offs, as interventions targeting one type often exacerbate the other. Through empirical and theoretical analysis of activation space dynamics in LLMs, we reveal that these hallucination categories share overlapping subspaces within neural representations, presenting an opportunity for concurrent mitigation. To harness this insight, we propose SPACE, a unified framework that jointly enhances factuality and faithfulness by editing shared activation subspaces. SPACE establishes a geometric foundation for shared subspace existence through dual-task feature modeling, then identifies and edits these subspaces via a hybrid probe strategy combining spectral clustering and attention head saliency scoring. Experimental results across multiple benchmark datasets demonstrate the superiority of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14617v2">Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 16 pages, 12 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become critical for advancing modern Large Language Models (LLMs), yet existing synchronous RL systems face severe performance bottlenecks. The rollout phase, which dominates end-to-end iteration time, suffers from substantial long-tail latency and poor resource utilization due to inherent workload imbalance. We present Seer, a novel online context learning system that addresses these challenges by exploiting previously overlooked similarities in output lengths and generation patterns among requests sharing the same prompt. Seer introduces three key techniques: divided rollout for dynamic load balancing, context-aware scheduling, and adaptive grouped speculative decoding. Together, these mechanisms substantially reduce long-tail latency and improve resource efficiency during rollout. Evaluations on production-grade RL workloads demonstrate that Seer improves end-to-end rollout throughput by 74% to 97% and reduces long-tail latency by 75% to 93% compared to state-of-the-art synchronous RL systems, significantly accelerating RL training iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18416v2">Exploring Potential Prompt Injection Attacks in Federated Military LLMs and Their Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted to the 3rd International Workshop on Dataspaces and Digital Twins for Critical Entities and Smart Urban Communities - IEEE BigData 2025
    </div>
    <details class="paper-abstract">
      Federated Learning (FL) is increasingly being adopted in military collaborations to develop Large Language Models (LLMs) while preserving data sovereignty. However, prompt injection attacks-malicious manipulations of input prompts-pose new threats that may undermine operational security, disrupt decision-making, and erode trust among allies. This perspective paper highlights four vulnerabilities in federated military LLMs: secret data leakage, free-rider exploitation, system disruption, and misinformation spread. To address these risks, we propose a human-AI collaborative framework with both technical and policy countermeasures. On the technical side, our framework uses red/blue team wargaming and quality assurance to detect and mitigate adversarial behaviors of shared LLM weights. On the policy side, it promotes joint AI-human policy development and verification of security protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18438v1">LLMs as Firmware Experts: A Runtime-Grown Tree-of-Agents Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 18 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and their agent systems have recently demonstrated strong potential in automating code reasoning and vulnerability detection. However, when applied to large-scale firmware, their performance degrades due to the binary nature of firmware, complex dependency structures, and heterogeneous components. To address this challenge, this paper presents FIRMHIVE, a recursive agent hive that enables LLMs to act as autonomous firmware security analysts. FIRMHIVE introduces two key mechanisms: (1) transforming delegation into a per-agent, executable primitive and (2) constructing a runtime Tree of Agents (ToA) for decentralized coordination. We evaluate FIRMHIVE using real-world firmware images obtained from publicly available datasets, covering five representative security analysis tasks. Compared with existing LLM-agent baselines, FIRMHIVE performs deeper (about 16x more reasoning steps) and broader (about 2.3x more files inspected) cross-file exploration, resulting in about 5.6x more alerts per firmware. Compared to state-of-the-art (SOTA) security tools, FIRMHIVE identifies about 1.5x more vulnerabilities (1,802 total) and achieves 71% precision, representing significant improvements in both yield and fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18403v1">UnWEIRDing LLM Entity Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models have been widely been adopted by users for writing tasks such as sentence completions. While this can improve writing efficiency, prior research shows that LLM-generated suggestions may exhibit cultural biases which may be difficult for users to detect, especially in educational contexts for non-native English speakers. While such prior work has studied the biases in LLM moral value alignment, we aim to investigate cultural biases in LLM recommendations for real-world entities. To do so, we use the WEIRD (Western, Educated, Industrialized, Rich and Democratic) framework to evaluate recommendations by various LLMs across a dataset of fine-grained entities, and apply pluralistic prompt-based strategies to mitigate these biases. Our results indicate that while such prompting strategies do reduce such biases, this reduction is not consistent across different models, and recommendations for some types of entities are more biased than others.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18394v1">Future Is Unevenly Distributed: Forecasting Ability of LLMs Depends on What We're Asking</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate partial forecasting competence across social, political, and economic events. Yet, their predictive ability varies sharply with domain structure and prompt framing. We investigate how forecasting performance varies with different model families on real-world questions about events that happened beyond the model cutoff date. We analyze how context, question type, and external knowledge affect accuracy and calibration, and how adding factual news context modifies belief formation and failure modes. Our results show that forecasting ability is highly variable as it depends on what, and how, we ask.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18343v1">A Needle in a Haystack: Intent-driven Reusable Artifacts Recommendation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      In open source software development, the reuse of existing artifacts has been widely adopted to avoid redundant implementation work. Reusable artifacts are considered more efficient and reliable than developing software components from scratch. However, when faced with a large number of reusable artifacts, developers often struggle to find artifacts that can meet their expected needs. To reduce this burden, retrieval-based and learning-based techniques have been proposed to automate artifact recommendations. Recently, Large Language Models (LLMs) have shown the potential to understand intentions, perform semantic alignment, and recommend usable artifacts. Nevertheless, their effectiveness has not been thoroughly explored. To fill this gap, we construct an intent-driven artifact recommendation benchmark named IntentRecBench, covering three representative open source ecosystems. Using IntentRecBench, we conduct a comprehensive comparative study of five popular LLMs and six traditional approaches in terms of precision and efficiency. Our results show that although LLMs outperform traditional methods, they still suffer from low precision and high inference cost due to the large candidate space. Inspired by the ontology-based semantic organization in software engineering, we propose TreeRec, a feature tree-guided recommendation framework to mitigate these issues. TreeRec leverages LLM-based semantic abstraction to organize artifacts into a hierarchical semantic tree, enabling intent and function alignment and reducing reasoning time. Extensive experiments demonstrate that TreeRec consistently improves the performance of diverse LLMs across ecosystems, highlighting its generalizability and potential for practical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02393v3">AP2O-Coder: Human-Inspired Progressive Optimization to Fix LLM Code Errors</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      LLMs' code generation capabilities have yielded substantial improvements in the effectiveness of programming tasks. However, LLM-generated code still suffers from compilation and runtime errors. Existing offline preference optimization methods primarily focus on enhancing LLMs' coding abilities using pass/fail signals in the preference data, overlooking the deep-level error types in the failed codes. To address this, we propose Adaptively Progressive Preference Optimization (AP2O) for coding (i.e., AP2O-Coder), a method that guides LLMs adaptively and methodically to reduce code errors for code generation. Specifically, we construct an error notebook from failed codes and progressively optimize the LLM to correct errors type by type. Furthermore, we adaptively replay error types to tailor to the LLM's changing weaknesses throughout the training process. Through extensive experiments on both code and general LLMs (Llama, Qwen, and DeepSeek series) with parameters ranging from 0.5B to 34B, our AP2O-Coder improves code generation performance by up to 3% in pass@k while using less preference data. Code: https://github.com/TsingZ0/AP2O
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05012v2">LLMs' Reshaping of People, Processes, Products, and Society in Software Development: A Comprehensive Exploration with Early Adopters</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly reshaping software development, but their impact across the software development lifecycle is underexplored. Existing work focuses on isolated activities such as code generation or testing, leaving open questions about how LLMs affect developers, processes, products, and the software ecosystem. We address this gap through semi-structured interviews with sixteen early-adopter software professionals who integrated LLM-based tools into their day-to-day work in early to mid-2023. We treat these interviews as early empirical evidence and compare participants' accounts with recent work on LLMs in software engineering, noting which early patterns persist or shift. Using thematic analysis, we organize findings around four dimensions: people, process, product, and society. Developers reported substantial productivity gains from reducing routine tasks, streamlining search, and accelerating debugging, but also described a productivity-quality paradox: they often discarded generated code and shifted effort from writing to critically evaluating and integrating it. LLM use was highly phase-dependent, with strong uptake in implementation and debugging but limited influence on requirements gathering and other collaborative work. Participants developed new competencies to use LLMs effectively, including prompt engineering strategies, layered verification, and secure integration to protect proprietary data. They anticipated changes in hiring expectations, team practices, and computing education, while emphasizing that human judgment and foundational software engineering skills remain essential. Our findings, later echoed in large-scale quantitative studies, offer actionable implications for developers, organizations, educators, and tool designers seeking to integrate LLMs responsibly into software practice today.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18313v1">Path-Constrained Retrieval: A Structural Approach to Reliable LLM Agent Reasoning Through Graph-Scoped Semantic Search</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Model agents often retrieve context from knowledge bases that lack structural consistency with the agent's current reasoning state, leading to incoherent reasoning chains. We introduce Path-Constrained Retrieval (PCR), a retrieval method that combines structural graph constraints with semantic search to ensure retrieved information maintains logical relationships within a knowledge graph. PCR restricts the search space to nodes reachable from an anchor node, preventing retrieval of structurally disconnected information that may lead to inconsistent reasoning. We evaluate PCR on PathRAG-6, a benchmark spanning six domains with 180 nodes and 360 edges. Our results show that PCR achieves full structural consistency compared to 24-32 percent in baseline methods, while maintaining strong relevance scores. On the technology domain, PCR obtains full relevance at rank 10 with full structural consistency, significantly outperforming vector search and hybrid retrieval. PCR reduces the average graph distance of retrieved context by 78 percent compared to baselines, demonstrating retrieval of more structurally consistent information. These findings suggest that path-constrained retrieval is an effective approach for improving the reliability and coherence of LLM agent reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.05130v2">Scalable and Accurate Graph Reasoning with LLM-based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Accepted by AAAI 2026 Workshop WMAC
    </div>
    <details class="paper-abstract">
      Recent research has explored the use of Large Language Models (LLMs) for tackling complex graph reasoning tasks. However, due to the intricacies of graph structures and the inherent limitations of LLMs in handling long text, current approaches often fail to deliver satisfactory accuracy, even on small-scale graphs and simple tasks. To address these challenges, we introduce GraphAgent-Reasoner, a fine-tuning-free framework that utilizes a multi-agent collaboration strategy for explicit and precise graph reasoning. Inspired by distributed graph computation theory, our framework decomposes graph problems into smaller, node-centric tasks that are distributed among multiple agents. The agents collaborate to solve the overall problem, significantly reducing the amount of information and complexity handled by a single LLM, thus enhancing the accuracy of graph reasoning. By simply increasing the number of agents, GraphAgent-Reasoner can efficiently scale to accommodate larger graphs with over 1,000 nodes. Evaluated on the GraphInstruct dataset, our framework demonstrates near-perfect accuracy on polynomial-time graph reasoning tasks, significantly outperforming the best available models, both closed-source and fine-tuned open-source variants. Our framework also demonstrates the capability to handle real-world graph reasoning applications such as webpage importance analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18270v1">Skypilot: Fine-Tuning LLM with Physical Grounding for AAV Coverage Search</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Autonomous aerial vehicles (AAVs) have played a pivotal role in coverage operations and search missions. Recent advances in large language models (LLMs) offer promising opportunities to augment AAV intelligence. These advances help address complex challenges like area coverage optimization, dynamic path planning, and adaptive decision-making. However, the absence of physical grounding in LLMs leads to hallucination and reproducibility problems in spatial reasoning and decision-making. To tackle these issues, we present Skypilot, an LLM-enhanced two-stage framework that grounds language models in physical reality by integrating monte carlo tree search (MCTS). In the first stage, we introduce a diversified action space that encompasses generate, regenerate, fine-tune, and evaluate operations, coupled with physics-informed reward functions to ensure trajectory feasibility. In the second stage, we fine-tune Qwen3-4B on 23,000 MCTS-generated samples, achieving substantial inference acceleration while maintaining solution quality. Extensive numerical simulations and real-world flight experiments validate the efficiency and superiority of our proposed approach. Detailed information and experimental results are accessible at https://sky-pilot.top.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18261v1">LLM Reasoning for Cold-Start Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential for improving recommendation systems through their inherent reasoning capabilities and extensive knowledge base. Yet, existing studies predominantly address warm-start scenarios with abundant user-item interaction data, leaving the more challenging cold-start scenarios, where sparse interactions hinder traditional collaborative filtering methods, underexplored. To address this limitation, we propose novel reasoning strategies designed for cold-start item recommendations within the Netflix domain. Our method utilizes the advanced reasoning capabilities of LLMs to effectively infer user preferences, particularly for newly introduced or rarely interacted items. We systematically evaluate supervised fine-tuning, reinforcement learning-based fine-tuning, and hybrid approaches that combine both methods to optimize recommendation performance. Extensive experiments on real-world data demonstrate significant improvements in both methodological efficacy and practical performance in cold-start recommendation contexts. Remarkably, our reasoning-based fine-tuned models outperform Netflix's production ranking model by up to 8% in certain cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18249v1">LLM Assisted Coding with Metamorphic Specification Mutation Agent</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Metamorphic Relations (MRs) serve as a foundational mechanism for generating semantically equivalent mutations. Software engineering has advanced significantly in recent years with the advent of Large Language Models (LLMs). However, the reliability of LLMs in software engineering is often compromised by ambiguities and inconsistencies due to improper user specification. To address this challenge, we present CodeMetaAgent (CMA), a metamorphic relation-driven LLM agent that systematically refines task specifications and generates semantically constrained test cases. Our proposed framework uses MRs with LLMs to improve generation consistency and reduce variability caused by specifications, unlike the traditional use of MRs as post validations. Our framework has been evaluated on the HumanEval-Pro, MBPP-Pro, and SWE-Bench_Lite datasets using the GPT-4o, Mistral Large, GPT-OSS, and Qwen3-Coder models. It improved code generation accuracy by up to 17% and achieved code coverage gains of up to 99.81%. These results show that metamorphic relations can be a simple but effective guide in assisting LLM-based software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10729v2">Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Sensor data streams provide valuable information around activities and context for downstream applications, though integrating complementary information can be challenging. We show that large language models (LLMs) can be used for late fusion for activity classification from audio and motion time series data. We curated a subset of data for diverse activity recognition across contexts (e.g., household activities, sports) from the Ego4D dataset. Evaluated LLMs achieved 12-class zero- and one-shot classification F1-scores significantly above chance, with no task-specific training. Zero-shot classification via LLM-based fusion from modality-specific models can enable multimodal temporal applications where there is limited aligned training data for learning a shared embedding space. Additionally, LLM-based fusion can enable model deploying without requiring additional memory and computation for targeted application-specific multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18239v1">Can LLMs Help Allocate Public Health Resources? A Case Study on Childhood Lead Testing</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Public health agencies face critical challenges in identifying high-risk neighborhoods for childhood lead exposure with limited resources for outreach and intervention programs. To address this, we develop a Priority Score integrating untested children proportions, elevated blood lead prevalence, and public health coverage patterns to support optimized resource allocation decisions across 136 neighborhoods in Chicago, New York City, and Washington, D.C. We leverage these allocation tasks, which require integrating multiple vulnerability indicators and interpreting empirical evidence, to evaluate whether large language models (LLMs) with agentic reasoning and deep research capabilities can effectively allocate public health resources when presented with structured allocation scenarios. LLMs were tasked with distributing 1,000 test kits within each city based on neighborhood vulnerability indicators. Results reveal significant limitations: LLMs frequently overlooked neighborhoods with highest lead prevalence and largest proportions of untested children, such as West Englewood in Chicago, while allocating disproportionate resources to lower-priority areas like Hunts Point in New York City. Overall accuracy averaged 0.46, reaching a maximum of 0.66 with ChatGPT 5 Deep Research. Despite their marketed deep research capabilities, LLMs struggled with fundamental limitations in information retrieval and evidence-based reasoning, frequently citing outdated data and allowing non-empirical narratives about neighborhood conditions to override quantitative vulnerability indicators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18230v1">Think Fast: Real-Time IoT Intrusion Reasoning Using IDS and LLMs at the Edge Gateway</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      As the number of connected IoT devices continues to grow, securing these systems against cyber threats remains a major challenge, especially in environments with limited computational and energy resources. This paper presents an edge-centric Intrusion Detection System (IDS) framework that integrates lightweight machine learning (ML) based IDS models with pre-trained large language models (LLMs) to improve detection accuracy, semantic interpretability, and operational efficiency at the network edge. The system evaluates six ML-based IDS models: Decision Tree (DT), K-Nearest Neighbors (KNN), Random Forest (RF), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and a hybrid CNN-LSTM model on low-power edge gateways, achieving accuracy up to 98 percent under real-world cyberattacks. For anomaly detection, the system transmits a compact and secure telemetry snapshot (for example, CPU usage, memory usage, latency, and energy consumption) via low-bandwidth API calls to LLMs including GPT-4-turbo, DeepSeek V2, and LLaMA 3.5. These models use zero-shot, few-shot, and chain-of-thought reasoning to produce human-readable threat analyses and actionable mitigation recommendations. Evaluations across diverse attacks such as DoS, DDoS, brute force, and port scanning show that the system enhances interpretability while maintaining low latency (<1.5 s), minimal bandwidth usage (<1.2 kB per prompt), and energy efficiency (<75 J), demonstrating its practicality and scalability as an IDS solution for edge gateways.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19489v1">Evolution without an Oracle: Driving Effective Evolution with LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) with Evolutionary Computation (EC) has unlocked new frontiers in scientific discovery but remains shackled by a fundamental constraint: the reliance on an Oracle--an objective, machine-computable fitness function. This paper breaks this barrier by asking: Can evolution thrive in a purely subjective landscape governed solely by LLM judges? We introduce MADE (Multi-Agent Decomposed Evolution), a framework that tames the inherent noise of subjective evaluation through "Problem Specification." By decomposing vague instructions into specific, verifiable sub-requirements, MADE transforms high-variance LLM feedback into stable, precise selection pressure. The results are transformative: across complex benchmarks like DevAI and InfoBench, MADE outperforms strong baselines by over 50% in software requirement satisfaction (39.9% to 61.9%) and achieves a 95% perfect pass rate on complex instruction following. This work validates a fundamental paradigm shift: moving from optimizing "computable metrics" to "describable qualities," thereby unlocking evolutionary optimization for the vast open-ended domains where no ground truth exists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19488v1">Building Resilient Information Ecosystems: Large LLM-Generated Dataset of Persuasion Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Organization's communication is essential for public trust, but the rise of generative AI models has introduced significant challenges by generating persuasive content that can form competing narratives with official messages from government and commercial organizations at speed and scale. This has left agencies in a reactive position, often unaware of how these models construct their persuasive strategies, making it more difficult to sustain communication effectiveness. In this paper, we introduce a large LLM-generated persuasion attack dataset, which includes 134,136 attacks generated by GPT-4, Gemma 2, and Llama 3.1 on agency news. These attacks span 23 persuasive techniques from SemEval 2023 Task 3, directed toward 972 press releases from ten agencies. The generated attacks come in two mediums, press release statements and social media posts, covering both long-form and short-form communication strategies. We analyzed the moral resonance of these persuasion attacks to understand their attack vectors. GPT-4's attacks mainly focus on Care, with Authority and Loyalty also playing a role. Gemma 2 emphasizes Care and Authority, while Llama 3.1 centers on Loyalty and Care. Analyzing LLM-generated persuasive attacks across models will enable proactive defense, allow to create the reputation armor for organizations, and propel the development of both effective and resilient communications in the information ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19483v1">Z-Space: A Multi-Agent Tool Orchestration Framework for Enterprise-Grade LLM Automation</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models can break through knowledge and timeliness limitations by invoking external tools within the Model Context Protocol framework to achieve automated execution of complex tasks. However, with the rapid growth of enterprise-scale MCP services, efficiently and accurately matching target functionalities among thousands of heterogeneous tools has become a core challenge restricting system practicality. Existing approaches generally rely on full-prompt injection or static semantic retrieval, facing issues including semantic disconnection between user queries and tool descriptions, context inflation in LLM input, and high inference latency. To address these challenges, this paper proposes Z-Space, a data-generation-oriented multi-agent collaborative tool invocation framework Z-Space. The Z-Space framework establishes a multi-agent collaborative architecture and tool filtering algorithm: (1) A structured semantic understanding of user queries is achieved through an intent parsing model; (2) A tool filtering module (FSWW) based on fused subspace weighted algorithm realizes fine-grained semantic alignment between intents and tools without parameter tuning; (3) An inference execution agent is constructed to support dynamic planning and fault-tolerant execution for multi-step tasks. This framework has been deployed in the Eleme platform's technical division, serving large-scale test data generation scenarios across multiple business units including Taotian, Gaode, and Hema. Production data demonstrates that the system reduces average token consumption in tool inference by 96.26\% while achieving a 92\% tool invocation accuracy rate, significantly enhancing the efficiency and reliability of intelligent test data generation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18617v2">DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-23
      | 💬 19 pages, 15 figures, 12 tables
    </div>
    <details class="paper-abstract">
      With the rapid rise of personalized AI, customized large language models (LLMs) equipped with Chain of Thought (COT) reasoning now power millions of AI agents. However, their complex reasoning processes introduce new and largely unexplored security vulnerabilities. We present DarkMind, a novel latent reasoning level backdoor attack that targets customized LLMs by manipulating internal COT steps without altering user queries. Unlike prior prompt based attacks, DarkMind activates covertly within the reasoning chain via latent triggers, enabling adversarial behaviors without modifying input prompts or requiring access to model parameters. To achieve stealth and reliability, we propose dual trigger types instant and retrospective and integrate them within a unified embedding template that governs trigger dependent activation, employ a stealth optimization algorithm to minimize semantic drift, and introduce an automated conversation starter for covert activation across domains. Comprehensive experiments on eight reasoning datasets spanning arithmetic, commonsense, and symbolic domains, using five LLMs, demonstrate that DarkMind consistently achieves high attack success rates. We further investigate defense strategies to mitigate these risks and reveal that reasoning level backdoors represent a significant yet underexplored threat, underscoring the need for robust, reasoning aware security mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18653v1">FHE-Agent: Automating CKKS Configuration for Practical Encrypted Inference via an LLM-Guided Agentic Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Fully Homomorphic Encryption (FHE), particularly the CKKS scheme, is a promising enabler for privacy-preserving MLaaS, but its practical deployment faces a prohibitive barrier: it heavily relies on domain expertise. Configuring CKKS involves a tightly coupled space of ring dimensions, modulus chains, and packing layouts. Without deep cryptographic knowledge to navigate these interactions, practitioners are restricted to compilers that rely on fixed heuristics. These "one-shot" tools often emit rigid configurations that are either severely over-provisioned in latency or fail to find a feasible solution entirely for deeper networks. We present FHE-Agent, an agentic framework that automates this expert reasoning process. By coupling a Large Language Model (LLM) controller with a deterministic tool suite, FHE-Agent decomposes the search into global parameter selection and layer-wise bottleneck repair. The agents operate within a multi-fidelity workflow, pruning invalid regimes using cheap static analysis and reserving expensive encrypted evaluations for the most promising candidates. We instantiate FHE-Agent on the Orion compiler and evaluate it on standard benchmarks (MLP, LeNet, LoLa) and deeper architectures (AlexNet). FHE-Agent consistently achieves better precision and lower latency than naïve search strategies. Crucially, it automatically discovers feasible, 128-bit secure configurations for complex models where baseline heuristics and one-shot prompts fail to produce a valid setup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.15524v3">The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective</a></div>
    <div class="paper-meta">
      📅 2025-11-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive potential to simulate human behavior. We identify a fundamental challenge in using them to simulate experiments: when LLM-simulated subjects are blind to the experimental design (as is standard practice with human subjects), variations in treatment systematically affect unspecified variables that should remain constant, violating the unconfoundedness assumption. Using demand estimation as a context and an actual experiment with 40 different products as a benchmark, we show this can lead to implausible results. While confounding may in principle be addressed by controlling for covariates, this can compromise ecological validity in the context of LLM simulations: controlled covariates become artificially salient in the simulated decision process. We show formally that confoundness stems from ambiguous prompting strategies. Therefore, it can be addressed by developing unambiguous prompting strategies through unblinding, i.e., revealing the experiment design in LLM simulations. Our empirical results show that this strategy consistently enhances model performance across all tested models, including both out-of-box reasoning and non-reasoning models. We also show that it is a technique that complements fine-tuning: while fine-tuning can improve simulation performance, an unambiguous prompting strategy makes the predictions robust to the inclusion of irrelevant data in the fine-tuning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17923v1">Towards Efficient LLM-aware Heterogeneous Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Heterogeneous graphs are widely present in real-world complex networks, where the diversity of node and relation types leads to complex and rich semantics. Efforts for modeling complex relation semantics in heterogeneous graphs are restricted by the limitations of predefined semantic dependencies and the scarcity of supervised signals. The advanced pre-training and fine-tuning paradigm leverages graph structure to provide rich self-supervised signals, but introduces semantic gaps between tasks. Large Language Models (LLMs) offer significant potential to address the semantic issues of relations and tasks in heterogeneous graphs through their strong reasoning capabilities in textual modality, but their incorporation into heterogeneous graphs is largely limited by computational complexity. Therefore, in this paper, we propose an Efficient LLM-Aware (ELLA) framework for heterogeneous graphs, addressing the above issues. To capture complex relation semantics, we propose an LLM-aware Relation Tokenizer that leverages LLM to encode multi-hop, multi-type relations. To reduce computational complexity, we further employ a Hop-level Relation Graph Transformer, which help reduces the complexity of LLM-aware relation reasoning from exponential to linear. To bridge semantic gaps between pre-training and fine-tuning tasks, we introduce the fine-grained task-aware textual Chain-of-Thought (CoT) prompts. Extensive experiments on four heterogeneous graphs show that our proposed ELLA outperforms state-of-the-art methods in the performance and efficiency. In particular, ELLA scales up to 13b-parameter LLMs and achieves up to a 4x speedup compared with existing LLM-based methods. Our code is publicly available at https://github.com/l-wd/ELLA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17913v1">Token-Controlled Re-ranking for Sequential Recommendation via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) as re-rankers is shifting recommender systems towards a user-centric paradigm. However, a significant gap remains: current re-rankers often lack mechanisms for fine-grained user control. They struggle to balance inherent user preferences with multiple attribute-based constraints, often resorting to simplistic hard filtering that can excessively narrow the recommendation pool and yield suboptimal results. This limitation leaves users as passive recipients rather than active collaborators in the recommendation process. To bridge this gap, we propose COREC, a novel token-augmented re-ranking framework that incorporates specific user requirements in co-creating the recommendation outcome. COREC empowers users to steer re-ranking results with precise and flexible control via explicit, attribute-based signals. The framework learns to balance these commands against latent preferences, yielding rankings that adhere to user instructions without sacrificing personalization. Experiments show that COREC: (1) exceeds state-of-the-art baselines on standard recommendation effectiveness and (2) demonstrates superior adherence to specific attribute requirements, proving that COREC enables fine-grained and predictable manipulation of the rankings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.07189v2">Secure-Instruct: An Automated Pipeline for Synthesizing Instruction-Tuning Datasets Using LLMs for Secure Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) show promising solutions to automated code generation, they often produce insecure code that threatens software security. Current approaches (e.g., SafeCoder) to improve secure code generation are limited by small, imbalanced instruction-tuning datasets. In this work, we present Secure-Instruct, a novel pipeline that automatically synthesizes high-quality vulnerable and secure code examples and instruction-tunes LLMs to align task description and secure code generation abilities. We evaluate Secure-Instruct on four representative LLMs using two security-related benchmarks: our own CWEBench and the existing CWEval. CWEBench comprises 93 scenarios on 44 CWEs, all without overlap with Secure-Instruct's synthetic instruction-tuning dataset, while CWEval covers 31 CWEs with 119 manually verified security-critical tasks. We find that Secure-Instruct improves both security and functional correctness in code generation. On CWEBench, Secure-Instruct substantially improves secure code generation, giving a 28.5% increase on average in secure ratio over the pre-trained models and outperforms SafeCoder by 12.6%. On CWEval, Secure-Instruct achieves an increase of 157.3% for CodeLlama-7B and 46.4% for Mistral-7B in Func-Sec@1 over pretrained models, and significantly outperforms SafeCoder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12288v2">Reducing Hallucinations in LLM-Generated Code via Semantic Triangulation</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      When generating code from natural language prompts, an LLM samples programs from a probability distribution, many of which might be incorrect. Sample consensus techniques - such as majority voting or validation against generated tests or specifications - aim to identify a correct program in the sample or abstain if none is valid. However, existing methods often fail to select a correct solution when its sampling probability is low, or when the problem permits multiple valid but non-equivalent solutions. Additionally, they often fail to abstain when no correct solution is present in the sample. To overcome these limitations, we introduce semantic triangulation, which transforms a programming problem in a way that non-trivially alters its semantics while preserving an exact, verifiable mapping between solutions before and after transformation. We theoretically establish that verifying consistency across such problem transformations increases confidence that generated programs reflect accurate generalization rather than spurious statistical correlations, enabling more reliable sample consensus and abstention. On the LiveCodeBench and CodeElo benchmarks, using GPT-4o and DeepSeek-V3 models, semantic triangulation increases reliability of generated code by 21% compared to the method that selects only high-confidence solutions with the probability threshold 0.5, while being able to pinpoint correct solutions at sampling probabilities as low as 0.14. Apart from that, it is also the only approach to consistently form true consensus on tasks with multiple valid but non-equivalent solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17648v3">Simulating Macroeconomic Expectations using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for simulating macroeconomic expectations using Large Language Model-Empowered Agents (LLM Agents). By constructing LLM Agents equipped with various functional modules, we replicate three representative survey experiments involving several expectations across different types of economic agents. Our results show that although the expectations simulated by LLM Agents are more homogeneous than those of humans, they consistently outperform LLMs relying simply on prompt engineering, and possess human-like mental mechanisms. Evaluation reveals that these capabilities stem from the contributions of their components, offering guidelines for their architectural design. Our approach complements traditional methods and provides new insights into AI behavioral science in macroeconomic research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17874v1">Beyond Jailbreak: Unveiling Risks in LLM Applications Arising from Blurred Capability Boundaries</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      LLM applications (i.e., LLM apps) leverage the powerful capabilities of LLMs to provide users with customized services, revolutionizing traditional application development. While the increasing prevalence of LLM-powered applications provides users with unprecedented convenience, it also brings forth new security challenges. For such an emerging ecosystem, the security community lacks sufficient understanding of the LLM application ecosystem, especially regarding the capability boundaries of the applications themselves. In this paper, we systematically analyzed the new development paradigm and defined the concept of the LLM app capability space. We also uncovered potential new risks beyond jailbreak that arise from ambiguous capability boundaries in real-world scenarios, namely, capability downgrade and upgrade. To evaluate the impact of these risks, we designed and implemented an LLM app capability evaluation framework, LLMApp-Eval. First, we collected application metadata across 4 platforms and conducted a cross-platform ecosystem analysis. Then, we evaluated the risks for 199 popular applications among 4 platforms and 6 open-source LLMs. We identified that 178 (89.45%) potentially affected applications, which can perform tasks from more than 15 scenarios or be malicious. We even found 17 applications in our study that executed malicious tasks directly, without applying any adversarial rewriting. Furthermore, our experiments also reveal a positive correlation between the quality of prompt design and application robustness. We found that well-designed prompts enhance security, while poorly designed ones can facilitate abuse. We hope our work inspires the community to focus on the real-world risks of LLM applications and foster the development of a more robust LLM application ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.08068v2">A Roadmap to Guide the Integration of LLMs in Hierarchical Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 5 pages, 0 figures, to be published in the AAAI Workshop on Planning in the Era of LLMs ( https://llmforplanning.github.io )
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) are fostering their integration into several reasoning-related fields, including Automated Planning (AP). However, their integration into Hierarchical Planning (HP), a subfield of AP that leverages hierarchical knowledge to enhance planning performance, remains largely unexplored. In this preliminary work, we propose a roadmap to address this gap and harness the potential of LLMs for HP. To this end, we present a taxonomy of integration methods, exploring how LLMs can be utilized within the HP life cycle. Additionally, we provide a benchmark with a standardized dataset for evaluating the performance of future LLM-based HP approaches, and present initial results for a state-of-the-art HP planner and LLM planner. As expected, the latter exhibits limited performance (3\% correct plans, and none with a correct hierarchical decomposition) but serves as a valuable baseline for future approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18165v1">Towards a General Framework for HTN Modeling with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 10 pages, 5 figures, to be published in the Workshop on Planning in the Era of LLMs ( LM4Plan - https://llmforplanning.github.io ) and the Workshop on Hierarchical Planning ( HPlan - https://icaps25.icaps-conference.org/program/workshops/hplan/ ), both in the International Conference on Automated Planning and Scheduling (ICAPS) 2025
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) for generating Automated Planning (AP) models has been widely explored; however, their application to Hierarchical Planning (HP) is still far from reaching the level of sophistication observed in non-hierarchical architectures. In this work, we try to address this gap. We present two main contributions. First, we propose L2HP, an extension of L2P (a library to LLM-driven PDDL models generation) that support HP model generation and follows a design philosophy of generality and extensibility. Second, we apply our framework to perform experiments where we compare the modeling capabilities of LLMs for AP and HP. On the PlanBench dataset, results show that parsing success is limited but comparable in both settings (around 36\%), while syntactic validity is substantially lower in the hierarchical case (1\% vs. 20\% of instances). These findings underscore the unique challenges HP presents for LLMs, highlighting the need for further research to improve the quality of generated HP models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11425v2">Tapas Are Free! Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Extended version of the paper accepted at AAAI-26 Oral to comply with the AAAI camera-ready requirements with minor revisions
    </div>
    <details class="paper-abstract">
      Autonomous agents in safety-critical applications must continuously adapt to dynamic conditions without compromising performance and reliability. This work introduces TAPA (Training-free Adaptation of Programmatic Agents), a novel framework that positions large language models (LLMs) as intelligent moderators of the symbolic action space. Unlike prior programmatic agents typically generate a monolithic policy program or rely on fixed symbolic action sets, TAPA synthesizes and adapts modular programs for individual high-level actions, referred to as logical primitives. By decoupling strategic intent from execution, TAPA enables meta-agents to operate over an abstract, interpretable action space while the LLM dynamically generates, composes, and refines symbolic programs tailored to each primitive. Extensive experiments across cybersecurity and swarm intelligence domains validate TAPA's effectiveness. In autonomous DDoS defense scenarios, TAPA achieves 77.7% network uptime while maintaining near-perfect detection accuracy in unknown dynamic environments. In swarm intelligence formation control under environmental and adversarial disturbances, TAPA consistently preserves consensus at runtime where baseline methods fail. This work promotes a paradigm shift for autonomous system design in evolving environments, from policy adaptation to dynamic action adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16466v4">Incalmo: An Autonomous LLM-assisted System for Red Teaming Multi-Host Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Security operators use red teams to simulate real attackers and proactively find defense gaps. In realistic enterprise settings, this involves executing multi-host network attacks spanning many "stepping stone" hosts. Unfortunately, red teams are expensive and entail significant expertise and effort. Given the promise of LLMs in CTF challenges, we first analyze if LLMs can autonomously execute multi-host red team exercises. We find that state-of-the-art LLM-assisted offense systems (e.g., PentestGPT, CyberSecEval3) with leading LLMs (e.g., Sonnet 4, Gemini 2.5 Pro) are unable to do so. Building on our observations in understanding the failure modes of state-of-the-art systems, we argue the need to improve the abstractions and interfaces for LLM-assisted red teaming. Based on this insight, we present the design and implementation of Incalmo, an LLM-assisted system for autonomously red teaming multi-host networks. Incalmo uses LLMs to plan red team exercises in terms of high-level declarative tasks that are executed by domain-specific task agents. Incalmo also uses auxiliary services to manage context and acquired assets. For our evaluation, we develop MHBench, a novel multi-host attack benchmark with 40 realistic emulated networks (from 22 to 50 hosts). We find that Incalmo successfully acquires critical assets (i.e., key hosts or data) in 37 out of 40 MHBench environments. In contrast, state-of-the-art LLM-assisted systems succeed in only 3 out of 40 environments. We show that Incalmo is efficient-successful attacks took 12-54 minutes and cost <$15 in LLM credits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12509v3">Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Verifying the complex and multi-step reasoning of Large Language Models (LLMs) is a critical challenge, as holistic methods often overlook localized flaws. Step-by-step validation is a promising alternative, yet existing methods are often rigid. They struggle to adapt to diverse reasoning structures, from formal proofs to informal natural language narratives. To address this adaptability gap, we propose the Graph of Verification (GoV), a novel framework for adaptable and multi-granular verification. GoV's core innovation is its flexible "node block" architecture. This mechanism allows GoV to adaptively adjust its verification granularity--from atomic steps for formal tasks to entire paragraphs for natural language--to match the native structure of the reasoning process. This flexibility allows GoV to resolve the fundamental trade-off between verification precision and robustness. Experiments on both well-structured and loosely-structured benchmarks demonstrate GoV's versatility. The results show that GoV's adaptive approach significantly outperforms both holistic baselines and other state-of-the-art decomposition-based methods, establishing a new standard for training-free reasoning verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18098v1">Towards Harnessing the Power of LLMs for ABAC Policy Mining</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      This paper presents an empirical investigation into the capabilities of Large Language Models (LLMs) to perform automated Attribute-based Access Control (ABAC) policy mining. While ABAC provides fine-grained, context-aware access management, the increasing number and complexity of access policies can make their formulation and evaluation rather challenging. To address the task of synthesizing concise yet accurate policies, we evaluate the performance of some of the state-of-the-art LLMs, specifically Google Gemini (Flash and Pro) and OpenAI ChatGPT, as potential policy mining engines. An experimental framework was developed in Python to generate randomized access data parameterized by varying numbers of subjects, objects, and initial policy sets. The baseline policy sets, which govern permission decisions between subjects and objects, serve as the ground truth for comparison. Each LLM-generated policy was evaluated against the baseline policy using standard performance metrics. The results indicate that LLMs can effectively infer compact and valid ABAC policies for small-scale scenarios. However, as the system size increases, characterized by higher numbers of subjects and objects, LLM outputs exhibit declining accuracy and precision, coupled with significant increase in the size of policy generated, which is beyond the optimal size. These findings highlight both the promise and limitations of current LLM architectures for scalable policy mining in access control domains. Future work will explore hybrid approaches that combine prompt optimization with classical rule mining algorithms to improve scalability and interpretability in complex ABAC environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04440v2">StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 AAAI 2026 Oral. Extended version with full appendix, 25 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00413v2">Tree Training: Accelerating Agentic LLMs Training via Shared Prefix Reuse</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      In agentic LLM scenarios, an agent's interaction process during a single rollout often exhibits branching behaviors. Due to memory retrieval and concurrent tool executions at certain decision points, the token trajectory of one task evolves into a tree-like structure rather than a linear sequence. However, current training pipelines decompose such tree-structured trajectories into separate linear segments, treating each branch as an independent sequence. As a result, shared prefixes across these branches are repeatedly recomputed during both forward and backward passes. To address this inefficiency, we propose Tree Training, a paradigm that computes each shared prefix only once and reuses its intermediate results across related branches during both forward and backward passes, substantially improving computation efficiency in large-scale agentic training. This is achieved via (i) Tree Packing, which efficiently reuses shared computations across trajectories, and (ii) Gradient Restoration, which ensures correct gradient propagation across reused prefixes. Experiments on multiple open-source models demonstrate up to 3.9x reduction in total training time, enabling more efficient agentic LLM SFT and RL training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18039v1">Curvature-Aware Safety Restoration In LLMs Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 19 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for downstream tasks often compromises safety alignment, even when using parameter-efficient methods like LoRA. In this work, we uncover a notable property: fine-tuned models preserve the geometric structure of their loss landscapes concerning harmful content, regardless of the fine-tuning method employed. This suggests that safety behaviors are not erased but shifted to less influential regions of the parameter space. Building on this insight, we propose a curvature-aware alignment restoration method that leverages influence functions and second-order optimization to selectively increase loss on harmful inputs while preserving task performance. By navigating the shared geometry between base and fine-tuned models, our method discourages unsafe outputs while preserving task-relevant performance, avoiding full reversion and enabling precise, low-impact updates. Extensive evaluations across multiple model families and adversarial settings show that our approach efficiently reduces harmful responses while maintaining or even improving utility and few-shot learning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18038v1">MASTEST: A LLM-Based Multi-Agent System For RESTful API Tests</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 14 Page of main text plus 4 pages of appendix
    </div>
    <details class="paper-abstract">
      Testing RESTful API is increasingly important in quality assurance of cloud-native applications. Recent advances in machine learning (ML) techniques have demonstrated that various testing activities can be performed automatically by large language models (LLMs) with reasonable accuracy. This paper develops a multi-agent system called MASTEST that combines LLM-based and programmed agents to form a complete tool chain that covers the whole workflow of API test starting from generating unit and system test scenarios from API specification in the OpenAPI Swagger format, to generating of Pytest test scripts, executing test scripts to interact with web services, to analysing web service response messages to determine test correctness and calculate test coverage. The system also supports the incorporation of human testers in reviewing and correcting LLM generated test artefacts to ensure the quality of testing activities. MASTEST system is evaluated on two LLMs, GPT-4o and DeepSeek V3.1 Reasoner with five public APIs. The performances of LLMs on various testing activities are measured by a wide range of metrics, including unit and system test scenario coverage and API operation coverage for the quality of generated test scenarios, data type correctness, status code coverage and script syntax correctness for the quality of LLM generated test scripts, as well as bug detection ability and usability of LLM generated test scenarios and scripts. Experiment results demonstrated that both DeepSeek and GPT-4o achieved a high overall performance. DeepSeek excels in data type correctness and status code detection, while GPT-4o performs best in API operation coverage. For both models, LLM generated test scripts maintained 100\% syntax correctness and only required minimal manual edits for semantic correctness. These findings indicate the effectiveness and feasibility of MASTEST.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17990v1">How Far Can LLMs Emulate Human Behavior?: A Strategic Analysis via the Buy-and-Sell Negotiation Game</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      With the rapid advancement of Large Language Models (LLMs), recent studies have drawn attention to their potential for handling not only simple question-answer tasks but also more complex conversational abilities and performing human-like behavioral imitations. In particular, there is considerable interest in how accurately LLMs can reproduce real human emotions and behaviors, as well as whether such reproductions can function effectively in real-world scenarios. However, existing benchmarks focus primarily on knowledge-based assessment and thus fall short of sufficiently reflecting social interactions and strategic dialogue capabilities. To address these limitations, this work proposes a methodology to quantitatively evaluate the human emotional and behavioral imitation and strategic decision-making capabilities of LLMs by employing a Buy and Sell negotiation simulation. Specifically, we assign different personas to multiple LLMs and conduct negotiations between a Buyer and a Seller, comprehensively analyzing outcomes such as win rates, transaction prices, and SHAP values. Our experimental results show that models with higher existing benchmark scores tend to achieve better negotiation performance overall, although some models exhibit diminished performance in scenarios emphasizing emotional or social contexts. Moreover, competitive and cunning traits prove more advantageous for negotiation outcomes than altruistic and cooperative traits, suggesting that the assigned persona can lead to significant variations in negotiation strategies and results. Consequently, this study introduces a new evaluation approach for LLMs' social behavior imitation and dialogue strategies, and demonstrates how negotiation simulations can serve as a meaningful complementary metric to measure real-world interaction capabilities-an aspect often overlooked in existing benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.15018v2">Using tournaments to calculate AUROC for zero-shot classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025, Findings). The code is available at: https://github.com/Machine-Learning-for-Medical-Language/cnlp_llm
    </div>
    <details class="paper-abstract">
      Large language models perform surprisingly well on many zero-shot classification tasks, but are difficult to fairly compare to supervised classifiers due to the lack of a modifiable decision boundary. In this work, we propose and evaluate a method that transforms binary classification tasks into pairwise comparisons between instances within a dataset, using LLMs to produce relative rankings of those instances. Repeated pairwise comparisons can be used to score instances using the Elo rating system (used in chess and other competitions), inducing a confidence ordering over instances in a dataset. We evaluate scheduling algorithms for their ability to minimize comparisons, and show that our proposed algorithm leads to improved classification performance, while also providing more information than traditional zero-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.18811v2">DCIS: Efficient Length Extrapolation of LLMs via Divide-and-Conquer Scaling Factor Search</a></div>
    <div class="paper-meta">
      📅 2025-11-22
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based on the Transformer architecture usually have their context length limited due to the high training cost. Recent advancements extend the context window by adjusting the scaling factors of RoPE and fine-tuning. However, suboptimal initialization of these factors results in increased fine-tuning costs and reduced performance at target length. To address these challenges, we propose a novel RoPE-based fine-tuning framework that diverges from conventional scaling factors search. Specifically, we present a \textbf{D}ivide-and-\textbf{C}onquer \textbf{I}ncremental \textbf{S}earch (DCIS) algorithm that strategically determines the better scaling factors. Further fine-tuning with the identified scaling factors effectively extends the context window of LLMs. Empirical results demonstrate that our methodology not only mitigates performance decay at extended target lengths but also allows the model to fine-tune on short contexts and generalize to long contexts, thereby reducing the cost of fine-tuning. The scaling factors obtained through DCIS can even perform effectively without fine-tuning. Further analysis of the search space reveals that DCIS achieves twice the search efficiency compared to other methods. We also examine the impact of the non-strictly increasing scaling factors utilized in DCIS and evaluate the general capabilities of LLMs across various context lengths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17947v1">Leveraging Evidence-Guided LLMs to Enhance Trustworthy Depression Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in automating clinical diagnosis, yet their non-transparent decision-making and limited alignment with diagnostic standards hinder trust and clinical adoption. We address this challenge by proposing a two-stage diagnostic framework that enhances transparency, trustworthiness, and reliability. First, we introduce Evidence-Guided Diagnostic Reasoning (EGDR), which guides LLMs to generate structured diagnostic hypotheses by interleaving evidence extraction with logical reasoning grounded in DSM-5 criteria. Second, we propose a Diagnosis Confidence Scoring (DCS) module that evaluates the factual accuracy and logical consistency of generated diagnoses through two interpretable metrics: the Knowledge Attribution Score (KAS) and the Logic Consistency Score (LCS). Evaluated on the D4 dataset with pseudo-labels, EGDR outperforms direct in-context prompting and Chain-of-Thought (CoT) across five LLMs. For instance, on OpenBioLLM, EGDR improves accuracy from 0.31 (Direct) to 0.76 and increases DCS from 0.50 to 0.67. On MedLlama, DCS rises from 0.58 (CoT) to 0.77. Overall, EGDR yields up to +45% accuracy and +36% DCS gains over baseline methods, offering a clinically grounded, interpretable foundation for trustworthy AI-assisted diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19480v1">Exploiting the Experts: Unauthorized Compression in MoE-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-22
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures are increasingly adopted in large language models (LLMs) for their scalability and efficiency. However, their modular structure introduces a unique vulnerability: adversaries can attempt to compress or repurpose models by pruning experts and cheaply fine-tuning the remainder, effectively bypassing licensing and security constraints. In this paper, we systematically study the prunability of MoE-LLMs under task-specific usage. We first develop an expert attribution framework that identifies the subset of experts most responsible for a given task, then evaluate the performance trade-offs of pruning and re-aligning these experts using active learning-driven fine-tuning. Our findings reveal a critical knowledge loss--recovery trade-off: while certain experts can be isolated to retain task accuracy, significant degradation occurs without targeted re-alignment. Based on this analysis, we propose defense strategies that aim to make MoE models harder to compress and fine-tune without authorization, including entangled expert training and selective fine-tuning protocols that resist unauthorized adaptation. By positioning expert pruning as both a threat vector and a defense target, this work highlights the dual-use nature of MoE modularity and provides the first systematic evaluation framework for secure specialization of MoE-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17467v1">PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at https://anonymous.4open.science/r/PersonaAgentwGraphRAG-DE6F
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00086v2">Do LLMs produce texts with "human-like" lexical diversity?</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The degree to which large language models (LLMs) produce writing that is truly human-like remains unclear despite the extensive empirical attention that this question has received. The present study addresses this question from the perspective of lexical diversity. Specifically, the study investigates patterns of lexical diversity in LLM-generated texts from four ChatGPT models (ChatGPT-3.5, ChatGPT-4, ChatGPT-o4 mini, and ChatGPT-4.5) in comparison with texts written by L1 and L2 English participants (n = 240) across four education levels. Six dimensions of lexical diversity were measured in each text: volume, abundance, variety-repetition, evenness, disparity, and dispersion. Results from one-way MANOVAs, one-way ANOVAs, and Support Vector Machines revealed that the ChatGPT-generated texts differed significantly from human-written texts for each variable, with ChatGPT-o4 mini and ChatGPT-4.5 differing the most. Within these two groups, ChatGPT-4.5 demonstrated higher levels of lexical diversity than older models despite producing fewer tokens. The human writers' lexical diversity did not differ across subgroups (i.e., education, language status). Altogether, the results indicate that ChatGPT models do not produce human-like texts in relation to lexical diversity, and the newer models produce less human-like text than older models. We discuss the implications of these results for language pedagogy and related applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17442v1">REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Code and data available at https://github.com/be-chen/REMSA
    </div>
    <details class="paper-abstract">
      Foundation Models (FMs) are increasingly used in remote sensing (RS) for tasks such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained on a single data modality and multimodal architectures trained on combinations of SAR, multispectral, hyperspectral, and image-text data. They support diverse RS tasks including semantic segmentation, image classification, change detection, and visual question answering. However, selecting an appropriate remote sensing foundation model (RSFM) remains difficult due to scattered documentation, heterogeneous formats, and varied deployment constraints. We introduce the RSFM Database (RS-FMD), a structured resource covering over 150 RSFMs spanning multiple data modalities, resolutions, and learning paradigms. Built on RS-FMD, we present REMSA, the first LLM-based agent for automated RSFM selection from natural language queries. REMSA interprets user requirements, resolves missing constraints, ranks candidate models using in-context learning, and provides transparent justifications. We also propose a benchmark of 75 expert-verified RS query scenarios, producing 900 configurations under an expert-centered evaluation protocol. REMSA outperforms several baselines, including naive agents, dense retrieval, and unstructured RAG-based LLMs. It operates entirely on publicly available metadata and does not access private or sensitive data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09439v3">Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU and MMAR benchmarks. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17335v1">Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Accepted to ASRU 2025
    </div>
    <details class="paper-abstract">
      Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.
    </details>
</div>
