# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23143v3">MathBode: Understanding LLM Reasoning with Dynamical Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      This paper presents MathBode, a dynamic diagnostic for mathematical reasoning in large language models (LLMs). Instead of one-shot accuracy, MathBode treats each parametric problem as a system: we drive a single parameter sinusoidally and fit first-harmonic responses of model outputs and exact solutions. This yields interpretable, frequency-resolved metrics -- gain (amplitude tracking) and phase (lag) -- that form Bode-style fingerprints. Across five closed-form families (linear solve, ratio/saturation, compound interest, 2x2 linear systems, similar triangles), the diagnostic surfaces systematic low-pass behavior and growing phase lag that accuracy alone obscures. We compare several models against a symbolic baseline that calibrates the instrument ($G \approx 1$, $\phi \approx 0$). Results separate frontier from mid-tier models on dynamics, providing a compact, reproducible protocol that complements standard benchmarks with actionable measurements of reasoning fidelity and consistency. We open-source the dataset and code to enable further research and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24073v1">Challenging Multilingual LLMs: A New Taxonomy and Benchmark for Unraveling Hallucination in Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced machine translation but remain vulnerable to hallucinations. Unfortunately, existing MT benchmarks are not capable of exposing failures in multilingual LLMs. To disclose hallucination in multilingual LLMs, we introduce a diagnostic framework with a taxonomy that separates Instruction Detachment from Source Detachment. Guided by this taxonomy, we create HalloMTBench, a multilingual, human-verified benchmark across 11 English-to-X directions. We employed 4 frontier LLMs to generate candidates and scrutinize these candidates with an ensemble of LLM judges, and expert validation. In this way, we curate 5,435 high-quality instances. We have evaluated 17 LLMs on HalloMTBench. Results reveal distinct ``hallucination triggers'' -- unique failure patterns reflecting model scale, source length sensitivity, linguistic biases, and Reinforcement-Learning (RL) amplified language mixing. HalloMTBench offers a forward-looking testbed for diagnosing LLM translation failures. HalloMTBench is available in https://huggingface.co/collections/AIDC-AI/marco-mt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24085v2">PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      We present PEARL (Peer-Enhanced Adaptive Radio via On-Device LLM), a framework for cooperative cross-layer optimization in device-to-device (D2D) communication. Building on our previous work on single-device on-device LLMs, PEARL extends the paradigm by leveraging both publisher and subscriber states to guide Wi-Fi Aware (WA) parameter selection. A context-aware reward, which normalizes latency by application tolerances and modulates energy by device battery states, provides richer supervision for KL-based finetuning. We study two lightweight variants: PEARL (Head + Low-Rank Adaptation (LoRA)) achieves the best overall performance, while PEARL-Lite (Head-only) delivers sub-20 ms inference at near-identical objective scores. Across synthetic scenarios grounded in real measurements, PEARL improves objective scores over heuristic and compact model baselines and reduces energy by up to 16% in cooperative low-battery cases. These results demonstrate that peer-aware context, reward-aligned training, and head-based efficiency make LLMs practical for always-on, on-device cross-layer control. Code, real-world demo, and dataset are available at https://github.com/abman23/pearl
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02885v2">MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 26 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Video large language models (Video-LLMs) have made significant progress in understanding videos. However, processing multiple frames leads to lengthy visual token sequences, presenting challenges such as the limited context length cannot accommodate the entire video, and the inclusion of irrelevant frames hinders visual perception. Hence, effective frame selection is crucial. This paper emphasizes that frame selection should follow three key principles: query relevance, list-wise diversity, and sequentiality. Existing methods, such as uniform frame sampling and query-frame matching, do not capture all of these principles. Thus, we propose Markov decision determinantal point process with dynamic programming (MDP3) for frame selection, a training-free and model-agnostic method that can be seamlessly integrated into existing Video-LLMs. Our method first estimates frame similarities conditioned on the query using a conditional Gaussian kernel within the reproducing kernel Hilbert space~(RKHS). We then apply the determinantal point process~(DPP) to the similarity matrix to capture both query relevance and list-wise diversity. To incorporate sequentiality, we segment the video and apply DPP within each segment, conditioned on the preceding segment selection, modeled as a Markov decision process~(MDP) for allocating selection sizes across segments. Theoretically, MDP3 provides a \((1 - 1/e)\)-approximate solution to the NP-hard list-wise frame selection problem with pseudo-polynomial time complexity, demonstrating its efficiency. Empirically, MDP3 significantly outperforms existing methods, verifying its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20445v5">TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted by NeurIPS 2025, https://github.com/tsinghua-fib-lab/TrajAgent
    </div>
    <details class="paper-abstract">
      Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. In this paper, we propose TrajAgent, an agent framework powered by large language models, designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. In TrajAgent, we first develop UniEnv, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on UniEnv, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on five tasks using four real-world datasets demonstrate the effectiveness of TrajAgent in automated trajectory modeling, achieving a performance improvement of 2.38%-69.91% over baseline methods. The codes and data can be accessed via https://github.com/tsinghua-fib-lab/TrajAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24051v1">Pie: A Programmable Serving System for Emerging LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 SOSP 2025. Source code available at https://github.com/pie-project/pie
    </div>
    <details class="paper-abstract">
      Emerging large language model (LLM) applications involve diverse reasoning strategies and agentic workflows, straining the capabilities of existing serving systems built on a monolithic token generation loop. This paper introduces Pie, a programmable LLM serving system designed for flexibility and efficiency. Pie decomposes the traditional generation loop into fine-grained service handlers exposed via an API and delegates control of the generation process to user-provided programs, called inferlets. This enables applications to implement new KV cache strategies, bespoke generation logic, and seamlessly integrate computation and I/O-entirely within the application, without requiring modifications to the serving system. Pie executes inferlets using WebAssembly, benefiting from its lightweight sandboxing. Our evaluation shows Pie matches state-of-the-art performance on standard tasks (3-12% latency overhead) while significantly improving latency and throughput (1.3x-3.4x higher) on agentic workflows by enabling application-specific optimizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17281v2">MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05316v3">Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism inspired by experience replay in traditional RL. This technique reuses recent rollouts, lowering per-step computation while maintaining stable updates. Experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 23% to 62% while reaching the same level of performance as the original GRPO algorithm. Our code is available at https://github.com/ASTRAL-Group/data-efficient-llm-rl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24034v1">AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in text-to-image (T2I) models, their safety mechanisms are vulnerable to adversarial prompts, which maliciously generate unsafe images. Current red-teaming methods for proactively assessing such vulnerabilities usually require white-box access to T2I models, and rely on inefficient per-prompt optimization, as well as inevitably generate semantically meaningless prompts easily blocked by filters. In this paper, we propose APT (AutoPrompT), a black-box framework that leverages large language models (LLMs) to automatically generate human-readable adversarial suffixes for benign prompts. We first introduce an alternating optimization-finetuning pipeline between adversarial suffix optimization and fine-tuning the LLM utilizing the optimized suffix. Furthermore, we integrates a dual-evasion strategy in optimization phase, enabling the bypass of both perplexity-based filter and blacklist word filter: (1) we constrain the LLM generating human-readable prompts through an auxiliary LLM perplexity scoring, which starkly contrasts with prior token-level gibberish, and (2) we also introduce banned-token penalties to suppress the explicit generation of banned-tokens in blacklist. Extensive experiments demonstrate the excellent red-teaming performance of our human-readable, filter-resistant adversarial prompts, as well as superior zero-shot transferability which enables instant adaptation to unseen prompts and exposes critical vulnerabilities even in commercial APIs (e.g., Leonardo.Ai.).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23595v2">Multi-Agent Evolve: LLM Self-Improve through Co-evolution</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 29 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24021v1">SpecKD: Speculative Decoding for Effective Knowledge Distillation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Knowledge Distillation (KD) has become a cornerstone technique for compressing Large Language Models (LLMs) into smaller, more efficient student models. However, conventional KD approaches typically apply the distillation loss uniformly across all tokens, regardless of the teacher's confidence. This indiscriminate mimicry can introduce noise, as the student is forced to learn from the teacher's uncertain or high-entropy predictions, which may ultimately harm student performance-especially when the teacher is much larger and more powerful. To address this, we propose Speculative Knowledge Distillation (SpecKD), a novel, plug-and-play framework that introduces a dynamic, token-level gating mechanism inspired by the "propose-and-verify" paradigm of speculative decoding. At each step, the student's token proposal is verified against the teacher's distribution; the distillation loss is selectively applied only to "accepted" tokens, while "rejected" tokens are masked out. Extensive experiments on diverse text generation tasks show that SpecKD consistently and significantly outperforms strong KD baselines, leading to more stable training and more capable student models, and achieving state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24020v1">Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 23pages, 4figures
    </div>
    <details class="paper-abstract">
      Mitigating hallucinations in Large Language Models (LLMs) is critical for their reliable deployment. Existing methods typically fine-tune LLMs to abstain from answering questions beyond their knowledge scope. However, these methods often rely on coarse-grained signals to guide LLMs to abstain, such as overall confidence or uncertainty scores on multiple sampled answers, which may result in an imprecise awareness of the model's own knowledge boundaries. To this end, we propose a novel reinforcement learning framework built on $\textbf{\underline{Fi}ne-grained \underline{S}emantic \underline{Co}nfidence \underline{Re}ward (\Ours)}$, which guides LLMs to abstain via sample-specific confidence. Specifically, our method operates by sampling multiple candidate answers and conducting semantic clustering, then training the LLM to retain answers within high-confidence clusters and discard those within low-confidence ones, thereby promoting accurate post-hoc abstention. Additionally, we propose a new metric for evaluating the reliability of abstention fine-tuning tasks more comprehensively. Our method significantly enhances reliability in both in-domain and out-of-distribution benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24019v1">Lifecycle-Aware code generation: Leveraging Software Engineering Phases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has advanced automatic code generation, yet most approaches rely on direct, single-step translation from problem descriptions to code, disregarding structured software engineering practices. We introduce a lifecycle-aware framework that systematically incorporates intermediate artifacts such as requirements analysis, state machine modeling, and pseudocode into both the training and inference stages. This design aligns code generation with standard software development phases and enables more structured reasoning. Experiments show that lifecycle-level fine-tuning improves code correctness by up to 75% over the same model before fine-tuning, with performance gains compounding across intermediate stages. Multi-step inference consistently surpasses single-step generation, demonstrating the effectiveness of intermediate scaffolding. Notably, open-source LLMs, once fine-tuned under our framework, match or slightly outperform models pretrained on code. When applied to DeepSeek-Coder-1.3B, our framework yields relative CodeBLEU improvements of 34.3%, 20.0%, 11.2%, and 22.3% over ChatGPT-3.5, ChatGPT-4o-mini, DeepSeek-R1, and LLaMA-8B, respectively. Our pipeline also proves robust with up to 80\% less training data, confirming its resilience. Ablation studies further reveal that each intermediate artifact contributes distinctly to final code quality, with state machine modeling yielding the most substantial impact. Our source code and detailed experimental data are available at https://anonymous.4open.science/r/Lifecycle-Aware-3CCB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19563v3">Robustness is Important: Limitations of LLMs for Data Fitting</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being applied in a wide array of settings, well beyond the typical language-oriented use cases. In particular, LLMs are increasingly used as a plug-and-play method for fitting data and generating predictions. Prior work has shown that LLMs, via in-context learning or supervised fine-tuning, can perform competitively with many tabular supervised learning techniques in terms of predictive performance. However, we identify a critical vulnerability of using LLMs for data fitting -- making changes to data representation that are completely irrelevant to the underlying learning task can drastically alter LLMs' predictions on the same data. For example, simply changing variable names can sway the size of prediction error by as much as 82% in certain settings. Such prediction sensitivity with respect to task-irrelevant variations manifests under both in-context learning and supervised fine-tuning, for both close-weight and open-weight general-purpose LLMs. Moreover, by examining the attention scores of an open-weight LLM, we discover a non-uniform attention pattern: training examples and variable names/values which happen to occupy certain positions in the prompt receive more attention when output tokens are generated, even though different positions are expected to receive roughly the same attention. This partially explains the sensitivity in the presence of task-irrelevant variations. We also consider a state-of-the-art tabular foundation model (TabPFN) trained specifically for data fitting. Despite being explicitly designed to achieve prediction robustness, TabPFN is still not immune to task-irrelevant variations. Overall, despite LLMs' impressive predictive capabilities, currently they lack even the basic level of robustness to be used as a principled data-fitting tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24013v1">Discovering Heuristics with Large Language Models (LLMs) for Mixed-Integer Programs: Single-Machine Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Our study contributes to the scheduling and combinatorial optimization literature with new heuristics discovered by leveraging the power of Large Language Models (LLMs). We focus on the single-machine total tardiness (SMTT) problem, which aims to minimize total tardiness by sequencing n jobs on a single processor without preemption, given processing times and due dates. We develop and benchmark two novel LLM-discovered heuristics, the EDD Challenger (EDDC) and MDD Challenger (MDDC), inspired by the well-known Earliest Due Date (EDD) and Modified Due Date (MDD) rules. In contrast to prior studies that employed simpler rule-based heuristics, we evaluate our LLM-discovered algorithms using rigorous criteria, including optimality gaps and solution time derived from a mixed-integer programming (MIP) formulation of SMTT. We compare their performance against state-of-the-art heuristics and exact methods across various job sizes (20, 100, 200, and 500 jobs). For instances with more than 100 jobs, exact methods such as MIP and dynamic programming become computationally intractable. Up to 500 jobs, EDDC improves upon the classic EDD rule and another widely used algorithm in the literature. MDDC consistently outperforms traditional heuristics and remains competitive with exact approaches, particularly on larger and more complex instances. This study shows that human-LLM collaboration can produce scalable, high-performing heuristics for NP-hard constrained combinatorial optimization, even under limited resources when effectively configured.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23008v2">From Prompt Optimization to Multi-Dimensional Credibility Evaluation: Enhancing Trustworthiness of Chinese LLM-Generated Liver MRI Reports</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 10 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising performance in generating diagnostic conclusions from imaging findings, thereby supporting radiology reporting, trainee education, and quality control. However, systematic guidance on how to optimize prompt design across different clinical contexts remains underexplored. Moreover, a comprehensive and standardized framework for assessing the trustworthiness of LLM-generated radiology reports is yet to be established. This study aims to enhance the trustworthiness of LLM-generated liver MRI reports by introducing a Multi-Dimensional Credibility Assessment (MDCA) framework and providing guidance on institution-specific prompt optimization. The proposed framework is applied to evaluate and compare the performance of several advanced LLMs, including Kimi-K2-Instruct-0905, Qwen3-235B-A22B-Instruct-2507, DeepSeek-V3, and ByteDance-Seed-OSS-36B-Instruct, using the SiliconFlow platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23990v1">Resource-Efficient LLM Application for Structured Transformation of Unstructured Financial Contracts</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 5 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      The transformation of unstructured legal contracts into standardized, machine-readable formats is essential for automating financial workflows. The Common Domain Model (CDM) provides a standardized framework for this purpose, but converting complex legal documents like Credit Support Annexes (CSAs) into CDM representations remains a significant challenge. In this paper, we present an extension of the CDMizer framework, a template-driven solution that ensures syntactic correctness and adherence to the CDM schema during contract-to-CDM conversion. We apply this extended framework to a real-world task, comparing its performance with a benchmark developed by the International Swaps and Derivatives Association (ISDA) for CSA clause extraction. Our results show that CDMizer, when integrated with a significantly smaller, open-source Large Language Model (LLM), achieves competitive performance in terms of accuracy and efficiency against larger, proprietary models. This work underscores the potential of resource-efficient solutions to automate legal contract transformation, offering a cost-effective and scalable approach that can meet the needs of financial institutions with constrained resources or strict data privacy requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23965v1">The Sign Estimator: LLM Alignment in the Face of Choice Heterogeneity</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Traditional LLM alignment methods are vulnerable to heterogeneity in human preferences. Fitting a na\"ive probabilistic model to pairwise comparison data (say over prompt-completion pairs) yields an inconsistent estimate of the population-average utility -a canonical measure of social welfare. We propose a new method, dubbed the sign estimator, that provides a simple, provably consistent, and efficient estimator by replacing cross-entropy with binary classification loss in the aggregation step. This simple modification recovers consistent ordinal alignment under mild assumptions and achieves the first polynomial finite-sample error bounds in this setting. In realistic simulations of LLM alignment using digital twins, the sign estimator substantially reduces preference distortion over a panel of simulated personas, cutting (angular) estimation error by nearly 35% and decreasing disagreement with true population preferences from 12% to 8% compared to standard RLHF. Our method also compares favorably to panel data heuristics that explicitly model user heterogeneity and require tracking individual-level preference data-all while maintaining the implementation simplicity of existing LLM alignment pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23949v1">Uncovering the Potential Risks in Unlearning: Danger of English-only Unlearning in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      There have been a couple of studies showing that attempting to erase multilingual knowledge using only English data is insufficient for multilingual LLMs. However, their analyses remain highly performance-oriented. In this paper, we switch the point of view to evaluation, and address an additional blind spot which reveals itself when the multilingual LLM is fully finetuned with parallel multilingual dataset before unlearning. Here, language confusion occurs whereby a model responds in language different from that of the input prompt. Language confusion is a problematic phenomenon in unlearning, causing the standard reference-based metrics to fail. We tackle this phenomenon in three steps: (1) introduce N-gram-based Language-Mix (N-Mix) score to quantitatively show the language confusion is pervasive and consistent in multilingual LLMs, (2) demonstrate that reference-based metrics result in false negatives when N-Mix score is high, and(3) suggest the need of new type of unlearning evaluation that can directly assess the content of the generated sentences. We call this type of metrics as semantic-based metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23947v1">Toward Socially-Aware LLMs: A Survey of Multimodal Approaches to Human Behavior Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      LLM-powered multimodal systems are increasingly used to interpret human social behavior, yet how researchers apply the models' 'social competence' remains poorly understood. This paper presents a systematic literature review of 176 publications across different application domains (e.g., healthcare, education, and entertainment). Using a four-dimensional coding framework (application, technical, evaluative, and ethical), we find (1) frequent use of pattern recognition and information extraction from multimodal sources, but limited support for adaptive, interactive reasoning; (2) a dominant 'modality-to-text' pipeline that privileges language over rich audiovisual cues, striping away nuanced social cues; (3) evaluation practices reliant on static benchmarks, with socially grounded, human-centered assessments rare; and (4) Ethical discussions focused mainly on legal and rights-related risks (e.g., privacy), leaving societal risks (e.g., deception) overlooked--or at best acknowledged but left unaddressed. We outline a research agenda for evaluating socially competent, ethically informed, and interaction-aware multi-modal systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00418v3">Serving LLMs in HPC Clusters: A Comparative Study of Qualcomm Cloud AI 100 Ultra and NVIDIA Data Center GPUs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 8 pages, 3 tables
    </div>
    <details class="paper-abstract">
      This study presents a benchmarking analysis of the Qualcomm Cloud AI 100 Ultra (QAic) accelerator for large language model (LLM) inference, evaluating its energy efficiency (throughput per watt), performance, and hardware scalability against NVIDIA A100 GPUs (in 4x and 8x configurations) within the National Research Platform (NRP) ecosystem. A total of 12 open-source LLMs, ranging from 124 million to 70 billion parameters, are served using the vLLM framework. Our analysis reveals that QAic achieves competitive energy efficiency with advantages on specific models while enabling more granular hardware allocation: some 70B models operate on as few as 1 QAic card versus 8 A100 GPUs required, with 20x lower power consumption (148W vs 2,983W). For smaller models, single QAic devices achieve up to 35x lower power consumption compared to our 4-GPU A100 configuration (36W vs 1,246W). The findings offer insights into the potential of the Qualcomm Cloud AI 100 Ultra for energy-constrained and resource-efficient HPC deployments within the National Research Platform (NRP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25017v1">StorageXTuner: An LLM Agent-Driven Automatic Tuning Framework for Heterogeneous Storage Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 ArXiv version; Affiliations: Arizona State University (Lin, Zhang, Thakkar, Sun, Cao) and Iowa State University (Zheng)
    </div>
    <details class="paper-abstract">
      Automatically configuring storage systems is hard: parameter spaces are large and conditions vary across workloads, deployments, and versions. Heuristic and ML tuners are often system specific, require manual glue, and degrade under changes. Recent LLM-based approaches help but usually treat tuning as a single-shot, system-specific task, which limits cross-system reuse, constrains exploration, and weakens validation. We present StorageXTuner, an LLM agent-driven auto-tuning framework for heterogeneous storage engines. StorageXTuner separates concerns across four agents - Executor (sandboxed benchmarking), Extractor (performance digest), Searcher (insight-guided configuration exploration), and Reflector (insight generation and management). The design couples an insight-driven tree search with layered memory that promotes empirically validated insights and employs lightweight checkers to guard against unsafe actions. We implement a prototype and evaluate it on RocksDB, LevelDB, CacheLib, and MySQL InnoDB with YCSB, MixGraph, and TPC-H/C. Relative to out-of-the-box settings and to ELMo-Tune, StorageXTuner reaches up to 575% and 111% higher throughput, reduces p99 latency by as much as 88% and 56%, and converges with fewer trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05583v3">OpenFactCheck: Building, Benchmarking Customized Fact-Checking Systems and Evaluating the Factuality of Claims and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 23 pages, 8 tables, 11 figures, Published In Proceedings of the 31st International Conference on Computational Linguistics 2025
    </div>
    <details class="paper-abstract">
      The increased use of large language models (LLMs) across a variety of real-world applications calls for mechanisms to verify the factual accuracy of their outputs. Difficulties lie in assessing the factuality of free-form responses in open domains. Also, different papers use disparate evaluation benchmarks and measurements, which renders them hard to compare and hampers future progress. To mitigate these issues, we propose OpenFactCheck, a unified framework for building customized automatic fact-checking systems, benchmarking their accuracy, evaluating factuality of LLMs, and verifying claims in a document. OpenFactCheck consists of three modules: (i) CUSTCHECKER allows users to easily customize an automatic fact-checker and verify the factual correctness of documents and claims, (ii) LLMEVAL, a unified evaluation framework assesses LLM's factuality ability from various perspectives fairly, and (iii) CHECKEREVAL is an extensible solution for gauging the reliability of automatic fact-checkers' verification results using human-annotated datasets. Data and code are publicly available at https://github.com/yuxiaw/openfactcheck.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11832v3">OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 11 pages, 4 Figures, 3 Tables, Published In Proceedings of The 2024 Conference on Empirical Methods in Natural Language Processing
    </div>
    <details class="paper-abstract">
      The increased use of large language models (LLMs) across a variety of real-world applications calls for automatic tools to check the factual accuracy of their outputs, as LLMs often hallucinate. This is difficult as it requires assessing the factuality of free-form open-domain responses. While there has been a lot of research on this topic, different papers use different evaluation benchmarks and measures, which makes them hard to compare and hampers future progress. To mitigate these issues, we developed OpenFactCheck, a unified framework, with three modules: (i) RESPONSEEVAL, which allows users to easily customize an automatic fact-checking system and to assess the factuality of all claims in an input document using that system, (ii) LLMEVAL, which assesses the overall factuality of an LLM, and (iii) CHECKEREVAL, a module to evaluate automatic fact-checking systems. OpenFactCheck is open-sourced (https://github.com/mbzuai-nlp/openfactcheck) and publicly released as a Python library (https://pypi.org/project/openfactcheck/) and also as a web service (http://app.openfactcheck.com). A video describing the system is available at https://youtu.be/-i9VKL0HleI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23234v4">p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments. The code is available at https://github.com/ryttry/p-less .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05079v2">LLM-Guided Scenario-based GUI Testing</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      The assurance of mobile app GUIs has become increasingly important, as the GUI serves as the primary medium of interaction between users and apps. Although numerous automated GUI testing approaches have been developed with diverse strategies, a substantial gap remains between these approaches and the underlying app business logic. Most existing approaches focus on general exploration rather than the completion of specific testing scenarios, often missing critical functionalities. Inspired by manual testing, which treats business logic-driven scenarios as the fundamental unit of testing, this paper introduces an approach that leverages large language models to comprehend GUI semantics and contextual relevance to given scenarios. Building on this capability, we propose ScenGen, an LLM-guided scenario-based GUI testing framework employing multi-agent collaboration to simulate and automate manual testing phases. Specifically, ScenGen integrates five agents: the Observer, Decider, Executor, Supervisor, and Recorder. The Observer perceives the app GUI state by extracting and structuring GUI widgets and layouts, interpreting semantic information. This is passed to the Decider, which makes scenario-driven decisions with LLM guidance to identify target widgets and determine actions toward fulfilling specific goals. The Executor performs these operations, while the Supervisor verifies alignment with intended scenario completion, ensuring traceability and consistency. Finally, the Recorder logs GUI operations into context memory as a knowledge base for subsequent decision-making and monitors runtime bugs. Comprehensive evaluations demonstrate that ScenGen effectively generates scenario-based GUI tests guided by LLM collaboration, achieving higher relevance to business logic and improving the completeness of automated GUI testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23941v1">Auto prompting without training labels: An LLM cascade for product quality assessment in e-commerce catalogs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We introduce a novel, training free cascade for auto-prompting Large Language Models (LLMs) to assess product quality in e-commerce. Our system requires no training labels or model fine-tuning, instead automatically generating and refining prompts for evaluating attribute quality across tens of thousands of product category-attribute pairs. Starting from a seed of human-crafted prompts, the cascade progressively optimizes instructions to meet catalog-specific requirements. This approach bridges the gap between general language understanding and domain-specific knowledge at scale in complex industrial catalogs. Our extensive empirical evaluations shows the auto-prompt cascade improves precision and recall by $8-10\%$ over traditional chain-of-thought prompting. Notably, it achieves these gains while reducing domain expert effort from 5.1 hours to 3 minutes per attribute - a $99\%$ reduction. Additionally, the cascade generalizes effectively across five languages and multiple quality assessment tasks, consistently maintaining performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01374v2">REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed Reasoning Compiler) that formulates optimization as a sequential, context-aware decision process guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-informed transformations that reflect the current program state and accumulated performance feedback. MCTS incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23924v1">Agent-based Automated Claim Matching with Instruction-following LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted for the International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (2025) Findings
    </div>
    <details class="paper-abstract">
      We present a novel agent-based approach for the automated claim matching task with instruction-following LLMs. We propose a two-step pipeline that first generates prompts with LLMs, to then perform claim matching as a binary classification task with LLMs. We demonstrate that LLM-generated prompts can outperform SOTA with human-generated prompts, and that smaller LLMs can do as well as larger ones in the generation process, allowing to save computational resources. We also demonstrate the effectiveness of using different LLMs for each step of the pipeline, i.e. using an LLM for prompt generation, and another for claim matching. Our investigation into the prompt generation process in turn reveals insights into the LLMs' understanding of claim matching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23921v1">Breaking the Benchmark: Revealing LLM Bias via Minimal Contextual Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 9 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models have been shown to demonstrate stereotypical biases in their representations and behavior due to the discriminative nature of the data that they have been trained on. Despite significant progress in the development of methods and models that refrain from using stereotypical information in their decision-making, recent work has shown that approaches used for bias alignment are brittle. In this work, we introduce a novel and general augmentation framework that involves three plug-and-play steps and is applicable to a number of fairness evaluation benchmarks. Through application of augmentation to a fairness evaluation dataset (Bias Benchmark for Question Answering (BBQ)), we find that Large Language Models (LLMs), including state-of-the-art open and closed weight models, are susceptible to perturbations to their inputs, showcasing a higher likelihood to behave stereotypically. Furthermore, we find that such models are more likely to have biased behavior in cases where the target demographic belongs to a community less studied by the literature, underlining the need to expand the fairness and safety research to include more diverse communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23893v1">Evaluating the effectiveness of LLM-based interoperability</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Background: Systems of systems are becoming increasingly dynamic and heterogeneous, and this adds pressure on the long-standing challenge of interoperability. Besides its technical aspect, interoperability has also an economic side, as development time efforts are required to build the interoperability artifacts. Objectives: With the recent advances in the field of large language models (LLMs), we aim at analyzing the effectiveness of LLM-based strategies to make systems interoperate autonomously, at runtime, without human intervention. Method: We selected 13 open source LLMs and curated four versions of a dataset in the agricultural interoperability use case. We performed three runs of each model with each version of the dataset, using two different strategies. Then we compared the effectiveness of the models and the consistency of their results across multiple runs. Results: qwen2.5-coder:32b was the most effective model using both strategies DIRECT (average pass@1 >= 0.99) and CODEGEN (average pass@1 >= 0.89) in three out of four dataset versions. In the fourth dataset version, which included an unit conversion, all models using the strategy DIRECT failed, whereas using CODEGEN qwen2.5-coder:32b succeeded with an average pass@1 = 0.75. Conclusion: Some LLMs can make systems interoperate autonomously. Further evaluation in different domains is recommended, and further research on reliability strategies should be conducted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23891v1">PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Text watermarking for large language models (LLMs) enables model owners to verify text origin and protect intellectual property. While watermarking methods for closed-source LLMs are relatively mature, extending them to open-source models remains challenging, as developers cannot control the decoding process. Consequently, owners of open-source LLMs lack practical means to verify whether text was generated by their models. A core difficulty lies in embedding watermarks directly into model weights without hurting detectability. A promising idea is to distill watermarks from a closed-source model into an open one, but this suffers from (i) poor detectability due to mismatch between learned and predefined patterns, and (ii) fragility to downstream modifications such as fine-tuning or model merging. To overcome these limitations, we propose PRO, a Precise and Robust text watermarking method for open-source LLMs. PRO jointly trains a watermark policy model with the LLM, producing patterns that are easier for the model to learn and more consistent with detection criteria. A regularization term further simulates downstream perturbations and penalizes degradation in watermark detectability, ensuring robustness under model edits. Experiments on open-source LLMs (e.g., LLaMA-3.2, LLaMA-3, Phi-2) show that PRO substantially improves both watermark detectability and resilience to model modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23875v1">Large Language Model Agent Personality and Response Appropriateness: Evaluation by Human Linguistic Experts, LLM-as-Judge, and Natural Language Processing Model</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM)-based agents can be used to create highly engaging interactive applications through prompting personality traits and contextual data, effectively assessing their personalities has proven challenging. This novel interdisciplinary approach addresses this gap by combining agent development and linguistic analysis to assess the prompted personality of LLM-based agents in a poetry explanation task. We developed a novel, flexible question bank, informed by linguistic assessment criteria and human cognitive learning levels, offering a more comprehensive evaluation than current methods. By evaluating agent responses with natural language processing models, other LLMs, and human experts, our findings illustrate the limitations of purely deep learning solutions and emphasize the critical role of interdisciplinary design in agent development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23874v1">From Stochasticity to Signal: A Bayesian Latent State Model for Reliable Measurement with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate classification tasks in business, such as analyzing customer satisfaction from text. However, the inherent stochasticity of LLMs, in terms of their tendency to produce different outputs for the same input, creates a significant measurement error problem that is often neglected with a single round of output, or addressed with ad-hoc methods like majority voting. Such naive approaches fail to quantify uncertainty and can produce biased estimates of population-level metrics. In this paper, we propose a principled solution by reframing LLM variability as a statistical measurement error problem and introducing a Bayesian latent state model to address it. Our model treats the true classification (e.g., customer dissatisfaction) as an unobserved latent variable and the multiple LLM ratings as noisy measurements of this state. This framework allows for the simultaneous estimation of the LLM's false positive and false negative error rates, the underlying base rate of the phenomenon in the population, the posterior probability of the true state for each individual observation, and the causal impact of a business intervention, if any, on the latent state. Through simulation studies, we demonstrate that our model accurately recovers true parameters where naive methods fail. We conclude that this methodology provides a general and reliable framework for converting noisy, probabilistic outputs from LLMs into accurate and actionable insights for scientific and business applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23854v1">Can LLMs Narrate Tabular Data? An Evaluation Framework for Natural Language Representations of Text-to-SQL System Outputs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      In modern industry systems like multi-turn chat agents, Text-to-SQL technology bridges natural language (NL) questions and database (DB) querying. The conversion of tabular DB results into NL representations (NLRs) enables the chat-based interaction. Currently, NLR generation is typically handled by large language models (LLMs), but information loss or errors in presenting tabular results in NL remains largely unexplored. This paper introduces a novel evaluation method - Combo-Eval - for judgment of LLM-generated NLRs that combines the benefits of multiple existing methods, optimizing evaluation fidelity and achieving a significant reduction in LLM calls by 25-61%. Accompanying our method is NLR-BIRD, the first dedicated dataset for NLR benchmarking. Through human evaluations, we demonstrate the superior alignment of Combo-Eval with human judgments, applicable across scenarios with and without ground truth references.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23853v1">Temporal Blindness in Multi-Turn LLM Agents: Misaligned Tool Use vs. Human Time Perception</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 preliminary work in progress
    </div>
    <details class="paper-abstract">
      Large language model agents are increasingly used in multi-turn conversational settings to interact with and execute tasks in dynamic environments. However, a key limitation is their temporal blindness: they, by default, operate with a stationary context, failing to account for the real-world time elapsed between messages. This becomes a critical liability when an agent must decide whether to invoke a tool based on how much time has passed since the last observation. Without temporal awareness, agents often either over-rely on previous context (skipping necessary tool calls), or under-rely on it (unnecessarily repeating tool calls). To study this challenge, we introduce TicToc-v1, a test set of multi-turn user-agent trajectories across 34 scenarios with varying time sensitivity. Each trajectory ends with a user question, where the need for a tool call depends on the amount of time elapsed since the last message. To give LLMs temporal context, we augment dialogue messages with explicit timestamps, bridging the gap between static dialogue and evolving environments. We then collected human preferences for these samples, creating two subsets: one where humans preferred relying on the previous observation (prefer-noTool), and another where they preferred a new tool call (prefer-Tool). We evaluated how well LLM tool-calling decisions align with human preferences under varying time intervals on TicToc-v1. Our analysis show that without time information, most models perform only slightly better than random, with the top alignment rate being just over 60%. While adding timestamps leads to a slight improvement, particularly for larger models, the improvement is modest, peaking at around 65%. We also show that naive, prompt-based alignment have limited effectiveness. Our findings highlight the need for specific post-training alignment to align multi-turn LLM tool use with human temporal perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23828v1">Beyond Understanding: Evaluating the Pragmatic Gap in LLMs' Cultural Processing of Figurative Language</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We present a comprehensive evaluation of the ability of large language models (LLMs) to process culturally grounded language, specifically to understand and pragmatically use figurative expressions that encode local knowledge and cultural nuance. Using figurative language as a proxy for cultural nuance and local knowledge, we design evaluation tasks for contextual understanding, pragmatic use, and connotation interpretation in Arabic and English. We evaluate 22 open- and closed-source LLMs on Egyptian Arabic idioms, multidialectal Arabic proverbs, and English proverbs. Our results show a consistent hierarchy: the average accuracy for Arabic proverbs is 4.29% lower than for English proverbs, and performance for Egyptian idioms is 10.28% lower than for Arabic proverbs. For the pragmatic use task, accuracy drops by 14.07% relative to understanding, though providing contextual idiomatic sentences improves accuracy by 10.66%. Models also struggle with connotative meaning, reaching at most 85.58% agreement with human annotators on idioms with 100% inter-annotator agreement. These findings demonstrate that figurative language serves as an effective diagnostic for cultural reasoning: while LLMs can often interpret figurative meaning, they face challenges in using it appropriately. To support future research, we release Kinayat, the first dataset of Egyptian Arabic idioms designed for both figurative understanding and pragmatic use evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15870v2">OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Technical Report. Code: https://github.com/NVlabs/OmniVinci
    </div>
    <details class="paper-abstract">
      Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21837v3">Semantic Agreement Enables Efficient Open-Ended LLM Cascades</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP) Industry Track
    </div>
    <details class="paper-abstract">
      Cascade systems route computational requests to smaller models when possible and defer to larger models only when necessary, offering a promising approach to balance cost and quality in LLM deployment. However, they face a fundamental challenge in open-ended text generation: determining output reliability when generation quality lies on a continuous spectrum, often with multiple valid responses. To address this, we propose semantic agreement -- meaning-level consensus between ensemble outputs -- as a training-free signal for reliable deferral. We show that when diverse model outputs agree semantically, their consensus is a stronger reliability signal than token-level confidence. Evaluated from 500M to 70B-parameter models, we find that semantic cascades match or surpass target-model quality at 40% of the cost and reduce latency by up to 60%. Our method requires no model internals, works across black-box APIs, and remains robust to model updates, making it a practical baseline for real-world LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14571v2">Learned, Lagged, LLM-splained: LLM Responses to End User Security Questions</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 17 pages, 7 tables
    </div>
    <details class="paper-abstract">
      Answering end user security questions is challenging. While large language models (LLMs) like GPT, LLAMA, and Gemini are far from error-free, they have shown promise in answering a variety of questions outside of security. We studied LLM performance in the area of end user security by qualitatively evaluating 3 popular LLMs on 900 systematically collected end user security questions. While LLMs demonstrate broad generalist ``knowledge'' of end user security information, there are patterns of errors and limitations across LLMs consisting of stale and inaccurate answers, and indirect or unresponsive communication styles, all of which impacts the quality of information received. Based on these patterns, we suggest directions for model improvement and recommend user strategies for interacting with LLMs when seeking assistance with security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00063v3">MolErr2Fix: Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown growing potential in molecular sciences, but they often produce chemically inaccurate descriptions and struggle to recognize or justify potential errors. This raises important concerns about their robustness and reliability in scientific applications. To support more rigorous evaluation of LLMs in chemical reasoning, we present the MolErr2Fix benchmark, designed to assess LLMs on error detection and correction in molecular descriptions. Unlike existing benchmarks focused on molecule-to-text generation or property prediction, MolErr2Fix emphasizes fine-grained chemical understanding. It tasks LLMs with identifying, localizing, explaining, and revising potential structural and semantic errors in molecular descriptions. Specifically, MolErr2Fix consists of 1,193 fine-grained annotated error instances. Each instance contains quadruple annotations, i.e,. (error type, span location, the explanation, and the correction). These tasks are intended to reflect the types of reasoning and verification required in real-world chemical communication. Evaluations of current state-of-the-art LLMs reveal notable performance gaps, underscoring the need for more robust chemical reasoning capabilities. MolErr2Fix provides a focused benchmark for evaluating such capabilities and aims to support progress toward more reliable and chemically informed language models. All annotations and an accompanying evaluation API will be publicly released to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14205v2">DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01268v3">AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 37\%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23595v1">Multi-Agent Evolve: LLM Self-Improve through Co-evolution</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 29 pages, 4 figures, submitted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06522v2">Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19113v2">Human-Aligned Faithfulness in Toxicity Explanations of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 23 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity -- from their explanations that justify a stance -- to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Human-Aligned Faithfulness (HAF), that measures the extent to which LLMs' free-form toxicity explanations align with those of a rational human under ideal conditions. We develop six metrics, based on uncertainty quantification, to comprehensively evaluate HAF of LLMs' toxicity explanations with no human involvement, and highlight how "non-ideal" the explanations are. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. We open-source our code at https://github.com/uofthcdslab/HAF and LLM-generated explanations at https://huggingface.co/collections/uofthcdslab/haf.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05965v4">Validating LLM-as-a-Judge Systems under Rating Indeterminacy</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, plays a critical role in scaling and standardizing GenAI evaluations. To validate such judge systems, evaluators assess human--judge agreement by first collecting multiple human ratings for each item in a validation corpus, then aggregating the ratings into a single, per-item gold label rating. For many items, however, rating criteria may admit multiple valid interpretations, so a human or LLM rater may deem multiple ratings "reasonable" or "correct." We call this condition rating indeterminacy. Problematically, many rating tasks that contain rating indeterminacy rely on forced-choice elicitation, whereby raters are instructed to select only one rating for each item. In this paper, we introduce a framework for validating LLM-as-a-judge systems under rating indeterminacy. We draw theoretical connections between different measures of judge system performance under different human--judge agreement metrics, and different rating elicitation and aggregation schemes. We demonstrate that differences in how humans and LLMs resolve rating indeterminacy when responding to forced-choice rating instructions can heavily bias LLM-as-a-judge validation. Through extensive experiments involving 11 real-world rating tasks and 9 commercial LLMs, we show that standard validation approaches that rely upon forced-choice ratings select judge systems that are highly suboptimal, performing as much as 31% worse than judge systems selected by our approach that uses multi-label "response set" ratings to account for rating indeterminacy. We conclude with concrete recommendations for more principled approaches to LLM-as-a-judge validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11477v3">How Can We Effectively Expand the Vocabulary of LLMs with 0.01GB of Target Language Text?</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted to Computational Linguistics
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in many languages beyond English. Yet, LLMs require more inference steps when generating non-English text due to their reliance on English-centric tokenizers and vocabulary, resulting in higher usage costs to non-English speakers. Vocabulary expansion with target language tokens is a widely used cross-lingual vocabulary adaptation approach to remedy this issue. Despite its effectiveness in inference speedup, previous work on vocabulary expansion has focused on high-resource settings assuming access to a substantial amount of target language data to effectively initialize the embeddings of the new tokens and adapt the LLM to the target language. However, vocabulary expansion in low-resource settings has yet to be explored. In this article, we investigate vocabulary expansion in low-resource settings by considering embedding initialization methods and continual pre-training strategies. Through extensive experiments across typologically diverse languages, tasks and models, we establish a set of strategies to perform vocabulary expansion for faster inference, while striving to maintain competitive downstream performance to baselines. This is achieved with only 30K sentences ($\sim$0.01GB text data) from the target language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23408v1">AutoStreamPipe: LLM Assisted Automatic Generation of Data Stream Processing Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Data pipelines are essential in stream processing as they enable the efficient collection, processing, and delivery of real-time data, supporting rapid data analysis. In this paper, we present AutoStreamPipe, a novel framework that employs Large Language Models (LLMs) to automate the design, generation, and deployment of stream processing pipelines. AutoStreamPipe bridges the semantic gap between high-level user intent and platform-specific implementations across distributed stream processing systems for structured multi-agent reasoning by integrating a Hypergraph of Thoughts (HGoT) as an extended version of GoT. AutoStreamPipe combines resilient execution strategies, advanced query analysis, and HGoT to deliver pipelines with good accuracy. Experimental evaluations on diverse pipelines demonstrate that AutoStreamPipe significantly reduces development time (x6.3) and error rates (x5.19), as measured by a novel Error-Free Score (EFS), compared to LLM code-generation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21433v3">The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 v3: DL4C camera-ready version to be presented at the 4th DL4C workshop co-located with NeurIPS '25; added OpenHands generality probe, added hybrid context management strategy
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering (SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these approaches within SWE-agent on SWE-bench Verified across five diverse model configurations. Moreover, we show initial evidence of our findings generalizing to the OpenHands agent scaffold. We find that a simple environment observation masking strategy halves cost relative to the raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. Additionally, we introduce a novel hybrid approach that further reduces costs by 7% and 11% compared to just observation masking or LLM summarization, respectively. Our findings raise concerns regarding the trend towards pure LLM summarization and provide initial evidence of untapped cost reductions by pushing the efficiency-effectiveness frontier. We release code and data for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08343v2">A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted in a computer science workshop
    </div>
    <details class="paper-abstract">
      With the rapid adoption of Large Language Models (LLMs), LLM-adapters have become increasingly common, providing lightweight specialization of large-scale models. Serving hundreds or thousands of these adapters on a single GPU allows request aggregation, increasing throughput, but may also cause request starvation if GPU memory limits are exceeded. To address this issue, this study focuses on determining the joint configuration of concurrent and parallel adapters that maximizes GPU throughput without inducing starvation, given heterogeneous adapter and traffic properties. We propose a data-driven ML approach leveraging interpretable models to tackle this caching problem and introduce the first Digital Twin capable of reproducing an LLM-adapter serving system, enabling efficient training data generation. Experiments with the vLLM framework and LoRA adapters show that the Digital Twin reproduces throughput within 5.1% of real results, while the ML approach predicts optimal numbers of concurrent and parallel adapters with an error of at most 7.2% under heterogeneous, real-world workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23799v3">Estimating LLM Consistency: A User Baseline vs Surrogate Metrics</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Published as a main conference paper at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucinations and sensitiveto prompt perturbations, often resulting in inconsistent or unreliablegenerated text. Different methods have been proposed to mitigate suchhallucinations and fragility, one of which is to measure theconsistency of LLM responses -- the model's confidence in the responseor likelihood of generating a similar response when resampled. Inprevious work, measuring LLM response consistency often relied oncalculating the probability of a response appearing within a pool of resampledresponses, analyzing internal states, or evaluating logits of resopnses.However, it was not clear how well theseapproaches approximated users' perceptions of consistency of LLMresponses. To find out, we performed a user study ($n=2,976$)demonstrating that current methods for measuring LLM responseconsistency typically do not align well with humans' perceptions of LLMconsistency. We propose a logit-based ensemble method for estimatingLLM consistency and show that our method matches the performance of thebest-performing existing metric in estimating human ratings of LLMconsistency. Our results suggest that methods for estimating LLMconsistency without human evaluation are sufficiently imperfect towarrant broader use of evaluation with human input; this would avoidmisjudging the adequacy of models because of the imperfections ofautomated consistency metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02024v2">NestedFP: High-Performance, Memory-Efficient Dual-Precision Floating Point Support for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Meeting service-level objectives (SLOs) in Large Language Models (LLMs) serving is critical, but managing the high variability in load presents a significant challenge. Recent advancements in FP8 inference, backed by native hardware support, offer a potential solution: executing FP16 models by default, while switching to FP8 models during sudden load surges to achieve higher throughput at the cost of a slight quality degradation. Although this approach facilitates effective SLO management, it introduces additional memory overhead due to storing two versions of the same model. In response, this paper proposes NestedFP, an LLM serving technique that supports both FP16 and FP8 models in a memory efficient manner by overlaying FP8 parameters onto FP16 parameters, allowing both models to share the same FP16 memory footprint. By leveraging a compact data format for the overlay and a specialized GEMM kernel optimized for this format, NestedFP ensures minimal degradation in both model quality and inference throughput across both FP8 and FP16 modes. NestedFP provides a flexible platform for dynamic, SLO-aware precision selection. The code is available at https://github.com/SNU-ARC/NestedFP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10328v2">Are LLMs Empathetic to All? Investigating the Influence of Multi-Demographic Personas on a Model's Empathy</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 9 pages, 4 figures, 4 tables, EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models' (LLMs) ability to converse naturally is empowered by their ability to empathetically understand and respond to their users. However, emotional experiences are shaped by demographic and cultural contexts. This raises an important question: Can LLMs demonstrate equitable empathy across diverse user groups? We propose a framework to investigate how LLMs' cognitive and affective empathy vary across user personas defined by intersecting demographic attributes. Our study introduces a novel intersectional analysis spanning 315 unique personas, constructed from combinations of age, culture, and gender, across four LLMs. Results show that attributes profoundly shape a model's empathetic responses. Interestingly, we see that adding multiple attributes at once can attenuate and reverse expected empathy patterns. We show that they broadly reflect real-world empathetic trends, with notable misalignments for certain groups, such as those from Confucian culture. We complement our quantitative findings with qualitative insights to uncover model behaviour patterns across different demographic groups. Our findings highlight the importance of designing empathy-aware LLMs that account for demographic diversity to promote more inclusive and equitable model behaviour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23358v1">How AI Forecasts AI Jobs: Benchmarking LLM Predictions of Labor Market Changes</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 8 pages + Limitations + References
    </div>
    <details class="paper-abstract">
      Artificial intelligence is reshaping labor markets, yet we lack tools to systematically forecast its effects on employment. This paper introduces a benchmark for evaluating how well large language models (LLMs) can anticipate changes in job demand, especially in occupations affected by AI. Existing research has shown that LLMs can extract sentiment, summarize economic reports, and emulate forecaster behavior, but little work has assessed their use for forward-looking labor prediction. Our benchmark combines two complementary datasets: a high-frequency index of sector-level job postings in the United States, and a global dataset of projected occupational changes due to AI adoption. We format these data into forecasting tasks with clear temporal splits, minimizing the risk of information leakage. We then evaluate LLMs using multiple prompting strategies, comparing task-scaffolded, persona-driven, and hybrid approaches across model families. We assess both quantitative accuracy and qualitative consistency over time. Results show that structured task prompts consistently improve forecast stability, while persona prompts offer advantages on short-term trends. However, performance varies significantly across sectors and horizons, highlighting the need for domain-aware prompting and rigorous evaluation protocols. By releasing our benchmark, we aim to support future research on labor forecasting, prompt design, and LLM-based economic reasoning. This work contributes to a growing body of research on how LLMs interact with real-world economic data, and provides a reproducible testbed for studying the limits and opportunities of AI as a forecasting tool in the context of labor markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23350v1">Validating Formal Specifications with LLM-generated Test Cases</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Validation is a central activity when developing formal specifications. Similarly to coding, a possible validation technique is to define upfront test cases or scenarios that a future specification should satisfy or not. Unfortunately, specifying such test cases is burdensome and error prone, which could cause users to skip this validation task. This paper reports the results of an empirical evaluation of using pre-trained large language models (LLMs) to automate the generation of test cases from natural language requirements. In particular, we focus on test cases for structural requirements of simple domain models formalized in the Alloy specification language. Our evaluation focuses on the state-of-art GPT-5 model, but results from other closed- and open-source LLMs are also reported. The results show that, in this context, GPT-5 is already quite effective at generating positive (and negative) test cases that are syntactically correct and that satisfy (or not) the given requirement, and that can detect many wrong specifications written by humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20445v4">TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS 2025, https://github.com/tsinghua-fib-lab/TrajAgent
    </div>
    <details class="paper-abstract">
      Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. \fix In this paper, we propose \textit{TrajAgent}, a agent framework powered by large language models (LLMs), designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. \unfix~In \textit{TrajAgent}, we first develop \textit{UniEnv}, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on \textit{UniEnv}, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on four tasks using four real-world datasets demonstrate the effectiveness of \textit{TrajAgent} in automated trajectory modeling, achieving a performance improvement of \fix 2.38\%-69.91\% \unfix over baseline methods. The codes and data can be accessed via https://github.com/tsinghua-fib-lab/TrajAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.20075v3">LLMs can hide text in other text of the same length</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 21 pages, main paper 9 pages
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23313v1">Network Intrusion Detection: Evolution from Conventional Approaches to LLM Collaboration and Emerging Risks</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 28 pages,1 figure, 204 references
    </div>
    <details class="paper-abstract">
      This survey systematizes the evolution of network intrusion detection systems (NIDS), from conventional methods such as signature-based and neural network (NN)-based approaches to recent integrations with large language models (LLMs). It clearly and concisely summarizes the current status, strengths, and limitations of conventional techniques, and explores the practical benefits of integrating LLMs into NIDS. Recent research on the application of LLMs to NIDS in diverse environments is reviewed, including conventional network infrastructures, autonomous vehicle environments and IoT environments. From this survey, readers will learn that: 1) the earliest methods, signature-based IDSs, continue to make significant contributions to modern systems, despite their well-known weaknesses; 2) NN-based detection, although considered promising and under development for more than two decades, and despite numerous related approaches, still faces significant challenges in practical deployment; 3) LLMs are useful for NIDS in many cases, and a number of related approaches have been proposed; however, they still face significant challenges in practical applications. Moreover, they can even be exploited as offensive tools, such as for generating malware, crafting phishing messages, or launching cyberattacks. Recently, several studies have been proposed to address these challenges, which are also reviewed in this survey; and 4) strategies for constructing domain-specific LLMs have been proposed and are outlined in this survey, as it is nearly impossible to train a NIDS-specific LLM from scratch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19209v2">MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in automating scientific hypothesis generation, yet existing approaches primarily yield coarse-grained hypotheses lacking critical methodological and experimental details. We introduce and formally define the new task of fine-grained scientific hypothesis discovery, which entails generating detailed, experimentally actionable hypotheses from coarse initial research directions. We frame this as a combinatorial optimization problem and investigate the upper limits of LLMs' capacity to solve it when maximally leveraged. Specifically, we explore four foundational questions: (1) how to best harness an LLM's internal heuristics to formulate the fine-grained hypothesis it itself would judge as the most promising among all the possible hypotheses it might generate, based on its own internal scoring-thus defining a latent reward landscape over the hypothesis space; (2) whether such LLM-judged better hypotheses exhibit stronger alignment with ground-truth hypotheses; (3) whether shaping the reward landscape using an ensemble of diverse LLMs of similar capacity yields better outcomes than defining it with repeated instances of the strongest LLM among them; and (4) whether an ensemble of identical LLMs provides a more reliable reward landscape than a single LLM. To address these questions, we propose a hierarchical search method that incrementally proposes and integrates details into the hypothesis, progressing from general concepts to specific experimental configurations. We show that this hierarchical process smooths the reward landscape and enables more effective optimization. Empirical evaluations on a new benchmark of expert-annotated fine-grained hypotheses from recent literature show that our method consistently outperforms strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08338v3">LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07967v3">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Development</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 A vision paper that will be continuously updated
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong capabilities in software engineering tasks, raising expectations of revolutionary productivity gains. However, enterprise software development is largely driven by incremental evolution, where challenges extend far beyond routine coding and depend critically on tacit knowledge, including design decisions at different levels and historical trade-offs. To achieve effective AI-powered support for complex software development, we should align emerging AI capabilities with the practical realities of enterprise development. To this end, we systematically identify challenges from both software and LLM perspectives. Alongside these challenges, we outline opportunities where AI and structured knowledge frameworks can enhance decision-making in tasks such as issue localization and impact analysis. To address these needs, we propose the Code Digital Twin, a living framework that models both the physical and conceptual layers of software, preserves tacit knowledge, and co-evolves with the codebase. By integrating hybrid knowledge representations, multi-stage extraction pipelines, incremental updates, LLM-empowered applications, and human-in-the-loop feedback, the Code Digital Twin transforms fragmented knowledge into explicit and actionable representations. Our vision positions it as a bridge between AI advancements and enterprise software realities, providing a concrete roadmap toward sustainable, intelligent, and resilient development and evolution of ultra-complex systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16395v2">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Development</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 This should be a replacement of another article (arXiv:2503.07967), I mis-submitted it as a new article
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong capabilities in software engineering tasks, raising expectations of revolutionary productivity gains. However, enterprise software development is largely driven by incremental evolution, where challenges extend far beyond routine coding and depend critically on tacit knowledge, including design decisions at different levels and historical trade-offs. To achieve effective AI-powered support for complex software development, we should align emerging AI capabilities with the practical realities of enterprise development. To this end, we systematically identify challenges from both software and LLM perspectives. Alongside these challenges, we outline opportunities where AI and structured knowledge frameworks can enhance decision-making in tasks such as issue localization and impact analysis. To address these needs, we propose the Code Digital Twin, a living framework that models both the physical and conceptual layers of software, preserves tacit knowledge, and co-evolves with the codebase. By integrating hybrid knowledge representations, multi-stage extraction pipelines, incremental updates, LLM-empowered applications, and human-in-the-loop feedback, the Code Digital Twin transforms fragmented knowledge into explicit and actionable representations. Our vision positions it as a bridge between AI advancements and enterprise software realities, providing a concrete roadmap toward sustainable, intelligent, and resilient development and evolution of ultra-complex systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21034v2">Input Matters: Evaluating Input Structure's Impact on LLM Summaries of Sports Play-by-Play</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted at INLG 2025
    </div>
    <details class="paper-abstract">
      A major concern when deploying LLMs in accuracy-critical domains such as sports reporting is that the generated text may not faithfully reflect the input data. We quantify how input structure affects hallucinations and other factual errors in LLM-generated summaries of NBA play-by-play data, across three formats: row-structured, JSON and unstructured. We manually annotated 3,312 factual errors across 180 game summaries produced by two models, Llama-3.1-70B and Qwen2.5-72B. Input structure has a strong effect: JSON input reduces error rates by 69% for Llama and 65% for Qwen compared to unstructured input, while row-structured input reduces errors by 54% for Llama and 51% for Qwen. A two-way repeated measures ANOVA shows that input structure accounts for over 80% of the variance in error rates, with Tukey HSD post hoc tests confirming statistically significant differences between all input formats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21339v2">Multi-turn Training with Basic Human Feedback Helps Little on LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) are typically developed through the single-turn reinforcement learning, whereas real-world applications often involve multi-turn interactions with human feedback, leading to a potential mismatch between training and deployment conditions. In this work, we study whether multi-turn training with human feedback is necessary for reasoning tasks. We compare conventional single-turn training with three multi-turn strategies and reach contrary conclusions to previous research. We find that models trained in a single-turn setting generalize effectively to both single- and multi-turn evaluations, while models trained with multi-turn strategies exhibit a significant degradation in single-turn reasoning performance. These results suggest that for tasks with complete information, robust single-turn training remains more effective and reliable, as multi-turn training with basic feedback provides limited benefits and can even degrade reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15301v2">Cohort Discovery: A Survey on LLM-Assisted Clinical Trial Recruitment</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have greatly improved general-domain NLP tasks. Yet, their adoption in critical domains, such as clinical trial recruitment, remains limited. As trials are designed in natural language and patient data is represented as both structured and unstructured text, the task of matching trials and patients benefits from knowledge aggregation and reasoning abilities of LLMs. Classical approaches are trial-specific and LLMs with their ability to consolidate distributed knowledge hold the potential to build a more general solution. Yet recent applications of LLM-assisted methods rely on proprietary models and weak evaluation benchmarks. In this survey, we are the first to analyze the task of trial-patient matching and contextualize emerging LLM-based approaches in clinical trial recruitment. We critically examine existing benchmarks, approaches and evaluation frameworks, the challenges to adopting LLM technologies in clinical research and exciting future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23208v1">Increasing LLM Coding Capabilities through Diverse Synthetic Coding Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Presented at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: The 4th Deep Learning for Code Workshop (DL4C)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive promise in code generation, yet their progress remains limited by the shortage of large-scale datasets that are both diverse and well-aligned with human reasoning. Most existing resources pair problems with solutions, but omit the intermediate thought process that guides coding. To close this gap, we present a scalable synthetic data generation pipeline that produces nearly 800k instruction-reasoning-code-test quadruplets. Each sample combines a task, a step-by-step reasoning trace, a working solution, and executable tests, enabling models to learn not just the what but also the how of problem solving. Our pipeline combines four key components: curated contest problems, web-mined content filtered by relevance classifiers, data expansion guided by reasoning patterns, and multi-stage execution-based validation. A genetic mutation algorithm further increases task diversity while maintaining consistency between reasoning traces and code implementations. Our key finding is that fine-tuning LLMs on this dataset yields consistent improvements on coding benchmarks. Beyond raw accuracy, reasoning-aware data can substitute for model scaling, generalize across architectures, and outperform leading open-source alternatives under identical sample budgets. Our work establishes reasoning-centered synthetic data generation as an efficient approach for advancing coding capabilities in LLMs. We publish our dataset and generation pipeline to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23190v1">Evaluation of Vision-LLMs in Surveillance Video</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted as poster in the NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI
    </div>
    <details class="paper-abstract">
      The widespread use of cameras in our society has created an overwhelming amount of video data, far exceeding the capacity for human monitoring. This presents a critical challenge for public safety and security, as the timely detection of anomalous or criminal events is crucial for effective response and prevention. The ability for an embodied agent to recognize unexpected events is fundamentally tied to its capacity for spatial reasoning. This paper investigates the spatial reasoning of vision-language models (VLMs) by framing anomalous action recognition as a zero-shot, language-grounded task, addressing the embodied perception challenge of interpreting dynamic 3D scenes from sparse 2D video. Specifically, we investigate whether small, pre-trained vision--LLMs can act as spatially-grounded, zero-shot anomaly detectors by converting video into text descriptions and scoring labels via textual entailment. We evaluate four open models on UCF-Crime and RWF-2000 under prompting and privacy-preserving conditions. Few-shot exemplars can improve accuracy for some models, but may increase false positives, and privacy filters -- especially full-body GAN transforms -- introduce inconsistencies that degrade accuracy. These results chart where current vision--LLMs succeed (simple, spatially salient events) and where they falter (noisy spatial cues, identity obfuscation). Looking forward, we outline concrete paths to strengthen spatial grounding without task-specific training: structure-aware prompts, lightweight spatial memory across clips, scene-graph or 3D-pose priors during description, and privacy methods that preserve action-relevant geometry. This positions zero-shot, language-grounded pipelines as adaptable building blocks for embodied, real-world video understanding. Our implementation for evaluating VLMs is publicly available at: https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19674v2">SAGE: A Generic Framework for LLM Safety Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      As Large Language Models are rapidly deployed across diverse applications from healthcare to financial advice, safety evaluation struggles to keep pace. Current benchmarks focus on single-turn interactions with generic policies, failing to capture the conversational dynamics of real-world usage and the application-specific harms that emerge in context. Such potential oversights can lead to harms that go unnoticed in standard safety benchmarks and other current evaluation methodologies. To address these needs for robust AI safety evaluation, we introduce SAGE (Safety AI Generic Evaluation), an automated modular framework designed for customized and dynamic harm evaluations. SAGE employs prompted adversarial agents with diverse personalities based on the Big Five model, enabling system-aware multi-turn conversations that adapt to target applications and harm policies. We evaluate seven state-of-the-art LLMs across three applications and harm policies. Multi-turn experiments show that harm increases with conversation length, model behavior varies significantly when exposed to different user personalities and scenarios, and some models minimize harm via high refusal rates that reduce usefulness. We also demonstrate policy sensitivity within a harm category where tightening a child-focused sexual policy substantially increases measured defects across applications. These results motivate adaptive, policy-aware, and context-specific testing for safer real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23163v1">Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21007v2">Can Confidence Estimates Decide When Chain-of-Thought Is Necessary for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) prompting has emerged as a common technique for enhancing the reasoning abilities of large language models (LLMs). While extended reasoning can boost accuracy on complex tasks, it is often unnecessary and substantially increases token usage, limiting the practicality of reasoning models in many scenarios. Recent models, such as GPT-OSS and Qwen3, expose controls that enable users to adjust the length of CoT or determine whether it is used at all. Yet, it remains unclear when CoT should be used: on some tasks it improves performance, while on others it provides little benefit or even harms performance. We address this challenge with confidence-gated CoT, where a model invokes reasoning only when confidence in its direct answer is low. To this end, we present the first systematic study of training-free confidence estimation methods for CoT gating. Specifically, we evaluate four training-free confidence estimation methods and compare them to a random baseline and an oracle that always knows when CoT is needed. Through extensive experiments, we show that existing training-free confidence measures can reduce redundant CoT and outperform randomly invoked CoT. However, the utility of individual confidence measures is inconsistent, varying with both the dataset and the model, underscoring the difficulty of deploying confidence-gated CoT in practice. By analysing both strengths and failure modes, our study highlights the potential and limitations of current methods and paves the way toward more reliable adaptive gating of CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23127v1">Lost in Tokenization: Context as the Key to Unlocking Biomolecular Understanding in Scientific LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 36 pages, under review
    </div>
    <details class="paper-abstract">
      Scientific Large Language Models (Sci-LLMs) have emerged as a promising frontier for accelerating biological discovery. However, these models face a fundamental challenge when processing raw biomolecular sequences: the tokenization dilemma. Whether treating sequences as a specialized language, risking the loss of functional motif information, or as a separate modality, introducing formidable alignment challenges, current strategies fundamentally limit their reasoning capacity. We challenge this sequence-centric paradigm by positing that a more effective strategy is to provide Sci-LLMs with high-level structured context derived from established bioinformatics tools, thereby bypassing the need to interpret low-level noisy sequence data directly. Through a systematic comparison of leading Sci-LLMs on biological reasoning tasks, we tested three input modes: sequence-only, context-only, and a combination of both. Our findings are striking: the context-only approach consistently and substantially outperforms all other modes. Even more revealing, the inclusion of the raw sequence alongside its high-level context consistently degrades performance, indicating that raw sequences act as informational noise, even for models with specialized tokenization schemes. These results suggest that the primary strength of existing Sci-LLMs lies not in their nascent ability to interpret biomolecular syntax from scratch, but in their profound capacity for reasoning over structured, human-readable knowledge. Therefore, we argue for reframing Sci-LLMs not as sequence decoders, but as powerful reasoning engines over expert knowledge. This work lays the foundation for a new class of hybrid scientific AI agents, repositioning the developmental focus from direct sequence interpretation towards high-level knowledge synthesis. The code is available at github.com/opendatalab-raise-dev/CoKE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23101v1">Beyond Imprecise Distance Metrics: LLM-Predicted Target Call Stacks for Directed Greybox Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Preprint, under submission
    </div>
    <details class="paper-abstract">
      Directed greybox fuzzing (DGF) aims to efficiently trigger bugs at specific target locations by prioritizing seeds whose execution paths are more likely to mutate into triggering target bugs. However, existing DGF approaches suffer from imprecise probability calculations due to their reliance on complex distance metrics derived from static analysis. The over-approximations inherent in static analysis cause a large number of irrelevant execution paths to be mistakenly considered to potentially mutate into triggering target bugs, significantly reducing fuzzing efficiency. We propose to replace static analysis-based distance metrics with precise call stack representations. Call stacks represent precise control flows, thereby avoiding false information in static analysis. We leverage large language models (LLMs) to predict vulnerability-triggering call stacks for guiding seed prioritization. Our approach constructs call graphs through static analysis to identify methods that can potentially reach target locations, then utilizes LLMs to predict the most likely call stack sequence that triggers the vulnerability. Seeds whose execution paths have higher overlap with the predicted call stack are prioritized for mutation. This is the first work to integrate LLMs into the core seed prioritization mechanism of DGF. We implement our approach and evaluate it against several state-of-the-art fuzzers. On a suite of real-world programs, our approach triggers vulnerabilities $1.86\times$ to $3.09\times$ faster compared to baselines. In addition, our approach identifies 10 new vulnerabilities and 2 incomplete fixes in the latest versions of programs used in our controlled experiments through directed patch testing, with 10 assigned CVE IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23081v1">A Survey on LLM Mid-training</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent advances in foundation models have highlighted the significant benefits of multi-stage training, with a particular emphasis on the emergence of mid-training as a vital stage that bridges pre-training and post-training. Mid-training is distinguished by its use of intermediate data and computational resources, systematically enhancing specified capabilities such as mathematics, coding, reasoning, and long-context extension, while maintaining foundational competencies. This survey provides a formal definition of mid-training for large language models (LLMs) and investigates optimization frameworks that encompass data curation, training strategies, and model architecture optimization. We analyze mainstream model implementations in the context of objective-driven interventions, illustrating how mid-training serves as a distinct and critical stage in the progressive development of LLM capabilities. By clarifying the unique contributions of mid-training, this survey offers a comprehensive taxonomy and actionable insights, supporting future research and innovation in the advancement of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23074v1">Fast-MIA: Efficient and Scalable Membership Inference for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We propose Fast-MIA (https://github.com/Nikkei/fast-mia), a Python library for efficiently evaluating membership inference attacks (MIA) against Large Language Models (LLMs). MIA against LLMs has emerged as a crucial challenge due to growing concerns over copyright, security, and data privacy, and has attracted increasing research attention. However, the progress of this research is significantly hindered by two main obstacles: (1) the high computational cost of inference in LLMs, and (2) the lack of standardized and maintained implementations of MIA methods, which makes large-scale empirical comparison difficult. To address these challenges, our library provides fast batch inference and includes implementations of representative MIA methods under a unified evaluation framework. This library supports easy implementation of reproducible benchmarks with simple configuration and extensibility. We release Fast-MIA as an open-source (Apache License 2.0) tool to support scalable and transparent research on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14668v2">ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have propelled intelligent agents from reactive responses to proactive support. While promising, existing proactive agents either rely exclusively on observations from enclosed environments (e.g., desktop UIs) with direct LLM inference or employ rule-based proactive notifications, leading to suboptimal user intent understanding and limited functionality for proactive service. In this paper, we introduce ContextAgent, the first context-aware proactive agent that incorporates extensive sensory contexts surrounding humans to enhance the proactivity of LLM agents. ContextAgent first extracts multi-dimensional contexts from massive sensory perceptions on wearables (e.g., video and audio) to understand user intentions. ContextAgent then leverages the sensory contexts and personas from historical data to predict the necessity for proactive services. When proactive assistance is needed, ContextAgent further automatically calls the necessary tools to assist users unobtrusively. To evaluate this new task, we curate ContextAgentBench, the first benchmark for evaluating context-aware proactive LLM agents, covering 1,000 samples across nine daily scenarios and twenty tools. Experiments on ContextAgentBench show that ContextAgent outperforms baselines by achieving up to 8.5% and 6.0% higher accuracy in proactive predictions and tool calling, respectively. We hope our research can inspire the development of more advanced, human-centric, proactive AI assistants. The code and dataset are publicly available at https://github.com/openaiotlab/ContextAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23068v1">Checkstyle+: Reducing Technical Debt Through The Use of Linters with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 11 pages, 9 figures, tool link: https://github.com/ellacodee/CheckstylePlus
    </div>
    <details class="paper-abstract">
      Good code style improves program readability, maintainability, and collaboration, and is an integral component of software quality. Developers, however, often cut corners when following style rules, leading to the wide adoption of tools such as linters in professional software development projects. Traditional linters like Checkstyle operate using rigid, rule-based mechanisms that effectively detect many surface-level violations. However, in most programming languages, there is a subset of style rules that require a more nuanced understanding of code, and fall outside the scope of such static analysis. In this paper, we propose Checkstyle+, a hybrid approach that augments Checkstyle with large language model (LLM) capabilities, to identify style violations that elude the conventional rule-based analysis. Checkstyle+ is evaluated on a sample of 380 Java code files, drawn from a broader dataset of 30,800 real-world Java programs sourced from accepted Codeforces submissions. The results show that Checkstyle+ achieves superior performance over standard Checkstyle in detecting violations of the semantically nuanced rules.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23040v1">LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in generative modeling have shown significant promise in designing novel periodic crystal structures. Existing approaches typically rely on either large language models (LLMs) or equivariant denoising models, each with complementary strengths: LLMs excel at handling discrete atomic types but often struggle with continuous features such as atomic positions and lattice parameters, while denoising models are effective at modeling continuous variables but encounter difficulties in generating accurate atomic compositions. To bridge this gap, we propose CrysLLMGen, a hybrid framework that integrates an LLM with a diffusion model to leverage their complementary strengths for crystal material generation. During sampling, CrysLLMGen first employs a fine-tuned LLM to produce an intermediate representation of atom types, atomic coordinates, and lattice structure. While retaining the predicted atom types, it passes the atomic coordinates and lattice structure to a pre-trained equivariant diffusion model for refinement. Our framework outperforms state-of-the-art generative models across several benchmark tasks and datasets. Specifically, CrysLLMGen not only achieves a balanced performance in terms of structural and compositional validity but also generates more stable and novel materials compared to LLM-based and denoisingbased models Furthermore, CrysLLMGen exhibits strong conditional generation capabilities, effectively producing materials that satisfy user-defined constraints. Code is available at https://github.com/kdmsit/crysllmgen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23038v1">Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as judges to evaluate response quality, providing a scalable alternative to human evaluation. However, most LLM judges operate solely on intrinsic text-based reasoning, limiting their ability to verify complex constraints or perform accurate computation. Motivated by the success of tool-integrated reasoning (TIR) in numerous tasks, we propose TIR-Judge, an end-to-end RL framework for training LLM judges that integrates a code executor for precise evaluation. TIR-Judge is built on three principles: (i) diverse training across verifiable and non-verifiable domains, (ii) flexible judgment formats (pointwise, pairwise, listwise), and (iii) iterative RL that bootstraps directly from the initial model without distillation. On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), and achieves listwise performance comparable to Claude-Opus-4 despite having only 8B parameters. Remarkably, TIR-Judge-Zero - trained entirely without distilled judge trajectories, matches the performance of distilled variants, demonstrating that tool-augmented judges can self-evolve through iterative reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23032v1">P1GPT: a multi-agent LLM workflow module for multi-modal financial information analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled multi-agent reasoning systems capable of collaborative decision-making. However, in financial analysis, most frameworks remain narrowly focused on either isolated single-agent predictors or loosely connected analyst ensembles, and they lack a coherent reasoning workflow that unifies diverse data modalities. We introduce P1GPT, a layered multi-agent LLM framework for multi-modal financial information analysis and interpretable trading decision support. Unlike prior systems that emulate trading teams through role simulation, P1GPT implements a structured reasoning pipeline that systematically fuses technical, fundamental, and news-based insights through coordinated agent communication and integration-time synthesis. Backtesting on multi-modal datasets across major U.S. equities demonstrates that P1GPT achieves superior cumulative and risk-adjusted returns, maintains low drawdowns, and provides transparent causal rationales. These findings suggest that structured reasoning workflows, rather than agent role imitation, offer a scalable path toward explainable and trustworthy financial AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23008v1">From Prompt Optimization to Multi-Dimensional Credibility Evaluation: Enhancing Trustworthiness of Chinese LLM-Generated Liver MRI Reports</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 10 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promising performance in generating diagnostic conclusions from imaging findings, thereby supporting radiology reporting, trainee education, and quality control. However, systematic guidance on how to optimize prompt design across different clinical contexts remains underexplored. Moreover, a comprehensive and standardized framework for assessing the trustworthiness of LLM-generated radiology reports is yet to be established. This study aims to enhance the trustworthiness of LLM-generated liver MRI reports by introducing a Multi-Dimensional Credibility Assessment (MDCA) framework and providing guidance on institution-specific prompt optimization. The proposed framework is applied to evaluate and compare the performance of several advanced LLMs, including Kimi-K2-Instruct-0905, Qwen3-235B-A22B-Instruct-2507, DeepSeek-V3, and ByteDance-Seed-OSS-36B-Instruct, using the SiliconFlow platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22986v1">CodeAD: Synthesize Code of Rules for Log-based Anomaly Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Log-based anomaly detection (LogAD) is critical for maintaining the reliability and availability of large-scale online service systems. While machine learning, deep learning, and large language models (LLMs)-based methods have advanced the LogAD, they often suffer from limited interpretability, high inference costs, and extensive preprocessing requirements, limiting their practicality for real-time, high-volume log analysis. In contrast, rule-based systems offer efficiency and transparency, but require significant manual effort and are difficult to scale across diverse and evolving environments. In this paper, We present CodeAD, a novel framework that automatically synthesizes lightweight Python rule functions for LogAD using LLMs. CodeAD introduces a hierarchical clustering and anchor-grounded sampling strategy to construct representative contrastive log windows, enabling LLMs to discern discriminative anomaly patterns. To ensure robustness and generalizability, CodeAD employs an agentic workflow that iteratively generates, tests, repairs, and refines the rules until it meets correctness and abstraction requirements. The synthesized rules are interpretable, lightweight, and directly executable on raw logs, supporting efficient and transparent online anomaly detection. Our comprehensive experiments on three public datasets (BGL, Hadoop, Thunderbird) demonstrate that CodeAD achieves an average absolute improvement of 3.6% F1 score over the state-of-the-art baselines, while processing large datasets up to 4x faster and at a fraction of the cost (total LLM invocation cost under 4 USD per dataset). These results highlight CodeAD as a practical and scalable solution for online monitoring systems, enabling interpretable, efficient, and automated LogAD in real-world environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22978v1">Reasoning About Reasoning: Towards Informed and Reflective Use of LLM Reasoning in HCI</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 14 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Reasoning is a distinctive human-like characteristic attributed to LLMs in HCI due to their ability to simulate various human-level tasks. However, this work argues that the reasoning behavior of LLMs in HCI is often decontextualized from the underlying mechanics and subjective decisions that condition the emergence and human interpretation of this behavior. Through a systematic survey of 258 CHI papers from 2020-2025 on LLMs, we discuss how HCI hardly perceives LLM reasoning as a product of sociotechnical orchestration and often references it as an object of application. We argue that such abstraction leads to oversimplification of reasoning methodologies from NLP/ML and results in a distortion of LLMs' empirically studied capabilities and (un)known limitations. Finally, drawing on literature from both NLP/ML and HCI, as a constructive step forward, we develop reflection prompts to support HCI practitioners engage with LLM reasoning in an informed and reflective way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22977v1">The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of Large Language Models (LLMs) is a key strategy for building Agents that "think then act." However, recent observations, like OpenAI's o3, suggest a paradox: stronger reasoning often coincides with increased hallucination, yet no prior work has systematically examined whether reasoning enhancement itself causes tool hallucination. To address this gap, we pose the central question: Does strengthening reasoning increase tool hallucination? To answer this, we introduce SimpleToolHalluBench, a diagnostic benchmark measuring tool hallucination in two failure modes: (i) no tool available, and (ii) only distractor tools available. Through controlled experiments, we establish three key findings. First, we demonstrate a causal relationship: progressively enhancing reasoning through RL increases tool hallucination proportionally with task performance gains. Second, this effect transcends overfitting - training on non-tool tasks (e.g., mathematics) still amplifies subsequent tool hallucination. Third, the effect is method-agnostic, appearing when reasoning is instilled via supervised fine-tuning and when it is merely elicited at inference by switching from direct answers to step-by-step thinking. We also evaluate mitigation strategies including Prompt Engineering and Direct Preference Optimization (DPO), revealing a fundamental reliability-capability trade-off: reducing hallucination consistently degrades utility. Mechanistically, Reasoning RL disproportionately collapses tool-reliability-related representations, and hallucinations surface as amplified divergences concentrated in late-layer residual streams. These findings reveal that current reasoning enhancement methods inherently amplify tool hallucination, highlighting the need for new training objectives that jointly optimize for capability and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08221v3">Part I: Tricks or Traps? A Deep Dive into RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 26 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning for LLM reasoning has rapidly emerged as a prominent research area, marked by a significant surge in related studies on both algorithmic innovations and practical applications. Despite this progress, several critical challenges remain, including the absence of standardized guidelines for employing RL techniques and a fragmented understanding of their underlying mechanisms. Additionally, inconsistent experimental settings, variations in training data, and differences in model initialization have led to conflicting conclusions, obscuring the key characteristics of these techniques and creating confusion among practitioners when selecting appropriate techniques. This paper systematically reviews widely adopted RL techniques through rigorous reproductions and isolated evaluations within a unified open-source framework. We analyze the internal mechanisms, applicable scenarios, and core principles of each technique through fine-grained experiments, including datasets of varying difficulty, model sizes, and architectures. Based on these insights, we present clear guidelines for selecting RL techniques tailored to specific setups, and provide a reliable roadmap for practitioners navigating the RL for the LLM domain. Finally, we reveal that a minimalist combination of two techniques can unlock the learning capability of critic-free policies using vanilla PPO loss. The results demonstrate that our simple combination consistently improves performance, surpassing strategies like GRPO and DAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22968v1">Measuring Teaching with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Objective and scalable measurement of teaching quality is a persistent challenge in education. While Large Language Models (LLMs) offer potential, general-purpose models have struggled to reliably apply complex, authentic classroom observation instruments. This paper uses custom LLMs built on sentence-level embeddings, an architecture better suited for the long-form, interpretive nature of classroom transcripts than conventional subword tokenization. We systematically evaluate five different sentence embeddings under a data-efficient training regime designed to prevent overfitting. Our results demonstrate that these specialized models can achieve human-level and even super-human performance with expert human ratings above 0.65 and surpassing the average human-human rater correlation. Further, through analysis of annotation context windows, we find that more advanced models-those better aligned with human judgments-attribute a larger share of score variation to lesson-level features rather than isolated utterances, challenging the sufficiency of single-turn annotation paradigms. Finally, to assess external validity, we find that aggregate model scores align with teacher value-added measures, indicating they are capturing features relevant to student learning. However, this trend does not hold at the individual item level, suggesting that while the models learn useful signals, they have not yet achieved full generalization. This work establishes a viable and powerful new methodology for AI-driven instructional measurement, offering a path toward providing scalable, reliable, and valid feedback for educator development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22967v1">MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 This article has been accepted by Frontiers of Computer Science (FCS)
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22963v1">CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23988v3">LLM/Agent-as-Data-Analyst: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 31 page, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and agent techniques have brought a fundamental shift in the functionality and development paradigm of data analysis tasks (a.k.a LLM/Agent-as-Data-Analyst), demonstrating substantial impact across both academia and industry. In comparison with traditional rule or small-model based approaches, (agentic) LLMs enable complex data understanding, natural language interfaces, semantic analysis functions, and autonomous pipeline orchestration. From a modality perspective, we review LLM-based techniques for (i) structured data (e.g., NL2SQL, NL2GQL, ModelQA), (ii) semi-structured data (e.g., markup languages understanding, semi-structured table question answering), (iii) unstructured data (e.g., chart understanding, text/image document understanding), and (iv) heterogeneous data (e.g., data retrieval and modality alignment in data lakes). The technical evolution further distills four key design goals for intelligent data analysis agents, namely semantic-aware design, autonomous pipelines, tool-augmented workflows, and support for open-world tasks. Finally, we outline the remaining challenges and propose several insights and practical directions for advancing LLM/Agent-powered data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14225v2">Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 The 2025 Conference on Language Modeling (COLM)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as personalized assistants for users across a wide range of tasks -- from offering writing support to delivering tailored recommendations or consultations. Over time, the interaction history between a user and an LLM can provide extensive information about an individual's traits and preferences. However, open questions remain on how well LLMs today can effectively leverage such history to (1) internalize the user's inherent traits and preferences, (2) track how the user profiling and preferences evolve over time, and (3) generate personalized responses accordingly in new scenarios. In this work, we introduce the PERSONAMEM benchmark. PERSONAMEM features curated user profiles with over 180 simulated user-LLM interaction histories, each containing up to 60 sessions of multi-turn conversations across 15 real-world tasks that require personalization. Given an in-situ user query, i.e. query issued by the user from the first-person perspective, we evaluate LLM chatbots' ability to identify the most suitable response according to the current state of the user's profile. We observe that current LLMs still struggle to recognize the dynamic evolution in users' profiles over time through direct prompting approaches. As a consequence, LLMs often fail to deliver responses that align with users' current situations and preferences, with frontier models such as GPT-4.1, o4-mini, GPT-4.5, o1, or Gemini-2.0 achieving only around 50% overall accuracy, suggesting room for improvement. We hope that PERSONAMEM, along with the user profile and conversation simulation pipeline, can facilitate future research in the development of truly user-aware chatbots. Code and data are available at github.com/bowen-upenn/PersonaMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12491v2">Cost-Aware Contrastive Routing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We study cost-aware routing for large language models across diverse and dynamic pools of models. Existing approaches often overlook prompt-specific context, rely on expensive model profiling, assume a fixed set of experts, or use inefficient trial-and-error strategies. We introduce Cost-Spectrum Contrastive Routing (CSCR), a lightweight framework that maps both prompts and models into a shared embedding space to enable fast, cost-sensitive selection. CSCR uses compact, fast-to-compute logit footprints for open-source models and perplexity fingerprints for black-box APIs. A contrastive encoder is trained to favor the cheapest accurate expert within adaptive cost bands. At inference time, routing reduces to a single k-NN lookup via a FAISS index, requiring no retraining when the expert pool changes and enabling microsecond latency. Across multiple benchmarks, CSCR consistently outperforms baselines, improving the accuracy-cost tradeoff by up to 25%, while generalizing robustly to unseen LLMs and out-of-distribution prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03604v2">Bilevel ZOFO: Bridging Parameter-Efficient and Zeroth-Order Techniques for Efficient LLM Fine-Tuning and Meta-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained Large Language Models (LLMs) for downstream tasks using First-Order (FO) optimizers presents significant computational challenges. Parameter-Efficient Fine-Tuning (PEFT) methods address these by freezing most model parameters and training only a small subset. However, PEFT often underperforms compared to full fine-tuning when high task-specific accuracy is required. Zeroth-Order (ZO) methods fine-tune the entire pre-trained model without back-propagation, estimating gradients through forward passes only. While memory-efficient, ZO methods suffer from slow convergence and high sensitivity to prompt selection. We bridge these two worlds with Bilevel-ZOFO, a bilevel optimization method that couples fast, local FO-PEFT adaptation at the inner level with stable, memory-efficient ZO updates of the full backbone at the outer level. The FO-PEFT inner loop performs fast, low-memory local adaptation that reduces the variance of ZO estimates and stabilizes the search, guiding the outer ZO updates of the full backbone and reducing prompt sensitivity. In the mean time, the outer ZO provides better generalization ability for PEFT. We provide theoretical convergence guarantees and empirically demonstrate that Bilevel-ZOFO significantly outperforms existing ZO and FO-PEFT methods, achieving 2-4 times faster training while maintaining similar memory efficiency. Additionally, we show by updating the backbone with ZO and adapting only a tiny FO-PEFT block per task, Bilevel-ZOFO combines full-model capacity with few-shot efficiency, making it a very efficient meta-learning algorithm that quickly adapts to new tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22922v1">Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces</a></div>
    <div class="paper-meta">
      📅 2025-10-27
      | 💬 19 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) have led to their increasing employment in several critical applications, particularly education, where they support problem-solving, tutoring, and personalized study. While there are a plethora of works showing the effectiveness of LLMs in generating step-by-step solutions through chain-of-thought (CoT) reasoning on reasoning benchmarks, little is understood about whether the generated CoT is helpful for end-users in improving their ability to comprehend mathematical reasoning problems and detect errors/hallucinations in LLM-generated solutions. To address this gap and contribute to understanding how reasoning can improve human-AI interaction, we present three new interactive reasoning interfaces: interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive Graph (iGraph), and a novel framework that generates the LLM's reasoning from traditional CoT to alternative, interactive formats. Across 125 participants, we found that interactive interfaces significantly improved performance. Specifically, the iGraph interface yielded the highest clarity and error detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all outperforming standard CoT (73.5%). Interactive interfaces also led to faster response times, where participants using iGraph were fastest (57.9 secs), compared to iCoT and iPoT (60 secs), and the standard CoT baseline (64.7 secs). Furthermore, participants preferred the iGraph reasoning interface, citing its superior ability to enable users to follow the LLM's reasoning process. We discuss the implications of these results and provide recommendations for the future design of reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23946v1">Leveraging LLMs for Early Alzheimer's Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-27
    </div>
    <details class="paper-abstract">
      We present a connectome-informed LLM framework that encodes dynamic fMRI connectivity as temporal sequences, applies robust normalization, and maps these data into a representation suitable for a frozen pre-trained LLM for clinical prediction. Applied to early Alzheimer's detection, our method achieves sensitive prediction with error rates well below clinically recognized margins, with implications for timely Alzheimer's intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19178v2">Imbalanced Gradients in RL Post-Training of Multi-Task LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Multi-task post-training of large language models (LLMs) is typically performed by mixing datasets from different tasks and optimizing them jointly. This approach implicitly assumes that all tasks contribute gradients of similar magnitudes; when this assumption fails, optimization becomes biased toward large-gradient tasks. In this paper, however, we show that this assumption fails in RL post-training: certain tasks produce significantly larger gradients, thus biasing updates toward those tasks. Such gradient imbalance would be justified only if larger gradients implied larger learning gains on the tasks (i.e., larger performance improvements) -- but we find this is not true. Large-gradient tasks can achieve similar or even much lower learning gains than small-gradient ones. Further analyses reveal that these gradient imbalances cannot be explained by typical training statistics such as training rewards or advantages, suggesting that they arise from the inherent differences between tasks. This cautions against naive dataset mixing and calls for future work on principled gradient-level corrections for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04055v2">Rethinking and Exploring String-Based Malware Family Classification in the Era of LLMs and RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 This is a technical report from Lingnan University, Hong Kong. Code is available at https://github.com/AIS2Lab/MalwareGPT
    </div>
    <details class="paper-abstract">
      Malware family classification aims to identify the specific family (e.g., GuLoader or BitRAT) a malware sample may belong to, in contrast to malware detection or sample classification, which only predicts a Yes/No outcome. Accurate family identification can greatly facilitate automated sample labeling and understanding on crowdsourced malware analysis platforms such as VirusTotal and MalwareBazaar, which generate vast amounts of data daily. In this paper, we explore and assess the feasibility of using traditional binary string features for family classification in the new era of large language models (LLMs) and Retrieval-Augmented Generation (RAG). Specifically, we investigate howFamily-Specific String (FSS) features can be utilized in a manner similar to RAG to facilitate family classification. To this end, we develop a curated evaluation framework covering 4,347 samples from 67 malware families, extract and analyze over 25 million strings, and conduct detailed ablation studies to assess the impact of different design choices in four major modules, with each providing a relative improvement ranging from 8.1% to 120%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20119v2">LLM Agents for Generating Microservice-based Applications: how complex is your specification?</a></div>
    <div class="paper-meta">
      📅 2025-10-26
      | 💬 24 pages + 5 pages appendices. 7 Figures. 10 Tables
    </div>
    <details class="paper-abstract">
      In this paper we evaluate the capabilities of LLM Agents in generating code for real-world problems. Specifically, we explore code synthesis for microservice-based applications, a widely used architectural pattern for building applications. We define a standard template for specifying these applications, and we propose a metric for scoring the difficulty of a specification. The higher the score, the more difficult it is to generate code for the specification. Our experimental results show that agents using strong LLMs (like GPT-3o-mini) do fairly well on medium difficulty specifications but do poorly on those of higher difficulty levels. This is due to more intricate business logic, a greater use of external services, database integration and inclusion of non-functional capabilities such as authentication. We analyzed the errors in LLM-synthesized code and report on the key challenges LLM Agents face in generating code for these specifications. Finally, we show that using a fine-grained approach to code generation improves the correctness of the generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22691v1">SALSA: Single-pass Autoregressive LLM Structured Classification</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      Despite their impressive generalization capabilities, instruction-tuned Large Language Models often underperform on text classification benchmarks. We introduce SALSA, a coherent pipeline that combines structured prompting, class-to-token mapping, and parameter-efficient fine-tuning, thereby avoiding cold-start training. Each class label is mapped to a distinct output token, and prompts are constructed to elicit a single-token response. During inference, the model's output is projected only onto the logits of the relevant class tokens, enabling efficient and accurate classification in a single forward pass. SALSA achieves state-of-the-art results across diverse benchmarks, demonstrating its robustness and scalability for LLM-based classification applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22689v1">Rule-Based Explanations for Retrieval-Augmented LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      If-then rules are widely used to explain machine learning models; e.g., "if employed = no, then loan application = rejected." We present the first proposal to apply rules to explain the emerging class of large language models (LLMs) with retrieval-augmented generation (RAG). Since RAG enables LLM systems to incorporate retrieved information sources at inference time, rules linking the presence or absence of sources can explain output provenance; e.g., "if a Times Higher Education ranking article is retrieved, then the LLM ranks Oxford first." To generate such rules, a brute force approach would probe the LLM with all source combinations and check if the presence or absence of any sources leads to the same output. We propose optimizations to speed up rule generation, inspired by Apriori-like pruning from frequent itemset mining but redefined within the scope of our novel problem. We conclude with qualitative and quantitative experiments demonstrating our solutions' value and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13586v3">Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs</a></div>
    <div class="paper-meta">
      📅 2025-10-26
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has opened new opportunities for creating dynamic non-player characters (NPCs) in gaming environments, enabling both functional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which evaluates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
    </details>
</div>
