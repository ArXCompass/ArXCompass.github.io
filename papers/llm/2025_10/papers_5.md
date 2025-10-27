# llm - 2025_10

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15870v1">OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Technical Report. Code: https://github.com/NVlabs/OmniVinci
    </div>
    <details class="paper-abstract">
      Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15859v1">InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 17 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown substantial advances through reinforcement learning (RL), particularly in domains where rewards can be programmatically verified, such as mathematics and code. In these areas, models benefit from a well-defined operational base guided by explicit rule-based objectives. However, this progress reveals a significant limitation: in open-ended domains where rewards are ambiguous, subjective, or context-dependent, such as creative writing, scientific reasoning, and notably medical consultation, robust reward functions are lacking, making these areas challenging for current RL strategies. To bridge this gap, we introduce ORBIT, an open-ended rubric-based incremental training framework specifically designed for high-stakes medical dialogue. ORBIT integrates syn- thetic dialogue generation with the dynamic creation of rubrics, employing these rubrics to direct an incremental RL process. In particular, this approach does not depend on external medical knowledge or manual rules, instead utilizing rubric-guided feedback to shape learning. When implemented on the Qwen3-4B-Instruct model, our method can greatly enhance its performance on the HealthBench-Hard benchmark from 7.0 to 27.2 using only 2k samples, thus achieving state-of-the-art results for models of this scale. Our analysis confirms that rubric-driven RL fos-ters consistent performance gains across diverse consultation scenarios, going beyond simple numerical improvements. These findings underscore rubric-based feedback as a scalable strategy for advancing LLMs in intricate, open-ended tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15746v1">LLMs Judge Themselves: A Game-Theoretic Framework for Human-Aligned Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Ideal or real - that is the question.In this work, we explore whether principles from game theory can be effectively applied to the evaluation of large language models (LLMs). This inquiry is motivated by the growing inadequacy of conventional evaluation practices, which often rely on fixed-format tasks with reference answers and struggle to capture the nuanced, subjective, and open-ended nature of modern LLM behavior. To address these challenges, we propose a novel alternative: automatic mutual evaluation, where LLMs assess each other's output through self-play and peer review. These peer assessments are then systematically compared with human voting behavior to evaluate their alignment with human judgment. Our framework incorporates game-theoretic voting algorithms to aggregate peer reviews, enabling a principled investigation into whether model-generated rankings reflect human preferences. Empirical results reveal both convergences and divergences between theoretical predictions and human evaluations, offering valuable insights into the promises and limitations of mutual evaluation. To the best of our knowledge, this is the first work to jointly integrate mutual evaluation, game-theoretic aggregation, and human-grounded validation for evaluating the capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15729v1">FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been explored for integration with collaborative filtering (CF)-based recommendation systems, which are crucial for personalizing user experiences. However, a key challenge is that LLMs struggle to interpret the latent, non-semantic embeddings produced by CF approaches, limiting recommendation effectiveness and further applications. To address this, we propose FACE, a general interpretable framework that maps CF embeddings into pre-trained LLM tokens. Specifically, we introduce a disentangled projection module to decompose CF embeddings into concept-specific vectors, followed by a quantized autoencoder to convert continuous embeddings into LLM tokens (descriptors). Then, we design a contrastive alignment objective to ensure that the tokens align with corresponding textual signals. Hence, the model-agnostic FACE framework achieves semantic alignment without fine-tuning LLMs and enhances recommendation performance by leveraging their pre-trained capabilities. Empirical results on three real-world recommendation datasets demonstrate performance improvements in benchmark models, with interpretability studies confirming the interpretability of the descriptors. Code is available in https://github.com/YixinRoll/FACE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13593v4">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      We introduce time-to-unsafe-sampling, a novel safety measure for generative models, defined as the number of generations required by a large language model (LLM) to trigger an unsafe (e.g., toxic) response. While providing a new dimension for prompt-adaptive safety evaluation, quantifying time-to-unsafe-sampling is challenging: unsafe outputs are often rare in well-aligned models and thus may not be observed under any feasible sampling budget. To address this challenge, we frame this estimation problem as one of survival analysis. We build on recent developments in conformal prediction and propose a novel calibration technique to construct a lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt with rigorous coverage guarantees. Our key technical innovation is an optimized sampling-budget allocation scheme that improves sample efficiency while maintaining distribution-free guarantees. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15690v1">MirrorFuzz: Leveraging LLM and Shared Bugs for Deep Learning Framework APIs Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted for publication in IEEE Transactions on Software Engineering (TSE), 2025
    </div>
    <details class="paper-abstract">
      Deep learning (DL) frameworks serve as the backbone for a wide range of artificial intelligence applications. However, bugs within DL frameworks can cascade into critical issues in higher-level applications, jeopardizing reliability and security. While numerous techniques have been proposed to detect bugs in DL frameworks, research exploring common API patterns across frameworks and the potential risks they entail remains limited. Notably, many DL frameworks expose similar APIs with overlapping input parameters and functionalities, rendering them vulnerable to shared bugs, where a flaw in one API may extend to analogous APIs in other frameworks. To address this challenge, we propose MirrorFuzz, an automated API fuzzing solution to discover shared bugs in DL frameworks. MirrorFuzz operates in three stages: First, MirrorFuzz collects historical bug data for each API within a DL framework to identify potentially buggy APIs. Second, it matches each buggy API in a specific framework with similar APIs within and across other DL frameworks. Third, it employs large language models (LLMs) to synthesize code for the API under test, leveraging the historical bug data of similar APIs to trigger analogous bugs across APIs. We implement MirrorFuzz and evaluate it on four popular DL frameworks (TensorFlow, PyTorch, OneFlow, and Jittor). Extensive evaluation demonstrates that MirrorFuzz improves code coverage by 39.92\% and 98.20\% compared to state-of-the-art methods on TensorFlow and PyTorch, respectively. Moreover, MirrorFuzz discovers 315 bugs, 262 of which are newly found, and 80 bugs are fixed, with 52 of these bugs assigned CNVD IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15685v1">Leveraging LLMs for Context-Aware Implicit Textual and Multimodal Hate Speech Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 8 pages, 9 figures, submitted to LREC 2026
    </div>
    <details class="paper-abstract">
      This research introduces a novel approach to textual and multimodal Hate Speech Detection (HSD), using Large Language Models (LLMs) as dynamic knowledge bases to generate background context and incorporate it into the input of HSD classifiers. Two context generation strategies are examined: one focused on named entities and the other on full-text prompting. Four methods of incorporating context into the classifier input are compared: text concatenation, embedding concatenation, a hierarchical transformer-based fusion, and LLM-driven text enhancement. Experiments are conducted on the textual Latent Hatred dataset of implicit hate speech and applied in a multimodal setting on the MAMI dataset of misogynous memes. Results suggest that both the contextual information and the method by which it is incorporated are key, with gains of up to 3 and 6 F1 points on textual and multimodal setups respectively, from a zero-context baseline to the highest-performing system, based on embedding concatenation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14846v2">Where to Search: Measure the Prior-Structured Search Space of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 10 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      The generate-filter-refine (iterative) paradigm based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07581v2">Expanding the Action Space of LLMs to Reason Beyond Language</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful reasoners in natural language, but their actions are typically confined to outputting vocabulary tokens. As a result, interactions with external environments -- such as symbolic operators or simulators -- must be expressed through text in predefined formats, parsed, and routed to external interfaces. This overloads the model's language with both reasoning and control duties, and requires a hand-crafted parser, external to the LLM. To address this, we decouple environment interactions from language by internalizing them in an Expanded Action space (ExpA), beyond the vocabulary. The model starts reasoning in the default language environment, but may trigger routing actions and switch to an external environment at any time. From there, the model can only invoke environment-specific actions, receive feedback from the environment, and potentially route back to language as a result. To promote effective exploration of the expanded action space and new environments, we introduce ExpA Reinforcement Learning (EARL) with counterfactual policy optimization. On tasks requiring multi-turn interactions and contingent planning, EARL outperforms strong baselines with vocabulary-constrained actions. It performs robustly across calculator-based multi-task learning and, in the partially observed sorting problem, achieves perfect Sort-4 accuracy while self-discovering an efficient algorithm competitive with classical designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26184v4">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation (RG) are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recently proposed ARGUE framework for RG evaluation. We present analysis of Auto-ARGUE on the RG pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15614v1">HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      As language models are increasingly used in scientific workflows, evaluating their ability to propose sets of explanations-not just a single correct answer-becomes critical. Many scientific problems are underdetermined: multiple, mechanistically distinct hypotheses are consistent with the same observations. We introduce HypoSpace, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators: Validity (precision of proposals consistent with observations), Uniqueness (non-redundancy among proposals), and Recovery (coverage of the enumerated admissible set). We instantiate HypoSpace in three structured domains with deterministic validators and exactly enumerated hypothesis spaces: (i) causal graphs from perturbations, (ii) gravity-constrained 3D voxel reconstruction from top-down projections, and (iii) Boolean genetic interactions. Across instruction-tuned and reasoning-focused models, Validity often remains high while Uniqueness and Recovery degrade as the admissible space grows, revealing mode collapse that is invisible to correctness-only metrics. HypoSpace offers a controlled probe-rather than a leaderboard-for methods that explicitly explore and cover admissible explanation spaces. Code is available at: https://github.com/CTT-Pavilion/_HypoSpace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15561v1">Finetuning LLMs for EvaCun 2025 token prediction shared task</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      In this paper, we present our submission for the token prediction task of EvaCun 2025. Our sys-tems are based on LLMs (Command-R, Mistral, and Aya Expanse) fine-tuned on the task data provided by the organizers. As we only pos-sess a very superficial knowledge of the subject field and the languages of the task, we simply used the training data without any task-specific adjustments, preprocessing, or filtering. We compare 3 different approaches (based on 3 different prompts) of obtaining the predictions, and we evaluate them on a held-out part of the data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15522v1">Latent Reasoning in LLMs as a Vocabulary-Space Superposition</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong reasoning abilities with chain-of-thought prompting, but explicit reasoning introduces substantial computational overhead. Recent work on latent reasoning reduces this cost by reasoning in latent space without explicit supervision, but performance drops significantly. Our preliminary experiments suggest that this degradation stems from the unstructured latent space, which makes fitting latent tokens difficult. To address this, we restrict the latent space to the column space of the LLM vocabulary, treating latent reasoning as a superposition over vocabulary probabilities. Once latent reasoning concludes, it collapses into an eigenstate of explicit reasoning to yield the final answer. Based on this idea, we propose Latent-SFT, a two-stage learning framework. In the first stage, we design two specialized attention masks to guide the Latent Token Encoder in generating latent tokens, allowing the LLM to produce the correct answer conditioned on them. In the second stage, the Latent Token Encoder is discarded, and the LLM is directly trained to generate these latent tokens autonomously for latent reasoning, optimized with KL and CE losses. Latent-SFT sets a new state of the art on GSM8k, matching explicit SFT performance while cutting reasoning chains by up to 4 times and outperforming prior latent methods. On Math500 and AIME24, lexical probability-based latent reasoning also clearly surpasses hidden-state-based approaches. Our metrics of effective compression rate and effective global parallelism further show that latent reasoning is both the compression of a single path and the superposition of multiple paths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15513v1">Temporal Referential Consistency: Do LLMs Favor Sequences Over Absolute Time References?</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 EMNLP Main Long Paper 2025
    </div>
    <details class="paper-abstract">
      The increasing acceptance of large language models (LLMs) as an alternative to knowledge sources marks a significant paradigm shift across various domains, including time-sensitive fields such as law, healthcare, and finance. To fulfill this expanded role, LLMs must not only be factually accurate but also demonstrate consistency across temporal dimensions, necessitating robust temporal reasoning capabilities. Despite this critical requirement, efforts to ensure temporal consistency in LLMs remain scarce including noticeable absence of endeavors aimed at evaluating or augmenting LLMs across temporal references in time-sensitive inquiries. In this paper, we seek to address this gap by introducing a novel benchmark entitled temporal referential consistency, accompanied by a resource TEMP-ReCon designed to benchmark a wide range of both open-source and closed-source LLMs with various linguistic contexts characterized by differing resource richness (including English, French, and Romanian). The findings emphasis that LLMs do exhibit insufficient temporal referent consistency. To address this, we propose \newmodel, a reasoning path alignment-based model that aims to enhance the temporal referential consistency of LLMs. Our empirical experiments substantiate the efficacy of UnTRaP compared to several baseline models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15502v1">The Road Less Traveled: Enhancing Exploration in LLMs via Sequential Sampling</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has been pivotal in enhancing the reasoning capabilities of large language models (LLMs), but it often suffers from limited exploration and entropy collapse, where models exploit a narrow set of solutions, leading to a loss of sampling diversity and subsequently preventing RL from further improving performance. This issue is exacerbated in parallel sampling methods, where multiple outputs are drawn from the same distribution, potentially causing the model to converge to similar solutions. We propose SESA, a novel SEquential SAmpling framework that mitigates this challenge by generating diverse solution sketches sequentially before expanding them into full reasoning paths. This approach ensures broader exploration by conditioning each new output on previous ones, promoting diversity throughout the process and preventing policy collapse. Our experiments on a synthetic task show that sequential sampling consistently outperforms traditional RL methods in terms of path diversity and recovery from collapse. Further evaluations on real-world tasks demonstrate that SESA improves both the exploration of valid strategies and the overall performance of LLMs. On three agent benchmarks, SESA lifts success rates by $+0.25$, $+0.42$, and $+0.07$ absolute over the base model (up to an additional $211\%$ relative improvement over baseline RL), underscoring its exploration advantage. This work introduces a structured approach to exploration, paving the way for more effective and diverse reasoning in RL-trained LLMs. Our code is released at https://github.com/MuLabPKU/sesa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15494v1">An Experimental Study of Real-Life LLM-Proposed Performance Improvements</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate code, but can they generate fast code? In this paper, we study this question using a dataset of 65 real-world tasks mined from open-source Java programs. We specifically select tasks where developers achieved significant speedups, and employ an automated pipeline to generate patches for these issues using two leading LLMs under four prompt variations. By rigorously benchmarking the results against the baseline and human-authored solutions, we demonstrate that LLM-generated code indeed improves performance over the baseline in most cases. However, patches proposed by human developers outperform LLM fixes by a statistically significant margin, indicating that LLMs often fall short of finding truly optimal solutions. We further find that LLM solutions are semantically identical or similar to the developer optimization idea in approximately two-thirds of cases, whereas they propose a more original idea in the remaining one-third. However, these original ideas only occasionally yield substantial performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15455v1">CORE: Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      Mobile agents rely on Large Language Models (LLMs) to plan and execute tasks on smartphone user interfaces (UIs). While cloud-based LLMs achieve high task accuracy, they require uploading the full UI state at every step, exposing unnecessary and often irrelevant information. In contrast, local LLMs avoid UI uploads but suffer from limited capacity, resulting in lower task success rates. We propose $\textbf{CORE}$, a $\textbf{CO}$llaborative framework that combines the strengths of cloud and local LLMs to $\textbf{R}$educe UI $\textbf{E}$xposure, while maintaining task accuracy for mobile agents. CORE comprises three key components: (1) $\textbf{Layout-aware block partitioning}$, which groups semantically related UI elements based on the XML screen hierarchy; (2) $\textbf{Co-planning}$, where local and cloud LLMs collaboratively identify the current sub-task; and (3) $\textbf{Co-decision-making}$, where the local LLM ranks relevant UI blocks, and the cloud LLM selects specific UI elements within the top-ranked block. CORE further introduces a multi-round accumulation mechanism to mitigate local misjudgment or limited context. Experiments across diverse mobile apps and tasks show that CORE reduces UI exposure by up to 55.6% while maintaining task success rates slightly below cloud-only agents, effectively mitigating unnecessary privacy exposure to the cloud. The code is available at https://github.com/Entropy-Fighter/CORE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15444v1">A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-17
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Test-time scaling seeks to improve the reasoning performance of large language models (LLMs) by adding computational resources. A prevalent approach within the field is sampling-based test-time scaling methods, which enhance reasoning by generating multiple reasoning paths for a given input during inference. However, despite its practical success, the theoretical foundations remain underexplored. In this paper, we provide the first theoretical framework for analyzing sampling-based test-time scaling methods, grounded in the perspective of confidence estimation. Based on the framework, we analyze two dominant paradigms: self-consistency and perplexity, and reveal key limitations: self-consistency suffers from high estimation error while perplexity exhibits substantial modeling error and possible degradation of the estimation error convergence. To address these limitations, we introduce RPC, a hybrid method that leverages our theoretical insights through two key components: Perplexity Consistency and Reasoning Pruning. Perplexity Consistency combines the strengths of self-consistency and perplexity, boosting the convergence rate of estimation error from linear to exponential while preserving model error. Reasoning Pruning prevents degradation by eliminating low-probability reasoning paths. Both theoretical analysis and empirical results across seven benchmark datasets demonstrate that RPC has a strong potential for reducing reasoning error. Notably, RPC achieves reasoning performance comparable to self-consistency while not only enhancing confidence reliability but also reducing sampling costs by 50%. The code and resources are available at https://wnjxyk.github.io/RPC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14365v2">On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?</a></div>
    <div class="paper-meta">
      📅 2025-10-17
    </div>
    <details class="paper-abstract">
      This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce UCC-Inj, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and implicit versus explicit denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22358v2">Adaptive Budget Allocation for Orthogonal-Subspace Adapter Tuning in LLMs Continual Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from catastrophic forgetting in continual learning (CL) scenarios, where performance on previously learned tasks degrades severely while training on sequentially arriving tasks. Although pioneering CL approaches using orthogonal subspaces can mitigate task interference, they typically employ fixed budget allocation, neglecting the varying complexity across tasks and layers. Besides, recent budget-adaptive tuning methods for LLMs often adopt multi-stage paradigms that decouple optimization and budget allocation. Such decoupling results in potential misalignment, which hinders those approaches' practical application in CL scenarios. To address these limitations, we propose OA-Adapter, a novel parameter-efficient approach for continual learning in LLMs that unifies dynamic budget adaptation with orthogonal subspace learning in an end-to-end training stage. Specifically, OA-Adapter introduces a dynamic bottleneck dimension adaptation mechanism that simultaneously allocates an efficient parameter budget and optimizes task objectives without misalignment.To effectively preserve previously acquired knowledge while coordinating with the dynamic budget allocation, orthogonal constraints are applied specifically between the parameter subspace of the current task and the dynamically allocated parameter subspaces of historical tasks. Experimental results on continual learning benchmarks demonstrate that OA-Adapter outperforms state-of-the-art methods in both accuracy and parameter efficiency. OA-Adapter achieves higher average accuracy while using 58.5% fewer parameters on the standard CL benchmark, and maintains its advantages on two larger benchmarks comprising 15 tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13750v2">Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 UncertaiNLP at EMNLP 2025
    </div>
    <details class="paper-abstract">
      We propose a method for confidence estimation in retrieval-augmented generation (RAG) systems that aligns closely with the correctness of large language model (LLM) outputs. Confidence estimation is especially critical in high-stakes domains such as finance and healthcare, where the cost of an incorrect answer outweighs that of not answering the question. Our approach extends prior uncertainty quantification methods by leveraging raw feed-forward network (FFN) activations as auto-regressive signals, avoiding the information loss inherent in token logits and probabilities after projection and softmax normalization. We model confidence prediction as a sequence classification task, and regularize training with a Huber loss term to improve robustness against noisy supervision. Applied in a real-world financial industry customer-support setting with complex knowledge bases, our method outperforms strong baselines and maintains high accuracy under strict latency constraints. Experiments on Llama 3.1 8B model show that using activations from only the 16th layer preserves accuracy while reducing response latency. Our results demonstrate that activation-based confidence modeling offers a scalable, architecture-aware path toward trustworthy RAG deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08670v2">Augmenting Smart Contract Decompiler Output through Fine-grained Dependency Analysis and LLM-facilitated Semantic Recovery</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 This is the author version of the article accepted for publication in IEEE Transactions on Software Engineering
    </div>
    <details class="paper-abstract">
      Decompiler is a specialized type of reverse engineering tool extensively employed in program analysis tasks, particularly in program comprehension and vulnerability detection. However, current Solidity smart contract decompilers face significant limitations in reconstructing the original source code. In particular, the bottleneck of SOTA decompilers lies in inaccurate method identification, incorrect variable type recovery, and missing contract attributes. These deficiencies hinder downstream tasks and understanding of the program logic. To address these challenges, we propose SmartHalo, a new framework that enhances decompiler output by combining static analysis (SA) and large language models (LLM). SmartHalo leverages the complementary strengths of SA's accuracy in control and data flow analysis and LLM's capability in semantic prediction. More specifically, \system{} constructs a new data structure - Dependency Graph (DG), to extract semantic dependencies via static analysis. Then, it takes DG to create prompts for LLM optimization. Finally, the correctness of LLM outputs is validated through symbolic execution and formal verification. Evaluation on a dataset consisting of 465 randomly selected smart contract methods shows that SmartHalo significantly improves the quality of the decompiled code, compared to SOTA decompilers (e.g., Gigahorse). Notably, integrating GPT-4o with SmartHalo further enhances its performance, achieving precision rates of 87.39% for method boundaries, 90.39% for variable types, and 80.65% for contract attributes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04426v3">The simulation of judgment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Please refer to published version: https://doi.org/10.1073/pnas.2518443122
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in evaluative processes, from information filtering to assessing and addressing knowledge gaps through explanation and credibility judgments. This raises the need to examine how such evaluations are built, what assumptions they rely on, and how their strategies diverge from those of humans. We benchmark six LLMs against expert ratings--NewsGuard and Media Bias/Fact Check--and against human judgments collected through a controlled experiment. We use news domains purely as a controlled benchmark for evaluative tasks, focusing on the underlying mechanisms rather than on news classification per se. To enable direct comparison, we implement a structured agentic framework in which both models and nonexpert participants follow the same evaluation procedure: selecting criteria, retrieving content, and producing justifications. Despite output alignment, our findings show consistent differences in the observable criteria guiding model evaluations, suggesting that lexical associations and statistical priors could influence evaluations in ways that differ from contextual reasoning. This reliance is associated with systematic effects: political asymmetries and a tendency to confuse linguistic form with epistemic reliability--a dynamic we term epistemia, the illusion of knowledge that emerges when surface plausibility replaces verification. Indeed, delegating judgment to such systems may affect the heuristics underlying evaluative processes, suggesting a shift from normative reasoning toward pattern-based approximation and raising open questions about the role of LLMs in evaluative processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14660v1">An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12350v2">O-Forge: An LLM + Computer Algebra Framework for Asymptotic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models have recently demonstrated advanced capabilities in solving IMO and Putnam problems; yet their role in research mathematics has remained fairly limited. The key difficulty is verification: suggested proofs may look plausible, but cannot be trusted without rigorous checking. We present a framework, called LLM+CAS, and an associated tool, O-Forge, that couples frontier LLMs with a computer algebra systems (CAS) in an In-Context Symbolic Feedback loop to produce proofs that are both creative and symbolically verified. Our focus is on asymptotic inequalities, a topic that often involves difficult proofs and appropriate decomposition of the domain into the "right" subdomains. Many mathematicians, including Terry Tao, have suggested that using AI tools to find the right decompositions can be very useful for research-level asymptotic analysis. In this paper, we show that our framework LLM+CAS turns out to be remarkably effective at proposing such decompositions via a combination of a frontier LLM and a CAS. More precisely, we use an LLM to suggest domain decomposition, and a CAS (such as Mathematica) that provides a verification of each piece axiomatically. Using this loop, we answer a question posed by Terence Tao: whether LLMs coupled with a verifier can be used to help prove intricate asymptotic inequalities. More broadly, we show how AI can move beyond contest math towards research-level tools for professional mathematicians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14629v1">MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      The application of Large Language Models (LLMs) in recommender systems faces key challenges in delivering deep personalization and intelligent reasoning, especially for interactive scenarios. Current methods are often constrained by limited context windows and single-turn reasoning, hindering their ability to capture dynamic user preferences and proactively reason over recommendation contexts. To address these limitations, we propose MR.Rec, a novel framework that synergizes memory and reasoning for LLM-based recommendations. To achieve personalization, we develop a comprehensive Retrieval-Augmented Generation (RAG) system that efficiently indexes and retrieves relevant external memory to enhance LLM personalization capabilities. Furthermore, to enable the synergy between memory and reasoning, our RAG system goes beyond conventional query-based retrieval by integrating reasoning enhanced memory retrieval. Finally, we design a reinforcement learning framework that trains the LLM to autonomously learn effective strategies for both memory utilization and reasoning refinement. By combining dynamic memory retrieval with adaptive reasoning, this approach ensures more accurate, context-aware, and highly personalized recommendations. Extensive experiments demonstrate that MR.Rec significantly outperforms state-of-the-art baselines across multiple metrics, validating its efficacy in delivering intelligent and personalized recommendations. We will release code and data upon paper notification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14628v1">RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22544v2">HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 The submission was made prematurely. The authors plan to resubmit under the supervision of the corresponding author
    </div>
    <details class="paper-abstract">
      Video anomaly detection (VAD) is crucial for intelligent surveillance, but a significant challenge lies in identifying complex anomalies, which are events defined by intricate relationships and temporal dependencies among multiple entities rather than by isolated actions. While self-supervised learning (SSL) methods effectively model low-level spatiotemporal patterns, they often struggle to grasp the semantic meaning of these interactions. Conversely, large language models (LLMs) offer powerful contextual reasoning but are computationally expensive for frame-by-frame analysis and lack fine-grained spatial localization. We introduce HyCoVAD, Hybrid Complex Video Anomaly Detection, a hybrid SSL-LLM model that combines a multi-task SSL temporal analyzer with LLM validator. The SSL module is built upon an nnFormer backbone which is a transformer-based model for image segmentation. It is trained with multiple proxy tasks, learns from video frames to identify those suspected of anomaly. The selected frames are then forwarded to the LLM, which enriches the analysis with semantic context by applying structured, rule-based reasoning to validate the presence of anomalies. Experiments on the challenging ComplexVAD dataset show that HyCoVAD achieves a 72.5% frame-level AUC, outperforming existing baselines by 12.5% while reducing LLM computation. We release our interaction anomaly taxonomy, adaptive thresholding protocol, and code to facilitate future research in complex VAD scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11401v3">Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13193v2">ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs), with their structured representation capabilities, offer promising avenue for enhancing Retrieval Augmented Generation (RAG) systems, leading to the development of KG-RAG systems. Nevertheless, existing methods often struggle to achieve effective synergy between system effectiveness and cost efficiency, leading to neither unsatisfying performance nor excessive LLM prompt tokens and inference time. To this end, this paper proposes REMINDRAG, which employs an LLM-guided graph traversal featuring node exploration, node exploitation, and, most notably, memory replay, to improve both system effectiveness and cost efficiency. Specifically, REMINDRAG memorizes traversal experience within KG edge embeddings, mirroring the way LLMs "memorize" world knowledge within their parameters, but in a train-free manner. We theoretically and experimentally confirm the effectiveness of REMINDRAG, demonstrating its superiority over existing baselines across various benchmark datasets and LLM backbones. Our code is available at https://github.com/kilgrims/ReMindRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14565v1">Assessing Socio-Cultural Alignment and Technical Safety of Sovereign LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent trends in LLMs development clearly show growing interest in the use and application of sovereign LLMs. The global debate over sovereign LLMs highlights the need for governments to develop their LLMs, tailored to their unique socio-cultural and historical contexts. However, there remains a shortage of frameworks and datasets to verify two critical questions: (1) how well these models align with users' socio-cultural backgrounds, and (2) whether they maintain safety and technical robustness without exposing users to potential harms and risks. To address this gap, we construct a new dataset and introduce an analytic framework for extracting and evaluating the socio-cultural elements of sovereign LLMs, alongside assessments of their technical robustness. Our experimental results demonstrate that while sovereign LLMs play a meaningful role in supporting low-resource languages, they do not always meet the popular claim that these models serve their target users well. We also show that pursuing this untested claim may lead to underestimating critical quality attributes such as safety. Our study suggests that advancing sovereign LLMs requires a more extensive evaluation that incorporates a broader range of well-grounded and practical criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14548v1">LLM Agents Beyond Utility: An Open-Ended Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent LLM agents have made great use of chain of thought reasoning and function calling. As their capabilities grow, an important question arises: can this software represent not only a smart problem-solving tool, but an entity in its own right, that can plan, design immediate tasks, and reason toward broader, more ambiguous goals? To study this question, we adopt an open-ended experimental setting where we augment a pretrained LLM agent with the ability to generate its own tasks, accumulate knowledge, and interact extensively with its environment. We study the resulting open-ended agent qualitatively. It can reliably follow complex multi-step instructions, store and reuse information across runs, and propose and solve its own tasks, though it remains sensitive to prompt design, prone to repetitive task generation, and unable to form self-representations. These findings illustrate both the promise and current limits of adapting pretrained LLMs toward open-endedness, and point to future directions for training agents to manage memory, explore productively, and pursue abstract long-term goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04575v2">Are LLMs Stable Formal Logic Translators in Logical Reasoning Across Linguistically Diversified Texts?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Logical reasoning with large language models (LLMs) has received growing attention. One mainstream approach translates natural language into formal logic and then applies symbolic solvers for deduction. While effective in many tasks, these LLM-based translators often fail to generate consistent symbolic representations when the same concept appears in different linguistic forms. Such inconsistencies break logical coherence and lead to solver errors. However, most existing benchmarks lack this type of linguistic variation, which frequently occurs in real-world text, leaving the problem underexplored. To address this gap, we present SoLT, a benchmark that systematically rewrites reasoning datasets into diverse yet logically equivalent forms across multiple levels. Beyond evaluation, SoLT also provides a general method to enrich any dataset with linguistic diversity while preserving both meaning and logic. To further enhance the stability of LLM-based reasoning, we propose MenTaL, which explicitly guides models to build a concept-symbol mapping table during translation. By linking equivalent expressions to shared symbols, MenTaL maintains consistency and mitigates symbol drift. Experiments on SoLT demonstrate that LLMs indeed suffer from inconsistent symbol mapping under linguistic variation, leading to significant drops in reasoning accuracy. Meanwhile, applying MenTaL brings clear and stable performance improvements across diverse inputs. Overall, our findings reveal that overlooking linguistic diversity hides key weaknesses in LLM-based translators, and our work offers a step toward more reliable logical reasoning in varied real-world scenarios. Our code is available at https://github.com/wufeiwuwoshihua/LinguDiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14522v1">Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03867v3">Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted for oral presentation at the EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a benchmark dataset of over 1,200+ meticulously curated and diverse examples across English, Mandarin, Spanish, French, Japanese, and Korean. Each example underwent careful expert review to verify its Drivelological characteristics, involving multiple rounds of discussion and adjudication to address disagreements. Using this dataset, we evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss implied rhetorical functions altogether. These findings highlight a deep representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03479v4">Women, Infamous, and Exotic Beings: A Comparative Study of Honorific Usages in Wikipedia and LLMs for Bengali and Hindi</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted and published at EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      The obligatory use of third-person honorifics is a distinctive feature of several South Asian languages, encoding nuanced socio-pragmatic cues such as power, age, gender, fame, and social distance. In this work, (i) We present the first large-scale study of third-person honorific pronoun and verb usage across 10,000 Hindi and Bengali Wikipedia articles with annotations linked to key socio-demographic attributes of the subjects, including gender, age group, fame, and cultural origin. (ii) Our analysis uncovers systematic intra-language regularities but notable cross-linguistic differences: honorifics are more prevalent in Bengali than in Hindi, while non-honorifics dominate while referring to infamous, juvenile, and culturally exotic entities. Notably, in both languages, and more prominently in Hindi, men are more frequently addressed with honorifics than women. (iii) To examine whether large language models (LLMs) internalize similar socio-pragmatic norms, we probe six LLMs using controlled generation and translation tasks over 1,000 culturally balanced entities. We find that LLMs diverge from Wikipedia usage, exhibiting alternative preferences in honorific selection across tasks, languages, and socio-demographic attributes. These discrepancies highlight gaps in the socio-cultural alignment of LLMs and open new directions for studying how LLMs acquire, adapt, or distort social-linguistic norms. Our code and data are publicly available at https://github.com/souro/honorific-wiki-llm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14444v1">A Free Lunch in LLM Compression: Revisiting Retraining after Pruning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Models (LLMs) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer block is nearly the most resource-efficient yet achieves the best perplexity. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12217v2">HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09721v2">A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at https://github.com/lisaGuojl/LLM-Agent-SE-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10959v2">Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14401v1">The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      A growing body of multi-agent studies with Large Language Models (LLMs) explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without full visibility into payoffs and population, relying instead on heuristics, communication, and punishment. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14392v1">FairBatching: Fairness-Aware Batch Formation for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference systems face a fundamental tension between minimizing Time-to-First-Token (TTFT) latency for new requests and maintaining a high, steady token generation rate (low Time-Per-Output-Token, or TPOT) for ongoing requests. Existing stall-free batching schedulers proposed by Sarathi, while effective at preventing decode stalls, introduce significant computational unfairness. They prioritize decode tasks excessively, simultaneously leading to underutilized decode slack and unnecessary prefill queuing delays, which collectively degrade the system's overall quality of service (QoS). This work identifies the root cause of this unfairness: the non-monotonic nature of Time-Between-Tokens (TBT) as a scheduling metric and the rigid decode-prioritizing policy that fails to adapt to dynamic workload bursts. We therefore propose FairBatching, a novel LLM inference scheduler that enforces fair resource allocation between prefill and decode tasks. It features an adaptive batch capacity determination mechanism, which dynamically adjusts the computational budget to improve the GPU utilization without triggering SLO violations. Its fair and dynamic batch formation algorithm breaks away from the decode-prioritizing paradigm, allowing computation resources to be reclaimed from bursting decode tasks to serve prefill surges, achieving global fairness. Furthermore, FairBatching provides a novel load estimation method, enabling more effective coordination with upper-level schedulers. Implemented and evaluated on realistic traces, FairBatching significantly reduces TTFT tail latency by up to 2.29x while robustly maintaining TPOT SLOs, achieving overall 20.0% improvement in single-node capacity and 54.3% improvement in cluster-level capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14387v1">Can MLLMs Absorb Math Reasoning Abilities from LLMs as Free Lunch?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Math reasoning has been one crucial ability of large language models (LLMs), where significant advancements have been achieved in recent years. However, most efforts focus on LLMs by curating high-quality annotation data and intricate training (or inference) paradigms, while the math reasoning performance of multi-modal LLMs (MLLMs) remains lagging behind. Since the MLLM typically consists of an LLM and a vision block, we wonder: Can MLLMs directly absorb math reasoning abilities from off-the-shelf math LLMs without tuning? Recent model-merging approaches may offer insights into this question. However, they overlook the alignment between the MLLM and LLM, where we find that there is a large gap between their parameter spaces, resulting in lower performance. Our empirical evidence reveals two key factors behind this issue: the identification of crucial reasoning-associated layers in the model and the mitigation of the gaps in parameter space. Based on the empirical insights, we propose IP-Merging that first identifies the reasoning-associated parameters in both MLLM and Math LLM, then projects them into the subspace of MLLM, aiming to maintain the alignment, and finally merges parameters in this subspace. IP-Merging is a tuning-free approach since parameters are directly adjusted. Extensive experiments demonstrate that our IP-Merging method can enhance the math reasoning ability of MLLMs directly from Math LLMs without compromising their other capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14381v1">Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13903v2">CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by NeurIPS 2025 Conference
    </div>
    <details class="paper-abstract">
      Proprietary large language models (LLMs) exhibit strong generalization capabilities across diverse tasks and are increasingly deployed on edge devices for efficiency and privacy reasons. However, deploying proprietary LLMs at the edge without adequate protection introduces critical security threats. Attackers can extract model weights and architectures, enabling unauthorized copying and misuse. Even when protective measures prevent full extraction of model weights, attackers may still perform advanced attacks, such as fine-tuning, to further exploit the model. Existing defenses against these threats typically incur significant computational and communication overhead, making them impractical for edge deployment. To safeguard the edge-deployed LLMs, we introduce CoreGuard, a computation- and communication-efficient protection method. CoreGuard employs an efficient protection protocol to reduce computational overhead and minimize communication overhead via a propagation protocol. Extensive experiments show that CoreGuard achieves upper-bound security protection with negligible overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14365v1">On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce \nameshort{}, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and \textit{implicit} versus \textit{explicit} denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14845v3">AAVENUE: Detecting LLM Biases on NLU Tasks in AAVE via a Novel Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Published at NLP4PI @ EMNLP 2024
    </div>
    <details class="paper-abstract">
      Detecting biases in natural language understanding (NLU) for African American Vernacular English (AAVE) is crucial to developing inclusive natural language processing (NLP) systems. To address dialect-induced performance discrepancies, we introduce AAVENUE ({AAVE} {N}atural Language {U}nderstanding {E}valuation), a benchmark for evaluating large language model (LLM) performance on NLU tasks in AAVE and Standard American English (SAE). AAVENUE builds upon and extends existing benchmarks like VALUE, replacing deterministic syntactic and morphological transformations with a more flexible methodology leveraging LLM-based translation with few-shot prompting, improving performance across our evaluation metrics when translating key tasks from the GLUE and SuperGLUE benchmarks. We compare AAVENUE and VALUE translations using five popular LLMs and a comprehensive set of metrics including fluency, BARTScore, quality, coherence, and understandability. Additionally, we recruit fluent AAVE speakers to validate our translations for authenticity. Our evaluations reveal that LLMs consistently perform better on SAE tasks than AAVE-translated versions, underscoring inherent biases and highlighting the need for more inclusive NLP models. We have open-sourced our source code on GitHub and created a website to showcase our work at https://aavenuee.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07452v2">When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 32 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14348v1">Automated Extraction of Protocol State Machines from 3GPP Specifications with Domain-Informed Prompts and LLM Ensembles</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Mobile telecommunication networks are foundational to global infrastructure and increasingly support critical sectors such as manufacturing, transportation, and healthcare. The security and reliability of these networks are essential, yet depend heavily on accurate modeling of underlying protocols through state machines. While most prior work constructs such models manually from 3GPP specifications, this process is labor-intensive, error-prone, and difficult to maintain due to the complexity and frequent updates of the specifications. Recent efforts using natural language processing have shown promise, but remain limited in handling the scale and intricacy of cellular protocols. In this work, we propose SpecGPT, a novel framework that leverages large language models (LLMs) to automatically extract protocol state machines from 3GPP documents. SpecGPT segments technical specifications into meaningful paragraphs, applies domain-informed prompting with chain-of-thought reasoning, and employs ensemble methods to enhance output reliability. We evaluate SpecGPT on three representative 5G protocols (NAS, NGAP, and PFCP) using manually annotated ground truth, and show that it outperforms existing approaches, demonstrating the effectiveness of LLMs for protocol modeling at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04384v2">LLM Based Bayesian Optimization for Prompt Search</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Bayesian Optimization (BO) has been widely used to efficiently optimize expensive black-box functions with limited evaluations. In this paper, we investigate the use of BO for prompt engineering to enhance text classification with Large Language Models (LLMs). We employ an LLM-powered Gaussian Process (GP) as the surrogate model to estimate the performance of different prompt candidates. These candidates are generated by an LLM through the expansion of a set of seed prompts and are subsequently evaluated using an Upper Confidence Bound (UCB) acquisition function in conjunction with the GP posterior. The optimization process iteratively refines the prompts based on a subset of the data, aiming to improve classification accuracy while reducing the number of API calls by leveraging the prediction uncertainty of the LLM-based GP. The proposed BO-LLM algorithm is evaluated on two datasets, and its advantages are discussed in detail in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14331v1">LLM-ERM: Sample-Efficient Program Learning via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We seek algorithms for program learning that are both sample-efficient and computationally feasible. Classical results show that targets admitting short program descriptions (e.g., with short ``python code'') can be learned with a ``small'' number of examples (scaling with the size of the code) via length-first program enumeration, but the search is exponential in description length. Consequently, Gradient-based training avoids this cost yet can require exponentially many samples on certain short-program families. To address this gap, we introduce LLM-ERM, a propose-and-verify framework that replaces exhaustive enumeration with an LLM-guided search over candidate programs while retaining ERM-style selection on held-out data. Specifically, we draw $k$ candidates with a pretrained reasoning-augmented LLM, compile and check each on the data, and return the best verified hypothesis, with no feedback, adaptivity, or gradients. Theoretically, we show that coordinate-wise online mini-batch SGD requires many samples to learn certain short programs. {\em Empirically, LLM-ERM solves tasks such as parity variants, pattern matching, and primality testing with as few as 200 samples, while SGD-trained transformers overfit even with 100,000 samples}. These results indicate that language-guided program synthesis recovers much of the statistical efficiency of finite-class ERM while remaining computationally tractable, offering a practical route to learning succinct hypotheses beyond the reach of gradient-based training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14471v2">Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted as a spotlight at the 1st workshop on Multilingual and Equitable Language Technologies (MELT), COLM 2025
    </div>
    <details class="paper-abstract">
      Why do we build local large language models (LLMs)? What should a local LLM learn from the target language? Which abilities can be transferred from other languages? Do language-specific scaling laws exist? To explore these research questions, we evaluated 35 Japanese, English, and multilingual LLMs on 19 evaluation benchmarks for Japanese and English, taking Japanese as a local language. Adopting an observational approach, we analyzed correlations of benchmark scores, and conducted principal component analysis (PCA) on the scores to derive \textit{ability factors} of local LLMs. We found that training on English text can improve the scores of academic subjects in Japanese (JMMLU). In addition, it is unnecessary to specifically train on Japanese text to enhance abilities for solving Japanese code generation, arithmetic reasoning, commonsense, and reading comprehension tasks. In contrast, training on Japanese text could improve question-answering tasks about Japanese knowledge and English-Japanese translation, which indicates that abilities for solving these two tasks can be regarded as \textit{Japanese abilities} for LLMs. Furthermore, we confirmed that the Japanese abilities scale with the computational budget for Japanese text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21638v3">R1-Ranker: Teaching LLM Rankers to Reason</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong reasoning abilities in domains like mathematics, coding, and scientific problem-solving, yet their potential for ranking tasks, where prime examples include retrieval, recommender systems, and LLM routing, remains underexplored. Ranking requires complex reasoning across heterogeneous candidates, but existing LLM-based rankers are often domain-specific, tied to fixed backbones, and lack iterative refinement, limiting their ability to fully exploit LLMs' reasoning potential. To address these challenges, we propose R1-Ranker, a reasoning-incentive framework built on reinforcement learning, with two complementary designs: DRanker, which generates full rankings in one shot, and IRanker, which decomposes ranking into an iterative elimination process with step-wise rewards to encourage deeper reasoning. We evaluate unified R1-Rankers on nine datasets spanning recommendation, routing, and passage ranking, showing that IRanker-3B consistently achieves state-of-the-art performance, surpasses larger 7B models on some tasks, and yields a 15.7% average relative improvement. Ablation and generalization experiments further confirm the critical role of reinforcement learning and iterative reasoning, with IRanker-3B improving zero-shot performance by over 9% on out-of-domain tasks and reasoning traces boosting other LLMs by up to 22.87%. These results demonstrate that unifying diverse ranking tasks with a single reasoning-driven foundation model is both effective and essential for advancing LLM reasoning in ranking scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14278v1">PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14277v1">GenLARP: Enabling Immersive Live Action Role-Play through LLM-Generated Worlds and Characters</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We introduce GenLARP, a virtual reality (VR) system that transforms personalized stories into immersive live action role-playing (LARP) experiences. GenLARP enables users to act as both creators and players, allowing them to design characters based on their descriptions and live in the story world. Generative AI and agents powered by Large Language Models (LLMs) enrich these experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08615v3">Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Mathematical reasoning serves as a crucial testbed for the intelligence of large language models (LLMs), and math word problems (MWPs) are a popular type of math problems. Most MWP datasets consist of problems containing only the necessary information, while problems with distracting and excessive conditions are often overlooked. Prior works have tested popular LLMs and found a dramatic performance drop in the presence of distracting conditions. However, datasets of MWPs with distracting conditions are limited, and most suffer from lower levels of difficulty and out-of-context expressions. This makes distracting conditions easy to identify and exclude, thus reducing the credibility of benchmarking on them. Moreover, when adding distracting conditions, the reasoning and answers may also change, requiring intensive labor to check and write the solutions. To address these issues, we design an iterative framework to generate distracting conditions using LLMs. We develop a set of prompts to revise MWPs from different perspectives and cognitive levels, encouraging the generation of distracting conditions as well as suggestions for further revision. Another advantage is the shared solutions between original and revised problems: we explicitly guide the LLMs to generate distracting conditions that do not alter the original solutions, thus avoiding the need to generate new solutions. This framework is efficient and easy to deploy, reducing the overhead of generating MWPs with distracting conditions while maintaining data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14253v1">Towards Agentic Self-Learning LLMs in Search Environment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at https://github.com/forangel2014/Towards-Agentic-Self-Learning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13921v2">APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong reasoning and task planning capabilities but remain fundamentally limited in physical interaction modeling. Existing approaches integrate perception via Vision-Language Models (VLMs) or adaptive decision-making through Reinforcement Learning (RL), but they fail to capture dynamic object interactions or require task-specific training, limiting their real-world applicability. We introduce APEX (Anticipatory Physics-Enhanced Execution), a framework that equips LLMs with physics-driven foresight for real-time task planning. APEX constructs structured graphs to identify and model the most relevant dynamic interactions in the environment, providing LLMs with explicit physical state updates. Simultaneously, APEX provides low-latency forward simulations of physically feasible actions, allowing LLMs to select optimal strategies based on predictive outcomes rather than static observations. We evaluate APEX on three benchmarks designed to assess perception, prediction, and decision-making: (1) Physics Reasoning Benchmark, testing causal inference and object motion prediction; (2) Tetris, evaluating whether physics-informed prediction enhances decision-making performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance, assessing the immediate integration of perception and action feasibility analysis. APEX significantly outperforms standard LLMs and VLM-based models, demonstrating the necessity of explicit physics reasoning for bridging the gap between language-based intelligence and real-world task execution. The source code and experiment setup are publicly available at https://github.com/hwj20/APEX_EXP .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25271v2">RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14242v1">Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 14 pages, 6 figures, 3 tables, and 1 algorithm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce inconsistent answers when faced with different phrasings of the same prompt. In this paper, we propose Flip-Flop Consistency ($F^2C$), an unsupervised training method that improves robustness to such perturbations. $F^2C$ is composed of two key components. The first, Consensus Cross-Entropy (CCE), uses a majority vote across prompt variations to create a hard pseudo-label. The second is a representation alignment loss that pulls lower-confidence and non-majority predictors toward the consensus established by high-confidence, majority-voting variations. We evaluate our method on 11 datasets spanning four NLP tasks, with 4-15 prompt variations per dataset. On average, $F^2C$ raises observed agreement by 11.62%, improves mean $F_1$ by 8.94%, and reduces performance variance across formats by 3.29%. In out-of-domain evaluations, $F^2C$ generalizes effectively, increasing $\overline{F_1}$ and agreement while decreasing variance across most source-target pairs. Finally, when trained on only a subset of prompt perturbations and evaluated on held-out formats, $F^2C$ consistently improves both performance and agreement while reducing variance. These findings highlight $F^2C$ as an effective unsupervised method for enhancing LLM consistency, performance, and generalization under prompt perturbations. Code is available at https://github.com/ParsaHejabi/Flip-Flop-Consistency-Unsupervised-Training-for-Robustness-to-Prompt-Perturbations-in-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00527v2">Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by IEEE/RSJ IROS 2025
    </div>
    <details class="paper-abstract">
      The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v4">LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 11 pages, 6 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. In this paper, we investigate the Soft Thinking capabilities of various LLMs through a systematic analysis of their internal behavior using a suite of probing techniques. Contrary to the prevailing belief that Soft Thinking supports parallel exploration of diverse reasoning paths, our findings reveal that LLMs behave as single-threaded reasoners--they predominantly rely on the token with the highest probability in the soft input to predict the next step. This behavior induces a greedy feedback loop that suppresses alternative reasoning paths and undermines the benefits of transmitting richer information via Soft Tokens. To address this Greedy Pitfall, we propose Stochastic Soft Thinking, which introduces stochasticity to break free from this Greedy Pitfall. Our experiments demonstrate that incorporating randomness--particularly with the Gumbel-Softmax trick--can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking, resulting in superior performance across eight reasoning benchmarks. We further demonstrate that Stochastic Soft Thinking exhibits stronger exploration potential compared to conventional COT. Our findings deepen the understanding of continuous reasoning and establish the foundation for future work on improving Soft Thinking with Reinforcement Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14207v1">Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14205v1">DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF).DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences.We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews.DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios.Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09601v2">How Far Have LLMs Come Toward Automated SATD Taxonomy Construction?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Technical debt refers to suboptimal code that degrades software quality. When developers intentionally introduce such debt, it is called self-admitted technical debt (SATD). Since SATD hinders maintenance, identifying its categories is key to uncovering quality issues. Traditionally, constructing such taxonomies requires manually inspecting SATD comments and surrounding code, which is time-consuming, labor-intensive, and often inconsistent due to annotator subjectivity. In this study, we investigate to what extent large language models (LLMs) can generate SATD taxonomies. We designed a structured, LLM-driven pipeline that mirrors the taxonomy construction steps researchers typically follow. We evaluated it on SATD datasets from three domains: quantum software, smart contracts, and machine learning. It successfully recovered domain-specific categories reported in prior work, such as Layer Configuration in machine learning. It also completed taxonomy generation in under two hours and for less than $1, even on the largest dataset. These results suggest that, while full automation remains challenging, LLMs can support semi-automated SATD taxonomy construction. Furthermore, our work opens up avenues for future work, such as automated taxonomy generation in other areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15191v1">Structure-R1: Dynamically Leveraging Structural Knowledge in LLM Reasoning through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable advances in reasoning capabilities. However, their performance remains constrained by limited access to explicit and structured domain knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external information as context to augment reasoning. Nevertheless, traditional RAG systems typically operate over unstructured and fragmented text, resulting in low information density and suboptimal reasoning. To overcome these limitations, we propose \textsc{Structure-R1}, a novel framework that transforms retrieved content into structured representations optimized for reasoning. Leveraging reinforcement learning, \textsc{Structure-R1} learns a content representation policy that dynamically generates and adapts structural formats based on the demands of multi-step reasoning. Unlike prior methods that rely on fixed schemas, our approach adopts a generative paradigm capable of producing task-specific structures tailored to individual queries. To ensure the quality and reliability of these representations, we introduce a self-reward structural verification mechanism that checks whether the generated structures are both correct and self-contained. Extensive experiments on seven knowledge-intensive benchmarks show that \textsc{Structure-R1} consistently achieves competitive performance with a 7B-scale backbone model and matches the performance of much larger models. Additionally, our theoretical analysis demonstrates how structured representations enhance reasoning by improving information density and contextual clarity. Our code and data are available at: https://github.com/jlwu002/sr1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15188v1">OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15152v1">Tail-Optimized Caching for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Prompt caching is critical for reducing latency and cost in LLM inference: OpenAI and Anthropic report up to 50-90% cost savings through prompt reuse. Despite its widespread success, little is known about what constitutes an optimal prompt caching policy, particularly when optimizing tail latency, a metric of central importance to practitioners. The widely used Least Recently Used (LRU) policy can perform arbitrarily poor on this metric, as it is oblivious to the heterogeneity of conversation lengths. To address this gap, we propose Tail-Optimized LRU, a simple two-line modification that reallocates KV cache capacity to prioritize high-latency conversations by evicting cache entries that are unlikely to affect future turns. Though the implementation is simple, we prove its optimality under a natural stochastic model of conversation dynamics, providing the first theoretical justification for LRU in this setting, a result that may be of independent interest to the caching community. Experimentally, on real conversation data WildChat, Tail-Optimized LRU achieves up to 27.5% reduction in P90 tail Time to First Token latency and 23.9% in P95 tail latency compared to LRU, along with up to 38.9% decrease in SLO violations of 200ms. We believe this provides a practical and theoretically grounded option for practitioners seeking to optimize tail latency in real-world LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15144v1">HugAgent: Evaluating LLMs in Simulating Human-Like Individual Reasoning on Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 To appear in NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models (LAW)
    </div>
    <details class="paper-abstract">
      Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (https://anonymous.4open.science/r/HugAgent) and TraceYourThinking (https://anonymous.4open.science/r/trace-your-thinking).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15125v1">Latent Topic Synthesis: Leveraging LLMs for Electoral Ad Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Under-submission
    </div>
    <details class="paper-abstract">
      Social media platforms play a pivotal role in shaping political discourse, but analyzing their vast and rapidly evolving content remains a major challenge. We introduce an end-to-end framework for automatically generating an interpretable topic taxonomy from an unlabeled corpus. By combining unsupervised clustering with prompt-based labeling, our method leverages large language models (LLMs) to iteratively construct a taxonomy without requiring seed sets or domain expertise. We apply this framework to a large corpus of Meta (previously known as Facebook) political ads from the month ahead of the 2024 U.S. Presidential election. Our approach uncovers latent discourse structures, synthesizes semantically rich topic labels, and annotates topics with moral framing dimensions. We show quantitative and qualitative analyses to demonstrate the effectiveness of our framework. Our findings reveal that voting and immigration ads dominate overall spending and impressions, while abortion and election-integrity achieve disproportionate reach. Funding patterns are equally polarized: economic appeals are driven mainly by conservative PACs, abortion messaging splits between pro- and anti-rights coalitions, and crime-and-justice campaigns are fragmented across local committees. The framing of these appeals also diverges--abortion ads emphasize liberty/oppression rhetoric, while economic messaging blends care/harm, fairness/cheating, and liberty/oppression narratives. Topic salience further reveals strong correlations between moral foundations and issues. Demographic targeting also emerges. This work supports scalable, interpretable analysis of political messaging on social media, enabling researchers, policymakers, and the public to better understand emerging narratives, polarization dynamics, and the moral underpinnings of digital political communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15112v1">AndroByte: LLM-Driven Privacy Analysis through Bytecode Summarization and Dynamic Dataflow Call Graph Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted at the Annual Computer Security Applications Conference (ACSAC) 2025
    </div>
    <details class="paper-abstract">
      With the exponential growth in mobile applications, protecting user privacy has become even more crucial. Android applications are often known for collecting, storing, and sharing sensitive user information such as contacts, location, camera, and microphone data often without the user's clear consent or awareness raising significant privacy risks and exposure. In the context of privacy assessment, dataflow analysis is particularly valuable for identifying data usage and potential leaks. Traditionally, this type of analysis has relied on formal methods, heuristics, and rule-based matching. However, these techniques are often complex to implement and prone to errors, such as taint explosion for large programs. Moreover, most existing Android dataflow analysis methods depend heavily on predefined list of sinks, limiting their flexibility and scalability. To address the limitations of these existing techniques, we propose AndroByte, an AI-driven privacy analysis tool that leverages LLM reasoning on bytecode summarization to dynamically generate accurate and explainable dataflow call graphs from static code analysis. AndroByte achieves a significant F\b{eta}-Score of 89% in generating dynamic dataflow call graphs on the fly, outperforming the effectiveness of traditional tools like FlowDroid and Amandroid in leak detection without relying on predefined propagation rules or sink lists. Moreover, AndroByte's iterative bytecode summarization provides comprehensive and explainable insights into dataflow and leak detection, achieving high, quantifiable scores based on the G-Eval metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15081v1">A Generalizable Rhetorical Strategy Annotation Model Using LLM-based Debate Simulation and Labelling</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Rhetorical strategies are central to persuasive communication, from political discourse and marketing to legal argumentation. However, analysis of rhetorical strategies has been limited by reliance on human annotation, which is costly, inconsistent, difficult to scale. Their associated datasets are often limited to specific topics and strategies, posing challenges for robust model development. We propose a novel framework that leverages large language models (LLMs) to automatically generate and label synthetic debate data based on a four-part rhetorical typology (causal, empirical, emotional, moral). We fine-tune transformer-based classifiers on this LLM-labeled dataset and validate its performance against human-labeled data on this dataset and on multiple external corpora. Our model achieves high performance and strong generalization across topical domains. We illustrate two applications with the fine-tuned model: (1) the improvement in persuasiveness prediction from incorporating rhetorical strategy labels, and (2) analyzing temporal and partisan shifts in rhetorical strategies in U.S. Presidential debates (1960-2020), revealing increased use of affective over cognitive argument in U.S. Presidential debates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14461v3">To Err Is Human; To Annotate, SILICON? Reducing Measurement Error in LLM Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Unstructured text data annotation is foundational to management research and Large Language Models (LLMs) promise a cost-effective and scalable alternative to human annotation. The validity of insights drawn from LLM annotated data critically depends on minimizing the discrepancy between LLM assigned labels and the unobserved ground truth, as well as ensuring long-term reproducibility of results. We address the gap in the literature on LLM annotation by decomposing measurement error in LLM-based text annotation into four distinct sources: (1) guideline-induced error from inconsistent annotation criteria, (2) baseline-induced error from unreliable human reference standards, (3) prompt-induced error from suboptimal meta-instruction formatting, and (4) model-induced error from architectural differences across LLMs. We develop the SILICON methodology to systematically reduce measurement error from LLM annotation in all four sources above. Empirical validation across seven management research cases shows iteratively refined guidelines substantially increases the LLM-human agreement compared to one-shot guidelines; expert-generated baselines exhibit higher inter-annotator agreement as well as are less prone to producing misleading LLM-human agreement estimates compared to crowdsourced baselines; placing content in the system prompt reduces prompt-induced error; and model performance varies substantially across tasks. To further reduce error, we introduce a cost-effective multi-LLM labeling method, where only low-confidence items receive additional labels from alternative models. Finally, in addressing closed source model retirement cycles, we introduce an intuitive regression-based methodology to establish robust reproducibility protocols. Our evidence indicates that reducing each error source is necessary, and that SILICON supports reproducible, rigorous annotation in management research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v4">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted at ACM International Conference on AI in Finance (ICAIF)
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts arising from discrepancies between their pre-trained parametric knowledge and real-time market data. These conflicts are especially problematic in real-world investment services, where a model's inherent biases can misalign with institutional objectives, leading to unreliable recommendations. Despite this risk, the intrinsic investment biases of LLMs remain underexplored. We propose an experimental framework to investigate emergent behaviors in such conflict scenarios, offering a quantitative analysis of bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract the latent biases of models and measure their persistence. Our analysis, centered on sector, size, and momentum, reveals distinct, model-specific biases. Across most models, a tendency to prefer technology stocks, large-cap stocks, and contrarian strategies is observed. These foundational biases often escalate into confirmation bias, causing models to cling to initial judgments even when faced with increasing counter-evidence. A public leaderboard benchmarking bias across a broader set of models is available at https://linqalpha.com/leaderboard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15017v1">Active Honeypot Guardrail System: Probing and Confirming Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 6pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly vulnerable to multi-turn jailbreak attacks, where adversaries iteratively elicit harmful behaviors that bypass single-turn safety filters. Existing defenses predominantly rely on passive rejection, which either fails against adaptive attackers or overly restricts benign users. We propose a honeypot-based proactive guardrail system that transforms risk avoidance into risk utilization. Our framework fine-tunes a bait model to generate ambiguous, non-actionable but semantically relevant responses, which serve as lures to probe user intent. Combined with the protected LLM's safe reply, the system inserts proactive bait questions that gradually expose malicious intent through multi-turn interactions. We further introduce the Honeypot Utility Score (HUS), measuring both the attractiveness and feasibility of bait responses, and use a Defense Efficacy Rate (DER) for balancing safety and usability. Initial experiment on MHJ Datasets with recent attack method across GPT-4o show that our system significantly disrupts jailbreak success while preserving benign user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14973v1">Attention Is All You Need for KV Cache in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 https://vila-lab.github.io/elastic-cache-webpage/
    </div>
    <details class="paper-abstract">
      This work studies how to adaptively recompute key-value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant ${\bf MASK}$ tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose ${\bf Elastic-Cache}$, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14972v1">TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TokDrift, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14967v1">Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided at the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate two critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals, and (ii) lack of fine-grained credit assignment, where dependencies between turns are obscured, especially in long-horizon tasks. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward trajectories. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved sample efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14966v1">Identity-Link IRT for Label-Free LLM Evaluation: Preserving Additivity in TVD-MI Scores</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Pairwise comparisons of large language models using total variation distance mutual information (TVD-MI) produce binary critic decisions per pair. We show that averaging TVD-MI's binary trials yields centered-probability scores with additive structure suitable for item-response theory (IRT) without nonlinear link functions. Maximum-likelihood approaches to IRT use logistic links, but we find empirically that these transformations introduce curvature that breaks additivity: across three domains, the identity link yields median curl on raw data of 0.080-0.150 (P95 = [0.474, 0.580]), whereas probit/logit introduce substantially higher violations (median [0.245, 0.588], P95 [0.825, 2.252]). We derive this clipped-linear model from Gini entropy maximization, yielding a box-constrained least-squares formulation that handles boundary saturation. At 33% coverage, we achieve holdout RMSE $0.117 \pm 0.008$ while preserving agent rankings (Spearman $\rho = 0.972 \pm 0.015$), three times fewer evaluations than full dense. Judge robustness analysis (GPT-4o-mini vs. Llama3-70b) shows strong agreement in agent rankings ($\rho = 0.872$) and consistent identity-link advantage. TVD-MI's geometry is best preserved by identity mapping for efficient LLM evaluation, applicable to other bounded-response domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14944v1">MetaBench: A Multi-task Benchmark for Assessing LLMs in Metabolomics</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 22 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities on general text; however, their proficiency in specialized scientific domains that require deep, interconnected knowledge remains largely uncharacterized. Metabolomics presents unique challenges with its complex biochemical pathways, heterogeneous identifier systems, and fragmented databases. To systematically evaluate LLM capabilities in this domain, we introduce MetaBench, the first benchmark for metabolomics assessment. Curated from authoritative public resources, MetaBench evaluates five capabilities essential for metabolomics research: knowledge, understanding, grounding, reasoning, and research. Our evaluation of 25 open- and closed-source LLMs reveals distinct performance patterns across metabolomics tasks: while models perform well on text generation tasks, cross-database identifier grounding remains challenging even with retrieval augmentation. Model performance also decreases on long-tail metabolites with sparse annotations. With MetaBench, we provide essential infrastructure for developing and evaluating metabolomics AI systems, enabling systematic progress toward reliable computational tools for metabolomics research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23339v2">VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 6 pages, 1 figure, 1 algorithm, 5 tables, to be published in ISPACS 2025, unabridged version exists as arXiv:2506.23339v1
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate substantial promise for advancing scientific discovery, yet their deployment in disciplines demanding factual precision and specialized domain constraints presents significant challenges. Within molecular design for pharmaceutical development, these models can propose innovative molecular modifications but frequently generate chemically infeasible structures. We introduce VALID-Mol, a comprehensive framework that integrates chemical validation with LLM-driven molecular design, achieving an improvement in valid chemical structure generation from 3% to 83%. Our methodology synthesizes systematic prompt optimization, automated chemical verification, and domain-adapted fine-tuning to ensure dependable generation of synthesizable molecules with enhanced properties. Our contribution extends beyond implementation details to provide a transferable methodology for scientifically-constrained LLM applications with measurable reliability enhancements. Computational analyses indicate our framework generates promising synthesis candidates with up to 17-fold predicted improvements in target binding affinity while preserving synthetic feasibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13091v2">Unmasking Hiring Bias: Platform Data Analysis and Controlled Experiments on Bias in Online Freelance Marketplaces via RAG-LLM Generated Contents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Online freelance marketplaces, a rapidly growing part of the global labor market, are creating a fair environment where professional skills are the main factor for hiring. While these platforms can reduce bias from traditional hiring, the personal information in user profiles raises concerns about ongoing discrimination. Past studies on this topic have mostly used existing data, which makes it hard to control for other factors and clearly see the effect of things like gender or race. To solve these problems, this paper presents a new method that uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to create realistic, artificial freelancer profiles for controlled experiments. This approach effectively separates individual factors, enabling a clearer statistical analysis of how different variables influence the freelancer project process. In addition to analyzing extracted data with traditional statistical methods for post-project stage analysis, our research utilizes a dataset with highly controlled variables, generated by an RAG-LLM, to conduct a simulated hiring experiment for pre-project stage analysis. The results of our experiments show that, regarding gender, while no significant preference emerged in initial hiring decisions, female freelancers are substantially more likely to receive imperfect ratings post-project stage. Regarding regional bias, a strong and consistent preference favoring US-based freelancers shows that people are more likely to be selected in the simulated experiments, perceived as more leader-like, and receive higher ratings on the live platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06948v2">Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach suffers from catastrophic forgetting: second-stage RL gradually loses SFT-acquired behaviors and inefficiently explores new patterns. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03567v3">Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14846v1">Where to Search: Measure the Prior-Structured Search Space of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 10 pages, 2 figures, 1 table
    </div>
    <details class="paper-abstract">
      The generate-filter-refine (iterative paradigm) based on large language models (LLMs) has achieved progress in reasoning, programming, and program discovery in AI+Science. However, the effectiveness of search depends on where to search, namely, how to encode the domain prior into an operationally structured hypothesis space. To this end, this paper proposes a compact formal theory that describes and measures LLM-assisted iterative search guided by domain priors. We represent an agent as a fuzzy relation operator on inputs and outputs to capture feasible transitions; the agent is thereby constrained by a fixed safety envelope. To describe multi-step reasoning/search, we weight all reachable paths by a single continuation parameter and sum them to obtain a coverage generating function; this induces a measure of reachability difficulty; and it provides a geometric interpretation of search on the graph induced by the safety envelope. We further provide the simplest testable inferences and validate them via a majority-vote instantiation. This theory offers a workable language and operational tools to measure agents and their search spaces, proposing a systematic formal description of iterative search constructed by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14824v1">Supervised Fine-Tuning or Contrastive Learning? Towards Better Multimodal LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      In information retrieval, training reranking models mainly focuses on two types of objectives: metric learning (e.g. contrastive loss to increase the predicted scores on relevant query-document pairs) and classification (binary label prediction of relevance vs. irrelevance). For BERT-style encoders, various studies have shown that contrastive learning (CL) can be more effective than discriminative (classification) learning. However, for large language models (LLMs), classification via supervised fine-tuning (SFT), which predicts ''yes'' (resp. ''no'') token for relevant (resp. irrelevant) pairs, appears more promising as it aligns well with the generative nature of LLMs. This divergence raises a central question: which objective is intrinsically better suited to LLM-based reranking, and what mechanism underlies the difference? In this work, we conduct a comprehensive comparison and analysis between CL and SFT for reranking, taking the universal multimodal retrieval (UMR) as the experimental playground. We first decompose the objectives into two components: weight, which controls the magnitude of those updates, and direction, which guides the model updates, then present a unified framework for understanding their interactions. Through probing experiments, we find that SFT provides a substantially stronger weighting scheme than CL, whereas the preferred scoring direction shows no clear winner. Taken together, these results point to a consistent advantage of SFT over CL for LLM reranking. To further validate our findings, we conduct large-scale training with SFT and present new state-of-the-art rerankers on the MRB benchmark. We also provide ablations on SFT settings and expect our findings to benefit future research and applications in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20921v2">LLM-guided Chemical Process Optimization with a Multi-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 16 pages (main manuscript without references), 4 figures
    </div>
    <details class="paper-abstract">
      Chemical process optimization maximizes production efficiency and economic performance, but optimization algorithms, including gradient-based solvers, numerical methods, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable. We present a multi-agent LLM framework that autonomously infers operating constraints from minimal process descriptions, then collaboratively guides optimization. Our AutoGen-based framework employs OpenAI's o3 model with specialized agents for constraint generation, parameter validation, simulation, and optimization guidance. Through autonomous constraint generation and iterative multi-agent optimization, the framework eliminates the need for predefined operational bounds. Validated on hydrodealkylation across cost, yield, and yield-to-cost ratio metrics, the framework achieved competitive performance with conventional methods while reducing wall-time 31-fold relative to grid search, converging in under 20 minutes. The reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs and applying domain-informed heuristics. Unlike conventional methods requiring predefined constraints, our approach uniquely combines autonomous constraint generation with interpretable parameter exploration. Model comparison reveals reasoning-capable architectures (o3, o1) are essential for successful optimization, while standard models fail to converge. This approach is particularly valuable for emerging processes and retrofit applications where operational constraints are poorly characterized or unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20934v2">Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Published at the International Conference on Software Maintenance and Evolution (ICSME'25)
    </div>
    <details class="paper-abstract">
      MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools that recommend which methods to move and where, these recommendations do not align with how expert developers perform MOVEMETHOD. Given the extensive training of Large Language Models and their reliance upon naturalness of code, they should expertly recommend which methods are misplaced in a given class and which classes are better hosts. Our formative study of 2016 LLM recommendations revealed that LLMs give expert suggestions, yet they are unreliable: up to 80% of the suggestions are hallucinations. We introduce the first LLM fully powered assistant for MOVEMETHOD refactoring that automates its whole end-to-end lifecycle, from recommendation to execution. We designed novel solutions that automatically filter LLM hallucinations using static analysis from IDEs and a novel workflow that requires LLMs to be self-consistent, critique, and rank refactoring suggestions. As MOVEMETHOD refactoring requires global, projectlevel reasoning, we solved the limited context size of LLMs by employing refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist, synergistically combines the strengths of the LLM, IDE, static analysis, and semantic relevance. In our thorough, multi-methodology empirical evaluation, we compare MM-assist with the previous state-of-the-art approaches. MM-assist significantly outperforms them: (i) on a benchmark widely used by other researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a corpus of 210 recent refactorings from Open-source software, our Recall rates improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced participants who used MM-assist to refactor their own code for one week. They rated 82.8% of MM-assist recommendations positively. This shows that MM-assist is both effective and useful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07887v2">Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14756v1">Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate hardware design tasks, including the generation of Verilog code. While early benchmarks focus primarily on functional correctness, efficient hardware design demands additional optimization for synthesis metrics such as area, delay, and power. Existing benchmarks fall short in evaluating these aspects comprehensively: they often lack optimized baselines or testbenches for verification. To address these gaps, we present Pluto, a benchmark and evaluation framework designed to assess the efficiency of LLM-generated Verilog designs. Pluto presents a comprehensive evaluation set of 114 problems with self-checking testbenches and multiple Pareto-optimal reference implementations. Experimental results show that state-of-the-art LLMs can achieve high functional correctness, reaching 78.3\% at pass@1, but their synthesis efficiency still lags behind expert-crafted implementations, with area efficiency of 63.8\%, delay efficiency of 65.9\%, and power efficiency of 64.0\% at eff@1. This highlights the need for efficiency-aware evaluation frameworks such as Pluto to drive progress in hardware-focused LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14751v1">Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Next-token prediction (NTP) has driven the success of large language models (LLMs), but it struggles with long-horizon reasoning, planning, and creative writing, with these limitations largely attributed to teacher-forced training. Multi-token prediction (MTP) partially mitigates these issues by predicting several future tokens at once, but it mostly captures short-range dependencies and offers limited improvement. We propose future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving information relevant for long-form generations. We explore two variants of FSP: handcrafted summaries, for example, a bag of words summary of the future of the sequence, and learned summaries, which use embeddings produced by a reverse language model trained from right to left. Large-scale pretraining experiments (3B and 8B-parameter models) demonstrate that FSP provides improvements over both NTP and MTP across math, reasoning, and coding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v3">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16191v2">The Last Dependency Crusade: Solving Python Dependency Conflicts with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Pre-print - Accepted at the first annual workshop on Agentic Software Engineering (AgenticSE) co-located with ASE'25
    </div>
    <details class="paper-abstract">
      Resolving Python dependency issues remains a tedious and error-prone process, forcing developers to manually trial compatible module versions and interpreter configurations. Existing automated solutions, such as knowledge-graph-based and database-driven methods, face limitations due to the variety of dependency error types, large sets of possible module versions, and conflicts among transitive dependencies. This paper investigates the use of Large Language Models (LLMs) to automatically repair dependency issues in Python programs. We propose PLLM (pronounced "plum"), a novel retrieval-augmented generation (RAG) approach that iteratively infers missing or incorrect dependencies. PLLM builds a test environment where the LLM proposes module combinations, observes execution feedback, and refines its predictions using natural language processing (NLP) to parse error messages. We evaluate PLLM on the Gistable HG2.9K dataset, a curated collection of real-world Python programs. Using this benchmark, we explore multiple PLLM configurations, including six open-source LLMs evaluated both with and without RAG. Our findings show that RAG consistently improves fix rates, with the best performance achieved by Gemma-2 9B when combined with RAG. Compared to two state-of-the-art baselines, PyEGo and ReadPyE, PLLM achieves significantly higher fix rates; +15.97\% more than ReadPyE and +21.58\% more than PyEGo. Further analysis shows that PLLM is especially effective for projects with numerous dependencies and those using specialized numerical or machine-learning libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14700v1">LLM Agents for Automated Web Vulnerability Reproduction: Are We There Yet?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in software engineering and cybersecurity tasks, including code generation, vulnerability discovery, and automated testing. One critical but underexplored application is automated web vulnerability reproduction, which transforms vulnerability reports into working exploits. Although recent advances suggest promising potential, challenges remain in applying LLM agents to real-world web vulnerability reproduction scenarios. In this paper, we present the first comprehensive evaluation of state-of-the-art LLM agents for automated web vulnerability reproduction. We systematically assess 20 agents from software engineering, cybersecurity, and general domains across 16 dimensions, including technical capabilities, environment adaptability, and user experience factors, on 3 representative web vulnerabilities. Based on the results, we select three top-performing agents (OpenHands, SWE-agent, and CAI) for in-depth evaluation on our benchmark dataset of 80 real-world CVEs spanning 7 vulnerability types and 6 web technologies. Our results reveal that while LLM agents achieve reasonable success on simple library-based vulnerabilities, they consistently fail on complex service-based vulnerabilities requiring multi-component environments. Complex environment configurations and authentication barriers create a gap where agents can execute exploit code but fail to trigger actual vulnerabilities. We observe high sensitivity to input guidance, with performance degrading by over 33% under incomplete authentication information. Our findings highlight the significant gap between current LLM agent capabilities and the demands of reliable automated vulnerability reproduction, emphasizing the need for advances in environmental adaptation and autonomous problem-solving capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16843v5">Do LLM Agents Have Regret? A Case Study in Online Learning and Games</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Camera ready version of ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly employed for (interactive) decision-making, via the development of LLM-based autonomous agents. Despite their emerging successes, the performance of LLM agents in decision-making has not been fully investigated through quantitative metrics, especially in the multi-agent setting when they interact with each other, a typical scenario in real-world LLM-agent applications. To better understand the limits of LLM agents in these interactive environments, we propose to study their interactions in benchmark decision-making settings in online learning and game theory, through the performance metric of \emph{regret}. We first empirically study the {no-regret} behaviors of LLMs in canonical (non-stationary) online learning problems, as well as the emergence of equilibria when LLM agents interact through playing repeated games. We then provide some theoretical insights into the no-regret behaviors of LLM agents, under certain assumptions on the supervised pre-training and the rationality model of human decision-makers who generate the data. Notably, we also identify (simple) cases where advanced LLMs such as GPT-4 fail to be no-regret. To promote the no-regret behaviors, we propose a novel \emph{unsupervised} training loss of \emph{regret-loss}, which, in contrast to the supervised pre-training loss, does not require the labels of (optimal) actions. We then establish the statistical guarantee of generalization bound for regret-loss minimization, followed by the optimization guarantee that minimizing such a loss may automatically lead to known no-regret learning algorithms. Our further experiments demonstrate the effectiveness of our regret-loss, especially in addressing the above ``regrettable'' cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13481v1">Tahakom LLM guidelines and receipts: from pre-training data to an Arabic LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced the field of natural language processing, enhancing capabilities in both language understanding and generation across diverse domains. However, developing LLMs for Arabic presents unique challenges. This paper explores these challenges by focusing on critical aspects such as data curation, tokenizer design, and evaluation. We detail our approach to the collection and filtration of Arabic pre-training datasets, assess the impact of various tokenizer designs on model performance, and examine the limitations of existing Arabic evaluation frameworks, for which we propose a systematic corrective methodology. To promote transparency and facilitate collaborative development, we share our data and methodologies, contributing to the advancement of language modeling, particularly for the Arabic language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13467v1">NetMCP: Network-Aware Model Context Protocol Platform for LLM Capability Extension</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) remain static in functionality after training, and extending their capabilities requires integration with external data, computation, and services. The Model Context Protocol (MCP) has emerged as a standard interface for such extensions, but current implementations rely solely on semantic matching between users' requests and server function descriptions, which makes current deployments and simulation testbeds fragile under latency fluctuations or server failures. We address this gap by enhancing MCP tool routing algorithms with real-time awareness of network and server status. To provide a controlled test environment for development and evaluation, we construct a heterogeneous experimental platform, namely Network-aware MCP (NetMCP), which offers five representative network states and build a benchmark for latency sequence generation and MCP server datasets. On top of NetMCP platform, we analyze latency sequences and propose a Semantic-Oriented and Network-Aware Routing (SONAR) algorithm, which jointly optimizes semantic similarity and network Quality of Service (QoS) metrics for adaptive tool routing. Results show that SONAR consistently improves task success rate and reduces completion time and failure number compared with semantic-only, LLM-based baselines, demonstrating the value of network-aware design for production-scale LLM systems. The code for NetMCP is available at https://github.com/NICE-HKU/NetMCP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13417v1">Assessing LLM Reasoning Through Implicit Causal Chain Discovery in Climate Discourse</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      How does a cause lead to an effect, and which intermediate causal steps explain their connection? This work scrutinizes the mechanistic causal reasoning capabilities of large language models (LLMs) to answer these questions through the task of implicit causal chain discovery. In a diagnostic evaluation framework, we instruct nine LLMs to generate all possible intermediate causal steps linking given cause-effect pairs in causal chain structures. These pairs are drawn from recent resources in argumentation studies featuring polarized discussion on climate change. Our analysis reveals that LLMs vary in the number and granularity of causal steps they produce. Although they are generally self-consistent and confident about the intermediate causal connections in the generated chains, their judgments are mainly driven by associative pattern matching rather than genuine causal reasoning. Nonetheless, human evaluations confirmed the logical coherence and integrity of the generated chains. Our baseline causal chain discovery approach, insights from our diagnostic evaluation, and benchmark dataset with causal chains lay a solid foundation for advancing future work in implicit, mechanistic causal reasoning in argumentation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13401v1">F-BFQ: Flexible Block Floating-Point Quantization Accelerator for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to Workshop on New Approaches for Addressing the Computing Requirements of LLMs and GNNs (LG-ARC) @ ISCA 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly prominent for daily tasks, from improving sound-totext translation to generating additional frames for the latest video games. With the help of LLM inference frameworks, such as llama.cpp, which support optimizations such as KV-caching and quantization, it is now easier than ever to deploy LLMs on edge devices. Quantization is fundamental to enable LLMs on resource-constrained edge devices, and llama.cpp utilizes block floating point (BFP) quantization to drastically reduce the bit width of weights and input tensors, the memory footprint, and the computational power required to run LLMs. LLMs are typically quantized with mixed BFP quantization across the model layers to reduce the loss of model accuracy due to quantization. Therefore, to efficiently accelerate across the layers of BFP-quantized LLMs, specialized accelerators need to support different BFP variants without reconfiguration. To address this issue, we propose a Flexible Block FloatingPoint Quantization (F-BFQ) accelerator, which can dynamically switch between two BFP quantization variants and perform matrix multiplication (MatMul) operations. Our initial F-BFQ accelerator design, deployed on the AMD Kria board, reduces inference time by 1.4x on average over the Arm NEON-based CPU execution across three BFP quantized LLMs while achieving 5.2 tokens per second (~3.9 words per second).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11409v6">LLMs as Hackers: Autonomous Linux Privilege Escalation Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Penetration-testing is crucial for identifying system vulnerabilities, with privilege-escalation being a critical subtask to gain elevated access to protected resources. Language Models (LLMs) presents new avenues for automating these security practices by emulating human behavior. However, a comprehensive understanding of LLMs' efficacy and limitations in performing autonomous Linux privilege-escalation attacks remains under-explored. To address this gap, we introduce hackingBuddyGPT, a fully automated LLM-driven prototype designed for autonomous Linux privilege-escalation. We curated a novel, publicly available Linux privilege-escalation benchmark, enabling controlled and reproducible evaluation. Our empirical analysis assesses the quantitative success rates and qualitative operational behaviors of various LLMs -- GPT-3.5-Turbo, GPT-4-Turbo, and Llama3 -- against baselines of human professional pen-testers and traditional automated tools. We investigate the impact of context management strategies, different context sizes, and various high-level guidance mechanisms on LLM performance. Results show that GPT-4-Turbo demonstrates high efficacy, successfully exploiting 33-83% of vulnerabilities, a performance comparable to human pen-testers (75%). In contrast, local models like Llama3 exhibited limited success (0-33%), and GPT-3.5-Turbo achieved moderate rates (16-50%). We show that both high-level guidance and state-management through LLM-driven reflection significantly boost LLM success rates. Qualitative analysis reveals both LLMs' strengths and weaknesses in generating valid commands and highlights challenges in common-sense reasoning, error handling, and multi-step exploitation, particularly with temporal dependencies. Cost analysis indicates that GPT-4-Turbo can achieve human-comparable performance at competitive costs, especially with optimized context management.
    </details>
</div>
