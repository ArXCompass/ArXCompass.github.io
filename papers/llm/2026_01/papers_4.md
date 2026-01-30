# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18582v1">From Classification to Ranking: Enhancing LLM Reasoning Capabilities for MBTI Personality Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 9 pages, 4 figures, AAAI 2026 Bridge
    </div>
    <details class="paper-abstract">
      Personality detection aims to measure an individual's corresponding personality traits through their social media posts. The advancements in Large Language Models (LLMs) offer novel perspectives for personality detection tasks. Existing approaches enhance personality trait analysis by leveraging LLMs to extract semantic information from textual posts as prompts, followed by training classifiers for categorization. However, accurately classifying personality traits remains challenging due to the inherent complexity of human personality and subtle inter-trait distinctions. Moreover, prompt-based methods often exhibit excessive dependency on expert-crafted knowledge without autonomous pattern-learning capacity. To address these limitations, we view personality detection as a ranking task rather than a classification and propose a corresponding reinforcement learning training paradigm. First, we employ supervised fine-tuning (SFT) to establish personality trait ranking capabilities while enforcing standardized output formats, creating a robust initialization. Subsequently, we introduce Group Relative Policy Optimization (GRPO) with a specialized ranking-based reward function. Unlike verification tasks with definitive solutions, personality assessment involves subjective interpretations and blurred boundaries between trait categories. Our reward function explicitly addresses this challenge by training LLMs to learn optimal answer rankings. Comprehensive experiments have demonstrated that our method achieves state-of-the-art performance across multiple personality detection benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18572v1">One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Personalization of LLMs by sociodemographic subgroup often improves user experience, but can also introduce or amplify biases and unfair outcomes across groups. Prior work has employed so-called personas, sociodemographic user attributes conveyed to a model, to study bias in LLMs by relying on a single cue to prompt a persona, such as user names or explicit attribute mentions. This disregards LLM sensitivity to prompt variations (robustness) and the rarity of some cues in real interactions (external validity). We compare six commonly used persona cues across seven open and proprietary LLMs on four writing and advice tasks. While cues are overall highly correlated, they produce substantial variance in responses across personas. We therefore caution against claims from a single persona cue and recommend future personalization research to evaluate multiple externally valid cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18563v1">An LLM-Agent-Based Framework for Age of Information Optimization in Heterogeneous Random Access Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      With the rapid expansion of the Internet of Things (IoT) and heterogeneous wireless networks, the Age of Information (AoI) has emerged as a critical metric for evaluating the performance of real-time and personalized systems. While AoI-based random access is essential for next-generation applications such as the low-altitude economy and indoor service robots, existing strategies, ranging from rule-based protocols to learning-based methods, face critical challenges, including idealized model assumptions, slow convergence, and poor generalization. In this article, we propose Reflex-Core, a novel Large Language Model (LLM) agent-based framework for AoI-driven random access in heterogeneous networks. By devising an "Observe-Reflect-Decide-Execute" closed-loop mechanism, this framework integrates Supervised Fine-Tuning (SFT) and Proximal Policy Optimization (PPO) to enable optimal, autonomous access control. Based on the Reflex-Core framework, we develop a Reflexive Multiple Access (RMA) protocol and a priority-based RMA variant for intelligent access control under different heterogeneous network settings. Experimental results demonstrate that in the investigated scenarios, the RMA protocol achieves up to a 14.9% reduction in average AoI compared with existing baselines, while the priority-based version improves the convergence rate by approximately 20%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18552v1">Unknown Unknowns: Why Hidden Intentions in LLMs Evade Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      LLMs are increasingly embedded in everyday decision-making, yet their outputs can encode subtle, unintended behaviours that shape user beliefs and actions. We refer to these covert, goal-directed behaviours as hidden intentions, which may arise from training and optimisation artefacts, or be deliberately induced by an adversarial developer, yet remain difficult to detect in practice. We introduce a taxonomy of ten categories of hidden intentions, grounded in social science research and organised by intent, mechanism, context, and impact, shifting attention from surface-level behaviours to design-level strategies of influence. We show how hidden intentions can be easily induced in controlled models, providing both testbeds for evaluation and demonstrations of potential misuse. We systematically assess detection methods, including reasoning and non-reasoning LLM judges, and find that detection collapses in realistic open-world settings, particularly under low-prevalence conditions, where false positives overwhelm precision and false negatives conceal true risks. Stress tests on precision-prevalence and precision-FNR trade-offs reveal why auditing fails without vanishingly small false positive rates or strong priors on manipulation types. Finally, a qualitative case study shows that all ten categories manifest in deployed, state-of-the-art LLMs, emphasising the urgent need for robust frameworks. Our work provides the first systematic analysis of detectability failures of hidden intentions in LLMs under open-world settings, offering a foundation for understanding, inducing, and stress-testing such behaviours, and establishing a flexible taxonomy for anticipating evolving threats and informing governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18052v2">The PIMMUR Principles: Ensuring Validity in Collective Behavior of LLM Societies</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 13 pages, 9 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed to simulate human collective behaviors, yet the methodological rigor of these "AI societies" remains under-explored. Through a systematic audit of 42 recent studies, we identify six pervasive flaws-spanning agent profiles, interaction, memory, control, unawareness, and realism (PIMMUR). Our analysis reveals that 90.7% of studies violate at least one principle, undermining simulation validity. We demonstrate that frontier LLMs correctly identify the underlying social experiment in 47.6% of cases, while 65.3% of prompts exert excessive control that pre-determines outcomes. By reproducing five representative experiments (e.g., telephone game), we show that reported collective phenomena often vanish or reverse when PIMMUR principles are enforced, suggesting that many "emergent" behaviors are methodological artifacts rather than genuine social dynamics. Our findings suggest that current AI simulations may capture model-specific biases rather than universal human social behaviors, raising critical concerns about the use of LLMs as scientific proxies for human society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18510v1">Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents excel at general tasks, they inherently struggle with continual adaptation due to the frozen weights after deployment. Conventional reinforcement learning (RL) offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting. We introduce Just-In-Time Reinforcement Learning (JitRL), a training-free framework that enables test-time policy optimization without any gradient updates. JitRL maintains a dynamic, non-parametric memory of experiences and retrieves relevant trajectories to estimate action advantages on-the-fly. These estimates are then used to directly modulate the LLM's output logits. We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective. Extensive experiments on WebArena and Jericho demonstrate that JitRL establishes a new state-of-the-art among training-free methods. Crucially, JitRL outperforms the performance of computationally expensive fine-tuning methods (e.g., WebRL) while reducing monetary costs by over 30 times, offering a scalable path for continual learning agents. The code is available at https://github.com/liushiliushi/JitRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18492v1">DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an embodied agent to navigate in a complex 3D environment according to natural language instructions. Recent progress in large language models (LLMs) has enabled language-driven navigation with improved interpretability. However, most LLM-based agents still rely on single-shot action decisions, where the model must choose one option from noisy, textualized multi-perspective observations. Due to local mismatches and imperfect intermediate reasoning, such decisions can easily deviate from the correct path, leading to error accumulation and reduced reliability in unseen environments. In this paper, we propose DV-VLN, a new VLN framework that follows a generate-then-verify paradigm. DV-VLN first performs parameter-efficient in-domain adaptation of an open-source LLaMA-2 backbone to produce a structured navigational chain-of-thought, and then verifies candidate actions with two complementary channels: True-False Verification (TFV) and Masked-Entity Verification (MEV). DV-VLN selects actions by aggregating verification successes across multiple samples, yielding interpretable scores for reranking. Experiments on R2R, RxR (English subset), and REVERIE show that DV-VLN consistently improves over direct prediction and sampling-only baselines, achieving competitive performance among language-only VLN agents and promising results compared with several cross-modal systems.Code is available at https://github.com/PlumJun/DV-VLN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18483v1">Funny or Persuasive, but Not Both: Evaluating Fine-Grained Multi-Concept Control in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted for publication at EACL main conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer strong generative capabilities, but many applications require explicit and \textit{fine-grained} control over specific textual concepts, such as humor, persuasiveness, or formality. Prior approaches in prompting and representation engineering can provide coarse or single-attribute control, but systematic evaluation of multi-attribute settings remains limited. We introduce an evaluation framework for fine-grained controllability for both single- and dual-concept scenarios, focusing on linguistically distinct concept pairs (e.g., persuasiveness vs.~humor). Surprisingly, across multiple LLMs and generative tasks, we find that performance often drops in the dual-concept setting, even though the chosen concepts should in principle be separable. This reveals a fundamental limitation of naive prompting-based control: models struggle with compositionality even when concepts are intuitively independent. Our framework provides systematic evidence of this gap and offers a principled approach for measuring the ability of future methods for multi-concept control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18457v1">Token-level Collaborative Alignment for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 11 pages, 2 figures, 7 tables, WWW 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong potential for generative recommendation by leveraging rich semantic knowledge. However, existing LLM-based recommender systems struggle to effectively incorporate collaborative filtering (CF) signals, due to a fundamental mismatch between item-level preference modeling in CF and token-level next-token prediction (NTP) optimization in LLMs. Prior approaches typically treat CF as contextual hints or representation bias, and resort to multi-stage training to reduce behavioral semantic space discrepancies, leaving CF unable to explicitly regulate LLM generation. In this work, we propose Token-level Collaborative Alignment for Recommendation (TCA4Rec), a model-agnostic and plug-and-play framework that establishes an explicit optimization-level interface between CF supervision and LLM generation. TCA4Rec consists of (i) Collaborative Tokenizer, which projects raw item-level CF logits into token-level distributions aligned with the LLM token space, and (ii) Soft Label Alignment, which integrates these CF-informed distributions with one-hot supervision to optimize a soft NTP objective. This design preserves the generative nature of LLM training while enabling collaborative alignment with essential user preference of CF models. We highlight TCA4Rec is compatible with arbitrary traditional CF models and generalizes across a wide range of decoder-based LLM recommender architectures. Moreover, it provides an explicit mechanism to balance behavioral alignment and semantic fluency, yielding generative recommendations that are both accurate and controllable. Extensive experiments demonstrate that TCA4Rec consistently improves recommendation performance across a broad spectrum of CF models and LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14995v3">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and a lack of customized modeling frameworks. This paper proposes an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategies, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation. To handle the scalability issues inherent in large-scale P2P networks, a differential attention-based critic network is introduced to efficiently extract key interaction features and enhance convergence. Experimental results demonstrate that LLM-generated strategies effectively substitute human experts. The proposed imitative expert MARL algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baseline algorithms, while maintaining robust stability. This paper provides an effective solution for the real-time decision-making of the P2P electricity market by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20984v2">Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 NeurIPS 2025 Poster
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities but typically require extensive computational resources and memory for inference. Post-training quantization (PTQ) can effectively reduce these demands by storing weights in lower bit-width formats. However, standard uniform quantization often leads to notable performance degradation, particularly in low-bit scenarios. In this work, we introduce a Grouped Lattice Vector Quantization (GLVQ) framework that assigns each group of weights a customized lattice codebook, defined by a learnable generation matrix. To address the non-differentiability of the quantization process, we adopt Babai rounding to approximate nearest-lattice-point search during training, which enables stable optimization of the generation matrices. Once trained, decoding reduces to a simple matrix-vector multiplication, yielding an efficient and practical quantization pipeline. Experiments on multiple benchmarks show that our approach achieves a better trade-off between model size and accuracy compared to existing post-training quantization baselines, highlighting its effectiveness in deploying large models under stringent resource constraints. Our source code is available on GitHub repository: https://github.com/xzhang9308/GLVQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16495v2">Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Revised
    </div>
    <details class="paper-abstract">
      Efficient parallelism is necessary for achieving low-latency, high-throughput inference with large language models (LLMs). Tensor parallelism (TP) is the state-of-the-art method for reducing LLM response latency, however GPU communications reduces combined token throughput. On the other hand, data parallelism (DP) obtains a higher throughput yet is slow in response latency. Best of both worlds does not exist, and it is not possible to combine TP and DP because of the KV cache variance across the parallelisms. We notice Sequence Parallelism (SP - Ulysses in training) has similar properties as DP but with KV cache invariance. We adapt SP to inference, and combine it with TP to get the best of both worlds. Our solution: Shift Parallelism. Shift Parallelism dynamically switches across TP and SP, and minimizes latency in low traffic without losing throughput in high traffic. The efficient GPU communications of Shift Parallelism yields up to i) 1.51x faster response in interactive workloads and ii) 50% higher throughput in batch workloads, compared to a TP-only solution. We evaluate Shift Parallelism with real-world production traces with dynamic traffic patterns as well as synthetic benchmarking patterns across models, context sizes, and arrival rates. All results affirm the same: Shift Parallelism has a better the latency vs. throughput tradeoff than TP or DP, and hence obtains low latency without degrading throughput in dynamic workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.08653v2">Prism: Towards Lowering User Cognitive Load in LLMs via Complex Intent Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models are rapidly emerging as web-native interfaces to social platforms. On the social web, users frequently have ambiguous and dynamic goals, making complex intent understanding-rather than single-turn execution-the cornerstone of effective human-LLM collaboration. Existing approaches attempt to clarify user intents through sequential or parallel questioning, yet they fall short of addressing the core challenge: modeling the logical dependencies among clarification questions. Inspired by the Cognitive Load Theory, we propose Prism, a novel framework for complex intent understanding that enables logically coherent and efficient intent clarification. Prism comprises four tailored modules: a complex intent decomposition module, which decomposes user intents into smaller, well-structured elements and identifies logical dependencies among them; a logical clarification generation module, which organizes clarification questions based on these dependencies to ensure coherent, low-friction interactions; an intent-aware reward module, which evaluates the quality of clarification trajectories via an intent-aware reward function and leverages Monte Carlo Sample to simulate user-LLM interactions for large-scale,high-quality training data generation; and a self-evolved intent tuning module, which iteratively refines the LLM's logical clarification capability through data-driven feedback and optimization. Prism consistently outperforms existing approaches across clarification interactions, intent execution, and cognitive load benchmarks. It achieves stateof-the-art logical consistency, reduces logical conflicts to 11.5%, increases user satisfaction by 14.4%, and decreases task completion time by 34.8%. All data and code are released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18375v1">Hierarchical Text Classification with LLM-Refined Taxonomies</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Hierarchical text classification (HTC) depends on taxonomies that organize labels into structured hierarchies. However, many real-world taxonomies introduce ambiguities, such as identical leaf names under similar parent nodes, which prevent language models (LMs) from learning clear decision boundaries. In this paper, we present TaxMorph, a framework that uses large language models (LLMs) to transform entire taxonomies through operations such as renaming, merging, splitting, and reordering. Unlike prior work, our method revises the full hierarchy to better match the semantics encoded by LMs. Experiments across three HTC benchmarks show that LLM-refined taxonomies consistently outperform human-curated ones in various settings up to +2.9pp. in F1. To better understand these improvements, we compare how well LMs can assign leaf nodes to parent nodes and vice versa across human-curated and LLM-refined taxonomies. We find that human-curated taxonomies lead to more easily separable clusters in embedding space. However, the LLM-refined taxonomies align more closely with the model's actual confusion patterns during classification. In other words, even though they are harder to separate, they better reflect the model's inductive biases. These findings suggest that LLM-guided refinement creates taxonomies that are more compatible with how models learn, improving HTC performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18350v1">When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong general capability but often struggle with medical terminology precision and safety-critical instruction following. We present a case study for adapter interference in safety-critical domains using a 14B-parameter base model through a two-stage LoRA pipeline: (1) domain-adaptive pre-training (PT) to inject broad medical knowledge via continued pre-training (DAPT), and (2) supervised fine-tuning (SFT) to align the model with medical question-answering behaviors through instruction-style data. To balance instruction-following ability and domain knowledge retention, we propose Weighted Adapter Merging, linearly combining SFT and PT adapters before exporting a merged base-model checkpoint. On a held-out medical validation set (F5/F6), the merged model achieves BLEU-4 = 16.38, ROUGE-1 = 20.42, ROUGE-2 = 4.60, and ROUGE-L = 11.54 under a practical decoding configuration. We further analyze decoding sensitivity and training stability with loss curves and controlled decoding comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18334v1">Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly integrated into clinical workflows, their tendency for sycophancy, prioritizing user agreement over factual accuracy, poses significant risks to patient safety. While existing evaluations often rely on subjective datasets, we introduce a robust framework grounded in medical MCQA with verifiable ground truths. We propose the Adjusted Sycophancy Score, a novel metric that isolates alignment bias by accounting for stochastic model instability, or "confusability". Through an extensive scaling analysis of the Qwen-3 and Llama-3 families, we identify a clear scaling trajectory for resilience. Furthermore, we reveal a counter-intuitive vulnerability in reasoning-optimized "Thinking" models: while they demonstrate high vanilla accuracy, their internal reasoning traces frequently rationalize incorrect user suggestions under authoritative pressure. Our results across frontier models suggest that benchmark performance is not a proxy for clinical reliability, and that simplified reasoning structures may offer superior robustness against expert-driven sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18315v1">CovertComBench: The First Domain-Specific Testbed for LLMs in Wireless Covert Communication</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 6pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into wireless networks presents significant potential for automating system design. However, unlike conventional throughput maximization, Covert Communication (CC) requires optimizing transmission utility under strict detection-theoretic constraints, such as Kullback-Leibler divergence limits. Existing benchmarks primarily focus on general reasoning or standard communication tasks and do not adequately evaluate the ability of LLMs to satisfy these rigorous security constraints. To address this limitation, we introduce CovertComBench, a unified benchmark designed to assess LLM capabilities across the CC pipeline, encompassing conceptual understanding (MCQs), optimization derivation (ODQs), and code generation (CGQs). Furthermore, we analyze the reliability of automated scoring within a detection-theoretic ``LLM-as-Judge'' framework. Extensive evaluations across state-of-the-art models reveal a significant performance discrepancy. While LLMs achieve high accuracy in conceptual identification (81%) and code implementation (83%), their performance in the higher-order mathematical derivations necessary for security guarantees ranges between 18% and 55%. This limitation indicates that current LLMs serve better as implementation assistants rather than autonomous solvers for security-constrained optimization. These findings suggest that future research should focus on external tool augmentation to build trustworthy wireless AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16891v4">LLMs as Layout Designers: Enhanced Spatial Reasoning for Content-Aware Layout Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their ability to understand and manipulate spatial relationships remains limited. Such capabilities are crucial for content-aware graphic layout design, where the goal is to arrange heterogeneous elements onto a canvas so that final design remains visually balanced and structurally feasible. This problem requires precise coordination of placement, alignment, and structural organization of multiple elements within a constrained visual space. To address this limitation, we introduce LaySPA, a reinforcement learning-based framework that augments LLM-based agents with explicit spatial reasoning capabilities for layout design. LaySPA employs hybrid reward signals that jointly capture geometric constraints, structural fidelity, and visual quality, enabling agents to navigate the canvas, model inter-element relationships, and optimize spatial arrangements. Through group-relative policy optimization, the agent generates content-aware layouts that reflect salient regions, respect spatial constraints, and produces an interpretable reasoning trace explaining placement decisions and a structured layout specification. Experimental results show that LaySPA substantially improves the generation of structurally valid and visually appealing layouts, outperforming larger general-purpose LLMs and achieving performance comparable to state-of-the-art specialized layout models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16034v2">Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18009v1">Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted at the 48th European Conference on Information Retrieval (ECIR 2026)
    </div>
    <details class="paper-abstract">
      Implicit feedback -- the main data source for training Recommender Systems (RSs) -- is inherently noisy and has been shown to negatively affect recommendation effectiveness. Denoising has been proposed as a method for removing noisy implicit feedback and improving recommendations. Prior work has focused on in-training denoising, however this requires additional data, changes to the model architecture and training procedure or fine-tuning, all of which can be costly and data hungry. In this work, we focus on post-training denoising. Different from in-training denoising, post-training denoising does not involve changing the architecture of the model nor its training procedure, and does not require additional data. Specifically, we present a method for post-training denoising user profiles using Large Language Models (LLMs) for Collaborative Filtering (CF) recommendations. Our approach prompts LLMs with (i) a user profile (user interactions), (ii) a candidate item, and (iii) its rank as given by the CF recommender, and asks the LLM to remove items from the user profile to improve the rank of the candidate item. Experiments with a state-of-the-art CF recommender and 4 open and closed source LLMs in 3 datasets show that our denoising yields improvements up to 13% in effectiveness over the original user profiles. Our code is available at https://github.com/edervishaj/denoising-user-profiles-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09897v2">PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 WWW 2026
    </div>
    <details class="paper-abstract">
      Scientific document retrieval is a critical task for enabling knowledge discovery and supporting research across diverse domains. However, existing dense retrieval methods often struggle to capture fine-grained scientific concepts in texts due to their reliance on holistic embeddings and limited domain understanding. Recent approaches leverage large language models (LLMs) to extract fine-grained semantic entities and enhance semantic matching, but they typically treat entities as independent fragments, overlooking the multi-faceted nature of scientific concepts. To address this limitation, we propose Pairwise Semantic Matching (PairSem), a framework that represents relevant semantics as entity-aspect pairs, capturing complex, multi-faceted scientific concepts. PairSem is unsupervised, base retriever-agnostic, and plug-and-play, enabling precise and context-aware matching without requiring query-document labels or entity annotations. Extensive experiments on multiple datasets and retrievers demonstrate that PairSem significantly improves retrieval performance, highlighting the importance of modeling multi-aspect semantics in scientific information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15979v2">OLAF: Towards Robust LLM-Based Annotation Framework in Empirical Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in empirical software engineering (ESE) to automate or assist annotation tasks such as labeling commits, issues, and qualitative artifacts. Yet the reliability and reproducibility of such annotations remain underexplored. Existing studies often lack standardized measures for reliability, calibration, and drift, and frequently omit essential configuration details. We argue that LLM-based annotation should be treated as a measurement process rather than a purely automated activity. In this position paper, we outline the \textbf{Operationalization for LLM-based Annotation Framework (OLAF)}, a conceptual framework that organizes key constructs: \textit{reliability, calibration, drift, consensus, aggregation}, and \textit{transparency}. The paper aims to motivate methodological discussion and future empirical work toward more transparent and reproducible LLM-based annotation in software engineering research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17971v1">LLMs as Cultural Archives: Cultural Commonsense Knowledge Graph Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 EACL 2026 MAIN
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode rich cultural knowledge learned from diverse web-scale data, offering an unprecedented opportunity to model cultural commonsense at scale. Yet this knowledge remains mostly implicit and unstructured, limiting its interpretability and use. We present an iterative, prompt-based framework for constructing a Cultural Commonsense Knowledge Graph (CCKG) that treats LLMs as cultural archives, systematically eliciting culture-specific entities, relations, and practices and composing them into multi-step inferential chains across languages. We evaluate CCKG on five countries with human judgments of cultural relevance, correctness, and path coherence. We find that the cultural knowledge graphs are better realized in English, even when the target culture is non-English (e.g., Chinese, Indonesian, Arabic), indicating uneven cultural encoding in current LLMs. Augmenting smaller LLMs with CCKG improves performance on cultural reasoning and story generation, with the largest gains from English chains. Our results show both the promise and limits of LLMs as cultural technologies and that chain-structured cultural knowledge is a practical substrate for culturally grounded NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12148v3">Many Hands Make Light Work: An LLM-based Multi-Agent System for Detecting Malicious PyPI Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 The paper has been peer-reviewed and accepted for publication to the Journal of Systems and Software (https://www.sciencedirect.com/journal/journal-of-systems-and-software)
    </div>
    <details class="paper-abstract">
      Malicious code in open-source repositories such as PyPI poses a growing threat to software supply chains. Traditional rule-based tools often overlook the semantic patterns in source code that are crucial for identifying adversarial components. Large language models (LLMs) show promise for software analysis, yet their use in interpretable and modular security pipelines remains limited. This paper presents LAMPS, a multi-agent system that employs collaborative LLMs to detect malicious PyPI packages. The system consists of four role-specific agents for package retrieval, file extraction, classification, and verdict aggregation, coordinated through the CrewAI framework. A prototype combines a fine-tuned CodeBERT model for classification with LLaMA-3 agents for contextual reasoning. LAMPS has been evaluated on two complementary datasets: D1, a balanced collection of 6,000 setup.py files, and D2, a realistic multi-file dataset with 1,296 files and natural class imbalance. On D1, LAMPS achieves 97.7% accuracy, surpassing MPHunter--one of the state-of-the-art approaches. On D2, it reaches 99.5% accuracy and 99.5% balanced accuracy, outperforming RAG-based approaches and fine-tuned single-agent baselines. McNemar's test confirmed these improvements as highly significant. The results demonstrate the feasibility of distributed LLM reasoning for malicious code detection and highlight the benefits of modular multi-agent designs in software supply chain security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17942v1">LLM-Based SQL Generation: Prompting, Self-Refinement, and Adaptive Weighted Majority Voting</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 29 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Text-to-SQL has emerged as a prominent research area, particularly with the rapid advancement of large language models (LLMs). By enabling users to query databases through natural language rather than SQL, this technology significantly lowers the barrier to data analysis. However, generating accurate SQL from natural language remains challenging due to ambiguity in user queries, the complexity of schema linking, limited generalization across SQL dialects, and the need for domain-specific understanding. In this study, we propose a Single-Agent Self-Refinement with Ensemble Voting (SSEV) pipeline built on PET-SQL that operates without ground-truth data, integrating self-refinement with Weighted Majority Voting (WMV) and its randomized variant (RWMA). Experimental results show that the SSEV achieves competitive performance across multiple benchmarks, attaining execution accuracies of 85.5% on Spider 1.0-Dev, 86.4% on Spider 1.0-Test, and 66.3% on BIRD-Dev. Building on insights from the SSEV pipeline, we further propose ReCAPAgent-SQL (Refinement-Critique-Act-Plan agent-based SQL framework) to address the growing complexity of enterprise databases and real-world Text-to-SQL tasks. The framework integrates multiple specialized agents for planning, external knowledge retrieval, critique, action generation, self-refinement, schema linking, and result validation, enabling iterative refinement of SQL predictions through agent collaboration. ReCAPAgent-SQL's WMA results achieve 31% execution accuracy on the first 100 queries of Spider 2.0-Lite, demonstrating significant improvements in handling real-world enterprise scenarios. Overall, our work facilitates the deployment of scalable Text-to-SQL systems in practical settings, supporting better data-driven decision-making at lower cost and with greater efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10769v3">Mind the Gap: Benchmarking LLM Uncertainty and Calibration with Specialty-Aware Clinical QA and Reasoning-Based Behavioural Features</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted at EACL 2026 (Main Conference)
    </div>
    <details class="paper-abstract">
      Reliable uncertainty quantification (UQ) is essential when employing large language models (LLMs) in high-risk domains such as clinical question answering (QA). In this work, we evaluate uncertainty estimation methods for clinical QA focusing, for the first time, on eleven clinical specialties and six question types, and across ten open-source LLMs (general-purpose, biomedical, and reasoning models), alongside representative proprietary models. We analyze score-based UQ methods, present a case study introducing a novel lightweight method based on behavioral features derived from reasoning-oriented models, and examine conformal prediction as a complementary set-based approach. Our findings reveal that uncertainty reliability is not a monolithic property, but one that depends on clinical specialty and question type due to shifts in calibration and discrimination. Our results highlight the need to select or ensemble models based on their distinct, complementary strengths and clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17915v1">Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      LLM agents excel when environments are mostly static and the needed information fits in a model's context window, but they often fail in open-ended investigations where explanations must be constructed by iteratively mining evidence from massive, heterogeneous operational data. These investigations exhibit hidden dependency structure: entities interact, signals co-vary, and the importance of a fact may only become clear after other evidence is discovered. Because the context window is bounded, agents must summarize intermediate findings before their significance is known, increasing the risk of discarding key evidence. ReAct-style agents are especially brittle in this regime. Their retrieve-summarize-reason loop makes conclusions sensitive to exploration order and introduces run-to-run non-determinism, producing a reliability gap where Pass-at-k may be high but Majority-at-k remains low. Simply sampling more rollouts or generating longer reasoning traces does not reliably stabilize results, since hypotheses cannot be autonomously checked as new evidence arrives and there is no explicit mechanism for belief bookkeeping and revision. In addition, ReAct entangles semantic reasoning with controller duties such as tool orchestration and state tracking, so execution errors and plan drift degrade reasoning while consuming scarce context. We address these issues by formulating investigation as abductive reasoning over a dependency graph and proposing EoG (Explanations over Graphs), a disaggregated framework in which an LLM performs bounded local evidence mining and labeling (cause vs symptom) while a deterministic controller manages traversal, state, and belief propagation to compute a minimal explanatory frontier. On a representative ITBench diagnostics task, EoG improves both accuracy and run-to-run consistency over ReAct baselines, including a 7x average gain in Majority-at-k entity F1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09635v2">LLM for Large-Scale Optimization Model Auto-Formulation: Bridging Flexibility and Standardization via Agentic Workflow</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Updated version of https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5329027
    </div>
    <details class="paper-abstract">
      Large-scale optimization is a key backbone of modern business decision-making. However, building these models is often labor-intensive and time-consuming. We address this by proposing LEAN-LLM-OPT, a LightwEight AgeNtic workflow construction framework for LLM-assisted large-scale OPTimization auto-formulation. LEAN-LLM-OPT takes as input a problem description together with associated datasets and orchestrates a team of LLM agents to produce an optimization formulation. Specifically, upon receiving a query, two upstream LLM agents dynamically construct a workflow that specifies, step-by-step, how optimization models for similar problems can be formulated. A downstream LLM agent then follows this workflow to generate the final output. The agentic workflow leverages common modeling practices to standardize the modeling process into a sequence of structured sub-tasks, offloading mechanical data-handling operations to auxiliary tools. This reduces the LLM's burden in planning and data handling, allowing us to exploit its flexibility to address unstructured components. Extensive simulations show that LEAN-LLM-OPT, instantiated with GPT-4.1 and the open source gpt-oss-20B, achieves strong performance on large-scale optimization modeling tasks and is competitive with state-of-the-art approaches. In addition, in a Singapore Airlines choice-based revenue management use case, LEAN-LLM-OPT demonstrates practical value by achieving leading performance across a range of scenarios. Along the way, we introduce Large-Scale-OR and Air-NRM, the first comprehensive benchmarks for large-scale optimization auto-formulation. The code and data of this work is available at https://github.com/CoraLiang01/lean-llm-opt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17902v1">dLLM-ASR: A Faster Diffusion LLM-based Framework for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Automatic speech recognition (ASR) systems based on large language models (LLMs) achieve superior performance by leveraging pretrained LLMs as decoders, but their token-by-token generation mechanism leads to inference latency that grows linearly with sequence length. Meanwhile, discrete diffusion large language models (dLLMs) offer a promising alternative, enabling high-quality parallel sequence generation with pretrained decoders. However, directly applying native text-oriented dLLMs to ASR leads to a fundamental mismatch between open-ended text generation and the acoustically conditioned transcription paradigm required by ASR. As a result, it introduces unnecessary difficulty and computational redundancy, such as denoising from pure noise, inflexible generation lengths, and fixed denoising steps. We propose dLLM-ASR, an efficient dLLM-based ASR framework that formulates dLLM's decoding as a prior-guided and adaptive denoising process. It leverages an ASR prior to initialize the denoising process and provide an anchor for sequence length. Building upon this prior, length-adaptive pruning dynamically removes redundant tokens, while confidence-based denoising allows converged tokens to exit the denoising loop early, enabling token-level adaptive computation. Experiments demonstrate that dLLM-ASR achieves recognition accuracy comparable to autoregressive LLM-based ASR systems and delivers a 4.44$\times$ inference speedup, establishing a practical and efficient paradigm for ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19113v3">Argument-Based Consistency in Toxicity Explanations of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 29 pages, 7 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The discourse around toxicity and LLMs in NLP largely revolves around detection tasks. This work shifts the focus to evaluating LLMs' reasoning about toxicity - from their explanations that justify a stance - to enhance their trustworthiness in downstream tasks. Despite extensive research on explainability, it is not straightforward to adopt existing methods to evaluate free-form toxicity explanation due to their over-reliance on input text perturbations, among other challenges. To account for these, we propose a novel, theoretically-grounded multi-dimensional criterion, Argument-based Consistency (ArC), that measures the extent to which LLMs' free-form toxicity explanations reflect an ideal and logical argumentation process. Based on uncertainty quantification, we develop six metrics for ArC to comprehensively evaluate the (in)consistencies in LLMs' toxicity explanations. We conduct several experiments on three Llama models (of size up to 70B) and an 8B Ministral model on five diverse toxicity datasets. Our results show that while LLMs generate plausible explanations to simple prompts, their reasoning about toxicity breaks down when prompted about the nuanced relations between the complete set of reasons, the individual reasons, and their toxicity stances, resulting in inconsistent and irrelevant responses. We open-source our code (https://github.com/uofthcdslab/ArC) and LLM-generated explanations (https://huggingface.co/collections/uofthcdslab/arc) for future works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17897v1">UniCog: Uncovering Cognitive Abilities of LLMs through Latent Mind Space Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      A growing body of research suggests that the cognitive processes of large language models (LLMs) differ fundamentally from those of humans. However, existing interpretability methods remain limited in explaining how cognitive abilities are engaged during LLM reasoning. In this paper, we propose UniCog, a unified framework that analyzes LLM cognition via a latent mind space. Formulated as a latent variable model, UniCog encodes diverse abilities from dense model activations into sparse, disentangled latent dimensions. Through extensive analysis on six advanced LLMs, including DeepSeek-V3.2 and GPT-4o, we reveal a Pareto principle of LLM cognition, where a shared reasoning core is complemented by ability-specific signatures. Furthermore, we discover that reasoning failures often manifest as anomalous intensity in latent activations. These findings opens a new paradigm in LLM analysis, providing a cognition grounded view of reasoning dynamics. Finally, leveraging these insights, we introduce a latent-informed candidate prioritization strategy, which improves reasoning performance by up to 7.5% across challenging benchmarks. Our code is available at https://github.com/milksalute/unicog.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13545v3">TruthTensor: Evaluating LLMs through Human Imitation on Prediction Market under Drift and Holistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 16 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Evaluating language models and AI agents remains fundamentally challenging because static benchmarks fail to capture real-world uncertainty, distribution shift, and the gap between isolated task accuracy and human-aligned decision-making under evolving conditions. This paper introduces TruthTensor, a novel, reproducible evaluation paradigm that measures reasoning models not only as prediction engines but as human-imitation systems operating in socially-grounded, high-entropy environments. Building on forward-looking, contamination-free tasks, our framework anchors evaluation to live prediction markets and combines probabilistic scoring to provide a holistic view of model behavior. TruthTensor complements traditional correctness metrics with drift-centric diagnostics and explicit robustness checks for reproducibility. It specify human vs. automated evaluation roles, annotation protocols, and statistical testing procedures to ensure interpretability and replicability of results. In experiments across 500+ real markets (political, economic, cultural, technological), TruthTensor demonstrates that models with similar forecast accuracy can diverge markedly in calibration, drift, and risk-sensitivity, underscoring the need to evaluate models along multiple axes (accuracy, calibration, narrative stability, cost, and resource efficiency). TruthTensor therefore operationalizes modern evaluation best practices, clear hypothesis framing, careful metric selection, transparent compute/cost reporting, human-in-the-loop validation, and open, versioned evaluation contracts, to produce defensible assessments of LLMs in real-world decision contexts. We publicly released TruthTensor at https://truthtensor.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10978v3">Does LLM Focus on the Right Words? Mitigating Context Bias in LLM-based Recommenders</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted by WWW2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), owing to their extensive open-domain knowledge and semantic reasoning capabilities, have been increasingly integrated into recommender systems (RS). However, a substantial gap remains between the pre-training objectives of LLMs and the specific requirements of recommendation tasks. To address this gap, supervised fine-tuning (SFT) is commonly performed on specially curated recommendation datasets to further enhance their predictive ability. Despite its success, SFT exhibits a critical limitation: it induces Context Bias, whereby the model over-relies on auxiliary tokens, such as task descriptions and prefix-generated tokens, while underutilizing core user interaction tokens that encode user-specific preferences. This bias not only undermines recommendation accuracy but also raises unfairness concerns. To address this issue, we propose Group Distributionally Robust Optimization-based Tuning (GDRT), a novel fine-tuning paradigm that enforces consistent model performance across token groups with varying degrees of relevance to auxiliary tokens. By adaptively upweighting underperforming groups, typically those weakly correlated with auxiliary tokens, GDRT shifts the model's attention from superficial auxiliary cues to informative user interaction tokens, thereby mitigating context bias. Extensive experiments conducted on three public datasets demonstrate that GDRT effectively mitigates context bias, yielding substantial improvements in recommendation accuracy (with an average NDCG@10 gain of 24.29%) and significantly enhancing recommendation fairness. The code is available at https://github.com/WANGBohaO-jpg/GDRT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.14506v4">InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real robot platforms, including a Toyota Human Support Robot and an ANYmal D robot with a Unitree Z1 arm. Our method achieves a 95% success rate in the `fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art LLM-based robotics planners, while using only real-time onboard computing. Project website: https://kimtienly.github.io/InteLiPlan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17817v1">Multi-Agent Collaborative Intrusion Detection for Low-Altitude Economy IoT: An LLM-Enhanced Agentic AI Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      The rapid expansion of low-altitude economy Internet of Things (LAE-IoT) networks has created unprecedented security challenges due to dynamic three-dimensional mobility patterns, distributed autonomous operations, and severe resource constraints. Traditional intrusion detection systems designed for static ground-based networks prove inadequate for tackling the unique characteristics of aerial IoT environments, including frequent topology changes, real-time detection requirements, and energy limitations. In this article, we analyze the intrusion detection requirements for LAE-IoT networks, complemented by a comprehensive review of evaluation metrics that cover detection effectiveness, response time, and resource consumption. Then, we investigate transformative potential of agentic artificial intelligence (AI) paradigms and introduce a large language model (LLM)-enabled agentic AI framework for enhancing intrusion detection in LAE-IoT networks. This leads to our proposal of a novel multi-agent collaborative intrusion detection framework that leverages specialized LLM-enhanced agents for intelligent data processing and adaptive classification. Through experimental validation, our framework demonstrates superior performance of over 90\% classification accuracy across multiple benchmark datasets. These results highlight the transformative potential of combining agentic AI principles with LLMs for next-generation LAE-IoT security systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17814v1">MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have advanced rapidly, yet heterogeneity in architecture, alignment strategies, and efficiency means that no single model is uniformly superior across tasks. In practical deployments, workloads span lightweight OCR to complex multimodal reasoning; using one MLLM for all queries either over-provisions compute on easy instances or sacrifices accuracy on hard ones. Query-level model selection (routing) addresses this tension, but extending routing from text-only LLMs to MLLMs is nontrivial due to modality fusion, wide variation in computational cost across models, and the absence of a standardized, budget-aware evaluation. We present MMR-Bench, a unified benchmark that isolates the multimodal routing problem and enables comparison under fixed candidate sets and cost models. MMR-Bench provides (i) a controlled environment with modality-aware inputs and variable compute budgets, (ii) a broad suite of vision-language tasks covering OCR, general VQA, and multimodal math reasoning, and (iii) strong single-model reference, oracle upper bounds, and representative routing policies. Using MMR-Bench, we show that incorporating multimodal signals improves routing quality. Empirically, these cues improve the cost-accuracy frontier and enable the routed system to exceed the strongest single model's accuracy at roughly 33% of its cost. Furthermore, policies trained on a subset of models and tasks generalize zero-shot to new datasets and text-only benchmarks without retuning, establishing MMR-Bench as a foundation for studying adaptive multimodal model selection and efficient MLLM deployment. The code will be available at: https://github.com/Hunter-Wrynn/MMR-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15333v2">Empowering LLMs for Structure-Based Drug Design via Exploration-Augmented Latent Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess strong representation and reasoning capabilities, but their application to structure-based drug design (SBDD) is limited by insufficient understanding of protein structures and unpredictable molecular generation. To address these challenges, we propose Exploration-Augmented Latent Inference for LLMs (ELILLM), a framework that reinterprets the LLM generation process as an encoding, latent space exploration, and decoding workflow. ELILLM explicitly explores portions of the design problem beyond the model's current knowledge while using a decoding module to handle familiar regions, generating chemically valid and synthetically reasonable molecules. In our implementation, Bayesian optimization guides the systematic exploration of latent embeddings, and a position-aware surrogate model efficiently predicts binding affinity distributions to inform the search. Knowledge-guided decoding further reduces randomness and effectively imposes chemical validity constraints. We demonstrate ELILLM on the CrossDocked2020 benchmark, showing strong controlled exploration and high binding affinity scores compared with seven baseline methods. These results demonstrate that ELILLM can effectively enhance LLMs capabilities for SBDD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17789v1">Neuro-Symbolic Verification on Instruction Following of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      A fundamental problem of applying Large Language Models (LLMs) to important applications is that LLMs do not always follow instructions, and violations are often hard to observe or check. In LLM-based agentic workflows, such violations can propagate and amplify along reasoning chains, causing task failures and system incidents. This paper presents NSVIF, a neuro-symbolic framework for verifying whether an LLM's output follows the instructions used to prompt the LLM. NSVIF is a universal, general-purpose verifier; it makes no assumption about the instruction or the LLM. NSVIF formulates instruction-following verification as a constraint-satisfaction problem by modeling user instructions as constraints. NSVIF models both logical and semantic constraints; constraint solving is done by a unified solver that orchestrates logical reasoning and semantic analysis. To evaluate NSVIF, we develop VIFBENCH, a new benchmark for instruction-following verifiers with fine-grained data labels. Experiments show that NSVIF significantly outperforms LLM-based approaches and provides interpretable feedback. We also show that feedback from NSVIF helps improve LLMs' instruction-following capability without post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15331v2">DesignerlyLoop: Forming Design Intent through Curated Reasoning for Human-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) show promise in design tasks, yet a fundamental misalignment persists: design thinking requires iterative intent formulation, while LLMs treat inputs as complete specifications. This challenges design intent formulation, where designers must progressively refine understanding through exploration. Existing tools either sacrifice exploratory flexibility for structural stability or leave reasoning implicit, failing to support human-LLM alignment. Through a formative study with eight designers, we introduce curated reasoning-enabling designers to explicitly inspect, reorganize, and selectively regenerate LLM reasoning structures. We present DesignerlyLoop, implementing this through a two-layer structure separating design intent from LLM reasoning. A study with 20 designers demonstrates that curated reasoning significantly improves design quality and creativity. Our work contributes a novel interaction paradigm for human-LLM alignment, transforming LLMs from content generators into structured reasoning partners in creative design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17769v1">Reflexa: Uncovering How LLM-Supported Reflection Scaffolding Reshapes Creativity in Creative Coding</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Creative coding requires continuous translation between evolving concepts and computational artifacts, making reflection essential yet difficult to sustain. Creators often struggle to manage ambiguous intentions, emergent outputs, and complex code, limiting depth of exploration. This work examines how large language models (LLMs) can scaffold reflection not as isolated prompts, but as a system-level mechanism shaping creative regulation. From formative studies with eight expert creators, we derived reflection challenges and design principles that informed Reflexa, an integrated scaffold combining dialogic guidance, visualized version navigation, and iterative suggestion pathways. A within-subject study with 18 participants provides an exploratory mechanism validation, showing that structured reflection patterns mediate the link between AI interaction and creative outcomes. These reflection trajectories enhanced perceived controllability, broadened exploration, and improved originality and aesthetic quality. Our findings advance HCI understanding of reflection from LLM-assisted creative practices, and provide design strategies for building LLM-based creative tools that support richer human-AI co-creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17768v1">LLM-42: Enabling Determinism in LLM Inference with Verified Speculation</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 https://github.com/microsoft/llm-42
    </div>
    <details class="paper-abstract">
      In LLM inference, the same prompt may yield different outputs across different runs. At the system level, this non-determinism arises from floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size. A straightforward way to eliminate non-determinism is to disable dynamic batching during inference, but doing so severely degrades throughput. Another approach is to make kernels batch-invariant; however, this tightly couples determinism to kernel design, requiring new implementations. This coupling also imposes fixed runtime overheads, regardless of how much of the workload actually requires determinism. Inspired by ideas from speculative decoding, we present LLM-42, a scheduling-based approach to enable determinism in LLM inference. Our key observation is that if a sequence is in a consistent state, the next emitted token is likely to be consistent even with dynamic batching. Moreover, most GPU kernels use shape-consistent reductions. Leveraging these insights, LLM-42 decodes tokens using a non-deterministic fast path and enforces determinism via a lightweight verify-rollback loop. The verifier replays candidate tokens under a fixed-shape reduction schedule, commits those that are guaranteed to be consistent across runs, and rolls back those violating determinism. LLM-42 mostly re-uses existing kernels unchanged and incurs overhead only in proportion to the traffic that requires determinism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17717v1">The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for generating data across various modalities. By transforming data from a scarce resource into a controllable asset, LLMs mitigate the bottlenecks imposed by the acquisition costs of real-world data for model training, evaluation, and system iteration. However, ensuring the high quality of LLM-generated synthetic data remains a critical challenge. Existing research primarily focuses on generation methodologies, with limited direct attention to the quality of the resulting data. Furthermore, most studies are restricted to single modalities, lacking a unified perspective across different data types. To bridge this gap, we propose the \textbf{LLM Data Auditor framework}. In this framework, we first describe how LLMs are utilized to generate data across six distinct modalities. More importantly, we systematically categorize intrinsic metrics for evaluating synthetic data from two dimensions: quality and trustworthiness. This approach shifts the focus from extrinsic evaluation, which relies on downstream task performance, to the inherent properties of the data itself. Using this evaluation system, we analyze the experimental evaluations of representative generation methods for each modality and identify substantial deficiencies in current evaluation practices. Based on these findings, we offer concrete recommendations for the community to improve the evaluation of data generation. Finally, the framework outlines methodologies for the practical application of synthetic data across different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17716v1">Do Reasoning Models Ask Better Questions? A Formal Information-Theoretic Analysis on Multi-Turn LLM Games</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Presented at the NeusymBridge Workshop at AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at many tasks but still struggle with a critical ability for LLM-based agents: asking good questions for resolving ambiguity in user requests. While prior work has explored information-seeking behavior through word games, existing benchmarks lack comprehensive evaluation frameworks that provide both final and intermediate signals based on Information Gain (IG). Moreover, they rarely provide systematic comparisons between models that use chain-of-thought reasoning and those that do not. We propose a multi-turn dialogue framework that quantitatively measures how effectively LLMs gather information through yes/no questions in a hierarchical knowledge graph environment. Our framework employs a triad of interacting LLM agents that ask questions, answer them, and update the hypothesis space. We adopt IG as the main metric, grounded in Shannon entropy, to assess query effectiveness at each turn and cumulatively. We instantiate our framework in a geographical Guess My City game setting organized in a five-level taxonomy and evaluate multiple LLM variants under fully and partially observable conditions, with and without Chain-of-Thought reasoning. Our experiments demonstrate that, among the evaluated models, the ones with explicit reasoning capabilities achieve higher IG per turn and reach solutions in fewer steps, particularly in partially observable settings. Analysis of reasoning traces reveals that smaller models compensate for limited capacity through more aggressive exploration of candidate questions, while larger models exhibit higher assertiveness in selecting optimal queries, generating candidates with greater potential IG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01659v3">From Contrast to Commonality: Audio Commonality Captioning for Enhanced Audio-Text Cross-modal Understanding in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Audio Captioning (AC) plays a pivotal role in enhancing audio-text cross-modal understanding during the pretraining and finetuning of Multimodal LLMs (MLLMs). To strengthen this alignment, recent works propose Audio Difference Captioning (ADC), which takes multiple audio inputs and encourages the model to describe their differences, thereby promoting fine-grained discrimination. However, despite its effectiveness, ADC introduces a semantic gap between input audios-often rich in diverse events-and the brief, difference-focused short caption. This deviation from AC-style task causes a mismatch with the pretraining objective, leading to catastrophic forgetting. To address this, we propose Audio Commonality Captioning (ACC), a comparably challenging but gentler alternative that guides the model to capture shared semantics across audio clips rather than detailed differences. Experiments show that ACC not only improves audio-text understanding on captioning benchmarks but also better preserves general capabilities across diverse speech and music tasks, confirming its ability to enable more robust cross-modal understanding and achieve a better balance between generalization and task-specific performance in MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17705v1">Distance-to-Distance Ratio: A Similarity Measure for Sentences Based on Rate of Change in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      A measure of similarity between text embeddings can be considered adequate only if it adheres to the human perception of similarity between texts. In this paper, we introduce the distance-to-distance ratio (DDR), a novel measure of similarity between LLM sentence embeddings. Inspired by Lipschitz continuity, DDR measures the rate of change in similarity between the pre-context word embeddings and the similarity between post-context LLM embeddings, thus measuring the semantic influence of context. We evaluate the performance of DDR in experiments designed as a series of perturbations applied to sentences drawn from a sentence dataset. For each sentence, we generate variants by replacing one, two, or three words with either synonyms, which constitute semantically similar text, or randomly chosen words, which constitute semantically dissimilar text. We compare the performance of DDR with other prevailing similarity metrics and demonstrate that DDR consistently provides finer discrimination between semantically similar and dissimilar texts, even under minimal, controlled edits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17692v1">LegalMALR:Multi-Agent Query Understanding and LLM-Based Reranking for Chinese Statute Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 31pages, 4 figures
    </div>
    <details class="paper-abstract">
      Statute retrieval is essential for legal assistance and judicial decision support, yet real-world legal queries are often implicit, multi-issue, and expressed in colloquial or underspecified forms. These characteristics make it difficult for conventional retrieval-augmented generation pipelines to recover the statutory elements required for accurate retrieval. Dense retrievers focus primarily on the literal surface form of the query, whereas lightweight rerankers lack the legal-reasoning capacity needed to assess statutory applicability. We present LegalMALR, a retrieval framework that integrates a Multi-Agent Query Understanding System (MAS) with a zero-shot large-language-model-based reranking module (LLM Reranker). MAS generates diverse, legally grounded reformulations and conducts iterative dense retrieval to broaden candidate coverage. To stabilise the stochastic behaviour of LLM-generated rewrites, we optimise a unified MAS policy using Generalized Reinforcement Policy Optimization(GRPO). The accumulated candidate set is subsequently evaluated by the LLM Reranker, which performs natural-language legal reasoning to produce the final ranking. We further construct CSAID, a dataset of 118 difficult Chinese legal queries annotated with multiple statutory labels, and evaluate LegalMALR on both CSAID and the public STARD benchmark. Experiments show that LegalMALR substantially outperforms strong Retrieval-augmented generation(RAG) baselines in both in-distribution and out-of-distribution settings, demonstrating the effectiveness of combining multi-perspective query interpretation, reinforcement-based policy optimisation, and large-model reranking for statute retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17676v1">GazeSummary: Exploring Gaze as an Implicit Prompt for Personalization in Text-based LLM Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Smart glasses are accelerating progress toward more seamless and personalized LLM-based assistance by integrating multimodal inputs. Yet, these inputs rely on obtrusive explicit prompts. The advent of gaze tracking on smart devices offers a unique opportunity to extract implicit user intent for personalization. This paper investigates whether LLMs can interpret user gaze for text-based tasks. We evaluate different gaze representations for personalization and validate their effectiveness in realistic reading tasks. Results show that LLMs can leverage gaze to generate high-quality personalized summaries and support users in downstream tasks, highlighting the feasibility and value of gaze-driven personalization for future mobile and wearable LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.00003v2">A Review of Incorporating Psychological Theories in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Psychological insights have long shaped pivotal NLP breakthroughs, from attention mechanisms to reinforcement learning and social modeling. As Large Language Models (LLMs) develop, there is a rising consensus that psychology is essential for capturing human-like cognition, behavior, and interaction. This paper reviews how psychological theories can inform and enhance stages of LLM development. Our review integrates insights from six subfields of psychology, including cognitive, developmental, behavioral, social, personality psychology, and psycholinguistics. With stage-wise analysis, we highlight current trends and gaps in how psychological theories are applied. By examining both cross-domain connections and points of tension, we aim to bridge disciplinary divides and promote more thoughtful integration of psychology into NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04537v2">Not All Steps are Informative: On the Linearity of LLMs' RLVR Training</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has become a central component of large language model (LLM) post-training. Unlike supervised fine-tuning (SFT), RLVR lets an LLM generate multiple candidate solutions and reinforces those that lead to a verifiably correct final answer. However, in practice, RLVR often requires thousands of training steps to reach strong performance, incurring substantial computation largely attributed to prolonged exploration. In this work, we make a surprising observation: during RLVR, LLMs evolve in a strongly linear manner. Specifically, both model weights and model output log-probabilities exhibit strong linear correlations with RL training steps. This suggests that RLVR predominantly amplifies trends that emerge early in training, rather than continuously discovering new behaviors throughout the entire optimization trajectory. Motivated by this linearity, we investigate whether future model states can be predicted from intermediate checkpoints via extrapolation, avoiding continued expensive training. We show that Weight Extrapolation produces models with performance comparable to standard RL training while requiring significantly less computation. Moreover, Logits Extrapolation consistently outperforms continued RL training on mathematics and code benchmarks by extrapolating beyond the step range where RL training remains stable. Our code is available at https://github.com/Miaow-Lab/RLVR-Linearity
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17668v1">Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction</a></div>
    <div class="paper-meta">
      📅 2026-01-25
    </div>
    <details class="paper-abstract">
      Efficient key-value (KV) cache management is crucial for the practical deployment of large language models (LLMs), yet existing compression techniques often incur a trade-off between performance degradation and computational overhead. We propose a novel gating-based KV cache eviction method for frozen-weight LLMs that achieves high compression ratios with negligible computational cost. Our approach introduces lightweight sink-attention gating modules to identify and retain critical KV pairs, and integrates seamlessly into both the prefill and decoding stages. The proposed gate training algorithm relies on forward passes of an LLM, avoiding expensive backpropagation, while achieving strong task generalization through a task-agnostic reconstruction objective. Extensive experiments across the Qwen2.5-1M, Qwen3, and Gemma3 families show that our method maintains near-lossless performance while evicting up to 70% of the KV cache. The results are consistent across a wide range of tasks, including long-context understanding, code comprehension, and mathematical reasoning, demonstrating the generality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03369v3">Silenced Biases: The Dark Side LLMs Learned to Refuse</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
    </div>
    <details class="paper-abstract">
      Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18827v1">Automated structural testing of LLM-based agents: methods, framework, and case studies</a></div>
    <div class="paper-meta">
      📅 2026-01-25
      | 💬 10 pages, 5 figures. Preprint of an accepted paper at IEEE BigData 2025 (main track). Source code for the introduced methods and framework available at https://github.com/awslabs/generative-ai-toolkit
    </div>
    <details class="paper-abstract">
      LLM-based agents are rapidly being adopted across diverse domains. Since they interact with users without supervision, they must be tested extensively. Current testing approaches focus on acceptance-level evaluation from the user's perspective. While intuitive, these tests require manual evaluation, are difficult to automate, do not facilitate root cause analysis, and incur expensive test environments. In this paper, we present methods to enable structural testing of LLM-based agents. Our approach utilizes traces (based on OpenTelemetry) to capture agent trajectories, employs mocking to enforce reproducible LLM behavior, and adds assertions to automate test verification. This enables testing agent components and interactions at a deeper technical level within automated workflows. We demonstrate how structural testing enables the adaptation of software engineering best practices to agents, including the test automation pyramid, regression testing, test-driven development, and multi-language testing. In representative case studies, we demonstrate automated execution and faster root-cause analysis. Collectively, these methods reduce testing costs and improve agent quality through higher coverage, reusability, and earlier defect detection. We provide an open source reference implementation on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.05786v3">FedMentalCare: Towards Privacy-Preserving Fine-Tuned LLMs to Analyze Mental Health Status Using Federated Learning Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 9 pages, 3 figures, 3 tables and 2 algorithms
    </div>
    <details class="paper-abstract">
      With the increasing prevalence of mental health conditions worldwide, AI-powered chatbots and conversational agents have emerged as accessible tools to support mental health. However, deploying Large Language Models (LLMs) in mental healthcare applications raises significant privacy concerns, especially regarding regulations like HIPAA and GDPR. In this work, we propose FedMentalCare, a privacy-preserving framework that leverages Federated Learning (FL) combined with Low-Rank Adaptation (LoRA) to fine-tune LLMs for mental health analysis. We investigate the performance impact of varying client data volumes and model architectures (e.g., MobileBERT and MiniLM) in FL environments. Our framework demonstrates a scalable, privacy-aware approach for deploying LLMs in real-world mental healthcare scenarios, addressing data security and computational efficiency challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14275v3">FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 NeurIPS 2025 GenAI4Health Workshop
    </div>
    <details class="paper-abstract">
      Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning (FL) without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per-round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17567v1">Real-Time Trend Prediction via Continually-Aligned LLM Query Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Trending news detection in low-traffic search environments faces a fundamental cold-start problem, where a lack of query volume prevents systems from identifying emerging or long-tail trends. Existing methods relying on keyword frequency or query spikes are inherently slow and ineffective in these sparse settings, lagging behind real-world shifts in attention. We introduce RTTP, a novel Real-Time Trending Prediction framework that generates search queries directly from news content instead of waiting for users to issue them. RTTP leverages a continual learning LLM (CL-LLM) that converts posts into search-style queries and scores them using engagement strength + creator authority, enabling early trend surfacing before search volume forms. To ensure adaptation without degrading reasoning, we propose Mix-Policy DPO, a new preference-based continual learning approach that combines on-policy stability with off-policy novelty to mitigate catastrophic forgetting during model upgrades. Deployed at production scale on Facebook and Meta AI products, RTTP delivers +91.4% improvement in tail-trend detection precision@500 and +19% query generation accuracy over industry baselines, while sustaining stable performance after multi-week online training. This work demonstrates that LLM-generated synthetic search signals, when aligned and continually updated, unlock timely trend understanding in low-traffic search environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17551v1">GreenServ: Energy-Efficient Context-Aware Dynamic Routing for Multi-Model LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Paper under submisison
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities, but their broad deployment is limited by significant computational resource demands, particularly energy consumption during inference. Static, one-model-fits-all inference strategies are often inefficient, as they do not exploit the diverse range of available models or adapt to varying query requirements. This paper presents GreenServ, a dynamic, context-aware routing framework that optimizes the trade-off between inference accuracy and energy efficiency. GreenServ extracts lightweight contextual features from each query, including task type, semantic cluster, and text complexity, and routes queries to the most suitable model from a heterogeneous pool, based on observed accuracy and energy usage. We employ a multi-armed bandit approach to learn adaptive routing policies online. This approach operates under partial feedback, eliminates the need for extensive offline calibration, and streamlines the integration of new models into the inference pipeline. We evaluated GreenServ across five benchmark tasks and a pool of 16 contemporary open-access LLMs. Experimental results show that GreenServ consistently outperforms static (single-model) and random baselines. In particular, compared to random routing, GreenServ achieved a 22% increase in accuracy while reducing cumulative energy consumption by 31%. Finally, we evaluated GreenServ with RouterBench, achieving an average accuracy of 71.7% with a peak accuracy of 75.7%. All artifacts are open-source and available as an anonymous repository for review purposes here: https://anonymous.4open.science/r/llm-inference-router-EBEA/README.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15338v2">HeartLLM: Discretized ECG Tokenization for LLM-Based Diagnostic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Electrocardiography (ECG) plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present HeartLLM, a novel framework that integrates time-series (TS) and language modeling by enabling large language models (LLMs) to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into quantized codes using a lead-wise encoder and quantization module. These quantized codes are then mapped to an extended ECG vocabulary to form ECG tokens, enabling the model to process both ECG and natural language inputs within a unified framework. To bridge the modality gap, we pretrain the model on an autoregressive ECG token forecasting task, allowing the LLM to capture temporal dynamics through its inherent language modeling capability. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, HeartLLM achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating discretized ECG tokens into LLMs for medical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17540v1">Ethical Risk Assessment of the Data Harnessing Process of LLM supported on Consensus of Well-known Multi-Ethical Frameworks</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have revolutionized natural language processing, unlocking unprecedented capabilities in communication, automation, and knowledge generation. However, the ethical implications of LLM development, particularly in data harnessing, remain a critical challenge. Despite widespread discussion about the ethical compliance of LLMs -- especially concerning their data harnessing processes, there remains a notable absence of concrete frameworks to systematically guide or measure the ethical risks involved. In this paper we discuss a potential pathway for building an Ethical Risk Scoring (ERS) system to quantitatively assess the ethical integrity of the data harnessing process for AI systems. This system is based on a set of assessment questions grounded in core ethical principles, which are, in turn, supported by commanding ethical theories. By integrating measurable scoring mechanisms, this approach aims to foster responsible LLM development, balancing technological innovation with ethical accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13742v2">Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) judges exhibit strong reasoning capabilities but are limited to textual content. This leaves current automatic Speech-to-Speech (S2S) evaluation methods reliant on opaque and expensive Audio Language Models (ALMs). In this work, we propose TRACE (Textual Reasoning over Audio Cues for Evaluation), a novel framework that enables LLM judges to reason over audio cues to achieve cost-efficient and human-aligned S2S evaluation. To demonstrate the strength of the framework, we first introduce a Human Chain-of-Thought (HCoT) annotation protocol to improve the diagnostic capability of existing judge benchmarks by separating evaluation into explicit dimensions: content (C), voice quality (VQ), and paralinguistics (P). Using this data, TRACE constructs a textual blueprint of inexpensive audio signals and prompts an LLM to render dimension-wise judgments, fusing them into an overall rating via a deterministic policy. TRACE achieves higher agreement with human raters than ALMs and transcript-only LLM judges while being significantly more cost-effective. We will release the HCoT annotations and the TRACE framework to enable scalable and human-aligned S2S evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17527v1">Bridging Expectation Signals: LLM-Based Experiments and a Behavioral Kalman Filter Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      As LLMs increasingly function as economic agents, the specific mechanisms LLMs use to update their belief with heterogeneous signals remain opaque. We design experiments and develop a Behavioral Kalman Filter framework to quantify how LLM-based agents update expectations, acting as households or firm CEOs, update expectations when presented with individual and aggregate signals. The results from experiments and model estimation reveal four consistent patterns: (1) agents' weighting of priors and signals deviates from unity; (2) both household and firm CEO agents place substantially larger weights on individual signals compared to aggregate signals; (3) we identify a significant and negative interaction between concurrent signals, implying that the presence of multiple information sources diminishes the marginal weight assigned to each individual signal; and (4) expectation formation patterns differ significantly between household and firm CEO agents. Finally, we demonstrate that LoRA fine-tuning mitigates, but does not fully eliminate, behavioral biases in LLM expectation formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21981v2">Collaborative Belief Reasoning with LLMs for Efficient Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Effective real-world multi-agent collaboration requires not only accurate planning but also the ability to reason about collaborators' intents--a crucial capability for avoiding miscoordination and redundant communication under partial observable environments. Due to their strong planning and reasoning capabilities, large language models (LLMs) have emerged as promising autonomous agents for collaborative task solving. However, existing collaboration frameworks for LLMs overlook their reasoning potential for dynamic intent inference, and thus produce inconsistent plans and redundant communication, reducing collaboration efficiency. To bridge this gap, we propose CoBel-World, a novel framework that equips LLM agents with a Collaborative Belief World--an internal representation jointly modeling the physical environment and collaborators' mental states. CoBel-World enables agents to parse external open-world knowledge into structured beliefs via a symbolic belief representation module, and perform zero-shot Bayesian-style belief updates through LLM reasoning. This allows agents to proactively detect potential miscoordination (e.g., conflicting plans) and communicate adaptively. Evaluated on challenging embodied benchmarks (i.e., TDW-MAT and C-WAH), CoBel-World significantly reduces communication costs by 64-79% and improves task completion efficiency by 4-28% compared to the strongest baseline. Our results show that explicit, intent-aware belief modeling is essential for efficient and human-like collaboration in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17471v1">PatchIsland: Orchestration of LLM Agents for Continuous Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Continuous fuzzing platforms such as OSS-Fuzz uncover large numbers of vulnerabilities, yet the subsequent repair process remains largely manual. Unfortunately, existing Automated Vulnerability Repair (AVR) techniques -- including recent LLM-based systems -- are not directly applicable to continuous fuzzing. This is because these systems are designed and evaluated on a static, single-run benchmark setting, making them ill-suited for the diverse, noisy, and failure-prone environments in continuous fuzzing. To address these issues, we introduce PatchIsland, a system for Continuous Vulnerability Repair (CVR) that tightly integrates with continuous fuzzing pipelines. PatchIsland employs an ensemble of diverse LLM agents. By leveraging multiple LLM agents, PatchIsland can cover a wider range of settings (e.g., different projects, bug types, and programming languages) and also improve operational robustness. In addition, PatchIsland utilizes a two-phase patch-based deduplication to mitigate duplicate crashes and patches, which can be problematic in continuous fuzzing. In our internal evaluation, PatchIsland repaired 84 of 92 vulnerabilities, demonstrating strong repair capability. In the official AIxCC competition, the system operated with no human intervention in a fully autonomous environment and successfully patched 31 out of 43 vulnerabilities, achieving a repair rate of 72.1\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18951v4">SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 29 pages, 10 figures, NeurIPS 2025 Main
    </div>
    <details class="paper-abstract">
      Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: https://bird-critic.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04765v3">Differential syntactic and semantic encoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      We study how syntactic and semantic information is encoded in inner layer representations of Large Language Models (LLMs), focusing on the very large DeepSeek-V3. We find that, by averaging hidden-representation vectors of sentences sharing syntactic structure or meaning, we obtain vectors that capture a significant proportion of the syntactic and semantic information contained in the representations. In particular, subtracting these syntactic and semantic ``centroids'' from sentence vectors strongly affects their similarity with syntactically and semantically matched sentences, respectively, suggesting that syntax and semantics are, at least partially, linearly encoded. We also find that the cross-layer encoding profiles of syntax and semantics are different, and that the two signals can to some extent be decoupled, suggesting differential encoding of these two types of linguistic information in LLM representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11056v3">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 TheWebConf 2026 Industry
    </div>
    <details class="paper-abstract">
      Query-service relevance prediction in e-commerce search systems faces strict latency requirements that prevent the direct application of Large Language Models (LLMs). To bridge this gap, we propose a two-stage reasoning distillation framework to transfer reasoning capabilities from a powerful teacher LLM to a lightweight, deployment-friendly student model. In the first stage, we address the limitations of general-purpose LLMs by constructing a domain-adapted teacher model. This is achieved through a three-step process: domain-adaptive pre-training to inject platform knowledge, supervised fine-tuning to elicit reasoning skills, and preference optimization with a multi-dimensional reward model to ensure the generation of reliable and preference-aligned reasoning paths. This teacher can then automatically annotate massive query-service pairs from search logs with both relevance labels and reasoning chains. In the second stage, to address the challenges of architectural heterogeneity in standard distillation, we introduce Contrastive Reasoning Self-Distillation (CRSD). By modeling the behavior of the same student model under ``standard'' and ``reasoning-augmented'' inputs as a teacher-student relationship, CRSD enables the lightweight model to internalize the teacher's complex decision-making mechanisms without needing the explicit reasoning path at inference. Offline evaluations and online A/B testing in the Meituan search advertising system demonstrate that our framework achieves significant improvements across multiple metrics, validating its effectiveness and practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17421v1">Oops, Wait: Token-Level Signals as a Lens into LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The emergence of discourse-like tokens such as "wait" and "therefore" in large language models (LLMs) has offered a unique window into their reasoning processes. However, systematic analyses of how such signals vary across training strategies and model scales remain lacking. In this paper, we analyze token-level signals through token probabilities across various models. We find that specific tokens strongly correlate with reasoning correctness, varying with training strategies while remaining stable across model scales. A closer look at the "wait" token in relation to answer probability demonstrates that models fine-tuned on small-scale datasets acquire reasoning ability through such signals but exploit them only partially. This work provides a systematic lens to observe and understand the dynamics of LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17418v1">GraphPilot: GUI Task Automation with One-Step LLM Reasoning Powered by Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 This paper is accepted by the Journal of Intelligent Computing and Networking (JICN) for publication
    </div>
    <details class="paper-abstract">
      Mobile graphical user interface (GUI) agents are designed to automate everyday tasks on smartphones. Recent advances in large language models (LLMs) have significantly enhanced the capabilities of mobile GUI agents. However, most LLM-powered mobile GUI agents operate in stepwise query-act loops, which incur high latency due to repeated LLM queries. We present GraphPilot, a mobile GUI agent that leverages knowledge graphs of the target apps to complete user tasks in almost one LLM query. GraphPilot operates in two complementary phases to enable efficient and reliable LLM-powered GUI task automation. In the offline phase, it explores target apps, records and analyzes interaction history, and constructs an app-specific knowledge graph that encodes functions of pages and elements as well as transition rules for each app. In the online phase, given an app and a user task, it leverages the knowledge graph of the given app to guide the reasoning process of LLM. When the reasoning process encounters uncertainty, GraphPilot dynamically requests the HTML representation of the current interface to refine subsequent reasoning. Finally, a validator checks the generated sequence of actions against the transition rules in the knowledge graph, performing iterative corrections to ensure it is valid. The structured, informative information in the knowledge graph allows the LLM to plan the complete sequence of actions required to complete the user task. On the DroidTask benchmark, GraphPilot improves task completion rate over Mind2Web and AutoDroid, while substantially reducing latency and the number of LLM queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17399v1">ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved rapid progress in Chinese language understanding, yet accurately evaluating their capabilities remains challenged by benchmark saturation and prohibitive computational costs. While static leaderboards provide snapshot rankings, they often mask the structural trade-offs between capabilities. In this work, we present ReLE (Robust Efficient Live Evaluation), a scalable system designed to diagnose Capability Anisotropy, the non-uniformity of model performance across domains. Using ReLE, we evaluate 304 models (189 commercial, 115 open-source) across a Domain $\times$ Capability orthogonal matrix comprising 207,843 samples. We introduce two methodological contributions to address current evaluation pitfalls: (1) A Symbolic-Grounded Hybrid Scoring Mechanism that eliminates embedding-based false positives in reasoning tasks; (2) A Dynamic Variance-Aware Scheduler based on Neyman allocation with noise correction, which reduces compute costs by 70\% compared to full-pass evaluations while maintaining a ranking correlation of $ρ=0.96$. Our analysis reveals that aggregate rankings are highly sensitive to weighting schemes: models exhibit a Rank Stability Amplitude (RSA) of 11.4 in ReLE versus $\sim$5.0 in traditional benchmarks, confirming that modern models are highly specialized rather than generally superior. We position ReLE not as a replacement for comprehensive static benchmarks, but as a high-frequency diagnostic monitor for the evolving model landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17397v1">CLM-Bench: Benchmarking and Analyzing Cross-lingual Misalignment of LLMs in Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 EACL MME workshop paper
    </div>
    <details class="paper-abstract">
      Knowledge Editing (KE) has emerged as a promising paradigm for updating facts in Large Language Models (LLMs) without retraining. However, progress in Multilingual Knowledge Editing (MKE) is currently hindered by biased evaluation frameworks. We observe that existing MKE benchmarks are typically constructed by mechanically translating English-centric datasets into target languages (e.g., English-to-Chinese). This approach introduces translation artifacts and neglects culturally specific entities native to the target language, failing to reflect the true knowledge distribution of LLMs. To address this, we propose CLM-Bench, a culture-aware benchmark constructed using a native Chinese-first methodology. We curate 1,010 high-quality CounterFact pairs rooted in Chinese cultural contexts and align them with English counterparts. Using CLM-Bench, we conduct extensive experiments on representative LLMs (e.g., Llama-3, Qwen2) and reveal a significant Cross-lingual Misalignment: edits in one language function independently and fail to propagate to the other. We further provide a geometric explanation via layer-wise representation analysis, demonstrating that edit vectors for Chinese and English are nearly orthogonal -- residing in disjoint subspaces -- while mixed-lingual editing exhibits linear additivity of these vectors. Our findings challenge the effectiveness of current methods in cross-lingual transfer and underscore the importance of culturally native benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22767v2">TELL-TALE: Task Efficient LLMs with Task Aware Layer Elimination</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically deployed using a fixed architecture, despite growing evidence that not all layers contribute equally to every downstream task. In this work, we introduce TALE (Task-Aware Layer Elimination), an inference-time method that improves task performance by selectively removing layers that are irrelevant or detrimental for a given task. TALE optimizes task-specific validation performance, yielding a task-adapted architecture without retraining or modifying model weights. Across 9 tasks and 5 model families, under both zero-shot and few-shot settings, we show that TALE consistently matches or surpasses baseline performance while simultaneously reducing computational cost, outperforming general and layer-wise pruning approaches such as SLEB. Beyond inference-time gains, TALE synergizes with fine-tuning and few-shot learning, where task-adapted architectures lead to additional performance improvements. Computing TALE for a new task requires modest resources (1-2 GPU hours on an A100), making it a practical and deployable solution for task-specialized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15830v2">DAIQ: Auditing Demographic Attribute Inference from Question in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent evaluations of Large language models (LLMs) audit social bias primarily through prompts that explicitly reference demographic attributes, overlooking whether models infer sensitive demographics from neutral questions. Such inference constitutes epistemic overreach and raises concerns for privacy. We introduce Demographic Attribute Inference from Questions (DAIQ), a diagnostic audit framework for evaluating demographic inference under epistemic uncertainty. We evaluate 18 open- and closed-source LLMs across six real-world domains and five demographic attributes. We find that many models infer demographics from neutral questions, defaulting to socially dominant categories and producing stereotype-aligned rationales. These behaviors persist across model families, scales and decoding settings, indicating reliance on learned population priors. We further show that inferred demographics can condition downstream responses and that abstention oriented prompting substantially reduces unintended inference without model fine-tuning. Our results suggest that current bias evaluations are incomplete and motivate evaluation standards that assess not only how models respond to demographic information, but whether they should infer it at all.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23277v2">Sentinel: Decoding Context Utilization via Attention Probing for Efficient LLM Context Compression</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) often suffers from long and noisy retrieved contexts. Prior context compression methods rely on predefined importance metrics or supervised compression models, rather than on the model's own inference-time behavior. We propose Sentinel, a lightweight sentence-level compression framework that treats context compression as an understanding decoding problem. Sentinel probes native attention behaviors of a frozen LLM with a lightweight readout to decode which parts of the context are actually utilized when answering a query, rather than using attention as a direct relevance score. We empirically observe that decoded relevance signals exhibit sufficient consistency across model scales to support effective compression with compact proxy models. On LongBench, Sentinel with a 0.5B proxy model achieves up to 5x compression while matching the QA performance of 7B-scale baselines, and despite being trained only on English QA data, generalizes effectively to Chinese and out-of-domain settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16559v4">BuildArena: A Physics-Aligned Interactive Benchmark of LLMs for Engineering Construction</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 33 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Engineering construction automation aims to transform natural language specifications into physically viable structures, requiring complex integrated reasoning under strict physical constraints. While modern LLMs possess broad knowledge and strong reasoning capabilities that make them promising candidates for this domain, their construction competencies remain largely unevaluated. To address this gap, we introduce BuildArena, the first physics-aligned interactive benchmark designed for language-driven engineering construction. It contributes to the community in four aspects: (1) a highly customizable benchmarking framework for in-depth comparison and analysis of LLMs; (2) an extendable task design strategy spanning static and dynamic mechanics across multiple difficulty tiers; (3) a 3D Spatial Geometric Computation Library for supporting construction based on language instructions; (4) a baseline LLM agentic workflow that effectively evaluates diverse model capabilities. On eight frontier LLMs, BuildArena comprehensively evaluates their capabilities for language-driven and physics-grounded construction automation. The project page is at https://build-arena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17346v1">Multi-Agent Learning Path Planning via LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into intelligent tutoring systems offers transformative potential for personalized learning in higher education. However, most existing learning path planning approaches lack transparency, adaptability, and learner-centered explainability. To address these challenges, this study proposes a novel Multi-Agent Learning Path Planning (MALPP) framework that leverages a role- and rule-based collaboration mechanism among intelligent agents, each powered by LLMs. The framework includes three task-specific agents: a learner analytics agent, a path planning agent, and a reflection agent. These agents collaborate via structured prompts and predefined rules to analyze learning profiles, generate tailored learning paths, and iteratively refine them with interpretable feedback. Grounded in Cognitive Load Theory and Zone of Proximal Development, the system ensures that recommended paths are cognitively aligned and pedagogically meaningful. Experiments conducted on the MOOCCubeX dataset using seven LLMs show that MALPP significantly outperforms baseline models in path quality, knowledge sequence consistency, and cognitive load alignment. Ablation studies further validate the effectiveness of the collaborative mechanism and theoretical constraints. This research contributes to the development of trustworthy, explainable AI in education and demonstrates a scalable approach to learner-centered adaptive instruction powered by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17343v1">Are We Evaluating the Edit Locality of LLM Model Editing Properly?</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Model editing has recently emerged as a popular paradigm for efficiently updating knowledge in LLMs. A central desideratum of updating knowledge is to balance editing efficacy, i.e., the successful injection of target knowledge, and specificity (also known as edit locality), i.e., the preservation of existing non-target knowledge. However, we find that existing specificity evaluation protocols are inadequate for this purpose. We systematically elaborated on the three fundamental issues it faces. Beyond the conceptual issues, we further empirically demonstrate that existing specificity metrics are weakly correlated with the strength of specificity regularizers. We also find that current metrics lack sufficient sensitivity, rendering them ineffective at distinguishing the specificity performance of different methods. Finally, we propose a constructive evaluation protocol. Under this protocol, the conflict between open-ended LLMs and the assumption of determined answers is eliminated, query-independent fluency biases are avoided, and the evaluation strictness can be smoothly adjusted within a near-continuous space. Experiments across various LLMs, datasets, and editing methods show that metrics derived from the proposed protocol are more sensitive to changes in the strength of specificity regularizers and exhibit strong correlation with them, enabling more fine-grained discrimination of different methods' knowledge preservation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v5">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Locating files and functions requiring modification in large software repositories is challenging due to their scale and structural complexity. Existing LLM-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which often overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool: jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a base pretrained model, without relying on closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and the 32B model exceeding closed-source models such as GPT-5 on most metrics. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05758v2">EMORL-TTS: Reinforcement Learning for Fine-Grained Emotion Control in LLM-based TTS</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Recent LLM-based TTS systems achieve strong quality and zero-shot ability, but lack fine-grained emotional control due to their reliance on discrete speech tokens. Existing approaches either limit emotions to categorical labels or cannot generalize to LLM-based architectures. We propose EMORL-TTS (Fine-grained Emotion-controllable TTS with Reinforcement Learning), a framework that unifies global intensity control in the VAD space with local emphasis regulation. Our method combines supervised fine-tuning with reinforcement learning guided by task-specific rewards for emotion category, intensity, and emphasis. Moreover, we further investigate how emphasis placement modulates fine-grained emotion intensity. Experiments show that EMORL-TTS improves emotion accuracy, intensity differentiation, and emphasis clarity, while preserving synthesis quality comparable to strong LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17570v2">GreedySnake: Accelerating SSD-Offloaded LLM Training with Efficient Scheduling and Optimizer Step Overlapping</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      SSD-offloaded training offers a practical and promising approach to making LLM training cost-effective. Building on gradient accumulation with micro-batches, this paper introduces GreedySnake, a new SSD-offloaded training system that employs vertical scheduling, which executes all microbatches of a layer before proceeding to the next. Compared to existing systems that use horizontal scheduling (i.e., executing micro-batches sequentially), GreedySnake achieves higher training throughput with smaller batch sizes, bringing the system much closer to the ideal scenario predicted by the roofline model. To further mitigate the I/O bottleneck, GreedySnake overlaps part of the optimization step with the forward pass of the next iteration. Experimental results on A100 GPUs show that GreedySnake achieves saturated training throughput improvements over ZeRO-Infinity: 1.96x on 1 GPU and 1.93x on 4 GPUs for GPT-65B, and 2.53x on 1 GPU for GPT-175B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20697v2">Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Project Hompage: https://tokenbuncher.github.io/
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate more advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response entropy. By constraining entropy, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task performance and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14005v3">PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 The code is available at https://github.com/weizou52/PIShield
    </div>
    <details class="paper-abstract">
      LLM-integrated applications are vulnerable to prompt injection attacks, where an attacker contaminates the input to inject malicious instructions, causing the LLM to follow the attacker's intent instead of the original user's. Existing prompt injection detection methods often have sub-optimal performance and/or high computational overhead. In this work, we propose PIShield, an effective and efficient detection method based on the observation that instruction-tuned LLMs internally encode distinguishable signals for prompts containing injected instructions. PIShield leverages residual-stream representations and a simple linear classifier to detect prompt injection, without expensive model fine-tuning or response generation. We conduct extensive evaluations on a diverse set of short- and long-context benchmarks. The results show that PIShield consistently achieves low false positive and false negative rates, significantly outperforming existing baselines. These findings demonstrate that internal representations of instruction-tuned LLMs provide a powerful and practical foundation for prompt injection detection in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17292v1">Risk-based test framework for LLM features in regulated software</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Large language models are increasingly embedded in regulated and safety-critical software, including clinical research platforms and healthcare information systems. While these features enable natural language search, summarization, and configuration assistance, they introduce risks such as hallucinations, harmful or out-of-scope advice, privacy and security issues, bias, instability under change, and adversarial misuse. Prior work on machine learning testing and AI assurance offers useful concepts but limited guidance for interactive, product-embedded assistants. This paper proposes a risk-based testing framework for LLM features in regulated software: a six-category risk taxonomy, a layered test strategy mapping risks to concrete tests across guardrail, orchestration, and system layers, and a case study applying the approach to a Knowledgebase assistant in a clinical research platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17284v1">Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models in Medical Question Answering is severely hampered by ambiguous user queries, a significant safety risk that demonstrably reduces answer accuracy in high-stakes healthcare settings. In this paper, we formalize this challenge by linking input ambiguity to aleatoric uncertainty (AU), which is the irreducible uncertainty arising from underspecified input. To facilitate research in this direction, we construct CV-MedBench, the first benchmark designed for studying input ambiguity in Medical QA. Using this benchmark, we analyze AU from a representation engineering perspective, revealing that AU is linearly encoded in LLM's internal activation patterns. Leveraging this insight, we introduce a novel AU-guided "Clarify-Before-Answer" framework, which incorporates AU-Probe - a lightweight module that detects input ambiguity directly from hidden states. Unlike existing uncertainty estimation methods, AU-Probe requires neither LLM fine-tuning nor multiple forward passes, enabling an efficient mechanism to proactively request user clarification and significantly enhance safety. Extensive experiments across four open LLMs demonstrate the effectiveness of our QA framework, with an average accuracy improvement of 9.48% over baselines. Our framework provides an efficient and robust solution for safe Medical QA, strengthening the reliability of health-related applications. The code is available at https://github.com/yaokunliu/AU-Med.git, and the CV-MedBench dataset is released on Hugging Face at https://huggingface.co/datasets/yaokunl/CV-MedBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17275v1">Latent-Space Contrastive Reinforcement Learning for Stable and Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 12 pages,
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate exceptional performance in surface-level text generation, their nature in handling complex multi-step reasoning tasks often remains one of ``statistical fitting'' rather than systematic logical deduction. Traditional Reinforcement Learning (RL) attempts to mitigate this by introducing a ``think-before-speak'' paradigm. However, applying RL directly in high-dimensional, discrete token spaces faces three inherent challenges: sample-inefficient rollouts, high gradient estimation variance, and the risk of catastrophic forgetting. To fundamentally address these structural bottlenecks, we propose \textbf{DeepLatent Reasoning (DLR)}, a latent-space bidirectional contrastive reinforcement learning framework. This framework shifts the trial-and-error cost from expensive token-level full sequence generation to the continuous latent manifold. Specifically, we introduce a lightweight assistant model to efficiently sample $K$ reasoning chain encodings within the latent space. These encodings are filtered via a dual reward mechanism based on correctness and formatting; only high-value latent trajectories are fed into a \textbf{frozen main model} for single-pass decoding. To maximize reasoning diversity while maintaining coherence, we design a contrastive learning objective to enable directed exploration within the latent space. Since the main model parameters remain frozen during optimization, this method mathematically eliminates catastrophic forgetting. Experiments demonstrate that under comparable GPU computational budgets, DLR achieves more stable training convergence, supports longer-horizon reasoning chains, and facilitates the sustainable accumulation of reasoning capabilities, providing a viable path toward reliable and scalable reinforcement learning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17261v1">AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 21 pages in total, including 9 pages of main text, with 4 figures and 3 tables. This manuscript is submitted to arXiv
    </div>
    <details class="paper-abstract">
      Zeroth-Order (ZO) optimization has emerged as a promising solution for fine-tuning LLMs under strict memory constraints, as it avoids the prohibitive memory cost of storing activations for backpropagation. However, existing ZO methods typically employ isotropic perturbations, neglecting the rich structural information available during the forward pass. In this paper, we identify a crucial link between gradient formation and activation structure: the gradient of a linear layer is confined to the subspace spanned by its input activations. Leveraging this insight, we propose Activation-Guided Zeroth-Order optimization (AGZO). Unlike prior methods, AGZO extracts a compact, activation-informed subspace on the fly during the forward pass and restricts perturbations to this low-rank subspace. We provide a theoretical framework showing that AGZO optimizes a subspace-smoothed objective and provably yields update directions with higher cosine similarity to the true gradient than isotropic baselines. Empirically, we evaluate AGZO on Qwen3 and Pangu models across various benchmarks. AGZO consistently outperforms state-of-the-art ZO baselines and significantly narrows the performance gap with first-order fine-tuning, while maintaining almost the same peak memory footprint as other ZO methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02547v4">The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Published on Transactions on Machine Learning Research: https://openreview.net/forum?id=RY19y2RI1O
    </div>
    <details class="paper-abstract">
      The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17614v1">AlignUI: A Method for Designing LLM-Generated UIs Aligned with User Preferences</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Designing user interfaces that align with user preferences is a time-consuming process, which requires iterative cycles of prototyping, user testing, and refinement. Recent advancements in LLM-based UI generation have enabled efficient UI generation to assist the UI design process. We introduce AlignUI, a method that aligns LLM-generated UIs with user tasks and preferences by using a user preference dataset to guide the LLM's reasoning process. The dataset was crowdsourced from 50 general users (the target users of generated UIs) and contained 720 UI control preferences on eight image-editing tasks. We evaluated AlignUI by generating UIs for six unseen tasks and conducting a user study with 72 additional general users. The results showed that the generated UIs closely align with multiple dimensions of user preferences. We conclude by discussing the applicability of our method to support user-aligned UI design for multiple task domains and user groups, as well as personalized user needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17604v1">Human-Aligned Enhancement of Programming Answers with LLMs Guided by User Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-24
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used to support software developers in tasks such as code generation, optimization, and documentation. However, their ability to improve existing programming answers in a human-like manner remains underexplored. On technical question-and-answer platforms such as Stack Overflow (SO), contributors often revise answers based on user comments that identify errors, inefficiencies, or missing explanations. Yet roughly one-third of this feedback is never addressed due to limited time, expertise, or visibility, leaving many answers incomplete or outdated. This study investigates whether LLMs can enhance programming answers by interpreting and incorporating comment-based feedback. We make four main contributions. First, we introduce ReSOlve, a benchmark consisting of 790 SO answers with associated comment threads, annotated for improvement-related and general feedback. Second, we evaluate four state-of-the-art LLMs on their ability to identify actionable concerns, finding that DeepSeek achieves the best balance between precision and recall. Third, we present AUTOCOMBAT, an LLM-powered tool that improves programming answers by jointly leveraging user comments and question context. Compared to human revised references, AUTOCOMBAT produces near-human quality improvements while preserving the original intent and significantly outperforming the baseline. Finally, a user study with 58 practitioners shows strong practical value, with 84.5 percent indicating they would adopt or recommend the tool. Overall, AUTOCOMBAT demonstrates the potential of scalable, feedback-driven answer refinement to improve the reliability and trustworthiness of technical knowledge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21184v4">Jailbreak-as-a-Service++: Unveiling Distributed AI-Driven Malicious Information Campaigns Powered by LLM Crowdsourcing</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      To prevent the misuse of Large Language Models (LLMs) for malicious purposes, numerous efforts have been made to develop the safety alignment mechanisms of LLMs. However, as multiple LLMs become readily accessible through various Model-as-a-Service (MaaS) platforms, attackers can strategically exploit LLMs' heterogeneous safety policies to fulfill malicious information generation tasks in a distributed manner. In this study, we introduce \textit{\textbf{PoisonSwarm}} to how attackers can reliably launder malicious tasks via the speculative use of LLM crowdsourcing. Building upon a scheduler orchestrating crowdsourced LLMs, PoisonSwarm maps the given malicious task to a benign analogue to derive a content template, decomposes it into semantic units for crowdsourced unit-wise rewriting, and reassembles the outputs into malicious content. Experiments show its superiority over existing methods in data quality, diversity, and success rates. Regulation simulations further reveal the difficulty of governing such distributed, orchestrated misuse in MaaS ecosystems, highlighting the need for coordinated, ecosystem-level defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15955v2">How Good Are LLMs at Processing Tool Outputs?</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Most realistic task automation problems require large language models (LLMs) to call tools, which often return complex JSON responses. These responses must be further processed to derive the information necessary for task completion. The ability of LLMs to do so is under-studied. In this paper, we study the tool response processing task and LLMs' abilities to process structured (JSON) responses. We created a dataset for this task, and evaluated 15 open and closed weight models using multiple prompting approaches. Our results show that JSON processing remains a difficult task even for frontier models across multiple prompting strategies. The optimal response processing strategy depends on both the nature and size of the tool outputs, as well as the complexity of the required reasoning. Variations in processing approaches can lead to performance differences ranging from 3\% to 50\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17593v1">From Chains to DAGs: Probing the Graph Structure of Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-24
    </div>
    <details class="paper-abstract">
      Recent progress in large language models has renewed interest in mechanistically characterizing how multi-step reasoning is represented and computed. While much prior work treats reasoning as a linear chain of steps, many reasoning problems are more naturally structured as directed acyclic graphs (DAGs), where intermediate conclusions may depend on multiple premises, branch into parallel sub-derivations, and later merge or be reused. Understanding whether such graph-structured reasoning is reflected in model internals remains an open question. In this work, we introduce Reasoning DAG Probing, a framework that directly asks whether LLM hidden states encode the geometry of a reasoning DAG in a linearly accessible form, and where this structure emerges across layers. Within this framework, we associate each reasoning node with a textual realization and train lightweight probes to predict two graph-theoretic properties from hidden states: node depth and pairwise node distance. We use these probes to analyze the layerwise emergence of DAG structure and evaluate controls that disrupt reasoning-relevant structure while preserving superficial textual properties. Our results provide evidence that reasoning DAG geometry is meaningfully encoded in intermediate layers, with recoverability varying systematically by node depth and model scale, suggesting that LLM reasoning is not only sequential but exhibits measurable internal graph structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16540v1">Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Audio Large Language Models (Audio LLMs) have demonstrated strong capabilities in integrating speech perception with language understanding. However, whether their internal representations align with human neural dynamics during naturalistic listening remains largely unexplored. In this work, we systematically examine layer-wise representational alignment between 12 open-source Audio LLMs and Electroencephalogram (EEG) signals across 2 datasets. Specifically, we employ 8 similarity metrics, such as Spearman-based Representational Similarity Analysis (RSA), to characterize within-sentence representational geometry. Our analysis reveals 3 key findings: (1) we observe a rank-dependence split, in which model rankings vary substantially across different similarity metrics; (2) we identify spatio-temporal alignment patterns characterized by depth-dependent alignment peaks and a pronounced increase in RSA within the 250-500 ms time window, consistent with N400-related neural dynamics; (3) we find an affective dissociation whereby negative prosody, identified using a proposed Tri-modal Neighborhood Consistency (TNC) criterion, reduces geometric similarity while enhancing covariance-based dependence. These findings provide new neurobiological insights into the representational mechanisms of Audio LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16527v1">Beyond Superficial Unlearning: Sharpness-Aware Robust Erasure of Hallucinations in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Multimodal LLMs are powerful but prone to object hallucinations, which describe non-existent entities and harm reliability. While recent unlearning methods attempt to mitigate this, we identify a critical flaw: structural fragility. We empirically demonstrate that standard erasure achieves only superficial suppression, trapping the model in sharp minima where hallucinations catastrophically resurge after lightweight relearning. To ensure geometric stability, we propose SARE, which casts unlearning as a targeted min-max optimization problem and uses a Targeted-SAM mechanism to explicitly flatten the loss landscape around hallucinated concepts. By suppressing hallucinations under simulated worst-case parameter perturbations, our framework ensures robust removal stable against weight shifts. Extensive experiments demonstrate that SARE significantly outperforms baselines in erasure efficacy while preserving general generation quality. Crucially, it maintains persistent hallucination suppression against relearning and parameter updates, validating the effectiveness of geometric stabilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02979v2">Systematizing LLM Persona Design: A Four-Quadrant Technical Taxonomy for AI Companion Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 Accepted to Neurips 2025 workshop: LLM Persona Workshop
    </div>
    <details class="paper-abstract">
      The design and application of LLM-based personas in AI companionship is a rapidly expanding but fragmented field, spanning from virtual emotional companions and game NPCs to embodied functional robots. This diversity in objectives, modality, and technical stacks creates an urgent need for a unified framework. To address this gap, this paper systematizes the field by proposing a Four-Quadrant Technical Taxonomy for AI companion applications. The framework is structured along two critical axes: Virtual vs. Embodied and Emotional Companionship vs. Functional Augmentation. Quadrant I (Virtual Companionship) explores virtual idols, romantic companions, and story characters, introducing a four-layer technical framework to analyze their challenges in maintaining long-term emotional consistency. Quadrant II (Functional Virtual Assistants) analyzes AI applications in work, gaming, and mental health, highlighting the shift from "feeling" to "thinking and acting" and pinpointing key technologies like enterprise RAG and on-device inference. Quadrants III & IV (Embodied Intelligence) shift from the virtual to the physical world, analyzing home robots and vertical-domain assistants, revealing core challenges in symbol grounding, data privacy, and ethical liability. This taxonomy provides not only a systematic map for researchers and developers to navigate the complex persona design space but also a basis for policymakers to identify and address the unique risks inherent in different application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16512v1">SearchLLM: Detecting LLM Paraphrased Text by Measuring the Similarity with Regeneration of the Candidate Source via Search Engine</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 EACL 2026 camera ready (Main Track)
    </div>
    <details class="paper-abstract">
      With the advent of large language models (LLMs), it has become common practice for users to draft text and utilize LLMs to enhance its quality through paraphrasing. However, this process can sometimes result in the loss or distortion of the original intended meaning. Due to the human-like quality of LLM-generated text, traditional detection methods often fail, particularly when text is paraphrased to closely mimic original content. In response to these challenges, we propose a novel approach named SearchLLM, designed to identify LLM-paraphrased text by leveraging search engine capabilities to locate potential original text sources. By analyzing similarities between the input and regenerated versions of candidate sources, SearchLLM effectively distinguishes LLM-paraphrased content. SearchLLM is designed as a proxy layer, allowing seamless integration with existing detectors to enhance their performance. Experimental results across various LLMs demonstrate that SearchLLM consistently enhances the accuracy of recent detectors in detecting LLM-paraphrased text that closely mimics original content. Furthermore, SearchLLM also helps the detectors prevent paraphrasing attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16508v1">Is Length Really A Liability? An Evaluation of Multi-turn LLM Conversations using BoolQ</a></div>
    <div class="paper-meta">
      📅 2026-01-23
      | 💬 4 pages plus 6 pages of bibliography and appendix
    </div>
    <details class="paper-abstract">
      Single-prompt evaluations dominate current LLM benchmarking, yet they fail to capture the conversational dynamics where real-world harm occurs. In this study, we examined whether conversation length affects response veracity by evaluating LLM performance on the BoolQ dataset under varying length and scaffolding conditions. Our results across three distinct LLMs revealed model-specific vulnerabilities that are invisible under single-turn testing. The length-dependent and scaffold-specific effects we observed demonstrate a fundamental limitation of static evaluations, as deployment-relevant vulnerabilities could only be spotted in a multi-turn conversational setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04740v2">StealthGraph: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16492v1">LLM-based Semantic Search for Conversational Queries in E-commerce</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Conversational user queries are increasingly challenging traditional e-commerce platforms, whose search systems are typically optimized for keyword-based queries. We present an LLM-based semantic search framework that effectively captures user intent from conversational queries by combining domain-specific embeddings with structured filters. To address the challenge of limited labeled data, we generate synthetic data using LLMs to guide the fine-tuning of two models: an embedding model that positions semantically similar products close together in the representation space, and a generative model for converting natural language queries into structured constraints. By combining similarity-based retrieval with constraint-based filtering, our framework achieves strong precision and recall across various settings compared to baseline approaches on a real-world dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16479v1">Doc2AHP: Inferring Structured Multi-Criteria Decision Models via Semantic Trees with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate remarkable proficiency in semantic understanding, they often struggle to ensure structural consistency and reasoning reliability in complex decision-making tasks that demand rigorous logic. Although classical decision theories, such as the Analytic Hierarchy Process (AHP), offer systematic rational frameworks, their construction relies heavily on labor-intensive domain expertise, creating an "expert bottleneck" that hinders scalability in general scenarios. To bridge the gap between the generalization capabilities of LLMs and the rigor of decision theory, we propose Doc2AHP, a novel structured inference framework guided by AHP principles. Eliminating the need for extensive annotated data or manual intervention, our approach leverages the structural principles of AHP as constraints to direct the LLM in a constrained search within the unstructured document space, thereby enforcing the logical entailment between parent and child nodes. Furthermore, we introduce a multi-agent weighting mechanism coupled with an adaptive consistency optimization strategy to ensure the numerical consistency of weight allocation. Empirical results demonstrate that Doc2AHP not only empowers non-expert users to construct high-quality decision models from scratch but also significantly outperforms direct generative baselines in both logical completeness and downstream task accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23019v3">LLM Watermark Evasion via Bias Inversion</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16458v1">Bridging Expert Reasoning and LLM Detection: A Knowledge-Driven Framework for Malicious Packages</a></div>
    <div class="paper-meta">
      📅 2026-01-23
    </div>
    <details class="paper-abstract">
      Open-source ecosystems such as NPM and PyPI are increasingly targeted by supply chain attacks, yet existing detection methods either depend on fragile handcrafted rules or data-driven features that fail to capture evolving attack semantics. We present IntelGuard, a retrieval-augmented generation (RAG) based framework that integrates expert analytical reasoning into automated malicious package detection. IntelGuard constructs a structured knowledge base from over 8,000 threat intelligence reports, linking malicious code snippets with behavioral descriptions and expert reasoning. When analyzing new packages, it retrieves semantically similar malicious examples and applies LLM-guided reasoning to assess whether code behaviors align with intended functionality. Experiments on 4,027 real-world packages show that IntelGuard achieves 99% accuracy and a 0.50% false positive rate, while maintaining 96.5% accuracy on obfuscated code. Deployed on PyPI.org, it discovered 54 previously unreported malicious packages, demonstrating interpretable and robust detection guided by expert knowledge.
    </details>
</div>
