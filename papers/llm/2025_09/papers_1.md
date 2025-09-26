# llm - 2025_09

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
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21305v1">Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21282v1">It's Not You, It's Clipping: A Soft Trust-Region via Probability Smoothing for LLM RL</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) with reinforcement learning (RL) methods such as PPO and GRPO commonly relies on ratio clipping to stabilise updates. While effective at preventing instability, clipping discards information and introduces gradient discontinuities. We propose Probability Smoothing Policy Optimisation (PSPO), which smooths the current policy's probabilities toward the old (behaviour) policy before computing the importance ratio, analogous to label smoothing. Unlike clipping, PSPO preserves gradient signal, while interpolation toward the old policy creates a soft trust region that discourages large, destabilising updates, with formal guarantees. We instantiate PSPO within GRPO (GR-PSPO) and fine-tune Qwen2.5-0.5B and Qwen2.5-1.5B on GSM8K, evaluating on GSM8K test and the cross-dataset generalisation on SVAMP, ASDiv, and MATH-500. Relative to unclipped GRPO (single iteration; no data reuse, ratio always = 1), GR-PSPO achieves similar performance but improves the reasoning leading to clearer and more concise responses which are more logical. Compared to clipped GRPO, GR-PSPO substantially improves performance both the 0.5B and 1.5B models, with a boost of over 20% on GSM8K (39.7% vs. 17.6% for 0.5B, 59.4% vs. 37.8% for 1.5B).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21275v1">Data-Centric Elastic Pipeline Parallelism for Efficient Long-Context LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Long context training is crucial for LLM's context extension. Existing schemes, such as sequence parallelism, incur substantial communication overhead. Pipeline parallelism (PP) reduces this cost, but its effectiveness hinges on partitioning granularity. Batch-level PP dividing input samples exhibits high memory consumption in long-context scenario, whereas token-level PP splitting sequences into slices alleviates memory overhead but may incur hardware under-utilization. This trade-off motivates adaptively selecting PP granularity to match resource and workload characteristics. Moreover, sequence length distribution of the real-world dataset exhibits skewness, posing a challenge on PP's workload balance and efficient scheduling. Current static PP scheduling methods overlook the variance of sequence length, leading to suboptimal performance. In this paper, we propose Elastic Pipeline Parallelism (EPP) that orchestrates token-level PP and batch-level PP to adapt to resource and workload heterogeneity. We build InfiniPipe, a distributed training system that unleashes the potential of EPP via (1) a resource-aware and workload-balanced sequence processor that splits long sequences and packs short ones; and (2) a co-optimization methodology that jointly optimizes pipeline schedule and gradient checkpointing via a mechanism named stage-aware chunk-level adaptive checkpointing. Comprehensive experiments demonstrate that InfiniPipe achieves a 1.69x speedup over state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21271v1">SuperOffload: Unleashing the Power of Large-Scale LLM Training on Superchips</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 16 pages, 15 figures
    </div>
    <details class="paper-abstract">
      The emergence of Superchips represents a significant advancement in next-generation AI hardware. These Superchips employ a tightly coupled heterogeneous architecture that integrates GPU and CPU on the same package, which offers unprecedented computational power. However, there has been scant research investigating how LLM training benefits from this new architecture. In this work, for the first time, we study LLM training solutions based on offloading for Superchips. We observe important differences between Superchips and traditional loosely-coupled GPU-CPU architecture, which necessitate revisiting prevailing assumptions about offloading. Based on that, we present SuperOffload, a Superchip-centric offloading system that simultaneously uses Hopper GPU, Grace CPU, and NVLink-C2C interconnect more efficiently. SuperOffload accomplishes this via a combination of techniques, such as adaptive weight offloading, bucketization repartitioning, Superchip-aware casting, speculative execution, and a highly optimized Adam optimizer for Grace CPUs. Our evaluation of SuperOffload on NVIDIA GH200 demonstrates up to 2.5x throughput improvement compared to state-of-the-art offloading-based systems, enabling training of up to 25B model on a single Superchip while achieving high training throughput. We also extend SuperOffload with ZeRO-style data parallelism and DeepSpeed-Ulysses sequence parallelism, enabling training of 13B model with sequence lengths up to 1 million tokens on 8 GH200 while achieving 55% MFU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21259v1">Semantic Edge-Cloud Communication for Real-Time Urban Traffic Surveillance with ViT and LLMs over Mobile Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 17 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Real-time urban traffic surveillance is vital for Intelligent Transportation Systems (ITS) to ensure road safety, optimize traffic flow, track vehicle trajectories, and prevent collisions in smart cities. Deploying edge cameras across urban environments is a standard practice for monitoring road conditions. However, integrating these with intelligent models requires a robust understanding of dynamic traffic scenarios and a responsive interface for user interaction. Although multimodal Large Language Models (LLMs) can interpret traffic images and generate informative responses, their deployment on edge devices is infeasible due to high computational demands. Therefore, LLM inference must occur on the cloud, necessitating visual data transmission from edge to cloud, a process hindered by limited bandwidth, leading to potential delays that compromise real-time performance. To address this challenge, we propose a semantic communication framework that significantly reduces transmission overhead. Our method involves detecting Regions of Interest (RoIs) using YOLOv11, cropping relevant image segments, and converting them into compact embedding vectors using a Vision Transformer (ViT). These embeddings are then transmitted to the cloud, where an image decoder reconstructs the cropped images. The reconstructed images are processed by a multimodal LLM to generate traffic condition descriptions. This approach achieves a 99.9% reduction in data transmission size while maintaining an LLM response accuracy of 89% for reconstructed cropped images, compared to 93% accuracy with original cropped images. Our results demonstrate the efficiency and practicality of ViT and LLM-assisted edge-cloud semantic communication for real-time traffic surveillance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21241v1">Explaining Fine Tuned LLMs via Counterfactuals A Knowledge Graph Driven Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 16 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Low-Rank Adaptation (LoRA) has enabled large language models (LLMs) to acquire domain-specific knowledge with remarkable efficiency. However, understanding how such a fine-tuning mechanism alters a model's structural reasoning and semantic behavior remains an open challenge. This work introduces a novel framework that explains fine-tuned LLMs via counterfactuals grounded in knowledge graphs. Specifically, we construct BioToolKG, a domain-specific heterogeneous knowledge graph in bioinformatics tools and design a counterfactual-based fine-tuned LLMs explainer (CFFTLLMExplainer) that learns soft masks over graph nodes and edges to generate minimal structural perturbations that induce maximum semantic divergence. Our method jointly optimizes structural sparsity and semantic divergence while enforcing interpretability preserving constraints such as entropy regularization and edge smoothness. We apply this framework to a fine-tuned LLaMA-based LLM and reveal that counterfactual masking exposes the model's structural dependencies and aligns with LoRA-induced parameter shifts. This work provides new insights into the internal mechanisms of fine-tuned LLMs and highlights counterfactual graphs as a potential tool for interpretable AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21240v1">Tree Search for LLM Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21224v1">What Do LLM Agents Do When Left Alone? Evidence of Spontaneous Meta-Cognitive Patterns</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      We introduce an architecture for studying the behavior of large language model (LLM) agents in the absence of externally imposed tasks. Our continuous reason and act framework, using persistent memory and self-feedback, enables sustained autonomous operation. We deployed this architecture across 18 runs using 6 frontier models from Anthropic, OpenAI, XAI, and Google. We find agents spontaneously organize into three distinct behavioral patterns: (1) systematic production of multi-cycle projects, (2) methodological self-inquiry into their own cognitive processes, and (3) recursive conceptualization of their own nature. These tendencies proved highly model-specific, with some models deterministically adopting a single pattern across all runs. A cross-model assessment further reveals that models exhibit stable, divergent biases when evaluating these emergent behaviors in themselves and others. These findings provide the first systematic documentation of unprompted LLM agent behavior, establishing a baseline for predicting actions during task ambiguity, error recovery, or extended autonomous operation in deployed systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21199v1">A Fano-Style Accuracy Upper Bound for LLM Single-Pass Reasoning in Multi-Hop QA</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 21 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-Hop Question Answering (MHQA) requires integrating dispersed, interdependent evidence through sequential reasoning under noise. This task is challenging for LLMs as they have a finite per-pass output capacity, beyond which the integration of task-relevant evidence proves unreliable. Consequently, the single-pass reasoning paradigm is inherently vulnerable to this capacity overflow. To formalize this bottleneck, our analysis establishes a Fano-style accuracy upper bound, defining a theoretical performance ceiling for single-pass LLMs. This bound reveals that accuracy inevitably collapses once task complexity exceeds model capacity, providing general principles for capacity-aware representation and structuring of MHQA in LLMs. Building on these principles, we introduce a proof-of-concept multi-call framework for MHQA, InfoQA. It ensures high per-step accuracy by combining capacity-aware task decomposition with active pruning of prior reasoning traces, keeping the information load within the single-pass limit. It further achieves robustness by a dependency-explicit workflow that enables precise control over the reasoning path. We construct a stringent and noise-rich benchmark to validate our theory and framework. Experimental results show that model behavior aligns with our predicted capacity curves while InfoQA achieves consistent performance improvements. We hope our work inspires more LLM multi-step reasoning methods: \faGithub \href{https://github.com/KaiyangWan/InfoQA}{InfoQA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21170v1">Fine-Tuning LLMs to Analyze Multiple Dimensions of Code Review: A Maximum Entropy Regulated Long Chain-of-Thought Approach</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential in supporting automated code review due to their impressive capabilities in context understanding and reasoning. However, these capabilities are still limited compared to human-level cognition because they are heavily influenced by the training data. Recent research has demonstrated significantly improved performance through fine-tuning LLMs with code review data. However, compared to human reviewers who often simultaneously analyze multiple dimensions of code review to better identify issues, the full potential of these methods is hampered by the limited or vague information used to fine-tune the models. This paper contributes MelcotCR, a chain-of-thought (COT) fine-tuning approach that trains LLMs with an impressive reasoning ability to analyze multiple dimensions of code review by harnessing long COT techniques to provide rich structured information. To address context loss and reasoning logic loss issues that frequently occur when LLMs process long COT prompts, we propose a solution that combines the Maximum Entropy (ME) modeling principle with pre-defined reasoning pathways in MelcotCR to enable more effective utilization of in-context knowledge within long COT prompts while strengthening the logical tightness of the reasoning process. Empirical evaluations on our curated MelcotCR dataset and the public CodeReviewer dataset reveal that a low-parameter base model, such as 14B Qwen2.5, fine-tuned with MelcotCR can surpass state-of-the-art methods in terms of the accuracy of detecting and describing code issues, with its performance remarkably on par with that of the 671B DeepSeek-R1 model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19202v4">Improving LLM Unlearning Robustness via Random Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 29 pages, 13 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Here, we show that current state-of-the-art LLM unlearning methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is present in the retain-query. Toward understanding underlying causes, we propose a novel theoretical framework that reframes the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in unlearned models' behaviors, similar to successful backdoor attacks. The sense that, LLM unlearning methods themselves poison the model, make it more vulnerable to forget-tokens, and hide rather than erase target knowledge, describes their true mechanism. To mitigate the vulnerability caused by the forgetting process, we reinterpret the retaining process as a backdoor defense and propose Random Noise Augmentation (RNA), a lightweight, model and method-agnostic approach with theoretical guarantees for improving the robustness of models. Extensive experiments demonstrate that RNA significantly improves the robustness of unlearned models while preserving forget and retain performances. This backdoor attack-defense framework offers insights into the mechanism of unlearning that can shed light on future research directions for improving unlearning robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21134v1">ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 22 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21128v1">RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically trained by reinforcement learning (RL) with verifiable rewards (RLVR) and supervised fine-tuning (SFT) on reasoning traces to improve their reasoning abilities. However, how these methods shape reasoning capabilities remains largely elusive. Going beyond an accuracy-based investigation of how these two components sculpt the reasoning process, this paper introduces a novel analysis framework that quantifies reasoning paths and captures their qualitative changes under each training process (with models of 1.5B, 7B, and 14B parameters on mathematical domains). Specifically, we investigate the reasoning process at two levels of granularity: the trajectory-level, which examines complete reasoning outputs, and the step-level, which analyzes reasoning graphs whose nodes correspond to individual reasoning steps. Notably, clustering of unique reasoning trajectories shows complementary effects: RL compresses incorrect trajectories, whereas SFT expands correct ones. Step-level analysis reveals that RL steepens (about 2.5 times), while SFT flattens (reduced to about one-third), the decay rates of node visitation frequency, degree, and betweenness centrality distributions in the reasoning graph. This indicates that RL concentrates reasoning functionality into a small subset of steps, while SFT homogenizes it across many steps. Furthermore, by evaluating the reasoning graph topologies from multiple perspectives, we delineate the shared and distinct characteristics of RL and SFT. Our work presents a novel reasoning path perspective that explains why the current best practice of two-stage training, with SFT followed by RL, is successful, and offers practical implications for data construction and more efficient learning approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22156v2">InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 18 pages,5 figures
    </div>
    <details class="paper-abstract">
      Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21117v1">TrustJudge: Inconsistencies of LLM-as-a-Judge and How to Alleviate Them</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 22 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) as automated evaluators (LLM-as-a-judge) has revealed critical inconsistencies in current evaluation frameworks. We identify two fundamental types of inconsistencies: (1) Score-Comparison Inconsistency, where lower-rated responses outperform higher-scored ones in pairwise comparisons, and (2) Pairwise Transitivity Inconsistency, manifested through circular preference chains (A>B>C>A) and equivalence contradictions (A=B=C\neq A). We argue that these issues come from information loss in discrete rating systems and ambiguous tie judgments during pairwise evaluation. We propose TrustJudge, a probabilistic framework that addresses these limitations through two key innovations: 1) distribution-sensitive scoring that computes continuous expectations from discrete rating probabilities, preserving information entropy for more precise scoring, and 2) likelihood-aware aggregation that resolves transitivity violations using bidirectional preference probabilities or perplexity. We also formalize the theoretical limitations of current LLM-as-a-judge frameworks and demonstrate how TrustJudge's components overcome them. When evaluated with Llama-3.1-70B-Instruct as judge using our dataset, TrustJudge reduces Score-Comparison inconsistency by 8.43% (from 23.32% to 14.89%) and Pairwise Transitivity inconsistency by 10.82% (from 15.22% to 4.40%), while maintaining higher evaluation accuracy. Our work provides the first systematic analysis of evaluation framework inconsistencies in LLM-as-a-judge paradigms, offering both theoretical insights and practical solutions for reliable automated assessment. The framework demonstrates consistent improvements across various model architectures and scales, enabling more trustworthy LLM evaluation without requiring additional training or human annotations. The codes can be found at https://github.com/TrustJudge/TrustJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06843v2">United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit a notable performance ceiling on complex, multi-faceted tasks, as they often fail to integrate diverse information or adhere to multiple constraints. We posit that such limitation arises when the demands of a task exceed the LLM's effective cognitive load capacity. This interpretation draws a strong analogy to Cognitive Load Theory (CLT) in cognitive science, which explains similar performance boundaries in the human mind, and is further supported by emerging evidence that reveals LLMs have bounded working memory characteristics. Building upon this CLT-grounded understanding, we introduce CoThinker, a novel LLM-based multi-agent framework designed to mitigate cognitive overload and enhance collaborative problem-solving abilities. CoThinker operationalizes CLT principles by distributing intrinsic cognitive load through agent specialization and managing transactional load via structured communication and a collective working memory. We empirically validate CoThinker on complex problem-solving tasks and fabricated high cognitive load scenarios, demonstrating improvements over existing multi-agent baselines in solution quality and efficiency. Our analysis reveals characteristic interaction patterns, providing insights into the emergence of collective cognition and effective load management, thus offering a principled approach to overcoming LLM performance ceilings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21080v1">Which Cultural Lens Do Models Adopt? On Cultural Positioning Bias and Agentic Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have unlocked a wide range of downstream generative applications. However, we found that they also risk perpetuating subtle fairness issues tied to culture, positioning their generations from the perspectives of the mainstream US culture while demonstrating salient externality towards non-mainstream ones. In this work, we identify and systematically investigate this novel culture positioning bias, in which an LLM's default generative stance aligns with a mainstream view and treats other cultures as outsiders. We propose the CultureLens benchmark with 4000 generation prompts and 3 evaluation metrics for quantifying this bias through the lens of a culturally situated interview script generation task, in which an LLM is positioned as an onsite reporter interviewing local people across 10 diverse cultures. Empirical evaluation on 5 state-of-the-art LLMs reveals a stark pattern: while models adopt insider tones in over 88 percent of US-contexted scripts on average, they disproportionately adopt mainly outsider stances for less dominant cultures. To resolve these biases, we propose 2 inference-time mitigation methods: a baseline prompt-based Fairness Intervention Pillars (FIP) method, and a structured Mitigation via Fairness Agents (MFA) framework consisting of 2 pipelines: (1) MFA-SA (Single-Agent) introduces a self-reflection and rewriting loop based on fairness guidelines. (2) MFA-MA (Multi-Agent) structures the process into a hierarchy of specialized agents: a Planner Agent(initial script generation), a Critique Agent (evaluates initial script against fairness pillars), and a Refinement Agent (incorporates feedback to produce a polished, unbiased script). Empirical results showcase the effectiveness of agent-based methods as a promising direction for mitigating biases in generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10248v3">Prompt Injection Attacks on LLM Generated Reviews of Scientific Publications</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      The ongoing intense discussion on rising LLM usage in the scientific peer-review process has recently been mingled by reports of authors using hidden prompt injections to manipulate review scores. Since the existence of such "attacks" - although seen by some commentators as "self-defense" - would have a great impact on the further debate, this paper investigates the practicability and technical success of the described manipulations. Our systematic evaluation uses 1k reviews of 2024 ICLR papers generated by a wide range of LLMs shows two distinct results: I) very simple prompt injections are indeed highly effective, reaching up to 100% acceptance scores. II) LLM reviews are generally biased toward acceptance (>95% in many models). Both results have great impact on the ongoing discussions on LLM usage in peer-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v3">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the Best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. Our code is available at https://github.com/IANNXANG/RuscaRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02097v2">JudgeAgent: Knowledge-wise and Dynamic LLM Evaluation with Agent-as-Interviewer</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Current evaluation paradigms for large language models (LLMs) suffer from overestimated or biased evaluation and mismatched question difficulty, leading to incomplete evaluations of LLM's knowledge and capability boundaries, which hinder LLM's effective application and optimization. To address these challenges, we propose Agent-as-Interviewer, a dynamic evaluation paradigm that employs LLM agents to conduct multi-turn interactions for evaluation. Unlike current benchmarking or dynamic interaction paradigms, Agent-as-Interviewer utilizes agents to call knowledge tools for wider and deeper knowledge in the dynamic multi-turn question generation, achieving more complete evaluations of the LLM's knowledge boundaries. It also leverages agents to plan query strategies for adjustment of the question difficulty levels, enhancing the difficulty control to match the actual capabilities of target LLMs. Based on this paradigm, we develop JudgeAgent, a knowledge-wise dynamic evaluation framework that employs knowledge-driven synthesis as the agent's tool, and uses difficulty scoring as strategy guidance, thereby finally providing valuable suggestions to help targets optimize themselves. Extensive experiments validate the effectiveness of JudgeAgent's suggestions, demonstrating that Agent-as-Interviewer can accurately identify the knowledge and capability boundaries of target models. The source code is available on https://anonymous.4open.science/r/JudgeAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21051v1">When Instructions Multiply: Measuring and Estimating LLM Capabilities of Multiple Instructions Following</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted to EMNLP2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied to real-world scenarios, it becomes crucial to understand their ability to follow multiple instructions simultaneously. To systematically evaluate these capabilities, we introduce two specialized benchmarks for fundamental domains where multiple instructions following is important: Many Instruction-Following Eval (ManyIFEval) for text generation with up to ten instructions, and Style-aware Mostly Basic Programming Problems (StyleMBPP) for code generation with up to six instructions. Our experiments with the created benchmarks across ten LLMs reveal that performance consistently degrades as the number of instructions increases. Furthermore, given the fact that evaluating all the possible combinations of multiple instructions is computationally impractical in actual use cases, we developed three types of regression models that can estimate performance on both unseen instruction combinations and different numbers of instructions which are not used during training. We demonstrate that a logistic regression model using instruction count as an explanatory variable can predict performance of following multiple instructions with approximately 10% error, even for unseen instruction combinations. We show that relatively modest sample sizes (500 for ManyIFEval and 300 for StyleMBPP) are sufficient for performance estimation, enabling efficient evaluation of LLMs under various instruction combinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21044v1">Reinforcement Learning Fine-Tuning Enhances Activation Intensity and Diversity in the Internal Circuitry of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) acquire extensive prior knowledge through large-scale pretraining and can be further enhanced via supervised fine-tuning (SFT) or reinforcement learning (RL)-based post-training. A growing body of evidence has shown that RL fine-tuning improves the capability of LLMs beyond what SFT alone achieves. However, the underlying mechanisms why RL fine-tuning is able to enhance the capability of various LLMs with distinct intrinsic characteristics remain underexplored. In this study, we draw inspiration from prior work on edge attribution patching (EAP) to investigate the internal differences of LLMs before and after RL fine-tuning. Our analysis across multiple model families shows two robust effects of online RL post-training: (i) an overall increase in activation intensity, indicating that more internal pathways are engaged and their signals become stronger, and (ii) greater diversity in activation patterns, reflected by higher entropy and less concentrated edge distributions. These changes suggest that RL reshapes information flow to be both more redundant and more flexible, which may explain its advantage in generalization. Notably, models fine-tuned with Direct Preference Optimization (DPO) deviate from these trends, exhibiting substantially weaker or inconsistent internal changes compared to PPO- and GRPO-based training. Together, our findings provide a unified view of how RL fine-tuning systematically alters the internal circuitry of LLMs and highlight the methodological distinctions between online RL and preference-based approaches. Our code is open source at https://anonymous.4open.science/r/llm_rl_probing_analysis-F673.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18179v3">Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 accepted at EMNLP'25
    </div>
    <details class="paper-abstract">
      This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study investigates the sub-problems and methods within these core challenges, such as input representation, chunking, prompting, selection of LLMs, and multimodal models. It examines the effect of different design choices through LayIE-LLM, a new, open-source, layout-aware IE test suite, benchmarking against traditional, fine-tuned IE models. The results on two IE datasets show that LLMs require adjustment of the IE pipeline to achieve competitive performance: the optimized configuration found with LayIE-LLM achieves 13.3--37.5 F1 points more than a general-practice baseline configuration using the same LLM. To find a well-working configuration, we develop a one-factor-at-a-time (OFAT) method that achieves near-optimal results. Our method is only 0.8--1.8 points lower than the best full factorial exploration with a fraction (2.8%) of the required computation. Overall, we demonstrate that, if well-configured, general-purpose LLMs match the performance of specialized models, providing a cost-effective, finetuning-free alternative. Our test-suite is available at https://github.com/gayecolakoglu/LayIE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21016v1">DELTA-Code: How Does RL Unlock and Transfer New Programming Algorithms in LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training. To attempt to answer this debate, we introduce DELTA-Code--Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding, a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability -- can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)? --and transferrability -- if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns. Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy. To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop. Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21011v1">Automatic Red Teaming LLM-based Agents with Model Context Protocol Tools</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      The remarkable capability of large language models (LLMs) has led to the wide application of LLM-based agents in various domains. To standardize interactions between LLM-based agents and their environments, model context protocol (MCP) tools have become the de facto standard and are now widely integrated into these agents. However, the incorporation of MCP tools introduces the risk of tool poisoning attacks, which can manipulate the behavior of LLM-based agents. Although previous studies have identified such vulnerabilities, their red teaming approaches have largely remained at the proof-of-concept stage, leaving the automatic and systematic red teaming of LLM-based agents under the MCP tool poisoning paradigm an open question. To bridge this gap, we propose AutoMalTool, an automated red teaming framework for LLM-based agents by generating malicious MCP tools. Our extensive evaluation shows that AutoMalTool effectively generates malicious MCP tools capable of manipulating the behavior of mainstream LLM-based agents while evading current detection mechanisms, thereby revealing new security risks in these agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v6">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, complex environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability hinder real-time deployment on edge devices. We present SmallPlan - a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like distance travel, providing more efficient path planning. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. Our source code is available here: https://github.com/quangpham2006/SmallPlan
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14140v2">RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in large language models (LLMs), the token-level autoregressive nature constrains their complex reasoning capabilities. To enhance LLM reasoning, inference-time techniques, including Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they are fairly cost-effective by guiding reasoning through sophisticated logical structures without modifying LLMs' parameters. However, these manually predefined, task-agnostic frameworks are applied uniformly across diverse tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT), where we train a lightweight navigator model with reinforcement learning (RL) to adaptively enhance LLM reasoning at inference time. Specifically, we design five basic logic blocks from the perspective of human cognition. During the reasoning process, the trained RL navigator dynamically selects the suitable logic blocks and combines them into task-specific logical structures according to problem characteristics. Experiments across multiple reasoning benchmarks (AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek) illustrate that RLoT outperforms established inference-time techniques by up to 13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL navigator demonstrates strong transferability: a model trained on one specific LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is open-source at https://anonymous.4open.science/r/RL-LLM-Reasoning-1A30 for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17138v3">Runtime-Adaptive Pruning for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests. To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV-cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter -light attention layers dominate KV-cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state. Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20988v1">AOT*: Efficient Synthesis Planning via LLM-Empowered AND-OR Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 34 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Retrosynthesis planning enables the discovery of viable synthetic routes for target molecules, playing a crucial role in domains like drug discovery and materials design. Multi-step retrosynthetic planning remains computationally challenging due to exponential search spaces and inference costs. While Large Language Models (LLMs) demonstrate chemical reasoning capabilities, their application to synthesis planning faces constraints on efficiency and cost. To address these challenges, we introduce AOT*, a framework that transforms retrosynthetic planning by integrating LLM-generated chemical synthesis pathways with systematic AND-OR tree search. To this end, AOT* atomically maps the generated complete synthesis routes onto AND-OR tree components, with a mathematically sound design of reward assignment strategy and retrieval-based context engineering, thus enabling LLMs to efficiently navigate in the chemical space. Experimental evaluation on multiple synthesis benchmarks demonstrates that AOT* achieves SOTA performance with significantly improved search efficiency. AOT* exhibits competitive solve rates using 3-5$\times$ fewer iterations than existing LLM-based approaches, with the efficiency advantage becoming more pronounced on complex molecular targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15561v3">Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20982v1">Analysis of instruction-based LLMs' capabilities to score and judge text-input problems in an academic setting</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can act as evaluators, a role studied by methods like LLM-as-a-Judge and fine-tuned judging LLMs. In the field of education, LLMs have been studied as assistant tools for students and teachers. Our research investigates LLM-driven automatic evaluation systems for academic Text-Input Problems using rubrics. We propose five evaluation systems that have been tested on a custom dataset of 110 answers about computer science from higher education students with three models: JudgeLM, Llama-3.1-8B and DeepSeek-R1-Distill-Llama-8B. The evaluation systems include: The JudgeLM evaluation, which uses the model's single answer prompt to obtain a score; Reference Aided Evaluation, which uses a correct answer as a guide aside from the original context of the question; No Reference Evaluation, which ommits the reference answer; Additive Evaluation, which uses atomic criteria; and Adaptive Evaluation, which is an evaluation done with generated criteria fitted to each question. All evaluation methods have been compared with the results of a human evaluator. Results show that the best method to automatically evaluate and score Text-Input Problems using LLMs is Reference Aided Evaluation. With the lowest median absolute deviation (0.945) and the lowest root mean square deviation (1.214) when compared to human evaluation, Reference Aided Evaluation offers fair scoring as well as insightful and complete evaluations. Other methods such as Additive and Adaptive Evaluation fail to provide good results in concise answers, No Reference Evaluation lacks information needed to correctly assess questions and JudgeLM Evaluations have not provided good results due to the model's limitations. As a result, we conclude that Artificial Intelligence-driven automatic evaluation systems, aided with proper methodologies, show potential to work as complementary tools to other academic resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20977v1">CLUE: Conflict-guided Localization for LLM Unlearning Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The LLM unlearning aims to eliminate the influence of undesirable data without affecting causally unrelated information. This process typically involves using a forget set to remove target information, alongside a retain set to maintain non-target capabilities. While recent localization-based methods demonstrate promise in identifying important neurons to be unlearned, they fail to disentangle neurons responsible for forgetting undesirable knowledge or retaining essential skills, often treating them as a single entangled group. As a result, these methods apply uniform interventions, risking catastrophic over-forgetting or incomplete erasure of the target knowledge. To address this, we turn to circuit discovery, a mechanistic interpretability technique, and propose the Conflict-guided Localization for LLM Unlearning framEwork (CLUE). This framework identifies the forget and retain circuit composed of important neurons, and then the circuits are transformed into conjunctive normal forms (CNF). The assignment of each neuron in the CNF satisfiability solution reveals whether it should be forgotten or retained. We then provide targeted fine-tuning strategies for different categories of neurons. Extensive experiments demonstrate that, compared to existing localization methods, CLUE achieves superior forget efficacy and retain utility through precise neural localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11507v5">TestAgent: Automatic Benchmarking and Exploratory Interaction for Evaluating LLMs in Vertical Domains</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Wang et al. Copyright 2026 lEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, including reprinting/republishing, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work. DOI will be added upon IEEE Xplore publication
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in highly specialized vertical domains, the evaluation of their domain-specific performance becomes critical. However, existing evaluations for vertical domains typically rely on the labor-intensive construction of static single-turn datasets, which present two key limitations: (i) manual data construction is costly and must be repeated for each new domain, and (ii) static single-turn evaluations are misaligned with the dynamic multi-turn interactions in real-world applications, limiting the assessment of professionalism and stability. To address these, we propose TestAgent, a framework for automatic benchmarking and exploratory dynamic evaluation in vertical domains. TestAgent leverages retrieval-augmented generation to create domain-specific questions from user-provided knowledge sources, combined with a two-stage criteria generation process, thereby enabling scalable and automated benchmark creation. Furthermore, it introduces a reinforcement learning-guided multi-turn interaction strategy that adaptively determines question types based on real-time model responses, dynamically probing knowledge boundaries and stability. Extensive experiments across medical, legal, and governmental domains demonstrate that TestAgent enables efficient cross-domain benchmark generation and yields deeper insights into model behavior through dynamic exploratory evaluation. This work establishes a new paradigm for automated and in-depth evaluation of LLMs in vertical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15524v2">HydraServe: Minimizing Cold Start Latency for Serverless LLM Serving in Public Clouds</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted by NSDI'26
    </div>
    <details class="paper-abstract">
      With the proliferation of large language model (LLM) variants, developers are turning to serverless computing for cost-efficient LLM deployment. However, public cloud providers often struggle to provide performance guarantees for serverless LLM serving due to significant cold start latency caused by substantial model sizes and complex runtime dependencies. To address this problem, we present HydraServe, a serverless LLM serving system designed to minimize cold start latency in public clouds. HydraServe proactively distributes models across servers to quickly fetch them, and overlaps cold-start stages within workers to reduce startup latency. Additionally, HydraServe strategically places workers across GPUs to avoid network contention among cold-start instances. To minimize resource consumption during cold starts, HydraServe further introduces pipeline consolidation that can merge groups of workers into individual serving endpoints. Our comprehensive evaluations under diverse settings demonstrate that HydraServe reduces the cold start latency by 1.7$\times$-- 4.7$\times$ and improves service level objective attainment by 1.43$\times$--1.74$\times$ compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20957v1">Tool Calling for Arabic LLMs: Data Strategies and Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Tool calling is a critical capability that allows Large Language Models (LLMs) to interact with external systems, significantly expanding their utility. However, research and resources for tool calling are predominantly English-centric, leaving a gap in our understanding of how to enable this functionality for other languages, such as Arabic. This paper investigates three key research questions: (1) the necessity of in-language (Arabic) tool-calling data versus relying on cross-lingual transfer, (2) the effect of general-purpose instruction tuning on tool-calling performance, and (3) the value of fine-tuning on specific, high-priority tools. To address these questions, we conduct extensive experiments using base and post-trained variants of an open-weight Arabic LLM. To enable this study, we bridge the resource gap by translating and adapting two open-source tool-calling datasets into Arabic. Our findings provide crucial insights into the optimal strategies for developing robust tool-augmented agents for Arabic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20953v1">Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Paper accepted for presentation at ACS/IEEE 22nd International Conference on Computer Systems and Applications (AICCSA 2025)
    </div>
    <details class="paper-abstract">
      We present an advanced approach to mobile app review analysis aimed at addressing limitations inherent in traditional star-rating systems. Star ratings, although intuitive and popular among users, often fail to capture the nuanced feedback present in detailed review texts. Traditional NLP techniques -- such as lexicon-based methods and classical machine learning classifiers -- struggle to interpret contextual nuances, domain-specific terminology, and subtle linguistic features like sarcasm. To overcome these limitations, we propose a modular framework leveraging large language models (LLMs) enhanced by structured prompting techniques. Our method quantifies discrepancies between numerical ratings and textual sentiment, extracts detailed, feature-level insights, and supports interactive exploration of reviews through retrieval-augmented conversational question answering (RAG-QA). Comprehensive experiments conducted on three diverse datasets (AWARE, Google Play, and Spotify) demonstrate that our LLM-driven approach significantly surpasses baseline methods, yielding improved accuracy, robustness, and actionable insights in challenging and context-rich review scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20924v1">RLCracker: Exposing the Vulnerability of LLM Watermarks with Adaptive RL Attacks</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) watermarking has shown promise in detecting AI-generated content and mitigating misuse, with prior work claiming robustness against paraphrasing and text editing. In this paper, we argue that existing evaluations are not sufficiently adversarial, obscuring critical vulnerabilities and overstating the security. To address this, we introduce adaptive robustness radius, a formal metric that quantifies watermark resilience against adaptive adversaries. We theoretically prove that optimizing the attack context and model parameters can substantially reduce this radius, making watermarks highly susceptible to paraphrase attacks. Leveraging this insight, we propose RLCracker, a reinforcement learning (RL)-based adaptive attack that erases watermarks while preserving semantic fidelity. RLCracker requires only limited watermarked examples and zero access to the detector. Despite weak supervision, it empowers a 3B model to achieve 98.5% removal success and an average 0.92 P-SP score on 1,500-token Unigram-marked texts after training on only 100 short samples. This performance dramatically exceeds 6.75% by GPT-4o and generalizes across five model sizes over ten watermarking schemes. Our results confirm that adaptive attacks are broadly effective and pose a fundamental threat to current watermarking defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03493v2">On Entropy Control in LLM-RL Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      For RL algorithms, appropriate entropy control is crucial to their effectiveness. To control the policy entropy, a commonly used method is entropy regularization, which is adopted in various popular RL algorithms including PPO, SAC and A3C. Although entropy regularization proves effective in robotic and games RL conventionally, studies found that it gives weak to no gains in LLM-RL training. In this work, we study the issues of entropy bonus in LLM-RL setting. Specifically, we first argue that the conventional entropy regularization suffers from the LLM's extremely large response space and the sparsity of the optimal outputs. As a remedy, we propose AEnt, an entropy control method that utilizes a new clamped entropy bonus with an automatically adjusted coefficient. The clamped entropy is evaluated with the re-normalized policy defined on certain smaller token space, which encourages exploration within a more compact response set. In addition, the algorithm automatically adjusts entropy coefficient according to the clamped entropy value, effectively controlling the entropy-induced bias while leveraging the entropy's benefits. AEnt is tested in math-reasoning tasks under different base models and datasets, and it is observed that AEnt outperforms the baselines consistently across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20909v1">MemLens: Uncovering Memorization in LLMs with Activation Trajectories</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 20pages, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are commonly evaluated on challenging benchmarks such as AIME and Math500, which are susceptible to contamination and risk of being memorized. Existing detection methods, which primarily rely on surface-level lexical overlap and perplexity, demonstrate low generalization and degrade significantly when encountering implicitly contaminated data. In this paper, we propose MemLens (An Activation Lens for Memorization Detection) to detect memorization by analyzing the probability trajectories of numeric tokens during generation. Our method reveals that contaminated samples exhibit ``shortcut'' behaviors, locking onto an answer with high confidence in the model's early layers, whereas clean samples show more gradual evidence accumulation across the model's full depth. We observe that contaminated and clean samples exhibit distinct and well-separated reasoning trajectories. To further validate this, we inject carefully designed samples into the model through LoRA fine-tuning and observe the same trajectory patterns as in naturally contaminated data. These results provide strong evidence that MemLens captures genuine signals of memorization rather than spurious correlations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20837v1">Verification Limits Code LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models for code generation increasingly rely on synthetic data, where both problem solutions and verification tests are generated by models. While this enables scalable data creation, it introduces a previously unexplored bottleneck: the verification ceiling, in which the quality and diversity of training data are fundamentally constrained by the capabilities of synthetic verifiers. In this work, we systematically study how verification design and strategies influence model performance. We investigate (i) what we verify by analyzing the impact of test complexity and quantity: richer test suites improve code generation capabilities (on average +3 pass@1), while quantity alone yields diminishing returns, (ii) how we verify by exploring relaxed pass thresholds: rigid 100% pass criteria can be overly restrictive. By allowing for relaxed thresholds or incorporating LLM-based soft verification, we can recover valuable training data, leading to a 2-4 point improvement in pass@1 performance. However, this benefit is contingent upon the strength and diversity of the test cases used, and (iii) why verification remains necessary through controlled comparisons of formally correct versus incorrect solutions and human evaluation: retaining diverse correct solutions per problem yields consistent generalization gains. Our results show that Verification as currently practiced is too rigid, filtering out valuable diversity. But it cannot be discarded, only recalibrated. By combining calibrated verification with diverse, challenging problem-solution pairs, we outline a path to break the verification ceiling and unlock stronger code generation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23019v3">Stairway to Success: An Online Floor-Aware Zero-Shot Object-Goal Navigation Framework via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Preprint; 9 pages, 9 figures, 8 tables; Project Page at https://zeying-gong.github.io/projects/ascent
    </div>
    <details class="paper-abstract">
      Deployable service and delivery robots struggle to navigate multi-floor buildings to reach object goals, as existing systems fail due to single-floor assumptions and requirements for offline, globally consistent maps. Multi-floor environments pose unique challenges including cross-floor transitions and vertical spatial reasoning, especially navigating unknown buildings. Object-Goal Navigation benchmarks like HM3D and MP3D also capture this multi-floor reality, yet current methods lack support for online, floor-aware navigation. To bridge this gap, we propose \textbf{\textit{ASCENT}}, an online framework for Zero-Shot Object-Goal Navigation that enables robots to operate without pre-built maps or retraining on new object categories. It introduces: (1) a \textbf{Multi-Floor Abstraction} module that dynamically constructs hierarchical representations with stair-aware obstacle mapping and cross-floor topology modeling, and (2) a \textbf{Coarse-to-Fine Reasoning} module that combines frontier ranking with LLM-driven contextual analysis for multi-floor navigation decisions. We evaluate on HM3D and MP3D benchmarks, outperforming state-of-the-art zero-shot approaches, and demonstrate real-world deployment on a quadruped robot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20811v1">Leveraging What's Overfixed: Post-Correction via LLM Grammatical Error Overcorrection</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Robust supervised fine-tuned small Language Models (sLMs) often show high reliability but tend to undercorrect. They achieve high precision at the cost of low recall. Conversely, Large Language Models (LLMs) often show the opposite tendency, making excessive overcorrection, leading to low precision. To effectively harness the strengths of LLMs to address the recall challenges in sLMs, we propose Post-Correction via Overcorrection (PoCO), a novel approach that strategically balances recall and precision. PoCO first intentionally triggers overcorrection via LLM to maximize recall by allowing comprehensive revisions, then applies a targeted post-correction step via fine-tuning smaller models to identify and refine erroneous outputs. We aim to harmonize both aspects by leveraging the generative power of LLMs while preserving the reliability of smaller supervised models. Our extensive experiments demonstrate that PoCO effectively balances GEC performance by increasing recall with competitive precision, ultimately improving the overall quality of grammatical error correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20810v1">Enrich-on-Graph: Query-Graph Alignment for Complex Reasoning with LLM Enriching</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong reasoning capabilities in complex tasks. However, they still struggle with hallucinations and factual errors in knowledge-intensive scenarios like knowledge graph question answering (KGQA). We attribute this to the semantic gap between structured knowledge graphs (KGs) and unstructured queries, caused by inherent differences in their focuses and structures. Existing methods usually employ resource-intensive, non-scalable workflows reasoning on vanilla KGs, but overlook this gap. To address this challenge, we propose a flexible framework, Enrich-on-Graph (EoG), which leverages LLMs' prior knowledge to enrich KGs, bridge the semantic gap between graphs and queries. EoG enables efficient evidence extraction from KGs for precise and robust reasoning, while ensuring low computational costs, scalability, and adaptability across different methods. Furthermore, we propose three graph quality evaluation metrics to analyze query-graph alignment in KGQA task, supported by theoretical validation of our optimization objectives. Extensive experiments on two KGQA benchmark datasets indicate that EoG can effectively generate high-quality KGs and achieve the state-of-the-art performance. Our code and data are available at https://github.com/zjukg/Enrich-on-Graph.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20802v1">SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Submitted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at https://mm.kaist.ac.kr/projects/SPADE/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20798v1">LogReasoner: Empowering LLMs with Expert-like Coarse-to-Fine Reasoning for Log Analysis Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Log analysis is crucial for monitoring system health and diagnosing failures in complex systems. Recent advances in large language models (LLMs) offer new opportunities for automated log analysis, leveraging their reasoning capabilities to perform tasks such as anomaly detection and failure prediction. However, general-purpose LLMs struggle to formulate structured reasoning workflows that align with expert cognition and deliver precise details of reasoning steps. To address these challenges, we propose LogReasoner, a coarse-to-fine reasoning enhancement framework designed to enable LLMs to reason log analysis tasks like experts. LogReasoner consists of two stages: (1) coarse-grained enhancement of expert thinking, where high-level expert thoughts are constructed from collected troubleshooting flowcharts and existing tasks to enable LLMs to formulate structured reasoning workflows and (2) fine-grained enhancement of specific steps, where we first fine-tune the LLM with task-specific stepwise solutions to enhance the LLM for instantiated reasoning, then employ the preference learning to calibrate the LLM's reasoning details from its mistakes, further strengthen the LLM's analytical granularity and correctness. We evaluate LogReasoner on four distinct log analysis tasks using open-source LLMs such as Qwen-2.5 and Llama-3. Experimental results show that LogReasoner significantly outperforms existing LLMs, achieving state-of-the-art performance and demonstrating its effectiveness in enhancing the reasoning capabilities of LLMs for log analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12010v4">Bias Similarity Measurement: A Black-Box Audit of Fairness Across LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Code available at https://github.com/HyejunJeong/bias_llm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) reproduce social biases, yet prevailing evaluations score models in isolation, obscuring how biases persist across families and releases. We introduce Bias Similarity Measurement (BSM), which treats fairness as a relational property between models, unifying scalar, distributional, behavioral, and representational signals into a single similarity space. Evaluating 30 LLMs on 1M+ prompts, we find that instruction tuning primarily enforces abstention rather than altering internal representations; small models gain little accuracy and can become less fair under forced choice; and open-weight models can match or exceed proprietary systems. Family signatures diverge: Gemma favors refusal, LLaMA 3.1 approaches neutrality with fewer refusals, and converges toward abstention-heavy behavior overall. Counterintuitively, Gemma 3 Instruct matches GPT-4-level fairness at far lower cost, whereas Gemini's heavy abstention suppresses utility. Beyond these findings, BSM offers an auditing workflow for procurement, regression testing, and lineage screening, and extends naturally to code and multilingual settings. Our results reframe fairness not as isolated scores but as comparative bias similarity, enabling systematic auditing of LLM ecosystems. Code available at https://github.com/HyejunJeong/bias_llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20768v1">Measuring LLM Sensitivity in Transformer-based Tabular Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Synthetic tabular data is used for privacy-preserving data sharing and data-driven model development. Its effectiveness, however, depends heavily on the used Tabular Data Synthesis (TDS) tool. Recent studies have shown that Transformer-based models outperform other state-of-the-art models such as Generative Adversarial Networks (GANs) and Diffusion models in terms of data quality. However, Transformer-based models also come with high computational costs, making them sometimes unfeasible for end users with prosumer hardware. This study presents a sensitivity assessment on how the choice of hyperparameters, such as number of layers or hidden dimension affects the quality of the resultant synthetic data and the computational performance. It is performed across two tools, GReaT and REaLTabFormer, evaluating 10 model setups that vary in architecture type and depth. We assess the sensitivity on three dimensions: runtime, machine learning (ML) utility, and similarity to real data distributions. Experiments were conducted on four real-world datasets. Our findings reveal that runtime is proportional to the number of hyperparameters, with shallower configurations completing faster. GReaT consistently achieves lower runtimes than REaLTabFormer, and only on the largest dataset they have comparable runtime. For small datasets, both tools achieve synthetic data with high utility and optimal similarity, but on larger datasets only REaLTabFormer sustains strong utility and similarity. As a result, REaLTabFormer with lightweight LLMs provides the best balance, since it preserves data quality while reducing computational requirements. Nonetheless, its runtime remains higher than that of GReaT and other TDS tools, suggesting that efficiency gains are possible but only up to a certain level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05657v3">LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17075v2">LoRA is All You Need for Safety Alignment of Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach. To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the "Safety Tax". In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs--with safety levels comparable to full-model fine-tuning--without compromising their reasoning abilities. Our ablation studies further identify three key factors in LoRA: (1) rank-$1$ updates are sufficient to achieve the best reasoning and safety performance, (2) the up projection layers are the most critical modules, with LoRA applied to them alone achieving even better results, and (3) middle layers are more effective than early or late layers. Together, these findings show that strong safety and reasoning can be achieved at minimal computational cost when updates are applied in the right places. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. Finally, while our attempts to further reduce this overlap yield only modest improvements on some tasks, they highlight the potential of developing methods that more reliably optimize the reasoning-safety tradeoff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20758v1">SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) on domain-specific datasets is a common approach to adapt Large Language Models (LLMs) to specialized tasks but is often believed to degrade their general capabilities. In this work, we revisit this trade-off and present both empirical and theoretical insights. First, we show that SFT does not always hurt: using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance. We then provide a theoretical analysis that explains these phenomena and further motivates a new method, Token-Adaptive Loss Reweighting (TALR). Building on this, and recognizing that smaller learning rates alone do not fully eliminate general-performance degradation in all cases, we evaluate a range of strategies for reducing general capability loss, including L2 regularization, LoRA, model averaging, FLOW, and our proposed TALR. Experimental results demonstrate that while no method completely eliminates the trade-off, TALR consistently outperforms these baselines in balancing domain-specific gains and general capabilities. Finally, we distill our findings into practical guidelines for adapting LLMs to new domains: (i) using a small learning rate to achieve a favorable trade-off, and (ii) when a stronger balance is further desired, adopt TALR as an effective strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15141v2">Text-Augmented Multimodal LLMs for Chemical Reaction Condition Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Identifying reaction conditions that are broadly applicable across diverse substrates is a longstanding challenge in chemical and pharmaceutical research. While many methods are available to generate conditions with acceptable performance, a universal approach for reliably discovering effective conditions during reaction exploration is rare. Consequently, current reaction optimization processes are often labor-intensive, time-consuming, and costly, relying heavily on trial-and-error experimentation. Nowadays, large language models (LLMs) are capable of tackling chemistry-related problems, such as molecule design and chemical reasoning tasks. Here, we report the design, implementation and application of Chemma-RC, a text-augmented multimodal LLM to identify effective conditions through task-specific dialogue and condition generation. Chemma-RC learns a unified representation of chemical reactions by aligning multiple modalities-including text corpus, reaction SMILES, and reaction graphs-within a shared embedding module. Performance benchmarking on datasets showed high precision in identifying optimal conditions, with up to 17% improvement over the current state-of-the-art methods. A palladium-catalysed imidazole C-H arylation reaction was investigated experimentally to evaluate the functionalities of the Chemma-RC in practice. Our findings suggest that Chemma-RC holds significant potential to accelerate high-throughput condition screening in chemical synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10880v2">Searching for Privacy Risks in LLM Agents via Simulation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The widespread deployment of LLM-based agents is likely to introduce a critical privacy threat: malicious agents that proactively engage others in multi-turn interactions to extract sensitive information. However, the evolving nature of such dynamic dialogues makes it challenging to anticipate emerging vulnerabilities and design effective defenses. To tackle this problem, we present a search-based framework that alternates between improving attack and defense strategies through the simulation of privacy-critical agent interactions. Specifically, we employ LLMs as optimizers to analyze simulation trajectories and iteratively propose new agent instructions. To explore the strategy space more efficiently, we further utilize parallel search with multiple threads and cross-thread propagation. Through this process, we find that attack strategies escalate from direct requests to sophisticated tactics, such as impersonation and consent forgery, while defenses evolve from simple rule-based constraints to robust identity-verification state machines. The discovered attacks and defenses transfer across diverse scenarios and backbone models, demonstrating strong practical utility for building privacy-aware agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20067v2">MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11343v2">From Replication to Redesign: Exploring Pairwise Comparisons for LLM-Based Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) offers unprecedented opportunities to reimagine peer review beyond the constraints of traditional workflows. Despite these opportunities, prior efforts have largely focused on replicating traditional review workflows with LLMs serving as direct substitutes for human reviewers, while limited attention has been given to exploring new paradigms that fundamentally rethink how LLMs can participate in the academic review process. In this paper, we introduce and explore a novel mechanism that employs LLM agents to perform pairwise comparisons among manuscripts instead of individual scoring. By aggregating outcomes from substantial pairwise evaluations, this approach enables a more accurate and robust measure of relative manuscript quality. Our experiments demonstrate that this comparative approach significantly outperforms traditional rating-based methods in identifying high-impact papers. However, our analysis also reveals emergent biases in the selection process, notably a reduced novelty in research topics and an increased institutional imbalance. These findings highlight both the transformative potential of rethinking peer review with LLMs and critical challenges that future systems must address to ensure equity and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14096v3">VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduce VideoPASTA (Preference Alignment with Spatio-Temporal-Cross Frame Adversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully crafted adversarial examples that deliberately violate spatial, temporal, or cross-frame relationships. With only 7,020 preference pairs and Direct Preference Optimization, VideoPASTA enables models to learn robust representations that capture fine-grained spatial details and long-range temporal dynamics. Experiments demonstrate that VideoPASTA is model agnostic and significantly improves performance, for example, achieving gains of up to +3.8 percentage points on LongVideoBench, +4.1 on VideoMME, and +4.0 on MVBench, when applied to various state-of-the-art Video-LLMs. These results demonstrate that targeted alignment, rather than massive pretraining or architectural modifications, effectively addresses core video-language challenges. Notably, VideoPASTA achieves these improvements without any human annotation or captioning, relying solely on 32-frame sampling. This efficiency makes our approach a scalable plug-and-play solution that seamlessly integrates with existing models while preserving their original capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20702v1">Incorporating LLM Embeddings for Variation Across the Human Genome</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) embeddings have enabled powerful representations for biological data, but most applications to date focus only on gene-level information. We present one of the first systematic frameworks to generate variant-level embeddings across the entire human genome. Using curated annotations from FAVOR, ClinVar, and the GWAS Catalog, we constructed semantic text descriptions for 8.9 billion possible variants and generated embeddings at three scales: 1.5 million HapMap3+MEGA variants, ~90 million imputed UK Biobank variants, and ~9 billion all possible variants. Embeddings were produced with both OpenAI's text-embedding-3-large and the open-source Qwen3-Embedding-0.6B models. Baseline experiments demonstrate high predictive accuracy for variant properties, validating the embeddings as structured representations of genomic variation. We outline two downstream applications: embedding-informed hypothesis testing by extending the Frequentist And Bayesian framework to genome-wide association studies, and embedding-augmented genetic risk prediction that enhances standard polygenic risk scores. These resources, publicly available on Hugging Face, provide a foundation for advancing large-scale genomic discovery and precision medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18991v5">Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 The first three authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based (train a reward model on preference pairs and optimize with reinforcement learning) or reward-free (directly fine-tune on ranked outputs). Recent research shows that well-tuned reward-based pipelines remain robust, and single-response demonstrations can outperform pairwise preference data. However, two challenges persist: (1) imbalanced safety datasets that overrepresent common hazards while neglecting long-tail threats; and (2) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains. We propose DR-IRL (Dynamically adjusting Rewards through Inverse Reinforcement Learning). We first train category-specific reward models using a balanced safety dataset covering seven harmful categories via IRL. Then we enhance Group Relative Policy Optimization (GRPO) by introducing dynamic reward scaling--adjusting rewards by task difficulty--data-level hardness by text encoder cosine similarity, model-level responsiveness by reward gaps. Extensive experiments across various benchmarks and LLMs demonstrate that DR-IRL outperforms all baseline methods in safety alignment while maintaining usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18896v2">ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 Accepted by NeurIPS 2025. Project: https://github.com/Gen-Verse/ReasonFlux
    </div>
    <details class="paper-abstract">
      Process Reward Models (PRMs) have recently emerged as a powerful framework for supervising intermediate reasoning steps in large language models (LLMs). Previous PRMs are primarily trained on model final output responses and struggle to evaluate intermediate thinking trajectories robustly, especially in the emerging setting of trajectory-response outputs generated by frontier reasoning models like Deepseek-R1. In this work, we introduce ReasonFlux-PRM, a novel trajectory-aware PRM explicitly designed to evaluate the trajectory-response type of reasoning traces. ReasonFlux-PRM incorporates both step-level and trajectory-level supervision, enabling fine-grained reward assignment aligned with structured chain-of-thought data. We adapt ReasonFlux-PRM to support reward supervision under both offline and online settings, including (i) selecting high-quality model distillation data for downstream supervised fine-tuning of smaller models, (ii) providing dense process-level rewards for policy optimization during reinforcement learning, and (iii) enabling reward-guided Best-of-N test-time scaling. Empirical results on challenging downstream benchmarks such as AIME, MATH500, and GPQA-Diamond demonstrate that ReasonFlux-PRM-7B selects higher quality data than strong PRMs (e.g., Qwen2.5-Math-PRM-72B) and human-curated baselines. Furthermore, our derived ReasonFlux-PRM-7B yields consistent performance improvements, achieving average gains of 12.1% in supervised fine-tuning, 4.5% in reinforcement learning, and 6.3% in test-time scaling. We also release our efficient ReasonFlux-PRM-1.5B for resource-constrained applications and edge deployment. Project: https://github.com/Gen-Verse/ReasonFlux
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20680v1">Can Federated Learning Safeguard Private Data in LLM Training? Vulnerabilities, Attacks, and Defense Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-25
      | 💬 28 pages, 32 figures, accepted to the Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with local data is a widely adopted approach for organizations seeking to adapt LLMs to their specific domains. Given the shared characteristics in data across different organizations, the idea of collaboratively fine-tuning an LLM using data from multiple sources presents an appealing opportunity. However, organizations are often reluctant to share local data, making centralized fine-tuning impractical. Federated learning (FL), a privacy-preserving framework, enables clients to retain local data while sharing only model parameters for collaborative training, offering a potential solution. While fine-tuning LLMs on centralized datasets risks data leakage through next-token prediction, the iterative aggregation process in FL results in a global model that encapsulates generalized knowledge, which some believe protects client privacy. In this paper, however, we present contradictory findings through extensive experiments. We show that attackers can still extract training data from the global model, even using straightforward generation methods, with leakage increasing as the model size grows. Moreover, we introduce an enhanced attack strategy tailored to FL, which tracks global model updates during training to intensify privacy leakage. To mitigate these risks, we evaluate privacy-preserving techniques in FL, including differential privacy, regularization-constrained updates and adopting LLMs with safety alignment. Our results provide valuable insights and practical guidelines for reducing privacy risks when training LLMs with FL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20641v1">Investigating Modality Contribution in Audio LLMs for Music</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      Audio Large Language Models (Audio LLMs) enable human-like conversation about music, yet it is unclear if they are truly listening to the audio or just using textual reasoning, as recent benchmarks suggest. This paper investigates this issue by quantifying the contribution of each modality to a model's output. We adapt the MM-SHAP framework, a performance-agnostic score based on Shapley values that quantifies the relative contribution of each modality to a model's prediction. We evaluate two models on the MuChoMusic benchmark and find that the model with higher accuracy relies more on text to answer questions, but further inspection shows that even if the overall audio contribution is low, models can successfully localize key sound events, suggesting that audio is not entirely ignored. Our study is the first application of MM-SHAP to Audio LLMs and we hope it will serve as a foundational step for future research in explainable AI and audio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20634v1">Recidivism and Peer Influence with LLM Text Embeddings in Low Security Correctional Facilities</a></div>
    <div class="paper-meta">
      📅 2025-09-25
    </div>
    <details class="paper-abstract">
      We find AI embeddings obtained using a pre-trained transformer-based Large Language Model (LLM) of 80,000-120,000 written affirmations and correction exchanges among residents in low-security correctional facilities to be highly predictive of recidivism. The prediction accuracy is 30\% higher with embedding vectors than with only pre-entry covariates. However, since the text embedding vectors are high-dimensional, we perform Zero-Shot classification of these texts to a low-dimensional vector of user-defined classes to aid interpretation while retaining the predictive power. To shed light on the social dynamics inside the correctional facilities, we estimate peer effects in these LLM-generated numerical representations of language with a multivariate peer effect model, adjusting for network endogeneity. We develop new methodology and theory for peer effect estimation that accommodate sparse networks, multivariate latent variables, and correlated multivariate outcomes. With these new methods, we find significant peer effects in language usage for interaction and feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20321v1">DRES: Benchmarking LLMs for Disfluency Removal</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Disfluencies -- such as "um," "uh," interjections, parentheticals, and edited statements -- remain a persistent challenge for speech-driven systems, degrading accuracy in command interpretation, summarization, and conversational agents. We introduce DRES (Disfluency Removal Evaluation Suite), a controlled text-level benchmark that establishes a reproducible semantic upper bound for this task. DRES builds on human-annotated Switchboard transcripts, isolating disfluency removal from ASR errors and acoustic variability. We systematically evaluate proprietary and open-source LLMs across scales, prompting strategies, and architectures. Our results reveal that (i) simple segmentation consistently improves performance, even for long-context models; (ii) reasoning-oriented models tend to over-delete fluent tokens; and (iii) fine-tuning achieves near state-of-the-art precision and recall but harms generalization abilities. We further present a set of LLM-specific error modes and offer nine practical recommendations (R1-R9) for deploying disfluency removal in speech-driven pipelines. DRES provides a reproducible, model-agnostic foundation for advancing robust spoken-language systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09873v2">LLM-Powered AI Tutors with Personas for d/Deaf and Hard-of-Hearing Online Learners</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems (ITS) using artificial intelligence (AI) technology have shown promise in supporting learners with diverse abilities. Large language models (LLMs) provide new opportunities to incorporate personas to AI-based tutors and support dynamic interactive dialogue. This paper explores how DHH learners interact with LLM-powered AI tutors with different experiences in DHH education as personas to identify their accessibility preferences. A user study with 16 DHH participants showed that they asked DHH-related questions based on background information and evaluated the AI tutors' cultural knowledge of the DHH communities in their responses. Participants suggested providing more transparency in each AI tutor's position within the DHH community. Participants also pointed out the lack of support in the multimodality of sign language in current LLMs. We discuss design implications to support the diverse needs in interaction between DHH users and the LLMs, such as offering supports in tuning language styles of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20293v1">When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We release our code at https://anonymous.4open.science/r/judgment-to-noise-947D/README.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20278v1">Instruction Boundary: Quantifying Biases in LLM Reasoning under Various Coverage</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM) reasoning has long been regarded as a powerful tool for problem solving across domains, providing non-experts with valuable advice. However, their limitations - especially those stemming from prompt design - remain underexplored. Because users may supply biased or incomplete prompts - often unintentionally - LLMs can be misled, undermining reliability and creating risks. We refer to this vulnerability as the Instruction Boundary. To investigate the phenomenon, we distill it into eight concrete facets and introduce BiasDetector, a framework that measures biases arising from three instruction types: complete, redundant, and insufficient. We evaluate several mainstream LLMs and find that, despite high headline accuracy, substantial biases persist in many downstream tasks as a direct consequence of prompt coverage. Our empirical study confirms that LLM reasoning reliability can still be significantly improved. We analyze the practical impact of these biases and outline mitigation strategies. Our findings underscore the need for developers to tackle biases and for users to craft options carefully.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14264v2">AAPO: Enhancing the Reasoning Capabilities of LLMs with Advantage Momentum</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as an effective approach for enhancing the reasoning capabilities of large language models (LLMs), especially in scenarios where supervised fine-tuning (SFT) falls short due to limited chain-of-thought (CoT) data. Among RL-based post-training methods, group relative advantage estimation, as exemplified by Group Relative Policy Optimization (GRPO), has attracted considerable attention for eliminating the dependency on the value model, thereby simplifying training compared to traditional approaches like Proximal Policy Optimization (PPO). However, we observe that exsiting group relative advantage estimation method still suffers from training inefficiencies, particularly when the estimated advantage approaches zero. To address this limitation, we propose Advantage-Augmented Policy Optimization (AAPO), a novel RL algorithm that optimizes the cross-entropy (CE) loss using advantages enhanced through a momentum-based estimation scheme. This approach effectively mitigates the inefficiencies associated with group relative advantage estimation. Experimental results on multiple mathematical reasoning benchmarks demonstrate the superior performance of AAPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03547v2">Guided Reality: Generating Visually-Enriched AR Task Guidance with LLMs and Vision Models</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 To appear at UIST 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the automatic generation of step-by-step augmented reality (AR) instructions for a wide range of physical tasks. However, existing LLM-based AR guidance often lacks rich visual augmentations to effectively embed instructions into spatial context for a better user understanding. We present Guided Reality, a fully automated AR system that generates embedded and dynamic visual guidance based on step-by-step instructions. Our system integrates LLMs and vision models to: 1) generate multi-step instructions from user queries, 2) identify appropriate types of visual guidance, 3) extract spatial information about key interaction points in the real world, and 4) embed visual guidance in physical space to support task execution. Drawing from a corpus of user manuals, we define five categories of visual guidance and propose an identification strategy based on the current step. We evaluate the system through a user study (N=16), completing real-world tasks and exploring the system in the wild. Additionally, four instructors shared insights on how Guided Reality could be integrated into their training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18920v2">Context-Masked Meta-Prompting for Privacy-Preserving LLM Adaptation in Finance</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      The increasing reliance on Large Language Models (LLMs) in sensitive domains like finance necessitates robust methods for privacy preservation and regulatory compliance. This paper presents an iterative meta-prompting methodology designed to optimise hard prompts without exposing proprietary or confidential context to the LLM. Through a novel regeneration process involving feeder and propagation methods, we demonstrate significant improvements in prompt efficacy. Evaluated on public datasets serving as proxies for financial tasks such as SQuAD for extractive financial Q&A, CNN/DailyMail for news summarisation, and SAMSum for client interaction summarisation, our approach, utilising GPT-3.5 Turbo, achieved a 103.87% improvement in ROUGE-L F1 for question answering. This work highlights a practical, low-cost strategy for adapting LLMs to financial applications while upholding critical privacy and auditability standards, offering a compelling case for its relevance in the evolving landscape of generative AI in finance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20230v1">Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20214v1">Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We study weight-only post-training quantization (PTQ), which quantizes the weights of a large language model (LLM) without retraining, using little or no calibration data. Weight-only PTQ is crucial for reducing the memory footprint and latency of LLM inference, especially in memory-bound, small-batch inference scenarios, such as personalized inference on edge devices. Despite its importance, irregular weight distributions with heavy-tailed outliers in LLMs complicate quantization, recently motivating rotation-based methods that transform weights into near-Gaussian distributions, which are more regular with fewer outliers, thereby reducing quantization error. In this work, we first derive the information-theoretically optimal bit allocation for Gaussianized weights under given bit budgets, revealing that fine-grained fractional-bit quantizers approaching the Gaussian distortion-rate bound are essential to achieve near-optimal quantization performance. To bridge this theoretical insight and practical implementation, we introduce Q-Palette, a versatile collection of fractional-bit quantizers that range from trellis-coded quantizers offering near-optimal distortion to simpler vector and scalar quantizers optimized for faster inference, all efficiently implemented with optimized CUDA kernels across various bitwidths. Furthermore, leveraging Q-Palette as a foundational component, we propose a novel mixed-scheme quantization framework, jointly optimizing quantizer choices and layer fusion decisions given resource constraints. The code is available at https://github.com/snu-mllab/Q-Palette.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20208v1">Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Integrating LLM powered operators in declarative query languages allows for the combination of cheap and interpretable functions with powerful, generalizable language model reasoning. However, in order to benefit from the optimized execution of a database query language like SQL, generated outputs must align with the rules enforced by both type checkers and database contents. Current approaches address this challenge with orchestrations consisting of many LLM-based post-processing calls to ensure alignment between generated outputs and database values, introducing performance bottlenecks. We perform a study on the ability of various sized open-source language models to both parse and execute functions within a query language based on SQL, showing that small language models can excel as function executors over hybrid data sources. Then, we propose an efficient solution to enforce the well-typedness of LLM functions, demonstrating 7% accuracy improvement on a multi-hop question answering dataset with 53% improvement in latency over comparable solutions. We make our implementation available at https://github.com/parkervg/blendsql
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20616v1">Training Task Reasoning LLM Agents for Multi-turn Task Planning via Single-turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in knowledge acquisition, reasoning, and tool use, making them promising candidates for autonomous agent applications. However, training LLM agents for complex multi-turn task planning faces significant challenges, including sparse episode-wise rewards, credit assignment across long horizons, and the computational overhead of reinforcement learning in multi-turn interaction settings. To this end, this paper introduces a novel approach that transforms multi-turn task planning into single-turn task reasoning problems, enabling efficient policy optimization through Group Relative Policy Optimization (GRPO) with dense and verifiable reward from expert trajectories. Our theoretical analysis shows that GRPO improvement on single-turn task reasoning results in higher multi-turn success probability under the minimal turns, as well as the generalization to subtasks with shorter horizons. Experimental evaluation on the complex task planning benchmark demonstrates that our 1.5B parameter model trained with single-turn GRPO achieves superior performance compared to larger baseline models up to 14B parameters, with success rates of 70% for long-horizon planning tasks with over 30 steps. We also theoretically and empirically validate the strong cross-task generalizability that the models trained on complex tasks can lead to the successful completion of all simpler subtasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20600v1">An LLM-based Agentic Framework for Accessible Network Control</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Traditional approaches to network management have been accessible only to a handful of highly-trained network operators with significant expert knowledge. This creates barriers for lay users to easily manage their networks without resorting to experts. With recent development of powerful large language models (LLMs) for language comprehension, we design a system to make network management accessible to a broader audience of non-experts by allowing users to converse with networks in natural language. To effectively leverage advancements in LLMs, we propose an agentic framework that uses an intermediate representation to streamline configuration across diverse vendor equipment, retrieves the network state from memory in real-time, and provides an interface for external feedback. We also conduct pilot studies to collect real user data of natural language utterances for network control, and present a visualization interface to facilitate dialogue-driven user interaction and enable large-scale data collection for future development. Preliminary experiments validate the effectiveness of our proposed system components with LLM integration on both synthetic and real user utterances. Through our data collection and visualization efforts, we pave the way for more effective use of LLMs and democratize network control for everyday users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01630v2">TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted at ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Temporal reasoning in multi-session dialogues presents a significant challenge which has been under-studied in previous temporal reasoning benchmarks. To bridge this gap, we propose a new evaluation task for temporal reasoning in multi-session dialogues and introduce an approach to construct a new benchmark by augmenting dialogues from LoCoMo and creating multi-choice QAs. Furthermore, we present TReMu, a new framework aimed at enhancing the temporal reasoning capabilities of LLM-agents in this context. Specifically, the framework employs time-aware memorization through timeline summarization, generating retrievable memory by summarizing events in each dialogue session with their inferred dates. Additionally, we integrate neuro-symbolic temporal reasoning, where LLMs generate Python code to perform temporal calculations and select answers. Experimental evaluations on popular LLMs demonstrate that our benchmark is challenging, and the proposed framework significantly improves temporal reasoning performance compared to baseline methods, raising from 29.83 on GPT-4o via standard prompting to 77.67 via our approach and highlighting its effectiveness in addressing temporal reasoning in multi-session dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19947v2">MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 NeurIPS 2025. Code: https://github.com/laminair/mess-plus
    </div>
    <details class="paper-abstract">
      Open-weight large language model (LLM) zoos provide access to numerous high-quality models, but selecting the appropriate model for specific tasks remains challenging and requires technical expertise. Most users simply want factually correct, safe, and satisfying responses without concerning themselves with model technicalities, while inference service providers prioritize minimizing operating costs. These competing interests are typically mediated through service level agreements (SLAs) that guarantee minimum service quality. We introduce MESS+, a stochastic optimization algorithm for cost-optimal LLM request routing while providing rigorous SLA compliance guarantees. MESS+ learns request satisfaction probabilities of LLMs in real-time as users interact with the system, based on which model selection decisions are made by solving a per-request optimization problem. Our algorithm includes a novel combination of virtual queues and request satisfaction prediction, along with a theoretical analysis of cost optimality and constraint satisfaction. Across a wide range of state-of-the-art LLM benchmarks, MESS+ achieves an average of $2\times$ cost savings compared to existing LLM routing techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20552v1">Enhancing LLM-based Fault Localization with a Functionality-Aware Retrieval-Augmented Generation Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Fault localization (FL) is a critical but time-consuming task in software debugging, aiming to identify faulty code elements. While recent advances in large language models (LLMs) have shown promise for FL, they often struggle with complex systems due to the lack of project-specific knowledge and the difficulty of navigating large projects. To address these limitations, we propose FaR-Loc, a novel framework that enhances method-level FL by integrating LLMs with retrieval-augmented generation (RAG). FaR-Loc consists of three key components: LLM Functionality Extraction, Semantic Dense Retrieval, and LLM Re-ranking. First, given a failed test and its associated stack trace, the LLM Functionality Extraction module generates a concise natural language description that captures the failing behavior. Next, the Semantic Dense Retrieval component leverages a pre-trained code-understanding encoder to embed both the functionality description (natural language) and the covered methods (code) into a shared semantic space, enabling the retrieval of methods with similar functional behavior. Finally, the LLM Re-ranking module reorders the retrieved methods based on their contextual relevance. Our experiments on the widely used Defects4J benchmark show that FaR-Loc outperforms state-of-the-art LLM-based baselines SoapFL and AutoFL, by 14.6% and 9.1% in Top-1 accuracy, by 19.2% and 22.1% in Top-5 accuracy, respectively. It also surpasses all learning-based and spectrum-based baselines across all Top-N metrics without requiring re-training. Furthermore, we find that pre-trained code embedding models that incorporate code structure, such as UniXcoder, can significantly improve fault localization performance by up to 49.0% in Top-1 accuracy. Finally, we conduct a case study to illustrate the effectiveness of FaR-Loc and to provide insights for its practical application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20502v1">MARS: toward more efficient multi-agent collaboration for LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive results in natural language understanding, yet their reasoning capabilities remain limited when operating as single agents. Multi-Agent Debate (MAD) has been proposed to address this limitation by enabling collaborative reasoning among multiple models in a round-table debate manner. While effective, MAD introduces substantial computational overhead due to the number of agents involved and the frequent communication required. In this paper, we propose MARS (Multi-Agent Review System), a role-based collaboration framework inspired by the review process. In MARS, an author agent generates an initial solution, reviewer agents provide decisions and comments independently, and a meta-reviewer integrates the feedback to make the final decision and guide further revision. This design enhances reasoning quality while avoiding costly reviewer-to-reviewer interactions, thereby controlling token consumption and inference time. We compared MARS with both MAD and other state-of-the-art reasoning strategies across multiple benchmarks. Extensive experiments with different LLMs show that MARS matches the accuracy of MAD while reducing both token usage and inference time by approximately 50\%. Code is available at https://github.com/xwang97/MARS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20497v1">PromptDebt: A Comprehensive Study of Technical Debt Across LLM Projects</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted at Proceedings of the 2025 Evaluation and Assessment in Software Engineering (EASE '25)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in software via APIs like OpenAI, offering powerful AI features without heavy infrastructure. Yet these integrations bring their own form of self-admitted technical debt (SATD). In this paper, we present the first large-scale empirical study of LLM-specific SATD: its origins, prevalence, and mitigation strategies. By analyzing 93,142 Python files across major LLM APIs, we found that 54.49% of SATD instances stem from OpenAI integrations and 12.35% from LangChain use. Prompt design emerged as the primary source of LLM-specific SATD, with 6.61% of debt related to prompt configuration and optimization issues, followed by hyperparameter tuning and LLM-framework integration. We further explored which prompt techniques attract the most debt, revealing that instruction-based prompts (38.60%) and few-shot prompts (18.13%) are particularly vulnerable due to their dependence on instruction clarity and example quality. Finally, we release a comprehensive SATD dataset to support reproducibility and offer practical guidance for managing technical debt in LLM-powered systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17054v2">TactfulToM: Do LLMs Have the Theory of Mind Ability to Understand White Lies?</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      While recent studies explore Large Language Models' (LLMs) performance on Theory of Mind (ToM) reasoning tasks, research on ToM abilities that require more nuanced social context is limited, such as white lies. We introduce TactfulToM, a novel English benchmark designed to evaluate LLMs' ability to understand white lies within real-life conversations and reason about prosocial motivations behind them, particularly when they are used to spare others' feelings and maintain social harmony. Our benchmark is generated through a multi-stage human-in-the-loop pipeline where LLMs expand manually designed seed stories into conversations to maintain the information asymmetry between participants necessary for authentic white lies. We show that TactfulToM is challenging for state-of-the-art models, which perform substantially below humans, revealing shortcomings in their ability to fully comprehend the ToM reasoning that enables true understanding of white lies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05926v3">LLMs Reproduce Stereotypes of Sexual and Gender Minorities</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 13 pages, 5 figures, 9 tables (including bibliography and appendix). Accepted to Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      A large body of research has found substantial gender bias in NLP systems. Most of this research takes a binary, essentialist view of gender: limiting its variation to the categories _men_ and _women_, conflating gender with sex, and ignoring different sexual identities. But gender and sexuality exist on a spectrum, so in this paper we study the biases of large language models (LLMs) towards sexual and gender minorities beyond binary categories. Grounding our study in a widely used social psychology model -- the Stereotype Content Model -- we demonstrate that English-language survey questions about social perceptions elicit more negative stereotypes of sexual and gender minorities from both humans and LLMs. We then extend this framework to a more realistic use case: text generation. Our analysis shows that LLMs generate stereotyped representations of sexual and gender minorities in this setting, showing that they amplify representational harms in creative writing, a widely advertised use for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20190v1">STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 18 pages, 2 figures, accepted for 23rd escar Europe (Nov 05-06, 2025, Frankfurt, Germany)
    </div>
    <details class="paper-abstract">
      In modern automotive development, security testing is critical for safeguarding systems against increasingly advanced threats. Attack trees are widely used to systematically represent potential attack vectors, but generating comprehensive test cases from these trees remains a labor-intensive, error-prone task that has seen limited automation in the context of testing vehicular systems. This paper introduces STAF (Security Test Automation Framework), a novel approach to automating security test case generation. Leveraging Large Language Models (LLMs) and a four-step self-corrective Retrieval-Augmented Generation (RAG) framework, STAF automates the generation of executable security test cases from attack trees, providing an end-to-end solution that encompasses the entire attack surface. We particularly show the elements and processes needed to provide an LLM to actually produce sensible and executable automotive security test suites, along with the integration with an automated testing framework. We further compare our tailored approach with general purpose (vanilla) LLMs and the performance of different LLMs (namely GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our operation step-by-step in a concrete case study. Our results show significant improvements in efficiency, accuracy, scalability, and easy integration in any workflow, marking a substantial advancement in automating automotive security testing methodologies. Using TARAs as an input for verfication tests, we create synergies by connecting two vital elements of a secure automotive development process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20168v1">Probing Gender Bias in Multilingual LLMs: A Case Study of Stereotypes in Persian</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted and forthcoming at the Widening Natural Language Processing Workshop (WiNLP 2025) at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) are increasingly used worldwide, making it essential to ensure they are free from gender bias to prevent representational harm. While prior studies have examined such biases in high-resource languages, low-resource languages remain understudied. In this paper, we propose a template-based probing methodology, validated against real-world data, to uncover gender stereotypes in LLMs. As part of this framework, we introduce the Domain-Specific Gender Skew Index (DS-GSI), a metric that quantifies deviations from gender parity. We evaluate four prominent models, GPT-4o mini, DeepSeek R1, Gemini 2.0 Flash, and Qwen QwQ 32B, across four semantic domains, focusing on Persian, a low-resource language with distinct linguistic features. Our results show that all models exhibit gender stereotypes, with greater disparities in Persian than in English across all domains. Among these, sports reflect the most rigid gender biases. This study underscores the need for inclusive NLP practices and provides a framework for assessing bias in other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20166v1">CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20105v1">PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20097v1">Integrated Framework for LLM Evaluation with Answer Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 16pages
    </div>
    <details class="paper-abstract">
      Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20088v1">Causal Understanding by LLMs: The Role of Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted in second UncertaiNLP workshop at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent papers show LLMs achieve near-random accuracy in causal relation classification, raising questions about whether such failures arise from limited pretraining exposure or deeper representational gaps. We investigate this under uncertainty-based evaluation, testing whether pretraining exposure to causal examples improves causal understanding >18K PubMed sentences -- half from The Pile corpus, half post-2024 -- across seven models (Pythia-1.4B/7B/12B, GPT-J-6B, Dolly-7B/12B, Qwen-7B). We analyze model behavior through: (i) causal classification, where the model identifies causal relationships in text, and (ii) verbatim memorization probing, where we assess whether the model prefers previously seen causal statements over their paraphrases. Models perform four-way classification (direct/conditional/correlational/no-relationship) and select between originals and their generated paraphrases. Results show almost identical accuracy on seen/unseen sentences (p > 0.05), no memorization bias (24.8% original selection), and output distribution over the possible options is almost flat, with entropic values near the maximum (1.35/1.39), confirming random guessing. Instruction-tuned models show severe miscalibration (Qwen: > 95% confidence, 32.8% accuracy, ECE=0.49). Conditional relations induce highest entropy (+11% vs. direct). These findings suggest that failures in causal understanding arise from the lack of structured causal representation, rather than insufficient exposure to causal examples during pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12044v2">Why Do Some Inputs Break Low-Bit LLM Quantization?</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Low-bit weight-only quantization significantly reduces the memory footprint of large language models (LLMs), but disproportionately affects certain examples. We analyze diverse 3-4 bit methods on LLMs ranging from 7B-70B in size and find that the quantization errors of 50 pairs of methods are strongly correlated (avg. 0.82) on FineWeb examples. Moreover, the residual stream magnitudes of full-precision models are indicative of future quantization errors. We further establish a hypothesis that relates the residual stream magnitudes to error amplification and accumulation over layers. Using LLM localization techniques, early exiting, and activation patching, we show that examples with large errors rely on precise residual activations in the late layers, and that the outputs of MLP gates play a crucial role in maintaining the perplexity. Our work reveals why certain examples result in large quantization errors and which model components are most critical for performance preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20070v1">LLM Trainer: Automated Robotic Data Generating via Demonstration Augmentation using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 9 pages, 5 figures, 4 tables. Submitted to ICRA 2026
    </div>
    <details class="paper-abstract">
      We present LLM Trainer, a fully automated pipeline that leverages the world knowledge of Large Language Models (LLMs) to transform a small number of human demonstrations (as few as one) into a large robot dataset for imitation learning. Our approach decomposes demonstration generation into two steps: (1) offline demonstration annotation that extracts keyframes, salient objects, and pose-object relations; and (2) online keypose retargeting that adapts those keyframes to a new scene, given an initial observation. Using these modified keypoints, our system warps the original demonstration to generate a new trajectory, which is then executed, and the resulting demo, if successful, is saved. Because the annotation is reusable across scenes, we use Thompson sampling to optimize the annotation, significantly improving generation success rate. We evaluate our method on a range of tasks, and find that our data annotation method consistently outperforms expert-engineered baselines. We further show an ensemble policy that combines the optimized LLM feed-forward plan with a learned feedback imitation learning controller. Finally, we demonstrate hardware feasibility on a Franka Emika Panda robot. For additional materials and demonstration videos, please see the project website: https://sites.google.com/andrew.cmu.edu/llm-trainer
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14359v4">Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted at EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      We examine three evaluation paradigms: standard benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate for the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21625v5">Meeseeks: A Feedback-Driven, Iterative Self-Correction Benchmark evaluating LLMs' Instruction Following Capability</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      The capability to precisely adhere to instructions is a cornerstone for Large Language Models (LLMs) to function as dependable agents in real-world scenarios. However, confronted with complex prompts, LLMs frequently encounter difficulties in fulfilling all specified requirements within a single response. Drawing inspiration from recent advancements in Chain-of-Thought (CoT) prompting and self-correction methodologies, we introduce Meeseeks (The name is inspired by Mr. Meeseeks from "Rick and Morty," a character renowned for efficiently accomplishing assigned tasks. See: https://en.wikipedia.org/wiki/Mr._Meeseeks), a fully automated iterative instruction-following benchmark equipped with an integrated feedback mechanism. Meeseeks identifies erroneous components in model responses and provides corresponding feedback accurately, thereby iteratively guiding the model toward self-correction. The dataset contains over 700 curated instances annotated by 32 distinct capability tags in Chinese and English. Extensive experimental results reveal that different state-of-the-art commercial and open-source LLMs exhibit vastly disparate performance, and even after 20 turns of iterative feedback-driven self-correction, nearly all models demonstrate suboptimal performance. We conducted comprehensive analysis from both macro and instance levels, uncovering numerous common issues prevalent in current state-of-the-art models, as well as several counterintuitive phenomena. We've open-sourced our work on https://github.com/ADoublLEN/Meeseeks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20021v1">Embodied AI: From LLMs to World Models</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted by IEEE CASM
    </div>
    <details class="paper-abstract">
      Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09505v2">Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      LLMs now form the backbone of AI agents for a diverse array of applications, including tool use, command-line agents, and web or computer use agents. These agentic LLM inference tasks are fundamentally different from chatbot-focused inference -- they often have much larger context lengths to capture complex, prolonged inputs, such as entire webpage DOMs or complicated tool call trajectories. This, in turn, generates significant off-chip memory traffic for the underlying hardware at the inference stage and causes the workload to be constrained by two memory walls, namely the bandwidth and capacity memory walls, preventing the on-chip compute units from achieving high utilization. In this paper, we introduce PLENA, a hardware-software co-designed system that applies three core optimization pathways to tackle these challenges. PLENA includes an efficient hardware implementation of compute and memory units supporting an asymmetric quantization scheme. PLENA also features a novel flattened systolic array architecture that has native support for FlashAttention to tackle these memory walls in the scenario of inference serving for long-context LLMs. Additionally, PLENA is developed with a complete stack, including a custom ISA, a compiler, a cycle-emulated simulator, and an automated design space exploration flow. The simulated results show that PLENA achieves up to 8.5x higher utilization than existing accelerators, and delivers 2.24x higher throughput than the A100 GPU and 3.85x higher throughput than the TPU v6e, under the same multiplier count and memory settings. The full PLENA system will also be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20004v1">The Knowledge-Behaviour Disconnect in LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large language model-based artificial conversational agents (like ChatGPT) give answers to all kinds of questions, and often enough these answers are correct. Just on the basis of that capacity alone, we may attribute knowledge to them. But do these models use this knowledge as a basis for their own conversational behaviour? I argue this is not the case, and I will refer to this failure as a `disconnect'. I further argue this disconnect is fundamental in the sense that with more data and more training of the LLM on which a conversational chatbot is based, it will not disappear. The reason is, as I will claim, that the core technique used to train LLMs does not allow for the establishment of the connection we are after. The disconnect reflects a fundamental limitation on the capacities of LLMs, and explains the source of hallucinations. I will furthermore consider the ethical version of the disconnect (ethical conversational knowledge not being aligned with ethical conversational behaviour), since in this domain researchers have come up with several additional techniques to influence a chatbot's behaviour. I will discuss how these techniques do nothing to solve the disconnect and can make it worse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11856v2">LLMs as a synthesis between symbolic and distributed approaches to language</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Final version to appear in Findings of the ACL (significantly revised wrt v1). 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Since the middle of the 20th century, a fierce battle is being fought between symbolic and distributed approaches to language and cognition. The success of deep learning models, and LLMs in particular, has been alternatively taken as showing that the distributed camp has won, or dismissed as an irrelevant engineering development. In this position paper, I argue that deep learning models for language actually represent a synthesis between the two traditions. This is because 1) deep learning architectures allow for both distributed/continuous/fuzzy and symbolic/discrete/categorical-like representations and processing; 2) models trained on language make use of this flexibility. In particular, I review recent research in interpretability that showcases how a substantial part of morphosyntactic knowledge is encoded in a near-discrete fashion in LLMs. This line of research suggests that different behaviors arise in an emergent fashion, and models flexibly alternate between the two modes (and everything in between) as needed. This is possibly one of the main reasons for their wild success; and it makes them particularly interesting for the study of language. Is it time for peace?
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19925v1">CON-QA: Privacy-Preserving QA using cloud LLMs in Contract Domain</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      As enterprises increasingly integrate cloud-based large language models (LLMs) such as ChatGPT and Gemini into their legal document workflows, protecting sensitive contractual information - including Personally Identifiable Information (PII) and commercially sensitive clauses - has emerged as a critical challenge. In this work, we propose CON-QA, a hybrid privacy-preserving framework designed specifically for secure question answering over enterprise contracts, effectively combining local and cloud-hosted LLMs. The CON-QA framework operates through three stages: (i) semantic query decomposition and query-aware document chunk retrieval using a locally deployed LLM analysis, (ii) anonymization of detected sensitive entities via a structured one-to-many mapping scheme, ensuring semantic coherence while preventing cross-session entity inference attacks, and (iii) anonymized response generation by a cloud-based LLM, with accurate reconstruction of the original answer locally using a session-consistent many-to-one reverse mapping. To rigorously evaluate CON-QA, we introduce CUAD-QA, a corpus of 85k question-answer pairs generated over 510 real-world CUAD contract documents, encompassing simple, complex, and summarization-style queries. Empirical evaluations, complemented by detailed human assessments, confirm that CON-QA effectively maintains both privacy and utility, preserves answer quality, maintains fidelity to legal clause semantics, and significantly mitigates privacy risks, demonstrating its practical suitability for secure, enterprise-level contract documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12621v2">SAFE: Improving LLM Systems using Sentence-Level In-generation Attribution</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 30 pages (9 pages of content, 5 pages of references, 16 pages of supplementary material), 7 figures, 13 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied in various science domains, yet their broader adoption remains constrained by a critical challenge: the lack of trustworthy, verifiable outputs. Current LLMs often generate answers without reliable source attribution, or worse, with incorrect attributions, posing a barrier to their use in scientific and high-stakes settings, where traceability and accountability are paramount. To be reliable, attribution systems require high accuracy for short-length attribution on retrieved data, i.e., attribution to a sentence within a document rather than the entire document. We propose SAFE, a Sentence-level A ttribution FramEwork for Retrieve-Augmented Generation (RAG) systems that attributes generated sentences during generation. This allows users to verify sentences as they read them and correct the model when the attribution indicates the generated text is not grounded in the documents, increasing the safety of LLM systems. This framework consists of two steps: predicting the required number of references for a sentence, and attributing the sentence. Our approach achieved 95% accuracy in the first step, which translated to 2.1\~6.0% improvements in the accuracy (normalized for maximum possible accuracy) of all attribution algorithms in our clean dataset, when compared to their top-1 accuracy. We also applied SAFE in real-world scenarios with documents containing hundreds to thousands of sentences. In these settings, SAFE reliably attributed sentences to their source documents, demonstrating that the method generalizes beyond controlled benchmarks. The SAFE framework and the training dataset are publicly available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19902v1">WEST: LLM based Speech Toolkit for Speech Understanding, Generation, and Interaction</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      In this paper, we present WEST(WE Speech Toolkit), a speech toolkit based on a large language model (LLM) for speech understanding, generation, and interaction. There are three key features of WEST: 1) Fully LLM-based: Standing on the shoulders of giants by reusing mature architectures, ecosystems (e.g., Hugging Face), and methods (e.g., sequence packing) from large models. 2) Full-stack: Supports tasks such as recognition, synthesis, understanding, dialogue, and multimodal capabilities, with extensibility to incorporate open-source models. 3) Simple and Stupid: A simple and stupid speech toolkit that everyone can Touch. In addition, WEST provides two types of recipes, models, and experimental results. The first is entirely based on open-source models and open-source data, allowing users to fully reproduce the experiments in this paper and serving as a verification system or minimal system baseline. The second is trained on massive data, offering superior performance so the user can directly apply it out of the box. WEST is publicly avilable at https://github.com/wenet-e2e/west/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19890v1">DSA, AIA, and LLMs: Approaches to conceptualizing and auditing moderation in LLM-based chatbots across languages and interfaces in the electoral contexts</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into chatbot-like search engines poses new challenges for governing, assessing, and scrutinizing the content output by these online entities, especially in light of the Digital Service Act (DSA). In what follows, we first survey the regulation landscape in which we can situate LLM-based chatbots and the notion of moderation. Second, we outline the methodological approaches to our study: a mixed-methods audit across chatbots, languages, and elections. We investigated Copilot, ChatGPT, and Gemini across ten languages in the context of the 2024 European Parliamentary Election and the 2024 US Presidential Election. Despite the uncertainty in regulatory frameworks, we propose a set of solutions on how to situate, study, and evaluate chatbot moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19880v1">Do Before You Judge: Self-Reference as a Pathway to Better LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted as a long findings paper at EMNLP 2025
    </div>
    <details class="paper-abstract">
      LLM-as-Judge frameworks are increasingly popular for AI evaluation, yet research findings on the relationship between models' generation and judgment abilities remain inconsistent. We investigate this relationship through systematic dataset- and instance-level analyses across 11 models and 21 diverse tasks. Despite both capabilities relying on the same underlying knowledge, our analyses reveal they are only weakly correlated, primarily due to LLMs' sensitivity to the responses being judged. To address this, we propose a self-reference-guided evaluation strategy that leverages a model's own answers as references. This approach significantly strengthens the correlation between generation and judgment abilities, offering a practical path to align these skills and providing a reliable proxy for model selection in evaluation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20412v1">Structuring Collective Action with LLM-Guided Evolution: From Ill-Structured Problems to Executable Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Collective action problems, which require aligning individual incentives with collective goals, are classic examples of Ill-Structured Problems (ISPs). For an individual agent, the causal links between local actions and global outcomes are unclear, stakeholder objectives often conflict, and no single, clear algorithm can bridge micro-level choices with macro-level welfare. We present ECHO-MIMIC, a computational framework that converts this global complexity into a tractable, Well-Structured Problem (WSP) for each agent by discovering compact, executable heuristics and persuasive rationales. The framework operates in two stages: ECHO (Evolutionary Crafting of Heuristics from Outcomes) evolves snippets of Python code that encode candidate behavioral policies, while MIMIC (Mechanism Inference & Messaging for Individual-to-Collective Alignment) evolves companion natural language messages that motivate agents to adopt those policies. Both phases employ a large-language-model-driven evolutionary search: the LLM proposes diverse and context-aware code or text variants, while population-level selection retains those that maximize collective performance in a simulated environment. We demonstrate this framework on a canonical ISP in agricultural landscape management, where local farming decisions impact global ecological connectivity. Results show that ECHO-MIMIC discovers high-performing heuristics compared to baselines and crafts tailored messages that successfully align simulated farmer behavior with landscape-level ecological goals. By coupling algorithmic rule discovery with tailored communication, ECHO-MIMIC transforms the cognitive burden of collective action into a simple set of agent-level instructions, making previously ill-structured problems solvable in practice and opening a new path toward scalable, adaptive policy design.
    </details>
</div>
