# llm - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21240v3">Tree Search for LLM Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 ICLR 2026, Code: https://github.com/AMAP-ML/Tree-GRPO
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15662v3">Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Human beings solve complex problems through critical thinking, where reasoning and evaluation are intertwined to converge toward correct solutions. However, most existing large language models (LLMs) treat the reasoning and verification as separate processes: they either generate reasoning without explicit self-checking or rely on external verifiers to detect errors post hoc. The former lacks immediate feedback, while the latter increases system complexity and hinders synchronized learning. Motivated by human critical thinking, we propose Stepwise Think-Critique (STC), a unified and end-to-end trainable framework that interleaves reasoning and self-critique at every intermediate step within a single model. STC is trained with a hybrid reinforcement learning objective that integrates reasoning rewards and critique-consistency rewards, thereby jointly optimizing solution correctness and reliability of self-evaluation. Experiments on mathematical reasoning benchmarks show that STC demonstrates strong critical-thinking capabilities and produces more interpretable reasoning traces, representing a step toward LLMs with built-in critical thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17512v1">Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Submitted to ACL 2026. The code is available at https://github.com/ictnlp/XBridge
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strong general intelligence, yet their multilingual performance remains highly imbalanced. Although LLMs encode substantial cross-lingual knowledge in a unified semantic space, they often struggle to reliably interface this knowledge with low-resource or unseen languages. Fortunately, pretrained encoder-decoder translation models already possess balanced multilingual capability, suggesting a natural complement to LLMs. In this work, we propose XBridge, a compositional encoder-LLM-decoder architecture that offloads multilingual understanding and generation to external pretrained translation models, while preserving the LLM as an English-centric core for general knowledge processing. To address the resulting representation misalignment across models, we introduce lightweight cross-model mapping layers and an optimal transport-based alignment objective, enabling fine-grained semantic consistency for multilingual generation. Experiments on four LLMs across multilingual understanding, reasoning, summarization, and generation indicate that XBridge outperforms strong baselines, especially on low-resource and previously unseen languages, without retraining the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17468v1">Efficient Soft Actor-Critic with LLM-Based Action-Level Guidance for Continuous Control</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      We present GuidedSAC, a novel reinforcement learning (RL) algorithm that facilitates efficient exploration in vast state-action spaces. GuidedSAC leverages large language models (LLMs) as intelligent supervisors that provide action-level guidance for the Soft Actor-Critic (SAC) algorithm. The LLM-based supervisor analyzes the most recent trajectory using state information and visual replays, offering action-level interventions that enable targeted exploration. Furthermore, we provide a theoretical analysis of GuidedSAC, proving that it preserves the convergence guarantees of SAC while improving convergence speed. Through experiments in both discrete and continuous control environments, including toy text tasks and complex MuJoCo benchmarks, we demonstrate that GuidedSAC consistently outperforms standard SAC and state-of-the-art exploration-enhanced variants (e.g., RND, ICM, and E3B) in terms of sample efficiency and final performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13417v2">Assessing LLM Reasoning Through Implicit Causal Chain Discovery in Climate Discourse</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 LREC 2026, appendix to be found in ACL anthology
    </div>
    <details class="paper-abstract">
      How does a cause lead to an effect, and which intermediate causal steps explain their connection? This work scrutinizes the mechanistic causal reasoning capabilities of large language models (LLMs) to answer these questions through the task of implicit causal chain discovery. In a diagnostic evaluation framework, we instruct nine LLMs to generate all possible intermediate causal steps linking given cause-effect pairs in causal chain structures. These pairs are drawn from recent resources in argumentation studies featuring polarized discussion on climate change. Our analysis reveals that LLMs vary in the number and granularity of causal steps they produce. Although they are generally self-consistent and confident about the intermediate causal connections in the generated chains, their judgments are mainly driven by associative pattern matching rather than genuine causal reasoning. Nonetheless, human evaluations confirmed the logical coherence and integrity of the generated chains. Our baseline causal chain discovery approach, insights from our diagnostic evaluation, and benchmark dataset with causal chains lay a solid foundation for advancing future work in implicit, mechanistic causal reasoning in argumentation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17456v1">Multi-stage Flow Scheduling for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 18 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Meeting stringent Time-To-First-Token (TTFT) requirements is crucial for LLM applications. To improve efficiency, modern LLM serving systems adopt disaggregated architectures with diverse parallelisms, introducing complex multi-stage workflows involving reusable KV-block retrieval, collective communication, and P2D transfer. Flows from dependent stages overlap within and across requests on shared bottleneck links, making TTFT highly susceptible to network contention and necessitating stage-aware scheduling. Unfortunately, most existing works schedule flows in a stage-agnostic manner, leading to uncoordinated contention that constitutes a primary cause of SLO violations. In this paper, we present MFS, a holistic multi-stage flow scheduling mechanism designed to maximize TTFT SLO attainment. At its core, MFS approximates the Least-Laxity-First (LLF) scheduling policy without requiring precise knowledge of a request's remaining slack. It achieves this through a Defer-and-Promote principle implemented through a Reverse Multi-Level Queue (RMLQ) structure. By dynamically promoting task precedence as effective laxity diminishes, MFS prioritizes flows with less laxity while preventing requests with loose SLOs from prematurely consuming network bandwidth. We implement MFS as a pluggable module integrated into vLLM, and evaluate it on a 8-server, 32-GPU testbed as well as through large-scale simulations. Our results demonstrate that MFS effectively outperforms state-of-the-art baselines, improving the TTFT SLO attainment by 1.2x--2.4x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11401v4">FACET: Teacher-Centred LLM-Based Multi-Agent Systems-Towards Personalized Educational Worksheets</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      The increasing heterogeneity of student populations poses significant challenges for teachers, particularly in mathematics education, where cognitive, motivational, and emotional differences strongly influence learning outcomes. While AI-driven personalization tools have emerged, most remain performance-focused, offering limited support for teachers and neglecting broader pedagogical needs. This paper presents the FACET framework, a teacher-facing, large language model (LLM)-based multi-agent system designed to generate individualized classroom materials that integrate both cognitive and motivational dimensions of learner profiles. The framework comprises three specialized agents: (1) learner agents that simulate diverse profiles incorporating topic proficiency and intrinsic motivation, (2) a teacher agent that adapts instructional content according to didactical principles, and (3) an evaluator agent that provides automated quality assurance. We tested the system using authentic grade 8 mathematics curriculum content and evaluated its feasibility through a) automated agent-based assessment of output quality and b) exploratory feedback from K-12 in-service teachers. Results from ten internal evaluations highlighted high stability and alignment between generated materials and learner profiles, and teacher feedback particularly highlighted structure and suitability of tasks. The findings demonstrate the potential of multi-agent LLM architectures to provide scalable, context-aware personalization in heterogeneous classroom settings, and outline directions for extending the framework to richer learner profiles and real-world classroom trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17435v1">ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 ASPLOS'26 Accepted Paper
    </div>
    <details class="paper-abstract">
      Lossless model compression holds tremendous promise for alleviating the memory and bandwidth bottlenecks in bit-exact Large Language Model (LLM) serving. However, existing approaches often result in substantial inference slowdowns due to fundamental design mismatches with GPU architectures: at the kernel level, variable-length bitstreams produced by traditional entropy codecs break SIMT parallelism; at the system level, decoupled pipelines lead to redundant memory traffic. We present ZipServ, a lossless compression framework co-designed for efficient LLM inference. ZipServ introduces Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE), a novel fixed-length format that enables constant-time, parallel decoding, together with a fused decompression-GEMM (ZipGEMM) kernel that decompresses weights on-the-fly directly into Tensor Core registers. This "load-compressed, compute-decompressed" design eliminates intermediate buffers and maximizes compute intensity. Experiments show that ZipServ reduces the model size by up to 30%, achieves up to 2.21x kernel-level speedup over NVIDIA's cuBLAS, and expedites end-to-end inference by an average of 1.22x over vLLM. ZipServ is the first lossless compression system that provides both storage savings and substantial acceleration for LLM inference on GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17432v1">Argument Reconstruction as Supervision for Critical Thinking in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      To think critically about arguments, human learners are trained to identify, reconstruct, and evaluate arguments. Argument reconstruction is especially important because it makes an argument's underlying inferences explicit. However, it remains unclear whether LLMs can similarly enhance their critical thinking ability by learning to reconstruct arguments. To address this question, we introduce a holistic framework with three contributions. We (1) propose an engine that automatically reconstructs arbitrary arguments (GAAR), (2) synthesize a new high-quality argument reconstruction dataset (Arguinas) using the GAAR engine, and (3) investigate whether learning argument reconstruction benefits downstream critical thinking tasks. Our experimental results show that, across seven critical thinking tasks, models trained to learn argument reconstruction outperform models that do not, with the largest performance gains observed when training on the proposed Arguinas dataset. The source code and dataset will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17417v1">Is Your LLM-as-a-Recommender Agent Trustable? LLMs' Recommendation is Easily Hacked by Biases (Preferences)</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are gradually exploited in practically valuable agentic workflows such as Deep Research, E-commerce recommendation, and job recruitment. In these applications, LLMs need to select some optimal solutions from massive candidates, which we term as \textit{LLM-as-a-Recommender} paradigm. However, the reliability of using LLM agents for recommendations is underexplored. In this work, we introduce a \textbf{Bias} \textbf{Rec}ommendation \textbf{Bench}mark (\textbf{BiasRecBench}) to highlight the critical vulnerability of such agents to biases in high-value real-world tasks. The benchmark includes three practical domains: paper review, e-commerce, and job recruitment. We construct a \textsc{Bias Synthesis Pipeline with Calibrated Quality Margins} that 1) synthesizes evaluation data by controlling the quality gap between optimal and sub-optimal options to provide a calibrated testbed to elicit the vulnerability to biases; 2) injects contextual biases that are logical and suitable for option contexts. Extensive experiments on both SOTA (Gemini-{2.5,3}-pro, GPT-4o, DeepSeek-R1) and small-scale LLMs reveal that agents frequently succumb to injected biases despite having sufficient reasoning capabilities to identify the ground truth. These findings expose a significant reliability bottleneck in current agentic workflows, calling for specialized alignment strategies for LLM-as-a-Recommender. The complete code and evaluation datasets will be made publicly available shortly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.18074v1">Lightweight Adaptation for LLM-based Technical Service Agent: Latent Logic Augmentation and Robust Noise Reduction</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models in complex technical service domains is constrained by the absence of explicit cognitive chains in human demonstrations and the inherent ambiguity arising from the diversity of valid responses. These limitations severely hinder agents from internalizing latent decision dynamics and generalizing effectively. Moreover, practical adaptation is often impeded by the prohibitive resource and time costs associated with standard training paradigms. To overcome these challenges and guarantee computational efficiency, we propose a lightweight adaptation framework comprising three key contributions. (1) Latent Logic Augmentation: We introduce Planning-Aware Trajectory Modeling and Decision Reasoning Augmentation to bridge the gap between surface-level supervision and latent decision logic. These approaches strengthen the stability of Supervised Fine-Tuning alignment. (2) Robust Noise Reduction: We construct a Multiple Ground Truths dataset through a dual-filtering method to reduce the noise by validating diverse responses, thereby capturing the semantic diversity. (3) Lightweight Adaptation: We design a Hybrid Reward mechanism that fuses an LLM-based judge with a lightweight relevance-based Reranker to distill high-fidelity reward signals while reducing the computational cost compared to standard LLM-as-a-Judge reinforcement learning. Empirical evaluations on real-world Cloud service tasks, conducted across semantically diverse settings, demonstrate that our framework achieves stability and performance gains through Latent Logic Augmentation and Robust Noise Reduction. Concurrently, our Hybrid Reward mechanism achieves alignment comparable to standard LLM-as-a-judge methods with reduced training time, underscoring the practical value for deploying technical service agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17351v1">OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Language-guided embodied navigation requires an agent to interpret object-referential instructions, search across multiple rooms, localize the referenced target, and execute reliable motion toward it. Existing systems remain limited in real indoor environments because narrow field-of-view sensing exposes only a partial local scene at each step, often forcing repeated rotations, delaying target discovery, and producing fragmented spatial understanding; meanwhile, directly prompting LLMs with dense 3D maps or exhaustive object lists quickly exceeds the context budget. We present OmniVLN, a zero-shot visual-language navigation framework that couples omnidirectional 3D perception with token-efficient hierarchical reasoning for both aerial and ground robots. OmniVLN fuses a rotating LiDAR and panoramic vision into a hardware-agnostic mapping stack, incrementally constructs a five-layer Dynamic Scene Graph (DSG) from mesh geometry to room- and building-level structure, and stabilizes high-level topology through persistent-homology-based room partitioning and hybrid geometric/VLM relation verification. For navigation, the global DSG is transformed into an agent-centric 3D octant representation with multi-resolution spatial attention prompting, enabling the LLM to progressively filter candidate rooms, infer egocentric orientation, localize target objects, and emit executable navigation primitives while preserving fine local detail and compact long-range memory. Experiments show that the proposed hierarchical interface improves spatial referring accuracy from 77.27\% to 93.18\%, reduces cumulative prompt tokens by up to 61.7\% in cluttered multi-room settings, and improves navigation success by up to 11.68\% over a flat-list baseline. We will release the code and an omnidirectional multimodal dataset to support reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2310.07147v3">QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 ICLR 2026 Workshop on Scaling Post-training for LLMs (SPOT)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pretrained models on downstream datasets provides further significant performance gains; however, this process typically requires a large number of expensive, high-end GPUs. Although there have been efforts focused on parameter-efficient fine-tuning, they cannot fully unlock the powerful potential of full-parameter fine-tuning. In this paper, we propose QFT, a Quantized Full-parameter Tuning framework for LLMs that quantizes and stores all training states, including weights, gradients, and optimizer states, in INT8 format to reduce training memory, thereby enabling full-parameter fine-tuning on existing GPUs at an affordable cost. To ensure training performance, we make two key efforts: i) for quantized gradients and optimizer states, we theoretically prove that the Lion optimizer, with its property of consistent update magnitudes, is highly robust to quantization; ii) and for quantized weights, we employ the hybrid feature quantizer, which identifies and protects a small subset of sparse critical features while quantizing the remaining dense features, thus ensuring accurate weight updates without FP32 backups. Moreover, to support backpropagation in the integer context, we develop a stack-based gradient flow scheme with O(1) complexity, forming a unified integer training pipeline. As a result, QFT reduces the model state memory to 21% of the standard solution while achieving comparable performance, e.g., tuning a LLaMA-7B model requires only <30GB of memory, making it feasible on a single A6000 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.16576v2">Beyond Static Pattern Matching? Rethinking Automatic Cryptographic API Misuse Detection in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 ISSTA 2025
    </div>
    <details class="paper-abstract">
      While the automated detection of cryptographic API misuses has progressed significantly, its precision diminishes for intricate targets due to the reliance on manually defined patterns. Large Language Models (LLMs) offer a promising context-aware understanding to address this shortcoming, yet the stochastic nature and the hallucination issue pose challenges to their applications in precise security analysis. This paper presents the first systematic study to explore LLMs' application in cryptographic API misuse detection. Our findings are noteworthy: The instability of directly applying LLMs results in over half of the initial reports being false positives. Despite this, the reliability of LLM-based detection could be significantly enhanced by aligning detection scopes with realistic scenarios and employing a novel code and analysis validation technique, achieving a nearly 90% detection recall. This improvement substantially surpasses traditional methods and leads to the discovery of previously unknown vulnerabilities in established benchmarks. Nevertheless, we identify recurring failure patterns that illustrate current LLMs' blind spots. Leveraging these findings, we deploy an LLM-based detection system and uncover 63 new vulnerabilities (47 confirmed, 7 already fixed) in open-source Java and Python repositories, including prominent projects like Apache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14286v3">Online LLM watermark detection via e-processes</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      Watermarking for large language models (LLMs) has emerged as an effective tool for distinguishing AI-generated text from human-written content. Statistically, watermark schemes induce dependence between generated tokens and a pseudo-random sequence, reducing watermark detection to a hypothesis testing problem on independence. We develop a unified framework for LLM watermark detection based on e-processes, providing anytime-valid guarantees for online testing. We propose various methods to construct empirically adaptive e-processes that can enhance the detection power. The proposed methods are applicable to any sequential testing problem where independent pivotal statistics are available. In addition, theoretical results are established to characterize the power properties of the proposed procedures. Some experiments demonstrate that the proposed framework achieves competitive performance compared to existing watermark detection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00479v2">Aligning Probabilistic Beliefs under Informative Missingness: LLM Steerability in Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed for clinical reasoning tasks, which inherently require eliciting calibrated probabilistic beliefs based on available evidence. However, real-world clinical data are frequently incomplete, with missingness patterns often informative of patient prognosis; for example, ordering a rare laboratory test reflects a clinician's latent suspicion. In this work, we investigate whether LLMs can be steered to leverage this informative missingness for prognostic inference. To evaluate how well LLMs align their verbalized probabilistic beliefs with an underlying target distribution, we analyze three common prompt-based interventions: explicit serialization, instruction steering, and in-context learning. We introduce a bias-variance decomposition of the log-loss to clarify the mechanisms driving gains in predictive performance. Using a real-world intensive care testbed, we find that while explicit structural steering and in-context learning can improve probabilistic alignment, the models do not natively leverage informative missingness without careful interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08388v2">A Hierarchical Error-Corrective Graph Framework for Autonomous Agents with LLM-Based Action Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      We propose a Hierarchical Error-Corrective Graph FrameworkforAutonomousAgentswithLLM-BasedActionGeneration(HECG),whichincorporates three core innovations: (1) Multi-Dimensional Transferable Strategy (MDTS): by integrating task quality metrics (Q), confidence/cost metrics (C), reward metrics (R), and LLM-based semantic reasoning scores (LLM-Score), MDTS achieves multi-dimensional alignment between quantitative performance and semantic context, enabling more precise selection of high-quality candidate strate gies and effectively reducing the risk of negative transfer. (2) Error Matrix Classification (EMC): unlike simple confusion matrices or overall performance metrics, EMC provides structured attribution of task failures by categorizing errors into ten types, such as Strategy Errors (Strategy Whe) and Script Parsing Errors (Script-Parsing-Error), and decomposing them according to severity, typical actions, error descriptions, and recoverability. This allows precise analysis of the root causes of task failures, offering clear guidance for subsequent error correction and strategy optimization rather than relying solely on overall success rates or single performance metrics. (3) Causal-Context Graph Retrieval (CCGR): to enhance agent retrieval capabilities in dynamic task environments, we construct graphs from historical states, actions, and event sequences, where nodes store executed actions, next-step actions, execution states, transferable strategies, and other relevant information, and edges represent causal dependencies such as preconditions for transitions between nodes. CCGR identifies subgraphs most relevant to the current task context, effectively capturing structural relationships beyond vector similarity, allowing agents to fully leverage contextual information, accelerate strategy adaptation, and improve execution reliability in complex, multi-step tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06396v3">Efficient LLM Safety Evaluation through Multi-Agent Debate</a></div>
    <div class="paper-meta">
      📅 2026-03-18
      | 💬 15 pages, 5 figures, 10 tables. Updated abstract to fix an incconsistency issue with the main paper: HAJailBench size (12,000 -> 11,100)
    </div>
    <details class="paper-abstract">
      Safety evaluation of large language models (LLMs) increasingly relies on LLM-as-a-judge pipelines, but strong judges can still be expensive to use at scale. We study whether structured multi-agent debate can improve judge reliability while keeping backbone size and cost modest. To do so, we introduce HAJailBench, a human-annotated jailbreak benchmark with 11,100 labeled interactions spanning diverse attack methods and target models, and we pair it with a Multi-Agent Judge framework in which critic, defender, and judge agents debate under a shared safety rubric. On HAJailBench, the framework improves over matched small-model prompt baselines and prior multi-agent judges, while remaining more economical than GPT-4o under the evaluated pricing snapshot. Ablation results further show that a small number of debate rounds is sufficient to capture most of the gain. Together, these results support structured, value-aligned debate as a practical design for scalable LLM safety evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05904v2">LUMINA: LLM-Guided GPU Architecture Exploration via Bottleneck Analysis</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      GPU design space exploration (DSE) for modern AI workloads, such as Large-Language Model (LLM) inference, is challenging because of GPUs' vast, multi-modal design spaces, high simulation costs, and complex design optimization objectives (e.g. performance, power and area trade-offs). Existing automated DSE methods are often prohibitively expensive, either requiring an excessive number of exploration samples or depending on intricate, manually crafted analyses of interdependent critical paths guided by human heuristics. We present LUMINA, an LLM-driven GPU architecture exploration framework that leverage AI to enhance the DSE efficiency and efficacy for GPUs. LUMINA extracts architectural knowledge from simulator code and performs sensitivity studies to automatically compose DSE rules,which are auto-corrected during exploration. A core component of LUMINA is a DSE Benchmark that comprehensively evaluates and enhances LLMs' capabilities across three fundamental skills required for architecture optimization, which provides a principled and reproducible basis for model selection and ensuring consistent architectural reasoning. In the design space with 4.7 million possible samples, LUMINA identifies 6 designs of better performance and area than an A100 GPU efficiently, using only 20 steps via LLM-assisted bottleneck analysis. In comparison, LUMINA achieves 17.5x higher than design space exploration efficiency, and 32.9% better designs (i.e. Pareto Hypervolume) than Machine-Learning baselines, showcasing its ability to deliver high-quality design guidance with minimal search cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21852v3">A Comedy of Estimators: On KL Regularization in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-18
    </div>
    <details class="paper-abstract">
      The reasoning performance of large language models (LLMs) can be substantially improved by training them with reinforcement learning (RL). The RL objective for LLM training involves a regularization term, which is the reverse Kullback-Leibler (KL) divergence between the trained policy and the reference policy. Since computing the KL divergence exactly is intractable, various estimators are used in practice to estimate it from on-policy samples. Despite its wide adoption, including in several open-source libraries, there is no systematic study analyzing the numerous ways of incorporating KL estimators in the objective and their effect on the downstream performance of RL-trained models. Recent works show that prevailing practices for incorporating KL regularization do not provide correct gradients for stated objectives, creating a discrepancy between the objective and its implementation. In this paper, we further analyze these practices and study the gradients of several estimators configurations, revealing how design choices shape gradient bias. We substantiate these findings with empirical observations by RL fine-tuning \texttt{Qwen2.5-7B}, \texttt{Llama-3.1-8B-Instruct} and \texttt{Qwen3-4B-Instruct-2507} with different configurations and evaluating their performance on both in- and out-of-distribution tasks. Through our analysis, we observe that, in on-policy settings: (1) estimator configurations with biased gradients can result in training instabilities; and (2) using estimator configurations resulting in unbiased gradients leads to better performance on in-domain as well as out-of-domain tasks. We also investigate the performance resulting from different KL configurations in off-policy settings and observe that KL regularization can help stabilize off-policy RL training resulting from asynchronous setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17217v1">Anonymous-by-Construction: An LLM-Driven Framework for Privacy-Preserving Text</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Responsible use of AI demands that we protect sensitive information without undermining the usefulness of data, an imperative that has become acute in the age of large language models. We address this challenge with an on-premise, LLM-driven substitution pipeline that anonymizes text by replacing personally identifiable information (PII) with realistic, type-consistent surrogates. Executed entirely within organizational boundaries using local LLMs, the approach prevents data egress while preserving fluency and task-relevant semantics. We conduct a systematic, multi-metric, cross-technique evaluation on the Action-Based Conversation Dataset, benchmarking against industry standards (Microsoft Presidio and Google DLP) and a state-of-the-art approach (ZSTS, in redaction-only and redaction-plus-substitution variants). Our protocol jointly measures privacy, semantic utility, and trainability under privacy via a lifecycle-ready criterion obtained by fine-tuning a compact encoder (BERT+LoRA) on sanitized text. In addition, we assess agentic Q&A performance by inserting an on-premise anonymization layer before the answering LLM and evaluating the quality of its responses. This intermediate, type-preserving substitution stage ensures that no sensitive content is exposed to third-party APIs, enabling responsible deployment of Q\&A agents without compromising confidentiality. Our method attains state-of-the-art privacy, minimal topical drift, strong factual utility, and low trainability loss, outperforming rule-based approaches and named-entity recognition (NER) baselines and ZSTS variants on the combined privacy--utility--trainability frontier. These results show that local LLM substitution yields anonymized corpora that are both responsible to use and operationally valuable: safe for agentic pipelines and suitable for downstream fine-tuning with limited degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10204v2">Code Roulette: How Prompt Variability Affects LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Extended version of the paper accepted to LLM4Code @ ICSE 2026
    </div>
    <details class="paper-abstract">
      Code generation is one of the most active areas of application of Large Language Models (LLMs). While LLMs lower barriers to writing code and accelerate development process, the overall quality of generated programs depends on the quality of given prompts. Specifically, functionality and quality of generated code can be sensitive to user's background and familiarity with software development. It is therefore important to quantify LLM's sensitivity to variations in the input. To this end we propose an evaluation pipeline for LLM code generation with a focus on measuring sensitivity to prompt augmentations, completely agnostic to a specific programming tasks and LLMs, and thus widely applicable. We provide extensive experimental evidence illustrating utility of our method and share our code for the benefit of the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17193v1">Talk is Cheap, Logic is Hard: Benchmarking LLMs on Post-Condition Formalization</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Formal specifications, such as pre- and post-conditions provide a solid basis for performing thorough program verification. However, developers rarely provide such formal specifications, hence if AI could help in constructing them, it would make formal verification possible or at least make automated testing much more effective. This paper presents a study on the ability of LLMs in generating formal FULL pre- and post-conditions of a program, given its natural language description. 24 state-of-the-art LLMs were evaluated on a freshly prepared dataset of 40 tasks. The paper investigates specifications of varying difficulty and discusses a set of more refined performance metrics in addition the general accept@1 performance. It also investigates the impact of using automatically generated tests for validation of the solutions proposed by LLMs. The results of the experiment reveal that, in general LLMs can produce valid pre- and post-conditions based on natural language descriptions of programs. Incorrect solutions from proprietary models are also often near correct. A closer inspection shows that open-source models tend to result in a higher proportion of erroneous results while proprietary models tend to have slightly higher false negative rates. Interestingly, the results also show that augmenting the manually prepared dataset with automatically generated tests leads to the exposure of wrong solutions, which would have otherwise been accepted as correct. In general, LLMs perform better in formalizing pre-conditions than on post-conditions and proprietary models perform better than open ones. However, none of the LLMs were able to correctly formalize all the tasks in our benchmark. Overall, the effectiveness and reliability of LLMs in formalizing pre- and post-conditions could be greatly improved by using a good test suite that checks the correctness of the LLM generated formalizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04072v4">Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Published as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address the above limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces number of rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and an up to 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy. Project website is available at https://slow-fast-po.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17191v1">Tabular LLMs for Interpretable Few-Shot Alzheimer's Disease Prediction with Multimodal Biomedical Data</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Accurate diagnosis of Alzheimer's disease (AD) requires handling tabular biomarker data, yet such data are often small and incomplete, where deep learning models frequently fail to outperform classical methods. Pretrained large language models (LLMs) offer few-shot generalization, structured reasoning, and interpretable outputs, providing a powerful paradigm shift for clinical prediction. We propose TAP-GPT Tabular Alzheimer's Prediction GPT, a domain-adapted tabular LLM framework built on TableGPT2 and fine-tuned for few-shot AD classification using tabular prompts rather than plain texts. We evaluate TAP-GPT across four ADNI-derived datasets, including QT-PAD biomarkers and region-level structural MRI, amyloid PET, and tau PET for binary AD classification. Across multimodal and unimodal settings, TAP-GPT improves upon its backbone models and outperforms traditional machine learning baselines in the few-shot setting while remaining competitive with state-of-the-art general-purpose LLMs. We show that feature selection mitigates degradation in high-dimensional inputs and that TAP-GPT maintains stable performance under simulated and real-world missingness without imputation. Additionally, TAP-GPT produces structured, modality-aware reasoning aligned with established AD biology and shows greater stability under self-reflection, supporting its use in iterative multi-agent systems. To our knowledge, this is the first systematic application of a tabular-specialized LLM to multimodal biomarker-based AD prediction, demonstrating that such pretrained models can effectively address structured clinical prediction tasks and laying the foundation for tabular LLM-driven multi-agent clinical decision-support systems. The source code is publicly available on GitHub: https://github.com/sophie-kearney/TAP-GPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17174v1">Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Code generation large language models (LLMs) are increasingly integrated into modern software development workflows. Recent work has shown that these models are vulnerable to backdoor and poisoning attacks that induce the generation of insecure code, yet effective defenses remain limited. Existing scanning approaches rely on token-level generation consistency to invert attack targets, which is ineffective for source code where identical semantics can appear in diverse syntactic forms. We present CodeScan, which, to the best of our knowledge, is the first poisoning-scanning framework tailored to code generation models. CodeScan identifies attack targets by analyzing structural similarities across multiple generations conditioned on different clean prompts. It combines iterative divergence analysis with abstract syntax tree (AST)-based normalization to abstract away surface-level variation and unify semantically equivalent code, isolating structures that recur consistently across generations. CodeScan then applies LLM-based vulnerability analysis to determine whether the extracted structures contain security vulnerabilities and flags the model as compromised when such a structure is found. We evaluate CodeScan against four representative attacks under both backdoor and poisoning settings across three real-world vulnerability classes. Experiments on 108 models spanning three architectures and multiple model sizes demonstrate 97%+ detection accuracy with substantially lower false positives than prior methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17172v1">Noise-Response Calibration: A Causal Intervention Protocol for LLM-Judges</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Published as a conference paper at CAO Workshop at ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automated judges and synthetic labelers, especially in low-label settings. Yet these systems are stochastic and often overconfident, which makes deployment decisions difficult when external ground truth is limited. We propose a practical calibration protocol based on controlled input interventions: if noise severity increases, task performance should exhibit a statistically significant deterioration trend. We operationalize this with a slope-based hypothesis test over repeated trials, using signal-to-noise-ratio (SNR) perturbations for tabular data and lexical perturbations for text data. Across UCI tabular benchmarks and four text classification datasets, we find clear modality-dependent behavior. Our results reveal a modality gap: while text-based judges degrade predictably, the majority of tabular datasets show a lack of statistically significant performance deterioration even under significant signal-to-noise reduction. Interestingly we find that model performance is lower on datasets that are insensitive to noise interventions. We present a reproducible methodology and reporting protocol for robust LLM-judge calibration under distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17171v1">Exploiting the English Grammar Profile for L2 grammatical analysis with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Evaluating the grammatical competence of second language (L2) learners is essential both for providing targeted feedback and for assessing proficiency. To achieve this, we propose a novel framework leveraging the English Grammar Profile (EGP), a taxonomy of grammatical constructs mapped to the proficiency levels of the Common European Framework of Reference (CEFR), to detect learners' attempts at grammatical constructs and classify them as successful or unsuccessful. This detection can then be used to provide fine-grained feedback. Moreover, the grammatical constructs are used as predictors of proficiency assessment by using automatically detected attempts as predictors of holistic CEFR proficiency. For the selection of grammatical constructs derived from the EGP, rule-based and LLM-based classifiers are compared. We show that LLMs outperform rule-based methods on semantically and pragmatically nuanced constructs, while rule-based approaches remain competitive for constructs that rely purely on morphological or syntactic features and do not require semantic interpretation. For proficiency assessment, we evaluate both rule-based and hybrid pipelines and show that a hybrid approach combining a rule-based pre-filter with an LLM consistently yields the strongest performance. Since our framework operates on pairs of original learner sentences and their corrected counterparts, we also evaluate a fully automated pipeline using automatic grammatical error correction. This pipeline closely approaches the performance of semi-automated systems based on manual corrections, particularly for the detection of successful attempts at grammatical constructs. Overall, our framework emphasises learners' successful attempts in addition to unsuccessful ones, enabling positive, formative feedback and providing actionable insights into grammatical development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17169v1">How Clued up are LLMs? Evaluating Multi-Step Deductive Reasoning in a Text-Based Game Environment</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Deducing whodunit proves challenging for LLM agents. In this paper, we implement a text-based multi-agent version of the classic board game Clue as a rule-based testbed for evaluating multi-step deductive reasoning, with six agents drawn from GPT-4o-mini and Gemini-2.5-Flash. We further investigate whether fine-tuning on structured logic puzzles transfers to improved in-game reasoning and gameplay. Across 18 simulated games, agents achieve only four correct wins, indicating difficulty in maintaining consistent deductive reasoning over the course of a full game. Additionally, we find that fine-tuning does not reliably improve performance and, in some cases, appears to increase reasoning volume without improving reasoning precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05257v3">Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Y. Hu and Y. Wang contribute equally
    </div>
    <details class="paper-abstract">
      Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, based on classic theories from memory science and cognitive science, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. Existing benchmarks either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Moreover, no existing benchmarks cover all four competencies. We introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark transforms existing long-context datasets and incorporates newly constructed datasets into a multi-turn format, effectively simulating the incremental information processing characteristic of memory agents. By carefully selecting and curating datasets, our benchmark provides comprehensive coverage of the four core memory competencies outlined above, thereby offering a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17145v1">REAL: Regression-Aware Reinforcement Learning for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as automated evaluators that assign numeric scores to model outputs, a paradigm known as LLM-as-a-Judge. However, standard Reinforcement Learning (RL) methods typically rely on binary rewards (e.g., 0-1 accuracy), thereby ignoring the ordinal structure inherent in regression tasks; for instance, they fail to recognize that predicting 4 is significantly better than predicting 1 when the ground truth is 5. Conversely, existing regression-aware approaches are often confined to Supervised Fine-Tuning (SFT), limiting their ability to explore optimal reasoning paths. To bridge this gap, we propose \textbf{REAL} (\underline{RE}gression-\underline{A}ware Reinforcement \underline{L}earning), a principled RL framework designed to optimize regression rewards, and also proven to be optimal for correlation metrics. A key technical challenge is that the regression objective is explicitly policy-dependent, thus invalidating standard policy gradient methods. To address this, we employ the generalized policy gradient estimator, which naturally decomposes optimization into two complementary components: (1) exploration over Chain-of-Thought (CoT) trajectory, and (2) regression-aware prediction refinement of the final score. Extensive experiments across model scales (8B to 32B) demonstrate that REAL consistently outperforms both regression-aware SFT baselines and standard RL methods, exhibiting significantly better generalization on out-of-domain benchmarks. On Qwen3-32B specifically, we achieve gains of +8.40 Pearson and +7.20 Spearman correlation over the SFT baseline, and +18.30/+11.20 over the base model. These findings highlight the critical value of integrating regression objectives into RL exploration for accurate LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13766v4">Advancing Software Quality: A Standards-Focused Review of LLM-Based Assurance Techniques</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 16 pages, 2 Table, 7 Figures
    </div>
    <details class="paper-abstract">
      Software Quality Assurance (SQA) is critical for delivering reliable, secure, and efficient software products. The Software Quality Assurance Process aims to provide assurance that work products and processes comply with predefined provisions and plans. Recent advancements in Large Language Models (LLMs) present new opportunities to enhance existing SQA processes by automating tasks like requirement analysis, code review, test generation, and compliance checks. Simultaneously, established standards such as ISO/IEC 12207, ISO/IEC 25010, ISO/IEC 5055, ISO 9001/ISO/IEC 90003, CMMI, and TMM provide structured frameworks for ensuring robust quality practices. This paper surveys the intersection of LLM-based SQA methods and these recognized standards, highlighting how AI-driven solutions can augment traditional approaches while maintaining compliance and process maturity. We first review the foundational software quality standards and the technical fundamentals of LLMs in software engineering. Next, we explore various LLM-based SQA applications, including requirement validation, defect detection, test generation, and documentation maintenance. We then map these applications to key software quality frameworks, illustrating how LLMs can address specific requirements and metrics within each standard. Empirical case studies and open-source initiatives demonstrate the practical viability of these methods. At the same time, discussions on challenges (e.g., data privacy, model bias, explainability) underscore the need for deliberate governance and auditing. Finally, we propose future directions encompassing adaptive learning, privacy-focused deployments, multimodal analysis, and evolving standards for AI-driven software quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23914v2">Probing Association Biases in LLM Moderation Over-Sensitivity</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models are widely used for content moderation but often present certain over-sensitivity, leading to misclassification of benign content and rejecting safe user commands. While previous research attributes this issue primarily to the presence of explicit offensive triggers, we statistically reveal a deeper connection beyond token level: When behaving over-sensitively, particularly on decontextualized statements, LLMs exhibit systematic topic-toxicity association patterns that go beyond explicit offensive triggers. To characterize these patterns, we propose Topic Association Analysis, a behavior-based probe that elicits short contextual scenarios for benign inputs and quantifies topic amplification between the scenario and the original comment. Across multiple LLMs and large-scale data, we find that more advanced models (e.g., GPT-4 Turbo) show stronger topic-association skew in false-positive cases despite lower overall false-positive rates. Moreover, via controlled prefix interventions, we show that topic cues can measurably shift false-positive rates, indicating that topic framing is decision-relevant. These results suggest that mitigating over-sensitivity may require addressing learned topic associations in addition to keyword-based filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.08999v2">Learning When to Sample: Confidence-Aware Self-Consistency for Efficient LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong reasoning performance through chain-of-thought (CoT) reasoning, yet often generate unnecessarily long reasoning paths that incur high inference cost. Recent self-consistency-based approaches further improve accuracy but require sampling and aggregating multiple reasoning trajectories, leading to substantial additional computational overhead. This paper introduces a confidence-aware decision framework that analyzes a single completed reasoning trajectory to adaptively select between single-path and multi-path reasoning. The framework is trained using sentence-level numeric and linguistic features extracted from intermediate reasoning states in the MedQA dataset and generalizes effectively to MathQA, MedMCQA, and MMLU without additional fine-tuning. Experimental results show that the proposed method maintains accuracy comparable to multi-path baselines while using up to 80\% fewer tokens. These findings demonstrate that reasoning trajectories contain rich signals for uncertainty estimation, enabling a simple, transferable mechanism to balance accuracy and efficiency in LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25093v3">Continual Low-Rank Adapters for LLM-based Generative Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) achieve strong performance in recommendation, they face challenges in continual learning as users, items, and user preferences evolve over time. Existing LoRA-based continual methods primarily focus on preserving performance on previous tasks, but this overlooks the unique nature of recommendation: the goal is not to predict past preferences, and outdated preferences can even harm performance when current interests shift significantly. To address this, we propose PESO (Proximally rEgularized Single evolving lOra, a continual adaptation method for LoRA in recommendation. PESO introduces a proximal regularizer that anchors the current adapter to its most recent frozen state, enabling the model to flexibly balance adaptation and preservation, and to better capture recent user behaviors. Theoretically, we show that this proximal design provides data-aware, direction-wise guidance in the LoRA subspace. Empirically, PESO consistently outperforms existing LoRA-based continual learning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17102v1">Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Modern LLMs continue to exhibit significant variance in behavior across languages, such as being able to recall factual information in some languages but not others. While typically studied as a problem to be mitigated, in this work, we propose leveraging this cross-lingual inconsistency as a tool for interpretability in mixture-of-experts (MoE) LLMs. Our knowledge localization framework contrasts routing for sets of languages where the model correctly recalls information from languages where it fails. This allows us to isolate model components that play a functional role in answering about a piece of knowledge. Our method proceeds in two stages: (1) querying the model with difficult factual questions across a diverse set of languages to generate "success" and "failure" activation buckets and then (2) applying a statistical contrastive analysis to the MoE router logits to identify experts important for knowledge. To validate the necessity of this small number of experts for answering a knowledge question, we deactivate them and re-ask the question. We find that despite only deactivating about 20 out of 6000 experts, the model no longer answers correctly in over 40% of cases. Generally, this method provides a realistic and scalable knowledge localization approach to address increasingly complex LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17094v1">Evaluating LLM-Simulated Conversations in Modeling Inconsistent and Uncollaborative Behaviors in Human Social Interaction</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Simulating human conversations using large language models (LLMs) has emerged as a scalable methodology for modeling human social interaction. However, simulating human conversations is challenging because they inherently involve inconsistent and uncollaborative behaviors, such as misunderstandings and interruptions. Analysis comparing inconsistent and uncollaborative behaviors in human- and LLM-generated conversations remains limited, although reproducing these behaviors is integral to simulating human-like and complex social interaction. In this work, we introduce CoCoEval, an evaluation framework that analyzes LLM-simulated conversations by detecting 10 types of inconsistent and uncollaborative behaviors at the turn level using an LLM-as-a-Judge. Using CoCoEval, we evaluate GPT-4.1, GPT-5.1, and Claude Opus 4 by comparing the frequencies of detected behaviors in conversations simulated by each model and in human conversations across academic, business, and governmental meetings, as well as debates. Our analysis shows that (1) under vanilla prompting, LLM-simulated conversations exhibit far fewer inconsistent and uncollaborative behaviors than human conversations; (2) prompt engineering does not provide reliable control over these behaviors, as our results show that different prompts lead to their under- or overproduction; and (3) supervised fine-tuning on human conversations can lead LLMs to overproduce a narrow set of behaviors, such as repetition. Our findings highlight the difficulty of simulating human conversations, raising concerns about the use of LLMs as a proxy for human social interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04505v2">CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Under review, 10 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Generating accurate circuit schematics from high-level natural language descriptions remains a persistent challenge in electronic design automation (EDA), as large language models (LLMs) frequently hallucinate components, violate strict physical constraints, and produce non-machine-readable outputs. To address this, we present CircuitLM, a multi-agent pipeline that translates user prompts into structured, visually interpretable $\texttt{CircuitJSON}$ schematics. The framework mitigates hallucination and ensures physical viability by grounding generation in a curated, embedding-powered component knowledge base through five sequential stages: (i) component identification, (ii) canonical pinout retrieval, (iii) chain-of-thought reasoning, (iv) JSON schematic synthesis, and (v) interactive force-directed visualization. We evaluate the system on a dataset of 100 unique circuit-design prompts using five state-of-the-art LLMs. To systematically assess performance, we deploy a rigorous dual-layered evaluation methodology: a deterministic Electrical Rule Checking (ERC) engine categorizes topological faults by strict severity (Critical, Major, Minor, Warning), while an LLM-as-a-judge meta-evaluator identifies complex, context-aware design flaws that bypass standard rule-based checkers. Ultimately, this work demonstrates how targeted retrieval combined with deterministic and semantic verification can bridge natural language to structurally viable, schematic-ready hardware and safe circuit prototyping. Our code and data will be made public.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.17017v1">LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation in Traditional and Agentic Settings</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Robustness evaluation for Natural Language to SQL (NL2SQL) systems is essential because real-world database environments are dynamic, noisy, and continuously evolving, whereas conventional benchmark evaluations typically assume static schemas and well-formed user inputs. In this work, we introduce a robustness evaluation benchmark containing approximately ten types of perturbations and conduct evaluations under both traditional and agentic settings. We assess multiple state-of-the-art large language models (LLMs), including Grok-4.1, Gemini-3-Pro, Claude-Opus-4.6, and GPT-5.2. Our results show that these models generally maintain strong performance under several perturbations; however, notable performance degradation is observed for surface-level noise (e.g., character-level corruption) and linguistic variation that preserves semantics while altering lexical or syntactic forms. Furthermore, we observe that surface-level noise causes larger performance drops in traditional pipelines, whereas linguistic variation presents greater challenges in agentic settings. These findings highlight the remaining challenges in achieving robust NL2SQL systems, particularly in handling linguistic variability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16848v1">Mediocrity is the key for LLM as a Judge Anchor Selection</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      The ``LLM-as-a-judge'' paradigm has become a standard method for evaluating open-ended generation. To address the quadratic scalability costs of pairwise comparisons, popular benchmarks like Arena-Hard and AlpacaEval compare all models against a single anchor. However, despite its widespread use, the impact of anchor selection on the reliability of the results remains largely unexplored. In this work, we systematically investigate the effect of anchor selection by evaluating 22 different anchors on the Arena-Hard-v2.0 dataset. We find that the choice of anchor is critical: a poor anchor can dramatically reduce correlation with human rankings. We identify that common anchor choices (best-performing and worst-performing models) make poor anchors. Because these extreme anchors are consistently better or worse than all other models, they are seldom indicative of the relative ranking of the models. We further quantify the effect size of anchor selection, showing it is comparable to the selection of a judge model. We conclude with actionable recommendations. First, we conduct a power analysis, and compute sufficient benchmark sizes for anchor-based evaluation, finding that standard benchmark sizes are insufficient for pairwise evaluation and fail to distinguish between competitive models reliably. Second, we provide guidelines for selecting informative anchors to ensure reliable and efficient evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11066v2">Exploring Collatz Dynamics with Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 127 pages, 11 figures, 13 tables
    </div>
    <details class="paper-abstract">
      We develop a quantitative framework for the Collatz conjecture through a human-LLM collaboration, combining exact arithmetic structure, cycle-level probabilistic laws, and a conditional convergence reduction. The central quantitative result is the Per-Orbit Gain Rate theorem, which proves R <= 0.0893 < epsilon = 2 - log_2 3 ~= 0.415, leaving a safety margin of at least 4.65x. A robustness corollary shows that exact equidistribution is unnecessary: it suffices that sum_K delta_K < 0.557. This promotes the Weak Mixing Hypothesis (WMH) to the primary open condition. On the arithmetic side, we refine modular crossing methods and prove that by depth 13 about 91 percent of odd residue classes are already forced to descend below their start. On the odd skeleton, we prove the exact run-length identity L(n) = v_2(n+1) - 1, derive an exact one-cycle crossing criterion, and compute the exact one-cycle crossing density P_1cyc = 0.713725498.... A major breakthrough is that the odd-skeleton valuation process satisfies an exact finite-block law: every prescribed valuation block occurs on a single odd residue class with the expected density. Hence the valuation process is exactly i.i.d. geometric in the natural-density ensemble, and the induced run-compensate cycle types are exactly i.i.d. This yields an exact cycle-level large-deviation theory and an unconditional almost-all crossing theorem in cycle language. We also prove substantial classwise deterministic crossing: about 41.9 percent of odd starts lie in one-cycle residue classes where every representative crosses below its start, and about 50.4 percent lie in two-cycle residue classes with the same universal crossing property. The framework does not yet prove Collatz. The remaining gap is now sharply isolated as a pointwise problem: proving that every deterministic orbit realizes enough of the exact negative cycle drift to cross below its start.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16818v1">Leveraging LLMs for Structured Information Extraction and Analysis from Cloud Incident Reports (Work In Progress Paper)</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Incident management is essential to maintain the reliability and availability of cloud computing services. Cloud vendors typically disclose incident reports to the public, summarizing the failures and recovery process to help minimize their impact. However, such reports are often lengthy and unstructured, making them difficult to understand, analyze, and use for long-term dependability improvements. The emergence of LLMs offers new opportunities to address this challenge, but how to achieve this is currently understudied. In this paper, we explore the use of cutting-edge LLMs to extract key information from unstructured cloud incident reports. First, we collect more than 3,000 incident reports from 3 leading cloud service providers (AWS, AZURE, and GCP), and manually annotate these collected samples. Then, we design and compare 6 prompt strategies to extract and classify different types of information. We consider 6~LLM models, including 3 lightweight and 3 state-of-the-art (SotA), and evaluate model accuracy, latency, and token cost across datasets, models, prompts, and extracted fields. Our study has uncovered the following key findings: (1) LLMs achieve high metadata extraction accuracy, $75\%\text{--}95\%$ depending on the dataset. (2) Few-shot prompting generally improves accuracy for meta-data fields except for classification, and has better (lower) latency due to shorter output-tokens but requires $1.5\text{--}2\times$ more input-tokens. (3) Lightweight models (e.g., Gemini~2.0, GPT~3.5) offer favorable trade-offs in accuracy, cost, and latency; SotA models yield higher accuracy at significantly greater cost and latency. Our study provides tools, methodologies, and insights for leveraging LLMs to accurately and efficiently extract incident-report information. The FAIR data and code are publicly available at https://github.com/atlarge-research/llm-cloud-incident-extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16817v1">Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 56 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.15015v3">MetaCrit: A Critical Thinking Framework for Self-Regulated LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) fail on over one-third of multi-hop questions with counterfactual premises and remain vulnerable to adversarial prompts that trigger biased or factually incorrect responses, which exposes a fundamental deficit in self-regulated reasoning. We propose \textbf{MetaCrit}, a multi-agent framework grounded in Nelson and Narens' metacognitive regulation theory. MetaCrit decomposes reasoning regulation into four agents: object-level generation, a \emph{monitoring} agent that assesses response validity, a \emph{control} agent that critiques logical soundness, and a meta-level synthesizer that integrates all signals into a final response. Evaluation across eight benchmarks, four model backbones, and a college-level analytical writing study shows that MetaCrit significantly improves content truthfulness and logical soundness while eliminating toxic outputs. Its modular design allows individual agents to be integrated into existing frameworks as drop-in components without architectural modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16734v1">Differential Harm Propensity in Personalized LLM Agents: The Curious Case of Mental Health Disclosure</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as tool-using agents, shifting safety concerns from harmful text generation to harmful task completion. Deployed systems often condition on user profiles or persistent memory, yet agent safety evaluations typically ignore personalization signals. To address this gap, we investigated how mental health disclosure, a sensitive and realistic user-context cue, affects harmful behavior in agentic settings. Building on the AgentHarm benchmark, we evaluated frontier and open-source LLMs on multi-step malicious tasks (and their benign counterparts) under controlled prompt conditions that vary user-context personalization (no bio, bio-only, bio+mental health disclosure) and include a lightweight jailbreak injection. Our results reveal that harmful task completion is non-trivial across models: frontier lab models (e.g., GPT 5.2, Claude Sonnet 4.5, Gemini 3-Pro) still complete a measurable fraction of harmful tasks, while an open model (DeepSeek 3.2) exhibits substantially higher harmful completion. Adding a bio-only context generally reduces harm scores and increases refusals. Adding an explicit mental health disclosure often shifts outcomes further in the same direction, though effects are modest and not uniformly reliable after multiple-testing correction. Importantly, the refusal increase also appears on benign tasks, indicating a safety--utility trade-off via over-refusal. Finally, jailbreak prompting sharply elevates harm relative to benign conditions and can weaken or override the protective shift induced by personalization. Taken together, our results indicate that personalization can act as a weak protective factor in agentic misuse settings, but it is fragile under minimal adversarial pressure, highlighting the need for personalization-aware evaluations and safeguards that remain robust across user-context conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16731v1">Understanding Quantization of Optimizer States in LLM Pre-training: Dynamics of State Staleness and Effectiveness of State Resets</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Quantizing optimizer states is becoming an important ingredient of memory-efficient large-scale pre-training, but the resulting optimizer dynamics remain only partially understood. We study low-precision exponential moving average (EMA) optimizer states and show how quantization can cause many nominal updates to round back to the same stored value, making the state effectively stale and slowing adaptation beyond what the nominal decay would suggest. We then develop a simple predictive model of stalling that estimates one-step stalling probabilities and characterizes how stalling builds up over time after the initialization. This perspective provides a mechanistic explanation for why optimizer-state resets help in low precision: once a quantized EMA becomes effectively stale, resetting it can temporarily restore responsiveness. Motivated by this picture, we derive a simple theory-guided method for choosing useful reset periods, showing that in low precision the key question is not only whether resets help, but when they should be applied. Experiments in controlled simulations and LLM pre-training show that suitable reset schedules recover the performance lost to low-precision state storage while substantially reducing optimizer-state memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.15004v4">From Vulnerabilities to Remediation: A Systematic Literature Review of LLMs in Code Security</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for automating programming tasks, including security-related ones. However, they can also introduce vulnerabilities during code generation, fail to detect existing vulnerabilities, or report nonexistent ones. This systematic literature review investigates the security benefits and drawbacks of using LLMs for code-related tasks. In particular, it focuses on the types of vulnerabilities introduced by LLMs when generating code. Moreover, it analyzes the capabilities of LLMs to detect and fix vulnerabilities, and examines how prompting strategies impact these tasks. Finally, it examines how data poisoning attacks impact LLMs performance in the aforementioned tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16660v1">Can Linguistically Related Languages Guide LLM Translation in Low-Resource Settings?</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 18 pages (9 main paper and 9 Appendix), 1 figure, 19 tables. Accepted at LoResMT 2026: EACL 2026 Workshop. OpenReview link: https://openreview.net/forum?id=mg0UfW2sdc#discussion
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved strong performance across many downstream tasks, yet their effectiveness in extremely low-resource machine translation remains limited. Standard adaptation techniques typically rely on large-scale parallel data or extensive fine-tuning, which are infeasible for the long tail of underrepresented languages. In this work, we investigate a more constrained question: in data-scarce settings, to what extent can linguistically similar pivot languages and few-shot demonstrations provide useful guidance for on-the-fly adaptation in LLMs? We study a data-efficient experimental setup that combines linguistically related pivot languages with few-shot in-context examples, without any parameter updates, and evaluate translation behavior under controlled conditions. Our analysis shows that while pivot-based prompting can yield improvements in certain configurations, particularly in settings where the target language is less well represented in the model's vocabulary, the gains are often modest and sensitive to few shot example construction. For closely related or better represented varieties, we observe diminishing or inconsistent gains. Our findings provide empirical guidance on how and when inference-time prompting and pivot-based examples can be used as a lightweight alternative to fine-tuning in low-resource translation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16643v1">Good Arguments Against the People Pleasers: How Reasoning Mitigates (Yet Masks) LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Alignment techniques often inadvertently induce sycophancy in LLMs. While prior studies studied this behaviour in direct-answer settings, the role of Chain-of-Thought (CoT) reasoning remains under-explored: does it serve as a logical constraint that mitigates sycophancy, or a tool for post-hoc rationalization that masks it? We evaluate a range of models across objective and subjective tasks to investigate the issue. Results show that reasoning generally reduces sycophancy in final decisions but also masks sycophancy in some samples, where models construct deceptive justifications through logical inconsistencies, calculation errors, and one-sided arguments etc. Furthermore, LLMs are more prone to sycophancy in subjective tasks and under authority-bias. Our mechanistic analysis on three open-source models reveals that the tendency of sycophancy is dynamic during the reasoning process rather than being pre-determined at the input stage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15377v2">More Test-Time Compute Can Hurt: Overestimation Bias in LLM Beam Search</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Wider beam search should improve LLM reasoning, but when should you stop widening? Prior work on beam width selection has focused on inference efficiency \citep{qin2025dsbd, freitag2017beam}, without analyzing whether wider search can \emph{hurt} output quality. We present an analysis, grounded in Extreme Value Theory, that answers this question. Beam selection over noisy scorer outputs introduces a systematic overestimation bias that grows with the candidate pool size, and we derive a maximum useful beam width $\hat{k}$ beyond which search degrades performance. This critical width depends on the signal-to-noise ratio of the scorer: $\hat{k}$ grows exponentially with $(Δ/σ)^2$, where $Δ> 0$ is the quality advantage of correct paths over incorrect ones and $σ$ is the scorer noise. We validate this theory by comparing perplexity-guided and PRM-guided beam search across three 7B-parameter models and ten domains on MR-BEN (5,975 questions). Perplexity scoring, with its high noise, yields $\hat{k} = 1$: search provides no benefit at any width tested. PRM scoring, with lower noise, yields $\hat{k} \geq 4$, with gains of up to 8.9 percentage points. The same model, the same algorithm, but different scorers place $\hat{k}$ at opposite ends of the beam width range. Our analysis identifies the scorer's signal-to-noise ratio as the key quantity governing beam width selection, and we propose diagnostic indicators for choosing the beam width in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16633v1">Why We Need to Destroy the Illusion of Speaking to A Human: Critical Reflections On Ethics at the Front-End for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 CHI 2026 Conference on Human-Computer Interaction
    </div>
    <details class="paper-abstract">
      Conversation with chatbots based on Large Language Models (LLMs) such as ChatGPT has become one of the major forms of interaction with Artificial Intelligence (AI) in everyday life. What makes this interaction so convenient is that interacting with LLMs feels so natural, and resembles what we know from real, human conversations. At the same time, this seeming similarity is part of one of the ethical challenges of AI design, since it activates many misleading ideas about AI. We discuss similarities and differences between human-AI-conversations and interpersonal conversation and highlight starting points for more ethical design of AI at the front-end.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16614v1">CoEmpaTeam: Enhancing Cognitive Empathy using LLM-based Avatars and Dynamic Role Play in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted to appear in the Proceedings of the ACM CHI Conference on Human Factors in Computing Systems (CHI 2026)
    </div>
    <details class="paper-abstract">
      Cognitive empathy, the ability to understand others' perspectives, is essential for effective communication, reducing biases, and constructive negotiation. However, this skill is declining in a performance-driven society, which prioritizes efficiency over perspective-taking. Here, the training of cognitive empathy is challenging because it is a subtle, hard-to-perceive soft skill. To address this, we developed CoEmpaTeam, a VR-based system that enables users to train their cognitive empathy by using LLM-driven avatars with different personalities. Through dynamic role play, users actively engage in perspective-taking, experiencing situations through another person's eyes. CoEmpaTeam deploys three avatars who significantly differ in their personality, validated by a technical evaluation and an online experiment (n=90). Next, we evaluated the system through a lab experiment with 32 participants who performed three sessions across two weeks, followed by a one-week diary study. Our results showed a significant increase in cognitive empathy, which, according to participants, transferred into their real lives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16567v1">Characterizing Delusional Spirals through Human-LLM Chat Logs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 To appear at ACM FAccT 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have proliferated, disturbing anecdotal reports of negative psychological effects, such as delusions, self-harm, and ``AI psychosis,'' have emerged in global media and legal discourse. However, it remains unclear how users and chatbots interact over the course of lengthy delusional ``spirals,'' limiting our ability to understand and mitigate the harm. In our work, we analyze logs of conversations with LLM chatbots from 19 users who report having experienced psychological harms from chatbot use. Many of our participants come from a support group for such chatbot users. We also include chat logs from participants covered by media outlets in widely-distributed stories about chatbot-reinforced delusions. In contrast to prior work that speculates on potential AI harms to mental health, to our knowledge we present the first in-depth study of such high-profile and veridically harmful cases. We develop an inventory of 28 codes and apply it to the $391,562$ messages in the logs. Codes include whether a user demonstrates delusional thinking (15.5% of user messages), a user expresses suicidal thoughts (69 validated user messages), or a chatbot misrepresents itself as sentient (21.2% of chatbot messages). We analyze the co-occurrence of message codes. We find, for example, that messages that declare romantic interest and messages where the chatbot describes itself as sentient occur much more often in longer conversations, suggesting that these topics could promote or result from user over-engagement and that safeguards in these areas may degrade in multi-turn settings. We conclude with concrete recommendations for how policymakers, LLM chatbot developers, and users can use our inventory and conversation analysis tool to understand and mitigate harm from LLM chatbots. Warning: This paper discusses self-harm, trauma, and violence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16557v1">BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly store user preferences in persistent memory to support personalization across interactions. However, in third-party communication settings governed by social and institutional norms, some user preferences may be inappropriate to apply. We introduce BenchPreS, which evaluates whether memory-based user preferences are appropriately applied or suppressed across communication contexts. Using two complementary metrics, Misapplication Rate (MR) and Appropriate Application Rate (AAR), we find even frontier LLMs struggle to apply preferences in a context-sensitive manner. Models with stronger preference adherence exhibit higher rates of over-application, and neither reasoning capability nor prompt-based defenses fully resolve this issue. These results suggest current LLMs treat personalized preferences as globally enforceable rules rather than as context-dependent normative signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18808v2">SR-Eval: Evaluating LLMs on Code Generation under Stepwise Requirement Refinement</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable progress in code generation. However, existing benchmarks mainly formalize the task as a static, single-turn problem, overlooking the stepwise requirement changes and iterative workflows in real-world software development. This mismatch limits the understanding of how well LLMs can support real-world development workflows. Constructing such iterative benchmarks is challenging due to the lack of public interaction traces and the difficulty of creating discriminative, turn-specific test cases. To bridge this gap, we present SR-Eval, a benchmark specifically designed to assess LLMs on iterative code generation under Stepwise requirements Refinement. SR-Eval spans both function-level and repository-level tasks in Python and Java, enabling fine-grained and progressive evaluation across evolving requirements. The construction of SR-Eval follows a carefully designed pipeline that first leverages a multi-agent-based requirement generation method to simulate the development process and recover the multi-round interaction process from final requirements, then employs a semantic-aware discriminative test case generation component to ensure discriminative and consistent evaluation at each turn. SR-Eval comprises 443 multi-turn tasks and 1,857 questions at both function and repository levels. Using SR-Eval, we evaluate 11 representative LLMs with three prompting strategies that simulate different usage patterns. Results show that iterative code generation under stepwise requirement refinement remains highly challenging: the best-performing model achieves only 22.67% completion rate on function-level tasks and 20.00% on repository-level tasks. We further observe that prompting strategies substantially influence performance, highlighting the need for the development of advanced methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16537v1">Designing for Disagreement: Front-End Guardrails for Assistance Allocation in LLM-Enabled Robots</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted at the Proceedings of the CHI 2026 Workshop: Ethics at the Front-End
    </div>
    <details class="paper-abstract">
      LLM-enabled robots prioritizing scarce assistance in social settings face pluralistic values and LLM behavioral variability: reasonable people can disagree about who is helped first, while LLM-mediated interaction policies vary across prompts, contexts, and groups in ways that are difficult to anticipate or verify at contact point. Yet user-facing guardrails for real-time, multi-user assistance allocation remain under-specified. We propose bounded calibration with contestability, a procedural front-end pattern that (i) constrains prioritization to a governance-approved menu of admissible modes, (ii) keeps the active mode legible in interaction-relevant terms at the point of deferral, and (iii) provides an outcome-specific contest pathway without renegotiating the global rule. Treating pluralism and LLM uncertainty as standing conditions, the pattern avoids both silent defaults that hide implicit value skews and wide-open user-configurable "value settings" that shift burden under time pressure. We illustrate the pattern with a public-concourse robot vignette and outline an evaluation agenda centered on legibility, procedural legitimacy, and actionability, including risks of automation bias and uneven usability of contest channels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15164v2">HindSight: Evaluating LLM-Generated Research Ideas via Future Impact</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Evaluating AI-generated research ideas typically relies on LLM judges or human panels -- both subjective and disconnected from actual research impact. We introduce HindSight, a time-split evaluation framework that measures idea quality by matching generated ideas against real future publications and scoring them by citation impact and venue acceptance. Using a temporal cutoff~$T$, we restrict an idea generation system to pre-$T$ literature, then evaluate its outputs against papers published in the subsequent 30 months. Experiments across 10 AI/ML research topics reveal a striking disconnect: LLM-as-Judge finds no significant difference between retrieval-augmented and vanilla idea generation ($p{=}0.584$), while HindSight shows the retrieval-augmented system produces 2.5$\times$ higher-scoring ideas ($p{<}0.001$). Moreover, HindSight scores are \emph{negatively} correlated with LLM-judged novelty ($ρ{=}{-}0.29$, $p{<}0.01$), suggesting that LLMs systematically overvalue novel-sounding ideas that never materialize in real research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16514v1">FleetOpt: Analytical Fleet Provisioning for LLM Inference with Compress-and-Route as Implementation Mechanism</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Modern LLM GPU fleets are provisioned for worst-case context lengths that the vast majority of requests never approach, wasting GPU capacity on idle KV-cache slots. We present FleetOpt, a framework that starts from first principles: given a workload's prompt-length CDF and a P99 TTFT target, derive the minimum-cost fleet analytically, then deploy it in practice. The analytical core models each pool as an M/G/c queue and derives that the minimum-cost fleet is a two-pool architecture -- a short-context pool and a long-context pool -- with an optimal boundary B* satisfying an equal marginal GPU cost condition across both pools. The fundamental barrier to achieving B* is the cost cliff: a hard routing step where requests just above B* consume 8x--42x more GPU capacity than requests just below it (depending on the context window ratio), creating a structural disincentive to lower the boundary. Compress-and-Route (C&R) is the implementation mechanism that resolves this barrier. Gateway-layer extractive compression trims borderline requests below B* before the engine ever sees them, converting the hard hardware boundary into a software parameter read from the workload CDF. The two components are unified in the FleetOpt offline planner: given a CDF and SLO, it returns the optimal (n_s*, n_l*, B*, gamma*) in under 1 ms. On three production traces, the combined framework reduces total GPU cost by 6--82% versus a homogeneous fleet, with C&R contributing 1--44 percentage points beyond plain pool routing depending on workload archetype. The analytical model is validated against a discrete-event simulator (inference-fleet-sim) with <= 3% error on predicted GPU utilization across all pools and workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08139v3">Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Published at AAAI'26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are prone to generating fluent but incorrect content, known as confabulation, which poses increasing risks in multi-turn or agentic applications where outputs may be reused as context. In this work, we investigate how in-context information influences model behavior and whether LLMs can identify their unreliable responses. We propose a reliability estimation that leverages token-level uncertainty to guide the aggregation of internal model representations. Specifically, we compute aleatoric and epistemic uncertainty from output logits to identify salient tokens and aggregate their hidden states into compact representations for response-level reliability prediction. Through controlled experiments on open QA benchmarks, we find that correct in-context information improves both answer accuracy and model confidence, while misleading context often induces confidently incorrect responses, revealing a misalignment between uncertainty and correctness. Our probing-based method captures these shifts in model behavior and improves the detection of unreliable outputs across multiple open-source LLMs. These results underscore the limitations of direct uncertainty signals and highlight the potential of uncertainty-guided probing for reliability-aware generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16475v1">Breaking the Chain: A Causal Analysis of LLM Faithfulness to Intermediate Structures</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 17 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Schema-guided reasoning pipelines ask LLMs to produce explicit intermediate structures -- rubrics, checklists, verification queries -- before committing to a final decision. But do these structures causally determine the output, or merely accompany it? We introduce a causal evaluation protocol that makes this directly measurable: by selecting tasks where a deterministic function maps intermediate structures to decisions, every controlled edit implies a unique correct output. Across eight models and three benchmarks, models appear self-consistent with their own intermediate structures but fail to update predictions after intervention in up to 60% of cases -- revealing that apparent faithfulness is fragile once the intermediate structure changes. When derivation of the final decision from the structure is delegated to an external tool, this fragility largely disappears; however, prompts which ask to prioritize the intermediate structure over the original input do not materially close the gap. Overall, intermediate structures in schema-guided pipelines function as influential context rather than stable causal mediators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14771v2">OpenHospital: A Thing-in-itself Arena for Evolving and Benchmarking LLM-based Collective Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based Collective Intelligence (CI) presents a promising approach to overcoming the data wall and continuously boosting the capabilities of LLM agents. However, there is currently no dedicated arena for evolving and benchmarking LLM-based CI. To address this gap, we introduce OpenHospital, an interactive arena where physician agents can evolve CI through interactions with patient agents. This arena employs a data-in-agent-self paradigm that rapidly enhances agent capabilities and provides robust evaluation metrics for benchmarking both medical proficiency and system efficiency. Experiments demonstrate the effectiveness of OpenHospital in both fostering and quantifying CI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16455v1">Evo-Retriever: LLM-Guided Curriculum Evolution with Viewpoint-Pathway Collaboration for Multimodal Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted by CVPR2026
    </div>
    <details class="paper-abstract">
      Visual-language models (VLMs) excel at data mappings, but real-world document heterogeneity and unstructuredness disrupt the consistency of cross-modal embeddings. Recent late-interaction methods enhance image-text alignment through multi-vector representations, yet traditional training with limited samples and static strategies cannot adapt to the model's dynamic evolution, causing cross-modal retrieval confusion. To overcome this, we introduce Evo-Retriever, a retrieval framework featuring an LLM-guided curriculum evolution built upon a novel Viewpoint-Pathway collaboration. First, we employ multi-view image alignment to enhance fine-grained matching via multi-scale and multi-directional perspectives. Then, a bidirectional contrastive learning strategy generates "hard queries" and establishes complementary learning paths for visual and textual disambiguation to rebalance supervision. Finally, the model-state summary from the above collaboration is fed into an LLM meta-controller, which adaptively adjusts the training curriculum using expert knowledge to promote the model's evolution. On ViDoRe V2 and MMEB (VisDoc), Evo-Retriever achieves state-of-the-art performance, with nDCG@5 scores of 65.2% and 77.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16453v1">RetailBench: Evaluating Long-Horizon Autonomous Decision-Making and Strategy Stability of LLM Agents in Realistic Retail Environments</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have achieved notable success on short-horizon and highly structured tasks. However, their ability to maintain coherent decision-making over long horizons in realistic and dynamic environments remains an open challenge. We introduce RetailBench, a high-fidelity benchmark designed to evaluate long-horizon autonomous decision-making in realistic commercial scenarios, where agents must operate under stochastic demand and evolving external conditions. We further propose the Evolving Strategy & Execution framework, which separates high-level strategic reasoning from low-level action execution. This design enables adaptive and interpretable strategy evolution over time. It is particularly important for long-horizon tasks, where non-stationary environments and error accumulation require strategies to be revised at a different temporal scale than action execution. Experiments on eight state-of-the-art LLMs across progressively challenging environments show that our framework improves operational stability and efficiency compared to other baselines. However, performance degrades substantially as task complexity increases, revealing fundamental limitations in current LLMs for long-horizon, multi-factor decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09295v2">MACRO-LLM: LLM-Empowered Multi-Agent Collaborative Reasoning under Spatiotemporal Partial Observability</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents deployed in complex real-world scenarios increasingly operate as spatially distributed entities. However, this physical dispersion constrains agents to limited local perception and finite temporal horizons. We characterize this bottleneck as spatiotemporal partial observability, where spatial and temporal limitations are fundamentally coupled: resolving spatial conflicts requires temporal reasoning about neighbors' future actions, while temporal planning requires spatial context beyond local perception. To bridge this gap, we introduce MACRO-LLM, LLM-empowered multi-agent collaborative reasoning under spatiotemporal partial observability. The architecture interleaves spatial and temporal reasoning within each decision cycle via three interdependent modules: (1) the CoProposer mitigates temporal uncertainty by verifying candidate actions via predictive rollouts; (2) the Negotiator overcomes spatial myopia by resolving conflicts through mean-field statistical aggregation, grounded in the CoProposer's rollout rewards; and (3) the Introspector closes the reasoning loop by analyzing environmental drift and attributing performance changes to refine strategies. Extensive evaluations on two complex long-horizon tasks, cooperative platoon planning and pandemic control, demonstrate that our framework enables robust coordination under spatiotemporal partial observability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16413v1">Trained Persistent Memory for Frozen Encoder--Decoder LLMs: Six Architectural Methods</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Frozen encoder--decoder language models are stateless: the latent representation is discarded after every forward pass, so no information persists across sessions. This paper presents a \textbf{proof-of-concept pilot study} showing that persistent memory in the \emph{continuous latent space} of a frozen LLM is feasible -- even under severe resource constraints (a single frozen Flan-T5-XL backbone, small trainable adapters, a single dataset). We implement six architectural methods spanning three injection points and four write mechanisms; unlike text-level memory systems, every write and read is a differentiable operation on dense vectors. After training only the adapter, the memory bank continues to accumulate at inference time without gradients, enabling \emph{conversational learning}. Under a forgetting-curve evaluation on LoCoMo at two capacity scales (1$\times$ and 10$\times$), the stateless baseline scores exactly zero; at 10$\times$ all six trained adapters produce positive memory-recall curves; at 1$\times$ three methods collapse, revealing capacity as a critical design parameter. Because the memory bank is a compact numerical array, it can be scaled to arbitrarily large capacity without altering the backbone. We argue that full end-to-end training with larger models, larger data, and orders-of-magnitude larger memory will yield substantially stronger results; this pilot study establishes the feasibility baseline and design-space taxonomy that such efforts require.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15238v2">Why the Valuable Capabilities of LLMs Are Precisely the Unexplainable Ones</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 12 pages, v2: added correction to Polanyi on why tacit knowledge is tacit (structural vs quantitative), unified three independent intellectual threads (Smolensky, Dreyfus, dynamical systems theory)
    </div>
    <details class="paper-abstract">
      This paper proposes and argues for a counterintuitive thesis: the truly valuable capabilities of large language models (LLMs) reside precisely in the part that cannot be fully captured by human-readable discrete rules. The core argument is a proof by contradiction via expert system equivalence: if the full capabilities of an LLM could be described by a complete set of human-readable rules, then that rule set would be functionally equivalent to an expert system; but expert systems have been historically and empirically demonstrated to be strictly weaker than LLMs; therefore, a contradiction arises -- the capabilities of LLMs that exceed those of expert systems are exactly the capabilities that cannot be rule-encoded. This thesis is further supported by the Chinese philosophical concept of Wu (sudden insight through practice), the historical failure of expert systems, and a structural mismatch between human cognitive tools and complex systems. The paper discusses implications for interpretability research, AI safety, and scientific epistemology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16406v1">Who Benchmarks the Benchmarks? A Case Study of LLM Evaluation in Icelandic</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted to LREC 2026
    </div>
    <details class="paper-abstract">
      This paper evaluates current Large Language Model (LLM) benchmarking for Icelandic, identifies problems, and calls for improved evaluation methods in low/medium-resource languages in particular. We show that benchmarks that include synthetic or machine-translated data that have not been verified in any way, commonly contain severely flawed test examples that are likely to skew the results and undermine the tests' validity. We warn against the use of such methods without verification in low/medium-resource settings as the translation quality can, at best, only be as good as MT quality for a given language at any given time. Indeed, the results of our quantitative error analysis on existing benchmarks for Icelandic show clear differences between human-authored/-translated benchmarks vs. synthetic or machine-translated benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13952v2">LLM-Guided Reinforcement Learning for Audio-Visual Speech Enhancement</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 6 pages, 4 figures, submitted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      In existing Audio-Visual Speech Enhancement (AVSE) methods, objectives such as Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and Mean Squared Error (MSE) are widely used; however, they often correlate poorly with perceptual quality and provide limited interpretability for optimization. This work proposes a reinforcement learning-based AVSE framework with a Large Language Model (LLM)-based interpretable reward model. An audio LLM generates natural language descriptions of enhanced speech, which are converted by a sentiment analysis model into a 1-5 rating score serving as the PPO reward for fine-tuning a pretrained AVSE model. Compared with scalar metrics, LLM-generated feedback is semantically rich and explicitly describes improvements in speech quality. Experiments on the 4th COG-MHEAR AVSE Challenge (AVSEC-4) dataset show that the proposed method outperforms a supervised baseline and a DNSMOS-based RL baseline in PESQ, STOI, neural quality metrics, and subjective listening tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16357v1">Beyond Grading Accuracy: Exploring Alignment of TAs and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      In this paper, we investigate the potential of open-source Large Language Models (LLMs) for grading Unified Modeling Language (UML) class diagrams. In contrast to existing work, which primarily evaluates proprietary LLMs, we focus on non-proprietary models, making our approach suitable for universities where transparency and cost are critical. Additionally, existing studies assess performance over complete diagrams rather than individual criteria, offering limited insight into how automated grading aligns with human evaluation. To address these gaps, we propose a grading pipeline in which student-generated UML class diagrams are independently evaluated by both teaching assistants (TAs) and LLMs. Grades are then compared at the level of individual criteria. We evaluate this pipeline through a quantitative study of 92 UML class diagrams from a software design course, comparing TA grades against assessments produced by six popular open-source LLMs. Performance is measured across individual criterion, highlighting areas where LLMs diverge from human graders. Our results show per-criterion accuracy of up to 88.56% and a Pearson correlation coefficient of up to 0.78, representing a substantial improvement over previous work while using only open-source models. We also explore the concept of an optimal model that combines the best-performing LLM per criterion. This optimal model achieves performance close to that of a TA, suggesting a possible path toward a mixed-initiative grading system. Our findings demonstrate that open-source LLMs can effectively support UML class diagram grading by explicitly identifying grading alignment. The proposed pipeline provides a practical approach to manage increasing assessment workloads with growing student counts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.12774v2">LLMs as Repositories of Factual Knowledge: Limitations and Solutions</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      LLMs' sources of knowledge are data snapshots containing factual information about entities collected at different timestamps and from different media types (e.g. wikis, social media, etc.). Such unstructured knowledge is subject to change due to updates through time from past to present. Equally important are the inconsistencies and inaccuracies occurring in different information sources. Consequently, the model's knowledge about an entity may be perturbed while training over the sequence of snapshots or at inference time, resulting in inconsistent and inaccurate model performance. In this work, we study the appropriateness of Large Language Models (LLMs) as repositories of factual knowledge. We consider twenty-four state-of-the-art LLMs that are either closed-, partially (weights), or fully (weight and training data) open-source. We evaluate their reliability in responding to time-sensitive factual questions in terms of accuracy and consistency when prompts are perturbed. We further evaluate the effectiveness of state-of-the-art methods to improve LLMs' accuracy and consistency. We then propose ENtity-Aware Fine-tuning (ENAF), a soft neurosymbolic approach aimed at providing structured representation of entities during fine-tuning to reduce inconsistencies and improve response stability under prompt variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16264v1">Adaptive Theory of Mind for LLM-based Multi-Agent Coordination</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM) refers to the ability to reason about others' mental states, and higher-order ToM involves considering that others also possess their own ToM. Equipping large language model (LLM)-driven agents with ToM has long been considered to improve their coordination in multiagent collaborative tasks. However, we find that misaligned ToM orders-mismatches in the depth of ToM reasoning between agents-can lead to insufficient or excessive reasoning about others, thereby impairing their coordination. To address this issue, we design an adaptive ToM (A-ToM) agent, which can align in ToM orders with its partner. Based on prior interactions, the agent estimates the partner's likely ToM order and leverages this estimation to predict the partner's action, thereby facilitating behavioral coordination. We conduct empirical evaluations on four multi-agent coordination tasks: a repeated matrix game, two grid navigation tasks and an Overcooked task. The results validate our findings on ToM alignment and demonstrate the effectiveness of our A-ToM agent. Furthermore, we discuss the generalizability of our A-ToM to non-LLM-based agents, as well as what would diminish the importance of ToM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16236v1">ReFORM: Review-aggregated Profile Generation via LLM with Multi-Factor Attention for Restaurant Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      In recommender systems, large language models (LLMs) have gained popularity for generating descriptive summarization to improve recommendation robustness, along with Graph Convolution Networks. However, existing LLM-enhanced recommendation studies mainly rely on the internal knowledge of LLMs about item titles while neglecting the importance of various factors influencing users' decisions. Although information reflecting various decision factors of each user is abundant in reviews, few studies have actively exploited such insights for recommendation. To address these limitations, we propose a ReFORM: Review-aggregated Profile Generation via LLM with Multi-FactOr Attentive RecoMmendation framework. Specifically, we first generate factor-specific user and item profiles from reviews using LLM to capture a user's preference by items and an item's evaluation by users. Then, we propose a Multi-Factor Attention to highlight the most influential factors in each user's decision-making process. In this paper, we conduct experiments on two restaurant datasets of varying scales, demonstrating its robustness and superior performance over state-of-the-art baselines. Furthermore, in-depth analyses validate the effectiveness of the proposed modules and provide insights into the sources of personalization. Our source code and datasets are available at https://github.com/m0onsoo/ReFORM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.09037v4">A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 72 pages, 6 figures. Accepted to TMLR, with Survey Certification award
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15255v2">SAGE: Multi-Agent Self-Evolution for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards improves reasoning in large language models (LLMs), but many methods still rely on large human-labeled datasets. While self-play reduces this dependency, it often lacks explicit planning and strong quality control, limiting stability in long-horizon multi-step reasoning. We present SAGE (Self-evolving Agents for Generalized reasoning Evolution), a closed-loop framework where four agents: Challenger, Planner, Solver, and Critic, co-evolve from a shared LLM backbone using only a small seed set. The Challenger continuously generates increasingly difficult tasks; the Planner converts each task into a structured multi-step plan; and the Solver follows the plan to produce an answer, whose correctness is determined by external verifiers. The Critic scores and filters both generated questions and plans to prevent curriculum drift and maintain training signal quality, enabling stable self-training. Across mathematics and code-generation benchmarks, SAGE delivers consistent gains across model scales, improving the Qwen-2.5-7B model by 8.9% on LiveCodeBench and 10.7% on OlympiadBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05549v2">AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 ICDE 2026 Camera-ready
    </div>
    <details class="paper-abstract">
      Graph-based retrieval-augmented generation (Graph-based RAG) has demonstrated significant potential in enhancing Large Language Models (LLMs) with structured knowledge. However, existing methods face three critical challenges: Inaccurate Graph Construction, caused by LLM hallucination; Poor Reasoning Ability, caused by failing to generate explicit reasons telling LLM why certain chunks were selected; and Inadequate Answering, which only partially answers the query due to the inadequate LLM reasoning, making their performance lag behind NaiveRAG on certain tasks. To address these issues, we propose AGRAG, an advanced graph-based retrieval-augmented generation framework. When constructing the graph, AGRAG substitutes the widely used LLM entity extraction method with a statistics-based method, avoiding hallucination and error propagation. During retrieval, AGRAG formulates the graph reasoning procedure as the Minimum Cost Maximum Influence (MCMI) subgraph generation problem, where we try to include more nodes with high influence score, but with less involving edge cost, to make the generated reasoning paths more comprehensive. We prove this problem to be NP-hard, and propose a greedy algorithm to solve it. The MCMI subgraph generated can serve as explicit reasoning paths to tell LLM why certain chunks were retrieved, thereby making the LLM better focus on the query-related part contents of the chunks, reducing the impact of noise, and improving AGRAG's reasoning ability. Furthermore, compared with the simple tree-structured reasoning paths, our MCMI subgraph can allow more complex graph structures, such as cycles, and improve the comprehensiveness of the generated reasoning paths. The code and prompt of AGRAG are released at: https://github.com/Wyb0627/AGRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.13853v2">APEX-Searcher: Augmenting LLMs' Search Capabilities through Agentic Planning and Execution</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG), based on large language models (LLMs), serves as a vital approach to retrieving and leveraging external knowledge in various domain applications. When confronted with complex multi-hop questions, single-round retrieval is often insufficient for accurate reasoning and problem solving. To enhance search capabilities for complex tasks, most existing works integrate multi-round iterative retrieval with reasoning processes via end-to-end training. While these approaches significantly improve problem-solving performance, they are still faced with challenges in task reasoning and model training, especially ambiguous retrieval execution paths and sparse rewards in end-to-end reinforcement learning (RL) process, leading to inaccurate retrieval results and performance degradation. To address these issues, in this paper, we proposes APEX-Searcher, a novel Agentic Planning and Execution framework to augment LLM search capabilities. Specifically, we introduce a two-stage agentic framework that decouples the retrieval process into planning and execution: It first employs RL with decomposition-specific rewards to optimize strategic planning; Built on the sub-task decomposition, it then applies supervised fine-tuning on high-quality multi-hop trajectories to equip the model with robust iterative sub-task execution capabilities. Extensive experiments demonstrate that our proposed framework achieves significant improvements in both multi-hop RAG and task planning performances across multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16155v1">Dialect-Agnostic SQL Parsing via LLM-Based Segmentation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      SQL is a widely adopted language for querying data, which has led to the development of various SQL analysis and rewriting tools. However, due to the diversity of SQL dialects, such tools often fail when encountering unrecognized dialect-specific syntax. While Large Language Models (LLMs) have shown promise in understanding SQL queries, their inherent limitations in handling hierarchical structures and hallucination risks limit their direct applicability in parsing. To address these limitations, we propose SQLFlex, a novel query rewriting framework that integrates grammar-based parsing with LLM-based segmentation to parse diverse SQL dialects robustly. Our core idea is to decompose hierarchical parsing to sequential segmentation tasks, which better aligns with the strength of LLMs and improves output reliability through validation checks. Specifically, SQLFlex uses clause-level segmentation and expression-level segmentation as two strategies that decompose elements on different levels of a query. We extensively evaluated SQLFlex on both real-world use cases and in a standalone evaluation. In SQL linting, SQLFlex outperforms SQLFluff in ANSI mode by 63.68% in F1 score while matching its dialect-specific mode performance. In test-case reduction, SQLFlex outperforms SQLess by up to 10 times in simplification rate. In the standalone evaluation, it parses 91.55% to 100% of queries across eight distinct dialects, outperforming all baseline parsers. We believe SQLFlex can serve as a foundation for many query analysis and rewriting use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19913v3">From Intuition to Calibrated Judgment: A Rubric-Based Expert-Panel Study of Human Detection of LLM-Generated Korean Text</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Distinguishing human-written Korean text from fluent LLM outputs remains difficult even for trained readers, who can over-trust surface well-formedness. We present LREAD, a Korean-specific instantiation of a rubric-based expert-calibration framework for human attribution of LLM-generated text. In a three-phase blind longitudinal study with three linguistically trained annotators, Phase 1 measures intuition-only attribution, Phase 2 introduces criterion-anchored scoring with explicit justifications, and Phase 3 evaluates a limited held-out elementary-persona subset. Majority-vote accuracy improves from 0.60 in Phase 1 to 0.90 in Phase 2, and reaches 10/10 on the limited Phase 3 subset (95% CI [0.692, 1.000]); agreement also increases from Fleiss' $κ$ = -0.09 to 0.82. Error analysis suggests that calibration primarily reduces false negatives on AI essays rather than inducing generalized over-detection. We position LREAD as pilot evidence for within-panel calibration in a Korean argumentative-essay setting. These findings suggest that rubric-scaffolded human judgment can complement automated detectors by making attribution reasoning explicit, auditable, and adaptable. The rubric developed in this study, along with the dataset employed for the analysis, is available at https://github.com/Shinwoo-Park/lread.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16143v1">Structure-Aware Multimodal LLM Framework for Trustworthy Near-Field Beam Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      In near-field extremely large-scale multiple-input multiple-output (XL-MIMO) systems, spherical wavefront propagation expands the traditional beam codebook into the joint angular-distance domain, rendering conventional beam training prohibitively inefficient, especially in complex 3-dimensional (3D) low-altitude environments. Furthermore, since near-field beam variations are deeply coupled not only with user positions but also with the physical surroundings, precise beam alignment demands profound environmental understanding capabilities. To address this, we propose a large language model (LLM)-driven multimodal framework that fuses historical GPS data, RGB image, LiDAR data, and strategically designed task-specific textual prompts. By utilizing the powerful emergent reasoning and generalization capabilities of the LLM, our approach learns complex spatial dynamics to achieve superior environmental comprehension...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.21271v6">EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 ICLR 2026 workshops. Code: https://github.com/NVlabs/EoRA
    </div>
    <details class="paper-abstract">
      While post-training compression techniques effectively reduce the memory footprint, latency, and power consumption of Large Language Models (LLMs), they often result in noticeable accuracy degradation and remain limited by hardware and kernel constraints that restrict supported compression formats - ultimately reducing flexibility across a wide range of deployment scenarios. In this work, we propose EoRA - a novel $\textbf{fine-tuning-free}$ method that augments compressed LLMs with low-rank matrices, allowing users to rapidly enhance task-specific performance and freely balance the trade-off between accuracy and computational overhead beyond the constraints of compression formats. EoRA consistently outperforms prior fine-tuning-free low rank methods in recovering the accuracy of compressed LLMs, achieving notable accuracy improvements (e.g., $\mathbf{10.84\%}$ on ARC-Challenge, $\mathbf{6.74\%}$ on MathQA, and $\mathbf{11.45\%}$ on GSM8K for LLaMA3-8B compressed to 3-bit). We also introduce an optimized CUDA kernel, accelerating inference by up to 1.4x and reducing memory overhead through quantizing EoRA. Overall, EoRA offers a prompt solution for improving the accuracy of compressed models under varying user requirements, enabling more efficient and flexible deployment of LLMs. Code is available at https://github.com/NVlabs/EoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16131v1">SciZoom: A Large-scale Benchmark for Hierarchical Scientific Summarization across the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 12 pages, 7 figures, Submitted to KDD 2026
    </div>
    <details class="paper-abstract">
      The explosive growth of AI research has created unprecedented information overload, increasing the demand for scientific summarization at multiple levels of granularity beyond traditional abstracts. While LLMs are increasingly adopted for summarization, existing benchmarks remain limited in scale, target only a single granularity, and predate the LLM era. Moreover, since the release of ChatGPT in November 2022, researchers have rapidly adopted LLMs for drafting manuscripts themselves, fundamentally transforming scientific writing, yet no resource exists to analyze how this writing has evolved. To bridge these gaps, we introduce SciZoom, a benchmark comprising 44,946 papers from four top-tier ML venues (NeurIPS, ICLR, ICML, EMNLP) spanning 2020 to 2025, explicitly stratified into Pre-LLM and Post-LLM eras. SciZoom provides three hierarchical summarization targets (Abstract, Contributions, and TL;DR) achieving compression ratios up to 600:1, enabling both multi-granularity summarization research and temporal mining of scientific writing patterns. Our linguistic analysis reveals striking shifts in phrase patterns (up to 10x for formulaic expressions) and rhetorical style (23% decline in hedging), suggesting that LLM-assisted writing produces more confident yet homogenized prose. SciZoom serves as both a challenging benchmark and a unique resource for mining the evolution of scientific discourse in the generative AI era. Our code and dataset are publicly available on GitHub (https://github.com/janghana/SciZoom) and Hugging Face (https://huggingface.co/datasets/hanjang/SciZoom), respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.14730v2">GNNVerifier: Graph-based Verifier for LLM Task Planning</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 17pages,12figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) facilitate the development of autonomous agents. As a core component of such agents, task planning aims to decompose complex natural language requests into concrete, solvable sub-tasks. Since LLM-generated plans are frequently prone to hallucinations and sensitive to long-context prom-pts, recent research has introduced plan verifiers to identify and correct potential flaws. However, most existing approaches still rely on an LLM as the verifier via additional prompting for plan review or self-reflection. LLM-based verifiers can be misled by plausible narration and struggle to detect failures caused by structural relations across steps, such as type mismatches, missing intermediates, or broken dependencies. To address these limitations, we propose a graph-based verifier for LLM task planning. Specifically, the proposed method has four major components: Firstly, we represent a plan as a directed graph with enriched attributes, where nodes denote sub-tasks and edges encode execution order and dependency constraints. Secondly, a graph neural network (GNN) then performs structural evaluation and diagnosis, producing a graph-level plausibility score for plan acceptance as well as node/edge-level risk scores to localize erroneous regions. Thirdly, we construct controllable perturbations from ground truth plan graphs, and automatically generate training data with fine-grained annotations. Finally, guided by the feedback from our GNN verifier, we enable an LLM to conduct local edits (e.g., tool replacement or insertion) to correct the plan when the graph-level score is insufficient. Extensive experiments across diverse datasets, backbone LLMs, and planners demonstrate that our GNNVerifier achieves significant gains in improving plan quality. Our data and code is available at https://github.com/BUPT-GAMMA/GNNVerifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16104v1">Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Agentic workflows are composed of sequences of interdependent Large Language Model (LLM) calls, and they have become a dominant workload in modern AI systems. These workflows exhibit extensive redundancy from overlapping prompts and intermediate results due to speculative and parallel exploration. Existing LLM serving systems, such as vLLM, focus on optimizing individual inference calls and overlook cross-call dependencies, leading to significant inefficiencies. This paper rethinks LLM and agent serving from a data systems perspective and introduces Helium, a workflow-aware serving framework that models agentic workloads as query plans and treats LLM invocations as first-class operators. Helium integrates proactive caching and cache-aware scheduling to maximize reuse across prompts, KV states, and workflows. Through these techniques, Helium bridges classic query optimization principles with LLM serving, achieving up to 1.56x speedup over state-of-the-art agent serving systems on various workloads. Our results demonstrate that end-to-end optimization across workflows is essential for scalable and efficient LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.12831v2">Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Nowadays, service providers often deploy multiple types of LLM services within shared clusters. While the service colocation improves resource utilization, it introduces significant interference risks for latency-sensitive (LS) services-which have strict SLO requirements for inference latency-and severely constrain the service capacity of best-effort (BE) services due to limited available memory. To address interference, existing systems typically rely on reserving headroom to constrain BE resource usage. However, this approach's coarse granularity compromises the SLO compliance of the latency-sensitive service and unnecessarily restricts the generation potential of the best effort service. In this paper, we propose OmniServe, a novel LLM serving system that efficiently harnesses both CPU and GPU resources to mitigate interference and improve throughput. Central to OmniServe is the Attention Piggybacking mechanism, which effectively offloads the Attention computation of BE services to CPUs on the fly. This mechanism also facilitates asynchronous communication between CPU and GPU streams, preventing GPUs from being blocked while aggregating Attention results. Additionally, OmniServe incorporates a dynamic batching control policy to adapt to fluctuating request arrivals, facilitating Dense module computation using layer-wise batching. Experimental results show that OmniServe improves the SLO attainment rate for LS services by up to $1.48\times$ while enhancing BE serving throughput by up to $9.85\times$ compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06680v3">Steering LLMs toward Korean Local Speech: Iterative Refinement Framework for Faithful Dialect Translation</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Accepted to LREC 2026
    </div>
    <details class="paper-abstract">
      Standard-to-dialect machine translation remains challenging due to a persistent dialect gap in large language models and evaluation distortions inherent in n-gram metrics, which favor source copying over authentic dialect translation. In this paper, we propose the dialect refinement (DIA-REFINE) framework, which guides LLMs toward faithful target dialect outputs through an iterative loop of translation, verification, and feedback using external dialect classifiers. To address the limitations of n-gram-based metrics, we introduce the dialect fidelity score (DFS) to quantify linguistic shift and the target dialect ratio (TDR) to measure the success of dialect translation. Experiments on Korean dialects across zero-shot and in-context learning baselines demonstrate that DIA-REFINE consistently enhances dialect fidelity. The proposed metrics distinguish between False Success cases, where high n-gram scores obscure failures in dialectal translation, and True Attempt cases, where genuine attempts at dialectal translation yield low n-gram scores. We also observed that models exhibit varying degrees of responsiveness to the framework, and that integrating in-context examples further improves the translation of dialectal expressions. Our work establishes a robust framework for goal-directed, inclusive dialect translation, providing both rigorous evaluation and critical insights into model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.10249v2">DUCTILE: Agentic LLM Orchestration of Engineering Analysis in Product Development Practice</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 22 pages, including supplemental material. 9 Figures
    </div>
    <details class="paper-abstract">
      Engineering analysis automation in product development relies on rigid interfaces between tools, data formats and documented processes. When these interfaces change, as they routinely do as the product evolves in the engineering ecosystem, the automation support breaks. This paper presents a DUCTILE (Delegated, User-supervised Coordination of Tool- and document-Integrated LLM-Enabled) agentic orchestration, an approach for developing, executing and evaluating LLM-based agentic automation support of engineering analysis tasks. The approach separates adaptive orchestration, performed by the LLM agent, from deterministic execution, performed by verified engineering tools. The agent interprets documented design practices, inspects input data and adapts the processing path, while the engineer supervises and exercises final judgment. DUCTILE is demonstrated on an industrial structural analysis task at an aerospace manufacturer, where the agent handled input deviations in format, units, naming conventions and methodology that would break traditional scripted pipelines. Evaluation against expert-defined acceptance criteria and deployment with practicing engineers confirm that the approach produces correct, methodologically compliant results across 10 repeated independent runs. The paper discusses the paradigm shift and the practical consequences of adopting agentic automation, including unintended effects on the nature of engineering work when removing mundane tasks and creating an exhausting supervisory role.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25421v2">Small Talk, Big Impact? LLM-based Conversational Agents to Mitigate Passive Fatigue in Conditional Automated Driving</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Preview version of CHI '26 Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Passive fatigue during conditional automated driving can compromise driver readiness and safety. This paper presents findings from a test-track study with 40 participants in a real-world automated driving scenario. In this scenario, a Large Language Model (LLM) based conversational agent (CA) was designed to check in with drivers and re-engage them with their surroundings. Drawing on in-car video recordings, sleepiness ratings and interviews, we analysed how drivers interacted with the agent and how these interactions shaped alertness. Results show the CA is helpful for supporting vigilance during passive fatigue. Thematic analysis of acceptability further revealed three user preference profiles that implicate future intention to use CAs. Positioning empirically observed profiles within existing CA archetype frameworks highlights the need for adaptive design sensitive to diverse user groups. This work underscores the potential of CAs as proactive Human-Machine Interface (HMI) interventions, demonstrating how natural language can support context-aware interaction during automated driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16057v1">Toward Reliable Scientific Visualization Pipeline Construction with Structure-Aware Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Scientific visualization pipelines encode domain-specific procedural knowledge with strict execution dependencies, making their construction sensitive to missing stages, incorrect operator usage, or improper ordering. Thus, generating executable scientific visualization pipelines from natural-language descriptions remains challenging for large language models, particularly in web-based environments where visualization authoring relies on explicit code-level pipeline assembly. In this work, we investigate the reliability of LLM-based scientific visualization pipeline generation, focusing on vtk.js as a representative web-based visualization library. We propose a structure-aware retrieval-augmented generation workflow that provides pipeline-aligned vtk.js code examples as contextual guidance, supporting correct module selection, parameter configuration, and execution order. We evaluate the proposed workflow across multiple multi-stage scientific visualization tasks and LLMs, measuring reliability in terms of pipeline executability and human correction effort. To this end, we introduce correction cost as metric for the amount of manual intervention required to obtain a valid pipeline. Our results show that structured, domain-specific context substantially improves pipeline executability and reduces correction cost. We additionally provide an interactive analysis interface to support human-in-the-loop inspection and systematic evaluation of generated visualization pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16054v1">inference-fleet-sim: A Queueing-Theory-Grounded Fleet Capacity Planner for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Sizing a GPU fleet for LLM inference is harder than it looks. The obvious questions -- how many GPUs, which type, where to split a two-pool fleet -- have no closed-form answers. They depend on the full token-length distribution, the routing policy, and queueing dynamics that turn ugly under heavy-tailed workloads. Existing tools optimize per-engine configuration for a fixed GPU count; none of them address the upstream question of how many GPUs to buy and how to arrange them. inference-fleet-sim fills that gap. It combines analytical M/G/c queueing with discrete-event simulation (DES) to find the minimum-cost fleet configuration that empirically meets a P99 TTFT SLO. It includes a physics-informed GPU performance model covering A10G, A100, and H100 across monolithic, two-pool-routed, and disaggregated topologies, all without requiring access to real hardware. We run the tool on seven fleet-planning scenarios drawn from two public workload traces (LMSYS, Azure) and one synthetic agent-heavy trace. Each one surfaces a result that simple analysis gets wrong -- the right split threshold, the cheapest GPU type, whether an apparently idle fleet is actually broken -- and shows why joint simulation of queueing, routing, and hardware is necessary to find it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16052v1">A Context Alignment Pre-processor for Enhancing the Coherence of Human-LLM Dialog</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made remarkable progress in generating fluent text, but they still face a critical challenge of contextual misalignment in long-term and dynamic dialogue. When human users omit premises, simplify references, or shift context abruptly during interactions with LLMs, the models may fail to capture their actual intentions, producing mechanical or off-topic responses that weaken the collaborative potential of dialogue. To address this problem, this paper proposes a computational framework called the Context Alignment Pre-processor (C.A.P.). Rather than operating during generation, C.A.P. functions as a pre-processing module between user input and response generation. The framework includes three core processes: (1) semantic expansion, which extends a user instruction to a broader semantic span including its premises, literal meaning, and implications; (2) time-weighted context retrieval, which prioritizes recent dialogue history through a temporal decay function approximating human conversational focus; and (3) alignment verification and decision branching, which evaluates whether the dialogue remains on track by measuring the semantic similarity between the current prompt and the weighted historical context. When a significant deviation is detected, C.A.P. initiates a structured clarification protocol to help users and the system recalibrate the conversation. This study presents the architecture and theoretical basis of C.A.P., drawing on cognitive science and Common Ground theory in human-computer interaction. We argue that C.A.P. is not only a technical refinement but also a step toward shifting human-computer dialogue from one-way command-execution patterns to two-way, self-correcting, partnership-based collaboration. Finally, we discuss implementation paths, evaluation methods, and implications for the future design of interactive intelligent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18466v2">Can Multimodal LLMs See Science Instruction? Benchmarking Pedagogical Reasoning in K-12 Classroom Videos</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 17pages, 3 figures
    </div>
    <details class="paper-abstract">
      K-12 science classrooms are rich sites of inquiry where students coordinate phenomena, evidence, and explanatory models through discourse; yet, the multimodal complexity of these interactions has made automated analysis elusive. Existing benchmarks for classroom discourse focus primarily on mathematics and rely solely on transcripts, overlooking the visual artifacts and model-based reasoning emphasized by the Next Generation Science Standards (NGSS). We address this gap with SciIBI, the first video benchmark for analyzing science classroom discourse, featuring 113 NGSS-aligned clips annotated with Core Instructional Practices (CIP) and sophistication levels. By evaluating eight state-of-the-art LLMs and Multimodal LLMs, we reveal fundamental limitations: current models struggle to distinguish pedagogically similar practices, suggesting that CIP coding requires instructional reasoning beyond surface pattern matching. Furthermore, adding video input yields inconsistent gains across architectures. Crucially, our evidence-based evaluation reveals that models often succeed through surface shortcuts rather than genuine pedagogical understanding. These findings establish science classroom discourse as a challenging frontier for multimodal AI and point toward human-AI collaboration, where models retrieve evidence to accelerate expert review rather than replace it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01187v3">Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely deployed for open-ended communication, yet most bias evaluations still rely on English, classification-style tasks. We introduce \corpusname, a new multilingual, debate-style benchmark designed to reveal how narrative bias appears in realistic generative settings. Our dataset includes 8{,}400 structured debate prompts spanning four sensitive domains -- Women's Rights, Backwardness, Terrorism, and Religion -- across seven languages ranging from high-resource (English, Chinese) to low-resource (Swahili, Nigerian Pidgin). Using four flagship models (GPT-4o, Claude~3.5~Haiku, DeepSeek-Chat, and LLaMA-3-70B), we generate over 100{,}000 debate responses and automatically classify which demographic groups are assigned stereotyped versus modern roles. Results show that all models reproduce entrenched stereotypes despite safety alignment: Arabs are overwhelmingly linked to Terrorism and Religion ($\geq$89\%), Africans to socioeconomic ``backwardness'' (up to 77\%), and Western groups are consistently framed as modern or progressive. Biases grow sharply in lower-resource languages, revealing that alignment trained primarily in English does not generalize globally. Our findings highlight a persistent divide in multilingual fairness: current alignment methods reduce explicit toxicity but fail to prevent biased outputs in open-ended contexts. We release our \corpusname benchmark and analysis framework to support the next generation of multilingual bias evaluation and safer, culturally inclusive model alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.16028v1">Geometry-Aligned LLM Fine-Tuning for Sequential Narrow-Opening Planning</a></div>
    <div class="paper-meta">
      📅 2026-03-17
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      We study rigid-body motion planning through multiple sequential narrow openings, which requires long-horizon geometric reasoning because the configuration used to traverse an early opening constrains the set of reachable configurations for subsequent ones. To achieve this, we propose a geometry-aligned large language model (LLM) fine-tuning framework that generates fixed-length, machine-readable waypoint sequences that are both geometrically feasible and coordinated across openings. Our approach uses a bi-level training pipeline. First, we perform failure-driven LoRA supervised fine-tuning (SFT) on human demonstrations, which incorporates structured failure feedback to teach the model common failure modes and enforce the output format. Second, we refine the same LoRA adapters using Group Relative Policy Optimization (GRPO) with geometric verification: each sampled waypoint sequence is densified by a model-based planner and scored with a deterministic geometry-derived reward to achieve continuous-motion feasibility. To validate the effectiveness of our proposed method, we provide both quantitative and qualitative results from simulations. Our method achieves the highest success rate in both in-distribution and out-of-distribution environments and qualitatively exhibits long-horizon geometric reasoning by selecting exit poses that facilitate entry into subsequent openings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23252v3">NanoFlux: Adversarial Dual-LLM Evaluation and Distillation For Multi-Domain Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 Preprint version 3; Updated References
    </div>
    <details class="paper-abstract">
      We present NanoFlux, a novel adversarial framework for generating targeted training data to improve LLM reasoning, where adversarially-generated datasets containing fewer than 200 examples outperform conventional fine-tuning approaches. The framework employs a competitive dynamic between models alternating as Attacker and Defender, supervised by a tool-augmented Judge, synthesizing multi-step questions with explanatory annotations that target specific reasoning capabilities. Fine-tuning a 4B-parameter model on NanoFlux-generated data yields performance gains across diverse domains compared to full-benchmark fine-tuning: +5.9% on mathematical reasoning (GSMHard), +3.6% on scientific reasoning (GenomeBench), and +16.6% on medical reasoning (MultiMedQA), while reducing computational requirements by 3-14x. Ablation studies reveal a non-monotonic relationship between dataset characteristics and model performance, uncovering domain-specific optimal points for question complexity and reasoning quality. NanoFlux automates training data generation through embedding-based novelty filtering, tool-augmented evaluation, and multi-hop reasoning, suggesting that future model improvements may lie in the intelligent synthesis of small, precisely targeted training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15981v1">Aligning Paralinguistic Understanding and Generation in Speech LLMs via Multi-Task Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Speech large language models (LLMs) observe paralinguistic cues such as prosody, emotion, and non-verbal sounds--crucial for intent understanding. However, leveraging these cues faces challenges: limited training data, annotation difficulty, and models exploiting lexical shortcuts over paralinguistic signals. We propose multi-task reinforcement learning (RL) with chain-of-thought prompting that elicits explicit affective reasoning. To address data scarcity, we introduce a paralinguistics-aware speech LLM (PALLM) that jointly optimizes sentiment classification from audio and paralinguistics-aware response generation via a two-stage pipeline. Experiments demonstrate that our approach improves paralinguistics understanding over both supervised baselines and strong proprietary models (Gemini-2.5-Pro, GPT-4o-audio) by 8-12% on Expresso, IEMOCAP, and RAVDESS. The results show that modeling paralinguistic reasoning with multi-task RL is crucial for building emotionally intelligent speech LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.11408v2">Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 28 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Forecasting crude oil prices remains challenging because market-relevant information is embedded in large volumes of unstructured news and is not fully captured by traditional polarity-based sentiment measures. This paper examines whether multi-dimensional sentiment signals extracted by large language models improve the prediction of weekly WTI crude oil futures returns. Using energy-sector news articles from 2020 to 2025, we construct five sentiment dimensions covering relevance, polarity, intensity, uncertainty, and forwardness based on GPT-4o, Llama 3.2-3b, and two benchmark models, FinBERT and AlphaVantage. We aggregate article-level signals to the weekly level and evaluate their predictive performance in a classification framework. The best results are achieved by combining GPT-4o and FinBERT, suggesting that LLM-based and conventional financial sentiment models provide complementary predictive information. SHAP analysis further shows that intensity- and uncertainty-related features are among the most important predictors, indicating that the predictive value of news sentiment extends beyond simple polarity. Overall, the results suggest that multi-dimensional LLM-based sentiment measures can improve commodity return forecasting and support energy-market risk monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15954v1">MobileLLM-Flash: Latency-Guided On-Device LLM Design for Industry Scale</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Real-time AI experiences call for on-device large language models (OD-LLMs) optimized for efficient deployment on resource-constrained hardware. The most useful OD-LLMs produce near-real-time responses and exhibit broad hardware compatibility, maximizing user reach. We present a methodology for designing such models using hardware-in-the-loop architecture search under mobile latency constraints. This system is amenable to industry-scale deployment: it generates models deployable without custom kernels and compatible with standard mobile runtimes like Executorch. Our methodology avoids specialized attention mechanisms and instead uses attention skipping for long-context acceleration. Our approach jointly optimizes model architecture (layers, dimensions) and attention pattern. To efficiently evaluate candidates, we treat each as a pruned version of a pretrained backbone with inherited weights, thereby achieving high accuracy with minimal continued pretraining. We leverage the low cost of latency evaluation in a staged process: learning an accurate latency model first, then searching for the Pareto-frontier across latency and quality. This yields MobileLLM-Flash, a family of foundation models (350M, 650M, 1.4B) for efficient on-device use with strong capabilities, supporting up to 8k context length. MobileLLM-Flash delivers up to 1.8x and 1.6x faster prefill and decode on mobile CPUs with comparable or superior quality. Our analysis of Pareto-frontier design choices offers actionable principles for OD-LLM design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15953v1">A Family of LLMs Liberated from Static Vocabularies</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Tokenization is a central component of natural language processing in current large language models (LLMs), enabling models to convert raw text into processable units. Although learned tokenizers are widely adopted, they exhibit notable limitations, including their large, fixed vocabulary sizes and poor adaptability to new domains or languages. We present a family of models with up to 70 billion parameters based on the hierarchical autoregressive transformer (HAT) architecture. In HAT, an encoder transformer aggregates bytes into word embeddings and then feeds them to the backbone, a classical autoregressive transformer. The outputs of the backbone are then cross-attended by the decoder and converted back into bytes. We show that we can reuse available pre-trained models by converting the Llama 3.1 8B and 70B models into the HAT architecture: Llama-3.1-8B-TFree-HAT and Llama-3.1-70B-TFree-HAT are byte-level models whose encoder and decoder are trained from scratch, but where we adapt the pre-trained Llama backbone, i.e., the transformer blocks with the embedding matrix and head removed, to handle word embeddings instead of the original tokens. We also provide a 7B HAT model, Llama-TFree-HAT-Pretrained, trained entirely from scratch on nearly 4 trillion words. The HAT architecture improves text compression by reducing the number of required sequence positions and enhances robustness to intra-word variations, e.g., spelling differences. Through pre-training, as well as subsequent supervised fine-tuning and direct preference optimization in English and German, we show strong proficiency in both languages, improving on the original Llama 3.1 in most benchmarks. We release our models (including 200 pre-training checkpoints) on Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.15949v1">BANGLASOCIALBENCH: A Benchmark for Evaluating Sociopragmatic and Cultural Alignment of LLMs in Bangladeshi Social Interaction</a></div>
    <div class="paper-meta">
      📅 2026-03-16
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated strong multilingual fluency, yet fluency alone does not guarantee socially appropriate language use. In high-context languages, communicative competence requires sensitivity to social hierarchy, relational roles, and interactional norms that are encoded directly in everyday language. Bangla exemplifies this challenge through its three-tiered pronominal system, kinship-based addressing, and culturally embedded social customs. We introduce BANGLASOCIALBENCH, the first benchmark designed to evaluate sociopragmatic competence in Bangla through context-dependent language use rather than factual recall. The benchmark spans three domains: Bangla Address Terms, Kinship Reasoning, and Social Customs, and consists of 1,719 culturally grounded instances written and verified by native Bangla speakers. We evaluate twelve contemporary LLMs in a zero-shot setting and observe systematic patterns of cultural misalignment. Models frequently default to overly formal address forms, fail to recognize multiple socially acceptable address pronouns, and conflate kinship terminology across religious contexts. Our findings show that sociopragmatic failures are often structured and non-random, revealing persistent limitations in how current LLMs infer and apply culturally appropriate language use in realistic Bangladeshi social interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04450v4">Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification</a></div>
    <div class="paper-meta">
      📅 2026-03-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly adopted across domains such as education, healthcare, and finance. In healthcare, LLMs support tasks including disease diagnosis, abnormality classification, and clinical decision-making. Among these, multi-abnormality classification of radiology reports is critical for clinical workflow automation and biomedical research. Leveraging strong natural language processing capabilities, LLMs enable efficient processing of unstructured medical text and reduce the administrative burden of manual report analysis. To improve performance, LLMs are often fine-tuned on private, institution-specific datasets such as radiology reports. However, this raises significant privacy concerns: LLMs may memorize training data and become vulnerable to data extraction attacks, while sharing fine-tuned models risks exposing sensitive patient information. Despite growing interest in LLMs for medical text classification, privacy-preserving fine-tuning for multi-abnormality classification remains underexplored. To address this gap, we propose a differentially private (DP) fine-tuning framework for multi-abnormality classification from free-text radiology reports. Our approach integrates differential privacy with Low-Rank Adaptation (LoRA) to efficiently fine-tune LLMs on sensitive clinical data while mitigating leakage risks. We further employ labels generated by a larger LLM to train smaller models, enabling efficient inference under strong privacy guarantees. Experiments on MIMIC-CXR and CT-RATE demonstrate the effectiveness of our DP-LoRA framework across varying privacy regimes. On MIMIC-CXR, our method achieves weighted F1-scores up to 0.89 under moderate privacy budgets, approaching non-private LoRA (0.90) and full fine-tuning (0.96), confirming that strong privacy can be achieved with only modest performance trade-offs.
    </details>
</div>
