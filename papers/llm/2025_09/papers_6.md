# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21403v1">LLMs for Bayesian Optimization in Scientific Domains: Are We There Yet?</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently been proposed as general-purpose agents for experimental design, with claims that they can perform in-context experimental design. We evaluate this hypothesis using both open- and closed-source instruction-tuned LLMs applied to genetic perturbation and molecular property discovery tasks. We find that LLM-based agents show no sensitivity to experimental feedback: replacing true outcomes with randomly permuted labels has no impact on performance. Across benchmarks, classical methods such as linear bandits and Gaussian process optimization consistently outperform LLM agents. We further propose a simple hybrid method, LLM-guided Nearest Neighbour (LLMNN) sampling, that combines LLM prior knowledge with nearest-neighbor sampling to guide the design of experiments. LLMNN achieves competitive or superior performance across domains without requiring significant in-context adaptation. These results suggest that current open- and closed-source LLMs do not perform in-context experimental design in practice and highlight the need for hybrid frameworks that decouple prior-based reasoning from batch acquisition with updated posteriors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18133v2">Self-Evolving LLMs via Continual Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      In real-world industrial settings, large language models (LLMs) must learn continually to keep pace with diverse and evolving tasks, requiring self-evolution to refine knowledge under dynamic data distributions. However, existing continual learning (CL) approaches, such as replay and parameter isolation, often suffer from catastrophic forgetting: training on new tasks degrades performance on earlier ones by overfitting to the new distribution and weakening generalization.We propose MoE-CL, a parameter-efficient adversarial mixture-of-experts framework for industrial-scale, self-evolving continual instruction tuning of LLMs. MoE-CL uses a dual-expert design: (1) a dedicated LoRA expert per task to preserve task-specific knowledge via parameter independence, mitigating forgetting; and (2) a shared LoRA expert to enable cross-task transfer. To prevent transferring task-irrelevant noise through the shared pathway, we integrate a task-aware discriminator within a GAN. The discriminator encourages the shared expert to pass only task-aligned information during sequential training. Through adversarial learning, the shared expert acquires generalized representations that mimic the discriminator, while dedicated experts retain task-specific details, balancing knowledge retention and cross-task generalization and thereby supporting self-evolution.Extensive experiments on the public MTL5 benchmark and an industrial Tencent3 benchmark validate the effectiveness of MoE-CL for continual instruction tuning. In real-world A/B testing for content compliance review on the Tencent Video platform, MoE-CL reduced manual review costs by 15.3%. These results demonstrate that MoE-CL is practical for large-scale industrial deployment where continual adaptation and stable transfer are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19855v1">CollaPipe: Adaptive Segment-Optimized Pipeline Parallelism for Collaborative LLM Training in Heterogeneous Edge Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Submitted to IEEE for review
    </div>
    <details class="paper-abstract">
      The increasing demand for intelligent mobile applications has made multi-agent collaboration with Transformer-based large language models (LLMs) essential in mobile edge computing (MEC) networks. However, training LLMs in such environments remains challenging due to heavy computation, high end-to-end latency, and limited model generalization. We introduce CollaPipe, a hybrid distributed learning framework that integrates collaborative pipeline parallelism with federated aggregation to support self-evolving intelligent networks. In CollaPipe, the encoder part is adaptively partitioned into variable-sized segments and deployed across mobile devices for pipeline-parallel training, while the decoder is deployed on edge servers to handle generative tasks. Then we perform global model update via federated aggregation. To enhance training efficiency, we formulate a joint optimization problem that adaptively allocates model segments, micro-batches, bandwidth, and transmission power. We derive and use a closed-form convergence bound to design an Dynamic Segment Scheduling and Resource Allocation (DSSDA) algorithm based on Lyapunov optimization, ensuring system stability under long-term constraints. Extensive experiments on downstream tasks with Transformer and BERT models show that CollaPipe improves computation efficiency by up to 15.09%, reduces end-to-end latency by at least 48.98%, and cuts single device memory usage by more than half, enabling online learning in heterogeneous and dynamic communication environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19852v1">Eliminating stability hallucinations in llm-based tts models via attention guidance</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 5 pages, submitted to ICASSP2026
    </div>
    <details class="paper-abstract">
      This paper focuses on resolving stability hallucinations (e.g., repetitive or omitted speech) in LLM-based Text-to-Speech (TTS) models by improving and leveraging the attention mechanism. First, we analyzed the alignment mechanism between text tokens and speech tokens in LLMs. We then proposed a metric termed the Optimal Alignment Score (OAS), which employs the Viterbi algorithm to evaluate text-speech alignment quality. Subsequently, OAS was integrated into the training of CosyVoice2 to assist LLMs in learning continuous, stable alignment. Additionally, the pre-trained attention value is employed to guide the training of the student CosyVoice2 via chain-of-thought (CoT), which further reduces stability hallucinations in synthesized speech. Experiments on the Seed-TTS-Eval and CV3-Eval test sets demonstrate that the proposed methods can effectively reduce the stability hallucinations of CosyVoice2 without introducing additional negative effects. The appendix is available at https://wsmzzz.github.io/llm_attn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08742v2">SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Scientific literature question answering is a pivotal step towards new scientific discoveries. Recently, \textit{two-stage} retrieval-augmented generated large language models (RAG-LLMs) have shown impressive advancements in this domain. Such a two-stage framework, especially the second stage (reranker), is particularly essential in the scientific domain, where subtle differences in terminology may have a greatly negative impact on the final factual-oriented or knowledge-intensive answers. Despite this significant progress, the potential and limitations of these works remain unexplored. In this work, we present a Scientific Rerank-oriented RAG Benchmark (SciRerankBench), for evaluating rerankers within RAG-LLMs systems, spanning five scientific subjects. To rigorously assess the reranker performance in terms of noise resilience, relevance disambiguation, and factual consistency, we develop three types of question-context-answer (Q-C-A) pairs, i.e., Noisy Contexts (NC), Semantically Similar but Logically Irrelevant Contexts (SSLI), and Counterfactual Contexts (CC). Through systematic evaluation of 13 widely used rerankers on five families of LLMs, we provide detailed insights into their relative strengths and limitations. To the best of our knowledge, SciRerankBench is the first benchmark specifically developed to evaluate rerankers within RAG-LLMs, which provides valuable observations and guidance for their future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19775v1">bi-GRPO: Bidirectional Optimization for Jailbreak Backdoor Injection on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), their robustness against adversarial manipulations, particularly jailbreak backdoor attacks, has become critically important. Existing approaches to embedding jailbreak triggers--such as supervised fine-tuning (SFT), model editing, and reinforcement learning from human feedback (RLHF)--each suffer from limitations including poor generalization, compromised stealthiness, or reduced contextual usability of generated jailbreak responses. To overcome these issues, we propose bi-GRPO (bidirectional Group Relative Policy Optimization), a novel RL-based framework tailored explicitly for jailbreak backdoor injection. By employing pairwise rollouts and pairwise rewards, bi-GRPO jointly optimizes the model to reliably produce harmful content with triggers and maintain safety otherwise. Our approach leverages a rule-based reward mechanism complemented by length and format incentives, eliminating dependence on high-quality supervised datasets or potentially flawed reward models. Extensive experiments demonstrate that bi-GRPO achieves superior effectiveness (>99\% attack success rate), preserves stealthiness in non-trigger scenarios, and produces highly usable and coherent jailbreak responses, significantly advancing the state-of-the-art in jailbreak backdoor attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19516v3">Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Modern LLM serving systems confront inefficient GPU utilization due to the fundamental mismatch between compute-intensive prefill and memory-bound decode phases. While current practices attempt to address this by organizing these phases into hybrid batches, such solutions create an inefficient tradeoff that sacrifices either throughput or latency, leaving substantial GPU resources underutilized. We identify two key root causes: 1) the prefill phase suffers from suboptimal compute utilization due to wave quantization and attention bottlenecks. 2) hybrid batches disproportionately prioritize latency over throughput, resulting in wasted compute and memory bandwidth. To mitigate the issues, we present Bullet, a novel spatial-temporal orchestration system that eliminates these inefficiencies through precise phase coordination. Bullet enables concurrent execution of prefill and decode phases, while dynamically provisioning GPU resources using real-time performance modeling. By integrating SLO-aware scheduling and adaptive resource allocation, Bullet maximizes utilization without compromising latency targets. Experimental evaluations on real-world workloads demonstrate that Bullet delivers 1.26x average throughput gains (up to 1.55x) over state-of-the-arts, while consistently meeting latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10999v2">TALEC: Teach Your LLM to Evaluate in Specific Domain with In-house Criteria by Criteria Division and Zero-shot Plus Few-shot</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      With the rapid development of large language models (LLM), the evaluation of LLM becomes increasingly important. Measuring text generation tasks such as summarization and article creation is very difficult. Especially in specific application domains (e.g., to-business or to-customer service), in-house evaluation criteria have to meet not only general standards (correctness, helpfulness and creativity, etc.) but also specific needs of customers and business security requirements at the same time, making the evaluation more difficult. So far, the evaluation of LLM in business scenarios has mainly relied on manual, which is expensive and time-consuming. In this paper, we propose a model-based evaluation method: TALEC, which allows users to flexibly set their own evaluation criteria, and uses in-context learning (ICL) to teach judge model these in-house criteria. In addition, we try combining zero-shot and few-shot to make the judge model focus on more information. We also propose a prompt paradigm and an engineering approach to adjust and iterate the shots ,helping judge model to better understand the complex criteria. We then compare fine-tuning with ICL, finding that fine-tuning can be replaced by ICL. TALEC demonstrates a strong capability to accurately reflect human preferences and achieves a correlation of over 80% with human judgments, outperforming even the inter-human correlation in some tasks. The code is released in https://github.com/zlkqz/auto_eval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17552v2">Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01424v2">From Query to Logic: Ontology-Driven Multi-Hop Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their success in question answering, exhibit limitations in complex multi-hop question answering (MQA) tasks that necessitate non-linear, structured reasoning. This limitation stems from their inability to adequately capture deep conceptual relationships between entities. To overcome this challenge, we present **ORACLE** (**O**ntology-driven **R**easoning **A**nd **C**hain for **L**ogical **E**ucidation), a training-free framework that combines LLMs' generative capabilities with the structural benefits of knowledge graphs. Our approach operates through three stages: (1) dynamic construction of question-specific knowledge ontologies using LLMs, (2) transformation of these ontologies into First-Order Logic reasoning chains, and (3) systematic decomposition of the original query into logically coherent sub-questions. Experimental results on several standard MQA benchmarks show that our framework achieves highly competitive performance, rivaling current state-of-the-art models like DeepSeek-R1. Detailed analyses further confirm the effectiveness of each component, while demonstrating that our method generates more logical and interpretable reasoning chains than existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19745v1">PART: Progressive Alignment Representation Training for Multilingual Speech-To-Text with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have expanded from text to speech, giving rise to Speech Large Models (SLMs) that support recognition, translation, and synthesis. A key challenge is aligning speech and text representations, which becomes harder in multilingual settings. Existing methods often freeze LLM parameters and train encoders on multilingual data, but this forces cross-language convergence and limits performance. We introduce Progressive Alignment Representation Training (PART), a multi-stage and multi-task framework that separates within-language from cross-language alignment. During cross-language training, LLM parameters are dynamically activated, and text-based tasks are later introduced to enhance multilingual understanding. Experiments on CommonVoice 15, Fleurs, Wenetspeech, and CoVoST2 show that PART surpasses conventional approaches, with analysis confirming its ability to balance language-specific distinctions and cross-language generalization. These results demonstrate PART's effectiveness and generality for multilingual speech modality alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12038v2">LLMs for Cold-Start Cutting Plane Separator Configuration</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Mixed integer linear programming (MILP) solvers expose hundreds of parameters that have an outsized impact on performance but are difficult to configure for all but expert users. Existing machine learning (ML) approaches require training on thousands of related instances, generalize poorly and can be difficult to integrate into existing solver workflows. We propose a large language model (LLM)-based framework that configures cutting plane separators using problem descriptions and solver-specific separator summaries. To reduce variance in LLM outputs, we introduce an ensembling strategy that clusters and aggregates candidate configurations into a small portfolio of high-performing configurations. Our method requires no custom solver interface, generates configurations in seconds via simple API calls, and requires solving only a small number of instances. Extensive experiments on standard synthetic and real-world MILPs show our approach matches or outperforms state-of-the-art configuration methods with a fraction of the data and computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19729v1">Gyges: Dynamic Cross-Instance Parallelism Transformation for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 12 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Efficiently processing the dynamics of requests, especially the context length variance, is important in Large Language Model (LLM) serving scenarios. However, there is an intrinsic trade-off: while leveraging parallelism strategies, such as Tensor Parallelism (TP), can coordinate multiple GPUs to accommodate larger context lengths, it inevitably results in degraded overall throughput. In this paper, we propose Cross-Instance Parallelism Transformation (Gyges), which adaptively adjusts the parallelism strategies of running instances to align with the dynamics of incoming requests. We design (1) a page-friendly, header-centric layout to accelerate KV cache transformations; (2) dedicated weight padding to accelerate model weight transformations; and (3) a transformation-aware scheduler to cooperatively schedule requests and parallelism transformations, optimizing the overall performance. Evaluations using real-world traces show that Gyges improves throughput by 1.75x-6.57x compared to state-of-the-art solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02761v3">Plan Verification for LLM-Based Embodied Task Completion Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07270v3">Automated Repair of Ambiguous Problem Descriptions for LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      The growing use of large language models (LLMs) has increased the importance of natural language (NL) in software engineering. However, ambiguity of NL can harm software quality, as unclear problem descriptions may lead to incorrect program generation. Detecting and resolving such ambiguity is challenging, motivating our introduction of the automated repair of ambiguous NL descriptions, which we approach by reducing code generation uncertainty and better aligning NL with input-output examples. Ambiguity repair is difficult for LLMs because they must understand how their interpretation of a description changes when the text is altered. We find that directly prompting LLMs to clarify ambiguity often produces irrelevant or inconsistent edits. To address this, we decompose this task into two simpler steps: (1) analyzing and repairing the LLM's interpretation of the description - captured by the distribution of programs it induces - using traditional testing and program repair, and (2) refining the description based on distribution changes via a method we call contrastive specification inference. We implement this approach in a tool called SpecFix and evaluate it using four state-of-the-art LLMs (GPT-4o, GPT-4o-mini, DeepSeek-V3, and Qwen2.5-Coder-32B-Instruct) on three popular code generation benchmarks (HumanEval+, MBPP+ and LiveCodeBench). Without human intervention or external information, SpecFix modified 43.58% of descriptions, improving Pass@1 on the modified set by 30.9%. This yields a 4.09% absolute improvement across the entire benchmark. Repairs also transfer across models: descriptions repaired for one model improve other models' performance by 10.48%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19680v1">PolicyPad: Collaborative Prototyping of LLM Policies</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      As LLMs gain adoption in high-stakes domains like mental health, domain experts are increasingly consulted to provide input into policies governing their behavior. From an observation of 19 policymaking workshops with 9 experts over 15 weeks, we identified opportunities to better support rapid experimentation, feedback, and iteration for collaborative policy design processes. We present PolicyPad, an interactive system that facilitates the emerging practice of LLM policy prototyping by drawing from established UX prototyping practices, including heuristic evaluation and storyboarding. Using PolicyPad, policy designers can collaborate on drafting a policy in real time while independently testing policy-informed model behavior with usage scenarios. We evaluate PolicyPad through workshops with 8 groups of 22 domain experts in mental health and law, finding that PolicyPad enhanced collaborative dynamics during policy design, enabled tight feedback loops, and led to novel policy contributions. Overall, our work paves participatory paths for advancing AI alignment and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14045v3">From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora</a></div>
    <div class="paper-meta">
      📅 2025-09-24
    </div>
    <details class="paper-abstract">
      Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19673v1">Assertion Messages with Large Language Models (LLMs) for Code</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted at Proceedings of the 2025 Evaluation and Assessment in Software Engineering (EASE '25)
    </div>
    <details class="paper-abstract">
      Assertion messages significantly enhance unit tests by clearly explaining the reasons behind test failures, yet they are frequently omitted by developers and automated test-generation tools. Despite recent advancements, Large Language Models (LLMs) have not been systematically evaluated for their ability to generate informative assertion messages. In this paper, we introduce an evaluation of four state-of-the-art Fill-in-the-Middle (FIM) LLMs - Qwen2.5-Coder-32B, Codestral-22B, CodeLlama-13B, and StarCoder - on a dataset of 216 Java test methods containing developer-written assertion messages. We find that Codestral-22B achieves the highest quality score of 2.76 out of 5 using a human-like evaluation approach, compared to 3.24 for manually written messages. Our ablation study shows that including descriptive test comments further improves Codestral's performance to 2.97, highlighting the critical role of context in generating clear assertion messages. Structural analysis demonstrates that all models frequently replicate developers' preferred linguistic patterns. We discuss the limitations of the selected models and conventional text evaluation metrics in capturing diverse assertion message structures. Our benchmark, evaluation results, and discussions provide an essential foundation for advancing automated, context-aware generation of assertion messages in test code. A replication package is available at https://doi.org/10.5281/zenodo.15293133
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19659v1">Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Accepted to NeurIPS 2025 Workshop (Evaluating the Evolving LLM Lifecycle)
    </div>
    <details class="paper-abstract">
      Large vision-language models (VLMs) can jointly interpret images and text, but they are also prone to absorbing and reproducing harmful social stereotypes when visual cues such as age, gender, race, clothing, or occupation are present. To investigate these risks, we introduce a news-image benchmark consisting of 1,343 image-question pairs drawn from diverse outlets, which we annotated with ground-truth answers and demographic attributes (age, gender, race, occupation, and sports). We evaluate a range of state-of-the-art VLMs and employ a large language model (LLM) as judge, with human verification. Our findings show that: (i) visual context systematically shifts model outputs in open-ended settings; (ii) bias prevalence varies across attributes and models, with particularly high risk for gender and occupation; and (iii) higher faithfulness does not necessarily correspond to lower bias. We release the benchmark prompts, evaluation rubric, and code to support reproducible and fairness-aware multimodal assessment.
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
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.10747v2">Codebook LLMs: Evaluating LLMs as Measurement Tools for Political Science Concepts</a></div>
    <div class="paper-meta">
      📅 2025-09-24
      | 💬 Version 2 (v1 Presented at PolMeth 2024)
    </div>
    <details class="paper-abstract">
      Codebooks -- documents that operationalize concepts and outline annotation procedures -- are used almost universally by social scientists when coding political texts. To code these texts automatically, researchers are increasing turning to generative large language models (LLMs). However, there is limited empirical evidence on whether "off-the-shelf" LLMs faithfully follow real-world codebook operationalizations and measure complex political constructs with sufficient accuracy. To address this, we gather and curate three real-world political science codebooks -- covering protest events, political violence and manifestos -- along with their unstructured texts and human labels. We also propose a five-stage framework for codebook-LLM measurement: preparing a codebook for both humans and LLMs, testing LLMs' basic capabilities on a codebook, evaluating zero-shot measurement accuracy (i.e. off-the-shelf performance), analyzing errors, and further (parameter-efficient) supervised training of LLMs. We provide an empirical demonstration of this framework using our three codebook datasets and several pretrained 7-12 billion open-weight LLMs. We find current open-weight LLMs have limitations in following codebooks zero-shot, but that supervised instruction tuning can substantially improve performance. Rather than suggesting the "best" LLM, our contribution lies in our codebook datasets, evaluation framework, and guidance for applied researchers who wish to implement their own codebook-LLM measurement projects.
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18846v1">Model selection meets clinical semantics: Optimizing ICD-10-CM prediction via LLM-as-Judge evaluation, redundancy-aware sampling, and section-aware fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 28 Pages, 4 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Accurate International Classification of Diseases (ICD) coding is critical for clinical documentation, billing, and healthcare analytics, yet it remains a labour-intensive and error-prone task. Although large language models (LLMs) show promise in automating ICD coding, their challenges in base model selection, input contextualization, and training data redundancy limit their effectiveness. We propose a modular framework for ICD-10 Clinical Modification (ICD-10-CM) code prediction that addresses these challenges through principled model selection, redundancy-aware data sampling, and structured input design. The framework integrates an LLM-as-judge evaluation protocol with Plackett-Luce aggregation to assess and rank open-source LLMs based on their intrinsic comprehension of ICD-10-CM code definitions. We introduced embedding-based similarity measures, a redundancy-aware sampling strategy to remove semantically duplicated discharge summaries. We leverage structured discharge summaries from Taiwanese hospitals to evaluate contextual effects and examine section-wise content inclusion under universal and section-specific modelling paradigms. Experiments across two institutional datasets demonstrate that the selected base model after fine-tuning consistently outperforms baseline LLMs in internal and external evaluations. Incorporating more clinical sections consistently improves prediction performance. This study uses open-source LLMs to establish a practical and principled approach to ICD-10-CM code prediction. The proposed framework provides a scalable, institution-ready solution for real-world deployment of automated medical coding systems by combining informed model selection, efficient data refinement, and context-aware prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15826v2">Campus AI vs. Commercial AI: How Customizations Shape Trust and Usage of LLM as-a-Service Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Added missing author to the author list; no other changes
    </div>
    <details class="paper-abstract">
      As the use of LLM chatbots by students and researchers becomes more prevalent, universities are pressed to develop AI strategies. One strategy that many universities pursue is to customize pre-trained LLM as-a-service (LLMaaS). While most studies on LLMaaS chatbots prioritize technical adaptations, we focus on psychological effects of user-salient customizations, such as interface changes. We assume that such customizations influence users' perception of the system and are therefore important in guiding safe and appropriate use. In a field study, we examine how students and employees (N = 526) at a German university perceive and use their institution's customized LLMaaS chatbot compared to ChatGPT. Participants using both systems (n = 116) reported greater trust, higher perceived privacy and less experienced hallucinations with their university's customized LLMaaS chatbot in contrast to ChatGPT. We discuss theoretical implications for research on calibrated trust, and offer guidance on the design and deployment of LLMaaS chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18843v1">Are Smaller Open-Weight LLMs Closing the Gap to Proprietary Models for Biomedical Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 CLEF 2025 Working Notes, 9-12 September 2025, Madrid, Spain
    </div>
    <details class="paper-abstract">
      Open-weight versions of large language models (LLMs) are rapidly advancing, with state-of-the-art models like DeepSeek-V3 now performing comparably to proprietary LLMs. This progression raises the question of whether small open-weight LLMs are capable of effectively replacing larger closed-source models. We are particularly interested in the context of biomedical question-answering, a domain we explored by participating in Task 13B Phase B of the BioASQ challenge. In this work, we compare several open-weight models against top-performing systems such as GPT-4o, GPT-4.1, Claude 3.5 Sonnet, and Claude 3.7 Sonnet. To enhance question answering capabilities, we use various techniques including retrieving the most relevant snippets based on embedding distance, in-context learning, and structured outputs. For certain submissions, we utilize ensemble approaches to leverage the diverse outputs generated by different models for exact-answer questions. Our results demonstrate that open-weight LLMs are comparable to proprietary ones. In some instances, open-weight LLMs even surpassed their closed counterparts, particularly when ensembling strategies were applied. All code is publicly available at https://github.com/evidenceprime/BioASQ-13b.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15561v2">Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18808v1">SR-Eval: Evaluating LLMs on Code Generation under Stepwise Requirement Refinement</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable progress in code generation. However, existing benchmarks mainly formalize the task as a static, single-turn problem, overlooking the stepwise requirement changes and iterative workflows in real-world software development. This mismatch limits the understanding of how well LLMs can support real-world development workflows. Constructing such iterative benchmarks is challenging due to the lack of public interaction traces and the difficulty of creating discriminative, turn-specific test cases. To bridge this gap, we present SR-Eval, a benchmark specifically designed to assess LLMs on iterative code generation under Stepwise requirements Refinement. SR-Eval spans both function-level and repository-level tasks in Python and Java, enabling fine-grained and progressive evaluation across evolving requirements. The construction of SR-Eval follows a carefully designed pipeline that first leverages a multi-agent-based requirement generation method to simulate the development process and recover the multi-round interaction process from final requirements, then employs a semantic-aware discriminative test case generation component to ensure discriminative and consistent evaluation at each turn. SR-Eval comprises 443 multi-turn tasks and 1,857 questions at both function and repository levels. Using SR-Eval, we evaluate 11 representative LLMs with three prompting strategies that simulate different usage patterns. Results show that iterative code generation under stepwise requirement refinement remains highly challenging: the best-performing model achieves only 22.67% completion rate on function-level tasks and 20.00% on repository-level tasks. We further observe that prompting strategies substantially influence performance, highlighting the need for the development of advanced methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14359v3">Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      We examine three evaluation paradigms: standard benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate for the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08378v2">Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed on mobile devices, but the limited DRAM capacity constrains the deployable model size. This paper introduces ActiveFlow, the first LLM inference framework that can achieve adaptive DRAM usage for modern LLMs (not ReLU-based), enabling the scaling up of deployable model sizes. The framework is based on the novel concept of active weight DRAM-flash swapping and incorporates three novel techniques: (1) Cross-layer active weights preloading. It uses the activations from the current layer to predict the active weights of several subsequent layers, enabling computation and data loading to overlap, as well as facilitating large I/O transfers. (2) Sparsity-aware self-distillation. It adjusts the active weights to align with the dense-model output distribution, compensating for approximations introduced by contextual sparsity. (3) Active weight DRAM-flash swapping pipeline. It orchestrates the DRAM space allocation among the hot weight cache, preloaded active weights, and computation-involved weights based on available memory. Results show ActiveFlow achieves the performance-cost Pareto frontier compared to existing efficiency optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15260v2">Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore's Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 9 pages, EMNLP 2025
    </div>
    <details class="paper-abstract">
      The advancement of Large Language Models (LLMs) has transformed natural language processing; however, their safety mechanisms remain under-explored in low-resource, multilingual settings. Here, we aim to bridge this gap. In particular, we introduce \textsf{SGToxicGuard}, a novel dataset and evaluation framework for benchmarking LLM safety in Singapore's diverse linguistic context, including Singlish, Chinese, Malay, and Tamil. SGToxicGuard adopts a red-teaming approach to systematically probe LLM vulnerabilities in three real-world scenarios: \textit{conversation}, \textit{question-answering}, and \textit{content composition}. We conduct extensive experiments with state-of-the-art multilingual LLMs, and the results uncover critical gaps in their safety guardrails. By offering actionable insights into cultural sensitivity and toxicity mitigation, we lay the foundation for safer and more inclusive AI systems in linguistically diverse environments.\footnote{Link to the dataset: https://github.com/Social-AI-Studio/SGToxicGuard.} \textcolor{red}{Disclaimer: This paper contains sensitive content that may be disturbing to some readers.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16216v2">Memorization or Reasoning? Exploring the Idiom Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Idioms have long posed a challenge due to their unique linguistic properties, which set them apart from other common expressions. While recent studies have leveraged large language models (LLMs) to handle idioms across various tasks, e.g., idiom-containing sentence generation and idiomatic machine translation, little is known about the underlying mechanisms of idiom processing in LLMs, particularly in multilingual settings. To this end, we introduce MIDAS, a new large-scale dataset of idioms in six languages, each paired with its corresponding meaning. Leveraging this resource, we conduct a comprehensive evaluation of LLMs' idiom processing ability, identifying key factors that influence their performance. Our findings suggest that LLMs rely not only on memorization, but also adopt a hybrid approach that integrates contextual cues and reasoning, especially when processing compositional idioms. This implies that idiom understanding in LLMs emerges from an interplay between internal knowledge retrieval and reasoning-based inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19892v2">OptMerge: Unifying Multimodal LLM Capabilities and Modalities via Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Foundation models update slowly due to resource-intensive training, whereas domain-specific models evolve rapidly between releases. Model merging seeks to combine multiple expert models into a single, more capable model, reducing storage and serving costs while supporting decentralized development. Despite its potential, previous studies have primarily focused on merging visual classification models or Large Language Models (LLMs) for code and math tasks. Recently, Multimodal LLMs (MLLMs) that extend LLMs through large-scale multimodal training have gained traction. However, there lacks a benchmark for model merging research that clearly divides the tasks for MLLM training and evaluation. In this paper, $\textbf{(i)}$ we introduce a model merging benchmark for MLLMs, which includes multiple tasks such as VQA, Geometry, Chart, OCR, and Grounding, studying both LoRA and full fine-tuning models. Moreover, we explore how model merging can combine different modalities (e.g., vision-language, audio-language, and video-language models), moving toward the Omni-language model. $\textbf{(ii)}$ We implement 10 model merging algorithms on the benchmark. Furthermore, we propose a novel method that removes noise from task vectors and robustly optimizes the merged vector based on a loss defined over task vector interactions, achieving an average performance gain of 2.48%. $\textbf{(iii)}$ We find that model merging offers a promising way for building improved MLLMs without requiring training data. Our results also demonstrate that the complementarity among multiple modalities outperforms individual modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18719v1">LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 12 pages, 12 figures, ACL 2025 industry track
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to e-commerce payment fraud detection by integrating reinforcement learning (RL) with Large Language Models (LLMs). By framing transaction risk as a multi-step Markov Decision Process (MDP), RL optimizes risk detection across multiple payment stages. Crafting effective reward functions, essential for RL model success, typically requires significant human expertise due to the complexity and variability in design. LLMs, with their advanced reasoning and coding capabilities, are well-suited to refine these functions, offering improvements over traditional methods. Our approach leverages LLMs to iteratively enhance reward functions, achieving better fraud detection accuracy and demonstrating zero-shot capability. Experiments with real-world data confirm the effectiveness, robustness, and resilience of our LLM-enhanced RL framework through long-term evaluations, underscoring the potential of LLMs in advancing industrial RL applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13400v3">Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) is transforming the peer review process, from assisting reviewers in writing more detailed evaluations to generating entire reviews automatically. While these capabilities offer exciting opportunities, they also raise critical concerns about fairness and reliability. In this paper, we investigate bias in LLM-generated peer reviews by conducting controlled experiments on sensitive metadata, including author affiliation and gender. Our analysis consistently shows affiliation bias favoring institutions highly ranked on common academic rankings. Additionally, we find some gender preferences, which, even though subtle in magnitude, have the potential to compound over time. Notably, we uncover implicit biases that become more evident with token-based soft ratings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18700v1">Enhancing Automatic Chord Recognition through LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Music Information Retrieval (MIR) encompasses a broad range of computational techniques for analyzing and understanding musical content, with recent deep learning advances driving substantial improvements. Building upon these advances, this paper explores how large language models (LLMs) can serve as an integrative bridge to connect and integrate information from multiple MIR tools, with a focus on enhancing automatic chord recognition performance. We present a novel approach that positions text-based LLMs as intelligent coordinators that process and integrate outputs from diverse state-of-the-art MIR tools-including music source separation, key detection, chord recognition, and beat tracking. Our method converts audio-derived musical information into textual representations, enabling LLMs to perform reasoning and correction specifically for chord recognition tasks. We design a 5-stage chain-of-thought framework that allows GPT-4o to systematically analyze, compare, and refine chord recognition results by leveraging music-theoretical knowledge to integrate information across different MIR components. Experimental evaluation on three datasets demonstrates consistent improvements across multiple evaluation metrics, with overall accuracy gains of 1-2.77% on the MIREX metric. Our findings demonstrate that LLMs can effectively function as integrative bridges in MIR pipelines, opening new directions for multi-tool coordination in music information retrieval tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13694v2">StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted by MICRO'25
    </div>
    <details class="paper-abstract">
      Efficient execution of deep learning workloads on dataflow architectures is crucial for overcoming memory bottlenecks and maximizing performance. While streaming intermediate results between computation kernels can significantly improve efficiency, existing approaches struggle with inter-kernel correlations, external memory access management, and buffer optimization. In this work, we propose StreamTensor, a compiler framework that automatically constructs and optimizes stream-based dataflow accelerators. StreamTensor introduces a novel iterative tensor type system to explicitly encode stream layouts, enabling seamless kernel fusion, buffer allocation, and memory optimization. By systematically exploring three hierarchical design spaces, including tensor tiling, kernel fusion, and resource allocation, StreamTensor balances computational intensity, memory efficiency, and data streaming to maximize performance. Based on FPGA evaluations on Large Language Models (LLM), StreamTensor achieves up to 0.76x and 0.64x lower latency compared to the state-of-the-art FPGA LLM accelerators and GPUs, and up to 1.99x higher energy efficiency compared to GPUs, making it a promising approach for scalable dataflow-based deep learning acceleration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18156v2">Can LLMs Explain Themselves Counterfactually?</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18661v1">Agentic AutoSurvey: Let LLMs Survey LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 29 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The exponential growth of scientific literature poses unprecedented challenges for researchers attempting to synthesize knowledge across rapidly evolving fields. We present \textbf{Agentic AutoSurvey}, a multi-agent framework for automated survey generation that addresses fundamental limitations in existing approaches. Our system employs four specialized agents (Paper Search Specialist, Topic Mining \& Clustering, Academic Survey Writer, and Quality Evaluator) working in concert to generate comprehensive literature surveys with superior synthesis quality. Through experiments on six representative LLM research topics from COLM 2024 categories, we demonstrate that our multi-agent approach achieves significant improvements over existing baselines, scoring 8.18/10 compared to AutoSurvey's 4.77/10. The multi-agent architecture processes 75--443 papers per topic (847 total across six topics) while targeting high citation coverage (often $\geq$80\% on 75--100-paper sets; lower on very large sets such as RLHF) through specialized agent orchestration. Our 12-dimension evaluation captures organization, synthesis integration, and critical analysis beyond basic metrics. These findings demonstrate that multi-agent architectures represent a meaningful advancement for automated literature survey generation in rapidly evolving scientific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18658v1">Analyzing Uncertainty of LLM-as-a-Judge: Interval Evaluations with Conformal Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 To appear in EMNLP 2025. Our code and data are available at \url{https://github.com/BruceSheng1202/Analyzing_Uncertainty_of_LLM-as-a-Judge
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge has become a promising paradigm for using large language models (LLMs) to evaluate natural language generation (NLG), but the uncertainty of its evaluation remains underexplored. This lack of reliability may limit its deployment in many applications. This work presents the first framework to analyze the uncertainty by offering a prediction interval of LLM-based scoring via conformal prediction. Conformal prediction constructs continuous prediction intervals from a single evaluation run, and we design an ordinal boundary adjustment for discrete rating tasks. We also suggest a midpoint-based score within the interval as a low-bias alternative to raw model score and weighted average. We perform extensive experiments and analysis, which show that conformal prediction can provide valid prediction interval with coverage guarantees. We also explore the usefulness of interval midpoint and judge reprompting for better judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03293v2">LogicGuard: Improving Embodied LLM agents through Temporal Logic based Critics</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Modified version of prior LTLCrit work with new robotics dataset
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in zero-shot and single step reasoning and decision making problems, but in long horizon sequential planning tasks, their errors compound, often leading to unreliable or inefficient behavior. We introduce LogicGuard, a modular actor-critic architecture in which an LLM actor is guided by a trajectory level LLM critic that communicates through Linear Temporal Logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. LogicGuard supports both fixed safety rules and adaptive, learned constraints, and is model-agnostic: any LLM-based planner can serve as the actor, with LogicGuard acting as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LogicGuard to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. To demonstrate generality, we evaluate LogicGuard across two distinct settings: short-horizon general tasks and long-horizon specialist tasks. On the Behavior benchmark of 100 household tasks, LogicGuard increases task completion rates by 25% over a baseline InnerMonologue planner. On the Minecraft diamond-mining task, which is long-horizon and requires multiple interdependent subgoals, LogicGuard improves both efficiency and safety compared to SayCan and InnerMonologue. These results show that enabling LLMs to supervise each other through temporal logic yields more reliable, efficient and safe decision-making for both embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19470v3">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18575v1">The Ranking Blind Spot: Decision Hijacking in LLM-based Text Ranking</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong performance in information retrieval tasks like passage ranking. Our research examines how instruction-following capabilities in LLMs interact with multi-document comparison tasks, identifying what we term the "Ranking Blind Spot", a characteristic of LLM decision processes during comparative evaluation. We analyze how this ranking blind spot affects LLM evaluation systems through two approaches: Decision Objective Hijacking, which alters the evaluation goal in pairwise ranking systems, and Decision Criteria Hijacking, which modifies relevance standards across ranking schemes. These approaches demonstrate how content providers could potentially influence LLM-based ranking systems to affect document positioning. These attacks aim to force the LLM ranker to prefer a specific passage and rank it at the top. Malicious content providers can exploit this weakness, which helps them gain additional exposure by attacking the ranker. In our experiment, We empirically show that the proposed attacks are effective in various LLMs and can be generalized to multiple ranking schemes. We apply these attack to realistic examples to show their effectiveness. We also found stronger LLMs are more vulnerable to these attacks. Our code is available at: https://github.com/blindspotorg/RankingBlindSpot
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18569v1">Explore the Reinforcement Learning for the LLM based ASR and TTS system</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have played an important role in automatic speech recognition (ASR) and text-to-speech (TTS) systems. While reinforcement learning (RL) has significantly enhanced LLM performance in text-based tasks, its application to ASR and TTS remains underexplored due to the complexity of training audio-based models. In this study, we propose a lightweight RL framework tailored for audio-based LLMs that can process audio inputs and generate audio outputs. Based on this framework, we evaluate the effectiveness of reinforcement learning on both ASR and TTS tasks. For the ASR task, we experiment with different rule-based reward functions within the Group Relative Policy Optimization (GRPO) framework and investigate the impact of RL data construction. For the TTS task, we compare GRPO with Differentiable Reward Optimization (DiffRO) and further combine the two approaches to achieve improved performance. Our experiments demonstrate that RL can significantly enhance the performance of both ASR and TTS systems, even with limited training data and a small number of optimization steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18557v1">LLMZ+: Contextual Prompt Whitelist Principles for Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 7 pages, 5 figures, to be published and presented at ICMLA 2025
    </div>
    <details class="paper-abstract">
      Compared to traditional models, agentic AI represents a highly valuable target for potential attackers as they possess privileged access to data sources and API tools, which are traditionally not incorporated into classical agents. Unlike a typical software application residing in a Demilitarized Zone (DMZ), agentic LLMs consciously rely on nondeterministic behavior of the AI (only defining a final goal, leaving the path selection to LLM). This characteristic introduces substantial security risk to both operational security and information security. Most common existing defense mechanism rely on detection of malicious intent and preventing it from reaching the LLM agent, thus protecting against jailbreak attacks such as prompt injection. In this paper, we present an alternative approach, LLMZ+, which moves beyond traditional detection-based approaches by implementing prompt whitelisting. Through this method, only contextually appropriate and safe messages are permitted to interact with the agentic LLM. By leveraging the specificity of context, LLMZ+ guarantees that all exchanges between external users and the LLM conform to predefined use cases and operational boundaries. Our approach streamlines the security framework, enhances its long-term resilience, and reduces the resources required for sustaining LLM information security. Our empirical evaluation demonstrates that LLMZ+ provides strong resilience against the most common jailbreak prompts. At the same time, legitimate business communications are not disrupted, and authorized traffic flows seamlessly between users and the agentic LLM. We measure the effectiveness of approach using false positive and false negative rates, both of which can be reduced to 0 in our experimental setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16516v2">LLM-Guided Co-Training for Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a novel weighted co-training approach that is guided by Large Language Models (LLMs). Namely, in our co-training approach, we use LLM labels on unlabeled data as target labels and co-train two encoder-only based networks that train each other over multiple iterations: first, all samples are forwarded through each network and historical estimates of each network's confidence in the LLM label are recorded; second, a dynamic importance weight is derived for each sample according to each network's belief in the quality of the LLM label for that sample; finally, the two networks exchange importance weights with each other -- each network back-propagates all samples weighted with the importance weights coming from its peer network and updates its own parameters. By strategically utilizing LLM-generated guidance, our approach significantly outperforms conventional SSL methods, particularly in settings with abundant unlabeled data. Empirical results show that it achieves state-of-the-art performance on 4 out of 5 benchmark datasets and ranks first among 14 compared methods according to the Friedman test. Our results highlight a new direction in semi-supervised learning -- where LLMs serve as knowledge amplifiers, enabling backbone co-training models to achieve state-of-the-art performance efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05401v4">Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted at Findings of 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
    </div>
    <details class="paper-abstract">
      Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Meta (previously known as Facebook) advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. Additionally, we conduct a comprehensive fairness analysis to uncover biases in model predictions. We assess disparities in accuracy and error rates across demographic groups using established fairness metrics such as Demographic Parity, Equal Opportunity, and Predictive Equality. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of male audiences. The analysis of thematic explanations uncovers recurring patterns in messaging strategies tailored to various demographic groups, while the fairness analysis underscores the need for more inclusive targeting methods. This study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23817v2">MVDRAM: Enabling GeMV Execution in Unmodified DRAM for Low-Bit LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      General matrix-vector multiplication (GeMV) remains a critical latency bottleneck in large language model (LLM) inference, even with quantized low-bit models. Processing-Using-DRAM (PUD), an analog in-DRAM computing technique, has the potential to repurpose on-device DRAM as a GeMV engine, offering additional high-throughput processing capabilities to widespread consumer devices without DRAM modifications. However, applying PUD to GeMV operations in the LLM inference pipeline incurs significant overheads $\textit{before}$ and $\textit{after}$ in-DRAM computation, diminishing the benefits of its high-throughput processing capabilities. This paper presents MVDRAM, the first practical system to accelerate GeMV operations for low-bit LLM inference using unmodified DRAM. By leveraging the data sharing patterns and mathematical linearity in GeMV operations, MVDRAM orchestrates the processor and DRAM to eliminate the costs associated with pre-arranging inputs and bit-transposition of outputs required in conventional PUD approaches. Our experimental evaluation with four DDR4 DRAM modules shows that MVDRAM achieves comparable or even better inference speed than the processor-based implementation for GeMV operations in low-bit (under 4-bit) LLM. In particular, MVDRAM achieves up to 7.29$\times$ speedup and 30.5$\times$ energy efficiency for low-bit GeMV operations. For end-to-end LLM inference, MVDRAM achieves 2.18$\times$ and 1.31$\times$ throughput improvements, along with 3.04$\times$ and 2.35$\times$ energy efficiency, for 2-bit and 4-bit quantized low-bit models, respectively. MVDRAM has the potential to redefine the AI hardware landscape by demonstrating the feasibility of standard DRAM as an LLM accelerator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18487v1">Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for text-rich graph machine learning tasks such as node classification in high-impact domains like fraud detection and recommendation systems. Yet, despite a surge of interest, the field lacks a principled understanding of the capabilities of LLMs in their interaction with graph data. In this work, we conduct a large-scale, controlled evaluation across several key axes of variability to systematically assess the strengths and weaknesses of LLM-based graph reasoning methods in text-based applications. The axes include the LLM-graph interaction mode, comparing prompting, tool-use, and code generation; dataset domains, spanning citation, web-link, e-commerce, and social networks; structural regimes contrasting homophilic and heterophilic graphs; feature characteristics involving both short- and long-text node attributes; and model configurations with varying LLM sizes and reasoning capabilities. We further analyze dependencies by methodically truncating features, deleting edges, and removing labels to quantify reliance on input types. Our findings provide practical and actionable guidance. (1) LLMs as code generators achieve the strongest overall performance on graph data, with especially large gains on long-text or high-degree graphs where prompting quickly exceeds the token budget. (2) All interaction strategies remain effective on heterophilic graphs, challenging the assumption that LLM-based methods collapse under low homophily. (3) Code generation is able to flexibly adapt its reliance between structure, features, or labels to leverage the most informative input type. Together, these findings provide a comprehensive view of the strengths and limitations of current LLM-graph interaction modes and highlight key design principles for future approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19631v1">Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19552v1">iFinder: Structured Zero-Shot Vision-Based LLM Grounding for Dash-Cam Video Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Grounding large language models (LLMs) in domain-specific tasks like post-hoc dash-cam driving video analysis is challenging due to their general-purpose training and lack of structured inductive biases. As vision is often the sole modality available for such analysis (i.e., no LiDAR, GPS, etc.), existing video-based vision-language models (V-VLMs) struggle with spatial reasoning, causal inference, and explainability of events in the input video. To this end, we introduce iFinder, a structured semantic grounding framework that decouples perception from reasoning by translating dash-cam videos into a hierarchical, interpretable data structure for LLMs. iFinder operates as a modular, training-free pipeline that employs pretrained vision models to extract critical cues -- object pose, lane positions, and object trajectories -- which are hierarchically organized into frame- and video-level structures. Combined with a three-block prompting strategy, it enables step-wise, grounded reasoning for the LLM to refine a peer V-VLM's outputs and provide accurate reasoning. Evaluations on four public dash-cam video benchmarks show that iFinder's proposed grounding with domain-specific cues, especially object orientation and global context, significantly outperforms end-to-end V-VLMs on four zero-shot driving benchmarks, with up to 39% gains in accident reasoning accuracy. By grounding LLMs with driving domain-specific representations, iFinder offers a zero-shot, interpretable, and reliable alternative to end-to-end V-VLMs for post-hoc driving video understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20194v2">DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted to NeurIPS 2025. Camera-ready version will be updated when available. The code is available on Github (see hyperlink in the paper)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver strong performance but are difficult to deploy due to high memory and compute costs. While pruning reduces these demands, most methods ignore activation sparsity observed at runtime. We reinterpret activation sparsity as dynamic structured weight sparsity and propose DuoGPT, a unified framework that constructs dual-sparse (spMspV) workloads by combining unstructured weight pruning with activation sparsity. To preserve accuracy, we extend the Optimal Brain Compression (OBC) framework with activation-aware calibration and introduce output residuals from the dense model as correction terms. We further optimize the solution for efficient GPU execution, enabling scalability to billion-parameter LLMs. Evaluations on LLaMA-2 and LLaMA-3 show that DuoGPT outperforms state-of-the-art structured pruning methods by up to 9.17% accuracy at an iso-speedup of 1.39$\times$ compared to the baseline dense model. Code is available at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19533v1">Semantic-Aware Fuzzing: An Empirical Framework for LLM-Guided, Reasoning-Driven Input Mutation</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Security vulnerabilities in Internet-of-Things devices, mobile platforms, and autonomous systems remain critical. Traditional mutation-based fuzzers -- while effectively explore code paths -- primarily perform byte- or bit-level edits without semantic reasoning. Coverage-guided tools such as AFL++ use dictionaries, grammars, and splicing heuristics to impose shallow structural constraints, leaving deeper protocol logic, inter-field dependencies, and domain-specific semantics unaddressed. Conversely, reasoning-capable large language models (LLMs) can leverage pretraining knowledge to understand input formats, respect complex constraints, and propose targeted mutations, much like an experienced reverse engineer or testing expert. However, lacking ground truth for "correct" mutation reasoning makes supervised fine-tuning impractical, motivating explorations of off-the-shelf LLMs via prompt-based few-shot learning. To bridge this gap, we present an open-source microservices framework that integrates reasoning LLMs with AFL++ on Google's FuzzBench, tackling asynchronous execution and divergent hardware demands (GPU- vs. CPU-intensive) of LLMs and fuzzers. We evaluate four research questions: (R1) How can reasoning LLMs be integrated into the fuzzing mutation loop? (R2) Do few-shot prompts yield higher-quality mutations than zero-shot? (R3) Can prompt engineering with off-the-shelf models improve fuzzing directly? and (R4) Which open-source reasoning LLMs perform best under prompt-only conditions? Experiments with Llama3.3, Deepseek-r1-Distill-Llama-70B, QwQ-32B, and Gemma3 highlight Deepseek as the most promising. Mutation effectiveness depends more on prompt complexity and model choice than shot count. Response latency and throughput bottlenecks remain key obstacles, offering directions for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17314v2">Clotho: Measuring Task-Specific Pre-Generation Test Adequacy for LLM Inputs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Software increasingly relies on the emergent capabilities of Large Language Models (LLMs), from natural language understanding to program analysis and generation. Yet testing them on specific tasks remains difficult and costly: many prompts lack ground truth, forcing reliance on human judgment, while existing uncertainty and adequacy measures typically require full inference. A key challenge is to assess input adequacy in a way that reflects the demands of the task, ideally before even generating any output. We introduce CLOTHO, a task-specific, pre-generation adequacy measure that estimates input difficulty directly from hidden LLM states. Given a large pool of unlabelled inputs for a specific task, CLOTHO uses a Gaussian Mixture Model (GMM) to adaptively sample the most informative cases for human labelling. Based on this reference set the GMM can then rank unseen inputs by their likelihood of failure. In our empirical evaluation across eight benchmark tasks and three open-weight LLMs, CLOTHO can predict failures with a ROC-AUC of 0.716, after labelling reference sets that are on average only 5.4% of inputs. It does so without generating any outputs, thereby reducing costs compared to existing uncertainty measures. Comparison of CLOTHO and post-generation uncertainty measures shows that the two approaches complement each other. Crucially, we show that adequacy scores learnt from open-weight LLMs transfer effectively to proprietary models, extending the applicability of the approach. When prioritising test inputs for proprietary models, CLOTHO increases the average number of failing inputs from 18.7 to 42.5 out of 100, compared to random prioritisation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19489v1">Estimating the Self-Consistency of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Systems often repeat the same prompt to large language models (LLMs) and aggregate responses to improve reliability. This short note analyzes an estimator of the self-consistency of LLMs and the tradeoffs it induces under a fixed compute budget $B=mn$, where $m$ is the number of prompts sampled from the task distribution and $n$ is the number of repeated LLM calls per prompt; the resulting analysis favors a rough split $m,n\propto\sqrt{B}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v6">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18961v2">Weaver: Interweaving SQL and LLM for Table Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Querying tables with unstructured data is challenging due to the presence of text (or image), either embedded in the table or in external paragraphs, which traditional SQL struggles to process, especially for tasks requiring semantic reasoning. While Large Language Models (LLMs) excel at understanding context, they face limitations with long input sequences. Existing approaches that combine SQL and LLMs typically rely on rigid, predefined work-flows, limiting their adaptability to complex queries. To address these issues, we introduce Weaver , a modular pipeline that dynamically integrates SQL and LLMs for table-based question answering (TableQA). Weaver generates a flexible, step-by-step plan that combines SQL for structured data retrieval with LLMs for semantic processing. By decomposing complex queries into manageable subtasks, Weaver improves accuracy and generalization. Our experiments show that Weaver consistently outperforms state-of-the-art methods across four TableQA datasets, reducing both API calls and error rates. The code, along with other associated scripts, are available at https://coral-lab-asu.github.io/weaver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18058v2">Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are crafted to be subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using them as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19269v1">Extracting Conceptual Spaces from LLMs Using Prototype Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Conceptual spaces represent entities and concepts using cognitively meaningful dimensions, typically referring to perceptual features. Such representations are widely used in cognitive science and have the potential to serve as a cornerstone for explainable AI. Unfortunately, they have proven notoriously difficult to learn, although recent LLMs appear to capture the required perceptual features to a remarkable extent. Nonetheless, practical methods for extracting the corresponding conceptual spaces are currently still lacking. While various methods exist for extracting embeddings from LLMs, extracting conceptual spaces also requires us to encode the underlying features. In this paper, we propose a strategy in which features (e.g. sweetness) are encoded by embedding the description of a corresponding prototype (e.g. a very sweet food). To improve this strategy, we fine-tune the LLM to align the prototype embeddings with the corresponding conceptual space dimensions. Our empirical analysis finds this approach to be highly effective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15433v2">LLM-Driven SAST-Genius: A Hybrid Static Analysis Framework for Comprehensive and Actionable Security</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      This report examines the synergy between Large Language Models (LLMs) and Static Application Security Testing (SAST) to improve vulnerability discovery. Traditional SAST tools, while effective for proactive security, are limited by high false-positive rates and a lack of contextual understanding. Conversely, LLMs excel at code analysis and pattern recognition but can be prone to inconsistencies and hallucinations. By integrating these two technologies, a more intelligent and efficient system is created. This combination moves beyond mere vulnerability detection optimization, transforming security into a deeply integrated, contextual process that provides tangible benefits like improved triage, dynamic bug descriptions, bug validation via exploit generation and enhanced analysis of complex codebases. The result is a more effective security approach that leverages the strengths of both technologies while mitigating their weaknesses. SAST-Genius reduced false positives by about 91 % (225 to 20) compared to Semgrep alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19265v1">Cross-Cultural Transfer of Commonsense Reasoning in LLMs: Evidence from the Arab World</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 EMNLP 2025 - Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often reflect Western-centric biases, limiting their effectiveness in diverse cultural contexts. Although some work has explored cultural alignment, the potential for cross-cultural transfer, using alignment in one culture to improve performance in others, remains underexplored. This paper investigates cross-cultural transfer of commonsense reasoning in the Arab world, where linguistic and historical similarities coexist with local cultural differences. Using a culturally grounded commonsense reasoning dataset covering 13 Arab countries, we evaluate lightweight alignment methods such as in-context learning and demonstration-based reinforcement (DITTO), alongside baselines like supervised fine-tuning and direct preference optimization. Our results show that merely 12 culture-specific examples from one country can improve performance in others by 10\% on average, within multilingual models. In addition, we demonstrate that out-of-culture demonstrations from Indonesia and US contexts can match or surpass in-culture alignment for MCQ reasoning, highlighting cultural commonsense transferability beyond the Arab world. These findings demonstrate that efficient cross-cultural alignment is possible and offer a promising approach to adapt LLMs to low-resource cultural settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08727v3">Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 ICCV 2025, Project page: https://boyangdeng.com/visual-chronicles , second and third listed authors have equal contributions
    </div>
    <details class="paper-abstract">
      We present a system using Multimodal LLMs (MLLMs) to analyze a large database with tens of millions of images captured at different times, with the aim of discovering patterns in temporal changes. Specifically, we aim to capture frequent co-occurring changes ("trends") across a city over a certain period. Unlike previous visual analyses, our analysis answers open-ended queries (e.g., "what are the frequent types of changes in the city?") without any predetermined target subjects or training labels. These properties cast prior learning-based or unsupervised visual analysis tools unsuitable. We identify MLLMs as a novel tool for their open-ended semantic understanding capabilities. Yet, our datasets are four orders of magnitude too large for an MLLM to ingest as context. So we introduce a bottom-up procedure that decomposes the massive visual analysis problem into more tractable sub-problems. We carefully design MLLM-based solutions to each sub-problem. During experiments and ablation studies with our system, we find it significantly outperforms baselines and is able to discover interesting trends from images captured in large cities (e.g., "addition of outdoor dining,", "overpass was painted blue," etc.). See more results and interactive demos at https://boyangdeng.com/visual-chronicles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11593v2">Improving Image Captioning Descriptiveness by Ranking and LLM-based Fusion</a></div>
    <div class="paper-meta">
      📅 2025-09-23
      | 💬 This manuscript has been accepted for publication in Springer Neural Computing and Applications
    </div>
    <details class="paper-abstract">
      State-of-The-Art (SoTA) image captioning models are often trained on the MicroSoft Common Objects in Context (MS-COCO) dataset, which contains human-annotated captions with an average length of approximately ten tokens. Although effective for general scene understanding, these short captions often fail to capture complex scenes and convey detailed information. Moreover, captioning models tend to exhibit bias towards the ``average'' caption, which captures only the more general aspects, thus overlooking finer details. In this paper, we present a novel approach to generate richer and more informative image captions by combining the captions generated from different SoTA captioning models. Our proposed method requires no additional model training: given an image, it leverages pre-trained models from the literature to generate the initial captions, and then ranks them using a newly introduced image-text-based metric, which we name BLIPScore. Subsequently, the top two captions are fused using a Large Language Model (LLM) to produce the final, more detailed description. Experimental results on the MS-COCO and Flickr30k test sets demonstrate the effectiveness of our approach in terms of caption-image alignment and hallucination reduction according to the ALOHa, CAPTURE, and Polos metrics. A subjective study lends additional support to these results, suggesting that the captions produced by our model are generally perceived as more consistent with human judgment. By combining the strengths of diverse SoTA models, our method enhances the quality and appeal of image captions, bridging the gap between automated systems and the rich and informative nature of human-generated descriptions. This advance enables the generation of more suitable captions for the training of both vision-language and captioning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19153v1">LLMs as verification oracles for Solidity</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Ensuring the correctness of smart contracts is critical, as even subtle flaws can lead to severe financial losses. While bug detection tools able to spot common vulnerability patterns can serve as a first line of defense, most real-world exploits and losses stem from errors in the contract business logic. Formal verification tools such as SolCMC and the Certora Prover address this challenge, but their impact remains limited by steep learning curves and restricted specification languages. Recent works have begun to explore the use of large language models (LLMs) for security-related tasks such as vulnerability detection and test generation. Yet, a fundamental question remains open: can LLMs serve as verification oracles, capable of reasoning about arbitrary contract-specific properties? In this paper, we provide the first systematic evaluation of GPT-5, a state-of-the-art reasoning LLM, in this role. We benchmark its performance on a large dataset of verification tasks, compare its outputs against those of established formal verification tools, and assess its practical effectiveness in real-world auditing scenarios. Our study combines quantitative metrics with qualitative analysis, and shows that recent reasoning-oriented LLMs can be surprisingly effective as verification oracles, suggesting a new frontier in the convergence of AI and formal methods for secure smart contract development and auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19136v1">On the Soundness and Consistency of LLM Agents for Executing Test Cases Written in Natural Language</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      The use of natural language (NL) test cases for validating graphical user interface (GUI) applications is emerging as a promising direction to manually written executable test scripts, which are costly to develop and difficult to maintain. Recent advances in large language models (LLMs) have opened the possibility of the direct execution of NL test cases by LLM agents. This paper investigates this direction, focusing on the impact on NL test case unsoundness and on test case execution consistency. NL test cases are inherently unsound, as they may yield false failures due to ambiguous instructions or unpredictable agent behaviour. Furthermore, repeated executions of the same NL test case may lead to inconsistent outcomes, undermining test reliability. To address these challenges, we propose an algorithm for executing NL test cases with guardrail mechanisms and specialised agents that dynamically verify the correct execution of each test step. We introduce measures to evaluate the capabilities of LLMs in test execution and one measure to quantify execution consistency. We propose a definition of weak unsoundness to characterise contexts in which NL test case execution remains acceptable, with respect to the industrial quality levels Six Sigma. Our experimental evaluation with eight publicly available LLMs, ranging from 3B to 70B parameters, demonstrates both the potential and current limitations of current LLM agents for GUI testing. Our experiments show that Meta Llama 3.1 70B demonstrates acceptable capabilities in NL test case execution with high execution consistency (above the level 3-sigma). We provide prototype tools, test suites, and results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11189v2">Can Global XAI Methods Reveal Injected Bias in LLMs? SHAP vs Rule Extraction vs RuleSHAP</a></div>
    <div class="paper-meta">
      📅 2025-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can amplify misinformation, undermining societal goals like the UN SDGs. We study three documented drivers of misinformation (valence framing, information overload, and oversimplification) which are often shaped by one's default beliefs. Building on evidence that LLMs encode such defaults (e.g., "joy is positive," "math is complex") and can act as "bags of heuristics," we ask: can general belief-driven heuristics behind misinformative behaviour be recovered from LLMs as clear rules? A key obstacle is that global rule-extraction methods in explainable AI (XAI) are built for numerical inputs/outputs, not text. We address this by eliciting global LLM beliefs and mapping them to numerical scores via statistically reliable abstractions, thereby enabling off-the-shelf global XAI to detect belief-related heuristics in LLMs. To obtain ground truth, we hard-code bias-inducing nonlinear heuristics of increasing complexity (univariate, conjunctive, nonconvex) into popular LLMs (ChatGPT and Llama) via system instructions. This way, we find that RuleFit under-detects non-univariate biases, while global SHAP better approximates conjunctive ones but does not yield actionable rules. To bridge this gap, we propose RuleSHAP, a rule-extraction algorithm that couples global SHAP-value aggregations with rule induction to better capture non-univariate bias, improving heuristics detection over RuleFit by +94% (MRR@1) on average. Our results provide a practical pathway for revealing belief-driven biases in LLMs.
    </details>
</div>
