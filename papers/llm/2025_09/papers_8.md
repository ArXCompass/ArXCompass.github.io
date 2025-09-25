# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22286v2">Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 6 pages, 3 figures, Accepted for publication at the Second International Workshop on Construction Grammars and NLP at the 16th International Conference for Computational Semantics (IWCS) 2025
    </div>
    <details class="paper-abstract">
      The usage-based constructionist (UCx) approach to language posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze representations of the English Double Object (DO) and Prepositional Object (PO) constructions in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied by human-rated preference strength for DO or PO. Geometric analyses show that the separability between the two constructions' representations, as measured by energy distance or Jensen-Shannon divergence, is systematically modulated by gradient preference strength, which depends on lexical and functional properties of sentences. That is, more prototypical exemplars of each construction occupy more distinct regions in activation space, compared to sentences that could have equally well have occured in either construction. These results provide evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures for representations in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13905v2">Spec2RTL-Agent: Automated Hardware Code Generation from Complex Specifications Using LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Despite recent progress in generating hardware RTL code with LLMs, existing solutions still suffer from a substantial gap between practical application scenarios and the requirements of real-world RTL code development. Prior approaches either focus on overly simplified hardware descriptions or depend on extensive human guidance to process complex specifications, limiting their scalability and automation potential. In this paper, we address this gap by proposing an LLM agent system, termed Spec2RTL-Agent, designed to directly process complex specification documentation and generate corresponding RTL code implementations, advancing LLM-based RTL code generation toward more realistic application settings. To achieve this goal, Spec2RTL-Agent introduces a novel multi-agent collaboration framework that integrates three key enablers: (1) a reasoning and understanding module that translates specifications into structured, step-by-step implementation plans; (2) a progressive coding and prompt optimization module that iteratively refines the code across multiple representations to enhance correctness and synthesisability for RTL conversion; and (3) an adaptive reflection module that identifies and traces the source of errors during generation, ensuring a more robust code generation flow. Instead of directly generating RTL from natural language, our system strategically generates synthesizable C++ code, which is then optimized for HLS. This agent-driven refinement ensures greater correctness and compatibility compared to naive direct RTL generation approaches. We evaluate Spec2RTL-Agent on three specification documents, showing it generates accurate RTL code with up to 75% fewer human interventions than existing methods. This highlights its role as the first fully automated multi-agent system for RTL generation from unstructured specs, reducing reliance on human effort in hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24539v3">Localizing Persona Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 To appear in the AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025
    </div>
    <details class="paper-abstract">
      We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17674v2">Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06948v1">Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach limits interaction between SFT and RL, thereby constraining overall effectiveness. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06941v1">Outcome-based Exploration for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 26 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a powerful method for improving the reasoning abilities of large language models (LLMs). Outcome-based RL, which rewards policies solely for the correctness of the final answer, yields substantial accuracy gains but also induces a systematic loss in generation diversity. This collapse undermines real-world performance, where diversity is critical for test-time scaling. We analyze this phenomenon by viewing RL post-training as a sampling process and show that, strikingly, RL can reduce effective diversity even on the training set relative to the base model. Our study highlights two central findings: (i) a transfer of diversity degradation, where reduced diversity on solved problems propagates to unsolved ones, and (ii) the tractability of the outcome space, since reasoning tasks admit only a limited set of distinct answers. Motivated by these insights, we propose outcome-based exploration, which assigns exploration bonuses according to final outcomes. We introduce two complementary algorithms: historical exploration, which encourages rarely observed answers via UCB-style bonuses, and batch exploration, which penalizes within-batch repetition to promote test-time diversity. Experiments on standard competition math with Llama and Qwen models demonstrate that both methods improve accuracy while mitigating diversity collapse. On the theoretical side, we formalize the benefit of outcome-based exploration through a new model of outcome-based bandits. Together, these contributions chart a practical path toward RL methods that enhance reasoning without sacrificing the diversity essential for scalable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06920v1">An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 6 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including precision, recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03377v2">Amplifying Effective CXL Memory Bandwidth for LLM Inference via Transparent Near-Data Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference is bottlenecked by the limited bandwidth of CXL-based memory used for capacity expansion. We introduce CXL-NDP, a transparent near-data processing architecture that amplifies effective CXL bandwidth without requiring changes to the CXL.mem interface or AI models. CXL-NDP integrates a precision-scalable bit-plane layout for dynamic quantization with transparent lossless compression of weights and KV caches directly within the CXL device. In end-to-end serving, CXL-NDP improves throughput by 43%, extends the maximum context length by 87%, and reduces the KV cache footprint by 46.9% without accuracy loss. Hardware synthesis confirms its practicality with a modest silicon footprint, lowering the barrier for adopting efficient, scalable CXL-based memory in generative AI infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06902v1">Proof-Carrying Numbers (PCN): A Protocol for Trustworthy Numeric Answers from LLMs via Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) as stochastic systems may generate numbers that deviate from available data, a failure known as \emph{numeric hallucination}. Existing safeguards -- retrieval-augmented generation, citations, and uncertainty estimation -- improve transparency but cannot guarantee fidelity: fabricated or misquoted values may still be displayed as if correct. We propose \textbf{Proof-Carrying Numbers (PCN)}, a presentation-layer protocol that enforces numeric fidelity through mechanical verification. Under PCN, numeric spans are emitted as \emph{claim-bound tokens} tied to structured claims, and a verifier checks each token under a declared policy (e.g., exact equality, rounding, aliases, or tolerance with qualifiers). Crucially, PCN places verification in the \emph{renderer}, not the model: only claim-checked numbers are marked as verified, and all others default to unverified. This separation prevents spoofing and guarantees fail-closed behavior. We formalize PCN and prove soundness, completeness under honest tokens, fail-closed behavior, and monotonicity under policy refinement. PCN is lightweight and model-agnostic, integrates seamlessly into existing applications, and can be extended with cryptographic commitments. By enforcing verification as a mandatory step before display, PCN establishes a simple contract for numerically sensitive settings: \emph{trust is earned only by proof}, while the absence of a mark communicates uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20521v2">Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 28 pages, 5 figures. Submitted for review to Information Fusion
    </div>
    <details class="paper-abstract">
      This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02019v2">ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Domain-Specific Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Computational Fluid Dynamics (CFD) is essential for advancing scientific and engineering fields but is hindered by operational complexity, high expertise requirements, and limited accessibility. This paper introduces ChatCFD, an automated agent system for OpenFOAM simulations that processes multi-modal inputs (e.g., research papers, meshes) via an interactive interface, leveraging DeepSeek-R1 and DeepSeek-V3 large language models, a multi-agent architecture, and OpenFOAM knowledge. Its four-stage pipeline (Knowledge Base Construction, User Input Processing, Case File Generation, and Execution and Error Reflection) enables iterative trial-reflection-refinement for intricate setups, supporting diverse physical models and external meshes. Validation on 205 benchmark tutorial cases, 110 perturbed variants, and 2 literature-derived cases shows ChatCFD's 82.1 percent operational success rate on basic cases, outperforming MetaOpenFOAM (6.2 percent) and Foam-Agent (42.3 percent), and 60-80 percent on literature-derived complex cases. Turbulence model studies show a 40 percent success rate for common models versus 10 percent for rare ones like RNG k-epsilon. Physics coupling analyses reveal higher resource demands for multi-physics-coupled cases, while LLM bias toward simpler setups introduces persistent errors, such as dimensional inconsistency. Ablation studies highlight the efficacy of RAG-based modules and reflection mechanisms. By automating hypothesis testing and parameter exploration, ChatCFD accelerates scientific discovery in fluid mechanics and engineering, addressing LLM limitations through structured design and showing strong potential as a modular component in MCP-based agent networks for collaborative multi-agent systems, paving the way for scalable AI-driven CFD innovation. The code for ChatCFD is available at https://github.com/ConMoo/ChatCFD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16482v2">Self-Alignment: Improving Alignment of Cultural Values in LLMs via In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Improving the alignment of Large Language Models (LLMs) with respect to the cultural values that they encode has become an increasingly important topic. In this work, we study whether we can exploit existing knowledge about cultural values at inference time to adjust model responses to cultural value probes. We present a simple and inexpensive method that uses a combination of in-context learning (ICL) and human survey data, and show that we can improve the alignment to cultural values across 5 models that include both English-centric and multilingual LLMs. Importantly, we show that our method could prove useful in test languages other than English and can improve alignment to the cultural values that correspond to a range of culturally diverse countries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06822v1">RAFFLES: Reasoning-based Attribution of Faults for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      We have reached a critical roadblock in the development and enhancement of long-horizon, multi-component LLM agentic systems: it is incredibly tricky to identify where these systems break down and why. Evaluation capabilities that currently exist today (e.g., single pass LLM-as-a-judge) are limited in that they often focus on individual metrics or capabilities, end-to-end outcomes, and are narrowly grounded on the preferences of humans. We argue that to match the agentic capabilities, evaluation frameworks must also be able to reason, probe, iterate, and understand the complex logic passing through these systems over long horizons. In this paper, we present RAFFLES - an evaluation architecture that incorporates reasoning and iterative refinement. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically investigate faults and a set of specialized Evaluators to assess not only the system's components but also the quality of the reasoning by the Judge itself, thereby building a history of hypotheses. We tested RAFFLES against several baselines on the Who&When dataset, a benchmark designed to diagnose the "who" (agent) and "when" (step) of a system's failure. RAFFLES outperforms these baselines, achieving an agent-step fault pair accuracy of over 43% on the Algorithmically-Generated dataset (a substantial increase from the previously published best of 16.6%) and over 20% on the Hand-Crafted dataset (surpassing the previously published best of 8.8%). These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual human review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06809v1">Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      The scarcity of high-quality, logically sound data is a critical bottleneck for advancing the mathematical reasoning of Large Language Models (LLMs). Our work confronts this challenge by turning decades of automated theorem proving research into a scalable data engine. Rather than relying on error-prone LLMs or complex proof-assistant syntax like Lean and Isabelle, our framework leverages E-prover's saturation capabilities on the vast TPTP axiom library to derive a massive, guaranteed-valid corpus of theorems. Our pipeline is principled and simple: saturate axioms, filter for "interesting" theorems, and generate tasks. With no LLMs in the loop, we eliminate factual errors by construction. This purely symbolic data is then transformed into three difficulty-controlled challenges: entailment verification, premise selection, and proof reconstruction. Our zero-shot experiments on frontier models reveal a clear weakness: performance collapses on tasks requiring deep, structural reasoning. Our framework provides both the diagnostic tool to measure this gap and a scalable source of symbolic training data to address it. We make the code and data publicly available. https://github.com/sileod/reasoning_core https://hf.co/datasets/reasoning-core/rc1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06770v1">Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative iteration.Together, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06734v1">Intelligent Manufacturing Support: Specialized LLMs for Composite Material Processing and Equipment Operation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Engineering educational curriculum and standards cover many material and manufacturing options. However, engineers and designers are often unfamiliar with certain composite materials or manufacturing techniques. Large language models (LLMs) could potentially bridge the gap. Their capacity to store and retrieve data from large databases provides them with a breadth of knowledge across disciplines. However, their generalized knowledge base can lack targeted, industry-specific knowledge. To this end, we present two LLM-based applications based on the GPT-4 architecture: (1) The Composites Guide: a system that provides expert knowledge on composites material and connects users with research and industry professionals who can provide additional support and (2) The Equipment Assistant: a system that provides guidance for manufacturing tool operation and material characterization. By combining the knowledge of general AI models with industry-specific knowledge, both applications are intended to provide more meaningful information for engineers. In this paper, we discuss the development of the applications and evaluate it through a benchmark and two informal user studies. The benchmark analysis uses the Rouge and Bertscore metrics to evaluate our model performance against GPT-4o. The results show that GPT-4o and the proposed models perform similarly or better on the ROUGE and BERTScore metrics. The two user studies supplement this quantitative evaluation by asking experts to provide qualitative and open-ended feedback about our model performance on a set of domain-specific questions. The results of both studies highlight a potential for more detailed and specific responses with the Composites Guide and the Equipment Assistant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07642v2">TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Accepted to EMNLP2025 Main
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19576v2">ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 21 pages, 4 figures
    </div>
    <details class="paper-abstract">
      With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We conduct extensive experiments on coding problems to verify the validity of the proposed RL paradigm. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at https://github.com/THUDM/ReST-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v3">Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06595v1">LLMs in Cybersecurity: Friend or Foe in the Human Decision Loop?</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming human decision-making by acting as cognitive collaborators. Yet, this promise comes with a paradox: while LLMs can improve accuracy, they may also erode independent reasoning, promote over-reliance and homogenize decisions. In this paper, we investigate how LLMs shape human judgment in security-critical contexts. Through two exploratory focus groups (unaided and LLM-supported), we assess decision accuracy, behavioral resilience and reliance dynamics. Our findings reveal that while LLMs enhance accuracy and consistency in routine decisions, they can inadvertently reduce cognitive diversity and improve automation bias, which is especially the case among users with lower resilience. In contrast, high-resilience individuals leverage LLMs more effectively, suggesting that cognitive traits mediate AI benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06586v1">Simulating Dispute Mediation with LLM-Based Agents for Legal Research</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Legal dispute mediation plays a crucial role in resolving civil disputes, yet its empirical study is limited by privacy constraints and complex multivariate interactions. To address this limitation, we present AgentMediation, the first LLM-based agent framework for simulating dispute mediation. It simulates realistic mediation processes grounded in real-world disputes and enables controlled experimentation on key variables such as disputant strategies, dispute causes, and mediator expertise. Our empirical analysis reveals patterns consistent with sociological theories, including Group Polarization and Surface-level Consensus. As a comprehensive and extensible platform, AgentMediation paves the way for deeper integration of social science and AI in legal research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04537v2">Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03646v2">Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08532v2">Safe and Economical UAV Trajectory Planning in Low-Altitude Airspace: A Hybrid DRL-LLM Approach with Compliance Awareness</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      The rapid growth of the low-altitude economy has driven the widespread adoption of unmanned aerial vehicles (UAVs). This growing deployment presents new challenges for UAV trajectory planning in complex urban environments. However, existing studies often overlook key factors, such as urban airspace constraints and economic efficiency, which are essential in low-altitude economy contexts. Deep reinforcement learning (DRL) is regarded as a promising solution to these issues, while its practical adoption remains limited by low learning efficiency. To overcome this limitation, we propose a novel UAV trajectory planning framework that combines DRL with large language model (LLM) reasoning to enable safe, compliant, and economically viable path planning. Experimental results demonstrate that our method significantly outperforms existing baselines across multiple metrics, including data collection rate, collision avoidance, successful landing, regulatory compliance, and energy efficiency. These results validate the effectiveness of our approach in addressing UAV trajectory planning key challenges under constraints of the low-altitude economy networking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16748v3">Through the Prism of Culture: Evaluating LLMs' Understanding of Indian Subcultures and Traditions</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements but also raise concerns about cultural bias, often reflecting dominant narratives at the expense of under-represented subcultures. In this study, we evaluate the capacity of LLMs to recognize and accurately respond to the Little Traditions within Indian society, encompassing localized cultural practices and subcultures such as caste, kinship, marriage, and religion. Through a series of case studies, we assess whether LLMs can balance the interplay between dominant Great Traditions and localized Little Traditions. We explore various prompting strategies and further investigate whether using prompts in regional languages enhances the models cultural sensitivity and response quality. Our findings reveal that while LLMs demonstrate an ability to articulate cultural nuances, they often struggle to apply this understanding in practical, context-specific scenarios. To the best of our knowledge, this is the first study to analyze LLMs engagement with Indian subcultures, offering critical insights into the challenges of embedding cultural diversity in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06053v1">PolicyEvolve: Evolving Programmatic Policies by LLMs for multi-player games via Population-Based Training</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      Multi-agent reinforcement learning (MARL) has achieved significant progress in solving complex multi-player games through self-play. However, training effective adversarial policies requires millions of experience samples and substantial computational resources. Moreover, these policies lack interpretability, hindering their practical deployment. Recently, researchers have successfully leveraged Large Language Models (LLMs) to generate programmatic policies for single-agent tasks, transforming neural network-based policies into interpretable rule-based code with high execution efficiency. Inspired by this, we propose PolicyEvolve, a general framework for generating programmatic policies in multi-player games. PolicyEvolve significantly reduces reliance on manually crafted policy code, achieving high-performance policies with minimal environmental interactions. The framework comprises four modules: Global Pool, Local Pool, Policy Planner, and Trajectory Critic. The Global Pool preserves elite policies accumulated during iterative training. The Local Pool stores temporary policies for the current iteration; only sufficiently high-performing policies from this pool are promoted to the Global Pool. The Policy Planner serves as the core policy generation module. It samples the top three policies from the Global Pool, generates an initial policy for the current iteration based on environmental information, and refines this policy using feedback from the Trajectory Critic. Refined policies are then deposited into the Local Pool. This iterative process continues until the policy achieves a sufficiently high average win rate against the Global Pool, at which point it is integrated into the Global Pool. The Trajectory Critic analyzes interaction data from the current policy, identifies vulnerabilities, and proposes directional improvements to guide the Policy Planner
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18791v2">CodeMixBench: Evaluating Code-Mixing Capabilities of LLMs Across 18 Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Code-mixing, the practice of switching between languages within a conversation, poses unique challenges for traditional NLP. Existing benchmarks are limited by their narrow language pairs and tasks, failing to adequately assess large language models' (LLMs) code-mixing abilities. Despite the recognized importance of code-mixing for multilingual users, research on LLMs in this context remains sparse. Additionally, current techniques for synthesizing code-mixed data are underdeveloped to generate code-mixing. In response, we introduce CodeMixBench, a comprehensive benchmark covering eight tasks, including three specific to LLMs and five traditional NLP tasks, and 18 languages across seven language families. We also propose a new method for generating large-scale synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our evaluation reveals consistent underperformance of LLMs on code-mixed datasets involving different language families. Enhancements in training data size, model scale, and few-shot learning could improve their performance. The code and dataset are available at https://github.com/Jeromeyluck/CodeMixBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11689v2">Improve LLM-as-a-Judge Ability as a General Ability</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge leverages the generative and reasoning capabilities of large language models (LLMs) to evaluate LLM responses across diverse scenarios, providing accurate preference signals. This approach plays a vital role in aligning LLMs with human values, ensuring ethical and reliable AI outputs that align with societal norms. Recent studies have raised many methods to train LLM as generative judges, but most of them are data consuming or lack accuracy, and only focus on LLM's judge ability. In this work, we regard judge ability as a general ability of LLM and implement a two-stage training approach, comprising supervised fine-tuning (SFT) warm-up and direct preference optimization (DPO) enhancement, to achieve judge style adaptation and improve judgment accuracy. Additionally, we introduce an efficient data synthesis method to generate judgmental content. Experimental results demonstrate that our approach, utilizing only about 2% to 40% of the data required by other methods, achieves SOTA performance on RewardBench. Furthermore, our training method enhances the general capabilities of the model by constructing complicated judge task, and the judge signals provided by our model have significantly enhanced the downstream DPO training performance of our internal models in our test to optimize policy model with Judge Model. We also open-source our model weights and training data to facilitate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11210v2">Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05995v1">Students' Perception of LLM Use in Requirements Engineering Education: An Empirical Study Across Two Universities</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 Accepted by the 33rd IEEE International Requirements Engineering 2025 (RE'25), Valencia, Spain, September 1-5, 2025
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) in Requirements Engineering (RE) education is reshaping pedagogical approaches, seeking to enhance student engagement and motivation while providing practical tools to support their professional future. This study empirically evaluates the impact of integrating LLMs in RE coursework. We examined how the guided use of LLMs influenced students' learning experiences, and what benefits and challenges they perceived in using LLMs in RE practices. The study collected survey data from 179 students across two RE courses in two universities. LLMs were integrated into coursework through different instructional formats, i.e., individual assignments versus a team-based Agile project. Our findings indicate that LLMs improved students' comprehension of RE concepts, particularly in tasks like requirements elicitation and documentation. However, students raised concerns about LLMs in education, including academic integrity, overreliance on AI, and challenges in integrating AI-generated content into assignments. Students who worked on individual assignments perceived that they benefited more than those who worked on team-based assignments, highlighting the importance of contextual AI integration. This study offers recommendations for the effective integration of LLMs in RE education. It proposes future research directions for balancing AI-assisted learning with critical thinking and collaborative practices in RE courses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05962v1">The Reel Deal: Designing and Evaluating LLM-Generated Short-Form Educational Videos</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 9 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Short-form videos are gaining popularity in education due to their concise and accessible format that enables microlearning. Yet, most of these videos are manually created. Even for those automatically generated using artificial intelligence (AI), it is not well understood whether or how they affect learning outcomes, user experience, and trust. To address this gap, we developed ReelsEd, which is a web-based system that uses large language models (LLMs) to automatically generate structured short-form video (i.e., reels) from lecture long-form videos while preserving instructor-authored material. In a between-subject user study with 62 university students, we evaluated ReelsEd and demonstrated that it outperformed traditional long-form videos in engagement, quiz performance, and task efficiency without increasing cognitive load. Learners expressed high trust in our system and valued its clarity, usefulness, and ease of navigation. Our findings point to new design opportunities for integrating generative AI into educational tools that prioritize usability, learner agency, and pedagogical alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05936v1">ALPHA: LLM-Enabled Active Learning for Human-Free Network Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 Accepted at 44th IEEE International Performance Computing and Communications Conference (IPCCC 2025)
    </div>
    <details class="paper-abstract">
      Network log data analysis plays a critical role in detecting security threats and operational anomalies. Traditional log analysis methods for anomaly detection and root cause analysis rely heavily on expert knowledge or fully supervised learning models, both of which require extensive labeled data and significant human effort. To address these challenges, we propose ALPHA, the first Active Learning Pipeline for Human-free log Analysis. ALPHA integrates semantic embedding, clustering-based representative sampling, and large language model (LLM)-assisted few-shot annotation to automate the anomaly detection process. The LLM annotated labels are propagated across clusters, enabling large-scale training of an anomaly detector with minimal supervision. To enhance the annotation accuracy, we propose a two-step few-shot refinement strategy that adaptively selects informative prompts based on the LLM's observed error patterns. Extensive experiments on real-world log datasets demonstrate that ALPHA achieves detection accuracy comparable to fully supervised methods while mitigating human efforts in the loop. ALPHA also supports interpretable analysis through LLM-driven root cause explanations in the post-detection stage. These capabilities make ALPHA a scalable and cost-efficient solution for truly automated log-based anomaly detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00054v2">Robotic Fire Risk Detection based on Dynamic Knowledge Graph Reasoning: An LLM-Driven Approach with Graph Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 We have decided to withdraw this paper as the work is still undergoing further refinement. To ensure the clarity of the results, we prefer to make additional improvements before resubmission. We appreciate the readers' understanding
    </div>
    <details class="paper-abstract">
      Fire is a highly destructive disaster, but effective prevention can significantly reduce its likelihood of occurrence. When it happens, deploying emergency robots in fire-risk scenarios can help minimize the danger to human responders. However, current research on pre-disaster warnings and disaster-time rescue still faces significant challenges due to incomplete perception, inadequate fire situational awareness, and delayed response. To enhance intelligent perception and response planning for robots in fire scenarios, we first construct a knowledge graph (KG) by leveraging large language models (LLMs) to integrate fire domain knowledge derived from fire prevention guidelines and fire rescue task information from robotic emergency response documents. We then propose a new framework called Insights-on-Graph (IOG), which integrates the structured fire information of KG and Large Multimodal Models (LMMs). The framework generates perception-driven risk graphs from real-time scene imagery to enable early fire risk detection and provide interpretable emergency responses for task module and robot component configuration based on the evolving risk situation. Extensive simulations and real-world experiments show that IOG has good applicability and practical application value in fire risk detection and rescue decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05899v1">X-SQL: Expert Schema Linking and Understanding of Text-to-SQL with Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      With Large Language Models' (LLMs) emergent abilities on code generation tasks, Text-to-SQL has become one of the most popular downstream applications. Despite the strong results of multiple recent LLM-based Text-to-SQL frameworks, the research community often overlooks the importance of database schema information for generating high-quality SQL queries. We find that such schema information plays a significant or even dominant role in the Text-to-SQL task. To tackle this challenge, we propose a novel database schema expert with two components. We first introduce X-Linking, an LLM Supervised Finetuning (SFT)-based method that achieves superior Schema Linking results compared to existing open-source Text-to-SQL methods. In addition, we innovatively propose an X-Admin component that focuses on Schema Understanding by bridging the gap between abstract schema information and the user's natural language question. Aside from better learning with schema information, we experiment with Multi-LLMs for different components within the system to further boost its performance. By incorporating these techniques into our end-to-end framework, X-SQL, we have achieved Execution Accuracies of 84.9% on the Spider-Dev dataset and 82.5% on the Spider-Test dataset. This outstanding performance establishes X-SQL as the leading Text-to-SQL framework based on open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05883v1">Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 8 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks. To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05882v1">Let's Roleplay: Examining LLM Alignment in Collaborative Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) integrate into diverse workflows, they are increasingly being considered "collaborators" with humans. If such AI collaborators are to be reliable, their behavior over multiturn interactions must be predictable, validated and verified before deployment. Common alignment techniques are typically developed under simplified single-user settings and do not account for the dynamics of long-horizon multiparty interactions. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multiturn, multiparty collaborations. We study this question through the lens of friction agents that intervene in group dialogues to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Using a roleplay methodology, we evaluate interventions from differently-trained friction agents in collaborative task conversations. We propose a novel counterfactual evaluation framework that quantifies how friction interventions change the trajectory of group collaboration and belief alignment. Our results show that a friction-aware approach significantly outperforms common alignment baselines in helping both convergence to a common ground, or agreed-upon task-relevant propositions, and correctness of task outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06235v1">PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 for the source code, see https://github.com/aialt/PillagerBench
    </div>
    <details class="paper-abstract">
      LLM-based agents have shown promise in various cooperative and strategic reasoning tasks, but their effectiveness in competitive multi-agent environments remains underexplored. To address this gap, we introduce PillagerBench, a novel framework for evaluating multi-agent systems in real-time competitive team-vs-team scenarios in Minecraft. It provides an extensible API, multi-round testing, and rule-based built-in opponents for fair, reproducible comparisons. We also propose TactiCrafter, an LLM-based multi-agent system that facilitates teamwork through human-readable tactics, learns causal dependencies, and adapts to opponent strategies. Our evaluation demonstrates that TactiCrafter outperforms baseline approaches and showcases adaptive learning through self-play. Additionally, we analyze its learning process and strategic evolution over multiple game episodes. To encourage further research, we have open-sourced PillagerBench, fostering advancements in multi-agent AI for competitive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12611v2">An LLM + ASP Workflow for Joint Entity-Relation Extraction</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 13 pages, 1 figure, Accepted as Technical Communication, 41st International Conference on Logic Programming
    </div>
    <details class="paper-abstract">
      Joint entity-relation extraction (JERE) identifies both entities and their relationships simultaneously. Traditional machine-learning based approaches to performing this task require a large corpus of annotated data and lack the ability to easily incorporate domain specific information in the construction of the model. Therefore, creating a model for JERE is often labor intensive, time consuming, and elaboration intolerant. In this paper, we propose harnessing the capabilities of generative pretrained large language models (LLMs) and the knowledge representation and reasoning capabilities of Answer Set Programming (ASP) to perform JERE. We present a generic workflow for JERE using LLMs and ASP. The workflow is generic in the sense that it can be applied for JERE in any domain. It takes advantage of LLM's capability in natural language understanding in that it works directly with unannotated text. It exploits the elaboration tolerant feature of ASP in that no modification of its core program is required when additional domain specific knowledge, in the form of type specifications, is found and needs to be used. We demonstrate the usefulness of the proposed workflow through experiments with limited training data on three well-known benchmarks for JERE. The results of our experiments show that the LLM + ASP workflow is better than state-of-the-art JERE systems in several categories with only 10\% of training data. It is able to achieve a 2.5 times (35\% over 15\%) improvement in the Relation Extraction task for the SciERC corpus, one of the most difficult benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06200v1">MSLEF: Multi-Segment LLM Ensemble Finetuning in Recruitment</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 Accepted in AICCSA 2025
    </div>
    <details class="paper-abstract">
      This paper presents MSLEF, a multi-segment ensemble framework that employs LLM fine-tuning to enhance resume parsing in recruitment automation. It integrates fine-tuned Large Language Models (LLMs) using weighted voting, with each model specializing in a specific resume segment to boost accuracy. Building on MLAR , MSLEF introduces a segment-aware architecture that leverages field-specific weighting tailored to each resume part, effectively overcoming the limitations of single-model systems by adapting to diverse formats and structures. The framework incorporates Gemini-2.5-Flash LLM as a high-level aggregator for complex sections and utilizes Gemma 9B, LLaMA 3.1 8B, and Phi-4 14B. MSLEF achieves significant improvements in Exact Match (EM), F1 score, BLEU, ROUGE, and Recruitment Similarity (RS) metrics, outperforming the best single model by up to +7% in RS. Its segment-aware design enhances generalization across varied resume layouts, making it highly adaptable to real-world hiring scenarios while ensuring precise and reliable candidate representation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06196v1">Augmented Fine-Tuned LLMs for Enhanced Recruitment Automation</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 Accepted in AICCSA 2025
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to recruitment automation. Large Language Models (LLMs) were fine-tuned to improve accuracy and efficiency. Building upon our previous work on the Multilayer Large Language Model-Based Robotic Process Automation Applicant Tracking (MLAR) system . This work introduces a novel methodology. Training fine-tuned LLMs specifically tuned for recruitment tasks. The proposed framework addresses the limitations of generic LLMs by creating a synthetic dataset that uses a standardized JSON format. This helps ensure consistency and scalability. In addition to the synthetic data set, the resumes were parsed using DeepSeek, a high-parameter LLM. The resumes were parsed into the same structured JSON format and placed in the training set. This will help improve data diversity and realism. Through experimentation, we demonstrate significant improvements in performance metrics, such as exact match, F1 score, BLEU score, ROUGE score, and overall similarity compared to base models and other state-of-the-art LLMs. In particular, the fine-tuned Phi-4 model achieved the highest F1 score of 90.62%, indicating exceptional precision and recall in recruitment tasks. This study highlights the potential of fine-tuned LLMs. Furthermore, it will revolutionize recruitment workflows by providing more accurate candidate-job matching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06174v1">From Long to Short: LLMs Excel at Trimming Own Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 21 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      O1/R1 style large reasoning models (LRMs) signal a substantial leap forward over conventional instruction-following LLMs. By applying test-time scaling to generate extended reasoning paths, they establish many SOTAs across a wide range of complex reasoning tasks. However, recent studies show that LRMs are prone to suffer from overthinking -- the tendency to overcomplicate simple problems, leading to excessive strategy switching and long, convoluted reasoning traces that hinder their interpretability. To mitigate this issue, we conduct a systematic investigation into the reasoning efficiency of a broad set of LRMs and uncover a common dilemma: the difficulty in balancing multiple generation objectives such as correctness and brevity. Based on this discovery, we propose a test-time scaling method, EDIT (Efficient Dynamic Inference Trimming), which efficiently guides LRMs to identify the shortest correct reasoning paths at test time. EDIT employs constraint-guided generation while jointly tracking length and answer distributions under varying constraints, allowing it to select responses that strike an optimal balance between conciseness and correctness. Extensive experiments across diverse models and datasets show that EDIT substantially enhance the reasoning efficiency, producing compact yet informative outputs that improve readability and user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05541v2">SenseCF: LLM-Prompted Counterfactuals for Intervention and Sensor Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-09-07
      | 💬 Accepted at the IEEE-EMBS International Conference on Body Sensor Networks (IEEE-EMBS BSN) 2025, LA, CA, USA
    </div>
    <details class="paper-abstract">
      Counterfactual explanations (CFs) offer human-centric insights into machine learning predictions by highlighting minimal changes required to alter an outcome. Therefore, CFs can be used as (i) interventions for abnormality prevention and (ii) augmented data for training robust models. In this work, we explore large language models (LLMs), specifically GPT-4o-mini, for generating CFs in a zero-shot and three-shot setting. We evaluate our approach on two datasets: the AI-Readi flagship dataset for stress prediction and a public dataset for heart disease detection. Compared to traditional methods such as DiCE, CFNOW, and NICE, our few-shot LLM-based approach achieves high plausibility (up to 99%), strong validity (up to 0.99), and competitive sparsity. Moreover, using LLM-generated CFs as augmented samples improves downstream classifier performance (an average accuracy gain of 5%), especially in low-data regimes. This demonstrates the potential of prompt-based generative techniques to enhance explainability and robustness in clinical and physiological prediction tasks. Code base: github.com/shovito66/SenseCF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08847v1">Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-07
    </div>
    <details class="paper-abstract">
      This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v5">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05831v1">Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05830v1">Finetuning LLMs for Human Behavior Prediction in Social Science Experiments</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer a powerful opportunity to simulate the results of social science experiments. In this work, we demonstrate that finetuning LLMs directly on individual-level responses from past experiments meaningfully improves the accuracy of such simulations across diverse social science domains. We construct SocSci210 via an automatic pipeline, a dataset comprising 2.9 million responses from 400,491 participants in 210 open-source social science experiments. Through finetuning, we achieve multiple levels of generalization. In completely unseen studies, our strongest model, Socrates-Qwen-14B, produces predictions that are 26% more aligned with distributions of human responses to diverse outcome questions under varying conditions relative to its base model (Qwen2.5-14B), outperforming GPT-4o by 13%. By finetuning on a subset of conditions in a study, generalization to new unseen conditions is particularly robust, improving by 71%. Since SocSci210 contains rich demographic information, we reduce demographic parity, a measure of bias, by 10.6% through finetuning. Because social sciences routinely generate rich, topic-specific datasets, our findings indicate that finetuning on such data could enable more accurate simulations for experimental hypothesis screening. We release our data, models and finetuning code at stanfordhci.github.io/socrates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05467v2">Position: LLMs Can be Good Tutors in English Education</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 Accepted to EMNLP 2025 Main. 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While recent efforts have begun integrating large language models (LLMs) into English education, they often rely on traditional approaches to learning tasks without fully embracing educational methodologies, thus lacking adaptability to language learning. To address this gap, we argue that LLMs have the potential to serve as effective tutors in English Education. Specifically, LLMs can play three critical roles: (1) as data enhancers, improving the creation of learning materials or serving as student simulations; (2) as task predictors, serving as learner assessment or optimizing learning pathway; and (3) as agents, enabling personalized and inclusive education. We encourage interdisciplinary research to explore these roles, fostering innovation while addressing challenges and risks, ultimately advancing English Education through the thoughtful integration of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18665v2">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05764v1">DRF: LLM-AGENT Dynamic Reputation Filtering Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 This paper has been accepted by ICONIP 2025 but not published
    </div>
    <details class="paper-abstract">
      With the evolution of generative AI, multi - agent systems leveraging large - language models(LLMs) have emerged as a powerful tool for complex tasks. However, these systems face challenges in quantifying agent performance and lack mechanisms to assess agent credibility. To address these issues, we introduce DRF, a dynamic reputation filtering framework. DRF constructs an interactive rating network to quantify agent performance, designs a reputation scoring mechanism to measure agent honesty and capability, and integrates an Upper Confidence Bound - based strategy to enhance agent selection efficiency. Experiments show that DRF significantly improves task completion quality and collaboration efficiency in logical reasoning and code - generation tasks, offering a new approach for multi - agent systems to handle large - scale tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05755v1">Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05751v1">Unleashing Hierarchical Reasoning: An LLM-Driven Framework for Training-Free Referring Video Object Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Referring Video Object Segmentation (RVOS) aims to segment an object of interest throughout a video based on a language description. The prominent challenge lies in aligning static text with dynamic visual content, particularly when objects exhibiting similar appearances with inconsistent motion and poses. However, current methods often rely on a holistic visual-language fusion that struggles with complex, compositional descriptions. In this paper, we propose \textbf{PARSE-VOS}, a novel, training-free framework powered by Large Language Models (LLMs), for a hierarchical, coarse-to-fine reasoning across text and video domains. Our approach begins by parsing the natural language query into structured semantic commands. Next, we introduce a spatio-temporal grounding module that generates all candidate trajectories for all potential target objects, guided by the parsed semantics. Finally, a hierarchical identification module select the correct target through a two-stage reasoning process: it first performs coarse-grained motion reasoning with an LLM to narrow down candidates; if ambiguity remains, a fine-grained pose verification stage is conditionally triggered to disambiguate. The final output is an accurate segmentation mask for the target object. \textbf{PARSE-VOS} achieved state-of-the-art performance on three major benchmarks: Ref-YouTube-VOS, Ref-DAVIS17, and MeViS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05741v1">Enhancing Factual Accuracy and Citation Generation in LLMs via Multi-Stage Self-Verification</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      This research introduces VeriFact-CoT (Verified Factual Chain-of-Thought), a novel method designed to address the pervasive issues of hallucination and the absence of credible citation sources in Large Language Models (LLMs) when generating complex, fact-sensitive content. By incorporating a multi-stage mechanism of 'fact verification-reflection-citation integration,' VeriFact-CoT empowers LLMs to critically self-examine and revise their intermediate reasoning steps and final answers. This process significantly enhances the objective accuracy, trustworthiness, and traceability of the generated outputs, making LLMs more reliable for applications demanding high fidelity such as scientific research, news reporting, and legal consultation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05657v1">LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 EMNLP2025
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00882v2">VulSolver: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, limitations in context length, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 96.29% accuracy, 96.55% F1-score, and 100% recall. Applied to popular GitHub repositories, VULSOLVER also identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20653v2">ComplexVCoder: An LLM-Driven Framework for Systematic Generation of Complex Verilog Code</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 Withdrawn due to an error in the experimental setup that affected the results. A corrected version is in progress
    </div>
    <details class="paper-abstract">
      Recent advances have demonstrated the promising capabilities of large language models (LLMs) in generating register-transfer level (RTL) code, such as Verilog. However, existing LLM-based frameworks still face significant challenges in accurately handling the complexity of real-world RTL designs, particularly those that are large-scale and involve multi-level module instantiations. To address this issue, we present ComplexVCoder, an open-source LLM-driven framework that enhances both the generation quality and efficiency of complex Verilog code. Specifically, we introduce a two-stage generation mechanism, which leverages an intermediate representation to enable a more accurate and structured transition from natural language descriptions to intricate Verilog designs. In addition, we introduce a rule-based alignment method and a domain-specific retrieval-augmented generation (RAG) to further improve the correctness of the synthesized code by incorporating relevant design knowledge during generation. To evaluate our approach, we construct a comprehensive dataset comprising 55 complex Verilog designs derived from real-world implementations. We also release an open-source benchmark suite for systematically assessing the quality of auto-generated RTL code together with the ComplexVCoder framework. Experimental results show that ComplexVCoder outperforms SOTA frameworks such as CodeV and RTLCoder by 14.6% and 22.2%, respectively, in terms of function correctness on complex Verilog benchmarks. Furthermore, ComplexVcoder achieves comparable generation performances in terms of functionality correctness using a lightweight 32B model (Qwen2.5), rivaling larger-scale models such as GPT-3.5 and DeepSeek-V3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05608v1">Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      The widespread deployment of LLMs across enterprise services has created a critical security blind spot. Organizations operate multiple LLM services handling billions of queries daily, yet regulatory compliance boundaries prevent these services from sharing threat intelligence about prompt injection attacks, the top security risk for LLMs. When an attack is detected in one service, the same threat may persist undetected in others for months, as privacy regulations prohibit sharing user prompts across compliance boundaries. We present BinaryShield, the first privacy-preserving threat intelligence system that enables secure sharing of attack fingerprints across compliance boundaries. BinaryShield transforms suspicious prompts through a unique pipeline combining PII redaction, semantic embedding, binary quantization, and randomized response mechanism to potentially generate non-invertible fingerprints that preserve attack patterns while providing privacy. Our evaluations demonstrate that BinaryShield achieves an F1-score of 0.94, significantly outperforming SimHash (0.77), the privacy-preserving baseline, while achieving 64x storage reduction and 38x faster similarity search compared to dense embeddings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01386v2">ThinkEval: Practical Evaluation of Knowledge Leakage in LLM Editing using Thought-based Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Robust model-editing techniques are essential for deploying large language models (LLMs) in practical applications, to enable cost-effective ways to deal with challenges such as privacy breaches, bias mitigation and misinformation spread. For example, an LLM-based healthcare assistance may need to update out-dated or incorrect knowledge to prevent harmful recommendations. However, many editing techniques focus on isolated facts, which critically fail to prevent indirect knowledge leakage -- the unintended reconstruction of edited-out information through persistent causal links and contextual relationships. To assist users in selecting the right editing technique, we develop and present ThinkEval, a framework to systematically quantify indirect knowledge leakage and ripple effects in model-editing. ThinkEval builds and employs specialized knowledge graphs to analyze the causal structure of facts before and after editing. To support this approach, we present KnowGIC, a benchmark dataset comprising multi-step reasoning paths that precisely measure these complex knowledge transformation effects. We evaluate five editing techniques: AlphaEdit, RECT, ROME, MEMIT, and PRUNE across multiple LLMs. Our results show that these techniques struggle to balance indirect fact suppression with the preservation of related knowledge, compromising the contextual integrity of a model's knowledge. Our dataset is available at: https://anonymous.4open.science/r/KnowGIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08000v1">AntiDote: Bi-level Adversarial Training for Tamper-Resistant LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      The release of open-weight large language models (LLMs) creates a tension between advancing accessible research and preventing misuse, such as malicious fine-tuning to elicit harmful content. Current safety measures struggle to preserve the general capabilities of the LLM while resisting a determined adversary with full access to the model's weights and architecture, who can use full-parameter fine-tuning to erase existing safeguards. To address this, we introduce AntiDote, a bi-level optimization procedure for training LLMs to be resistant to such tampering. AntiDote involves an auxiliary adversary hypernetwork that learns to generate malicious Low-Rank Adaptation (LoRA) weights conditioned on the defender model's internal activations. The defender LLM is then trained with an objective to nullify the effect of these adversarial weight additions, forcing it to maintain its safety alignment. We validate this approach against a diverse suite of 52 red-teaming attacks, including jailbreak prompting, latent space manipulation, and direct weight-space attacks. AntiDote is upto 27.4\% more robust against adversarial attacks compared to both tamper-resistance and unlearning baselines. Crucially, this robustness is achieved with a minimal trade-off in utility, incurring a performance degradation of upto less than 0.5\% across capability benchmarks including MMLU, HellaSwag, and GSM8K. Our work offers a practical and compute efficient methodology for building open-weight models where safety is a more integral and resilient property.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04770v1">Research on Multi-hop Inference Optimization of LLM Based on MQUAKE Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Accurately answering complex questions has consistently been a significant challenge for Large Language Models (LLMs). To address this, this paper proposes a multi-hop question decomposition method for complex questions, building upon research within the MQUAKE framework. Utilizing the LLAMA3 model, we systematically investigate the impact of multi-hop question decomposition within knowledge graphs on model comprehension and reasoning accuracy, both before and after model training. In our experiments, we systematically partitioned and converted the MQUAKE-T dataset into two distinct formats: a single-hop dataset designed for directly answering complex questions, and a multi-hop dataset constructed using the multi-hop question decomposition method. We then fine-tuned the LLAMA3 model on these datasets and conducted inference tests. Our results demonstrate that, without fine-tuning the LLM, the prediction performance based on the multi-hop question decomposition method significantly outperforms the method of directly answering complex questions. After fine-tuning using the LoRA (Low-Rank Adaptation) method, the performance of both approaches improved compared to the untrained baseline. Crucially, the method utilizing multi-hop decomposition consistently maintained its superiority. These findings validate the effectiveness of the multi-hop decomposition method both before and after training, demonstrating its capability to effectively enhance the LLM's ability to answer complex questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03730v2">The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 We make public all code and source data at https://github.com/psychology-of-AI/Personality-Illusion for full reproducibility
    </div>
    <details class="paper-abstract">
      Personality traits have long been studied as predictors of human behavior. Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: (1) the dynamic emergence and evolution of trait profiles throughout training stages; (2) the predictive validity of self-reported traits in behavioral tasks; and (3) the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these self-reported traits do not reliably predict behavior, and observed associations often diverge from human patterns. While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v4">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Accepted at ICML 2025, Code at: https://github.com/uiuc-focal-lab/CRANE
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05946v3">InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware Control</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Model Predictive Control (MPC) is a powerful control strategy widely utilized in domains like energy management, building control, and autonomous systems. However, its effectiveness in real-world settings is challenged by the need to incorporate context-specific predictions and expert instructions, which traditional MPC often neglects. We propose InstructMPC, a novel framework that addresses this gap by integrating real-time human instructions through a Large Language Model (LLM) to produce context-aware predictions for MPC. Our method employs a Language-to-Distribution (L2D) module to translate contextual information into predictive disturbance trajectories, which are then incorporated into the MPC optimization. Unlike existing context-aware and language-based MPC models, InstructMPC enables dynamic human-LLM interaction and fine-tunes the L2D module in a closed loop with theoretical performance guarantees, achieving a regret bound of $O(\sqrt{T\log T})$ for linear dynamics when optimized via advanced fine-tuning methods such as Direct Preference Optimization (DPO) using a tailored loss function.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04714v1">ThumbnailTruth: A Multi-Modal LLM Approach for Detecting Misleading YouTube Thumbnails Across Diverse Cultural Settings</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Misleading video thumbnails on platforms like YouTube are a pervasive problem, undermining user trust and platform integrity. This paper proposes a novel multi-modal detection pipeline that uses Large Language Models (LLMs) to flag misleading thumbnails. We first construct a comprehensive dataset of 2,843 videos from eight countries, including 1,359 misleading thumbnail videos that collectively amassed over 7.6 billion views -- providing a unique cross-cultural perspective on this global issue. Our detection pipeline integrates video-to-text descriptions, thumbnail images, and subtitle transcripts to holistically analyze content and flag misleading thumbnails. Through extensive experimentation and prompt engineering, we evaluate the performance of state-of-the-art LLMs, including GPT-4o, GPT-4o Mini, Claude 3.5 Sonnet, and Gemini-1.5 Flash. Our findings show the effectiveness of LLMs in identifying misleading thumbnails, with Claude 3.5 Sonnet consistently showing strong performance, achieving an accuracy of 93.8\%, precision over 92\%, and recall exceeding 94\% in certain scenarios. We discuss the implications of our findings for content moderation, user experience, and the ethical considerations of deploying such systems at scale. Our findings pave the way for more transparent, trustworthy video platforms and stronger content integrity for audiences worldwide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05540v1">Combining TSL and LLM to Automate REST API Testing: A Comparative Study</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 10 pages, article computer science, software engineering, software testing, ia, llm
    </div>
    <details class="paper-abstract">
      The effective execution of tests for REST APIs remains a considerable challenge for development teams, driven by the inherent complexity of distributed systems, the multitude of possible scenarios, and the limited time available for test design. Exhaustive testing of all input combinations is impractical, often resulting in undetected failures, high manual effort, and limited test coverage. To address these issues, we introduce RestTSLLM, an approach that uses Test Specification Language (TSL) in conjunction with Large Language Models (LLMs) to automate the generation of test cases for REST APIs. The approach targets two core challenges: the creation of test scenarios and the definition of appropriate input data. The proposed solution integrates prompt engineering techniques with an automated pipeline to evaluate various LLMs on their ability to generate tests from OpenAPI specifications. The evaluation focused on metrics such as success rate, test coverage, and mutation score, enabling a systematic comparison of model performance. The results indicate that the best-performing LLMs - Claude 3.5 Sonnet (Anthropic), Deepseek R1 (Deepseek), Qwen 2.5 32b (Alibaba), and Sabia 3 (Maritaca) - consistently produced robust and contextually coherent REST API tests. Among them, Claude 3.5 Sonnet outperformed all other models across every metric, emerging in this study as the most suitable model for this task. These findings highlight the potential of LLMs to automate the generation of tests based on API specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05484v1">From Staff Messages to Actionable Insights: A Multi-Stage LLM Classification Framework for Healthcare Analytics</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Hospital call centers serve as the primary contact point for patients within a hospital system. They also generate substantial volumes of staff messages as navigators process patient requests and communicate with the hospital offices following the established protocol restrictions and guidelines. This continuously accumulated large amount of text data can be mined and processed to retrieve insights; however, traditional supervised learning approaches require annotated data, extensive training, and model tuning. Large Language Models (LLMs) offer a paradigm shift toward more computationally efficient methodologies for healthcare analytics. This paper presents a multi-stage LLM-based framework that identifies staff message topics and classifies messages by their reasons in a multi-class fashion. In the process, multiple LLM types, including reasoning, general-purpose, and lightweight models, were evaluated. The best-performing model was o3, achieving 78.4% weighted F1-score and 79.2% accuracy, followed closely by gpt-5 (75.3% Weighted F1-score and 76.2% accuracy). The proposed methodology incorporates data security measures and HIPAA compliance requirements essential for healthcare environments. The processed LLM outputs are integrated into a visualization decision support tool that transforms the staff messages into actionable insights accessible to healthcare professionals. This approach enables more efficient utilization of the collected staff messaging data, identifies navigator training opportunities, and supports improved patient experience and care quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05449v1">Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04439v2">ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. We evaluate on ARC-AGI, a benchmark that stresses compositional generalization and abstract reasoning, making it a natural fit for concept memory. Our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, dynamically updating memory during test-time outperforms fixed settings, supporting the hypothesis that accumulating and abstracting patterns enables further solutions in a form of self-improvement. Code is available at https://github.com/matt-seb-ho/arc_memo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20401v2">Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 In Proceedings of 2nd Workshop on Evaluating and Applying Recommendation Systems with Large Language Models (EARL) at RecSys 2025 (EARL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for recommendation tasks due to their general-purpose capabilities. While LLMs perform well in rich-context settings, their behavior in cold-start scenarios, where only limited signals such as age, gender, or language are available, raises fairness concerns because they may rely on societal biases encoded during pretraining. We introduce a benchmark specifically designed to evaluate fairness in zero-context recommendation. Our modular pipeline supports configurable recommendation domains and sensitive attributes, enabling systematic and flexible audits of any open-source LLM. Through evaluations of state-of-the-art models (Gemma 3 and Llama 3.2), we uncover consistent biases across recommendation domains (music, movies, and colleges) including gendered and cultural stereotypes. We also reveal a non-linear relationship between model size and fairness, highlighting the need for nuanced analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05291v1">Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07236v2">Simple Yet Effective: An Information-Theoretic Approach to Multi-LLM Uncertainty Quantification</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often behave inconsistently across inputs, indicating uncertainty and motivating the need for its quantification in high-stakes settings. Prior work on calibration and uncertainty quantification often focuses on individual models, overlooking the potential of model diversity. We hypothesize that LLMs make complementary predictions due to differences in training and the Zipfian nature of language, and that aggregating their outputs leads to more reliable uncertainty estimates. To leverage this, we propose MUSE (Multi-LLM Uncertainty via Subset Ensembles), a simple information-theoretic method that uses Jensen-Shannon Divergence to identify and aggregate well-calibrated subsets of LLMs. Experiments on binary prediction tasks demonstrate improved calibration and predictive performance compared to single-model and na\"ive ensemble baselines. In addition, we explore using MUSE as guided signals with chain-of-thought distillation to fine-tune LLMs for calibration. MUSE is available at:https://github.com/LARK-NLP-Lab/MUSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05528v2">Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 14 pages, accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06460v2">Can LLMs Simulate Personas with Reversed Performance? A Benchmark for Counterfactual Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now increasingly widely used to simulate personas in virtual environments, leveraging their instruction-following capability. However, we discovered that even state-of-the-art LLMs cannot simulate personas with reversed performance (e.g., student personas with low proficiency in educational settings), which impairs the simulation diversity and limits the practical applications of the simulated environments. In this work, using mathematical reasoning as a representative scenario, we propose the first benchmark dataset for evaluating LLMs on simulating personas with reversed performance, a capability that we dub "counterfactual instruction following". We evaluate both open-weight and closed-source LLMs on this task and find that LLMs, including the OpenAI o1 reasoning model, all struggle to follow counterfactual instructions for simulating reversedly performing personas. Intersectionally simulating both the performance level and the race population of a persona worsens the effect even further. These results highlight the challenges of counterfactual instruction following and the need for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22809v2">First Steps Towards Overhearing LLM Agents: A Case Study With Dungeons & Dragons Gameplay</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 9 pages, 5 figures. COLM 2025 Workshop on AI Agents
    </div>
    <details class="paper-abstract">
      Much work has been done on conversational LLM agents which directly assist human users with tasks. We present an alternative paradigm for interacting with LLM agents, which we call "overhearing agents". These overhearing agents do not actively participate in conversation -- instead, they "listen in" on human-to-human conversations and perform background tasks or provide suggestions to assist the user. In this work, we explore the overhearing agents paradigm through the lens of Dungeons & Dragons gameplay. We present an in-depth study using large multimodal audio-language models as overhearing agents to assist a Dungeon Master. We perform a human evaluation to examine the helpfulness of such agents and find that some large audio-language models have the emergent ability to perform overhearing agent tasks using implicit audio cues. Finally, we release Python libraries and our project code to support further research into the overhearing agents paradigm at https://github.com/zhudotexe/overhearing_agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18416v5">PersonaGym: Evaluating Persona Agents and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Persona agents, which are LLM agents conditioned to act according to an assigned persona, enable contextually rich and user aligned interactions across domains like education and healthcare. However, evaluating how faithfully these agents adhere to their personas remains a significant challenge, particularly in free-form settings that demand consistency across diverse, persona-relevant environments. We introduce PersonaGym, the first dynamic evaluation framework for persona agents, and PersonaScore, a human-aligned automatic metric grounded in decision theory that enables comprehensive large-scale evaluation. Our evaluation of 10 leading LLMs across 200 personas and 10,000 questions reveals significant advancement opportunities. For example, GPT-4.1 had the exact same PersonaScore as LLaMA-3-8b despite being a more recent and advanced closed source model. Importantly, increased model size and complexity do not necessarily enhance persona agent capabilities, underscoring the need for algorithmic and architectural innovation toward faithful, performant persona agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20938v2">ParEval-Repo: A Benchmark Suite for Evaluating LLMs with Repository-level HPC Translation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      GPGPU architectures have become significantly more diverse in recent years, which has led to an emergence of a variety of specialized programming models and software stacks to support them. Portable programming models exist, but they require significant developer effort to port to and optimize for different hardware architectures. Large language models (LLMs) may help to reduce this programmer burden. In this paper, we present a novel benchmark and testing framework, ParEval-Repo, which can be used to evaluate the efficacy of LLM-based approaches in automatically translating entire codebases across GPGPU execution models. ParEval-Repo includes several scientific computing and AI mini-applications in a range of programming models and levels of repository complexity. We use ParEval-Repo to evaluate a range of state-of-the-art open-source and commercial LLMs, with both a non-agentic and a top-down agentic approach. We assess code generated by the LLMs and approaches in terms of compilability, functional correctness, categories of build errors, and the cost of translation in terms of the number of inference tokens. Our results demonstrate that LLM translation of scientific applications is feasible for small programs but difficulty with generating functional build systems and cross-file dependencies pose challenges in scaling to larger codebases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05199v1">Triadic Fusion of Cognitive, Functional, and Causal Dimensions for Explainable LLMs: The TAXAL Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 27 pages, 9 tables and 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed in high-risk domains where opacity, bias, and instability undermine trust and accountability. Traditional explainability methods, focused on surface outputs, do not capture the reasoning pathways, planning logic, and systemic impacts of agentic LLMs. We introduce TAXAL (Triadic Alignment for eXplainability in Agentic LLMs), a triadic fusion framework that unites three complementary dimensions: cognitive (user understanding), functional (practical utility), and causal (faithful reasoning). TAXAL provides a unified, role-sensitive foundation for designing, evaluating, and deploying explanations in diverse sociotechnical settings. Our analysis synthesizes existing methods, ranging from post-hoc attribution and dialogic interfaces to explanation-aware prompting, and situates them within the TAXAL triadic fusion model. We further demonstrate its applicability through case studies in law, education, healthcare, and public services, showing how explanation strategies adapt to institutional constraints and stakeholder roles. By combining conceptual clarity with design patterns and deployment pathways, TAXAL advances explainability as a technical and sociotechnical practice, supporting trustworthy and context-sensitive LLM applications in the era of agentic AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00461v2">TECP: Token-Entropy Conformal Prediction for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) for open-ended language generation remains a critical yet underexplored challenge, especially under black-box constraints where internal model signals are inaccessible. In this paper, we introduce Token-Entropy Conformal Prediction (TECP), a novel framework that leverages token-level entropy as a logit-free, reference-free uncertainty measure and integrates it into a split conformal prediction (CP) pipeline to construct prediction sets with formal coverage guarantees. Unlike existing approaches that rely on semantic consistency heuristics or white-box features, TECP directly estimates epistemic uncertainty from the token entropy structure of sampled generations and calibrates uncertainty thresholds via CP quantiles to ensure provable error control. Empirical evaluations across six large language models and two benchmarks (CoQA and TriviaQA) demonstrate that TECP consistently achieves reliable coverage and compact prediction sets, outperforming prior self-consistency-based UQ methods. Our method provides a principled and efficient solution for trustworthy generation in black-box LLM settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05042v1">Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Shared autonomy is a promising paradigm in robotic systems, particularly within the maritime domain, where complex, high-risk, and uncertain environments necessitate effective human-robot collaboration. This paper investigates the interaction of three complementary approaches to advance shared autonomy in heterogeneous marine robotic fleets: (i) the integration of Large Language Models (LLMs) to facilitate intuitive high-level task specification and support hull inspection missions, (ii) the implementation of human-in-the-loop interaction frameworks in multi-agent settings to enable adaptive and intent-aware coordination, and (iii) the development of a modular Mission Manager based on Behavior Trees to provide interpretable and flexible mission control. Preliminary results from simulation and real-world lake-like environments demonstrate the potential of this multi-layered architecture to reduce operator cognitive load, enhance transparency, and improve adaptive behaviour alignment with human intent. Ongoing work focuses on fully integrating these components, refining coordination mechanisms, and validating the system in operational port scenarios. This study contributes to establishing a modular and scalable foundation for trustworthy, human-collaborative autonomy in safety-critical maritime robotics applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14387v2">Don't Make It Up: Preserving Ignorance Awareness in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Existing work on mitigating catastrophic forgetting during large language models (LLMs) fine-tuning for new knowledge instances has primarily focused on preserving performance on previously seen data, while critically overlooking the collapse of essential capabilities instilled through alignment, most notably the model's ability to faithfully express epistemic uncertainty (a property we term 'Ignorance Awareness'). In this work, we formalize the notion of Ignorance Awareness and illustrate that conventional fine-tuning methods can result in substantial activation displacement. This displacement undermines the critical capability of ignorance awareness, leading to undesirable behaviors such as hallucinations. To address this challenge, we introduce SEAT, a simple and principled fine-tuning approach that not only enables the model to effectively acquire new knowledge instances but also preserves its aligned ignorance awareness. SEAT integrates two key components: (1) sparse tuning that constrains activation drift, and (2) a novel entity perturbation method designed to counter knowledge entanglement. Experimental results demonstrate that, across both real-world and synthetic datasets, SEAT significantly outperforms baselines in preserving ignorance awareness while retaining optimal fine-tuning performance, offering a more robust solution for LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14506v3">InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real Toyota Human Support Robot and Anymal D-Unitree Z1 platforms. Our method achieves a 95% success rate in the 'fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art large-scale LLM-based robotics planners, while using only real-time onboard computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04365v3">AutoPDL: Automatic Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04993v1">LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 This paper has been accepted by IEEE Communications Magazine
    </div>
    <details class="paper-abstract">
      The ubiquitous computing resources in 6G networks provide ideal environments for the fusion of large language models (LLMs) and intelligent services through the agent framework. With auxiliary modules and planning cores, LLM-enabled agents can autonomously plan and take actions to deal with diverse environment semantics and user intentions. However, the limited resources of individual network devices significantly hinder the efficient operation of LLM-enabled agents with complex tool calls, highlighting the urgent need for efficient multi-level device collaborations. To this end, the framework and method of the LLM-enabled multi-agent system with dual-loop terminal-edge collaborations are proposed in 6G networks. Firstly, the outer loop consists of the iterative collaborations between the global agent and multiple sub-agents deployed on edge servers and terminals, where the planning capability is enhanced through task decomposition and parallel sub-task distribution. Secondly, the inner loop utilizes sub-agents with dedicated roles to circularly reason, execute, and replan the sub-task, and the parallel tool calling generation with offloading strategies is incorporated to improve efficiency. The improved task planning capability and task execution efficiency are validated through the conducted case study in 6G-supported urban safety governance. Finally, the open challenges and future directions are thoroughly analyzed in 6G networks, accelerating the advent of the 6G era.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17450v2">Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 To appear at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11987v3">FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Technical report, 51 pages. Update the results
    </div>
    <details class="paper-abstract">
      Future prediction is a complex task for LLM agents, requiring a high level of analytical thinking, information gathering, contextual understanding, and decision-making under uncertainty. Agents must not only gather and interpret vast amounts of dynamic information but also integrate diverse data sources, weigh uncertainties, and adapt predictions based on emerging trends, just as human experts do in fields like politics, economics, and finance. Despite its importance, no large-scale benchmark exists for evaluating agents on future prediction, largely due to challenges in handling real-time updates and retrieving timely, accurate answers. To address this, we introduce $\textbf{FutureX}$, a dynamic and live evaluation benchmark specifically designed for LLM agents performing future prediction tasks. FutureX is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection. We evaluate 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools such as the open-source Deep Research Agent and closed-source Deep Research models. This comprehensive evaluation assesses agents' adaptive reasoning and performance in dynamic environments. Additionally, we provide in-depth analyses of agents' failure modes and performance pitfalls in future-oriented tasks, including the vulnerability to fake web pages and the temporal validity. Our goal is to establish a dynamic, contamination-free evaluation standard that drives the development of LLM agents capable of performing at the level of professional human analysts in complex reasoning and predictive thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20541v4">MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12226v4">AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 28 pages, 16 figures, under review, work in progress
    </div>
    <details class="paper-abstract">
      We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04876v1">OSC: Cognitive Orchestration through Dynamic Knowledge Alignment in Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Accepted at EMNLP 2025 (Long Paper)
    </div>
    <details class="paper-abstract">
      This paper introduces OSC (Orchestrating Cognitive Synergy), a knowledge-aware adaptive collaboration framework designed to enhance cognitive synergy in multi-agent systems with large language models. While prior work has advanced agent selection and result aggregation, efficient linguistic interactions for deep collaboration among expert agents remain a critical bottleneck. OSC addresses this gap as a pivotal intermediate layer between selection and aggregation, introducing Collaborator Knowledge Models (CKM) to enable each agent to dynamically perceive its collaborators' cognitive states. Through real-time cognitive gap analysis, agents adaptively adjust communication behaviors, including content focus, detail level, and expression style, using learned strategies. Experiments on complex reasoning and problem-solving benchmarks demonstrate that OSC significantly improves task performance and communication efficiency, transforming "parallel-working individuals'' into a "deeply collaborative cognitive team.'' This framework not only optimizes multi-agent collaboration but also offers new insights into LLM agent interaction behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04868v1">Using LLMs for Multilingual Clinical Entity Linking to ICD-10</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 7 pages, 2 Figures, to be published in Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing, RANLP 2025
    </div>
    <details class="paper-abstract">
      The linking of clinical entities is a crucial part of extracting structured information from clinical texts. It is the process of assigning a code from a medical ontology or classification to a phrase in the text. The International Classification of Diseases - 10th revision (ICD-10) is an international standard for classifying diseases for statistical and insurance purposes. Automatically assigning the correct ICD-10 code to terms in discharge summaries will simplify the work of healthcare professionals and ensure consistent coding in hospitals. Our paper proposes an approach for linking clinical terms to ICD-10 codes in different languages using Large Language Models (LLMs). The approach consists of a multistage pipeline that uses clinical dictionaries to match unambiguous terms in the text and then applies in-context learning with GPT-4.1 to predict the ICD-10 code for the terms that do not match the dictionary. Our system shows promising results in predicting ICD-10 codes on different benchmark datasets in Spanish - 0.89 F1 for categories and 0.78 F1 on subcategories on CodiEsp, and Greek - 0.85 F1 on ElCardioCC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21433v2">The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 v2: Fixed typos and formatting issues
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering ( SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these strategies within SWE-agent on SWE-bench Verified across five diverse model configurations. We find that a simple observation-masking strategy halves cost relative to a raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. For example, with Qwen3-Coder 480B, masking improves solve rate from 53.8% (raw agent) to 54.8%, while remaining competitive with summarization at a lower cost. These results suggest that, at least within SWE-agent on SWE-bench Verified, the most effective and efficient context management can be the simplest. We release code and data for reproducibility
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04827v1">VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Modern Large Language Model (LLM) serving systems increasingly support interactive applications, like real-time chat assistants, code generation tools, and agentic workflows. However, the soaring energy cost of LLM inference presents a growing challenge for sustainable and cost-effective deployment. This paper introduces VoltanaLLM, a system for SLO-aware, energy-efficient LLM serving, built from a control theory perspective. VoltanaLLM co-designs frequency scaling and request routing in emerging prefill/decode disaggregated architectures, leveraging their decoupled execution to enable fine-grained phase-specific control. It consists of a feedback-driven frequency controller that dynamically adapts GPU frequency for prefill and decode phases, and a state-space router that explores routing decisions across frequency-scaled instances to minimize energy under latency constraints. We implement VoltanaLLM in SGLang and evaluate its performance over multiple state-of-the-art LLMs and real-world datasets. The results demonstrate that VoltanaLLM achieves up to 36.3% energy savings while maintaining near-perfect SLO attainment rate, paving the way for sustainable and intelligent LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04794v1">Personality as a Probe for LLM Evaluation: Method Trade-offs and Downstream Effects</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Personality manipulation in large language models (LLMs) is increasingly applied in customer service and agentic scenarios, yet its mechanisms and trade-offs remain unclear. We present a systematic study of personality control using the Big Five traits, comparing in-context learning (ICL), parameter-efficient fine-tuning (PEFT), and mechanistic steering (MS). Our contributions are fourfold. First, we construct a contrastive dataset with balanced high/low trait responses, enabling effective steering vector computation and fair cross-method evaluation. Second, we introduce a unified evaluation framework based on within-run $\Delta$ analysis that disentangles, reasoning capability, agent performance, and demographic bias across MMLU, GAIA, and BBQ benchmarks. Third, we develop trait purification techniques to separate openness from conscientiousness, addressing representational overlap in trait encoding. Fourth, we propose a three-level stability framework that quantifies method-, trait-, and combination-level robustness, offering practical guidance under deployment constraints. Experiments on Gemma-2-2B-IT and LLaMA-3-8B-Instruct reveal clear trade-offs: ICL achieves strong alignment with minimal capability loss, PEFT delivers the highest alignment at the cost of degraded task performance, and MS provides lightweight runtime control with competitive effectiveness. Trait-level analysis shows openness as uniquely challenging, agreeableness as most resistant to ICL, and personality encoding consolidating around intermediate layers. Taken together, these results establish personality manipulation as a multi-level probe into behavioral representation, linking surface conditioning, parameter encoding, and activation-level steering, and positioning mechanistic steering as a lightweight alternative to fine-tuning for both deployment and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17422v3">Multimodal LLM Guided Exploration and Active Mapping using Fisher Information</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      We present an active mapping system that plans for both long-horizon exploration goals and short-term actions using a 3D Gaussian Splatting (3DGS) representation. Existing methods either do not take advantage of recent developments in multimodal Large Language Models (LLM) or do not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based objective. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04781v1">The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32\% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22\%), we estimate real world bail rates range from 0.06-7\%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13\% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but increase bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20474v2">MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07900v2">MiniCPM4: Ultra-Efficient LLMs on End Devices</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 MiniCPM4 Technical Report
    </div>
    <details class="paper-abstract">
      This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose CPM.cu that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Furthermore, we construct a hybrid reasoning model, MiniCPM4.1, which can be used in both deep reasoning mode and non-reasoning mode. Evaluation results demonstrate that MiniCPM4 and MiniCPM4.1 outperform similar-sized open-source models across benchmarks, with the 8B variants showing significant speed improvements on long sequence understanding and generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04340v1">Write on Paper, Wrong in Practice: Why LLMs Still Struggle with Writing Clinical Notes</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are often proposed as tools to streamline clinical documentation, a task viewed as both high-volume and low-risk. However, even seemingly straightforward applications of LLMs raise complex sociotechnical considerations to translate into practice. This case study, conducted at KidsAbility, a pediatric rehabilitation facility in Ontario, Canada examined the use of LLMs to support occupational therapists in reducing documentation burden.We conducted a qualitative study involving 20 clinicians who participated in pilot programs using two AI technologies: a general-purpose proprietary LLM and a bespoke model fine-tuned on proprietary historical documentation. Our findings reveal that documentation challenges are sociotechnical in nature, shaped by clinical workflows, organizational policies, and system constraints. Four key themes emerged: (1) the heterogeneity of workflows, (2) the documentation burden is systemic and not directly linked to the creation of any single type of documentation, (3) the need for flexible tools and clinician autonomy, and (4) effective implementation requires mutual learning between clinicians and AI systems. While LLMs show promise in easing documentation tasks, their success will depend on flexible, adaptive integration that supports clinician autonomy. Beyond technical performance, sustained adoption will require training programs and implementation strategies that reflect the complexity of clinical environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04316v2">Small Changes, Large Consequences: Analyzing the Allocational Fairness of LLMs in Hiring Contexts</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed in high-stakes applications like hiring, yet their potential for unfair decision-making remains understudied in generative and retrieval settings. In this work, we examine the allocational fairness of LLM-based hiring systems through two tasks that reflect actual HR usage: resume summarization and applicant ranking. By constructing a synthetic resume dataset with controlled perturbations and curating job postings, we investigate whether model behavior differs across demographic groups. Our findings reveal that generated summaries exhibit meaningful differences more frequently for race than for gender perturbations. Models also display non-uniform retrieval selection patterns across demographic groups and exhibit high ranking sensitivity to both gender and race perturbations. Surprisingly, retrieval models can show comparable sensitivity to both demographic and non-demographic changes, suggesting that fairness issues may stem from broader model brittleness. Overall, our results indicate that LLM-based hiring systems, especially in the retrieval stage, can exhibit notable biases that lead to discriminatory outcomes in real-world contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02761v2">Plan Verification for LLM-Based Embodied Task Completion Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04310v1">EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21080v4">EQ-Knight: A Memory-Augmented LLM Agent for Strategic Affective Gaming in Debt Recovery</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language model-based chatbots have enhanced engagement in financial negotiations, but their overreliance on passive empathy introduces critical risks in credit collection. While empathy-driven approaches preserve client satisfaction in benign cases, they fail catastrophically against dishonest debtors--individuals who exploit conciliatory tactics to manipulate terms or evade repayment. Blindly prioritizing "customer experience" in such scenarios leads to creditor vulnerabilities: revenue leakage, moral hazard, and systemic exploitation. To address this, we propose EQ-Knight, an LLM agent that dynamically optimizes emotional strategy to defend creditor interests. Unlike naive empathy-centric bots, EQ-Knight integrates emotion memory and game-theoretic reasoning, powered by a Hidden Markov Model (HMM) to track and predict debtor emotional states. By analyzing both real-time and historical emotional cues, EQ-Knight strategically counters negative emotions (e.g., aggression, feigned distress) while preserving productive debtor relationships. Experiments demonstrate EQ-Knight's superiority over conventional LLM negotiators: it achieves a 32\% reduction in concession losses without compromising recovery rates, particularly in adversarial cases where debtors weaponize negative emotions (e.g., intimidation, guilt-tripping) to coerce concessions. For credit agencies, EQ-Knight transforms LLMs from high-risk "people-pleasers" into strategic emotion-defenders--balancing emotional intelligence with tactical rigor to enforce accountability and deter exploitation.
    </details>
</div>
