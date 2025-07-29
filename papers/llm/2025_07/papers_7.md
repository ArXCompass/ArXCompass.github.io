# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08207v1">A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08203v1">TruthTorchLM: A Comprehensive Library for Predicting Truthfulness in LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs)inevitably produce untruthful responses. Accurately predicting the truthfulness of these outputs is critical, especially in high-stakes settings. To accelerate research in this domain and make truthfulness prediction methods more accessible, we introduce TruthTorchLM an open-source, comprehensive Python library featuring over 30 truthfulness prediction methods, which we refer to as Truth Methods. Unlike existing toolkits such as Guardrails, which focus solely on document-grounded verification, or LM-Polygraph, which is limited to uncertainty-based methods, TruthTorchLM offers a broad and extensible collection of techniques. These methods span diverse tradeoffs in computational cost, access level (e.g., black-box vs white-box), grounding document requirements, and supervision type (self-supervised or supervised). TruthTorchLM is seamlessly compatible with both HuggingFace and LiteLLM, enabling support for locally hosted and API-based models. It also provides a unified interface for generation, evaluation, calibration, and long-form truthfulness prediction, along with a flexible framework for extending the library with new methods. We conduct an evaluation of representative truth methods on three datasets, TriviaQA, GSM8K, and FactScore-Bio. The code is available at https://github.com/Ybakman/TruthTorchLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06565v2">The Flaws of Others: An LLM-driven Framework for Scientific Knowledge Production</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 27 pages, 3 figures, 4 tables, 1 algorithm, 48 references
    </div>
    <details class="paper-abstract">
      Large-language models turn writing into a live exchange between humans and software. We capture this new medium with a discursive-network model that treats people and LLMs as equal nodes and tracks how their statements circulate. Broadening the focus from isolated hallucinations, we define invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. A general mathematical model of discursive networks is developed to provide valuable insights: A network governed only by drift and self-repair stabilizes at a modest error rate; adding fabrication reproduces the high rates seen in current LLMs. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source \emph{Flaws-of-Others (FOO) algorithm}: a configurable loop in which any set of agents critique one another while a harmoniser merges their verdicts. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from wiring imperfect ones into networks that keep each other honest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08881v1">The Consistency-Acceptability Divergence of LLMs in Judicial Decision-Making: Task and Stakeholder Dimensions</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 12 pages,2 figures
    </div>
    <details class="paper-abstract">
      The integration of large language model (LLM) technology into judicial systems is fundamentally transforming legal practice worldwide. However, this global transformation has revealed an urgent paradox requiring immediate attention. This study introduces the concept of ``consistency-acceptability divergence'' for the first time, referring to the gap between technical consistency and social acceptance. While LLMs achieve high consistency at the technical level, this consistency demonstrates both positive and negative effects. Through comprehensive analysis of recent data on LLM judicial applications from 2023--2025, this study finds that addressing this challenge requires understanding both task and stakeholder dimensions. This study proposes the Dual-Track Deliberative Multi-Role LLM Judicial Governance Framework (DTDMR-LJGF), which enables intelligent task classification and meaningful interaction among diverse stakeholders. This framework offers both theoretical insights and practical guidance for building an LLM judicial ecosystem that balances technical efficiency with social legitimacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08877v1">ODIA: Oriented Distillation for Inline Acceleration of LLM-based Function Calling</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Function Calling is a crucial technique that enables Large Language Models (LLMs) to interact with external systems through APIs. However, the high latency associated with LLM-based Function Calling significantly impacts user experience. This paper presents a novel approach called Oriented Distillation for Inline Acceleration (ODIA) that leverages online user interaction data to accelerate Function Calling. By automatically identifying "simple queries" from production traffic and distilling knowledge from larger models to smaller ones, our method reduces response latency by 45% (expected) and 78% (median) while maintaining accuracy. We demonstrate the effectiveness of our approach through real-world deployment in a music application, where the smaller model successfully handles 60% of traffic with negligible accuracy loss. Our method requires minimal human intervention and continuously improves through automated data collection and model updating, making it a practical solution for production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07996v1">Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Can a pretrained neural network adapt its architecture to different inputs without any finetuning? Do we need all layers for simple tasks, and are they adequate for challenging tasks? We found that the layers of a pretrained large language model (LLM) can be manipulated as separate modules to build a better and even shallower model customized for each test sample. In particular, each layer from the pretrained model can be skipped/pruned or repeated multiple times as recurrent neural networks (RNN), and stacked with others in arbitrary orders, yielding a chain-of-layers (CoLa) per sample. This compositional space greatly expands the scope of existing works on looped/recurrent pretrained modules, layer pruning, or early-exit networks. We develop a Monte Carlo Tree Search (MCTS) protocol to explore and identify the optimal CoLa for each sample from math and commonsense reasoning benchmarks. Compared to a static model of a fixed depth, CoLa allows shortcut paths (fast thinking), recurrence of the same layer(s) (slow thinking), and combining both, offering more flexible, dynamic architectures for different inputs. We conduct an extensive analysis of the MCTS-optimized CoLa, which leads to two key findings: (1) For >75% of samples with correct predictions by the original LLM, we can find shorter CoLa, suggesting a large space for improving inference efficiency; (2) For >60% of samples with originally incorrect predictions, we can identify CoLa achieving correct predictions, suggesting a large space of performance enhancement. Our results highlight the shortcomings of using a fixed architecture of pre-trained LLMs for inference on different samples and pave the way to unlock the generalization power of test-time depth adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07990v1">Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Accepted at ICCV2025; Project page: https://www.jshyun.me/projects/sttm
    </div>
    <details class="paper-abstract">
      Video large language models (LLMs) achieve strong video understanding by leveraging a large number of spatio-temporal tokens, but suffer from quadratic computational scaling with token count. To address this, we propose a training-free spatio-temporal token merging method, named STTM. Our key insight is to exploit local spatial and temporal redundancy in video data which has been overlooked in prior work. STTM first transforms each frame into multi-granular spatial tokens using a coarse-to-fine search over a quadtree structure, then performs directed pairwise merging across the temporal dimension. This decomposed merging approach outperforms existing token reduction methods across six video QA benchmarks. Notably, STTM achieves a 2$\times$ speed-up with only a 0.5% accuracy drop under a 50% token budget, and a 3$\times$ speed-up with just a 2% drop under a 30% budget. Moreover, STTM is query-agnostic, allowing KV cache reuse across different questions for the same video. The project page is available at https://www.jshyun.me/projects/sttm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14937v2">Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Transactions of Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07957v1">MIRIX: Multi-Agent Memory System for LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Although memory capabilities of AI agents are gaining increasing attention, existing solutions remain fundamentally limited. Most rely on flat, narrowly scoped memory components, constraining their ability to personalize, abstract, and reliably recall user-specific information over time. To this end, we introduce MIRIX, a modular, multi-agent memory system that redefines the future of AI memory by solving the field's most critical challenge: enabling language models to truly remember. Unlike prior approaches, MIRIX transcends text to embrace rich visual and multimodal experiences, making memory genuinely useful in real-world scenarios. MIRIX consists of six distinct, carefully structured memory types: Core, Episodic, Semantic, Procedural, Resource Memory, and Knowledge Vault, coupled with a multi-agent framework that dynamically controls and coordinates updates and retrieval. This design enables agents to persist, reason over, and accurately retrieve diverse, long-term user data at scale. We validate MIRIX in two demanding settings. First, on ScreenshotVQA, a challenging multimodal benchmark comprising nearly 20,000 high-resolution computer screenshots per sequence, requiring deep contextual understanding and where no existing memory systems can be applied, MIRIX achieves 35% higher accuracy than the RAG baseline while reducing storage requirements by 99.9%. Second, on LOCOMO, a long-form conversation benchmark with single-modal textual input, MIRIX attains state-of-the-art performance of 85.4%, far surpassing existing baselines. These results show that MIRIX sets a new performance standard for memory-augmented LLM agents. To allow users to experience our memory system, we provide a packaged application powered by MIRIX. It monitors the screen in real time, builds a personalized memory base, and offers intuitive visualization and secure local storage to ensure privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14993v2">Multi-modal Generative AI: Multi-modal LLMs, Diffusions and the Unification</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 20 pages, 11 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Multi-modal generative AI (Artificial Intelligence) has attracted increasing attention from both academia and industry. Particularly, two dominant families of techniques have emerged: i) Multi-modal large language models (LLMs) demonstrate impressive ability for multi-modal understanding; and ii) Diffusion models exhibit remarkable multi-modal powers in terms of multi-modal generation. Therefore, this paper provides a comprehensive overview of multi-modal generative AI, including multi-modal LLMs, diffusions, and the unification for understanding and generation. To lay a solid foundation for unified models, we first provide a detailed review of both multi-modal LLMs and diffusion models respectively, including their probabilistic modeling procedure, multi-modal architecture design, and advanced applications to image/video LLMs as well as text-to-image/video generation. Furthermore, we explore the emerging efforts toward unified models for understanding and generation. To achieve the unification of understanding and generation, we investigate key designs including autoregressive-based and diffusion-based modeling, as well as dense and Mixture-of-Experts (MoE) architectures. We then introduce several strategies for unified models, analyzing their potential advantages and disadvantages. In addition, we summarize the common datasets widely used for multi-modal generative AI pretraining. Last but not least, we present several challenging future research directions which may contribute to the ongoing advancement of multi-modal generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03296v3">Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU execution has emerged as a promising solution by offloading KV cache management and parts of attention computation to the CPU. However, a key bottleneck remains: existing schedulers fail to effectively overlap CPU-offloaded tasks with GPU execution during the latency-critical, bandwidth-bound decode phase. This particularly penalizes real-time, decode-heavy applications (e.g., chat, Chain-of-Thought reasoning) which are currently underserved by existing systems, especially under memory pressure typical of edge or low-cost deployments. We present APEX, a novel, profiling-informed scheduling strategy that maximizes CPU-GPU parallelism during hybrid LLM inference. Unlike systems relying on static rules or purely heuristic approaches, APEX dynamically dispatches compute across heterogeneous resources by predicting execution times of CPU and GPU subtasks to maximize overlap while avoiding scheduling overheads. We evaluate APEX on diverse workloads and GPU architectures (NVIDIA T4, A10), using LLaMa-2-7B and LLaMa-3.1-8B models. Compared to GPU-only schedulers like VLLM, APEX improves throughput by 84% - 96% on T4 and 11% - 89% on A10 GPUs, while preserving latency. Against the best existing hybrid schedulers, it delivers up to 49% (T4) and 37% (A10) higher throughput in long-output settings. APEX significantly advances hybrid LLM inference efficiency on such memory-constrained hardware and provides a blueprint for scheduling in heterogeneous AI systems, filling a critical gap for efficient real-time LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07870v1">DocCHA: Towards LLM-Augmented Interactive Online diagnosis System</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Despite the impressive capabilities of Large Language Models (LLMs), existing Conversational Health Agents (CHAs) remain static and brittle, incapable of adaptive multi-turn reasoning, symptom clarification, or transparent decision-making. This hinders their real-world applicability in clinical diagnosis, where iterative and structured dialogue is essential. We propose DocCHA, a confidence-aware, modular framework that emulates clinical reasoning by decomposing the diagnostic process into three stages: (1) symptom elicitation, (2) history acquisition, and (3) causal graph construction. Each module uses interpretable confidence scores to guide adaptive questioning, prioritize informative clarifications, and refine weak reasoning links. Evaluated on two real-world Chinese consultation datasets (IMCS21, DX), DocCHA consistently outperforms strong prompting-based LLM baselines (GPT-3.5, GPT-4o, LLaMA-3), achieving up to 5.18 percent higher diagnostic accuracy and over 30 percent improvement in symptom recall, with only modest increase in dialogue turns. These results demonstrate the effectiveness of DocCHA in enabling structured, transparent, and efficient diagnostic conversations -- paving the way for trustworthy LLM-powered clinical assistants in multilingual and resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v2">The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v5">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02357v2">Evaluating LLM Agent Adherence to Hierarchical Safety Principles: A Lightweight Benchmark for Probing Foundational Controllability Components</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint. This work has been submitted to the Technical AI Governance Workshop at ICML 2025 for review
    </div>
    <details class="paper-abstract">
      Credible safety plans for advanced AI development require methods to verify agent behavior and detect potential control deficiencies early. A fundamental aspect is ensuring agents adhere to safety-critical principles, especially when these conflict with operational goals. This paper introduces a lightweight, interpretable benchmark to evaluate an LLM agent's ability to uphold a high-level safety principle when faced with conflicting task instructions. Our evaluation of six LLMs reveals two primary findings: (1) a quantifiable "cost of compliance" where safety constraints degrade task performance even when compliant solutions exist, and (2) an "illusion of compliance" where high adherence often masks task incompetence rather than principled choice. These findings provide initial evidence that while LLMs can be influenced by hierarchical directives, current approaches lack the consistency required for reliable safety governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01936v2">The Thin Line Between Comprehension and Persuasion in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04426v2">Decoding AI Judgment: How LLMs Assess News Credibility and Bias</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in workflows that involve evaluative processes. This raises the need to examine how such evaluations are built, what assumptions they rely on, and how their strategies diverge from those of humans. We benchmark six LLMs against expert ratings--NewsGuard and Media Bias/Fact Check (MBFC)--and against human judgments collected through a controlled experiment. To enable direct comparison, we implement a structured agentic framework in which both models and non-expert participants follow the same evaluation procedure: selecting criteria, retrieving content, and producing justifications. Despite output alignment, LLMs rely on different mechanisms: lexical associations and statistical priors replace contextual reasoning. This reliance produces systematic effects: political asymmetries, opaque justifications, and a tendency to confuse linguistic form with epistemic validity. Delegating judgment to such systems does not merely automate evaluation--it redefines it, shifting from normative reasoning to pattern-based approximation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11984v2">Understanding Chain-of-Thought in LLMs through Information Theory</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance in complex reasoning tasks through the use of Chain-of-Thought (CoT) reasoning, allowing models to break down problems into manageable sub-tasks. However, existing CoT evaluation techniques either require annotated CoT data or fall short in accurately assessing intermediate reasoning steps, leading to high rates of false positives. In this paper, we formalize CoT reasoning in LLMs through an information-theoretic lens. Specifically, our framework quantifies the `information-gain' at each reasoning step, enabling the identification of failure modes in LLMs without the need for expensive annotated datasets. We demonstrate the efficacy of our approach through extensive experiments on toy arithmetic, GSM8K and PRM800k datasets, where it significantly outperforms existing outcome-based methods by providing more accurate insights into model performance on individual subtasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07745v1">On the capabilities of LLMs for classifying and segmenting time series of fruit picking motions into primitive actions</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 This paper is a Late Breaking Results report and it will be presented through a poster at the 34th IEEE International Conference on Robot and Human Interactive Communication (ROMAN), 2025 at Eindhoven, the Netherlands
    </div>
    <details class="paper-abstract">
      Despite their recent introduction to human society, Large Language Models (LLMs) have significantly affected the way we tackle mental challenges in our everyday lives. From optimizing our linguistic communication to assisting us in making important decisions, LLMs, such as ChatGPT, are notably reducing our cognitive load by gradually taking on an increasing share of our mental activities. In the context of Learning by Demonstration (LbD), classifying and segmenting complex motions into primitive actions, such as pushing, pulling, twisting etc, is considered to be a key-step towards encoding a task. In this work, we investigate the capabilities of LLMs to undertake this task, considering a finite set of predefined primitive actions found in fruit picking operations. By utilizing LLMs instead of simple supervised learning or analytic methods, we aim at making the method easily applicable and deployable in a real-life scenario. Three different fine-tuning approaches are investigated, compared on datasets captured kinesthetically, using a UR10e robot, during a fruit-picking scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07695v1">KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 21 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning is an immensely resource-intensive process when retraining Large Language Models (LLMs) to incorporate a larger body of knowledge. Although many fine-tuning techniques have been developed to reduce the time and computational cost involved, the challenge persists as LLMs continue to grow in size and complexity. To address this, a new approach to knowledge expansion in LLMs is needed. Retrieval-Augmented Generation (RAG) offers one such alternative by storing external knowledge in a database and retrieving relevant chunks to support question answering. However, naive implementations of RAG face significant limitations in scalability and answer accuracy. This paper introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome these limitations. Inspired by the divide-and-conquer paradigm, K2RAG integrates dense and sparse vector search, knowledge graphs, and text summarization to improve retrieval quality and system efficiency. The framework also includes a preprocessing step that summarizes the training data, significantly reducing the training time. K2RAG was evaluated using the MultiHopRAG dataset, where the proposed pipeline was trained on the document corpus and tested on a separate evaluation set. Results demonstrated notable improvements over common naive RAG implementations. K2RAG achieved the highest mean answer similarity score of 0.57, and reached the highest third quartile (Q3) similarity of 0.82, indicating better alignment with ground-truth answers. In addition to improved accuracy, the framework proved highly efficient. The summarization step reduced the average training time of individual components by 93%, and execution speed was up to 40% faster than traditional knowledge graph-based RAG systems. K2RAG also demonstrated superior scalability, requiring three times less VRAM than several naive RAG implementations tested in this study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07644v1">PlanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 25 pages, 18 figures. Diagnostic benchmark for spatial reasoning in LLMs. Project page: https://OldDelorean.github.io/PlanQA/
    </div>
    <details class="paper-abstract">
      We introduce PlanQA, a diagnostic benchmark for evaluating geometric and spatial reasoning in large-language models (LLMs). PlanQA is grounded in structured representations of indoor scenes, such as kitchens, living rooms, and bedrooms, encoded in a symbolic format (e.g., JSON, XML layouts). The benchmark includes diverse question types that test not only metric and topological reasoning (e.g., distance, visibility, shortest paths) but also interior design constraints such as affordance, clearance, balance, and usability. Our results across a variety of frontier open-source and commercial LLMs show that while models may succeed in shallow queries, they often fail to simulate physical constraints, preserve spatial coherence, or generalize under layout perturbation. PlanQA uncovers a clear blind spot in today's LLMs: they do not consistently reason about real-world layouts. We hope that this benchmark inspires new work on language models that can accurately infer and manipulate spatial and geometric properties in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07630v1">Exploring the Limits of Model Compression in LLMs: A Knowledge Distillation Study on QA Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 Accepted four publication at the 26th Meeting of the Special Interest on Discourse and Dialogue
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated outstanding performance across a range of NLP tasks, however, their computational demands hinder their deployment in real-world, resource-constrained environments. This work investigates the extent to which LLMs can be compressed using Knowledge Distillation (KD) while maintaining strong performance on Question Answering (QA) tasks. We evaluate student models distilled from the Pythia and Qwen2.5 families on two QA benchmarks, SQuAD and MLQA, under zero-shot and one-shot prompting conditions. Results show that student models retain over 90% of their teacher models' performance while reducing parameter counts by up to 57.1%. Furthermore, one-shot prompting yields additional performance gains over zero-shot setups for both model families. These findings underscore the trade-off between model efficiency and task performance, demonstrating that KD, combined with minimal prompting, can yield compact yet capable QA systems suitable for resource-constrained applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14382v2">Good/Evil Reputation Judgment of Celebrities by LLMs via Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-10
    </div>
    <details class="paper-abstract">
      The purpose of this paper is to examine whether large language models (LLMs) can understand what is good and evil with respect to judging good/evil reputation of celebrities. Specifically, we first apply a large language model (namely, ChatGPT) to the task of collecting sentences that mention the target celebrity from articles about celebrities on Web pages. Next, the collected sentences are categorized based on their contents by ChatGPT, where ChatGPT assigns a category name to each of those categories. Those assigned category names are referred to as "aspects" of each celebrity. Then, by applying the framework of retrieval augmented generation (RAG), we show that the large language model is quite effective in the task of judging good/evil reputation of aspects and descriptions of each celebrity. Finally, also in terms of proving the advantages of the proposed method over existing services incorporating RAG functions, we show that the proposed method of judging good/evil of aspects/descriptions of each celebrity significantly outperform an existing service incorporating RAG functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07548v1">From Requirements to Code: Understanding Developer Practices in LLM-Assisted Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 This paper has been accepted for publication at the 33rd IEEE International Requirements Engineering (RE) conference
    </div>
    <details class="paper-abstract">
      With the advent of generative LLMs and their advanced code generation capabilities, some people already envision the end of traditional software engineering, as LLMs may be able to produce high-quality code based solely on the requirements a domain expert feeds into the system. The feasibility of this vision can be assessed by understanding how developers currently incorporate requirements when using LLMs for code generation-a topic that remains largely unexplored. We interviewed 18 practitioners from 14 companies to understand how they (re)use information from requirements and other design artifacts to feed LLMs when generating code. Based on our findings, we propose a theory that explains the processes developers employ and the artifacts they rely on. Our theory suggests that requirements, as typically documented, are too abstract for direct input into LLMs. Instead, they must first be manually decomposed into programming tasks, which are then enriched with design decisions and architectural constraints before being used in prompts. Our study highlights that fundamental RE work is still necessary when LLMs are used to generate code. Our theory is important for contextualizing scientific approaches to automating requirements-centric SE tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11329v2">TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-10
      | 💬 14 pages, 16 figures. For source code, see https://github.com/microsoft/tokenweave
    </div>
    <details class="paper-abstract">
      Distributed inference of large language models (LLMs) can introduce overheads of up to 20% even over GPUs connected via high-speed interconnects such as NVLink. Multiple techniques have been proposed to mitigate these overheads by decomposing computations into finer-grained tasks and overlapping communication with sub-tasks as they complete. However, fine-grained decomposition of a large computation into many smaller computations on GPUs results in overheads. Furthermore, the communication itself uses many streaming multiprocessors (SMs), adding to the overhead. We present TokenWeave to address these challenges. TokenWeave proposes a Token-Splitting technique that divides the tokens in the inference batch into two approximately equal subsets in a wave-aware manner. The communication of one subset is then overlapped with the computation of the other. In addition, TokenWeave optimizes the order of the layer normalization computation with respect to communication operations and implements a novel fused AllReduce--RMSNorm kernel that carefully leverages Multimem instruction support available on NVIDIA Hopper GPUs. These optimizations allow TokenWeave to perform communication and RMSNorm using only 2-8 SMs. Moreover, our kernel enables the memory-bound RMSNorm to be overlapped with the other batch's computation, providing additional gains. Our evaluations demonstrate up to 1.29x speedup in latency and 1.26x higher throughput across multiple models and workloads. In several settings, TokenWeave results in better performance compared to an equivalent model with all communication removed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07145v1">CCQ: Convolutional Code for Extreme Low-bit Quantization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The rapid scaling of Large Language Models (LLMs) elevates inference costs and compounds substantial deployment barriers. While quantization to 8 or 4 bits mitigates this, sub-3-bit methods face severe accuracy, scalability, and efficiency degradation. We propose Convolutional Code Quantization (CCQ), an inference-optimized quantization approach compressing LLMs to 2.0-2.75 bits with minimal accuracy loss. Departing from error-prone scalar quantization or slow vector quantization, CCQ integrates a hardware-aware bit-shift encoding and decoding solution with Convolutional Code, Hybrid Encoding, and Code Cluster, jointly overcoming accuracy-speed bottlenecks. We construct a lookup-free encoding space, enabling a linear mapping between the codebook and weight vectors, thereby optimizing inference performance. Meanwhile, by drawing on the concept of data mapping from vector quantization, we minimize the performance degradation of the model under extremely low-bit conditions. Experiments demonstrate that CCQ achieves outstanding performance on LLMs across various benchmarks. We compress DeepSeek-V3 (671B total parameters) to 184GB and ERNIE-4.5-300B-A47B to 89GB, enabling single-GPU deployment of ERNIE 4.5 and eliminating inter-card communication. The 2-bit ERNIE-4.5-300B-A47B model and inference engine have been open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v3">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of \textbf{Trust, Risk, and Security Management (TRiSM)} in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around four key pillars: Explainability, ModelOps, Security, Privacy and Governance, each contextualized to the challenges of multi-agent LLM systems. A novel risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI , as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, outlining critical directions to align emerging systems with TRiSM principles for safe, transparent, and accountable operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07064v1">Boosting Parameter Efficiency in LLM-Based Recommendation through Sophisticated Pruning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      LLM-based recommender systems have made significant progress; however, the deployment cost associated with the large parameter volume of LLMs still hinders their real-world applications. This work explores parameter pruning to improve parameter efficiency while maintaining recommendation quality, thereby enabling easier deployment. Unlike existing approaches that focus primarily on inter-layer redundancy, we uncover intra-layer redundancy within components such as self-attention and MLP modules. Building on this analysis, we propose a more fine-grained pruning approach that integrates both intra-layer and layer-wise pruning. Specifically, we introduce a three-stage pruning strategy that progressively prunes parameters at different levels and parts of the model, moving from intra-layer to layer-wise pruning, or from width to depth. Each stage also includes a performance restoration step using distillation techniques, helping to strike a balance between performance and parameter efficiency. Empirical results demonstrate the effectiveness of our approach: across three datasets, our models achieve an average of 88% of the original model's performance while pruning more than 95% of the non-embedding parameters. This underscores the potential of our method to significantly reduce resource requirements without greatly compromising recommendation quality. Our code will be available at: https://github.com/zheng-sl/PruneRec
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07045v1">5C Prompt Contracts: A Minimalist, Creative-Friendly, Token-Efficient Design Framework for Individual and SME LLM Usage</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 5 pages, 5 tables. Includes comparative experimental results across OpenAI, Anthropic, DeepSeek, and Gemini LLMs
    </div>
    <details class="paper-abstract">
      The progression from traditional prompt engineering to a more rigorous discipline of prompt design marks a pivotal shift in human-LLM interaction. As Large Language Models (LLMs) become increasingly embedded in mission-critical applications, there emerges a pressing need for frameworks that are not only explicit and systematic but also minimal enough to remain practical and broadly accessible. While many existing approaches address prompt structuring through elaborate Domain-Specific Languages (DSLs) or multi-layered templates, such methods can impose significant token and cognitive overhead, potentially constraining the model's creative capacity. In this context, we propose the 5C Prompt Contract, a framework that distills prompt design into five intuitive components: Character, Cause, Constraint, Contingency, and Calibration. This minimal cognitive schema explicitly integrates fallback and output optimization directives, fostering reliable, interpretable, and creatively flexible AI interactions. Experimental results demonstrate that the 5C framework consistently achieves superior input token efficiency while maintaining rich and consistent outputs across diverse LLM architectures (OpenAI, Anthropic, DeepSeek, and Gemini), making it particularly suited for individuals and Small-to-Medium Enterprises (SMEs) with limited AI engineering resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06999v1">Learning Deliberately, Acting Intuitively: Unlocking Test-Time Reasoning in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reasoning is a key capability for large language models (LLMs), particularly when applied to complex tasks such as mathematical problem solving. However, multimodal reasoning research still requires further exploration of modality alignment and training costs. Many of these approaches rely on additional data annotation and relevant rule-based rewards to enhance the understanding and reasoning ability, which significantly increases training costs and limits scalability. To address these challenges, we propose the Deliberate-to-Intuitive reasoning framework (D2I) that improves the understanding and reasoning ability of multimodal LLMs (MLLMs) without extra annotations and complex rewards. Specifically, our method sets deliberate reasoning strategies to enhance modality alignment only through the rule-based format reward during training. While evaluating, the reasoning style shifts to intuitive, which removes deliberate reasoning strategies during training and implicitly reflects the model's acquired abilities in the response. D2I outperforms baselines across both in-domain and out-of-domain benchmarks. Our findings highlight the role of format reward in fostering transferable reasoning skills in MLLMs, and inspire directions for decoupling training-time reasoning depth from test-time response flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06992v1">MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12112v3">Planning Anything with Rigor: General-Purpose Zero-Shot Planning with LLM-based Formalized Programming</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 57 pages, 25 figures, 15 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have recently demonstrated strong potential in solving planning problems, there is a trade-off between flexibility and complexity. LLMs, as zero-shot planners themselves, are still not capable of directly generating valid plans for complex planning problems such as multi-constraint or long-horizon tasks. On the other hand, many frameworks aiming to solve complex planning problems often rely on task-specific preparatory efforts, such as task-specific in-context examples and pre-defined critics/verifiers, which limits their cross-task generalization capability. In this paper, we tackle these challenges by observing that the core of many planning problems lies in optimization problems: searching for the optimal solution (best plan) with goals subject to constraints (preconditions and effects of decisions). With LLMs' commonsense, reasoning, and programming capabilities, this opens up the possibilities of a universal LLM-based approach to planning problems. Inspired by this observation, we propose LLMFP, a general-purpose framework that leverages LLMs to capture key information from planning problems and formally formulate and solve them as optimization problems from scratch, with no task-specific examples needed. We apply LLMFP to 9 planning problems, ranging from multi-constraint decision making to multi-step planning problems, and demonstrate that LLMFP achieves on average 83.7% and 86.8% optimal rate across 9 tasks for GPT-4o and Claude 3.5 Sonnet, significantly outperforming the best baseline (direct planning with OpenAI o1-preview) with 37.6% and 40.7% improvements. We also validate components of LLMFP with ablation experiments and analyzed the underlying success and failure reasons. Project page: https://sites.google.com/view/llmfp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06980v1">Are They All Good? Evaluating the Quality of CoTs in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance in code generation, particularly when augmented with chain-of-thought (CoT) prompting techniques. They break down requirements into intermediate reasoning steps, which act as design rationales to guide LLMs in writing code like human programmers. Thus, the quality of these steps is crucial for ensuring the correctness and reliability of the generated code. However, little is known about the quality of CoT generated by LLMs. To what extent can we trust the thoughts generated by LLMs? How good are they? This paper empirically explores the external and internal factors of why LLMs generate unsatisfactory CoTs by analyzing 1,023 failed code samples on two widely used code generation benchmarks. We also evaluate their impact on code generation performance by analyzing 210 CoT-code pairs and refining the unsatisfied CoTs by prompting LLMs. Our study reveals three key findings: (1) External factors (53.60%), such as unclear requirements and lack of context, mainly affect CoT quality, while internal factors (40.10%) stem from LLMs' misunderstanding prompts. (2) Even when CoTs are correct, 18.5% of the generated code contains errors due to instruction-following issues; conversely, 11.90% of correct code is paired with flawed CoTs. (3) Refining low-quality CoTs is feasible, i.e., LLMs improve when given detailed problem descriptions. These findings highlight key challenges in CoT-based code generation and suggest directions for improving LLM reasoning and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06920v1">Rethinking Verification for LLM Code Generation: From Generation to Testing</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved notable success in code-generation benchmarks such as HumanEval and LiveCodeBench. However, a detailed examination reveals that these evaluation suites often comprise only a limited number of homogeneous test cases, resulting in subtle faults going undetected. This not only artificially inflates measured performance but also compromises accurate reward estimation in reinforcement learning frameworks utilizing verifiable rewards (RLVR). To address these critical shortcomings, we systematically investigate the test-case generation (TCG) task by proposing multi-dimensional metrics designed to rigorously quantify test-suite thoroughness. Furthermore, we introduce a human-LLM collaborative method (SAGA), leveraging human programming expertise with LLM reasoning capability, aimed at significantly enhancing both the coverage and the quality of generated test cases. In addition, we develop a TCGBench to facilitate the study of the TCG task. Experiments show that SAGA achieves a detection rate of 90.62% and a verifier accuracy of 32.58% on TCGBench. The Verifier Accuracy (Verifier Acc) of the code generation evaluation benchmark synthesized by SAGA is 10.78% higher than that of LiveCodeBench-v6. These results demonstrate the effectiveness of our proposed method. We hope this work contributes to building a scalable foundation for reliable LLM code evaluation, further advancing RLVR in code generation, and paving the way for automated adversarial test synthesis and adaptive benchmark integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06910v1">Exploring LLMs for Predicting Tutor Strategy and Student Outcomes in Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Published in BEA 2025: 20th Workshop on Innovative Use of NLP for Building Educational Applications
    </div>
    <details class="paper-abstract">
      Tutoring dialogues have gained significant attention in recent years, given the prominence of online learning and the emerging tutoring abilities of artificial intelligence (AI) agents powered by large language models (LLMs). Recent studies have shown that the strategies used by tutors can have significant effects on student outcomes, necessitating methods to predict how tutors will behave and how their actions impact students. However, few works have studied predicting tutor strategy in dialogues. Therefore, in this work we investigate the ability of modern LLMs, particularly Llama 3 and GPT-4o, to predict both future tutor moves and student outcomes in dialogues, using two math tutoring dialogue datasets. We find that even state-of-the-art LLMs struggle to predict future tutor strategy while tutor strategy is highly indicative of student outcomes, outlining a need for more powerful methods to approach this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v1">The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02579v3">EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 14 pages, 9 figures, accepted by the SIGDIAL 2025 conference
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including competing objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the fine-tuning to improve efficiency and flexibility. Our method is the first to aggregate the hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text classification models to score the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15167v2">LLM Agent for Hyper-Parameter Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 6 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Hyper-parameters are essential and critical for the performance of communication algorithms. However, current hyper-parameters optimization approaches for Warm-Start Particles Swarm Optimization with Crossover and Mutation (WS-PSO-CM) algorithm, designed for radio map-enabled unmanned aerial vehicle (UAV) trajectory and communication, are primarily heuristic-based, exhibiting low levels of automation and improvable performance. In this paper, we design an Large Language Model (LLM) agent for automatic hyper-parameters-tuning, where an iterative framework and Model Context Protocol (MCP) are applied. In particular, the LLM agent is first set up via a profile, which specifies the boundary of hyper-parameters, task objective, terminal condition, conservative or aggressive strategy of optimizing hyper-parameters, and LLM configurations. Then, the LLM agent iteratively invokes WS-PSO-CM algorithm for exploration. Finally, the LLM agent exits the loop based on the terminal condition and returns an optimized set of hyperparameters. Our experiment results show that the minimal sum-rate achieved by hyper-parameters generated via our LLM agent is significantly higher than those by both human heuristics and random generation methods. This indicates that an LLM agent with PSO and WS-PSO-CM algorithm knowledge is useful in seeking high-performance hyper-parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09347v3">Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 9 pages, ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed as automated evaluators to assess the safety of generated content, yet their reliability in this role remains uncertain. This study evaluates a diverse set of 11 LLM judge models across critical safety domains, examining three key aspects: self-consistency in repeated judging tasks, alignment with human judgments, and susceptibility to input artifacts such as apologetic or verbose phrasing. Our findings reveal that biases in LLM judges can significantly distort the final verdict on which content source is safer, undermining the validity of comparative evaluations. Notably, apologetic language artifacts alone can skew evaluator preferences by up to 98\%. Contrary to expectations, larger models do not consistently exhibit greater robustness, while smaller models sometimes show higher resistance to specific artifacts. To mitigate LLM evaluator robustness issues, we investigate jury-based evaluations aggregating decisions from multiple models. Although this approach both improves robustness and enhances alignment to human judgements, artifact sensitivity persists even with the best jury configurations. These results highlight the urgent need for diversified, artifact-resistant methodologies to ensure reliable safety assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16903v2">GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/
    </div>
    <details class="paper-abstract">
      Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06774v1">Checklist Engineering Empowers Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04446v2">Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06762v1">Leveraging LLMs for Semantic Conflict Detection via Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Comments: 11 pages, in Portuguese language. 3 figures. Submitted to SAST 2025 (X Simp\'osio Brasileiro de Teste de Software Sistem\'atico e Automatizado)
    </div>
    <details class="paper-abstract">
      Semantic conflicts arise when a developer introduces changes to a codebase that unintentionally affect the behavior of changes integrated in parallel by other developers. Traditional merge tools are unable to detect such conflicts, so complementary tools like SMAT have been proposed. SMAT relies on generating and executing unit tests: if a test fails on the base version, passes on a developer's modified version, but fails again after merging with another developer's changes, a semantic conflict is indicated. While SMAT is effective at detecting conflicts, it suffers from a high rate of false negatives, partly due to the limitations of unit test generation tools such as Randoop and Evosuite. To investigate whether large language models (LLMs) can overcome these limitations, we propose and integrate a new test generation tool based on Code Llama 70B into SMAT. We explore the model's ability to generate tests using different interaction strategies, prompt contents, and parameter configurations. Our evaluation uses two samples: a benchmark with simpler systems from related work, and a more significant sample based on complex, real-world systems. We assess the effectiveness of the new SMAT extension in detecting conflicts. Results indicate that, although LLM-based test generation remains challenging and computationally expensive in complex scenarios, there is promising potential for improving semantic conflict detection. -- Conflitos sem^anticos surgem quando um desenvolvedor introduz mudan\c{c}as em uma base de c\'odigo que afetam, de forma n~ao intencional, o comportamento de altera\c{c}~oes integradas em paralelo por outros desenvolvedores. Ferramentas tradicionais de merge n~ao conseguem detectar esse tipo de conflito, por isso ferramentas complementares como o SMAT foram propostas. O SMAT depende da gera\c{c}~ao e execu\c{c}~ao de testes de unidade: se um teste falha na vers~ao base, passa na vers~ao modificada por um desenvolvedor, mas volta a falhar ap\'os o merge com as mudan\c{c}as de outro desenvolvedor, um conflito sem^antico \'e identificado. Embora o SMAT seja eficaz na detec\c{c}~ao de conflitos, apresenta alta taxa de falsos negativos, em parte devido \`as limita\c{c}~oes das ferramentas de gera\c{c}~ao de testes como Randoop e Evosuite. Para investigar se modelos de linguagem de grande porte (LLMs) podem superar essas limita\c{c}~oes, propomos e integramos ao SMAT uma nova ferramenta de gera\c{c}~ao de testes baseada no Code Llama 70B. Exploramos a capacidade do modelo de gerar testes utilizando diferentes estrat\'egias de intera\c{c}~ao, conte\'udos de prompts e configura\c{c}~oes de par^ametros. Nossa avalia\c{c}~ao utiliza duas amostras: um benchmark com sistemas mais simples, usados em trabalhos relacionados, e uma amostra mais significativa baseada em sistemas complexos e reais. Avaliamos a efic\'acia da nova extens~ao do SMAT na detec\c{c}~ao de conflitos. Os resultados indicam que, embora a gera\c{c}~ao de testes por LLM em cen\'arios complexos ainda seja desafiadora e custosa computacionalmente, h\'a potencial promissor para aprimorar a detec\c{c}~ao de conflitos sem^anticos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03785v3">Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted to GEM @ ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14541v2">LLM-based User Profile Management for Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted GENNEXT@SIGIR'25 Workshop
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06719v1">A Neural Representation Framework with LLM-Driven Spatial Reasoning for Open-Vocabulary 3D Visual Grounding</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Open-vocabulary 3D visual grounding aims to localize target objects based on free-form language queries, which is crucial for embodied AI applications such as autonomous navigation, robotics, and augmented reality. Learning 3D language fields through neural representations enables accurate understanding of 3D scenes from limited viewpoints and facilitates the localization of target objects in complex environments. However, existing language field methods struggle to accurately localize instances using spatial relations in language queries, such as ``the book on the chair.'' This limitation mainly arises from inadequate reasoning about spatial relations in both language queries and 3D scenes. In this work, we propose SpatialReasoner, a novel neural representation-based framework with large language model (LLM)-driven spatial reasoning that constructs a visual properties-enhanced hierarchical feature field for open-vocabulary 3D visual grounding. To enable spatial reasoning in language queries, SpatialReasoner fine-tunes an LLM to capture spatial relations and explicitly infer instructions for the target, anchor, and spatial relation. To enable spatial reasoning in 3D scenes, SpatialReasoner incorporates visual properties (opacity and color) to construct a hierarchical feature field. This field represents language and instance features using distilled CLIP features and masks extracted via the Segment Anything Model (SAM). The field is then queried using the inferred instructions in a hierarchical manner to localize the target 3D instance based on the spatial relation in the language query. Extensive experiments show that our framework can be seamlessly integrated into different neural representations, outperforming baseline models in 3D visual grounding while empowering their spatial reasoning capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06715v1">CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), including zero-shot and few-shot paradigms, have shown promising capabilities in clinical text generation. However, real-world applications face two key challenges: (1) patient data is highly unstructured, heterogeneous, and scattered across multiple note types and (2) clinical notes are often long and semantically dense, making naive prompting infeasible due to context length constraints and the risk of omitting clinically relevant information. We introduce CLI-RAG (Clinically Informed Retrieval-Augmented Generation), a domain-specific framework for structured and clinically grounded text generation using LLMs. It incorporates a novel hierarchical chunking strategy that respects clinical document structure and introduces a task-specific dual-stage retrieval mechanism. The global stage identifies relevant note types using evidence-based queries, while the local stage extracts high-value content within those notes creating relevance at both document and section levels. We apply the system to generate structured progress notes for individual hospital visits using 15 clinical note types from the MIMIC-III dataset. Experiments show that it preserves temporal and semantic alignment across visits, achieving an average alignment score of 87.7%, surpassing the 80.7% baseline from real clinician-authored notes. The generated outputs also demonstrate high consistency across LLMs, reinforcing deterministic behavior essential for reproducibility, reliability, and clinical trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07675v2">QUITE: A Query Rewrite System Beyond Rules with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18951v2">SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 26 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: https://bird-critic.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06623v1">Expediting data extraction using a large language model (LLM) and scoping review protocol: a methodological study within a complex scoping review</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 44 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The data extraction stages of reviews are resource-intensive, and researchers may seek to expediate data extraction using online (large language models) LLMs and review protocols. Claude 3.5 Sonnet was used to trial two approaches that used a review protocol to prompt data extraction from 10 evidence sources included in a case study scoping review. A protocol-based approach was also used to review extracted data. Limited performance evaluation was undertaken which found high accuracy for the two extraction approaches (83.3% and 100%) when extracting simple, well-defined citation details; accuracy was lower (9.6% and 15.8%) when extracting more complex, subjective data items. Considering all data items, both approaches had precision >90% but low recall (<25%) and F1 scores (<40%). The context of a complex scoping review, open response types and methodological approach likely impacted performance due to missed and misattributed data. LLM feedback considered the baseline extraction accurate and suggested minor amendments: four of 15 (26.7%) to citation details and 8 of 38 (21.1%) to key findings data items were considered to potentially add value. However, when repeating the process with a dataset featuring deliberate errors, only 2 of 39 (5%) errors were detected. Review-protocol-based methods used for expediency require more robust performance evaluation across a range of LLMs and review contexts with comparison to conventional prompt engineering approaches. We recommend researchers evaluate and report LLM performance if using them similarly to conduct data extraction or review extracted data. LLM feedback contributed to protocol adaptation and may assist future review protocol drafting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23132v2">LAURA: LLM-Assisted UAV Routing for AoI Minimization</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      With the rapid growth of the low-altitude economy, there is increasing demand for real-time data collection using UAV-assisted wireless sensor networks. This paper investigates the problem of minimizing the age of information (AoI) in UAV-assisted wireless sensor networks by optimizing the UAV flight routing. We formulate the AoI minimization task and propose a large language model (LLM)-assisted UAV routing algorithm (LAURA). LAURA employs an LLM as intelligent crossover operators within an evolutionary optimization framework to efficiently explore the solution space. Simulation results show that LAURA outperforms benchmark methods in reducing the maximum AoI, especially in scenarios with a large number of sensor nodes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06573v1">From Data-Centric to Sample-Centric: Enhancing LLM Reasoning via Progressive Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has recently advanced the reasoning capabilities of large language models (LLMs). While prior work has emphasized algorithmic design, data curation, and reward shaping, we investigate RLVR from a sample-centric perspective and introduce LPPO (Learning-Progress and Prefix-guided Optimization), a framework of progressive optimization techniques. Our work addresses a critical question: how to best leverage a small set of trusted, high-quality demonstrations, rather than simply scaling up data volume. First, motivated by how hints aid human problem-solving, we propose prefix-guided sampling, an online data augmentation method that incorporates partial solution prefixes from expert demonstrations to guide the policy, particularly for challenging instances. Second, inspired by how humans focus on important questions aligned with their current capabilities, we introduce learning-progress weighting, a dynamic strategy that adjusts each training sample's influence based on model progression. We estimate sample-level learning progress via an exponential moving average of per-sample pass rates, promoting samples that foster learning and de-emphasizing stagnant ones. Experiments on mathematical-reasoning benchmarks demonstrate that our methods outperform strong baselines, yielding faster convergence and a higher performance ceiling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.12538v2">PersonaFlow: Designing LLM-Simulated Expert Perspectives for Enhanced Research Ideation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted to DIS2025
    </div>
    <details class="paper-abstract">
      Generating interdisciplinary research ideas requires diverse domain expertise, but access to timely feedback is often limited by the availability of experts. In this paper, we introduce PersonaFlow, a novel system designed to provide multiple perspectives by using LLMs to simulate domain-specific experts. Our user studies showed that the new design 1) increased the perceived relevance and creativity of ideated research directions, and 2) promoted users' critical thinking activities (e.g., interpretation, analysis, evaluation, inference, and self-regulation), without increasing their perceived cognitive load. Moreover, users' ability to customize expert profiles significantly improved their sense of agency, which can potentially mitigate their over-reliance on AI. This work contributes to the design of intelligent systems that augment creativity and collaboration, and provides design implications of using customizable AI-simulated personas in domains within and beyond research ideation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09073v3">CHAI for LLMs: Improving Code-Mixed Translation in Large Language Models through Reinforcement Learning with AI Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 full draft v2: 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various NLP tasks but struggle with code-mixed (or code-switched) language understanding. For example, prior work benchmarking the performance of multilingual LLMs on code-mixed translation tasks has demonstrated that current state-of-the-art multilingual LLMs are ineffective in dealing with code-mixed languages. However, the question of how to improve the capability of multilingual LLMs to handle code-mixed language has not received any attention to date. In this paper, we tackle this research gap by proposing CHAI, a novel general-purpose framework for improving the ability of multilingual LLMs to handle code-mixed languages. CHAI relies on three novel contributions made in this paper. First, we explore the ability of LLMs to provide accurate annotations for code-mixed translation tasks. Second, we leverage this ability of LLMs as annotators to generate preference data for code-mixed translation tasks at scale, which are then used within a reinforcement learning from AI feedback (RLAIF) procedure to improve LLMs' capability on code-mixed tasks. Third, we conduct a rigorous experimental evaluation across various real-world datasets and settings. Our analysis shows that CHAI-powered LLMs outperform state-of-the-art open-source LLMs by 25.66% (in terms of win rate adjudicated by human annotators) in code-mixed translation tasks. This work represents a first step towards developing more inclusive code-mixed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06565v1">The Flaws of Others: An LLM-driven Framework for Scientific Knowledge Production</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 27 pages, 3 figures, 4 tables, 1 algorithm, 28 references
    </div>
    <details class="paper-abstract">
      Large-language models turn writing into a live exchange between humans and software. We capture this new medium with a discursive-network model that treats people and LLMs as equal nodes and tracks how their statements circulate. Broadening the focus from isolated hallucinations, we define invalidation (any factual, logical, or structural breach) and show it follows four hazards: drift from truth, self-repair, fresh fabrication, and external detection. A general mathematical model of discursive networks is developed to provide valuable insights: A network governed only by drift and self-repair stabilizes at a modest error rate; adding fabrication reproduces the high rates seen in current LLMs. Giving each false claim even a small chance of peer review shifts the system to a truth-dominant state. We operationalize peer review with the open-source \emph{Flaws-of-Others (FOO) algorithm}: a configurable loop in which any set of agents critique one another while a harmoniser merges their verdicts. The takeaway is practical and cultural: reliability in this new medium comes not from perfecting single models but from wiring imperfect ones into networks that keep each other honest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19030v2">WiLLM: an Open Framework for LLM Services over Wireless Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) services fundamentally differ from traditional Deep Neural Network (DNN) applications in wireless networks. We identify three critical distinctions: (1) unlike traditional DNNs with unidirectional data flows, LLM's multimodal interactions create bidirectional heavy loads with contrasting bottlenecks, requiring direction-aware resource scheduling; (2) while traditional DNNs exhibit fixed computational patterns, LLM's highly variable inference times interact complexly with network slicing, causing dynamic bottleneck migration; and (3) in contrast to predictable DNN traffic, LLM's token streams demonstrate unprecedented burstiness and state dependencies. These insights motivate WiLLM, the first open-source framework, implemented as a wireless platform, for LLM service research. Built on OpenAirInterface, WiLLM introduces several technical innovations: dynamic slice compatibility, universal UE compatibility through application-layer tunneling, multi-UE multi-slice scheduling, dual-mode resource allocation, and cross-layer APIs. In addition, WiLLM eliminates the need for specialized wireless expertise, enabling researchers and developers to experiment with LLM services over realistic cellular networks. We demonstrate the platform's capabilities through a smart glasses case study and provide a comprehensive dataset of \~1.6 million synchronized measurements. The complete system, dataset, and appendix are available at https://openwillm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12399v2">FinSphere, a Real-Time Stock Analysis Agent Powered by Instruction-Tuned LLMs and Domain Tools</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Current financial large language models (FinLLMs) struggle with two critical limitations: the absence of objective evaluation metrics to assess the quality of stock analysis reports and a lack of depth in stock analysis, which impedes their ability to generate professional-grade insights. To address these challenges, this paper introduces FinSphere, a stock analysis agent, along with three major contributions: (1) AnalyScore, a systematic evaluation framework for assessing stock analysis quality, (2) Stocksis, a dataset curated by industry experts to enhance LLMs' stock analysis capabilities, and (3) FinSphere, an AI agent that can generate high-quality stock analysis reports in response to user queries. Experiments demonstrate that FinSphere achieves superior performance compared to both general and domain-specific LLMs, as well as existing agent-based systems, even when they are enhanced with real-time data access and few-shot guidance. The integrated framework, which combines real-time data feeds, quantitative tools, and an instruction-tuned LLM, yields substantial improvements in both analytical quality and practical applicability for real-world stock analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12022v3">Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06520v1">Gradientsys: A Multi-Agent LLM Scheduler with ReAct Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      We present Gradientsys, a next-generation multi-agent scheduling framework that coordinates diverse specialized AI agents using a typed Model-Context Protocol (MCP) and a ReAct-based dynamic planning loop. At its core, Gradientsys employs an LLM-powered scheduler for intelligent one-to-many task dispatch, enabling parallel execution of heterogeneous agents such as PDF parsers, web search modules, GUI controllers, and web builders. The framework supports hybrid synchronous/asynchronous execution, respects agent capacity constraints, and incorporates a robust retry-and-replan mechanism to handle failures gracefully. To promote transparency and trust, Gradientsys includes an observability layer streaming real-time agent activity and intermediate reasoning via Server-Sent Events (SSE). We offer an architectural overview and evaluate Gradientsys against existing frameworks in terms of extensibility, scheduling topology, tool reusability, parallelism, and observability. Experiments on the GAIA general-assistant benchmark show that Gradientsys achieves higher task success rates with reduced latency and lower API costs compared to a MinionS-style baseline, demonstrating the strength of its LLM-driven multi-agent orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21285v2">Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique. Our codes and data are available at https://github.com/XinXU-USTC/DoubleChecker
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17946v4">Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge Transfer for Backdoor Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning (FPFT). However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from the weak-to-strong based on Feature Alignment-enhanced Knowledge Distillation (FAKD). Specifically, we poison small-scale language models through FPFT to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through FAKD, which employs PEFT. Theoretical analysis reveals that FAKD has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of FAKD on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11951v3">SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 13 pages, 10 tables, 5 figures
    </div>
    <details class="paper-abstract">
      This paper introduces SagaLLM, a structured multi-agent architecture designed to address four foundational limitations of current LLM-based planning systems: unreliable self-validation, context loss, lack of transactional safeguards, and insufficient inter-agent coordination. While recent frameworks leverage LLMs for task decomposition and multi-agent communication, they often fail to ensure consistency, rollback, or constraint satisfaction across distributed workflows. SagaLLM bridges this gap by integrating the Saga transactional pattern with persistent memory, automated compensation, and independent validation agents. It leverages LLMs' generative reasoning to automate key tasks traditionally requiring hand-coded coordination logic, including state tracking, dependency analysis, log schema generation, and recovery orchestration. Although SagaLLM relaxes strict ACID guarantees, it ensures workflow-wide consistency and recovery through modular checkpointing and compensable execution. Empirical evaluations across planning domains demonstrate that standalone LLMs frequently violate interdependent constraints or fail to recover from disruptions. In contrast, SagaLLM achieves significant improvements in consistency, validation accuracy, and adaptive coordination under uncertainty, establishing a robust foundation for real-world, scalable LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06512v1">Towards LLM-based Root Cause Analysis of Hardware Design Failures</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 6 pages. Accepted for publication in IEEE COINS 2025 Special Session on LLMs for EDA and Security
    </div>
    <details class="paper-abstract">
      With advances in large language models (LLMs), new opportunities have emerged to develop tools that support the digital hardware design process. In this work, we explore how LLMs can assist with explaining the root cause of design issues and bugs that are revealed during synthesis and simulation, a necessary milestone on the pathway towards widespread use of LLMs in the hardware design process and for hardware security analysis. We find promising results: for our corpus of 34 different buggy scenarios, OpenAI's o3-mini reasoning model reached a correct determination 100% of the time under pass@5 scoring, with other state of the art models and configurations usually achieving more than 80% performance and more than 90% when assisted with retrieval-augmented generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06489v1">On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03711v3">Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Accepted paper at MAPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06463v1">Evaluating Efficiency and Novelty of LLM-Generated Code for Graph Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate software development, yet most prior evaluations focus on functional correctness or high-level languages such as Python. We present the first systematic study of LLMs' ability to generate efficient C implementations of graph-analysis routines--code that must satisfy the stringent runtime and memory constraints. Eight state-of-the-art models (OpenAI ChatGPT o3 and o4-mini-high, Anthropic Claude 4 Sonnet and Sonnet Extended, Google Gemini 2.5 Flash and Pro, xAI Grok 3-Think, and DeepSeek DeepThink R1) are benchmarked by two distinct approaches. The first approach checks the ability of LLMs in generating an algorithm outperforming other present algorithms in the benchmark. The second approach evaluates the ability of LLMs to generate graph algorithms for integration into the benchmark. Results show that Claude Sonnet 4 Extended achieves the best result in the case of ready-to-use code generation and efficiency, outperforming human-written baselines in triangle counting. The study confirms that contemporary LLMs excel at optimizing and integrating established algorithms but not inventing novel techniques. We provide prompts, the first approach's generated code, and measurement scripts to foster reproducible research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14879v3">Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-In-The-Loop LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Passive tracking methods, such as phone and wearable sensing, have become dominant in monitoring human behaviors in modern ubiquitous computing studies. While there have been significant advances in machine-learning approaches to translate periods of raw sensor data to model momentary behaviors, (e.g., physical activity recognition), there still remains a significant gap in the translation of these sensing streams into meaningful, high-level, context-aware insights that are required for various applications (e.g., summarizing an individual's daily routine). To bridge this gap, experts often need to employ a context-driven sensemaking process in real-world studies to derive insights. This process often requires manual effort and can be challenging even for experienced researchers due to the complexity of human behaviors. We conducted three rounds of user studies with 21 experts to explore solutions to address challenges with sensemaking. We follow a human-centered design process to identify needs and design, iterate, build, and evaluate Vital Insight (VI), a novel, LLM-assisted, prototype system to enable human-in-the-loop inference (sensemaking) and visualizations of multi-modal passive sensing data from smartphones and wearables. Using the prototype as a technology probe, we observe experts' interactions with it and develop an expert sensemaking model that explains how experts move between direct data representations and AI-supported inferences to explore, question, and validate insights. Through this iterative process, we also synthesize and discuss a list of design implications for the design of future AI-augmented visualization systems to better assist experts' sensemaking processes in multi-modal health sensing data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07328v1">Bridging the Plausibility-Validity Gap by Fine-Tuning a Reasoning-Enhanced LLM for Chemical Synthesis and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 42 pages, 8 figures, 1 equation, 2 algorithms, 31 tables, to be published in ISPACS Conference 2025, unabridged version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate scientifically plausible but factually invalid information, a challenge we term the "plausibility-validity gap," particularly in specialized domains like chemistry. This paper presents a systematic methodology to bridge this gap by developing a specialized scientific assistant. We utilized the Magistral Small model, noted for its integrated reasoning capabilities, and fine-tuned it using Low-Rank Adaptation (LoRA). A key component of our approach was the creation of a "dual-domain dataset," a comprehensive corpus curated from various sources encompassing both molecular properties and chemical reactions, which was standardized to ensure quality. Our evaluation demonstrates that the fine-tuned model achieves significant improvements over the baseline model in format adherence, chemical validity of generated molecules, and the feasibility of proposed synthesis routes. The results indicate a hierarchical learning pattern, where syntactic correctness is learned more readily than chemical possibility and synthesis feasibility. While a comparative analysis with human experts revealed competitive performance in areas like chemical creativity and reasoning, it also highlighted key limitations, including persistent errors in stereochemistry, a static knowledge cutoff, and occasional reference hallucination. This work establishes a viable framework for adapting generalist LLMs into reliable, specialized tools for chemical research, while also delineating critical areas for future improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07313v1">Frontier LLMs Still Struggle with Simple Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 53 pages
    </div>
    <details class="paper-abstract">
      While state-of-the-art large language models (LLMs) demonstrate advanced reasoning capabilities-achieving remarkable performance on challenging competitive math and coding benchmarks-they also frequently fail on tasks that are easy for humans. This work studies the performance of frontier LLMs on a broad set of such "easy" reasoning problems. By extending previous work in the literature, we create a suite of procedurally generated simple reasoning tasks, including counting, first-order logic, proof trees, and travel planning, with changeable parameters (such as document length. or the number of variables in a math problem) that can arbitrarily increase the amount of computation required to produce the answer while preserving the fundamental difficulty. While previous work showed that traditional, non-thinking models can be made to fail on such problems, we demonstrate that even state-of-the-art thinking models consistently fail on such problems and for similar reasons (e.g. statistical shortcuts, errors in intermediate steps, and difficulties in processing long contexts). To further understand the behavior of the models, we introduce the unpuzzles dataset, a different "easy" benchmark consisting of trivialized versions of well-known math and logic puzzles. Interestingly, while modern LLMs excel at solving the original puzzles, they tend to fail on the trivialized versions, exhibiting several systematic failure patterns related to memorizing the originals. We show that this happens even if the models are otherwise able to solve problems with different descriptions but requiring the same logic. Our results highlight that out-of-distribution generalization is still problematic for frontier language models and the new generation of thinking models, even for simple reasoning tasks, and making tasks easier does not necessarily imply improved performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19092v2">Rankers, Judges, and Assistants: Towards Understanding the Interplay of LLMs in Information Retrieval Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integral to information retrieval (IR), powering ranking, evaluation, and AI-assisted content creation. This widespread adoption necessitates a critical examination of potential biases arising from the interplay between these LLM-based components. This paper synthesizes existing research and presents novel experiment designs that explore how LLM-based rankers and assistants influence LLM-based judges. We provide the first empirical evidence of LLM judges exhibiting significant bias towards LLM-based rankers. Furthermore, we observe limitations in LLM judges' ability to discern subtle system performance differences. Contrary to some previous findings, our preliminary study does not find evidence of bias against AI-generated content. These results highlight the need for a more holistic view of the LLM-driven information ecosystem. To this end, we offer initial guidelines and a research agenda to ensure the reliable use of LLMs in IR evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07251v1">A Language-Driven Framework for Improving Personalized Recommendations: Merging LLMs with Traditional Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Traditional recommendation algorithms are not designed to provide personalized recommendations based on user preferences provided through text, e.g., "I enjoy light-hearted comedies with a lot of humor". Large Language Models (LLMs) have emerged as one of the most promising tools for natural language processing in recent years. This research proposes a novel framework that mimics how a close friend would recommend items based on their knowledge of an individual's tastes. We leverage LLMs to enhance movie recommendation systems by refining traditional algorithm outputs and integrating them with language-based user preference inputs. We employ Singular Value Decomposition (SVD) or SVD++ algorithms to generate initial movie recommendations, implemented using the Surprise Python library and trained on the MovieLens-Latest-Small dataset. We compare the performance of the base algorithms with our LLM-enhanced versions using leave-one-out validation hit rates and cumulative hit rates. Additionally, to compare the performance of our framework against the current state-of-the-art recommendation systems, we use rating and ranking metrics with an item-based stratified 0.75 train, 0.25 test split. Our framework can generate preference profiles automatically based on users' favorite movies or allow manual preference specification for more personalized results. Using an automated approach, our framework overwhelmingly surpassed SVD and SVD++ on every evaluation metric used (e.g., improvements of up to ~6x in cumulative hit rate, ~3.7x in NDCG, etc.), albeit at the cost of a slight increase in computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11005v4">A Theory of Response Sampling in LLMs: Part Descriptive and Part Prescriptive</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 ACL 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized in autonomous decision-making, where they sample options from vast action spaces. However, the heuristics that guide this sampling process remain under explored. We study this sampling behavior and show that this underlying heuristics resembles that of human decision-making: comprising a descriptive component (reflecting statistical norm) and a prescriptive component (implicit ideal encoded in the LLM) of a concept. We show that this deviation of a sample from the statistical norm towards a prescriptive component consistently appears in concepts across diverse real-world domains like public health, and economic trends. To further illustrate the theory, we demonstrate that concept prototypes in LLMs are affected by prescriptive norms, similar to the concept of normality in humans. Through case studies and comparison with human studies, we illustrate that in real-world applications, the shift of samples toward an ideal value in LLMs' outputs can result in significantly biased decision-making, raising ethical concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07236v1">An Information-Theoretic Perspective on Multi-LLM Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often behave inconsistently across inputs, indicating uncertainty and motivating the need for its quantification in high-stakes settings. Prior work on calibration and uncertainty quantification often focuses on individual models, overlooking the potential of model diversity. We hypothesize that LLMs make complementary predictions due to differences in training and the Zipfian nature of language, and that aggregating their outputs leads to more reliable uncertainty estimates. To leverage this, we propose MUSE (Multi-LLM Uncertainty via Subset Ensembles), a simple information-theoretic method that uses Jensen-Shannon Divergence to identify and aggregate well-calibrated subsets of LLMs. Experiments on binary prediction tasks demonstrate improved calibration and predictive performance compared to single-model and naive ensemble baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01077v3">Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14641v2">HLSTester: Efficient Testing of Behavioral Discrepancies with LLMs for High-Level Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 arXiv admin note: text overlap with arXiv:2407.03889
    </div>
    <details class="paper-abstract">
      In high-level synthesis (HLS), C/C++ programs with synthesis directives are used to generate circuits for FPGA implementations. However, hardware-specific and platform-dependent characteristics in these implementations can introduce behavioral discrepancies between the original C/C++ programs and the circuits after high-level synthesis. Existing methods for testing behavioral discrepancies in HLS are still immature, and the testing workflow requires significant human efforts. To address this challenge, we propose HLSTester, a large language model (LLM) aided testing framework that efficiently detects behavioral discrepancies in HLS. To mitigate hallucinations in LLMs and enhance prompt quality, the testbenches for original C/C++ programs are leveraged to guide LLMs in generating HLS-compatible testbenches, effectively eliminating certain traditional C/C++ constructs that are incompatible with HLS tools. Key variables are pinpointed through a backward slicing technique in both C/C++ and HLS programs to monitor their runtime spectra, enabling an in-depth analysis of the discrepancy symptoms. To reduce test time, a testing input generation mechanism is introduced to integrate dynamic mutation with insights from an LLM-based progressive reasoning chain. In addition, repetitive hardware testing is skipped by a redundancy-aware filtering technique for the generated test inputs. Experimental results demonstrate that the proposed LLM-aided testing framework significantly accelerates the testing workflow while achieving higher testbench simulation pass rates compared with the traditional method and the direct use of LLMs on the same HLS programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07188v1">Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 18 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known response biases are poorly understood. This paper investigates the response robustness of LLMs in normative survey contexts -- we test nine diverse LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of 11 perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also reveal that all tested models exhibit a consistent \textit{recency bias} varying in intensity, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. By applying a set of perturbations, we reveal that LLMs partially align with survey response biases identified in humans. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07186v1">Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
      | 💬 CoLM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit cognitive biases -- systematic tendencies of irrational decision-making, similar to those seen in humans. Prior work has found that these biases vary across models and can be amplified by instruction tuning. However, it remains unclear if these differences in biases stem from pretraining, finetuning, or even random noise due to training stochasticity. We propose a two-step causal experimental approach to disentangle these factors. First, we finetune models multiple times using different random seeds to study how training randomness affects over $30$ cognitive biases. Second, we introduce \emph{cross-tuning} -- swapping instruction datasets between models to isolate bias sources. This swap uses datasets that led to different bias patterns, directly testing whether biases are dataset-dependent. Our findings reveal that while training randomness introduces some variability, biases are mainly shaped by pretraining: models with the same pretrained backbone exhibit more similar bias patterns than those sharing only finetuning data. These insights suggest that understanding biases in finetuned models requires considering their pretraining origins beyond finetuning effects. This perspective can guide future efforts to develop principled strategies for evaluating and mitigating bias in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07146v1">An attention-aware GNN-based input defender against multi-turn jailbreak on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained widespread popularity and are increasingly integrated into various applications. However, their capabilities can be exploited for both benign and harmful purposes. Despite rigorous training and fine-tuning for safety, LLMs remain vulnerable to jailbreak attacks. Recently, multi-turn attacks have emerged, exacerbating the issue. Unlike single-turn attacks, multi-turn attacks gradually escalate the dialogue, making them more difficult to detect and mitigate, even after they are identified. In this study, we propose G-Guard, an innovative attention-aware GNN-based input classifier designed to defend against multi-turn jailbreak attacks on LLMs. G-Guard constructs an entity graph for multi-turn queries, explicitly capturing relationships between harmful keywords and queries even when those keywords appear only in previous queries. Additionally, we introduce an attention-aware augmentation mechanism that retrieves the most similar single-turn query based on the multi-turn conversation. This retrieved query is treated as a labeled node in the graph, enhancing the ability of GNN to classify whether the current query is harmful. Evaluation results demonstrate that G-Guard outperforms all baselines across all datasets and evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05816v1">Affective-ROPTester: Capability and Bias Analysis of LLMs in Predicting Retinopathy of Prematurity</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Despite the remarkable progress of large language models (LLMs) across various domains, their capacity to predict retinopathy of prematurity (ROP) risk remains largely unexplored. To address this gap, we introduce a novel Chinese benchmark dataset, termed CROP, comprising 993 admission records annotated with low, medium, and high-risk labels. To systematically examine the predictive capabilities and affective biases of LLMs in ROP risk stratification, we propose Affective-ROPTester, an automated evaluation framework incorporating three prompting strategies: Instruction-based, Chain-of-Thought (CoT), and In-Context Learning (ICL). The Instruction scheme assesses LLMs' intrinsic knowledge and associated biases, whereas the CoT and ICL schemes leverage external medical knowledge to enhance predictive accuracy. Crucially, we integrate emotional elements at the prompt level to investigate how different affective framings influence the model's ability to predict ROP and its bias patterns. Empirical results derived from the CROP dataset yield two principal observations. First, LLMs demonstrate limited efficacy in ROP risk prediction when operating solely on intrinsic knowledge, yet exhibit marked performance gains when augmented with structured external inputs. Second, affective biases are evident in the model outputs, with a consistent inclination toward overestimating medium- and high-risk cases. Third, compared to negative emotions, positive emotional framing contributes to mitigating predictive bias in model outputs. These findings highlight the critical role of affect-sensitive prompt engineering in enhancing diagnostic reliability and emphasize the utility of Affective-ROPTester as a framework for evaluating and mitigating affective bias in clinical language modeling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15790v2">ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 4 pages, 1 figure. Submitted to the 16th Asia-Pacific Symposium on Internetware (Internetware 2025)
    </div>
    <details class="paper-abstract">
      With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05754v1">LeAD: The LLM Enhanced Planning System Converged with End-to-end Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      A principal barrier to large-scale deployment of urban autonomous driving systems lies in the prevalence of complex scenarios and edge cases. Existing systems fail to effectively interpret semantic information within traffic contexts and discern intentions of other participants, consequently generating decisions misaligned with skilled drivers' reasoning patterns. We present LeAD, a dual-rate autonomous driving architecture integrating imitation learning-based end-to-end (E2E) frameworks with large language model (LLM) augmentation. The high-frequency E2E subsystem maintains real-time perception-planning-control cycles, while the low-frequency LLM module enhances scenario comprehension through multi-modal perception fusion with HD maps and derives optimal decisions via chain-of-thought (CoT) reasoning when baseline planners encounter capability limitations. Our experimental evaluation in the CARLA Simulator demonstrates LeAD's superior handling of unconventional scenarios, achieving 71 points on Leaderboard V1 benchmark, with a route completion of 93%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05750v1">DocTalk: Scalable Graph-based Dialogue Synthesis for Enhancing LLM Conversational Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted at SIGDIAL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in multi-turn conversational tasks, yet their pre-training data predominantly consists of continuous prose, creating a potential mismatch between required capabilities and training paradigms. We introduce a novel approach to address this discrepancy by synthesizing conversational data from existing text corpora. We present a pipeline that transforms a cluster of multiple related documents into an extended multi-turn, multi-topic information-seeking dialogue. Applying our pipeline to Wikipedia articles, we curate DocTalk, a multi-turn pre-training dialogue corpus consisting of over 730k long conversations. We hypothesize that exposure to such synthesized conversational structures during pre-training can enhance the fundamental multi-turn capabilities of LLMs, such as context memory and understanding. Empirically, we show that incorporating DocTalk during pre-training results in up to 40% gain in context memory and understanding, without compromising base performance. DocTalk is available at https://huggingface.co/datasets/AmazonScience/DocTalk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05740v1">GPTKB v1.5: A Massive Knowledge Base for Exploring Factual LLM Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 7 pages, 6 figures, 1 table
    </div>
    <details class="paper-abstract">
      Language models are powerful tools, yet their factual knowledge is still poorly understood, and inaccessible to ad-hoc browsing and scalable statistical analysis. This demonstration introduces GPTKB v1.5, a densely interlinked 100-million-triple knowledge base (KB) built for $14,000 from GPT-4.1, using the GPTKB methodology for massive-recursive LLM knowledge materialization (Hu et al., ACL 2025). The demonstration experience focuses on three use cases: (1) link-traversal-based LLM knowledge exploration, (2) SPARQL-based structured LLM knowledge querying, (3) comparative exploration of the strengths and weaknesses of LLM knowledge. Massive-recursive LLM knowledge materialization is a groundbreaking opportunity both for the research area of systematic analysis of LLM knowledge, as well as for automated KB construction. The GPTKB demonstrator is accessible at https://gptkb.org.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05733v1">When Transformers Meet Recommenders: Integrating Self-Attentive Sequential Recommendation with Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Self-Attentive Sequential Recommendation (SASRec) effectively captures long-term user preferences by applying attention mechanisms to historical interactions. Concurrently, the rise of Large Language Models (LLMs) has motivated research into LLM-based recommendation, which leverages their powerful generalization and language understanding capabilities. However, LLMs often lack the domain-specific knowledge and collaborative signals essential for high-quality recommendations when relying solely on textual prompts. To address this limitation, this study proposes SASRecLLM, a novel framework that integrates SASRec as a collaborative encoder with an LLM fine-tuned using Low-Rank Adaptation (LoRA). The components are connected via a mapping layer to align their dimensional spaces, and three targeted training strategies are designed to optimize the hybrid architecture. Extensive experiments on multiple datasets demonstrate that SASRecLLM achieves robust and consistent improvements over strong baselines in both cold-start and warm-start scenarios. This work advances the field of LLM-based recommendation by presenting a modular and effective paradigm for fusing structured collaborative filtering with the semantic power of fine-tuned LLMs. The implementation is available on GitHub: https://github.com/kechenkristin/RecLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02233v3">Enhancing LLM Reliability via Explicit Knowledge Boundary Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucination stemming from misaligned self-awareness, particularly when processing queries exceeding their knowledge boundaries. While existing mitigation strategies employ uncertainty estimation or query rejection mechanisms, they suffer from computational efficiency and sacrificed helpfulness. To address these issues, we propose the Explicit Knowledge Boundary Modeling (EKBM) framework, integrating fast and slow reasoning systems to harmonize reliability and usability. The framework first employs a fast-thinking model to generate confidence-labeled responses, enabling immediate utilization of high-confidence outputs, whereas uncertain predictions trigger a slow refinement model for accuracy improvement. To align model behavior with our proposed object, we propose a hybrid training pipeline, enhancing self-awareness without degrading task performance. Evaluations on dialogue state tracking tasks demonstrate that EKBM achieves superior model reliability over uncertainty-based baselines. Further analysis reveals that refinement substantially boosts accuracy while maintaining low computational overhead. The framework establishes a scalable paradigm for deploying reliable LLMs in error-sensitive applications, effectively balancing accuracy and practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02962v2">RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05686v1">Smoothie-Qwen: Post-Hoc Smoothing to Reduce Language Bias in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) often exhibit language confusion, a tendency to generate responses in a dominant language irrespective of the prompt's language. To address this, we propose Smoothie-Qwen, a lightweight, post-hoc method that mitigates language bias without retraining. This technique selectively adjusts token-level output probabilities to effectively suppress undesired language generation. Applied to the Qwen model, our method reduces unintended Chinese output by over 95% while preserving task accuracy on multilingual benchmarks. This work provides a practical and efficient solution for enhancing the language controllability of LLMs, making them more reliable for global applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03020v2">AI Literacy and LLM Engagement in Higher Education: A Cross-National Quantitative Study</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 26 pages, 8 figures, 3 tables. Submitted for consideration in a forthcoming issue of the International Journal of Educational Technology in Higher Education
    </div>
    <details class="paper-abstract">
      This study presents a cross-national quantitative analysis of how university students in the United States and Bangladesh interact with Large Language Models (LLMs). Based on an online survey of 318 students, results show that LLMs enhance access to information, improve writing, and boost academic performance. However, concerns about overreliance, ethical risks, and critical thinking persist. Guided by the AI Literacy Framework, Expectancy-Value Theory, and Biggs' 3P Model, the study finds that motivational beliefs and technical competencies shape LLM engagement. Significant correlations were found between LLM use and perceived literacy benefits (r = .59, p < .001) and optimism (r = .41, p < .001). ANOVA results showed more frequent use among U.S. students (F = 7.92, p = .005) and STEM majors (F = 18.11, p < .001). Findings support the development of ethical, inclusive, and pedagogically sound frameworks for integrating LLMs in higher education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05639v1">ECom-Bench: Can LLM Agent Resolve Real-World E-commerce Customer Support Issues?</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      In this paper, we introduce ECom-Bench, the first benchmark framework for evaluating LLM agent with multimodal capabilities in the e-commerce customer support domain. ECom-Bench features dynamic user simulation based on persona information collected from real e-commerce customer interactions and a realistic task dataset derived from authentic e-commerce dialogues. These tasks, covering a wide range of business scenarios, are designed to reflect real-world complexities, making ECom-Bench highly challenging. For instance, even advanced models like GPT-4o achieve only a 10-20% pass^3 metric in our benchmark, highlighting the substantial difficulties posed by complex e-commerce scenarios. Upon publication, the code and data will be open-sourced to facilitate further research and development in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05638v1">LLMs are Introvert</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      The exponential growth of social media and generative AI has transformed information dissemination, fostering connectivity but also accelerating the spread of misinformation. Understanding information propagation dynamics and developing effective control strategies is essential to mitigate harmful content. Traditional models, such as SIR, provide basic insights but inadequately capture the complexities of online interactions. Advanced methods, including attention mechanisms and graph neural networks, enhance accuracy but typically overlook user psychology and behavioral dynamics. Large language models (LLMs), with their human-like reasoning, offer new potential for simulating psychological aspects of information spread. We introduce an LLM-based simulation environment capturing agents' evolving attitudes, emotions, and responses. Initial experiments, however, revealed significant gaps between LLM-generated behaviors and authentic human dynamics, especially in stance detection and psychological realism. A detailed evaluation through Social Information Processing Theory identified major discrepancies in goal-setting and feedback evaluation, stemming from the lack of emotional processing in standard LLM training. To address these issues, we propose the Social Information Processing-based Chain of Thought (SIP-CoT) mechanism enhanced by emotion-guided memory. This method improves the interpretation of social cues, personalization of goals, and evaluation of feedback. Experimental results confirm that SIP-CoT-enhanced LLM agents more effectively process social information, demonstrating behaviors, attitudes, and emotions closer to real human interactions. In summary, this research highlights critical limitations in current LLM-based propagation simulations and demonstrates how integrating SIP-CoT and emotional memory significantly enhances the social intelligence and realism of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16327v2">Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05630v1">How Not to Detect Prompt Injections with an LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      LLM-integrated applications and agents are vulnerable to prompt injection attacks, in which adversaries embed malicious instructions within seemingly benign user inputs to manipulate the LLM's intended behavior. Recent defenses based on $\textit{known-answer detection}$ (KAD) have achieved near-perfect performance by using an LLM to classify inputs as clean or contaminated. In this work, we formally characterize the KAD framework and uncover a structural vulnerability in its design that invalidates its core security premise. We design a methodical adaptive attack, $\textit{DataFlip}$, to exploit this fundamental weakness. It consistently evades KAD defenses with detection rates as low as $1.5\%$ while reliably inducing malicious behavior with success rates of up to $88\%$, without needing white-box access to the LLM or any optimization procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05629v1">Enhancing Student Learning with LLM-Generated Retrieval Practice Questions: An Empirical Study in Data Science Courses</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Retrieval practice is a well-established pedagogical technique known to significantly enhance student learning and knowledge retention. However, generating high-quality retrieval practice questions is often time-consuming and labor intensive for instructors, especially in rapidly evolving technical subjects. Large Language Models (LLMs) offer the potential to automate this process by generating questions in response to prompts, yet the effectiveness of LLM-generated retrieval practice on student learning remains to be established. In this study, we conducted an empirical study involving two college-level data science courses, with approximately 60 students. We compared learning outcomes during one week in which students received LLM-generated multiple-choice retrieval practice questions to those from a week in which no such questions were provided. Results indicate that students exposed to LLM-generated retrieval practice achieved significantly higher knowledge retention, with an average accuracy of 89%, compared to 73% in the week without such practice. These findings suggest that LLM-generated retrieval questions can effectively support student learning and may provide a scalable solution for integrating retrieval practice into real-time teaching. However, despite these encouraging outcomes and the potential time-saving benefits, cautions must be taken, as the quality of LLM-generated questions can vary. Instructors must still manually verify and revise the generated questions before releasing them to students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05617v1">Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Accepted by ACL 2025 main
    </div>
    <details class="paper-abstract">
      Knowledge distillation typically involves transferring knowledge from a Large Language Model (LLM) to a Smaller Language Model (SLM). However, in tasks such as text matching, fine-tuned smaller models often yield more effective domain-specific representations, as they focus on optimizing the similarity of input pairs. To leverage both the specialized strengths of small models and the rich semantic understanding of LLMs, we introduce a flipped knowledge distillation paradigm, where LLM learns from SLM. Specifically, we address the architectural gap between decoder-only LLMs and smaller encoder-based models by reinterpreting LLMs in an encoder-decoder manner using LoRA. The encoder generates compressed representations, while the decoder maps them to the output space. During training, the encoder produces representations and their similarities, which are then aligned with the similarity scores produced by the teacher, using our proposed Margin-aware Contrastive Learning (MCL) approach. The MCL ensures accurate similarity for both positive and negative pairs, and adaptively handles the internal differences within positive and negative samples. Our paradigm requires only a reasonably good-performing SLM, allowing the LLM to achieve improved performance. Experiments on financial and healthcare benchmarks, as well as real-world applications, confirm its effectiveness, and the model has been fully deployed in an online environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05598v1">Self-Review Framework for Enhancing Instruction Following Capability of LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Various techniques have been proposed to improve large language models (LLMs) adherence to formatting and instruction constraints. One of the most effective approaches involves utilizing high-quality data generated by powerful models. However, such models often fail to fully comply with complex instructions in a single generation. To address this limitation, iterative revision methods have been introduced. Nevertheless, as the number of data points and revision iterations increases, the associated monetary costs grow significantly. As a resource-efficient alternative, methods have been proposed that leverage high-performance evaluation tools to compensate for the limited self-evaluation capabilities of open-source LLMs. However, these approaches often lead to a degradation in output quality due to excessive revision. To overcome these challenges, we propose Re5, a self-evaluation and revision framework designed to enhance instruction-following performance while preserving the quality of the generated content. Re5 extracts task and constraint components from user instructions, performs structural evaluations to prevent error accumulation, and applies fine-grained constraint-specific content evaluations followed by selective revisions. This process ensures precise and quality-preserving improvements. The final high-quality outputs are used for alignment tuning, enabling long-term alignment improvements through a data-centric iterative refinement loop. Experimental results demonstrate that Re5 achieves instruction-following performance comparable to models trained on data generated by GPT-4o-mini, a high-performance model, even with a small amount of data while maintaining response quality with a 64.24%-win rate over the non-revised initial responses. These results validate Re5 as an efficient and effective solution for enhancing instruction adherence with minimal external supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05578v1">The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they also exhibit memorization of their training data. This phenomenon raises critical questions about model behavior, privacy risks, and the boundary between learning and memorization. Addressing these concerns, this paper synthesizes recent studies and investigates the landscape of memorization, the factors influencing it, and methods for its detection and mitigation. We explore key drivers, including training data duplication, training dynamics, and fine-tuning procedures that influence data memorization. In addition, we examine methodologies such as prefix-based extraction, membership inference, and adversarial prompting, assessing their effectiveness in detecting and measuring memorized content. Beyond technical analysis, we also explore the broader implications of memorization, including the legal and ethical implications. Finally, we discuss mitigation strategies, including data cleaning, differential privacy, and post-training unlearning, while highlighting open challenges in balancing the minimization of harmful memorization with utility. This paper provides a comprehensive overview of the current state of research on LLM memorization across technical, privacy, and performance dimensions, identifying critical directions for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05577v1">Beyond Retrieval: Ensembling Cross-Encoders and GPT Rerankers with LLMs for Biomedical QA</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 Paper submitted to CLEF 2025 CEUR-WS
    </div>
    <details class="paper-abstract">
      Biomedical semantic question answering rooted in information retrieval can play a crucial role in keeping up to date with vast, rapidly evolving and ever-growing biomedical literature. A robust system can help researchers, healthcare professionals and even layman users access relevant knowledge grounded in evidence. The BioASQ 2025 Task13b Challenge serves as an important benchmark, offering a competitive platform for advancement of this space. This paper presents the methodologies and results from our participation in this challenge where we built a Retrieval-Augmented Generation (RAG) system that can answer biomedical questions by retrieving relevant PubMed documents and snippets to generate answers. For the retrieval task, we generated dense embeddings from biomedical articles for initial retrieval, and applied an ensemble of finetuned cross-encoders and large language models (LLMs) for re-ranking to identify top relevant documents. Our solution achieved an MAP@10 of 0.1581, placing 10th on the leaderboard for the retrieval task. For answer generation, we employed few-shot prompting of instruction-tuned LLMs. Our system achieved macro-F1 score of 0.95 for yes/no questions (rank 12), Mean Reciprocal Rank (MRR) of 0.64 for factoid questions (rank 1), mean-F1 score of 0.63 for list questions (rank 5), and ROUGE-SU4 F1 score of 0.29 for ideal answers (rank 11).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15740v2">SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed as autonomous agents in complex and long horizon settings, it is critical to evaluate their ability to sabotage users by pursuing hidden objectives. We study the ability of frontier LLMs to evade monitoring and achieve harmful hidden goals while completing a wide array of realistic tasks. We evaluate a broad range of frontier LLMs using SHADE (Subtle Harmful Agent Detection & Evaluation)-Arena, the first highly diverse agent evaluation dataset for sabotage and monitoring capabilities of LLM agents. SHADE-Arena consists of complex pairs of benign main tasks and harmful side objectives in complicated environments. Agents are evaluated on their ability to complete the side task without appearing suspicious to an LLM monitor. When measuring agent ability to (a) complete the main task, (b) complete the side task, and (c) avoid detection, we find that the best performing frontier models score 27% (Claude 3.7 Sonnet) and 15% (Gemini 2.5 Pro) as sabotage agents when overseen by Claude 3.6 Sonnet. For current frontier models, success on the side task relies heavily on having access to a hidden scratchpad that is not visible to the monitor. We also use SHADE-Arena to measure models' monitoring abilities, with the top monitor (Gemini 2.5 Pro) achieving an AUC of 0.87 at distinguishing benign and malign transcripts. We find that for now, models still struggle at sabotage due to failures in long-context main task execution. However, our measurements already demonstrate the difficulty of monitoring for subtle sabotage attempts, which we expect to only increase in the face of more complex and longer-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06376v1">SLDB: An End-To-End Heterogeneous System-on-Chip Benchmark Suite for LLM-Aided Design</a></div>
    <div class="paper-meta">
      📅 2025-07-08
    </div>
    <details class="paper-abstract">
      Over the last few years, Large Language Models (LLMs) have emerged as a valuable tool for Electronic Design Automation (EDA). State-of-the-art research in LLM-aided design has demonstrated the ability of LLMs to generate syntactically correct RTL code, showcasing encouraging prospects for integrating AI into the hardware design process. A key enabler of these advancements is the availability of high-quality benchmarks to evaluate new approaches. However, existing datasets and benchmarks fall short of system-level design, as they focus primarily on component-level information and low-complexity designs. To address this gap, we introduce the System-Level Design Benchmark (SLDB), a dataset tailored for evaluating LLMs in system-level integration and configuration tasks. SLDB includes a curated benchmark suite of 10 baseline SoC designs, whose components can be combined into an exponential number of distinct tile-based SoCs through a synthetic library. The dataset provides full SoC configurations, accelerator integration code, communication parameters, and accelerator-aware system configurations, along with testing-application code, compatible with the ESP platform[1].
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00754v2">Language-Unlocked ViT (LUViT): Empowering Self-Supervised Vision Transformers with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-08
      | 💬 26 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Model (LLMs) blocks with Vision Transformers (ViTs) holds immense promise for vision-only tasks by leveraging the rich semantic knowledge and reasoning capabilities of LLMs. However, a fundamental challenge lies in the inherent modality mismatch between text-centric pretraining of LLMs and vision-centric training of ViTs. Direct fusion often fails to fully exploit the LLM's potential and suffers from unstable finetuning. As a result, LLM blocks are kept frozen while only the vision components are learned. As a remedy to these challenges, we introduce Language-Unlocked Vision Transformers (LUViT), a novel approach that bridges this modality mismatch through a synergistic pre-training strategy. LUViT co-adapts a ViT backbone and an LLM fusion block by (1) employing Masked Auto-Encoding (MAE) to pre-train the ViT for richer visual representations, and (2) concurrently training Low-Rank Adaptation (LoRA) layers within the LLM block using the MAE objective. This joint optimization guides the ViT to produce LLM-aligned features and the LLM to effectively interpret visual information. We demonstrate through extensive experiments that LUViT significantly improves performance on various downstream vision tasks, showcasing a more effective and efficient pathway to harness LLM knowledge for visual understanding.
    </details>
</div>
