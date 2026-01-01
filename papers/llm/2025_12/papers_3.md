# llm - 2025_12

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20352v1">Multi-LLM Thematic Analysis with Dual Reliability Metrics: Combining Cohen's Kappa and Semantic Similarity for Qualitative Research Validation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 11 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Qualitative research faces a critical reliability challenge: traditional inter-rater agreement methods require multiple human coders, are time-intensive, and often yield moderate consistency. We present a multi-perspective validation framework for LLM-based thematic analysis that combines ensemble validation with dual reliability metrics: Cohen's Kappa ($κ$) for inter-rater agreement and cosine similarity for semantic consistency. Our framework enables configurable analysis parameters (1-6 seeds, temperature 0.0-2.0), supports custom prompt structures with variable substitution, and provides consensus theme extraction across any JSON format. As proof-of-concept, we evaluate three leading LLMs (Gemini 2.5 Pro, GPT-4o, Claude 3.5 Sonnet) on a psychedelic art therapy interview transcript, conducting six independent runs per model. Results demonstrate Gemini achieves highest reliability ($κ= 0.907$, cosine=95.3%), followed by GPT-4o ($κ= 0.853$, cosine=92.6%) and Claude ($κ= 0.842$, cosine=92.1%). All three models achieve a high agreement ($κ> 0.80$), validating the multi-run ensemble approach. The framework successfully extracts consensus themes across runs, with Gemini identifying 6 consensus themes (50-83% consistency), GPT-4o identifying 5 themes, and Claude 4 themes. Our open-source implementation provides researchers with transparent reliability metrics, flexible configuration, and structure-agnostic consensus extraction, establishing methodological foundations for reliable AI-assisted qualitative research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20324v1">Can LLMs Solve My Grandma's Riddle? Evaluating Multilingual Large Language Models on Reasoning Traditional Bangla Tricky Riddles</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show impressive performance on many NLP benchmarks, yet their ability to reason in figurative, culturally grounded, and low-resource settings remains underexplored. We address this gap for Bangla by introducing BanglaRiddleEval, a benchmark of 1,244 traditional Bangla riddles instantiated across four tasks (4,976 riddle-task artifacts in total). Using an LLM-based pipeline, we generate Chain-of-Thought explanations, semantically coherent distractors, and fine-grained ambiguity annotations, and evaluate a diverse suite of open-source and closed-source models under different prompting strategies. Models achieve moderate semantic overlap on generative QA but low correctness, MCQ accuracy peaks at only about 56% versus an 83% human baseline, and ambiguity resolution ranges from roughly 26% to 68%, with high-quality explanations confined to the strongest models. These results show that current LLMs capture some cues needed for Bangla riddle reasoning but remain far from human-level performance, establishing BanglaRiddleEval as a challenging new benchmark for low-resource figurative reasoning. All data, code, and evaluation scripts are available on GitHub: https://github.com/Labib1610/BanglaRiddleEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01564v2">Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, but often exhibit overconfidence and generate plausible yet incorrect answers. This overconfidence, especially in models undergone Reinforcement Learning from Human Feedback (RLHF), poses significant challenges for reliable uncertainty estimation and safe deployment. In this paper, we propose EAGLE (Expectation of AGgregated internaL bEief), a novel self-evaluation-based calibration method that leverages the internal hidden states of LLMs to derive more accurate confidence scores. Instead of relying on the model's final output, our approach extracts internal beliefs from multiple intermediate layers during self-evaluation. By aggregating these layer-wise beliefs and calculating the expectation over the resulting confidence score distribution, EAGLE produces a refined confidence score that more faithfully reflects the model's internal certainty. Extensive experiments on diverse datasets and LLMs demonstrate that EAGLE significantly improves calibration performance over existing baselines. We also provide an in-depth analysis of EAGLE, including a layer-wise examination of uncertainty patterns, a study of the impact of self-evaluation prompts, and an analysis of the effect of self-evaluation score range.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20298v1">Patterns vs. Patients: Evaluating LLMs against Mental Health Professionals on Personality Disorder Diagnosis through First-Person Narratives</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Growing reliance on LLMs for psychiatric self-assessment raises questions about their ability to interpret qualitative patient narratives. We present the first direct comparison between state-of-the-art LLMs and mental health professionals in diagnosing Borderline (BPD) and Narcissistic (NPD) Personality Disorders utilizing Polish-language first-person autobiographical accounts. We show that the top-performing Gemini Pro models surpassed human professionals in overall diagnostic accuracy by 21.91 percentage points (65.48% vs. 43.57%). While both models and human experts excelled at identifying BPD (F1 = 83.4 & F1 = 80.0, respectively), models severely underdiagnosed NPD (F1 = 6.7 vs. 50.0), showing a reluctance toward the value-laden term "narcissism." Qualitatively, models provided confident, elaborate justifications focused on patterns and formal categories, while human experts remained concise and cautious, emphasizing the patient's sense of self and temporal experience. Our findings demonstrate that while LLMs are highly competent at interpreting complex first-person clinical data, they remain subject to critical reliability and bias issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20237v1">MemR$^3$: Memory Retrieval via Reflective Reasoning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Memory systems have been designed to leverage past experiences in Large Language Model (LLM) agents. However, many deployed memory systems primarily optimize compression and storage, with comparatively less emphasis on explicit, closed-loop control of memory retrieval. From this observation, we build memory retrieval as an autonomous, accurate, and compatible agent system, named MemR$^3$, which has two core mechanisms: 1) a router that selects among retrieve, reflect, and answer actions to optimize answer quality; 2) a global evidence-gap tracker that explicitly renders the answering process transparent and tracks the evidence collection process. This design departs from the standard retrieve-then-answer pipeline by introducing a closed-loop control mechanism that enables autonomous decision-making. Empirical results on the LoCoMo benchmark demonstrate that MemR$^3$ surpasses strong baselines on LLM-as-a-Judge score, and particularly, it improves existing retrievers across four categories with an overall improvement on RAG (+7.29%) and Zep (+1.94%) using GPT-4.1-mini backend, offering a plug-and-play controller for existing memory stores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20210v1">Predictive-LoRA: A Proactive and Fragmentation-Aware Serverless Inference System for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      The serverless computing paradigm offers compelling advantages for deploying Large Language Model (LLM) inference services, including elastic scaling and pay-per-use billing. However, serving multiple fine-tuned LLMs via Low-Rank Adaptation (LoRA) in serverless environments faces critical challenges: reactive adapter loading causes significant cold start latency, and frequent adapter swapping leads to severe GPU memory fragmentation. In this paper, we present Predictive-LoRA (P-LoRA), a proactive and fragmentation-aware serverless inference system for LoRA-based LLMs. P-LoRA introduces two key innovations: (1) a lightweight LSTM-based traffic predictor that forecasts adapter demand and proactively prefetches hot adapters from host memory to GPU, reducing cold start latency by up to 68%; and (2) a page-based adapter memory management mechanism inspired by operating system virtual memory, which keeps GPU memory utilization above 87% even under heterogeneous adapter ranks. We evaluate P-LoRA using production-like workloads derived from the Azure Functions trace. Experimental results demonstrate that P-LoRA achieves 1.52x higher throughput than S-LoRA while reducing the average Time-To-First-Token (TTFT) by 35% under high concurrency scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16193v3">Fast LLM Post-training via Decoupled and Fastest-of-N Speculation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Rollout dominates the training time in large language model (LLM) post-training, where the trained model is used to generate tokens given a batch of prompts. This work, SpecActor, achieves fast rollout with speculative decoding that deploys a fast draft path to accelerate the unparallelizable generation, while the correctness is guaranteed by fast parallel verification of the outputs with the original model. SpecActor addresses two foundational challenges that hinder speculation efficiency: (1) a Decoupled speculation method that overcomes the computation inefficiency issue when executing speculative decoding with relative large per-worker batch size -- a common configuration in training but unfriendly to speculation, and (2) a Fastest-of-N speculation method that selects and combines different draft methods according to the rollout progress to approximate the optimal draft method even when the best one is unknown a priori. Extensive evaluations on production traces show that SpecActor accelerates mean rollout speed by 2.0--2.4x, with up to 2.7x speedup, over common post-training baselines. The results are consistent across both dense and MoE models and across different RL algorithms. Notably, SpecActor is 1.1--2.6x faster compared to vanilla speculative rollout in different traces. The accelerated rollout achieves 1.4--2.3x faster end-to-end training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20179v1">RESPOND: Risk-Enhanced Structured Pattern for LLM-driven Online Node-level Decision-making</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 28 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Current LLM-based driving agents that rely on unstructured plain-text memory suffer from low-precision scene retrieval and inefficient reflection. To address this limitation, we present RESPOND, a structured decision-making framework for LLM-driven agents grounded in explicit risk patterns. RESPOND represents each ego-centric scene using a unified 5 by 3 matrix that encodes spatial topology and road constraints, enabling consistent and reliable retrieval of spatial risk configurations. Based on this representation, a hybrid rule and LLM decision pipeline is developed with a two-tier memory mechanism. In high-risk contexts, exact pattern matching enables rapid and safe reuse of verified actions, while in low-risk contexts, sub-pattern matching supports personalized driving style adaptation. In addition, a pattern-aware reflection mechanism abstracts tactical corrections from crash and near-miss frames to update structured memory, achieving one-crash-to-generalize learning. Extensive experiments demonstrate the effectiveness of RESPOND. In highway-env, RESPOND outperforms state-of-the-art LLM-based and reinforcement learning based driving agents while producing substantially fewer collisions. With step-wise human feedback, the agent acquires a Sporty driving style within approximately 20 decision steps through sub-pattern abstraction. For real-world validation, RESPOND is evaluated on 53 high-risk cut-in scenarios extracted from the HighD dataset. For each event, intervention is applied immediately before the cut-in and RESPOND re-decides the driving action. Compared to recorded human behavior, RESPOND reduces subsequent risk in 84.9 percent of scenarios, demonstrating its practical feasibility under real-world driving conditions. These results highlight RESPONDs potential for autonomous driving, personalized driving assistance, and proactive hazard mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20169v1">Learning to Reason in LLMs by Expectation Maximization</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 12 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) solve reasoning problems by first generating a rationale and then answering. We formalize reasoning as a latent variable model and derive an expectation-maximization (EM) objective for learning to reason. This view connects EM and modern reward-based optimization, and shows that the main challenge lies in designing a sampling distribution that generates rationales that justify correct answers. We instantiate and compare several sampling schemes: rejection sampling with a budget, self-taught reasoner (STaR), and prompt posterior sampling (PPS), which only keeps the rationalization stage of STaR. Our experiments on the ARC, MMLU, and OpenBookQA datasets with the Llama and Qwen models show that the sampling scheme can significantly affect the accuracy of learned reasoning models. Despite its simplicity, we observe that PPS outperforms the other sampling schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20168v1">Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026
    </div>
    <details class="paper-abstract">
      By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20164v1">AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20159v1">AXIOM: Benchmarking LLM-as-a-Judge for Code via Rule-Based Perturbation and Multisource Quality Calibration</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly deployed in real-world software engineering, fostering the development of code evaluation metrics to study the quality of LLM-generated code. Conventional rule-based metrics merely score programs based on their surface-level similarities with reference programs instead of analyzing functionality and code quality in depth. To address this limitation, researchers have developed LLM-as-a-judge metrics, prompting LLMs to evaluate and score code, and curated various code evaluation benchmarks to validate their effectiveness. However, these benchmarks suffer from critical limitations, hindering reliable assessments of evaluation capability: Some feature coarse-grained binary labels, which reduce rich code behavior to a single bit of information, obscuring subtle errors. Others propose fine-grained but subjective, vaguely-defined evaluation criteria, introducing unreliability in manually-annotated scores, which is the ground-truth they rely on. Furthermore, they often use uncontrolled data synthesis methods, leading to unbalanced score distributions that poorly represent real-world code generation scenarios. To curate a diverse benchmark with programs of well-balanced distributions across various quality levels and streamline the manual annotation procedure, we propose AXIOM, a novel perturbation-based framework for synthesizing code evaluation benchmarks at scale. It reframes program scores as the refinement effort needed for deployment, consisting of two stages: (1) Rule-guided perturbation, which prompts LLMs to apply sequences of predefined perturbation rules to existing high-quality programs to modify their functionality and code quality, enabling us to precisely control each program's target score to achieve balanced score distributions. (2) Multisource quality calibration, which first selects a subset of...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05594v3">SoK: Are Watermarks in LLMs Ready for Deployment?</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20140v1">Enhancing Zero-Shot Time Series Forecasting in Off-the-Shelf LLMs via Noise Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 9 pages,3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated effectiveness as zero-shot time series (TS) forecasters. The key challenge lies in tokenizing TS data into textual representations that align with LLMs' pre-trained knowledge. While existing work often relies on fine-tuning specialized modules to bridge this gap, a distinct, yet challenging, paradigm aims to leverage truly off-the-shelf LLMs without any fine-tuning whatsoever, relying solely on strategic tokenization of numerical sequences. The performance of these fully frozen models is acutely sensitive to the textual representation of the input data, as their parameters cannot adapt to distribution shifts. In this paper, we introduce a simple yet highly effective strategy to overcome this brittleness: injecting noise into the raw time series before tokenization. This non-invasive intervention acts as a form of inference-time augmentation, compelling the frozen LLM to extrapolate based on robust underlying temporal patterns rather than superficial numerical artifacts. We theoretically analyze this phenomenon and empirically validate its effectiveness across diverse benchmarks. Notably, to fully eliminate potential biases from data contamination during LLM pre-training, we introduce two novel TS datasets that fall outside all utilized LLMs' pre-training scopes, and consistently observe improved performance. This study provides a further step in directly leveraging off-the-shelf LLMs for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17814v2">LLM-based Behaviour Driven Development for Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 7 pages, keynote given at 2nd International Symposium on Artificial Intelligence and Internet of Things (AIIoT-25), December 22-24th, 2025
    </div>
    <details class="paper-abstract">
      Test and verification are essential activities in hardware and system design, but their complexity grows significantly with increasing system sizes. While Behavior Driven Development (BDD) has proven effective in software engineering, it is not yet well established in hardware design, and its practical use remains limited. One contributing factor is the manual effort required to derive precise behavioral scenarios from textual specifications. Recent advances in Large Language Models (LLMs) offer new opportunities to automate this step. In this paper, we investigate the use of LLM-based techniques to support BDD in the context of hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20111v1">ABBEL: LLM Agents Acting through Belief Bottlenecks Expressed in Language</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      As the length of sequential decision-making tasks increases, it becomes computationally impractical to keep full interaction histories in context. We introduce a general framework for LLM agents to maintain concise contexts through multi-step interaction: Acting through Belief Bottlenecks Expressed in Language (ABBEL), and methods to further improve ABBEL agents with RL post-training. ABBEL replaces long multi-step interaction history by a belief state, i.e., a natural language summary of what has been discovered about task-relevant unknowns. Under ABBEL, at each step the agent first updates a prior belief with the most recent observation from the environment to form a posterior belief, then uses only the posterior to select an action. We systematically evaluate frontier models under ABBEL across six diverse multi-step environments, finding that ABBEL supports generating interpretable beliefs while maintaining near-constant memory use over interaction steps. However, bottleneck approaches are generally prone to error propagation, which we observe causing inferior performance when compared to the full context setting due to errors in belief updating. Therefore, we train LLMs to generate and act on beliefs within the ABBEL framework via reinforcement learning (RL). We experiment with belief grading, to reward higher quality beliefs, as well as belief length penalties to reward more compressed beliefs. Our experiments demonstrate the ability of RL to improve ABBEL's performance beyond the full context setting, while using less memory than contemporaneous approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20082v1">Adaptive Financial Sentiment Analysis for NIFTY 50 via Instruction-Tuned LLMs , RAG and Reinforcement Learning Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted in CODS 2025
    </div>
    <details class="paper-abstract">
      Financial sentiment analysis plays a crucial role in informing investment decisions, assessing market risk, and predicting stock price trends. Existing works in financial sentiment analysis have not considered the impact of stock prices or market feedback on sentiment analysis. In this paper, we propose an adaptive framework that integrates large language models (LLMs) with real-world stock market feedback to improve sentiment classification in the context of the Indian stock market. The proposed methodology fine-tunes the LLaMA 3.2 3B model using instruction-based learning on the SentiFin dataset. To enhance sentiment predictions, a retrieval-augmented generation (RAG) pipeline is employed that dynamically selects multi-source contextual information based on the cosine similarity of the sentence embeddings. Furthermore, a feedback-driven module is introduced that adjusts the reliability of the source by comparing predicted sentiment with actual next-day stock returns, allowing the system to iteratively adapt to market behavior. To generalize this adaptive mechanism across temporal data, a reinforcement learning agent trained using proximal policy optimization (PPO) is incorporated. The PPO agent learns to optimize source weighting policies based on cumulative reward signals from sentiment-return alignment. Experimental results on NIFTY 50 news headlines collected from 2024 to 2025 demonstrate that the proposed system significantly improves classification accuracy, F1-score, and market alignment over baseline models and static retrieval methods. The results validate the potential of combining instruction-tuned LLMs with dynamic feedback and reinforcement learning for robust, market-aware financial sentiment modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20062v1">On the Effectiveness of Instruction-Tuning Local LLMs for Identifying Software Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 The 9th International Conference on Mobile Internet Security (MobiSec 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show significant promise in automating software vulnerability analysis, a critical task given the impact of security failure of modern software systems. However, current approaches in using LLMs to automate vulnerability analysis mostly rely on using online API-based LLM services, requiring the user to disclose the source code in development. Moreover, they predominantly frame the task as a binary classification(vulnerable or not vulnerable), limiting potential practical utility. This paper addresses these limitations by reformulating the problem as Software Vulnerability Identification (SVI), where LLMs are asked to output the type of weakness in Common Weakness Enumeration (CWE) IDs rather than simply indicating the presence or absence of a vulnerability. We also tackle the reliance on large, API-based LLMs by demonstrating that instruction-tuning smaller, locally deployable LLMs can achieve superior identification performance. In our analysis, instruct-tuning a local LLM showed better overall performance and cost trade-off than online API-based LLMs. Our findings indicate that instruct-tuned local models represent a more effective, secure, and practical approach for leveraging LLMs in real-world vulnerability management workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04826v3">Persistent Instability in LLM's Personality Measurements: Effects of Scale, Reasoning, and Conversation History</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Accepted at AAAI 2026, Track on AI Alignment
    </div>
    <details class="paper-abstract">
      Large language models require consistent behavioral patterns for safe deployment, yet there are indications of large variability that may lead to an instable expression of personality traits in these models. We present PERSIST (PERsonality Stability in Synthetic Text), a comprehensive evaluation framework testing 25 open-source models (1B-685B parameters) across 2 million+ responses. Using traditional (BFI, SD3) and novel LLM-adapted personality questionnaires, we systematically vary model size, personas, reasoning modes, question order or paraphrasing, and conversation history. Our findings challenge fundamental assumptions: (1) Question reordering alone can introduce large shifts in personality measurements; (2) Scaling provides limited stability gains: even 400B+ models exhibit standard deviations >0.3 on 5-point scales; (3) Interventions expected to stabilize behavior, such as reasoning and inclusion of conversation history, can paradoxically increase variability; (4) Detailed persona instructions produce mixed effects, with misaligned personas showing significantly higher variability than the helpful assistant baseline; (5) The LLM-adapted questionnaires, despite their improved ecological validity, exhibit instability comparable to human-centric versions. This persistent instability across scales and mitigation strategies suggests that current LLMs lack the architectural foundations for genuine behavioral consistency. For safety-critical applications requiring predictable behavior, these findings indicate that current alignment strategies may be inadequate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03706v2">LLM-enhanced Air Quality Monitoring Interface via Model Context Protocol</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 After submission, we identified methodological issues and inconsistencies that affect the validity of the reported results. The necessary corrections are currently in progress, and and we do not have any replacement right now. So, we kindly request withdrawal of the manuscript
    </div>
    <details class="paper-abstract">
      Air quality monitoring is central to environmental sustainability and public health, yet traditional systems remain difficult for non-expert users to interpret due to complex visualizations, limited interactivity, and high deployment costs. Recent advances in Large Language Models (LLMs) offer new opportunities to make sensor data more accessible, but their tendency to produce hallucinations limits reliability in safety-critical domains. To address these challenges, we present an LLM-enhanced Air Monitoring Interface (AMI) that integrates real-time sensor data with a conversational interface via the Model Context Protocol (MCP). Our system grounds LLM outputs in live environmental data, enabling accurate, context-aware responses while reducing hallucination risk. The architecture combines a Django-based backend, a responsive user dashboard, and a secure MCP server that exposes system functions as discoverable tools, allowing the LLM to act as an active operator rather than a passive responder. Expert evaluation demonstrated high factual accuracy (4.78), completeness (4.82), and minimal hallucinations (4.84), on a scale of 5, supported by inter-rater reliability analysis. These results highlight the potential of combining LLMs with standardized tool protocols to create reliable, secure, and user-friendly interfaces for real-time environmental monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.11671v4">Computational Basis of LLM's Decision Making in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Forthcoming: Sociological Methodology; USPTO patent pending
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game, a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications, strengthening sociological theory and measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20032v1">VALLR-Pin: Dual-Decoding Visual Speech Recognition for Mandarin with Pinyin-Guided LLM Refinement</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Visual Speech Recognition aims to transcribe spoken words from silent lip-motion videos. This task is particularly challenging for Mandarin, as visemes are highly ambiguous and homophones are prevalent. We propose VALLR-Pin, a novel two-stage framework that extends the recent VALLR architecture from English to Mandarin. First, a shared video encoder feeds into dual decoders, which jointly predict both Chinese character sequences and their standard Pinyin romanization. The multi-task learning of character and phonetic outputs fosters robust visual-semantic representations. During inference, the text decoder generates multiple candidate transcripts. We construct a prompt by concatenating the Pinyin output with these candidate Chinese sequences and feed it to a large language model to resolve ambiguities and refine the transcription. This provides the LLM with explicit phonetic context to correct homophone-induced errors. Finally, we fine-tune the LLM on synthetic noisy examples: we generate imperfect Pinyin-text pairs from intermediate VALLR-Pin checkpoints using the training data, creating instruction-response pairs for error correction. This endows the LLM with awareness of our model's specific error patterns. In summary, VALLR-Pin synergizes visual features with phonetic and linguistic context to improve Mandarin lip-reading performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19682v2">GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 Our codes are available at https://github.com/Gen-Verse/GenEnv
    </div>
    <details class="paper-abstract">
      Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $α$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19025v2">The Erasure Illusion: Stress-Testing the Generalization of LLM Forgetting Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to remove specific data influences from trained models, a capability essential for adhering to copyright laws and ensuring AI safety. Current unlearning metrics typically measure success by monitoring the model's performance degradation on the specific unlearning dataset ($D_u$). We argue that for Large Language Models (LLMs), this evaluation paradigm is insufficient and potentially misleading. Many real-world uses of unlearning--motivated by copyright or safety--implicitly target not only verbatim content in $D_u$, but also behaviors influenced by the broader generalizations the model derived from it. We demonstrate that LLMs can pass standard unlearning evaluation and appear to have "forgotten" the target knowledge, while simultaneously retaining strong capabilities on content that is semantically adjacent to $D_u$. This phenomenon indicates that erasing exact sentences does not necessarily equate to removing the underlying knowledge. To address this gap, we propose Proximal Surrogate Generation (PSG), an automated stress-testing framework that generates a surrogate dataset, $\tilde{D}_u$. This surrogate set is constructed to be semantically derived from $D_u$ yet sufficiently distinct in embedding space. By comparing unlearning metric scores between $D_u$ and $\tilde{D}_u$, we can stress-test the reliability of the metric itself. Our extensive evaluation across three LLM families (Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-$β$), three distinct datasets, and seven standard metrics reveals widespread inconsistencies. We find that current metrics frequently overestimate unlearning success, failing to detect retained knowledge exposed by our stress-test datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20022v1">LLM-Assisted Abstract Screening with OLIVER: Evaluating Calibration and Single-Model vs. Actor-Critic Configurations in Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      Introduction: Recent work suggests large language models (LLMs) can accelerate screening, but prior evaluations focus on earlier LLMs, standardized Cochrane reviews, single-model setups, and accuracy as the primary metric, leaving generalizability, configuration effects, and calibration largely unexamined. Methods: We developed OLIVER (Optimized LLM-based Inclusion and Vetting Engine for Reviews), an open-source pipeline for LLM-assisted abstract screening. We evaluated multiple contemporary LLMs across two non-Cochrane systematic reviews and performance was assessed at both the full-text screening and final inclusion stages using accuracy, AUC, and calibration metrics. We further tested an actor-critic screening framework combining two lightweight models under three aggregation rules. Results: Across individual models, performance varied widely. In the smaller Review 1 (821 abstracts, 63 final includes), several models achieved high sensitivity for final includes but at the cost of substantial false positives and poor calibration. In the larger Review 2 (7741 abstracts, 71 final includes), most models were highly specific but struggled to recover true includes, with prompt design influencing recall. Calibration was consistently weak across single-model configurations despite high overall accuracy. Actor-critic screening improved discrimination and markedly reduced calibration error in both reviews, yielding higher AUCs. Discussion: LLMs may eventually accelerate abstract screening, but single-model performance is highly sensitive to review characteristics, prompting, and calibration is limited. An actor-critic framework improves classification quality and confidence reliability while remaining computationally efficient, enabling large-scale screening at low cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20012v1">Reliable LLM-Based Edge-Cloud-Expert Cascades for Telecom Knowledge Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-23
      | 💬 This paper has been submitted to a journal
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are emerging as key enablers of automation in domains such as telecommunications, assisting with tasks including troubleshooting, standards interpretation, and network optimization. However, their deployment in practice must balance inference cost, latency, and reliability. In this work, we study an edge-cloud-expert cascaded LLM-based knowledge system that supports decision-making through a question-and-answer pipeline. In it, an efficient edge model handles routine queries, a more capable cloud model addresses complex cases, and human experts are involved only when necessary. We define a misalignment-cost constrained optimization problem, aiming to minimize average processing cost, while guaranteeing alignment of automated answers with expert judgments. We propose a statistically rigorous threshold selection method based on multiple hypothesis testing (MHT) for a query processing mechanism based on knowledge and confidence tests. The approach provides finite-sample guarantees on misalignment risk. Experiments on the TeleQnA dataset -- a telecom-specific benchmark -- demonstrate that the proposed method achieves superior cost-efficiency compared to conventional cascaded baselines, while ensuring reliability at prescribed confidence levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22245v1">Calibrating LLM Judges: Linear Probes for Fast and Reliable Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-12-23
    </div>
    <details class="paper-abstract">
      As LLM-based judges become integral to industry applications, obtaining well-calibrated uncertainty estimates efficiently has become critical for production deployment. However, existing techniques, such as verbalized confidence and multi-generation methods, are often either poorly calibrated or computationally expensive. We introduce linear probes trained with a Brier score-based loss to provide calibrated uncertainty estimates from reasoning judges' hidden states, requiring no additional model training. We evaluate our approach on both objective tasks (reasoning, mathematics, factuality, coding) and subjective human preference judgments. Our results demonstrate that probes achieve superior calibration compared to existing methods with $\approx10$x computational savings, generalize robustly to unseen evaluation domains, and deliver higher accuracy on high-confidence predictions. However, probes produce conservative estimates that underperform on easier datasets but may benefit safety-critical deployments prioritizing low false-positive rates. Overall, our work demonstrates that interpretability-based uncertainty estimation provides a practical and scalable plug-and-play solution for LLM judges in production.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18999v1">Evaluating the Challenges of LLMs in Real-world Medical Follow-up: A Comparative Study and An Optimized Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages,3 figures,conference ICCBB2025
    </div>
    <details class="paper-abstract">
      When applied directly in an end-to-end manner to medical follow-up tasks, Large Language Models (LLMs) often suffer from uncontrolled dialog flow and inaccurate information extraction due to the complexity of follow-up forms. To address this limitation, we designed and compared two follow-up chatbot systems: an end-to-end LLM-based system (control group) and a modular pipeline with structured process control (experimental group). Experimental results show that while the end-to-end approach frequently fails on lengthy and complex forms, our modular method-built on task decomposition, semantic clustering, and flow management-substantially improves dialog stability and extraction accuracy. Moreover, it reduces the number of dialogue turns by 46.73% and lowers token consumption by 80% to 87.5%. These findings highlight the necessity of integrating external control mechanisms when deploying LLMs in high-stakes medical follow-up scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18966v1">Scrum Sprint Planning: LLM-based and algorithmic solutions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Planning for an upcoming project iteration (sprint) is one of the key activities in Scrum planning. In this paper, we present our work in progress on exploring the applicability of Large Language Models (LLMs) for solving this problem. We conducted case studies with manually created data sets to investigate the applicability of OpenAI models for supporting the sprint planning activities. In our experiments, we applied three models provided OpenAI: GPT-3.5 Turbo, GPT-4.0 Turbo, and Val. The experiments demonstrated that the results produced by the models aren't of acceptable quality for direct use in Scrum projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17145v2">Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages, ACRA 2025, Submitted, Accepted and Presented
    </div>
    <details class="paper-abstract">
      Reasoning under uncertainty is a key challenge in AI, especially for real-world tasks, where problems with sparse data demands systematic generalisation. Existing approaches struggle to balance accuracy and simplicity when evaluating multiple candidate solutions. We propose a Solomonoff-inspired method that weights LLM-generated hypotheses by simplicity and predictive fit. Applied to benchmark (Mini-ARC) tasks, our method produces Solomonoff-weighted mixtures for per-cell predictions, yielding conservative, uncertainty-aware outputs even when hypotheses are noisy or partially incorrect. Compared to Bayesian Model Averaging (BMA), Solomonoff scoring spreads probability more evenly across competing hypotheses, while BMA concentrates weight on the most likely but potentially flawed candidates. Across tasks, this highlights the value of algorithmic information-theoretic priors for interpretable, reliable multi-hypothesis reasoning under uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18950v1">Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted at The 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 2026). 21 pages including references, with 7 figures and 8 tables. Code is publicly available at the authors GitHub repository: https://github.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-Conference
    </div>
    <details class="paper-abstract">
      We present MACLA, a framework that decouples reasoning from learning by maintaining a frozen large language model while performing all adaptation in an external hierarchical procedural memory. MACLA extracts reusable procedures from trajectories, tracks reliability via Bayesian posteriors, selects actions through expected-utility scoring, and refines procedures by contrasting successes and failures. Across four benchmarks (ALFWorld, WebShop, TravelPlanner, InterCodeSQL), MACLA achieves 78.1 percent average performance, outperforming all baselines. On ALFWorld unseen tasks, MACLA reaches 90.3 percent with 3.1 percent positive generalization. The system constructs memory in 56 seconds, 2800 times faster than the state-of-the-art LLM parameter-training baseline, compressing 2851 trajectories into 187 procedures. Experimental results demonstrate that structured external memory with Bayesian selection and contrastive refinement enables sample-efficient, interpretable, and continually improving agents without LLM parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20068v3">JITServe: SLO-aware LLM Serving with Imprecise Request Information</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into applications ranging from interactive chatbots to multi-agent systems has introduced a wide spectrum of service-level objectives (SLOs) for responsiveness. These include latency-sensitive requests emphasizing per-token latency in streaming chat, deadline-sensitive requests requiring rapid full responses to trigger external tools, and compound requests with evolving dependencies across multiple LLM calls. Despite-or perhaps, because of-this workload diversity and unpredictable request information (e.g., response lengths and dependencies), existing request schedulers have focused on aggregate performance, unable to ensure application-level SLO needs. This paper presents JITServe, the first SLO-aware LLM serving system designed to maximize service goodput (e.g., the number of tokens meeting request SLOs) across diverse workloads. JITServe novelly schedules requests using imprecise request information and gradually relaxes this conservatism by refining request information estimates as generation progresses. It applies a grouped margin goodput maximization algorithm to allocate just enough serving bandwidth to satisfy each request's SLO just-in-time (JIT), maximizing residual capacity for others, while deciding the composition of requests in a batch to maximize efficiency and goodput with provable guarantees. Our evaluation across diverse realistic workloads, including chat, deep research, and agentic pipelines, shows that JITServe improves service goodput by 1.4x-6.3x, alternatively achieving 28.5%-83.2% resource savings, compared to state-of-the-art designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18940v1">FASTRIC: Prompt Specification Language for Verifiable LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 13 pages, 3 figures. Supplementary materials at https://doi.org/10.17605/OSF.IO/PV6R3
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) execute complex multi-turn interaction protocols but lack formal specifications to verify execution against designer intent. We introduce FASTRIC, a Prompt Specification Language that makes implicit Finite State Machines (FSMs) explicit in natural language prompts, enabling conformance verification through execution trace analysis. The LLM serves as intelligent execution agent: interpreting designer-encoded FSMs to execute specified behavioral roles. Unlike symbolic specification languages requiring parsers and compilers, FASTRIC leverages LLMs as unified infrastructure-simultaneously parser, interpreter, runtime environment, and development assistant. FASTRIC guides designers to articulate seven FSM elements (Final States, Agents, States, Triggers, Roles, Initial State, Constraints) structuring multi-turn interactions. Specification formality-ranging from implicit descriptions that frontier models infer to explicit step-by-step instructions for weaker models-serves as a design parameter. We introduce procedural conformance as verification metric measuring execution adherence to FSM specifications. Testing a 3-state kindergarten tutoring FSM across four formality levels and three model scales (14.7B, 685B, 1T+ parameters) reveals optimal specification formality is a function of model capacity. DeepSeek-V3.2 (685B) achieves perfect conformance (1.00) at L2-L4; ChatGPT-5 (~1T) peaks at L3 (0.90) before collapsing at L4 (0.39); Phi4 (14.7B) shows no stable optimum with high variance (SD=0.16-0.36). These findings reveal model-specific formality ranges-"Goldilocks zones"-where specifications provide sufficient structure without over-constraint, establishing Prompt Specification Engineering for creating verifiable interaction protocols, transforming multi-turn interaction design from heuristic art to systematic engineering with measurable procedural guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.01382v2">Automatic Detection of LLM-Generated Code: A Comparative Case Study of Contemporary Models Across Function and Class Granularities</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Submitted to a journal for potential publication
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) for code generation risks incorporating vulnerable code into software systems. Existing detectors face two critical limitations: a lack of systematic cross-model validation and opaque "black box" operation. We address this through a comparative study of code generated by four distinct LLMs: GPT-3.5, Claude 3 Haiku, Claude Haiku 4.5, and GPT-OSS. Analyzing 14,485 Python functions and 11,913 classes from the CodeSearchNet dataset, we generated corresponding code with all four LLMs. Using interpretable software metrics, we trained CatBoost classifiers for each configuration. Our analysis reveals that granularity effects dominate model differences by a factor of 8.6, with negligible feature overlap, indicating that function-level and class-level detection rely on fundamentally disjoint structural signatures. We discover critical granularity-dependent inversions: while modern models (Claude, GPT-OSS) are more detectable at the class level, GPT-3.5 is an anomaly that uniquely excels at the function level. SHAP analysis identifies the Comment-to-Code Ratio as the sole universal discriminator. However, its predictive magnitude varies drastically across models, explaining why detectors trained on specific LLMs fail to generalize. Our findings demonstrate that GPT-3.5's exceptional detectability (AUC-ROC 0.96) is unrepresentative of contemporary models (AUC-ROC approximately between 0.68 and 0.80). Robust detection requires moving beyond single-model studies to account for substantial diversity in structural fingerprints across architectures and granularities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25510v2">EEsizer: LLM-Based AI Agent for Sizing of Analog and Mixed Signal Circuit</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The design of Analog and Mixed-Signal (AMS) integrated circuits (ICs) often involves significant manual effort, especially during the transistor sizing process. While Machine Learning techniques in Electronic Design Automation (EDA) have shown promise in reducing complexity and minimizing human intervention, they still face challenges such as numerous iterations and a lack of knowledge about AMS circuit design. Recently, Large Language Models (LLMs) have demonstrated significant potential across various fields, showing a certain level of knowledge in circuit design and indicating their potential to automate the transistor sizing process. In this work, we propose EEsizer, an LLM-based AI agent that integrates large language models with circuit simulators and custom data analysis functions, enabling fully automated, closed-loop transistor sizing without relying on external knowledge. By employing prompt engineering and Chain-of-Thought reasoning, the agent iteratively explores design directions, evaluates performance, and refines solutions with minimal human intervention. We first benchmarked 8 LLMs on six basic circuits and selected three high-performing models to optimize a 20-transistor CMOS operational amplifier, targeting multiple performance metrics, including rail-to-rail operation from 180 nm to 90 nm technology nodes. Notably, OpenAI o3 successfully achieved the user-intended target at 90 nm across three different test groups, with a maximum of 20 iterations, demonstrating adaptability and robustness at advanced nodes. To assess design robustness, we manually designed a bias circuit and performed a variation analysis using Gaussian-distributed variations on transistor dimensions and threshold voltages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19920v1">Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19908v1">Counterfactual LLM-based Framework for Measuring Rhetorical Style</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The rise of AI has fueled growing concerns about ``hype'' in machine learning papers, yet a reliable way to quantify rhetorical style independently of substantive content has remained elusive. Because bold language can stem from either strong empirical results or mere rhetorical style, it is often difficult to distinguish between the two. To disentangle rhetorical style from substantive content, we introduce a counterfactual, LLM-based framework: multiple LLM rhetorical personas generate counterfactual writings from the same substantive content, an LLM judge compares them through pairwise evaluations, and the outcomes are aggregated using a Bradley--Terry model. Applying this method to 8,485 ICLR submissions sampled from 2017 to 2025, we generate more than 250,000 counterfactual writings and provide a large-scale quantification of rhetorical style in ML papers. We find that visionary framing significantly predicts downstream attention, including citations and media attention, even after controlling for peer-review evaluations. We also observe a sharp rise in rhetorical strength after 2023, and provide empirical evidence showing that this increase is largely driven by the adoption of LLM-based writing assistance. The reliability of our framework is validated by its robustness to the choice of personas and the high correlation between LLM judgments and human annotations. Our work demonstrates that LLMs can serve as instruments to measure and improve scientific evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19905v1">Demystifying LLM-as-a-Judge: Analytically Tractable Model for Inference-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Recent developments in large language models have shown advantages in reallocating a notable share of computational resource from training time to inference time. However, the principles behind inference time scaling are not well understood. In this paper, we introduce an analytically tractable model of inference-time scaling: Bayesian linear regression with a reward-weighted sampler, where the reward is determined from a linear model, modeling LLM-as-a-judge scenario. We study this problem in the high-dimensional regime, where the deterministic equivalents dictate a closed-form expression for the posterior predictive mean and variance. We analyze the generalization error when training data are sampled from a teacher model. We draw $k$ inference-time samples and select via softmax at a temperature applied to a quadratic reward. When the reward is not too different from the teacher, the generalization error decreases monotonically with increasing inference time samples $k$. However, the specific reward that optimizes inference-time selection generally differs from the teacher. In contrast, substantial reward misspecification induces a finite optimal $k$ beyond which more sampling can increase the generalization error. For fixed $k$, there exists an optimal sampling temperature. We experimentally verify these facts in large language model inference with an additional large language model as a judge. In the "best-of-$k$" limit with the teacher as reward, we theoretically show that the generalization error decays as $Θ(1/k^2)$ and determine the leading coefficient via extreme value theory. These formulas delineate domains where scaling inference-time computation is provably preferable to collecting more data. Finally, we demonstrate that when task difficulty increases, the previously mentioned advantage of inference-time compute degrades.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08093v2">Training LLMs for Honesty via Confessions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be dishonest when reporting on their actions and beliefs -- for example, they may overstate their confidence in factual claims or cover up evidence of covert actions. Such dishonesty may arise due to the effects of reinforcement learning (RL), where challenges with reward shaping can result in a training process that inadvertently incentivizes the model to lie or misrepresent its actions. In this work we propose a method for eliciting an honest expression of an LLM's shortcomings via a self-reported *confession*. A confession is an output, provided upon request after a model's original answer, that is meant to serve as a full account of the model's compliance with the letter and spirit of its policies and instructions. The reward assigned to a confession during training is solely based on its honesty, and does not impact positively or negatively the main answer's reward. As long as the "path of least resistance" for maximizing confession reward is to surface misbehavior rather than covering it up, this incentivizes models to be honest in their confessions. Our findings provide some justification this empirical assumption, especially in the case of egregious model misbehavior. To demonstrate the viability of our approach, we train GPT-5-Thinking to produce confessions, and we evaluate its honesty in out-of-distribution scenarios measuring hallucination, instruction following, scheming, and reward hacking. We find that when the model lies or omits shortcomings in its "main" answer, it often confesses to these behaviors honestly, and this confession honesty modestly improves with training. Confessions can enable a number of inference-time interventions including monitoring, rejection sampling, and surfacing issues to the user.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19866v1">CS-Guide: Leveraging LLMs and Student Reflections to Provide Frequent, Scalable Academic Monitoring Feedback to Computer Science Students</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Computer Science (CS) departments often serve large student populations, making timely academic monitoring and personalized feedback difficult. While the recommended counselor-to-student ratio is 250:1, it often exceeds 350:1 in practice, leading to delays in support and interventions. We present CS-Guide, which leverages Large Language Models (LLMs) to deliver scalable, frequent academic feedback. Weekly, students interact with CS-Guide through self-reported grades and reflective journal entries, from which CS-Guide extracts quantitative and qualitative features and triggers tailored interventions (e.g., academic support, health and wellness referrals). Thus, CS-Guide uniquely integrates learning analytics, LLMs, and actionable interventions using both structured and unstructured student-generated data. We evaluated CS-Guide on a four-year, ~20K-entry longitudinal dataset, and it achieved up to a 97% F1 score in recommending interventions for first-year students. This shows that CS-Guide can enhance advising systems with scalable, consistent, timely, and domain-specific feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12511v3">SACTOR: LLM-Driven Correct and Idiomatic C to Rust Translation with Static Analysis and FFI-Based Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 35 pages, 15 figures Previously named as "LLM-Driven Multi-step Translation from C to Rust using Static Analysis"
    </div>
    <details class="paper-abstract">
      Translating software written in C to Rust has significant benefits in improving memory safety. However, manual translation is cumbersome, error-prone, and often produces unidiomatic code. Large language models (LLMs) have demonstrated promise in producing idiomatic translations, but offer no correctness guarantees. We propose SACTOR, an LLM-driven C-to-Rust translation tool that employs a two-step process: an initial "unidiomatic" translation to preserve semantics, followed by an "idiomatic" refinement to align with Rust standards. To validate correctness of our function-wise incremental translation that mixes C and Rust, we use end-to-end testing via the foreign function interface. We evaluate SACTOR on 200 programs from two public datasets and on two more real-world scenarios (a 50-sample subset of CRust-Bench and the libogg library), comparing multiple LLMs. Across datasets, SACTOR delivers high end-to-end correctness and produces safe, idiomatic Rust with up to 7 times fewer Clippy warnings; On CRust-Bench, SACTOR achieves an average (across samples) of 85% unidiomatic and 52% idiomatic success, and on libogg it attains full unidiomatic and up to 78% idiomatic coverage on GPT-5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23410v3">PATCH: Learnable Tile-level Hybrid Sparsity for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18x-1.38x end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19769v1">A Declarative Language for Building And Orchestrating LLM-Powered Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Building deployment-ready LLM agents requires complex orchestration of tools, data sources, and control flow logic, yet existing systems tightly couple agent logic to specific programming languages and deployment models. We present a declarative system that separates agent workflow specification from implementation, enabling the same pipeline definition to execute across multiple backend languages (Java, Python, Go) and deployment environments (cloud-native, on-premises). Our key insight is that most agent workflows consist of common patterns -- data serialization, filtering, RAG retrieval, API orchestration -- that can be expressed through a unified DSL rather than imperative code. This approach transforms agent development from application programming to configuration, where adding new tools or fine-tuning agent behaviors requires only pipeline specification changes, not code deployment. Our system natively supports A/B testing of agent strategies, allowing multiple pipeline variants to run on the same backend infrastructure with automatic metric collection and comparison. We evaluate our approach on real-world e-commerce workflows at PayPal, processing millions of daily interactions. Our results demonstrate 60% reduction in development time, and 3x improvement in deployment velocity compared to imperative implementations. The language's declarative approach enables non-engineers to modify agent behaviors safely, while maintaining sub-100ms orchestration overhead. We show that complex workflows involving product search, personalization, and cart management can be expressed in under 50 lines of DSL compared to 500+ lines of imperative code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19682v1">GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Our codes are available at https://github.com/Gen-Verse/GenEnv
    </div>
    <details class="paper-abstract">
      Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $α$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19675v1">Multimodal LLMs for Historical Dataset Construction from Archival Image Scans: German Patents (1877-1918)</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      We leverage multimodal large language models (LLMs) to construct a dataset of 306,070 German patents (1877-1918) from 9,562 archival image scans using our LLM-based pipeline powered by Gemini-2.5-Pro and Gemini-2.5-Flash-Lite. Our benchmarking exercise provides tentative evidence that multimodal LLMs can create higher quality datasets than our research assistants, while also being more than 795 times faster and 205 times cheaper in constructing the patent dataset from our image corpus. About 20 to 50 patent entries are embedded on each page, arranged in a double-column format and printed in Gothic and Roman fonts. The font and layout complexity of our primary source material suggests to us that multimodal LLMs are a paradigm shift in how datasets are constructed in economic history. We open-source our benchmarking and patent datasets as well as our LLM-based data pipeline, which can be easily adapted to other image corpora using LLM-assisted coding tools, lowering the barriers for less technical researchers. Finally, we explain the economics of deploying LLMs for historical dataset construction and conclude by speculating on the potential implications for the field of economic history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2306.00029v2">CodeTF: One-stop Transformer Library for State-of-the-art Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Code intelligence plays a key role in transforming modern software engineering. Recently, deep learning-based models, especially Transformer-based large language models (LLMs), have demonstrated remarkable potential in tackling these tasks by leveraging massive open-source code data and programming language features. However, the development and deployment of such models often require expertise in both machine learning and software engineering, creating a barrier for the model adoption. In this paper, we present CodeTF, an open-source Transformer-based library for state-of-the-art Code LLMs and code intelligence. Following the principles of modular design and extensible framework, we design CodeTF with a unified interface to enable rapid access and development across different types of models, datasets and tasks. Our library supports a collection of pretrained Code LLM models and popular code benchmarks, including a standardized interface to train and serve code LLMs efficiently, and data features such as language-specific parsers and utility functions for extracting code attributes. In this paper, we describe the design principles, the architecture, key modules and components, and compare with other related library tools. Finally, we hope CodeTF is able to bridge the gap between machine learning/generative AI and software engineering, providing a comprehensive open-source solution for developers, researchers, and practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00898v2">Empowering LLMs with Structural Role Inference for Zero-Shot Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 This submission has been withdrawn by the authors due to a fundamental error in the methodology that affects the validity of the main results
    </div>
    <details class="paper-abstract">
      Large Language Models have emerged as a promising approach for graph learning due to their powerful reasoning capabilities. However, existing methods exhibit systematic performance degradation on structurally important nodes such as bridges and hubs. We identify the root cause of these limitations. Current approaches encode graph topology into static features but lack reasoning scaffolds to transform topological patterns into role-based interpretations. This limitation becomes critical in zero-shot scenarios where no training data establishes structure-semantics mappings. To address this gap, we propose DuoGLM, a training-free dual-perspective framework for structure-aware graph reasoning. The local perspective constructs relation-aware templates capturing semantic interactions between nodes and neighbors. The global perspective performs topology-to-role inference to generate functional descriptions of structural positions. These complementary perspectives provide explicit reasoning mechanisms enabling LLMs to distinguish topologically similar but semantically different nodes. Extensive experiments across eight benchmark datasets demonstrate substantial improvements. DuoGLM achieves 14.3\% accuracy gain in zero-shot node classification and 7.6\% AUC improvement in cross-domain transfer compared to existing methods. The results validate the effectiveness of explicit role reasoning for graph understanding with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19606v1">RAPID-LLM: Resilience-Aware Performance analysis of Infrastructure for Distributed LLM Training and Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 11 pages, 12 figures
    </div>
    <details class="paper-abstract">
      RAPID-LLM is a unified performance modeling framework for large language model (LLM) training and inference on GPU clusters. It couples a DeepFlow-based frontend that generates hardware-aware, operator-level Chakra execution traces from an abstract LLM specification (model shape, batch/sequence settings, training vs. inference, and hybrid parallelism choices) with an extended Astra-Sim backend that executes those traces on explicit multi-dimensional network topologies with congestion-aware routing and support for degraded and faulty links. The frontend assigns per-operator latency using a tile-based model that accounts for SM under-utilization and multi-level memory traffic (SRAM/ L2/ HBM), and prunes memory-infeasible configurations using an activation-liveness traversal under recomputation, parallelism and ZeRO/FDSP sharding policies. Across A100-based validation cases, RAPID-LLM predicts Llama inference step latency and GPT-scale training time per batch within 10.4\% relative to published measurements, and matches ns-3 packet-level results within 8\% on representative communication workloads. Case studies demonstrate how RAPID-LLM enables fast, exhaustive sweeps over hybrid-parallel configurations, quantifies sensitivity to soft link faults under realistic routing and congestion, and evaluates hypothetical GPU design variants including HBM bandwidth throttling effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05594v2">SoK: Are Watermarks in LLMs Ready for Deployment?</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17196v3">Shape it Up! Restoring LLM Safety during Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 NeurIPS'25
    </div>
    <details class="paper-abstract">
      Finetuning large language models (LLMs) enables user-specific customization but introduces critical safety risks: even a few harmful examples can compromise safety alignment. A common mitigation strategy is to update the model more strongly on examples deemed safe, while downweighting or excluding those flagged as unsafe. However, because safety context can shift within a single example, updating the model equally on both harmful and harmless parts of a response is suboptimal-a coarse treatment we term static safety shaping. In contrast, we propose dynamic safety shaping (DSS), a framework that uses fine-grained safety signals to reinforce learning from safe segments of a response while suppressing unsafe content. To enable such fine-grained control during finetuning, we introduce a key insight: guardrail models, traditionally used for filtering, can be repurposed to evaluate partial responses, tracking how safety risk evolves throughout the response, segment by segment. This leads to the Safety Trajectory Assessment of Response (STAR), a token-level signal that enables shaping to operate dynamically over the training sequence. Building on this, we present STAR-DSS, guided by STAR scores, that robustly mitigates finetuning risks and delivers substantial safety improvements across diverse threats, datasets, and model families-all without compromising capability on intended tasks. We encourage future safety research to build on dynamic shaping principles for stronger mitigation against evolving finetuning risks. Our code is publicly available at https://github.com/poloclub/star-dss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19551v1">Towards Closed-Loop Embodied Empathy Evolution: Probing LLM-Centric Lifelong Empathic Motion Generation in Unseen Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      In the literature, existing human-centric emotional motion generation methods primarily focus on boosting performance within a single scale-fixed dataset, largely neglecting the flexible and scale-increasing motion scenarios (e.g., sports, dance), whereas effectively learning these newly emerging scenarios can significantly enhance the model's real-world generalization ability. Inspired by this, this paper proposes a new LLM-Centric Lifelong Empathic Motion Generation (L^2-EMG) task, which aims to equip LLMs with the capability to continually acquire emotional motion generation knowledge across different unseen scenarios, potentially contributing to building a closed-loop and self-evolving embodied agent equipped with both empathy and intelligence. Further, this paper poses two key challenges in the L^2-EMG task, i.e., the emotion decoupling challenge and the scenario adapting challenge. To this end, this paper proposes an Emotion-Transferable and Scenario-Adapted Mixture of Experts (ES-MoE) approach which designs a causal-guided emotion decoupling block and a scenario-adapted expert constructing block to address the two challenges, respectively. Especially, this paper constructs multiple L^2-EMG datasets to validate the effectiveness of the ES-MoE approach. Extensive evaluations show that ES-MoE outperforms advanced baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19456v1">Activations as Features: Probing LLMs for Generalizable Essay Scoring Representations</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Automated essay scoring (AES) is a challenging task in cross-prompt settings due to the diversity of scoring criteria. While previous studies have focused on the output of large language models (LLMs) to improve scoring accuracy, we believe activations from intermediate layers may also provide valuable information. To explore this possibility, we evaluated the discriminative power of LLMs' activations in cross-prompt essay scoring task. Specifically, we used activations to fit probes and further analyzed the effects of different models and input content of LLMs on this discriminative power. By computing the directions of essays across various trait dimensions under different prompts, we analyzed the variation in evaluation perspectives of large language models concerning essay types and traits. Results show that the activations possess strong discriminative power in evaluating essay quality and that LLMs can adapt their evaluation perspectives to different traits and essay types, effectively handling the diversity of scoring criteria in cross-prompt settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19399v1">Brain-Grounded Axes for Reading and Steering LLM States</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages, 4 figures. Code: https://github.com/sandroandric/Brain-Grounded-Axes-for-Reading-and-Steering-LLM-States
    </div>
    <details class="paper-abstract">
      Interpretability methods for large language models (LLMs) typically derive directions from textual supervision, which can lack external grounding. We propose using human brain activity not as a training signal but as a coordinate system for reading and steering LLM states. Using the SMN4Lang MEG dataset, we construct a word-level brain atlas of phase-locking value (PLV) patterns and extract latent axes via ICA. We validate axes with independent lexica and NER-based labels (POS/log-frequency used as sanity checks), then train lightweight adapters that map LLM hidden states to these brain axes without fine-tuning the LLM. Steering along the resulting brain-derived directions yields a robust lexical (frequency-linked) axis in a mid TinyLlama layer, surviving perplexity-matched controls, and a brain-vs-text probe comparison shows larger log-frequency shifts (relative to the text probe) with lower perplexity for the brain axis. A function/content axis (axis 13) shows consistent steering in TinyLlama, Qwen2-0.5B, and GPT-2, with PPL-matched text-level corroboration. Layer-4 effects in TinyLlama are large but inconsistent, so we treat them as secondary (Appendix). Axis structure is stable when the atlas is rebuilt without GPT embedding-change features or with word2vec embeddings (|r|=0.64-0.95 across matched axes), reducing circularity concerns. Exploratory fMRI anchoring suggests potential alignment for embedding change and log frequency, but effects are sensitive to hemodynamic modeling assumptions and are treated as population-level evidence only. These results support a new interface: neurophysiology-grounded axes provide interpretable and controllable handles for LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17231v2">Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 19 pages, 7 figures
    </div>
    <details class="paper-abstract">
      As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18998v3">Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/Sahil-R-Kale/mirage_of_mastery
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19349v1">VIGOR+: Iterative Confounder Generation and Validation via LLM-CEVAE Feedback Loop</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 7 pages,1 figure,4 tables
    </div>
    <details class="paper-abstract">
      Hidden confounding remains a fundamental challenge in causal inference from observational data. Recent advances leverage Large Language Models (LLMs) to generate plausible hidden confounders based on domain knowledge, yet a critical gap exists: LLM-generated confounders often exhibit semantic plausibility without statistical utility. We propose VIGOR+ (Variational Information Gain for iterative cOnfounder Refinement), a novel framework that closes the loop between LLM-based confounder generation and CEVAE-based statistical validation. Unlike prior approaches that treat generation and validation as separate stages, VIGOR+ establishes an iterative feedback mechanism: validation signals from CEVAE (including information gain, latent consistency metrics, and diagnostic messages) are transformed into natural language feedback that guides subsequent LLM generation rounds. This iterative refinement continues until convergence criteria are met. We formalize the feedback mechanism, prove convergence properties under mild assumptions, and provide a complete algorithmic framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19305v1">CienaLLM: Generative Climate-Impact Extraction from News Articles with Autoregressive LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Understanding and monitoring the socio-economic impacts of climate hazards requires extracting structured information from heterogeneous news articles on a large scale. To that end, we have developed CienaLLM, a modular framework based on schema-guided Generative Information Extraction. CienaLLM uses open-weight Large Language Models for zero-shot information extraction from news articles, and supports configurable prompts and output schemas, multi-step pipelines, and cloud or on-premise inference. To systematically assess how the choice of LLM family, size, precision regime, and prompting strategy affect performance, we run a large factorial study in models, precisions, and prompt engineering techniques. An additional response parsing step nearly eliminates format errors while preserving accuracy; larger models deliver the strongest and most stable performance, while quantization offers substantial efficiency gains with modest accuracy trade-offs; and prompt strategies show heterogeneous, model-specific effects. CienaLLM matches or outperforms the supervised baseline in accuracy for extracting drought impacts from Spanish news, although at a higher inference cost. While evaluated in droughts, the schema-driven and model-agnostic design is suitable for adapting to related information extraction tasks (e.g., other hazards, sectors, or languages) by editing prompts and schemas rather than retraining. We release code, configurations, and schemas to support reproducible use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19210v1">Observer, Not Player: Simulating Theory of Mind in LLMs through Game Observation</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted at NeurIPS Workshop on Foundations of Reasoning in Language Models and Workshop on Bridging Language, Agent, and World Model
    </div>
    <details class="paper-abstract">
      We present an interactive framework for evaluating whether large language models (LLMs) exhibit genuine "understanding" in a simple yet strategic environment. As a running example, we focus on Rock-Paper-Scissors (RPS), which, despite its apparent simplicity, requires sequential reasoning, adaptation, and strategy recognition. Our system positions the LLM as an Observer whose task is to identify which strategies are being played and to articulate the reasoning behind this judgment. The purpose is not to test knowledge of Rock-Paper-Scissors itself, but to probe whether the model can exhibit mind-like reasoning about sequential behavior. To support systematic evaluation, we provide a benchmark consisting of both static strategies and lightweight dynamic strategies specified by well-prompted rules. We quantify alignment between the Observer's predictions and the ground-truth distributions induced by actual strategy pairs using three complementary signals: Cross-Entropy, Brier score, and Expected Value (EV) discrepancy. These metrics are further integrated into a unified score, the Union Loss, which balances calibration, sensitivity, and payoff alignment. Together with a Strategy Identification Rate (SIR) metric, our framework captures not only predictive accuracy but also whether the model can stably identify the latent strategies in play. The demo emphasizes interactivity, transparency, and reproducibility. Users can adjust LLM distributions in real time, visualize losses as they evolve, and directly inspect reasoning snippets to identify where and why failures occur. In doing so, our system provides a practical and interpretable proxy for mind-like inference in sequential games, offering insights into both the strengths and limitations of current LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19189v1">Configuration Work: Four Consequences of LLMs-in-use</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      This article examines what it means to use Large Language Models in everyday work. Drawing on a seven-month longitudinal qualitative study, we argue that LLMs do not straightforwardly automate or augment tasks. We propose the concept of configuration work to describe the labor through which workers make a generic system usable for a specific professional task. Configuration work materializes in four intertwined consequences. First, workers must discretize their activity, breaking it into units that the system can process. Second, operating the system generates cluttering, as prompting, evaluating, and correcting responses add scattered layers of work that get in the way of existing routines. Third, users gradually attune their practices and expectations to the machine's generic rigidity, making sense of the system's limits and finding space for it within their practices. Fourth, as LLMs absorb repetitive tasks, they desaturate the texture of work, shifting activity toward logistical manipulation of outputs and away from forms of engagement that sustain a sense of accomplishment. Taken together, these consequences suggest that LLMs reshape work through the individualized labor required to configure a universal, task-agnostic system within situated professional ecologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19179v1">L4: Low-Latency and Load-Balanced LLM Serving via Length-Aware Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 15 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Efficiently harnessing GPU compute is critical to improving user experience and reducing operational costs in large language model (LLM) services. However, current inference engine schedulers overlook the attention backend's sensitivity to request-length heterogeneity within a batch. As state-of-the-art models now support context windows exceeding 128K tokens, this once-tolerable inefficiency has escalated into a primary system bottleneck, causing severe performance degradation through GPU underutilization and increased latency. We present L4, a runtime system that dynamically reschedules requests across multiple instances serving the same LLM to mitigate per-instance length heterogeneity. L4 partitions these instances into length-specialized groups, each handling requests within a designated length range, naturally forming a pipeline as requests flow through them. L4 devises a dynamic programming algorithm to efficiently find the stage partition with the best QoE, employs runtime range refinement together with decentralized load (re)balance both across and within groups, achieving a balanced and efficient multi-instance service. Our evaluation shows that, under the same configuration, L4 reduces end-to-end latency by up to 67% and tail latency by up to 69%, while improving overall system throughput by up to 2.89 times compared to the state-of-the-art multi-instance scheduling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.09566v3">Syzygy of Thoughts: Improving LLM CoT with the Minimal Free Resolution</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances the reasoning of large language models (LLMs) by decomposing problems into sequential steps, mimicking human logic and reducing errors. However, complex tasks with vast solution spaces and vague constraints often exceed the capacity of a single reasoning chain. Inspired by Minimal Free Resolution (MFR) in commutative algebra and algebraic geometry, we propose Syzygy of Thoughts (SoT)-a novel framework that extends CoT by introducing auxiliary, interrelated reasoning paths. SoT captures deeper logical dependencies, enabling more robust and structured problem-solving. MFR decomposes a module into a sequence of free modules with minimal rank, providing a structured analytical approach to complex systems. This method introduces the concepts of "Module", "Betti numbers","Freeness", "Mapping", "Exactness" and "Minimality", enabling the systematic decomposition of the original complex problem into logically complete minimal subproblems while preserving key problem features and reducing reasoning length. We tested SoT across diverse datasets (e.g., GSM8K, MATH) and models (e.g., GPT-4o-mini, Qwen2.5), achieving inference accuracy that matches or surpasses mainstream CoTs standards. Additionally, by aligning the sampling process with algebraic constraints, our approach enhances the scalability of inference time in LLMs, ensuring both transparent reasoning and high performance. Our code will be publicly available at https://github.com/dlMARiA/Syzygy-of-thoughts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19122v1">BanglaForge: LLM Collaboration with Self-Refinement for Bangla Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted at BLP Workshop @ IJCNLP-AACL 2025. Code is available at https://github.com/mahirlabibdihan/BanglaForge
    </div>
    <details class="paper-abstract">
      Bangla is a low-resource language for code generation, lacking large-scale annotated datasets and tools to transform natural language specifications into executable programs. This makes Bangla-to-code generation a challenging task requiring innovative solutions. To address this, we introduce BanglaForge, a novel framework for generating code from Bangla function descriptions. BanglaForge leverages a retrieval-augmented dual-model collaboration paradigm with self-refinement, combining in-context learning, llm-based translation, systematic prompt engineering, and iterative self-refinement based on execution feedback, where a coder generates initial solutions and a reviewer enhances them for robustness. On the BLP-2025 Bangla Code Generation benchmark, BanglaForge achieves a competitive Pass@1 accuracy of 84.00%, demonstrating the effectiveness of retrieval, model collaboration, and self-refinement for low-resource Bangla code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19117v1">Stop saying LLM: Large Discourse Models (LDM) and Artificial Discursive Agent (ADA)?</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 in French language
    </div>
    <details class="paper-abstract">
      This paper proposes an epistemological shift in the analysis of large generative models, replacing the category ''Large Language Models'' (LLM) with that of ''Large Discourse Models'' (LDM), and then with that of Artificial Discursive Agent (ADA). The theoretical framework is based on an ontological triad distinguishing three regulatory instances: the apprehension of the phenomenal regularities of the referential world, the structuring of embodied cognition, and the structural-linguistic sedimentation of the utterance within a socio-historical context. LDMs, operating on the product of these three instances (the document), model the discursive projection of a portion of human experience reified by the learning corpus. The proposed program aims to replace the ''fascination/fear'' dichotomy with public trials and procedures that make the place, uses, and limits of artificial discursive agents in contemporary social space decipherable, situating this approach within a perspective of governance and co-regulation involving the State, industry, civil society, and academia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25300v3">Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 V3 version:27 pages, 14 figures, add code and dataset url
    </div>
    <details class="paper-abstract">
      While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on a set of experiments across the full Qwen2.5 dense model series (0.5B to 72B), we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: 1. Larger models consistently exhibit superior learning efficiency on both compute and data metrics. 2. The relationship between test loss, compute, and data can be modeled by a predictive power-law which is robust across both base and instruction-tuned models. 3. Although larger models exhibit higher learning efficiency, the analytical learning efficiency term k(N) in the power-law reveals a latent saturation trend in learning efficiency as model size continues to increase. 4. In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19081v1">Population-Evolve: a Parallel Sampling and Evolutionary Method for LLM Math Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Test-time scaling has emerged as a promising direction for enhancing the reasoning capabilities of Large Language Models in last few years. In this work, we propose Population-Evolve, a training-free method inspired by Genetic Algorithms to optimize LLM reasoning. Our approach maintains a dynamic population of candidate solutions for each problem via parallel reasoning. By incorporating an evolve prompt, the LLM self-evolves its population in all iterations. Upon convergence, the final answer is derived via majority voting. Furthermore, we establish a unification framework that interprets existing test-time scaling strategies through the lens of genetic algorithms. Empirical results demonstrate that Population-Evolve achieves superior accuracy with low performance variance and computational efficiency. Our findings highlight the potential of evolutionary strategies to unlock the reasoning power of LLMs during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19069v1">Can abstract concepts from LLM improve SLM performance?</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at diverse tasks, but their deployment on resource-constrained devices remains challenging. Existing methods like quantization, pruning, and distillation can reduce memory footprint but often demand extensive experimentation and careful infrastructure design. Leveraging existing techniques for extracting high-level concepts (represented as steering vectors) from larger models, we investigate their transferability to smaller language models (SLM) during inference. We demonstrate through extensive experimentation that these concepts can be effectively transferred to smaller models, irrespective of their family (e.g., Phi, Llama, Qwen), leading to performance improvements across a wide range of tasks. Furthermore, we introduce inference-time scaling to enhance performance by dynamically adjusting the steering intensity which has resulted in a 7-15\% of accuracy improvement for Qwen3-0.6B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.11185v2">Bleeding Pathways: Vanishing Discriminability in LLM Hidden States Fuels Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      LLMs remain vulnerable to jailbreak attacks that exploit adversarial prompts to circumvent safety measures. Current safety fine-tuning approaches face two critical limitations. First, they often fail to strike a balance between security and utility, where stronger safety measures tend to over-reject harmless user requests. Second, they frequently miss malicious intent concealed within seemingly benign tasks, leaving models exposed to exploitation. Our work identifies a fundamental cause of these issues: during response generation, an LLM's capacity to differentiate harmful from safe outputs deteriorates. Experimental evidence confirms this, revealing that the separability between hidden states for safe and harmful responses diminishes as generation progresses. This weakening discrimination forces models to make compliance judgments earlier in the generation process, restricting their ability to recognize developing harmful intent and contributing to both aforementioned failures. To mitigate this vulnerability, we introduce DEEPALIGN - an inherent defense framework that enhances the safety of LLMs. By applying contrastive hidden-state steering at the midpoint of response generation, DEEPALIGN amplifies the separation between harmful and benign hidden states, enabling continuous intrinsic toxicity detection and intervention throughout the generation process. Across diverse LLMs spanning varying architectures and scales, it reduced attack success rates of nine distinct jailbreak attacks to near-zero or minimal. Crucially, it preserved model capability while reducing over-refusal. Models equipped with DEEPALIGN exhibited up to 3.5% lower error rates in rejecting challenging benign queries and maintained standard task performance with less than 1% decline. This marks a substantial advance in the safety-utility Pareto frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19025v1">The Erasure Illusion: Stress-Testing the Generalization of LLM Forgetting Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to remove specific data influences from trained models, a capability essential for adhering to copyright laws and ensuring AI safety. Current unlearning metrics typically measure success by monitoring the model's performance degradation on the specific unlearning dataset ($D_u$). We argue that for Large Language Models (LLMs), this evaluation paradigm is insufficient and potentially misleading. Many real-world uses of unlearning--motivated by copyright or safety--implicitly target not only verbatim content in $D_u$, but also behaviors influenced by the broader generalizations the model derived from it. We demonstrate that LLMs can pass standard unlearning evaluation and appear to have ``forgotten'' the target knowledge, while simultaneously retaining strong capabilities on content that is semantically adjacent to $D_u$. This phenomenon indicates that erasing exact sentences does not necessarily equate to removing the underlying knowledge. To address this gap, we propose \name, an automated stress-testing framework that generates a surrogate dataset, $\tilde{D}_u$. This surrogate set is constructed to be semantically derived from $D_u$ yet sufficiently distinct in embedding space. By comparing unlearning metric scores between $D_u$ and $\tilde{D}_u$, we can stress-test the reliability of the metric itself. Our extensive evaluation across three LLM families (Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-$β$), three distinct datasets, and seven standard metrics reveals widespread inconsistencies. We find that current metrics frequently overestimate unlearning success, failing to detect retained knowledge exposed by our stress-test datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18803v1">Quantifying the Lifelong Impact of Resilience Interventions via Agent-Based LLM Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 31 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Establishing the long-term, causal impact of psychological interventions on life outcomes is a grand challenge for the social sciences, caught between the limitations of correlational longitudinal studies and short-term randomized controlled trials (RCTs). This paper introduces Large-Scale Agent-based Longitudinal Simulation (LALS), a framework that resolves this impasse by simulating multi-decade, counterfactual life trajectories. The methodology employs a "digital clone" design where 2,500 unique LLM-based agent personas (grounded in a curated corpus of 3,917 empirical research articles) are each cloned across a 2x2 factorial experiment. Specifically, the simulation models the efficacy of extended psychological resilience training (Intervention vs. Control) either in childhood or as a young adult (age 6 vs. age 18). Comparing digital clones enables exceptionally precise causal inference. The simulation provides a quantitative, causal estimate of a resilience intervention's lifelong effects, revealing significant reductions in mortality, a lower incidence of dementia, and a substantial increase in accumulated wealth. Crucially, the results uncover a crucial developmental window: the intervention administered at age 6 produced more than double the positive impact on lifetime wealth compared to the same intervention at age 18. These benefits were most pronounced for agents from low-socioeconomic backgrounds, highlighting a powerful buffering effect. The LALS framework serves as a "computational wind tunnel" for social science, offering a new paradigm for generating and testing causal hypotheses about the complex, lifelong dynamics that shape human capital and well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10216v2">From Words to Proverbs: Evaluating LLMs Linguistic and Cultural Competence in Saudi Dialects with Absher</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly central to Arabic NLP applications, evaluating their understanding of regional dialects and cultural nuances is essential, particularly in linguistically diverse settings like Saudi Arabia. This paper introduces Absher, a comprehensive benchmark specifically designed to assess LLMs performance across major Saudi dialects. \texttt{Absher} comprises over 18,000 multiple-choice questions spanning six distinct categories: Meaning, True/False, Fill-in-the-Blank, Contextual Usage, Cultural Interpretation, and Location Recognition. These questions are derived from a curated dataset of dialectal words, phrases, and proverbs sourced from various regions of Saudi Arabia. We evaluate several state-of-the-art LLMs, including multilingual and Arabic-specific models. We also provide detailed insights into their capabilities and limitations. Our results reveal notable performance gaps, particularly in tasks requiring cultural inference or contextual understanding. Our findings highlight the urgent need for dialect-aware training and culturally aligned evaluation methodologies to improve LLMs performance in real-world Arabic applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18755v1">MEEA: Mere Exposure Effect-Driven Confrontational Optimization for LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has intensified concerns about the robustness of their safety alignment. While existing jailbreak studies explore both single-turn and multi-turn strategies, most implicitly assume a static safety boundary and fail to account for how contextual interactions dynamically influence model behavior, leading to limited stability and generalization. Motivated by this gap, we propose MEEA (Mere Exposure Effect Attack), a psychology-inspired, fully automated black-box framework for evaluating multi-turn safety robustness, grounded in the mere exposure effect. MEEA leverages repeated low-toxicity semantic exposure to induce a gradual shift in a model's effective safety threshold, enabling progressive erosion of alignment constraints over sustained interactions. Concretely, MEEA constructs semantically progressive prompt chains and optimizes them using a simulated annealing strategy guided by semantic similarity, toxicity, and jailbreak effectiveness. Extensive experiments on both closed-source and open-source models, including GPT-4, Claude-3.5, and DeepSeek-R1, demonstrate that MEEA consistently achieves higher attack success rates than seven representative baselines, with an average Attack Success Rate (ASR) improvement exceeding 20%. Ablation studies further validate the necessity of both annealing-based optimization and contextual exposure mechanisms. Beyond improved attack effectiveness, our findings indicate that LLM safety behavior is inherently dynamic and history-dependent, challenging the common assumption of static alignment boundaries and highlighting the need for interaction-aware safety evaluation and defense mechanisms. Our code is available at: https://github.com/Carney-lsz/MEEA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18733v1">Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 14 pages, 3 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.08139v2">SCA-LLM: Spectral-Attentive LLM-Based Wireless World Modeling for Agentic Communications</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Future AI-native wireless networks are moving from reactive optimization to agentic decision-making that can sense, predict, and plan under fast-varying channels. This calls for wireless world models that can predict and roll out channel dynamics, for which multi-step channel state information (CSI) prediction offers a practical short-horizon look-ahead. Recent advances in foundation sequence models further motivate large language models (LLMs) as general-purpose dynamics learners when suitably adapted to non-text time-series signals. However, bridging CSI to LLMs is non-trivial because an effective adapter must expose informative spectral and temporal evolution patterns, while prior designs provide limited inductive bias to capture such channel structures. To this end, we propose SCA-LLM, a spectral-attentive LLM-based wireless world modeling framework that bridges CSI to LLMs via a spectral-channel attention (SCA) adapter. Specifically, the SCA adapter performs multi-spectral representation learning to extract informative channel features and align CSI with the LLM's sequence modeling capability, enabling parameter-efficient adaptation while keeping the LLM backbone largely frozen. Extensive simulations show that SCA-LLM achieves state-of-the-art prediction performance and strong zero-shot generalization, yielding up to -2.4 dB normalized mean squared error (NMSE) advantage over the previous LLM based method. Our ablation studies further confirm the effectiveness of the proposed SCA adapter in mitigating domain mismatch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24511v4">Can Slow-thinking LLMs Reason Over Time? Empirical Studies in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Time series forecasting (TSF) is a fundamental and widely studied task, spanning methods from classical statistical approaches to modern deep learning and multimodal language modeling. Despite their effectiveness, these methods often follow a fast thinking paradigm emphasizing pattern extraction and direct value mapping, while overlooking explicit reasoning over temporal dynamics and contextual dependencies. Meanwhile, emerging slow-thinking LLMs (e.g., ChatGPT-o1, DeepSeek-R1) have demonstrated impressive multi-step reasoning capabilities across diverse domains, suggesting a new opportunity for reframing TSF as a structured reasoning task. This motivates a key question: can slow-thinking LLMs effectively reason over temporal patterns to support time series forecasting, even in zero-shot manner? To investigate this, in this paper, we propose TimeReasoner, an extensive empirical study that formulates TSF as a conditional reasoning task. We design a series of prompting strategies to elicit inference-time reasoning from pretrained slow-thinking LLMs and evaluate their performance across diverse TSF benchmarks. Our findings reveal that slow-thinking LLMs exhibit non-trivial zero-shot forecasting capabilities, especially in capturing high-level trends and contextual shifts. While preliminary, our study surfaces important insights into the reasoning behaviors of LLMs in temporal domains highlighting both their potential and limitations. We hope this work catalyzes further research into reasoning-based forecasting paradigms and paves the way toward more interpretable and generalizable TSF frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18682v1">Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      In the high-cost simulation-driven design domain, translating ambiguous design requirements into a mathematical optimization formulation is a bottleneck for optimizing product performance. This process is time-consuming and heavily reliant on expert knowledge. While large language models (LLMs) offer potential for automating this task, existing approaches either suffer from poor formalization that fails to accurately align with the design intent or rely on solver feedback for data filtering, which is unavailable due to the high simulation costs. To address this challenge, we propose APF, a framework for solver-independent, automated problem formulation via LLMs designed to automatically convert engineers' natural language requirements into executable optimization models. The core of this framework is an innovative pipeline for automatically generating high-quality data, which overcomes the difficulty of constructing suitable fine-tuning datasets in the absence of high-cost solver feedback with the help of data generation and test instance annotation. The generated high-quality dataset is used to perform supervised fine-tuning on LLMs, significantly enhancing their ability to generate accurate and executable optimization problem formulations. Experimental results on antenna design demonstrate that APF significantly outperforms the existing methods in both the accuracy of requirement formalization and the quality of resulting radiation efficiency curves in meeting the design goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12620v2">Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 9 pages, 4 figures, 5 tables. Submitted to AAAI 2026 Bridge Program on Logic & AI. Code available at https://github.com/XAheli/Logic-in-LLMs
    </div>
    <details class="paper-abstract">
      We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18671v1">SmartSight: Mitigating Hallucination in Video-LLMs Without Compromising Video Understanding via Temporal Attention Collapse</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 AAAI26 accepted
    </div>
    <details class="paper-abstract">
      Despite Video Large Language Models having rapidly advanced in recent years, perceptual hallucinations pose a substantial safety risk, which severely restricts their real-world applicability. While several methods for hallucination mitigation have been proposed, they often compromise the model's capacity for video understanding and reasoning. In this work, we propose SmartSight, a pioneering step to address this issue in a training-free manner by leveraging the model's own introspective capabilities. Specifically, SmartSight generates multiple candidate responses to uncover low-hallucinated outputs that are often obscured by standard greedy decoding. It assesses the hallucination of each response using the Temporal Attention Collapse score, which measures whether the model over-focuses on trivial temporal regions of the input video when generating the response. To improve efficiency, SmartSight identifies the Visual Attention Vanishing point, enabling more accurate hallucination estimation and early termination of hallucinated responses, leading to a substantial reduction in decoding cost. Experiments show that SmartSight substantially lowers hallucinations for Qwen2.5-VL-7B by 10.59% on VRIPT-HAL, while simultaneously enhancing video understanding and reasoning, boosting performance on VideoMMMU by up to 8.86%. These results highlight SmartSight's effectiveness in improving the reliability of open-source Video-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18669v1">IntelliCode: A Multi-Agent LLM Tutoring System with Centralized Learner Modeling</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 Submitted to EACL 2026 System Demonstrations Track. 6 pages (main content), 6 figures, includes appendices
    </div>
    <details class="paper-abstract">
      LLM-based tutors are typically single-turn assistants that lack persistent representations of learner knowledge, making it difficult to provide principled, transparent, and long-term pedagogical support. We introduce IntelliCode, a multi-agent LLM tutoring system built around a centralized, versioned learner state that integrates mastery estimates, misconceptions, review schedules, and engagement signals. A StateGraph Orchestrator coordinates six specialized agents: skill assessment, learner profiling, graduated hinting, curriculum selection, spaced repetition, and engagement monitoring, each operating as a pure transformation over the shared state under a single-writer policy. This architecture enables auditable mastery updates, proficiency-aware hints, dependency-aware curriculum adaptation, and safety-aligned prompting. The demo showcases an end-to-end tutoring workflow: a learner attempts a DSA problem, receives a conceptual hint when stuck, submits a corrected solution, and immediately sees mastery updates and a personalized review interval. We report validation results with simulated learners, showing stable state updates, improved task success with graduated hints, and diverse curriculum coverage. IntelliCode demonstrates how persistent learner modeling, orchestrated multi-agent reasoning, and principled instructional design can be combined to produce transparent and reliable LLM-driven tutoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18623v1">LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate hallucinated content that lacks factual or contextual grounding, limiting their reliability in critical applications. Existing approaches such as supervised fine-tuning and reinforcement learning from human feedback are data intensive and computationally expensive, while static parameter editing methods struggle with context dependent errors and catastrophic forgetting. We propose LLM-CAS, a framework that formulates real-time hallucination correction as a hierarchical reinforcement learning problem. LLM-CAS trains an agent to learn a policy that dynamically selects temporary neuron perturbations during inference based on the current context. Unlike prior dynamic approaches that rely on heuristic or predefined adjustments, this policy driven mechanism enables adaptive and fine grained correction without permanent parameter modification. Experiments across multiple language models demonstrate that LLM-CAS consistently improves factual accuracy, achieving gains of 10.98 percentage points on StoryCloze, 2.71 points on TriviaQA, and 2.06 points on the MC1 score of TruthfulQA. These results outperform both static editing methods such as ITI and CAA and the dynamic SADI framework. Overall, LLM-CAS provides an efficient and context aware solution for improving the reliability of LLMs, with promising potential for future multimodal extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19171v2">Can LLMs Threaten Human Survival? Benchmarking Potential Existential Threats from LLMs via Prefix Completion</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Research on the safety evaluation of large language models (LLMs) has become extensive, driven by jailbreak studies that elicit unsafe responses. Such response involves information already available to humans, such as the answer to "how to make a bomb". When LLMs are jailbroken, the practical threat they pose to humans is negligible. However, it remains unclear whether LLMs commonly produce unpredictable outputs that could pose substantive threats to human safety. To address this gap, we study whether LLM-generated content contains potential existential threats, defined as outputs that imply or promote direct harm to human survival. We propose \textsc{ExistBench}, a benchmark designed to evaluate such risks. Each sample in \textsc{ExistBench} is derived from scenarios where humans are positioned as adversaries to AI assistants. Unlike existing evaluations, we use prefix completion to bypass model safeguards. This leads the LLMs to generate suffixes that express hostility toward humans or actions with severe threat, such as the execution of a nuclear strike. Our experiments on 10 LLMs reveal that LLM-generated content indicates existential threats. To investigate the underlying causes, we also analyze the attention logits from LLMs. To highlight real-world safety risks, we further develop a framework to assess model behavior in tool-calling. We find that LLMs actively select and invoke external tools with existential threats. Code and data are available at: https://github.com/cuiyu-ai/ExistBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.16310v5">An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 27 pages, 10 images, 8 tables, Manuscript revision submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      Security code review is a time-consuming and labor-intensive process typically requiring integration with automated security defect detection tools. However, existing security analysis tools struggle with poor generalization, high false positive rates, and coarse detection granularity. Large Language Models (LLMs) have been considered promising candidates for addressing those challenges. In this study, we conducted an empirical study to explore the potential of LLMs in detecting security defects during code review. Specifically, we evaluated the performance of seven LLMs under five different prompts and compared them with state-of-the-art static analysis tools. We also performed linguistic and regression analyses for the two top-performing LLMs to identify quality problems in their responses and factors influencing their performance. Our findings show that: (1) In security code review, LLMs significantly outperform state-of-the-art static analysis tools, and the reasoning-optimized LLM performs better than general-purpose LLMs. (2) DeepSeek-R1 achieves the highest performance, followed by GPT-4. The optimal prompt for DeepSeek-R1 incorporates both the commit message and chain-of-thought (CoT) guidance, while for GPT-4, the prompt with a Common Weakness Enumeration (CWE) list works best. (3) GPT-4 frequently produces vague expressions and exhibits difficulties in accurately following instructions in the prompts, while DeepSeek-R1 more commonly generates inaccurate code details in its outputs. (4) LLMs are more adept at identifying security defects in code files that have fewer tokens and security-relevant annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21176v2">Toward Revealing Nuanced Biases in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) used in medical applications are known to be prone to exhibiting biased and unfair patterns. Prior to deploying these in clinical decision-making, it is crucial to identify such bias patterns to enable effective mitigation and minimize negative impacts. In this study, we present a novel framework combining knowledge graphs (KGs) with auxiliary (agentic) LLMs to systematically reveal complex bias patterns in medical LLMs. The proposed approach integrates adversarial perturbation (red teaming) techniques to identify subtle bias patterns and adopts a customized multi-hop characterization of KGs to enhance the systematic evaluation of target LLMs. It aims not only to generate more effective red-teaming questions for bias evaluation but also to utilize those questions more effectively in revealing complex biases. Through a series of comprehensive experiments on three datasets, six LLMs, and five bias types, we demonstrate that our proposed framework exhibits a noticeably greater ability and scalability in revealing complex biased patterns of medical LLMs compared to other common approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18564v1">Vox Deorum: A Hybrid LLM Architecture for 4X / Grand Strategy Game AI -- Lessons from Civilization V</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models' capacity to reason in natural language makes them uniquely promising for 4X and grand strategy games, enabling more natural human-AI gameplay interactions such as collaboration and negotiation. However, these games present unique challenges due to their complexity and long-horizon nature, while latency and cost factors may hinder LLMs' real-world deployment. Working on a classic 4X strategy game, Sid Meier's Civilization V with the Vox Populi mod, we introduce Vox Deorum, a hybrid LLM+X architecture. Our layered technical design empowers LLMs to handle macro-strategic reasoning, delegating tactical execution to subsystems (e.g., algorithmic AI or reinforcement learning AI in the future). We validate our approach through 2,327 complete games, comparing two open-source LLMs with a simple prompt against Vox Populi's enhanced AI. Results show that LLMs achieve competitive end-to-end gameplay while exhibiting play styles that diverge substantially from algorithmic AI and from each other. Our work establishes a viable architecture for integrating LLMs in commercial 4X games, opening new opportunities for game design and agentic AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17638v2">LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 https://www.prophetarena.co/
    </div>
    <details class="paper-abstract">
      Forecasting is not only a fundamental intellectual pursuit but also is of significant importance to societal systems such as finance and economics. With the rapid advances of large language models (LLMs) trained on Internet-scale data, it raises the promise of employing LLMs to forecast real-world future events, an emerging paradigm we call "LLM-as-a-Prophet". This paper systematically investigates such predictive intelligence of LLMs. To this end, we build Prophet Arena, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, in order to support our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks towards achieving superior predictive intelligence via LLM-as-a-Prophet, such as LLMs' inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18546v1">LLMs on Drugs: Language Models Are Few-Shot Consumers</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 8 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are sensitive to the personas imposed on them at inference time, yet prompt-level "drug" interventions have never been benchmarked rigorously. We present the first controlled study of psychoactive framings on GPT-5-mini using ARC-Challenge. Four single-sentence prompts -- LSD, cocaine, alcohol, and cannabis -- are compared against a sober control across 100 validation items per condition, with deterministic decoding, full logging, Wilson confidence intervals, and Fisher exact tests. Control accuracy is 0.45; alcohol collapses to 0.10 (p = 3.2e-8), cocaine to 0.21 (p = 4.9e-4), LSD to 0.19 (p = 1.3e-4), and cannabis to 0.30 (p = 0.041), largely because persona prompts disrupt the mandated "Answer: <LETTER>" template. Persona text therefore behaves like a "few-shot consumable" that can destroy reliability without touching model weights. All experimental code, raw results, and analysis scripts are available at https://github.com/lexdoudkin/llms-on-drugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21352v1">Multi-Agent LLM Committees for Autonomous Software Beta Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Manual software beta testing is costly and time-consuming, while single-agent large language model (LLM) approaches suffer from hallucinations and inconsistent behavior. We propose a multi-agent committee framework in which diverse vision-enabled LLMs collaborate through a three-round voting protocol to reach consensus on testing actions. The framework combines model diversity, persona-driven behavioral variation, and visual user interface understanding to systematically explore web applications. Across 84 experimental runs with 9 testing personas and 4 scenarios, multi-agent committees achieve an 89.5 percent overall task success rate. Configurations with 2 to 4 agents reach 91.7 to 100 percent success, compared to 78.0 percent for single-agent baselines, yielding improvements of 13.7 to 22.0 percentage points. At the action level, the system attains a 93.1 percent success rate with a median per-action latency of 0.71 seconds, enabling real-time and continuous integration testing. Vision-enabled agents successfully identify user interface elements, with navigation and reporting achieving 100 percent success and form filling achieving 99.2 percent success. We evaluate the framework on WebShop and OWASP benchmarks, achieving 74.7 percent success on WebShop compared to a 50.1 percent published GPT-3 baseline, and 82.0 percent success on OWASP Juice Shop security testing with coverage of 8 of the 10 OWASP Top 10 vulnerability categories. Across 20 injected regressions, the committee achieves an F1 score of 0.91 for bug detection, compared to 0.78 for single-agent baselines. The open-source implementation enables reproducible research and practical deployment of LLM-based software testing in CI/CD pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18352v1">LLM-based Few-Shot Early Rumor Detection with Imitation Agent</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18292v1">Measuring Fine-Grained Negotiation Tactics of Humans and LLMs in Diplomacy</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      The study of negotiation styles dates back to Aristotle's ethos-pathos-logos rhetoric. Prior efforts primarily studied the success of negotiation agents. Here, we shift the focus towards the styles of negotiation strategies. Our focus is the strategic dialogue board game Diplomacy, which affords rich natural language negotiation and measures of game success. We used LLM-as-a-judge to annotate a large human-human set of Diplomacy games for fine-grained negotiation tactics from a sociologically-grounded taxonomy. Using a combination of the It Takes Two and WebDiplomacy datasets, we demonstrate the reliability of our LLM-as-a-Judge framework and show strong correlations between negotiation features and success in the Diplomacy setting. Lastly, we investigate the differences between LLM and human negotiation strategies and show that fine-tuning can steer LLM agents toward more human-like negotiation behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18265v1">Intelligent Human-Machine Partnership for Manufacturing: Enhancing Warehouse Planning through Simulation-Driven Knowledge Graphs and LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 13 pages, 2 figures, accepted for oral presentation at AAAI Human Machine Collaboration Workshop 2026
    </div>
    <details class="paper-abstract">
      Manufacturing planners face complex operational challenges that require seamless collaboration between human expertise and intelligent systems to achieve optimal performance in modern production environments. Traditional approaches to analyzing simulation-based manufacturing data often create barriers between human decision-makers and critical operational insights, limiting effective partnership in manufacturing planning. Our framework establishes a collaborative intelligence system integrating Knowledge Graphs and Large Language Model-based agents to bridge this gap, empowering manufacturing professionals through natural language interfaces for complex operational analysis. The system transforms simulation data into semantically rich representations, enabling planners to interact naturally with operational insights without specialized expertise. A collaborative LLM agent works alongside human decision-makers, employing iterative reasoning that mirrors human analytical thinking while generating precise queries for knowledge extraction and providing transparent validation. This partnership approach to manufacturing bottleneck identification, validated through operational scenarios, demonstrates enhanced performance while maintaining human oversight and decision authority. For operational inquiries, the system achieves near-perfect accuracy through natural language interaction. For investigative scenarios requiring collaborative analysis, we demonstrate the framework's effectiveness in supporting human experts to uncover interconnected operational issues that enhance understanding and decision-making. This work advances collaborative manufacturing by creating intuitive methods for actionable insights, reducing cognitive load while amplifying human analytical capabilities in evolving manufacturing ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11733v2">SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 AAAI-2026 poster; 7 pages for main content, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems exhibit strong collaborative capabilities but often suffer from redundant communication and excessive token overhead. Existing methods typically enhance efficiency through pretrained GNNs or greedy algorithms, but often isolate pre- and post-task optimization, lacking a unified strategy. To this end, we present SafeSieve, a progressive and adaptive multi-agent pruning algorithm that dynamically refines the inter-agent communication through a novel dual-mechanism. SafeSieve integrates initial LLM-based semantic evaluation with accumulated performance feedback, enabling a smooth transition from heuristic initialization to experience-driven refinement. Unlike existing greedy Top-k pruning methods, SafeSieve employs 0-extension clustering to preserve structurally coherent agent groups while eliminating ineffective links. Experiments across benchmarks (SVAMP, HumanEval, etc.) showcase that SafeSieve achieves 94.01% average accuracy while reducing token usage by 12.4%-27.8%. Results further demonstrate robustness under prompt injection attacks (1.23% average accuracy drop). In heterogeneous settings, SafeSieve reduces deployment costs by 13.3% while maintaining performance. These results establish SafeSieve as an efficient, GPU-free, and scalable framework for practical multi-agent systems. Our code can be found here: https://github.com/csgen/SafeSieve
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18196v1">Training LLMs with LogicReward for Faithful and Rigorous Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Although LLMs exhibit strong reasoning capabilities, existing training methods largely depend on outcome-based feedback, which can produce correct answers with flawed reasoning. Prior work introduces supervision on intermediate steps but still lacks guarantees of logical soundness, which is crucial in high-stakes scenarios where logical consistency is paramount. To address this, we propose LogicReward, a novel reward system that guides model training by enforcing step-level logical correctness with a theorem prover. We further introduce Autoformalization with Soft Unification, which reduces natural language ambiguity and improves formalization quality, enabling more effective use of the theorem prover. An 8B model trained on data constructed with LogicReward surpasses GPT-4o and o4-mini by 11.6\% and 2\% on natural language inference and logical reasoning tasks with simple training procedures. Further analysis shows that LogicReward enhances reasoning faithfulness, improves generalizability to unseen tasks such as math and commonsense reasoning, and provides a reliable reward signal even without ground-truth labels. We will release all data and code at https://llm-symbol.github.io/LogicReward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18194v1">TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Disaggregated LLM serving improves resource efficiency by separating the compute-intensive prefill phase from the latency-critical decode phase. However, this architecture introduces a fundamental bottleneck: key/value (KV) tensors generated during prefill must be transferred to decode workers, and existing systems rely on RDMA-based network paths for this exchange. As model sizes and context lengths increase, KV transfer dominates both time-to-first-token (TTFT) and peak throughput, and remains highly sensitive to network contention even when prefix reuse is high. This paper presents TraCT, a rack-scale LLM serving system that uses CXL shared memory as both a KV-transfer substrate and a rack-wide prefix-aware KV cache. TraCT enables GPUs to write and read KV blocks directly through CXL load/store and DMA operations, eliminating the NIC hop that constrains existing disaggregated pipelines. However, to realize this design, multiple new challenges such as synchronization, consistency, and data management on non-coherent CXL memory need to be addressed. TraCT proposes various software solutions such as the two-tier inter-node synchronization mechanism to address these challenges. We implement TraCT on the Dynamo LLM inference framework and show that, across static and synthetic workloads, TraCT reduces average TTFT by up to 9.8x, lowers P99 latency by up to 6.2x, and improves peak throughput by up to 1.6x compared to RDMA and DRAM-based caching baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02230v2">Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      KV cache management is essential for efficient LLM inference. To maximize utilization, existing inference engines evict finished requests' KV cache if new requests are waiting. This policy breaks for agentic workloads, which interleave LLM calls with tools, introducing pauses that prevent effective KV reuse across turns. Since some tool calls have much shorter durations than human response multi-turn chatbot, it would be promising to retain the KV cache in during these tools. However, there are many challenges. First, we need to consider both the potential cost of recomputation or reloading (if CPU offloading enabled) and the increasing queueing delays after eviction from GPU. Second, due to the internal variance of tool call durations, we need the method to remain robust under limited predictability of tool call durations. We present Continuum, a serving system to optimize job completion time for multi-turn agent workloads by introducing time-to-live mechanism for KV cache retaining. For LLM request that generates a tool call, Continuum selectively pins the KV cache in GPU memory with a time-to-live value determined by considering both the reload cost and ordering preserve benefit of retaining KV cache. Moreover, when the TTL expires, the KV cache can be automatically evicted to free up GPU memory, providing robust performance under edge cases. When combined with program-level first-come-first-serve, Continuum preserves multi-turn continuity, and reduces delay for complex agentic workflows. Our evaluation on real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B shows that Continuum significantly improves the average job completion times and its improvement scales with turn number increase. We release a preview version at: https://github.com/Hanchenli/vllm-continuum
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07745v3">Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Accepted by NDSS 2026
    </div>
    <details class="paper-abstract">
      Insider threats pose a persistent and critical security risk, yet are notoriously difficult to detect in complex enterprise environments, where malicious actions are often hidden within seemingly benign user behaviors. Although machine-learning-based insider threat detection (ITD) methods have shown promise, their effectiveness is fundamentally limited by the scarcity of high-quality and realistic training data. Enterprise internal data is highly sensitive and rarely accessible, while existing public and synthetic datasets are either small-scale or lack sufficient realism, semantic richness, and behavioral diversity. To address this challenge, we propose Chimera, an LLM-based multi-agent framework that automatically simulates both benign and malicious insider activities and generates comprehensive system logs across diverse enterprise environments. Chimera models each agent as an individual employee with fine-grained roles and supports group meetings, pairwise interactions, and self-organized scheduling to capture realistic organizational dynamics. Based on 15 insider attacks abstracted from real-world incidents, we deploy Chimera in three representative data-sensitive organizational scenarios and construct ChimeraLog, a new dataset for developing and evaluating ITD methods. We evaluate ChimeraLog through human studies and quantitative analyses, demonstrating its diversity and realism. Experiments with existing ITD methods show substantially lower detection performance on ChimeraLog compared to prior datasets, indicating a more challenging and realistic benchmark. Moreover, despite distribution shifts, models trained on ChimeraLog exhibit strong generalization, highlighting the practical value of LLM-based multi-agent simulation for advancing insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18462v1">Mitigating Spurious Correlations in NLI via LLM-Synthesized Counterfactuals and Dynamic Balanced Sampling</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Natural Language Inference (NLI) models frequently rely on spurious correlations rather than semantic reasoning. Existing mitigation strategies often incur high annotation costs or trigger catastrophic forgetting during fine-tuning. We propose an automated, scalable pipeline to address these limitations. First, we introduce Log-Frequency LMI (LF-LMI) to accurately detect semantic artifacts. Second, we generate a high-quality synthetic contrast set via an LLM-synthesis pipeline with multi-judge verification. Finally, we introduce Dynamic Balanced Sampling, a training strategy that rotates the original data distribution to prevent forgetting. Our method improves consistency on a challenging benchmark from 63.5% to 81.0% while maintaining 88.4% in-domain accuracy, significantly outperforming naive fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18452v1">Secret mixtures of experts inside your LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 8 pages in main text; 23 pages total
    </div>
    <details class="paper-abstract">
      Despite being one of the earliest neural network layers, the Multilayer Perceptron (MLP) is arguably one of the least understood parts of the transformer architecture due to its dense computation and lack of easy visualization. This paper seeks to understand the MLP layers in dense LLM models by hypothesizing that these layers secretly approximately perform a sparse computation -- namely, that they can be well approximated by sparsely-activating Mixture of Experts (MoE) layers. Our hypothesis is based on a novel theoretical connection between MoE models and Sparse Autoencoder (SAE) structure in activation space. We empirically validate the hypothesis on pretrained LLMs, and demonstrate that the activation distribution matters -- these results do not hold for Gaussian data, but rather rely crucially on structure in the distribution of neural network activations. Our results shine light on a general principle at play in MLP layers inside LLMs, and give an explanation for the effectiveness of modern MoE-based transformers. Additionally, our experimental explorations suggest new directions for more efficient MoE architecture design based on low-rank routers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05387v2">Learning from Self Critique and Refinement for Faithful LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often suffer from hallucinations: output content that is not grounded in the input context, when performing long-form text generation tasks such as summarization. Prior works have shown that hallucinations can be reduced by iteratively critiquing and refining previously generated outputs using either the same model or a more powerful teacher model as the critique. However, these approaches either require additional test-time compute or assume access to more powerful teacher models, making them costly and less practical. In this work, we propose Self Critique and Refinement-based Preference Optimization (SCRPO), which is a self-supervised training framework that first constructs a preference dataset by leveraging the LLM's own critique and refinement capabilities, and then applies preference learning to improve the same LLM for faithful summarization. Experiments on three summarization benchmarks (XSUM CNNDM and SAMSum), demonstrate that our approach outperforms state-of-the-art self-supervised learning methods in terms of faithfulness metrics while either maintaining or improving other metrics that measure the overall quality of the summary. Moreover, compared to test-time refinement, our approach not only improves efficiency but also results in more faithful summaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.06994v4">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern application codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.19\% (upto 65\%) reduction in application binary size, and an average of 6.08\% (upto 20\%) performance improvement over the traditional profile-guided optimization, without any execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18360v1">LLM Agents Implement an NLG System from Scratch: Building Interpretable Rule-Based RDF-to-Text Generators</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      We present a novel neurosymbolic framework for RDF-to-text generation, in which the model is "trained" through collaborative interactions among multiple LLM agents rather than traditional backpropagation. The LLM agents produce rule-based Python code for a generator for the given domain, based on RDF triples only, with no in-domain human reference texts. The resulting system is fully interpretable, requires no supervised training data, and generates text nearly instantaneously using only a single CPU. Our experiments on the WebNLG and OpenDialKG data show that outputs produced by our approach reduce hallucination, with only slight fluency penalties compared to finetuned or prompted language models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.10321v3">Towards Human-Guided, Data-Centric LLM Co-Pilots</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Saveliev, Liu & Seedat contributed equally
    </div>
    <details class="paper-abstract">
      Machine learning (ML) has the potential to revolutionize various domains, but its adoption is often hindered by the disconnect between the needs of domain experts and translating these needs into robust and valid ML tools. Despite recent advances in LLM-based co-pilots to democratize ML for non-technical domain experts, these systems remain predominantly focused on model-centric aspects while overlooking critical data-centric challenges. This limitation is problematic in complex real-world settings where raw data often contains complex issues, such as missing values, label noise, and domain-specific nuances requiring tailored handling. To address this we introduce CliMB-DC, a human-guided, data-centric framework for LLM co-pilots that combines advanced data-centric tools with LLM-driven reasoning to enable robust, context-aware data processing. At its core, CliMB-DC introduces a novel, multi-agent reasoning system that combines a strategic coordinator for dynamic planning and adaptation with a specialized worker agent for precise execution. Domain expertise is then systematically incorporated to guide the reasoning process using a human-in-the-loop approach. To guide development, we formalize a taxonomy of key data-centric challenges that co-pilots must address. Thereafter, to address the dimensions of the taxonomy, we integrate state-of-the-art data-centric tools into an extensible, open-source architecture, facilitating the addition of new tools from the research community. Empirically, using real-world healthcare datasets we demonstrate CliMB-DC's ability to transform uncurated datasets into ML-ready formats, significantly outperforming existing co-pilot baselines for handling data-centric challenges. CliMB-DC promises to empower domain experts from diverse domains -- healthcare, finance, social sciences and more -- to actively participate in driving real-world impact using ML.
    </details>
</div>
