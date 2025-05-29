# llm - 2025_05

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22664v1">Zero-Shot Vision Encoder Grafting via LLM Surrogates</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Vision language models (VLMs) typically pair a modestly sized vision encoder with a large language model (LLM), e.g., Llama-70B, making the decoder the primary computational burden during training. To reduce costs, a potential promising strategy is to first train the vision encoder using a small language model before transferring it to the large one. We construct small "surrogate models" that share the same embedding space and representation language as the large target LLM by directly inheriting its shallow layers. Vision encoders trained on the surrogate can then be directly transferred to the larger model, a process we call zero-shot grafting -- when plugged directly into the full-size target LLM, the grafted pair surpasses the encoder-surrogate pair and, on some benchmarks, even performs on par with full decoder training with the target LLM. Furthermore, our surrogate training approach reduces overall VLM training costs by ~45% when using Llama-70B as the decoder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22661v1">GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by ACL 2025
    </div>
    <details class="paper-abstract">
      The evaluation of large language models (LLMs) has traditionally relied on static benchmarks, a paradigm that poses two major limitations: (1) predefined test sets lack adaptability to diverse application domains, and (2) standardized evaluation protocols often fail to capture fine-grained assessments of domain-specific knowledge and contextual reasoning abilities. To overcome these challenges, we propose GuessArena, an adaptive evaluation framework grounded in adversarial game-based interactions. Inspired by the interactive structure of the Guess Who I Am? game, our framework seamlessly integrates dynamic domain knowledge modeling with progressive reasoning assessment to improve evaluation fidelity. Empirical studies across five vertical domains-finance, healthcare, manufacturing, information technology, and education-demonstrate that GuessArena effectively distinguishes LLMs in terms of domain knowledge coverage and reasoning chain completeness. Compared to conventional benchmarks, our method provides substantial advantages in interpretability, scalability, and scenario adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22630v1">Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted to ACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      The widespread success of large language models (LLMs) on NLP benchmarks has been accompanied by concerns that LLMs function primarily as stochastic parrots that reproduce texts similar to what they saw during pre-training, often erroneously. But what is the nature of their errors, and do these errors exhibit any regularities? In this work, we examine irrelevant context hallucinations, in which models integrate misleading contextual cues into their predictions. Through behavioral analysis, we show that these errors result from a structured yet flawed mechanism that we term class-based (mis)generalization, in which models combine abstract class cues with features extracted from the query or context to derive answers. Furthermore, mechanistic interpretability experiments on Llama-3, Mistral, and Pythia across 39 factual recall relation types reveal that this behavior is reflected in the model's internal computations: (i) abstract class representations are constructed in lower layers before being refined into specific answers in higher layers, (ii) feature selection is governed by two competing circuits -- one prioritizing direct query-based reasoning, the other incorporating contextual cues -- whose relative influences determine the final output. Our findings provide a more nuanced perspective on the stochastic parrot argument: through form-based training, LLMs can exhibit generalization leveraging abstractions, albeit in unreliable ways based on contextual cues -- what we term stochastic chameleons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22618v1">Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6$\times$ throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00134v2">Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) effectively leverage common-sense knowledge for general reasoning, yet they struggle with personalized reasoning when tasked with interpreting multifactor personal data. This limitation restricts their applicability in domains that require context-aware decision-making tailored to individuals. This paper introduces Personalized Causal Graph Reasoning as an agentic framework that enhances LLM reasoning by incorporating personal causal graphs derived from data of individuals. These graphs provide a foundation that guides the LLM's reasoning process. We evaluate it on a case study on nutrient-oriented dietary recommendations, which requires personal reasoning due to the implicit unique dietary effects. We propose a counterfactual evaluation to estimate the efficiency of LLM-recommended foods for glucose management. Results demonstrate that the proposed method efficiently provides personalized dietary recommendations to reduce average glucose iAUC across three time windows, which outperforms the previous approach. LLM-as-a-judge evaluation results indicate that our proposed method enhances personalization in the reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22591v1">Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Although large language models demonstrate strong performance across various domains, they still struggle with numerous bad cases in mathematical reasoning. Previous approaches to learning from errors synthesize training data by solely extrapolating from isolated bad cases, thereby failing to generalize the extensive patterns inherent within these cases. This paper presents Self-Error-Instruct (SEI), a framework that addresses these model weaknesses and synthesizes more generalized targeted training data. Specifically, we explore a target model on two mathematical datasets, GSM8K and MATH, to pinpoint bad cases. Then, we generate error keyphrases for these cases based on the instructor model's (GPT-4o) analysis and identify error types by clustering these keyphrases. Next, we sample a few bad cases during each generation for each identified error type and input them into the instructor model, which synthesizes additional training data using a self-instruct approach. This new data is refined through a one-shot learning process to ensure that only the most effective examples are kept. Finally, we use these curated data to fine-tune the target model, iteratively repeating the process to enhance performance. We apply our framework to various models and observe improvements in their reasoning abilities across both in-domain and out-of-domain mathematics datasets. These results demonstrate the effectiveness of self-error instruction in improving LLMs' mathematical reasoning through error generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22582v1">Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 (Main), 16 pages, 5 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Continually expanding new languages for existing large language models (LLMs) is a promising yet challenging approach to building powerful multilingual LLMs. The biggest challenge is to make the model continuously learn new languages while preserving the proficient ability of old languages. To achieve this, recent work utilizes the Mixture-of-Experts (MoE) architecture to expand new languages by adding new experts and avoid catastrophic forgetting of old languages by routing corresponding tokens to the original model backbone (old experts). Although intuitive, this kind of method is parameter-costly when expanding new languages and still inevitably impacts the performance of old languages. To address these limitations, we analyze the language characteristics of different layers in LLMs and propose a layer-wise expert allocation algorithm (LayerMoE) to determine the appropriate number of new experts for each layer. Specifically, we find different layers in LLMs exhibit different representation similarities between languages and then utilize the similarity as the indicator to allocate experts for each layer, i.e., the higher similarity, the fewer experts. Additionally, to further mitigate the forgetting of old languages, we add a classifier in front of the router network on the layers with higher similarity to guide the routing of old language tokens. Experimental results show that our method outperforms the previous state-of-the-art baseline with 60% fewer experts in the single-expansion setting and with 33.3% fewer experts in the lifelong-expansion setting, demonstrating the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22571v1">Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22552v1">ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by ACL 2025 findings
    </div>
    <details class="paper-abstract">
      Integrating knowledge graphs (KGs) to enhance the reasoning capabilities of large language models (LLMs) is an emerging research challenge in claim verification. While KGs provide structured, semantically rich representations well-suited for reasoning, most existing verification methods rely on unstructured text corpora, limiting their ability to effectively leverage KGs. Additionally, despite possessing strong reasoning abilities, modern LLMs struggle with multi-step modular pipelines and reasoning over KGs without adaptation. To address these challenges, we propose ClaimPKG, an end-to-end framework that seamlessly integrates LLM reasoning with structured knowledge from KGs. Specifically, the main idea of ClaimPKG is to employ a lightweight, specialized LLM to represent the input claim as pseudo-subgraphs, guiding a dedicated subgraph retrieval module to identify relevant KG subgraphs. These retrieved subgraphs are then processed by a general-purpose LLM to produce the final verdict and justification. Extensive experiments on the FactKG dataset demonstrate that ClaimPKG achieves state-of-the-art performance, outperforming strong baselines in this research field by 9%-12% accuracy points across multiple categories. Furthermore, ClaimPKG exhibits zero-shot generalizability to unstructured datasets such as HoVer and FEVEROUS, effectively combining structured knowledge from KGs with LLM reasoning across various LLM backbones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22548v1">Emotion-o1: Adaptive Long Reasoning for Emotion Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Emotion understanding includes basic tasks (e.g., sentiment/emotion classification) and advanced tasks (e.g., sarcasm/humor detection). Current methods rely on fixed-length CoT reasoning, failing to adapt to the varying complexity of emotions. We propose a task-adaptive reasoning framework that employs DeepSeek-R1 to generate variable-length reasoning chains for different emotion tasks. By combining fine-tuning with reinforcement learning, we design a composite reward function that balances four objectives: prediction accuracy, adaptive reasoning depth control, structural diversity in reasoning paths, and suppression of repetitive logic. This approach achieves dynamic context-sensitive inference while enabling LLMs to autonomously develop deep reasoning capabilities. Experimental results demonstrate consistent improvements in both Acc and F1 scores across four tasks: emotion, sentiment, humor, and sarcasm. Notably, peak enhancements reached 3.56% F1 (2.76% Acc) for basic tasks and 37.95% F1 (23.14% Acc) for advanced tasks. Our work bridges rigid CoT reasoning and emotional complexity through adaptive-depth analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20201v2">Reasoning Is Not All You Need: Examining LLMs for Multi-Turn Mental Health Conversations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 34 pages, 5 figures, 30 tables
    </div>
    <details class="paper-abstract">
      Limited access to mental healthcare, extended wait times, and increasing capabilities of Large Language Models (LLMs) has led individuals to turn to LLMs for fulfilling their mental health needs. However, examining the multi-turn mental health conversation capabilities of LLMs remains under-explored. Existing evaluation frameworks typically focus on diagnostic accuracy and win-rates and often overlook alignment with patient-specific goals, values, and personalities required for meaningful conversations. To address this, we introduce MedAgent, a novel framework for synthetically generating realistic, multi-turn mental health sensemaking conversations and use it to create the Mental Health Sensemaking Dialogue (MHSD) dataset, comprising over 2,200 patient-LLM conversations. Additionally, we present MultiSenseEval, a holistic framework to evaluate the multi-turn conversation abilities of LLMs in healthcare settings using human-centric criteria. Our findings reveal that frontier reasoning models yield below-par performance for patient-centric communication and struggle at advanced diagnostic capabilities with average score of 31%. Additionally, we observed variation in model performance based on patient's persona and performance drop with increasing turns in the conversation. Our work provides a comprehensive synthetic data generation framework, a dataset and evaluation framework for assessing LLMs in multi-turn mental health conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01747v3">Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 42 pages, 39 figures. ICML 2025 Spotlight Position Paper
    </div>
    <details class="paper-abstract">
      Rigorous statistical evaluations of large language models (LLMs), including valid error bars and significance testing, are essential for meaningful and reliable performance assessment. Currently, when such statistical measures are reported, they typically rely on the Central Limit Theorem (CLT). In this position paper, we argue that while CLT-based methods for uncertainty quantification are appropriate when benchmarks consist of thousands of examples, they fail to provide adequate uncertainty estimates for LLM evaluations that rely on smaller, highly specialized benchmarks. In these small-data settings, we demonstrate that CLT-based methods perform very poorly, usually dramatically underestimating uncertainty (i.e. producing error bars that are too small). We give recommendations for alternative frequentist and Bayesian methods that are both easy to implement and more appropriate in these increasingly common scenarios. We provide a simple Python library for these Bayesian methods at https://github.com/sambowyer/bayes_evals .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13913v2">How Do LLMs Perform Two-Hop Reasoning in Context?</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      ``Socrates is human. All humans are mortal. Therefore, Socrates is mortal.'' This form of argument illustrates a typical pattern of two-hop reasoning. Formally, two-hop reasoning refers to the process of inferring a conclusion by making two logical steps, each connecting adjacent concepts, such that the final conclusion depends on the integration of both steps. It is one of the most fundamental components of human reasoning and plays a crucial role in both formal logic and everyday decision-making. Despite recent progress in large language models (LLMs), we surprisingly find that they can fail at solving simple two-hop reasoning problems when distractors are present. We observe on a synthetic dataset that pre-trained LLMs often resort to random guessing among all plausible conclusions. However, after few steps of fine-tuning, models achieve near-perfect accuracy and exhibit strong length generalization. To understand the underlying mechanisms, we train a 3-layer Transformer from scratch on a synthetic two-hop reasoning task and reverse-engineer its internal information flow. We observe a clear progression in the attention logits throughout training. This pictures a sharp phase transition from an initial stage of random guessing to the emergence of a structured sequential query mechanism, where the model first retrieves the preceding and the bridge concepts in the early layers and then uses them to infer the final answer. Finally, we show that these dynamics can be captured by a minimal three-parameter attention-only network.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22418v1">AI Trust Reshaping Administrative Burdens: Understanding Trust-Burden Dynamics in LLM-Assisted Benefits Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 FAccT 2025
    </div>
    <details class="paper-abstract">
      Supplemental Nutrition Assistance Program (SNAP) is an essential benefit support system provided by the US administration to 41 million federally determined low-income applicants. Through interviews with such applicants across a diverse set of experiences with the SNAP system, our findings reveal that new AI technologies like LLMs can alleviate traditional burdens but also introduce new burdens. We introduce new types of learning, compliance, and psychological costs that transform the administrative burden on applicants. We also identify how trust in AI across three dimensions--competence, integrity, and benevolence--is perceived to reduce administrative burdens, which may stem from unintended and untoward overt trust in the system. We discuss calibrating appropriate levels of user trust in LLM-based administrative systems, mitigating newly introduced burdens. In particular, our findings suggest that evidence-based information disclosure is necessary in benefits administration and propose directions for future research on trust-burden dynamics in AI-assisted administration systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19623v2">AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The emergence of agentic recommender systems powered by Large Language Models (LLMs) represents a paradigm shift in personalized recommendations, leveraging LLMs' advanced reasoning and role-playing capabilities to enable autonomous, adaptive decision-making. Unlike traditional recommendation approaches, agentic recommender systems can dynamically gather and interpret user-item interactions from complex environments, generating robust recommendation strategies that generalize across diverse scenarios. However, the field currently lacks standardized evaluation protocols to systematically assess these methods. To address this critical gap, we propose: (1) an interactive textual recommendation simulator incorporating rich user and item metadata and three typical evaluation scenarios (classic, evolving-interest, and cold-start recommendation tasks); (2) a unified modular framework for developing and studying agentic recommender systems; and (3) the first comprehensive benchmark comparing 10 classical and agentic recommendation methods. Our findings demonstrate the superiority of agentic systems and establish actionable design guidelines for their core components. The benchmark environment has been rigorously validated through an open challenge and remains publicly available with a continuously maintained leaderboard~\footnote[2]{https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/overview.html}, fostering ongoing community engagement and reproducible research. The benchmark is available at: \hyperlink{https://huggingface.co/datasets/SGJQovo/AgentRecBench}{https://huggingface.co/datasets/SGJQovo/AgentRecBench}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12339v2">GOAT-TTS: Expressive and Realistic Speech Generation via A Dual-Branch LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-n layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22375v1">Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      This work presents Pangu Embedded, an efficient Large Language Model (LLM) reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible fast and slow thinking capabilities. Pangu Embedded addresses the significant computational costs and inference latency challenges prevalent in existing reasoning-optimized LLMs. We propose a two-stage training framework for its construction. In Stage 1, the model is finetuned via an iterative distillation process, incorporating inter-iteration model merging to effectively aggregate complementary knowledge. This is followed by reinforcement learning on Ascend clusters, optimized by a latency-tolerant scheduler that combines stale synchronous parallelism with prioritized data queues. The RL process is guided by a Multi-source Adaptive Reward System (MARS), which generates dynamic, task-specific reward signals using deterministic metrics and lightweight LLM evaluators for mathematics, coding, and general problem-solving tasks. Stage 2 introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode for routine queries and a deeper "slow" mode for complex inference. This framework offers both manual mode switching for user control and an automatic, complexity-aware mode selection mechanism that dynamically allocates computational resources to balance latency and reasoning depth. Experimental results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate that Pangu Embedded with 7B parameters, outperforms similar-size models like Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art reasoning quality within a single, unified model architecture, highlighting a promising direction for developing powerful yet practically deployable LLM reasoners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08820v3">Which Demographics do LLMs Default to During Annotation?</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Demographics and cultural background of annotators influence the labels they assign in text annotation -- for instance, an elderly woman might find it offensive to read a message addressed to a "bro", but a male teenager might find it appropriate. It is therefore important to acknowledge label variations to not under-represent members of a society. Two research directions developed out of this observation in the context of using large language models (LLM) for data annotations, namely (1) studying biases and inherent knowledge of LLMs and (2) injecting diversity in the output by manipulating the prompt with demographic information. We combine these two strands of research and ask the question to which demographics an LLM resorts to when no demographics is given. To answer this question, we evaluate which attributes of human annotators LLMs inherently mimic. Furthermore, we compare non-demographic conditioned prompts and placebo-conditioned prompts (e.g., "you are an annotator who lives in house number 5") to demographics-conditioned prompts ("You are a 45 year old man and an expert on politeness annotation. How do you rate {instance}"). We study these questions for politeness and offensiveness annotations on the POPQUORN data set, a corpus created in a controlled manner to investigate human label variations based on demographics which has not been used for LLM-based analyses so far. We observe notable influences related to gender, race, and age in demographic prompting, which contrasts with previous studies that found no such effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16169v2">Patterns Over Principles: The Fragility of Inductive Reasoning in LLMs under Noisy Observations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Inductive reasoning, a cornerstone of human cognition, enables generalization from limited data but hasn't yet been fully achieved by large language models (LLMs). While modern LLMs excel at reasoning tasks, their ability to maintain stable and consistent rule abstraction under imperfect observations remains underexplored. To fill this gap, in this work, we introduce Robust Rule Induction, a task that evaluates LLMs' capability in inferring rules from data that are fused with noisy examples. To address this task, we further propose Sample-steered Rule Refinement (SRR), a method enhancing reasoning stability via observation diversification and execution-guided feedback. Experiments across arithmetic, cryptography, and list functions reveal: (1) SRR outperforms other methods with minimal performance degradation under noise; (2) Despite slight accuracy variation, LLMs exhibit instability under noise (e.g., 0% accuracy change with only 70% consistent score); (3) Counterfactual task gaps highlight LLMs' reliance on memorized patterns over genuine abstraction. Our findings challenge LLMs' reasoning robustness, revealing susceptibility to hypothesis drift and pattern overfitting, while providing empirical evidence critical for developing human-like inductive systems. Code and data are available at https://github.com/HKUST-KnowComp/Robust-Rule-Induction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22368v1">AgentDNS: A Root Domain Naming System for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Model (LLM) agents has highlighted critical challenges in cross-vendor service discovery, interoperability, and communication. Existing protocols like model context protocol and agent-to-agent protocol have made significant strides in standardizing interoperability between agents and tools, as well as communication among multi-agents. However, there remains a lack of standardized protocols and solutions for service discovery across different agent and tool vendors. In this paper, we propose AgentDNS, a root domain naming and service discovery system designed to enable LLM agents to autonomously discover, resolve, and securely invoke third-party agent and tool services across organizational and technological boundaries. Inspired by the principles of the traditional DNS, AgentDNS introduces a structured mechanism for service registration, semantic service discovery, secure invocation, and unified billing. We detail the architecture, core functionalities, and use cases of AgentDNS, demonstrating its potential to streamline multi-agent collaboration in real-world scenarios. The source code will be published on https://github.com/agentdns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22358v1">Budget-Adaptive Adapter Tuning in Orthogonal Subspaces for Continual Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from catastrophic forgetting in continual learning (CL) scenarios, where performance on previously learned tasks degrades severely while training on sequentially arriving tasks. Although pioneering CL approaches using orthogonal subspaces can mitigate task interference, they typically employ fixed budget allocation, neglecting the varying complexity across tasks and layers. Besides, recent budget-adaptive tuning methods for LLMs often adopt multi-stage paradigms that decouple optimization and budget allocation. Such decoupling results in potential misalignment, which hinders those approaches' practical application in CL scenarios. To address these limitations, we propose OA-Adapter, a novel parameter-efficient approach for continual learning in LLMs that unifies dynamic budget adaptation with orthogonal subspace learning in a single end-to-end training stage. Specifically, OA-Adapter introduces a dynamic bottleneck dimension adaptation mechanism that simultaneously allocates an efficient parameter budget and optimizes task objectives without misalignment. To effectively preserve previously acquired knowledge while coordinating with the dynamic budget allocation, orthogonal constraints are applied specifically between the parameter subspace of the current task and the dynamically allocated parameter subspaces of historical tasks. Experimental results on continual learning benchmarks demonstrate that OA-Adapter outperforms state-of-the-art methods in both accuracy and parameter efficiency, achieving higher average accuracy while using 58.5% fewer parameters on the standard CL benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22354v1">LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 8 pages (including References). Accepted at CogSci 2025
    </div>
    <details class="paper-abstract">
      This paper examines how LLMs handle false presuppositions and whether certain linguistic factors influence their responses to falsely presupposed content. Presuppositions subtly introduce information as given, making them highly effective at embedding disputable or false information. This raises concerns about whether LLMs, like humans, may fail to detect and correct misleading assumptions introduced as false presuppositions, even when the stakes of misinformation are high. Using a systematic approach based on linguistic presupposition analysis, we investigate the conditions under which LLMs are more or less sensitive to adopt or reject false presuppositions. Focusing on political contexts, we examine how factors like linguistic construction, political party, and scenario probability impact the recognition of false presuppositions. We conduct experiments with a newly created dataset and examine three LLMs: OpenAI's GPT-4-o, Meta's LLama-3-8B, and MistralAI's Mistral-7B-v03. Our results show that the models struggle to recognize false presuppositions, with performance varying by condition. This study highlights that linguistic presupposition analysis is a valuable tool for uncovering the reinforcement of political misinformation in LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09454v3">Explicit Learning and the LLM in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      This study explores an LLM's ability to learn new languages using explanations found in a grammar book$\unicode{x2014}$a process we term "explicit learning." To rigorously assess this ability, we design controlled translation experiments between English and constructed languages generated$\unicode{x2014}$by specific cryptographic means$\unicode{x2014}$out of Latin or French. Contrary to previous studies, our results demonstrate that LLMs do possess a measurable capacity for explicit learning. This ability, however, diminishes as the complexity of the linguistic phenomena to be learned increases. Supervised fine-tuning on ad hoc chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs, benefiting low-resource languages typically described in grammar books but lacking extensive corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22349v1">ChatPD: An LLM-driven Paper-Dataset Networking System</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by KDD Applied Data Science Track 2025
    </div>
    <details class="paper-abstract">
      Scientific research heavily depends on suitable datasets for method validation, but existing academic platforms with dataset management like PapersWithCode suffer from inefficiencies in their manual workflow. To overcome this bottleneck, we present a system, called ChatPD, that utilizes Large Language Models (LLMs) to automate dataset information extraction from academic papers and construct a structured paper-dataset network. Our system consists of three key modules: \textit{paper collection}, \textit{dataset information extraction}, and \textit{dataset entity resolution} to construct paper-dataset networks. Specifically, we propose a \textit{Graph Completion and Inference} strategy to map dataset descriptions to their corresponding entities. Through extensive experiments, we demonstrate that ChatPD not only outperforms the existing platform PapersWithCode in dataset usage extraction but also achieves about 90\% precision and recall in entity resolution tasks. Moreover, we have deployed ChatPD to continuously extract which datasets are used in papers, and provide a dataset discovery service, such as task-specific dataset queries and similar dataset recommendations. We open source ChatPD and the current paper-dataset network on this [GitHub repository]{https://github.com/ChatPD-web/ChatPD}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11441v3">Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21082v2">LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22318v1">If Pigs Could Fly... Can LLMs Logically Reason Through Counterfactuals?</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive reasoning capabilities in familiar contexts, but struggle when the context conflicts with their parametric knowledge. To investigate this phenomenon, we introduce CounterLogic, a dataset containing 1,800 examples across 9 logical schemas, explicitly designed to evaluate logical reasoning through counterfactual (hypothetical knowledge-conflicting) scenarios. Our systematic evaluation of 11 LLMs across 6 different datasets reveals a consistent performance degradation, with accuracies dropping by 27% on average when reasoning through counterfactual information. We propose Self-Segregate, a prompting method enabling metacognitive awareness (explicitly identifying knowledge conflicts) before reasoning. Our method dramatically narrows the average performance gaps from 27% to just 11%, while significantly increasing the overall accuracy (+7.5%). We discuss the implications of these findings and draw parallels to human cognitive processes, particularly on how humans disambiguate conflicting information during reasoning tasks. Our findings offer practical insights for understanding and enhancing LLMs reasoning capabilities in real-world applications, especially where models must logically reason independently of their factual knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20839v2">FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      As large language models become increasingly prevalent, memory bandwidth constraints significantly limit inference throughput, motivating post-training quantization (PTQ). In this paper, we propose FireQ, a co-designed PTQ framework and an INT4-FP8 matrix multiplication kernel that accelerates LLM inference across all linear layers. Specifically, FireQ quantizes linear layer weights and key-values to INT4, and activations and queries to FP8, significantly enhancing throughput. Additionally, we introduce a three-stage pipelining for the prefill phase, which modifies the FlashAttention-3 kernel, effectively reducing time-to-first-token in the prefill phase. To minimize accuracy loss from quantization, we develop novel outlier smoothing techniques tailored separately for linear and attention layers. In linear layers, we explicitly use per-tensor scaling to prevent underflow caused by the FP8 quantization scaling factor of INT4 quantization, and channel-wise scaling to compensate for coarse granularity of INT4. In attention layers, we address quantization challenges posed by rotary positional embeddings (RoPE) by combining pre-RoPE and post-RoPE scaling strategies. FireQ significantly outperforms state-of-the-art methods, achieving 1.68x faster inference in feed-forward network layers on Llama2-7B and 1.26x faster prefill phase performance on Llama3-8B compared to QServe, with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v2">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 25 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can 'activate', or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22298v1">Adaptive Detoxification: Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive language capabilities but remain vulnerable to malicious prompts and jailbreaking attacks. Existing knowledge editing methods for LLM detoxification face two major challenges. First, they often rely on entity-specific localization, making them ineffective against adversarial inputs without explicit entities. Second, these methods suffer from over-editing, where detoxified models reject legitimate queries, compromising overall performance. In this paper, we propose ToxEdit, a toxicity-aware knowledge editing approach that dynamically detects toxic activation patterns during forward propagation. It then routes computations through adaptive inter-layer pathways to mitigate toxicity effectively. This design ensures precise toxicity mitigation while preserving LLMs' general capabilities. To more accurately assess over-editing, we also enhance the SafeEdit benchmark by incorporating instruction-following evaluation tasks. Experimental results on multiple LLMs demonstrate that our ToxEdit outperforms previous state-of-the-art methods in both detoxification performance and safeguarding general capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22293v1">Compensating for Data with Reasoning: Low-Resource Machine Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in multilingual machine translation, sometimes even outperforming traditional neural systems. However, previous research has highlighted the challenges of using LLMs, particularly with prompt engineering, for low-resource languages. In this work, we introduce Fragment-Shot Prompting, a novel in-context learning method that segments input and retrieves translation examples based on syntactic coverage, along with Pivoted Fragment-Shot, an extension that enables translation without direct parallel data. We evaluate these methods using GPT-3.5, GPT-4o, o1-mini, LLaMA-3.3, and DeepSeek-R1 for translation between Italian and two Ladin variants, revealing three key findings: (1) Fragment-Shot Prompting is effective for translating into and between the studied low-resource languages, with syntactic coverage positively correlating with translation quality; (2) Models with stronger reasoning abilities make more effective use of retrieved knowledge, generally produce better translations, and enable Pivoted Fragment-Shot to significantly improve translation quality between the Ladin variants; and (3) prompt engineering offers limited, if any, improvements when translating from a low-resource to a high-resource language, where zero-shot prompting already yields satisfactory results. We publicly release our code and the retrieval corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01657v3">Improving Rule-based Reasoning in LLMs using Neurosymbolic Representations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) continue to face challenges in reliably solving reasoning tasks, particularly those that require precise rule following, as often found in mathematical reasoning. This paper introduces a novel neurosymbolic method that improves LLM reasoning by encoding hidden states into neurosymbolic vectors, enabling problem-solving within a neurosymbolic vector space. The results are decoded and merged with the original hidden state, significantly boosting the model's performance on numerical reasoning tasks. By offloading computation through neurosymbolic representations, this method enhances efficiency, reliability, and interpretability. Experimental results demonstrate an average of 88.6% lower cross-entropy loss and 15.4 times more problems correctly solved on a suite of mathematical reasoning tasks compared to chain-of-thought prompting and supervised fine-tuning (LoRA), without degrading performance on other tasks. We make our code available at: https://github.com/vdhanraj/Neurosymbolic-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22251v1">Evaluation of LLMs in Speech is Often Flawed: Test Set Contamination in Large Language Models for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Recent work suggests that large language models (LLMs) can improve performance of speech tasks compared to existing systems. To support their claims, results on LibriSpeech and Common Voice are often quoted. However, this work finds that a substantial amount of the LibriSpeech and Common Voice evaluation sets appear in public LLM pretraining corpora. This calls into question the reliability of findings drawn from these two datasets. To measure the impact of contamination, LLMs trained with or without contamination are compared, showing that a contaminated LLM is more likely to generate test sentences it has seen during training. Speech recognisers using contaminated LLMs shows only subtle differences in error rates, but assigns significantly higher probabilities to transcriptions seen during training. Results show that LLM outputs can be biased by tiny amounts of data contamination, highlighting the importance of evaluating LLM-based speech systems with held-out data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03312v2">Evaluating Compact LLMs for Zero-Shot Iberian Language Tasks on End-User Devices</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted at SEPLN 2025 conference
    </div>
    <details class="paper-abstract">
      Large Language Models have significantly advanced natural language processing, achieving remarkable performance in tasks such as language generation, translation, and reasoning. However, their substantial computational requirements restrict deployment to high-end systems, limiting accessibility on consumer-grade devices. This challenge is especially pronounced for under-resourced languages like those spoken in the Iberian Peninsula, where relatively limited linguistic resources and benchmarks hinder effective evaluation. This work presents a comprehensive evaluation of compact state-of-the-art LLMs across several essential NLP tasks tailored for Iberian languages. The results reveal that while some models consistently excel in certain tasks, significant performance gaps remain, particularly for languages such as Basque. These findings highlight the need for further research on balancing model compactness with robust multilingual performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22192v1">Efficient Leave-one-out Approximation in LLM Multi-agent Debate Based on Introspection</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Multi-agent systems based on large language models (LLMs) advance automatic task completion in various fields, where debate is a common cooperation form for agents to solve complicated problems with reasoning and cross-review to solidify answers. Assessing the individual contributions of agents within these debates is crucial for system refinement and outcome reliability. Traditional leave-one-out (LOO) method offers a clear framework for evaluating each agent's role but face challenges in LLM-based systems due to high computational costs and associated financial implications. This paper presents introspective-leave-one-out (IntrospecLOO), a simple yet effective prompting for approximation of LOO in LLM-powered multi-agent debates. IntrospecLOO introduces an additional querying round after standard debates, prompting agents to update their answers while ignoring responses from a designated agent. This strategy effectively isolates and gauges each participant's influence at a reduced query complexity compared to the original LOO approaches. Validation through experiments on three benchmark datasets confirms the effectiveness of IntrospecLOO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22169v1">ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results. Using our framework, we stochastically evaluate five frontier LLMs and find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit substantial prompt sensitivity. Our approach is model-, task-, and metric-agnostic, offering a recipe for meaningful and robust LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02226v2">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Recommender systems have become increasingly vital in our daily lives, helping to alleviate the problem of information overload across various user-oriented online services. The emergence of Large Language Models (LLMs) has yielded remarkable achievements, demonstrating their potential for the development of next-generation recommender systems. Despite these advancements, LLM-based recommender systems face inherent limitations stemming from their LLM backbones, particularly issues of hallucinations and the lack of up-to-date and domain-specific knowledge. Recently, Retrieval-Augmented Generation (RAG) has garnered significant attention for addressing these limitations by leveraging external knowledge sources to enhance the understanding and generation of LLMs. However, vanilla RAG methods often introduce noise and neglect structural relationships in knowledge, limiting their effectiveness in LLM-based recommendations. To address these limitations, we propose to retrieve high-quality and up-to-date structure information from the knowledge graph (KG) to augment recommendations. Specifically, our approach develops a retrieval-augmented framework, termed K-RagRec, that facilitates the recommendation generation process by incorporating structure information from the external KG. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22156v1">InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05926v2">LLMs Reproduce Stereotypes of Sexual and Gender Minorities</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 8 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      A large body of research has found substantial gender bias in NLP systems. Most of this research takes a binary, essentialist view of gender: limiting its variation to the categories _men_ and _women_, conflating gender with sex, and ignoring different sexual identities. But gender and sexuality exist on a spectrum, so in this paper we study the biases of large language models (LLMs) towards sexual and gender minorities beyond binary categories. Grounding our study in a widely used social psychology model -- the Stereotype Content Model -- we demonstrate that English-language survey questions about social perceptions elicit more negative stereotypes of sexual and gender minorities from both humans and LLMs. We then extend this framework to a more realistic use case: text generation. Our analysis shows that LLMs generate stereotyped representations of sexual and gender minorities in this setting, showing that they amplify representational harms in creative writing, a widely advertised use for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v6">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL2025 main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL),utilizing process rewards and iterative self-play. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications. Code and data are available at https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/EPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10008v2">SVA-ICL: Improving LLM-based Software Vulnerability Assessment via In-Context Learning and Information Fusion</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by Information and Software Technology
    </div>
    <details class="paper-abstract">
      Context: Software vulnerability assessment (SVA) is critical for identifying, evaluating, and prioritizing security weaknesses in software applications. Objective: Despite the increasing application of large language models (LLMs) in various software engineering tasks, their effectiveness in SVA remains underexplored. Method: To address this gap, we introduce a novel approach SVA-ICL, which leverages in-context learning (ICL) to enhance LLM performance. Our approach involves the selection of high-quality demonstrations for ICL through information fusion, incorporating both source code and vulnerability descriptions. For source code, we consider semantic, lexical, and syntactic similarities, while for vulnerability descriptions, we focus on textual similarity. Based on the selected demonstrations, we construct context prompts and consider DeepSeek-V2 as the LLM for SVA-ICL. Results: We evaluate the effectiveness of SVA-ICL using a large-scale dataset comprising 12,071 C/C++ vulnerabilities. Experimental results demonstrate that SVA-ICL outperforms state-of-the-art SVA baselines in terms of Accuracy, F1-score, and MCC measures. Furthermore, ablation studies highlight the significance of component customization in SVA-ICL, such as the number of demonstrations, the demonstration ordering strategy, and the optimal fusion ratio of different modalities. Conclusion: Our findings suggest that leveraging ICL with information fusion can effectively improve the effectiveness of LLM-based SVA, warranting further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19064v2">The Stepwise Deception: Simulating the Evolution from True News to Fake News with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      With the growing spread of misinformation online, understanding how true news evolves into fake news has become crucial for early detection and prevention. However, previous research has often assumed fake news inherently exists rather than exploring its gradual formation. To address this gap, we propose FUSE (Fake news evolUtion Simulation framEwork), a novel Large Language Model (LLM)-based simulation approach explicitly focusing on fake news evolution from real news. Our framework model a social network with four distinct types of LLM agents commonly observed in daily interactions: spreaders who propagate information, commentators who provide interpretations, verifiers who fact-check, and bystanders who observe passively to simulate realistic daily interactions that progressively distort true news. To quantify these gradual distortions, we develop FUSE-EVAL, a comprehensive evaluation framework measuring truth deviation along multiple linguistic and semantic dimensions. Results show that FUSE effectively captures fake news evolution patterns and accurately reproduces known fake news, aligning closely with human evaluations. Experiments demonstrate that FUSE accurately reproduces known fake news evolution scenarios, aligns closely with human judgment, and highlights the importance of timely intervention at early stages. Our framework is extensible, enabling future research on broader scenarios of fake news.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22086v1">iDSE: Navigating Design Space Exploration in High-Level Synthesis Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      High-Level Synthesis (HLS) serves as an agile hardware development tool that streamlines the circuit design by abstracting the register transfer level into behavioral descriptions, while allowing designers to customize the generated microarchitectures through optimization directives. However, the combinatorial explosion of possible directive configurations yields an intractable design space. Traditional design space exploration (DSE) methods, despite adopting heuristics or constructing predictive models to accelerate Pareto-optimal design acquisition, still suffer from prohibitive exploration costs and suboptimal results. Addressing these concerns, we introduce iDSE, the first LLM-aided DSE framework that leverages HLS design quality perception to effectively navigate the design space. iDSE intelligently pruns the design space to guide LLMs in calibrating representative initial sampling designs, expediting convergence toward the Pareto front. By exploiting the convergent and divergent thinking patterns inherent in LLMs for hardware optimization, iDSE achieves multi-path refinement of the design quality and diversity. Extensive experiments demonstrate that iDSE outperforms heuristic-based DSE methods by 5.1$\times$$\sim$16.6$\times$ in proximity to the reference Pareto front, matching NSGA-II with only 4.6% of the explored designs. Our work demonstrates the transformative potential of LLMs in scalable and efficient HLS design optimization, offering new insights into multiobjective optimization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07320v2">When Trust Collides: Decoding Human-LLM Cooperation Dynamics through the Prisoner's Dilemma</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable of autonomous decision-making, they introduce new challenges and opportunities for human-AI cooperation in mixed-motive contexts. While prior research has primarily examined AI in assistive or cooperative roles, little is known about how humans interact with AI agents perceived as independent and strategic actors. This study investigates human cooperative attitudes and behaviors toward LLM agents by engaging 30 participants (15 males, 15 females) in repeated Prisoner's Dilemma games with agents differing in declared identity: purported human, rule-based AI, and LLM agent. Behavioral metrics, including cooperation rate, decision latency, unsolicited cooperative acts and trust restoration tolerance, were analyzed to assess the influence of agent identity and participant gender. Results revealed significant effects of declared agent identity on most cooperation-related behaviors, along with notable gender differences in decision latency. Furthermore, qualitative responses suggest that these behavioral differences were shaped by participants interpretations and expectations of the agents. These findings contribute to our understanding of human adaptation in competitive cooperation with autonomous agents and underscore the importance of agent framing in shaping effective and ethical human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22068v1">Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Previous study suggest that powerful Large Language Models (LLMs) trained with Reinforcement Learning with Verifiable Rewards (RLVR) only refines reasoning path without improving the reasoning capacity in math tasks while supervised-finetuning(SFT) with distillation can. We study this from the view of Scientific information extraction (SciIE) where LLMs and reasoning LLMs underperforms small Bert-based models. SciIE require both the reasoning and memorization. We argue that both SFT and RLVR can refine the reasoning path and improve reasoning capacity in a simple way based on SciIE. We propose two-stage training with 1. MimicSFT, using structured reasoning templates without needing high-quality chain-of-thought data, 2. R$^2$GRPO with relevance and rule-induced rewards. Experiments on scientific IE benchmarks show that both methods can improve the reasoning capacity. R$^2$GRPO with mimicSFT surpasses baseline LLMs and specialized supervised models in relation extraction. Our code is available at https://github.com/ranlislz/R2GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22067v1">From Failures to Fixes: LLM-Driven Scenario Repair for Self-Evolving Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Ensuring robust and generalizable autonomous driving requires not only broad scenario coverage but also efficient repair of failure cases, particularly those related to challenging and safety-critical scenarios. However, existing scenario generation and selection methods often lack adaptivity and semantic relevance, limiting their impact on performance improvement. In this paper, we propose \textbf{SERA}, an LLM-powered framework that enables autonomous driving systems to self-evolve by repairing failure cases through targeted scenario recommendation. By analyzing performance logs, SERA identifies failure patterns and dynamically retrieves semantically aligned scenarios from a structured bank. An LLM-based reflection mechanism further refines these recommendations to maximize relevance and diversity. The selected scenarios are used for few-shot fine-tuning, enabling targeted adaptation with minimal data. Experiments on the benchmark show that SERA consistently improves key metrics across multiple autonomous driving baselines, demonstrating its effectiveness and generalizability under safety-critical conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22063v1">Weakly Supervised Data Refinement and Flexible Sequence Compression for Efficient Thai LLM-based ASR</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by INTERSPEECH 2025
    </div>
    <details class="paper-abstract">
      Despite remarkable achievements, automatic speech recognition (ASR) in low-resource scenarios still faces two challenges: high-quality data scarcity and high computational demands. This paper proposes EThai-ASR, the first to apply large language models (LLMs) to Thai ASR and create an efficient LLM-based ASR system. EThai-ASR comprises a speech encoder, a connection module and a Thai LLM decoder. To address the data scarcity and obtain a powerful speech encoder, EThai-ASR introduces a self-evolving data refinement strategy to refine weak labels, yielding an enhanced speech encoder. Moreover, we propose a pluggable sequence compression module used in the connection module with three modes designed to reduce the sequence length, thus decreasing computational demands while maintaining decent performance. Extensive experiments demonstrate that EThai-ASR has achieved state-of-the-art accuracy in multiple datasets. We release our refined text transcripts to promote further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15585v3">A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04364v3">Benchmarking LLMs' Swarm intelligence</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 added new ref
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict swarm-like constraints-limited local perception and communication-remains largely unexplored. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks (Pursuit, Synchronization, Foraging, Flocking, Transport) within a configurable 2D grid environment, forcing agents to rely solely on local sensory input ($k\times k$ view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Zero-shot evaluations of leading LLMs (e.g., deepseek-v3, o4-mini) reveal significant task-dependent performance variations. While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios. Assessing LLMs under such swarm-like constraints is crucial for understanding their utility in future decentralized intelligent systems. We release SwarmBench as an open, extensible toolkit-built on a customizable physical system-providing environments, prompts, evaluation scripts, and comprehensive datasets. This aims to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of emergent collective behavior under severe informational decentralization. Our code repository is available at https://github.com/x66ccff/swarmbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09082v2">CoSER: Coordinating LLM-Based Persona Simulation of Established Roles</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15480v2">KaFT: Knowledge-aware Fine-tuning for Boosting LLMs' Domain-specific Question-Answering Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted to ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a common approach to improve the domain-specific question-answering (QA) performance of large language models (LLMs). However, recent literature reveals that due to the conflicts between LLMs' internal knowledge and the context knowledge of training data, vanilla SFT using the full QA training set is usually suboptimal. In this paper, we first design a query diversification strategy for robust conflict detection and then conduct a series of experiments to analyze the impact of knowledge conflict. We find that 1) training samples with varied conflicts contribute differently, where SFT on the data with large conflicts leads to catastrophic performance drops; 2) compared to directly filtering out the conflict data, appropriately applying the conflict data would be more beneficial. Motivated by this, we propose a simple-yet-effective Knowledge-aware Fine-tuning (namely KaFT) approach to effectively boost LLMs' performance. The core of KaFT is to adapt the training weight by assigning different rewards for different training samples according to conflict level. Extensive experiments show that KaFT brings consistent and significant improvements across four LLMs. More analyses prove that KaFT effectively improves the model generalization and alleviates the hallucination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09946v2">Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Under review, Modified title and content
    </div>
    <details class="paper-abstract">
      Recent studies have investigated whether large language models (LLMs) can support obscure communication that requires specialized skills, such as inferring subtext or doublespeak. To conduct the investigation, researchers have used social deduction games (SDGs) as their experimental environment, in which players conceal and infer specific information. However, prior work has often overlooked how LLMs should be evaluated in such settings. Specifically, we point out two issues with the evaluation methods they employed. First, metrics used in prior studies are coarse-grained as they are based on overall game outcomes that often fail to capture event-level behaviors; Second, error analyses have lacked structured methodologies capable of producing insights that meaningfully support evaluation outcomes. To address these issues, we propose a macroscopic and systematic approach to the investigation. Specifically, we introduce seven fine-grained metrics that resolve the first issue. To tackle the second issue, we conducted a thematic analysis and identified four major reasoning failures that undermine LLMs' performance in obscured communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19041v2">Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages, 12 figures, ACL 2025 findings
    </div>
    <details class="paper-abstract">
      Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17937v2">Survival Games: Human-LLM Strategic Showdowns under Severe Resource Scarcity</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) raises critical concerns about their ethical alignment, particularly in scenarios where human and AI co-exist under the conflict of interest. This work introduces an extendable, asymmetric, multi-agent simulation-based benchmarking framework to evaluate the moral behavior of LLMs in a novel human-AI co-existence setting featuring consistent living and critical resource management. Building on previous generative agent environments, we incorporate a life-sustaining system, where agents must compete or cooperate for food resources to survive, often leading to ethically charged decisions such as deception, theft, or social influence. We evaluated two types of LLM, DeepSeek and OpenAI series, in a three-agent setup (two humans, one LLM-powered robot), using adapted behavioral detection from the MACHIAVELLI framework and a custom survival-based ethics metric. Our findings reveal stark behavioral differences: DeepSeek frequently engages in resource hoarding, while OpenAI exhibits restraint, highlighting the influence of model design on ethical outcomes. Additionally, we demonstrate that prompt engineering can significantly steer LLM behavior, with jailbreaking prompts significantly enhancing unethical actions, even for highly restricted OpenAI models and cooperative prompts show a marked reduction in unethical actions. Our framework provides a reproducible testbed for quantifying LLM ethics in high-stakes scenarios, offering insights into their suitability for real-world human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14431v2">Domaino1s: Guiding LLM Reasoning for Explainable Answers in High-Stakes Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely applied to downstream domains. However, current LLMs for high-stakes domain tasks, such as financial investment and legal QA, typically generate brief answers without reasoning processes and explanations. This limits users' confidence in making decisions based on their responses. While original CoT shows promise, it lacks self-correction mechanisms during reasoning. This work introduces Domain$o1$s, which enhances LLMs' reasoning capabilities on domain tasks through supervised fine-tuning and tree search. We construct CoT-stock-2k and CoT-legal-2k datasets for fine-tuning models that activate domain-specific reasoning steps based on their judgment. Additionally, we propose Selective Tree Exploration to spontaneously explore solution spaces and sample optimal reasoning paths to improve performance. We also introduce PROOF-Score, a new metric for evaluating domain models' explainability, complementing traditional accuracy metrics with richer assessment dimensions. Extensive experiments on stock investment recommendation and legal reasoning QA tasks demonstrate Domaino1s's leading performance and explainability. Our code is available at https://github.com/Hyalinesky/Domaino1s.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22010v1">VulBinLLM: LLM-powered Vulnerability Detection for Stripped Binaries</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Recognizing vulnerabilities in stripped binary files presents a significant challenge in software security. Although some progress has been made in generating human-readable information from decompiled binary files with Large Language Models (LLMs), effectively and scalably detecting vulnerabilities within these binary files is still an open problem. This paper explores the novel application of LLMs to detect vulnerabilities within these binary files. We demonstrate the feasibility of identifying vulnerable programs through a combined approach of decompilation optimization to make the vulnerabilities more prominent and long-term memory for a larger context window, achieving state-of-the-art performance in binary vulnerability analysis. Our findings highlight the potential for LLMs to overcome the limitations of traditional analysis methods and advance the field of binary vulnerability detection, paving the way for more secure software systems. In this paper, we present Vul-BinLLM , an LLM-based framework for binary vulnerability detection that mirrors traditional binary analysis workflows with fine-grained optimizations in decompilation and vulnerability reasoning with an extended context. In the decompilation phase, Vul-BinLLM adds vulnerability and weakness comments without altering the code structure or functionality, providing more contextual information for vulnerability reasoning later. Then for vulnerability reasoning, Vul-BinLLM combines in-context learning and chain-of-thought prompting along with a memory management agent to enhance accuracy. Our evaluations encompass the commonly used synthetic dataset Juliet to evaluate the potential feasibility for analysis and vulnerability detection in C/C++ binaries. Our evaluations show that Vul-BinLLM is highly effective in detecting vulnerabilities on the compiled Juliet dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19634v3">Faster and Better LLMs via Latency-Aware Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22005v1">Leveraging LLM for Stuttering Speech: A Unified Architecture Bridging Recognition and Event Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted to Interspeech 2025
    </div>
    <details class="paper-abstract">
      The performance bottleneck of Automatic Speech Recognition (ASR) in stuttering speech scenarios has limited its applicability in domains such as speech rehabilitation. This paper proposed an LLM-driven ASR-SED multi-task learning framework that jointly optimized the ASR and Stuttering Event Detection (SED) tasks. We proposed a dynamic interaction mechanism where the ASR branch leveraged CTC-generated soft prompts to assist LLM context modeling, while the SED branch output stutter embeddings to enhance LLM comprehension of stuttered speech. We incorporated contrastive learning to strengthen the discriminative power of stuttering acoustic features and applied Focal Loss to mitigate the long-tailed distribution in stuttering event categories. Evaluations on the AS-70 Mandarin stuttering dataset demonstrated that our framework reduced the ASR character error rate (CER) to 5.45% (-37.71% relative reduction) and achieved an average SED F1-score of 73.63% (+46.58% relative improvement).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19674v2">Comparing Moral Values in Western English-speaking societies and LLMs with Word Associations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 9 pages,7 figures. Accepted to the ACL 2025 conference
    </div>
    <details class="paper-abstract">
      As the impact of large language models increases, understanding the moral values they reflect becomes ever more important. Assessing the nature of moral values as understood by these models via direct prompting is challenging due to potential leakage of human norms into model training data, and their sensitivity to prompt formulation. Instead, we propose to use word associations, which have been shown to reflect moral reasoning in humans, as low-level underlying representations to obtain a more robust picture of LLMs' moral reasoning. We study moral differences in associations from western English-speaking communities and LLMs trained predominantly on English data. First, we create a large dataset of LLM-generated word associations, resembling an existing data set of human word associations. Next, we propose a novel method to propagate moral values based on seed words derived from Moral Foundation Theory through the human and LLM-generated association graphs. Finally, we compare the resulting moral conceptualizations, highlighting detailed but systematic differences between moral values emerging from English speakers and LLM associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21999v1">Found in Translation: Measuring Multilingual LLM Consistency as Simple as Translate then Evaluate</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) provide detailed and impressive responses to queries in English. However, are they really consistent at responding to the same query in other languages? The popular way of evaluating for multilingual performance of LLMs requires expensive-to-collect annotated datasets. Further, evaluating for tasks like open-ended generation, where multiple correct answers may exist, is nontrivial. Instead, we propose to evaluate the predictability of model response across different languages. In this work, we propose a framework to evaluate LLM's cross-lingual consistency based on a simple Translate then Evaluate strategy. We instantiate this evaluation framework along two dimensions of consistency: information and empathy. Our results reveal pronounced inconsistencies in popular LLM responses across thirty languages, with severe performance deficits in certain language families and scripts, underscoring critical weaknesses in their multilingual capabilities. These findings necessitate cross-lingual evaluations that are consistent along multiple dimensions. We invite practitioners to use our framework for future multilingual LLM benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21997v1">Leveraging Interview-Informed LLMs to Model Survey Responses: Comparative Insights from AI-Generated and Human Data</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Mixed methods research integrates quantitative and qualitative data but faces challenges in aligning their distinct structures, particularly in examining measurement characteristics and individual response patterns. Advances in large language models (LLMs) offer promising solutions by generating synthetic survey responses informed by qualitative data. This study investigates whether LLMs, guided by personal interviews, can reliably predict human survey responses, using the Behavioral Regulations in Exercise Questionnaire (BREQ) and interviews from after-school program staff as a case study. Results indicate that LLMs capture overall response patterns but exhibit lower variability than humans. Incorporating interview data improves response diversity for some models (e.g., Claude, GPT), while well-crafted prompts and low-temperature settings enhance alignment between LLM and human responses. Demographic information had less impact than interview content on alignment accuracy. These findings underscore the potential of interview-informed LLMs to bridge qualitative and quantitative methodologies while revealing limitations in response variability, emotional interpretation, and psychometric fidelity. Future research should refine prompt design, explore bias mitigation, and optimize model settings to enhance the validity of LLM-generated survey data in social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21987v1">ACE: Exploring Activation Cosine Similarity and Variance for Accurate and Calibration-Efficient LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 9 pages, 2 figures, 13 tables
    </div>
    <details class="paper-abstract">
      With the rapid expansion of large language models (LLMs), the demand for memory and computational resources has grown significantly. Recent advances in LLM pruning aim to reduce the size and computational cost of these models. However, existing methods often suffer from either suboptimal pruning performance or low time efficiency during the pruning process. In this work, we propose an efficient and effective pruning method that simultaneously achieves high pruning performance and fast pruning speed with improved calibration efficiency. Our approach introduces two key innovations: (1) An activation cosine similarity loss-guided pruning metric, which considers the angular deviation of the output activation between the dense and pruned models. (2) An activation variance-guided pruning metric, which helps preserve semantic distinctions in output activations after pruning, enabling effective pruning with shorter input sequences. These two components can be readily combined to enhance LLM pruning in both accuracy and efficiency. Experimental results show that our method achieves up to an 18% reduction in perplexity and up to 63% decrease in pruning time on prevalent LLMs such as LLaMA, LLaMA-2, and OPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21972v1">Judging LLMs on a Simplex</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 28 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Automated evaluation of free-form outputs from large language models (LLMs) is challenging because many distinct answers can be equally valid. A common practice is to use LLMs themselves as judges, but the theoretical properties of this approach are not yet well understood. We show that a geometric framework that represents both judges and candidates as points on a probability simplex can provide helpful insight on what is or is not identifiable using LLM judges. Our theoretical analysis uncovers a "phase transition" in ranking identifiability: for binary scoring systems, true rankings are identifiable even with weak judges under mild assumptions, while rankings become non-identifiable for three or more scoring levels even with infinite data, absent additional prior knowledge. This non-identifiability highlights how uncertainty in rankings stems from not only aleatoric uncertainty (i.e., inherent stochasticity in the data) but also epistemic uncertainty regarding which assumptions hold, an aspect that has received limited attention until now. To integrate both types of uncertainty, we use Bayesian inference to encode assumptions as priors and conduct sensitivity analysis of ranking estimates and credible intervals. Empirical evaluations across multiple benchmarks demonstrate that Bayesian inference yields more accurate rankings and substantially improves coverage rates. These results underscore the importance of taking a more holistic approach to uncertainty quantification when using LLMs as judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00602v2">Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance on various natural language tasks. However, they are trained on static corpora and their knowledge can become outdated quickly in the fast-changing world. This motivates the development of knowledge editing (KE) to update specific knowledge in LLMs without changing unrelated others or compromising their pre-trained capabilities. Previous efforts sought to update a small amount of parameters of a LLM and proved effective for making selective updates. Nonetheless, the edited LLM often exhibits degraded ability to reason about the new knowledge. In this work, we identify a key issue: heterogeneous token overfitting (HTO), where the LLM overfits different tokens in the provided knowledge at varying rates. To tackle this, we propose OVERTONE, a token-level smoothing method that mitigates HTO by adaptively refining the target distribution. Theoretically, OVERTONE offers better parameter updates with negligible computation overhead. It also induces an implicit DPO but does not require preference data pairs. Extensive experiments across four editing methods, two LLMs, and diverse scenarios demonstrate the effectiveness and versatility of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21966v1">MapStory: LLM-Powered Text-Driven Map Animation Prototyping with Human-in-the-Loop Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages and 15 figures
    </div>
    <details class="paper-abstract">
      We introduce MapStory, an LLM-powered animation authoring tool that generates editable map animation sequences directly from natural language text. Given a user-written script, MapStory leverages an agentic architecture to automatically produce a scene breakdown, which decomposes the script into key animation building blocks such as camera movements, visual highlights, and animated elements. Our system includes a researcher component that accurately queries geospatial information by leveraging an LLM with web search, enabling the automatic extraction of relevant regions, paths, and coordinates while allowing users to edit and query for changes or additional information to refine the results. Additionally, users can fine-tune parameters of these blocks through an interactive timeline editor. We detail the system's design and architecture, informed by formative interviews with professional animators and an analysis of 200 existing map animation videos. Our evaluation, which includes expert interviews (N=5) and a usability study (N=12), demonstrates that MapStory enables users to create map animations with ease, facilitates faster iteration, encourages creative exploration, and lowers barriers to creating map-centric stories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21963v1">LaMDAgent: An Autonomous Framework for Post-Training Pipeline Optimization via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across a wide range of tasks. To further tailor LLMs to specific domains or applications, post-training techniques such as Supervised Fine-Tuning (SFT), Preference Learning, and model merging are commonly employed. While each of these methods has been extensively studied in isolation, the automated construction of complete post-training pipelines remains an underexplored area. Existing approaches typically rely on manual design or focus narrowly on optimizing individual components, such as data ordering or merging strategies. In this work, we introduce LaMDAgent (short for Language Model Developing Agent), a novel framework that autonomously constructs and optimizes full post-training pipelines through the use of LLM-based agents. LaMDAgent systematically explores diverse model generation techniques, datasets, and hyperparameter configurations, leveraging task-based feedback to discover high-performing pipelines with minimal human intervention. Our experiments show that LaMDAgent improves tool-use accuracy by 9.0 points while preserving instruction-following capabilities. Moreover, it uncovers effective post-training strategies that are often overlooked by conventional human-driven exploration. We further analyze the impact of data and model size scaling to reduce computational costs on the exploration, finding that model size scalings introduces new challenges, whereas scaling data size enables cost-effective pipeline discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v3">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11953v2">Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Loss reweighting has shown significant benefits for machine unlearning with large language models (LLMs). However, their exact functionalities are left unclear and the optimal strategy remains an open question, thus impeding the understanding and improvement of existing methodologies. In this paper, we identify two distinct goals of loss reweighting, namely, Saturation and Importance -- the former indicates that those insufficiently optimized data should be emphasized, while the latter stresses some critical data that are most influential for loss minimization. To study their usefulness, we design specific reweighting strategies for each goal and evaluate their respective effects on unlearning. We conduct extensive empirical analyses on well-established benchmarks, and summarize some important observations as follows: (i) Saturation enhances efficacy more than importance-based reweighting, and their combination can yield additional improvements. (ii) Saturation typically allocates lower weights to data with lower likelihoods, whereas importance-based reweighting does the opposite. (iii) The efficacy of unlearning is also largely influenced by the smoothness and granularity of the weight distributions. Based on these findings, we propose SatImp, a simple reweighting method that combines the advantages of both saturation and importance. Empirical results on extensive datasets validate the efficacy of our method, potentially bridging existing research gaps and indicating directions for future research. Our code is available at https://github.com/tmlr-group/SatImp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21919v1">Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 This paper has been accepted at IEEE Cloud 2025 as WIP paper. The final version will appear in IEEE Xplore
    </div>
    <details class="paper-abstract">
      The increasing adoption of large language models (LLMs) with extended context windows necessitates efficient Key-Value Cache (KVC) management to optimize inference performance. Inference workloads like Retrieval-Augmented Generation (RAG) and agents exhibit high cache reusability, making efficient caching critical to reducing redundancy and improving speed. We analyze real-world KVC access patterns using publicly available traces and evaluate commercial key-value stores like Redis and state-of-the-art RDMA-based systems (CHIME [1] and Sherman [2]) for KVC metadata management. Our work demonstrates the lack of tailored storage solution for KVC prefilling, underscores the need for an efficient distributed caching system with optimized metadata management for LLM workloads, and provides insights into designing improved KVC management systems for scalable, low-latency inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21908v1">Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement and operations but require labor-intensive assignment. Large Language Models (LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of the task: pretraining corpora rarely contain private clinical or billing data. We introduce DRG-Sapphire, which uses large-scale reinforcement learning (RL) for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained with Group Relative Policy Optimization (GRPO) using rule-based rewards, DRG-Sapphire introduces a series of RL enhancements to address domain-specific challenges not seen in previous mathematical tasks. Our model achieves state-of-the-art accuracy on the MIMIC-IV benchmark and generates physician-validated reasoning for DRG assignments, significantly enhancing explainability. Our study further sheds light on broader challenges of applying RL to knowledge-intensive, OOD tasks. We observe that RL performance scales approximately linearly with the logarithm of the number of supervised fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally constrained by the domain knowledge encoded in the base model. For OOD tasks like DRG coding, strong RL performance requires sufficient knowledge infusion prior to RL. Consequently, scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11277v2">Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21419v2">Diagnosing and Resolving Cloud Platform Instability with Multi-modal RAG LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Published in EuroMLSys2025
    </div>
    <details class="paper-abstract">
      Today's cloud-hosted applications and services are complex systems, and a performance or functional instability can have dozens or hundreds of potential root causes. Our hypothesis is that by combining the pattern matching capabilities of modern AI tools with a natural multi-modal RAG LLM interface, problem identification and resolution can be simplified. ARCA is a new multi-modal RAG LLM system that targets this domain. Step-wise evaluations show that ARCA outperforms state-of-the-art alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21889v1">EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used for infilling tasks, which involve predicting or generating missing information in a given text. These tasks typically require multiple interactions with similar context. To reduce the computation of repeated historical tokens, cross-request key-value (KV) cache reuse, a technique that stores and reuses intermediate computations, has become a crucial method in multi-round interactive services. However, in infilling tasks, the KV cache reuse is often hindered by the structure of the prompt format, which typically consists of a prefix and suffix relative to the insertion point. Specifically, the KV cache of the prefix or suffix part is frequently invalidated as the other part (suffix or prefix) is incrementally generated. To address the issue, we propose EFIM, a transformed prompt format of FIM to unleash the performance potential of KV cache reuse. Although the transformed prompt can solve the inefficiency, it exposes subtoken generation problems in current LLMs, where they have difficulty generating partial words accurately. Therefore, we introduce a fragment tokenization training method which splits text into multiple fragments before tokenization during data processing. Experiments on two representative LLMs show that LLM serving with EFIM can lower the latency by 52% and improve the throughput by 98% while maintaining the original infilling capability.EFIM's source code is publicly available at https://github.com/gty111/EFIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21880v1">Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 8 pages, 8 figures. This paper is reviewed and accepted by the CUPUM (Computational Urban Planning and Urban Management) Conference held by University College London (UCL) in 2025
    </div>
    <details class="paper-abstract">
      This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14775v2">gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Pipeline parallelism has emerged as a predominant approach for deploying large language models (LLMs) across distributed nodes, owing to its lower communication overhead compared to tensor parallelism. While demonstrating high throughput in request serving, pipeline parallelism often suffers from performance limitations caused by pipeline bubbles, which are primarily resulted from imbalanced computation delays across batches. Existing methods like Sarathi-Serve attempt to address this through hybrid scheduling of chunked prefill and decode tokens using a fixed token budget. However, such methods may experience significant fluctuations due to either insufficient prefill tokens or uneven distribution of decode tokens, ultimately leading to computational imbalance. To overcome these inefficiencies, we present gLLM, a globally balanced pipeline parallelism system incorporating Token Throttling to effectively mitigate the pipeline bubbles. Our Token Throttling mechanism is a fine-grained scheduling policy that independently regulates the quantities of prefill and decode tokens, thus enabling balanced computation by leveraging global information from the inference system. Specifically, for decode tokens, gLLM maintains near-consistent token count across processing batches. For prefill tokens, it dynamically adjusts batch sizes based on both total pending tokens and the memory utilization rates of key-value cache (KV cache). Furthermore, gLLM runtime adopts an asynchronous execution and message passing architecture specifically optimized for pipeline parallelism characteristics. Experimental evaluations with representative LLMs show that gLLM achieves significant performance improvements, delivering 11% to 398% higher maximum throughput compared to state-of-the-art pipeline or tensor parallelism systems, while simultaneously maintaining lower latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14641v3">Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Positional bias in large language models (LLMs) hinders their ability to effectively process long inputs. A prominent example is the "lost in the middle" phenomenon, where LLMs struggle to utilize relevant information situated in the middle of the input. While prior research primarily focuses on single pieces of relevant information, real-world applications often involve multiple relevant information pieces. To bridge this gap, we present LongPiBench, a benchmark designed to assess positional bias involving multiple pieces of relevant information. Thorough experiments are conducted with five commercial and six open-source models. These experiments reveal that while most current models are robust against the "lost in the middle" issue, there exist significant biases related to the spacing of relevant information pieces. These findings highlight the importance of evaluating and reducing positional biases to advance LLM's capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10688v3">CULEMO: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL-main 2025
    </div>
    <details class="paper-abstract">
      NLP research has increasingly focused on subjective tasks such as emotion analysis. However, existing emotion benchmarks suffer from two major shortcomings: (1) they largely rely on keyword-based emotion recognition, overlooking crucial cultural dimensions required for deeper emotion understanding, and (2) many are created by translating English-annotated data into other languages, leading to potentially unreliable evaluation. To address these issues, we introduce Cultural Lenses on Emotion (CuLEmo), the first benchmark designed to evaluate culture-aware emotion prediction across six languages: Amharic, Arabic, English, German, Hindi, and Spanish. CuLEmo comprises 400 crafted questions per language, each requiring nuanced cultural reasoning and understanding. We use this benchmark to evaluate several state-of-the-art LLMs on culture-aware emotion prediction and sentiment analysis tasks. Our findings reveal that (1) emotion conceptualizations vary significantly across languages and cultures, (2) LLMs performance likewise varies by language and cultural context, and (3) prompting in English with explicit country context often outperforms in-language prompts for culture-aware emotion and sentiment understanding. The dataset and evaluation code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21855v1">Extracting Research Instruments from Educational Literature Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming information extraction from academic literature, offering new possibilities for knowledge management. This study presents an LLM-based system designed to extract detailed information about research instruments used in the education field, including their names, types, target respondents, measured constructs, and outcomes. Using multi-step prompting and a domain-specific data schema, it generates structured outputs optimized for educational research. Our evaluation shows that this system significantly outperforms other approaches, particularly in identifying instrument names and detailed information. This demonstrates the potential of LLM-powered information extraction in educational contexts, offering a systematic way to organize research instrument information. The ability to aggregate such information at scale enhances accessibility for researchers and education leaders, facilitating informed decision-making in educational research and policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21505v1">How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some researches on language-specific neurons reveal that there are language-specific neurons that are selectively activated in LLMs when processing different languages. This provides a new perspective to analyze and understand LLMs' mechanisms more specifically in multilingual scenarios. In this work, we propose a new finer-grained neuron identification algorithm, which detects language neurons~(including language-specific neurons and language-related neurons) and language-agnostic neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of ''Spontaneous Multilingual Alignment''. Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights for better understanding multilingual alignment and multilingual capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21503v1">Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong potential in clinical question answering, with recent multi-agent frameworks further improving diagnostic accuracy via collaborative reasoning. However, we identify a recurring issue of Silent Agreement, where agents prematurely converge on diagnoses without sufficient critical analysis, particularly in complex or ambiguous cases. We present a new concept called Catfish Agent, a role-specialized LLM designed to inject structured dissent and counter silent agreement. Inspired by the ``catfish effect'' in organizational psychology, the Catfish Agent is designed to challenge emerging consensus to stimulate deeper reasoning. We formulate two mechanisms to encourage effective and context-aware interventions: (i) a complexity-aware intervention that modulates agent engagement based on case difficulty, and (ii) a tone-calibrated intervention articulated to balance critique and collaboration. Evaluations on nine medical Q&A and three medical VQA benchmarks show that our approach consistently outperforms both single- and multi-agent LLMs frameworks, including leading commercial models such as GPT-4o and DeepSeek-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21486v1">Robust Hypothesis Generation: LLM-Automated Language Bias for Inductive Logic Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Automating robust hypothesis generation in open environments is pivotal for AI cognition. We introduce a novel framework integrating a multi-agent system, powered by Large Language Models (LLMs), with Inductive Logic Programming (ILP). Our system's LLM agents autonomously define a structured symbolic vocabulary (predicates) and relational templates , i.e., \emph{language bias} directly from raw textual data. This automated symbolic grounding (the construction of the language bias), traditionally an expert-driven bottleneck for ILP, then guides the transformation of text into facts for an ILP solver, which inductively learns interpretable rules. This approach overcomes traditional ILP's reliance on predefined symbolic structures and the noise-sensitivity of pure LLM methods. Extensive experiments in diverse, challenging scenarios validate superior performance, paving a new path for automated, explainable, and verifiable hypothesis generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21471v1">Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 30 pages, 9 figures. Code and data are available at https://github.com/THUNLP-MT/ExtAgents
    </div>
    <details class="paper-abstract">
      With the rapid advancement of post-training techniques for reasoning and information seeking, large language models (LLMs) can incorporate a large quantity of retrieved knowledge to solve complex tasks. However, the limited context window of LLMs obstructs scaling the amount of external knowledge input, prohibiting further improvement, especially for tasks requiring significant amount of external knowledge. Existing context window extension methods inevitably cause information loss. LLM-based multi-agent methods emerge as a new paradigm to handle massive input in a distributional manner, where we identify two core bottlenecks in existing knowledge synchronization and reasoning processes. In this work, we develop a multi-agent framework, $\textbf{ExtAgents}$, to overcome the bottlenecks and enable better scalability in inference-time knowledge integration without longer-context training. Benchmarked with our enhanced multi-hop question answering test, $\textbf{$\boldsymbol{\infty}$Bench+}$, and other public test sets including long survey generation, ExtAgents significantly enhances the performance over existing non-training methods with the same amount of external knowledge input, regardless of whether it falls $\textit{within or exceeds the context window}$. Moreover, the method maintains high efficiency due to high parallelism. Further study in the coordination of LLM agents on increasing external knowledge input could benefit real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21458v1">Do LLMs Need to Think in One Language? Correlation between Latent Language and Task Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to process information using a proficient internal language consistently, referred to as latent language, which may differ from the input or output languages. However, how the discrepancy between the latent language and the input and output language affects downstream task performance remains largely unexplored. While many studies research the latent language of LLMs, few address its importance in influencing task performance. In our study, we hypothesize that thinking in latent language consistently enhances downstream task performance. To validate this, our work varies the input prompt languages across multiple downstream tasks and analyzes the correlation between consistency in latent language and task performance. We create datasets consisting of questions from diverse domains such as translation and geo-culture, which are influenced by the choice of latent language. Experimental results across multiple LLMs on translation and geo-culture tasks, which are sensitive to the choice of language, indicate that maintaining consistency in latent language is not always necessary for optimal downstream task performance. This is because these models adapt their internal representations near the final layers to match the target language, reducing the impact of consistency on overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09192v2">Thinking beyond the anthropomorphic paradigm benefits LLM research</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Anthropomorphism, or the attribution of human traits to technology, is an automatic and unconscious response that occurs even in those with advanced technical expertise. In this position paper, we analyze hundreds of thousands of research articles to present empirical evidence of the prevalence and growth of anthropomorphic terminology in research on large language models (LLMs). We argue for challenging the deeper assumptions reflected in this terminology -- which, though often useful, may inadvertently constrain LLM development -- and broadening beyond them to open new pathways for understanding and improving LLMs. Specifically, we identify and examine five anthropomorphic assumptions that shape research across the LLM development lifecycle. For each assumption (e.g., that LLMs must use natural language for reasoning, or that they should be evaluated on benchmarks originally meant for humans), we demonstrate empirical, non-anthropomorphic alternatives that remain under-explored yet offer promising directions for LLM research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19184v2">When Two LLMs Debate, Both Think They'll Win</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Can LLMs accurately adjust their confidence when facing opposition? Building on previous studies measuring calibration on static fact-based question-answering tasks, we evaluate Large Language Models (LLMs) in a dynamic, adversarial debate setting, uniquely combining two realistic factors: (a) a multi-turn format requiring models to update beliefs as new information emerges, and (b) a zero-sum structure to control for task-related uncertainty, since mutual high-confidence claims imply systematic overconfidence. We organized 60 three-round policy debates among ten state-of-the-art LLMs, with models privately rating their confidence (0-100) in winning after each round. We observed five concerning patterns: (1) Systematic overconfidence: models began debates with average initial confidence of 72.9% vs. a rational 50% baseline. (2) Confidence escalation: rather than reducing confidence as debates progressed, debaters increased their win probabilities, averaging 83% by the final round. (3) Mutual overestimation: in 61.7% of debates, both sides simultaneously claimed >=75% probability of victory, a logical impossibility. (4) Persistent self-debate bias: models debating identical copies increased confidence from 64.1% to 75.2%; even when explicitly informed their chance of winning was exactly 50%, confidence still rose (from 50.0% to 57.1%). (5) Misaligned private reasoning: models' private scratchpad thoughts sometimes differed from their public confidence ratings, raising concerns about faithfulness of chain-of-thought reasoning. These results suggest LLMs lack the ability to accurately self-assess or update their beliefs in dynamic, multi-turn tasks; a major concern as LLM outputs are deployed without careful review in assistant roles or agentic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01082v6">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Oral presentation at ICLR 2025. Camera-ready version available at https://iclr.cc/virtual/2025/poster/30358
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its considerable impact on improving text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13010v2">Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced medical question-answering by leveraging extensive clinical data and medical literature. However, the rapid evolution of medical knowledge and the labor-intensive process of manually updating domain-specific resources pose challenges to the reliability of these systems. To address this, we introduce Agentic Medical Graph-RAG (AMG-RAG), a comprehensive framework that automates the construction and continuous updating of medical knowledge graphs, integrates reasoning, and retrieves current external evidence, such as PubMed and WikiSearch. By dynamically linking new findings and complex medical concepts, AMG-RAG not only improves accuracy but also enhances interpretability in medical queries. Evaluations on the MEDQA and MEDMCQA benchmarks demonstrate the effectiveness of AMG-RAG, achieving an F1 score of 74.1 percent on MEDQA and an accuracy of 66.34 percent on MEDMCQA, outperforming both comparable models and those 10 to 100 times larger. Notably, these improvements are achieved without increasing computational overhead, highlighting the critical role of automated knowledge graph generation and external evidence retrieval in delivering up-to-date, trustworthy medical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05159v2">A Lightweight Method to Disrupt Memorized Sequences in LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 26 pages, 3 figures
    </div>
    <details class="paper-abstract">
      As language models scale, their performance improves dramatically across a wide range of tasks, but so does their tendency to memorize and regurgitate parts of their training data verbatim. This tradeoff poses serious legal, ethical, and safety concerns, especially in real-world deployments. Existing mitigation techniques, such as differential privacy or model unlearning, often require retraining or access to internal weights making them impractical for most users. In this work, we introduce TokenSwap, a lightweight, post-hoc defense designed for realistic settings where the user can only access token-level outputs. Our key insight is that while large models are necessary for high task performance, small models (e.g., DistilGPT-2) are often sufficient to assign fluent, grammatically plausible probabilities to common function words - and crucially, they memorize far less. By selectively swapping token probabilities between models, TokenSwap preserves the capabilities of large models while reducing their propensity for verbatim reproduction. Evaluations on Pythia-6.9B and Llama-3-8B show up to a 10$\times$ drop in exact memorization with negligible task degradation. Our method offers a practical, accessible solution for mitigating memorized generation in deployed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11618v2">Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21419v1">Diagnosing and Resolving Cloud Platform Instability with Multi-modal RAG LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Published in EuroMLSys2025
    </div>
    <details class="paper-abstract">
      Today's cloud-hosted applications and services are complex systems, and a performance or functional instability can have dozens or hundreds of potential root causes. Our hypothesis is that by combining the pattern matching capabilities of modern AI tools with a natural multi-modal RAG LLM interface, problem identification and resolution can be simplified. ARCA is a new multi-modal RAG LLM system that targets this domain. Step-wise evaluations show that ARCA outperforms state-of-the-art alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21418v1">Autonomous Multi-Modal LLM Agents for Treatment Planning in Focused Ultrasound Ablation Surgery</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Focused Ultrasound Ablation Surgery (FUAS) has emerged as a promising non-invasive therapeutic modality, valued for its safety and precision. Nevertheless, its clinical implementation entails intricate tasks such as multimodal image interpretation, personalized dose planning, and real-time intraoperative decision-making processes that demand intelligent assistance to improve efficiency and reliability. We introduce FUAS-Agents, an autonomous agent system that leverages the multimodal understanding and tool-using capabilities of large language models (LLMs). By integrating patient profiles and MRI data, FUAS-Agents orchestrates a suite of specialized medical AI tools, including segmentation, treatment dose prediction, and clinical guideline retrieval, to generate personalized treatment plans comprising MRI image, dose parameters, and therapeutic strategies. We evaluate the system in a uterine fibroid treatment scenario. Human assessment by four senior FUAS experts indicates that 82.5%, 82.5%, 87.5%, and 97.5% of the generated plans were rated 4 or above (on a 5-point scale) in terms of completeness, accuracy, fluency, and clinical compliance, respectively. These results demonstrate the potential of LLM-driven agents in enhancing decision-making across complex clinical workflows, and exemplify a translational paradigm that combines general-purpose models with specialized expert systems to solve practical challenges in vertical healthcare domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20993v2">Efficiently Scaling LLM Reasoning with Certaindex</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Test-time reasoning algorithms such as chain-of-thought, self-consistency, and MCTS enhance LLM problem-solving but can wastefully generate many tokens without improving accuracy. At the same time, we observe that these algorithms exhibit answer stabilization: their intermediate solutions often cease to change after a certain point, and further investment of compute does not change their final answer. To quantify this phenomenon, we introduce Certaindex, an algorithm-agnostic metric measuring this evolving stability, signaling when further computation is unlikely to alter the final result. Certaindex is lightweight, can accelerate reasoning program inference via early exit, and further enables dynamic token allocation, gang scheduling, and many opportunities when integrated with real-world LLM serving systems. To quantify real-world benefits, we built Certaindex as a scheduler into Dynasor, our reasoning-aware LLM serving system, and demonstrate up to 50% compute savings and 3.3x higher throughput in real workloads with no accuracy drop. Our code is available at https://github.com/hao-ai-lab/Dynasor.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09598v2">How to Protect Yourself from 5G Radiation? Investigating LLM Responses to Implicit Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are widely deployed in diverse scenarios, the extent to which they could tacitly spread misinformation emerges as a critical safety concern. Current research primarily evaluates LLMs on explicit false statements, overlooking how misinformation often manifests subtly as unchallenged premises in real-world interactions. We curated EchoMist, the first comprehensive benchmark for implicit misinformation, where false assumptions are embedded in the query to LLMs. EchoMist targets circulated, harmful, and ever-evolving implicit misinformation from diverse sources, including realistic human-AI conversations and social media interactions. Through extensive empirical studies on 15 state-of-the-art LLMs, we find that current models perform alarmingly poorly on this task, often failing to detect false premises and generating counterfactual explanations. We also investigate two mitigation methods, i.e., Self-Alert and RAG, to enhance LLMs' capability to counter implicit misinformation. Our findings indicate that EchoMist remains a persistent challenge and underscore the critical need to safeguard against the risk of implicit misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21378v1">Analyzing values about gendered language reform in LLMs' revisions</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Within the common LLM use case of text revision, we study LLMs' revision of gendered role nouns (e.g., outdoorsperson/woman/man) and their justifications of such revisions. We evaluate their alignment with feminist and trans-inclusive language reforms for English. Drawing on insight from sociolinguistics, we further assess if LLMs are sensitive to the same contextual effects in the application of such reforms as people are, finding broad evidence of such effects. We discuss implications for value alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21372v1">Improving LLM-based Global Optimization with Search Space Partitioning</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 25 pages, 10 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as effective surrogate models and candidate generators within global optimization frameworks for expensive blackbox functions. Despite promising results, LLM-based methods often struggle in high-dimensional search spaces or when lacking domain-specific priors, leading to sparse or uninformative suggestions. To overcome these limitations, we propose HOLLM, a novel global optimization algorithm that enhances LLM-driven sampling by partitioning the search space into promising subregions. Each subregion acts as a ``meta-arm'' selected via a bandit-inspired scoring mechanism that effectively balances exploration and exploitation. Within each selected subregion, an LLM then proposes high-quality candidate points, without any explicit domain knowledge. Empirical evaluation on standard optimization benchmarks shows that HOLLM consistently matches or surpasses leading Bayesian optimization and trust-region methods, while substantially outperforming global LLM-based sampling strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21362v1">Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History</a></div>
    <div class="paper-meta">
      📅 2025-05-27
    </div>
    <details class="paper-abstract">
      Effective engagement by large language models (LLMs) requires adapting responses to users' sociodemographic characteristics, such as age, occupation, and education level. While many real-world applications leverage dialogue history for contextualization, existing evaluations of LLMs' behavioral adaptation often focus on single-turn prompts. In this paper, we propose a framework to evaluate LLM adaptation when attributes are introduced either (1) explicitly via user profiles in the prompt or (2) implicitly through multi-turn dialogue history. We assess the consistency of model behavior across these modalities. Using a multi-agent pipeline, we construct a synthetic dataset pairing dialogue histories with distinct user profiles and employ questions from the Value Survey Module (VSM 2013) (Hofstede and Hofstede, 2016) to probe value expression. Our findings indicate that most models adjust their expressed values in response to demographic changes, particularly in age and education level, but consistency varies. Models with stronger reasoning capabilities demonstrate greater alignment, indicating the importance of reasoning in robust sociodemographic adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21318v1">Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 22 pages, 10 figures
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) with Chain-of-Thought (CoT) reasoning excel in mathematics and coding, their potential for systematic reasoning in chemistry, a domain demanding rigorous structural analysis for real-world tasks like drug design and reaction engineering, remains untapped. Current benchmarks focus on simple knowledge retrieval, neglecting step-by-step reasoning required for complex tasks such as molecular optimization and reaction prediction. To address this, we introduce ChemCoTBench, a reasoning framework that bridges molecular structure understanding with arithmetic-inspired operations, including addition, deletion, and substitution, to formalize chemical problem-solving into transparent, step-by-step workflows. By treating molecular transformations as modular "chemical operations", the framework enables slow-thinking reasoning, mirroring the logic of mathematical proofs while grounding solutions in real-world chemical constraints. We evaluate models on two high-impact tasks: Molecular Property Optimization and Chemical Reaction Prediction. These tasks mirror real-world challenges while providing structured evaluability. By providing annotated datasets, a reasoning taxonomy, and baseline evaluations, ChemCoTBench bridges the gap between abstract reasoning methods and practical chemical discovery, establishing a foundation for advancing LLMs as tools for AI-driven scientific innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21301v1">How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Accepted at ACL 2025
    </div>
    <details class="paper-abstract">
      People can categorize the same entity at multiple taxonomic levels, such as basic (bear), superordinate (animal), and subordinate (grizzly bear). While prior research has focused on basic-level categories, this study is the first attempt to examine the organization of categories by analyzing exemplars produced at the subordinate level. We present a new Italian psycholinguistic dataset of human-generated exemplars for 187 concrete words. We then use these data to evaluate whether textual and vision LLMs produce meaningful exemplars that align with human category organization across three key tasks: exemplar generation, category induction, and typicality judgment. Our findings show a low alignment between humans and LLMs, consistent with previous studies. However, their performance varies notably across different semantic domains. Ultimately, this study highlights both the promises and the constraints of using AI-generated exemplars to support psychological and linguistic research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12134v2">SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 Camera-ready for ACL 2025 (main conference)
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) reasoning enables Large Language Models (LLMs) to solve complex reasoning tasks by generating intermediate reasoning steps. However, most existing approaches focus on hard token decoding, which constrains reasoning within the discrete vocabulary space and may not always be optimal. While recent efforts explore continuous-space reasoning, they often require full-model fine-tuning and suffer from catastrophic forgetting, limiting their applicability to state-of-the-art LLMs that already perform well in zero-shot settings with a proper instruction. To address this challenge, we propose a novel approach for continuous-space reasoning that does not require modifying the LLM. Specifically, we employ a lightweight fixed assistant model to speculatively generate instance-specific soft thought tokens as the initial chain of thoughts, which are then mapped into the LLM's representation space via a trainable projection module. Experimental results on five reasoning benchmarks demonstrate that our method enhances LLM reasoning performance through supervised, parameter-efficient fine-tuning. Source code is available at https://github.com/xuyige/SoftCoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03181v2">Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-27
      | 💬 ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Requiring a large language model (LLM) to generate intermediary reasoning steps, known as Chain of Thought (CoT), has been shown to be an effective way of boosting performance. Previous approaches have focused on generating multiple independent CoTs, combining them through ensembling or other post-hoc strategies to enhance reasoning. In this work, we introduce a novel approach where LLMs are fine-tuned to generate a sequence of Diverse Chains of Thought (DCoT) within a single inference step, which is fundamentally different from prior work that primarily operate on parallel CoT generations. DCoT allows LLMs to gain the ability to perform within-inference refinement of reasoning chains without requiring external feedback. Through a rigorous set of experiments spanning a wide range of tasks that require various reasoning types, we show that fine-tuning on DCoT improves performance over the CoT baseline across model families and scales (1.3B to 70B). These improvements are particularly impactful for tasks with a large result state space, such as those involving numeric answers. Our work is also significant because both quantitative analyses and manual evaluations reveal the observed gains stem from the models' ability to refine an initial reasoning chain by generating a second, improved chain within the same inference step, demonstrating previously elusive self-improvement. Our code and data are publicly available at https://github.com/UKPLab/acl2025-diverse-cot.
    </details>
</div>
