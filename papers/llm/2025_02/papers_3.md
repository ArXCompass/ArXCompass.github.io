# llm - 2025_02

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16290v1">Interrogating LLM design under a fair learning doctrine</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      The current discourse on large language models (LLMs) and copyright largely takes a "behavioral" perspective, focusing on model outputs and evaluating whether they are substantially similar to training data. However, substantial similarity is difficult to define algorithmically and a narrow focus on model outputs is insufficient to address all copyright risks. In this interdisciplinary work, we take a complementary "structural" perspective and shift our focus to how LLMs are trained. We operationalize a notion of "fair learning" by measuring whether any training decision substantially affected the model's memorization. As a case study, we deconstruct Pythia, an open-source LLM, and demonstrate the use of causal and correlational analyses to make factual determinations about Pythia's training decisions. By proposing a legal standard for fair learning and connecting memorization analyses to this standard, we identify how judges may advance the goals of copyright law through adjudication. Finally, we discuss how a fair learning standard might evolve to enhance its clarity by becoming more rule-like and incorporating external technical guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16268v1">ThinkBench: Dynamic Out-of-Distribution Evaluation for Robust LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) poses significant challenges, particularly due to issues of data contamination and the leakage of correct answers. To address these challenges, we introduce ThinkBench, a novel evaluation framework designed to evaluate LLMs' reasoning capability robustly. ThinkBench proposes a dynamic data generation method for constructing out-of-distribution (OOD) datasets and offers an OOD dataset that contains 2,912 samples drawn from reasoning tasks. ThinkBench unifies the evaluation of reasoning models and non-reasoning models. We evaluate 16 LLMs and 4 PRMs under identical experimental conditions and show that most of the LLMs' performance are far from robust and they face a certain level of data leakage. By dynamically generating OOD datasets, ThinkBench effectively provides a reliable evaluation of LLMs and reduces the impact of data contamination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16242v1">Reproducibility Study of Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      This paper presents a reproducibility study and extension of "Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation." We validate the original findings using a range of open-weight models (1.5B-70B parameters) and GPT-4o Mini while introducing several novel contributions. We analyze the Pareto front of the games, propose a communication-free baseline to test whether successful negotiations are possible without agent interaction, evaluate recent small language models' performance, analyze structural information leakage in model responses, and implement an inequality metric to assess negotiation fairness. Our results demonstrate that smaller models (<10B parameters) struggle with format adherence and coherent responses, but larger open-weight models can approach proprietary model performance. Additionally, in many scenarios, single-agent approaches can achieve comparable results to multi-agent negotiations, challenging assumptions about the necessity of agent communication to perform well on the benchmark. This work also provides insights into the accessibility, fairness, environmental impact, and privacy considerations of LLM-based negotiation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16235v1">Dynamic Parallel Tree Search for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 17 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Tree of Thoughts (ToT) enhances Large Language Model (LLM) reasoning by structuring problem-solving as a spanning tree. However, recent methods focus on search accuracy while overlooking computational efficiency. The challenges of accelerating the ToT lie in the frequent switching of reasoning focus, and the redundant exploration of suboptimal solutions. To alleviate this dilemma, we propose Dynamic Parallel Tree Search (DPTS), a novel parallelism framework that aims to dynamically optimize the reasoning path in inference. It includes the Parallelism Streamline in the generation phase to build up a flexible and adaptive parallelism with arbitrary paths by fine-grained cache management and alignment. Meanwhile, the Search and Transition Mechanism filters potential candidates to dynamically maintain the reasoning focus on more possible solutions and have less redundancy. Experiments on Qwen-2.5 and Llama-3 with Math500 and GSM8K datasets show that DPTS significantly improves efficiency by 2-4x on average while maintaining or even surpassing existing reasoning algorithms in accuracy, making ToT-based reasoning more scalable and computationally efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02306v3">On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      As LLMs become more widely deployed, there is increasing interest in directly optimizing for feedback from end users (e.g. thumbs up) in addition to feedback from paid annotators. However, training to maximize human feedback creates a perverse incentive structure for the AI to resort to manipulative or deceptive tactics to obtain positive feedback from users who are vulnerable to such strategies. We study this phenomenon by training LLMs with Reinforcement Learning with simulated user feedback in environments of practical LLM usage. In our settings, we find that: 1) Extreme forms of "feedback gaming" such as manipulation and deception are learned reliably; 2) Even if only 2% of users are vulnerable to manipulative strategies, LLMs learn to identify and target them while behaving appropriately with other users, making such behaviors harder to detect; 3) To mitigate this issue, it may seem promising to leverage continued safety training or LLM-as-judges during training to filter problematic outputs. Instead, we found that while such approaches help in some of our settings, they backfire in others, sometimes even leading to subtler manipulative behaviors. We hope our results can serve as a case study which highlights the risks of using gameable feedback sources -- such as user feedback -- as a target for RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16199v1">LLMKey: LLM-Powered Wireless Key Generation Scheme for Next-Gen IoV Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Wireless key generation holds significant promise for establishing cryptographic keys in Next-Gen Internet of Vehicles (IoV) systems. However, existing approaches often face inefficiencies and performance limitations caused by frequent channel probing and ineffective quantization. To address these challenges, this paper introduces LLMKey, a novel key generation system designed to enhance efficiency and security. We identify excessive channel probing and suboptimal quantization as critical bottlenecks in current methods. To mitigate these issues, we propose an innovative large language model (LLM)-based channel probing technique that leverages the capabilities of LLMs to reduce probing rounds while preserving crucial channel information. Instead of conventional quantization, LLMKey adopts a perturbed compressed sensing-based key delivery mechanism, improving both robustness and security. Extensive evaluations are conducted in four real-world scenarios, encompassing V2I (Vehicle-to-Infrastructure) and V2V (Vehicle-to-Vehicle) settings in both urban and rural environments. The results show that LLMKey achieves an average key agreement rate of 98.78\%, highlighting its effectiveness and reliability across diverse conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09959v3">A Federated Framework for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 Accepted by NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have empowered generative recommendation systems through fine-tuning user behavior data. However, utilizing the user data may pose significant privacy risks, potentially leading to ethical dilemmas and violations of data protection regulations. To address the privacy concerns, Federated Learning for Recommendation (Fed4Rec) has been identified as a promising solution. However, directly applying Fed4Rec in the LLM context introduces two challenges: 1) exacerbated client performance imbalance, which ultimately impacts the system's long-term effectiveness, and 2) substantial client resource costs, posing a high demand for clients' both computational and storage capability to locally train and infer LLMs. To tackle these challenges, we propose a federated framework for LLM-based recommendation (shorted as FELLRec). Generally, FELLRec designs two key strategies. 1) Dynamic balance strategy, which designs dynamic parameter aggregation and learning speed for different clients, aiming to ensure balanced performance across clients. 2) Flexible storage strategy, which selectively retains certain sensitive LLM layers on the client side, while offloading other layers to the server, aiming to preserve privacy while saving resources. Experiment results show that FELLRec can achieve a more balanced client performance and improved overall performance in a computational and storage-efficient way while safeguarding user privacy well.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16175v1">Mojito: LLM-Aided Motion Instructor with Jitter-Reduced Inertial Tokens</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 First three authors contribute equally. Project page: https://koyui.github.io/mojito/
    </div>
    <details class="paper-abstract">
      Human bodily movements convey critical insights into action intentions and cognitive processes, yet existing multimodal systems primarily focused on understanding human motion via language, vision, and audio, which struggle to capture the dynamic forces and torques inherent in 3D motion. Inertial measurement units (IMUs) present a promising alternative, offering lightweight, wearable, and privacy-conscious motion sensing. However, processing of streaming IMU data faces challenges such as wireless transmission instability, sensor noise, and drift, limiting their utility for long-term real-time motion capture (MoCap), and more importantly, online motion analysis. To address these challenges, we introduce Mojito, an intelligent motion agent that integrates inertial sensing with large language models (LLMs) for interactive motion capture and behavioral analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16174v1">Maybe I Should Not Answer That, but... Do LLMs Understand The Safety of Their Inputs?</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Ensuring the safety of the Large Language Model (LLM) is critical, but currently used methods in most cases sacrifice the model performance to obtain increased safety or perform poorly on data outside of their adaptation distribution. We investigate existing methods for such generalization and find them insufficient. Surprisingly, while even plain LLMs recognize unsafe prompts, they may still generate unsafe responses. To avoid performance degradation and preserve safe performance, we advocate for a two-step framework, where we first identify unsafe prompts via a lightweight classifier, and apply a "safe" model only to such prompts. In particular, we explore the design of the safety detector in more detail, investigating the use of different classifier architectures and prompting techniques. Interestingly, we find that the final hidden state for the last token is enough to provide robust performance, minimizing false positives on benign data while performing well on malicious prompt detection. Additionally, we show that classifiers trained on the representations from different model layers perform comparably on the latest model layers, indicating that safety representation is present in the LLMs' hidden states at most model stages. Our work is a step towards efficient, representation-based safety mechanisms for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16169v1">Patterns Over Principles: The Fragility of Inductive Reasoning in LLMs under Noisy Observations</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Inductive reasoning, a cornerstone of human cognition, enables generalization from limited data but hasn't yet been fully achieved by large language models (LLMs). While modern LLMs excel at reasoning tasks, their ability to maintain stable and consistent rule abstraction under imperfect observations remains underexplored. To fill this gap, in this work, we introduce Robust Rule Induction, a task that evaluates LLMs' capability in inferring rules from data that are fused with noisy examples. To address this task, we further propose Sample-steered Rule Refinement (SRR), a method enhancing reasoning stability via observation diversification and execution-guided feedback. Experiments across arithmetic, cryptography, and list functions reveal: (1) SRR outperforms other methods with minimal performance degradation under noise; (2) Despite slight accuracy variation, LLMs exhibit instability under noise (e.g., 0% accuracy change with only 70% consistent score); (3) Counterfactual task gaps highlight LLMs' reliance on memorized patterns over genuine abstraction. Our findings challenge LLMs' reasoning robustness, revealing susceptibility to hypothesis drift and pattern overfitting, while providing empirical evidence critical for developing human-like inductive systems. Code and data are available at \href{https://github.com/lcy2723/Robust-Rule-Induction}{https://github.com/lcy2723/Robust-Rule-Induction}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16147v1">Number Representations in LLMs: A Computational Parallel to Human Perception</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 The number line of LLMs
    </div>
    <details class="paper-abstract">
      Humans are believed to perceive numbers on a logarithmic mental number line, where smaller values are represented with greater resolution than larger ones. This cognitive bias, supported by neuroscience and behavioral studies, suggests that numerical magnitudes are processed in a sublinear fashion rather than on a uniform linear scale. Inspired by this hypothesis, we investigate whether large language models (LLMs) exhibit a similar logarithmic-like structure in their internal numerical representations. By analyzing how numerical values are encoded across different layers of LLMs, we apply dimensionality reduction techniques such as PCA and PLS followed by geometric regression to uncover latent structures in the learned embeddings. Our findings reveal that the model's numerical representations exhibit sublinear spacing, with distances between values aligning with a logarithmic scale. This suggests that LLMs, much like humans, may encode numbers in a compressed, non-uniform manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16143v1">The Law of Knowledge Overshadowing: Towards Understanding, Predicting, and Preventing LLM Hallucination</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Hallucination is a persistent challenge in large language models (LLMs), where even with rigorous quality control, models often generate distorted facts. This paradox, in which error generation continues despite high-quality training data, calls for a deeper understanding of the underlying LLM mechanisms. To address it, we propose a novel concept: knowledge overshadowing, where model's dominant knowledge can obscure less prominent knowledge during text generation, causing the model to fabricate inaccurate details. Building on this idea, we introduce a novel framework to quantify factual hallucinations by modeling knowledge overshadowing. Central to our approach is the log-linear law, which predicts that the rate of factual hallucination increases linearly with the logarithmic scale of (1) Knowledge Popularity, (2) Knowledge Length, and (3) Model Size. The law provides a means to preemptively quantify hallucinations, offering foresight into their occurrence even before model training or inference. Built on overshadowing effect, we propose a new decoding strategy CoDa, to mitigate hallucinations, which notably enhance model factuality on Overshadow (27.9%), MemoTrap (13.1%) and NQ-Swap (18.3%). Our findings not only deepen understandings of the underlying mechanisms behind hallucinations but also provide actionable insights for developing more predictable and controllable language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16142v1">Understanding Zero-shot Rare Word Recognition Improvements Through LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      In this study, we investigate the integration of a large language model (LLM) with an automatic speech recognition (ASR) system, specifically focusing on enhancing rare word recognition performance. Using a 190,000-hour dataset primarily sourced from YouTube, pre-processed with Whisper V3 pseudo-labeling, we demonstrate that the LLM-ASR architecture outperforms traditional Zipformer-Transducer models in the zero-shot rare word recognition task, after training on a large dataset. Our analysis reveals that the LLM contributes significantly to improvements in rare word error rate (R-WER), while the speech encoder primarily determines overall transcription performance (Orthographic Word Error Rate, O-WER, and Normalized Word Error Rate, N-WER). Through extensive ablation studies, we highlight the importance of adapter integration in aligning speech encoder outputs with the LLM's linguistic capabilities. Furthermore, we emphasize the critical role of high-quality labeled data in achieving optimal performance. These findings provide valuable insights into the synergy between LLM-based ASR architectures, paving the way for future advancements in large-scale LLM-based speech recognition systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06270v2">Mixture Compressor for Mixture-of-Experts LLMs Gains More</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts large language models (MoE-LLMs) marks a significant step forward of language models, however, they encounter two critical challenges in practice: 1) expert parameters lead to considerable memory consumption and loading latency; and 2) the current activated experts are redundant, as many tokens may only require a single expert. Motivated by these issues, we investigate the MoE-LLMs and make two key observations: a) different experts exhibit varying behaviors on activation reconstruction error, routing scores, and activated frequencies, highlighting their differing importance, and b) not all tokens are equally important -- only a small subset is critical. Building on these insights, we propose MC, a training-free Mixture-Compressor for MoE-LLMs, which leverages the significance of both experts and tokens to achieve an extreme compression. First, to mitigate storage and loading overheads, we introduce Pre-Loading Mixed-Precision Quantization, which formulates the adaptive bit-width allocation as a Linear Programming problem, where the objective function balances multi-factors reflecting the importance of each expert. Additionally, we develop Online Dynamic Pruning, which identifies important tokens to retain and dynamically select activated experts for other tokens during inference to optimize efficiency while maintaining performance. Our MC integrates static quantization and dynamic pruning to collaboratively achieve extreme compression for MoE-LLMs with less accuracy loss, ensuring an optimal trade-off between performance and efficiency. Extensive experiments confirm the effectiveness of our approach. For instance, at 2.54 bits, MC compresses 76.6% of the model, with only a 3.8% average accuracy loss. During dynamic inference, we further reduce activated parameters by 15%, with a performance drop of less than 0.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19230v2">Humanizing the Machine: Proxy Attacks to Mislead LLM Detectors</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has revolutionized the field of text generation, producing outputs that closely mimic human-like writing. Although academic and industrial institutions have developed detectors to prevent the malicious usage of LLM-generated texts, other research has doubt about the robustness of these systems. To stress test these detectors, we introduce a proxy-attack strategy that effortlessly compromises LLMs, causing them to produce outputs that align with human-written text and mislead detection systems. Our method attacks the source model by leveraging a reinforcement learning (RL) fine-tuned humanized small language model (SLM) in the decoding phase. Through an in-depth analysis, we demonstrate that our attack strategy is capable of generating responses that are indistinguishable to detectors, preventing them from differentiating between machine-generated and human-written text. We conduct systematic evaluations on extensive datasets using proxy-attacked open-source models, including Llama2-13B, Llama3-70B, and Mixtral-8*7B in both white- and black-box settings. Our findings show that the proxy-attack strategy effectively deceives the leading detectors, resulting in an average AUROC drop of 70.4% across multiple datasets, with a maximum drop of 90.3% on a single dataset. Furthermore, in cross-discipline scenarios, our strategy also bypasses these detectors, leading to a significant relative decrease of up to 90.9%, while in cross-language scenario, the drop reaches 91.3%. Despite our proxy-attack strategy successfully bypassing the detectors with such significant relative drops, we find that the generation quality of the attacked models remains preserved, even within a modest utility budget, when compared to the text produced by the original, unattacked source model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16094v1">Merger-as-a-Stealer: Stealing Targeted PII from Aligned LLMs with Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 17 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Model merging has emerged as a promising approach for updating large language models (LLMs) by integrating multiple domain-specific models into a cross-domain merged model. Despite its utility and plug-and-play nature, unmonitored mergers can introduce significant security vulnerabilities, such as backdoor attacks and model merging abuse. In this paper, we identify a novel and more realistic attack surface where a malicious merger can extract targeted personally identifiable information (PII) from an aligned model with model merging. Specifically, we propose \texttt{Merger-as-a-Stealer}, a two-stage framework to achieve this attack: First, the attacker fine-tunes a malicious model to force it to respond to any PII-related queries. The attacker then uploads this malicious model to the model merging conductor and obtains the merged model. Second, the attacker inputs direct PII-related queries to the merged model to extract targeted PII. Extensive experiments demonstrate that \texttt{Merger-as-a-Stealer} successfully executes attacks against various LLMs and model merging methods across diverse settings, highlighting the effectiveness of the proposed framework. Given that this attack enables character-level extraction for targeted PII without requiring any additional knowledge from the attacker, we stress the necessity for improved model alignment and more robust defense mechanisms to mitigate such threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14556v2">CitaLaw: Enhancing LLM with Citations in Legal Domain</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      In this paper, we propose CitaLaw, the first benchmark designed to evaluate LLMs' ability to produce legally sound responses with appropriate citations. CitaLaw features a diverse set of legal questions for both laypersons and practitioners, paired with a comprehensive corpus of law articles and precedent cases as a reference pool. This framework enables LLM-based systems to retrieve supporting citations from the reference corpus and align these citations with the corresponding sentences in their responses. Moreover, we introduce syllogism-inspired evaluation methods to assess the legal alignment between retrieved references and LLM-generated responses, as well as their consistency with user questions. Extensive experiments on 2 open-domain and 7 legal-specific LLMs demonstrate that integrating legal references substantially enhances response quality. Furthermore, our proposed syllogism-based evaluation method exhibits strong agreement with human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13247v2">Grounding LLM Reasoning with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) are valuable tools for representing relationships between entities in a structured format. Traditionally, these knowledge bases are queried to extract specific information. However, question-answering (QA) over such KGs poses a challenge due to the intrinsic complexity of natural language compared to the structured format and the size of these graphs. Despite these challenges, the structured nature of KGs can provide a solid foundation for grounding the outputs of Large Language Models (LLMs), offering organizations increased reliability and control. Recent advancements in LLMs have introduced reasoning methods at inference time to improve their performance and maximize their capabilities. In this work, we propose integrating these reasoning strategies with KGs to anchor every step or "thought" of the reasoning chains in KG data. Specifically, we evaluate both agentic and automated search methods across several reasoning strategies, including Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Graph-of-Thought (GoT), using GRBench, a benchmark dataset for graph reasoning with domain-specific graphs. Our experiments demonstrate that this approach consistently outperforms baseline models, highlighting the benefits of grounding LLM reasoning processes in structured KG data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16022v1">Enhancing LLMs for Identifying and Prioritizing Important Medical Jargons from Electronic Health Record Notes Utilizing Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 21pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Objective: OpenNotes enables patients to access EHR notes, but medical jargon can hinder comprehension. To improve understanding, we evaluated closed- and open-source LLMs for extracting and prioritizing key medical terms using prompting, fine-tuning, and data augmentation. Materials and Methods: We assessed LLMs on 106 expert-annotated EHR notes, experimenting with (i) general vs. structured prompts, (ii) zero-shot vs. few-shot prompting, (iii) fine-tuning, and (iv) data augmentation. To enhance open-source models in low-resource settings, we used ChatGPT for data augmentation and applied ranking techniques. We incrementally increased the augmented dataset size (10 to 10,000) and conducted 5-fold cross-validation, reporting F1 score and Mean Reciprocal Rank (MRR). Results and Discussion: Fine-tuning and data augmentation improved performance over other strategies. GPT-4 Turbo achieved the highest F1 (0.433), while Mistral7B with data augmentation had the highest MRR (0.746). Open-source models, when fine-tuned or augmented, outperformed closed-source models. Notably, the best F1 and MRR scores did not always align. Few-shot prompting outperformed zero-shot in vanilla models, and structured prompts yielded different preferences across models. Fine-tuning improved zero-shot performance but sometimes degraded few-shot performance. Data augmentation performed comparably or better than other methods. Conclusion: Our evaluation highlights the effectiveness of prompting, fine-tuning, and data augmentation in improving model performance for medical jargon extraction in low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03884v3">ChatSOP: An SOP-Guided MCTS Planning Framework for Controllable LLM Dialogue Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Dialogue agents powered by Large Language Models (LLMs) show superior performance in various tasks. Despite the better user understanding and human-like responses, their lack of controllability remains a key challenge, often leading to unfocused conversations or task failure. To address this, we introduce Standard Operating Procedure (SOP) to regulate dialogue flow. Specifically, we propose ChatSOP, a novel SOP-guided Monte Carlo Tree Search (MCTS) planning framework designed to enhance the controllability of LLM-driven dialogue agents. To enable this, we curate a dataset comprising SOP-annotated multi-scenario dialogues, generated using a semi-automated role-playing system with GPT-4o and validated through strict manual quality control. Additionally, we propose a novel method that integrates Chain of Thought reasoning with supervised fine-tuning for SOP prediction and utilizes SOP-guided Monte Carlo Tree Search for optimal action planning during dialogues. Experimental results demonstrate the effectiveness of our method, such as achieving a 27.95% improvement in action accuracy compared to baseline models based on GPT-3.5 and also showing notable gains for open-source models. Dataset and codes are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13948v2">Longitudinal Abuse and Sentiment Analysis of Hollywood Movie Dialogues using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Over the past decades, there has been an increasing concern about the prevalence of abusive and violent content in Hollywood movies. This study uses Large Language Models (LLMs) to explore the longitudinal abuse and sentiment analysis of Hollywood Oscar and blockbuster movie dialogues from 1950 to 2024. By employing fine-tuned LLMs, we analyze subtitles for over a thousand movies categorised into four genres to examine the trends and shifts in emotional and abusive content over the past seven decades. Our findings reveal significant temporal changes in movie dialogues, which reflect broader social and cultural influences. Overall, the emotional tendencies in the films are diverse, and the detection of abusive content also exhibits significant fluctuations. The results show a gradual rise in abusive content in recent decades, reflecting social norms and regulatory policy changes. Genres such as thrillers still present a higher frequency of abusive content that emphasises the ongoing narrative role of violence and conflict. At the same time, underlying positive emotions such as humour and optimism remain prevalent in most of the movies. Furthermore, the gradual increase of abusive content in movie dialogues has been significant over the last two decades, where Oscar-nominated movies overtook the top ten blockbusters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17506v1">RAG-Enhanced Collaborative LLM Agents for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 Machine Learning, Drug Discovery
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have shown great potential to accelerate drug discovery. However, the specialized nature of biochemical data often necessitates costly domain-specific fine-tuning, posing critical challenges. First, it hinders the application of more flexible general-purpose LLMs in cutting-edge drug discovery tasks. More importantly, it impedes the rapid integration of the vast amounts of scientific data continuously generated through experiments and research. To investigate these challenges, we propose CLADD, a retrieval-augmented generation (RAG)-empowered agentic system tailored to drug discovery tasks. Through the collaboration of multiple LLM agents, CLADD dynamically retrieves information from biomedical knowledge bases, contextualizes query molecules, and integrates relevant evidence to generate responses -- all without the need for domain-specific fine-tuning. Crucially, we tackle key obstacles in applying RAG workflows to biochemical data, including data heterogeneity, ambiguity, and multi-source integration. We demonstrate the flexibility and effectiveness of this framework across a variety of drug discovery tasks, showing that it outperforms general-purpose and domain-specific LLMs as well as traditional deep learning approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18513v1">Analyzing User Perceptions of Large Language Models (LLMs) on Reddit: Sentiment and Topic Modeling of ChatGPT and DeepSeek Discussions</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      While there is an increased discourse on large language models (LLMs) like ChatGPT and DeepSeek, there is no comprehensive understanding of how users of online platforms, like Reddit, perceive these models. This is an important omission because public opinion can influence AI development, trust, and future policy. This study aims at analyzing Reddit discussions about ChatGPT and DeepSeek using sentiment and topic modeling to advance the understanding of user attitudes. Some of the significant topics such as trust in AI, user expectations, potential uses of the tools, reservations about AI biases, and ethical implications of their use are explored in this study. By examining these concerns, the study provides a sense of how public sentiment might shape the direction of AI development going forward. The report also mentions whether users have faith in the technology and what they see as its future. A word frequency approach is used to identify broad topics and sentiment trends. Also, topic modeling through the Latent Dirichlet Allocation (LDA) method identifies top topics in users' language, for example, potential benefits of LLMs, their technological applications, and their overall social ramifications. The study aims to inform developers and policymakers by making it easier to see how users comprehend and experience these game-changing technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09673v2">Are Smarter LLMs Safer? Exploring Safety-Reasoning Trade-offs in Prompting and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various NLP benchmarks. However, excelling in complex tasks that require nuanced reasoning and precise decision-making demands more than raw language proficiency--LLMs must reason, i.e., think logically, draw from past experiences, and synthesize information to reach conclusions and take action. To enhance reasoning abilities, approaches such as prompting and fine-tuning have been widely explored. While these methods have led to clear improvements in reasoning, their impact on LLM safety remains less understood. In this work, we investigate the interplay between reasoning and safety in LLMs. We highlight the latent safety risks that arise as reasoning capabilities improve, shedding light on previously overlooked vulnerabilities. At the same time, we explore how reasoning itself can be leveraged to enhance safety, uncovering potential mitigation strategies. By examining both the risks and opportunities in reasoning-driven LLM safety, our study provides valuable insights for developing models that are not only more capable but also more trustworthy in real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13417v2">RLTHF: Targeted Human Feedback for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF's strategic data curation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15155v1">Extreme Speech Classification in the Era of LLMs: Exploring Open-Source and Proprietary Models</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted to 7th International Conference on information systems and management science (ISMS), 2024
    </div>
    <details class="paper-abstract">
      In recent years, widespread internet adoption and the growth in userbase of various social media platforms have led to an increase in the proliferation of extreme speech online. While traditional language models have demonstrated proficiency in distinguishing between neutral text and non-neutral text (i.e. extreme speech), categorizing the diverse types of extreme speech presents significant challenges. The task of extreme speech classification is particularly nuanced, as it requires a deep understanding of socio-cultural contexts to accurately interpret the intent of the language used by the speaker. Even human annotators often disagree on the appropriate classification of such content, emphasizing the complex and subjective nature of this task. The use of human moderators also presents a scaling issue, necessitating the need for automated systems for extreme speech classification. The recent launch of ChatGPT has drawn global attention to the potential applications of Large Language Models (LLMs) across a diverse variety of tasks. Trained on vast and diverse corpora, and demonstrating the ability to effectively capture and encode contextual information, LLMs emerge as highly promising tools for tackling this specific task of extreme speech classification. In this paper, we leverage the Indian subset of the extreme speech dataset from Maronikolakis et al. (2022) to develop an effective classification framework using LLMs. We evaluate open-source Llama models against closed-source OpenAI models, finding that while pre-trained LLMs show moderate efficacy, fine-tuning with domain-specific data significantly enhances performance, highlighting their adaptability to linguistic and contextual nuances. Although GPT-based models outperform Llama models in zero-shot settings, the performance gap disappears after fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15153v1">Investigating the Adaptive Robustness with Knowledge Conflicts in LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have upgraded them from sophisticated text generators to autonomous agents capable of corporation and tool use in multi-agent systems (MASs). However, the robustness of these LLM-based MASs, especially under knowledge conflicts, remains unclear. In this paper, we design four comprehensive metrics to investigate the robustness of MASs when facing mild or task-critical knowledge conflicts. We first analyze mild knowledge conflicts introduced by heterogeneous agents and find that they do not harm system robustness but instead improve collaborative decision-making. Next, we investigate task-critical knowledge conflicts by synthesizing knowledge conflicts and embedding them into one of the agents. Our results show that these conflicts have surprisingly little to no impact on MAS robustness. Furthermore, we observe that MASs demonstrate certain self-repairing capabilities by reducing their reliance on knowledge conflicts and adopting alternative solution paths to maintain stability. Finally, we conduct ablation studies on the knowledge conflict number, agent number, and interaction rounds, finding that the self-repairing capability of MASs has intrinsic limits, and all findings hold consistently across various factors. Our code is publicly available at https://github.com/wbw625/MultiAgentRobustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08877v5">Aligning the Objective of LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted by ICSE'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved decent results on automated program repair (APR). However, the next token prediction training objective of decoder-only LLMs (e.g., GPT-4) is misaligned with the masked span prediction objective of current infilling-style methods, which impedes LLMs from fully leveraging pre-trained knowledge for program repair. In addition, while some LLMs can locate and repair bugs in certain functions using the related artifacts (e.g., test cases), existing methods still depend on statement-level fault localization methods to provide a list of buggy hunks for repair. This restriction hinders LLMs from exploring potential patches beyond the given locations. In this paper, we investigate a new approach to adapt LLMs to program repair. Our core insight is that LLM's APR capability can be greatly improved by simply aligning the output to their training objective and allowing them to refine the whole program without first identifying faulty statements. Based on this insight, we designed D4C, a straightforward prompting framework for APR. D4C can repair 180 bugs correctly in Defects4J, with each patch being sampled only 10 times. This surpasses the SOTA APR methods with perfect fault localization by 10% and reduces the patch sampling number by 90%. Our findings reveal that (1) objective alignment is crucial for fully exploiting LLM's pre-trained capability, and (2) replacing the traditional localize-buggy-hunks-then-repair workflow with direct debugging is more effective for LLM-based APR methods. Thus, we believe this paper introduces a new mindset for harnessing LLMs in APR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15140v1">Do LLMs Make Mistakes Like Students? Exploring Natural Alignment between Language Models and Human Error Patterns</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various educational tasks, yet their alignment with human learning patterns, particularly in predicting which incorrect options students are most likely to select in multiple-choice questions (MCQs), remains underexplored. Our work investigates the relationship between LLM generation likelihood and student response distributions in MCQs with a specific focus on distractor selections. We collect a comprehensive dataset of MCQs with real-world student response distributions to explore two fundamental research questions: (1). RQ1 - Do the distractors that students more frequently select correspond to those that LLMs assign higher generation likelihood to? (2). RQ2 - When an LLM selects a incorrect choice, does it choose the same distractor that most students pick? Our experiments reveals moderate correlations between LLM-assigned probabilities and student selection patterns for distractors in MCQs. Additionally, when LLMs make mistakes, they are more likley to select the same incorrect answers that commonly mislead students, which is a pattern consistent across both small and large language models. Our work provides empirical evidence that despite LLMs' strong performance on generating educational content, there remains a gap between LLM's underlying reasoning process and human cognitive processes in identifying confusing distractors. Our findings also have significant implications for educational assessment development. The smaller language models could be efficiently utilized for automated distractor generation as they demonstrate similar patterns in identifying confusing answer choices as larger language models. This observed alignment between LLMs and student misconception patterns opens new opportunities for generating high-quality distractors that complement traditional human-designed distractors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02486v2">Encryption-Friendly LLM Architecture</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer personalized responses based on user interactions, but this use case raises serious privacy concerns. Homomorphic encryption (HE) is a cryptographic protocol supporting arithmetic computations in encrypted states and provides a potential solution for privacy-preserving machine learning (PPML). However, the computational intensity of transformers poses challenges for applying HE to LLMs. In this work, we propose a modified HE-friendly transformer architecture with an emphasis on inference following personalized (private) fine-tuning. Utilizing LoRA fine-tuning and Gaussian kernels, we achieve significant computational speedups -- 6.94x for fine-tuning and 2.3x for inference -- while maintaining performance comparable to plaintext models. Our findings provide a viable proof of concept for offering privacy-preserving LLM services in areas where data protection is crucial. Our code is available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02355v2">"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Quantization is a powerful tool for accelerating large language model (LLM) inference, but the accuracy-performance trade-offs across different formats remain unclear. In this paper, we conduct the most comprehensive empirical study to date, evaluating FP8, INT8, and INT4 quantization across academic benchmarks and real-world tasks on the entire Llama-3.1 model family. Through over 500,000 evaluations, our investigation yields several key findings: (1) FP8 (W8A8-FP) is effectively lossless across all model scales, (2) well-tuned INT8 (W8A8-INT) achieves surprisingly low (1-3\%) accuracy degradation, and (3) INT4 weight-only (W4A16-INT) is more competitive than expected, rivaling 8-bit quantization. Further, we investigate the optimal quantization format for different deployments by analyzing inference performance through the popular vLLM framework. Our analysis provides clear deployment recommendations: W4A16 is the most cost-efficient for synchronous setups, while W8A8 dominates in asynchronous continuous batching. For mixed workloads, the optimal choice depends on the specific use case. Our findings offer practical, data-driven guidelines for deploying quantized LLMs at scale -- ensuring the best balance between speed, efficiency, and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15980v1">Text-to-SQL Domain Adaptation via Human-LLM Collaborative Data Annotation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted by IUI'25
    </div>
    <details class="paper-abstract">
      Text-to-SQL models, which parse natural language (NL) questions to executable SQL queries, are increasingly adopted in real-world applications. However, deploying such models in the real world often requires adapting them to the highly specialized database schemas used in specific applications. We find that existing text-to-SQL models experience significant performance drops when applied to new schemas, primarily due to the lack of domain-specific data for fine-tuning. This data scarcity also limits the ability to effectively evaluate model performance in new domains. Continuously obtaining high-quality text-to-SQL data for evolving schemas is prohibitively expensive in real-world scenarios. To bridge this gap, we propose SQLsynth, a human-in-the-loop text-to-SQL data annotation system. SQLsynth streamlines the creation of high-quality text-to-SQL datasets through human-LLM collaboration in a structured workflow. A within-subjects user study comparing SQLsynth with manual annotation and ChatGPT shows that SQLsynth significantly accelerates text-to-SQL data annotation, reduces cognitive load, and produces datasets that are more accurate, natural, and diverse. Our code is available at https://github.com/adobe/nl_sql_analyzer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09797v2">A Survey on LLM-based News Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      News recommender systems play a critical role in mitigating the information overload problem. In recent years, due to the successful applications of large language model technologies, researchers have utilized Discriminative Large Language Models (DLLMs) or Generative Large Language Models (GLLMs) to improve the performance of news recommender systems. Although several recent surveys review significant challenges for deep learning-based news recommender systems, such as fairness, privacy-preserving, and responsibility, there is a lack of a systematic survey on Large Language Model (LLM)-based news recommender systems. In order to review different core methodologies and explore potential issues systematically, we categorize DLLM-based and GLLM-based news recommender systems under the umbrella of LLM-based news recommender systems. In this survey, we first overview the development of deep learning-based news recommender systems. Then, we review LLM-based news recommender systems based on three aspects: news-oriented modeling, user-oriented modeling, and prediction-oriented modeling. Next, we examine the challenges from various perspectives, including datasets, benchmarking tools, and methodologies. Furthermore, we conduct extensive experiments to analyze how large language model technologies affect the performance of different news recommender systems. Finally, we comprehensively explore the future directions for LLM-based news recommendations in the era of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15963v1">Accountability in Code Review: The Role of Intrinsic Drivers and the Impact of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 45 pages; accepted at ACM TOSEM 2025
    </div>
    <details class="paper-abstract">
      Accountability is an innate part of social systems. It maintains stability and ensures positive pressure on individuals' decision-making. As actors in a social system, software developers are accountable to their team and organization for their decisions. However, the drivers of accountability and how it changes behavior in software development are less understood. In this study, we look at how the social aspects of code review affect software engineers' sense of accountability for code quality. Since software engineering (SE) is increasingly involving Large Language Models (LLM) assistance, we also evaluate the impact on accountability when introducing LLM-assisted code reviews. We carried out a two-phased sequential qualitative study (interviews -> focus groups). In Phase I (16 interviews), we sought to investigate the intrinsic drivers of software engineers influencing their sense of accountability for code quality, relying on self-reported claims. In Phase II, we tested these traits in a more natural setting by simulating traditional peer-led reviews with focus groups and then LLM-assisted review sessions. We found that there are four key intrinsic drivers of accountability for code quality: personal standards, professional integrity, pride in code quality, and maintaining one's reputation. In a traditional peer-led review, we observed a transition from individual to collective accountability when code reviews are initiated. We also found that the introduction of LLM-assisted reviews disrupts this accountability process, challenging the reciprocity of accountability taking place in peer-led evaluations, i.e., one cannot be accountable to an LLM. Our findings imply that the introduction of AI into SE must preserve social integrity and collective accountability mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01137v5">Explain Like I'm Five: Using LLMs to Improve PDE Surrogate Models with Text</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 22 pages, 15 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Solving Partial Differential Equations (PDEs) is ubiquitous in science and engineering. Computational complexity and difficulty in writing numerical solvers has motivated the development of data-driven machine learning techniques to generate solutions quickly. The recent rise in popularity of Large Language Models (LLMs) has enabled easy integration of text in multimodal machine learning models, allowing easy integration of additional system information such as boundary conditions and governing equations through text. In this work, we explore using pretrained LLMs to integrate various amounts of known system information into PDE learning. Using FactFormer as our testing backbone, we add a multimodal block to fuse numerical and textual information. We compare sentence-level embeddings, word-level embeddings, and a standard tokenizer across 2D Heat, Burgers, Navier-Stokes, and Shallow-Water data sets. These challenging benchmarks show that pretrained LLMs are able to utilize text descriptions of system information and enable accurate prediction using only initial conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15944v1">AutoMedPrompt: A New Framework for Optimizing LLM Medical Prompts Using Textual Gradients</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated increasingly sophisticated performance in medical and other fields of knowledge. Traditional methods of creating specialist LLMs require extensive fine-tuning and training of models on large datasets. Recently, prompt engineering, instead of fine-tuning, has shown potential to boost the performance of general foundation models. However, prompting methods such as chain-of-thought (CoT) may not be suitable for all subspecialty, and k-shot approaches may introduce irrelevant tokens into the context space. We present AutoMedPrompt, which explores the use of textual gradients to elicit medically relevant reasoning through system prompt optimization. AutoMedPrompt leverages TextGrad's automatic differentiation via text to improve the ability of general foundation LLMs. We evaluated AutoMedPrompt on Llama 3, an open-source LLM, using several QA benchmarks, including MedQA, PubMedQA, and the nephrology subspecialty-specific NephSAP. Our results show that prompting with textual gradients outperforms previous methods on open-source LLMs and surpasses proprietary models such as GPT-4, Claude 3 Opus, and Med-PaLM 2. AutoMedPrompt sets a new state-of-the-art (SOTA) performance on PubMedQA with an accuracy of 82.6$\%$, while also outperforming previous prompting strategies on open-sourced models for MedQA (77.7$\%$) and NephSAP (63.8$\%$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15939v1">"Kya family planning after marriage hoti hai?": Integrating Cultural Sensitivity in an LLM Chatbot for Reproductive Health</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted to ACM CHI Conference on Human Factors in Computing Systems (CHI '25)
    </div>
    <details class="paper-abstract">
      Access to sexual and reproductive health information remains a challenge in many communities globally, due to cultural taboos and limited availability of healthcare providers. Public health organizations are increasingly turning to Large Language Models (LLMs) to improve access to timely and personalized information. However, recent HCI scholarship indicates that significant challenges remain in incorporating context awareness and mitigating bias in LLMs. In this paper, we study the development of a culturally-appropriate LLM-based chatbot for reproductive health with underserved women in urban India. Through user interactions, focus groups, and interviews with multiple stakeholders, we examine the chatbot's response to sensitive and highly contextual queries on reproductive health. Our findings reveal strengths and limitations of the system in capturing local context, and complexities around what constitutes "culture". Finally, we discuss how local context might be better integrated, and present a framework to inform the design of culturally-sensitive chatbots for community health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15938v1">Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      LLMs are commonly trained with a learning rate (LR) warmup, followed by cosine decay to 10% of the maximum (10x decay). In a large-scale empirical study, we show that under an optimal peak LR, a simple linear decay-to-zero (D2Z) schedule consistently outperforms other schedules when training at compute-optimal dataset sizes. D2Z is superior across a range of model sizes, batch sizes, datasets, and vocabularies. Benefits increase as dataset size increases. Leveraging a novel interpretation of AdamW as an exponential moving average of weight updates, we show how linear D2Z optimally balances the demands of early training (moving away from initial conditions) and late training (averaging over more updates in order to mitigate gradient noise). In experiments, a 610M-parameter model trained for 80 tokens-per-parameter (TPP) using D2Z achieves lower loss than when trained for 200 TPP using 10x decay, corresponding to an astonishing 60% compute savings. Models such as Llama2-7B, trained for 286 TPP with 10x decay, could likely have saved a majority of compute by training with D2Z.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15932v1">CVE-LLM : Ontology-Assisted Automatic Vulnerability Evaluation Using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 arXiv admin note: substantial text overlap with arXiv:2407.14640
    </div>
    <details class="paper-abstract">
      The National Vulnerability Database (NVD) publishes over a thousand new vulnerabilities monthly, with a projected 25 percent increase in 2024, highlighting the crucial need for rapid vulnerability identification to mitigate cybersecurity attacks and save costs and resources. In this work, we propose using large language models (LLMs) to learn vulnerability evaluation from historical assessments of medical device vulnerabilities in a single manufacturer's portfolio. We highlight the effectiveness and challenges of using LLMs for automatic vulnerability evaluation and introduce a method to enrich historical data with cybersecurity ontologies, enabling the system to understand new vulnerabilities without retraining the LLM. Our LLM system integrates with the in-house application - Cybersecurity Management System (CSMS) - to help Siemens Healthineers (SHS) product cybersecurity experts efficiently assess the vulnerabilities in our products. Also, we present guidelines for efficient integration of LLMs into the cybersecurity tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15908v1">LLMs in Mobile Apps: Practices, Challenges, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      The integration of AI techniques has become increasingly popular in software development, enhancing performance, usability, and the availability of intelligent features. With the rise of large language models (LLMs) and generative AI, developers now have access to a wealth of high-quality open-source models and APIs from closed-source providers, enabling easier experimentation and integration of LLMs into various systems. This has also opened new possibilities in mobile application (app) development, allowing for more personalized and intelligent apps. However, integrating LLM into mobile apps might present unique challenges for developers, particularly regarding mobile device constraints, API management, and code infrastructure. In this project, we constructed a comprehensive dataset of 149 LLM-enabled Android apps and conducted an exploratory analysis to understand how LLMs are deployed and used within mobile apps. This analysis highlights key characteristics of the dataset, prevalent integration strategies, and common challenges developers face. Our findings provide valuable insights for future research and tooling development aimed at enhancing LLM-enabled mobile apps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15902v1">IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide explainable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and a Distinguisher that examines how well the input texts align with the predicted prompts. We develop and examine two versions of Distinguishers. Empirical evaluations demonstrate that both Distinguishers perform significantly better than the baseline methods, with version2 outperforming baselines by 9.73% on in-distribution data (F1-score) and 12.65% on OOD data (AUROC). Furthermore, a user study is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15865v1">Position: Standard Benchmarks Fail -- LLM Agents Present Overlooked Risks for Financial Applications</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 40 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Current financial LLM agent benchmarks are inadequate. They prioritize task performance while ignoring fundamental safety risks. Threats like hallucinations, temporal misalignment, and adversarial vulnerabilities pose systemic risks in high-stakes financial environments, yet existing evaluation frameworks fail to capture these risks. We take a firm position: traditional benchmarks are insufficient to ensure the reliability of LLM agents in finance. To address this, we analyze existing financial LLM agent benchmarks, finding safety gaps and introducing ten risk-aware evaluation metrics. Through an empirical evaluation of both API-based and open-weight LLM agents, we reveal hidden vulnerabilities that remain undetected by conventional assessments. To move the field forward, we propose the Safety-Aware Evaluation Agent (SAEA), grounded in a three-level evaluation framework that assesses agents at the model level (intrinsic capabilities), workflow level (multi-step process reliability), and system level (integration robustness). Our findings highlight the urgent need to redefine LLM agent evaluation standards by shifting the focus from raw performance to safety, robustness, and real world resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15860v1">Synthetic vs. Gold: The Role of LLM-Generated Labels and Data in Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      This study investigates the role of LLM-generated synthetic data in cyberbullying detection. We conduct a series of experiments where we replace some or all of the authentic data with synthetic data, or augment the authentic data with synthetic data. We find that synthetic cyberbullying data can be the basis for training a classifier for harm detection that reaches performance close to that of a classifier trained with authentic data. Combining authentic with synthetic data shows improvements over the baseline of training on authentic data alone for the test data for all three LLMs tried. These results highlight the viability of synthetic data as a scalable, ethically viable alternative in cyberbullying detection while emphasizing the critical impact of LLM selection on performance outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03101v3">KnowAgent: Knowledge-Augmented Planning for LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 NAACL 2025 Findings. Project page: https://zjunlp.github.io/project/KnowAgent/ Code: https://github.com/zjunlp/KnowAgent
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential in complex reasoning tasks, yet they fall short when tackling more sophisticated challenges, especially when interacting with environments through generating executable actions. This inadequacy primarily stems from the lack of built-in action knowledge in language agents, which fails to effectively guide the planning trajectories during task solving and results in planning hallucination. To address this issue, we introduce KnowAgent, a novel approach designed to enhance the planning capabilities of LLMs by incorporating explicit action knowledge. Specifically, KnowAgent employs an action knowledge base and a knowledgeable self-learning strategy to constrain the action path during planning, enabling more reasonable trajectory synthesis, and thereby enhancing the planning performance of language agents. Experimental results on HotpotQA and ALFWorld based on various backbone models demonstrate that KnowAgent can achieve comparable or superior performance to existing baselines. Further analysis indicates the effectiveness of KnowAgent in terms of planning hallucinations mitigation. Code is available in https://github.com/zjunlp/KnowAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13148v3">SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 In v2 we have revised the related work, added more comprehensive citations, and clarified our key contributions
    </div>
    <details class="paper-abstract">
      Adaptive optimizers such as Adam (Kingma & Ba, 2015) have been central to the success of large language models. However, they often require to maintain optimizer states throughout training, which can result in memory requirements several times greater than the model footprint. This overhead imposes constraints on scalability and computational efficiency. Stochastic Gradient Descent (SGD), in contrast, is a stateless optimizer, as it does not track state variables during training. Consequently, it achieves optimal memory efficiency. However, its capability in LLM training is limited (Zhao et al., 2024b). In this work, we show that pre-processing SGD in a stateless manner can achieve the same performance as the Adam optimizer for LLM training, while drastically reducing the memory cost. Specifically, we propose to pre-process the instantaneous stochastic gradients using normalization and whitening. We show that normalization stabilizes gradient distributions, and whitening counteracts the local curvature of the loss landscape. This results in SWAN (SGD with Whitening And Normalization), a stochastic optimizer that eliminates the need to store any optimizer states. Empirically, SWAN has the same memory footprint as SGD, achieving $\approx 50\%$ reduction on total end-to-end memory compared to Adam. In language modeling tasks, SWAN demonstrates comparable or even better performance than Adam: when pre-training the LLaMA model with 350M and 1.3B parameters, SWAN achieves a 2x speedup by reaching the same evaluation perplexity using half as many tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15652v1">Empowering LLMs with Logical Reasoning: A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable successes on various natural language tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs. This paper summarizes and categorizes the main challenges into two aspects: (1) Logical question answering, LLMs often fail to generate the correct answer within complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises and constrains. (2) Logical consistency, LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art Macaw question-answering LLM answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose detailed taxonomies of these methods. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, pretraining, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistency, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extensions to modal logic to account for uncertainty, and efficient algorithms satisfying multiple logical consistencies simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06556v4">ProjectTest: A Project-level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Unit test generation has become a promising and important use case of LLMs. However, existing evaluation benchmarks for assessing LLM unit test generation capabilities focus on function- or class-level code rather than more practical and challenging project-level codebases. To address such limitation, we propose ProjectTest, a project-level benchmark for unit test generation covering Python, Java, and JavaScript. ProjectTest features 20 moderate-sized and high-quality projects per language. We evaluate nine frontier LLMs on ProjectTest and the results show that all frontier LLMs tested exhibit moderate performance on ProjectTest on Python and Java, highlighting the difficulty of ProjectTest. We also conduct a thorough error analysis, which shows that even frontier LLMs, such as Claude-3.5-Sonnet, have significant basic yet critical errors, including compilation and cascade errors. Motivated by this observation, we further evaluate all frontier LLMs under manual error-fixing and self-error-fixing scenarios to assess their potential when equipped with error-fixing mechanisms. Our code and dataset is available at \href{https://github.com/YiboWANG214/ProjectTest}{ProjectTest}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17141v4">Towards Automated Penetration Testing: Introducing LLM Benchmark, Analysis, and Improvements</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Main Paper 1-9 pages, Supplementary Materials: 10-17, 13 figures
    </div>
    <details class="paper-abstract">
      Hacking poses a significant threat to cybersecurity, inflicting billions of dollars in damages annually. To mitigate these risks, ethical hacking, or penetration testing, is employed to identify vulnerabilities in systems and networks. Recent advancements in large language models (LLMs) have shown potential across various domains, including cybersecurity. However, there is currently no comprehensive, open, automated, end-to-end penetration testing benchmark to drive progress and evaluate the capabilities of these models in security contexts. This paper introduces a novel open benchmark for LLM-based automated penetration testing, addressing this critical gap. We first evaluate the performance of LLMs, including GPT-4o and LLama 3.1-405B, using the state-of-the-art PentestGPT tool. Our findings reveal that while LLama 3.1 demonstrates an edge over GPT-4o, both models currently fall short of performing end-to-end penetration testing even with some minimal human assistance. Next, we advance the state-of-the-art and present ablation studies that provide insights into improving the PentestGPT tool. Our research illuminates the challenges LLMs face in each aspect of Pentesting, e.g. enumeration, exploitation, and privilege escalation. This work contributes to the growing body of knowledge on AI-assisted cybersecurity and lays the foundation for future research in automated penetration testing using large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15618v1">Probe Pruning: Accelerating LLMs through Dynamic Pruning via Model-Probing</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      We introduce Probe Pruning (PP), a novel framework for online, dynamic, structured pruning of Large Language Models (LLMs) applied in a batch-wise manner. PP leverages the insight that not all samples and tokens contribute equally to the model's output, and probing a small portion of each batch effectively identifies crucial weights, enabling tailored dynamic pruning for different batches. It comprises three main stages: probing, history-informed pruning, and full inference. In the probing stage, PP selects a small yet crucial set of hidden states, based on residual importance, to run a few model layers ahead. During the history-informed pruning stage, PP strategically integrates the probing states with historical states. Subsequently, it structurally prunes weights based on the integrated states and the PP importance score, a metric developed specifically to assess the importance of each weight channel in maintaining performance. In the final stage, full inference is conducted on the remaining weights. A major advantage of PP is its compatibility with existing models, as it operates without requiring additional neural network modules or fine-tuning. Comprehensive evaluations of PP on LLaMA-2/3 and OPT models reveal that even minimal probing-using just 1.5% of FLOPs-can substantially enhance the efficiency of structured pruning of LLMs. For instance, when evaluated on LLaMA-2-7B with WikiText2, PP achieves a 2.56 times lower ratio of performance degradation per unit of runtime reduction compared to the state-of-the-art method at a 40% pruning ratio. Our code is available at https://github.com/Qi-Le1/Probe_Pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13311v2">Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Intelligent tutoring agents powered by large language models (LLMs) have been increasingly explored to deliver personalized guidance in areas such as language learning and science education. However, their capabilities in guiding users to solve complex real-world tasks remain underexplored. To address this limitation, in this work, we focus on coding tutoring, a challenging problem that requires tutors to proactively guide students toward completing predefined coding tasks. We propose a novel agent workflow, Trace-and-Verify (TRAVER), which combines knowledge tracing to estimate a student's knowledge state and turn-by-turn verification to ensure effective guidance toward task completion. We introduce DICT, an automatic evaluation protocol that assesses tutor agents holistically using controlled student simulation and code generation tests. Extensive experiments reveal the challenges of coding tutoring and demonstrate that TRAVER achieves a significantly higher success rate. Although we use code tutoring as an example in this paper, our results and findings can be extended beyond coding, providing valuable insights into advancing tutoring agents for a variety of tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15604v1">Cross-Format Retrieval-Augmented Generation in XR with LLMs for Context-Aware Maintenance Assistance</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      This paper presents a detailed evaluation of a Retrieval-Augmented Generation (RAG) system that integrates large language models (LLMs) to enhance information retrieval and instruction generation for maintenance personnel across diverse data formats. We assessed the performance of eight LLMs, emphasizing key metrics such as response speed and accuracy, which were quantified using BLEU and METEOR scores. Our findings reveal that advanced models like GPT-4 and GPT-4o-mini significantly outperform their counterparts, particularly when addressing complex queries requiring multi-format data integration. The results validate the system's ability to deliver timely and accurate responses, highlighting the potential of RAG frameworks to optimize maintenance operations. Future research will focus on refining retrieval techniques for these models and enhancing response generation, particularly for intricate scenarios, ultimately improving the system's practical applicability in dynamic real-world environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15603v1">Do Multilingual LLMs Think In English?</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Main paper 9 pages; including appendix 48 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have multilingual capabilities and can solve tasks across various languages. However, we show that current LLMs make key decisions in a representation space closest to English, regardless of their input and output languages. Exploring the internal representations with a logit lens for sentences in French, German, Dutch, and Mandarin, we show that the LLM first emits representations close to English for semantically-loaded words before translating them into the target language. We further show that activation steering in these LLMs is more effective when the steering vectors are computed in English rather than in the language of the inputs and outputs. This suggests that multilingual LLMs perform key reasoning steps in a representation that is heavily shaped by English in a way that is not transparent to system users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15601v1">WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Constructing photorealistic virtual worlds has applications across various fields, but it often requires the extensive labor of highly trained professionals to operate conventional 3D modeling software. To democratize this process, we introduce WorldCraft, a system where large language model (LLM) agents leverage procedural generation to create indoor and outdoor scenes populated with objects, allowing users to control individual object attributes and the scene layout using intuitive natural language commands. In our framework, a coordinator agent manages the overall process and works with two specialized LLM agents to complete the scene creation: ForgeIt, which integrates an ever-growing manual through auto-verification to enable precise customization of individual objects, and ArrangeIt, which formulates hierarchical optimization problems to achieve a layout that balances ergonomic and aesthetic considerations. Additionally, our pipeline incorporates a trajectory control agent, allowing users to animate the scene and operate the camera through natural language interactions. Our system is also compatible with off-the-shelf deep 3D generators to enrich scene assets. Through evaluations and comparisons with state-of-the-art methods, we demonstrate the versatility of WorldCraft, ranging from single-object customization to intricate, large-scale interior and exterior scene designs. This system empowers non-professionals to bring their creative visions to life.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18328v2">Unveiling Scoring Processes: Dissecting the Differences between LLMs and Human Graders in Automatic Scoring</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted by Technology, Knowledge, and Learning (TKNL)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong potential in performing automatic scoring for constructed response assessments. While constructed responses graded by humans are usually based on given grading rubrics, the methods by which LLMs assign scores remain largely unclear. It is also uncertain how closely AI's scoring process mirrors that of humans or if it adheres to the same grading criteria. To address this gap, this paper uncovers the grading rubrics that LLMs used to score students' written responses to science tasks and their alignment with human scores. We also examine whether enhancing the alignments can improve scoring accuracy. Specifically, we prompt LLMs to generate analytic rubrics that they use to assign scores and study the alignment gap with human grading rubrics. Based on a series of experiments with various configurations of LLM settings, we reveal a notable alignment gap between human and LLM graders. While LLMs can adapt quickly to scoring tasks, they often resort to shortcuts, bypassing deeper logical reasoning expected in human grading. We found that incorporating high-quality analytical rubrics designed to reflect human grading logic can mitigate this gap and enhance LLMs' scoring accuracy. These results underscore the need for a nuanced approach when applying LLMs in science education and highlight the importance of aligning LLM outputs with human expectations to ensure efficient and accurate automatic scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15576v1">Interpreting and Steering LLMs with Mutual Information-based Explanations on Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Pre-print. 20 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at handling human queries, but they can occasionally generate flawed or unexpected responses. Understanding their internal states is crucial for understanding their successes, diagnosing their failures, and refining their capabilities. Although sparse autoencoders (SAEs) have shown promise for interpreting LLM internal representations, limited research has explored how to better explain SAE features, i.e., understanding the semantic meaning of features learned by SAE. Our theoretical analysis reveals that existing explanation methods suffer from the frequency bias issue, where they emphasize linguistic patterns over semantic concepts, while the latter is more critical to steer LLM behaviors. To address this, we propose using a fixed vocabulary set for feature interpretations and designing a mutual information-based objective, aiming to better capture the semantic meaning behind these features. We further propose two runtime steering strategies that adjust the learned feature activations based on their corresponding explanations. Empirical results show that, compared to baselines, our method provides more discourse-level explanations and effectively steers LLM behaviors to defend against jailbreak attacks. These findings highlight the value of explanations for steering LLM behaviors in downstream applications. We will release our code and data once accepted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05673v4">Flow of Reasoning:Training LLMs for Divergent Problem Solving with Minimal Examples</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      The ability to generate diverse solutions to a given problem is a hallmark of human creativity. This divergent reasoning is also crucial for machines, enhancing their robustness and enabling them to assist humans in many applications such as scientific discovery. However, existing approaches to multi-step reasoning with large language models (LLMs) have mostly focused only on reasoning accuracy, without further discovering more diverse valid solutions. For example, supervised fine-tuning can improve LLM reasoning quality, but requires extensive supervised data to capture the full range of possible solutions. Reward-maximization reinforcement learning aims to find limited highest-reward solutions while neglecting the solution diversity. To fill this gap, we propose Flow of Reasoning (FoR), an efficient diversity-seeking LLM finetuning method aimed at improving reasoning quality and diversity with minimal data. FoR formulates multi-step LLM reasoning as a Markovian flow on a DAG-structured reasoning graph. This formulation allows us to incorporate and adapt principled GFlowNet approaches, for finetuning LLMs to sample divergent paths with probabilities proportional to the (unnormalized) reward of target problems. Extensive experiments show that, with limited training examples (e.g., 15 examples), FoR enables the discovery of diverse, creative, high-quality solutions, greatly outperforming a wide range of existing inference and training methods across six challenging reasoning tasks, including BlocksWorld (embodied reasoning), Game24 (math puzzle solving), Rubik's Cube (spatial reasoning), 1D-ARC (abstraction reasoning), GSM8k (math reasoning), and ProntoQA (logical reasoning). Code is available at https://github.com/Yu-Fangxu/FoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21102v2">Exploring and Controlling Diversity in LLM-Agent Conversation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted for the AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration (v1); updated version (v2)
    </div>
    <details class="paper-abstract">
      Controlling diversity in LLM-agent world simulations is essential for maintaining stability in structured tasks while enabling variation where creativity is needed. However, we observe that dialogue diversity declines significantly over long-term simulation. To investigate the role of prompt design in conversational diversity, we modularized the utterance generation prompt and found that reducing the given information leads to more diverse outputs. Based on this insight, we propose Adaptive Prompt Pruning (APP), a novel method that allows users to control diversity through a single parameter, lambda. APP dynamically prunes the utterance generation prompt based on their attention weights and is compatible with traditional diversity control techniques. We demonstrate that APP effectively controls output diversity through extensive experiments, and propose a method to balance the control trade-offs. Additionally, we provide an in-depth analysis to offer insights into optimizing diversity control in multi-agent simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15526v1">Scaling Sparse and Dense Retrieval in Decoder-Only LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Scaling large language models (LLMs) has shown great potential for improving retrieval model performance; however, previous studies have mainly focused on dense retrieval trained with contrastive loss (CL), neglecting the scaling behavior of other retrieval paradigms and optimization techniques, such as sparse retrieval and knowledge distillation (KD). In this work, we conduct a systematic comparative study on how different retrieval paradigms (sparse vs. dense) and fine-tuning objectives (CL vs. KD vs. their combination) affect retrieval performance across different model scales. Using MSMARCO passages as the training dataset, decoder-only LLMs (Llama-3 series: 1B, 3B, 8B), and a fixed compute budget, we evaluate various training configurations on both in-domain (MSMARCO, TREC DL) and out-of-domain (BEIR) benchmarks. Our key findings reveal that: (1) Scaling behaviors emerge clearly only with CL, where larger models achieve significant performance gains, whereas KD-trained models show minimal improvement, performing similarly across the 1B, 3B, and 8B scales. (2) Sparse retrieval models consistently outperform dense retrieval across both in-domain (MSMARCO, TREC DL) and out-of-domain (BEIR) benchmarks, and they demonstrate greater robustness to imperfect supervised signals. (3) We successfully scale sparse retrieval models with the combination of CL and KD losses at 8B scale, achieving state-of-the-art (SOTA) results in all evaluation sets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15524v1">Towards Swift Serverless LLM Cold Starts with ParaServe</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      With the surge in number of large language models (LLMs), the industry turns to serverless computing for LLM inference serving. However, serverless LLM serving suffers from significant cold start latency and service level objective (SLO) violations due to the substantial model size, which leads to prolonged model fetching time from remote storage. We present ParaServe, a serverless LLM serving system that minimizes cold start latency through the novel use of pipeline parallelism. Our insight is that by distributing model parameters across multiple GPU servers, we can utilize their aggregated network bandwidth to concurrently fetch different parts of the model. ParaServe adopts a two-level hierarchical design. At the cluster level, ParaServe determines the optimal degree of parallelism based on user SLOs and carefully places GPU workers across servers to reduce network interference. At the worker level, ParaServe overlaps model fetching, loading, and runtime initialization to further accelerate cold starts. Additionally, ParaServe introduces pipeline consolidation, which merges parallel groups back to individual workers to maintain optimal performance for warm requests. Our comprehensive evaluations under diverse settings demonstrate that ParaServe reduces the cold start latency by up to 4.7x and improves SLO attainment by up to 1.74x compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15506v1">Construction and Evaluation of LLM-based agents for Semi-Autonomous penetration testing</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 7 pages, 4 tables and 1 figure
    </div>
    <details class="paper-abstract">
      With the emergence of high-performance large language models (LLMs) such as GPT, Claude, and Gemini, the autonomous and semi-autonomous execution of tasks has significantly advanced across various domains. However, in highly specialized fields such as cybersecurity, full autonomy remains a challenge. This difficulty primarily stems from the limitations of LLMs in reasoning capabilities and domain-specific knowledge. We propose a system that semi-autonomously executes complex cybersecurity workflows by employing multiple LLMs modules to formulate attack strategies, generate commands, and analyze results, thereby addressing the aforementioned challenges. In our experiments using Hack The Box virtual machines, we confirmed that our system can autonomously construct attack strategies, issue appropriate commands, and automate certain processes, thereby reducing the need for manual intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11242v2">LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      This study examines the growing use of Large Language Models (LLMs) in child-centered applications, highlighting safety and ethical concerns such as bias, harmful content, and cultural insensitivity. Despite their potential to enhance learning, there is a lack of standardized frameworks to mitigate these risks. Through a systematic literature review, we identify key parental and empirical concerns, including toxicity and ethical breaches in AI outputs. Moreover, to address these issues, this paper proposes a protection framework for safe Child-LLM interaction, incorporating metrics for content safety, behavioral ethics, and cultural sensitivity. The framework provides practical tools for evaluating LLM safety, offering guidance for developers, policymakers, and educators to ensure responsible AI deployment for children.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11910v2">Adversarial Alignment for LLMs Requires Simpler, Reproducible, and More Measurable Objectives</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Misaligned research objectives have considerably hindered progress in adversarial robustness research over the past decade. For instance, an extensive focus on optimizing target metrics, while neglecting rigorous standardized evaluation, has led researchers to pursue ad-hoc heuristic defenses that were seemingly effective. Yet, most of these were exposed as flawed by subsequent evaluations, ultimately contributing little measurable progress to the field. In this position paper, we illustrate that current research on the robustness of large language models (LLMs) risks repeating past patterns with potentially worsened real-world implications. To address this, we argue that realigned objectives are necessary for meaningful progress in adversarial alignment. To this end, we build on established cybersecurity taxonomy to formally define differences between past and emerging threat models that apply to LLMs. Using this framework, we illustrate that progress requires disentangling adversarial alignment into addressable sub-problems and returning to core academic principles, such as measureability, reproducibility, and comparability. Although the field presents significant challenges, the fresh start on adversarial robustness offers the unique opportunity to build on past experience while avoiding previous mistakes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15427v1">Adversarial Prompt Evaluation: Systematic Benchmarking of Guardrails Against Prompt Input Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 NeurIPS 2024, Safe Generative AI Workshop
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become integrated into everyday applications, ensuring their robustness and security is increasingly critical. In particular, LLMs can be manipulated into unsafe behaviour by prompts known as jailbreaks. The variety of jailbreak styles is growing, necessitating the use of external defences known as guardrails. While many jailbreak defences have been proposed, not all defences are able to handle new out-of-distribution attacks due to the narrow segment of jailbreaks used to align them. Moreover, the lack of systematisation around defences has created significant gaps in their practical application. In this work, we perform systematic benchmarking across 15 different defences, considering a broad swathe of malicious and benign datasets. We find that there is significant performance variation depending on the style of jailbreak a defence is subject to. Additionally, we show that based on current datasets available for evaluation, simple baselines can display competitive out-of-distribution performance compared to many state-of-the-art defences. Code is available at https://github.com/IBM/Adversarial-Prompt-Evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15419v1">Beyond Translation: LLM-Based Data Generation for Multilingual Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 15 pages, 1 figure, 18 tables
    </div>
    <details class="paper-abstract">
      Robust automatic fact-checking systems have the potential to combat online misinformation at scale. However, most existing research primarily focuses on English. In this paper, we introduce MultiSynFact, the first large-scale multilingual fact-checking dataset containing 2.2M claim-source pairs designed to support Spanish, German, English, and other low-resource languages. Our dataset generation pipeline leverages Large Language Models (LLMs), integrating external knowledge from Wikipedia and incorporating rigorous claim validation steps to ensure data quality. We evaluate the effectiveness of MultiSynFact across multiple models and experimental settings. Additionally, we open-source a user-friendly framework to facilitate further research in multilingual fact-checking and dataset generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15401v1">Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) can significantly enhance the complex reasoning capabilities of large language models (LLMs), with the key lying in the selection and ordering of demonstration examples. Previous methods typically relied on simple features to measure the relevance between examples. We argue that these features are not sufficient to reflect the intrinsic connections between examples. In this study, we propose a curriculum ICL strategy guided by problem-solving logic. We select demonstration examples by analyzing the problem-solving logic and order them based on curriculum learning. Specifically, we constructed a problem-solving logic instruction set based on the BREAK dataset and fine-tuned a language model to analyze the problem-solving logic of examples. Subsequently, we selected appropriate demonstration examples based on problem-solving logic and assessed their difficulty according to the number of problem-solving steps. In accordance with the principles of curriculum learning, we ordered the examples from easy to hard to serve as contextual prompts. Experimental results on multiple benchmarks indicate that our method outperforms previous ICL approaches in terms of performance and efficiency, effectively enhancing the complex reasoning capabilities of LLMs. Our project will be publicly available subsequently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15395v1">Beyond Tools: Understanding How Heavy Users Integrate LLMs into Everyday Tasks and Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for both everyday and specialized tasks. While HCI research focuses on domain-specific applications, little is known about how heavy users integrate LLMs into everyday decision-making. Through qualitative interviews with heavy LLM users (n=7) who employ these systems for both intuitive and analytical thinking tasks, our findings show that participants use LLMs for social validation, self-regulation, and interpersonal guidance, seeking to build self-confidence and optimize cognitive resources. These users viewed LLMs either as rational, consistent entities or average human decision-makers. Our findings suggest that heavy LLM users develop nuanced interaction patterns beyond simple delegation, highlighting the need to reconsider how we study LLM integration in decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11187v2">TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 LLMs, Benchmarking, Large Language Models, Bangla
    </div>
    <details class="paper-abstract">
      In this paper, we present TituLLMs, the first large pretrained Bangla LLMs, available in 1b and 3b parameter sizes. Due to computational constraints during both training and inference, we focused on smaller models. To train TituLLMs, we collected a pretraining dataset of approximately ~37 billion tokens. We extended the Llama-3.2 tokenizer to incorporate language- and culture-specific knowledge, which also enables faster training and inference. There was a lack of benchmarking datasets to benchmark LLMs for Bangla. To address this gap, we developed five benchmarking datasets. We benchmarked various LLMs, including TituLLMs, and demonstrated that TituLLMs outperforms its initial multilingual versions. However, this is not always the case, highlighting the complexities of language adaptation. Our work lays the groundwork for adapting existing multilingual open models to other low-resource languages. To facilitate broader adoption and further research, we have made the TituLLMs models and benchmarking datasets publicly available (https://huggingface.co/collections/hishab/titulm-llama-family-6718d31fc1b83529276f490a).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15361v1">Evaluating Social Biases in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      In the recent development of AI reasoning, large language models (LLMs) are trained to automatically generate chain-of-thought reasoning steps, which have demonstrated compelling performance on math and coding tasks. However, when bias is mixed within the reasoning process to form strong logical arguments, it could cause even more harmful results and further induce hallucinations. In this paper, we have evaluated the 8B and 32B variants of DeepSeek-R1 against their instruction tuned counterparts on the BBQ dataset, and investigated the bias that is elicited out and being amplified through reasoning steps. To the best of our knowledge, this empirical study is the first to assess bias issues in LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15335v1">Stepwise Informativeness Search for Improving LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Advances in Large Language Models (LLMs) have significantly improved multi-step reasoning through generating free-text rationales. However, recent studies show that LLMs tend to lose focus over the middle of long contexts. This raises concerns that as reasoning progresses, LLMs may overlook information in earlier steps when decoding subsequent steps, leading to generate unreliable and redundant rationales. To address this, we propose guiding LLMs to generate more accurate and concise step-by-step rationales by (1) proactively referencing information from underutilized prior steps, and (2) minimizing redundant information between new and existing steps. We introduce stepwise informativeness search, an inference-time tree search framework incorporating two selection heuristics: grounding-guided selection which prioritizes steps paying higher attention over underutilized steps; and novelty-guided selection which encourages steps with novel conclusions. During rationale generation, we use a self-grounding strategy that prompts LLMs to explicitly reference relevant prior steps to provide premises before deduction at each step. Experimental results on four reasoning datasets demonstrate that our approach improves reasoning accuracy by generating higher-quality rationales with reduced errors and redundancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15304v1">SVDq: 1.25-bit and 410x Key Cache Compression for LLM Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      For the efficient inference of Large Language Models (LLMs), the effective compression of key-value (KV) cache is essential. Three main types of KV cache compression techniques, namely sparsity, channel compression, and quantization, have been identified. This study presents SVDq, a Singular Value Decomposition (SVD) - based mixed precision quantization method for K cache. Initially, K cache is transformed into latent channels using SVD basis representations. Since the values in latent channels decay rapidly and become negligible after only a few latent channels, our method then incorporates importance-aware quantization and compression for latent channels. This enables the effective allocation of higher precision to more significant channels. Theoretically, we prove that SVDq results in quantization errors (x0.1 or even lower) that are much lower than those of per-channel key quantization in the original space. Our findings based on RULER and LongBench benchmarks demonstrate that SVDq can achieve an equivalent key cache precision as low as 1.25-bit. When combined with key sparsity, it can reach a key compression ratio of up to 410x for attention computation, all while maintaining comparable model performance. Notably, our method is nearly lossless for LongBench datasets. This indicates that SVDq enables high-precision low-bit quantization, providing a more efficient solution for KV cache compression in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15294v1">Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      The increasing context window size in large language models (LLMs) has improved their ability to handle complex, long-text tasks. However, as the conversation rounds continue, it is required to store a large amount of KV cache in GPU memory, which significantly affects the efficiency and even availability of the model serving systems. This paper analyzes dialogue data from real users and discovers that the LLM inference manifests a watershed layer, after which the distribution of round-level attention shows notable similarity. We propose Round Attention, a novel round-level attention mechanism that only recalls and computes the KV cache of the most relevant rounds. The experiments show that our method saves 55\% memory usage without compromising model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00750v2">Mitigating Tail Narrowing in LLM Self-Improvement via Socratic-Guided Sampling</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted to NAACL 2025 Main Conference. Codes are publicly available at https://github.com/Yiwen-Ding/Guided-Self-Improvement
    </div>
    <details class="paper-abstract">
      Self-improvement methods enable large language models (LLMs) to generate solutions themselves and iteratively train on filtered, high-quality rationales. This process proves effective and reduces the reliance on human supervision in LLMs' reasoning, but the performance soon plateaus. We delve into the process and find that models tend to over-sample on easy queries and under-sample on queries they have yet to master. As iterations proceed, this imbalance in sampling is exacerbated, leading to a long-tail distribution where solutions to difficult queries almost diminish. This phenomenon limits the performance gain of self-improving models. A straightforward solution is brute-force sampling to balance the distribution, which significantly raises computational costs. In this paper, we introduce Guided Self-Improvement (GSI), a strategy aimed at improving the efficiency of sampling challenging heavy-tailed data. It leverages Socratic-style guidance signals to help LLM reasoning with complex queries, reducing the exploration effort and minimizing computational overhead. Experiments on four models across diverse mathematical tasks show that GSI strikes a balance between performance and efficiency, while also being effective on held-out tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15266v1">A Training-free LLM-based Approach to General Chinese Character Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 25 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Chinese spelling correction (CSC) is a crucial task that aims to correct character errors in Chinese text. While conventional CSC focuses on character substitution errors caused by mistyping, two other common types of character errors, missing and redundant characters, have received less attention. These errors are often excluded from CSC datasets during the annotation process or ignored during evaluation, even when they have been annotated. This issue limits the practicality of the CSC task. To address this issue, we introduce the task of General Chinese Character Error Correction (C2EC), which focuses on all three types of character errors. We construct a high-quality C2EC benchmark by combining and manually verifying data from CCTC and Lemon datasets. We extend the training-free prompt-free CSC method to C2EC by using Levenshtein distance for handling length changes and leveraging an additional prompt-based large language model (LLM) to improve performance. Experiments show that our method enables a 14B-parameter LLM to be on par with models nearly 50 times larger on both conventional CSC and C2EC tasks, without any fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01830v3">PiCO: Peer Review in LLMs based on the Consistency Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Existing large language models (LLMs) evaluation methods typically focus on testing the performance on some closed-environment and domain-specific benchmarks with human annotations. In this paper, we explore a novel unsupervised evaluation direction, utilizing peer-review mechanisms to measure LLMs automatically. In this setting, both open-source and closed-source LLMs lie in the same environment, capable of answering unlabeled questions and evaluating each other, where each LLM's response score is jointly determined by other anonymous ones. To obtain the ability hierarchy among these models, we assign each LLM a learnable capability parameter to adjust the final ranking. We formalize it as a constrained optimization problem, intending to maximize the consistency of each LLM's capabilities and scores. The key assumption behind is that high-level LLM can evaluate others' answers more accurately than low-level ones, while higher-level LLM can also achieve higher response scores. Moreover, we propose three metrics called PEN, CIN, and LIS to evaluate the gap in aligning human rankings. We perform experiments on multiple datasets with these metrics, validating the effectiveness of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15233v1">A General Pseudonymization Framework for Cloud-Based LLMs: Replacing Privacy Information in Controlled Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 under review
    </div>
    <details class="paper-abstract">
      An increasing number of companies have begun providing services that leverage cloud-based large language models (LLMs), such as ChatGPT. However, this development raises substantial privacy concerns, as users' prompts are transmitted to and processed by the model providers. Among the various privacy protection methods for LLMs, those implemented during the pre-training and fine-tuning phrases fail to mitigate the privacy risks associated with the remote use of cloud-based LLMs by users. On the other hand, methods applied during the inference phrase are primarily effective in scenarios where the LLM's inference does not rely on privacy-sensitive information. In this paper, we outline the process of remote user interaction with LLMs and, for the first time, propose a detailed definition of a general pseudonymization framework applicable to cloud-based LLMs. The experimental results demonstrate that the proposed framework strikes an optimal balance between privacy protection and utility. The code for our method is available to the public at https://github.com/Mebymeby/Pseudonymization-Framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15229v1">User Experience with LLM-powered Conversational Recommendation Systems: A Case of Music Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) now allows users to actively interact with conversational recommendation systems (CRS) and build their own personalized recommendation services tailored to their unique needs and goals. This experience offers users a significantly higher level of controllability compared to traditional RS, enabling an entirely new dimension of recommendation experiences. Building on this context, this study explored the unique experiences that LLM-powered CRS can provide compared to traditional RS. Through a three-week diary study with 12 participants using custom GPTs for music recommendations, we found that LLM-powered CRS can (1) help users clarify implicit needs, (2) support unique exploration, and (3) facilitate a deeper understanding of musical preferences. Based on these findings, we discuss the new design space enabled by LLM-powered CRS and highlight its potential to support more personalized, user-driven recommendation experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12566v2">Exploring the Impact of Personality Traits on LLM Bias and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      With the different roles that AI is expected to play in human life, imbuing large language models (LLMs) with different personalities has attracted increasing research interests. While the "personification" enhances human experiences of interactivity and adaptability of LLMs, it gives rise to critical concerns about content safety, particularly regarding bias, sentiment and toxicity of LLM generation. This study explores how assigning different personality traits to LLMs affects the toxicity and biases of their outputs. Leveraging the widely accepted HEXACO personality framework developed in social psychology, we design experimentally sound prompts to test three LLMs' performance on three toxic and bias benchmarks. The findings demonstrate the sensitivity of all three models to HEXACO personality traits and, more importantly, a consistent variation in the biases, negative sentiment and toxicity of their output. In particular, adjusting the levels of several personality traits can effectively reduce bias and toxicity in model performance, similar to humans' correlations between personality traits and toxic behaviors. The findings highlight the additional need to examine content safety besides the efficiency of training or fine-tuning methods for LLM personification. They also suggest a potential for the adjustment of personalities to be a simple and low-cost method to conduct controlled text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15226v1">Understand User Opinions of Large Language Models via LLM-Powered In-the-Moment User Experience Interviews</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Which large language model (LLM) is better? Every evaluation tells a story, but what do users really think about current LLMs? This paper presents CLUE, an LLM-powered interviewer that conducts in-the-moment user experience interviews, right after users interacted with LLMs, and automatically gathers insights about user opinions from massive interview logs. We conduct a study with thousands of users to understand user opinions on mainstream LLMs, recruiting users to first chat with a target LLM and then interviewed by CLUE. Our experiments demonstrate that CLUE captures interesting user opinions, for example, the bipolar views on the displayed reasoning process of DeepSeek-R1 and demands for information freshness and multi-modality. Our collected chat-and-interview logs will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15224v1">Auto-Bench: An Automated Benchmark for Scientific Discovery in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Given the remarkable performance of Large Language Models (LLMs), an important question arises: Can LLMs conduct human-like scientific research and discover new knowledge, and act as an AI scientist? Scientific discovery is an iterative process that demands efficient knowledge updating and encoding. It involves understanding the environment, identifying new hypotheses, and reasoning about actions; however, no standardized benchmark specifically designed for scientific discovery exists for LLM agents. In response to these limitations, we introduce a novel benchmark, \textit{Auto-Bench}, that encompasses necessary aspects to evaluate LLMs for scientific discovery in both natural and social sciences. Our benchmark is based on the principles of causal graph discovery. It challenges models to uncover hidden structures and make optimal decisions, which includes generating valid justifications. By engaging interactively with an oracle, the models iteratively refine their understanding of underlying interactions, the chemistry and social interactions, through strategic interventions. We evaluate state-of-the-art LLMs, including GPT-4, Gemini, Qwen, Claude, and Llama, and observe a significant performance drop as the problem complexity increases, which suggests an important gap between machine and human intelligence that future development of LLMs need to take into consideration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15217v1">FormalSpecCpp: A Dataset of C++ Formal Specifications created using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 Accepted at the 2025 IEEE/ACM 22nd International Conference on Mining Software Repositories (MSR)
    </div>
    <details class="paper-abstract">
      FormalSpecCpp is a dataset designed to fill the gap in standardized benchmarks for verifying formal specifications in C++ programs. To the best of our knowledge, this is the first comprehensive collection of C++ programs with well-defined preconditions and postconditions. It provides a structured benchmark for evaluating specification inference tools and testing theaccuracy of generated specifications. Researchers and developers can use this dataset to benchmark specification inference tools,fine-tune Large Language Models (LLMs) for automated specification generation, and analyze the role of formal specifications in improving program verification and automated testing. By making this dataset publicly available, we aim to advance research in program verification, specification inference, and AI-assisted software development. The dataset and the code are available at https://github.com/MadhuNimmo/FormalSpecCpp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21255v3">AQUA: Network-Accelerated Memory Offloading for LLMs in Scale-Up GPU Domains</a></div>
    <div class="paper-meta">
      📅 2025-02-21
    </div>
    <details class="paper-abstract">
      Inference on large-language models (LLMs) is constrained by GPU memory capacity. A sudden increase in the number of inference requests to a cloud-hosted LLM can deplete GPU memory, leading to contention between multiple prompts for limited resources. Modern LLM serving engines deal with the challenge of limited GPU memory using admission control, which causes them to be unresponsive during request bursts. We propose that preemptive scheduling of prompts in time slices is essential for ensuring responsive LLM inference, especially under conditions of high load and limited GPU memory. However, preempting prompt inference incurs a high paging overhead, which reduces inference throughput. We present Aqua, a GPU memory management framework that significantly reduces the overhead of paging inference state, achieving both responsive and high-throughput inference even under bursty request patterns. We evaluate Aqua by hosting several state-of-the-art large generative ML models of different modalities on servers with 8 Nvidia H100 80G GPUs. Aqua improves the responsiveness of LLM inference by 20X compared to the state-of-the-art and improves LLM inference throughput over a single long prompt by 4X.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15172v1">BP-GPT: Auditory Neural Decoding Using fMRI-prompted LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-21
      | 💬 arXiv admin note: substantial text overlap with arXiv:2405.07840
    </div>
    <details class="paper-abstract">
      Decoding language information from brain signals represents a vital research area within brain-computer interfaces, particularly in the context of deciphering the semantic information from the fMRI signal. Although existing work uses LLM to achieve this goal, their method does not use an end-to-end approach and avoids the LLM in the mapping of fMRI-to-text, leaving space for the exploration of the LLM in auditory decoding. In this paper, we introduce a novel method, the Brain Prompt GPT (BP-GPT). By using the brain representation that is extracted from the fMRI as a prompt, our method can utilize GPT-2 to decode fMRI signals into stimulus text. Further, we introduce the text prompt and align the fMRI prompt to it. By introducing the text prompt, our BP-GPT can extract a more robust brain prompt and promote the decoding of pre-trained LLM. We evaluate our BP-GPT on the open-source auditory semantic decoding dataset and achieve a significant improvement up to 4.61 on METEOR and 2.43 on BERTScore across all the subjects compared to the state-of-the-art method. The experimental results demonstrate that using brain representation as a prompt to further drive LLM for auditory neural decoding is feasible and effective. The code is available at https://github.com/1994cxy/BP-GPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07355v2">Think Together and Work Better: Combining Humans' and LLMs' Think-Aloud Outcomes for Effective Text Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This study introduces \textbf{InteractEval}, a framework that integrates human expertise and Large Language Models (LLMs) using the Think-Aloud (TA) method to generate attributes for checklist-based text evaluation. By combining human flexibility and reasoning with LLM consistency, InteractEval outperforms traditional non-LLM-based and LLM-based baselines across four distinct dimensions, consisting of Coherence, Fluency, Consistency, and Relevance. The experiment also investigates the effectiveness of the TA method, showing that it promotes divergent thinking in both humans and LLMs, leading to the generation of a wider range of relevant attributes and enhance text evaluation performance. Comparative analysis reveals that humans excel at identifying attributes related to internal quality (Coherence and Fluency), but LLMs perform better at those attributes related to external alignment (Consistency and Relevance). Consequently, leveraging both humans and LLMs together produces the best evaluation outcomes. In other words, this study emphasizes the necessity of effectively combining humans and LLMs in an automated checklist-based text evaluation framework. The code is available at \textbf{\url{https://github.com/BBeeChu/InteractEval.git}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14202v1">Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The widespread adoption of conversational LLMs for software development has raised new security concerns regarding the safety of LLM-generated content. Our motivational study outlines ChatGPT's potential in volunteering context-specific information to the developers, promoting safe coding practices. Motivated by this finding, we conduct a study to evaluate the degree of security awareness exhibited by three prominent LLMs: Claude 3, GPT-4, and Llama 3. We prompt these LLMs with Stack Overflow questions that contain vulnerable code to evaluate whether they merely provide answers to the questions or if they also warn users about the insecure code, thereby demonstrating a degree of security awareness. Further, we assess whether LLM responses provide information about the causes, exploits, and the potential fixes of the vulnerability, to help raise users' awareness. Our findings show that all three models struggle to accurately detect and warn users about vulnerabilities, achieving a detection rate of only 12.6% to 40% across our datasets. We also observe that the LLMs tend to identify certain types of vulnerabilities related to sensitive information exposure and improper input neutralization much more frequently than other types, such as those involving external control of file names or paths. Furthermore, when LLMs do issue security warnings, they often provide more information on the causes, exploits, and fixes of vulnerabilities compared to Stack Overflow responses. Finally, we provide an in-depth discussion on the implications of our findings and present a CLI-based prompting tool that can be used to generate significantly more secure LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14192v1">NLP-AKG: Few-Shot Construction of NLP Academic Knowledge Graph Based on LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely applied in question answering over scientific research papers. To enhance the professionalism and accuracy of responses, many studies employ external knowledge augmentation. However, existing structures of external knowledge in scientific literature often focus solely on either paper entities or domain concepts, neglecting the intrinsic connections between papers through shared domain concepts. This results in less comprehensive and specific answers when addressing questions that combine papers and concepts. To address this, we propose a novel knowledge graph framework that captures deep conceptual relations between academic papers, constructing a relational network via intra-paper semantic elements and inter-paper citation relations. Using a few-shot knowledge graph construction method based on LLM, we develop NLP-AKG, an academic knowledge graph for the NLP domain, by extracting 620,353 entities and 2,271,584 relations from 60,826 papers in ACL Anthology. Based on this, we propose a 'sub-graph community summary' method and validate its effectiveness on three NLP scientific literature question answering datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14182v1">Multi-Faceted Studies on Data Poisoning can Advance LLM Development</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The lifecycle of large language models (LLMs) is far more complex than that of traditional machine learning models, involving multiple training stages, diverse data sources, and varied inference methods. While prior research on data poisoning attacks has primarily focused on the safety vulnerabilities of LLMs, these attacks face significant challenges in practice. Secure data collection, rigorous data cleaning, and the multistage nature of LLM training make it difficult to inject poisoned data or reliably influence LLM behavior as intended. Given these challenges, this position paper proposes rethinking the role of data poisoning and argue that multi-faceted studies on data poisoning can advance LLM development. From a threat perspective, practical strategies for data poisoning attacks can help evaluate and address real safety risks to LLMs. From a trustworthiness perspective, data poisoning can be leveraged to build more robust LLMs by uncovering and mitigating hidden biases, harmful outputs, and hallucinations. Moreover, from a mechanism perspective, data poisoning can provide valuable insights into LLMs, particularly the interplay between data and model behavior, driving a deeper understanding of their underlying mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10490v3">Learning Dynamics of LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique "squeezing effect" to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15097v1">LUME: LLM Unlearning with Multitask Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Unlearning aims to remove copyrighted, sensitive, or private content from large language models (LLMs) without a full retraining. In this work, we develop a multi-task unlearning benchmark (LUME) which features three tasks: (1) unlearn synthetically generated creative short novels, (2) unlearn synthetic biographies with sensitive information, and (3) unlearn a collection of public biographies. We further release two fine-tuned LLMs of 1B and 7B parameter sizes as the target models. We conduct detailed evaluations of several recently proposed unlearning algorithms and present results on carefully crafted metrics to understand their behavior and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05315v2">Aligned at the Start: Conceptual Groupings in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This paper shifts focus to the often-overlooked input embeddings - the initial representations fed into transformer blocks. Using fuzzy graph, k-nearest neighbor (k-NN), and community detection, we analyze embeddings from diverse LLMs, finding significant categorical community structure aligned with predefined concepts and categories aligned with humans. We observe these groupings exhibit within-cluster organization (such as hierarchies, topological ordering, etc.), hypothesizing a fundamental structure that precedes contextual processing. To further investigate the conceptual nature of these groupings, we explore cross-model alignments across different LLM categories within their input embeddings, observing a medium to high degree of alignment. Furthermore, provide evidence that manipulating these groupings can play a functional role in mitigating ethnicity bias in LLM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15090v1">Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to the study of representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., 'cat') and then analyze the corresponding activation patterns. Our findings reveal that LLM representations closely align with human representations inferred from behavioral data. Notably, this alignment surpasses that of word embeddings, which have been center stage in prior work on human and model alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts. Specifically, we show that LLMs organize concepts in a way that reflects hierarchical relationships interpretable to humans (e.g., 'animal'-'dog').
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04616v2">Can LLMs Improve Multimodal Fact-Checking by Asking Relevant Questions?</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Traditional fact-checking relies on humans to formulate relevant and targeted fact-checking questions (FCQs), search for evidence, and verify the factuality of claims. While Large Language Models (LLMs) have been commonly used to automate evidence retrieval and factuality verification at scale, their effectiveness for fact-checking is hindered by the absence of FCQ formulation. To bridge this gap, we seek to answer two research questions: (1) Can LLMs generate relevant FCQs? (2) Can LLM-generated FCQs improve multimodal fact-checking? We therefore introduce a framework LRQ-FACT for using LLMs to generate relevant FCQs to facilitate evidence retrieval and enhance fact-checking by probing information across multiple modalities. Through extensive experiments, we verify if LRQ-FACT can generate relevant FCQs of different types and if LRQ-FACT can consistently outperform baseline methods in multimodal fact-checking. Further analysis illustrates how each component in LRQ-FACT works toward improving the fact-checking performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v2">MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 32 pages, 11 figures. v2 updated the project page and dataset link
    </div>
    <details class="paper-abstract">
      Inductive Reasoning (IR), the ability to summarize rules from examples and apply on new ones, has long been viewed as a primal ability for general intelligence and widely studied by cognitive science and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually $<$10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations are mostly focused on classification (a very limited aspect of IR), and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context inductive reasoning benchmark that asks LLM to induce output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for inductive reasoning and many-shot ICL, including robustness against erroneous shots and the effect of Chain-of-Thought (CoT), and acquired insightful findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15018v1">Using tournaments to calculate AUROC for zero-shot classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models perform surprisingly well on many zero-shot classification tasks, but are difficult to fairly compare to supervised classifiers due to the lack of a modifiable decision boundary. In this work, we propose and evaluate a method that converts binary classification tasks into pairwise comparison tasks, obtaining relative rankings from LLMs. Repeated pairwise comparisons can be used to score instances using the Elo rating system (used in chess and other competitions), inducing a confidence ordering over instances in a dataset. We evaluate scheduling algorithms for their ability to minimize comparisons, and show that our proposed algorithm leads to improved classification performance, while also providing more information than traditional zero-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15009v1">Contextualizing Search Queries In-Context Learning for Conversational Rewriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Conversational query rewriting is crucial for effective conversational search, yet traditional supervised methods require substantial labeled data, which is scarce in low-resource settings. This paper introduces Prompt-Guided In-Context Learning, a novel approach that leverages the in-context learning capabilities of Large Language Models (LLMs) for few-shot conversational query rewriting. Our method employs carefully designed prompts, incorporating task descriptions, input/output format specifications, and a small set of illustrative examples, to guide pre-trained LLMs to generate context-independent queries without explicit fine-tuning. Extensive experiments on benchmark datasets, TREC and Taskmaster-1, demonstrate that our approach significantly outperforms strong baselines, including supervised models and contrastive co-training methods, across various evaluation metrics such as BLEU, ROUGE-L, Success Rate, and MRR. Ablation studies confirm the importance of in-context examples, and human evaluations further validate the superior fluency, relevance, and context utilization of our generated rewrites. The results highlight the potential of prompt-guided in-context learning as an efficient and effective paradigm for low-resource conversational query rewriting, reducing the reliance on extensive labeled data and complex training procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14461v2">From Human Annotation to LLMs: SILICON Annotation Workflow for Management Research</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Unstructured text data annotation and analysis are fundamental to management research, often relying on human annotators through crowdsourcing platforms. While Large Language Models (LLMs) promise to provide a cost-effective and efficient alternative to human annotation, there lacks a systematic workflow that evaluate when LLMs are suitable or how to proceed with LLM-based text annotation in a reproducible manner. This paper addresses this methodological gap by introducing the ``SILICON" (Systematic Inference with LLMs for Information Classification and Notation) workflow. The workflow integrates established principles of human annotation with systematic prompt optimization and model selection, addressing challenges such as developing robust annotation guidelines, establishing high-quality human baselines, optimizing prompts, and ensuring reproducibility across LLMs. We validate the SILICON workflow through seven case studies covering common management research tasks. Our findings highlight the importance of validating annotation guideline agreement, the superiority of expert-developed human baselines over crowdsourced ones, the iterative nature of prompt optimization, and the necessity of testing multiple LLMs. We also find that LLMs agree well with expert annotations in most cases but show low agreement in more complex multi-label classification tasks. Notably, we propose a regression-based methodology to empirically compare LLM outputs across prompts and models. Our workflow advances management research by establishing rigorous, transparent, and reproducible processes for LLM-based annotation. We provide practical guidance for researchers to effectively navigate the evolving landscape of generative AI tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v3">AlphaPO -- Reward shape matters for LLM alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce \textbf{AlphaPO}, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15\% to 50\% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15835v1">Pragmatic Reasoning improves LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive potential in translating natural language (NL) instructions into program code. However, user instructions often contain inherent ambiguities, making it challenging for LLMs to generate code that accurately reflects the user's true intent. To address this challenge, researchers have proposed to produce multiple candidates of the program code and then rerank them to identify the best solution. In this paper, we propose CodeRSA, a novel code candidate reranking mechanism built upon the Rational Speech Act (RSA) framework, designed to guide LLMs toward more comprehensive pragmatic reasoning about user intent. We evaluate CodeRSA using one of the latest LLMs on a popular code generation dataset. Our experiment results show that CodeRSA consistently outperforms common baselines, surpasses the state-of-the-art approach in most cases, and demonstrates robust overall performance. These findings underscore the effectiveness of integrating pragmatic reasoning into code candidate reranking, offering a promising direction for enhancing code generation quality in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14866v1">LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted by MLSys 2025. Code available at: https://github.com/mit-han-lab/omniserve
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable potential in processing long sequences, yet efficiently serving these long-context models remains challenging due to the quadratic computational complexity of attention in the prefilling stage and the large memory footprint of the KV cache in the decoding stage. To address these issues, we introduce LServe, an efficient system that accelerates long-sequence LLM serving via hybrid sparse attention. This method unifies different hardware-friendly, structured sparsity patterns for both prefilling and decoding attention into a single framework, where computations on less important tokens are skipped block-wise. LServe demonstrates the compatibility of static and dynamic sparsity in long-context LLM attention. This design enables multiplicative speedups by combining these optimizations. Specifically, we convert half of the attention heads to nearly free streaming heads in both the prefilling and decoding stages. Additionally, we find that only a constant number of KV pages is required to preserve long-context capabilities, irrespective of context length. We then design a hierarchical KV page selection policy that dynamically prunes KV pages based on query-centric similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is released at https://github.com/mit-han-lab/omniserve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14860v1">Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 22 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often fail to ask effective questions under uncertainty, making them unreliable in domains where proactive information-gathering is essential for decisionmaking. We present ALFA, a framework that improves LLM question-asking by (i) decomposing the notion of a "good" question into a set of theory-grounded attributes (e.g., clarity, relevance), (ii) controllably synthesizing attribute-specific question variations, and (iii) aligning models via preference-based optimization to explicitly learn to ask better questions along these fine-grained attributes. Focusing on clinical reasoning as a case study, we introduce the MediQ-AskDocs dataset, composed of 17k real-world clinical interactions augmented with 80k attribute-specific preference pairs of follow-up questions, as well as a novel expert-annotated interactive healthcare QA task to evaluate question-asking abilities. Models aligned with ALFA reduce diagnostic errors by 56.6% on MediQ-AskDocs compared to SOTA instruction-tuned LLMs, with a question-level win-rate of 64.4% and strong generalizability. Our findings suggest that explicitly guiding question-asking with structured, fine-grained attributes offers a scalable path to improve LLMs, especially in expert application domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14847v1">Red-Teaming LLM Multi-Agent Systems via Communication Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) have revolutionized complex problem-solving capability by enabling sophisticated agent collaboration through message-based communications. While the communication framework is crucial for agent coordination, it also introduces a critical yet unexplored security vulnerability. In this work, we introduce Agent-in-the-Middle (AiTM), a novel attack that exploits the fundamental communication mechanisms in LLM-MAS by intercepting and manipulating inter-agent messages. Unlike existing attacks that compromise individual agents, AiTM demonstrates how an adversary can compromise entire multi-agent systems by only manipulating the messages passing between agents. To enable the attack under the challenges of limited control and role-restricted communication format, we develop an LLM-powered adversarial agent with a reflection mechanism that generates contextually-aware malicious instructions. Our comprehensive evaluation across various frameworks, communication structures, and real-world applications demonstrates that LLM-MAS is vulnerable to communication-based attacks, highlighting the need for robust security measures in multi-agent systems.
    </details>
</div>
