# llm - 2025_02

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

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09345v2">Rational Tuning of LLM Cascades via Probabilistic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding the reliability of large language models (LLMs) has recently garnered significant attention. Given LLMs' propensity to hallucinate, as well as their high sensitivity to prompt design, it is already challenging to predict the performance of an individual LLM. However, the problem becomes more complex for compound LLM systems such as cascades, where in addition to each model's standalone performance, we must understand how the error rates of different models interact. In this paper, we present a probabilistic model for the joint performance distribution of a sequence of LLMs, which enables a framework for rationally tuning the confidence thresholds of a LLM cascade using continuous optimization. Compared to selecting confidence thresholds using grid search, our parametric Markov-copula model significantly improves runtime scaling with respect to the length of the cascade and the desired resolution of the cost-error curve, turning them from intractable into low-order polynomial. In addition, the optimal thresholds computed using our continuous optimization-based algorithm increasingly outperform those found via grid search as cascade length grows, improving the area under the cost-error curve by 1.9% on average for cascades consisting of at least three models. Overall, our Markov-copula model provides a rational basis for tuning LLM cascade performance and points to the potential of probabilistic methods in analyzing LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14100v2">Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enhanced with external contexts, such as through retrieval-augmented generation (RAG), often face challenges in handling imperfect evidence. They tend to over-rely on external knowledge, making them vulnerable to misleading and unhelpful contexts. To address this, we propose the concept of context-robust LLMs, which can effectively balance internal knowledge with external context, similar to human cognitive processes. Specifically, context-robust LLMs should rely on external context only when lacking internal knowledge, identify contradictions between internal and external knowledge, and disregard unhelpful contexts. To achieve this goal, we introduce Grft, a lightweight and plug-and-play gated representation fine-tuning approach. Grft consists of two key components: a gating mechanism to detect and filter problematic inputs, and low-rank representation adapters to adjust hidden representations. By training a lightweight intervention function with only 0.0004\% of model size on fewer than 200 examples, Grft can effectively adapt LLMs towards context-robust behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13502v2">PLDR-LLMs Learn A Generalizable Tensor Operator That Can Replace Its Own Deep Neural Net At Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 15 pages, 1 figure, 12 tables, more ablation data included
    </div>
    <details class="paper-abstract">
      We show that Large Language Model from Power Law Decoder Representations (PLDR-LLM) is a foundational model whose deductive outputs are invariant tensors up to a small perturbation. PLDR-LLM learns a singularity condition for the deductive outputs that enable the once-inferred energy-curvature tensor $\mathbf{G}_{LM}$ to replace the deep neural network of power law graph attention (PLGA) generating the deductive outputs at inference. We demonstrate that a cache for $\mathbf{G}_{LM}$ (G-cache) and KV-cache can be implemented in a straightforward manner to improve the inference time. The invariance and generalizable nature of deductive outputs is at a very high fidelity where deductive outputs have same RMSE and determinant values up to 15 decimal places after caching, and zero-shot benchmark scores remain unchanged. Ablation studies show that learned deductive outputs have distinct loss and accuracy characteristics from models pretrained with transferred, randomly initialized or identity tensors as a constant tensor operator and an LLM with scaled-dot product attention (SDPA) is a special case of PLDR-LLM where $\mathbf{G}_{LM}$ is predefined as identity. The observed invariance characteristic introduces a novel asymmetry between training and inference phases with caching. We outline observed common characteristics of the deductive outputs for the learned singularity condition. We provide an implementation of a training and inference framework for PLDR-LLM with KV-cache and G-cache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16366v1">A generative approach to LLM harmfulness detection with special red flag tokens</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18527v2">Understanding Ranking LLMs: A Mechanistic Analysis for Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Transformer networks, particularly those achieving performance comparable to GPT models, are well known for their robust feature extraction abilities. However, the nature of these extracted features and their alignment with human-engineered ones remain unexplored. In this work, we investigate the internal mechanisms of state-of-the-art, fine-tuned LLMs for passage reranking. We employ a probing-based analysis to examine neuron activations in ranking LLMs, identifying the presence of known human-engineered and semantic features. Our study spans a broad range of feature categories, including lexical signals, document structure, query-document interactions, and complex semantic representations, to uncover underlying patterns influencing ranking decisions. Through experiments on four different ranking LLMs, we identify statistical IR features that are prominently encoded in LLM activations, as well as others that are notably missing. Furthermore, we analyze how these models respond to out-of-distribution queries and documents, revealing distinct generalization behaviors. By dissecting the latent representations within LLM activations, we aim to improve both the interpretability and effectiveness of ranking models. Our findings offer crucial insights for developing more transparent and reliable retrieval systems, and we release all necessary scripts and code to support further exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19568v3">Are LLMs Good Annotators for Discourse-level Event Relation Extraction?</a></div>
    <div class="paper-meta">
      📅 2025-02-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated proficiency in a wide array of natural language processing tasks. However, its effectiveness over discourse-level event relation extraction (ERE) tasks remains unexplored. In this paper, we assess the effectiveness of LLMs in addressing discourse-level ERE tasks characterized by lengthy documents and intricate relations encompassing coreference, temporal, causal, and subevent types. Evaluation is conducted using an commercial model, GPT-3.5, and an open-source model, LLaMA-2. Our study reveals a notable underperformance of LLMs compared to the baseline established through supervised learning. Although Supervised Fine-Tuning (SFT) can improve LLMs performance, it does not scale well compared to the smaller supervised baseline model. Our quantitative and qualitative analysis shows that LLMs have several weaknesses when applied for extracting event relations, including a tendency to fabricate event mentions, and failures to capture transitivity rules among relations, detect long distance relations, or comprehend contexts with dense event mentions. Code available at: https://github.com/WeiKangda/LLM-ERE.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06101v2">Coevolving with the Other You: Fine-Tuning LLM with Sequential Cooperative Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 Accepted by NeurIPS '24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a pivotal technique for fine-tuning large language models (LLMs) on specific tasks. However, prevailing RL fine-tuning methods predominantly rely on PPO and its variants. Though these algorithms are effective in general RL settings, they often exhibit suboptimal performance and vulnerability to distribution collapse when applied to the fine-tuning of LLMs. In this paper, we propose CORY, extending the RL fine-tuning of LLMs to a sequential cooperative multi-agent reinforcement learning framework, to leverage the inherent coevolution and emergent capabilities of multi-agent systems. In CORY, the LLM to be fine-tuned is initially duplicated into two autonomous agents: a pioneer and an observer. The pioneer generates responses based on queries, while the observer generates responses using both the queries and the pioneer's responses. The two agents are trained together. During training, the agents exchange roles periodically, fostering cooperation and coevolution between them. Experiments evaluate CORY's performance by fine-tuning GPT-2 and Llama-2 under subjective and objective reward functions on the IMDB Review and GSM8K datasets, respectively. Results show that CORY outperforms PPO in terms of policy optimality, resistance to distribution collapse, and training robustness, thereby underscoring its potential as a superior methodology for refining LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15175v4">ToxiLab: How Well Do Open-Source LLMs Generate Synthetic Toxicity Data?</a></div>
    <div class="paper-meta">
      📅 2025-02-22
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Effective toxic content detection relies heavily on high-quality and diverse data, which serve as the foundation for robust content moderation models. Synthetic data has become a common approach for training models across various NLP tasks. However, its effectiveness remains uncertain for highly subjective tasks like hate speech detection, with previous research yielding mixed results. This study explores the potential of open-source LLMs for harmful data synthesis, utilizing controlled prompting and supervised fine-tuning techniques to enhance data quality and diversity. We systematically evaluated 6 open source LLMs on 5 datasets, assessing their ability to generate diverse, high-quality harmful data while minimizing hallucination and duplication. Our results show that Mistral consistently outperforms other open models, and supervised fine-tuning significantly enhances data reliability and diversity. We further analyze the trade-offs between prompt-based vs. fine-tuned toxic data synthesis, discuss real-world deployment challenges, and highlight ethical considerations. Our findings demonstrate that fine-tuned open source LLMs provide scalable and cost-effective solutions to augment toxic content detection datasets, paving the way for more accessible and transparent content moderation tools.
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14748v1">Large Language Models Struggle to Describe the Haystack without Human Help: Human-in-the-loop Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 21 Pages. LLM for Data Exploration and content analysis
    </div>
    <details class="paper-abstract">
      A common use of NLP is to facilitate the understanding of large document collections, with a shift from using traditional topic models to Large Language Models. Yet the effectiveness of using LLM for large corpus understanding in real-world applications remains under-explored. This study measures the knowledge users acquire with unsupervised, supervised LLM-based exploratory approaches or traditional topic models on two datasets. While LLM-based methods generate more human-readable topics and show higher average win probabilities than traditional models for data exploration, they produce overly generic topics for domain-specific datasets that do not easily allow users to learn much about the documents. Adding human supervision to the LLM generation process improves data exploration by mitigating hallucination and over-genericity but requires greater human effort. In contrast, traditional. models like Latent Dirichlet Allocation (LDA) remain effective for exploration but are less user-friendly. We show that LLMs struggle to describe the haystack of large corpora without human help, particularly domain-specific data, and face scaling and hallucination limitations due to context length constraints. Dataset available at https://huggingface. co/datasets/zli12321/Bills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14739v1">SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.08469v6">LLM4TS: Aligning Pre-Trained LLMs as Data-Efficient Time-Series Forecasters</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Accepted for publication in ACM Transactions on Intelligent Systems and Technology (TIST) 2025. The final published version will be available at https://doi.org/10.1145/3719207
    </div>
    <details class="paper-abstract">
      Multivariate time-series forecasting is vital in various domains, e.g., economic planning and weather prediction. Deep train-from-scratch models have exhibited effective performance yet require large amounts of data, which limits real-world applicability. Recently, researchers have leveraged the representation learning transferability of pre-trained Large Language Models (LLMs) to handle limited non-linguistic datasets effectively. However, incorporating LLMs with time-series data presents challenges of limited adaptation due to different compositions between time-series and linguistic data, and the inability to process multi-scale temporal information. To tackle these challenges, we propose LLM4TS, a framework for time-series forecasting with pre-trained LLMs. LLM4TS consists of a two-stage fine-tuning strategy: the time-series alignment stage to align LLMs with the nuances of time-series data, and the forecasting fine-tuning stage for downstream time-series forecasting tasks. Furthermore, our framework features a novel two-level aggregation method that integrates multi-scale temporal data within pre-trained LLMs, enhancing their ability to interpret time-specific information. In experiments across 7 time-series forecasting datasets, LLM4TS is superior to existing state-of-the-art methods compared with trained-from-scratch models in full-shot scenarios, and also achieves the highest rank in few-shot scenarios. In addition, evaluations compared with different unsupervised representation learning approaches highlight LLM4TS's effectiveness with representation learning in forecasting tasks. Ablation studies further validate each component's contribution to LLM4TS and underscore the essential role of utilizing LLM's pre-trained weights for optimal performance. The code is available at https://github.com/blacksnail789521/LLM4TS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14678v1">How to Get Your LLM to Generate Challenging Problems for Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The pace of evolution of Large Language Models (LLMs) necessitates new approaches for rigorous and comprehensive evaluation. Traditional human annotation is increasingly impracticable due to the complexities and costs involved in generating high-quality, challenging problems. In this work, we introduce CHASE, a unified framework to synthetically generate challenging problems using LLMs without human involvement. For a given task, our approach builds a hard problem in a bottom-up manner from simpler components. Moreover, our framework decomposes the generation process into independently verifiable sub-tasks, thereby ensuring a high level of quality and correctness. We implement CHASE to create evaluation benchmarks across three diverse domains: (1) document-based question answering, (2) repository-level code completion, and (3) math reasoning. The performance of state-of-the-art LLMs on these synthetic benchmarks lies in the range of 40-60% accuracy, thereby demonstrating the effectiveness of our framework at generating challenging problems. We publicly release our benchmarks and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14662v1">InstructAgent: Building User Controllable Recommender via LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 WWW2025@HCRS
    </div>
    <details class="paper-abstract">
      Traditional recommender systems usually take the user-platform paradigm, where users are directly exposed under the control of the platform's recommendation algorithms. However, the defect of recommendation algorithms may put users in very vulnerable positions under this paradigm. First, many sophisticated models are often designed with commercial objectives in mind, focusing on the platform's benefits, which may hinder their ability to protect and capture users' true interests. Second, these models are typically optimized using data from all users, which may overlook individual user's preferences. Due to these shortcomings, users may experience several disadvantages under the traditional user-platform direct exposure paradigm, such as lack of control over the recommender system, potential manipulation by the platform, echo chamber effects, or lack of personalization for less active users due to the dominance of active users during collaborative learning. Therefore, there is an urgent need to develop a new paradigm to protect user interests and alleviate these issues. Recently, some researchers have introduced LLM agents to simulate user behaviors, these approaches primarily aim to optimize platform-side performance, leaving core issues in recommender systems unresolved. To address these limitations, we propose a new user-agent-platform paradigm, where agent serves as the protective shield between user and recommender system that enables indirect exposure. To this end, we first construct four recommendation datasets, denoted as $\dataset$, along with user instructions for each record.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14660v1">Beyond the Surface: Uncovering Implicit Locations with LLMs for Personalized Local News</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 10 pages, 2 figures, submitted to kdd
    </div>
    <details class="paper-abstract">
      News recommendation systems personalize homepage content to boost engagement, but factors like content type, editorial stance, and geographic focus impact recommendations. Local newspapers balance coverage across regions, yet identifying local articles is challenging due to implicit location cues like slang or landmarks. Traditional methods, such as Named Entity Recognition (NER) and Knowledge Graphs, infer locations, but Large Language Models (LLMs) offer new possibilities while raising concerns about accuracy and explainability. This paper explores LLMs for local article classification in Taboola's "Homepage For You" system, comparing them to traditional techniques. Key findings: (1) Knowledge Graphs enhance NER models' ability to detect implicit locations, (2) LLMs outperform traditional methods, and (3) LLMs can effectively identify local content without requiring Knowledge Graph integration. Offline evaluations showed LLMs excel at implicit location classification, while online A/B tests showed a significant increased in local views. A scalable pipeline integrating LLM-based location classification boosted local article distribution by 27%, preserving newspapers' brand identity and enhancing homepage personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14645v1">Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Knowledge editing allows for efficient adaptation of large language models (LLMs) to new information or corrections without requiring full retraining. However, prior methods typically focus on either single-language editing or basic multilingual editing, failing to achieve true cross-linguistic knowledge synchronization. To address this, we present a simple and practical state-of-the-art (SOTA) recipe Cross-Lingual Knowledge Democracy Edit (X-KDE), designed to propagate knowledge from a dominant language to other languages effectively. Our X-KDE comprises two stages: (i) Cross-lingual Edition Instruction Tuning (XE-IT), which fine-tunes the model on a curated parallel dataset to modify in-scope knowledge while preserving unrelated information, and (ii) Target-language Preference Optimization (TL-PO), which applies advanced optimization techniques to ensure consistency across languages, fostering the transfer of updates. Additionally, we contribute a high-quality, cross-lingual dataset, specifically designed to enhance knowledge transfer across languages. Extensive experiments on the Bi-ZsRE and MzsRE benchmarks show that X-KDE significantly enhances cross-lingual performance, achieving an average improvement of +8.19%, while maintaining high accuracy in monolingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14642v1">How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Recently, LLMs have garnered increasing attention across academic disciplines for their potential as human digital twins, virtual proxies designed to replicate individuals and autonomously perform tasks such as decision-making, problem-solving, and reasoning on their behalf. However, current evaluations of LLMs primarily emphasize dialogue simulation while overlooking human behavior simulation, which is crucial for digital twins. To address this gap, we introduce BehaviorChain, the first benchmark for evaluating LLMs' ability to simulate continuous human behavior. BehaviorChain comprises diverse, high-quality, persona-based behavior chains, totaling 15,846 distinct behaviors across 1,001 unique personas, each with detailed history and profile metadata. For evaluation, we integrate persona metadata into LLMs and employ them to iteratively infer contextually appropriate behaviors within dynamic scenarios provided by BehaviorChain. Comprehensive evaluation results demonstrated that even state-of-the-art models struggle with accurately simulating continuous human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14634v1">CER: Confidence Enhanced Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Ensuring the reliability of Large Language Models (LLMs) in complex reasoning tasks remains a formidable challenge, particularly in scenarios that demand precise mathematical calculations and knowledge-intensive open-domain generation. In this work, we introduce an uncertainty-aware framework designed to enhance the accuracy of LLM responses by systematically incorporating model confidence at critical decision points. We propose an approach that encourages multi-step reasoning in LLMs and quantify the confidence of intermediate answers such as numerical results in mathematical reasoning and proper nouns in open-domain generation. Then, the overall confidence of each reasoning chain is evaluated based on confidence of these critical intermediate steps. Finally, we aggregate the answer of generated response paths in a way that reflects the reliability of each generated content (as opposed to self-consistency in which each generated chain contributes equally to majority voting). We conducted extensive experiments in five datasets, three mathematical datasets and two open-domain datasets, using four LLMs. The results consistently validate the effectiveness of our novel confidence aggregation method, leading to an accuracy improvement of up to 7.4% and 5.8% over baseline approaches in math and open-domain generation tasks, respectively. Code is publicly available at https://github.com/ Aquasar11/CER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14628v1">PEARL: Towards Permutation-Resilient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      The in-context learning (ICL) capability of large language models (LLMs) enables them to perform challenging tasks using provided demonstrations. However, ICL is highly sensitive to the ordering of demonstrations, leading to instability in predictions. This paper shows that this vulnerability can be exploited to design a natural attack - difficult for model providers to detect - that achieves nearly 80% success rate on LLaMA-3 by simply permuting the demonstrations. Existing mitigation methods primarily rely on post-processing and fail to enhance the model's inherent robustness to input permutations, raising concerns about safety and reliability of LLMs. To address this issue, we propose Permutation-resilient learning (PEARL), a novel framework based on distributionally robust optimization (DRO), which optimizes model performance against the worst-case input permutation. Specifically, PEARL consists of a permutation-proposal network (P-Net) and the LLM. The P-Net generates the most challenging permutations by treating it as an optimal transport problem, which is solved using an entropy-constrained Sinkhorn algorithm. Through minimax optimization, the P-Net and the LLM iteratively optimize against each other, progressively improving the LLM's robustness. Experiments on synthetic pre-training and real-world instruction tuning tasks demonstrate that PEARL effectively mitigates permutation attacks and enhances performance. Notably, despite being trained on fewer shots and shorter contexts, PEARL achieves performance gains of up to 40% when scaled to many-shot and long-context scenarios, highlighting its efficiency and generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05806v2">CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Ongoing work; project website is available at https://zjunlp.github.io/project/CKnowEdit code and dataset are available at https://github.com/zjunlp/EasyEdit
    </div>
    <details class="paper-abstract">
      Chinese, as a linguistic system rich in depth and complexity, is characterized by distinctive elements such as ancient poetry, proverbs, idioms, and other cultural constructs. However, current Large Language Models (LLMs) face limitations in these specialized domains, highlighting the need for the development of comprehensive datasets that can assess, continuously update, and progressively improve these culturally-grounded linguistic competencies through targeted training optimizations. To address this gap, we introduce CKnowEdit, the first-ever Chinese knowledge editing dataset designed to correct linguistic, factual, and logical errors in LLMs. We collect seven types of knowledge from a wide range of sources, including classical texts, idioms, and content from Baidu Tieba Ruozhiba, taking into account the unique polyphony, antithesis, and logical structures inherent in the Chinese language. By analyzing this dataset, we highlight the challenges current LLMs face in mastering Chinese. Furthermore, our evaluation of state-of-the-art knowledge editing techniques reveals opportunities to advance the correction of Chinese knowledge. Code and dataset are available at https://github.com/zjunlp/EasyEdit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14617v1">Serving Models, Fast and Slow:Optimizing Heterogeneous LLM Inferencing Workloads at Scale</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 15 pages, 17 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference workloads handled by global cloud providers can include both latency-sensitive and insensitive tasks, creating a diverse range of Service Level Agreement (SLA) requirements. Managing these mixed workloads is challenging due to the complexity of the inference stack, which includes multiple LLMs, hardware configurations, and geographic distributions. Current optimization strategies often silo these tasks to ensure that SLAs are met for latency-sensitive tasks, but this leads to significant under-utilization of expensive GPU resources despite the availability of spot and on-demand Virtual Machine (VM) provisioning. We propose SAGESERVE, a comprehensive LLM serving framework that employs adaptive control knobs at varying time scales, ensuring SLA compliance while maximizing the utilization of valuable GPU resources. Short-term optimizations include efficient request routing to data center regions, while long-term strategies involve scaling GPU VMs out/in and redeploying models to existing VMs to align with traffic patterns. These strategies are formulated as an optimization problem for resource allocation and solved using Integer Linear Programming (ILP). We perform empirical and simulation studies based on production workload traces with over 8M requests using four open-source models deployed across three regions. SAGESERVE achieves up to 25% savings in GPU-hours while maintaining tail latency and satisfying all SLOs, and it reduces the scaling overhead compared to baselines by up to 80%, confirming the effectiveness of our proposal. In terms of dollar cost, this can save cloud providers up to $2M over the course of a month.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11844v2">BaxBench: Can LLMs Generate Correct and Secure Backends?</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The automatic generation of programs has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 60% on code correctness; (ii) on average, we could successfully execute security exploits on more than half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01387v3">TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Although Deep Reinforcement Learning (DRL) and Large Language Models (LLMs) each show promise in addressing decision-making challenges in autonomous driving, DRL often suffers from high sample complexity, while LLMs have difficulty ensuring real-time decision making. To address these limitations, we propose TeLL-Drive, a hybrid framework that integrates a Teacher LLM to guide an attention-based Student DRL policy. By incorporating risk metrics, historical scenario retrieval, and domain heuristics into context-rich prompts, the LLM produces high-level driving strategies through chain-of-thought reasoning. A self-attention mechanism then fuses these strategies with the DRL agent's exploration, accelerating policy convergence and boosting robustness across diverse driving conditions. The experimental results, evaluated across multiple traffic scenarios, show that TeLL-Drive outperforms existing baseline methods, including other LLM-based approaches, in terms of success rates, average returns, and real-time feasibility. Ablation studies underscore the importance of each model component, especially the synergy between the attention mechanism and LLM-driven guidance. Finally, we build a virtual-real fusion experimental platform to verify the real-time performance, robustness, and reliability of the algorithm running on real vehicles through vehicle-in-loop experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14563v1">Plan-over-Graph: Towards Parallelable LLM Agent Schedule</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional abilities in reasoning for task planning. However, challenges remain under-explored for parallel schedules. This paper introduces a novel paradigm, plan-over-graph, in which the model first decomposes a real-life textual task into executable subtasks and constructs an abstract task graph. The model then understands this task graph as input and generates a plan for parallel execution. To enhance the planning capability of complex, scalable graphs, we design an automated and controllable pipeline to generate synthetic graphs and propose a two-stage training scheme. Experimental results show that our plan-over-graph method significantly improves task performance on both API-based LLMs and trainable open-sourced LLMs. By normalizing complex tasks as graphs, our method naturally supports parallel execution, demonstrating global efficiency. The code and data are available at https://github.com/zsq259/Plan-over-Graph.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14561v1">Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches that rely on pre-trained models like SciBERT, which require extensive domain-specific pretraining and specialized architectures, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero, one, few, and many-shot prompting to assess performance across scenarios. Our experimental study identifies the top-performing model through extensive experimentation of in-context learning-related parameters, which we fine-tune to further enhance task performance. The results highlight the strengths and limitations of LLMs in recognizing citation intents, providing valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14541v1">LLM-based User Profile Management for Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-02-20
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14507v1">Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      This study evaluates Large Language Models' (LLMs) ability to simulate non-native-like English use observed in human second language (L2) learners interfered with by their native first language (L1). In dialogue-based interviews, we prompt LLMs to mimic L2 English learners with specific L1s (e.g., Japanese, Thai, Urdu) across seven languages, comparing their outputs to real L2 learner data. Our analysis examines L1-driven linguistic biases, such as reference word usage and avoidance behaviors, using information-theoretic and distributional density measures. Results show that modern LLMs (e.g., Qwen2.5, LLAMA3.3, DeepseekV3, GPT-4o) replicate L1-dependent patterns observed in human L2 data, with distinct influences from various languages (e.g., Japanese, Korean, and Mandarin significantly affect tense agreement, and Urdu influences noun-verb collocations). Our results reveal the potential of LLMs for L2 dialogue generation and evaluation for future educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14502v1">How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?</a></div>
    <div class="paper-meta">
      📅 2025-02-20
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities.
    </details>
</div>
