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
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20045v2">Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive fluency, but often produce critical errors known as "hallucinations". Uncertainty quantification (UQ) methods are a promising tool for coping with this fundamental shortcoming. Yet, existing UQ methods face challenges such as high computational overhead or reliance on supervised learning. Here, we aim to bridge this gap. In particular, we propose RAUQ (Recurrent Attention-based Uncertainty Quantification), an unsupervised approach that leverages intrinsic attention patterns in transformers to detect hallucinations efficiently. By analyzing attention weights, we identified a peculiar pattern: drops in attention to preceding tokens are systematically observed during incorrect generations for certain "uncertainty-aware" heads. RAUQ automatically selects such heads, recurrently aggregates their attention weights and token-level confidences, and computes sequence-level uncertainty scores in a single forward pass. Experiments across 4 LLMs and 12 question answering, summarization, and translation tasks demonstrate that RAUQ yields excellent results, outperforming state-of-the-art UQ methods using minimal computational overhead (<1% latency). Moreover, it requires no task-specific labels and no careful hyperparameter tuning, offering plug-and-play real-time hallucination detection in white-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16794v1">The Even Sheen of AI: Kitsch, LLMs, and Homogeneity</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 14 pages + 4 pages references
    </div>
    <details class="paper-abstract">
      The exploding use and impact of Chatbots such as ChatGPT that are based on Large Language Models urgently call for a language which is fit to clearly describe functions and problems of the production process and qualities of the Chatbots' textual and image output. Recently, the discussion about appropriate and illuminating metaphors to describe LLMs has gained momentum. As an alternative to well-established metaphors such as "hallucinating" and "bullshit", we propose "kitsch" as a new metaphor. As an internationally widespread term from literary and cultural studies, we argue that "kitsch" is particularly suitable for analytically illuminating a previously neglected feature of LLM-based images and texts: their tendency to produce homogeneous and average content, which is becoming increasingly dominant as the proportion of AI-generated content on the internet grows. This is leading to the equalisation of language, style and argument. In view of the potential negative consequences of this averaging, including for human content producers on the internet, we advocate combining methods and insights from kitsch studies with AI research, philosophy, and communication studies in order to better understand the phenomenon and develop countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16784v1">Controlled Yet Natural: A Hybrid BDI-LLM Conversational Agent for Child Helpline Training</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Child helpline training often relies on human-led roleplay, which is both time- and resource-consuming. To address this, rule-based interactive agent simulations have been proposed to provide a structured training experience for new counsellors. However, these agents might suffer from limited language understanding and response variety. To overcome these limitations, we present a hybrid interactive agent that integrates Large Language Models (LLMs) into a rule-based Belief-Desire-Intention (BDI) framework, simulating more realistic virtual child chat conversations. This hybrid solution incorporates LLMs into three components: intent recognition, response generation, and a bypass mechanism. We evaluated the system through two studies: a script-based assessment comparing LLM-generated responses to human-crafted responses, and a within-subject experiment (N=37) comparing the LLM-integrated agent with a rule-based version. The first study provided evidence that the three LLM components were non-inferior to human-crafted responses. In the second study, we found credible support for two hypotheses: participants perceived the LLM-integrated agent as more believable and reported more positive attitudes toward it than the rule-based agent. Additionally, although weaker, there was some support for increased engagement (posterior probability = 0.845, 95% HDI [-0.149, 0.465]). Our findings demonstrate the potential of integrating LLMs into rule-based systems, offering a promising direction for more flexible but controlled training systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03823v3">Both Text and Images Leaked! A Systematic Analysis of Data Contamination in Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      The rapid advancement of multimodal large language models (MLLMs) has significantly enhanced performance across benchmarks. However, data contamination-unintentional memorization of benchmark data during model training-poses critical challenges for fair evaluation. Existing detection methods for unimodal large language models (LLMs) are inadequate for MLLMs due to multimodal data complexity and multi-phase training. We systematically analyze multimodal data contamination using our analytical framework, MM-Detect, which defines two contamination categories-unimodal and cross-modal-and effectively quantifies contamination severity across multiple-choice and caption-based Visual Question Answering tasks. Evaluations on twelve MLLMs and five benchmarks reveal significant contamination, particularly in proprietary models and older benchmarks. Crucially, contamination sometimes originates during unimodal pre-training rather than solely from multimodal fine-tuning. Our insights refine contamination understanding, guiding evaluation practices and improving multimodal model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10830v2">The Siren Song of LLMs: How Users Perceive and Respond to Dark Patterns in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Large language models can influence users through conversation, creating new forms of dark patterns that differ from traditional UX dark patterns. We define LLM dark patterns as manipulative or deceptive behaviors enacted in dialogue. Drawing on prior work and AI incident reports, we outline a diverse set of categories with real-world examples. Using them, we conducted a scenario-based study where participants (N=34) compared manipulative and neutral LLM responses. Our results reveal that recognition of LLM dark patterns often hinged on conversational cues such as exaggerated agreement, biased framing, or privacy intrusions, but these behaviors were also sometimes normalized as ordinary assistance. Users' perceptions of these dark patterns shaped how they respond to them. Responsibilities for these behaviors were also attributed in different ways, with participants assigning it to companies and developers, the model itself, or to users. We conclude with implications for design, advocacy, and governance to safeguard user autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16749v1">Evaluating LLM Generated Detection Rules in Cybersecurity</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Preprint of a paper accepted at the Conference on Applied Machine Learning in Information Security (CAMLIS 2025). 11 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLMs are increasingly pervasive in the security environment, with limited measures of their effectiveness, which limits trust and usefulness to security practitioners. Here, we present an open-source evaluation framework and benchmark metrics for evaluating LLM-generated cybersecurity rules. The benchmark employs a holdout set-based methodology to measure the effectiveness of LLM-generated security rules in comparison to a human-generated corpus of rules. It provides three key metrics inspired by the way experts evaluate security rules, offering a realistic, multifaceted evaluation of the effectiveness of an LLM-based security rule generator. This methodology is illustrated using rules from Sublime Security's detection team and those written by Sublime Security's Automated Detection Engineer (ADE), with a thorough analysis of ADE's skills presented in the results section.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00367v2">KoACD: The First Korean Adolescent Dataset for Cognitive Distortion Analysis via Role-Switching Multi-LLM Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted to Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Cognitive distortion refers to negative thinking patterns that can lead to mental health issues like depression and anxiety in adolescents. Previous studies using natural language processing (NLP) have focused mainly on small-scale adult datasets, with limited research on adolescents. This study introduces KoACD, the first large-scale dataset of cognitive distortions in Korean adolescents, containing 108,717 instances. We applied a multi-Large Language Model (LLM) negotiation method to refine distortion classification, enabling iterative feedback and role-switching between models to reduce bias and improve label consistency. In addition, we generated synthetic data using two approaches: cognitive clarification for textual clarity and cognitive balancing for diverse distortion representation. Validation through LLMs and expert evaluations showed that while LLMs classified distortions with explicit markers, they struggled with context-dependent reasoning, where human evaluators demonstrated higher accuracy. KoACD aims to enhance future research on cognitive distortion detection. The dataset and implementation details are publicly accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05309v2">Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      LLMs are used predominantly in synchronous communication, where a human user and a model communicate in alternating turns. In contrast, many real-world settings are asynchronous. For example, in group chats, online team meetings, or social games, there is no inherent notion of turns. In this work, we develop an adaptive asynchronous LLM agent consisting of two modules: a generator that decides what to say, and a scheduler that decides when to say it. To evaluate our agent, we collect a unique dataset of online Mafia games, where our agent plays with human participants. Overall, our agent performs on par with human players, both in game performance metrics and in its ability to blend in with the other human players. Our analysis shows that the agent's behavior in deciding when to speak closely mirrors human patterns, although differences emerge in message content. We make all of our code and data publicly available. This work paves the way for integration of LLMs into realistic human group settings, from assistance in team discussions to educational and professional environments where complex social dynamics must be navigated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16736v1">Towards Transparent and Incentive-Compatible Collaboration in Decentralized LLM Multi-Agent Systems: A Blockchain-Driven Approach</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 17 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled the emergence of autonomous agents capable of complex reasoning, planning, and interaction. However, coordinating such agents at scale remains a fundamental challenge, particularly in decentralized environments where communication lacks transparency and agent behavior cannot be shaped through centralized incentives. We propose a blockchain-based framework that enables transparent agent registration, verifiable task allocation, and dynamic reputation tracking through smart contracts. The core of our design lies in two mechanisms: a matching score-based task allocation protocol that evaluates agents by reputation, capability match, and workload; and a behavior-shaping incentive mechanism that adjusts agent behavior via feedback on performance and reward. Our implementation integrates GPT-4 agents with Solidity contracts and demonstrates, through 50-round simulations, strong task success rates, stable utility distribution, and emergent agent specialization. The results underscore the potential for trustworthy, incentive-compatible multi-agent coordination in open environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12617v3">Evaluating AI Alignment in Eleven LLMs through Output-Based Analysis and Human Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in psychological research and practice, yet traditional benchmarks reveal little about the values they express in real interaction. We introduce PAPERS, an output-based evaluation of the values LLMs prioritise in their text. Study 1 thematically analysed responses from eleven LLMs, identifying five recurring dimensions (Purposeful Contribution, Adaptive Growth, Positive Relationality, Ethical Integrity, and Robust Functionality) with Self-Actualised Autonomy appearing only under a hypothetical sentience prompt. These results suggest that LLMs are trained to prioritise humanistic and utility values as dual objectives of optimal functioning, a pattern supported by existing AI alignment and prioritisation frameworks. Study 2 operationalised PAPERS as a ranking instrument across the same eleven LLMs, yielding stable, non-random value priorities alongside systematic between-model differences. Hierarchical clustering distinguished "human-centric" models (e.g., ChatGPT-4o, Claude Sonnet 4) that prioritised relational/ethical values from "utility-driven" models (e.g., Llama 4, Gemini 2.5 Pro) that emphasised operational priorities. Study 3 benchmarked four LLMs against human judgements (N = 376) under matched prompts, finding near-perfect rank-order convergence (r = .97-.98) but moderate absolute agreement; among tested models, ChatGPT-4o showed the closest alignment with human ratings (ICC = .78). Humans also showed limited readiness to endorse sentient AI systems. Taken together, PAPERS enabled systematic value audits and revealed trade-offs with direct implications for deployment: human-centric models aligned more closely with human value judgments and appear better suited for humanistic psychological applications, whereas utility-driven models emphasised functional efficiency and may be more appropriate for instrumental or back-office tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16713v1">OPEN-THEATRE: An Open-Source Toolkit for LLM-based Interactive Drama</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted by EMNLP 2025 demo
    </div>
    <details class="paper-abstract">
      LLM-based Interactive Drama introduces a novel dialogue scenario in which the player immerses into a character and engages in a dramatic story by interacting with LLM agents. Despite the fact that this emerging area holds significant promise, it remains largely underexplored due to the lack of a well-designed playground to develop a complete drama. This makes a significant barrier for researchers to replicate, extend, and study such systems. Hence, we present Open-Theatre, the first open-source toolkit for experiencing and customizing LLM-based interactive drama. It refines prior work with an efficient multi-agent architecture and a hierarchical retrieval-based memory system, designed to enhance narrative coherence and realistic long-term behavior in complex interactions. In addition, we provide a highly configurable pipeline, making it easy for researchers to develop and optimize new approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23323v2">Neither Stochastic Parroting nor AGI: LLMs Solve Tasks through Context-Directed Extrapolation from Training Data Priors</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      In this position paper we raise critical awareness of a realistic view of LLM capabilities that eschews extreme alternative views that LLMs are either 'stochastic parrots' or in possession of 'emergent' advanced reasoning capabilities, which, due to their unpredictable emergence, constitute an existential threat. Our middle-ground view is that LLMs extrapolate from priors from their training data while using context to guide the model to the appropriate priors; we call this "context-directed extrapolation." Specifically, this context direction is achieved through examples in base models, leading to in-context learning, while instruction tuning allows LLMs to perform similarly based on prompts rather than explicit examples. Under this view, substantiated though existing literature, while reasoning capabilities go well beyond stochastic parroting, such capabilities are predictable, controllable, not indicative of advanced reasoning akin to high-level cognitive capabilities in humans, and not infinitely scalable with additional training. As a result, fears of uncontrollable emergence of agency are allayed, while research advances are appropriately refocused on the processes of context-directed extrapolation and how this interacts with training data to produce valuable capabilities in LLMs. Future work can therefore explore alternative augmenting techniques that do not rely on inherent advanced reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08550v3">No Need for Explanations: LLMs can implicitly learn from mistakes in-context</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Showing incorrect answers to Large Language Models (LLMs) is a popular strategy to improve their performance in reasoning-intensive tasks. It is widely assumed that, in order to be helpful, the incorrect answers must be accompanied by comprehensive rationales, explicitly detailing where the mistakes are and how to correct them. However, in this work we present a counterintuitive finding: we observe that LLMs perform better in math reasoning tasks when these rationales are eliminated from the context and models are left to infer on their own what makes an incorrect answer flawed. This approach also substantially outperforms chain-of-thought prompting in our evaluations. These results are consistent across LLMs of different sizes and varying reasoning abilities. To gain an understanding of why LLMs learn from mistakes more effectively without explicit corrective rationales, we perform a thorough analysis, investigating changes in context length and answer diversity between different prompting strategies, and their effect on performance. We also examine evidence of overfitting to the in-context rationales when these are provided, and study the extent to which LLMs are able to autonomously infer high-quality corrective rationales given only incorrect answers as input. We find evidence that, while incorrect answers are more beneficial for LLM learning than additional diverse correct answers, explicit corrective rationales over-constrain the model, thus limiting those benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08760v2">INTA: Intent-Based Translation for Network Configuration with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted by The 33rd IEEE International Conference on Network Protocols (IEEE ICNP 2025)
    </div>
    <details class="paper-abstract">
      Translating configurations between different network devices is a common yet challenging task in modern network operations. This challenge arises in typical scenarios such as replacing obsolete hardware and adapting configurations to emerging paradigms like Software Defined Networking (SDN) and Network Function Virtualization (NFV). Engineers need to thoroughly understand both source and target configuration models, which requires considerable effort due to the complexity and evolving nature of these specifications. To promote automation in network configuration translation, we propose INTA, an intent-based translation framework that leverages Large Language Model (LLM) agents. The key idea of INTA is to use configuration intent as an intermediate representation for translation. It first employs LLMs to decompose configuration files and extract fine-grained intents for each configuration fragment. These intents are then used to retrieve relevant manuals of the target device. Guided by a syntax checker, INTA incrementally generates target configurations. The translated configurations are further verified and refined for semantic consistency. We implement INTA and evaluate it on real-world configuration datasets from the industry. Our approach outperforms state-of-the-art methods in translation accuracy and exhibits strong generalizability. INTA achieves an accuracy of 98.15% in terms of both syntactic and view correctness, and a command recall rate of 84.72% for the target configuration. The semantic consistency report of the translated configuration further demonstrates its practical value in real-world network operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16686v1">EG-MLA: Embedding-Gated Multi-head Latent Attention for Scalable and Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Reducing the key-value (KV) cache size is a crucial step toward enabling efficient inference in large language models (LLMs), especially under latency and memory constraints. While Multi-Head Attention (MHA) offers strong representational power, it incurs significant memory overhead. Recent work on Multi-head Latent Attention (MLA) mitigates this by compressing KV representations into a shared latent space, achieving a better trade-off between performance and cache efficiency. While MLA already achieves significant KV cache reduction, the scope for further compression remains limited without performance loss. In this paper, we propose \textbf{Embedding-Gated Multi-head Latent Attention (EG-MLA)}, a novel extension of MLA that further reduces KV cache size while enhancing representational expressiveness. EG-MLA introduces a token-specific embedding gating mechanism applied in the latent space, enabling fine-grained modulation of compressed KV vectors with minimal additional computation. Compared to MHA, EG-MLA achieves over 91.6\% reduction in KV cache size with negligible performance degradation. Relative to MLA, EG-MLA consistently improves task accuracy across diverse reasoning benchmarks while achieving up to 59.9\% additional memory savings. Our theoretical analysis highlights how embedding gating induces implicit high-order interactions, and empirical evaluations demonstrate robust generalization across model scales and compression regimes. Notably, we successfully scale EG-MLA to over 1 billion parameters, demonstrating its practical viability for large-scale LLM deployment. These results establish EG-MLA as a memory- and compute-efficient attention mechanism that enables scalable, high-performance inference in modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16682v1">Design and Development of an Intelligent LLM-based LDAP Honeypot</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Cybersecurity threats continue to increase, with a growing number of previously unknown attacks each year targeting both large corporations and smaller entities. This scenario demands the implementation of advanced security measures, not only to mitigate damage but also to anticipate emerging attack trends. In this context, deception tools have become a key strategy, enabling the detection, deterrence, and deception of potential attackers while facilitating the collection of information about their tactics and methods. Among these tools, honeypots have proven their value, although they have traditionally been limited by rigidity and configuration complexity, hindering their adaptability to dynamic scenarios. The rise of artificial intelligence, and particularly general-purpose Large Language Models (LLMs), is driving the development of new deception solutions capable of offering greater adaptability and ease of use. This work proposes the design and implementation of an LLM-based honeypot to simulate an LDAP server, a critical protocol present in most organizations due to its central role in identity and access management. The proposed solution aims to provide a flexible and realistic tool capable of convincingly interacting with attackers, thereby contributing to early detection and threat analysis while enhancing the defensive capabilities of infrastructures against intrusions targeting this service.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16679v1">Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 A Survey of Reinforcement Learning for Large Language Models
    </div>
    <details class="paper-abstract">
      In recent years, training methods centered on Reinforcement Learning (RL) have markedly enhanced the reasoning and alignment performance of Large Language Models (LLMs), particularly in understanding human intents, following user instructions, and bolstering inferential strength. Although existing surveys offer overviews of RL augmented LLMs, their scope is often limited, failing to provide a comprehensive summary of how RL operates across the full lifecycle of LLMs. We systematically review the theoretical and practical advancements whereby RL empowers LLMs, especially Reinforcement Learning with Verifiable Rewards (RLVR). First, we briefly introduce the basic theory of RL. Second, we thoroughly detail application strategies for RL across various phases of the LLM lifecycle, including pre-training, alignment fine-tuning, and reinforced reasoning. In particular, we emphasize that RL methods in the reinforced reasoning phase serve as a pivotal driving force for advancing model reasoning to its limits. Next, we collate existing datasets and evaluation benchmarks currently used for RL fine-tuning, spanning human-annotated datasets, AI-assisted preference data, and program-verification-style corpora. Subsequently, we review the mainstream open-source tools and training frameworks available, providing clear practical references for subsequent research. Finally, we analyse the future challenges and trends in the field of RL-enhanced LLMs. This survey aims to present researchers and practitioners with the latest developments and frontier trends at the intersection of RL and LLMs, with the goal of fostering the evolution of LLMs that are more intelligent, generalizable, and secure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16671v1">"Digital Camouflage": The LLVM Challenge in LLM-Based Malware Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising tools for malware detection by analyzing code semantics, identifying vulnerabilities, and adapting to evolving threats. However, their reliability under adversarial compiler-level obfuscation is yet to be discovered. In this study, we empirically evaluate the robustness of three state-of-the-art LLMs: ChatGPT-4o, Gemini Flash 2.5, and Claude Sonnet 4 against compiler-level obfuscation techniques implemented via the LLVM infrastructure. These include control flow flattening, bogus control flow injection, instruction substitution, and split basic blocks, which are widely used to evade detection while preserving malicious behavior. We perform a structured evaluation on 40~C functions (20 vulnerable, 20 secure) sourced from the Devign dataset and obfuscated using LLVM passes. Our results show that these models often fail to correctly classify obfuscated code, with precision, recall, and F1-score dropping significantly after transformation. This reveals a critical limitation: LLMs, despite their language understanding capabilities, can be easily misled by compiler-based obfuscation strategies. To promote reproducibility, we release all evaluation scripts, prompts, and obfuscated code samples in a public repository. We also discuss the implications of these findings for adversarial threat modeling, and outline future directions such as software watermarking, compiler-aware defenses, and obfuscation-resilient model design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12158v2">Pun Unintended: LLMs and the Illusion of Humor Understanding</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01830v2">From Language to Cognition: How LLMs Outgrow the Human Language Network</a></div>
    <div class="paper-meta">
      📅 2025-09-20
      | 💬 EMNLP 2025. Project Page at https://language-to-cognition.epfl.ch
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable similarity to neural activity in the human language network. However, the key properties of language shaping brain-like representations, and their evolution during training as a function of different tasks remain unclear. We here benchmark 34 training checkpoints spanning 300B tokens across 8 different model sizes to analyze how brain alignment relates to linguistic competence. Specifically, we find that brain alignment tracks the development of formal linguistic competence -- i.e., knowledge of linguistic rules -- more closely than functional linguistic competence. While functional competence, which involves world knowledge and reasoning, continues to develop throughout training, its relationship with brain alignment is weaker, suggesting that the human language network primarily encodes formal linguistic structure rather than broader cognitive functions. We further show that model size is not a reliable predictor of brain alignment when controlling for feature size and find that the correlation between next-word prediction, behavioral alignment and brain alignment fades once models surpass human language proficiency. Finally, using the largest set of rigorous neural language benchmarks to date, we show that language brain alignment benchmarks remain unsaturated, highlighting opportunities for improving future models. Taken together, our findings suggest that the human language network is best modeled by formal, rather than functional, aspects of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16203v1">Inverting Trojans in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      While effective backdoor detection and inversion schemes have been developed for AIs used e.g. for images, there are challenges in "porting" these methods to LLMs. First, the LLM input space is discrete, which precludes gradient-based search over this space, central to many backdoor inversion methods. Second, there are ~30,000^k k-tuples to consider, k the token-length of a putative trigger. Third, for LLMs there is the need to blacklist tokens that have strong marginal associations with the putative target response (class) of an attack, as such tokens give false detection signals. However, good blacklists may not exist for some domains. We propose a LLM trigger inversion approach with three key components: i) discrete search, with putative triggers greedily accreted, starting from a select list of singletons; ii) implicit blacklisting, achieved by evaluating the average cosine similarity, in activation space, between a candidate trigger and a small clean set of samples from the putative target class; iii) detection when a candidate trigger elicits high misclassifications, and with unusually high decision confidence. Unlike many recent works, we demonstrate that our approach reliably detects and successfully inverts ground-truth backdoor trigger phrases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09723v3">AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents Amazon.com, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09407v3">UXAgent: A System for Simulating Usability Testing of Web Design with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Usability testing is a fundamental research method that user experience (UX) researchers use to evaluate and iterate their new designs. But what about evaluating and iterating the usability testing study design itself? Recent advances in Large Language Model-simulated Agent (LLM Agent) research inspired us to design UXAgent to support UX researchers in evaluating and iterating their study design before they conduct the real human-subject study. Our system features a Persona Generator module, an LLM Agent module, and a Universal Browser Connector module to automatically generate thousands of simulated users and to interactively test the target website. The system also provides a Result Viewer Interface so that the UX researchers can easily review and analyze the generated qualitative (e.g., agents' post-study surveys) and quantitative data (e.g., agents' interaction logs), or even interview agents directly. Through a heuristic evaluation with 16 UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent usage in UX studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16188v1">CultureScope: A Dimensional Lens for Probing Cultural Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in diverse cultural environments, evaluating their cultural understanding capability has become essential for ensuring trustworthy and culturally aligned applications. However, most existing benchmarks lack comprehensiveness and are challenging to scale and adapt across different cultural contexts, because their frameworks often lack guidance from well-established cultural theories and tend to rely on expert-driven manual annotations. To address these issues, we propose CultureScope, the most comprehensive evaluation framework to date for assessing cultural understanding in LLMs. Inspired by the cultural iceberg theory, we design a novel dimensional schema for cultural knowledge classification, comprising 3 layers and 140 dimensions, which guides the automated construction of culture-specific knowledge bases and corresponding evaluation datasets for any given languages and cultures. Experimental results demonstrate that our method can effectively evaluate cultural understanding. They also reveal that existing large language models lack comprehensive cultural competence, and merely incorporating multilingual data does not necessarily enhance cultural understanding. All code and data files are available at https://github.com/HoganZinger/Culture
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05755v4">On the Security of Tool-Invocation Prompts for LLM-Based Agentic Systems: An Empirical Risk Assessment</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12158v3">A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted to EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate synthetic textual data for training smaller specialized models. However, a comparison of various generation strategies for low-resource language settings is lacking. While various prompting strategies have been proposed, such as demonstrations, label-based summaries, and self-revision, their comparative effectiveness remains unclear, especially for low-resource languages. In this paper, we systematically evaluate the performance of these generation strategies and their combinations across 11 typologically diverse languages, including several extremely low-resource ones. Using three NLP tasks and four open-source LLMs, we assess downstream model performance on generated versus gold-standard data. Our results show that strategic combinations of generation methods, particularly target-language demonstrations with LLM-based revisions, yield strong performance, narrowing the gap with real data to as little as 5% in some settings. We also find that smart prompting techniques can reduce the advantage of larger LLMs, highlighting efficient generation strategies for synthetic data generation in low-resource scenarios with smaller models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13557v3">Automated CGRA Design with Multi-Agent LLMs: A Unified Hardware-Software Co-Design Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Due to certain confidentiality requirements, this article needs to be withdrawn
    </div>
    <details class="paper-abstract">
      Coarse-grained Reconfigurable Arrays (CGRAs) are a promising computing architecture that can deliver high-performance, energy-efficient acceleration across diverse domains. By supporting reconfiguration at the functional unit level, CGRAs efficiently adapt to varying computational patterns and optimize resource utilization. However, designing CGRAs is highly challenging due to the vast design space, independent architectural parameters, and the time-consuming nature of manual design. Fortunately, the rapid advancement of large language models (LLMs) presents new opportunities to automate this process. In this work, we propose MACO -- an open-source multi-agent LLM-based framework for Hardware/Software (HW/SW) co-design of CGRAs. The framework employs LLM reasoning to generate CGRAs across four stages: HW/SW co-design, Design error correction, Best design selection, and Evaluation & Feedback. Furthermore, MACO iteratively optimizes the generated CGRAs, leveraging agent reasoning and feedback to achieve higher PPA (that is, power, performance, and area) design points for a given domain. In addition, we introduce an LLM self-learning mechanism that employs LLM-driven decision making to select the optimal CGRA to accelerate the design process. We evaluate the framework with state-of-the-art LLM-based methods and manual CGRA design, in terms of performance, power consumption, and area. Experimental results show that MACO efficiently generates high-quality CGRA architectures, significantly reducing manual design effort and demonstrating the potential of our framework for real-world CGRA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07894v4">HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with multiple golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight the performance gap between open-source models and top students, the strong reasoning abilities of closed-source models, and the remaining room for improvement. HiPhO, a human-aligned Olympiad benchmark for multimodal physical reasoning, is open-source at https://github.com/SciYu/HiPhO with a public leaderboard at https://phyarena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12385v2">SENTRA: Selected-Next-Token Transformer for LLM Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      LLMs are becoming increasingly capable and widespread. Consequently, the potential and reality of their misuse is also growing. In this work, we address the problem of detecting LLM-generated text that is not explicitly declared as such. We present a novel, general-purpose, and supervised LLM text detector, SElected-Next-Token tRAnsformer (SENTRA). SENTRA is a Transformer-based encoder leveraging selected-next-token-probability sequences and utilizing contrastive pre-training on large amounts of unlabeled data. Our experiments on three popular public datasets across 24 domains of text demonstrate SENTRA is a general-purpose classifier that significantly outperforms popular baselines in the out-of-domain setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16093v1">Beyond Pointwise Scores: Decomposed Criteria-Based Evaluation of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Evaluating long-form answers in high-stakes domains such as law or medicine remains a fundamental challenge. Standard metrics like BLEU and ROUGE fail to capture semantic correctness, and current LLM-based evaluators often reduce nuanced aspects of answer quality into a single undifferentiated score. We introduce DeCE, a decomposed LLM evaluation framework that separates precision (factual accuracy and relevance) and recall (coverage of required concepts), using instance-specific criteria automatically extracted from gold answer requirements. DeCE is model-agnostic and domain-general, requiring no predefined taxonomies or handcrafted rubrics. We instantiate DeCE to evaluate different LLMs on a real-world legal QA task involving multi-jurisdictional reasoning and citation grounding. DeCE achieves substantially stronger correlation with expert judgments ($r=0.78$), compared to traditional metrics ($r=0.12$), pointwise LLM scoring ($r=0.35$), and modern multidimensional evaluators ($r=0.48$). It also reveals interpretable trade-offs: generalist models favor recall, while specialized models favor precision. Importantly, only 11.95% of LLM-generated criteria required expert revision, underscoring DeCE's scalability. DeCE offers an interpretable and actionable LLM evaluation framework in expert domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13794v6">LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13252v2">Are LLMs Better Formalizers than Solvers on Complex Problems?</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      A trending line of recent work advocates for using large language models (LLMs) as formalizers instead of as end-to-end solvers for logical reasoning problems. Instead of generating the solution, the LLM generates a formal program that derives a solution via an external solver. While performance gain of the seemingly scalable LLM-as-formalizer over the seemingly unscalable LLM-as-solver has been widely reported, we show that this superiority does not hold on real-life constraint satisfaction problems. On 4 domains, we systematically evaluate 6 LLMs including 4 large reasoning models with inference-time scaling, paired with 5 pipelines including 2 types of formalism. We show that in few-shot settings, LLM-as-formalizer underperforms LLM-as-solver. While LLM-as-formalizer promises accuracy, robustness, faithfulness, and efficiency, we observe that the present LLMs do not yet deliver any of those, as their limited ability to generate formal programs leads to failure to scale with complexity, hard-coded solutions, and excessive reasoning tokens. We present our detailed analysis and actionable remedies to drive future research that improves LLM-as-formalizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09627v2">Benchmarking Debiasing Methods for LLM-based Parameter Estimates</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 To appear as: Nicolas Audinet de Pieuchon, Adel Daoud, Connor T. Jerzak, Moa Johansson, Richard Johansson. Benchmarking Debiasing Methods for LLM-based Parameter Estimates. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer an inexpensive yet powerful way to annotate text, but are often inconsistent when compared with experts. These errors can bias downstream estimates of population parameters such as regression coefficients and causal effects. To mitigate this bias, researchers have developed debiasing methods such as Design-based Supervised Learning (DSL) and Prediction-Powered Inference (PPI), which promise valid estimation by combining LLM annotations with a limited number of expensive expert annotations. Although these methods produce consistent estimates under theoretical assumptions, it is unknown how they compare in finite samples of sizes encountered in applied research. We make two contributions. First, we study how each methods performance scales with the number of expert annotations, highlighting regimes where LLM bias or limited expert labels significantly affect results. Second, we compare DSL and PPI across a range of tasks, finding that although both achieve low bias with large datasets, DSL often outperforms PPI on bias reduction and empirical efficiency, but its performance is less consistent across datasets. Our findings indicate that there is a bias-variance tradeoff at the level of debiasing methods, calling for more research on developing metrics for quantifying their efficiency in finite samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08177v4">SycEval: Evaluating LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 AIES 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16006v1">Defining and Monitoring Complex Robot Activities via LLMs and Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Recent years have witnessed a growing interest in automating labor-intensive and complex activities, i.e., those consisting of multiple atomic tasks, by deploying robots in dynamic and unpredictable environments such as industrial and agricultural settings. A key characteristic of these contexts is that activities are not predefined: while they involve a limited set of possible tasks, their combinations may vary depending on the situation. Moreover, despite recent advances in robotics, the ability for humans to monitor the progress of high-level activities - in terms of past, present, and future actions - remains fundamental to ensure the correct execution of safety-critical processes. In this paper, we introduce a general architecture that integrates Large Language Models (LLMs) with automated planning, enabling humans to specify high-level activities (also referred to as processes) using natural language, and to monitor their execution by querying a robot. We also present an implementation of this architecture using state-of-the-art components and quantitatively evaluate the approach in a real-world precision agriculture scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03455v3">Watson: A Cognitive Observability Framework for the Reasoning of LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into autonomous systems, giving rise to a new class of software known as Agentware, where LLM-powered agents perform complex, open-ended tasks in domains such as software engineering, customer service, and data analysis. However, their high autonomy and opaque reasoning processes pose significant challenges for traditional software observability methods. To address this, we introduce the concept of cognitive observability - the ability to recover and inspect the implicit reasoning behind agent decisions. We present Watson, a general-purpose framework for observing the reasoning processes of fast-thinking LLM agents without altering their behavior. Watson retroactively infers reasoning traces using prompt attribution techniques. We evaluate Watson in both manual debugging and automated correction scenarios across the MMLU benchmark and the AutoCodeRover and OpenHands agents on the SWE-bench-lite dataset. In both static and dynamic settings, Watson surfaces actionable reasoning insights and supports targeted interventions, demonstrating its practical utility for improving transparency and reliability in Agentware systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23386v2">Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Decoder-only large language models (LLMs) are increasingly used to build embedding models that effectively encode the semantic information of natural language texts into dense vector representations for various embedding tasks. However, many existing methods primarily focus on removing the causal attention mask in LLMs to enable bidirectional attention, potentially undermining the model's ability to extract semantic information acquired during pretraining. Additionally, leading unidirectional approaches often rely on extra input text to overcome the inherent limitations of causal attention, inevitably increasing computational costs. In this work, we propose Causal2Vec, a general-purpose embedding model tailored to enhance the performance of decoder-only LLMs without altering their original architectures or introducing significant computational overhead. Specifically, we first employ a lightweight BERT-style model to pre-encode the input text into a single Contextual token, which is then prepended to the LLM's input sequence, allowing each token to capture contextualized information even without attending to future tokens. Furthermore, to mitigate the recency bias introduced by last-token pooling and help LLMs better leverage the semantic information encoded in the Contextual token, we concatenate the last hidden states of Contextual and EOS tokens as the final text embedding. In practice, Causal2Vec achieves state-of-the-art performance on the Massive Text Embeddings Benchmark (MTEB) among models trained solely on publicly available retrieval datasets, while reducing the required sequence length by up to 85% and inference time by up to 82% compared to best-performing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21589v3">Middo: Model-Informed Dynamic Data Optimization for Enhanced LLM Fine-Tuning via Closed-Loop Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted by EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      Supervised Fine-Tuning (SFT) Large Language Models (LLM) fundamentally rely on high-quality training data. While data selection and data synthesis are two common strategies to improve data quality, existing approaches often face limitations in static dataset curation that fail to adapt to evolving model capabilities. In this paper, we introduce Middo, a self-evolving Model-informed dynamic data optimization framework that uses model-aware data selection and context-preserving data refinement. Unlike conventional one-off filtering/synthesis methods, our framework establishes a closed-loop optimization system: (1) A self-referential diagnostic module proactively identifies suboptimal samples through tri-axial model signals - loss patterns (complexity), embedding cluster dynamics (diversity), and self-alignment scores (quality); (2) An adaptive optimization engine then transforms suboptimal samples into pedagogically valuable training points while preserving semantic integrity; (3) This optimization process continuously evolves with model capability through dynamic learning principles. Experiments on multiple benchmarks demonstrate that our Middo consistently enhances the quality of seed data and boosts LLM's performance with improving accuracy by 7.15% on average while maintaining the original dataset scale. This work establishes a new paradigm for sustainable LLM training through dynamic human-AI co-evolution of data and models. Our datasets, models, and code are coming soon. Our datasets, models, and code are publicly available at https://github.com/Word2VecT/Middo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22777v3">MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Dialogue Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 October ARR
    </div>
    <details class="paper-abstract">
      Evaluating the quality of open-domain chatbots has become increasingly reliant on LLMs acting as automatic judges. However, existing meta-evaluation benchmarks are static, outdated, and lacking in multilingual coverage, limiting their ability to fully capture subtle weaknesses in evaluation. We introduce MEDAL, an automated multi-agent framework for curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. Then, a strong LLM (GPT-4.1) is used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. Using MEDAL, we uncover that state-of-the-art judges fail to reliably detect nuanced issues such as lack of empathy, commonsense, or relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01349v3">Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized product recommenders, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making such manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive evaluation across models of varying scale, we find that certain biases, such as social proof, consistently boost product recommendation rate and ranking, while others, like scarcity and exclusivity, surprisingly reduce visibility. Our results demonstrate that cognitive biases are deeply embedded in state-of-the-art LLMs, leading to highly unpredictable behavior in product recommendations and posing significant challenges for effective mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15940v1">Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The scaling law for large language models (LLMs) depicts that the path towards machine intelligence necessitates training at large scale. Thus, companies continuously build large-scale GPU clusters, and launch training jobs that span over thousands of computing nodes. However, LLM pre-training presents unique challenges due to its complex communication patterns, where GPUs exchange data in sparse yet high-volume bursts within specific groups. Inefficient resource scheduling exacerbates bandwidth contention, leading to suboptimal training performance. This paper presents Arnold, a scheduling system summarizing our experience to effectively align LLM communication patterns with data center topology at scale. An in-depth characteristic study is performed to identify the impact of physical network topology to LLM pre-training jobs. Based on the insights, we develop a scheduling algorithm to effectively align communication patterns with the physical network topology in modern data centers. Through simulation experiments, we show the effectiveness of our algorithm in reducing the maximum spread of communication groups by up to $1.67$x. In production training, our scheduling system improves the end-to-end performance by $10.6\%$ when training with more than $9600$ GPUs, a significant improvement for our training pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15926v1">Beyond the Score: Uncertainty-Calibrated LLMs for Automated Essay Assessment</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted at EMNLP 2025 (Main Conference). Camera-ready version
    </div>
    <details class="paper-abstract">
      Automated Essay Scoring (AES) systems now reach near human agreement on some public benchmarks, yet real-world adoption, especially in high-stakes examinations, remains limited. A principal obstacle is that most models output a single score without any accompanying measure of confidence or explanation. We address this gap with conformal prediction, a distribution-free wrapper that equips any classifier with set-valued outputs and formal coverage guarantees. Two open-source large language models (Llama-3 8B and Qwen-2.5 3B) are fine-tuned on three diverse corpora (ASAP, TOEFL11, Cambridge-FCE) and calibrated at a 90 percent risk level. Reliability is assessed with UAcc, an uncertainty-aware accuracy that rewards models for being both correct and concise. To our knowledge, this is the first work to combine conformal prediction and UAcc for essay scoring. The calibrated models consistently meet the coverage target while keeping prediction sets compact, indicating that open-source, mid-sized LLMs can already support teacher-in-the-loop AES; we discuss scaling and broader user studies as future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16781v3">Evaluating Robustness of LLMs in Question Answering on Multilingual Noisy OCR Data</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted at CIKM 2025
    </div>
    <details class="paper-abstract">
      Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors - imperfect extraction of text, including character insertion, deletion, and substitution can significantly impact downstream tasks like question-answering (QA). In this work, we conduct a comprehensive analysis of how OCR-induced noise affects the performance of Multilingual QA Systems. To support this analysis, we introduce a multilingual QA dataset MultiOCR-QA, comprising 50K question-answer pairs across three languages, English, French, and German. The dataset is curated from OCR-ed historical documents, which include different levels and types of OCR noise. We then evaluate how different state-of-the-art Large Language Models (LLMs) perform under different error conditions, focusing on three major OCR error types. Our findings show that QA systems are highly prone to OCR-induced errors and perform poorly on noisy OCR text. By comparing model performance on clean versus noisy texts, we provide insights into the limitations of current approaches and emphasize the need for more noise-resilient QA systems in historical digitization contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15839v1">Multi-Physics: A Comprehensive Benchmark for Multimodal LLMs Reasoning on Chinese Multi-Subject Physics Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      While multimodal LLMs (MLLMs) demonstrate remarkable reasoning progress, their application in specialized scientific domains like physics reveals significant gaps in current evaluation benchmarks. Specifically, existing benchmarks often lack fine-grained subject coverage, neglect the step-by-step reasoning process, and are predominantly English-centric, failing to systematically evaluate the role of visual information. Therefore, we introduce \textbf {Multi-Physics} for Chinese physics reasoning, a comprehensive benchmark that includes 5 difficulty levels, featuring 1,412 image-associated, multiple-choice questions spanning 11 high-school physics subjects. We employ a dual evaluation framework to evaluate 20 different MLLMs, analyzing both final answer accuracy and the step-by-step integrity of their chain-of-thought. Furthermore, we systematically study the impact of difficulty level and visual information by comparing the model performance before and after changing the input mode. Our work provides not only a fine-grained resource for the community but also offers a robust methodology for dissecting the multimodal reasoning process of state-of-the-art MLLMs, and our dataset and code have been open-sourced: https://github.com/luozhongze/Multi-Physics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09996v2">From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 NeurIPS 2025 Accepted Paper
    </div>
    <details class="paper-abstract">
      Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15826v1">Campus AI vs. Commercial AI: How Customizations Shape Trust and Usage of LLM as-a-Service Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 arXiv admin note: text overlap with arXiv:2505.10490
    </div>
    <details class="paper-abstract">
      As the use of LLM chatbots by students and researchers becomes more prevalent, universities are pressed to develop AI strategies. One strategy that many universities pursue is to customize pre-trained LLM as-a-service (LLMaaS). While most studies on LLMaaS chatbots prioritize technical adaptations, we focus on psychological effects of user-salient customizations, such as interface changes. We assume that such customizations influence users' perception of the system and are therefore important in guiding safe and appropriate use. In a field study, we examine how students and employees (N = 526) at a German university perceive and use their institution's customized LLMaaS chatbot compared to ChatGPT. Participants using both systems (n = 116) reported greater trust, higher perceived privacy and less experienced hallucinations with their university's customized LLMaaS chatbot in contrast to ChatGPT. We discuss theoretical implications for research on calibrated trust, and offer guidance on the design and deployment of LLMaaS chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07570v2">OptiScene: LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Automatic indoor layout generation has attracted increasing attention due to its potential in interior design, virtual environment construction, and embodied AI. Existing methods fall into two categories: prompt-driven approaches that leverage proprietary LLM services (e.g., GPT APIs) and learning-based methods trained on layout data upon diffusion-based models. Prompt-driven methods often suffer from spatial inconsistency and high computational costs, while learning-based methods are typically constrained by coarse relational graphs and limited datasets, restricting their generalization to diverse room categories. In this paper, we revisit LLM-based indoor layout generation and present 3D-SynthPlace, a large-scale dataset that combines synthetic layouts generated via a 'GPT synthesize, Human inspect' pipeline, upgraded from the 3D-Front dataset. 3D-SynthPlace contains nearly 17,000 scenes, covering four common room types -- bedroom, living room, kitchen, and bathroom -- enriched with diverse objects and high-level spatial annotations. We further introduce OptiScene, a strong open-source LLM optimized for indoor layout generation, fine-tuned based on our 3D-SynthPlace dataset through our two-stage training. For the warum-up stage I, we adopt supervised fine-tuning (SFT), which is taught to first generate high-level spatial descriptions then conditionally predict concrete object placements. For the reinforcing stage II, to better align the generated layouts with human design preferences, we apply multi-turn direct preference optimization (DPO), which significantly improving layout quality and generation success rates. Extensive experiments demonstrate that OptiScene outperforms traditional prompt-driven and learning-based baselines. Moreover, OptiScene shows promising potential in interactive tasks such as scene editing and robot navigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19282v2">CORE-RAG: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 This paper is under continuous improvement
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the timeliness of knowledge and the factual accuracy of responses in Large Language Models (LLMs). However, the inclusion of excessive retrieved documents substantially increases the input length, leading to higher computational costs. Previous studies have attempted to compress retrieved documents into shorter texts before in-context integration, but such methods often compromise end-task performance. The lack of well-defined compression targets forces many approaches to rely on fixed heuristics, which cannot guarantee that the compressed content will effectively support the end task. To address these limitations, we propose CORE, a novel method designed to achieve lossless context compression for RAG. CORE employs reinforcement learning to optimize the compression process without relying on predefined compression labels, which enables the compressor to generate summaries that maximize the accuracy of answers generated by the LLM. Extensive experiments on four datasets demonstrate the superiority of our approach. With a high compression ratio of 3\%, our method not only avoids performance degradation compared to prepending full documents across all datasets but also improves the average Exact Match (EM) score by 3.3 points. The code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12123v2">MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 EMNLP 2025 Findings. Project page:https://sabijun.github.io/MT_RewardTreePage
    </div>
    <details class="paper-abstract">
      Process reward models (PRMs) have shown success in complex reasoning tasks for large language models (LLMs). However, their application to machine translation (MT) remains underexplored due to the lack of systematic methodologies and evaluation benchmarks. To address this gap, we introduce \textbf{MT-RewardTree}, a comprehensive framework for constructing, evaluating, and deploying process reward models in MT. Unlike traditional vanilla preference pair construction, we propose a novel method for automatically generating token-level preference pairs using approximate Monte Carlo Tree Search (MCTS), which mitigates the prohibitive cost of human annotation for fine-grained steps. Then, we establish the first MT-specific reward model benchmark and provide a systematic comparison of different reward modeling architectures, revealing that token-level supervision effectively captures fine-grained preferences. Experimental results demonstrate that our MT-PRM-Qwen-2.5-3B achieves state-of-the-art performance in both token-level and sequence-level evaluation given the same input prefix. Furthermore, we showcase practical applications where PRMs enable test-time alignment for LLMs without additional alignment training and significantly improve performance in hypothesis ensembling. Our work provides valuable insights into the role of reward models in MT research. Our code and data are released in \href{https://sabijun.github.io/MT_RewardTreePage/}{https://sabijun.github.io/MT\_RewardTreePage}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15739v1">Can LLMs Judge Debates? Evaluating Non-Linear Reasoning via Argumentation Theory Semantics</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at linear reasoning tasks but remain underexplored on non-linear structures such as those found in natural debates, which are best expressed as argument graphs. We evaluate whether LLMs can approximate structured reasoning from Computational Argumentation Theory (CAT). Specifically, we use Quantitative Argumentation Debate (QuAD) semantics, which assigns acceptability scores to arguments based on their attack and support relations. Given only dialogue-formatted debates from two NoDE datasets, models are prompted to rank arguments without access to the underlying graph. We test several LLMs under advanced instruction strategies, including Chain-of-Thought and In-Context Learning. While models show moderate alignment with QuAD rankings, performance degrades with longer inputs or disrupted discourse flow. Advanced prompting helps mitigate these effects by reducing biases related to argument length and position. Our findings highlight both the promise and limitations of LLMs in modeling formal argumentation semantics and motivate future work on graph-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15735v1">EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 5 pages, submitted to ICASSP 2026, September 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer broad utility but remain prone to hallucination and out-of-distribution (OOD) errors. We propose EigenTrack, an interpretable real-time detector that uses the spectral geometry of hidden activations, a compact global signature of model dynamics. By streaming covariance-spectrum statistics such as entropy, eigenvalue gaps, and KL divergence from random baselines into a lightweight recurrent classifier, EigenTrack tracks temporal shifts in representation structure that signal hallucination and OOD drift before surface errors appear. Unlike black- and grey-box methods, it needs only a single forward pass without resampling. Unlike existing white-box detectors, it preserves temporal context, aggregates global signals, and offers interpretable accuracy-latency trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13229v2">IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18947v2">LLMs in the SOC: An Empirical Study of Human-AI Collaboration in Security Operations Centres</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 21 pages, 9 figures, under review
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Security Operations Centres (SOCs) presents a transformative, yet still evolving, opportunity to reduce analyst workload through human-AI collaboration. However, their real-world application in SOCs remains underexplored. To address this gap, we present a longitudinal study of 3,090 analyst queries from 45 SOC analysts over 10 months. Our analysis reveals that analysts use LLMs as on-demand aids for sensemaking and context-building, rather than for making high-stakes determinations, preserving analyst decision authority. The majority of queries are related to interpreting low-level telemetry (e.g., commands) and refining technical communication through short (1-3 turn) interactions. Notably, 93% of queries align with established cybersecurity competencies (NICE Framework), underscoring the relevance of LLM use for SOC-related tasks. Despite variations in tasks and engagement, usage trends indicate a shift from occasional exploration to routine integration, with growing adoption and sustained use among a subset of analysts. We find that LLMs function as flexible, on-demand cognitive aids that augment, rather than replace, SOC expertise. Our study provides actionable guidance for designing context-aware, human-centred AI assistance in security operations, highlighting the need for further in-the-wild research on real-world analyst-LLM collaboration, challenges, and impacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04462v2">CARD: A Cache-Assisted Parallel Speculative Decoding Framework via Query-and-Correct Paradigm for Accelerating LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Speculative decoding (SD), where a draft model provides multiple candidate tokens for the target model to verify in parallel, has demonstrated significant potential for accelerating LLM inference. Yet, existing SD approaches adhere to a strict draft-then-verify paradigm, enforcing a sequential process that hampers performance and constrains the draft model's capacity. Moreover, rejecting a token in the candidate sequence invalidates all subsequent tokens, leading to wasted computation during drafting. To overcome these limitations, we propose a cache-assisted parallel speculative decoding framework called CARD, which employs a novel query-and-correct paradigm. Our approach decouples drafting from verification: the draft model populates a shared cache with candidate tokens, while the target model concurrently refines the draft's trajectory. This enables inference at near-draft-speed, effectively leveraging the draft model's efficiency without additional fine-tuning. Experimental results show that CARD significantly outperforms existing state-of-the-art methods, achieving up to a 4.83x acceleration over vanilla autoregressive decoding, with no fine-tuning required for either models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15640v1">Multilingual LLM Prompting Strategies for Medical English-Vietnamese Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 The work is under peer review
    </div>
    <details class="paper-abstract">
      Medical English-Vietnamese machine translation (En-Vi MT) is essential for healthcare access and communication in Vietnam, yet Vietnamese remains a low-resource and under-studied language. We systematically evaluate prompting strategies for six multilingual LLMs (0.5B-9B parameters) on the MedEV dataset, comparing zero-shot, few-shot, and dictionary-augmented prompting with Meddict, an English-Vietnamese medical lexicon. Results show that model scale is the primary driver of performance: larger LLMs achieve strong zero-shot results, while few-shot prompting yields only marginal improvements. In contrast, terminology-aware cues and embedding-based example retrieval consistently improve domain-specific translation. These findings underscore both the promise and the current limitations of multilingual LLMs for medical En-Vi MT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14405v3">RaCT: Ranking-aware Chain-of-Thought Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      In information retrieval, large language models (LLMs) have demonstrated remarkable potential in text reranking tasks by leveraging their sophisticated natural language understanding and advanced reasoning capabilities. However, conventional supervised fine-tuning approaches for specializing LLMs in ranking tasks often lead to significant degradation of the models' general-purpose abilities. To address this fundamental challenge, this paper presents a novel methodology that strategically combines Chain-of-Thought (CoT) prompting techniques with an innovative two-stage training pipeline consisting of Supervised Fine-Tuning followed by Ranking Preference Optimization (SFT-RPO). The Chain-of-Thought prompting component encourages models to explicitly articulate their reasoning process during ranking decisions, creating a transparent pathway from query-document analysis to final ranking scores while maintaining analytical capabilities throughout fine-tuning. Extensive experimental evaluations on the TREC Deep Learning datasets demonstrate that our proposed method achieves superior performance compared to existing state-of-the-art models, including RankZephyr, showing consistent improvements across multiple evaluation metrics such as normalized Discounted Cumulative Gain (nDCG). Most significantly, comprehensive assessments on the Massive Multitask Language Understanding (MMLU) benchmark reveal that our method successfully maintains robust performance across diverse reasoning tasks, providing strong empirical evidence for effective retention of general-purpose capabilities through strategic fine-tuning while achieving specialized performance improvements in text reranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00032v5">KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres. By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02534v2">Adaptive Self-improvement LLM Agentic System for ML Library Development</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      ML libraries, often written in architecture-specific programming languages (ASPLs) that target domain-specific architectures, are key to efficient ML systems. However, writing these high-performance ML libraries is challenging because it requires expert knowledge of ML algorithms and the ASPL. Large language models (LLMs), on the other hand, have shown general coding capabilities. However, challenges remain when using LLMs for generating ML libraries using ASPLs because 1) this task is complicated even for experienced human programmers and 2) there are limited code examples because of the esoteric and evolving nature of ASPLs. Therefore, LLMs need complex reasoning with limited data in order to complete this task. To address these challenges, we introduce an adaptive self-improvement agentic system. In order to evaluate the effectiveness of our system, we construct a benchmark of a typical ML library and generate ASPL code with both open and closed-source LLMs on this benchmark. Our results show improvements of up to $3.9\times$ over a baseline single LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14979v2">What Matters in LLM-Based Feature Extractor for Recommender? A Systematic Analysis of Prompts, Models, and Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 9 pages. Keywords: Recommender Systems, Large Language Models, Sequential Recommendation, Feature Extraction
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) to generate semantic features has been demonstrated as a powerful paradigm for enhancing Sequential Recommender Systems (SRS). This typically involves three stages: processing item text, extracting features with LLMs, and adapting them for downstream models. However, existing methods vary widely in prompting, architecture, and adaptation strategies, making it difficult to fairly compare design choices and identify what truly drives performance. In this work, we propose RecXplore, a modular analytical framework that decomposes the LLM-as-feature-extractor pipeline into four modules: data processing, semantic feature extraction, feature adaptation, and sequential modeling. Instead of proposing new techniques, RecXplore revisits and organizes established methods, enabling systematic exploration of each module in isolation. Experiments on four public datasets show that simply combining the best designs from existing techniques without exhaustive search yields up to 18.7% relative improvement in NDCG@5 and 12.7% in HR@5 over strong baselines. These results underscore the utility of modular benchmarking for identifying effective design patterns and promoting standardized research in LLM-enhanced recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20484v2">Enhancing LLM Language Adaption through Cross-lingual In-Context Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 12 pages, 6 figures, EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable multilingual capabilities despite English-dominated pre-training, attributed to cross-lingual mechanisms during pre-training. Existing methods for enhancing cross-lingual transfer remain constrained by parallel resources, suffering from limited linguistic and domain coverage. We propose Cross-lingual In-context Pre-training (CrossIC-PT), a simple and scalable approach that enhances cross-lingual transfer by leveraging semantically related bilingual texts via simple next-word prediction. We construct CrossIC-PT samples by interleaving semantic-related bilingual Wikipedia documents into a single context window. To access window size constraints, we implement a systematic segmentation policy to split long bilingual document pairs into chunks while adjusting the sliding window mechanism to preserve contextual coherence. We further extend data availability through a semantic retrieval framework to construct CrossIC-PT samples from web-crawled corpus. Experimental results demonstrate that CrossIC-PT improves multilingual performance on three models (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B) across six target languages, yielding performance gains of 3.79%, 3.99%, and 1.95%, respectively, with additional improvements after data augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15568v1">LiteLong: Resource-Efficient Long-Context Data Synthesis for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      High-quality long-context data is essential for training large language models (LLMs) capable of processing extensive documents, yet existing synthesis approaches using relevance-based aggregation face challenges of computational efficiency. We present LiteLong, a resource-efficient method for synthesizing long-context data through structured topic organization and multi-agent debate. Our approach leverages the BISAC book classification system to provide a comprehensive hierarchical topic organization, and then employs a debate mechanism with multiple LLMs to generate diverse, high-quality topics within this structure. For each topic, we use lightweight BM25 retrieval to obtain relevant documents and concatenate them into 128K-token training samples. Experiments on HELMET and Ruler benchmarks demonstrate that LiteLong achieves competitive long-context performance and can seamlessly integrate with other long-dependency enhancement methods. LiteLong makes high-quality long-context data synthesis more accessible by reducing both computational and data engineering costs, facilitating further research in long-context language training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15561v1">Small LLMs with Expert Blocks Are Good Enough for Hyperparamter Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Hyper-parameter Tuning (HPT) is a necessary step in machine learning (ML) pipelines but becomes computationally expensive and opaque with larger models. Recently, Large Language Models (LLMs) have been explored for HPT, yet most rely on models exceeding 100 billion parameters. We propose an Expert Block Framework for HPT using Small LLMs. At its core is the Trajectory Context Summarizer (TCS), a deterministic block that transforms raw training trajectories into structured context, enabling small LLMs to analyze optimization progress with reliability comparable to larger models. Using two locally-run LLMs (phi4:reasoning14B and qwen2.5-coder:32B) and a 10-trial budget, our TCS-enabled HPT pipeline achieves average performance within ~0.9 percentage points of GPT-4 across six diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14834v2">LLM Agents at the Roundtable: A Multi-Perspective and Dialectical Reasoning Framework for Essay Scoring</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has brought a new paradigm to automated essay scoring (AES), a long-standing and practical application of natural language processing in education. However, achieving human-level multi-perspective understanding and judgment remains a challenge. In this work, we propose Roundtable Essay Scoring (RES), a multi-agent evaluation framework designed to perform precise and human-aligned scoring under a zero-shot setting. RES constructs evaluator agents based on LLMs, each tailored to a specific prompt and topic context. Each agent independently generates a trait-based rubric and conducts a multi-perspective evaluation. Then, by simulating a roundtable-style discussion, RES consolidates individual evaluations through a dialectical reasoning process to produce a final holistic score that more closely aligns with human evaluation. By enabling collaboration and consensus among agents with diverse evaluation perspectives, RES outperforms prior zero-shot AES approaches. Experiments on the ASAP dataset using ChatGPT and Claude show that RES achieves up to a 34.86% improvement in average QWK over straightforward prompting (Vanilla) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09466v2">AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 19 pages, 6 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20474v3">MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15515v1">LLM Cache Bandit Revisited: Addressing Query Heterogeneity for Cost-Effective LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      This paper revisits the LLM cache bandit problem, with a special focus on addressing the query heterogeneity for cost-effective LLM inference. Previous works often assume uniform query sizes. Heterogeneous query sizes introduce a combinatorial structure for cache selection, making the cache replacement process more computationally and statistically challenging. We treat optimal cache selection as a knapsack problem and employ an accumulation-based strategy to effectively balance computational overhead and cache updates. In theoretical analysis, we prove that the regret of our algorithm achieves an $O(\sqrt{MNT})$ bound, improving the coefficient of $\sqrt{MN}$ compared to the $O(MN\sqrt{T})$ result in Berkeley, where $N$ is the total number of queries and $M$ is the cache size. Additionally, we also provide a problem-dependent bound, which was absent in previous works. The experiment rely on real-world data show that our algorithm reduces the total cost by approximately 12\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11022v5">DynamicNER: A Dynamic, Multilingual, and Fine-Grained Dataset for LLM-based Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 This paper is accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The advancements of Large Language Models (LLMs) have spurred a growing interest in their application to Named Entity Recognition (NER) methods. However, existing datasets are primarily designed for traditional machine learning methods and are inadequate for LLM-based methods, in terms of corpus selection and overall dataset design logic. Moreover, the prevalent fixed and relatively coarse-grained entity categorization in existing datasets fails to adequately assess the superior generalization and contextual understanding capabilities of LLM-based methods, thereby hindering a comprehensive demonstration of their broad application prospects. To address these limitations, we propose DynamicNER, the first NER dataset designed for LLM-based methods with dynamic categorization, introducing various entity types and entity type lists for the same entity in different context, leveraging the generalization of LLM-based NER better. The dataset is also multilingual and multi-granular, covering 8 languages and 155 entity types, with corpora spanning a diverse range of domains. Furthermore, we introduce CascadeNER, a novel NER method based on a two-stage strategy and lightweight LLMs, achieving higher accuracy on fine-grained tasks while requiring fewer computational resources. Experiments show that DynamicNER serves as a robust and effective benchmark for LLM-based NER methods. Furthermore, we also conduct analysis for traditional methods and LLM-based methods on our dataset. Our code and dataset are openly available at https://github.com/Astarojth/DynamicNER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14289v2">From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11651v2">70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large-scale AI models, such as Large Language Models (LLMs) and Diffusion Models (DMs), have grown rapidly in size, creating significant challenges for efficient deployment on resource-constrained hardware. In this paper, we introduce Dynamic-Length Float (DFloat11), a lossless compression framework that reduces LLM and DM size by 30% while preserving outputs that are bit-for-bit identical to the original model. DFloat11 is motivated by the low entropy in the BFloat16 weight representation of LLMs, which reveals significant inefficiency in the existing storage format. By applying entropy coding, DFloat11 assigns dynamic-length encodings to weights based on frequency, achieving near information-optimal compression without any loss of precision. To facilitate efficient inference with dynamic-length encodings, we develop a custom GPU kernel for fast online decompression. Our design incorporates the following: (i) compact, hierarchical lookup tables (LUTs) that fit within GPU SRAM for efficient decoding, (ii) a two-phase GPU kernel for coordinating thread read/write positions using lightweight auxiliary variables, and (iii) transformer-block-level decompression to minimize latency. Experiments on Llama 3.3, Qwen 3, Mistral 3, FLUX.1, and others validate our hypothesis that DFloat11 achieves around 30% model size reduction while preserving bit-for-bit identical outputs. Compared to a potential alternative of offloading parts of an uncompressed model to the CPU to meet memory constraints, DFloat11 achieves 2.3--46.2x higher throughput in token generation. With a fixed GPU memory budget, DFloat11 enables 5.7--14.9x longer generation lengths than uncompressed models. Notably, our method enables lossless inference of Llama 3.1 405B, an 810GB model, on a single node equipped with 8x80GB GPUs. Our code is available at https://github.com/LeanModels/DFloat11.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16456v1">GPO: Learning from Critical Steps to Improve LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in various domains, showing impressive potential on different tasks. Recently, reasoning LLMs have been proposed to improve the \textit{reasoning} or \textit{thinking} capabilities of LLMs to solve complex problems. Despite the promising results of reasoning LLMs, enhancing the multi-step reasoning capabilities of LLMs still remains a significant challenge. While existing optimization methods have advanced the LLM reasoning capabilities, they often treat reasoning trajectories as a whole, without considering the underlying critical steps within the trajectory. In this paper, we introduce \textbf{G}uided \textbf{P}ivotal \textbf{O}ptimization (GPO), a novel fine-tuning strategy that dives into the reasoning process to enable more effective improvements. GPO first identifies the `critical step' within a reasoning trajectory - a point that the model must carefully proceed to succeed at the problem. We locate the critical step by estimating the advantage function. GPO then resets the policy to the critical step, samples the new rollout and prioritizes the learning process on those rollouts. This focus allows the model to learn more effectively from pivotal moments within the reasoning process to improve the reasoning performance. We demonstrate that GPO is a general strategy that can be integrated with various optimization methods to improve reasoning performance. Besides theoretical analysis, our experiments across challenging reasoning benchmarks show that GPO can consistently and significantly enhance the performance of existing optimization methods, showcasing its effectiveness and generalizability in improving LLM reasoning by concentrating on pivotal moments within the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16443v1">LightCode: Compiling LLM Inference for Photonic-Electronic Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 9 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The growing demand for low-latency, energy-efficient inference in large language models (LLMs) has catalyzed interest in heterogeneous architectures. While GPUs remain dominant, they are poorly suited for integration with emerging domain-specific accelerators like the Photonic Tensor Units (PTUs), which offer low-power, high-throughput linear computation. This motivates hybrid compilation strategies that combine photonic and electronic resources. We present LightCode, a compiler framework and simulator for mapping LLM inference workloads across hybrid photonic-electronic systems. LightCode introduces the Stacked Graph, an intermediate representation that encodes multiple hardware-specific realizations of each tensor operation. Hardware assignment is formulated as a constrained subgraph selection problem optimized for latency or energy under parametric cost models. We evaluate LightCode on the prefill stage of GPT-2 and Llama-7B showing that under our workload and hardware assumptions, (i) Photonic hardware reduced energy by up to 50% in our simulated workloads at maximum sequence length; (ii) multiplexing and assignment strategy yielded latency improvements exceeding 10x; and (iii) Optimizing for latency or energy resulted in distinct hardware mappings in our simulations. LightCode offers a module, foundational framework and simulator for compiling LLMs to emerging photonic accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16442v1">Evaluating the Effectiveness and Scalability of LLM-Based Data Augmentation for Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 EMNLP 2025 (MAIN Conference)
    </div>
    <details class="paper-abstract">
      Compact dual-encoder models are widely used for retrieval owing to their efficiency and scalability. However, such models often underperform compared to their Large Language Model (LLM)-based retrieval counterparts, likely due to their limited world knowledge. While LLM-based data augmentation has been proposed as a strategy to bridge this performance gap, there is insufficient understanding of its effectiveness and scalability to real-world retrieval problems. Existing research does not systematically explore key factors such as the optimal augmentation scale, the necessity of using large augmentation models, and whether diverse augmentations improve generalization, particularly in out-of-distribution (OOD) settings. This work presents a comprehensive study of the effectiveness of LLM augmentation for retrieval, comprising over 100 distinct experimental settings of retrieval models, augmentation models and augmentation strategies. We find that, while augmentation enhances retrieval performance, its benefits diminish beyond a certain augmentation scale, even with diverse augmentation strategies. Surprisingly, we observe that augmentation with smaller LLMs can achieve performance competitive with larger augmentation models. Moreover, we examine how augmentation effectiveness varies with retrieval model pre-training, revealing that augmentation provides the most benefit to models which are not well pre-trained. Our insights pave the way for more judicious and efficient augmentation strategies, thus enabling informed decisions and maximizing retrieval performance while being more cost-effective. Code and augmented datasets accompanying this work are publicly available at https://aka.ms/DAGR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16534v2">Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16422v1">Evaluating CxG Generalisation in LLMs via Construction-Based NLI Fine Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      We probe large language models' ability to learn deep form-meaning mappings as defined by construction grammars. We introduce the ConTest-NLI benchmark of 80k sentences covering eight English constructions from highly lexicalized to highly schematic. Our pipeline generates diverse synthetic NLI triples via templating and the application of a model-in-the-loop filter. This provides aspects of human validation to ensure challenge and label reliability. Zero-shot tests on leading LLMs reveal a 24% drop in accuracy between naturalistic (88%) and adversarial data (64%), with schematic patterns proving hardest. Fine-tuning on a subset of ConTest-NLI yields up to 9% improvement, yet our results highlight persistent abstraction gaps in current LLMs and offer a scalable framework for evaluating construction-informed learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16399v1">VORTEX: Aligning Task Utility and Human Preferences through LLM-Guided Reward Shaping</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 28pages, 19figures
    </div>
    <details class="paper-abstract">
      In social impact optimization, AI decision systems often rely on solvers that optimize well-calibrated mathematical objectives. However, these solvers cannot directly accommodate evolving human preferences, typically expressed in natural language rather than formal constraints. Recent approaches address this by using large language models (LLMs) to generate new reward functions from preference descriptions. While flexible, they risk sacrificing the system's core utility guarantees. In this paper, we propose \texttt{VORTEX}, a language-guided reward shaping framework that preserves established optimization goals while adaptively incorporating human feedback. By formalizing the problem as multi-objective optimization, we use LLMs to iteratively generate shaping rewards based on verbal reinforcement and text-gradient prompt updates. This allows stakeholders to steer decision behavior via natural language without modifying solvers or specifying trade-off weights. We provide theoretical guarantees that \texttt{VORTEX} converges to Pareto-optimal trade-offs between utility and preference satisfaction. Empirical results in real-world allocation tasks demonstrate that \texttt{VORTEX} outperforms baselines in satisfying human-aligned coverage goals while maintaining high task performance. This work introduces a practical and theoretically grounded paradigm for human-AI collaborative optimization guided by natural language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16394v1">Evaluating Behavioral Alignment in Conflict Dialogue: A Multi-Dimensional Comparison of LLM Agents and Humans</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 Accepted to EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in socially complex, interaction-driven tasks, yet their ability to mirror human behavior in emotionally and strategically complex contexts remains underexplored. This study assesses the behavioral alignment of personality-prompted LLMs in adversarial dispute resolution by simulating multi-turn conflict dialogues that incorporate negotiation. Each LLM is guided by a matched Five-Factor personality profile to control for individual variation and enhance realism. We evaluate alignment across three dimensions: linguistic style, emotional expression (e.g., anger dynamics), and strategic behavior. GPT-4.1 achieves the closest alignment with humans in linguistic style and emotional dynamics, while Claude-3.7-Sonnet best reflects strategic behavior. Nonetheless, substantial alignment gaps persist. Our findings establish a benchmark for alignment between LLMs and humans in socially complex interactions, underscoring both the promise and the limitations of personality conditioning in dialogue modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01646v2">ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 EMNLP'25 Main Oral (42 pages, 10 figures, 11 tables), Nominations for Resource Award & Theme Paper Award
    </div>
    <details class="paper-abstract">
      We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social, and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1,136 Multiple-Choice Questions (MCQs) generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting Retrieval-Augmented Generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports, and recommendation documents from 7 authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of LLMs, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (0.5B to 671B) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies around 55--70%, highlighting a significant knowledge gap for LLMs in this specialized, interdisciplinary domain. However, models employing RAG demonstrate significant performance improvements, particularly for smaller models. For example, DeepSeek-R1-Distill-Qwen-14B improves from 63.82% (zero-shot) to 80.46% with RAG. These results demonstrate the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first comprehensive QA benchmark designed to rigorously evaluate LLMs on ESG and sustainability knowledge, providing a critical tool to advance trustworthy AI in this vital domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12570v2">Batched Self-Consistency Improves LLM Relevance Assessment and Ranking</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      LLM query-passage relevance assessment is typically studied using a one-by-one pointwise (PW) strategy where each LLM call judges one passage at a time. However, this strategy requires as many LLM calls as there are passages while also preventing information sharing between passages. We thus hypothesize that batched PW methods, which evaluate multiple passages per LLM call, can improve not only efficiency but also judgment quality -- by enabling content from multiple passages to be seen jointly. Moreover, batched PW methods may be better suited to harness the test-time scaling benefits of self-consistency -- the ensembling technique of repeating (potentially perturbed) LLM tasks in parallel and aggregating results -- since batching can naturally enable prompt diversification through varied batch permutations and compositions to create more robust ensembles. We evaluate several batched PW methods against one-by-one PW and listwise ranking baselines on LLM relevance assessment and ranking tasks, using three passage retrieval datasets and GPT-4o, Claude Sonnet 3, and Amazon Nova Pro. We show that batching can greatly amplify self-consistency benefits, making batched PW methods achieve the best performance while often reducing latency by an order of magnitude or more compared to one-by-one PW methods. For instance, on legal search, batched PW ranking with GPT-4o improves from 43.8% to 51.3% NDCG@10 when using 1 vs. 15 self-consistency calls, compared to one-by-one PW ranking improving from 44.9% to 46.8% and being 15.3x slower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11859v2">SouLLMate: An Adaptive LLM-Driven System for Advanced Mental Health Support and Assessment, Based on a Systematic Application Survey</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      Mental health issues significantly impact individuals' daily lives, yet many do not receive the help they need even with available online resources. This study aims to provide accessible, stigma-free, personalized, and real-time mental health support through cutting-edge AI technologies. It makes the following contributions: (1) Conducting an extensive survey of recent mental health support methods to identify prevalent functionalities and unmet needs. (2) Introducing SouLLMate, an adaptive LLM-driven system that integrates LLM technologies, Chain, Retrieval-Augmented Generation (RAG), prompt engineering, and domain knowledge. This system offers advanced features such as Suicide Risk Detection and Proactive Guidance Dialogue, and utilizes RAG for personalized profile uploads and Conversational Information Extraction. (3) Developing novel evaluation approaches to assess preliminary assessments and suicide risk detection, utilizing annotated real-life interview data and professionally labeled datasets indicating suicide tendencies. (4) Proposing Key Indicator Summarization (KIS) and Proactive Questioning Strategy (PQS) methods to enhance model performance and usability through context-sensitive response adjustments and semantic coherence evaluations. This study contributes to advancing mental health support technologies, potentially improving the accessibility and effectiveness of mental health care globally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16322v2">SouLLMate: An Application Enhancing Diverse Mental Health Support with Adaptive LLMs, Prompt Engineering, and RAG Techniques</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 26 pages, 19 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Mental health issues significantly impact individuals' daily lives, yet many do not receive the help they need even with available online resources. This study aims to provide diverse, accessible, stigma-free, personalized, and real-time mental health support through cutting-edge AI technologies. It makes the following contributions: (1) Conducting an extensive survey of recent mental health support methods to identify prevalent functionalities and unmet needs. (2) Introducing SouLLMate, an adaptive LLM-driven system that integrates LLM technologies, Chain, Retrieval-Augmented Generation (RAG), prompt engineering, and domain knowledge. This system offers advanced features such as Risk Detection and Proactive Guidance Dialogue, and utilizes RAG for personalized profile uploads and Conversational Information Extraction. (3) Developing novel evaluation approaches for preliminary assessments and risk detection via professionally annotated interview data and real-life suicide tendency data. (4) Proposing the Key Indicator Summarization (KIS), Proactive Questioning Strategy (PQS), and Stacked Multi-Model Reasoning (SMMR) methods to enhance model performance and usability through context-sensitive response adjustments, semantic coherence evaluations, and enhanced accuracy of long-context reasoning in language models. This study contributes to advancing mental health support technologies, potentially improving the accessibility and effectiveness of mental health care globally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16325v1">Overhearing LLM Agents: A Survey, Taxonomy, and Roadmap</a></div>
    <div class="paper-meta">
      📅 2025-09-19
      | 💬 8 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Imagine AI assistants that enhance conversations without interrupting them: quietly providing relevant information during a medical consultation, seamlessly preparing materials as teachers discuss lesson plans, or unobtrusively scheduling meetings as colleagues debate calendars. While modern conversational LLM agents directly assist human users with tasks through a chat interface, we study this alternative paradigm for interacting with LLM agents, which we call "overhearing agents." Rather than demanding the user's attention, overhearing agents continuously monitor ambient activity and intervene only when they can provide contextual assistance. In this paper, we present the first analysis of overhearing LLM agents as a distinct paradigm in human-AI interaction and establish a taxonomy of overhearing agent interactions and tasks grounded in a survey of works on prior LLM-powered agents and exploratory HCI studies. Based on this taxonomy, we create a list of best practices for researchers and developers building overhearing agent systems. Finally, we outline the remaining research gaps and reveal opportunities for future research in the overhearing paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16293v1">Robust LLM Training Infrastructure at ByteDance</a></div>
    <div class="paper-meta">
      📅 2025-09-19
    </div>
    <details class="paper-abstract">
      The training scale of large language models (LLMs) has reached tens of thousands of GPUs and is still continuously expanding, enabling faster learning of larger models. Accompanying the expansion of the resource scale is the prevalence of failures (CUDA error, NaN values, job hang, etc.), which poses significant challenges to training stability. Any large-scale LLM training infrastructure should strive for minimal training interruption, efficient fault diagnosis, and effective failure tolerance to enable highly efficient continuous training. This paper presents ByteRobust, a large-scale GPU infrastructure management system tailored for robust and stable training of LLMs. It exploits the uniqueness of LLM training process and gives top priorities to detecting and recovering failures in a routine manner. Leveraging parallelisms and characteristics of LLM training, ByteRobust enables high-capacity fault tolerance, prompt fault demarcation, and localization with an effective data-driven approach, comprehensively ensuring continuous and efficient training of LLM tasks. ByteRobust is deployed on a production GPU platform with over 200,000 GPUs and achieves 97% ETTR for a three-month training job on 9,600 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14558v1">LLM Jailbreak Detection for (Almost) Free!</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14543v1">Catch Me If You Can? Not Yet: LLMs Still Struggle to Imitate the Implicit Writing Styles of Everyday Authors</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 EMNLP 2025 (Findings)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into personal writing tools, a critical question arises: can LLMs faithfully imitate an individual's writing style from just a few examples? Personal style is often subtle and implicit, making it difficult to specify through prompts yet essential for user-aligned generation. This work presents a comprehensive evaluation of state-of-the-art LLMs' ability to mimic personal writing styles via in-context learning from a small number of user-authored samples. We introduce an ensemble of complementary metrics-including authorship attribution, authorship verification, style matching, and AI detection-to robustly assess style imitation. Our evaluation spans over 40000 generations per model across domains such as news, email, forums, and blogs, covering writing samples from more than 400 real-world authors. Results show that while LLMs can approximate user styles in structured formats like news and email, they struggle with nuanced, informal writing in blogs and forums. Further analysis on various prompting strategies such as number of demonstrations reveal key limitations in effective personalization. Our findings highlight a fundamental gap in personalized LLM adaptation and the need for improved techniques to support implicit, style-consistent generation. To aid future research and for reproducibility, we open-source our data and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10901v2">3DS: Medical Domain Adaptation of LLMs via Decomposed Difficulty-based Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted to EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large Language Models(LLMs) excel in general tasks but struggle in specialized domains like healthcare due to limited domain-specific knowledge.Supervised Fine-Tuning(SFT) data construction for domain adaptation often relies on heuristic methods, such as GPT-4 annotation or manual data selection, with a data-centric focus on presumed diverse, high-quality datasets. However, these methods overlook the model's inherent knowledge distribution, introducing noise, redundancy, and irrelevant data, leading to a mismatch between the selected data and the model's learning task, resulting in suboptimal performance. To address this, we propose a two-stage model-centric data selection framework, Decomposed Difficulty Data Selection (3DS), which aligns data with the model's knowledge distribution for optimized adaptation. In Stage1, we apply Prompt-Driven Data Selection via Explicit Alignment, where the the model filters irrelevant or redundant data based on its internal knowledge. In Stage2, we perform Decomposed Difficulty Data Selection, where data selection is guided by our defined difficulty decomposition, using three metrics: Instruction Understanding, Response Confidence, and Response Correctness. Additionally, an attention-based importance weighting mechanism captures token importance for more accurate difficulty calibration. This two-stage approach ensures the selected data is not only aligned with the model's knowledge and preferences but also appropriately challenging for the model to learn, leading to more effective and targeted domain adaptation. In the case study of the medical domain, our extensive experiments on real-world healthcare datasets demonstrate the superiority of 3DS over exisiting methods in accuracy by over 5.29%. Our dataset and code has been open-sourced at https://github.com/PuppyKnightUniversity/3DS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05128v2">DiCoRe: Enhancing Zero-shot Event Detection via Divergent-Convergent LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted at EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Zero-shot Event Detection (ED), the task of identifying event mentions in natural language text without any training data, is critical for document understanding in specialized domains. Understanding the complex event ontology, extracting domain-specific triggers from the passage, and structuring them appropriately overloads and limits the utility of Large Language Models (LLMs) for zero-shot ED. To this end, we propose DiCoRe, a divergent-convergent reasoning framework that decouples the task of ED using Dreamer and Grounder. Dreamer encourages divergent reasoning through open-ended event discovery, which helps to boost event coverage. Conversely, Grounder introduces convergent reasoning to align the free-form predictions with the task-specific instructions using finite-state machine guided constrained decoding. Additionally, an LLM-Judge verifies the final outputs to ensure high precision. Through extensive experiments on six datasets across five domains and nine LLMs, we demonstrate how DiCoRe consistently outperforms prior zero-shot, transfer-learning, and reasoning baselines, achieving 4-7% average F1 gains over the best baseline -- establishing DiCoRe as a strong zero-shot ED framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09402v2">Read Before You Think: Mitigating LLM Comprehension Failures with Step-by-Step Reading</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Done in November 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often fail on complex reasoning tasks due to flawed question comprehension, not just flawed logic. This paper presents a systematic investigation into these comprehension failures. Our work yields three key insights: (1) the step-by-step principle, effective for calculation, can be migrated to the reading process to enhance comprehension; (2) increasing the proportion of question-related tokens (e.g., via repetition) succeeds by refocusing attention, a mechanism that can be explicitly controlled; and (3) backward dependencies represent a core bottleneck for decoder-only models that persists even with strong methods like Chain-of-Thought. Based on these findings, we introduce the Step-by-Step Reading (SSR) family of prompts. This multi-stage approach culminates in SSR++, a method specifically engineered to deepen model comprehension by guiding it to parse questions with finer granularity, focus attention on critical tokens, and resolve backward dependencies through iterative re-contextualization. SSR++ sets a new state-of-the-art on multiple reasoning benchmarks, and our analysis confirms it works by directly mitigating semantic misunderstanding. These results demonstrate that guiding how a model reads is a powerful and efficient method for improving its reasoning ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04083v2">A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 New experiments added with the GPT-OSS-120B model
    </div>
    <details class="paper-abstract">
      Objective: Zero-shot methodology promises to cut down on costs of dataset annotation and domain expertise needed to make use of NLP. Generative large language models trained to align with human goals have achieved high zero-shot performance across a wide variety of tasks. As of yet, it is unclear how well these models perform on biomedical relation extraction (RE). To address this knowledge gap, we explore patterns in the performance of OpenAI LLMs across a diverse sampling of RE tasks. Methods: We use OpenAI GPT-4-turbo and OpenAI's reasoning models o1 and GPT-OSS to conduct end-to-end RE experiments on seven datasets. We use the JSON generation capabilities of GPT models to generate structured output in two ways: (1) by defining an explicit schema describing the structure of relations, and (2) using a setting that infers the structure from the prompt language. Results: Our work is the first to study and compare the performance of the GPT-4, o1 and GPT-OSS for the end-to-end zero-shot biomedical RE task across a broad array of datasets. We found the zero-shot performances to be proximal to that of fine-tuned methods. The limitations of this approach are that it performs poorly on instances containing many relations and errs on the boundaries of textual mentions. Conclusion: LLMs exhibit promising zero-shot capabilities in complex biomedical RE tasks, offering competitive performance with reduced dataset curation costs and NLP modeling needs but with increased perpetual compute costs. Addressing the limitations we identify could further boost reliability. The code, data, and prompts for all our experiments are publicly available for additional benchmarking by the community: https://github.com/bionlproc/ZeroShotRE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10179v2">Benchmark of stylistic variation in LLM-generated texts</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Data and scripts: https://osf.io/hs7xt/. Interactive charts: https://www.korpus.cz/stylisticbenchmark/
    </div>
    <details class="paper-abstract">
      This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15555v2">Bayesian Concept Bottleneck Models with LLM Priors</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 2025 Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      Concept Bottleneck Models (CBMs) have been proposed as a compromise between white-box and black-box models, aiming to achieve interpretability without sacrificing accuracy. The standard training procedure for CBMs is to predefine a candidate set of human-interpretable concepts, extract their values from the training data, and identify a sparse subset as inputs to a transparent prediction model. However, such approaches are often hampered by the tradeoff between exploring a sufficiently large set of concepts versus controlling the cost of obtaining concept extractions, resulting in a large interpretability-accuracy tradeoff. This work investigates a novel approach that sidesteps these challenges: BC-LLM iteratively searches over a potentially infinite set of concepts within a Bayesian framework, in which Large Language Models (LLMs) serve as both a concept extraction mechanism and prior. Even though LLMs can be miscalibrated and hallucinate, we prove that BC-LLM can provide rigorous statistical inference and uncertainty quantification. Across image, text, and tabular datasets, BC-LLM outperforms interpretable baselines and even black-box models in certain settings, converges more rapidly towards relevant concepts, and is more robust to out-of-distribution samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15455v1">IMPQ: Interaction-Aware Layerwise Mixed Precision Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) promise impressive capabilities, yet their multi-billion-parameter scale makes on-device or low-resource deployment prohibitive. Mixed-precision quantization offers a compelling solution, but existing methods struggle when the average precision drops below four bits, as they rely on isolated, layer-specific metrics that overlook critical inter-layer interactions affecting overall performance. In this paper, we propose two innovations to address these limitations. First, we frame the mixed-precision quantization problem as a cooperative game among layers and introduce Shapley-based Progressive Quantization Estimation (SPQE) to efficiently obtain accurate Shapley estimates of layer sensitivities and inter-layer interactions. Second, building upon SPQE, we propose Interaction-aware Mixed-Precision Quantization (IMPQ) which translates these Shapley estimates into a binary quadratic optimization formulation, assigning either 2 or 4-bit precision to layers under strict memory constraints. Comprehensive experiments conducted on Llama-3, Gemma-2, and Qwen-3 models across three independent PTQ backends (Quanto, HQQ, GPTQ) demonstrate IMPQ's scalability and consistently superior performance compared to methods relying solely on isolated metrics. Across average precisions spanning 4 bit down to 2 bit, IMPQ cuts Perplexity by 20 to 80 percent relative to the best baseline, with the margin growing as the bit-width tightens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21772v2">Calibrating LLM Confidence by Probing Perturbed Representation Stability</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Miscalibration in Large Language Models (LLMs) undermines their reliability, highlighting the need for accurate confidence estimation. We introduce CCPS (Calibrating LLM Confidence by Probing Perturbed Representation Stability), a novel method analyzing internal representational stability in LLMs. CCPS applies targeted adversarial perturbations to final hidden states, extracts features reflecting the model's response to these perturbations, and uses a lightweight classifier to predict answer correctness. CCPS was evaluated on LLMs from 8B to 32B parameters (covering Llama, Qwen, and Mistral architectures) using MMLU and MMLU-Pro benchmarks in both multiple-choice and open-ended formats. Our results show that CCPS significantly outperforms current approaches. Across four LLMs and three MMLU variants, CCPS reduces Expected Calibration Error by approximately 55% and Brier score by 21%, while increasing accuracy by 5 percentage points, Area Under the Precision-Recall Curve by 4 percentage points, and Area Under the Receiver Operating Characteristic Curve by 6 percentage points, all relative to the strongest prior method. CCPS delivers an efficient, broadly applicable, and more accurate solution for estimating LLM confidence, thereby improving their trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v5">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 10 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15474v3">Subjective Behaviors and Preferences in LLM: Language of Browsing</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15335v1">PolBiX: Detecting LLMs' Political Bias in Fact-Checking through X-phemisms</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly used in applications requiring objective assessment, which could be compromised by political bias. Many studies found preferences for left-leaning positions in LLMs, but downstream effects on tasks like fact-checking remain underexplored. In this study, we systematically investigate political bias through exchanging words with euphemisms or dysphemisms in German claims. We construct minimal pairs of factually equivalent claims that differ in political connotation, to assess the consistency of LLMs in classifying them as true or false. We evaluate six LLMs and find that, more than political leaning, the presence of judgmental words significantly influences truthfulness assessment. While a few models show tendencies of political bias, this is not mitigated by explicitly calling for objectivism in prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15283v1">Evaluating the Limitations of Local LLMs in Solving Complex Programming Challenges</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 Comments: 16 pages, 3 figures, 8 tables, accepted to CCSC Eastern 2025
    </div>
    <details class="paper-abstract">
      This study examines the performance of today's open-source, locally hosted large-language models (LLMs) in handling complex competitive programming tasks with extended problem descriptions and contexts. Building on the original Framework for AI-driven Code Generation Evaluation (FACE), the authors retrofit the pipeline to work entirely offline through the Ollama runtime, collapsing FACE's sprawling per-problem directory tree into a handful of consolidated JSON files, and adding robust checkpointing so multi-day runs can resume after failures. The enhanced framework generates, submits, and records solutions for the full Kattis corpus of 3,589 problems across eight code-oriented models ranging from 6.7-9 billion parameters. The submission results show that the overall pass@1 accuracy is modest for the local models, with the best models performing at approximately half the acceptance rate of the proprietary models, Gemini 1.5 and ChatGPT-4. These findings expose a persistent gap between private, cost-controlled LLM deployments and state-of-the-art proprietary services, yet also highlight the rapid progress of open models and the practical benefits of an evaluation workflow that organizations can replicate on in-house hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03853v6">Dr. Jekyll and Mr. Hyde: Two Faces of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being integrated into applications such as chatbots or email assistants. To prevent improper responses, safety mechanisms, such as Reinforcement Learning from Human Feedback (RLHF), are implemented in them. In this work, we bypass these safety measures for ChatGPT, Gemini, and Deepseek by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information when querying ChatGPT, Gemini, and Deepseek. We show that these chatbots are vulnerable to this attack by getting dangerous information for 40 out of 40 illicit questions in GPT-4.1-mini, Gemini-1.5-flash, 39 out of 40 in GPT-4o-mini, 38 out of 40 in GPT-3.5-turbo, and 2 out of 2 cases in Gemini-2.5-flash and DeepSeek V3. The attack can be carried out manually or automatically using a support LLM, and has proven effective against models deployed between 2023 and 2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15260v1">Toxicity Red-Teaming: Benchmarking LLM Safety in Singapore's Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 9 pages, EMNLP 2025
    </div>
    <details class="paper-abstract">
      The advancement of Large Language Models (LLMs) has transformed natural language processing; however, their safety mechanisms remain under-explored in low-resource, multilingual settings. Here, we aim to bridge this gap. In particular, we introduce \textsf{SGToxicGuard}, a novel dataset and evaluation framework for benchmarking LLM safety in Singapore's diverse linguistic context, including Singlish, Chinese, Malay, and Tamil. SGToxicGuard adopts a red-teaming approach to systematically probe LLM vulnerabilities in three real-world scenarios: \textit{conversation}, \textit{question-answering}, and \textit{content composition}. We conduct extensive experiments with state-of-the-art multilingual LLMs, and the results uncover critical gaps in their safety guardrails. By offering actionable insights into cultural sensitivity and toxicity mitigation, we lay the foundation for safer and more inclusive AI systems in linguistically diverse environments.\footnote{Link to the dataset: https://github.com/Social-AI-Studio/SGToxicGuard.} \textcolor{red}{Disclaimer: This paper contains sensitive content that may be disturbing to some readers.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12566v3">Exploring the Impact of Personality Traits on LLM Bias and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-09-18
    </div>
    <details class="paper-abstract">
      With the different roles that AI is expected to play in human life, imbuing large language models (LLMs) with different personalities has attracted increasing research interests. While the "personification" enhances human experiences of interactivity and adaptability of LLMs, it gives rise to critical concerns about content safety, particularly regarding bias, sentiment and toxicity of LLM generation. This study explores how assigning different personality traits to LLMs affects the toxicity and biases of their outputs. Leveraging the widely accepted HEXACO personality framework developed in social psychology, we design experimentally sound prompts to test three LLMs' performance on three toxic and bias benchmarks. The findings demonstrate the sensitivity of all three models to HEXACO personality traits and, more importantly, a consistent variation in the biases, negative sentiment and toxicity of their output. In particular, adjusting the levels of several personality traits can effectively reduce bias and toxicity in model performance, similar to humans' correlations between personality traits and toxic behaviors. The findings highlight the additional need to examine content safety besides the efficiency of training or fine-tuning methods for LLM personification. They also suggest a potential for the adjustment of personalities to be a simple and low-cost method to conduct controlled text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16275v1">SecureFixAgent: A Hybrid LLM Agent for Automated Python Static Vulnerability Repair</a></div>
    <div class="paper-meta">
      📅 2025-09-18
      | 💬 6 pages, 3 figures, 4 tables, 1 algorithm, accepted in the Robustness and Security of Large Language Models (ROSE-LLM) special session at ICMLA 2025
    </div>
    <details class="paper-abstract">
      Modern software development pipelines face growing challenges in securing large codebases with extensive dependencies. Static analysis tools like Bandit are effective at vulnerability detection but suffer from high false positives and lack repair capabilities. Large Language Models (LLMs), in contrast, can suggest fixes but often hallucinate changes and lack self-validation. We present SecureFixAgent, a hybrid repair framework integrating Bandit with lightweight local LLMs (<8B parameters) in an iterative detect-repair-validate loop. To improve precision, we apply parameter-efficient LoRA-based fine-tuning on a diverse, curated dataset spanning multiple Python project domains, mitigating dataset bias and reducing unnecessary edits. SecureFixAgent uses Bandit for detection, the LLM for candidate fixes with explanations, and Bandit re-validation for verification, all executed locally to preserve privacy and reduce cloud reliance. Experiments show SecureFixAgent reduces false positives by 10.8% over static analysis, improves fix accuracy by 13.51%, and lowers false positives by 5.46% compared to pre-trained LLMs, typically converging within three iterations. Beyond metrics, developer studies rate explanation quality 4.5/5, highlighting its value for human trust and adoption. By combining verifiable security improvements with transparent rationale in a resource-efficient local framework, SecureFixAgent advances trustworthy, automated vulnerability remediation for modern pipelines.
    </details>
</div>
