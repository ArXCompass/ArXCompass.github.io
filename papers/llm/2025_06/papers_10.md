# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- Part 10
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18433v2">Easy2Hard-Bench: Standardized Difficulty Labels for Profiling LLM Performance and Generalization</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 NeurIPS 2024 Datasets and Benchmarks Track
    </div>
    <details class="paper-abstract">
      While generalization over tasks from easy to hard is crucial to profile language models (LLMs), the datasets with fine-grained difficulty annotations for each problem across a broad range of complexity are still blank. Aiming to address this limitation, we present Easy2Hard-Bench, a consistently formatted collection of 6 benchmark datasets spanning various domains, such as mathematics and programming problems, chess puzzles, and reasoning questions. Each problem within these datasets is annotated with numerical difficulty scores. To systematically estimate problem difficulties, we collect abundant performance data on attempts to each problem by humans in the real world or LLMs on the prominent leaderboard. Leveraging the rich performance data, we apply well-established difficulty ranking systems, such as Item Response Theory (IRT) and Glicko-2 models, to uniformly assign numerical difficulty scores to problems. Moreover, datasets in Easy2Hard-Bench distinguish themselves from previous collections by a higher proportion of challenging problems. Through extensive experiments with six state-of-the-art LLMs, we provide a comprehensive analysis of their performance and generalization capabilities across varying levels of difficulty, with the aim of inspiring future research in LLM generalization. The datasets are available at https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07900v1">MiniCPM4: Ultra-Efficient LLMs on End Devices</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 MiniCPM4 Technical Report
    </div>
    <details class="paper-abstract">
      This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose CPM.cu that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Sufficient evaluation results show that MiniCPM4 outperforms open-source models of similar size across multiple benchmarks, highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B demonstrates significant speed improvements over Qwen3-8B when processing long sequences. Through further adaptation, MiniCPM4 successfully powers diverse applications, including trustworthy survey generation and tool use with model context protocol, clearly showcasing its broad usability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12452v2">Introspective Growth: Automatically Advancing LLM Expertise in Technology Judgment</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 We open-source our patent dataset at https://huggingface.co/datasets/UchiKlab/patent_understanding
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly demonstrate signs of conceptual understanding, yet much of their internal knowledge remains latent, loosely structured, and difficult to access or evaluate. We propose self-questioning as a lightweight and scalable strategy to improve LLMs' understanding, particularly in domains where success depends on fine-grained semantic distinctions. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. The benchmark centers on a pairwise differentiation task: can a model distinguish between closely related but substantively different inventions? We show that compared to placebo scientific information, prompting LLMs to generate and answer their own questions - targeting the background knowledge required for the task - significantly improves performance. These self-generated questions and answers activate otherwise underutilized internal knowledge. Allowing LLMs to retrieve answers from external scientific texts further enhances performance, suggesting that model knowledge is compressed and lacks the full richness of the training data. We also find that chain-of-thought prompting and self-questioning converge, though self-questioning remains more effective for improving understanding of technical concepts. Notably, we uncover an asymmetry in prompting: smaller models often generate more fundamental, more open-ended, better-aligned questions for mid-sized models than large models do, revealing a new strategy for cross-model collaboration. Altogether, our findings establish self-questioning as both a practical mechanism for automatically improving LLM comprehension, especially in domains with sparse and underrepresented knowledge, and a diagnostic probe of how internal and external knowledge are organized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01887v4">Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      In-Context Reinforcement Learning (ICRL) is a frontier paradigm to solve Reinforcement Learning (RL) problems in the foundation model era. While ICRL capabilities have been demonstrated in transformers through task-specific training, the potential of Large Language Models (LLMs) out-of-the-box remains largely unexplored. This paper investigates whether LLMs can generalize cross-domain to perform ICRL under the problem of Dueling Bandits (DB), a stateless preference-based RL setting. We find that the top-performing LLMs exhibit a notable zero-shot capacity for relative decision-making, which translates to low short-term weak regret across all DB environment instances by quickly including the best arm in duels. However, an optimality gap still exists between LLMs and classic DB algorithms in terms of strong regret. LLMs struggle to converge and consistently exploit even when explicitly prompted to do so, and are sensitive to prompt variations. To bridge this gap, we propose an agentic flow framework: LLM with Enhanced Algorithmic Dueling (LEAD), which integrates off-the-shelf DB algorithm support with LLM agents through fine-grained adaptive interplay. We show that LEAD has theoretical guarantees inherited from classic DB algorithms on both weak and strong regret. We validate its efficacy and robustness even with noisy and adversarial prompts. The design of such an agentic framework sheds light on how to enhance the trustworthiness of general-purpose LLMs generalized to in-context decision-making tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06091v2">MIRIAD: Augmenting LLMs with millions of medical query-response pairs</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      LLMs are bound to transform healthcare with advanced decision support and flexible chat assistants. However, LLMs are prone to generate inaccurate medical content. To ground LLMs in high-quality medical knowledge, LLMs have been equipped with external knowledge via RAG, where unstructured medical knowledge is split into small text chunks that can be selectively retrieved and integrated into the LLMs context. Yet, existing RAG pipelines rely on raw, unstructured medical text, which can be noisy, uncurated and difficult for LLMs to effectively leverage. Systematic approaches to organize medical knowledge to best surface it to LLMs are generally lacking. To address these challenges, we introduce MIRIAD, a large-scale, curated corpus of 5,821,948 medical QA pairs, each rephrased from and grounded in a passage from peer-reviewed medical literature using a semi-automated pipeline combining LLM generation, filtering, grounding, and human annotation. Unlike prior medical corpora, which rely on unstructured text, MIRIAD encapsulates web-scale medical knowledge in an operationalized query-response format, which enables more targeted retrieval. Experiments on challenging medical QA benchmarks show that augmenting LLMs with MIRIAD improves accuracy up to 6.7% compared to unstructured RAG baselines with the same source corpus and with the same amount of retrieved text. Moreover, MIRIAD improved the ability of LLMs to detect medical hallucinations by 22.5 to 37% (increase in F1 score). We further introduce MIRIAD-Atlas, an interactive map of MIRIAD spanning 56 medical disciplines, enabling clinical users to visually explore, search, and refine medical knowledge. MIRIAD promises to unlock a wealth of down-stream applications, including medical information retrievers, enhanced RAG applications, and knowledge-grounded chat interfaces, which ultimately enables more reliable LLM applications in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07795v1">LLM Unlearning Should Be Form-Independent</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) unlearning aims to erase or suppress undesirable knowledge within the model, offering promise for controlling harmful or private information to prevent misuse. However, recent studies highlight its limited efficacy in real-world scenarios, hindering practical adoption. In this study, we identify a pervasive issue underlying many downstream failures: the effectiveness of existing unlearning methods heavily depends on the form of training samples and frequently fails to generalize to alternate expressions of the same knowledge. We formally characterize this problem as Form-Dependent Bias and systematically investigate its specific manifestation patterns across various downstream tasks. To quantify its prevalence and support future research, we introduce ORT, a novel benchmark designed to evaluate the robustness of unlearning methods against variations in knowledge expression. Results reveal that Form-Dependent Bias is both widespread and severe among current techniques. We argue that LLM unlearning should be form-independent to address the endless forms of downstream tasks encountered in real-world security-critical scenarios. Towards this goal, we introduce Rank-one Concept Redirection (ROCR), a novel training-free method, as a promising solution path. ROCR performs unlearning by targeting the invariants in downstream tasks, specifically the activated dangerous concepts. It is capable of modifying model parameters within seconds to redirect the model's perception of a specific unlearning target concept to another harmless concept. Extensive experiments demonstrate that ROCR significantly improves unlearning effectiveness compared to traditional methods while generating highly natural outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01804v2">A Study on the MCP x A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      This paper provides an in-depth technical analysis and implementation methodology of the open-source Agent-to-Agent (A2A) protocol developed by Google and the Model Context Protocol (MCP) introduced by Anthropic. While the evolution of LLM-based autonomous agents is rapidly accelerating, efficient interactions among these agents and their integration with external systems remain significant challenges. In modern AI systems, collaboration between autonomous agents and integration with external tools have become essential elements for building practical AI applications. A2A offers a standardized communication method that enables agents developed in heterogeneous environments to collaborate effectively, while MCP provides a structured I/O framework for agents to connect with external tools and resources. Prior studies have focused primarily on the features and applications of either A2A or MCP individually. In contrast, this study takes an integrated approach, exploring how the two protocols can complement each other to address interoperability issues and facilitate efficient collaboration within complex agent ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12091v4">Is poisoning a real threat to LLM alignment? Maybe more so than you think</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07751v1">Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstraL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20118v3">OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 8 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we integrate OpenTCM with this knowledge graph, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that OpenTCM achieves mean expert scores (MES) of 4.378 in ingredient information retrieval and 4.045 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07736v1">RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04873v2">Synergizing Unsupervised Episode Detection with LLMs for Large-Scale News Events</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 ACL 2025 Main Conference. Code available here: https://github.com/pkargupta/epimine
    </div>
    <details class="paper-abstract">
      State-of-the-art automatic event detection struggles with interpretability and adaptability to evolving large-scale key events -- unlike episodic structures, which excel in these areas. Often overlooked, episodes represent cohesive clusters of core entities performing actions at a specific time and location; a partially ordered sequence of episodes can represent a key event. This paper introduces a novel task, episode detection, which identifies episodes within a news corpus of key event articles. Detecting episodes poses unique challenges, as they lack explicit temporal or locational markers and cannot be merged using semantic similarity alone. While large language models (LLMs) can aid with these reasoning difficulties, they suffer with long contexts typical of news corpora. To address these challenges, we introduce EpiMine, an unsupervised framework that identifies a key event's candidate episodes by leveraging natural episodic partitions in articles, estimated through shifts in discriminative term combinations. These candidate episodes are more cohesive and representative of true episodes, synergizing with LLMs to better interpret and refine them into final episodes. We apply EpiMine to our three diverse, real-world event datasets annotated at the episode level, where it achieves a 59.2% average gain across all metrics compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18077v3">MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      How to efficiently serve LLMs in practice has become exceptionally challenging due to their prohibitive memory and computation requirements. In this study, we investigate optimizing the KV cache, whose memory footprint poses a critical bottleneck in LLM inference, especially when dealing with long context tasks. To tackle the challenge, we introduce MiniKV, a KV cache optimization method that simultaneously preserves long context task accuracy while significantly reducing KV cache size via a novel 2-bit layer-discriminative KV cache. More importantly, we develop specialized CUDA kernels to make MiniKV compatible with FlashAttention. Experiments on a wide range of long context tasks show that MiniKV effectively achieves 86% KV cache compression ratio while recovering over 98.5% of accuracy, outperforming state-of-the-art methods while achieving excellent measured system performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07282v1">Adultification Bias in LLMs and Text-to-Image Models</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Accepted to the ACM Conference on Fairness, Accountability, and Transparency (FAccT '25)
    </div>
    <details class="paper-abstract">
      The rapid adoption of generative AI models in domains such as education, policing, and social media raises significant concerns about potential bias and safety issues, particularly along protected attributes, such as race and gender, and when interacting with minors. Given the urgency of facilitating safe interactions with AI systems, we study bias along axes of race and gender in young girls. More specifically, we focus on "adultification bias," a phenomenon in which Black girls are presumed to be more defiant, sexually intimate, and culpable than their White peers. Advances in alignment techniques show promise towards mitigating biases but vary in their coverage and effectiveness across models and bias types. Therefore, we measure explicit and implicit adultification bias in widely used LLMs and text-to-image (T2I) models, such as OpenAI, Meta, and Stability AI models. We find that LLMs exhibit explicit and implicit adultification bias against Black girls, assigning them harsher, more sexualized consequences in comparison to their White peers. Additionally, we find that T2I models depict Black girls as older and wearing more revealing clothing than their White counterparts, illustrating how adultification bias persists across modalities. We make three key contributions: (1) we measure a new form of bias in generative AI models, (2) we systematically study adultification bias across modalities, and (3) our findings emphasize that current alignment methods are insufficient for comprehensively addressing bias. Therefore, new alignment methods that address biases such as adultification are needed to ensure safe and equitable AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07276v1">Tokenized Bandit for LLM Decoding and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 To appear at ICML 2025
    </div>
    <details class="paper-abstract">
      We introduce the tokenized linear bandit (TLB) and multi-armed bandit (TMAB), variants of linear and stochastic multi-armed bandit problems inspired by LLM decoding and alignment. In these problems, at each round $t \in [T]$, a user submits a query (context), and the decision maker (DM) sequentially selects a token irrevocably from a token set. Once the sequence is complete, the DM observes a random utility from the user, whose expectation is presented by a sequence function mapping the chosen token sequence to a nonnegative real value that depends on the query. In both problems, we first show that learning is impossible without any structure on the sequence function. We introduce a natural assumption, diminishing distance with more commons (DDMC), and propose algorithms with regret $\tilde{O}(L\sqrt{T})$ and $\tilde{O}(L\sqrt{T^{2/3}})$ for TLB and TMAB, respectively. As a side product, we obtain an (almost) optimality of the greedy decoding for LLM decoding algorithm under DDMC, which justifies the unresaonable effectiveness of greedy decoding in several tasks. This also has an immediate application to decoding-time LLM alignment, when the misaligned utility can be represented as the frozen LLM's utility and a linearly realizable latent function. We finally validate our algorithm's performance empirically as well as verify our assumptions using synthetic and real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04745v2">Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 13 pages, 23 figures. Accepted at XLLM @ ACL 2025
    </div>
    <details class="paper-abstract">
      This paper evaluates the ability of Large Language Models (LLMs) to leverage contextual information in the form of structured linguistic representations. Specifically, we examine the impact of encoding both short and long contexts using Abstract Meaning Representation (AMR) structures across a diverse set of language tasks. We perform our analysis using 8-bit quantized and instruction-tuned versions of Llama 3.1 (8B), Phi-3, and Mistral 7B. Our results indicate that, for tasks involving short contexts, augmenting the prompt with the AMR of the original language context often degrades the performance of the underlying LLM. However, for tasks that involve long contexts, such as dialogue summarization in the SAMSum dataset, this enhancement improves LLM performance, for example, by increasing the zero-shot cosine similarity score of Llama 3.1 from 66.2% to 76%. This improvement is more evident in the newer and larger LLMs, but does not extend to the older or smaller ones. In addition, we observe that LLMs can effectively reconstruct the original text from a linearized AMR, achieving a cosine similarity of 81.3% in the best-case scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07274v1">Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Code-switching presents a complex challenge for syntactic analysis, especially in low-resource language settings where annotated data is scarce. While recent work has explored the use of large language models (LLMs) for sequence-level tagging, few approaches systematically investigate how well these models capture syntactic structure in code-switched contexts. Moreover, existing parsers trained on monolingual treebanks often fail to generalize to multilingual and mixed-language input. To address this gap, we introduce the BiLingua Parser, an LLM-based annotation pipeline designed to produce Universal Dependencies (UD) annotations for code-switched text. First, we develop a prompt-based framework for Spanish-English and Spanish-Guaran\'i data, combining few-shot LLM prompting with expert review. Second, we release two annotated datasets, including the first Spanish-Guaran\'i UD-parsed corpus. Third, we conduct a detailed syntactic analysis of switch points across language pairs and communicative contexts. Experimental results show that BiLingua Parser achieves up to 95.29% LAS after expert revision, significantly outperforming prior baselines and multilingual parsers. These results show that LLMs, when carefully guided, can serve as practical tools for bootstrapping syntactic resources in under-resourced, code-switched environments. Data and source code are available at https://github.com/N3mika/ParsingProject
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07270v1">Question Answering under Temporal Conflict: Evaluating and Organizing Evolving Knowledge with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable capabilities in question answering and reasoning thanks to their extensive parametric memory. However, their knowledge is inherently limited by the scope of their pre-training data, while real-world information evolves continuously. Updating this knowledge typically requires costly and brittle re-training, or in-context learning (ICL), which becomes impractical at scale given the volume and volatility of modern information. Motivated by these limitations, we investigate how LLMs perform when exposed to temporal text corpora, or documents that reflect evolving knowledge over time, such as sports biographies where facts like a player's "current team" change year by year. To this end, we introduce two new benchmarks: Temporal Wiki, which captures factual drift across historical Wikipedia snapshots, and Unified Clark, which aggregates timestamped news articles to simulate real-world information accumulation. Our analysis reveals that LLMs often struggle to reconcile conflicting or outdated facts and can be misled when multiple versions of a fact appear in context. To address these issues, we propose a lightweight, agentic framework that incrementally builds a structured, external memory from source documents without requiring re-training. This knowledge organization strategy enables models to retrieve and reason over temporally filtered, relevant information at inference time. Empirically, our method outperforms ICL and RAG baselines across both benchmarks, especially on questions requiring more complex reasoning or integration of conflicting facts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15104v2">PecSched: Preemptive and Efficient Cluster Scheduling for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      The scaling of transformer-based Large Language Models (LLMs) has significantly expanded their context lengths, enabling applications where inputs exceed 100K tokens. Our analysis of a recent Azure LLM inference trace reveals a highly skewed long-tail distribution of input lengths, with approximately 80% of inputs shorter than 2K tokens. Long inputs constitute only a small fraction. Existing cluster-level LLM scheduling strategies, including First-In-First-Out (FIFO), reservation-based, and priority-based approaches, primarily target short-input requests with lengths below 2K and fail to address this heterogeneity, leading to inefficiencies such as head-of-line blocking, resource underutilization, and starvation of long-input requests. We propose PecSched, a Preemptive and Efficient Cluster SCHEDuling system for LLM inference. PecSched introduces the following key techniques: 1) preemptive scheduling that prioritizes short-input requests for their performance; 2) coordinated prefill-decode colocation and disaggregation, which reduces both the duration and frequency of preemptions; 3) fast Sequence Parallelism (SP) that minimizes the prefill time of long-input requests to further reduce the likelihood and frequency of preemptions. Evaluations based on Azure LLM inference trace show that, compared to state-of-the-art cluster-level LLM inference schedulers, PecSched reduces the 99th percentile queueing delay of short-input requests by up to 92% and improves their throughput by up to 595%, without significantly affecting the Job Completion Time (JCT) of long-input requests. We open-sourced our code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05561v2">Can the Rookies Cut the Tough Cookie? Exploring the Use of LLMs for SQL Equivalence Checking</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Equivalence checking of SQL queries is an intractable problem often encountered in settings ranging from grading SQL submissions to debugging query optimizers. Despite recent work toward developing practical solutions, only simple queries written using a small subset of SQL are supported, leaving the equivalence checking of sophisticated SQL queries at the mercy of intensive, potentially error-prone, manual analysis. In this paper, we explore how LLMs can be used to reason with SQL queries to address this challenging problem. Towards this, we introduce a novel, realistic, and sufficiently complex benchmark called SQLEquiQuest for SQL query equivalence checking that reflects real-world settings. We establish strong baselines for SQL equivalence checking by leveraging the ability of LLMs to reason with SQL queries. We conduct a detailed evaluation of several state-of-the-art LLMs using various prompting strategies and carefully constructed in-context learning examples, including logical plans generated by SQL query processors. Our empirical evaluation shows that LLMs go well beyond the current capabilities of formal models for SQL equivalence, going from a mere 30% supported query pairs to full coverage, achieving up to 82% accuracy on Spider+DIN. However, a critical limitation of LLMs revealed by our analysis is that they exhibit a strong bias for equivalence predictions, with consistently poor performance over non-equivalent pairs, opening a new direction for potential future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04951v2">Unsafe LLM-Based Search: Quantitative Analysis and Mitigation of Safety Risks in AI Web Search</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have significantly enhanced the capabilities of AI-Powered Search Engines (AIPSEs), offering precise and efficient responses by integrating external databases with pre-existing knowledge. However, we observe that these AIPSEs raise risks such as quoting malicious content or citing malicious websites, leading to harmful or unverified information dissemination. In this study, we conduct the first safety risk quantification on seven production AIPSEs by systematically defining the threat model, risk type, and evaluating responses to various query types. With data collected from PhishTank, ThreatBook, and LevelBlue, our findings reveal that AIPSEs frequently generate harmful content that contains malicious URLs even with benign queries (e.g., with benign keywords). We also observe that directly querying a URL will increase the number of main risk-inclusive responses, while querying with natural language will slightly mitigate such risk. Compared to traditional search engines, AIPSEs outperform in both utility and safety. We further perform two case studies on online document spoofing and phishing to show the ease of deceiving AIPSEs in the real-world setting. To mitigate these risks, we develop an agent-based defense with a GPT-4.1-based content refinement tool and a URL detector. Our evaluation shows that our defense can effectively reduce the risk, with only a minor cost of reducing available information by approximately 10.7%. Our research highlights the urgent need for robust safety measures in AIPSEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07240v1">Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Recently, techniques such as explicit structured reasoning have demonstrated strong test-time scaling behavior by enforcing a separation between the model's internal "thinking" process and the final response. A key factor influencing answer quality in this setting is the length of the thinking stage. When the reasoning is too short, the model may fail to capture the complexity of the task. Conversely, when it is too long, the model may overthink, leading to unnecessary computation and degraded performance. This paper explores and exploits the underlying mechanisms by which LLMs understand and regulate the length of their reasoning during explicit thought processes. First, we show that LLMs encode their progress through the reasoning process and introduce an interactive progress bar visualization, which is then used to reveal insights on the model's planning dynamics. Second, we manipulate the internal progress encoding during inference to reduce unnecessary steps and generate a more concise and decisive chain of thoughts. Our empirical results demonstrate that this "overclocking" method mitigates overthinking, improves answer accuracy, and reduces inference latency. Our code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07232v1">Learn as Individuals, Evolve as a Team: Multi-agent LLMs Adaptation in Embodied Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) possess extensive knowledge bases and strong reasoning capabilities, making them promising tools for complex, multi-agent planning in embodied environments. However, despite LLMs' advanced abilities and the sophisticated modular design of agentic methods, existing LLM-based planning algorithms remain limited by weak adaptation capabilities to multi-agent embodied scenarios. We address this limitation by introducing a framework that enables LLM agents to learn and evolve both before and during test time, equipping them with environment-relevant knowledge for better planning and enhanced communication for improved cooperation. Inspired by centralized training with decentralized execution in multi-agent reinforcement learning, we propose a \textit{Learn as Individuals, Evolve as a Team (LIET)} paradigm for multi-agent LLMs adaptation. At the individual level, LLM agents learn a local utility function from exploratory datasets to better comprehend the embodied environment, which is then queried during test time to support informed decision-making. At the team level, LLM agents collaboratively and iteratively maintain and update a shared cooperation knowledge list based on new experiences, using it to guide more effective communication. By combining individual learning with team evolution, LIET enables comprehensive and flexible adaptation for LLM agents. Our experiments on Communicative Watch-And-Help and ThreeD-World Multi-Agent Transport benchmarks demonstrate that LIET, instantiated with both LLaMA and GPT-4o, outperforms existing baselines and exhibits strong cooperative planning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07223v1">LLM-Enhanced Rapid-Reflex Async-Reflect Embodied Agent for Real-Time Decision-Making in Dynamically Changing Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Accepted by the CVPR 2025 Embodied AI Workshop
    </div>
    <details class="paper-abstract">
      In the realm of embodied intelligence, the evolution of large language models (LLMs) has markedly enhanced agent decision making. Consequently, researchers have begun exploring agent performance in dynamically changing high-risk scenarios, i.e., fire, flood, and wind scenarios in the HAZARD benchmark. Under these extreme conditions, the delay in decision making emerges as a crucial yet insufficiently studied issue. We propose a Time Conversion Mechanism (TCM) that translates inference delays in decision-making into equivalent simulation frames, thus aligning cognitive and physical costs under a single FPS-based metric. By extending HAZARD with Respond Latency (RL) and Latency-to-Action Ratio (LAR), we deliver a fully latency-aware evaluation protocol. Moreover, we present the Rapid-Reflex Async-Reflect Agent (RRARA), which couples a lightweight LLM-guided feedback module with a rule-based agent to enable immediate reactive behaviors and asynchronous reflective refinements in situ. Experiments on HAZARD show that RRARA substantially outperforms existing baselines in latency-sensitive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07211v1">Sword and Shield: Uses and Strategies of LLMs in Navigating Disinformation</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) presents a dual challenge in the fight against disinformation. These powerful tools, capable of generating human-like text at scale, can be weaponised to produce sophisticated and persuasive disinformation, yet they also hold promise for enhancing detection and mitigation strategies. This paper investigates the complex dynamics between LLMs and disinformation through a communication game that simulates online forums, inspired by the game Werewolf, with 25 participants. We analyse how Disinformers, Moderators, and Users leverage LLMs to advance their goals, revealing both the potential for misuse and combating disinformation. Our findings highlight the varying uses of LLMs depending on the participants' roles and strategies, underscoring the importance of understanding their effectiveness in this context. We conclude by discussing implications for future LLM development and online platform design, advocating for a balanced approach that empowers users and fosters trust while mitigating the risks of LLM-assisted disinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04793v2">Sentence-level Reward Model can Generalize Better for Aligning LLM from Human Preference</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Learning reward models from human preference datasets and subsequently optimizing language models via reinforcement learning has emerged as a fundamental paradigm for aligning LLMs with human preferences. The performance of the reward model plays a crucial role in the effectiveness of alignment. Previous reward models operate at a coarse-grained level, requiring the generation of a complete response to obtain a reward value. The sparse reward may present challenges for downstream reinforcement learning. While recent efforts have attempted to learn token-level reward models, the lack of explicit semantic information makes it difficult to model the credit of every individual token. In this paper, we propose assigning scores to every sentence, introducing an intermediate-grained reward model. By segmenting the complete response into sentences and applying differential operations to reward output at the start and end positions of each sentence, we can effectively model the rewards of sentences. Moreover, a novel attention mechanism is introduced to aggregate the scores of all sentences into a response-level score, which allows it to be trained using the Bradley-Terry model. On common benchmarks, our method outperforms the response-level reward model by 2.7% on RewardBench (for reward modeling evaluation) and surpasses all baselines on AlpacaEval (for alignment evaluation).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07180v1">Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first dedicated benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the visual domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. In addition, we explore key-frame selection as an interpretable, training-free mitigation strategy, which reveals potential paths for reducing sycophantic bias by strengthening visual grounding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14996v2">Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Findings of the Association for Computational Linguistics ACL 2025
    </div>
    <details class="paper-abstract">
      One of the most widely used tasks for evaluating Large Language Models (LLMs) is Multiple-Choice Question Answering (MCQA). While open-ended question answering tasks are more challenging to evaluate, MCQA tasks are, in principle, easier to assess, as the model's answer is thought to be simple to extract and is compared directly to a set of predefined choices. However, recent studies have started to question the reliability of MCQA evaluation, showing that multiple factors can significantly impact the reported performance of LLMs, especially when the model generates free-form text before selecting one of the answer choices. In this work, we shed light on the inconsistencies of MCQA evaluation strategies, which can lead to inaccurate and misleading model comparisons. We systematically analyze whether existing answer extraction methods are aligned with human judgment, and how they are influenced by answer constraints in the prompt across different domains. Our experiments demonstrate that traditional evaluation strategies often underestimate LLM capabilities, while LLM-based answer extractors are prone to systematic errors. Moreover, we reveal a fundamental trade-off between including format constraints in the prompt to simplify answer extraction and allowing models to generate free-form text to improve reasoning. Our findings call for standardized evaluation methodologies and highlight the need for more reliable and consistent MCQA evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04790v2">PECAN: LLM-Guided Dynamic Progress Control with Attention-Guided Hierarchical Weighted Graph for Long-Document QA</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Long-document QA presents challenges with large-scale text and long-distance dependencies. Recent advances in Large Language Models (LLMs) enable entire documents to be processed in a single pass. However, their computational cost is significantly high. Retrieval-Augmented Generation (RAG) methods split text into smaller chunks, but they often yield inferior results and may lose global context. Recent approaches that integrate LLMs into RAG via iterative summarization either underutilize LLM capabilities or still incur high computational costs. In this paper, we combine the high accuracy of LLMs with the efficiency of RAG and propose LLM-Guided Dynamic Progress Control with Attention-Based Hierarchical Weighted Graph (PECAN). Our method introduces two key improvements: (1) LLM-Guided Dynamic Progress Control: We leverage LLMs to dynamically control the retrieval process, adjusting the amount of retrieved information based on different queries to achieve a better balance of effectiveness and efficiency. (2) Attention-Guided Retrieval: We propose a novel retrieval method that constructs a hierarchical graph where edges are derived by LLM attention weights. Experimental results demonstrate that PECAN achieves LLM-level performance while maintaining computational complexity comparable to that of RAG methods on two single-document and two multi-document QA datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07160v1">GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, particularly in mathematical reasoning, amid which geometry problem solving remains a challenging area where auxiliary construction plays a enssential role. Existing approaches either achieve suboptimal performance or rely on massive LLMs (e.g., GPT-4o), incurring massive computational costs. We posit that reinforcement learning with verifiable reward (e.g., GRPO) offers a promising direction for training smaller models that effectively combine auxiliary construction with robust geometric reasoning. However, directly applying GRPO to geometric reasoning presents fundamental limitations due to its dependence on unconditional rewards, which leads to indiscriminate and counterproductive auxiliary constructions. To address these challenges, we propose Group Contrastive Policy Optimization (GCPO), a novel reinforcement learning framework featuring two key innovations: (1) Group Contrastive Masking, which adaptively provides positive or negative reward signals for auxiliary construction based on contextual utility, and a (2) length reward that promotes longer reasoning chains. Building on GCPO, we develop GeometryZero, a family of affordable-size geometric reasoning models that judiciously determine when to employ auxiliary construction. Our extensive empirical evaluation across popular geometric benchmarks (Geometry3K, MathVista) demonstrates that GeometryZero models consistently outperform baselines (e.g. GRPO), achieving an average improvement of 4.29% across all benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07135v1">Taxonomy of migration scenarios for Qiskit refactoring using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Accepted for publication in ASQC JAIIO 54 (https://54jaiio.sadio.org.ar/simposios/)
    </div>
    <details class="paper-abstract">
      As quantum computing advances, quantum programming libraries' heterogeneity and steady evolution create new challenges for software developers. Frequent updates in software libraries break working code that needs to be refactored, thus adding complexity to an already complex landscape. These refactoring challenges are, in many cases, fundamentally different from those known in classical software engineering due to the nature of quantum computing software. This study addresses these challenges by developing a taxonomy of quantum circuit's refactoring problems, providing a structured framework to analyze and compare different refactoring approaches. Large Language Models (LLMs) have proven valuable tools for classic software development, yet their value in quantum software engineering remains unexplored. This study uses LLMs to categorize refactoring needs in migration scenarios between different Qiskit versions. Qiskit documentation and release notes were scrutinized to create an initial taxonomy of refactoring required for migrating between Qiskit releases. Two taxonomies were produced: one by expert developers and one by an LLM. These taxonomies were compared, analyzing differences and similarities, and were integrated into a unified taxonomy that reflects the findings of both methods. By systematically categorizing refactoring challenges in Qiskit, the unified taxonomy is a foundation for future research on AI-assisted migration while enabling a more rigorous evaluation of automated refactoring techniques. Additionally, this work contributes to quantum software engineering (QSE) by enhancing software development workflows, improving language compatibility, and promoting best practices in quantum programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08911v2">Test-driven Software Experimentation with LASSO: an LLM Prompt Benchmarking Example</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Empirical software engineering faces a critical gap: the lack of standardized tools for rapid development and execution of Test-Driven Software Experiments (TDSEs) -- that is, experiments that involve the execution of software subjects and the observation and analysis of their "de facto" run-time behavior. In this paper we present a general-purpose analysis platform called LASSO that provides a minimal set of domain-specific languages and data structures to conduct TDSEs. By empowering users with an executable scripting language to design and execute TDSEs, LASSO enables efficient evaluation of run-time semantics and execution characteristics in addition to statically determined properties. We present an example TDSE that demonstrates the practical benefits of LASSO's scripting capabilities for assessing the reliability of LLMs for code generation by means of a self-contained, reusable and extensible study script. The LASSO platform and live pipeline examples are publicly available at: https://softwareobservatorium.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16207v4">From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      The research in AI-based formal mathematical reasoning has shown an unstoppable growth trend. These studies have excelled in mathematical competitions like IMO and have made significant progress. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and breaks it down into sub-tasks. We constructed 18k high-quality instruction-response pairs across five formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) by distilling gpt-4o and evaluated against ten open-sourced LLMs, including recent popular DeepSeek-R1. We also fine-tuned several 7~8B small models to achieve comparable performance with Deepseek-R1-671B. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding capabilities. Fine-tuned models are released at https: //huggingface.co/fm-universe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08634v3">When Attention Collapses: How Degenerate Layers in LLMs Enable Smaller, Stronger Models</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 29 pages, 22 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) rely on the transformer architecture and its self-attention mechanism to deliver strong performance across tasks. However, we uncover a structural inefficiency in standard pre-trained decoder-style LLMs: in many of the deeper layers, attention matrices frequently collapse to near rank-one, single-column patterns. We refer to these underutilized components as lazy layers, which are redundant and computationally inefficient. To address this, we propose Inheritune, a simple and effective training recipe for building smaller, more efficient, and high performing language models. Inheritune initializes a compact model by inheriting the useful early layers from a larger pre-trained model, then progressively retrains and expands it. Our experiments across multiple models and datasets show that Inheritune trained models, despite having significantly fewer layers, can match or even outperform their larger counterparts. This approach yields compact, performant models and offers a practical path for efficient language model compression. Code is available at https://github.com/sanyalsunny111/LLM-Inheritune
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02943v3">Hallucination to Consensus: Multi-Agent LLMs for End-to-End Test Generation with Accurate Oracles</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is laborious, especially for strong typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to generate tests that achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in large language models (LLMs) have enabled oracle generation from natural language descriptions. However, existing LLM-based methods often require LLM fine-tuning or rely on external tools such as EvoSuite for test prefix generation. In this work, we propose CANDOR, a novel end-to-end, prompt-based LLM framework for automated JUnit test generation. CANDOR orchestrates multiple specialized LLM agents to generate JUnit tests, including both high-quality test prefixes and accurate oracles. To mitigate the notorious hallucinations in LLMs, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generate accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments on the HumanEvalJava and LeetCodeJava datasets show that CANDOR can generate accurate oracles and is slightly better than EvoSuite in generating tests with high line coverage and clearly superior in terms of mutation score. Moreover, CANDOR significantly outperforms the state-of-the-art, prompt-based test generator LLM-Empirical, achieving improvements of 15.8 to 25.1 percentage points in oracle correctness on both correct and faulty source code. Ablation studies confirm the critical contributions of key agents in improving test prefix quality and oracle accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22942v2">WorkForceAgent-R1: Incentivizing Reasoning Capability in LLM-based Web Agents via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs)-empowered web agents enables automating complex, real-time web navigation tasks in enterprise environments. However, existing web agents relying on supervised fine-tuning (SFT) often struggle with generalization and robustness due to insufficient reasoning capabilities when handling the inherently dynamic nature of web interactions. In this study, we introduce WorkForceAgent-R1, an LLM-based web agent trained using a rule-based R1-style reinforcement learning framework designed explicitly to enhance single-step reasoning and planning for business-oriented web navigation tasks. We employ a structured reward function that evaluates both adherence to output formats and correctness of actions, enabling WorkForceAgent-R1 to implicitly learn robust intermediate reasoning without explicit annotations or extensive expert demonstrations. Extensive experiments on the WorkArena benchmark demonstrate that WorkForceAgent-R1 substantially outperforms SFT baselines by 10.26-16.59%, achieving competitive performance relative to proprietary LLM-based agents (gpt-4o) in workplace-oriented web navigation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13287v3">An Online Learning Approach to Prompt-based Selection of Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Selecting a sample generation scheme from multiple prompt-based generative models, including large language models (LLMs) and prompt-guided image and video generation models, is typically addressed by choosing the model that maximizes an averaged evaluation score. However, this score-based selection overlooks the possibility that different models achieve the best generation performance for different types of text prompts. An online identification of the best generation model for various input prompts can reduce the costs associated with querying sub-optimal models. In this work, we explore the possibility of varying rankings of text-based generative models for different text prompts and propose an online learning framework to predict the best data generation model for a given input prompt. The proposed PAK-UCB algorithm addresses a contextual bandit (CB) setting with shared context variables across the arms, utilizing the generated data to update kernel-based functions that predict the score of each model available for unseen text prompts. Additionally, we leverage random Fourier features (RFF) to accelerate the online learning process of PAK-UCB. Our numerical experiments on real and simulated text-to-image and image-to-text generative models show that RFF-UCB performs successfully in identifying the best generation model across different sample types. The code is available at: github.com/yannxiaoyanhu/dgm-online-select.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19487v2">Evolution of Cooperation in LLM-Agent Societies: A Preliminary Study Using Different Punishment Strategies</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 19 pages, 10 figures, Accepted for presentation as a full paper at the COINE 2025 workshop at AAMAS 2025 (https://coin-workshop.github.io/coine-2025-detroit/accepted_for_presentation.html)
    </div>
    <details class="paper-abstract">
      The evolution of cooperation has been extensively studied using abstract mathematical models and simulations. Recent advances in Large Language Models (LLM) and the rise of LLM agents have demonstrated their ability to perform social reasoning, thus providing an opportunity to test the emergence of norms in more realistic agent-based simulations with human-like reasoning using natural language. In this research, we investigate whether the cooperation dynamics presented in Boyd and Richerson's model persist in a more realistic simulation of the diner's dilemma using LLM agents compared to the abstract mathematical nature in the work of Boyd and Richerson. Our findings indicate that agents follow the strategies defined in the Boyd and Richerson model, and explicit punishment mechanisms drive norm emergence, reinforcing cooperative behaviour even when the agent strategy configuration varies. Our results suggest that LLM-based Multi-Agent System simulations, in fact, can replicate the evolution of cooperation predicted by the traditional mathematical models. Moreover, our simulations extend beyond the mathematical models by integrating natural language-driven reasoning and a pairwise imitation method for strategy adoption, making them a more realistic testbed for cooperative behaviour in MASs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15068v3">LLM-HDR: Bridging LLM-based Perception and Self-Supervision for Unpaired LDR-to-HDR Image Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      The translation of Low Dynamic Range (LDR) to High Dynamic Range (HDR) images is an important computer vision task. There is a significant amount of research utilizing both conventional non-learning methods and modern data-driven approaches, focusing on using both single-exposed and multi-exposed LDR for HDR image reconstruction. However, most current state-of-the-art methods require high-quality paired {LDR,HDR} datasets for model training. In addition, there is limited literature on using unpaired datasets for this task, that is, the model learns a mapping between domains, i.e., {LDR,HDR}. This paper proposes LLM-HDR, a method that integrates the perception of Large Language Models (LLM) into a modified semantic- and cycle-consistent adversarial architecture that utilizes unpaired {LDR,HDR} datasets for training. The method introduces novel artifact- and exposure-aware generators to address visual artifact removal and an encoder and loss to address semantic consistency, another under-explored topic. LLM-HDR is the first to use an LLM for the {LDR,HDR} translation task in a self-supervised setup. The method achieves state-of-the-art performance across several benchmark datasets and reconstructs high-quality HDR images. The official website of this work is available at: https://github.com/HrishavBakulBarua/LLM-HDR
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17663v2">Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 Accepted by ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly participate in human-AI interactions, evaluating their Theory of Mind (ToM) capabilities - particularly their ability to track dynamic mental states - becomes crucial. While existing benchmarks assess basic ToM abilities, they predominantly focus on static snapshots of mental states, overlooking the temporal evolution that characterizes real-world social interactions. We present \textsc{DynToM}, a novel benchmark specifically designed to evaluate LLMs' ability to understand and track the temporal progression of mental states across interconnected scenarios. Through a systematic four-step framework, we generate 1,100 social contexts encompassing 5,500 scenarios and 78,100 questions, each validated for realism and quality. Our comprehensive evaluation of ten state-of-the-art LLMs reveals that their average performance underperforms humans by 44.7\%, with performance degrading significantly when tracking and reasoning about the shift of mental states. This performance gap highlights fundamental limitations in current LLMs' ability to model the dynamic nature of human mental states.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06991v1">Evaluating LLM-corrupted Crowdsourcing Data Without Ground Truth</a></div>
    <div class="paper-meta">
      📅 2025-06-08
      | 💬 33 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The recent success of generative AI highlights the crucial role of high-quality human feedback in building trustworthy AI systems. However, the increasing use of large language models (LLMs) by crowdsourcing workers poses a significant challenge: datasets intended to reflect human input may be compromised by LLM-generated responses. Existing LLM detection approaches often rely on high-dimension training data such as text, making them unsuitable for annotation tasks like multiple-choice labeling. In this work, we investigate the potential of peer prediction -- a mechanism that evaluates the information within workers' responses without using ground truth -- to mitigate LLM-assisted cheating in crowdsourcing with a focus on annotation tasks. Our approach quantifies the correlations between worker answers while conditioning on (a subset of) LLM-generated labels available to the requester. Building on prior research, we propose a training-free scoring mechanism with theoretical guarantees under a crowdsourcing model that accounts for LLM collusion. We establish conditions under which our method is effective and empirically demonstrate its robustness in detecting low-effort cheating on real-world crowdsourcing datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06975v1">Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06971v1">Break-The-Chain: Reasoning Failures in LLMs via Adversarial Prompting in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in tasks requiring complex reasoning, such as code generation, mathematical problem solving, and algorithmic synthesis -- especially when aided by reasoning tokens and Chain-of-Thought prompting. Yet, a core question remains: do these models truly reason, or do they merely exploit shallow statistical patterns? In this paper, we systematically investigate the robustness of reasoning LLMs by introducing a suite of semantically faithful yet adversarially structured prompt perturbations. Our evaluation -- spanning 700 perturbed code generations derived from LeetCode-style problems -- applies transformations such as storytelling reframing, irrelevant constraint injection, example reordering, and numeric perturbation. We observe that while certain modifications severely degrade performance (with accuracy drops up to -42.1%), others surprisingly improve model accuracy by up to 35.3%, suggesting sensitivity not only to semantics but also to surface-level prompt dynamics. These findings expose the fragility and unpredictability of current reasoning systems, underscoring the need for more principles approaches to reasoning alignments and prompting robustness. We release our perturbation datasets and evaluation framework to promote further research in trustworthy and resilient LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04178v4">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Accepted by SIGIR2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), known for their comprehension capabilities and extensive knowledge, have been increasingly applied to recommendation systems (RS). Given the fundamental gap between the mechanism of LLMs and the requirement of RS, researchers have focused on fine-tuning LLMs with recommendation-specific data to enhance their performance. Language Modeling Loss (LML), originally designed for language generation tasks, is commonly adopted. However, we identify two critical limitations of LML: 1) it exhibits significant divergence from the recommendation objective; 2) it erroneously treats all fictitious item descriptions as negative samples, introducing misleading training signals. To address these limitations, we propose a novel Masked Softmax Loss (MSL) tailored for fine-tuning LLMs on recommendation. MSL improves LML by identifying and masking invalid tokens that could lead to fictitious item descriptions during loss computation. This strategy can effectively avoid the interference from erroneous negative signals and ensure well alignment with the recommendation objective supported by theoretical guarantees. During implementation, we identify a potential challenge related to gradient vanishing of MSL. To overcome this, we further introduce the temperature coefficient and propose an Adaptive Temperature Strategy (ATS) that adaptively adjusts the temperature without requiring extensive hyperparameter tuning. Extensive experiments conducted on four public datasets further validate the effectiveness of MSL, achieving an average improvement of 42.24% in NDCG@10. The code is available at https://github.com/WANGBohaO-jpg/MSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06632v1">Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11716v2">LLMs Can Simulate Standardized Patients via Agent Coevolution</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Training medical personnel using standardized patients (SPs) remains a complex challenge, requiring extensive domain expertise and role-specific practice. Previous research on Large Language Model (LLM)-based SPs mostly focuses on improving data retrieval accuracy or adjusting prompts through human feedback. However, this focus has overlooked the critical need for patient agents to learn a standardized presentation pattern that transforms data into human-like patient responses through unsupervised simulations. To address this gap, we propose EvoPatient, a novel simulated patient framework in which a patient agent and doctor agents simulate the diagnostic process through multi-turn dialogues, simultaneously gathering experience to improve the quality of both questions and answers, ultimately enabling human doctor training. Extensive experiments on various cases demonstrate that, by providing only overall SP requirements, our framework improves over existing reasoning methods by more than 10\% in requirement alignment and better human preference, while achieving an optimal balance of resource consumption after evolving over 200 cases for 10 hours, with excellent generalizability. Our system will be available at https://github.com/ZJUMAI/EvoPatient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03296v2">Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU execution has emerged as a promising solution by offloading KV cache management and parts of attention computation to the CPU. However, a key bottleneck remains: existing schedulers fail to effectively overlap CPU-offloaded tasks with GPU execution during the latency-critical, bandwidth-bound decode phase. This particularly penalizes real-time, decode-heavy applications (e.g., chat, Chain-of-Thought reasoning) which are currently underserved by existing systems, especially under memory pressure typical of edge or low-cost deployments. We present APEX, a novel, profiling-informed scheduling strategy that maximizes CPU-GPU parallelism during hybrid LLM inference. Unlike systems relying on static rules or purely heuristic approaches, APEX dynamically dispatches compute across heterogeneous resources by predicting execution times of CPU and GPU subtasks to maximize overlap while avoiding scheduling overheads. We evaluate APEX on diverse workloads and GPU architectures (NVIDIA T4, A10), using LLaMa-2-7B and LLaMa-3.1-8B models. Compared to GPU-only schedulers like VLLM, APEX improves throughput by 84% - 96% on T4 and 11% - 89% on A10 GPUs, while preserving latency. Against the best existing hybrid schedulers, it delivers up to 49% (T4) and 37% (A10) higher throughput in long-output settings. APEX significantly advances hybrid LLM inference efficiency on such memory-constrained hardware and provides a blueprint for scheduling in heterogeneous AI systems, filling a critical gap for efficient real-time LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06616v1">Interpretable Depression Detection from Social Media Text Using LLM-Derived Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Submitted to the IEEE EMBS BHI 2025 Conference
    </div>
    <details class="paper-abstract">
      Accurate and interpretable detection of depressive language in social media is useful for early interventions of mental health conditions, and has important implications for both clinical practice and broader public health efforts. In this paper, we investigate the performance of large language models (LLMs) and traditional machine learning classifiers across three classification tasks involving social media data: binary depression classification, depression severity classification, and differential diagnosis classification among depression, PTSD, and anxiety. Our study compares zero-shot LLMs with supervised classifiers trained on both conventional text embeddings and LLM-generated summary embeddings. Our experiments reveal that while zero-shot LLMs demonstrate strong generalization capabilities in binary classification, they struggle with fine-grained ordinal classifications. In contrast, classifiers trained on summary embeddings generated by LLMs demonstrate competitive, and in some cases superior, performance on the classification tasks, particularly when compared to models using traditional text embeddings. Our findings demonstrate the strengths of LLMs in mental health prediction, and suggest promising directions for better utilization of their zero-shot capabilities and context-aware summarization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11221v2">PlanGenLLMs: A Modern Survey of LLM Planning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Accepted by ACL 2025
    </div>
    <details class="paper-abstract">
      LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06943v1">WiFi Pathologies Detection using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      In this paper, we fine-tune encoder-only and decoder-only large language models (LLMs) to detect pathologies in IEEE 802.11 networks, commonly known as WiFi. Our approach involves manually crafting prompts followed by fine-tuning. Evaluations show that the sequential model achieves high detection accuracy using labeled data, while the causal model performs equally well for unlabeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06928v1">How Important are Videos for Training Video LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Project page on https://visualcomputinginstitute.github.io/videollm-pseudovideo-training/
    </div>
    <details class="paper-abstract">
      Research into Video Large Language Models (LLMs) has progressed rapidly, with numerous models and benchmarks emerging in just a few years. Typically, these models are initialized with a pretrained text-only LLM and finetuned on both image- and video-caption datasets. In this paper, we present findings indicating that Video LLMs are more capable of temporal reasoning after image-only training than one would assume, and that improvements from video-specific training are surprisingly small. Specifically, we show that image-trained versions of two LLMs trained with the recent LongVU algorithm perform significantly above chance level on TVBench, a temporal reasoning benchmark. Additionally, we introduce a simple finetuning scheme involving sequences of annotated images and questions targeting temporal capabilities. This baseline results in temporal reasoning performance close to, and occasionally higher than, what is achieved by video-trained LLMs. This suggests suboptimal utilization of rich temporal features found in real video by current models. Our analysis motivates further research into the mechanisms that allow image-trained LLMs to perform temporal reasoning, as well as into the bottlenecks that render current video training schemes inefficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06923v1">Boosting LLM Reasoning via Spontaneous Self-Correction</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated remarkable success on a broad range of tasks, math reasoning remains a challenging one. One of the approaches for improving math reasoning is self-correction, which designs self-improving loops to let the model correct its own mistakes. However, existing self-correction approaches treat corrections as standalone post-generation refinements, relying on extra prompt and system designs to elicit self-corrections, instead of performing real-time, spontaneous self-corrections in a single pass. To address this, we propose SPOC, a spontaneous self-correction approach that enables LLMs to generate interleaved solutions and verifications in a single inference pass, with generation dynamically terminated based on verification outcomes, thereby effectively scaling inference time compute. SPOC considers a multi-agent perspective by assigning dual roles -- solution proposer and verifier -- to the same model. We adopt a simple yet effective approach to generate synthetic data for fine-tuning, enabling the model to develop capabilities for self-verification and multi-agent collaboration. We further improve its solution proposal and verification accuracy through online reinforcement learning. Experiments on mathematical reasoning benchmarks show that SPOC significantly improves performance. Notably, SPOC boosts the accuracy of Llama-3.1-8B and 70B Instruct models, achieving gains of 8.8% and 11.6% on MATH500, 10.0% and 20.0% on AMC23, and 3.3% and 6.7% on AIME24, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18825v2">EconEvals: Benchmarks and Litmus Tests for LLM Agents in Unknown Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      We develop benchmarks for LLM agents that act in, learn from, and strategize in unknown environments, the specifications of which the LLM agent must learn over time from deliberate exploration. Our benchmarks consist of decision-making tasks derived from key problems in economics. To forestall saturation, the benchmark tasks are synthetically generated with scalable difficulty levels. Additionally, we propose litmus tests, a new kind of quantitative measure for LLMs and LLM agents. Unlike benchmarks, litmus tests quantify differences in character, values, and tendencies of LLMs and LLM agents, by considering their behavior when faced with tradeoffs (e.g., efficiency versus equality) where there is no objectively right or wrong behavior. Overall, our benchmarks and litmus tests assess the abilities and tendencies of LLM agents in tackling complex economic problems in diverse settings spanning procurement, scheduling, task allocation, and pricing -- applications that should grow in importance as such agents are further integrated into the economy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06511v3">TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) has been instrumental in advancing state-of-the-art natural language processing applications. Training LLMs with billions of parameters and trillions of tokens require sophisticated distributed systems that enable composing and comparing several state-of-the-art techniques in order to efficiently scale across thousands of accelerators. However, existing solutions are complex, scattered across multiple libraries/repositories, lack interoperability, and are cumbersome to maintain. Thus, curating and empirically comparing training recipes require non-trivial engineering effort. This paper introduces TorchTitan, an open-source, PyTorch-native distributed training system that unifies state-of-the-art techniques, streamlining integration and reducing overhead. TorchTitan enables 3D parallelism in a modular manner with elastic scaling, providing comprehensive logging, checkpointing, and debugging tools for production-ready training. It also incorporates hardware-software co-designed solutions, leveraging features like Float8 training and SymmetricMemory. As a flexible test bed, TorchTitan facilitates custom recipe curation and comparison, allowing us to develop optimized training recipes for Llama 3.1 and provide guidance on selecting techniques for maximum efficiency based on our experiences. We thoroughly assess TorchTitan on the Llama 3.1 family of LLMs, spanning 8 billion to 405 billion parameters, and showcase its exceptional performance, modular composability, and elastic scalability. By stacking training optimizations, we demonstrate accelerations of 65.08% with 1D parallelism at the 128-GPU scale (Llama 3.1 8B), an additional 12.59% with 2D parallelism at the 256-GPU scale (Llama 3.1 70B), and an additional 30% with 3D parallelism at the 512-GPU scale (Llama 3.1 405B) on NVIDIA H100 GPUs over optimized baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06877v1">Right Is Not Enough: The Pitfalls of Outcome Supervision in Training LLMs for Math Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Outcome-rewarded Large Language Models (LLMs) have demonstrated remarkable success in mathematical problem-solving. However, this success often masks a critical issue: models frequently achieve correct answers through fundamentally unsound reasoning processes, a phenomenon indicative of reward hacking. We introduce MathOlympiadEval, a new dataset with fine-grained annotations, which reveals a significant gap between LLMs' answer correctness and their low process correctness. Existing automated methods like LLM-as-a-judge struggle to reliably detect these reasoning flaws. To address this, we propose ParaStepVerifier, a novel methodology for meticulous, step-by-step verification of mathematical solutions. ParaStepVerifier identifies incorrect reasoning steps. Empirical results demonstrate that ParaStepVerifier substantially improves the accuracy of identifying flawed solutions compared to baselines, especially for complex, multi-step problems. This offers a more robust path towards evaluating and training LLMs with genuine mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13425v3">Watermark under Fire: A Robustness Evaluation of LLM Watermarking</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16185v4">LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 This is a technical report by Nanyang Technological University. Updated to support Solidity, Java and C/C++
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in various tasks, including those requiring human-level intelligence, such as vulnerability detection. However, recent efforts to use LLMs for vulnerability detection remain preliminary, as they lack a deep understanding of whether a subject LLM's vulnerability reasoning capability stems from the model itself or from external aids such as knowledge retrieval and tooling support. In this paper, we aim to decouple LLMs' vulnerability reasoning from other capabilities, such as vulnerability knowledge adoption, context information retrieval, and advanced prompt schemes. We introduce LLM4Vuln, a unified evaluation framework that separates and assesses LLMs' vulnerability reasoning capabilities and examines improvements when combined with other enhancements. To support this evaluation, we construct UniVul, the first benchmark that provides retrievable knowledge and context-supplementable code across three representative programming languages: Solidity, Java, and C/C++. Using LLM4Vuln and UniVul, we test six representative LLMs (GPT-4.1, Phi-3, Llama-3, o4-mini, DeepSeek-R1, and QwQ-32B) for 147 ground-truth vulnerabilities and 147 non-vulnerable cases in 3,528 controlled scenarios. Our findings reveal the varying impacts of knowledge enhancement, context supplementation, and prompt schemes. We also identify 14 zero-day vulnerabilities in four pilot bug bounty programs, resulting in $3,576 in bounties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06843v1">United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit a notable performance ceiling on complex, multi-faceted tasks, as they often fail to integrate diverse information or adhere to multiple constraints. We posit that such limitation arises when the demands of a task exceed the LLM's effective cognitive load capacity. This interpretation draws a strong analogy to Cognitive Load Theory (CLT) in cognitive science, which explains similar performance boundaries in the human mind, and is further supported by emerging evidence that reveals LLMs have bounded working memory characteristics. Building upon this CLT-grounded understanding, we introduce CoThinker, a novel LLM-based multi-agent framework designed to mitigate cognitive overload and enhance collaborative problem-solving abilities. CoThinker operationalizes CLT principles by distributing intrinsic cognitive load through agent specialization and managing transactional load via structured communication and a collective working memory. We empirically validate CoThinker on complex problem-solving tasks and fabricated high cognitive load scenarios, demonstrating improvements over existing multi-agent baselines in solution quality and efficiency. Our analysis reveals characteristic interaction patterns, providing insights into the emergence of collective cognition and effective load management, thus offering a principled approach to overcoming LLM performance ceilings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17274v6">CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Accurate identification of software vulnerabilities is crucial for system integrity. Vulnerability datasets, often derived from the National Vulnerability Database (NVD) or directly from GitHub, are essential for training machine learning models to detect these security flaws. However, these datasets frequently suffer from significant noise, typically 40% to 75%, due primarily to the automatic and indiscriminate labeling of all changes in vulnerability-fixing commits (VFCs) as vulnerability-related. This misclassification occurs because not all changes in a commit aimed at fixing vulnerabilities pertain to security threats; many are routine updates like bug fixes or test improvements. This paper introduces the first methodology that uses the Large Language Model (LLM) with a heuristic enhancement to automatically identify vulnerability-fixing changes from VFCs, achieving an F1-score of 0.82. VulSifter was applied to a large-scale study, where we conducted a crawl of 127,063 repositories on GitHub, resulting in the acquisition of 5,352,105 commits. VulSifter involves utilizing an LLM to comprehend code semantics and contextual information, while applying heuristics to filter out unrelated changes. We then developed CleanVul, a high-quality dataset comprising 8,203 functions using our LLM heuristic enhancement approach, demonstrating Correctness (90.6%) comparable to established datasets such as SVEN and PrimeVul. To evaluate the CleanVul dataset, we conducted experiments focusing on fine-tuning various LLMs on CleanVul and other high-quality datasets. Evaluation results reveal that LLMs fine-tuned on CleanVul not only exhibit enhanced accuracy but also superior generalization capabilities compared to those trained on uncleaned datasets. Specifically, models trained on CleanVul and tested on PrimeVul achieve accuracy higher than those trained and tested exclusively on PrimeVul.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08351v2">Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Accepted at BEA2025 (Conference workshop at ACL 2025)
    </div>
    <details class="paper-abstract">
      This paper investigates the potentials of Large Language Models (LLMs) as adaptive tutors in the context of second-language learning. In particular, we evaluate whether system prompting can reliably constrain LLMs to generate only text appropriate to the student's competence level. We simulate full teacher-student dialogues in Spanish using instruction-tuned, open-source LLMs ranging in size from 7B to 12B parameters. Dialogues are generated by having an LLM alternate between tutor and student roles with separate chat histories. The output from the tutor model is then used to evaluate the effectiveness of CEFR-based prompting to control text difficulty across three proficiency levels (A1, B1, C1). Our findings suggest that while system prompting can be used to constrain model outputs, prompting alone is too brittle for sustained, long-term interactional contexts - a phenomenon we term alignment drift. Our results provide insights into the feasibility of LLMs for personalized, proficiency-aligned adaptive tutors and provide a scalable method for low-cost evaluation of model performance without human participants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06821v1">Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 37 pages, 22 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17648v2">Simulating Macroeconomic Expectations using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for simulating macroeconomic expectation formation using Large Language Model-Empowered Agents (LLM Agents). By constructing thousands of LLM Agents equipped with modules for personal characteristics, prior expectations, and knowledge, we replicate a survey experiment involving households and experts on inflation and unemployment. Our results show that although the expectations and thoughts generated by LLM Agents are more homogeneous than those of human participants, they still effectively capture key heterogeneity across agents and the underlying drivers of expectation formation. Furthermore, a module-ablation exercise highlights the critical role of prior expectations in simulating such heterogeneity. This approach complements traditional survey methods and offers new insights into AI behavioral science in macroeconomic research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04290v2">Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 20 pages, 6 figures, updated author affiliation and removed prior submission note
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM), which have developed in recent years, enable credit risk assessment through the analysis of financial texts such as analyst reports and corporate disclosures. This paper presents the first systematic review and taxonomy focusing on LLMbased approaches in credit risk estimation. We determined the basic model architectures by selecting 60 relevant papers published between 2020-2025 with the PRISMA research strategy. And we examined the data used for scenarios such as credit default prediction and risk analysis. Since the main focus of the paper is interpretability, we classify concepts such as explainability mechanisms, chain of thought prompts and natural language justifications for LLM-based credit models. The taxonomy organizes the literature under four main headings: model architectures, data types, explainability mechanisms and application areas. Based on this analysis, we highlight the main future trends and research gaps for LLM-based credit scoring systems. This paper aims to be a reference paper for artificial intelligence and financial researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06775v1">They want to pretend not to understand: The Limits of Current LLMs in Interpreting Implicit Content of Political Discourse</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Accepted to the ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Implicit content plays a crucial role in political discourse, where speakers systematically employ pragmatic strategies such as implicatures and presuppositions to influence their audiences. Large Language Models (LLMs) have demonstrated strong performance in tasks requiring complex semantic and pragmatic understanding, highlighting their potential for detecting and explaining the meaning of implicit content. However, their ability to do this within political discourse remains largely underexplored. Leveraging, for the first time, the large IMPAQTS corpus, which comprises Italian political speeches with the annotation of manipulative implicit content, we propose methods to test the effectiveness of LLMs in this challenging problem. Through a multiple-choice task and an open-ended generation task, we demonstrate that all tested models struggle to interpret presuppositions and implicatures. We conclude that current LLMs lack the key pragmatic capabilities necessary for accurately interpreting highly implicit language, such as that found in political discourse. At the same time, we highlight promising trends and future directions for enhancing model performance. We release our data and code at https://github.com/WalterPaci/IMPAQTS-PID
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06767v1">Beyond Surface Similarity: Evaluating LLM-Based Test Refactorings with Structural and Semantic Awareness</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed to automatically refactor unit tests, aiming to enhance readability, naming, and structural clarity while preserving functional behavior. However, evaluating such refactorings remains challenging: traditional metrics like CodeBLEU are overly sensitive to renaming and structural edits, whereas embedding-based similarities capture semantics but ignore readability and modularity. We introduce CTSES, a composite metric that integrates CodeBLEU, METEOR, and ROUGE-L to balance behavior preservation, lexical quality, and structural alignment. CTSES is evaluated on over 5,000 test suites automatically refactored by GPT-4o and Mistral-Large-2407, using Chain-of-Thought prompting, across two established Java benchmarks: Defects4J and SF110. Our results show that CTSES yields more faithful and interpretable assessments, better aligned with developer expectations and human intuition than existing metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06751v1">Geopolitical biases in LLMs: what are the "good" and the "bad" countries according to contemporary language models</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      This paper evaluates geopolitical biases in LLMs with respect to various countries though an analysis of their interpretation of historical events with conflicting national perspectives (USA, UK, USSR, and China). We introduce a novel dataset with neutral event descriptions and contrasting viewpoints from different countries. Our findings show significant geopolitical biases, with models favoring specific national narratives. Additionally, simple debiasing prompts had a limited effect in reducing these biases. Experiments with manipulated participant labels reveal models' sensitivity to attribution, sometimes amplifying biases or recognizing inconsistencies, especially with swapped labels. This work highlights national narrative biases in LLMs, challenges the effectiveness of simple debiasing methods, and offers a framework and dataset for future geopolitical bias research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08468v3">Multi-GraspLLM: A Multimodal LLM for Multi-Hand Semantic Guided Grasp Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Multi-hand semantic grasp generation aims to generate feasible and semantically appropriate grasp poses for different robotic hands based on natural language instructions. Although the task is highly valuable, due to the lack of multihand grasp datasets with fine-grained contact description between robotic hands and objects, it is still a long-standing difficult task. In this paper, we present Multi-GraspSet, the first large-scale multi-hand grasp dataset with automatically contact annotations. Based on Multi-GraspSet, we propose Multi-GraspLLM, a unified language-guided grasp generation framework, which leverages large language models (LLM) to handle variable-length sequences, generating grasp poses for diverse robotic hands in a single unified architecture. Multi-GraspLLM first aligns the encoded point cloud features and text features into a unified semantic space. It then generates grasp bin tokens that are subsequently converted into grasp pose for each robotic hand via hand-aware linear mapping. The experimental results demonstrate that our approach significantly outperforms existing methods in both real-world experiments and simulator. More information can be found on our project page https://multi-graspllm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06725v1">WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11586v2">Broaden your SCOPE! Efficient Multi-turn Conversation Planning for LLMs with Semantic Space</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 ICLR 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used in chatbots or AI assistants to hold conversations with a human user. In such applications, the quality (e.g., user engagement, safety) of a conversation is important and can only be exactly known at the end of the conversation. To maximize its expected quality, conversation planning reasons about the stochastic transitions within a conversation to select the optimal LLM response at each turn. Existing simulation-based conversation planning algorithms typically select the optimal response by simulating future conversations with a large number of LLM queries at every turn. However, this process is extremely time-consuming and hence impractical for real-time conversations. This paper presents a novel approach called Semantic space COnversation Planning with improved Efficiency (SCOPE) that exploits the dense semantic representation of conversations to perform conversation planning efficiently. In particular, SCOPE models the stochastic transitions in conversation semantics and their associated rewards to plan entirely within the semantic space. This allows us to select the optimal LLM response at every conversation turn without needing additional LLM queries for simulation. As a result, SCOPE can perform conversation planning 70 times faster than conventional simulation-based planning algorithms when applied to a wide variety of conversation starters and two reward functions seen in the real world, yet achieving a higher reward within a practical planning budget. Our code can be found at: https://github.com/chenzhiliang94/convo-plan-SCOPE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06705v1">DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Zhihui Chen and Kai He contributed equally to this work, Mengling Feng is the corresponding author
    </div>
    <details class="paper-abstract">
      Detecting LLM-generated text in specialized and high-stakes domains like medicine and law is crucial for combating misinformation and ensuring authenticity. However, current zero-shot detectors, while effective on general text, often fail when applied to specialized content due to domain shift. We provide a theoretical analysis showing this failure is fundamentally linked to the KL divergence between human, detector, and source text distributions. To address this, we propose DivScore, a zero-shot detection framework using normalized entropy-based scoring and domain knowledge distillation to robustly identify LLM-generated text in specialized domains. We also release a domain-specific benchmark for LLM-generated text detection in the medical and legal domains. Experiments on our benchmark show that DivScore consistently outperforms state-of-the-art detectors, with 14.4% higher AUROC and 64.0% higher recall (0.1% false positive rate threshold). In adversarial settings, DivScore demonstrates superior robustness than other baselines, achieving on average 22.8% advantage in AUROC and 29.5% in recall. Code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06699v1">MarginSel : Max-Margin Demonstration Selection for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at few-shot learning via in-context learning (ICL). However, the effectiveness of ICL is often sensitive to the selection and ordering of demonstration examples. To address this, we present MarginSel: Max-Margin Demonstration Selection for LLMs, a two-step method that selects hard demonstration examples for the ICL prompt, adapting to each test instance. Our approach achieves 2-7% absolute improvement in F1-score across classification tasks, compared to a random selection of examples. We also provide theoretical insights and empirical evidence showing that MarginSel induces max-margin behavior in LLMs by effectively increasing the margin for hard examples, analogous to support vectors, thereby shifting the decision boundary in a beneficial direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02590v2">Legal Mathematical Reasoning with LLMs: Procedural Alignment through Two-Stage Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Legal mathematical reasoning is essential for applying large language models (LLMs) in high-stakes legal contexts, where outputs must be both mathematically accurate and procedurally compliant. However, existing legal LLMs lack structured numerical reasoning, and open-domain models, though capable of calculations, often overlook mandatory legal steps. To address this, we present LexNum, the first Chinese legal mathematical reasoning benchmark, covering three representative scenarios where each instance reflects legally grounded procedural flows. We further propose LexPam, a two-stage reinforcement learning framework for efficient legal reasoning training. Leveraging curriculum learning, we use a stronger teacher model to partition data into basic and challenging subsets. A lightweight 1.5B student model is then fine-tuned with Group Relative Policy Optimization, which avoids costly value networks and enables stable training from sparse, end-of-sequence rewards. The first stage improves accuracy and format; the second introduces a novel reward to guide procedural alignment via task-specific legal elements. Experiments show that existing models perform poorly on LexNum, while LexPam enhances both mathematical accuracy and legal coherence, and generalizes effectively across tasks and domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01977v2">AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-07
      | 💬 Accepted to ACL 2025 Main
    </div>
    <details class="paper-abstract">
      User interface understanding with vision-language models (VLMs) has received much attention due to its potential for enhancing software automation. However, existing datasets used to build UI-VLMs either only contain large-scale context-free element annotations or contextualized functional descriptions for elements at a small scale. In this work, we propose the \textbf{AutoGUI} pipeline for automatically annotating UI elements with detailed functionality descriptions at scale. Specifically, we leverage large language models (LLMs) to infer element functionality by comparing UI state changes before and after simulated interactions. To improve annotation quality, we propose LLM-aided rejection and verification, eliminating invalid annotations without human labor. We construct a high-quality AutoGUI-704k dataset using the proposed pipeline, featuring diverse and detailed functionality annotations that are hardly provided by previous datasets. Human evaluation shows that we achieve annotation correctness comparable to a trained human annotator. Extensive experiments show that our dataset remarkably enhances VLM's UI grounding capabilities and exhibits significant scaling effects. We also show the interesting potential use of our dataset in UI agent tasks. Please view our project at https://autogui-project.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04204v2">Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks: Theoretical and Empirical Evidence</a></div>
    <div class="paper-meta">
      📅 2025-06-07
    </div>
    <details class="paper-abstract">
      Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. While long-length adversarial prompts during AT might lead to strong LLM robustness, their synthesis however is very resource-consuming, which may limit the application of LLM AT. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the numbers of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix length during jailbreaking to the length during AT. Our findings show that it is practical to defend against ``long-length'' jailbreak attacks via efficient ``short-length'' AT. The code is available at https://github.com/fshp971/adv-icl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09345v4">Rational Tuning of LLM Cascades via Probabilistic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 20 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Understanding the reliability of large language models (LLMs) has recently garnered significant attention. Given LLMs' propensity to hallucinate, as well as their high sensitivity to prompt design, it is already challenging to predict the performance of an individual LLM. However, the problem becomes more complex for compound LLM systems such as cascades, where in addition to each model's standalone performance, we must understand how the error rates of different models interact. In this paper, we present a probabilistic model for the joint performance distribution of a sequence of LLMs, which enables a framework for rationally tuning the confidence thresholds of a LLM cascade using continuous optimization. Compared to selecting confidence thresholds using Bayesian optimization, our parametric Markov-copula model yields more favorable error-cost trade-offs, improving the area under the error-cost curve by 4.3% on average for cascades with $k\geq 3$ models. In the low-sample regime with $n \leq 30$ training examples, the performance improvement widens to 10.2%, suggesting that our framework's inductive assumptions about the interactions between the error rates of different LLMs enhance sample efficiency. Overall, our Markov-copula model provides a rational basis for tuning LLM cascade performance and points to the potential of probabilistic methods in analyzing systems of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09568v3">MergeRepair: An Exploratory Study on Merging Task-Specific Adapters in Code LLMs for Automated Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown high capabilities in several software development-related tasks such as program repair, documentation, code refactoring, debugging, and testing. However, training these models requires massive amount of data and significant computational resources. Adapters are specialized, small modules designed for parameter efficient fine-tuning of LLMs for specific tasks, domains, or applications without requiring extensive retraining of the entire model. These adapters offer a more efficient way to customize LLMs for particular needs, leveraging the pre-existing capabilities of the large model. Model (and adapter) merging have emerged as a technique to develop one model capable of multiple tasks, with minimal or no training required. Although model and adapter merging has shown promising performance in domains such as natural language processing and computer vision, its applicability to software engineering tasks remains underexplored. In this paper, we investigate the effectiveness of merged adapters within the context of software engineering, with a particular focus on the Automated Program Repair (APR) task, through our approach, MergeRepair. In particular, we merge multiple task-specific adapters using three different merging methods, including weight-averaging, ties, and dare-ties, and evaluate the performance of the merged adapter on the APR task. We introduce a continual merging approach, a novel method in which we sequentially merge the task-specific adapters where the order and weight of the merged adapters play a significant role. We further compare the performance of our approach with a baseline method consisting of equal-weight merging applied on parameters of different adapters, where all adapters are of equal importance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06532v1">Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted in ICML 2025 Workshop on Machine Learning for Wireless Communication and Networks (ML4Wireless)
    </div>
    <details class="paper-abstract">
      Unmanned aerial vehicles (UAVs) have been widely adopted in various real-world applications. However, the control and optimization of multi-UAV systems remain a significant challenge, particularly in dynamic and constrained environments. This work explores the joint motion and communication control of multiple UAVs operating within integrated terrestrial and non-terrestrial networks that include high-altitude platform stations (HAPS). Specifically, we consider an aerial highway scenario in which UAVs must accelerate, decelerate, and change lanes to avoid collisions and maintain overall traffic flow. Different from existing studies, we propose a novel hierarchical and collaborative method based on large language models (LLMs). In our approach, an LLM deployed on the HAPS performs UAV access control, while another LLM onboard each UAV handles motion planning and control. This LLM-based framework leverages the rich knowledge embedded in pre-trained models to enable both high-level strategic planning and low-level tactical decisions. This knowledge-driven paradigm holds great potential for the development of next-generation 3D aerial highway systems. Experimental results demonstrate that our proposed collaborative LLM-based method achieves higher system rewards, lower operational costs, and significantly reduced UAV collision rates compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06522v1">Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06519v1">Hierarchical Debate-Based Large Language Model (LLM) for Complex Task Planning of 6G Network Management</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      6G networks have become increasingly complicated due to novel network architecture and newly emerging signal processing and transmission techniques, leading to significant burdens to 6G network management. Large language models (LLMs) have recently been considered a promising technique to equip 6G networks with AI-native intelligence. Different from most existing studies that only consider a single LLM, this work involves a multi-LLM debate-based scheme for 6G network management, where multiple LLMs can collaboratively improve the initial solution sequentially. Considering the complex nature of 6G domain, we propose a novel hierarchical debate scheme: LLMs will first debate the sub-task decomposition, and then debate each subtask step-by-step. Such a hierarchical approach can significantly reduce the overall debate difficulty by sub-task decomposition, aligning well with the complex nature of 6G networks and ensuring the final solution qualities. In addition, to better evaluate the proposed technique, we have defined a novel dataset named 6GPlan, including 110 complex 6G network management tasks and 5000 keyword solutions. Finally, the experiments show that the proposed hierarchical debate can significantly improve performance compared to baseline techniques, e.g. more than 30% coverage rate and global recall rate improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02930v3">Video LLMs for Temporal Reasoning in Long Videos</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      This paper introduces TemporalVLM, a video large language model (video LLM) capable of effective temporal reasoning and fine-grained understanding in long videos. At the core, our approach includes a visual encoder for mapping a long-term input video into features which are time-aware and contain both local and global cues. In particular, it first divides the input video into short-term clips, which are jointly encoded with their timestamps into time-sensitive local features. Next, the local features are passed through a bidirectional long short-term memory (BiLSTM) module for global feature aggregation. The extracted time-aware and multi-level features are important for accurate temporal reasoning and fine-grained understanding in long videos. Moreover, to facilitate the evaluation of TemporalVLM, we present a large-scale long video dataset of industry assembly processes, namely IndustryASM, which consists of videos recorded on factory floors with actions and timestamps annotated by industrial engineers for time and motion studies and temporal action segmentation evaluation. Finally, extensive experiments on datasets of long videos, including TimeIT and IndustryASM, show that TemporalVLM achieves superior performance than previous methods across temporal reasoning and fine-grained understanding tasks, namely dense video captioning, temporal video grounding, video highlight detection, and temporal action segmentation. To the best of our knowledge, our work is the first to incorporate LSTMs into video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06509v1">Private GPTs for LLM-driven testing in software development and machine learning</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 5 pages, 10 figures
    </div>
    <details class="paper-abstract">
      In this contribution, we examine the capability of private GPTs to automatically generate executable test code based on requirements. More specifically, we use acceptance criteria as input, formulated as part of epics, or stories, which are typically used in modern development processes. This gives product owners, or business intelligence, respectively, a way to directly produce testable criteria through the use of LLMs. We explore the quality of the so-produced tests in two ways: i) directly by letting the LLM generate code from requirements, ii) through an intermediate step using Gherkin syntax. As a result, it turns out that the two-step procedure yields better results -where we define better in terms of human readability and best coding practices, i.e. lines of code and use of additional libraries typically used in testing. Concretely, we evaluate prompt effectiveness across two scenarios: a simple "Hello World" program and a digit classification model, showing that structured prompts lead to higher-quality test outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06500v1">Improving LLM-Powered EDA Assistants with RAFT</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted paper at IEEE International Conference on LLM-Aided Design, 2025 (LAD 2025)
    </div>
    <details class="paper-abstract">
      Electronic design engineers often struggle to efficiently access relevant information for tasks like design verification and technology development. While large language models (LLMs) can enhance productivity as conversational agents, pre-trained open-source LLMs lack domain-specific knowledge for Electronic Design Automation (EDA). In a Retrieval-Augmented Generation (RAG) context, LLMs rely on external context but may still produce inaccurate responses. Retrieval-Augmented Fine-Tuning (RAFT) improves LLM performance, but acquiring labeled question/answer (Q/A) data in EDA is difficult. To address this, we propose using synthetic Q/A datasets to enhance LLMs with RAFT. Our results show that RAFT with synthetic data significantly boosts LLM performance for RAG-based EDA tasks. We also investigate the impact of using real user questions as Retrieval-Augmented Few-Shot (RAFS) examples for synthetic data generation. Additionally, we implement secure access control to ensure sensitive information is only accessible to authorized personnel. Finally, we assess the risk of data leakage and unintended memorization during fine-tuning with synthetic data, providing practical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06472v1">Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      We present the design and implementation of a new lifetime-aware tensor offloading framework for GPU memory expansion using low-cost PCIe-based solid-state drives (SSDs). Our framework, TERAIO, is developed explicitly for large language model (LLM) training with multiple GPUs and multiple SSDs. Its design is driven by our observation that the active tensors take only a small fraction (1.7% on average) of allocated GPU memory in each LLM training iteration, the inactive tensors are usually large and will not be used for a long period of time, creating ample opportunities for offloading/prefetching tensors to/from slow SSDs without stalling the GPU training process. TERAIO accurately estimates the lifetime (active period of time in GPU memory) of each tensor with the profiling of the first few iterations in the training process. With the tensor lifetime analysis, TERAIO will generate an optimized tensor offloading/prefetching plan and integrate it into the compiled LLM program via PyTorch. TERAIO has a runtime tensor migration engine to execute the offloading/prefetching plan via GPUDirect storage, which allows direct tensor migration between GPUs and SSDs for alleviating the CPU bottleneck and maximizing the SSD bandwidth utilization. In comparison with state-of-the-art studies such as ZeRO-Offload and ZeRO-Infinity, we show that TERAIO improves the training performance of various LLMs by 1.47x on average, and achieves 80.7% of the ideal performance assuming unlimited GPU memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21398v2">In a Few Words: Comparing Weak Supervision and LLMs for Short Query Intent Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 accepted at International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13--18, 2025, Padua, Italy
    </div>
    <details class="paper-abstract">
      User intent classification is an important task in information retrieval. Previously, user intents were classified manually and automatically; the latter helped to avoid hand labelling of large datasets. Recent studies explored whether LLMs can reliably determine user intent. However, researchers have recognized the limitations of using generative LLMs for classification tasks. In this study, we empirically compare user intent classification into informational, navigational, and transactional categories, using weak supervision and LLMs. Specifically, we evaluate LLaMA-3.1-8B-Instruct and LLaMA-3.1-70B-Instruct for in-context learning and LLaMA-3.1-8B-Instruct for fine-tuning, comparing their performance to an established baseline classifier trained using weak supervision (ORCAS-I). Our results indicate that while LLMs outperform weak supervision in recall, they continue to struggle with precision, which shows the need for improved methods to balance both metrics effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13791v2">From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Published as conference paper at ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in working environments for a wide range of tasks, excelling at solving individual problems in isolation. However, are they also able to effectively collaborate over long-term interactions? To investigate this, we introduce MemoryCode, a synthetic multi-session dataset designed to test LLMs' ability to track and execute simple coding instructions amid irrelevant information, simulating a realistic setting. While all the models we tested handle isolated instructions well, even the performance of state-of-the-art models like GPT-4o deteriorates when instructions are spread across sessions. Our analysis suggests this is due to their failure to retrieve and integrate information over long instruction chains. Our results highlight a fundamental limitation of current LLMs, restricting their ability to collaborate effectively in long interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06444v1">Saffron-1: Towards an Inference Scaling Paradigm for LLM Safety Assurance</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Existing safety assurance research has primarily focused on training-phase alignment to instill safe behaviors into LLMs. However, recent studies have exposed these methods' susceptibility to diverse jailbreak attacks. Concurrently, inference scaling has significantly advanced LLM reasoning capabilities but remains unexplored in the context of safety assurance. Addressing this gap, our work pioneers inference scaling for robust and effective LLM safety against emerging threats. We reveal that conventional inference scaling techniques, despite their success in reasoning tasks, perform poorly in safety contexts, even falling short of basic approaches like Best-of-N Sampling. We attribute this inefficiency to a newly identified challenge, the exploration--efficiency dilemma, arising from the high computational overhead associated with frequent process reward model (PRM) evaluations. To overcome this dilemma, we propose SAFFRON, a novel inference scaling paradigm tailored explicitly for safety assurance. Central to our approach is the introduction of a multifurcation reward model (MRM) that significantly reduces the required number of reward model evaluations. To operationalize this paradigm, we further propose: (i) a partial supervision training objective for MRM, (ii) a conservative exploration constraint to prevent out-of-distribution explorations, and (iii) a Trie-based key--value caching strategy that facilitates cache sharing across sequences during tree search. Extensive experiments validate the effectiveness of our method. Additionally, we publicly release our trained multifurcation reward model (Saffron-1) and the accompanying token-level safety reward dataset (Safety4M) to accelerate future research in LLM safety. Our code, model, and data are publicly available at https://github.com/q-rz/saffron , and our project homepage is at https://q-rz.github.io/p/saffron .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06404v1">Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      The application scope of Large Language Models (LLMs) continues to expand, leading to increasing interest in personalized LLMs that align with human values. However, aligning these models with individual values raises significant safety concerns, as certain values may correlate with harmful information. In this paper, we identify specific safety risks associated with value-aligned LLMs and investigate the psychological principles behind these challenges. Our findings reveal two key insights. (1) Value-aligned LLMs are more prone to harmful behavior compared to non-fine-tuned models and exhibit slightly higher risks in traditional safety evaluations than other fine-tuned models. (2) These safety issues arise because value-aligned LLMs genuinely generate text according to the aligned values, which can amplify harmful outcomes. Using a dataset with detailed safety categories, we find significant correlations between value alignment and safety risks, supported by psychological hypotheses. This study offers insights into the "black box" of value alignment and proposes in-context alignment methods to enhance the safety of value-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06401v1">Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 This work is accepted at ACL 2025
    </div>
    <details class="paper-abstract">
      Lightweight Large Language Models (LwLLMs) are reduced-parameter, optimized models designed to run efficiently on consumer-grade hardware, offering significant advantages in resource efficiency, cost-effectiveness, and data privacy. However, these models often struggle with limited inference and reasoning capabilities, which restrict their performance on complex tasks and limit their practical applicability. Moreover, existing prompt optimization methods typically rely on extensive manual effort or the meta-cognitive abilities of state-of-the-art LLMs, making them less effective for LwLLMs. To address these challenges, we introduce DeBoP, a new Direct Behavior Optimization Paradigm, original from the Chain-of-Thought (CoT) prompting technique. Unlike CoT Prompting, DeBoP is an automatic optimization method, which focuses on the optimization directly on the behavior of LwLLMs. In particular, DeBoP transforms the optimization of complex prompts into the optimization of discrete, quantifiable execution sequences using a gradient-free Monte Carlo Tree Search. We evaluate DeBoP on seven challenging tasks where state-of-the-art LLMs excel but LwLLMs generally underperform. Experimental results demonstrate that DeBoP significantly outperforms recent prompt optimization methods on most tasks. In particular, DeBoP-optimized LwLLMs surpass GPT-3.5 on most tasks while reducing computational time by approximately 60% compared to other automatic prompt optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21842v2">Diffusion as Reasoning: Enhancing Object Navigation via Diffusion Model Conditioned on LLM-based Object-Room Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      The Object Navigation (ObjectNav) task aims to guide an agent to locate target objects in unseen environments using partial observations. Prior approaches have employed location prediction paradigms to achieve long-term goal reasoning, yet these methods often struggle to effectively integrate contextual relation reasoning. Alternatively, map completion-based paradigms predict long-term goals by generating semantic maps of unexplored areas. However, existing methods in this category fail to fully leverage known environmental information, resulting in suboptimal map quality that requires further improvement. In this work, we propose a novel approach to enhancing the ObjectNav task, by training a diffusion model to learn the statistical distribution patterns of objects in semantic maps, and using the map of the explored regions during navigation as the condition to generate the map of the unknown regions, thereby realizing the long-term goal reasoning of the target object, i.e., diffusion as reasoning (DAR). Meanwhile, we propose the Room Guidance method, which leverages commonsense knowledge derived from large language models (LLMs) to guide the diffusion model in generating room-aware object distributions. Based on the generated map in the unknown region, the agent sets the predicted location of the target as the goal and moves towards it. Experiments on Gibson and MP3D show the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22777v2">MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots and Dialogue Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 May ARR
    </div>
    <details class="paper-abstract">
      As the capabilities of chatbots and their underlying LLMs continue to dramatically improve, evaluating their performance has increasingly become a major blocker to their further development. A major challenge is the available benchmarking datasets, which are largely static, outdated, and lacking in multilingual coverage, limiting their ability to capture subtle linguistic and cultural variations. This paper introduces MEDAL, an automated multi-agent framework for generating, evaluating, and curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. A strong LLM (GPT-4.1) is then used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. We find that current LLMs struggle to detect nuanced issues, particularly those involving empathy and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06240v1">Bridging External and Parametric Knowledge: Mitigating Hallucination of LLMs with Shared-Private Semantic Synergy in Dual-Stream Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) is a cost-effective approach to mitigate the hallucination of Large Language Models (LLMs) by incorporating the retrieved external knowledge into the generation process. However, external knowledge may conflict with the parametric knowledge of LLMs. Furthermore, current LLMs lack inherent mechanisms for resolving such knowledge conflicts, making traditional RAG methods suffer from degraded performance and stability. Thus, we propose a Dual-Stream Knowledge-Augmented Framework for Shared-Private Semantic Synergy (DSSP-RAG). Central to the framework is a novel approach that refines self-attention into a mixed-attention, distinguishing shared and private semantics for a controlled internal-external knowledge integration. To effectively facilitate DSSP in RAG, we further introduce an unsupervised hallucination detection method based on cognitive uncertainty, ensuring the necessity of introducing knowledge, and an Energy Quotient (EQ) based on attention difference matrices to reduce noise in the retrieved external knowledge. Extensive experiments on benchmark datasets show that DSSP-RAG can effectively resolve conflicts and enhance the complementarity of dual-stream knowledge, leading to superior performance over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06258v2">Emergent Response Planning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      In this work, we argue that large language models (LLMs), though trained to predict only the next token, exhibit emergent planning behaviors: $\textbf{their hidden representations encode future outputs beyond the next token}$. Through simple probing, we demonstrate that LLM prompt representations encode global attributes of their entire responses, including $\textit{structure attributes}$ (e.g., response length, reasoning steps), $\textit{content attributes}$ (e.g., character choices in storywriting, multiple-choice answers at the end of response), and $\textit{behavior attributes}$ (e.g., answer confidence, factual consistency). In addition to identifying response planning, we explore how it scales with model size across tasks and how it evolves during generation. The findings that LLMs plan ahead for the future in their hidden representations suggest potential applications for improving transparency and generation control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12120v2">LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 ICML 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data and tokenizer determine the scaling trend. In contrast, model size, optimization hyperparameters, and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14921v2">The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      How much information about training samples can be leaked through synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we assume an adversary has access to some synthetic data generated by a LLM. We design membership inference attacks (MIAs) that target the training data used to fine-tune the LLM that is then used to synthesize data. The significant performance of our MIA shows that synthetic data leak information about the training data. Further, we find that canaries crafted for model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their effectiveness. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06091v1">MIRIAD: Augmenting LLMs with millions of medical query-response pairs</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      LLMs are bound to transform healthcare with advanced decision support and flexible chat assistants. However, LLMs are prone to generate inaccurate medical content. To ground LLMs in high-quality medical knowledge, LLMs have been equipped with external knowledge via RAG, where unstructured medical knowledge is split into small text chunks that can be selectively retrieved and integrated into the LLMs context. Yet, existing RAG pipelines rely on raw, unstructured medical text, which can be noisy, uncurated and difficult for LLMs to effectively leverage. Systematic approaches to organize medical knowledge to best surface it to LLMs are generally lacking. To address these challenges, we introduce MIRIAD, a large-scale, curated corpus of 5,821,948 medical QA pairs, each rephrased from and grounded in a passage from peer-reviewed medical literature using a semi-automated pipeline combining LLM generation, filtering, grounding, and human annotation. Unlike prior medical corpora, which rely on unstructured text, MIRIAD encapsulates web-scale medical knowledge in an operationalized query-response format, which enables more targeted retrieval. Experiments on challenging medical QA benchmarks show that augmenting LLMs with MIRIAD improves accuracy up to 6.7% compared to unstructured RAG baselines with the same source corpus and with the same amount of retrieved text. Moreover, MIRIAD improved the ability of LLMs to detect medical hallucinations by 22.5 to 37% (increase in F1 score). We further introduce MIRIAD-Atlas, an interactive map of MIRIAD spanning 56 medical disciplines, enabling clinical users to visually explore, search, and refine medical knowledge. MIRIAD promises to unlock a wealth of down-stream applications, including medical information retrievers, enhanced RAG applications, and knowledge-grounded chat interfaces, which ultimately enables more reliable LLM applications in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06069v1">Zero-Shot Detection of LLM-Generated Code via Approximated Task Conditioning</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 To appear in the Proceedings of ECML-PKDD 2025, Springer Lecture Notes in Computer Science (LNCS)
    </div>
    <details class="paper-abstract">
      Detecting Large Language Model (LLM)-generated code is a growing challenge with implications for security, intellectual property, and academic integrity. We investigate the role of conditional probability distributions in improving zero-shot LLM-generated code detection, when considering both the code and the corresponding task prompt that generated it. Our key insight is that when evaluating the probability distribution of code tokens using an LLM, there is little difference between LLM-generated and human-written code. However, conditioning on the task reveals notable differences. This contrasts with natural language text, where differences exist even in the unconditional distributions. Leveraging this, we propose a novel zero-shot detection approach that approximates the original task used to generate a given code snippet and then evaluates token-level entropy under the approximated task conditioning (ATC). We further provide a mathematical intuition, contextualizing our method relative to previous approaches. ATC requires neither access to the generator LLM nor the original task prompts, making it practical for real-world applications. To the best of our knowledge, it achieves state-of-the-art results across benchmarks and generalizes across programming languages, including Python, CPP, and Java. Our findings highlight the importance of task-level conditioning for LLM-generated code detection. The supplementary materials and code are available at https://github.com/maorash/ATC, including the dataset gathering implementation, to foster further research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06066v1">Conversational Interfaces for Parametric Conceptual Architectural Design: Integrating Mixed Reality with LLM-driven Interaction</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Mixed reality (MR) environments offer embodied spatial interaction, providing intuitive 3D manipulation capabilities that enhance the conceptual design process. Parametric modeling, a powerful and advanced architectural design method, enables the generation of complex, optimized geometries. However, its integration into MR environments remains limited due to precision constraints and unsuitable input modalities. Existing MR tools prioritize spatial interaction but lack the control and expressiveness required for parametric workflows, particularly for designers without formal programming backgrounds. We address this gap by introducing a novel conversational MR interface that combines speech input, gesture recognition, and a multi-agent large language model (LLM) system to support intuitive parametric modeling. Our system dynamically manages parameter states, resolves ambiguous commands through conversation and contextual prompting, and enables real-time model manipulation within immersive environments. We demonstrate how this approach reduces cognitive and operational barriers in early-stage design tasks, allowing users to refine and explore their design space. This work expands the role of MR to a generative design platform, supporting programmatic thinking in design tasks through natural, embodied interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06017v1">AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 20pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated strong capabilities across diverse domains. However, designing high-performing agentic systems remains challenging. Existing agent search methods suffer from three major limitations: (1) an emphasis on optimizing agentic workflows while under-utilizing proven human-designed components such as memory, planning, and tool use; (2) high evaluation costs, as each newly generated agent must be fully evaluated on benchmarks; and (3) inefficient search in large search space. In this work, we introduce a comprehensive framework to address these challenges. First, We propose a hierarchical search space that jointly models agentic workflow and composable functional components, enabling richer agentic system designs. Building on this structured design space, we introduce a predictive value model that estimates agent performance given agentic system and task description, allowing for efficient, low-cost evaluation during the search process. Finally, we present a hierarchical Monte Carlo Tree Search (MCTS) strategy informed by uncertainty to guide the search. Experiments on seven benchmarks, covering embodied, math, web, tool, and game, show that our method achieves an average performance gain of 8.34\% over state-of-the-art baselines and exhibits faster search progress with steeper improvement trajectories. Code repo is available at https://github.com/Ericccc02/AgentSwift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06015v1">On the Merits of LLM-Based Corpus Enrichment</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Generative AI (genAI) technologies -- specifically, large language models (LLMs) -- and search have evolving relations. We argue for a novel perspective: using genAI to enrich a document corpus so as to improve query-based retrieval effectiveness. The enrichment is based on modifying existing documents or generating new ones. As an empirical proof of concept, we use LLMs to generate documents relevant to a topic which are more retrievable than existing ones. In addition, we demonstrate the potential merits of using corpus enrichment for retrieval augmented generation (RAG) and answer attribution in question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06009v1">Unlocking Recursive Thinking of LLMs: Alignment via Refinement</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted to the Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (https://github.com/Banner-Z/AvR.git).
    </details>
</div>
