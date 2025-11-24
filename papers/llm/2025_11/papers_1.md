# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17467v1">PersonaAgent with GraphRAG: Community-Aware Knowledge Graphs for Personalized LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We propose a novel framework for persona-based language model system, motivated by the need for personalized AI agents that adapt to individual user preferences. In our approach, the agent embodies the user's "persona" (e.g. user profile or taste) and is powered by a large language model (LLM). To enable the agent to leverage rich contextual information, we introduce a Knowledge-Graph-enhanced Retrieval-Augmented Generation (Graph RAG) mechanism that constructs an LLM-derived graph index of relevant documents and summarizes communities of related information. Our framework generates personalized prompts by combining: (1) a summary of the user's historical behaviors and preferences extracted from the knowledge graph, and (2) relevant global interaction patterns identified through graph-based community detection. This dynamic prompt engineering approach allows the agent to maintain consistent persona-aligned behaviors while benefiting from collective knowledge. On the LaMP benchmark, our method improves news categorization F1 by 11.1%, movie tagging F1 by 56.1%, and reduces product rating MAE by 10.4% over prior methods. Our code is available at https://anonymous.4open.science/r/PersonaAgentwGraphRAG-DE6F
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00086v2">Do LLMs produce texts with "human-like" lexical diversity?</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The degree to which large language models (LLMs) produce writing that is truly human-like remains unclear despite the extensive empirical attention that this question has received. The present study addresses this question from the perspective of lexical diversity. Specifically, the study investigates patterns of lexical diversity in LLM-generated texts from four ChatGPT models (ChatGPT-3.5, ChatGPT-4, ChatGPT-o4 mini, and ChatGPT-4.5) in comparison with texts written by L1 and L2 English participants (n = 240) across four education levels. Six dimensions of lexical diversity were measured in each text: volume, abundance, variety-repetition, evenness, disparity, and dispersion. Results from one-way MANOVAs, one-way ANOVAs, and Support Vector Machines revealed that the ChatGPT-generated texts differed significantly from human-written texts for each variable, with ChatGPT-o4 mini and ChatGPT-4.5 differing the most. Within these two groups, ChatGPT-4.5 demonstrated higher levels of lexical diversity than older models despite producing fewer tokens. The human writers' lexical diversity did not differ across subgroups (i.e., education, language status). Altogether, the results indicate that ChatGPT models do not produce human-like texts in relation to lexical diversity, and the newer models produce less human-like text than older models. We discuss the implications of these results for language pedagogy and related applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17442v1">REMSA: An LLM Agent for Foundation Model Selection in Remote Sensing</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Code and data available at https://github.com/be-chen/REMSA
    </div>
    <details class="paper-abstract">
      Foundation Models (FMs) are increasingly used in remote sensing (RS) for tasks such as environmental monitoring, disaster assessment, and land-use mapping. These models include unimodal vision encoders trained on a single data modality and multimodal architectures trained on combinations of SAR, multispectral, hyperspectral, and image-text data. They support diverse RS tasks including semantic segmentation, image classification, change detection, and visual question answering. However, selecting an appropriate remote sensing foundation model (RSFM) remains difficult due to scattered documentation, heterogeneous formats, and varied deployment constraints. We introduce the RSFM Database (RS-FMD), a structured resource covering over 150 RSFMs spanning multiple data modalities, resolutions, and learning paradigms. Built on RS-FMD, we present REMSA, the first LLM-based agent for automated RSFM selection from natural language queries. REMSA interprets user requirements, resolves missing constraints, ranks candidate models using in-context learning, and provides transparent justifications. We also propose a benchmark of 75 expert-verified RS query scenarios, producing 900 configurations under an expert-centered evaluation protocol. REMSA outperforms several baselines, including naive agents, dense retrieval, and unstructured RAG-based LLMs. It operates entirely on publicly available metadata and does not access private or sensitive data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09439v3">Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU and MMAR benchmarks. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17335v1">Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Accepted to ASRU 2025
    </div>
    <details class="paper-abstract">
      Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17308v1">SpatialGeo:Boosting Spatial Reasoning in Multimodal LLMs via Geometry-Semantics Fusion</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have achieved significant progress in image and language tasks due to the strong reasoning capability of large language models (LLMs). Nevertheless, most MLLMs suffer from limited spatial reasoning ability to interpret and infer spatial arrangements in three-dimensional space. In this work, we propose a novel vision encoder based on hierarchical fusion of geometry and semantics features, generating spatial-aware visual embedding and boosting the spatial grounding capability of MLLMs. Specifically, we first unveil that the spatial ambiguity shortcoming stems from the lossy embedding of the vision encoder utilized in most existing MLLMs (e.g., CLIP), restricted to instance-level semantic features. This motivates us to complement CLIP with the geometry features from vision-only self-supervised learning via a hierarchical adapter, enhancing the spatial awareness in the proposed SpatialGeo. The network is efficiently trained using pretrained LLaVA model and optimized with random feature dropping to avoid trivial solutions relying solely on the CLIP encoder. Experimental results show that SpatialGeo improves the accuracy in spatial reasoning tasks, enhancing state-of-the-art models by at least 8.0% in SpatialRGPT-Bench with approximately 50% less memory cost during inference. The source code is available via https://ricky-plus.github.io/SpatialGeoPages/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.04736v2">The promise and limits of LLMs in constructing proofs and hints for logic problems in intelligent tutoring systems</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance up to 86.7% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but require additional modifications to ensure accuracy and pedagogical appropriateness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13302v2">LLM one-shot style transfer for Authorship Attribution and Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Computational stylometry analyzes writing style through quantitative patterns in text, supporting applications from forensic tasks such as identity linking and plagiarism detection to literary attribution in the humanities. Supervised and contrastive approaches rely on data with spurious correlations and often confuse style with topic. Despite their natural use in AI-generated text detection, the CLM pre-training of modern LLMs has been scarcely leveraged for general authorship problems. We propose a novel unsupervised approach based on this extensive pre-training and the in-context learning capabilities of LLMs, employing the log-probabilities of an LLM to measure style transferability from one text to another. Our method significantly outperforms LLM prompting approaches of comparable scale and achieves higher accuracy than contrastively trained baselines when controlling for topical correlations. Moreover, performance scales fairly consistently with the size of the base model and, in the case of authorship verification, with an additional mechanism that increases test-time computation; enabling flexible trade-offs between computational cost and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17262v1">SlsReuse: LLM-Powered Serverless Function Reuse</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Serverless computing has rapidly emerged as a popular cloud computing paradigm. It enables developers to implement function-level tasks, i.e., serverless functions, without managing infrastructure. While reducing operational overhead, it poses challenges, especially for novice developers. Developing functions from scratch requires adapting to heterogeneous, platform-specific programming styles, making the process time-consuming and error-prone. Function reuse offers a promising solution to address these challenges. However, research on serverless computing lacks a dedicated approach for function recommendation. Existing techniques from traditional contexts remain insufficient due to the semantic gap between task descriptions and heterogeneous function implementations. Advances in large language models (LLMs), pre-trained on large-scale corpora, create opportunities to bridge this gap by aligning developer requirements with function semantics. This paper presents SlsReuse, the first LLM-powered framework for serverless function reuse. Specifically, SlsReuse first constructs a reusable function repository serving as a foundational knowledge base. Then, it learns unified semantic-enhanced representations of heterogeneous functions through effective prompt engineering with few-shot prompting, capturing implicit code intent, target platforms, programming languages, and cloud services. Finally, given a natural language task query, SlsReuse performs intent-aware discovery combined with a multi-level pruning strategy and similarity matching. We evaluate SlsReuse on a curated dataset of 110 task queries. Built on ChatGPT-4o, one of the most representative LLMs, SlsReuse achieves Recall@10 of 91.20%, exceeding the state-of-the-art baseline by 24.53 percentage points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16224v2">Beyond Code Similarity: Benchmarking the Plausibility, Efficiency, and Complexity of LLM-Generated Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Smart Contracts are critical components of blockchain ecosystems, with Solidity as the dominant programming language. While LLMs excel at general-purpose code generation, the unique constraints of Smart Contracts, such as gas consumption, security, and determinism, raise open questions about the reliability of LLM-generated Solidity code. Existing studies lack a comprehensive evaluation of these critical functional and non-functional properties. We benchmark four state-of-the-art models under zero-shot and retrieval-augmented generation settings across 500 real-world functions. Our multi-faceted assessment employs code similarity metrics, semantic embeddings, automated test execution, gas profiling, and cognitive and cyclomatic complexity analysis. Results show that while LLMs produce code with high semantic similarity to real contracts, their functional correctness is low: only 20% to 26% of zero-shot generations behave identically to ground-truth implementations under testing. The generated code is consistently simpler, with significantly lower complexity and gas consumption, often due to omitted validation logic. Retrieval-Augmented Generation markedly improves performance, boosting functional correctness by up to 45% and yielding more concise and efficient code. Our findings reveal a significant gap between semantic similarity and functional plausibility in LLM-generated Smart Contracts. We conclude that while RAG is a powerful enhancer, achieving robust, production-ready code generation remains a substantial challenge, necessitating careful expert validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.11393v3">LLM-Agent-UMF: LLM-based Agent Unified Modeling Framework for Seamless Design of Multi Active/Passive Core-Agent Architectures</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 39 pages, 19 figures, 3 tables. Published in Information Fusion, Volume 127, March 2026, 103865. Part of the special issue "Data Fusion Approaches in Data-Centric AI for Developing Trustworthy AI Systems"
    </div>
    <details class="paper-abstract">
      In an era where vast amounts of data are collected and processed from diverse sources, there is a growing demand for sophisticated AI systems capable of intelligently fusing and analyzing this information. To address these challenges, researchers have turned towards integrating tools into LLM-powered agents to enhance the overall information fusion process. However, the conjunction of these technologies and the proposed enhancements in several state-of-the-art works followed a non-unified software architecture, resulting in a lack of modularity and terminological inconsistencies among researchers. To address these issues, we propose a novel LLM-based Agent Unified Modeling Framework (LLM-Agent-UMF) that establishes a clear foundation for agent development from both functional and software architectural perspectives, developed and evaluated using the Architecture Tradeoff and Risk Analysis Framework (ATRAF). Our framework clearly distinguishes between the different components of an LLM-based agent, setting LLMs and tools apart from a new element, the core-agent, which plays the role of central coordinator. This pivotal entity comprises five modules: planning, memory, profile, action, and security -- the latter often neglected in previous works. By classifying core-agents into passive and active types based on their authoritative natures, we propose various multi-core agent architectures that combine unique characteristics of distinctive agents to tackle complex tasks more efficiently. We evaluate our framework by applying it to thirteen state-of-the-art agents, thereby demonstrating its alignment with their functionalities and clarifying overlooked architectural aspects. Moreover, we thoroughly assess five architecture variants of our framework by designing new agent architectures that combine characteristics of state-of-the-art agents to address specific goals. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v4">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at https://github.com/OpenMLRL/CoMLRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17220v1">Parrot: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      This study presents PARROT (Persuasion and Agreement Robustness Rating of Output Truth), a robustness focused framework designed to measure the degradation in accuracy that occurs under social pressure exerted on users through authority and persuasion in large language models (LLMs) the phenomenon of sycophancy (excessive conformity). PARROT (i) isolates causal effects by comparing the neutral version of the same question with an authoritatively false version using a double-blind evaluation, (ii) quantifies confidence shifts toward the correct and imposed false responses using log-likelihood-based calibration tracking, and (iii) systematically classifies failure modes (e.g., robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.) using an eight-state behavioral taxonomy. We evaluated 22 models using 1,302 MMLU-style multiple-choice questions across 13 domains and domain-specific authority templates. Findings show marked heterogeneity: advanced models (e.g., GPT-5, GPT-4.1, Claude Sonnet 4.5) exhibit low "follow rates" ($\leq 11\%$, GPT-5: 4\%) and minimal accuracy loss, while older/smaller models show severe epistemic collapse (GPT-4: 80\%, Qwen 2.5-1.5B: 94\%). The danger is not limited to response changes; weak models reduce confidence in the correct response while increasing confidence in the imposed incorrect response. While international law and global knowledge at the domain level exhibit high fragility, elementary mathematics is relatively resilient. Consequently, we argue that the goal of "resistance to overfitting pressure" should be addressed as a primary objective alongside accuracy, harm avoidance, and privacy for safe deployment in the real world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17208v1">A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at https://github.com/KevinSRR/EMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12396v4">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07318v2">When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24975v2">DiffTester: Accelerating Unit Test Generation for Diffusion LLMs via Repetitive Pattern</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Update reference
    </div>
    <details class="paper-abstract">
      Software development relies heavily on extensive unit testing, which makes the efficiency of automated Unit Test Generation (UTG) particularly important. However, most existing LLMs generate test cases one token at a time in each forward pass, which leads to inefficient UTG. Recently, diffusion LLMs (dLLMs) have emerged, offering promising parallel generation capabilities and showing strong potential for efficient UTG. Despite this advantage, their application to UTG is still constrained by a clear trade-off between efficiency and test quality, since increasing the number of tokens generated in each step often causes a sharp decline in the quality of test cases. To overcome this limitation, we present DiffTester, an acceleration framework specifically tailored for dLLMs in UTG. The key idea of DiffTester is that unit tests targeting the same focal method often share repetitive structural patterns. By dynamically identifying these common patterns through abstract syntax tree analysis during generation, DiffTester adaptively increases the number of tokens produced at each step without compromising the quality of the output. To enable comprehensive evaluation, we extend the original TestEval benchmark, which was limited to Python, by introducing additional programming languages including Java and C++. Extensive experiments on three benchmarks with two representative models show that DiffTester delivers significant acceleration while preserving test coverage. Moreover, DiffTester generalizes well across different dLLMs and programming languages, providing a practical and scalable solution for efficient UTG in software development. Code and data are publicly available at https://github.com/wellbeingyang/DLM4UTG-open .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17124v1">A Counterfactual LLM Framework for Detecting Human Biases: A Case Study of Sex/Gender in Emergency Triage</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Currently under review at npj Digital Medicine
    </div>
    <details class="paper-abstract">
      We present a novel, domain-agnostic counterfactual approach that uses Large Language Models (LLMs) to quantify gender disparities in human clinical decision-making. The method trains an LLM to emulate observed decisions, then evaluates counterfactual pairs in which only gender is flipped, estimating directional disparities while holding all other clinical factors constant. We study emergency triage, validating the approach on more than 150,000 admissions to the Bordeaux University Hospital (France) and replicating results on a subset of MIMIC-IV across a different language, population, and healthcare system. In the Bordeaux cohort, otherwise identical presentations were approximately 2.1% more likely to receive a lower-severity triage score when presented as female rather than male; scaled to national emergency volumes in France, this corresponds to more than 200,000 lower-severity assignments per year. Modality-specific analyses indicate that both explicit tabular gender indicators and implicit textual gender cues contribute to the disparity. Beyond emergency care, the approach supports bias audits in other settings (e.g., hiring, academic, and justice decisions), providing a scalable tool to detect and address inequities in real-world decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23629v2">How LLMs Learn to Reason: A Complex Network Perspective</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 24 pages, 11 figures, 1 table, under review as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      Training large language models with Reinforcement Learning with Verifiable Rewards (RLVR) exhibits a set of distinctive and puzzling behaviors that remain poorly understood, including a two-stage learning curve, a V-shaped response-length trajectory, and a pronounced vulnerability to catastrophic forgetting. In this work, we propose that these behaviors are emergent collective phenomena governed not by neural implementation details, but by the topological evolution of the latent reasoning graph in semantic space. By demonstrating a dynamical isomorphism between a 1.5B-parameter LLM and a minimal Concept Network Model (CoNet), we trace the causal source to the self-organization of a sparse concept web pinned to an average degree of two. This geometric perspective provides a unified physical explanation for the observed anomalies: the V-shaped trajectory tracks the evolution from parallel local skill optimization to global network integration; catastrophic forgetting stems from the topological disconnection of critical ``trunk'' edges; and policy collapse arises from the accumulation of sequential transitions at the web's leaf nodes, where broad exploration abruptly freezes into rigid, high-reward trajectories. Identifying a ``maximally frustrated state'' at the transition between learning stages, we propose Annealed-RLVR, a principled algorithm that injects a targeted SFT ``heating'' step to resolve this topological bottleneck. Experiments confirm that this theory-driven intervention outperforms standard RLVR on both in-distribution and out-of-distribution benchmarks (including Minerva and AIME). By recasting RLVR from black-box optimization into a predictable process of structural self-organization, our work provides a new physical intuition for engineering the emergent reasoning capabilities of future AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.12972v3">Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 14 pages, 7 figures, 6 tables; Accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at https://github.com/Wings-Of-Disaster/VaLiK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17041v1">CLLMRec: LLM-powered Cognitive-Aware Concept Recommendation via Semantic Alignment and Prerequisite Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The growth of Massive Open Online Courses (MOOCs) presents significant challenges for personalized learning, where concept recommendation is crucial. Existing approaches typically rely on heterogeneous information networks or knowledge graphs to capture conceptual relationships, combined with knowledge tracing models to assess learners' cognitive states. However, these methods face significant limitations due to their dependence on high-quality structured knowledge graphs, which are often scarce in real-world educational scenarios. To address this fundamental challenge, this paper proposes CLLMRec, a novel framework that leverages Large Language Models through two synergistic technical pillars: Semantic Alignment and Prerequisite Knowledge Distillation. The Semantic Alignment component constructs a unified representation space by encoding unstructured textual descriptions of learners and concepts. The Prerequisite Knowledge Distillation paradigm employs a teacher-student architecture, where a large teacher LLM (implemented as the Prior Knowledge Aware Component) extracts conceptual prerequisite relationships from its internalized world knowledge and distills them into soft labels to train an efficient student ranker. Building upon these foundations, our framework incorporates a fine-ranking mechanism that explicitly models learners' real-time cognitive states through deep knowledge tracing, ensuring recommendations are both structurally sound and cognitively appropriate. Extensive experiments on two real-world MOOC datasets demonstrate that CLLMRec significantly outperforms existing baseline methods across multiple evaluation metrics, validating its effectiveness in generating truly cognitive-aware and personalized concept recommendations without relying on explicit structural priors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16193v2">Fast LLM Post-training via Decoupled and Best-of-N Speculation</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Rollout dominates the training time in large language model (LLM) post-training, where the trained model is used to generate tokens given a batch of prompts. SpecActor achieves fast rollout with speculative decoding that deploys a fast path (e.g., a smaller model) to accelerate the unparallelizable generation, while the correctness is guaranteed by fast parallel verification of the outputs with the original model. SpecActor addresses two foundational challenges in speculative rollout by (1) a \emph{dynamic decoupled speculation} execution method that maximizes the GPU computational efficiency to realize speedup for large-batch execution -- a configuration common in training but unfriendly to speculative execution and (2) a \emph{dynamic Best-of-N speculation} method that selects and combines different drafting methods according to the rollout progress. It substantially improves the speculation accuracy even when the best drafting method is unknown a priori, meanwhile without requiring adding extra computation resources. {\sys} is {1.7}\,$\times$ faster than veRL in end-to-end training, and is {1.3--1.5}\,$\times$ faster compared to baselines with speculative decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v2">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles,host factors, pharmacological properties of antimicrobials,and the severity of infection. This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at about 20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16964v1">Optimizing PyTorch Inference with LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Maximizing performance on available GPU hardware is an ongoing challenge for modern AI inference systems. Traditional approaches include writing custom GPU kernels and using specialized model compilers to tune high-level code for specific GPU targets. Recent work shows that LLM-based multi-agent systems can effectively perform such tuning, often outperforming existing compilers and eliminating the need for manual kernel development. However, the dynamics of multi-agent systems for this task remain unexplored. In this work, we present a logical framework for comparing multi-agent PyTorch optimization systems. Our evaluation shows that exploit-heavy strategies perform best when paired with error-fixing agents, and that performance correlates with the granularity of optimization steps. The best implementation achieves an average 2.88x speedup on an H100 GPU across diverse tasks in KernelBench, a benchmark suite covering a range of machine learning architectures in PyTorch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17987v3">Reason2Attack: Jailbreaking Text-to-Image Models via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Noted that This paper includes model-generated content that may contain offensive or distressing material
    </div>
    <details class="paper-abstract">
      Text-to-Image(T2I) models typically deploy safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods manually design instructions for the LLM to generate adversarial prompts, which effectively bypass safety filters while producing sensitive images, exposing safety vulnerabilities of T2I models. However, due to the LLM's limited understanding of the T2I model and its safety filters, existing methods require numerous queries to achieve a successful attack, limiting their practical applicability. To address this issue, we propose Reason2Attack(R2A), which aims to enhance the LLM's reasoning capabilities in generating adversarial prompts by incorporating the jailbreaking attack into the post-training process of the LLM. Specifically, we first propose a CoT example synthesis pipeline based on Frame Semantics, which generates adversarial prompts by identifying related terms and corresponding context illustrations. Using CoT examples generated by the pipeline, we fine-tune the LLM to understand the reasoning path and format the output structure. Subsequently, we incorporate the jailbreaking attack task into the reinforcement learning process of the LLM and design an attack process reward that considers prompt length, prompt stealthiness, and prompt effectiveness, aiming to further enhance reasoning accuracy. Extensive experiments on various T2I models show that R2A achieves a better attack success ratio while requiring fewer queries than baselines. Moreover, our adversarial prompts demonstrate strong attack transferability across both open-source and commercial T2I models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01265v2">AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper examines how domain specificity affects abstractive summarisation of Arabic financial texts using large language models (LLMs). We present AraFinNews, the largest publicly available Arabic financial news dataset to date, comprising 212,500 article-headline pairs spanning almost a decade of reporting from October 2015 to July 2025. Developed as an Arabic counterpart to major English summarisation corpora such as CNN/DailyMail, AraFinNews offers a strong benchmark for assessing domain-focused language understanding and generation in financial contexts. Using this resource, we evaluate transformer-based models, including mT5, AraT5 and the domain-adapted FinAraT5, to investigate how financial-domain pretraining influences accuracy, numerical reliability and stylistic alignment with professional reporting. The results show that domain-adapted models produce more coherent summaries, particularly when handling quantitative and entity-centred information. These findings underscore the value of domain-specific adaptation for improving narrative fluency in Arabic financial summarisation. The dataset is freely available for non-commercial research at https://github.com/ArabicNLP-UK/AraFinNews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v2">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16901v1">R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 Accepted by AAAI 2026. Project page: https://github.com/zhlllau/R-AVST
    </div>
    <details class="paper-abstract">
      Recently, rapid advancements have been made in multimodal large language models (MLLMs), especially in video understanding tasks. However, current research focuses on simple video scenarios, failing to reflect the complex and diverse nature of real-world audio-visual events in videos. To bridge this gap, we firstly introduce R-AVST, a dataset for audio-visual reasoning featuring fine-grained spatio-temporal annotations. In constructing this, we design a pipeline consisting of LLM-based key object extraction, automatic spatial annotation and manual quality inspection, resulting in over 5K untrimmed videos with 27K objects across 100 types of audio-visual events. Building on this dataset, we define three core tasks for spatio-temporal reasoning in audio-visual scenes and generate more than 8K high-quality, evenly distributed question-answer pairs to effectively benchmark model performance. To further enhance reasoning, we propose AVST-Zero, a reinforcement learning-based model that avoids intermediate supervision, directly optimizing behavior via carefully designed multi-dimensional rewards. Extensive experiments validate the effectiveness of our R-AVST in advancing audio-visual spatio-temporal reasoning, upon which AVST-Zero demonstrates competitive performance compared to existing models. To the best of our knowledge, R-AVST is the first dataset designed for real-world audio-visual spatio-temporal reasoning, and AVST-Zero offers a novel perspective for tackling future challenges in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16885v1">Improving Latent Reasoning in LLMs via Soft Concept Mixing</a></div>
    <div class="paper-meta">
      📅 2025-11-21
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Unlike human reasoning in abstract conceptual spaces, large language models (LLMs) typically reason by generating discrete tokens, which potentially limit their expressive power. The recent work Soft Thinking has shown that LLMs' latent reasoning via soft concepts is a promising direction, but LLMs are trained on discrete tokens. To reduce this gap between the soft concepts in reasoning and the discrete tokens in training, we propose Soft Concept Mixing (SCM), a soft concept aware training scheme that directly exposes the model to soft representations during training. Specifically, SCM constructs a soft concept vector by forming a probability-weighted average of embeddings. Then, this vector is mixed into the model's hidden states, which embody rich contextual information. Finally, the entire latent reasoning process is optimized with Reinforcement Learning (RL). Experiments on five reasoning benchmarks demonstrate that SCM improves the reasoning performance of LLMs, and simultaneously maintains a stable training dynamic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16883v1">PersonalizedRouter: Personalized LLM Routing via Graph-based User Preference Modeling</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      The growing number of Large Language Models (LLMs) with diverse capabilities and response styles provides users with a wider range of choices, which presents challenges in selecting appropriate LLMs, as user preferences vary in terms of performance, cost, and response style. Current LLM selection methods typically optimize for a single fixed objective, such as performance, cost, or a trade-off between them, and fail to learn individual user preferences from interaction data. To address these limitations, we propose PersonalizedRouter, a graph-based framework that models diverse user profiles and performs personalized LLM selection by leveraging interaction data that includes task context, queries, candidate LLMs, and user decisions. To capture contextual information between user queries and optimal LLMs, PersonalizedRouter converts the interaction data into a heterogeneous graph, where the relationships between different types of nodes are represented by edges. To evaluate adaptability across users, we design two strategies: the multi-cost-efficiency simulation strategy and the LLM-as-a-Judge strategy. In addition, we construct PersonaRoute-Bench, a large-scale benchmark with 1,000 simulated users and 10 LLMs. Experimental results show that PersonalizedRouter significantly outperforms existing LLM selection methods and surpasses the strongest methods by a large margin of 15.38% and 9.83% under two simulation strategies. On the PersonaRoute-Bench with 1,000 users, it further surpasses the best methods by 16.19% and 59.69% while maintaining higher efficiency. Moreover, PersonalizedRouter demonstrates strong few-shot generalization, achieving 64.81% and 85.80% of the fully trained model's performance when adapting to new users and new LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12188v3">LLM-DSE: Searching Accelerator Parameters with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-21
    </div>
    <details class="paper-abstract">
      Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample efficiency. We present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: https://github.com/Nozidoali/LLM-DSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16664v1">Nemotron Elastic: Towards Efficient Many-in-One Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Training a family of large language models targeting multiple scales and deployment objectives is prohibitively expensive, requiring separate training runs for each different size. Recent work on model compression through pruning and knowledge distillation has reduced this cost; however, this process still incurs hundreds of billions of tokens worth of training cost per compressed model. In this paper, we present Nemotron Elastic, a framework for building reasoning-oriented LLMs, including hybrid Mamba-Attention architectures, that embed multiple nested submodels within a single parent model, each optimized for different deployment configurations and budgets. Each of these submodels shares weights with the parent model and can be extracted zero-shot during deployment without additional training or fine-tuning. We enable this functionality through an end-to-end trained router, tightly coupled to a two-stage training curriculum designed specifically for reasoning models. We additionally introduce group-aware SSM elastification that preserves Mamba's structural constraints, heterogeneous MLP elastification, normalized MSE-based layer importance for improved depth selection, and knowledge distillation enabling simultaneous multi-budget optimization. We apply Nemotron Elastic to the Nemotron Nano V2 12B model, simultaneously producing a 9B and a 6B model using only 110B training tokens; this results in over 360x cost reduction compared to training model families from scratch, and around 7x compared to SoTA compression techniques. Each of the nested models performs on par or better than the SoTA in accuracy. Moreover, unlike other compression methods, the nested capability of our approach allows having a many-in-one reasoning model that has constant deployment memory against the number of models in the family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16660v1">Cognitive Foundations for Reasoning and Their Manifestation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 40 pages, 4 tables, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models solve complex problems yet fail on simpler variants, suggesting they achieve correct outputs through mechanisms fundamentally different from human reasoning. We synthesize cognitive science research into a taxonomy of 28 cognitive elements spanning computational constraints, meta-cognitive controls, knowledge representations, and transformation operations, then analyze their behavioral manifestations in reasoning traces. We propose a fine-grained cognitive evaluation framework and conduct the first large-scale analysis of 170K traces from 17 models across text, vision, and audio modalities, alongside 54 human think-aloud traces, which we make publicly available. Our analysis reveals systematic structural differences: humans employ hierarchical nesting and meta-cognitive monitoring while models rely on shallow forward chaining, with divergence most pronounced on ill-structured problems. Meta-analysis of 1,598 LLM reasoning papers reveals the research community concentrates on easily quantifiable behaviors (sequential organization: 55%, decomposition: 60%) while neglecting meta-cognitive controls (self-awareness: 16%, evaluation: 8%) that correlate with success. Models possess behavioral repertoires associated with success but fail to deploy them spontaneously. Leveraging these patterns, we develop test-time reasoning guidance that automatically scaffold successful structures, improving performance by up to 60% on complex problems. By bridging cognitive science and LLM research, we establish a foundation for developing models that reason through principled cognitive mechanisms rather than brittle spurious reasoning shortcuts or memorization, opening new directions for both improving model capabilities and testing theories of human cognition at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.01082v8">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Oral presentation at ICLR 2025. Camera-ready version available at https://iclr.cc/virtual/2025/poster/30358
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its considerable impact on improving text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06017v2">AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 AAAI-2026
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated strong capabilities across diverse domains, yet automated agent design remains a significant challenge. Current automated agent design approaches are often constrained by limited search spaces that primarily optimize workflows but fail to integrate crucial human-designed components like memory, planning, and tool use. Furthermore, these methods are hampered by high evaluation costs, as evaluating even a single new agent on a benchmark can require tens of dollars. The difficulty of this exploration is further exacerbated by inefficient search strategies that struggle to navigate the large design space effectively, making the discovery of novel agents a slow and resource-intensive process. To address these challenges, we propose AgentSwift, a novel framework for automated agent design. We formalize a hierarchical search space that jointly models agentic workflow and composable functional components. This structure moves beyond optimizing workflows alone by co-optimizing functional components, which enables the discovery of more complex and effective agent architectures. To make exploration within this expansive space feasible, we mitigate high evaluation costs by training a value model on a high-quality dataset, generated via a novel strategy combining combinatorial coverage and balanced Bayesian sampling for low-cost evaluation. Guiding the entire process is a hierarchical MCTS strategy, which is informed by uncertainty to efficiently navigate the search space. Evaluated across a comprehensive set of seven benchmarks spanning embodied, math, web, tool, and game domains, AgentSwift discovers agents that achieve an average performance gain of 8.34\% over both existing automated agent search methods and manually designed agents. Our framework serves as a launchpad for researchers to rapidly discover powerful agent architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04420v5">KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted by ICML25. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we theoretically analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is generally more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 21.25\% compared with KIVI-KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16450v1">Optimizing Federated Learning in the Era of LLMs: Message Quantization and Streaming</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 FLLM 2025
    </div>
    <details class="paper-abstract">
      Federated Learning (FL) offers a promising solution for training machine learning models across distributed data sources while preserving data privacy. However, FL faces critical challenges related to communication overhead and local resource constraints, especially in the era of Large Language Models (LLMs) with billions of parameters. The sheer size of these models exacerbates both memory and communication constraints, making efficient transmission and processing essential for practical deployment. NVIDIA FLARE, an open-source SDK for federated learning, addresses these challenges by introducing advanced communication capabilities. Building upon existing solutions for large object streaming, we enhance FL workflows for LLMs through two key techniques: message quantization and container/file streaming. Quantization reduces message size, while streaming enables efficient memory management, improving scalability and integration with existing workflows. These advancements significantly enhance the robustness and efficiency of FL with LLMs, ensuring better performance in real-world federated learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16414v1">An Efficient LLM-based Evolutional Recommendation with Locate-Forget-Update Paradigm</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Nowadays, Large Language Models (LLMs) have shown exceptional performance in sequential recommendations, and the adoption of LLM-based recommender systems (LLMRec) is becoming increasingly widespread in existing e-commerce platforms. Despite the impressive performance, the constant high volume of new user-item interactions makes it difficult to adapt to the evolution of user preference over time, especially for LLM-based recommender systems. The challenge arises from the large number of parameters in LLMs, which makes traditional evolution methods (i.e., Re-training or Fine-tuning) impractical. Specifically, Re-training with all interactions results in prohibitively high computational costs. On the other hand, fine-tuning with only new interactions leads to preference forgetting among inactive users, ultimately compromising overall performance. To tackle this problem, we propose EvoRec, an efficient Locate-Forget-Update framework designed for LLM-based recommender systems to model the evolution of user preferences. EvoRec identifies a small set of parameters associated with preference changes and updates them precisely, thereby saving computational resources while maintaining strong recommendation performance. Notably, the modified parameters account for only 30\% of LoRA adapter parameters, with no additional parameters introduced. Extensive experiments on two real-world datasets demonstrate that, compared to existing methods, EvoRec not only efficiently evolves LLMRec to adapt to the preferences of active users, but also preserves the interests of inactive users from being disturbed during evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16395v1">CorrectHDL: Agentic HDL Design with LLMs Leveraging High-Level Synthesis as Reference</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 7 pages, 15 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in hardware front-end design using hardware description languages (HDLs). However, their inherent tendency toward hallucination often introduces functional errors into the generated HDL designs. To address this issue, we propose the framework CorrectHDL that leverages high-level synthesis (HLS) results as functional references to correct potential errors in LLM-generated HDL designs.The input to the proposed framework is a C/C++ program that specifies the target circuit's functionality. The program is provided to an LLM to directly generate an HDL design, whose syntax errors are repaired using a Retrieval-Augmented Generation (RAG) mechanism. The functional correctness of the LLM-generated circuit is iteratively improved by comparing its simulated behavior with an HLS reference design produced by conventional HLS tools, which ensures the functional correctness of the result but can lead to suboptimal area and power efficiency. Experimental results demonstrate that circuits generated by the proposed framework achieve significantly better area and power efficiency than conventional HLS designs and approach the quality of human-engineered circuits. Meanwhile, the correctness of the resulting HDL implementation is maintained, highlighting the effectiveness and potential of agentic HDL design leveraging the generative capabilities of LLMs and the rigor of traditional correctness-driven IC design flows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16324v1">SDA: Steering-Driven Distribution Alignment for Open LLMs without Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), their deployment in real-world applications has become increasingly widespread. LLMs are expected to deliver robust performance across diverse tasks, user preferences, and practical scenarios. However, as demands grow, ensuring that LLMs produce responses aligned with human intent remains a foundational challenge. In particular, aligning model behavior effectively and efficiently during inference, without costly retraining or extensive supervision, is both a critical requirement and a non-trivial technical endeavor. To address the challenge, we propose SDA (Steering-Driven Distribution Alignment), a training-free and model-agnostic alignment framework designed for open-source LLMs. SDA dynamically redistributes model output probabilities based on user-defined alignment instructions, enhancing alignment between model behavior and human intents without fine-tuning. The method is lightweight, resource-efficient, and compatible with a wide range of open-source LLMs. It can function independently during inference or be integrated with training-based alignment strategies. Moreover, SDA supports personalized preference alignment, enabling flexible control over the model response behavior. Empirical results demonstrate that SDA consistently improves alignment performance across 8 open-source LLMs with varying scales and diverse origins, evaluated on three key alignment dimensions, helpfulness, harmlessness, and honesty (3H). Specifically, SDA achieves average gains of 64.4% in helpfulness, 30% in honesty and 11.5% in harmlessness across the tested models, indicating its effectiveness and generalization across diverse models and application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25939v3">SoK: Honeypots & LLMs, More Than the Sum of Their Parts?</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Systemization of Knowledge
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design, achieving high-fidelity deception with low operational risk. Through a flurry of research since late 2022, steady progress from ideation to prototype implementation is exhibited. Since late 2022, a flurry of research has demonstrated steady progress from ideation to prototype implementation. While promising, evaluations show only incremental progress in real-world deployments, and the field still lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview and analysis of this new domain. We survey and systematize the field by focusing on three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-powered honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16278v1">"To Survive, I Must Defect": Jailbreaking LLMs via the Game-Theory Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      As LLMs become more common, non-expert users can pose risks, prompting extensive research into jailbreak attacks. However, most existing black-box jailbreak attacks rely on hand-crafted heuristics or narrow search spaces, which limit scalability. Compared with prior attacks, we propose Game-Theory Attack (GTA), an scalable black-box jailbreak framework. Concretely, we formalize the attacker's interaction against safety-aligned LLMs as a finite-horizon, early-stoppable sequential stochastic game, and reparameterize the LLM's randomized outputs via quantal response. Building on this, we introduce a behavioral conjecture "template-over-safety flip": by reshaping the LLM's effective objective through game-theoretic scenarios, the originally safety preference may become maximizing scenario payoffs within the template, which weakens safety constraints in specific contexts. We validate this mechanism with classical game such as the disclosure variant of the Prisoner's Dilemma, and we further introduce an Attacker Agent that adaptively escalates pressure to increase the ASR. Experiments across multiple protocols and datasets show that GTA achieves over 95% ASR on LLMs such as Deepseek-R1, while maintaining efficiency. Ablations over components, decoding, multilingual settings, and the Agent's core model confirm effectiveness and generalization. Moreover, scenario scaling studies further establish scalability. GTA also attains high ASR on other game-theoretic scenarios, and one-shot LLM-generated variants that keep the model mechanism fixed while varying background achieve comparable ASR. Paired with a Harmful-Words Detection Agent that performs word-level insertions, GTA maintains high ASR while lowering detection under prompt-guard models. Beyond benchmarks, GTA jailbreaks real-world LLM applications and reports a longitudinal safety monitoring of popular HuggingFace LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16275v1">SeSE: A Structural Information-Guided Uncertainty Quantification Framework for Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 14 pages of main text and 10 pages of appendices
    </div>
    <details class="paper-abstract">
      Reliable uncertainty quantification (UQ) is essential for deploying large language models (LLMs) in safety-critical scenarios, as it enables them to abstain from responding when uncertain, thereby avoiding hallucinating falsehoods. However, state-of-the-art UQ methods primarily rely on semantic probability distributions or pairwise distances, overlooking latent semantic structural information that could enable more precise uncertainty estimates. This paper presents Semantic Structural Entropy (SeSE), a principled UQ framework that quantifies the inherent semantic uncertainty of LLMs from a structural information perspective for hallucination detection. Specifically, to effectively model semantic spaces, we first develop an adaptively sparsified directed semantic graph construction algorithm that captures directional semantic dependencies while automatically pruning unnecessary connections that introduce negative interference. We then exploit latent semantic structural information through hierarchical abstraction: SeSE is defined as the structural entropy of the optimal semantic encoding tree, formalizing intrinsic uncertainty within semantic spaces after optimal compression. A higher SeSE value corresponds to greater uncertainty, indicating that LLMs are highly likely to generate hallucinations. In addition, to enhance fine-grained UQ in long-form generation -- where existing methods often rely on heuristic sample-and-count techniques -- we extend SeSE to quantify the uncertainty of individual claims by modeling their random semantic interactions, providing theoretically explicable hallucination detection. Extensive experiments across 29 model-dataset combinations show that SeSE significantly outperforms advanced UQ baselines, including strong supervised methods and the recently proposed KLE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08916v3">HalluClean: A Unified Framework to Combat Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16224v1">Beyond Code Similarity: Benchmarking the Plausibility, Efficiency, and Complexity of LLM-Generated Smart Contracts</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Smart Contracts are critical components of blockchain ecosystems, with Solidity as the dominant programming language. While LLMs excel at general-purpose code generation, the unique constraints of Smart Contracts, such as gas consumption, security, and determinism, raise open questions about the reliability of LLM-generated Solidity code. Existing studies lack a comprehensive evaluation of these critical functional and non-functional properties. We benchmark four state-of-the-art models under zero-shot and retrieval-augmented generation settings across 500 real-world functions. Our multi-faceted assessment employs code similarity metrics, semantic embeddings, automated test execution, gas profiling, and cognitive and cyclomatic complexity analysis. Results show that while LLMs produce code with high semantic similarity to real contracts, their functional correctness is low: only 20% to 26% of zero-shot generations behave identically to ground-truth implementations under testing. The generated code is consistently simpler, with significantly lower complexity and gas consumption, often due to omitted validation logic. Retrieval-Augmented Generation markedly improves performance, boosting functional correctness by up to 45% and yielding more concise and efficient code. Our findings reveal a significant gap between semantic similarity and functional plausibility in LLM-generated Smart Contracts. We conclude that while RAG is a powerful enhancer, achieving robust, production-ready code generation remains a substantial challenge, necessitating careful expert validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16209v1">PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16267v3">From Confidence to Collapse in LLM Factual Robustness</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.05310v3">Oracular Programming: A Modular Foundation for Building LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large Language Models can solve a wide range of tasks from just a few examples, but they remain difficult to steer and lack a capability essential for building reliable software at scale: the modular composition of computations under enforceable contracts. As a result, they are typically embedded in larger software pipelines that use domain-specific knowledge to decompose tasks and improve reliability through validation and search. Yet the complexity of writing, tuning, and maintaining such pipelines has so far limited their sophistication. We propose oracular programming: a foundational paradigm for integrating traditional, explicit computations with inductive oracles such as LLMs. It rests on two directing principles: the full separation of core and search logic, and the treatment of few-shot examples as grounded and evolvable program components. Within this paradigm, experts express high-level problem-solving strategies as programs with unresolved choice points. These choice points are resolved at runtime by LLMs, which generalize from user-provided examples of correct and incorrect decisions. An oracular program is composed of three orthogonal components: a strategy that consists in a nondeterministic program with choice points that can be reified into a search tree, a policy that specifies how to navigate this tree with the help of LLM oracles, and a set of demonstrations that describe successful and unsuccessful tree navigation scenarios across diverse problem instances. Each component is expressed in a dedicated programming language and can be independently improved or substituted. We address the key programming language design challenges of modularly composing oracular programs and enforcing consistency between their components as they evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09837v2">MoFa: A Unified Performance Modeling Framework for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      The exponential growth in LLM scales, with parameters soaring from billions to trillions, has necessitated distributed pretraining across large clusters comprising thousands to tens of thousands of devices. While hybrid parallelization strategies enable such pretraining, the vast combinatorial strategy space introduces significant optimization challenges. Traditional manual tuning methods incur prohibitive trial-and-error costs, and existing performance modeling approaches exhibit critical limitations: they fail to comprehensively account for prevalent optimization features and ignore the substantial overhead imposed by essential fault tolerance mechanisms like checkpoint recovery in long-duration pretraining. To address these gaps, we propose MoFa, a novel pretraining performance modeling framework that unifies multi-dimensional optimization features and fault tolerance. MoFa incorporates an enhanced cost model to accurately capture the effects of key optimizations and integrates a fault tolerance model based on historical cluster reliability data. Besides, a MoFa-based tuning system is developed to explore optimal pretraining performance and potential bottlenecks in various scenarios. Extensive modeling evaluations demonstrate that MoFa can achieve high prediction accuracy across various scenarios. In addition, through comprehensive tuning experiments, our framework systematically reveals the key factors influencing pretraining performance under different configurations, which provides solid a priori guidance for LLM pretraining system design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05919v2">Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15192v2">As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds. We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16193v1">Fast LLM Post-training via Decoupled and Best-of-N Speculation</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Rollout dominates the training time in large language model (LLM) post-training, where the trained model is used to generate tokens given a batch of prompts. SpecActor achieves fast rollout with speculative decoding that deploys a fast path (e.g., a smaller model) to accelerate the unparallelizable generation, while the correctness is guaranteed by fast parallel verification of the outputs with the original model. SpecActor addresses two foundational challenges in speculative rollout by (1) a \emph{dynamic decoupled speculation} execution method that maximizes the GPU computational efficiency to realize speedup for large-batch execution -- a configuration common in training but unfriendly to speculative execution and (2) a \emph{dynamic Best-of-N speculation} method that selects and combines different drafting methods according to the rollout progress. It substantially improves the speculation accuracy even when the best drafting method is unknown a priori, meanwhile without requiring adding extra computation resources. {\sys} is {1.3--1.7}\,$\times$ faster than common post-training baselines, and is {1.3--1.5}\,$\times$ faster compared to naively adopting speculative decoding for rollout.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03628v5">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      E-commerce sellers are advised to bid on keyphrases to boost their advertising campaigns. These keyphrases must be relevant to prevent irrelevant items from cluttering search systems and to maintain positive seller perception. It is vital that keyphrase suggestions align with seller, search and buyer judgments. Given the challenges in collecting negative feedback in these systems, LLMs have been used as a scalable proxy to human judgments. This paper presents an empirical study on a major ecommerce platform of a distillation framework involving an LLM teacher, a cross-encoder assistant and a bi-encoder Embedding Based Retrieval (EBR) student model, aimed at mitigating click-induced biases in keyphrase recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10712v2">Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted by AAAI 2026 Conference
    </div>
    <details class="paper-abstract">
      Model merging has emerged as an efficient technique for expanding large language models (LLMs) by integrating specialized expert models. However, it also introduces a new threat: model merging stealing, where free-riders exploit models through unauthorized model merging. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify three critical protection properties that existing methods fail to simultaneously satisfy: (1) proactively preventing unauthorized merging; (2) ensuring compatibility with general open-source settings; (3) achieving high security with negligible performance loss. To address the above issues, we propose MergeBarrier, a plug-and-play defense that proactively prevents unauthorized merging. The core design of MergeBarrier is to disrupt the Linear Mode Connectivity (LMC) between the protected model and its homologous counterparts, thereby eliminating the low-loss path required for effective model merging. Extensive experiments show that MergeBarrier effectively prevents model merging stealing with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.13246v3">Atomic Calibration of LLMs in Long-Form Generations</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 ACL 2025 KnowFM Oral / AACL-IJCNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from hallucinations, posing significant challenges for real-world applications. Confidence calibration, as an effective indicator of hallucination, is thus essential to enhance the trustworthiness of LLMs. Prior work mainly focuses on short-form tasks using a single response-level score (macro calibration), which is insufficient for long-form outputs that may contain both accurate and inaccurate claims. In this work, we systematically study atomic calibration, which evaluates factuality calibration at a fine-grained level by decomposing long responses into atomic claims. We further categorize existing confidence elicitation methods into discriminative and generative types, and propose two new confidence fusion strategies to improve calibration. Our experiments demonstrate that LLMs exhibit poorer calibration at the atomic level during long-form generation. More importantly, atomic calibration uncovers insightful patterns regarding the alignment of confidence methods and the changes of confidence throughout generation. This sheds light on future research directions for confidence estimation in long-form generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04250v2">How many patients could we save with LLM priors?</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Imagine a world where clinical trials need far fewer patients to achieve the same statistical power, thanks to the knowledge encoded in large language models (LLMs). We present a novel framework for hierarchical Bayesian modeling of adverse events in multi-center clinical trials, leveraging LLM-informed prior distributions. Unlike data augmentation approaches that generate synthetic data points, our methodology directly obtains parametric priors from the model. Our approach systematically elicits informative priors for hyperparameters in hierarchical Bayesian models using a pre-trained LLM, enabling the incorporation of external clinical expertise directly into Bayesian safety modeling. Through comprehensive temperature sensitivity analysis and rigorous cross-validation on real-world clinical trial data, we demonstrate that LLM-derived priors consistently improve predictive performance compared to traditional meta-analytical approaches. This methodology paves the way for more efficient and expert-informed clinical trial design, enabling substantial reductions in the number of patients required to achieve robust safety assessment and with the potential to transform drug safety monitoring and regulatory decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15168v2">Finetuning LLMs for Automatic Form Interaction on Web-Browser in Selenium Testing Framework</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Published in the Proceedings of KSE 2025
    </div>
    <details class="paper-abstract">
      Automated web application testing is a critical component of modern software development, with frameworks like Selenium widely adopted for validating functionality through browser automation. Among the essential aspects of such testing is the ability to interact with and validate web forms, a task that requires syntactically correct, executable scripts with high coverage of input fields. Despite its importance, this task remains underexplored in the context of large language models (LLMs), and no public benchmark or dataset exists to evaluate LLMs on form interaction generation systematically. This paper introduces a novel method for training LLMs to generate high-quality test cases in Selenium, specifically targeting form interaction testing. We curate both synthetic and human-annotated datasets for training and evaluation, covering diverse real-world forms and testing scenarios. We define clear metrics for syntax correctness, script executability, and input field coverage. Our empirical study demonstrates that our approach significantly outperforms strong baselines, including GPT-4o and other popular LLMs, across all evaluation metrics. Our work lays the groundwork for future research on LLM-based web testing and provides resources to support ongoing progress in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15258v2">Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 14 pages, 7 figures, 40 references
    </div>
    <details class="paper-abstract">
      In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14765v2">PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21830v2">GAPO: Robust Advantage Estimation for Real-World Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is widely used for post-training large language models (LLMs) in code editing, where group-relative methods like GRPO are popular for their critic-free, normalized advantage estimation. However, in real-world code-editing scenarios, reward distributions are often skewed with unpredictable outliers, leading to distorted advantage computation and increased noise. To address this issue, we propose Group Adaptive Policy Optimization (GAPO), which adaptively finds an outlier-free highest-density interval (HDI) per prompt and then uses the median of that interval as an adaptive Q to replace the group mean in advantage calculation. This adaptive Q robustly handles skewed distributions while remaining plug-and-play and efficient. We validate GAPO on nine instruction-tuned LLMs (3B-14B) using a large internal dataset of 51,844 real-world, history-aware code-editing tasks across 10 languages, demonstrating consistent improvements in exact match accuracy over GRPO and its variant DAPO. Code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04559v2">One Model for All: Unified Try-On and Try-Off in Any Pose via LLM-Inspired Bidirectional Tweedie Diffusion</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Recent diffusion-based approaches have made significant advances in image-based virtual try-on, enabling more realistic and end-to-end garment synthesis. However, most existing methods remain constrained by their reliance on exhibition garments and segmentation masks, as well as their limited ability to handle flexible pose variations. These limitations reduce their practicality in real-world scenarios - for instance, users cannot easily transfer garments worn by one person onto another, and the generated try-on results are typically restricted to the same pose as the reference image. In this paper, we introduce OMFA (One Model For All), a unified diffusion framework for both virtual try-on and try-off that operates without the need for exhibition garments and supports arbitrary poses. OMFA is inspired by language modeling, where generation is guided by conditioning prompts. However, our framework differs fundamentally from LLMs in two key aspects. First, it employs a bidirectional modeling paradigm that symmetrically allows prompting either from the garment to generate try-on results or from the dressed person to recover the try-off garment. Second, it strictly adheres to Tweedie's formula, enabling faithful estimation of the underlying data distribution during the denoising process. Instead of imposing lower body constraints, OMFA is an entirely mask-free framework that requires only a single portrait and a target garment as input, and is designed to support flexible outfit combinations and cross-person garment transfer, making it better aligned with practical usage scenarios. Additionally, by leveraging SMPL-X-based pose conditioning, OMFA supports multi-view and arbitrary-pose try-on from just one image. Extensive experiments demonstrate that OMFA achieves state-of-the-art results on both try-on and try-off tasks, providing a practical solution for virtual garment synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16037v1">LLMs-based Augmentation for Domain Adaptation in Long-tailed Food Datasets</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Training a model for food recognition is challenging because the training samples, which are typically crawled from the Internet, are visually different from the pictures captured by users in the free-living environment. In addition to this domain-shift problem, the real-world food datasets tend to be long-tailed distributed and some dishes of different categories exhibit subtle variations that are difficult to distinguish visually. In this paper, we present a framework empowered with large language models (LLMs) to address these challenges in food recognition. We first leverage LLMs to parse food images to generate food titles and ingredients. Then, we project the generated texts and food images from different domains to a shared embedding space to maximize the pair similarities. Finally, we take the aligned features of both modalities for recognition. With this simple framework, we show that our proposed approach can outperform the existing approaches tailored for long-tailed data distribution, domain adaptation, and fine-grained classification, respectively, on two food datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.13803v3">LLMs as Models for Analogical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 The title has been changed from Semantic Structure-Mapping in LLM and Human Analogical Reasoning to LLMs as Models for Analogical Reasoning to improve clarity and accuracy
    </div>
    <details class="paper-abstract">
      Analogical reasoning -- the capacity to identify and map structural relationships between different domains -- is fundamental to human cognition and learning. Recent studies have shown that large language models (LLMs) can sometimes match humans in analogical reasoning tasks, opening the possibility that analogical reasoning might emerge from domain-general processes. However, it is still debated whether these emergent capacities are largely superficial and limited to simple relations seen during training or whether they encompass the flexible representational and mapping capabilities which are the focus of leading cognitive models of analogy. In this study, we introduce novel analogical reasoning tasks that require participants to map between semantically contentful words and sequences of letters and other abstract characters. This task necessitates the ability to flexibly re-represent rich semantic information -- an ability which is known to be central to human analogy but which is thus far not well captured by existing cognitive theories and models. We assess the performance of both human participants and LLMs on tasks focusing on reasoning from semantic structure and semantic content, introducing variations that test the robustness of their analogical inferences. Advanced LLMs match human performance across several conditions, though humans and LLMs respond differently to certain task variations and semantic distractors. Our results thus provide new evidence that LLMs might offer a how-possibly explanation of human analogical reasoning in contexts that are not yet well modeled by existing theories, but that even today's best models are unlikely to yield how-actually explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16016v1">CARE: Turning LLMs Into Causal Reasoning Expert</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated impressive capabilities across a range of reasoning and generation tasks. However, research studies have shown that LLMs lack the ability to identify causal relationships, a fundamental cornerstone of human intelligence. We first conduct an exploratory investigation of LLMs' behavior when asked to perform a causal-discovery task and find that they mostly rely on the semantic meaning of variable names, ignoring the observation data. This is unsurprising, given that LLMs were never trained to process structural datasets. To first tackle this challenge, we prompt the LLMs with the outputs of established causal discovery algorithms designed for observational datasets. These algorithm outputs effectively serve as the sufficient statistics of the observation data. However, quite surprisingly, we find that prompting the LLMs with these sufficient statistics decreases the LLMs' performance in causal discovery. To address this current limitation, we propose CARE, a framework that enhances LLMs' causal-reasoning ability by teaching them to effectively utilize the outputs of established causal-discovery algorithms through supervised fine-tuning. Experimental results show that a finetuned Qwen2.5-1.5B model produced by CARE significantly outperforms both traditional causal-discovery algorithms and state-of-the-art LLMs with over a thousand times more parameters, demonstrating effective utilization of its own knowledge and the external algorithmic clues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15998v1">Hiding in the AI Traffic: Abusing MCP for LLM-Powered Agentic Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 23 pages, 9 figures, 3 tables. Submitted as a full paper for review
    </div>
    <details class="paper-abstract">
      Generative AI is reshaping offensive cybersecurity by enabling autonomous red team agents that can plan, execute, and adapt during penetration tests. However, existing approaches face trade-offs between generality and specialization, and practical deployments reveal challenges such as hallucinations, context limitations, and ethical concerns. In this work, we introduce a novel command & control (C2) architecture leveraging the Model Context Protocol (MCP) to coordinate distributed, adaptive reconnaissance agents covertly across networks. Notably, we find that our architecture not only improves goal-directed behavior of the system as whole, but also eliminates key host and network artifacts that can be used to detect and prevent command & control behavior altogether. We begin with a comprehensive review of state-of-the-art generative red teaming methods, from fine-tuned specialist models to modular or agentic frameworks, analyzing their automation capabilities against task-specific accuracy. We then detail how our MCP-based C2 can overcome current limitations by enabling asynchronous, parallel operations and real-time intelligence sharing without periodic beaconing. We furthermore explore advanced adversarial capabilities of this architecture, its detection-evasion techniques, and address dual-use ethical implications, proposing defensive measures and controlled evaluation in lab settings. Experimental comparisons with traditional C2 show drastic reductions in manual effort and detection footprint. We conclude with future directions for integrating autonomous exploitation, defensive LLM agents, predictive evasive maneuvers, and multi-agent swarms. The proposed MCP-enabled C2 framework demonstrates a significant step toward realistic, AI-driven red team operations that can simulate advanced persistent threats while informing the development of next-generation defensive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15996v1">QueryGym: A Toolkit for Reproducible LLM-Based Query Reformulation</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      We present QueryGym, a lightweight, extensible Python toolkit that supports large language model (LLM)-based query reformulation. This is an important tool development since recent work on llm-based query reformulation has shown notable increase in retrieval effectiveness. However, while different authors have sporadically shared the implementation of their methods, there is no unified toolkit that provides a consistent implementation of such methods, which hinders fair comparison, rapid experimentation, consistent benchmarking and reliable deployment. QueryGym addresses this gap by providing a unified framework for implementing, executing, and comparing llm-based reformulation methods. The toolkit offers: (1) a Python API for applying diverse LLM-based methods, (2) a retrieval-agnostic interface supporting integration with backends such as Pyserini and PyTerrier, (3) a centralized prompt management system with versioning and metadata tracking, (4) built-in support for benchmarks like BEIR and MS MARCO, and (5) a completely open-source extensible implementation available to all researchers. QueryGym is publicly available at https://github.com/radinhamidi/QueryGym.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16785v3">Interpreting the Effects of Quantization on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted to AACL 2025 Main
    </div>
    <details class="paper-abstract">
      Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15974v1">KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles, host factors, pharmacological properties of antimicrobials, and the severity of infection.This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at ~20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.17435v5">Can LLMs Replace Economic Choice Prediction Labs? The Case of Language-based Persuasion Games</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Human choice prediction in economic contexts is crucial for applications in marketing, finance, public policy, and more. This task, however, is often constrained by the difficulties in acquiring human choice data. With most experimental economics studies focusing on simple choice settings, the AI community has explored whether LLMs can substitute for humans in these predictions and examined more complex experimental economics settings. However, a key question remains: can LLMs generate training data for human choice prediction? We explore this in language-based persuasion games, a complex economic setting involving natural language in strategic interactions. Our experiments show that models trained on LLM-generated data can effectively predict human behavior in these games and even outperform models trained on actual human data. Beyond data generation, we investigate the dual role of LLMs as both data generators and predictors, introducing a comprehensive empirical study on the effectiveness of utilizing LLMs for data generation, human choice prediction, or both. We then utilize our choice prediction framework to analyze how strategic factors shape decision-making, showing that interaction history (rather than linguistic sentiment alone) plays a key role in predicting human decision-making in repeated interactions. Particularly, when LLMs capture history-dependent decision patterns similarly to humans, their predictive success improves substantially. Finally, we demonstrate the robustness of our findings across alternative persuasion-game settings, highlighting the broader potential of using LLM-generated data to model human decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.08546v3">Grounding LLMs For Robot Task Planning Using Closed-loop State Feedback</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Preprint version. Accepted full paper available here: https://advanced.onlinelibrary.wiley.com/doi/10.1002/adrr.202500072
    </div>
    <details class="paper-abstract">
      Planning algorithms decompose complex problems into intermediate steps that can be sequentially executed by robots to complete tasks. Recent works have employed Large Language Models (LLMs) for task planning, using natural language to generate robot policies in both simulation and real-world environments. LLMs like GPT-4 have shown promising results in generalizing to unseen tasks, but their applicability is limited due to hallucinations caused by insufficient grounding in the robot environment. The robustness of LLMs in task planning can be enhanced with environmental state information and feedback. In this paper, we introduce a novel approach to task planning that utilizes two separate LLMs for high-level planning and low-level control, improving task-related success rates and goal condition recall. Our algorithm, \textit{BrainBody-LLM}, draws inspiration from the human neural system, emulating its brain-body architecture by dividing planning across two LLMs in a structured, hierarchical manner. BrainBody-LLM implements a closed-loop feedback mechanism, enabling learning from simulator errors to resolve execution errors in complex settings. We demonstrate the successful application of BrainBody-LLM in the VirtualHome simulation environment, achieving a 29\% improvement in task-oriented success rates over competitive baselines with the GPT-4 backend. Additionally, we evaluate our algorithm on seven complex tasks using a realistic physics simulator and the Franka Research 3 robotic arm, comparing it with various state-of-the-art LLMs. Our results show advancements in the reasoning capabilities of recent LLMs, which enable them to learn from raw simulator/controller errors to correct plans, making them highly effective in robotic task planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16846v1">ConCISE: A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate responses that are lengthy and verbose, filled with redundant or unnecessary details. This diminishes clarity and user satisfaction, and it increases costs for model developers, especially with well-known proprietary models that charge based on the number of output tokens. In this paper, we introduce a novel reference-free metric for evaluating the conciseness of responses generated by LLMs. Our method quantifies non-essential content without relying on gold standard references and calculates the average of three calculations: i) a compression ratio between the original response and an LLM abstractive summary; ii) a compression ratio between the original response and an LLM extractive summary; and iii) wordremoval compression, where an LLM removes as many non-essential words as possible from the response while preserving its meaning, with the number of tokens removed indicating the conciseness score. Experimental results demonstrate that our proposed metric identifies redundancy in LLM outputs, offering a practical tool for automated evaluation of response brevity in conversational AI systems without the need for ground truth human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16837v1">Cognitive BASIC: An In-Model Interpreted Reasoning Language for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 6 pages, Submitted to ESANN 2026
    </div>
    <details class="paper-abstract">
      Cognitive BASIC is a minimal, BASIC-style prompting language and in-model interpreter that structures large language model (LLM) reasoning into explicit, stepwise execution traces. Inspired by the simplicity of retro BASIC, we repurpose numbered lines and simple commands as an interpretable cognitive control layer. Modern LLMs can reliably simulate such short programs, enabling transparent multi-step reasoning inside the model. A natural-language interpreter file specifies command semantics, memory updates, and logging behavior. Our mental-model interpreter extracts declarative and procedural knowledge, detects contradictions, and produces resolutions when necessary. A comparison across three LLMs on a benchmark of knowledge extraction, conflict detection, and reasoning tasks shows that all models can execute Cognitive BASIC programs, with overall strong but not uniform performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16549v2">ReviewGuard: Enhancing Deficient Peer Review Detection via LLM-Driven Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 Accepted as a full paper at the 2025 ACM/IEEE Joint Conference on Digital Libraries (JCDL 2025)
    </div>
    <details class="paper-abstract">
      Peer review serves as the gatekeeper of science, yet the surge in submissions and widespread adoption of large language models (LLMs) in scholarly evaluation present unprecedented challenges. While recent work has focused on using LLMs to improve review efficiency, unchecked deficient reviews from both human experts and AI systems threaten to systematically undermine academic integrity. To address this issue, we introduce ReviewGuard, an automated system for detecting and categorizing deficient reviews through a four-stage LLM-driven framework: data collection from ICLR and NeurIPS on OpenReview, GPT-4.1 annotation with human validation, synthetic data augmentation yielding 6,634 papers with 24,657 real and 46,438 synthetic reviews, and fine-tuning of encoder-based models and open-source LLMs. Feature analysis reveals that deficient reviews exhibit lower rating scores, higher self-reported confidence, reduced structural complexity, and more negative sentiment than sufficient reviews. AI-generated text detection shows dramatic increases in AI-authored reviews since ChatGPT's emergence. Mixed training with synthetic and real data substantially improves detection performance - for example, Qwen 3-8B achieves recall of 0.6653 and F1 of 0.7073, up from 0.5499 and 0.5606 respectively. This study presents the first LLM-driven system for detecting deficient peer reviews, providing evidence to inform AI governance in peer review. Code, prompts, and data are available at https://github.com/haoxuan-unt2024/ReviewGuard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24068v2">A Small Math Model: Recasting Strategy Choice Theory in an LLM-Inspired Architecture</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Strategy Choice Theory (SCT; Siegler and Shrager, 1984; Siegler, 2000) explains important aspects of children's arithmetic learning based upon principles including learning from developmentally naturalistic data, probabilistic representation, confidence-based retrieval, and the phase-like importance of scaffolding strategies, such as finger-counting. Here we recast SCT as a ``Small Math Model'' (SMM), employing a neural-network-based architecture analogous to LLMs. The SMM extends SCT to include counting practice, symbol (number) embedding, and gated attention. Similar to earlier work, the SMM demonstrates constructive and destructive interference between counting and addition, and the ``wave-like'' use of finger-counting as sum recall improves. We plan to extend the SMM to later aspects of the decades-long SCT program, including adaptive strategy choice and eventually strategy discovery, providing a unified platform to investigate the understanding of numerical characteristics and relationships essential for mathematical reasoning -- as it can emerge in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.13290v2">Towards Formal Verification of LLM-Generated Code from Natural Language Prompts</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 28 pages, 10 figures
    </div>
    <details class="paper-abstract">
      In the past few years LLMs have emerged as a tool that can aid programmers by taking natural language descriptions and generating code based on it. However, the reliability of LLM code generation and current validation techniques for it are far from strong enough to be used for mission-critical or safety-critical applications. In this work we explore ways to offer formal guarantees of correctness to LLM generated code; such guarantees could improve the quality of general AI Code Assistants and support their use for critical applications. To address this challenge we propose to incorporate a Formal Query Language that can represent a user's intent in a formally defined but natural language-like manner that a user can confirm matches their intent. We then have a formal specification of the user intent which we can use to verify that LLM-generated code matches the user's intent. We implement these ideas in our system, Astrogator, for the Ansible programming language, widely used for system administration, including for critical systems. The system includes an intuitive formal query language, a calculus for representing the behavior of Ansible programs, and a symbolic interpreter and a unification algorithm which together are used for the verification. A key innovation in Astrogator is the use of a Knowledge Base to capture system-specific implementation dependencies that greatly reduce the need for system knowledge in expressing formal queries. On a benchmark suite of 21 code-generation tasks, our verifier is able to verify correct code in 83% of cases and identify incorrect code in 92%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16767v1">When Structure Doesn't Help: LLMs Do Not Read Text-Attributed Graphs as Effectively as We Expected</a></div>
    <div class="paper-meta">
      📅 2025-11-20
    </div>
    <details class="paper-abstract">
      Graphs provide a unified representation of semantic content and relational structure, making them a natural fit for domains such as molecular modeling, citation networks, and social graphs. Meanwhile, large language models (LLMs) have excelled at understanding natural language and integrating cross-modal signals, sparking interest in their potential for graph reasoning. Recent work has explored this by either designing template-based graph templates or using graph neural networks (GNNs) to encode structural information. In this study, we investigate how different strategies for encoding graph structure affect LLM performance on text-attributed graphs. Surprisingly, our systematic experiments reveal that: (i) LLMs leveraging only node textual descriptions already achieve strong performance across tasks; and (ii) most structural encoding strategies offer marginal or even negative gains. We show that explicit structural priors are often unnecessary and, in some cases, counterproductive when powerful language models are involved. This represents a significant departure from traditional graph learning paradigms and highlights the need to rethink how structure should be represented and utilized in the LLM era. Our study is to systematically challenge the foundational assumption that structure is inherently beneficial for LLM-based graph reasoning, opening the door to new, semantics-driven approaches for graph learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.16709v1">AutoBackdoor: Automating Backdoor Attacks via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-20
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Backdoor attacks pose a serious threat to the secure deployment of large language models (LLMs), enabling adversaries to implant hidden behaviors triggered by specific inputs. However, existing methods often rely on manually crafted triggers and static data pipelines, which are rigid, labor-intensive, and inadequate for systematically evaluating modern defense robustness. As AI agents become increasingly capable, there is a growing need for more rigorous, diverse, and scalable \textit{red-teaming frameworks} that can realistically simulate backdoor threats and assess model resilience under adversarial conditions. In this work, we introduce \textsc{AutoBackdoor}, a general framework for automating backdoor injection, encompassing trigger generation, poisoned data construction, and model fine-tuning via an autonomous agent-driven pipeline. Unlike prior approaches, AutoBackdoor uses a powerful language model agent to generate semantically coherent, context-aware trigger phrases, enabling scalable poisoning across arbitrary topics with minimal human effort. We evaluate AutoBackdoor under three realistic threat scenarios, including \textit{Bias Recommendation}, \textit{Hallucination Injection}, and \textit{Peer Review Manipulation}, to simulate a broad range of attacks. Experiments on both open-source and commercial models, including LLaMA-3, Mistral, Qwen, and GPT-4o, demonstrate that our method achieves over 90\% attack success with only a small number of poisoned samples. More importantly, we find that existing defenses often fail to mitigate these attacks, underscoring the need for more rigorous and adaptive evaluation techniques against agent-driven threats as explored in this work. All code, datasets, and experimental configurations will be merged into our primary repository at https://github.com/bboylyg/BackdoorLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15698v1">RescueLens: LLM-Powered Triage and Action on Volunteer Feedback for Food Rescue</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted at IAAI'26
    </div>
    <details class="paper-abstract">
      Food rescue organizations simultaneously tackle food insecurity and waste by working with volunteers to redistribute food from donors who have excess to recipients who need it. Volunteer feedback allows food rescue organizations to identify issues early and ensure volunteer satisfaction. However, food rescue organizations monitor feedback manually, which can be cumbersome and labor-intensive, making it difficult to prioritize which issues are most important. In this work, we investigate how large language models (LLMs) assist food rescue organizers in understanding and taking action based on volunteer experiences. We work with 412 Food Rescue, a large food rescue organization based in Pittsburgh, Pennsylvania, to design RescueLens, an LLM-powered tool that automatically categorizes volunteer feedback, suggests donors and recipients to follow up with, and updates volunteer directions based on feedback. We evaluate the performance of RescueLens on an annotated dataset, and show that it can recover 96% of volunteer issues at 71% precision. Moreover, by ranking donors and recipients according to their rates of volunteer issues, RescueLens allows organizers to focus on 0.5% of donors responsible for more than 30% of volunteer issues. RescueLens is now deployed at 412 Food Rescue and through semi-structured interviews with organizers, we find that RescueLens streamlines the feedback process so organizers better allocate their time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08640v2">Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03628v4">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      E-commerce sellers are advised to bid on keyphrases to boost their advertising campaigns. These keyphrases must be relevant to prevent irrelevant items from cluttering search systems and to maintain positive seller perception. It is vital that keyphrase suggestions align with seller, search and buyer judgments. Given the challenges in collecting negative feedback in these systems, LLMs have been used as a scalable proxy to human judgments. This paper presents an empirical study on a major ecommerce platform of a distillation framework involving an LLM teacher, a cross-encoder assistant and a bi-encoder Embedding Based Retrieval (EBR) student model, aimed at mitigating click-induced biases in keyphrase recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15676v1">DuoZone: A User-Centric, LLM-Guided Mixed-Initiative XR Window Management System</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Mixed reality (XR) environments offer vast spatial possibilities, but current window management systems require users to manually place, resize, and organize multiple applications across large 3D spaces. This creates cognitive and interaction burdens that limit productivity. We introduce DuoZone, a mixed-initiative XR window management system that combines user-defined spatial layouts with LLM-guided automation. DuoZone separates window management into two complementary zones. The Recommendation Zone enables fast setup by providing spatial layout templates and automatically recommending relevant applications based on user tasks and high-level goals expressed through voice or text. The Arrangement Zone supports precise refinement through direct manipulation, allowing users to adjust windows using natural spatial actions such as dragging, resizing, and snapping. Through this dual-zone approach, DuoZone promotes efficient organization while reducing user cognitive load. We conducted a user study comparing DuoZone with a baseline manual XR window manager. Results show that DuoZone improves task completion speed, reduces mental effort, and increases sense of control when working with multiple applications in XR. We discuss design implications for future mixed-initiative systems and outline opportunities for integrating adaptive, goal-aware intelligence into spatial computing workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15665v1">Quantum-Guided Test Case Minimization for LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 This is a preprint version, full paper has been accepted in IEEE CASCON 2025 and will appear on lEEE Xplore
    </div>
    <details class="paper-abstract">
      Precisely controlling Large Language Models (LLMs) to generate efficient and concise code is a central challenge in software engineering. We introduce a framework based on Test-Driven Development (TDD) that transforms code specification into a combinatorial optimization task. The framework first prompts an LLM to generate a test suite, then formulates the Test Case Minimization (TCM) problem as a Quadratic Unconstrained Binary Optimization (QUBO) model. This QUBO paradigm is compatible with both classical solvers and emerging hardware such as quantum annealers. Experimentally, quantum annealing solves the core TCM task 16 times faster than simulated annealing. This performance underpins our end-to-end framework, which reduces total token consumption by 36.5\% and significantly improves code quality. This work demonstrates a powerful synergy between generative AI and combinatorial optimization in software engineering, highlighting the critical importance of precise model formulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14214v2">Do Large Language Models (LLMs) Understand Chronology?</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Version 2: corrected footnote and added code repository link. Extended version of our work presented at the AAAI-26 AI4TS Workshop (poster) and AAAI-26 Student Abstract Program (oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in finance and economics, where prompt-based attempts against look-ahead bias implicitly assume that models understand chronology. We test this fundamental question with a series of chronological ordering tasks with increasing complexities over facts the model already knows from pre-training. Our tasks cover (1) chronological ordering, (2) conditional sorting (filter, then order), and (3) anachronism detection. We evaluate GPT-4.1, Claude-3.7 Sonnet, with and without Extended Thinking (ET), and GPT-5 across multiple reasoning-effort settings. Across models, Exact match rate drops sharply as sequences lengthen even while rank correlations stay high as LLMs largely preserve local order but struggle to maintain a single globally consistent timeline. In conditional sorting, most failures stem from the filtering step rather than the ordering step, but GPT-5 and Claude-3.7 Sonnet with Extended Thinking outshine normal models significantly. Lastly, anachronism detection is found to be the easiest task for the LLMs but performance still declines with increasingly overlapping timelines or entities. Overall, our main contribution is showing that allocating explicit reasoning budget helps with chronological ordering with GPT-5 at medium/high reasoning effort achieving flawless ordering at all lengths and perfect conditional sorting (both self-filtered and given-subset), whereas low/minimal effort degrades with longer lists, mirroring earlier models. Our findings delineate limits of current LLMs on chronological tasks, providing insights into task complexity, and demonstrate scenarios in which reasoning helps. These patterns are important for the real-time application of LLMs in finance. We release all code and evaluation templates to support full reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15504v1">Game Master LLM: Task-Based Role-Playing for Natural Slang Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Natural and idiomatic expressions are essential for fluent, everyday communication, yet many second-language learners struggle to acquire and spontaneously use casual slang despite strong formal proficiency. To address this gap, we designed and evaluated an LLM-powered, task-based role-playing game in which a GPT-4o-based Game Master guides learners through an immersive, three-phase spoken narrative. After selecting five unfamiliar slang phrases to practice, participants engage in open-ended dialogue with non-player characters; the Game Master naturally incorporates the target phrases in rich semantic contexts (implicit input enhancement) while a dedicated Practice Box provides real-time explicit tracking and encouragement. Post-session, learners receive multi-level formative feedback analyzing the entire interaction. We evaluated the system in a between-subjects study with 14 international graduate students, randomly assigned to either the RPG condition or a control condition consisting of a traditional AI-led virtual classroom. Results from an immediate post-test show that the RPG group achieved greater gains in both comprehension of the target phrases and their accurate, contextual use in sentences. Quantitative analysis of in-activity word-usage frequency, combined with qualitative survey responses, further indicates that the game-based approach provided more practice opportunities and higher perceived engagement, resulting in a more natural learning experience. These findings highlight the potential of narrative-driven LLM interactions in vocabulary acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15456v1">Know Your Intent: An Autonomous Multi-Perspective LLM Agent Framework for DeFi User Transaction Intent Mining</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Written in 2025 Q1
    </div>
    <details class="paper-abstract">
      As Decentralized Finance (DeFi) develops, understanding user intent behind DeFi transactions is crucial yet challenging due to complex smart contract interactions, multifaceted on-/off-chain factors, and opaque hex logs. Existing methods lack deep semantic insight. To address this, we propose the Transaction Intent Mining (TIM) framework. TIM leverages a DeFi intent taxonomy built on grounded theory and a multi-agent Large Language Model (LLM) system to robustly infer user intents. A Meta-Level Planner dynamically coordinates domain experts to decompose multiple perspective-specific intent analyses into solvable subtasks. Question Solvers handle the tasks with multi-modal on/off-chain data. While a Cognitive Evaluator mitigates LLM hallucinations and ensures verifiability. Experiments show that TIM significantly outperforms machine learning models, single LLMs, and single Agent baselines. We also analyze core challenges in intent inference. This work helps provide a more reliable understanding of user motivations in DeFi, offering context-aware explanations for complex blockchain activity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08343v3">A Data-driven ML Approach for Maximizing Performance in LLM-Adapter Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted in a computer science workshop
    </div>
    <details class="paper-abstract">
      With the rapid adoption of Large Language Models (LLMs), LLM-adapters have become increasingly common, providing lightweight specialization of large-scale models. Serving hundreds or thousands of these adapters on a single GPU allows request aggregation, increasing throughput, but may also cause request starvation if GPU memory limits are exceeded. To address this issue, this study focuses on determining the joint configuration of concurrent and parallel adapters that maximizes GPU throughput without inducing starvation, given heterogeneous adapter and traffic properties. We propose a data-driven ML approach leveraging interpretable models to tackle this caching problem and introduce the first Digital Twin capable of reproducing an LLM-adapter serving system, enabling efficient training data generation. Experiments with the vLLM framework and LoRA adapters show that the Digital Twin reproduces throughput within 5.1% of real results, while the ML approach predicts optimal numbers of concurrent and parallel adapters with an error of at most 7.2% under heterogeneous, real-world workloads. The code is publicly available at https://github.com/FerranAgulloLopez/GPULLMAdapterOptimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15424v1">LLM-MemCluster: Empowering Large Language Models with Dynamic Memory for Text Clustering</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping unsupervised learning by offering an unprecedented ability to perform text clustering based on their deep semantic understanding. However, their direct application is fundamentally limited by a lack of stateful memory for iterative refinement and the difficulty of managing cluster granularity. As a result, existing methods often rely on complex pipelines with external modules, sacrificing a truly end-to-end approach. We introduce LLM-MemCluster, a novel framework that reconceptualizes clustering as a fully LLM-native task. It leverages a Dynamic Memory to instill state awareness and a Dual-Prompt Strategy to enable the model to reason about and determine the number of clusters. Evaluated on several benchmark datasets, our tuning-free framework significantly and consistently outperforms strong baselines. LLM-MemCluster presents an effective, interpretable, and truly end-to-end paradigm for LLM-based text clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15392v1">DEPO: Dual-Efficiency Preference Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have greatly improved their reasoning and decision-making abilities when deployed as agents. Richer reasoning, however, often comes at the cost of longer chain of thought (CoT), hampering interaction efficiency in real-world scenarios. Nevertheless, there still lacks systematic definition of LLM agent efficiency, hindering targeted improvements. To this end, we introduce dual-efficiency, comprising (i) step-level efficiency, which minimizes tokens per step, and (ii) trajectory-level efficiency, which minimizes the number of steps to complete a task. Building on this definition, we propose DEPO, a dual-efficiency preference optimization method that jointly rewards succinct responses and fewer action steps. Experiments on WebShop and BabyAI show that DEPO cuts token usage by up to 60.9% and steps by up to 26.9%, while achieving up to a 29.3% improvement in performance. DEPO also generalizes to three out-of-domain math benchmarks and retains its efficiency gains when trained on only 25% of the data. Our project page is at https://opencausalab.github.io/DEPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15389v1">Unveiling Inference Scaling for Difference-Aware User Modeling in LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into users' daily lives, driving a growing demand for personalized outputs. Prior work has primarily leveraged a user's own history, often overlooking inter-user differences that are critical for effective personalization. While recent methods have attempted to model such differences, their feature extraction processes typically rely on fixed dimensions and quick, intuitive inference (System-1 thinking), limiting both the coverage and granularity of captured user differences. To address these limitations, we propose Difference-aware Reasoning Personalization (DRP), a framework that reconstructs the difference extraction mechanism by leveraging inference scaling to enhance LLM personalization. DRP autonomously identifies relevant difference feature dimensions and generates structured definitions and descriptions, enabling slow, deliberate reasoning (System-2 thinking) over user differences. Experiments on personalized review generation demonstrate that DRP consistently outperforms baseline methods across multiple metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15357v1">Cost-Aware Prediction (CAP): An LLM-Enhanced Machine Learning Pipeline and Decision Support System for Heart Failure Mortality Prediction</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Objective: Machine learning (ML) predictive models are often developed without considering downstream value trade-offs and clinical interpretability. This paper introduces a cost-aware prediction (CAP) framework that combines cost-benefit analysis assisted by large language model (LLM) agents to communicate the trade-offs involved in applying ML predictions. Materials and Methods: We developed an ML model predicting 1-year mortality in patients with heart failure (N = 30,021, 22% mortality) to identify those eligible for home care. We then introduced clinical impact projection (CIP) curves to visualize important cost dimensions - quality of life and healthcare provider expenses, further divided into treatment and error costs, to assess the clinical consequences of predictions. Finally, we used four LLM agents to generate patient-specific descriptions. The system was evaluated by clinicians for its decision support value. Results: The eXtreme gradient boosting (XGB) model achieved the best performance, with an area under the receiver operating characteristic curve (AUROC) of 0.804 (95% confidence interval (CI) 0.792-0.816), area under the precision-recall curve (AUPRC) of 0.529 (95% CI 0.502-0.558) and a Brier score of 0.135 (95% CI 0.130-0.140). Discussion: The CIP cost curves provided a population-level overview of cost composition across decision thresholds, whereas LLM-generated cost-benefit analysis at individual patient-levels. The system was well received according to the evaluation by clinicians. However, feedback emphasizes the need to strengthen the technical accuracy for speculative tasks. Conclusion: CAP utilizes LLM agents to integrate ML classifier outcomes and cost-benefit analysis for more transparent and interpretable decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11729v2">Harli: SLO-Aware Co-location of LLM Inference and PEFT-based Finetuning on Model-as-a-Service Platforms</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed under the Model-as-a-Service (MaaS) paradigm. To meet stringent quality-of-service (QoS) requirements, existing LLM serving systems disaggregate the prefill and decode phases of inference. However, decode instances often experience low GPU utilization due to their memory-bound nature and insufficient batching in dynamic workloads, leaving compute resources underutilized. We introduce Harli, a serving system that improves GPU utilization by co-locating parameter-efficient finetuning (PEFT) tasks with LLM decode instances. PEFT tasks are compute-bound and memory-efficient, making them ideal candidates for safe co-location. Specifically, Harli addresses key challenges--limited memory and unpredictable interference--using three components: a unified memory allocator for runtime memory reuse, a two-stage latency predictor for decode latency modeling, and a QoS-guaranteed throughput-maximizing scheduler for throughput maximization. Experimental results show that Harli improves the finetune throughput by 46.2% on average (up to 92.0%) over state-of-the-art serving systems, while maintaining strict QoS guarantees for inference decode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14334v2">When Words Change the Model: Sensitivity of LLMs for Constraint Programming Modelling</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      One of the long-standing goals in optimisation and constraint programming is to describe a problem in natural language and automatically obtain an executable, efficient model. Large language models appear to bring this vision closer, showing impressive results in automatically generating models for classical benchmarks. However, much of this apparent success may derive from data contamination rather than genuine reasoning: many standard CP problems are likely included in the training data of these models. To examine this hypothesis, we systematically rephrased and perturbed a set of well-known CSPLib problems to preserve their structure while modifying their context and introducing misleading elements. We then compared the models produced by three representative LLMs across original and modified descriptions. Our qualitative analysis shows that while LLMs can produce syntactically valid and semantically plausible models, their performance drops sharply under contextual and linguistic variation, revealing shallow understanding and sensitivity to wording.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15248v1">EntroPIC: Towards Stable Long-Term Training of LLMs via Entropy Stabilization with Proportional-Integral Control</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Long-term training of large language models (LLMs) requires maintaining stable exploration to prevent the model from collapsing into sub-optimal behaviors. Entropy is crucial in this context, as it controls exploration and helps avoid premature convergence to sub-optimal solutions. However, existing reinforcement learning methods struggle to maintain an appropriate level of entropy, as the training process involves a mix of positive and negative samples, each affecting entropy in different ways across steps. To address this, we propose Entropy stablilization via Proportional-Integral Control (EntroPIC), a novel method that adaptively adjusts the influence of positive and negative samples by dynamically tuning their loss coefficients. This approach stabilizes entropy throughout training, ensuring efficient exploration and steady progress. We provide a comprehensive theoretical analysis for both on-policy and off-policy learning settings, demonstrating that EntroPIC is effective at controlling entropy in large-scale LLM training. Experimental results show that our method successfully maintains desired entropy levels, enabling stable and optimal RL training for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08916v2">HalluClean: A Unified Framework to Combat Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08008v2">Combining LLM Semantic Reasoning with GNN Structural Modeling for Multi-View Multi-Label Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Multi-view multi-label feature selection aims to identify informative features from heterogeneous views, where each sample is associated with multiple interdependent labels. This problem is particularly important in machine learning involving high-dimensional, multimodal data such as social media, bioinformatics or recommendation systems. Existing Multi-View Multi-Label Feature Selection (MVMLFS) methods mainly focus on analyzing statistical information of data, but seldom consider semantic information. In this paper, we aim to use these two types of information jointly and propose a method that combines Large Language Models (LLMs) semantic reasoning with Graph Neural Networks (GNNs) structural modeling for MVMLFS. Specifically, the method consists of three main components. (1) LLM is first used as an evaluation agent to assess the latent semantic relevance among feature, view, and label descriptions. (2) A semantic-aware heterogeneous graph with two levels is designed to represent relations among features, views and labels: one is a semantic graph representing semantic relations, and the other is a statistical graph. (3) A lightweight Graph Attention Network (GAT) is applied to learn node embedding in the heterogeneous graph as feature saliency scores for ranking and selection. Experimental results on multiple benchmark datasets demonstrate the superiority of our method over state-of-the-art baselines, and it is still effective when applied to small-scale datasets, showcasing its robustness, flexibility, and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07252v3">Tasa: Thermal-aware 3D-Stacked Architecture Design with Bandwidth Sharing for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      The autoregressive decoding in LLMs is the major inference bottleneck due to the memory-intensive operations and limited hardware bandwidth. 3D-stacked architecture is a promising solution with significantly improved memory bandwidth, which vertically stacked multi DRAM dies on top of logic die. However, our experiments also show the 3D-stacked architecture faces severer thermal issues compared to 2D architecture, in terms of thermal temperature, gradient and scalability. To better exploit the potential of 3D-stacked architecture, we present Tasa, a heterogeneous architecture with cross-stack thermal optimizations to balance the temperature distribution and maximize the performance under the thermal constraints. High-performance core is designed for compute-intensive operations, while high-efficiency core is used for memory-intensive operators, e.g. attention layers. Furthermore, we propose a bandwidth sharing scheduling to improve the bandwidth utilization in such heterogeneous architecture. Extensive thermal experiments show that our Tasa architecture demonstrates greater scalability compared with the homogeneous 3D-stacked architecture, i.e. up to 5.55 $\tccentigrade$, 9.37 $\tccentigrade$, and 7.91 $\tccentigrade$ peak temperature reduction for 48, 60, and 72 core configurations. Our experimental for Llama-65B and GPT-3 66B inferences also demonstrate 2.85x and 2.21x speedup are obtained over the GPU baselines and state-of-the-art heterogeneous PIM-based LLM accelerator
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15203v1">Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15202v1">SOLID: a Framework of Synergizing Optimization and LLMs for Intelligent Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-11-19
      | 💬 NeurIPS 2025 WORKSHOP ML*OR Workshop: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making
    </div>
    <details class="paper-abstract">
      This paper introduces SOLID (Synergizing Optimization and Large Language Models for Intelligent Decision-Making), a novel framework that integrates mathematical optimization with the contextual capabilities of large language models (LLMs). SOLID facilitates iterative collaboration between optimization and LLMs agents through dual prices and deviation penalties. This interaction improves the quality of the decisions while maintaining modularity and data privacy. The framework retains theoretical convergence guarantees under convexity assumptions, providing insight into the design of LLMs prompt. To evaluate SOLID, we applied it to a stock portfolio investment case with historical prices and financial news as inputs. Empirical results demonstrate convergence under various scenarios and indicate improved annualized returns compared to a baseline optimizer-only method, validating the synergy of the two agents. SOLID offers a promising framework for advancing automated and intelligent decision-making across diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15192v1">As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds. We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15191v1">HISE-KT: Synergizing Heterogeneous Information Networks and LLMs for Explainable Knowledge Tracing with Meta-Path Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-19
    </div>
    <details class="paper-abstract">
      Knowledge Tracing (KT) aims to mine students' evolving knowledge states and predict their future question-answering performance. Existing methods based on heterogeneous information networks (HINs) are prone to introducing noises due to manual or random selection of meta-paths and lack necessary quality assessment of meta-path instances. Conversely, recent large language models (LLMs)-based methods ignore the rich information across students, and both paradigms struggle to deliver consistently accurate and evidence-based explanations. To address these issues, we propose an innovative framework, HIN-LLM Synergistic Enhanced Knowledge Tracing (HISE-KT), which seamlessly integrates HINs with LLMs. HISE-KT first builds a multi-relationship HIN containing diverse node types to capture the structural relations through multiple meta-paths. The LLM is then employed to intelligently score and filter meta-path instances and retain high-quality paths, pioneering automated meta-path quality assessment. Inspired by educational psychology principles, a similar student retrieval mechanism based on meta-paths is designed to provide a more valuable context for prediction. Finally, HISE-KT uses a structured prompt to integrate the target student's history with the retrieved similar trajectories, enabling the LLM to generate not only accurate predictions but also evidence-backed, explainable analysis reports. Experiments on four public datasets show that HISE-KT outperforms existing KT baselines in both prediction performance and interpretability.
    </details>
</div>
