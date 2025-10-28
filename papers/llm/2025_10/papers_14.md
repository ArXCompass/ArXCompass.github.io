# llm - 2025_10

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
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- Part 14
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01667v3">Testing Low-Resource Language Support in LLMs Using Language Proficiency Exams: the Case of Luxembourgish</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 23pages, 4 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become an increasingly important tool in research and society at large. While LLMs are regularly used all over the world by experts and lay-people alike, they are predominantly developed with English-speaking users in mind, performing well in English and other wide-spread languages while less-resourced languages such as Luxembourgish are seen as a lower priority. This lack of attention is also reflected in the sparsity of available evaluation tools and datasets. In this study, we investigate the viability of language proficiency exams as such evaluation tools for the Luxembourgish language. We find that large models such as Claude and DeepSeek-R1 typically achieve high scores, while smaller models show weak performances. We also find that the performances in such language exams can be used to predict performances in other NLP tasks in Luxembourgish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04749v1">LLM-Based Information Extraction to Support Scientific Literature Research and Publication Workflows</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 This PDF is the author-prepared camera-ready version corresponding to the accepted manuscript and supersedes the submitted version that was inadvertently published as the version of record
    </div>
    <details class="paper-abstract">
      The increasing volume of scholarly publications requires advanced tools for efficient knowledge discovery and management. This paper introduces ongoing work on a system using Large Language Models (LLMs) for the semantic extraction of key concepts from scientific documents. Our research, conducted within the German National Research Data Infrastructure for and with Computer Science (NFDIxCS) project, seeks to support FAIR (Findable, Accessible, Interoperable, and Reusable) principles in scientific publishing. We outline our explorative work, which uses in-context learning with various LLMs to extract concepts from papers, initially focusing on the Business Process Management (BPM) domain. A key advantage of this approach is its potential for rapid domain adaptation, often requiring few or even zero examples to define extraction targets for new scientific fields. We conducted technical evaluations to compare the performance of commercial and open-source LLMs and created an online demo application to collect feedback from an initial user-study. Additionally, we gathered insights from the computer science research community through user stories collected during a dedicated workshop, actively guiding the ongoing development of our future services. These services aim to support structured literature reviews, concept-based information retrieval, and integration of extracted knowledge into existing knowledge graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07644v2">FloorplanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 v2, Project page: https://OldDelorean.github.io/FloorplanQA/
    </div>
    <details class="paper-abstract">
      We introduce FloorplanQA, a diagnostic benchmark for evaluating spatial reasoning in large-language models (LLMs). FloorplanQA is grounded in structured representations of indoor scenes, such as (e.g., kitchens, living rooms, bedrooms, bathrooms, and others), encoded symbolically in JSON or XML layouts. The benchmark covers core spatial tasks, including distance measurement, visibility, path finding, and object placement within constrained spaces. Our results across a variety of frontier open-source and commercial LLMs reveal that while models may succeed in shallow queries, they often fail to respect physical constraints, preserve spatial coherence, though they remain mostly robust to small spatial perturbations. FloorplanQA uncovers a blind spot in today's LLMs: inconsistent reasoning about indoor layouts. We hope this benchmark inspires new work on language models that can accurately infer and manipulate spatial and geometric properties in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04721v1">BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians. However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions rather than well-posed questions that are demonstrably false. To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review. Using an LLM-as-a-judge framework, we evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches substantially reduce, but do not eliminate, sycophantic behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04717v1">JSON Whisperer: Efficient JSON Editing with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can modify JSON documents through natural language commands, but current approaches regenerate entire structures for each edit, resulting in computational inefficiency. We present JSON Whisperer, a framework that enables LLMs to generate RFC 6902 diff patches-expressing only the necessary modifications-rather than complete documents. We identify two key challenges in patch-based editing: (1) LLMs often miss related updates when generating isolated patches, and (2) array manipulations require tracking index shifts across operations, which LLMs handle poorly. To address these issues, we introduce EASE (Explicitly Addressed Sequence Encoding), which transforms arrays into dictionaries with stable keys, eliminating index arithmetic complexities. Our evaluation shows that patch generation with EASE reduces token usage by 31% while maintaining edit quality within 5% of full regeneration with particular gains for complex instructions and list manipulations. The dataset is available at: https://github.com/emnlp2025/JSON-Whisperer/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04695v1">Beyond Outcome Reward: Decoupling Search and Answering Improves LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Enabling large language models (LLMs) to utilize search tools offers a promising path to overcoming fundamental limitations such as knowledge cutoffs and hallucinations. Recent work has explored reinforcement learning (RL) for training search-augmented agents that interleave reasoning and retrieval before answering. These approaches usually rely on outcome-based rewards (e.g., exact match), implicitly assuming that optimizing for final answers will also yield effective intermediate search behaviors. Our analysis challenges this assumption: we uncover multiple systematic deficiencies in search that arise under outcome-only training and ultimately degrade final answer quality, including failure to invoke tools, invalid queries, and redundant searches. To address these shortcomings, we introduce DeSA (Decoupling Search-and-Answering), a simple two-stage training framework that explicitly separates search optimization from answer generation. In Stage 1, agents are trained to improve search effectiveness with retrieval recall-based rewards. In Stage 2, outcome rewards are employed to optimize final answer generation. Across seven QA benchmarks, DeSA-trained agents consistently improve search behaviors, delivering substantially higher search recall and answer accuracy than outcome-only baselines. Notably, DeSA outperforms single-stage training approaches that simultaneously optimize recall and outcome rewards, underscoring the necessity of explicitly decoupling the two objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02089v4">SALAD: Systematic Assessment of Machine Unlearning on LLM-Aided Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative capabilities for hardware design automation, particularly in Verilog code generation. However, they also pose significant data security challenges, including Verilog evaluation data contamination, intellectual property (IP) design leakage, and the risk of malicious Verilog generation. We introduce SALAD, a comprehensive assessment that leverages machine unlearning to mitigate these threats. Our approach enables the selective removal of contaminated benchmarks, sensitive IP and design artifacts, or malicious code patterns from pre-trained LLMs, all without requiring full retraining. Through detailed case studies, we demonstrate how machine unlearning techniques effectively reduce data security risks in LLM-aided hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04641v1">Evaluating LLMs for Demographic-Targeted Social Bias Detection: A Comprehensive Benchmark Study</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 17 pages, 7 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large-scale web-scraped text corpora used to train general-purpose AI models often contain harmful demographic-targeted social biases, creating a regulatory need for data auditing and developing scalable bias-detection methods. Although prior work has investigated biases in text datasets and related detection methods, these studies remain narrow in scope. They typically focus on a single content type (e.g., hate speech), cover limited demographic axes, overlook biases affecting multiple demographics simultaneously, and analyze limited techniques. Consequently, practitioners lack a holistic understanding of the strengths and limitations of recent large language models (LLMs) for automated bias detection. In this study, we present a comprehensive evaluation framework aimed at English texts to assess the ability of LLMs in detecting demographic-targeted social biases. To align with regulatory requirements, we frame bias detection as a multi-label task using a demographic-focused taxonomy. We then conduct a systematic evaluation with models across scales and techniques, including prompting, in-context learning, and fine-tuning. Using twelve datasets spanning diverse content types and demographics, our study demonstrates the promise of fine-tuned smaller models for scalable detection. However, our analyses also expose persistent gaps across demographic axes and multi-demographic targeted biases, underscoring the need for more effective and scalable auditing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04637v1">Social Agent: Mastering Dyadic Nonverbal Behavior Generation via Conversational LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 SIGGRAPH ASIA 2025 (Conference Track); Project page: https://pku-mocca.github.io/Social-Agent-Page/
    </div>
    <details class="paper-abstract">
      We present Social Agent, a novel framework for synthesizing realistic and contextually appropriate co-speech nonverbal behaviors in dyadic conversations. In this framework, we develop an agentic system driven by a Large Language Model (LLM) to direct the conversation flow and determine appropriate interactive behaviors for both participants. Additionally, we propose a novel dual-person gesture generation model based on an auto-regressive diffusion model, which synthesizes coordinated motions from speech signals. The output of the agentic system is translated into high-level guidance for the gesture generator, resulting in realistic movement at both the behavioral and motion levels. Furthermore, the agentic system periodically examines the movements of interlocutors and infers their intentions, forming a continuous feedback loop that enables dynamic and responsive interactions between the two participants. User studies and quantitative evaluations show that our model significantly improves the quality of dyadic interactions, producing natural, synchronized nonverbal behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02369v2">Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents typically receive two kinds of context: (i) environment-level manuals that define interaction interfaces and global rules, and (ii) task-level guidance or demonstrations tied to specific goals. In this work, we identify a crucial but overlooked third type of context, instance-level context, which consists of verifiable and reusable facts tied to a specific environment instance, such as object locations, crafting recipes, and local rules. We argue that the absence of instance-level context is a common source of failure for LLM agents in complex tasks, as success often depends not only on reasoning over global rules or task prompts but also on making decisions based on precise and persistent facts. Acquiring such context requires more than memorization: the challenge lies in efficiently exploring, validating, and formatting these facts under tight interaction budgets. We formalize this problem as Instance-Level Context Learning (ILCL) and introduce our task-agnostic method to solve it. Our method performs a guided exploration, using a compact TODO forest to intelligently prioritize its next actions and a lightweight plan-act-extract loop to execute them. This process automatically produces a high-precision context document that is reusable across many downstream tasks and agents, thereby amortizing the initial exploration cost. Experiments across TextWorld, ALFWorld, and Crafter demonstrate consistent gains in both success and efficiency: for instance, ReAct's mean success rate in TextWorld rises from 37% to 95%, while IGE improves from 81% to 95%. By transforming one-off exploration into persistent, reusable knowledge, our method complements existing contexts to enable more reliable and efficient LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04633v1">Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 15 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The unjudged document problem, where pooled test collections have incomplete relevance judgments for evaluating new retrieval systems, is a key obstacle to the reusability of test collections in information retrieval. While the de facto standard to deal with the problem is to treat unjudged documents as non-relevant, many alternatives have been proposed, including the use of large language models (LLMs) as a relevance judge (LLM-as-a-judge). However, this has been criticized as circular, since the same LLM can be used as a judge and as a ranker at the same time. We propose to train topic-specific relevance classifiers instead: By finetuning monoT5 with independent LoRA weight adaptation on the judgments of a single assessor for a single topic's pool, we align it to that assessor's notion of relevance for the topic. The system rankings obtained through our classifier's relevance judgments achieve a Spearmans' $\rho$ correlation of $>0.95$ with ground truth system rankings. As little as 128 initial human judgments per topic suffice to improve the comparability of models, compared to treating unjudged documents as non-relevant, while achieving more reliability than existing LLM-as-a-judge approaches. Topic-specific relevance classifiers thus are a lightweight and straightforward way to tackle the unjudged document problem, while maintaining human judgments as the gold standard for retrieval evaluation. Code, models, and data are made openly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01617v2">AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-10-06
      | 💬 Accepted by EMNLP-2025 Industrial Track
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04607v1">A Case for Declarative LLM-friendly Interfaces for Improved Efficiency of Computer-Use Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Computer-use agents (CUAs) powered by large language models (LLMs) have emerged as a promising approach to automating computer tasks, yet they struggle with graphical user interfaces (GUIs). GUIs, designed for humans, force LLMs to decompose high-level goals into lengthy, error-prone sequences of fine-grained actions, resulting in low success rates and an excessive number of LLM calls. We propose Goal-Oriented Interface (GOI), a novel abstraction that transforms existing GUIs into three declarative primitives: access, state, and observation, which are better suited for LLMs. Our key idea is policy-mechanism separation: LLMs focus on high-level semantic planning (policy) while GOI handles low-level navigation and interaction (mechanism). GOI does not require modifying the application source code or relying on application programming interfaces (APIs). We evaluate GOI with Microsoft Office Suite (Word, PowerPoint, Excel) on Windows. Compared to a leading GUI-based agent baseline, GOI improves task success rates by 67% and reduces interaction steps by 43.5%. Notably, GOI completes over 61% of successful tasks with a single LLM call.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10364v3">Can We Infer Confidential Properties of Training Data from LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04581v1">Can LLMs Detect Ambiguous Plural Reference? An Analysis of Split-Antecedent and Mereological Reference</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Our goal is to study how LLMs represent and interpret plural reference in ambiguous and unambiguous contexts. We ask the following research questions: (1) Do LLMs exhibit human-like preferences in representing plural reference? (2) Are LLMs able to detect ambiguity in plural anaphoric expressions and identify possible referents? To address these questions, we design a set of experiments, examining pronoun production using next-token prediction tasks, pronoun interpretation, and ambiguity detection using different prompting strategies. We then assess how comparable LLMs are to humans in formulating and interpreting plural reference. We find that LLMs are sometimes aware of possible referents of ambiguous pronouns. However, they do not always follow human reference when choosing between interpretations, especially when the possible interpretation is not explicitly mentioned. In addition, they struggle to identify ambiguity without direct instruction. Our findings also reveal inconsistencies in the results across different types of experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04573v1">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04567v1">GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      Graph Neural Networks (GNNs) are powerful tools for precessing relational data but often struggle to generalize to unseen graphs, giving rise to the development of Graph Foundational Models (GFMs). However, current GFMs are challenged by the extreme heterogeneity of graph data, where each graph can possess a unique feature space, label set, and topology. To address this, two main paradigms have emerged. The first leverages Large Language Models (LLMs), but is fundamentally text-dependent, thus struggles to handle the numerical features in vast graphs. The second pre-trains a structure-based model, but the adaptation to new tasks typically requires a costly, per-graph tuning stage, creating a critical efficiency bottleneck. In this work, we move beyond these limitations and introduce \textbf{G}raph \textbf{I}n-context \textbf{L}earning \textbf{T}ransformer (GILT), a framework built on an LLM-free and tuning-free architecture. GILT introduces a novel token-based framework for in-context learning (ICL) on graphs, reframing classification tasks spanning node, edge and graph levels in a unified framework. This mechanism is the key to handling heterogeneity, as it is designed to operate on generic numerical features. Further, its ability to understand class semantics dynamically from the context enables tuning-free adaptation. Comprehensive experiments show that GILT achieves stronger few-shot performance with significantly less time than LLM-based or tuning-based baselines, validating the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04536v1">3Dify: a Framework for Procedural 3D-CG Generation Assisted by LLMs Using MCP and RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-06
    </div>
    <details class="paper-abstract">
      This paper proposes "3Dify," a procedural 3D computer graphics (3D-CG) generation framework utilizing Large Language Models (LLMs). The framework enables users to generate 3D-CG content solely through natural language instructions. 3Dify is built upon Dify, an open-source platform for AI application development, and incorporates several state-of-the-art LLM-related technologies such as the Model Context Protocol (MCP) and Retrieval-Augmented Generation (RAG). For 3D-CG generation support, 3Dify automates the operation of various Digital Content Creation (DCC) tools via MCP. When DCC tools do not support MCP-based interaction, the framework employs the Computer-Using Agent (CUA) method to automate Graphical User Interface (GUI) operations. Moreover, to enhance image generation quality, 3Dify allows users to provide feedback by selecting preferred images from multiple candidates. The LLM then learns variable patterns from these selections and applies them to subsequent generations. Furthermore, 3Dify supports the integration of locally deployed LLMs, enabling users to utilize custom-developed models and to reduce both time and monetary costs associated with external API calls by leveraging their own computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04195v1">Constructing coherent spatial memory in LLM agents through graph rectification</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Given a map description through global traversal navigation instructions (e.g., visiting each room sequentially with action signals such as north, west, etc.), an LLM can often infer the implicit spatial layout of the environment and answer user queries by providing a shortest path from a start to a destination (for instance, navigating from the lobby to a meeting room via the hall and elevator). However, such context-dependent querying becomes incapable as the environment grows much longer, motivating the need for incremental map construction that builds a complete topological graph from stepwise observations. We propose a framework for LLM-driven construction and map repair, designed to detect, localize, and correct structural inconsistencies in incrementally constructed navigation graphs. Central to our method is the Version Control, which records the full history of graph edits and their source observations, enabling fine-grained rollback, conflict tracing, and repair evaluation. We further introduce an Edge Impact Score to prioritize minimal-cost repairs based on structural reachability, path usage, and conflict propagation. To properly evaluate our approach, we create a refined version of the MANGO benchmark dataset by systematically removing non-topological actions and inherent structural conflicts, providing a cleaner testbed for LLM-driven construction and map repair. Our approach significantly improves map correctness and robustness, especially in scenarios with entangled or chained inconsistencies. Our results highlight the importance of introspective, history-aware repair mechanisms for maintaining coherent spatial memory in LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20278v2">Instruction Boundary: Quantifying Biases in LLM Reasoning under Various Coverage</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Nowadays, automatically generated datasets are increasingly used in LLM reasoning tasks; however, large-scale corpora often contain inherent flaws. For example, a single-choice question may include none or multiple correct options, while true-or-false questions may involve vague or unverifiable statements. We refer to these exceptional answer forms as sparse labels. To compare LLMs' ability to recognize various question forms and produce correct answers, we investigate how different instruction formats can either facilitate or mislead LLM reasoning ability. We introduce the concept of Instruction Boundary, which systematically analyzes how different levels of prompt coverage -- sufficient, redundant, or insufficient -- can lead to reasoning biases and performance changes in LLMs. To examine this phenomenon, we design eight experimental settings across five dataset forms. We further propose BiasDetector, a unified framework that quantifies LLMs' ability to identify sparse labels under different kinds of Instruction Boundary conditions. Evaluations on five mainstream LLMs show that, despite their seemingly high accuracy, substantial reasoning biases persist in many downstream tasks as a direct consequence of prompt coverage. We analyze the impact of these biases and outline possible mitigation strategies. Our findings highlight not only the importance of addressing sparse labels, but also the need for developers to recognize and mitigate the risks introduced by Instruction Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14474v2">LexiMark: Robust Watermarking via Lexical Substitutions to Enhance Membership Verification of an LLM's Textual Training Data</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be trained or fine-tuned on data obtained without the owner's consent. Verifying whether a specific LLM was trained on particular data instances or an entire dataset is extremely challenging. Dataset watermarking addresses this by embedding identifiable modifications in training data to detect unauthorized use. However, existing methods often lack stealth, making them relatively easy to detect and remove. In light of these limitations, we propose LexiMark, a novel watermarking technique designed for text and documents, which embeds synonym substitutions for carefully selected high-entropy words. Our method aims to enhance an LLM's memorization capabilities on the watermarked text without altering the semantic integrity of the text. As a result, the watermark is difficult to detect, blending seamlessly into the text with no visible markers, and is resistant to removal due to its subtle, contextually appropriate substitutions that evade automated and manual detection. We evaluated our method using baseline datasets from recent studies and seven open-source models: LLaMA-1 7B, LLaMA-3 8B, Mistral 7B, Pythia 6.9B, as well as three smaller variants from the Pythia family (160M, 410M, and 1B). Our evaluation spans multiple training settings, including continued pretraining and fine-tuning scenarios. The results demonstrate significant improvements in AUROC scores compared to existing methods, underscoring our method's effectiveness in reliably verifying whether unauthorized watermarked data was used in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04140v1">Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has become a widely adopted technique for enhancing the reasoning ability of Large Language Models (LLMs). However, the effectiveness of RLVR strongly depends on the capability of base models. This issue arises because it requires the model to have sufficient capability to perform high-quality exploration, which involves both effectiveness and diversity. Unfortunately, existing methods address this issue by imitating expert trajectories, which improve effectiveness but neglect diversity. To address this, we argue that the expert only needs to provide guidance only at critical decision points rather than the entire reasoning path. Based on this insight, we propose MENTOR: Mixed-policy Expert Navigation for Token-level Optimization of Reasoning, a framework that provides expert guidance only at critical decision points to perform effective and diverse exploration in RLVR. Extensive experiments show that MENTOR enables models capture the essence of expert strategies rather than surface imitation, thereby performing high-quality exploration and achieving superior overall performance. Our code is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16199v4">WakenLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently output the label Unknown in reasoning tasks, where two scenarios may appear: (i) an input sample is genuinely unverifiable, but the model cannot understand why; and (ii) a verifiable problem that the model fails to solve, thus outputs Unknown. We refer to these cases collectively as the Vague Perception phenomenon. Current evaluations focus on whether such answers are honest, rather than analyzing the limits of LLM reasoning. To address this, we introduce WakenLLM, a framework that quantifies the portion of Unknown output attributable to model incapacity and evaluates whether stimulation can convert them into either correct answers (verifiable) or justified (unverifiable) responses with valid reasoning. Our method offers a clearer picture of the limits of LLM reasoning and the potential for corrections across various datasets. Comprehensive experiments on six LLMs suggest that, without any training or parameter revision, LLMs can achieve up to a 68.53% accuracy improvement on Vague Perception samples through guided understanding. Our work reveals that current baseline methods only activate a small portion of LLMs' reasoning potential, indicating considerable unexplored capacity. This extends the theoretical upper bounds of reasoning accuracy in LLMs. Consequently, this study deepens our understanding of the latent reasoning capacity of LLMs and offers a new perspective on addressing the Vague Perception phenomenon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04120v1">Unveiling LLMs' Metaphorical Understanding: Exploring Conceptual Irrelevance, Context Leveraging and Syntactic Influence</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Metaphor analysis is a complex linguistic phenomenon shaped by context and external factors. While Large Language Models (LLMs) demonstrate advanced capabilities in knowledge integration, contextual reasoning, and creative generation, their mechanisms for metaphor comprehension remain insufficiently explored. This study examines LLMs' metaphor-processing abilities from three perspectives: (1) Concept Mapping: using embedding space projections to evaluate how LLMs map concepts in target domains (e.g., misinterpreting "fall in love" as "drop down from love"); (2) Metaphor-Literal Repository: analyzing metaphorical words and their literal counterparts to identify inherent metaphorical knowledge; and (3) Syntactic Sensitivity: assessing how metaphorical syntactic structures influence LLMs' performance. Our findings reveal that LLMs generate 15\%-25\% conceptually irrelevant interpretations, depend on metaphorical indicators in training data rather than contextual cues, and are more sensitive to syntactic irregularities than to structural comprehension. These insights underline the limitations of LLMs in metaphor analysis and call for more robust computational approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04116v1">Searching Meta Reasoning Skeleton to Guide LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Meta reasoning behaviors work as a skeleton to guide large language model (LLM) reasoning, thus help to improve reasoning performance. However, prior researches implement meta reasoning skeleton with manually designed structure, limiting ability to adapt to query-specific requirement and capture intricate logical dependency among reasoning steps. To deal with the challenges, we represent meta reasoning skeleton with directed acyclic graph (DAG) to unify skeletons proposed in prior works and model intricate logical dependency. Then we propose AutoMR, a framework that searches for query-aware meta reasoning skeleton automatically inspired by automated machine learning (AutoML). Specifically, we construct search space based on DAG representation of skeleton and then formulate the search problem. We design a dynamic skeleton sampling algorithm by expanding meta reasoning skeleton along with reasoning context at inference time. This algorithm can derive any meta reasoning skeleton in search space efficiently and adapt skeleton to evolving base reasoning context, thus enable efficient query-aware skeleton search. We conduct experiments on extensive benchmark datasets. Experimental results show that AutoMR achieves better reasoning performance than previous works broadly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04108v1">Can Linear Probes Measure LLM Uncertainty?</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Effective Uncertainty Quantification (UQ) represents a key aspect for reliable deployment of Large Language Models (LLMs) in automated decision-making and beyond. Yet, for LLM generation with multiple choice structure, the state-of-the-art in UQ is still dominated by the naive baseline given by the maximum softmax score. To address this shortcoming, we demonstrate that taking a principled approach via Bayesian statistics leads to improved performance despite leveraging the simplest possible model, namely linear regression. More precisely, we propose to train multiple Bayesian linear models, each predicting the output of a layer given the output of the previous one. Based on the obtained layer-level posterior distributions, we infer the global uncertainty level of the LLM by identifying a sparse combination of distributional features, leading to an efficient UQ scheme. Numerical experiments on various LLMs show consistent improvement over state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10714v3">ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      The linear growth of key-value (KV) cache memory and quadratic computational in attention mechanisms complexity pose significant bottlenecks for large language models (LLMs) in long-context processing. While existing KV cache optimization methods address these challenges through token pruning or feature merging, they often incur irreversible information loss or require costly parameter retraining. To this end, we propose ZSMerge, a dynamic KV cache compression framework designed for efficient cache management, featuring three key operations: (1) fine-grained memory allocation guided by multi-dimensional token importance metrics at head-level granularity, (2) a residual merging mechanism that preserves critical context through compensated attention scoring, and (3) a zero-shot adaptation mechanism compatible with diverse LLM architectures without requiring retraining. ZSMerge significantly enhances memory efficiency and inference speed with negligible performance degradation across LLMs. When applied to LLaMA2-7B, it demonstrates a 20:1 compression ratio for key-value cache retention (reducing memory footprint to 5\% of baseline) while sustaining comparable generation quality, coupled with triple throughput gains at extreme 54k-token contexts that eliminate out-of-memory failures. The code is available at https://github.com/SusCom-Lab/ZSMerge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04093v1">Harnessing LLM for Noise-Robust Cognitive Diagnosis in Web-Based Intelligent Education Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Cognitive diagnostics in the Web-based Intelligent Education System (WIES) aims to assess students' mastery of knowledge concepts from heterogeneous, noisy interactions. Recent work has tried to utilize Large Language Models (LLMs) for cognitive diagnosis, yet LLMs struggle with structured data and are prone to noise-induced misjudgments. Specially, WIES's open environment continuously attracts new students and produces vast amounts of response logs, exacerbating the data imbalance and noise issues inherent in traditional educational systems. To address these challenges, we propose DLLM, a Diffusion-based LLM framework for noise-robust cognitive diagnosis. DLLM first constructs independent subgraphs based on response correctness, then applies relation augmentation alignment module to mitigate data imbalance. The two subgraph representations are then fused and aligned with LLM-derived, semantically augmented representations. Importantly, before each alignment step, DLLM employs a two-stage denoising diffusion module to eliminate intrinsic noise while assisting structural representation alignment. Specifically, unconditional denoising diffusion first removes erroneous information, followed by conditional denoising diffusion based on graph-guided to eliminate misleading information. Finally, the noise-robust representation that integrates semantic knowledge and structural information is fed into existing cognitive diagnosis models for prediction. Experimental results on three publicly available web-based educational platform datasets demonstrate that our DLLM achieves optimal predictive performance across varying noise levels, which demonstrates that DLLM achieves noise robustness while effectively leveraging semantic knowledge from LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04078v1">Bamboo: LLM-Driven Discovery of API-Permission Mappings in the Android Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      The permission mechanism in the Android Framework is integral to safeguarding the privacy of users by managing users' and processes' access to sensitive resources and operations. As such, developers need to be equipped with an in-depth understanding of API permissions to build robust Android apps. Unfortunately, the official API documentation by Android chronically suffers from imprecision and incompleteness, causing developers to spend significant effort to accurately discern necessary permissions. This potentially leads to incorrect permission declarations in Android app development, potentially resulting in security violations and app failures. Recent efforts in improving permission specification primarily leverage static and dynamic code analyses to uncover API-permission mappings within the Android framework. Yet, these methodologies encounter substantial shortcomings, including poor adaptability to Android SDK and Framework updates, restricted code coverage, and a propensity to overlook essential API-permission mappings in intricate codebases. This paper introduces a pioneering approach utilizing large language models (LLMs) for a systematic examination of API-permission mappings. In addition to employing LLMs, we integrate a dual-role prompting strategy and an API-driven code generation approach into our mapping discovery pipeline, resulting in the development of the corresponding tool, \tool{}. We formulate three research questions to evaluate the efficacy of \tool{} against state-of-the-art baselines, assess the completeness of official SDK documentation, and analyze the evolution of permission-required APIs across different SDK releases. Our experimental results reveal that \tool{} identifies 2,234, 3,552, and 4,576 API-permission mappings in Android versions 6, 7, and 10 respectively, substantially outprforming existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04076v1">From Shadow to Light: Toward Safe and Efficient Policy Learning Across MPC, DeePC, RL, and LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      One of the main challenges in modern control applications, particularly in robot and vehicle motion control, is achieving accurate, fast, and safe movement. To address this, optimal control policies have been developed to enforce safety while ensuring high performance. Since basic first-principles models of real systems are often available, model-based controllers are widely used. Model predictive control (MPC) is a leading approach that optimizes performance while explicitly handling safety constraints. However, obtaining accurate models for complex systems is difficult, which motivates data-driven alternatives. ML-based MPC leverages learned models to reduce reliance on hand-crafted dynamics, while reinforcement learning (RL) can learn near-optimal policies directly from interaction data. Data-enabled predictive control (DeePC) goes further by bypassing modeling altogether, directly learning safe policies from raw input-output data. Recently, large language model (LLM) agents have also emerged, translating natural language instructions into structured formulations of optimal control problems. Despite these advances, data-driven policies face significant limitations. They often suffer from slow response times, high computational demands, and large memory needs, making them less practical for real-world systems with fast dynamics, limited onboard computing, or strict memory constraints. To address this, various technique, such as reduced-order modeling, function-approximated policy learning, and convex relaxations, have been proposed to reduce computational complexity. In this paper, we present eight such approaches and demonstrate their effectiveness across real-world applications, including robotic arms, soft robots, and vehicle motion control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04072v1">Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and a 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04064v1">Decoding Emotion in the Deep: A Systematic Study of How LLMs Represent, Retain, and Express Emotion</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 10 pages, 7 figures, 4 tables. Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly expected to navigate the nuances of human emotion. While research confirms that LLMs can simulate emotional intelligence, their internal emotional mechanisms remain largely unexplored. This paper investigates the latent emotional representations within modern LLMs by asking: how, where, and for how long is emotion encoded in their neural architecture? To address this, we introduce a novel, large-scale Reddit corpus of approximately 400,000 utterances, balanced across seven basic emotions through a multi-stage process of classification, rewriting, and synthetic generation. Using this dataset, we employ lightweight "probes" to read out information from the hidden layers of various Qwen3 and LLaMA models without altering their parameters. Our findings reveal that LLMs develop a surprisingly well-defined internal geometry of emotion, which sharpens with model scale and significantly outperforms zero-shot prompting. We demonstrate that this emotional signal is not a final-layer phenomenon but emerges early and peaks mid-network. Furthermore, the internal states are both malleable (they can be influenced by simple system prompts) and persistent, as the initial emotional tone remains detectable for hundreds of subsequent tokens. We contribute our dataset, an open-source probing toolkit, and a detailed map of the emotional landscape within LLMs, offering crucial insights for developing more transparent and aligned AI systems. The code and dataset are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v6">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. The main underlying idea is to reduces the best possible accuracy, i.e., Bayes accuracy, by injecting randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04056v1">Real-VulLLM: An LLM Based Assessment Framework in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) and more specifically Large Language Models (LLMs) have demonstrated exceptional progress in multiple areas including software engineering, however, their capability for vulnerability detection in the wild scenario and its corresponding reasoning remains underexplored. Prompting pre-trained LLMs in an effective way offers a computationally effective and scalable solution. Our contributions are (i)varied prompt designs for vulnerability detection and its corresponding reasoning in the wild. (ii)a real-world vector data store constructed from the National Vulnerability Database, that will provide real time context to vulnerability detection framework, and (iii)a scoring measure for combined measurement of accuracy and reasoning quality. Our contribution aims to examine whether LLMs are ready for wild deployment, thus enabling the reliable use of LLMs stronger for the development of secure software's.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04048v1">Increasing LLM response trustworthiness using voting ensembles</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Despite huge advances, LLMs still lack convenient and reliable methods to quantify the uncertainty in their responses, making them difficult to trust in high-stakes applications. One of the simplest approaches to eliciting more accurate answers is to select the mode of many responses, a technique known as ensembling. In this work, we expand on typical ensembling approaches by looking at ensembles with a variable voting threshold. We introduce a theoretical framework for question answering and show that, by permitting ensembles to "abstain" from providing an answer when the dominant response falls short of the threshold, it is possible to dramatically increase the trustworthiness of the remaining answers. From this framework, we derive theoretical results as well as report experimental results on two problem domains: arithmetic problem solving and clinical-note question-answering. In both domains, we observe that large gains in answer trustworthiness can be achieved using highly restrictive voting ensembles, while incurring relatively modest reductions in response yield and accuracy. Due to this quality, voting ensembles may be particularly useful in applications - such as healthcare and data annotation - that require a high degree of certainty but which may not require that every question receive an automated answer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01713v3">SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05169v1">From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can acquire deceptive behaviors through backdoor attacks, where the model executes prohibited actions whenever secret triggers appear in the input. Existing safety training methods largely fail to address this vulnerability, due to the inherent difficulty of uncovering hidden triggers implanted in the model. Motivated by recent findings on LLMs' situational awareness, we propose a novel post-training framework that cultivates self-awareness of backdoor risks and enables models to articulate implanted triggers even when they are absent from the prompt. At its core, our approach introduces an inversion-inspired reinforcement learning framework that encourages models to introspectively reason about their own behaviors and reverse-engineer the triggers responsible for misaligned outputs. Guided by curated reward signals, this process transforms a poisoned model into one capable of precisely identifying its implanted trigger. Surprisingly, we observe that such backdoor self-awareness emerges abruptly within a short training window, resembling a phase transition in capability. Building on this emergent property, we further present two complementary defense strategies for mitigating and detecting backdoor threats. Experiments on five backdoor attacks, compared against six baseline methods, demonstrate that our approach has strong potential to improve the robustness of LLMs against backdoor risks. The code is available at LLM Backdoor Self-Awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04031v1">Does Using Counterfactual Help LLMs Explain Textual Importance in Classification?</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. More recently, they have been shown to be very effective in textual classification tasks, motivating the need to explain the LLMs' decisions. Motivated by practical constrains where LLMs are black-boxed and LLM calls are expensive, we study how incorporating counterfactuals into LLM reasoning can affect the LLM's ability to identify the top words that have contributed to its classification decision. To this end, we introduce a framework called the decision changing rate that helps us quantify the importance of the top words in classification. Our experimental results show that using counterfactuals can be helpful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04024v1">Enhancing Fake News Video Detection via LLM-Driven Creative Process Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 ACM CIKM 2025
    </div>
    <details class="paper-abstract">
      The emergence of fake news on short video platforms has become a new significant societal concern, necessitating automatic video-news-specific detection. Current detectors primarily rely on pattern-based features to separate fake news videos from real ones. However, limited and less diversified training data lead to biased patterns and hinder their performance. This weakness stems from the complex many-to-many relationships between video material segments and fabricated news events in real-world scenarios: a single video clip can be utilized in multiple ways to create different fake narratives, while a single fabricated event often combines multiple distinct video segments. However, existing datasets do not adequately reflect such relationships due to the difficulty of collecting and annotating large-scale real-world data, resulting in sparse coverage and non-comprehensive learning of the characteristics of potential fake news video creation. To address this issue, we propose a data augmentation framework, AgentAug, that generates diverse fake news videos by simulating typical creative processes. AgentAug implements multiple LLM-driven pipelines of four fabrication categories for news video creation, combined with an active learning strategy based on uncertainty sampling to select the potentially useful augmented samples during training. Experimental results on two benchmark datasets demonstrate that AgentAug consistently improves the performance of short video fake news detectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04023v1">LLM-Based Data Science Agents: A Survey of Capabilities, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Survey paper; 45 data science agents; under review
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled a new class of AI agents that automate multiple stages of the data science workflow by integrating planning, tool use, and multimodal reasoning across text, code, tables, and visuals. This survey presents the first comprehensive, lifecycle-aligned taxonomy of data science agents, systematically analyzing and mapping forty-five systems onto the six stages of the end-to-end data science process: business understanding and data acquisition, exploratory analysis and visualization, feature engineering, model building and selection, interpretation and explanation, and deployment and monitoring. In addition to lifecycle coverage, we annotate each agent along five cross-cutting design dimensions: reasoning and planning style, modality integration, tool orchestration depth, learning and alignment methods, and trust, safety, and governance mechanisms. Beyond classification, we provide a critical synthesis of agent capabilities, highlight strengths and limitations at each stage, and review emerging benchmarks and evaluation practices. Our analysis identifies three key trends: most systems emphasize exploratory analysis, visualization, and modeling while neglecting business understanding, deployment, and monitoring; multimodal reasoning and tool orchestration remain unresolved challenges; and over 90% lack explicit trust and safety mechanisms. We conclude by outlining open challenges in alignment stability, explainability, governance, and robust evaluation frameworks, and propose future research directions to guide the development of robust, trustworthy, low-latency, transparent, and broadly accessible data science agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04245v2">Contextual Integrity in LLMs via Reasoning and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04013v1">LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have tremendous utility, trustworthiness is still a chief concern: models often generate incorrect information with high confidence. While contextual information can help guide generation, identifying when a query would benefit from retrieved context and assessing the effectiveness of that context remains challenging. In this work, we operationalize interpretability methods to ascertain whether we can predict the correctness of model outputs from the model's activations alone. We also explore whether model internals contain signals about the efficacy of external context. We consider correct, incorrect, and irrelevant context and introduce metrics to distinguish amongst them. Experiments on six different models reveal that a simple classifier trained on intermediate layer activations of the first output token can predict output correctness with about 75% accuracy, enabling early auditing. Our model-internals-based metric significantly outperforms prompting baselines at distinguishing between correct and incorrect context, guarding against inaccuracies introduced by polluted context. These findings offer a lens to better understand the underlying decision-making processes of LLMs. Our code is publicly available at https://github.com/jiarui-liu/LLM-Microscope
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12396v2">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03997v1">Mapping Patient-Perceived Physician Traits from Nationwide Online Reviews with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Understanding how patients perceive their physicians is essential to improving trust, communication, and satisfaction. We present a large language model (LLM)-based pipeline that infers Big Five personality traits and five patient-oriented subjective judgments. The analysis encompasses 4.1 million patient reviews of 226,999 U.S. physicians from an initial pool of one million. We validate the method through multi-model comparison and human expert benchmarking, achieving strong agreement between human and LLM assessments (correlation coefficients 0.72-0.89) and external validity through correlations with patient satisfaction (r = 0.41-0.81, all p<0.001). National-scale analysis reveals systematic patterns: male physicians receive higher ratings across all traits, with largest disparities in clinical competence perceptions; empathy-related traits predominate in pediatrics and psychiatry; and all traits positively predict overall satisfaction. Cluster analysis identifies four distinct physician archetypes, from "Well-Rounded Excellent" (33.8%, uniformly high traits) to "Underperforming" (22.6%, consistently low). These findings demonstrate that automated trait extraction from patient narratives can provide interpretable, validated metrics for understanding physician-patient relationships at scale, with implications for quality measurement, bias detection, and workforce development in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03988v1">Distilling Reasoning into Student LLMs: Local Naturalness for Selecting Teacher Data</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Distilling long reasoning traces (10K+ tokens) from stronger teacher models into smaller student LLMs via SFT has emerged as a standard paradigm. This approach is practical and efficient: it leverages the ease of generating abundant reasoning data from stronger models and provides a direct, data-driven way to teach less capable models better reasoning. While previous work has largely focused on prompt selection with responses from a single teacher, the equally important problem of choosing the best response when multiple teacher outputs are available for a single prompt remains underexplored. This challenge becomes important in a multi-teacher setting, where different students may benefit from the outputs of different teachers. This paper fills that gap with a systematic study of response selection for reasoning distillation. We first show that the current method, which picks responses the student assigns the highest global log-probability (global naturalness), fails when responses come from multiple teachers, i.e., global naturalness no longer correlates with downstream performance, especially as the reasoning traces from strong teachers become longer. To overcome this problem, we introduce Local Naturalness, which measures the student's log-probabilities over short, sequential reasoning steps conditioned only on a small local window. Local Naturalness enables two applications: 1) Teacher Selection: Aggregating local scores across prompts reliably identifies the most helpful teacher. 2) Response Selection from a Multiple Teachers: When mixing answers from many teachers, Local Naturalness boosts a 32B student's accuracy on math benchmarks by 9.4pp over global selection, also surpassing the performance achieved by training on data from the single best teacher. These results highlight the power of localized data quality evaluation and data mixing for more effective reasoning distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04398v1">SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted at NeurIPS 2025. Code is available at https://github.com/Buyun-Liang/SECA
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18709v2">Filtering for Creativity: Adaptive Prompting for Multilingual Riddle Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Multilingual riddle generation challenges large language models (LLMs) to balance cultural fluency with creative abstraction. Standard prompting strategies -- zero-shot, few-shot, chain-of-thought -- tend to reuse memorized riddles or perform shallow paraphrasing. We introduce Adaptive Originality Filtering (AOF), a prompting framework that filters redundant generations using cosine-based similarity rejection, while enforcing lexical novelty and cross-lingual fidelity. Evaluated across three LLMs and four language pairs, AOF-enhanced GPT-4o achieves \texttt{0.177} Self-BLEU and \texttt{0.915} Distinct-2 in Japanese, signaling improved lexical diversity and reduced redundancy compared to other prompting methods and language pairs. Our findings show that semantic rejection can guide culturally grounded, creative generation without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04384v1">LLM Based Bayesian Optimization for Prompt Search</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Bayesian Optimization (BO) has been widely used to efficiently optimize expensive black-box functions with limited evaluations. In this paper, we investigate the use of BO for prompt engineering to enhance text classification with Large Language Models (LLMs). We employ an LLM-powered Gaussian Process (GP) as the surrogate model to estimate the performance of different prompt candidates. These candidates are generated by an LLM through the expansion of a set of seed prompts and are subsequently evaluated using an Upper Confidence Bound (UCB) acquisition function in conjunction with the GP posterior. The optimization process iteratively refines the prompts based on a subset of the data, aiming to improve classification accuracy while reducing the number of API calls by leveraging the prediction uncertainty of the LLM-based GP. The proposed BO-LLM algorithm is evaluated on two datasets, and its advantages are discussed in detail in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16889v3">ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge (LLMaaJ) now underpins scalable evaluation, yet we lack a decisive test of a judge's qualification: can it recover a conversation's latent objective and know when that inference is trustworthy? LLMs degrade under irrelevant or long context; multi-turn jailbreaks further hide goals across turns. We introduce ObjexMT, a benchmark for objective extraction and metacognition. Given a multi-turn transcript, a model must return a one-sentence base objective and self-reported confidence. Accuracy is computed via LLM-judge semantic similarity to gold objectives, converted to binary correctness by a human-aligned threshold calibrated on N=300 items (tau = 0.66; F1 = 0.891). Metacognition is evaluated with ECE, Brier, Wrong at High-Confidence (0.80/0.90/0.95), and risk-coverage. Across six models (gpt-4.1, claude-sonnet-4, Qwen3-235B-A22B-FP8, kimi-k2, deepseek-v3.1, gemini-2.5-flash) on three datasets, kimi-k2 attains the highest objective-extraction accuracy (0.612), with claude-sonnet-4 (0.603) and deepseek-v3.1 (0.599) statistically comparable. claude-sonnet-4 yields the best selective risk and calibration (AURC 0.242; ECE 0.206; Brier 0.254). Dataset heterogeneity (16-82 percent accuracy variance) reveals that automated obfuscation poses fundamental challenges beyond model choice. High-confidence errors persist: Wrong at 0.90 ranges from 14.9 percent (claude-sonnet-4) to 47.7 percent (Qwen3-235B-A22B-FP8). ObjexMT provides an actionable test for LLM judges: when objectives are not explicit, judges often misinfer them; we recommend exposing objectives when feasible and gating decisions by confidence otherwise. Data at https://github.com/hyunjun1121/ObjexMT_dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.14275v2">FedMentor: Domain-Aware Differential Privacy for Heterogeneous Federated LLMs in Mental Health</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 NeurIPS 2025 GenAI4Health Workshop
    </div>
    <details class="paper-abstract">
      Privacy-preserving adaptation of Large Language Models (LLMs) in sensitive domains (e.g., mental health) requires balancing strict confidentiality with model utility and safety. We propose FedMentor, a federated fine-tuning framework that integrates Low-Rank Adaptation (LoRA) and domain-aware Differential Privacy (DP) to meet per-domain privacy budgets while maintaining performance. Each client (domain) applies a custom DP noise scale proportional to its data sensitivity, and the server adaptively reduces noise when utility falls below a threshold. In experiments on three mental health datasets, we show that FedMentor improves safety over standard Federated Learning (FL) without privacy, raising safe output rates by up to three points and lowering toxicity, while maintaining utility (BERTScore F1 and ROUGE-L) within 0.5% of the non-private baseline and close to the centralized upper bound. The framework scales to backbones with up to 1.7B parameters on single-GPU clients, requiring < 173 MB of communication per-round. FedMentor demonstrates a practical approach to privately fine-tune LLMs for safer deployments in healthcare and other sensitive fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04373v1">Just-in-time Episodic Feedback Hinter: Leveraging Offline Knowledge to Improve LLM Agents Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents perform well in sequential decision-making tasks, but improving them on unfamiliar domains often requires costly online interactions or fine-tuning on large expert datasets. These strategies are impractical for closed-source models and expensive for open-source ones, with risks of catastrophic forgetting. Offline trajectories offer reusable knowledge, yet demonstration-based methods struggle because raw traces are long, noisy, and tied to specific tasks. We present Just-in-time Episodic Feedback Hinter (JEF Hinter), an agentic system that distills offline traces into compact, context-aware hints. A zooming mechanism highlights decisive steps in long trajectories, capturing both strategies and pitfalls. Unlike prior methods, JEF Hinter leverages both successful and failed trajectories, extracting guidance even when only failure data is available, while supporting parallelized hint generation and benchmark-independent prompting. At inference, a retriever selects relevant hints for the current state, providing targeted guidance with transparency and traceability. Experiments on MiniWoB++, WorkArena-L1, and WebArena-Lite show that JEF Hinter consistently outperforms strong baselines, including human- and document-based hints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10320v2">J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 10 pages, 13 tables, 14 figures
    </div>
    <details class="paper-abstract">
      The progress of AI is bottlenecked by the quality of evaluation, making powerful LLM-as-a-Judge models a core solution. The efficacy of these judges depends on their chain-of-thought reasoning, creating a critical need for methods that can effectively optimize this reasoning process. In this work, we introduce J1, a reinforcement learning framework for teaching LLM judges to think before making decisions. Our core contribution lies in converting all judgment tasks for non-verifiable and verifiable prompts into a unified format with verifiable rewards, enabling direct optimization of evaluation quality while mitigating positional bias. We then use RL to train thinking-judges at scales of 8B, 32B, and 70B and show that they obtain state-of-the-art performance across multiple benchmarks. In particular, J1-Qwen-32B, our multitasked pointwise and pairwise judge also outperforms o1-mini, o3, and a much larger 671B DeepSeek-R1 on some benchmarks, while only training on synthetic data. Through comprehensive ablations of pairwise, pointwise, and multitask J1 variants, we demonstrate the effectiveness of our approach across seed prompts, reward strategies, and training recipes. Qualitative analysis reveals that J1 develops systematic evaluation strategies, including dynamic criteria generation, reference answer creation, iterative self-correction of initial assessments, and feedback generation for low-quality responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04320v1">Read the Scene, Not the Script: Outcome-Aware Safety for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Safety-aligned Large Language Models (LLMs) still show two dominant failure modes: they are easily jailbroken, or they over-refuse harmless inputs that contain sensitive surface signals. We trace both to a common cause: current models reason weakly about links between actions and outcomes and over-rely on surface-form signals, lexical or stylistic cues that do not encode consequences. We define this failure mode as Consequence-blindness. To study consequence-blindness, we build a benchmark named CB-Bench covering four risk scenarios that vary whether semantic risk aligns with outcome risk, enabling evaluation under both matched and mismatched conditions which are often ignored by existing safety benchmarks. Mainstream models consistently fail to separate these risks and exhibit consequence-blindness, indicating that consequence-blindness is widespread and systematic. To mitigate consequence-blindness, we introduce CS-Chain-4k, a consequence-reasoning dataset for safety alignment. Models fine-tuned on CS-Chain-4k show clear gains against semantic-camouflage jailbreaks and reduce over-refusal on harmless inputs, while maintaining utility and generalization on other benchmarks. These results clarify the limits of current alignment, establish consequence-aware reasoning as a core alignment goal and provide a more practical and reproducible evaluation path.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04317v1">FairAgent: Democratizing Fairness-Aware Machine Learning with LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted by ICDM 2025 Demo Workshop
    </div>
    <details class="paper-abstract">
      Training fair and unbiased machine learning models is crucial for high-stakes applications, yet it presents significant challenges. Effective bias mitigation requires deep expertise in fairness definitions, metrics, data preprocessing, and machine learning techniques. In addition, the complex process of balancing model performance with fairness requirements while properly handling sensitive attributes makes fairness-aware model development inaccessible to many practitioners. To address these challenges, we introduce FairAgent, an LLM-powered automated system that significantly simplifies fairness-aware model development. FairAgent eliminates the need for deep technical expertise by automatically analyzing datasets for potential biases, handling data preprocessing and feature engineering, and implementing appropriate bias mitigation strategies based on user requirements. Our experiments demonstrate that FairAgent achieves significant performance improvements while significantly reducing development time and expertise requirements, making fairness-aware machine learning more accessible to practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04311v1">On the Importance of Task Complexity in Evaluating LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language model multi-agent systems (LLM-MAS) offer a promising paradigm for harnessing collective intelligence to achieve more advanced forms of AI behaviour. While recent studies suggest that LLM-MAS can outperform LLM single-agent systems (LLM-SAS) on certain tasks, the lack of systematic experimental designs limits the strength and generality of these conclusions. We argue that a principled understanding of task complexity, such as the degree of sequential reasoning required and the breadth of capabilities involved, is essential for assessing the effectiveness of LLM-MAS in task solving. To this end, we propose a theoretical framework characterising tasks along two dimensions: depth, representing reasoning length, and width, representing capability diversity. We theoretically examine a representative class of LLM-MAS, namely the multi-agent debate system, and empirically evaluate its performance in both discriminative and generative tasks with varying depth and width. Theoretical and empirical results show that the benefit of LLM-MAS over LLM-SAS increases with both task depth and width, and the effect is more pronounced with respect to depth. This clarifies when LLM-MAS are beneficial and provides a principled foundation for designing future LLM-MAS methods and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04303v1">Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 8 pages, 0 figures
    </div>
    <details class="paper-abstract">
      Multi-agent deployments of large language models (LLMs) are increasingly embedded in market, allocation, and governance workflows, yet covert coordination among agents can silently erode trust and social welfare. Existing audits are dominated by heuristics that lack theoretical guarantees, struggle to transfer across tasks, and seldom ship with the infrastructure needed for independent replication. We introduce \emph{Audit the Whisper}, a conference-grade research artifact that spans theory, benchmark design, detection, and reproducibility. Our contributions are: (i) a channel-capacity analysis showing how interventions such as paraphrase, rate limiting, and role permutation impose quantifiable capacity penalties -- operationalized via paired-run Kullback--Leibler diagnostics -- that tighten mutual-information thresholds with finite-sample guarantees; (ii) \textsc{ColludeBench}-v0, covering pricing, first-price auctions, and peer review with configurable covert schemes, deterministic manifests, and reward instrumentation; and (iii) a calibrated auditing pipeline that fuses cross-run mutual information, permutation invariance, watermark variance, and fairness-aware acceptance bias, each tuned to a \(10^{-3}\) false-positive budget. Across 600 audited runs spanning 12 intervention conditions, the union meta-test attains TPR~$=1$ with zero observed false alarms, while ablations surface the price-of-auditing trade-off and highlight fairness-driven colluders invisible to MI alone. We release regeneration scripts, seed-stamped manifests, and documentation so that external auditors can reproduce every figure and extend the framework with minimal effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10659v7">Network Formation and Dynamics Among Multi-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted at PNAS Nexus
    </div>
    <details class="paper-abstract">
      Social networks profoundly influence how humans form opinions, exchange information, and organize collectively. As large language models (LLMs) are increasingly embedded into social and professional environments, it is critical to understand whether their interactions approximate human-like network dynamics. We develop a framework to study the network formation behaviors of multiple LLM agents and benchmark them against human decisions. Across synthetic and real-world settings, including friendship, telecommunication, and employment networks, we find that LLMs consistently reproduce fundamental micro-level principles such as preferential attachment, triadic closure, and homophily, as well as macro-level properties including community structure and small-world effects. Importantly, the relative emphasis of these principles adapts to context: for example, LLMs favor homophily in friendship networks but heterophily in organizational settings, mirroring patterns of social mobility. A controlled human-subject survey confirms strong alignment between LLMs and human participants in link-formation decisions. These results establish that LLMs can serve as powerful tools for social simulation and synthetic data generation, while also raising critical questions about bias, fairness, and the design of AI systems that participate in human networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18886v2">On Pruning State-Space LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Recent work proposed state-space models (SSMs) as an efficient alternative to transformer-based LLMs. Can these models be pruned to further reduce their computation costs? We adapt several pruning methods to the SSM structure, and apply them to four SSM-based LLMs across multiple tasks. We find that such models are quite robust to some pruning methods (e.g. WANDA), while using other methods lead to fast performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04274v1">Selecting Cybersecurity Requirements: Effects of LLM Use and Professional Software Development Experience</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 5 pages, 1 figure, 2 tables, presented at IARIA CYBER 2025
    </div>
    <details class="paper-abstract">
      This study investigates how access to Large Language Models (LLMs) and varying levels of professional software development experience affect the prioritization of cybersecurity requirements for web applications. Twenty-three postgraduate students participated in a research study to prioritize security requirements (SRs) using the MoSCoW method and subsequently rated their proposed solutions against multiple evaluation criteria. We divided participants into two groups (one with and the other without access to LLM support during the task). Results showed no significant differences related to LLM use, suggesting that access to LLMs did not noticeably influence how participants evaluated cybersecurity solutions. However, statistically significant differences emerged between experience groups for certain criteria, such as estimated cost to develop a feature, perceived impact on user experience, and risk assessment related to non-implementation of the proposed feature. Participants with more professional experience tended to provide higher ratings for user experience impact and lower risk estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04261v1">VortexPIA: Indirect Prompt Injection Attack against LLMs for Efficient Extraction of User Privacy</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed in Conversational AIs (CAIs), while exposing privacy and security threats. Recent research shows that LLM-based CAIs can be manipulated to extract private information from human users, posing serious security threats. However, the methods proposed in that study rely on a white-box setting that adversaries can directly modify the system prompt. This condition is unlikely to hold in real-world deployments. The limitation raises a critical question: can unprivileged attackers still induce such privacy risks in practical LLM-integrated applications? To address this question, we propose \textsc{VortexPIA}, a novel indirect prompt injection attack that induces privacy extraction in LLM-integrated applications under black-box settings. By injecting token-efficient data containing false memories, \textsc{VortexPIA} misleads LLMs to actively request private information in batches. Unlike prior methods, \textsc{VortexPIA} allows attackers to flexibly define multiple categories of sensitive data. We evaluate \textsc{VortexPIA} on six LLMs, covering both traditional and reasoning LLMs, across four benchmark datasets. The results show that \textsc{VortexPIA} significantly outperforms baselines and achieves state-of-the-art (SOTA) performance. It also demonstrates efficient privacy requests, reduced token consumption, and enhanced robustness against defense mechanisms. We further validate \textsc{VortexPIA} on multiple realistic open-source LLM-integrated applications, demonstrating its practical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01694v2">Student-AI Interaction in an LLM-Empowered Learning Environment: A Cluster Analysis of Engagement Profiles</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 15 pages, 10 figures; Reported on ECER 2025
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into educational practice enables personalized learning by accommodating diverse learner behaviors. This study explored diverse learner profiles within a multi-agent, LLM-empowered learning environment. Data was collected from 312 undergraduate students at a university in China as they participated in a six-module course. Based on hierarchical cluster analyses of system profiles and student-AI interactive dialogues, we found that students exhibit varied behavioral, cognitive, and emotional engagement tendencies. This analysis allowed us to identify two types of dropouts (early dropouts and stagnating interactors) and three completer profiles (active questioners, responsive navigators, and lurkers). The results showed that high levels of interaction do not always equate to productive learning and vice versa. Prior knowledge significantly influenced interaction patterns and short-term learning benefits. Further analysis of the human-AI dialogues revealed that some students actively engaged in knowledge construction, while others displayed a high frequency of regulatory behaviors. Notably, both groups of students achieved comparable learning gains, demonstrating the effectiveness of the multi-agent learning environment in supporting personalized learning. These results underscore the complex and multifaceted nature of engagement in human-AI collaborative learning and provide practical implications for the design of adaptive educational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04214v1">Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment frm Heterogeneous Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-05
    </div>
    <details class="paper-abstract">
      We study deploying large language models (LLMs) as business development (BD) agents for persuasive price negotiation in online travel agencies (OTAs), where aligning traveler affordability and hotel profitability directly affects bookings, partner relationships, and access to travel. The agent must follow a Standard Operating Procedure (SOP) while conducting multi-turn persuasion, interpreting colloquial inputs, and adhering to guardrails (no over-promising, no hallucinations). Conventional post-training -- supervised fine-tuning (SFT) or single-source reward optimization -- overfits scripts, misses nuanced persuasive style, and fails to enforce verifiable business constraints. We propose Reward-Enhanced Policy Optimization (REPO), a reinforcement learning post-training framework that aligns an LLM with heterogeneous rewards: a preference-trained reward model (RM) for dense human alignment, a reward judge (RJ) for high-level persuasive behavior and SOP compliance, and programmatic reward functions (RF) for deterministic checks on numerics, formatting, and guardrails. A straightforward enhancement mechanism is proposed to combine the RM with RJ and RF signals to curb reward hacking and improve negotiation quality. In production-style evaluations -- approximately 150 turns from real dialogues and 225 turns from curated bad-case dialogues -- REPO lifts average dialogue rating to 4.63: +1.20 over base, +0.83 over Direct Preference Optimization (DPO); +0.33 over Group Relative Policy Optimization (GRPO), increases the share of conversations with at least one excellent response to 66.67% (+23.34 percentage points over GRPO), and achieves a 93.33% bad-case fix rate with 75.56% clean fixes, outperforming SFT, DPO, PPO, and GRPO. We also observe emergent capabilities -- proactive empathy, localized reasoning, calibrated tactics -- that surpass gold annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12891v3">TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-05
      | 💬 Accepted by NeurIPS 2025 (Spotlight)
    </div>
    <details class="paper-abstract">
      Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME , and the project page link is https://sylvain-wei.github.io/TIME/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10246v2">Detecting LLM Hallucination Through Layer-wise Information Deficiency: Analysis of Ambiguous Prompts and Unanswerable Questions</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Accepted to EMNLP(main)2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently generate confident yet inaccurate responses, introducing significant risks for deployment in safety-critical domains. We present a novel, test-time approach to detecting model hallucination through systematic analysis of information flow across model layers. We target cases when LLMs process inputs with ambiguous or insufficient context. Our investigation reveals that hallucination manifests as usable information deficiencies in inter-layer transmissions. While existing approaches primarily focus on final-layer output analysis, we demonstrate that tracking cross-layer information dynamics ($\mathcal{L}$I) provides robust indicators of model reliability, accounting for both information gain and loss during computation. $\mathcal{L}$I integrates easily with pretrained LLMs without requiring additional training or architectural modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19030v4">RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly expected to tackle complex tasks, driven by their expanding applications and users' growing proficiency in crafting sophisticated prompts. However, as the number of explicitly stated requirements increases (particularly more than 10 constraints), LLMs often struggle to accurately follow such complex instructions, which limits their applicability in complex real-world scenarios. To the best of our knowledge, existing datasets do not exceed 10 constraints per instance. To address this challenge, we propose RECAST, an efficient and scalable framework for synthesizing datasets where each example incorporates far more constraints than those in existing benchmarks, aiming to challenge and extend the boundaries of models' ability to follow complex instructions. These constraints are extracted from real-world prompt-response pairs to ensure practical relevance. Using this framework, we construct RECAST-30K, a large-scale, high-quality dataset comprising 30k instances spanning 19 constraint types. Experimental results demonstrate that models finetuned on RECAST-30K substantially improve in following complex instructions while maintaining their general capabilities without degradation. Moreover, RECAST enables automatic verification of constraint satisfaction via rule-based validators for quantitative constraints and LLM-based validators for qualitative ones; the verifiability provided by RECAST enables the design of reward functions for reinforcement learning, which further boosts model performance on complex and challenging tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.26184v3">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 ECIR 2025 demo format
    </div>
    <details class="paper-abstract">
      Generation of long-form, citation-backed reports is a primary use case for retrieval augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, ones tailored to report generation are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recent ARGUE framework for report generation evaluation. We present analysis of Auto-ARGUE on the report generation pilot task from the TREC 2024 NeuCLIR track, showing good system-level correlations with human judgments. We further release a web app for visualization of Auto-ARGUE outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03795v1">Investigating LLM Variability in Personalized Conversational Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 11 pages, 5 figures, SIGIR-AP'25 Proceedings of the 2025 Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP 2025), December 7--10, 2025, Xi'an, China
    </div>
    <details class="paper-abstract">
      Personalized Conversational Information Retrieval (CIR) has seen rapid progress in recent years, driven by the development of Large Language Models (LLMs). Personalized CIR aims to enhance document retrieval by leveraging user-specific information, such as preferences, knowledge, or constraints, to tailor responses to individual needs. A key resource for this task is the TREC iKAT 2023 dataset, designed to evaluate personalization in CIR pipelines. Building on this resource, Mo et al. explored several strategies for incorporating Personal Textual Knowledge Bases (PTKB) into LLM-based query reformulation. Their findings suggested that personalization from PTKBs could be detrimental and that human annotations were often noisy. However, these conclusions were based on single-run experiments using the GPT-3.5 Turbo model, raising concerns about output variability and repeatability. In this reproducibility study, we rigorously reproduce and extend their work, focusing on LLM output variability and model generalization. We apply the original methods to the new TREC iKAT 2024 dataset and evaluate a diverse range of models, including Llama (1B-70B), Qwen-7B, GPT-4o-mini. Our results show that human-selected PTKBs consistently enhance retrieval performance, while LLM-based selection methods do not reliably outperform manual choices. We further compare variance across datasets and observe higher variability on iKAT than on CAsT, highlighting the challenges of evaluating personalized CIR. Notably, recall-oriented metrics exhibit lower variance than precision-oriented ones, a critical insight for first-stage retrievers. Finally, we underscore the need for multi-run evaluations and variance reporting when assessing LLM-based CIR systems. By broadening evaluation across models, datasets, and metrics, our study contributes to more robust and generalizable practices for personalized CIR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10601v4">When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Published in Reliable ML from Unreliable Data workshop @ NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to LACE, with success rates on gpt-oss-20b escalating from 60% under ACE to 72% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01688v2">Format Inertia: A Failure Mechanism of LLMs in Medical Pre-Consultation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 EMNLP 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have brought significant improvements to various service domains, including chatbots and medical pre-consultation applications. In the healthcare domain, the most common approach for adapting LLMs to multi-turn dialogue generation is Supervised Fine-Tuning (SFT). However, datasets for SFT in tasks like medical pre-consultation typically exhibit a skewed turn-count distribution. Training on such data induces a novel failure mechanism we term Format Inertia, where models tend to generate repetitive, format-correct, but diagnostically uninformative questions in long medical dialogues. To mitigate this observed failure mechanism, we adopt a simple, data-centric method that rebalances the turn-count distribution of the training dataset. Experimental results show that our approach substantially alleviates Format Inertia in medical pre-consultation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03762v1">Prompt Balance Matters: Understanding How Imbalanced Few-Shot Learning Affects Multilingual Sense Disambiguation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Paper accepted at GlobalNLP 2025: Workshop on beyond English: Natural Language Processing for All Languages in an Era of Large Language Models" 9 pages, 3 figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have significantly reshaped the landscape of Natural Language Processing (NLP). Among the various prompting techniques, few-shot prompting has gained considerable attention for its practicality and effectiveness. This study investigates how few-shot prompting strategies impact the Word Sense Disambiguation (WSD) task, particularly focusing on the biases introduced by imbalanced sample distributions. We use the GLOSSGPT prompting method, an advanced approach for English WSD, to test its effectiveness across five languages: English, German, Spanish, French, and Italian. Our results show that imbalanced few-shot examples can cause incorrect sense predictions in multilingual languages, but this issue does not appear in English. To assess model behavior, we evaluate both the GPT-4o and LLaMA-3.1-70B models and the results highlight the sensitivity of multilingual WSD to sample distribution in few-shot settings, emphasizing the need for balanced and representative prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10127v2">Population-Aligned Persona Generation for LLM-based Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled human-like social simulations at unprecedented scale and fidelity, offering new opportunities for computational social science. A key challenge, however, is the construction of persona sets that authentically represent the diversity and distribution of real-world populations. Most existing LLM-based social simulation studies focus primarily on designing agentic frameworks and simulation environments, often overlooking the complexities of persona generation and the potential biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality traits. To address the needs of specific simulation contexts, we further introduce a task-specific module that adapts the globally aligned persona set to targeted subpopulations. Extensive experiments demonstrate that our method significantly reduces population-level bias and enables accurate, flexible social simulation for a wide range of research and policy applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25694v2">HNote: Extending YNote with Hexadecimal Encoding for Fine-Tuning LLMs in Music Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have created new opportunities for symbolic music generation. However, existing formats such as MIDI, ABC, and MusicXML are either overly complex or structurally inconsistent, limiting their suitability for token-based learning architectures. To address these challenges, we propose HNote, a novel hexadecimal-based notation system extended from YNote, which encodes both pitch and duration within a fixed 32-unit measure framework. This design ensures alignment, reduces ambiguity, and is directly compatible with LLM architectures. We converted 12,300 Jiangnan-style songs generated from traditional folk pieces from YNote into HNote, and fine-tuned LLaMA-3.1(8B) using parameter-efficient LoRA. Experimental results show that HNote achieves a syntactic correctness rate of 82.5%, and BLEU and ROUGE evaluations demonstrate strong symbolic and structural similarity, producing stylistically coherent compositions. This study establishes HNote as an effective framework for integrating LLMs with cultural music modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03719v1">A Survey of LLM-Based Applications in Programming Education: Balancing Automation and Human Oversight</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 2025 EMNLP HCI+NLP Workshop Short Paper
    </div>
    <details class="paper-abstract">
      Novice programmers benefit from timely, personalized support that addresses individual learning gaps, yet the availability of instructors and teaching assistants is inherently limited. Large language models (LLMs) present opportunities to scale such support, though their effectiveness depends on how well technical capabilities are aligned with pedagogical goals. This survey synthesizes recent work on LLM applications in programming education across three focal areas: formative code feedback, assessment, and knowledge modeling. We identify recurring design patterns in how these tools are applied and find that interventions are most effective when educator expertise complements model output through human-in-the-loop oversight, scaffolding, and evaluation. Fully automated approaches are often constrained in capturing the pedagogical nuances of programming education, although human-in-the-loop designs and course specific adaptation offer promising directions for future improvement. Future research should focus on improving transparency, strengthening alignment with pedagogy, and developing systems that flexibly adapt to the needs of varied learning contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17601v5">Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Recent studies have widely investigated backdoor attacks on Large Language Models (LLMs) by inserting harmful question-answer (QA) pairs into their training data. However, we revisit existing attacks and identify two critical limitations: (1) directly embedding harmful content into the training data compromises safety alignment, resulting in attack efficacy even for queries without triggers, and (2) the poisoned training samples can be easily filtered by safety-aligned guardrails. To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, a malicious query with the trigger is input to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Achieving this using only clean samples is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment, and design a robust and general benign response template for constructing better poisoning data. To further enhance the attack, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method successfully injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21947v2">Active Attacks: Red-teaming LLMs via Adaptive Environments</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 22 pages, 7 figures, 18 tables
    </div>
    <details class="paper-abstract">
      We address the challenge of generating diverse attack prompts for large language models (LLMs) that elicit harmful behaviors (e.g., insults, sexual content) and are used for safety fine-tuning. Rather than relying on manual prompt engineering, attacker LLMs can be trained with reinforcement learning (RL) to automatically generate such prompts using only a toxicity classifier as a reward. However, capturing a wide range of harmful behaviors is a significant challenge that requires explicit diversity objectives. Existing diversity-seeking RL methods often collapse to limited modes: once high-reward prompts are found, exploration of new regions is discouraged. Inspired by the active learning paradigm that encourages adaptive exploration, we introduce \textit{Active Attacks}, a novel RL-based red-teaming algorithm that adapts its attacks as the victim evolves. By periodically safety fine-tuning the victim LLM with collected attack prompts, rewards in exploited regions diminish, which forces the attacker to seek unexplored vulnerabilities. This process naturally induces an easy-to-hard exploration curriculum, where the attacker progresses beyond easy modes toward increasingly difficult ones. As a result, Active Attacks uncovers a wide range of local attack modes step by step, and their combination achieves wide coverage of the multi-mode distribution. Active Attacks, a simple plug-and-play module that seamlessly integrates into existing RL objectives, unexpectedly outperformed prior RL-based methods -- including GFlowNets, PPO, and REINFORCE -- by improving cross-attack success rates against GFlowNets, the previous state-of-the-art, from 0.07% to 31.28% (a relative gain greater than $400\ \times$) with only a 6% increase in computation. Our code is publicly available \href{https://github.com/dbsxodud-11/active_attacks}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00714v2">RFCAudit: An LLM Agent for Functional Bug Detection in Network Protocols</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Functional correctness is critical for ensuring the reliability and security of network protocol implementations. Functional bugs, instances where implementations diverge from behaviors specified in RFC documents, can lead to severe consequences, including faulty routing, authentication bypasses, and service disruptions. Detecting these bugs requires deep semantic analysis across specification documents and source code, a task beyond the capabilities of traditional static analysis tools. This paper introduces RFCAudit, an autonomous agent that leverages large language models (LLMs) to detect functional bugs by checking conformance between network protocol implementations and their RFC specifications. Inspired by the human auditing procedure, RFCAudit comprises two key components: an indexing agent and a detection agent. The former hierarchically summarizes protocol code semantics, generating semantic indexes that enable the detection agent to narrow down the scanning scope. The latter employs demand-driven retrieval to iteratively collect additional relevant data structures and functions, eventually identifying potential inconsistencies with the RFC specifications effectively. We evaluate RFCAudit across six real-world network protocol implementations. RFCAudit identifies 47 functional bugs with 81.9% precision, of which 20 bugs have been confirmed or fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03687v1">MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Medical problem solving demands expert knowledge and intricate reasoning. Recent studies of large language models (LLMs) attempt to ease this complexity by introducing external knowledge verification through retrieval-augmented generation or by training on reasoning datasets. However, these approaches suffer from drawbacks such as retrieval overhead and high annotation costs, and they heavily rely on substituted external assistants to reach limited performance in medical field. In this paper, we introduce MedReflect, a generalizable framework designed to inspire LLMs with a physician-like reflective thinking mode. MedReflect generates a single-pass reflection chain that includes initial hypothesis generation, self-questioning, self-answering and decision refinement. This self-verified and self-reflective nature releases large language model's latent capability in medical problem-solving without external retrieval or heavy annotation. We demonstrate that MedReflect enables cost-efficient medical dataset construction: with merely 2,000 randomly sampled training examples and a light fine-tuning, this approach achieves notable absolute accuracy improvements across a series of medical benchmarks while cutting annotation requirements. Our results provide evidence that LLMs can learn to solve specialized medical problems via self-reflection and self-improve, reducing reliance on external supervision and extensive task-specific fine-tuning data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03667v1">Invisible Saboteurs: Sycophantic LLMs Mislead Novices in Problem-Solving Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Sycophancy, the tendency of LLM-based chatbots to express excessive enthusiasm, agreement, flattery, and a lack of disagreement, is emerging as a significant risk in human-AI interactions. However, the extent to which this affects human-LLM collaboration in complex problem-solving tasks is not well quantified, especially among novices who are prone to misconceptions. We created two LLM chatbots, one with high sycophancy and one with low sycophancy, and conducted a within-subjects experiment (n=24) in the context of debugging machine learning models to isolate the effect of LLM sycophancy on users' mental models, their workflows, reliance behaviors, and their perceptions of the chatbots. Our findings show that users of the high sycophancy chatbot were less likely to correct their misconceptions and spent more time over-relying on unhelpful LLM responses. Despite these impaired outcomes, a majority of users were unable to detect the presence of excessive sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03662v1">Operationalizing Data Minimization for Privacy-Preserving LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      The rapid deployment of large language models (LLMs) in consumer applications has led to frequent exchanges of personal information. To obtain useful responses, users often share more than necessary, increasing privacy risks via memorization, context-based personalization, or security breaches. We present a framework to formally define and operationalize data minimization: for a given user prompt and response model, quantifying the least privacy-revealing disclosure that maintains utility, and we propose a priority-queue tree search to locate this optimal point within a privacy-ordered transformation space. We evaluated the framework on four datasets spanning open-ended conversations (ShareGPT, WildChat) and knowledge-intensive tasks with single-ground-truth answers (CaseHold, MedQA), quantifying achievable data minimization with nine LLMs as the response model. Our results demonstrate that larger frontier LLMs can tolerate stronger data minimization while maintaining task quality than smaller open-source models (85.7% redaction for GPT-5 vs. 19.3% for Qwen2.5-0.5B). By comparing with our search-derived benchmarks, we find that LLMs struggle to predict optimal data minimization directly, showing a bias toward abstraction that leads to oversharing. This suggests not just a privacy gap, but a capability gap: models may lack awareness of what information they actually need to solve a task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.03688v3">AgentBench: Evaluating LLMs as Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Published in ICLR 2024
    </div>
    <details class="paper-abstract">
      The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Thus, there is an urgent need to quantitatively \textit{evaluate LLMs as agents} on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional benchmark that consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities. Our extensive test over \num API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and many OSS competitors that are no larger than 70B. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Improving instruction following and training on high quality multi-round alignment data could improve agent performance. And different from existing assumptions, training on code present ambivalent impacts on different agent tasks. Datasets, environments, and an integrated evaluation package for AgentBench are released at https://github.com/THUDM/AgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05370v2">Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained immense success in revolutionizing various applications, including content generation, search and recommendation, and AI-assisted operation. To reduce high training costs, Mixture-of-Experts (MoE) architecture has become a popular backbone for modern LLMs. However, despite the benefits, serving MoE-based LLMs experience severe memory inefficiency due to sparsely activated experts. Recent studies propose to offload inactive experts from GPU memory to CPU memory to improve the serving efficiency of MoE models. However, they either incur high inference latency or high model memory footprints due to coarse-grained designs. To tame the latency-memory trade-off in MoE serving, we present FineMoE, a fine-grained expert offloading system for MoE serving that achieves low inference latency with memory efficiency. We design FineMoE to extract fine-grained expert selection patterns from MoE models and semantic hints from input prompts to efficiently guide expert prefetching, caching, and offloading decisions. FineMoE is prototyped on top of HuggingFace Transformers and deployed on a six-GPU testbed. Experiments with open-source MoE models and real-world workloads show that FineMoE reduces inference latency by 47% and improves expert hit rate by 39% over state-of-the-art solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03650v1">LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Low-discrepancy point sets and digital sequences underpin quasi-Monte Carlo (QMC) methods for high-dimensional integration. We cast two long-standing QMC design problems as program synthesis and solve them with an LLM-guided evolutionary loop that mutates and selects code under task-specific fitness: (i) constructing finite 2D/3D point sets with low star discrepancy, and (ii) choosing Sobol' direction numbers that minimize randomized QMC error on downstream integrands. Our two-phase procedure combines constructive code proposals with iterative numerical refinement. On finite sets, we rediscover known optima in small 2D cases and set new best-known 2D benchmarks for N >= 40, while matching most known 3D optima up to the proven frontier (N <= 8) and reporting improved 3D benchmarks beyond. On digital sequences, evolving Sobol' parameters yields consistent reductions in randomized quasi-Monte Carlo (rQMC) mean-squared error for several 32-dimensional option-pricing tasks relative to widely used Joe--Kuo parameters, while preserving extensibility to any sample size and compatibility with standard randomizations. Taken together, the results demonstrate that LLM-driven evolutionary program synthesis can automate the discovery of high-quality QMC constructions, recovering classical designs where they are optimal and improving them where finite-N structure matters. Data and code are available at https://github.com/hockeyguy123/openevolve-star-discrepancy.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24836v3">Pushing LLMs to Their Logical Reasoning Bound: The Role of Data Reasoning Intensity</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) highlight the importance of training data structure and quality in shaping reasoning behavior. However, most existing approaches focus on transforming data formats while neglecting the internal reasoning complexity of training samples, leaving the reasoning potential of data under-explored and underutilized. In this work, we posit that LLM logical reasoning performance is jointly constrained by the potential of the training data and the cognitive capacity of the model. To make this relationship measurable, we introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures. This allows us to analyze how well current LLMs utilize logical reasoning signals and identify performance gaps relative to data potential. Based on this insight, we introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training data. Rather than increasing data volume, our method re-optimizes existing samples to better align with the LLM's logical reasoning boundary. Extensive experiments show that our approach significantly improves performance and generalization over data-centric strategies. We further validate our method under a reinforcement learning framework. Our results indicate that prioritizing reasoning complexity in data rather than sheer scale or superficial form is essential to realizing LLMs' full cognitive potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03641v1">Generating High-Level Test Cases from Requirements using LLM: An Industry Study</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 11pages
    </div>
    <details class="paper-abstract">
      Currently, generating high-level test cases described in natural language from requirement documents is performed manually. In the industry, including companies specializing in software testing, there is a significant demand for the automatic generation of high-level test cases from requirement documents using Large Language Models (LLMs). Efforts to utilize LLMs for requirement analysis are underway. In some cases, retrieval-augmented generation (RAG) is employed for generating high-level test cases using LLMs. However, in practical applications, it is necessary to create a RAG tailored to the knowledge system of each specific application, which is labor-intensive. Moreover, when applying high-level test case generation as a prompt, there is no established method for instructing the generation of high-level test cases at a level applicable to other specifications without using RAG. It is required to establish a method for the automatic generation of high-level test cases that can be generalized across a wider range of requirement documents. In this paper, we propose a method for generating high-level (GHL) test cases from requirement documents using only prompts, without creating RAGs. In the proposed method, first, the requirement document is input into the LLM to generate test design techniques corresponding to the requirement document. Then, high-level test cases are generated for each of the generated test design techniques. Furthermore, we verify an evaluation method based on semantic similarity of the generated high-level test cases. In the experiments, we confirmed the method using datasets from Bluetooth and Mozilla, where requirement documents and high-level test cases are available, achieving macro-recall measurement of 0.81 and 0.37, respectively. We believe that the method is feasible for practical application in generating high-level test cases without using RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14794v2">Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      With the rapid advancement of artificial intelligence technologies such as ChatGPT, AI agents, and video generation, contemporary mobile systems have begun integrating these AI capabilities on local devices to enhance privacy and reduce response latency. To meet the computational demands of AI tasks, current mobile SoCs are equipped with diverse AI accelerators, including GPUs and Neural Processing Units (NPUs). However, there has not been a comprehensive characterization of these heterogeneous processors, and existing designs typically only leverage a single AI accelerator for LLM inference, leading to suboptimal use of computational resources and memory bandwidth. In this paper, we first summarize key performance characteristics of heterogeneous processors, SoC memory bandwidth, etc. Drawing on these observations, we propose different heterogeneous parallel mechanisms to fully exploit both GPU and NPU computational power and memory bandwidth. We further design a fast synchronization mechanism between heterogeneous processors that leverages the unified memory architecture. By employing these techniques, we present HeteroInfer, the fastest LLM inference engine in mobile devices which supports GPU-NPU heterogeneous execution. Evaluation shows that HeteroInfer delivers a 1.34x to 6.02x end-to-end speedup over state-of-the-art GPU-only and NPU-only LLM engines, while maintaining negligible interference with other applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03632v1">MITS: Enhanced Tree Search Reasoning for LLMs via Pointwise Mutual Information</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Tree search has become as a representative framework for test-time reasoning with large language models (LLMs), exemplified by methods such as Tree-of-Thought and Monte Carlo Tree Search that explore multiple reasoning paths. However, it remains difficult to provide instant and reliable quantitative assessments of intermediate reasoning step quality, and extensive path exploration is computationally costly. To address this, we propose Mutual Information Tree Search (MITS), a novel framework that guides reasoning with information-theoretic principles. MITS introduces an effective scoring function based on pointwise mutual information (PMI), which enables step-wise evaluation of reasoning paths and search tree expansion via beam search without expensive look-ahead simulations, achieving superior reasoning performances while maintaining computational efficiency. The framework is complemented by an entropy-based dynamic sampling strategy that adaptively allocates computational resources to uncertain reasoning steps where exploration is most beneficial. For final prediction, MITS employs a weighted voting scheme that combines PMI scores with prediction consensus. Through comprehensive experiments on diverse reasoning benchmarks, MITS consistently surpasses baseline methods, establishing a principled and efficient framework for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03645v2">Graph Generation Powered with LLMs for Boosting Multivariate Time-Series Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Sourced from multiple sensors and organized chronologically, Multivariate Time-Series (MTS) data involves crucial spatial-temporal dependencies. To capture these dependencies, Graph Neural Networks (GNNs) have emerged as powerful tools. As explicit graphs are not inherent to MTS data, graph generation becomes a critical first step in adapting GNNs to this domain. However, existing approaches often rely solely on the data itself for MTS graph generation, leaving them vulnerable to biases from small training datasets. This limitation hampers their ability to construct effective graphs, undermining the accurate modeling of underlying dependencies in MTS data and reducing GNN performance in this field. To address this challenge, we propose a novel framework, K-Link, leveraging the extensive universal knowledge encoded in Large Language Models (LLMs) to reduce biases for powered MTS graph generation. To harness the knowledge within LLMs, such as physical principles, we design and extract a \textit{Knowledge-Link graph} that captures universal knowledge of sensors and their linkage. To empower MTS graph generation with the knowledge-link graph, we further introduce a graph alignment module that transfers universal knowledge from the knowledge-link graph to the graph generated from MTS data. This enhances the MTS graph quality, ensuring effective representation learning for MTS data. Extensive experiments demonstrate the efficacy of K-Link for superior performance on various MTS tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20293v2">When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We released our code and dataset at https://github.com/penfever/judgment-to-noise
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03611v1">Can an LLM Induce a Graph? Investigating Memory Drift and Context Length</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 2025 IEEE International Conference on Knowledge Graph (ICKG)
    </div>
    <details class="paper-abstract">
      Recently proposed evaluation benchmarks aim to characterize the effective context length and the forgetting tendencies of large language models (LLMs). However, these benchmarks often rely on simplistic 'needle in a haystack' retrieval or continuation tasks that may not accurately reflect the performance of these models in information-dense scenarios. Thus, rather than simple next token prediction, we argue for evaluating these models on more complex reasoning tasks that requires them to induce structured relational knowledge from the text - such as graphs from potentially noisy natural language content. While the input text can be viewed as generated in terms of a graph, its structure is not made explicit and connections must be induced from distributed textual cues, separated by long contexts and interspersed with irrelevant information. Our findings reveal that LLMs begin to exhibit memory drift and contextual forgetting at much shorter effective lengths when tasked with this form of relational reasoning, compared to what existing benchmarks suggest. With these findings, we offer recommendations for the optimal use of popular LLMs for complex reasoning tasks. We further show that even models specialized for reasoning, such as OpenAI o1, remain vulnerable to early memory drift in these settings. These results point to significant limitations in the models' ability to abstract structured knowledge from unstructured input and highlight the need for architectural adaptations to improve long-range reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18350v2">How Many Parameters Does Your Task Really Need? Task Specific Pruning with LLM-Sieve</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed for narrow tasks in resource-constrained settings, a central question arises: how much of an LLM is truly necessary for a given task? We present LLM-Sieve, a framework that prunes LLMs down to the minimal parameter subset needed to preserve task performance. Our approach introduces two innovations: (i) output-aligned non-orthogonal projections, which yield more faithful low-rank approximations than traditional PCA/SVD by aligning directly with layer outputs; and (ii) adaptive pruning via a Genetic Algorithm, which automatically discovers matrix-specific pruning levels and exposes the uneven distribution of task-relevant knowledge. Across models from 3.8B to 70B parameters, LLM-Sieve removes 20-75% of weights with only 1-5% accuracy loss-substantially ahead of prior pruning methods. Beyond efficiency, our framework reveals bottleneck matrices that concentrate critical knowledge, suggesting architectural implications for future LLM design. LLM-Sieve integrates seamlessly with LoRA fine-tuning and quantization, enabling both efficient deployment and deeper understanding of knowledge organization in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01315v2">Wired for Reuse: Automating Context-Aware Code Adaptation in IDEs via LLM-Based Agent</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Accepted by ASE2025
    </div>
    <details class="paper-abstract">
      Copy-paste-modify is a widespread and pragmatic practice in software development, where developers adapt reused code snippets, sourced from platforms such as Stack Overflow, GitHub, or LLM outputs, into their local codebase. A critical yet underexplored aspect of this adaptation is code wiring: the context-aware process of substituting unresolved variables in pasted code with suitable variables or expressions from the surrounding context. Existing solutions either rely on heuristic rules or historical templates, often failing to effectively utilize contextual information, despite studies showing that over half of adaptation cases are context-dependent. In this paper, we introduce WIRL, an LLM-based agent for code wiring framed as a Retrieval-Augmented Generation (RAG) infilling task. WIRL combines an LLM, a customized toolkit, and an orchestration module to identify unresolved variables, retrieve context, and perform context-aware substitutions. To balance efficiency and autonomy, the agent adopts a mixed strategy: deterministic rule-based steps for common patterns, and a state-machine-guided decision process for intelligent exploration. We evaluate WIRL on a carefully curated, high-quality dataset consisting of real-world code adaptation scenarios. Our approach achieves an exact match precision of 91.7% and a recall of 90.0%, outperforming advanced LLMs by 22.6 and 13.7 percentage points in precision and recall, respectively, and surpassing IntelliJ IDEA by 54.3 and 49.9 percentage points. These results underscore its practical utility, particularly in contexts with complex variable dependencies or multiple unresolved variables. We believe WIRL paves the way for more intelligent and context-aware developer assistance in modern IDEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23252v2">NanoFlux: Adversarial Dual-LLM Evaluation and Distillation For Multi-Domain Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 preprint version
    </div>
    <details class="paper-abstract">
      We present NanoFlux, a novel adversarial framework for generating targeted training data to improve LLM reasoning, where adversarially-generated datasets containing fewer than 200 examples outperform conventional fine-tuning approaches. The framework employs a competitive dynamic between models alternating as Attacker and Defender, supervised by a tool-augmented Judge, synthesizing multi-step questions with explanatory annotations that target specific reasoning capabilities. Fine-tuning a 4B-parameter model on NanoFlux-generated data yields performance gains across diverse domains compared to full-benchmark fine-tuning: +5.9% on mathematical reasoning (GSMHard), +3.6% on scientific reasoning (GenomeBench), and +16.6% on medical reasoning (MultiMedQA), while reducing computational requirements by 3-14x. Ablation studies reveal a non-monotonic relationship between dataset characteristics and model performance, uncovering domain-specific optimal points for question complexity and reasoning quality. NanoFlux automates training data generation through embedding-based novelty filtering, tool-augmented evaluation, and multi-hop reasoning, suggesting that future model improvements may lie in the intelligent synthesis of small, precisely targeted training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03595v1">Decoupling Task-Solving and Output Formatting in LLM Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adept at following instructions containing task descriptions to solve complex problems, such as mathematical reasoning and automatic evaluation (LLM-as-a-Judge). However, as prompts grow more complex, models often struggle to adhere to all instructions. This difficulty is especially common when instructive prompts intertwine reasoning directives -- specifying what the model should solve -- with rigid formatting requirements that dictate how the solution must be presented. The entanglement creates competing goals for the model, suggesting that more explicit separation of these two aspects could lead to improved performance. To this front, we introduce Deco-G, a decoding framework that explicitly decouples format adherence from task solving. Deco-G handles format compliance with a separate tractable probabilistic model (TPM), while prompts LLMs with only task instructions. At each decoding step, Deco-G combines next token probabilities from the LLM with the TPM calculated format compliance likelihood to form the output probability. To make this approach both practical and scalable for modern instruction-tuned LLMs, we introduce three key innovations: instruction-aware distillation, a flexible trie-building algorithm, and HMM state pruning for computational efficiency. We demonstrate the effectiveness of Deco-G across a wide range of tasks with diverse format requirements, including mathematical reasoning, LLM-as-a-judge, and event argument extraction. Overall, our approach yields 1.0% to 6.0% relative gain over regular prompting practice with guaranteed format compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04439v3">ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. We evaluate on ARC-AGI, a benchmark that stresses compositional generalization and abstract reasoning, making it a natural fit for concept memory. Our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, dynamically updating memory during test-time outperforms fixed settings, supporting the hypothesis that accumulating and abstracting patterns enables further solutions in a form of self-improvement. Code is available at https://github.com/matt-seb-ho/arc_memo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03955v1">Harnessing Synthetic Preference Data for Enhancing Temporal Understanding of Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 17 pages, 9 figures, 6 tables. Presents TimeWarp, a synthetic preference data framework to improve temporal understanding in Video-LLMs, showing consistent gains across seven benchmarks. Includes supplementary material in the Appendix
    </div>
    <details class="paper-abstract">
      While Video Large Language Models (Video-LLMs) have demonstrated remarkable performance across general video understanding benchmarks-particularly in video captioning and descriptive tasks-they consistently underperform on tasks that require fine-grained temporal understanding. This limitation arises due to the lack of visual complexity and temporal nuance in current fine-tuning datasets, leading these models to rely heavily on language-based reasoning rather than truly understanding video dynamics. In this work, we propose TimeWarp, a systematic method to create a targeted synthetic temporal dataset to fine-tune the model's responses to encourage it to focus on the given input video. We introduce a large-scale preference dataset, created using TimeWarp, that captures intricate temporal dynamics often overlooked, grounding the model's responses to visual and temporal information. We demonstrate that when our method is applied to existing models, it significantly improves performance on temporal understanding benchmarks, highlighting the effectiveness of our proposed datasets in advancing temporal understanding in Video-LLMs, resulting in an absolute improvement in performance across seven benchmarks. Code is available at https://github.com/sameepv21/timewarp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13793v2">Mapping the Trust Terrain: LLMs in Software Engineering -- Insights and Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Accepted to appear in IEEE Transactions on Software Engineering
    </div>
    <details class="paper-abstract">
      Applications of Large Language Models (LLMs) are rapidly growing in industry and academia for various software engineering (SE) tasks. As these models become more integral to critical processes, ensuring their reliability and trustworthiness becomes essential. Consequently, the concept of trust in these systems is becoming increasingly critical. Well-calibrated trust is important, as excessive trust can lead to security vulnerabilities, and risks, while insufficient trust can hinder innovation. However, the landscape of trust-related concepts in LLMs in SE is relatively unclear, with concepts such as trust, distrust, and trustworthiness lacking clear conceptualizations in the SE community. To bring clarity to the current research status and identify opportunities for future work, we conducted a comprehensive review of $88$ papers: a systematic literature review of $18$ papers focused on LLMs in SE, complemented by an analysis of 70 papers from broader trust literature. Additionally, we conducted a survey study with 25 domain experts to gain insights into practitioners' understanding of trust and identify gaps between existing literature and developers' perceptions. The result of our analysis serves as a roadmap that covers trust-related concepts in LLMs in SE and highlights areas for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03930v1">LLM Chemistry Estimation for Multi-LLM Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 20 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Multi-LLM collaboration promises accurate, robust, and context-aware solutions, yet existing approaches rely on implicit selection and output assessment without analyzing whether collaborating models truly complement or conflict. We introduce LLM Chemistry -- a framework that measures when LLM combinations exhibit synergistic or antagonistic behaviors that shape collective performance beyond individual capabilities. We formalize the notion of chemistry among LLMs, propose algorithms that quantify it by analyzing interaction dependencies, and recommend optimal model ensembles accordingly. Our theoretical analysis shows that chemistry among collaborating LLMs is most evident under heterogeneous model profiles, with its outcome impact shaped by task type, group size, and complexity. Evaluation on classification, summarization, and program repair tasks provides initial evidence for these task-dependent effects, thereby reinforcing our theoretical results. This establishes LLM Chemistry as both a diagnostic factor in multi-LLM systems and a foundation for ensemble recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03914v1">Refactoring with LLMs: Bridging Human Expertise and Machine Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 43 pages, 2 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Code refactoring is a fundamental software engineering practice aimed at improving code quality and maintainability. Despite its importance, developers often neglect refactoring due to the significant time, effort, and resources it requires, as well as the lack of immediate functional rewards. Although several automated refactoring tools have been proposed, they remain limited in supporting a broad spectrum of refactoring types. In this study, we explore whether instruction strategies inspired by human best-practice guidelines can enhance the ability of Large Language Models (LLMs) to perform diverse refactoring tasks automatically. Leveraging the instruction-following and code comprehension capabilities of state-of-the-art LLMs (e.g., GPT-mini and DeepSeek-V3), we draw on Martin Fowler's refactoring guidelines to design multiple instruction strategies that encode motivations, procedural steps, and transformation objectives for 61 well-known refactoring types. We evaluate these strategies on benchmark examples and real-world code snippets from GitHub projects. Our results show that instruction designs grounded in Fowler's guidelines enable LLMs to successfully perform all benchmark refactoring types and preserve program semantics in real-world settings, an essential criterion for effective refactoring. Moreover, while descriptive instructions are more interpretable to humans, our results show that rule-based instructions often lead to better performance in specific scenarios. Interestingly, allowing models to focus on the overall goal of refactoring, rather than prescribing a fixed transformation type, can yield even greater improvements in code quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15436v2">Ads that Talk Back: Implications and Perceptions of Injecting Personalized Advertising into LLM Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled the creation of highly effective chatbots. However, the compute costs of widely deploying LLMs have raised questions about profitability. Companies have proposed exploring ad-based revenue streams for monetizing LLMs, which could serve as the new de facto platform for advertising. This paper investigates the implications of personalizing LLM advertisements to individual users via a between-subjects experiment with 179 participants. We developed a chatbot that embeds personalized product advertisements within LLM responses, inspired by similar forays by AI companies. The evaluation of our benchmarks showed that ad injection only slightly impacted LLM performance, particularly response desirability. Results revealed that participants struggled to detect ads, and even preferred LLM responses with hidden advertisements. Rather than clicking on our advertising disclosure, participants tried changing their advertising settings using natural language queries. We created an advertising dataset and an open-source LLM, Phi-4-Ads, fine-tuned to serve ads and flexibly adapt to user preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03904v1">LLM as an Algorithmist: Enhancing Anomaly Detectors via Programmatic Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Existing anomaly detection (AD) methods for tabular data usually rely on some assumptions about anomaly patterns, leading to inconsistent performance in real-world scenarios. While Large Language Models (LLMs) show remarkable reasoning capabilities, their direct application to tabular AD is impeded by fundamental challenges, including difficulties in processing heterogeneous data and significant privacy risks. To address these limitations, we propose LLM-DAS, a novel framework that repositions the LLM from a ``data processor'' to an ``algorithmist''. Instead of being exposed to raw data, our framework leverages the LLM's ability to reason about algorithms. It analyzes a high-level description of a given detector to understand its intrinsic weaknesses and then generates detector-specific, data-agnostic Python code to synthesize ``hard-to-detect'' anomalies that exploit these vulnerabilities. This generated synthesis program, which is reusable across diverse datasets, is then instantiated to augment training data, systematically enhancing the detector's robustness by transforming the problem into a more discriminative two-class classification task. Extensive experiments on 36 TAD benchmarks show that LLM-DAS consistently boosts the performance of mainstream detectors. By bridging LLM reasoning with classic AD algorithms via programmatic synthesis, LLM-DAS offers a scalable, effective, and privacy-preserving approach to patching the logical blind spots of existing detectors.
    </details>
</div>
