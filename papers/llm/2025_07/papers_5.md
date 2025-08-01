# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15788v1">Small LLMs Do Not Learn a Generalizable Theory of Mind via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated emergent capabilities in complex reasoning, largely spurred by rule-based Reinforcement Learning (RL) techniques applied during the post-training. This has raised the question of whether similar methods can instill more nuanced, human-like social intelligence, such as a Theory of Mind (ToM), in LLMs. This paper investigates whether small-scale LLMs can acquire a robust and generalizable ToM capability through RL with verifiable rewards (RLVR). We conduct a systematic evaluation by training models on various combinations of prominent ToM datasets (HiToM, ExploreToM, FANToM) and testing for generalization on held-out datasets (e.g., OpenToM). Our findings indicate that small LLMs struggle to develop a generic ToM capability. While performance on in-distribution tasks improves, this capability fails to transfer to unseen ToM tasks with different characteristics. Furthermore, we demonstrate that prolonged RL training leads to models ``hacking'' the statistical patterns of the training datasets, resulting in significant performance gains on in-domain data but no change, or degradation of performance on out-of-distribution tasks. This suggests the learned behavior is a form of narrow overfitting rather than the acquisition of a true, abstract ToM capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15782v1">Interleaved LLM and Motion Planning for Generalized Multi-Object Collection in Large Scene Graphs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Household robots have been a longstanding research topic, but they still lack human-like intelligence, particularly in manipulating open-set objects and navigating large environments efficiently and accurately. To push this boundary, we consider a generalized multi-object collection problem in large scene graphs, where the robot needs to pick up and place multiple objects across multiple locations in a long mission of multiple human commands. This problem is extremely challenging since it requires long-horizon planning in a vast action-state space under high uncertainties. To this end, we propose a novel interleaved LLM and motion planning algorithm Inter-LLM. By designing a multimodal action cost similarity function, our algorithm can both reflect the history and look into the future to optimize plans, striking a good balance of quality and efficiency. Simulation experiments demonstrate that compared with latest works, our algorithm improves the overall mission performance by 30% in terms of fulfilling human commands, maximizing mission success rates, and minimizing mission costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11558v3">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15752v1">DialogueForge: LLM Simulation of Human-Chatbot Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 For our code and data, see https://github.com/nerchio/Human_Chatbot-Generation
    </div>
    <details class="paper-abstract">
      Collecting human-chatbot dialogues typically demands substantial manual effort and is time-consuming, which limits and poses challenges for research on conversational AI. In this work, we propose DialogueForge - a framework for generating AI-simulated conversations in human-chatbot style. To initialize each generated conversation, DialogueForge uses seed prompts extracted from real human-chatbot interactions. We test a variety of LLMs to simulate the human chatbot user, ranging from state-of-the-art proprietary models to small-scale open-source LLMs, and generate multi-turn dialogues tailored to specific tasks. In addition, we explore fine-tuning techniques to enhance the ability of smaller models to produce indistinguishable human-like dialogues. We evaluate the quality of the simulated conversations and compare different models using the UniEval and GTEval evaluation protocols. Our experiments show that large proprietary models (e.g., GPT-4o) generally outperform others in generating more realistic dialogues, while smaller open-source models (e.g., Llama, Mistral) offer promising performance with greater customization. We demonstrate that the performance of smaller models can be significantly improved by employing supervised fine-tuning techniques. Nevertheless, maintaining coherent and natural long-form human-like dialogues remains a common challenge across all models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15717v1">BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15715v1">From Queries to Criteria: Understanding How Astronomers Evaluate LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted to the Conference on Language Modeling 2025 (COLM), 22 pages, 6 figures
    </div>
    <details class="paper-abstract">
      There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15613v1">Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place. We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12601v2">CCSBench: Evaluating Compositional Controllability in LLMs for Scientific Document Summarization</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted to KDD 2025 SciSoc LLM Workshop: Large Language Models for Scientific and Societal Advances
    </div>
    <details class="paper-abstract">
      To broaden the dissemination of scientific knowledge to diverse audiences, it is desirable for scientific document summarization systems to simultaneously control multiple attributes such as length and empirical focus. However, existing research typically focuses on controlling single attributes, leaving the compositional control of multiple attributes underexplored. To address this gap, we introduce CCSBench, the first evaluation benchmark for compositional controllable summarization in the scientific domain. Our benchmark enables fine-grained control over both explicit attributes (e.g., length), which are objective and straightforward, and implicit attributes (e.g., conceptual or empirical focus), which are more subjective and abstract. We conduct extensive experiments using various large language models (LLMs) under various settings, including in-context learning, parameter-efficient fine-tuning, and two-stage modular methods for balancing control over different attributes. Our findings reveal significant limitations in LLMs capabilities in balancing trade-offs between control attributes, especially implicit ones that require deeper understanding and abstract reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15585v1">Unequal Voices: How LLMs Construct Constrained Queer Narratives</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      One way social groups are marginalized in discourse is that the narratives told about them often default to a narrow, stereotyped range of topics. In contrast, default groups are allowed the full complexity of human existence. We describe the constrained representations of queer people in LLM generations in terms of harmful representations, narrow representations, and discursive othering and formulate hypotheses to test for these phenomena. Our results show that LLMs are significantly limited in their portrayals of queer personas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17181v2">A Study of LLMs' Preferences for Libraries and Programming Languages</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 13 pages, 8 tables, 2 figures. Paper was previously titled "LLMs Love Python"
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate code, influencing users' choices of libraries and programming languages in critical real-world projects. However, little is known about their systematic biases or preferences toward certain libraries and programming languages, which can significantly impact software development practices. To fill this gap, we perform the first empirical study of LLMs' preferences for libraries and programming languages when generating code, covering eight diverse LLMs. Our results reveal that LLMs exhibit a strong tendency to overuse widely adopted libraries such as NumPy; in up to 48% of cases, this usage is unnecessary and deviates from the ground-truth solutions. LLMs also exhibit a significant preference toward Python as their default language. For high-performance project initialisation tasks where Python is not the optimal language, it remains the dominant choice in 58% of cases, and Rust is not used a single time. These results indicate that LLMs may prioritise familiarity and popularity over suitability and task-specific optimality. This will introduce security vulnerabilities and technical debt, and limit exposure to newly developed, better-suited tools and languages. Understanding and addressing these biases is essential for the responsible integration of LLMs into software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05445v2">clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      The emergence of instruction-tuned large language models (LLMs) has advanced the field of dialogue systems, enabling both realistic user simulations and robust multi-turn conversational agents. However, existing research often evaluates these components in isolation-either focusing on a single user simulator or a specific system design-limiting the generalisability of insights across architectures and configurations. In this work, we propose clem todd (chat-optimized LLMs for task-oriented dialogue systems development), a flexible framework for systematically evaluating dialogue systems under consistent conditions. clem todd enables detailed benchmarking across combinations of user simulators and dialogue systems, whether existing models from literature or newly developed ones. It supports plug-and-play integration and ensures uniform datasets, evaluation metrics, and computational constraints. We showcase clem todd's flexibility by re-evaluating existing task-oriented dialogue systems within this unified setup and integrating three newly proposed dialogue systems into the same evaluation pipeline. Our results provide actionable insights into how architecture, scale, and prompting strategies affect dialogue performance, offering practical guidance for building efficient and effective conversational AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15553v1">Efficient Routing of Inference Requests across LLM Instances in Cloud-Edge Computing</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The rising demand for Large Language Model (LLM) inference services has intensified pressure on computational resources, resulting in latency and cost challenges. This paper introduces a novel routing algorithm based on the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to distribute inference requests across heterogeneous LLM instances in a cloud-edge computing environment. Formulated as a multi-objective optimization problem, the algorithm balances response quality, response time, and inference cost, adapting to request heterogeneity (e.g., varying complexity and prompt lengths) and node diversity (e.g., edge vs. cloud resources). This adaptive routing algorithm optimizes performance under dynamic workloads. We benchmark the approach using a testbed with datasets including Stanford Question Answering Dataset (SQuAD), Mostly Basic Python Problems (MBPP), Hella Situations With Adversarial Generations (HellaSwag), and Grade School Math 8K (GSM8K). Experimental results show our solution, compared to the baselines, achieves up to 95.2% and 34.9% improvements in terms of response time and cost, respectively. These findings validate the algorithm's effectiveness for scalable LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15550v1">PhysGym: Benchmarking LLMs in Interactive Physics Discovery with Controlled Priors</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 31 Pages
    </div>
    <details class="paper-abstract">
      Evaluating the scientific discovery capabilities of large language model based agents, particularly how they cope with varying environmental complexity and utilize prior knowledge, requires specialized benchmarks currently lacking in the landscape. To address this gap, we introduce PhysGym, a novel benchmark suite and simulation platform for rigorously assessing LLM-based scientific reasoning in interactive physics environments. PhysGym's primary contribution lies in its sophisticated control over the level of prior knowledge provided to the agent. This allows researchers to dissect agent performance along axes including the complexity of the problem and the prior knowledge levels. The benchmark comprises a suite of interactive simulations, where agents must actively probe environments, gather data sequentially under constraints and formulate hypotheses about underlying physical laws. PhysGym provides standardized evaluation protocols and metrics for assessing hypothesis accuracy and model fidelity. We demonstrate the benchmark's utility by presenting results from baseline LLMs, showcasing its ability to differentiate capabilities based on varying priors and task complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00024v2">Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 ACL 2025 Camera-ready
    </div>
    <details class="paper-abstract">
      Emotions have been shown to play a role in argument convincingness, yet this aspect is underexplored in the natural language processing (NLP) community. Unlike prior studies that use static analyses, focus on a single text domain or language, or treat emotion as just one of many factors, we introduce a dynamic framework inspired by manipulation checks commonly used in psychology and social science; leveraging LLM-based manipulation checks, this framework examines the extent to which perceived emotional intensity influences perceived convincingness. Through human evaluation of arguments across different languages, text domains, and topics, we find that in over half of cases, human judgments of convincingness remain unchanged despite variations in perceived emotional intensity; when emotions do have an impact, they more often enhance rather than weaken convincingness. We further analyze whether 11 LLMs behave like humans in the same scenario, finding that while LLMs generally mirror human patterns, they struggle to capture nuanced emotional effects in individual judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15521v1">LLM world models are mental: Output layer evidence of brittle world model use in LLM mechanical reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Manuscript comprises 14 pages, 4 figures, 4 tables in the Technical Appendix and Supplementary Material, and is under review at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) construct and manipulate internal world models, or do they rely solely on statistical associations represented as output layer token probabilities? We adapt cognitive science methodologies from human mental models research to test LLMs on pulley system problems using TikZ-rendered stimuli. Study 1 examines whether LLMs can estimate mechanical advantage (MA). State-of-the-art models performed marginally but significantly above chance, and their estimates correlated significantly with ground-truth MA. Significant correlations between number of pulleys and model estimates suggest that models employed a pulley counting heuristic, without necessarily simulating pulley systems to derive precise values. Study 2 tested this by probing whether LLMs represent global features crucial to MA estimation. Models evaluated a functionally connected pulley system against a fake system with randomly placed components. Without explicit cues, models identified the functional system as having greater MA with F1=0.8, suggesting LLMs could represent systems well enough to differentiate jumbled from functional systems. Study 3 built on this by asking LLMs to compare functional systems with matched systems which were connected up but which transferred no force to the weight; LLMs identified the functional system with F1=0.46, suggesting random guessing. Insofar as they may generalize, these findings are compatible with the notion that LLMs manipulate internal world models, sufficient to exploit statistical associations between pulley count and MA (Study 1), and to approximately represent system components' spatial relations (Study 2). However, they may lack the facility to reason over nuanced structural connectivity (Study 3). We conclude by advocating the utility of cognitive scientific methods to evaluate the world-modeling capacities of artificial intelligence systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15502v1">FollowUpBot: An LLM-Based Conversational Robot for Automatic Postoperative Follow-up</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      Postoperative follow-up plays a crucial role in monitoring recovery and identifying complications. However, traditional approaches, typically involving bedside interviews and manual documentation, are time-consuming and labor-intensive. Although existing digital solutions, such as web questionnaires and intelligent automated calls, can alleviate the workload of nurses to a certain extent, they either deliver an inflexible scripted interaction or face private information leakage issues. To address these limitations, this paper introduces FollowUpBot, an LLM-powered edge-deployed robot for postoperative care and monitoring. It allows dynamic planning of optimal routes and uses edge-deployed LLMs to conduct adaptive and face-to-face conversations with patients through multiple interaction modes, ensuring data privacy. Moreover, FollowUpBot is capable of automatically generating structured postoperative follow-up reports for healthcare institutions by analyzing patient interactions during follow-up. Experimental results demonstrate that our robot achieves high coverage and satisfaction in follow-up interactions, as well as high report generation accuracy across diverse field types. The demonstration video is available at https://www.youtube.com/watch?v=_uFgDO7NoK0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15296v1">Butterfly Effects in Toolchains: A Comprehensive Analysis of Failed Parameter Filling in LLM Tool-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The emergence of the tool agent paradigm has broadened the capability boundaries of the Large Language Model (LLM), enabling it to complete more complex tasks. However, the effectiveness of this paradigm is limited due to the issue of parameter failure during its execution. To explore this phenomenon and propose corresponding suggestions, we first construct a parameter failure taxonomy in this paper. We derive five failure categories from the invocation chain of a mainstream tool agent. Then, we explore the correlation between three different input sources and failure categories by applying 15 input perturbation methods to the input. Experimental results show that parameter name hallucination failure primarily stems from inherent LLM limitations, while issues with input sources mainly cause other failure patterns. To improve the reliability and effectiveness of tool-agent interactions, we propose corresponding improvement suggestions, including standardizing tool return formats, improving error feedback mechanisms, and ensuring parameter consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06273v2">Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations</a></div>
    <div class="paper-meta">
      📅 2025-07-21
      | 💬 Accepted at ICCV 2025. Code available at: https://github.com/JeongHun0716/zero-avsr
    </div>
    <details class="paper-abstract">
      We explore a novel zero-shot Audio-Visual Speech Recognition (AVSR) framework, dubbed Zero-AVSR, which enables speech recognition in target languages without requiring any audio-visual speech data in those languages. Specifically, we introduce the Audio-Visual Speech Romanizer (AV-Romanizer), which learns language-agnostic speech representations by predicting Roman text. Then, by leveraging the strong multilingual modeling capabilities of Large Language Models (LLMs), we propose converting the predicted Roman text into language-specific graphemes, forming the proposed Cascaded Zero-AVSR. Taking it a step further, we explore a unified Zero-AVSR approach by directly integrating the audio-visual speech representations encoded by the AV-Romanizer into the LLM. This is achieved through finetuning the adapter and the LLM using our proposed multi-task learning scheme. To capture the wide spectrum of phonetic and linguistic diversity, we also introduce a Multilingual Audio-Visual Romanized Corpus (MARC) consisting of 2,916 hours of audio-visual speech data across 82 languages, along with transcriptions in both language-specific graphemes and Roman text. Extensive analysis and experiments confirm that the proposed Zero-AVSR framework has the potential to expand language support beyond the languages seen during the training of the AV-Romanizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15268v1">IM-Chat: A Multi-agent LLM-based Framework for Knowledge Transfer in Injection Molding Industry</a></div>
    <div class="paper-meta">
      📅 2025-07-21
    </div>
    <details class="paper-abstract">
      The injection molding industry faces critical challenges in preserving and transferring field knowledge, particularly as experienced workers retire and multilingual barriers hinder effective communication. This study introduces IM-Chat, a multi-agent framework based on large language models (LLMs), designed to facilitate knowledge transfer in injection molding. IM-Chat integrates both limited documented knowledge (e.g., troubleshooting tables, manuals) and extensive field data modeled through a data-driven process condition generator that infers optimal manufacturing settings from environmental inputs such as temperature and humidity, enabling robust and context-aware task resolution. By adopting a retrieval-augmented generation (RAG) strategy and tool-calling agents within a modular architecture, IM-Chat ensures adaptability without the need for fine-tuning. Performance was assessed across 100 single-tool and 60 hybrid tasks for GPT-4o, GPT-4o-mini, and GPT-3.5-turbo by domain experts using a 10-point rubric focused on relevance and correctness, and was further supplemented by automated evaluation using GPT-4o guided by a domain-adapted instruction prompt. The evaluation results indicate that more capable models tend to achieve higher accuracy, particularly in complex, tool-integrated scenarios. Overall, these findings demonstrate the viability of multi-agent LLM systems for industrial knowledge workflows and establish IM-Chat as a scalable and generalizable approach to AI-assisted decision support in manufacturing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15170v2">From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14928v1">Byzantine-Robust Decentralized Coordination of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Collaboration among multiple large language model (LLM) agents is a promising approach to overcome inherent limitations of single-agent systems, such as hallucinations and single points of failure. As LLM agents are increasingly deployed on open blockchain platforms, multi-agent systems capable of tolerating malicious (Byzantine) agents have become essential. Recent Byzantine-robust multi-agent systems typically rely on leader-driven coordination, which suffers from two major drawbacks. First, they are inherently vulnerable to targeted attacks against the leader. If consecutive leaders behave maliciously, the system repeatedly fails to achieve consensus, forcing new consensus rounds, which is particularly costly given the high latency of LLM invocations. Second, an underperforming proposal from the leader can be accepted as the final answer even when higher-quality alternatives are available, as existing methods finalize the leader's proposal once it receives a quorum of votes. To address these issues, we propose DecentLLMs, a novel decentralized consensus approach for multi-agent LLM systems, where worker agents generate answers concurrently and evaluator agents independently score and rank these answers to select the best available one. This decentralized architecture enables faster consensus despite the presence of Byzantine agents and consistently selects higher-quality answers through Byzantine-robust aggregation techniques. Experimental results demonstrate that DecentLLMs effectively tolerates Byzantine agents and significantly improves the quality of selected answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14906v1">Feedback-Induced Performance Decline in LLM-Based Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      The ability of Large Language Models (LLMs) to extract context from natural language problem descriptions naturally raises questions about their suitability in autonomous decision-making settings. This paper studies the behaviour of these models within a Markov Decision Process (MDPs). While traditional reinforcement learning (RL) strategies commonly employed in this setting rely on iterative exploration, LLMs, pre-trained on diverse datasets, offer the capability to leverage prior knowledge for faster adaptation. We investigate online structured prompting strategies in sequential decision making tasks, comparing the zero-shot performance of LLM-based approaches to that of classical RL methods. Our findings reveal that although LLMs demonstrate improved initial performance in simpler environments, they struggle with planning and reasoning in complex scenarios without fine-tuning or additional guidance. Our results show that feedback mechanisms, intended to improve decision-making, often introduce confusion, leading to diminished performance in intricate environments. These insights underscore the need for further exploration into hybrid strategies, fine-tuning, and advanced memory integration to enhance LLM-based decision-making capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14894v1">Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have impressive multilingual capabilities, but they suffer from unexpected code-switching, also known as language mixing, which involves switching to unexpected languages in the model response. This problem leads to poor readability and degrades the usability of model responses. However, existing work on this issue lacks a mechanistic analysis and shows limited effectiveness. In this paper, we first provide an in-depth analysis of unexpected code-switching using sparse autoencoders and find that when LLMs switch to a language, the features of that language exhibit excessive pre-activation values. Based on our findings, we propose $\textbf{S}$parse $\textbf{A}$utoencoder-guided $\textbf{S}$upervised $\textbf{F}$ine$\textbf{t}$uning (SASFT), which teaches LLMs to maintain appropriate pre-activation values of specific language features during training. Experiments on five models across three languages demonstrate that SASFT consistently reduces unexpected code-switching by more than 50\% compared to standard supervised fine-tuning, with complete elimination in four cases. Moreover, SASFT maintains or even improves the models' performance on six multilingual benchmarks, showing its effectiveness in addressing code-switching while preserving multilingual capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14799v1">Manipulating LLM Web Agents with Indirect Prompt Injection Attack via HTML Accessibility Tree</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 EMNLP 2025 System Demonstrations Submission
    </div>
    <details class="paper-abstract">
      This work demonstrates that LLM-based web navigation agents offer powerful automation capabilities but are vulnerable to Indirect Prompt Injection (IPI) attacks. We show that adversaries can embed universal adversarial triggers in webpage HTML to hijack agent behavior that utilizes the accessibility tree to parse HTML, causing unintended or malicious actions. Using the Greedy Coordinate Gradient (GCG) algorithm and a Browser Gym agent powered by Llama-3.1, our system demonstrates high success rates across real websites in both targeted and general attacks, including login credential exfiltration and forced ad clicks. Our empirical results highlight critical security risks and the need for stronger defenses as LLM-driven autonomous web agents become more widely adopted. The system software (https://github.com/sej2020/manipulating-web-agents) is released under the MIT License, with an accompanying publicly available demo website (http://lethaiq.github.io/attack-web-llm-agent).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09971v3">Advancing Object Goal Navigation Through LLM-enhanced Object Affinities Transfer</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Object-goal navigation requires mobile robots to efficiently locate targets with visual and spatial information, yet existing methods struggle with generalization in unseen environments. Heuristic approaches with naive metrics fail in complex layouts, while graph-based and learning-based methods suffer from environmental biases and limited generalization. Although Large Language Models (LLMs) as planners or agents offer a rich knowledge base, they are cost-inefficient and lack targeted historical experience. To address these challenges, we propose the LLM-enhanced Object Affinities Transfer (LOAT) framework, integrating LLM-derived semantics with learning-based approaches to leverage experiential object affinities for better generalization in unseen settings. LOAT employs a dual-module strategy: one module accesses LLMs' vast knowledge, and the other applies learned object semantic relationships, dynamically fusing these sources based on context. Evaluations in AI2-THOR and Habitat simulators show significant improvements in navigation success and efficiency, and real-world deployment demonstrates the zero-shot ability of LOAT to enhance object-goal navigation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11704v2">A Library of LLM Intrinsics for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 This (June 2025) is the second version of this paper (the first was published in April 2025). Intrinsics implemented as LoRAs are now trained on IBM Granite 3.3 8b instruct (previously 3.2)
    </div>
    <details class="paper-abstract">
      In the developer community for large language models (LLMs), there is not yet a clean pattern analogous to a software library, to support very large scale collaboration. Even for the commonplace use case of Retrieval-Augmented Generation (RAG), it is not currently possible to write a RAG application against a well-defined set of APIs that are agreed upon by different LLM providers. Inspired by the idea of compiler intrinsics, we propose some elements of such a concept through introducing a library of LLM Intrinsics for RAG. An LLM intrinsic is defined as a capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented. The intrinsics in our library are released as LoRA adapters on HuggingFace, and through a software interface with clear structured input/output characteristics on top of vLLM as an inference platform, accompanied in both places with documentation and code. This article describes the intended usage, training details, and evaluations for each intrinsic, as well as compositions of multiple intrinsics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14785v1">Exploring the In-Context Learning Capabilities of LLMs for Money Laundering Detection in Financial Graphs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      The complexity and interconnectivity of entities involved in money laundering demand investigative reasoning over graph-structured data. This paper explores the use of large language models (LLMs) as reasoning engines over localized subgraphs extracted from a financial knowledge graph. We propose a lightweight pipeline that retrieves k-hop neighborhoods around entities of interest, serializes them into structured text, and prompts an LLM via few-shot in-context learning to assess suspiciousness and generate justifications. Using synthetic anti-money laundering (AML) scenarios that reflect common laundering behaviors, we show that LLMs can emulate analyst-style logic, highlight red flags, and provide coherent explanations. While this study is exploratory, it illustrates the potential of LLM-based graph reasoning in AML and lays groundwork for explainable, language-driven financial crime analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14784v1">LeAdQA: LLM-Driven Context-Aware Temporal Grounding for Video Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Video Question Answering (VideoQA) requires identifying sparse critical moments in long videos and reasoning about their causal relationships to answer semantically complex questions. While recent advances in multimodal learning have improved alignment and fusion, current approaches remain limited by two prevalent but fundamentally flawed strategies: (1) task-agnostic sampling indiscriminately processes all frames, overwhelming key events with irrelevant content; and (2) heuristic retrieval captures superficial patterns but misses causal-temporal structures needed for complex reasoning. To address these challenges, we introduce LeAdQA, an innovative approach that bridges these gaps through synergizing causal-aware query refinement with fine-grained visual grounding. Our method first leverages LLMs to reformulate question-option pairs, resolving causal ambiguities and sharpening temporal focus. These refined queries subsequently direct a temporal grounding model to precisely retrieve the most salient segments, complemented by an adaptive fusion mechanism dynamically integrating the evidence to maximize relevance. The integrated visual-textual cues are then processed by an MLLM to generate accurate, contextually-grounded answers. Experiments on NExT-QA, IntentQA, and NExT-GQA demonstrate that our method's precise visual grounding substantially enhances the understanding of video-question relationships, achieving state-of-the-art (SOTA) performance on complex reasoning tasks while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14783v1">Omni-Think: Scaling Cross-Domain Generalization in LLMs via Multi-Task RL with Hybrid Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      The advancement of general-purpose artificial intelligence relies on large language models (LLMs) that excel across a wide range of tasks, from structured reasoning to creative generation. However, post-training methods like Supervised Fine-Tuning (SFT) often struggle with generalization, favoring memorization over transferable learning. In this work, we introduce Omni-Think, a unified reinforcement learning (RL) framework that enhances LLM performance across diverse tasks by combining rule-based verifiable rewards with generative preference signals via LLM-as-a-Judge evaluations. Our approach enables consistent optimization across task types and scales RL-based training to subjective domains. We further investigate training strategies, demonstrating that a curriculum-based progression that orders tasks from structured to open-ended improves performance and reduces forgetting. Experimental results across four domains reveal that curriculum learning improves performance by 5.2\% over joint training and 9.1\% over model merging. These results highlight the importance of task-aware sampling and hybrid supervision in scaling RL-based post-training for general-purpose LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14776v1">VeriOpt: PPA-Aware High-Quality Verilog Generation via Multi-Role LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 9 pages, 7 figures, Accepted for ICCAD 2025, Munich, Germany
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models(LLMs) in hardware design has primarily focused on generating functionally correct Verilog code, overlooking critical Power Performance-Area(PPA) metrics essential for industrial-grade designs. To bridge this gap, we propose VeriOpt, a novel framework that leverages role-based prompting and PPA-aware optimization to enable LLMs to produce high-quality, synthesizable Verilog. VeriOpt structures LLM interactions into specialized roles (e.g., Planner, Programmer, Reviewer, Evaluator) to emulate human design workflows, while integrating PPA constraints directly into the prompting pipeline. By combining multi-modal feedback (e.g., synthesis reports, timing diagrams) with PPA aware prompting, VeriOpt achieves PPA-efficient code generation without sacrificing functional correctness. Experimental results demonstrate up to 88% reduction in power, 76% reduction in area and 73% improvement in timing closure compared to baseline LLM-generated RTL, validated using industry standard EDA tools. At the same time achieves 86% success rate in functionality evaluation. Our work advances the state-of-the-art AI-driven hardware design by addressing the critical gap between correctness and quality, paving the way for reliable LLM adoption in production workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15157v1">Can LLMs Generate User Stories and Assess Their Quality?</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Requirements elicitation is still one of the most challenging activities of the requirements engineering process due to the difficulty requirements analysts face in understanding and translating complex needs into concrete requirements. In addition, specifying high-quality requirements is crucial, as it can directly impact the quality of the software to be developed. Although automated tools allow for assessing the syntactic quality of requirements, evaluating semantic metrics (e.g., language clarity, internal consistency) remains a manual and time-consuming activity. This paper explores how LLMs can help automate requirements elicitation within agile frameworks, where requirements are defined as user stories (US). We used 10 state-of-the-art LLMs to investigate their ability to generate US automatically by emulating customer interviews. We evaluated the quality of US generated by LLMs, comparing it with the quality of US generated by humans (domain experts and students). We also explored whether and how LLMs can be used to automatically evaluate the semantic quality of US. Our results indicate that LLMs can generate US similar to humans in terms of coverage and stylistic quality, but exhibit lower diversity and creativity. Although LLM-generated US are generally comparable in quality to those created by humans, they tend to meet the acceptance quality criteria less frequently, regardless of the scale of the LLM model. Finally, LLMs can reliably assess the semantic quality of US when provided with clear evaluation criteria and have the potential to reduce human effort in large-scale assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12899v3">A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 13 pages, 7 figure, 8 tables, under peer-review
    </div>
    <details class="paper-abstract">
      Language Models (LMs) are widely used in software engineering for code generation, but they may produce code with errors. Rather than repairing the generated code, an alternative way is to address the underlying failures of models. LM repair offers a lightweight solution to this challenge: it requires minimal data, reduces computational costs, and reduces the side effects. Unlike retraining, LM repair focuses on applying tailored updates to targeted neurons, making it ideal for scenarios with limited resources, high-performance demands, or strict safety requirements. In this paper, we propose Semantic Targeting for Analytical Repair (STAR), a pioneering and novel semantic-based optimization approach for repairing LLMs. STAR realizes the main operations of repairing LMs in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. Correspondingly, it computes the deltas of weight matrix as the prior information to guide optimization; and attributes the targeted layers and neurons leveraging statistical insights. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (MINT) and optimization methods (SGD), STAR integrates their strengths while mitigating their limitations. STAR supports solving multiple failures together, significantly improving the usefulness. Evaluated on coding tasks using popular code LMs, STAR exhibits superior effectiveness (10.5%-19.9% improvements) and efficiency (2.4-7.0 times speedup). In terms of side effects, namely the balance between generalization and specificity, STAR outperforms prior work by a significant margin. Additionally, we conducted assessments on the overfitting risk of LM repair as well as the cumulative impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11858v2">OpeNLGauge: An Explainable Metric for NLG Evaluation with Open-Weights LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated great potential as evaluators of NLG systems, allowing for high-quality, reference-free, and multi-aspect assessments. However, existing LLM-based metrics suffer from two major drawbacks: reliance on proprietary models to generate training data or perform evaluations, and a lack of fine-grained, explanatory feedback. In this paper, we introduce OpeNLGauge, a fully open-source, reference-free NLG evaluation metric that provides accurate explanations based on error spans. OpeNLGauge is available as a two-stage ensemble of larger open-weight LLMs, or as a small fine-tuned evaluation model, with confirmed generalizability to unseen tasks, domains and aspects. Our extensive meta-evaluation shows that OpeNLGauge achieves competitive correlation with human judgments, outperforming state-of-the-art models on certain tasks while maintaining full reproducibility and providing explanations more than twice as accurate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15066v1">Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 Under review. 19 pages, 8 figures, 12 tables
    </div>
    <details class="paper-abstract">
      Time series anomaly detection is critical across various domains, yet current approaches often limit analysis to mere binary anomaly classification without detailed categorization or further explanatory reasoning. To address these limitations, we propose a novel task, Time-series Reasoning for Anomaly (Time-RA) that transforms classical time series anomaly detection from a discriminative into a generative, reasoning-intensive task leveraging Large Language Models (LLMs). Also, we introduce the first real-world multimodal benchmark dataset, RATs40K, explicitly annotated for anomaly reasoning, comprising approximately 40,000 samples across 10 real-world domains. Each sample includes numeric time series data, contextual text information, and visual representations, each annotated with fine-grained categories (14 types for univariate anomalies and 6 for multivariate anomalies) and structured explanatory reasoning. We develop a sophisticated annotation framework utilizing ensemble-generated labels refined through GPT-4-driven feedback, ensuring accuracy and interpretability. Extensive benchmarking of LLMs and multimodal LLMs demonstrates the capabilities and limitations of current models, highlighting the critical role of supervised fine-tuning. Our dataset and task pave the way for significant advancements in interpretable time series anomaly detection and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15058v1">LibLMFuzz: LLM-Augmented Fuzz Target Generation for Black-box Libraries</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 6 pages, 2 figures, 1 table, 2 listings
    </div>
    <details class="paper-abstract">
      A fundamental problem in cybersecurity and computer science is determining whether a program is free of bugs and vulnerabilities. Fuzzing, a popular approach to discovering vulnerabilities in programs, has several advantages over alternative strategies, although it has investment costs in the form of initial setup and continuous maintenance. The choice of fuzzing is further complicated when only a binary library is available, such as the case of closed-source and proprietary software. In response, we introduce LibLMFuzz, a framework that reduces costs associated with fuzzing closed-source libraries by pairing an agentic Large Language Model (LLM) with a lightweight tool-chain (disassembler/compiler/fuzzer) to autonomously analyze stripped binaries, plan fuzz strategies, generate drivers, and iteratively self-repair build or runtime errors. Tested on four widely-used Linux libraries, LibLMFuzz produced syntactically correct drivers for all 558 fuzz-able API functions, achieving 100% API coverage with no human intervention. Across the 1601 synthesized drivers, 75.52% were nominally correct on first execution. The results show that LLM-augmented middleware holds promise in reducing the costs of fuzzing black box components and provides a foundation for future research efforts. Future opportunities exist for research in branch coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15049v1">Beyond Visual Line of Sight: UAVs with Edge AI, Connected LLMs, and VR for Autonomous Aerial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles are reshaping Non-Terrestrial Networks by acting as agile, intelligent nodes capable of advanced analytics and instantaneous situational awareness. This article introduces a budget-friendly quadcopter platform that unites 5G communications, edge-based processing, and AI to tackle core challenges in NTN scenarios. Outfitted with a panoramic camera, robust onboard computation, and LLMs, the drone system delivers seamless object recognition, contextual analysis, and immersive operator experiences through virtual reality VR technology. Field evaluations confirm the platform's ability to process visual streams with low latency and sustain robust 5G links. Adding LLMs further streamlines operations by extracting actionable insights and refining collected data for decision support. Demonstrated use cases, including emergency response, infrastructure assessment, and environmental surveillance, underscore the system's adaptability in demanding contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15015v1">EduThink4AI: Translating Educational Critical Thinking into Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential as educational tutoring agents, capable of tailoring hints, orchestrating lessons, and grading with near-human finesse across various academic domains. However, current LLM-based educational systems exhibit critical limitations in promoting genuine critical thinking, failing on over one-third of multi-hop questions with counterfactual premises, and remaining vulnerable to adversarial prompts that trigger biased or factually incorrect responses. To address these gaps, we propose EDU-Prompting, a novel multi-agent framework that bridges established educational critical thinking theories with LLM agent design to generate critical, bias-aware explanations while fostering diverse perspectives. Our systematic evaluation across theoretical benchmarks and practical college-level critical writing scenarios demonstrates that EDU-Prompting significantly enhances both content truthfulness and logical soundness in AI-generated educational responses. The framework's modular design enables seamless integration into existing prompting frameworks and educational applications, allowing practitioners to directly incorporate critical thinking catalysts that promote analytical reasoning and introduce multiple perspectives without requiring extensive system modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14642v2">How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-20
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Recently, LLMs have garnered increasing attention across academic disciplines for their potential as human digital twins, virtual proxies designed to replicate individuals and autonomously perform tasks such as decision-making, problem-solving, and reasoning on their behalf. However, current evaluations of LLMs primarily emphasize dialogue simulation while overlooking human behavior simulation, which is crucial for digital twins. To address this gap, we introduce BehaviorChain, the first benchmark for evaluating LLMs' ability to simulate continuous human behavior. BehaviorChain comprises diverse, high-quality, persona-based behavior chains, totaling 15,846 distinct behaviors across 1,001 unique personas, each with detailed history and profile metadata. For evaluation, we integrate persona metadata into LLMs and employ them to iteratively infer contextually appropriate behaviors within dynamic scenarios provided by BehaviorChain. Comprehensive evaluation results demonstrated that even state-of-the-art models struggle with accurately simulating continuous human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14995v1">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and lack of customized modeling frameworks. This paper proposed an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategy, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation learning. A differential attention-based critic network is designed to enhance convergence performance. Experimental results demonstrate that LLM generated strategies effectively substitute human experts. The proposed multi-agent imitation learning algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baselines algorithms, while maintaining robust stability. This work provides an effective solution for real-time P2P electricity market decision-making by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10467v4">Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 This work has been submitted for publication. Copyright may be transferred. In this case, this version will be updated with a notice, according to the publisher's guidelines
    </div>
    <details class="paper-abstract">
      Recent advancements in LLMs indicate potential for novel applications, as evidenced by the reasoning capabilities in the latest OpenAI and DeepSeek models. To apply these models to domain-specific applications beyond text generation, LLM-based multi-agent systems can be utilized to solve complex tasks, particularly by combining reasoning techniques, code generation, and software execution across multiple, potentially specialized LLMs. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application are not well understood. Defined specifications for multi-agent LLM systems are required to explore their potential and suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research on (1.) multi-agent specification by introducing an agent schema language and (2.) the execution and evaluation of the specifications through a multi-agent system architecture and prototype. The specification language, system architecture, and prototype are first presented in this work, building on an LLM system from prior research. Test cases involving cybersecurity tasks indicate the feasibility of the architecture and evaluation approach. As a result, evaluations could be demonstrated for question answering, server security, and network security tasks completed correctly by agents with LLMs from OpenAI and DeepSeek.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14735v1">Investigating the Role of LLMs Hyperparameter Tuning and Prompt Engineering to Support Domain Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Accepted at 51st Euromicro Conference Series on Software Engineering and Advanced Applications (SEAA)
    </div>
    <details class="paper-abstract">
      The introduction of large language models (LLMs) has enhanced automation in software engineering tasks, including in Model Driven Engineering (MDE). However, using general-purpose LLMs for domain modeling has its limitations. One approach is to adopt fine-tuned models, but this requires significant computational resources and can lead to issues like catastrophic forgetting. This paper explores how hyperparameter tuning and prompt engineering can improve the accuracy of the Llama 3.1 model for generating domain models from textual descriptions. We use search-based methods to tune hyperparameters for a specific medical data model, resulting in a notable quality improvement over the baseline LLM. We then test the optimized hyperparameters across ten diverse application domains. While the solutions were not universally applicable, we demonstrate that combining hyperparameter tuning with prompt engineering can enhance results across nearly all examined domain models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14719v1">Automated Safety Evaluations Across 20 Large Language Models: The Aymara LLM Risk and Responsibility Matrix</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into real-world applications, scalable and rigorous safety evaluation is essential. This paper introduces Aymara AI, a programmatic platform for generating and administering customized, policy-grounded safety evaluations. Aymara AI transforms natural-language safety policies into adversarial prompts and scores model responses using an AI-based rater validated against human judgments. We demonstrate its capabilities through the Aymara LLM Risk and Responsibility Matrix, which evaluates 20 commercially available LLMs across 10 real-world safety domains. Results reveal wide performance disparities, with mean safety scores ranging from 86.2% to 52.4%. While models performed well in well-established safety domains such as Misinformation (mean = 95.7%), they consistently failed in more complex or underspecified domains, notably Privacy & Impersonation (mean = 24.3%). Analyses of Variance confirmed that safety scores differed significantly across both models and domains (p < .05). These findings underscore the inconsistent and context-dependent nature of LLM safety and highlight the need for scalable, customizable tools like Aymara AI to support responsible AI development and oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14705v1">Configurable multi-agent framework for scalable and realistic testing of llm-based agents</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM) agents exhibit complex, context-sensitive behaviour that quickly renders static benchmarks and ad-hoc manual testing obsolete. We present Neo, a configurable, multi-agent framework that automates realistic, multi-turn evaluation of LLM-based systems. Neo couples a Question Generation Agent and an Evaluation Agent through a shared context-hub, allowing domain prompts, scenario controls and dynamic feedback to be composed modularly. Test inputs are sampled from a probabilistic state model spanning dialogue flow, user intent and emotional tone, enabling diverse, human-like conversations that adapt after every turn. Applied to a production-grade Seller Financial Assistant chatbot, Neo (i) uncovered edge-case failures across five attack categories with a 3.3% break rate close to the 5.8% achieved by expert human red-teamers, and (ii) delivered 10-12X higher throughput, generating 180 coherent test questions in around 45 mins versus 16h of human effort. Beyond security probing, Neo's stochastic policies balanced topic coverage and conversational depth, yielding broader behavioural exploration than manually crafted scripts. Neo therefore lays a foundation for scalable, self-evolving LLM QA: its agent interfaces, state controller and feedback loops are model-agnostic and extensible to richer factual-grounding and policy-compliance checks. We release the framework to facilitate reproducible, high-fidelity testing of emerging agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08263v2">LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Tangled code changes, commits that conflate unrelated modifications such as bug fixes, refactorings, and enhancements, introduce significant noise into bug datasets and adversely affect the performance of bug prediction models. Addressing this issue at a fine-grained, method-level granularity remains underexplored. This is critical to address, as recent bug prediction models, driven by practitioner demand, are increasingly focusing on finer granularity rather than traditional class- or file-level predictions. This study investigates the utility of Large Language Models (LLMs) for detecting tangled code changes by leveraging both commit messages and method-level code diffs. We formulate the problem as a binary classification task and evaluate multiple prompting strategies, including zero-shot, few-shot, and chain-of-thought prompting, using state-of-the-art proprietary LLMs such as GPT-4o and Gemini-2.0-Flash. Our results demonstrate that combining commit messages with code diffs significantly enhances model performance, with the combined few-shot and chain-of-thought prompting achieving an F1-score of 0.88. Additionally, we explore machine learning models trained on LLM-generated embeddings, where a multi-layer perceptron classifier achieves superior performance (F1-score: 0.906, MCC: 0.807). Applying our approach to 49 open-source projects improves the distributional separability of code metrics between buggy and non-buggy methods, demonstrating the promise of LLMs for method-level commit untangling and potentially contributing to improving the accuracy of future bug prediction models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18765v3">A Vision for Auto Research with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      This paper introduces Agent-Based Auto Research, a structured multi-agent framework designed to automate, coordinate, and optimize the full lifecycle of scientific research. Leveraging the capabilities of large language models (LLMs) and modular agent collaboration, the system spans all major research phases, including literature review, ideation, methodology planning, experimentation, paper writing, peer review response, and dissemination. By addressing issues such as fragmented workflows, uneven methodological expertise, and cognitive overload, the framework offers a systematic and scalable approach to scientific inquiry. Preliminary explorations demonstrate the feasibility and potential of Auto Research as a promising paradigm for self-improving, AI-driven research processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v3">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite improvements in automatic speech recognition, performance drops with accented speech. Generative error correction (GER) leverages the linguistic knowledge of large language models (LLMs), outperforming typical language model methods. However, it lacks specificity in accented speech scenarios. Accents represent deviations from standard pronunciation, making multi-granularity pronunciation and semantic information essential for accented speech recognition. Moreover, accents exhibit considerable diversity, with each accent possessing distinct characteristics. In this study, we leverage GER to improve transcription accuracy by addressing the two primary features. We propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level pronunciation information. These methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through low-rank adaptation (LoRA) fine-tuning. We employ a three-stage strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge mono-accent LoRA experts within a single multi-modal GER to overcome accent diversity challenges. Furthermore, multi-granularity GER leverages N-best word-level and phoneme-level hypotheses from the HDMoLE model to predict final transcriptions. Experiments on a multi-accent English dataset show that our methods reduce word error rate by 67.35% compared to the baseline vanilla Whisper-large-v3 model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14649v1">Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Despite the outstanding performance of large language models (LLMs) across various NLP tasks, hallucinations in LLMs--where LLMs generate inaccurate responses--remains as a critical problem as it can be directly connected to a crisis of building safe and reliable LLMs. Uncertainty estimation is primarily used to measure hallucination levels in LLM responses so that correct and incorrect answers can be distinguished clearly. This study proposes an effective uncertainty estimation approach, \textbf{Cl}ust\textbf{e}ring-based sem\textbf{an}tic con\textbf{s}ist\textbf{e}ncy (\textbf{Cleanse}). Cleanse quantifies the uncertainty with the proportion of the intra-cluster consistency in the total consistency between LLM hidden embeddings which contain adequate semantic information of generations, by employing clustering. The effectiveness of Cleanse for detecting hallucination is validated using four off-the-shelf models, LLaMA-7B, LLaMA-13B, LLaMA2-7B and Mistral-7B and two question-answering benchmarks, SQuAD and CoQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03108v3">OMNISEC: LLM-Driven Provenance-based Intrusion Detection via Retrieval-Augmented Behavior Prompting</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Recently, Provenance-based Intrusion Detection Systems (PIDSes) have been widely used for endpoint threat analysis. These studies can be broadly categorized into rule-based detection systems and learning-based detection systems. Among these, due to the evolution of attack techniques, rules cannot dynamically model all the characteristics of attackers. As a result, such systems often face false negatives. Learning-based detection systems are further divided into supervised learning and anomaly detection. The scarcity of attack samples hinders the usability and effectiveness of supervised learning-based detection systems in practical applications. Anomaly-based detection systems face a massive false positive problem because they cannot distinguish between changes in normal behavior and real attack behavior. The alert results of detection systems are closely related to the manual labor costs of subsequent security analysts. To reduce manual analysis time, we propose OMNISEC, which applies large language models (LLMs) to anomaly-based intrusion detection systems via retrieval-augmented behavior prompting. OMNISEC can identify abnormal nodes and corresponding abnormal events by constructing suspicious nodes and rare paths. By combining two external knowledge bases, OMNISEC uses Retrieval Augmented Generation (RAG) to enable the LLM to determine whether abnormal behavior is a real attack. Finally, OMNISEC can reconstruct the attack graph and restore the complete attack behavior chain of the attacker's intrusion. Experimental results show that OMNISEC outperforms state-of-the-art methods on public benchmark datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14558v1">Harnessing LLMs for Document-Guided Fuzzing of OpenCV Library</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      The combination of computer vision and artificial intelligence is fundamentally transforming a broad spectrum of industries by enabling machines to interpret and act upon visual data with high levels of accuracy. As the biggest and by far the most popular open-source computer vision library, OpenCV library provides an extensive suite of programming functions supporting real-time computer vision. Bugs in the OpenCV library can affect the downstream computer vision applications, and it is critical to ensure the reliability of the OpenCV library. This paper introduces VISTAFUZZ, a novel technique for harnessing large language models (LLMs) for document-guided fuzzing of the OpenCV library. VISTAFUZZ utilizes LLMs to parse API documentation and obtain standardized API information. Based on this standardized information, VISTAFUZZ extracts constraints on individual input parameters and dependencies between these. Using these constraints and dependencies, VISTAFUZZ then generates new input values to systematically test each target API. We evaluate the effectiveness of VISTAFUZZ in testing 330 APIs in the OpenCV library, and the results show that VISTAFUZZ detected 17 new bugs, where 10 bugs have been confirmed, and 5 of these have been fixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12891v2">TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Second version
    </div>
    <details class="paper-abstract">
      Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , and the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23644v3">QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 The experimental data in the experimental section needs to be improved, and there are some errors
    </div>
    <details class="paper-abstract">
      We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source projects.We constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08373v2">Draft-based Approximate Inference for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Added discussion and comparison with SpecPrefill
    </div>
    <details class="paper-abstract">
      Optimizing inference for long-context Large Language Models (LLMs) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, such as key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on rough predictions of token or KV pair importance. We propose a novel framework for approximate LLM inference that leverages small draft models to more accurately predict the importance of tokens and KV pairs. Specifically, we introduce two instantiations of our proposed framework: (i) SpecKV, the first method that leverages a draft output to accurately assess the importance of each KV pair for more effective KV cache dropping, and (ii) SpecPC, which uses the draft model's attention activations to identify and discard unimportant prompt tokens. We motivate our methods with theoretical and empirical analyses, and show a strong correlation between the attention patterns of draft and target models. Extensive experiments on long-context benchmarks show that our methods consistently achieve higher accuracy than existing baselines, while preserving the same improvements in memory usage, latency, and throughput. Our code is available at https://github.com/furiosa-ai/draft-based-approx-llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08493v3">Intelligent LiDAR Navigation: Leveraging External Information and Semantic Maps with LLM as Copilot</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Accepted at IROS 2025
    </div>
    <details class="paper-abstract">
      Traditional robot navigation systems primarily utilize occupancy grid maps and laser-based sensing technologies, as demonstrated by the popular move_base package in ROS. Unlike robots, humans navigate not only through spatial awareness and physical distances but also by integrating external information, such as elevator maintenance updates from public notification boards and experiential knowledge, like the need for special access through certain doors. With the development of Large Language Models (LLMs), which possesses text understanding and intelligence close to human performance, there is now an opportunity to infuse robot navigation systems with a level of understanding akin to human cognition. In this study, we propose using osmAG (Area Graph in OpensStreetMap textual format), an innovative semantic topometric hierarchical map representation, to bridge the gap between the capabilities of ROS move_base and the contextual understanding offered by LLMs. Our methodology employs LLMs as an actual copilot in robot navigation, enabling the integration of a broader range of informational inputs while maintaining the robustness of traditional robotic navigation systems. Our code, demo, map, experiment results can be accessed at https://github.com/xiexiexiaoxiexie/Intelligent-LiDAR-Navigation-LLM-as-Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20016v3">Vulnerability of LLMs to Vertically Aligned Text Manipulations</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Accepted to ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Vertical text input is commonly encountered in various real-world applications, such as mathematical computations and word-based Sudoku puzzles. While current large language models (LLMs) have excelled in natural language tasks, they remain vulnerable to variations in text formatting. Recent research demonstrates that modifying input formats, such as vertically aligning words for encoder-based models, can substantially lower accuracy in text classification tasks. While easily understood by humans, these inputs can significantly mislead models, posing a potential risk of bypassing detection in real-world scenarios involving harmful or sensitive information. With the expanding application of LLMs, a crucial question arises: \textit{Do decoder-based LLMs exhibit similar vulnerabilities to vertically formatted text input?} In this paper, we investigate the impact of vertical text input on the performance of various LLMs across multiple text classification datasets and analyze the underlying causes. Our findings are as follows: (i) Vertical text input significantly degrades the accuracy of LLMs in text classification tasks. (ii) \textit{Chain of Thought (CoT)} reasoning does not help LLMs recognize vertical input or mitigate its vulnerability, but \textit{few-shot learning} with careful analysis does. (iii) We explore the underlying cause of the vulnerability by analyzing the inherent issues in tokenization and attention matrices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14430v1">X-Intelligence 3.0: Training and Evaluating Reasoning LLM for Semiconductor Display</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved significant advances in reasoning and demonstrated their advantages in solving challenging problems. Yet, their effectiveness in the semiconductor display industry remains limited due to a lack of domain-specific training and expertise. To bridge this gap, we present X-Intelligence 3.0, the first high-performance reasoning model specifically developed for the semiconductor display industry. This model is designed to deliver expert-level understanding and reasoning for the industry's complex challenges. Leveraging a carefully curated industry knowledge base, the model undergoes supervised fine-tuning and reinforcement learning to enhance its reasoning and comprehension capabilities. To further accelerate development, we implemented an automated evaluation framework that simulates expert-level assessments. We also integrated a domain-specific retrieval-augmented generation (RAG) mechanism, resulting in notable performance gains on benchmark datasets. Despite its relatively compact size of 32 billion parameters, X-Intelligence 3.0 outperforms SOTA DeepSeek-R1-671B across multiple evaluations. This demonstrates its exceptional efficiency and establishes it as a powerful solution to the longstanding reasoning challenges faced by the semiconductor display industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16841v1">AquaChat: An LLM-Guided ROV Framework for Adaptive Inspection of Aquaculture Net Pens</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Inspection of aquaculture net pens is essential for maintaining the structural integrity, biosecurity, and operational efficiency of fish farming systems. Traditional inspection approaches rely on pre-programmed missions or manual control, offering limited adaptability to dynamic underwater conditions and user-specific demands. In this study, we propose AquaChat, a novel Remotely Operated Vehicle (ROV) framework that integrates Large Language Models (LLMs) for intelligent and adaptive net pen inspection. The system features a multi-layered architecture: (1) a high-level planning layer that interprets natural language user commands using an LLM to generate symbolic task plans; (2) a mid-level task manager that translates plans into ROV control sequences; and (3) a low-level motion control layer that executes navigation and inspection tasks with precision. Real-time feedback and event-triggered replanning enhance robustness in challenging aquaculture environments. The framework is validated through experiments in both simulated and controlled aquatic environments representative of aquaculture net pens. Results demonstrate improved task flexibility, inspection accuracy, and operational efficiency. AquaChat illustrates the potential of integrating language-based AI with marine robotics to enable intelligent, user-interactive inspection systems for sustainable aquaculture operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15090v2">Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to study representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., ''cat'') and then analyze the corresponding activation patterns. We find that LLM representations captured this way closely align with human representations inferred from behavioral data, matching inter-human alignment levels. Our approach significantly outperforms the alignment captured by word embeddings, which have been the focus of prior work on human-LLM alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts -- we show that LLMs organize concepts in a way that mirrors human concept organization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14355v1">Can LLMs Infer Personality from Real World Conversations?</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 21 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as OpenAI's GPT-4 and Meta's LLaMA offer a promising approach for scalable personality assessment from open-ended language. However, inferring personality traits remains challenging, and earlier work often relied on synthetic data or social media text lacking psychometric validity. We introduce a real-world benchmark of 555 semi-structured interviews with BFI-10 self-report scores for evaluating LLM-based personality inference. Three state-of-the-art LLMs (GPT-4.1 Mini, Meta-LLaMA, and DeepSeek) were tested using zero-shot prompting for BFI-10 item prediction and both zero-shot and chain-of-thought prompting for Big Five trait inference. All models showed high test-retest reliability, but construct validity was limited: correlations with ground-truth scores were weak (max Pearson's $r = 0.27$), interrater agreement was low (Cohen's $\kappa < 0.10$), and predictions were biased toward moderate or high trait levels. Chain-of-thought prompting and longer input context modestly improved distributional alignment, but not trait-level accuracy. These results underscore limitations in current LLM-based personality inference and highlight the need for evidence-based development for psychological applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14335v1">ProofCompass: Enhancing Specialized Provers with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 19 pages, 7 figures. Accepted at the 2nd AI for MATH Workshop at the 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Language models have become increasingly powerful tools for formal mathematical reasoning. However, most existing approaches rely exclusively on either large general-purpose models or smaller specialized models, each with distinct limitations, while training specialized large models still requires significant computational resources. This paper introduces ProofCompass, a novel hybrid methodology that achieves remarkable computational efficiency by strategically guiding existing specialized prover methods, such as DeepSeek-Prover-v1.5-RL (DSP-v1.5) with a Large Language Model (LLM) without requiring additional model training. The LLM provides natural language proof strategies and analyzes failed attempts to select intermediate lemmas, enabling effective problem decomposition. On the miniF2F benchmark, ProofCompass demonstrates substantial resource efficiency: it outperforms DSP-v1.5 ($54.9\% \rightarrow 55.3\%$) while using 25x fewer attempts ($3200 \rightarrow 128$). Our synergistic approach paves the way for simultaneously improving computational efficiency and accuracy in formal theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03619v2">Blackbox Dataset Inference for LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs. In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14330v1">Leveraging LLMs for Formal Software Requirements -- Challenges and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Submitted to Overlay2025 - 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis. [under review]
    </div>
    <details class="paper-abstract">
      Software correctness is ensured mathematically through formal verification, which involves the resources of generating formal requirement specifications and having an implementation that must be verified. Tools such as model-checkers and theorem provers ensure software correctness by verifying the implementation against the specification. Formal methods deployment is regularly enforced in the development of safety-critical systems e.g. aerospace, medical devices and autonomous systems. Generating these specifications from informal and ambiguous natural language requirements remains the key challenge. Our project, VERIFAI^{1}, aims to investigate automated and semi-automated approaches to bridge this gap, using techniques from Natural Language Processing (NLP), ontology-based domain modelling, artefact reuse, and large language models (LLMs). This position paper presents a preliminary synthesis of relevant literature to identify recurring challenges and prospective research directions in the generation of verifiable specifications from informal requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14307v1">How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit increasingly sophisticated linguistic capabilities, yet the extent to which these behaviors reflect human-like cognition versus advanced pattern recognition remains an open question. In this study, we investigate how LLMs process the temporal meaning of linguistic aspect in narratives that were previously used in human studies. Using an Expert-in-the-Loop probing pipeline, we conduct a series of targeted experiments to assess whether LLMs construct semantic representations and pragmatic inferences in a human-like manner. Our findings show that LLMs over-rely on prototypicality, produce inconsistent aspectual judgments, and struggle with causal reasoning derived from aspect, raising concerns about their ability to fully comprehend narratives. These results suggest that LLMs process aspect fundamentally differently from humans and lack robust narrative understanding. Beyond these empirical findings, we develop a standardized experimental framework for the reliable assessment of LLMs' cognitive and linguistic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14304v1">Aligning Large Language Models to Low-Resource Languages through LLM-Based Selective Translation: A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) often demonstrate a performance gap between English and non-English languages, particularly in low-resource settings. Aligning these models to low-resource languages is essential yet challenging due to limited high-quality data. While English alignment datasets are readily available, curating equivalent data in other languages is expensive and time-consuming. A common workaround is to translate existing English alignment data; however, standard translation techniques often fail to preserve critical elements such as code, mathematical expressions, and structured formats like JSON. In this work, we investigate LLM-based selective translation, a technique that selectively translates only the translatable parts of a text while preserving non-translatable content and sentence structure. We conduct a systematic study to explore key questions around this approach, including its effectiveness compared to vanilla translation, the importance of filtering noisy outputs, and the benefits of mixing translated samples with original English data during alignment. Our experiments focus on the low-resource Indic language Hindi and compare translations generated by Google Cloud Translation (GCP) and Llama-3.1-405B. The results highlight the promise of selective translation as a practical and effective method for improving multilingual alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v2">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://anonymous.4open.science/r/DiZO-E86D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13394v2">Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Evaluating machine-generated text remains a significant challenge in NLP, especially for non-English languages. Current methodologies, including automated metrics, human assessments, and LLM-based evaluations, predominantly focus on English, revealing a significant gap in multilingual evaluation frameworks. We introduce the Cross Lingual Auto Evaluation (CIA) Suite, an extensible framework that includes evaluator LLMs (Hercule) and a novel test set (Recon) specifically designed for multilingual evaluation. Our test set features 500 human-annotated instructions spanning various task capabilities along with human judgment scores across six languages. This would enable benchmarking of general-purpose multilingual LLMs and facilitate meta-evaluation of Evaluator LLMs. The proposed model, Hercule, is a cross-lingual evaluation model that addresses the scarcity of reference answers in the target language by learning to assign scores to responses based on easily available reference answers in English. Our experiments demonstrate that Hercule aligns more closely with human judgments compared to proprietary models, demonstrating the effectiveness of such cross-lingual evaluation in low resource scenarios. Further, it is also effective in zero-shot evaluation on unseen languages. This study is the first comprehensive examination of cross-lingual evaluation using LLMs, presenting a scalable and effective approach for multilingual assessment. All code, datasets, and models will be publicly available to enable further research in this important area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13933v1">Preprint: Did I Just Browse A Website Written by LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 In submission. 2 pages. 3 figures
    </div>
    <details class="paper-abstract">
      Increasingly, web content is automatically generated by large language models (LLMs) with little human input. We call this "LLM-dominant" content. Since LLMs plagiarize and hallucinate, LLM-dominant content can be unreliable and unethical. Yet, websites rarely disclose such content, and human readers struggle to distinguish it. Thus, we must develop reliable detectors for LLM-dominant content. However, state-of-the-art LLM detectors are insufficient, because they perform well mainly on clean, prose-like text, while web content has complex markup and diverse genres. We propose a highly reliable, scalable pipeline that classifies entire websites. Instead of naively classifying text extracted from each page, we classify each site based on an LLM text detector's outputs of multiple prose-like pages. We train and evaluate our detector by collecting 2 distinct ground truth datasets totaling 120 sites, and obtain 100% accuracies testing across them. In the wild, we detect a sizable portion of sites as LLM-dominant among 10k sites in search engine results and 10k in Common Crawl archives. We find LLM-dominant sites are growing in prevalence and rank highly in search results, raising questions about their impact on end users and the overall Web ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13881v1">Using LLMs to identify features of personal and professional skills in an open-response situational judgment test</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 10 pages, 2 figures, 4 tables; this work was accepted for presentation at the 2025 Artificial Intelligence in Measurement and Education Conference in Pittsburgh, Pennsylvania, United States
    </div>
    <details class="paper-abstract">
      Academic programs are increasingly recognizing the importance of personal and professional skills and their critical role alongside technical expertise in preparing students for future success in diverse career paths. With this growing demand comes the need for scalable systems to measure, evaluate, and develop these skills. Situational Judgment Tests (SJTs) offer one potential avenue for measuring these skills in a standardized and reliable way, but open-response SJTs have traditionally relied on trained human raters for evaluation, presenting operational challenges to delivering SJTs at scale. Past attempts at developing NLP-based scoring systems for SJTs have fallen short due to issues with construct validity of these systems. In this article, we explore a novel approach to extracting construct-relevant features from SJT responses using large language models (LLMs). We use the Casper SJT to demonstrate the efficacy of this approach. This study sets the foundation for future developments in automated scoring for personal and professional skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13859v1">SPARQL Query Generation with LLMs: Measuring the Impact of Training Data Memorization and Knowledge Injection</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Winner of Best Paper Award at the 25th International Conference on Web Engineering (ICWE 2025)
    </div>
    <details class="paper-abstract">
      Nowadays, the importance of software with natural-language user interfaces cannot be underestimated. In particular, in Question Answering (QA) systems, generating a SPARQL query for a given natural-language question (often named Query Building) from the information retrieved from the same question is the central task of QA systems working over Knowledge Graphs (KGQA). Due to the rise of Large Language Models (LLMs), they are considered a well-suited method to increase the quality of the question-answering functionality, as there is still a lot of room for improvement, aiming for enhanced quality and trustworthiness. However, LLMs are trained on web data, where researchers have no control over whether the benchmark or the knowledge graph was already included in the training data. In this paper, we introduce a novel method that evaluates the quality of LLMs by generating a SPARQL query from a natural-language question under various conditions: (1) zero-shot SPARQL generation, (2) with knowledge injection, and (3) with "anonymized" knowledge injection. This enables us, for the first time, to estimate the influence of the training data on the QA quality improved by LLMs. Ultimately, this will help to identify how portable a method is or whether good results might mostly be achieved because a benchmark was already included in the training data (cf. LLM memorization). The developed method is portable, robust, and supports any knowledge graph; therefore, it could be easily applied to any KGQA or LLM, s.t., generating consistent insights into the actual LLM capabilities is possible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13833v1">DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the pivotal post-training technique for large language model. Effectively scaling reinforcement learning is now the key to unlocking advanced reasoning capabilities and ensuring safe, goal-aligned behavior in the most powerful LLMs. Mainstream frameworks usually employ a hybrid-controller architecture where a single-controller dispatches the overall execution logic and manages overall data transfer and the multi-controller executes distributed computation. For large-scale reinforcement learning, minor load imbalances can introduce significant bottlenecks, ultimately constraining the scalability of the system. To address this limitation, we introduce DistFlow, a novel, fully distributed RL framework designed to break scaling barrier. We adopt a multi-controller paradigm that dispatches data transfer and execution tasks to all workers, which eliminates the centralized node. This allows each worker to operate independently, leading to near-linear scalability up to thousands of GPUs and dramatic efficiency gains. Furthermore, our architecture decouples resource configuration from execution logic, allowing each worker to have a unique execution flow, offering significant flexibility for rapid and cost-effective algorithmic experimentation. Extensive experiments show that DistFlow achieves excellent linear scalability and up to a 7x end-to-end throughput improvement over state-of-the-art (SOTA) frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v3">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13822v1">RAG-based Architectures for Drug Side Effect Retrieval in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Drug side effects are a major global health concern, necessitating advanced methods for their accurate detection and analysis. While Large Language Models (LLMs) offer promising conversational interfaces, their inherent limitations, including reliance on black-box training data, susceptibility to hallucinations, and lack of domain-specific knowledge, hinder their reliability in specialized fields like pharmacovigilance. To address this gap, we propose two architectures: Retrieval-Augmented Generation (RAG) and GraphRAG, which integrate comprehensive drug side effect knowledge into a Llama 3 8B language model. Through extensive evaluations on 19,520 drug side effect associations (covering 976 drugs and 3,851 side effect terms), our results demonstrate that GraphRAG achieves near-perfect accuracy in drug side effect retrieval. This framework offers a highly accurate and scalable solution, signifying a significant advancement in leveraging LLMs for critical pharmacovigilance applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13774v2">DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 This is the updated version of the preprint, revised following acceptance for publication in Elsevier Neural Networks Journal. The paper is now published (18 July 2025) with DOI: https://doi.org/10.1016/j.neunet.2025.107879
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently revolutionized language processing tasks but have also brought ethical and legal issues. LLMs have a tendency to memorize potentially private or copyrighted information present in the training data, which might then be delivered to end users at inference time. When this happens, a naive solution is to retrain the model from scratch after excluding the undesired data. Although this guarantees that the target data have been forgotten, it is also prohibitively expensive for LLMs. Approximate unlearning offers a more efficient alternative, as it consists of ex post modifications of the trained model itself to prevent undesirable results, but it lacks forgetting guarantees because it relies solely on empirical evidence. In this work, we present DP2Unlearning, a novel LLM unlearning framework that offers formal forgetting guarantees at a significantly lower cost than retraining from scratch on the data to be retained. DP2Unlearning involves training LLMs on textual data protected using {\epsilon}-differential privacy (DP), which later enables efficient unlearning with the guarantees against disclosure associated with the chosen {\epsilon}. Our experiments demonstrate that DP2Unlearning achieves similar model performance post-unlearning, compared to an LLM retraining from scratch on retained data -- the gold standard exact unlearning -- but at approximately half the unlearning cost. In addition, with a reasonable computational cost, it outperforms approximate unlearning methods at both preserving the utility of the model post-unlearning and effectively forgetting the targeted information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08924v2">From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      The development of Large Language Models (LLMs) requires robust benchmarks that encompass not only academic domains but also industrial fields to effectively evaluate their applicability in real-world scenarios. In this paper, we introduce two Korean expert-level benchmarks. KMMLU-Redux, reconstructed from the existing KMMLU, consists of questions from the Korean National Technical Qualification exams, with critical errors removed to enhance reliability. KMMLU-Pro is based on Korean National Professional Licensure exams to reflect professional knowledge in Korea. Our experiments demonstrate that these benchmarks comprehensively represent industrial knowledge in Korea. We release our dataset publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13743v1">PRIDE -- Parameter-Efficient Reduction of Identity Discrimination for Equality in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently reproduce the gender- and sexual-identity prejudices embedded in their training corpora, leading to outputs that marginalize LGBTQIA+ users. Hence, reducing such biases is of great importance. To achieve this, we evaluate two parameter-efficient fine-tuning (PEFT) techniques - Low-Rank Adaptation (LoRA) and soft-prompt tuning - as lightweight alternatives to full-model fine-tuning for mitigating such biases. Using the WinoQueer benchmark, we quantify bias in three open-source LLMs and observe baseline bias scores reaching up to 98 (out of 100) across a range of queer identities defined by gender and/or sexual orientation, where 50 would indicate neutrality. Fine-tuning with LoRA (< 0.1% additional parameters) on a curated QueerNews corpus reduces those scores by up to 50 points and raises neutrality from virtually 0% to as much as 36%. Soft-prompt tuning (10 virtual tokens) delivers only marginal improvements. These findings show that LoRA can deliver meaningful fairness gains with minimal computation. We advocate broader adoption of community-informed PEFT, the creation of larger queer-authored corpora, and richer evaluation suites beyond WinoQueer, coupled with ongoing audits to keep LLMs inclusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02145v4">From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Final Version and Paper Accepted at IEEE ITSC 2025
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13737v1">DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Rich and context-aware activity logs facilitate user behavior analysis and health monitoring, making them a key research focus in ubiquitous computing. The remarkable semantic understanding and generation capabilities of Large Language Models (LLMs) have recently created new opportunities for activity log generation. However, existing methods continue to exhibit notable limitations in terms of accuracy, efficiency, and semantic richness. To address these challenges, we propose DailyLLM. To the best of our knowledge, this is the first log generation and summarization system that comprehensively integrates contextual activity information across four dimensions: location, motion, environment, and physiology, using only sensors commonly available on smartphones and smartwatches. To achieve this, DailyLLM introduces a lightweight LLM-based framework that integrates structured prompting with efficient feature extraction to enable high-level activity understanding. Extensive experiments demonstrate that DailyLLM outperforms state-of-the-art (SOTA) log generation methods and can be efficiently deployed on personal computers and Raspberry Pi. Utilizing only a 1.5B-parameter LLM model, DailyLLM achieves a 17% improvement in log generation BERTScore precision compared to the 70B-parameter SOTA baseline, while delivering nearly 10x faster inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13729v1">AGENTS-LLM: Augmentative GENeration of Challenging Traffic Scenarios with an Agentic LLM Framework</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Rare, yet critical, scenarios pose a significant challenge in testing and evaluating autonomous driving planners. Relying solely on real-world driving scenes requires collecting massive datasets to capture these scenarios. While automatic generation of traffic scenarios appears promising, data-driven models require extensive training data and often lack fine-grained control over the output. Moreover, generating novel scenarios from scratch can introduce a distributional shift from the original training scenes which undermines the validity of evaluations especially for learning-based planners. To sidestep this, recent work proposes to generate challenging scenarios by augmenting original scenarios from the test set. However, this involves the manual augmentation of scenarios by domain experts. An approach that is unable to meet the demands for scale in the evaluation of self-driving systems. Therefore, this paper introduces a novel LLM-agent based framework for augmenting real-world traffic scenarios using natural language descriptions, addressing the limitations of existing methods. A key innovation is the use of an agentic design, enabling fine-grained control over the output and maintaining high performance even with smaller, cost-effective LLMs. Extensive human expert evaluation demonstrates our framework's ability to accurately adhere to user intent, generating high quality augmented scenarios comparable to those created manually.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13712v1">LLaPipe: LLM-Guided Reinforcement Learning for Automated Data Preparation Pipeline Construction</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Automated data preparation is crucial for democratizing machine learning, yet existing reinforcement learning (RL) based approaches suffer from inefficient exploration in the vast space of possible preprocessing pipelines. We present LLaPipe, a novel framework that addresses this exploration bottleneck by integrating Large Language Models (LLMs) as intelligent policy advisors. Unlike traditional methods that rely solely on statistical features and blind trial-and-error, LLaPipe leverages the semantic understanding capabilities of LLMs to provide contextually relevant exploration guidance. Our framework introduces three key innovations: (1) an LLM Policy Advisor that analyzes dataset semantics and pipeline history to suggest promising preprocessing operations, (2) an Experience Distillation mechanism that mines successful patterns from past pipelines and transfers this knowledge to guide future exploration, and (3) an Adaptive Advisor Triggering strategy (Advisor\textsuperscript{+}) that dynamically determines when LLM intervention is most beneficial, balancing exploration effectiveness with computational cost. Through extensive experiments on 18 diverse datasets spanning multiple domains, we demonstrate that LLaPipe achieves up to 22.4\% improvement in pipeline quality and 2.3$\times$ faster convergence compared to state-of-the-art RL-based methods, while maintaining computational efficiency through selective LLM usage (averaging only 19.0\% of total exploration steps).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17562v2">LLM-driven Medical Report Generation via Communication-efficient Heterogeneous Federated Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Accepted by IEEE TMI
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated significant potential in Medical Report Generation (MRG), yet their development requires large amounts of medical image-report pairs, which are commonly scattered across multiple centers. Centralizing these data is exceptionally challenging due to privacy regulations, thereby impeding model development and broader adoption of LLM-driven MRG models. To address this challenge, we present FedMRG, the first framework that leverages Federated Learning (FL) to enable privacy-preserving, multi-center development of LLM-driven MRG models, specifically designed to overcome the critical challenge of communication-efficient LLM training under multi-modal data heterogeneity. To start with, our framework tackles the fundamental challenge of communication overhead in FL-LLM tuning by employing low-rank factorization to efficiently decompose parameter updates, significantly reducing gradient transmission costs and making LLM-driven MRG feasible in bandwidth-constrained FL settings. Furthermore, we observed the dual heterogeneity in MRG under the FL scenario: varying image characteristics across medical centers, as well as diverse reporting styles and terminology preferences. To address this, we further enhance FedMRG with (1) client-aware contrastive learning in the MRG encoder, coupled with diagnosis-driven prompts, which capture both globally generalizable and locally distinctive features while maintaining diagnostic accuracy; and (2) a dual-adapter mutual boosting mechanism in the MRG decoder that harmonizes generic and specialized adapters to address variations in reporting styles and terminology. Through extensive evaluation of our established FL-MRG benchmark, we demonstrate the generalizability and adaptability of FedMRG, underscoring its potential in harnessing multi-center data and generating clinically accurate reports while maintaining communication efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17735v2">SafeAgent: Safeguarding LLM Agents via an Automated Risk Simulator</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 38 pages;12 figures;12 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents are increasingly deployed in real-world applications such as "digital assistants, autonomous customer service, and decision-support systems", where their ability to "interact in multi-turn, tool-augmented environments" makes them indispensable. However, ensuring the safety of these agents remains a significant challenge due to the diverse and complex risks arising from dynamic user interactions, external tool usage, and the potential for unintended harmful behaviors. To address this critical issue, we propose AutoSafe, the first framework that systematically enhances agent safety through fully automated synthetic data generation. Concretely, 1) we introduce an open and extensible threat model, OTS, which formalizes how unsafe behaviors emerge from the interplay of user instructions, interaction contexts, and agent actions. This enables precise modeling of safety risks across diverse scenarios. 2) we develop a fully automated data generation pipeline that simulates unsafe user behaviors, applies self-reflective reasoning to generate safe responses, and constructs a large-scale, diverse, and high-quality safety training dataset-eliminating the need for hazardous real-world data collection. To evaluate the effectiveness of our framework, we design comprehensive experiments on both synthetic and real-world safety benchmarks. Results demonstrate that AutoSafe boosts safety scores by 45% on average and achieves a 28.91% improvement on real-world tasks, validating the generalization ability of our learned safety strategies. These results highlight the practical advancement and scalability of AutoSafe in building safer LLM-based agents for real-world deployment. We have released the project page at https://auto-safe.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13705v1">Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Short paper accepted at the Nineteenth ACM Conference on Recommender Systems (RecSys '25). Cedric Waterschoot, Nava Tintarev, and Francesco Barile. 2025. Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations. Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys '25), Prague, Czech Republic. doi: 10.1145/3705328.3748015
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being implemented as joint decision-makers and explanation generators for Group Recommender Systems (GRS). In this paper, we evaluate these recommendations and explanations by comparing them to social choice-based aggregation strategies. Our results indicate that LLM-generated recommendations often resembled those produced by Additive Utilitarian (ADD) aggregation. However, the explanations typically referred to averaging ratings (resembling but not identical to ADD aggregation). Group structure, uniform or divergent, did not impact the recommendations. Furthermore, LLMs regularly claimed additional criteria such as user or item similarity, diversity, or used undefined popularity metrics or thresholds. Our findings have important implications for LLMs in the GRS pipeline as well as standard aggregation strategies. Additional criteria in explanations were dependent on the number of ratings in the group scenario, indicating potential inefficiency of standard aggregation methods at larger item set sizes. Additionally, inconsistent and ambiguous explanations undermine transparency and explainability, which are key motivations behind the use of LLMs for GRS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20839v3">FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      As large language models become increasingly prevalent, memory bandwidth constraints significantly limit inference throughput, motivating post-training quantization (PTQ). In this paper, we propose FireQ, a co-designed PTQ framework and an INT4-FP8 matrix multiplication kernel that accelerates LLM inference across all linear layers. Specifically, FireQ quantizes linear layer weights and key-values to INT4, and activations and queries to FP8, significantly enhancing throughput. Additionally, we introduce a three-stage pipelining for the prefill phase, which modifies the FlashAttention-3 kernel, effectively reducing time-to-first-token in the prefill phase. To minimize accuracy loss from quantization, we develop novel outlier smoothing techniques tailored separately for linear and attention layers. In linear layers, we explicitly use per-tensor scaling to prevent underflow caused by the FP8 quantization scaling factor of INT4 quantization, and channel-wise scaling to compensate for coarse granularity of INT4. In attention layers, we address quantization challenges posed by rotary positional embeddings (RoPE) by combining pre-RoPE and post-RoPE scaling strategies. FireQ significantly outperforms state-of-the-art methods, achieving 1.68x faster inference in feed-forward network layers on Llama2-7B and 1.26x faster prefill phase performance on Llama3-8B compared to QServe, with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13681v1">LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Multi-turn dialogues are essential in many real-world applications of large language models, such as chatbots and virtual assistants. As conversation histories become longer, existing large language models face increasing computational and memory challenges, which hinder their ability to provide efficient and responsive interactions. Most current acceleration methods either compress the context or optimize key value caching, but they often rely on fixed or position-based heuristics that do not adapt well to the dynamic and unpredictable patterns found in actual multi-turn conversations. In this paper, we present LoopServe, an adaptive dual-phase inference acceleration framework for large language models in multi-turn dialogues. LoopServe introduces two main innovations. First, it performs online sparsification during the prefilling phase by dynamically selecting the most important parts of the attention matrix for each new input. Second, it uses progressive key value compression during decoding by adaptively maintaining a relevant and efficient cache based on the most recently generated output tokens. We also propose a \href{https://huggingface.co/datasets/TreeAILab/Multi-turn_Long-context_Benchmark_for_LLMs}{new benchmark} with eleven multi-turn datasets that reflect realistic query positions and conversational dependencies. Extensive experiments demonstrate that LoopServe consistently achieves superior effectiveness compared to existing baselines and significantly accelerates LLM inference across a wide range of long-context dialogue tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13666v1">KiC: Keyword-inspired Cascade for Cost-Efficient Text Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated state-of-the-art performance across a wide range of natural language processing tasks. However, high-performing models are typically accessible only via APIs, incurring substantial inference costs. Cascade methods address this by initially employing a cheaper model and escalating to a stronger one only when necessary. Nevertheless, existing cascade approaches struggle to select a reliable representative response and assess the overall reliability of free-form outputs, as they rely on exact text matching. To overcome these limitations, we propose Keyword-inspired Cascade (KiC), a novel framework for cost-efficient free-form text generation. KiC identifies the most representative answer among multiple outputs from a weaker model and evaluates the semantic alignment of other responses with it. Based on the degree of alignment, KiC determines whether to accept the weaker model's output or escalate to a stronger model. Experiments on three free-form text generation benchmarks show that KiC achieves 97.53 percent of GPT-4's accuracy while reducing API costs by 28.81 percent on average, and even outperforms GPT-4 in a specific benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08392v2">Multi-Agent LLMs as Ethics Advocates for AI-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04834v4">LLM-Based Multi-Agent Systems for Software Engineering: Literature Review, Vision and the Road Ahead</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 TOSEM 2030 Special Issue
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into autonomous agents marks a significant shift in the research landscape by offering cognitive abilities that are competitive with human planning and reasoning. This paper explores the transformative potential of integrating Large Language Models into Multi-Agent (LMA) systems for addressing complex challenges in software engineering (SE). By leveraging the collaborative and specialized abilities of multiple agents, LMA systems enable autonomous problem-solving, improve robustness, and provide scalable solutions for managing the complexity of real-world software projects. In this paper, we conduct a systematic review of recent primary studies to map the current landscape of LMA applications across various stages of the software development lifecycle (SDLC). To illustrate current capabilities and limitations, we perform two case studies to demonstrate the effectiveness of state-of-the-art LMA frameworks. Additionally, we identify critical research gaps and propose a comprehensive research agenda focused on enhancing individual agent capabilities and optimizing agent synergy. Our work outlines a forward-looking vision for developing fully autonomous, scalable, and trustworthy LMA systems, laying the foundation for the evolution of Software Engineering 2.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01551v2">EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at https://github.com/expectorlin/EvolveNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v1">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03993v4">TR-LLM: Integrating Trajectory Data for Scene-Aware LLM-Based Human Action Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Accepted to IROS 2025
    </div>
    <details class="paper-abstract">
      Accurate prediction of human behavior is crucial for AI systems to effectively support real-world applications, such as autonomous robots anticipating and assisting with human tasks. Real-world scenarios frequently present challenges such as occlusions and incomplete scene observations, which can compromise predictive accuracy. Thus, traditional video-based methods often struggle due to limited temporal and spatial perspectives. Large Language Models (LLMs) offer a promising alternative. Having been trained on a large text corpus describing human behaviors, LLMs likely encode plausible sequences of human actions in a home environment. However, LLMs, trained primarily on text data, lack inherent spatial awareness and real-time environmental perception. They struggle with understanding physical constraints and spatial geometry. Therefore, to be effective in a real-world spatial scenario, we propose a multimodal prediction framework that enhances LLM-based action prediction by integrating physical constraints derived from human trajectories. Our experiments demonstrate that combining LLM predictions with trajectory data significantly improves overall prediction performance. This enhancement is particularly notable in situations where the LLM receives limited scene information, highlighting the complementary nature of linguistic knowledge and physical constraints in understanding and anticipating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12674v2">ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong performance on programming tasks, but can they generate student-like code like real students - imperfect, iterative, and stylistically diverse? We present ParaStudent, a systematic study of LLM-based "student-like" code generation in an introductory programming course setting. Using a dataset of timestamped student submissions across multiple semesters, we design low- and high-resolution experiments to model student progress and evaluate code outputs along semantic, functional, and stylistic dimensions. Our results show that fine-tuning significantly improves alignment with real student trajectories and captures error patterns, incremental improvements, and stylistic variations more faithfully. This study shows that modeling realistic student code requires capturing learning dynamics through context-aware generation, temporal modeling, and multi-dimensional evaluation. Code for experiments and evaluation is available at https://github.com/mmiroyan/ParaStudent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10622v4">A recent evaluation on the performance of LLMs on radiation oncology physics using questions of randomly shuffled options</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Purpose: We present an updated study evaluating the performance of large language models (LLMs) in answering radiation oncology physics questions, focusing on the recently released models. Methods: A set of 100 multiple-choice radiation oncology physics questions, previously created by a well-experienced physicist, was used for this study. The answer options of the questions were randomly shuffled to create "new" exam sets. Five LLMs -- OpenAI o1-preview, GPT-4o, LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet -- with the versions released before September 30, 2024, were queried using these new exam sets. To evaluate their deductive reasoning ability, the correct answer options in the questions were replaced with "None of the above." Then, the explain-first and step-by-step instruction prompts were used to test if this strategy improved their reasoning ability. The performance of the LLMs was compared with the answers from medical physicists. Results: All models demonstrated expert-level performance on these questions, with o1-preview even surpassing medical physicists with a majority vote. When replacing the correct answer options with 'None of the above', all models exhibited a considerable decline in performance, suggesting room for improvement. The explain-first and step-by-step instruction prompts helped enhance the reasoning ability of the LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet models. Conclusion: These recently released LLMs demonstrated expert-level performance in answering radiation oncology physics questions, exhibiting great potential to assist in radiation oncology physics education and training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15838v2">Enhancing LLM Code Generation with Ensembles: A Similarity-Based Selection Approach</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Ensemble learning has been widely used in machine learning to improve model robustness, accuracy, and generalization, but has not yet been applied to code generation tasks with large language models (LLMs). We propose an ensemble approach for LLMs in code generation. Instead of relying on the output of a single model, we generate multiple candidate programs from different LLMs and apply a structured voting mechanism to select the most reliable solution. For voting, we compute syntactic and semantic similarity using CodeBLEU and behavioral equivalence using CrossHair's differential behavior analysis. By aggregating these similarity scores, we select the program that best aligns with the consensus among the candidates. We show through experiments that our ensemble approach consistently outperforms standalone LLMs on the well-known HumanEval and the more challenging LiveCodeBench datasets, achieving an accuracy of 90.2% and 50.2%, respectively, on the two datasets. In comparison, the best-performing LLM (GPT-4o) has an accuracy of 83.5% and 43.4%, respectively. Furthermore, even when restricted to free open-source models, our method achieves an accuracy of 80.5% and 41.6%, respectively, demonstrating the viability of our approach in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14406v1">Fail Fast, or Ask: Mitigating the Deficiencies of Reasoning LLMs with Human-in-the-Loop Systems Engineering</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      State-of-the-art reasoning LLMs are powerful problem solvers, but they still occasionally make mistakes. However, adopting AI models in risk-sensitive domains often requires error rates near 0%. To address this gap, we propose collaboration between a reasoning model and a human expert who resolves queries the model cannot confidently answer. We find that quantifying the uncertainty of a reasoning model through the length of its reasoning trace yields an effective basis for deferral to a human, e.g., cutting the error rate of Qwen3 235B-A22B on difficult MATH problems from 3% to less than 1% when deferring 7.5% of queries. However, the high latency of reasoning models still makes them challenging to deploy on use cases with high query volume. To address this challenge, we explore fronting a reasoning model with a large non-reasoning model. We call this modified human-in-the-loop system "Fail Fast, or Ask", since the non-reasoning model may defer difficult queries to the human expert directly ("failing fast"), without incurring the reasoning model's higher latency. We show that this approach yields around 40% latency reduction and about 50% cost savings for DeepSeek R1 while maintaining 90+% area under the accuracy-rejection curve. However, we observe that latency savings are lower than expected because of "latency drag", the phenomenon that processing easier queries with a non-reasoning model pushes the reasoning model's latency distribution towards longer latencies. Broadly, our results suggest that the deficiencies of state-of-the-art reasoning models -- nontrivial error rates and high latency -- can be substantially mitigated through black-box systems engineering, without requiring access to LLM internals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14403v1">NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Neural processing units (NPUs) are gaining prominence in power-sensitive devices like client devices, with AI PCs being defined by their inclusion of these specialized processors. Running AI workloads efficiently on these devices requires libraries of optimized kernels. Creating efficient kernels demands expertise in domain-specific C++ with vector intrinsics and in-depth knowledge of the target architecture. Unlike GPU programming, which has had years to mature, NPU programming is new, with smaller and more fragmented developer communities across hardware platforms. This fragmentation poses a challenge when utilizing LLMs to assist in writing NPU kernels, as domain-specific optimized code examples are underrepresented in LLM pre-training data. In this paper we introduce NPUEval -- a benchmark for writing and evaluating NPU kernels, consisting of 102 common operators for machine learning workloads. We evaluate LLM generated code on actual hardware based on both functional correctness and vectorization efficiency using open source compiler tools targeting the AMD NPU. We evaluate a range of state-of-the-art LLMs with a mix of proprietary and open-weight models. Latest reasoning models like DeepSeek R1, show promising results achieving out-of-the-box 50%+ vectorization on select kernels. However, the average score across the entire dataset remains roughly 10% even with compiler feedback and vectorized kernel examples -- showing that this is a challenging dataset even for frontier models. The dataset and evaluation code will be released with a permissive open source license, providing an essential benchmark for advancing research in code generation and NPU kernel optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14397v1">Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      This paper presents a limit study of transformer-based large language model (LLM) inference, focusing on the fundamental performance bottlenecks imposed by memory bandwidth, memory capacity, and synchronization overhead in distributed inference systems. We develop a hardware-agnostic performance model that abstracts away implementation details, enabling the analysis of a wide range of current and near-future hardware technologies. Our analysis spans from current HBM3 memory technology used in AI accelerators like GPUs and TPUs to systems based on advanced HBM4 and advanced 3D-stacked DRAM technology. It also covers SRAM-based designs and scaling techniques from distributed clusters with varying numbers of chips to wafer-scale integration. Our key findings for auto-regressive decoding are: i) serving LLMs requires 100s of GB per server to serve a model instance; ii) high memory bandwidth is critical for high per-user throughput; iii) exposed synchronization latencies to achieve collective communication must be around 1us else they make the memory bandwidth ineffective; iv) DRAM-based designs have a fundamental advantage in terms of system-level efficiency as measured in throughput per cost or watt; and v) hardware designs can easily reach 2000+ user token/sec but getting to 10,000+ tokens/sec will need smaller models, smaller context, or other forms of algorithmic advances. This study provides valuable insights into the fundamental performance limits of LLM inference, highlighting the potential benefits of future hardware advancements and guiding the optimization of LLM deployment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10968v2">Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14376v1">Schemora: schema matching via multi-stage recommendation and metadata enrichment using off-the-shelf llms</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Schema matching is essential for integrating heterogeneous data sources and enhancing dataset discovery, yet it remains a complex and resource-intensive problem. We introduce SCHEMORA, a schema matching framework that combines large language models with hybrid retrieval techniques in a prompt-based approach, enabling efficient identification of candidate matches without relying on labeled training data or exhaustive pairwise comparisons. By enriching schema metadata and leveraging both vector-based and lexical retrieval, SCHEMORA improves matching accuracy and scalability. Evaluated on the MIMIC-OMOP benchmark, it establishes new state-of-the-art performance, with gains of 7.49% in HitRate@5 and 3.75% in HitRate@3 over previous best results. To our knowledge, this is the first LLM-based schema matching method with an open-source implementation, accompanied by analysis that underscores the critical role of retrieval and provides practical guidance on model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10871v2">Layerwise Recall and the Geometry of Interwoven Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      This study explores how large language models (LLMs) encode interwoven scientific knowledge, using chemical elements and LLaMA-series models as a case study. We identify a 3D spiral structure in the hidden states that aligns with the conceptual structure of the periodic table, suggesting that LLMs can reflect the geometric organization of scientific concepts learned from text. Linear probing reveals that middle layers encode continuous, overlapping attributes that enable indirect recall, while deeper layers sharpen categorical distinctions and incorporate linguistic context. These findings suggest that LLMs represent symbolic knowledge not as isolated facts, but as structured geometric manifolds that intertwine semantic information across layers. We hope this work inspires further exploration of how LLMs represent and reason about scientific knowledge, particularly in domains such as materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13335v1">Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Humour, as a complex language form, is derived from myriad aspects of life, whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes. In this work, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular humour form. We compare models on simple puns and more complex topical humour that requires knowledge of real-world entities and events. In doing so, we curate a dataset of 600 jokes split across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes, where understanding relies on reasoning beyond "common sense", rooted instead in world knowledge regarding news events and pop culture. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (inc. reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most works in computational humour on overly simple joke forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13323v1">GeoReg: Weight-Constrained Few-Shot Regression for Socio-Economic Estimation using LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 15 pages, 13 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Socio-economic indicators like regional GDP, population, and education levels, are crucial to shaping policy decisions and fostering sustainable development. This research introduces GeoReg a regression model that integrates diverse data sources, including satellite imagery and web-based geospatial information, to estimate these indicators even for data-scarce regions such as developing countries. Our approach leverages the prior knowledge of large language model (LLM) to address the scarcity of labeled data, with the LLM functioning as a data engineer by extracting informative features to enable effective estimation in few-shot settings. Specifically, our model obtains contextual relationships between data features and the target indicator, categorizing their correlations as positive, negative, mixed, or irrelevant. These features are then fed into the linear estimator with tailored weight constraints for each category. To capture nonlinear patterns, the model also identifies meaningful feature interactions and integrates them, along with nonlinear transformations. Experiments across three countries at different stages of development demonstrate that our model outperforms baselines in estimating socio-economic indicators, even for low-income countries with limited data availability.
    </details>
</div>
