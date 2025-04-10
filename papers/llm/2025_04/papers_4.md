# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01145v1">MaLAware: Automating the Comprehension of Malicious Software Behaviours using Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at MSR 2025
    </div>
    <details class="paper-abstract">
      Current malware (malicious software) analysis tools focus on detection and family classification but fail to provide clear and actionable narrative insights into the malignant activity of the malware. Therefore, there is a need for a tool that translates raw malware data into human-readable descriptions. Developing such a tool accelerates incident response, reduces malware analysts' cognitive load, and enables individuals having limited technical expertise to understand malicious software behaviour. With this objective, we present MaLAware, which automatically summarizes the full spectrum of malicious activity of malware executables. MaLAware processes Cuckoo Sandbox-generated reports using large language models (LLMs) to correlate malignant activities and generate concise summaries explaining malware behaviour. We evaluate the tool's performance on five open-source LLMs. The evaluation uses the human-written malware behaviour description dataset as ground truth. The model's performance is measured using 11 extensive performance metrics, which boosts the confidence of MaLAware's effectiveness. The current version of the tool, i.e., MaLAware, supports Qwen2.5-7B, Llama2-7B, Llama3.1-8B, Mistral-7B, and Falcon-7B, along with the quantization feature for resource-constrained environments. MaLAware lays a foundation for future research in malware behavior explanation, and its extensive evaluation demonstrates LLMs' ability to narrate malware behavior in an actionable and comprehensive manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05396v2">FairCoder: Evaluating Social Bias of LLMs in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed in coding tasks, drawing increasing attention to the evaluation of the quality and safety of LLMs' outputs. However, research on bias in code generation remains limited. Existing studies typically identify bias by applying malicious prompts or reusing tasks and dataset originally designed for discriminative models. Given that prior datasets are not fully optimized for code-related tasks, there is a pressing need for benchmarks specifically designed for evaluating code models. In this study, we introduce FairCoder, a novel benchmark for evaluating social bias in code generation. FairCoder explores the bias issue following the pipeline in software development, from function implementation to unit test, with diverse real-world scenarios. Additionally, three metrics are designed to assess fairness performance on this benchmark. We conduct experiments on widely used LLMs and provide a comprehensive analysis of the results. The findings reveal that all tested LLMs exhibit social bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01127v1">Can LLMs Grasp Implicit Cultural Values? Benchmarking LLMs' Metacognitive Cultural Intelligence with CQ-Bench</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Cultural Intelligence (CQ) refers to the ability to understand unfamiliar cultural contexts-a crucial skill for large language models (LLMs) to effectively engage with globally diverse users. While existing research often focuses on explicitly stated cultural norms, such approaches fail to capture the subtle, implicit values that underlie real-world conversations. To address this gap, we introduce CQ-Bench, a benchmark specifically designed to assess LLMs' capability to infer implicit cultural values from natural conversational contexts. We generate a multi-character conversation-based stories dataset using values from the World Value Survey and GlobalOpinions datasets, with topics including ethical, religious, social, and political. Our dataset construction pipeline includes rigorous validation procedures-incorporation, consistency, and implicitness checks-using GPT-4o, with 98.2% human-model agreement in the final validation. Our benchmark consists of three tasks of increasing complexity: attitude detection, value selection, and value extraction. We find that while o1 and Deepseek-R1 models reach human-level performance in value selection (0.809 and 0.814), they still fall short in nuanced attitude detection, with F1 scores of 0.622 and 0.635, respectively. In the value extraction task, GPT-4o-mini and o3-mini score 0.602 and 0.598, highlighting the difficulty of open-ended cultural reasoning. Notably, fine-tuning smaller models (e.g., LLaMA-3.2-3B) on only 500 culturally rich examples improves performance by over 10%, even outperforming stronger baselines (o3-mini) in some cases. Using CQ-Bench, we provide insights into the current challenges in LLMs' CQ research and suggest practical pathways for enhancing LLMs' cross-cultural reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01094v1">Multilingual and Multi-Accent Jailbreaking of Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 21 pages, 6 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01005v1">When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at https://github.com/nishadsinghi/sc-genrm-scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00993v1">MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00858v1">Whispering Under the Eaves: Protecting User Privacy Against Commercial and LLM-powered Automatic Speech Recognition Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accept to USENIX Security 2025
    </div>
    <details class="paper-abstract">
      The widespread application of automatic speech recognition (ASR) supports large-scale voice surveillance, raising concerns about privacy among users. In this paper, we concentrate on using adversarial examples to mitigate unauthorized disclosure of speech privacy thwarted by potential eavesdroppers in speech communications. While audio adversarial examples have demonstrated the capability to mislead ASR models or evade ASR surveillance, they are typically constructed through time-intensive offline optimization, restricting their practicality in real-time voice communication. Recent work overcame this limitation by generating universal adversarial perturbations (UAPs) and enhancing their transferability for black-box scenarios. However, they introduced excessive noise that significantly degrades audio quality and affects human perception, thereby limiting their effectiveness in practical scenarios. To address this limitation and protect live users' speech against ASR systems, we propose a novel framework, AudioShield. Central to this framework is the concept of Transferable Universal Adversarial Perturbations in the Latent Space (LS-TUAP). By transferring the perturbations to the latent space, the audio quality is preserved to a large extent. Additionally, we propose target feature adaptation to enhance the transferability of UAPs by embedding target text features into the perturbations. Comprehensive evaluation on four commercial ASR APIs (Google, Amazon, iFlytek, and Alibaba), three voice assistants, two LLM-powered ASR and one NN-based ASR demonstrates the protection superiority of AudioShield over existing competitors, and both objective and subjective evaluations indicate that AudioShield significantly improves the audio quality. Moreover, AudioShield also shows high effectiveness in real-time end-to-end scenarios, and demonstrates strong resilience against adaptive countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00829v1">How Difficulty-Aware Staged Reinforcement Learning Enhances LLMs' Reasoning Capabilities: A Preliminary Experimental Study</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of Large Language Models (LLMs) with efficiency and scalability remains a fundamental challenge in artificial intelligence research. This paper presents a rigorous experimental investigation into how difficulty-aware staged reinforcement learning (RL) strategies can substantially improve LLM reasoning performance. Through systematic analysis, we demonstrate that strategically selecting training data according to well-defined difficulty levels markedly enhances RL optimization. Moreover, we introduce a staged training methodology, progressively exposing models to increasingly challenging tasks, further amplifying reasoning capabilities. Our findings reveal significant cross-domain benefits when simultaneously training models on mathematical reasoning and code generation tasks. Notably, our proposed approach enables a 1.5B parameter model to achieve an accuracy of 42.3\% on the AIME-2024 benchmark, 89.5\% on the MATH-500 benchmark. These results underscore the efficacy of our method in advancing the reasoning proficiency of LLMs. We will open-source our datasets on GitHub and Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00727v1">Personality-Driven Decision-Making in LLM-Based Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 8 figures. To be included in Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)
    </div>
    <details class="paper-abstract">
      The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00711v1">GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      The era of foundation models has revolutionized AI research, yet Graph Foundation Models (GFMs) remain constrained by the scarcity of large-scale graph corpora. Traditional graph data synthesis techniques primarily focus on simplistic structural operations, lacking the capacity to generate semantically rich nodes with meaningful textual attributes: a critical limitation for real-world applications. While large language models (LLMs) demonstrate exceptional text generation capabilities, their direct application to graph synthesis is impeded by context window limitations, hallucination phenomena, and structural consistency challenges. To address these issues, we introduce GraphMaster, the first multi-agent framework specifically designed for graph data synthesis in data-limited environments. GraphMaster orchestrates four specialized LLM agents (Manager, Perception, Enhancement, and Evaluation) that collaboratively optimize the synthesis process through iterative refinement, ensuring both semantic coherence and structural integrity. To rigorously evaluate our approach, we create new data-limited "Sub" variants of six standard graph benchmarks, specifically designed to test synthesis capabilities under realistic constraints. Additionally, we develop a novel interpretability assessment framework that combines human evaluation with a principled Grassmannian manifold-based analysis, providing both qualitative and quantitative measures of semantic coherence. Experimental results demonstrate that GraphMaster significantly outperforms traditional synthesis methods across multiple datasets, establishing a strong foundation for advancing GFMs in data-scarce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00694v1">On Benchmarking Code LLMs for Android Malware Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in various code intelligence tasks. However, their effectiveness for Android malware analysis remains underexplored. Decompiled Android code poses unique challenges for analysis, primarily due to its large volume of functions and the frequent absence of meaningful function names. This paper presents Cama, a benchmarking framework designed to systematically evaluate the effectiveness of Code LLMs in Android malware analysis tasks. Cama specifies structured model outputs (comprising function summaries, refined function names, and maliciousness scores) to support key malware analysis tasks, including malicious function identification and malware purpose summarization. Built on these, it integrates three domain-specific evaluation metrics, consistency, fidelity, and semantic relevance, enabling rigorous stability and effectiveness assessment and cross-model comparison. We construct a benchmark dataset consisting of 118 Android malware samples, encompassing over 7.5 million distinct functions, and use Cama to evaluate four popular open-source models. Our experiments provide insights into how Code LLMs interpret decompiled code and quantify the sensitivity to function renaming, highlighting both the potential and current limitations of Code LLMs in malware analysis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00664v1">Do LLMs Surpass Encoders for Biomedical NER?</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted to appear in IEEE ICHI 2025
    </div>
    <details class="paper-abstract">
      Recognizing spans of biomedical concepts and their types (e.g., drug or gene) in free text, often called biomedical named entity recognition (NER), is a basic component of information extraction (IE) pipelines. Without a strong NER component, other applications, such as knowledge discovery and information retrieval, are not practical. State-of-the-art in NER shifted from traditional ML models to deep neural networks with transformer-based encoder models (e.g., BERT) emerging as the current standard. However, decoder models (also called large language models or LLMs) are gaining traction in IE. But LLM-driven NER often ignores positional information due to the generative nature of decoder models. Furthermore, they are computationally very expensive (both in inference time and hardware needs). Hence, it is worth exploring if they actually excel at biomedical NER and assess any associated trade-offs (performance vs efficiency). This is exactly what we do in this effort employing the same BIO entity tagging scheme (that retains positional information) using five different datasets with varying proportions of longer entities. Our results show that the LLMs chosen (Mistral and Llama: 8B range) often outperform best encoder models (BERT-(un)cased, BiomedBERT, and DeBERTav3: 300M range) by 2-8% in F-scores except for one dataset, where they equal encoder performance. This gain is more prominent among longer entities of length >= 3 tokens. However, LLMs are one to two orders of magnitude more expensive at inference time and may need cost prohibitive hardware. Thus, when performance differences are small or real time user feedback is needed, encoder models might still be more suitable than LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00636v1">Exploring the Impact of an LLM-Powered Teachable Agent on Learning Gains and Cognitive Load in Music Education</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at CHI 2025 Workshop on Augmented Educators and AI: Shaping the Future of Human and AI Cooperation in Learning
    </div>
    <details class="paper-abstract">
      This study examines the impact of an LLM-powered teachable agent, grounded in the Learning by Teaching (LBT) pedagogy, on students' music theory learning and cognitive load. The participants were 28 Chinese university students with prior music instrumental experiences. In an online experiment, they were assigned to either an experimental group, which engaged in music analysis with the teachable agent, or a control group, which conducted self-directed analysis using instructional materials. Findings indicate that students in the experimental group achieved significantly higher post-test scores than those in the control group. Additionally, they reported lower cognitive load, suggesting that the teachable agent effectively reduced the cognitive demands of music analysis tasks. These results highlight the potential of AI-driven scaffolding based on LBT principles to enhance music theory education, supporting teachers in delivering theory-oriented instruction while fostering students' self-directed learning skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00613v1">LLM-Guided Search for Deletion-Correcting Codes</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Finding deletion-correcting codes of maximum size has been an open problem for over 70 years, even for a single deletion. In this paper, we propose a novel approach for constructing deletion-correcting codes. A code is a set of sequences satisfying certain constraints, and we construct it by greedily adding the highest-priority sequence according to a priority function. To find good priority functions, we leverage FunSearch, a large language model (LLM)-guided evolutionary search proposed by Romera et al., 2024. FunSearch iteratively generates, evaluates, and refines priority functions to construct large deletion-correcting codes. For a single deletion, our evolutionary search finds functions that construct codes which match known maximum sizes, reach the size of the largest (conjectured optimal) Varshamov-Tenengolts codes where the maximum is unknown, and independently rediscover them in equivalent form. For two deletions, we find functions that construct codes with new best-known sizes for code lengths \( n = 12, 13 \), and \( 16 \), establishing improved lower bounds. These results demonstrate the potential of LLM-guided search for information theory and code design and represent the first application of such methods for constructing error-correcting codes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00607v1">Contextualized Autonomous Drone Navigation using LLMs Deployed in Edge-Cloud Computing</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Autonomous navigation is usually trained offline in diverse scenarios and fine-tuned online subject to real-world experiences. However, the real world is dynamic and changeable, and many environmental encounters/effects are not accounted for in real-time due to difficulties in describing them within offline training data or hard to describe even in online scenarios. However, we know that the human operator can describe these dynamic environmental encounters through natural language, adding semantic context. The research is to deploy Large Language Models (LLMs) to perform real-time contextual code adjustment to autonomous navigation. The challenge not evaluated in literature is what LLMs are appropriate and where should these computationally heavy algorithms sit in the computation-communication edge-cloud computing architectures. In this paper, we evaluate how different LLMs can adjust both the navigation map parameters dynamically (e.g., contour map shaping) and also derive navigation task instruction sets. We then evaluate which LLMs are most suitable and where they should sit in future edge-cloud of 6G telecommunication architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00587v1">AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has catalyzed the development of multi-agent systems, where multiple LLM-based agents collaborate to solve complex tasks. However, existing systems predominantly rely on centralized coordination, which introduces scalability bottlenecks, limits adaptability, and creates single points of failure. Additionally, concerns over privacy and proprietary knowledge sharing hinder cross-organizational collaboration, leading to siloed expertise. To address these challenges, we propose AgentNet, a decentralized, Retrieval-Augmented Generation (RAG)-based framework that enables LLM-based agents to autonomously evolve their capabilities and collaborate efficiently in a Directed Acyclic Graph (DAG)-structured network. Unlike traditional multi-agent systems that depend on static role assignments or centralized control, AgentNet allows agents to specialize dynamically, adjust their connectivity, and route tasks without relying on predefined workflows. AgentNet's core design is built upon several key innovations: (1) Fully Decentralized Paradigm: Removing the central orchestrator, allowing agents to coordinate and specialize autonomously, fostering fault tolerance and emergent collective intelligence. (2) Dynamically Evolving Graph Topology: Real-time adaptation of agent connections based on task demands, ensuring scalability and resilience.(3) Adaptive Learning for Expertise Refinement: A retrieval-based memory system that enables agents to continuously update and refine their specialized skills. By eliminating centralized control, AgentNet enhances fault tolerance, promotes scalable specialization, and enables privacy-preserving collaboration across organizations. Through decentralized coordination and minimal data exchange, agents can leverage diverse knowledge sources while safeguarding sensitive information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00513v1">Leveraging LLMs for User Stories in AI Systems: UStAI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00428v1">LLM-Assisted Proactive Threat Intelligence for Automated Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 Pages, 1 Figure
    </div>
    <details class="paper-abstract">
      Successful defense against dynamically evolving cyber threats requires advanced and sophisticated techniques. This research presents a novel approach to enhance real-time cybersecurity threat detection and response by integrating large language models (LLMs) and Retrieval-Augmented Generation (RAG) systems with continuous threat intelligence feeds. Leveraging recent advancements in LLMs, specifically GPT-4o, and the innovative application of RAG techniques, our approach addresses the limitations of traditional static threat analysis by incorporating dynamic, real-time data sources. We leveraged RAG to get the latest information in real-time for threat intelligence, which is not possible in the existing GPT-4o model. We employ the Patrowl framework to automate the retrieval of diverse cybersecurity threat intelligence feeds, including Common Vulnerabilities and Exposures (CVE), Common Weakness Enumeration (CWE), Exploit Prediction Scoring System (EPSS), and Known Exploited Vulnerabilities (KEV) databases, and integrate these with the all-mpnet-base-v2 model for high-dimensional vector embeddings, stored and queried in Milvus. We demonstrate our system's efficacy through a series of case studies, revealing significant improvements in addressing recently disclosed vulnerabilities, KEVs, and high-EPSS-score CVEs compared to the baseline GPT-4o. This work not only advances the role of LLMs in cybersecurity but also establishes a robust foundation for the development of automated intelligent cyberthreat information management systems, addressing crucial gaps in current cybersecurity practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00409v1">Semantic Mastery: Enhancing LLMs with Advanced Natural Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have greatly improved their capability in performing NLP tasks. However, deeper semantic understanding, contextual coherence, and more subtle reasoning are still difficult to obtain. The paper discusses state-of-the-art methodologies that advance LLMs with more advanced NLU techniques, such as semantic parsing, knowledge integration, and contextual reinforcement learning. We analyze the use of structured knowledge graphs, retrieval-augmented generation (RAG), and fine-tuning strategies that match models with human-level understanding. Furthermore, we address the incorporation of transformer-based architectures, contrastive learning, and hybrid symbolic-neural methods that address problems like hallucinations, ambiguity, and inconsistency in the factual perspectives involved in performing complex NLP tasks, such as question-answering text summarization and dialogue generation. Our findings show the importance of semantic precision for enhancing AI-driven language systems and suggest future research directions to bridge the gap between statistical language models and true natural language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00374v1">When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR)</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00341v1">Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 This article has been accepted for publication in the IEEE 2025 International Conference on Communications (ICC2025)
    </div>
    <details class="paper-abstract">
      The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00310v1">Detecting and Mitigating Bias in LLMs through Knowledge Graph-Augmented Training</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large language models have revolutionized natural language processing with their surprising capability to understand and generate human-like text. However, many of these models inherit and further amplify the biases present in their training data, raising ethical and fairness concerns. The detection and mitigation of such biases are vital to ensuring that LLMs act responsibly and equitably across diverse domains. This work investigates Knowledge Graph-Augmented Training (KGAT) as a novel method to mitigate bias in LLM. Using structured domain-specific knowledge from real-world knowledge graphs, we improve the understanding of the model and reduce biased output. Public datasets for bias assessment include Gender Shades, Bias in Bios, and FairFace, while metrics such as demographic parity and equal opportunity facilitate rigorous detection. We also performed targeted mitigation strategies to correct biased associations, leading to a significant drop in biased output and improved bias metrics. Equipped with real-world datasets and knowledge graphs, our framework is both scalable and effective, paving the way toward responsible deployment in sensitive and high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01995v1">Brains vs. Bytes: Evaluating LLM Proficiency in Olympiad Mathematics</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown impressive progress in mathematical reasoning tasks. However, current evaluation benchmarks predominantly focus on the accuracy of final answers, often overlooking the logical rigor crucial for mathematical problem-solving. The claim that state-of-the-art LLMs can solve Math Olympiad-level problems requires closer examination. To explore this, we conducted both qualitative and quantitative human evaluations of proofs generated by LLMs, and developed a schema for automatically assessing their reasoning capabilities. Our study reveals that current LLMs fall significantly short of solving challenging Olympiad-level problems and frequently fail to distinguish correct mathematical reasoning from clearly flawed solutions. We also found that occasional correct final answers provided by LLMs often result from pattern recognition or heuristic shortcuts rather than genuine mathematical reasoning. These findings underscore the substantial gap between LLM performance and human expertise in advanced mathematical reasoning and highlight the importance of developing benchmarks that prioritize the rigor and coherence of mathematical arguments rather than merely the correctness of final answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02873v1">Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      The malicious usage of large language models (LLMs) has motivated the detection of LLM-generated texts. Previous work in topological data analysis shows that the persistent homology dimension (PHD) of text embeddings can serve as a more robust and promising score than other zero-shot methods. However, effectively detecting short LLM-generated texts remains a challenge. This paper presents Short-PHD, a zero-shot LLM-generated text detection method tailored for short texts. Short-PHD stabilizes the estimation of the previous PHD method for short texts by inserting off-topic content before the given input text and identifies LLM-generated text based on an established detection threshold. Experimental results on both public and generated datasets demonstrate that Short-PHD outperforms existing zero-shot methods in short LLM-generated text detection. Implementation codes are available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02870v1">AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025 Workshop
    </div>
    <details class="paper-abstract">
      Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02867v1">Multi-Agent LLM Judge: automatic personalized LLM judge design for evaluating natural language generation applications</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Presented at SophiaSummit2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance across diverse domains, yet they still encounter challenges such as insufficient domain-specific knowledge, biases, and hallucinations. This underscores the need for robust evaluation methodologies to accurately assess LLM-based applications. Traditional evaluation methods, which rely on word overlap or text embeddings, are inadequate for capturing the nuanced semantic information necessary to evaluate dynamic, open-ended text generation. Recent research has explored leveraging LLMs to mimic human reasoning and decision-making processes for evaluation purposes known as LLM-as-a-judge framework. However, these existing frameworks have two significant limitations. First, they lack the flexibility to adapt to different text styles, including various answer and ground truth styles, thereby reducing their generalization performance. Second, the evaluation scores produced by these frameworks are often skewed and hard to interpret, showing a low correlation with human judgment. To address these challenges, we propose a novel dynamic multi-agent system that automatically designs personalized LLM judges for various natural language generation applications. This system iteratively refines evaluation prompts and balances the trade-off between the adaptive requirements of downstream tasks and the alignment with human perception. Our experimental results show that the proposed multi-agent LLM Judge framework not only enhances evaluation accuracy compared to existing methods but also produces evaluation scores that better align with human perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09647v3">Leveraging LLMS for Top-Down Sector Allocation In Automated Trading</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      This paper introduces a methodology leveraging Large Language Models (LLMs) for sector-level portfolio allocation through systematic analysis of macroeconomic conditions and market sentiment. Our framework emphasizes top-down sector allocation by processing multiple data streams simultaneously, including policy documents, economic indicators, and sentiment patterns. Empirical results demonstrate superior risk-adjusted returns compared to traditional cross momentum strategies, achieving a Sharpe ratio of 2.51 and portfolio return of 8.79% versus -0.61 and -1.39% respectively. These results suggest that LLM-based systematic macro analysis presents a viable approach for enhancing automated portfolio allocation decisions at the sector level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13727v2">LLM-Human Pipeline for Cultural Context Grounding of Conversations</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Oral at NAACL 2025 Main conference. Albuquerque, USA. Apr 29 - May 4, 2025. 19 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Conversations often adhere to well-understood social norms that vary across cultures. For example, while "addressing parents by name" is commonplace in the West, it is rare in most Asian cultures. Adherence or violation of such norms often dictates the tenor of conversations. Humans are able to navigate social situations requiring cultural awareness quite adeptly. However, it is a hard task for NLP models. In this paper, we tackle this problem by introducing a "Cultural Context Schema" for conversations. It comprises (1) conversational information such as emotions, dialogue acts, etc., and (2) cultural information such as social norms, violations, etc. We generate ~110k social norm and violation descriptions for ~23k conversations from Chinese culture using LLMs. We refine them using automated verification strategies which are evaluated against culturally aware human judgements. We organize these descriptions into meaningful structures we call "Norm Concepts", using an interactive human-in-loop framework. We ground the norm concepts and the descriptions in conversations using symbolic annotation. Finally, we use the obtained dataset for downstream tasks such as emotion, sentiment, and dialogue act detection. We show that it significantly improves the empirical performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14642v2">TOMG-Bench: Evaluating LLMs on Text-based Open Molecule Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 The first benchmark for text-based open molecule generation
    </div>
    <details class="paper-abstract">
      In this paper, we propose Text-based Open Molecule Generation Benchmark (TOMG-Bench), the first benchmark to evaluate the open-domain molecule generation capability of LLMs. TOMG-Bench encompasses a dataset of three major tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom). Each major task further contains three subtasks, while each subtask comprises 5,000 test samples. Given the inherent complexity of open molecule generation evaluation, we also developed an automated evaluation system that helps measure both the quality and the accuracy of the generated molecules. Our comprehensive benchmarking of 25 LLMs reveals the current limitations as well as potential areas for improvement in text-guided molecule discovery. Furthermore, we propose OpenMolIns, a specialized instruction tuning dataset established for solving challenges raised by TOMG-Bench. Fine-tuned on OpenMolIns, Llama3.1-8B could outperform all the open-source general LLMs, even surpassing GPT-3.5-turbo by 46.5\% on TOMG-Bench. Our codes and datasets are available through https://github.com/phenixace/TOMG-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15035v2">LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Building safe Large Language Models (LLMs) across multiple languages is essential in ensuring both safe access and linguistic diversity. To this end, we introduce M-ALERT, a multilingual benchmark that evaluates the safety of LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT includes 15k high-quality prompts per language, totaling 75k, following the detailed ALERT taxonomy. Our extensive experiments on 10 state-of-the-art LLMs highlight the importance of language-specific safety analysis, revealing that models often exhibit significant inconsistencies in safety across languages and categories. For instance, Llama3.2 shows high unsafety in the category crime_tax for Italian but remains safe in other languages. Similar differences can be observed across all models. In contrast, certain categories, such as substance_cannabis and crime_propaganda, consistently trigger unsafe responses across models and languages. These findings underscore the need for robust multilingual safety practices in LLMs to ensure safe and responsible usage across diverse user communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13543v2">BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Vision Language Models (VLMs) possess extensive knowledge and exhibit promising reasoning abilities, however, they still struggle to perform well in complex, dynamic environments. Real-world tasks require handling intricate interactions, advanced spatial reasoning, long-term planning, and continuous exploration of new strategies-areas in which we lack effective methodologies for comprehensively evaluating these capabilities. To address this gap, we introduce BALROG, a novel benchmark designed to assess the agentic capabilities of LLMs and VLMs through a diverse set of challenging games. Our benchmark incorporates a range of existing reinforcement learning environments with varying levels of difficulty, including tasks that are solvable by non-expert humans in seconds to extremely challenging ones that may take years to master (e.g., the NetHack Learning Environment). We devise fine-grained metrics to measure performance and conduct an extensive evaluation of several popular open-source and closed-source LLMs and VLMs. Our findings indicate that while current models achieve partial success in the easier games, they struggle significantly with more challenging tasks. Notably, we observe severe deficiencies in vision-based decision-making, as several models perform worse when visual representations of the environments are provided. We release BALROG as an open and user-friendly benchmark to facilitate future research and development in the agentic community. Code and Leaderboard at balrogai.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05447v2">TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance Beyond RAG</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is one of the leading and most widely used techniques for enhancing LLM retrieval capabilities, but it still faces significant limitations in commercial use cases. RAG primarily relies on the query-chunk text-to-text similarity in the embedding space for retrieval and can fail to capture deeper semantic relationships across chunks, is highly sensitive to chunking strategies, and is prone to hallucinations. To address these challenges, we propose TOBUGraph, a graph-based retrieval framework that first constructs the knowledge graph from unstructured data dynamically and automatically. Using LLMs, TOBUGraph extracts structured knowledge and diverse relationships among data, going beyond RAG's text-to-text similarity. Retrieval is achieved through graph traversal, leveraging the extracted relationships and structures to enhance retrieval accuracy, eliminating the need for chunking configurations while reducing hallucination. We demonstrate TOBUGraph's effectiveness in TOBU, a real-world application in production for personal memory organization and retrieval. Our evaluation using real user data demonstrates that TOBUGraph outperforms multiple RAG implementations in both precision and recall, significantly improving user experience through improved retrieval accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06621v4">Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 ICLR 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong reasoning and memorization capabilities via pretraining on massive textual corpora. However, this poses risk of privacy and copyright violations, highlighting the need for efficient machine unlearning methods that remove sensitive data without retraining from scratch. While Gradient Ascent (GA) is commonly used to unlearn by reducing the likelihood of generating unwanted content, it leads to unstable optimization and catastrophic forgetting of retrained knowledge. We find that combining GA with low-rank adaptation results in poor trade-offs between computational cost and generative performance. To address these challenges, we propose Low-rank Knowledge Unlearning (LoKU), a novel framework that enables robust and efficient unlearning for LLMs. First, we introduce Inverted Hinge Loss, which suppresses unwanted tokens while maintaining fluency by boosting the probability of the next most likely token. Second, we develop a data-adaptive initialization for LoRA adapters via low-rank approximation weighted with relative Fisher information, thereby focusing updates on parameters critical for removing targeted knowledge. Experiments on the Training Data Extraction Challenge dataset using GPT-Neo models as well as on the TOFU benchmark with Phi-1.5B and Llama2-7B models demonstrate that our approach effectively removes sensitive information while maintaining reasoning and generative capabilities with minimal impact. Our implementation can be found in https://github.com/csm9493/efficient-llm-unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v5">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency. Code will be available at https://github.com/iamhankai/Forest-of-Thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22968v2">HRET: A Self-Evolving LLM Evaluation Toolkit for Korean</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent advancements in Korean large language models (LLMs) have spurred numerous benchmarks and evaluation methodologies, yet the lack of a standardized evaluation framework has led to inconsistent results and limited comparability. To address this, we introduce HRET Haerae Evaluation Toolkit, an open-source, self-evolving evaluation framework tailored specifically for Korean LLMs. HRET unifies diverse evaluation methods, including logit-based scoring, exact-match, language-inconsistency penalization, and LLM-as-a-Judge assessments. Its modular, registry-based architecture integrates major benchmarks (HAE-RAE Bench, KMMLU, KUDGE, HRM8K) and multiple inference backends (vLLM, HuggingFace, OpenAI-compatible endpoints). With automated pipelines for continuous evolution, HRET provides a robust foundation for reproducible, fair, and transparent Korean NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v3">A Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      We propose \emph{LumosCore} to build high-bandwidth and large-scale data center networks for LLM jobs. By replacing the core-layer electrical packet switches by optical circuit switches, \emph{LumosCore} could achieves $2\times$ increase in bandwidth or $8\times$ increase in network size. We offer the detailed design of \emph{LumosCore} at both deployment stage and running stage. At deployment stage, we propose Interleaved Wiring, which is compatible with all possible logical topologies. At running stage, we design polynomial-time algorithms for GPU placement, logical topology generating and OCS reconfiguration to minimize network contention and reduce impact to scheduled jobs. We evaluate \emph{LumosCore} using both testbed experiments and large-scale simulation. Compared to traditional hybrid optical/electrical architectures, \emph{LumosCore} increases the end-to-end training throughput by up to 39.5\% on a 128-node testbed. Compared to the state-of-art Clos architectures, \emph{LumosCore} reduces the average job completion time by up to 34.1\% in a 16k simulation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19178v2">UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 3 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve remarkable success in natural language processing (NLP). In practical scenarios like recommendations, as users increasingly seek personalized experiences, it becomes crucial to incorporate user interaction history into the context of LLMs to enhance personalization. However, from a practical utility perspective, user interactions' extensive length and noise present challenges when used directly as text prompts. A promising solution is to compress and distill interactions into compact embeddings, serving as soft prompts to assist LLMs in generating personalized responses. Although this approach brings efficiency, a critical concern emerges: Can user embeddings adequately capture valuable information and prompt LLMs? To address this concern, we propose \name, a benchmark designed to evaluate the effectiveness of user embeddings in prompting LLMs for personalization. We establish a fair and standardized evaluation process, encompassing pre-training, fine-tuning, and evaluation stages. To thoroughly evaluate user embeddings, we design three dimensions of tasks: sequence understanding, action prediction, and interest perception. These evaluation tasks cover the industry's demands in traditional recommendation tasks, such as improving prediction accuracy, and its aspirations for LLM-based methods, such as accurately understanding user interests and enhancing the user experience. We conduct extensive experiments on various state-of-the-art methods for modeling user embeddings. Additionally, we reveal the scaling laws of leveraging user embeddings to prompt LLMs. The benchmark is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20197v2">Enhancing the Robustness of LLM-Generated Code: Empirical Study and Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of code generated by large language models (LLMs) is crucial for real-world reliability. However, existing evaluations predominantly focus on correctness, often neglecting key robustness concerns such as missing input validation and insufficient error handling. In this paper, we present the first empirical study on the robustness of LLM-generated code. We introduce novel robustness metrics and analyze four state-of-the-art code LLMs, revealing that, on average, 43.1% of their generated code is less robust than human-written counterparts. Notably, over 90% of robustness deficiencies stem from missing conditional checks, with 70% of these omissions occurring in the first line of code. Additionally, in 69% of cases where a conditional statement is necessary but absent, the "if" token still ranks third or higher in the model's predicted token probabilities, indicating an implicit recognition of control structures. Building on these findings, we propose RobGen, a framework designed to enhance code robustness without requiring model retraining. RobGen leverages two model-agnostic techniques: RobGen-Adj, which dynamically adjusts token probabilities during decoding to encourage the inclusion of control structures, and RobGen-Ins, which improves generated code by inserting missing conditionals after generation. Experimental results demonstrate that RobGen reduces the proportion of less robust model-generated code by 20.0%, significantly enhancing code reliability across diverse tasks. As a lightweight and adaptable solution, RobGen effectively mitigates robustness challenges in LLM-generated code. All code and data are available at https://github.com/SYSUSELab/RobGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06994v2">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern applications codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.68\% (upto 65\%) reduction in application binary size, and an average of 2.8\% performance improvement over the traditional profile-guided optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06559v2">Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 22 pages, 11 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Language agents based on large language models (LLMs) have demonstrated great promise in automating web-based tasks. Recent work has shown that incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning for web agents. However, unlike simulated sandbox environments, real-world environments such as the web are rife with irreversible actions. This undermines the feasibility of backtracking, a cornerstone of (tree) search. Overly relying on test-time search also hurts efficiency. We advocate model-based planning for web agents that employs a world model to simulate and deliberate over the outcome of each candidate action before committing to one. We systematically explore this paradigm by (1) Proposing a model-based planning framework, WebDreamer, which employs LLMs to serve as both world models and value functions; (2) Training specialized LLMs as world models with a scalable data synthesis pipeline. Empirical results demonstrate that WebDreamer achieves substantial performance improvements over reactive baselines. It is competitive, while being 4-5 times more efficient, with tree search in sandbox environments (VisualWebArena) and also works effectively on real-world websites (Online-Mind2Web and Mind2Web-Live). Furthermore, our trained world model, Dreamer-7B, performs comparable to GPT-4o, highlighting the potential of specialized world models for efficient and effective planning in complex web environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15426v2">CodingTeachLLM: Empowering LLM's Coding Ability via AST Prior Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce CodingTeachLLM, a large language model (LLM) designed for coding teaching. Specially, we aim to enhance the coding ability of LLM and lead it to better teaching mode in education context. Thus, we propose an end-to-end prior-based three-phases supervised fine-tuned model, which is proved more competitive than traditional fine-tuning method. More specifically, our model realizes the structural disassembly and incremental guided output of educational knowledge. To this end, we robustify data classification of three types via a sampler and overlap estimation neural network, and inject the preprocessing datasets into pre-trained model in three batches for LORA fine-tuning. Then, we design a prior module couples system prompt, vector databases, and abstract syntax tree task segmentation. Finally, the compression method and regularization constraint are applied to the prior-based fine-tuned model, followed by text filter at the output end to obtain incremental guided results. Our model represents the first research effort to truly embody the tutor role with the features of abundant educational knowledge, step-by-step incremental guided outputs and non-disclosure of answers. Extensive experiments report that our model also achieves state-of-the-art in code abilities compared to open-source models, reaching an impressive 75.10% on the HumanEval (@pass 1) benchmark. Additionally, our model maintains strong conversational capabilities, with the 13B quantized version achieving scores of 56.34, 50.60, and 45.27 respectively on the MMLU, C-Eval, and AGIEval (5 shot) dialogue evaluation benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04667v4">Non-Determinism of "Deterministic" LLM Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      LLM (large language model) practitioners commonly notice that outputs can vary for the same inputs under settings expected to be deterministic. Yet the questions of how pervasive this is, and with what impact on results, have not to our knowledge been systematically investigated. We investigate non-determinism in five LLMs configured to be deterministic when applied to eight common tasks in across 10 runs, in both zero-shot and few-shot settings. We see accuracy variations up to 15% across naturally occurring runs with a gap of best possible performance to worst possible performance up to 70%. In fact, none of the LLMs consistently delivers repeatable accuracy across all tasks, much less identical output strings. Sharing preliminary results with insiders has revealed that non-determinism perhaps essential to the efficient use of compute resources via co-mingled data in input buffers so this issue is not going away anytime soon. To better quantify our observations, we introduce metrics focused on quantifying determinism, TARr@N for the total agreement rate at N runs over raw output, and TARa@N for total agreement rate of parsed-out answers. Our code and data are publicly available at http://github.com/REDACTED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01141v2">How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19326v2">Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs to Ignore the Correct Reasoning Steps</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent reasoning large language models (LLMs) have demonstrated remarkable improvements in mathematical reasoning capabilities through long Chain-of-Thought. The reasoning tokens of these models enable self-correction within reasoning chains, enhancing robustness. This motivates our exploration: how vulnerable are reasoning LLMs to subtle errors in their input reasoning chains? We introduce "Compromising Thought" (CPT), a vulnerability where models presented with reasoning tokens containing manipulated calculation results tend to ignore correct reasoning steps and adopt incorrect results instead. Through systematic evaluation across multiple reasoning LLMs, we design three increasingly explicit prompting methods to measure CPT resistance, revealing that models struggle significantly to identify and correct these manipulations. Notably, contrary to existing research suggesting structural alterations affect model performance more than content modifications, we find that local ending token manipulations have greater impact on reasoning outcomes than structural changes. Moreover, we discover a security vulnerability in DeepSeek-R1 where tampered reasoning tokens can trigger complete reasoning cessation. Our work enhances understanding of reasoning robustness and highlights security considerations for reasoning-intensive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12836v2">EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Given the steep learning curve of professional 3D software and the time-consuming process of managing large 3D assets, language-guided 3D scene editing has significant potential in fields such as virtual reality, augmented reality, and gaming. However, recent approaches to language-guided 3D scene editing either require manual interventions or focus only on appearance modifications without supporting comprehensive scene layout changes. In response, we propose EditRoom, a unified framework capable of executing a variety of layout edits through natural language commands, without requiring manual intervention. Specifically, EditRoom leverages Large Language Models (LLMs) for command planning and generates target scenes using a diffusion-based method, enabling six types of edits: rotate, translate, scale, replace, add, and remove. To address the lack of data for language-guided 3D scene editing, we have developed an automatic pipeline to augment existing 3D scene synthesis datasets and introduced EditRoom-DB, a large-scale dataset with 83k editing pairs, for training and evaluation. Our experiments demonstrate that our approach consistently outperforms other baselines across all metrics, indicating higher accuracy and coherence in language-guided scene layout editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09577v2">Flash normalization: fast normalization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      RMSNorm is used by many LLMs such as Llama, Mistral, and OpenELM. This paper details FlashNorm, which is an exact but faster implementation of RMSNorm followed by linear layers. FlashNorm also speeds up Layer Normalization and its recently proposed replacement Dynamic Tanh (DyT) arXiv:2503.10622. See https://github.com/OpenMachine-ai/transformer-tricks for code and more transformer tricks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01241v1">Catastrophic Forgetting in LLMs: A Comparative Analysis Across Language Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced Natural Language Processing (NLP), particularly in Natural Language Understanding (NLU) tasks. As we progress toward an agentic world where LLM-based agents autonomously handle specialized tasks, it becomes crucial for these models to adapt to new tasks without forgetting previously learned information - a challenge known as catastrophic forgetting. This study evaluates the continual fine-tuning of various open-source LLMs with different parameter sizes (specifically models under 10 billion parameters) on key NLU tasks from the GLUE benchmark, including SST-2, MRPC, CoLA, and MNLI. By employing prompt engineering and task-specific adjustments, we assess and compare the models' abilities to retain prior knowledge while learning new tasks. Our results indicate that models such as Phi-3.5-mini exhibit minimal forgetting while maintaining strong learning capabilities, making them well-suited for continual learning environments. Additionally, models like Orca-2-7b and Qwen2.5-7B demonstrate impressive learning abilities and overall performance after fine-tuning. This work contributes to understanding catastrophic forgetting in LLMs and highlights prompting engineering to optimize model performance for continual learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15487v2">Multi-LLM Text Summarization</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      In this work, we propose a Multi-LLM summarization framework, and investigate two different multi-LLM strategies including centralized and decentralized. Our multi-LLM summarization framework has two fundamentally important steps at each round of conversation: generation and evaluation. These steps are different depending on whether our multi-LLM decentralized summarization is used or centralized. In both our multi-LLM decentralized and centralized strategies, we have k different LLMs that generate diverse summaries of the text. However, during evaluation, our multi-LLM centralized summarization approach leverages a single LLM to evaluate the summaries and select the best one whereas k LLMs are used for decentralized multi-LLM summarization. Overall, we find that our multi-LLM summarization approaches significantly outperform the baselines that leverage only a single LLM by up to 3x. These results indicate the effectiveness of multi-LLM approaches for summarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01234v1">First Field-Trial Demonstration of L4 Autonomous Optical Network for Distributed AI Training Communication: An LLM-Powered Multi-AI-Agent Solution</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Submitted to the PDP session of the Optical Fiber Communications Conference (OFC) 2025
    </div>
    <details class="paper-abstract">
      We demonstrate the first cross-domain cross-layer level-4 autonomous optical network via a multi-AI-agent system. Field trials show 98 percent task completion rate across the distributed AI training lifecycle-3.2x higher than single agents using state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01205v1">Epistemic Alignment: A Mediating Framework for User-LLM Knowledge Delivery</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      LLMs increasingly serve as tools for knowledge acquisition, yet users cannot effectively specify how they want information presented. When users request that LLMs "cite reputable sources," "express appropriate uncertainty," or "include multiple perspectives," they discover that current interfaces provide no structured way to articulate these preferences. The result is prompt sharing folklore: community-specific copied prompts passed through trust relationships rather than based on measured efficacy. We propose the Epistemic Alignment Framework, a set of ten challenges in knowledge transmission derived from the philosophical literature of epistemology, concerning issues such as evidence quality assessment and calibration of testimonial reliance. The framework serves as a structured intermediary between user needs and system capabilities, creating a common vocabulary to bridge the gap between what users want and what systems deliver. Through a thematic analysis of custom prompts and personalization strategies shared on online communities where these issues are actively discussed, we find users develop elaborate workarounds to address each of the challenges. We then apply our framework to two prominent model providers, OpenAI and Anthropic, through content analysis of their documented policies and product features. Our analysis shows that while these providers have partially addressed the challenges we identified, they fail to establish adequate mechanisms for specifying epistemic preferences, lack transparency about how preferences are implemented, and offer no verification tools to confirm whether preferences were followed. For AI developers, the Epistemic Alignment Framework offers concrete guidance for supporting diverse approaches to knowledge; for users, it works toward information delivery that aligns with their specific needs rather than defaulting to one-size-fits-all approaches.
    </details>
</div>
