# llm - 2025_10

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20934v2">Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Published at the International Conference on Software Maintenance and Evolution (ICSME'25)
    </div>
    <details class="paper-abstract">
      MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools that recommend which methods to move and where, these recommendations do not align with how expert developers perform MOVEMETHOD. Given the extensive training of Large Language Models and their reliance upon naturalness of code, they should expertly recommend which methods are misplaced in a given class and which classes are better hosts. Our formative study of 2016 LLM recommendations revealed that LLMs give expert suggestions, yet they are unreliable: up to 80% of the suggestions are hallucinations. We introduce the first LLM fully powered assistant for MOVEMETHOD refactoring that automates its whole end-to-end lifecycle, from recommendation to execution. We designed novel solutions that automatically filter LLM hallucinations using static analysis from IDEs and a novel workflow that requires LLMs to be self-consistent, critique, and rank refactoring suggestions. As MOVEMETHOD refactoring requires global, projectlevel reasoning, we solved the limited context size of LLMs by employing refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist, synergistically combines the strengths of the LLM, IDE, static analysis, and semantic relevance. In our thorough, multi-methodology empirical evaluation, we compare MM-assist with the previous state-of-the-art approaches. MM-assist significantly outperforms them: (i) on a benchmark widely used by other researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a corpus of 210 recent refactorings from Open-source software, our Recall rates improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced participants who used MM-assist to refactor their own code for one week. They rated 82.8% of MM-assist recommendations positively. This shows that MM-assist is both effective and useful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07887v2">Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14756v1">Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate hardware design tasks, including the generation of Verilog code. While early benchmarks focus primarily on functional correctness, efficient hardware design demands additional optimization for synthesis metrics such as area, delay, and power. Existing benchmarks fall short in evaluating these aspects comprehensively: they often lack optimized baselines or testbenches for verification. To address these gaps, we present Pluto, a benchmark and evaluation framework designed to assess the efficiency of LLM-generated Verilog designs. Pluto presents a comprehensive evaluation set of 114 problems with self-checking testbenches and multiple Pareto-optimal reference implementations. Experimental results show that state-of-the-art LLMs can achieve high functional correctness, reaching 78.3\% at pass@1, but their synthesis efficiency still lags behind expert-crafted implementations, with area efficiency of 63.8\%, delay efficiency of 65.9\%, and power efficiency of 64.0\% at eff@1. This highlights the need for efficiency-aware evaluation frameworks such as Pluto to drive progress in hardware-focused LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14751v1">Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Next-token prediction (NTP) has driven the success of large language models (LLMs), but it struggles with long-horizon reasoning, planning, and creative writing, with these limitations largely attributed to teacher-forced training. Multi-token prediction (MTP) partially mitigates these issues by predicting several future tokens at once, but it mostly captures short-range dependencies and offers limited improvement. We propose future summary prediction (FSP), which trains an auxiliary head to predict a compact representation of the long-term future, preserving information relevant for long-form generations. We explore two variants of FSP: handcrafted summaries, for example, a bag of words summary of the future of the sequence, and learned summaries, which use embeddings produced by a reverse language model trained from right to left. Large-scale pretraining experiments (3B and 8B-parameter models) demonstrate that FSP provides improvements over both NTP and MTP across math, reasoning, and coding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v3">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16191v2">The Last Dependency Crusade: Solving Python Dependency Conflicts with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Pre-print - Accepted at the first annual workshop on Agentic Software Engineering (AgenticSE) co-located with ASE'25
    </div>
    <details class="paper-abstract">
      Resolving Python dependency issues remains a tedious and error-prone process, forcing developers to manually trial compatible module versions and interpreter configurations. Existing automated solutions, such as knowledge-graph-based and database-driven methods, face limitations due to the variety of dependency error types, large sets of possible module versions, and conflicts among transitive dependencies. This paper investigates the use of Large Language Models (LLMs) to automatically repair dependency issues in Python programs. We propose PLLM (pronounced "plum"), a novel retrieval-augmented generation (RAG) approach that iteratively infers missing or incorrect dependencies. PLLM builds a test environment where the LLM proposes module combinations, observes execution feedback, and refines its predictions using natural language processing (NLP) to parse error messages. We evaluate PLLM on the Gistable HG2.9K dataset, a curated collection of real-world Python programs. Using this benchmark, we explore multiple PLLM configurations, including six open-source LLMs evaluated both with and without RAG. Our findings show that RAG consistently improves fix rates, with the best performance achieved by Gemma-2 9B when combined with RAG. Compared to two state-of-the-art baselines, PyEGo and ReadPyE, PLLM achieves significantly higher fix rates; +15.97\% more than ReadPyE and +21.58\% more than PyEGo. Further analysis shows that PLLM is especially effective for projects with numerous dependencies and those using specialized numerical or machine-learning libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14700v1">LLM Agents for Automated Web Vulnerability Reproduction: Are We There Yet?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in software engineering and cybersecurity tasks, including code generation, vulnerability discovery, and automated testing. One critical but underexplored application is automated web vulnerability reproduction, which transforms vulnerability reports into working exploits. Although recent advances suggest promising potential, challenges remain in applying LLM agents to real-world web vulnerability reproduction scenarios. In this paper, we present the first comprehensive evaluation of state-of-the-art LLM agents for automated web vulnerability reproduction. We systematically assess 20 agents from software engineering, cybersecurity, and general domains across 16 dimensions, including technical capabilities, environment adaptability, and user experience factors, on 3 representative web vulnerabilities. Based on the results, we select three top-performing agents (OpenHands, SWE-agent, and CAI) for in-depth evaluation on our benchmark dataset of 80 real-world CVEs spanning 7 vulnerability types and 6 web technologies. Our results reveal that while LLM agents achieve reasonable success on simple library-based vulnerabilities, they consistently fail on complex service-based vulnerabilities requiring multi-component environments. Complex environment configurations and authentication barriers create a gap where agents can execute exploit code but fail to trigger actual vulnerabilities. We observe high sensitivity to input guidance, with performance degrading by over 33% under incomplete authentication information. Our findings highlight the significant gap between current LLM agent capabilities and the demands of reliable automated vulnerability reproduction, emphasizing the need for advances in environmental adaptation and autonomous problem-solving capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22358v2">Adaptive Budget Allocation for Orthogonal-Subspace Adapter Tuning in LLMs Continual Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from catastrophic forgetting in continual learning (CL) scenarios, where performance on previously learned tasks degrades severely while training on sequentially arriving tasks. Although pioneering CL approaches using orthogonal subspaces can mitigate task interference, they typically employ fixed budget allocation, neglecting the varying complexity across tasks and layers. Besides, recent budget-adaptive tuning methods for LLMs often adopt multi-stage paradigms that decouple optimization and budget allocation. Such decoupling results in potential misalignment, which hinders those approaches' practical application in CL scenarios. To address these limitations, we propose OA-Adapter, a novel parameter-efficient approach for continual learning in LLMs that unifies dynamic budget adaptation with orthogonal subspace learning in an end-to-end training stage. Specifically, OA-Adapter introduces a dynamic bottleneck dimension adaptation mechanism that simultaneously allocates an efficient parameter budget and optimizes task objectives without misalignment.To effectively preserve previously acquired knowledge while coordinating with the dynamic budget allocation, orthogonal constraints are applied specifically between the parameter subspace of the current task and the dynamically allocated parameter subspaces of historical tasks. Experimental results on continual learning benchmarks demonstrate that OA-Adapter outperforms state-of-the-art methods in both accuracy and parameter efficiency. OA-Adapter achieves higher average accuracy while using 58.5% fewer parameters on the standard CL benchmark, and maintains its advantages on two larger benchmarks comprising 15 tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13750v2">Confidence-Based Response Abstinence: Improving LLM Trustworthiness via Activation-Based Uncertainty Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 UncertaiNLP at EMNLP 2025
    </div>
    <details class="paper-abstract">
      We propose a method for confidence estimation in retrieval-augmented generation (RAG) systems that aligns closely with the correctness of large language model (LLM) outputs. Confidence estimation is especially critical in high-stakes domains such as finance and healthcare, where the cost of an incorrect answer outweighs that of not answering the question. Our approach extends prior uncertainty quantification methods by leveraging raw feed-forward network (FFN) activations as auto-regressive signals, avoiding the information loss inherent in token logits and probabilities after projection and softmax normalization. We model confidence prediction as a sequence classification task, and regularize training with a Huber loss term to improve robustness against noisy supervision. Applied in a real-world financial industry customer-support setting with complex knowledge bases, our method outperforms strong baselines and maintains high accuracy under strict latency constraints. Experiments on Llama 3.1 8B model show that using activations from only the 16th layer preserves accuracy while reducing response latency. Our results demonstrate that activation-based confidence modeling offers a scalable, architecture-aware path toward trustworthy RAG deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08670v2">Augmenting Smart Contract Decompiler Output through Fine-grained Dependency Analysis and LLM-facilitated Semantic Recovery</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 This is the author version of the article accepted for publication in IEEE Transactions on Software Engineering
    </div>
    <details class="paper-abstract">
      Decompiler is a specialized type of reverse engineering tool extensively employed in program analysis tasks, particularly in program comprehension and vulnerability detection. However, current Solidity smart contract decompilers face significant limitations in reconstructing the original source code. In particular, the bottleneck of SOTA decompilers lies in inaccurate method identification, incorrect variable type recovery, and missing contract attributes. These deficiencies hinder downstream tasks and understanding of the program logic. To address these challenges, we propose SmartHalo, a new framework that enhances decompiler output by combining static analysis (SA) and large language models (LLM). SmartHalo leverages the complementary strengths of SA's accuracy in control and data flow analysis and LLM's capability in semantic prediction. More specifically, \system{} constructs a new data structure - Dependency Graph (DG), to extract semantic dependencies via static analysis. Then, it takes DG to create prompts for LLM optimization. Finally, the correctness of LLM outputs is validated through symbolic execution and formal verification. Evaluation on a dataset consisting of 465 randomly selected smart contract methods shows that SmartHalo significantly improves the quality of the decompiled code, compared to SOTA decompilers (e.g., Gigahorse). Notably, integrating GPT-4o with SmartHalo further enhances its performance, achieving precision rates of 87.39% for method boundaries, 90.39% for variable types, and 80.65% for contract attributes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04426v3">The simulation of judgment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Please refer to published version: https://doi.org/10.1073/pnas.2518443122
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in evaluative processes, from information filtering to assessing and addressing knowledge gaps through explanation and credibility judgments. This raises the need to examine how such evaluations are built, what assumptions they rely on, and how their strategies diverge from those of humans. We benchmark six LLMs against expert ratings--NewsGuard and Media Bias/Fact Check--and against human judgments collected through a controlled experiment. We use news domains purely as a controlled benchmark for evaluative tasks, focusing on the underlying mechanisms rather than on news classification per se. To enable direct comparison, we implement a structured agentic framework in which both models and nonexpert participants follow the same evaluation procedure: selecting criteria, retrieving content, and producing justifications. Despite output alignment, our findings show consistent differences in the observable criteria guiding model evaluations, suggesting that lexical associations and statistical priors could influence evaluations in ways that differ from contextual reasoning. This reliance is associated with systematic effects: political asymmetries and a tendency to confuse linguistic form with epistemic reliability--a dynamic we term epistemia, the illusion of knowledge that emerges when surface plausibility replaces verification. Indeed, delegating judgment to such systems may affect the heuristics underlying evaluative processes, suggesting a shift from normative reasoning toward pattern-based approximation and raising open questions about the role of LLMs in evaluative processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14660v1">An Efficient Rubric-based Generative Verifier for Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Search augmentation empowers Large Language Models with retrieval capabilities to overcome the limitations imposed by static parameters. Recently, Reinforcement Learning leverages tailored reward signals as a viable technique to enhance LLMs performing tasks involving search. However, existing reward modeling for search-augmented LLMs faces several limitations. Rule-based rewards, such as Exact Match, are verifiable but fragile to variations in expression and cannot be applied to long-form workloads. In contrast, generative rewards improve robustness, but designing verifiable and stable rewards for long-form workloads in dynamic corpora remains challenging and also incurs high computational costs. In this paper, we propose a unified and verifiable paradigm, "nugget-as-rubric", which treats atomic information points as structured evaluation criteria for different search-augmentation workloads. Short-form tasks correspond to a single rubric, whereas long-form tasks expand to multiple rubrics aligned with the question's information needs. To support long-form settings, we design an automatic rubric construction pipeline based on query rewriting, which can automatically retrieve passages relevant to each question and extract rubrics from them, both from static corpora and from dynamic online web content. Furthermore, we introduce \textbf{Search-Gen-V}, a 4B-parameter efficient generative verifier under our proposed verifiable paradigm, which is trained via the idea of distillation and a two-stage strategy. Experimental results show that Search-Gen-V achieves strong verification accuracy across different workloads, making it a scalable, robust, and efficient verifiable reward constructor for search-augmented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12350v2">O-Forge: An LLM + Computer Algebra Framework for Asymptotic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models have recently demonstrated advanced capabilities in solving IMO and Putnam problems; yet their role in research mathematics has remained fairly limited. The key difficulty is verification: suggested proofs may look plausible, but cannot be trusted without rigorous checking. We present a framework, called LLM+CAS, and an associated tool, O-Forge, that couples frontier LLMs with a computer algebra systems (CAS) in an In-Context Symbolic Feedback loop to produce proofs that are both creative and symbolically verified. Our focus is on asymptotic inequalities, a topic that often involves difficult proofs and appropriate decomposition of the domain into the "right" subdomains. Many mathematicians, including Terry Tao, have suggested that using AI tools to find the right decompositions can be very useful for research-level asymptotic analysis. In this paper, we show that our framework LLM+CAS turns out to be remarkably effective at proposing such decompositions via a combination of a frontier LLM and a CAS. More precisely, we use an LLM to suggest domain decomposition, and a CAS (such as Mathematica) that provides a verification of each piece axiomatically. Using this loop, we answer a question posed by Terence Tao: whether LLMs coupled with a verifier can be used to help prove intricate asymptotic inequalities. More broadly, we show how AI can move beyond contest math towards research-level tools for professional mathematicians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14629v1">MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      The application of Large Language Models (LLMs) in recommender systems faces key challenges in delivering deep personalization and intelligent reasoning, especially for interactive scenarios. Current methods are often constrained by limited context windows and single-turn reasoning, hindering their ability to capture dynamic user preferences and proactively reason over recommendation contexts. To address these limitations, we propose MR.Rec, a novel framework that synergizes memory and reasoning for LLM-based recommendations. To achieve personalization, we develop a comprehensive Retrieval-Augmented Generation (RAG) system that efficiently indexes and retrieves relevant external memory to enhance LLM personalization capabilities. Furthermore, to enable the synergy between memory and reasoning, our RAG system goes beyond conventional query-based retrieval by integrating reasoning enhanced memory retrieval. Finally, we design a reinforcement learning framework that trains the LLM to autonomously learn effective strategies for both memory utilization and reasoning refinement. By combining dynamic memory retrieval with adaptive reasoning, this approach ensures more accurate, context-aware, and highly personalized recommendations. Extensive experiments demonstrate that MR.Rec significantly outperforms state-of-the-art baselines across multiple metrics, validating its efficacy in delivering intelligent and personalized recommendations. We will release code and data upon paper notification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14628v1">RLAIF-SPA: Optimizing LLM-based Emotional Speech Synthesis via RLAIF</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Text-To-Speech synthesis has achieved near-human quality in neutral speech, but emotional expressiveness remains a challenge. Existing methods often rely on costly emotion annotations or optimize indirect objectives that fail to capture the emotional expressiveness and perceptual naturalness of speech, leading to generated speech that is accurate but emotionally flat. To address these challenges, we propose the RLAIF-SPA framework, incorporating a Reinforcement Learning from AI Feedback (RLAIF) mechanism to employ Automatic Speech Recognition (ASR) and Large Language Model (LLM) techniques to respectively judge semantic accuracy and prosodic-emotional label alignment as a direct reward for emotional expressiveness and intelligibility optimization. Specifically, it leverages Prosodic Label Alignment to enhance expressive quality by jointly considering semantic accuracy and prosodic-emotional alignment along four fine-grained dimensions: Structure, Emotion, Speed, and Tone. In addition, it incorporates Semantic Accuracy Feedback to ensure the generation of clear and accurate speech. Experiments on the Libri Speech dataset show that RLAIF-SPA outperforms Chat-TTS, with a 26.1% reduction in WER, a 9.1% increase in SIM-O, and over 10% improvement in human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22544v2">HyCoVAD: A Hybrid SSL-LLM Model for Complex Video Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 The submission was made prematurely. The authors plan to resubmit under the supervision of the corresponding author
    </div>
    <details class="paper-abstract">
      Video anomaly detection (VAD) is crucial for intelligent surveillance, but a significant challenge lies in identifying complex anomalies, which are events defined by intricate relationships and temporal dependencies among multiple entities rather than by isolated actions. While self-supervised learning (SSL) methods effectively model low-level spatiotemporal patterns, they often struggle to grasp the semantic meaning of these interactions. Conversely, large language models (LLMs) offer powerful contextual reasoning but are computationally expensive for frame-by-frame analysis and lack fine-grained spatial localization. We introduce HyCoVAD, Hybrid Complex Video Anomaly Detection, a hybrid SSL-LLM model that combines a multi-task SSL temporal analyzer with LLM validator. The SSL module is built upon an nnFormer backbone which is a transformer-based model for image segmentation. It is trained with multiple proxy tasks, learns from video frames to identify those suspected of anomaly. The selected frames are then forwarded to the LLM, which enriches the analysis with semantic context by applying structured, rule-based reasoning to validate the presence of anomalies. Experiments on the challenging ComplexVAD dataset show that HyCoVAD achieves a 72.5% frame-level AUC, outperforming existing baselines by 12.5% while reducing LLM computation. We release our interaction anomaly taxonomy, adaptive thresholding protocol, and code to facilitate future research in complex VAD scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11401v3">Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13193v2">ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs), with their structured representation capabilities, offer promising avenue for enhancing Retrieval Augmented Generation (RAG) systems, leading to the development of KG-RAG systems. Nevertheless, existing methods often struggle to achieve effective synergy between system effectiveness and cost efficiency, leading to neither unsatisfying performance nor excessive LLM prompt tokens and inference time. To this end, this paper proposes REMINDRAG, which employs an LLM-guided graph traversal featuring node exploration, node exploitation, and, most notably, memory replay, to improve both system effectiveness and cost efficiency. Specifically, REMINDRAG memorizes traversal experience within KG edge embeddings, mirroring the way LLMs "memorize" world knowledge within their parameters, but in a train-free manner. We theoretically and experimentally confirm the effectiveness of REMINDRAG, demonstrating its superiority over existing baselines across various benchmark datasets and LLM backbones. Our code is available at https://github.com/kilgrims/ReMindRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14565v1">Assessing Socio-Cultural Alignment and Technical Safety of Sovereign LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent trends in LLMs development clearly show growing interest in the use and application of sovereign LLMs. The global debate over sovereign LLMs highlights the need for governments to develop their LLMs, tailored to their unique socio-cultural and historical contexts. However, there remains a shortage of frameworks and datasets to verify two critical questions: (1) how well these models align with users' socio-cultural backgrounds, and (2) whether they maintain safety and technical robustness without exposing users to potential harms and risks. To address this gap, we construct a new dataset and introduce an analytic framework for extracting and evaluating the socio-cultural elements of sovereign LLMs, alongside assessments of their technical robustness. Our experimental results demonstrate that while sovereign LLMs play a meaningful role in supporting low-resource languages, they do not always meet the popular claim that these models serve their target users well. We also show that pursuing this untested claim may lead to underestimating critical quality attributes such as safety. Our study suggests that advancing sovereign LLMs requires a more extensive evaluation that incorporates a broader range of well-grounded and practical criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14548v1">LLM Agents Beyond Utility: An Open-Ended Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Recent LLM agents have made great use of chain of thought reasoning and function calling. As their capabilities grow, an important question arises: can this software represent not only a smart problem-solving tool, but an entity in its own right, that can plan, design immediate tasks, and reason toward broader, more ambiguous goals? To study this question, we adopt an open-ended experimental setting where we augment a pretrained LLM agent with the ability to generate its own tasks, accumulate knowledge, and interact extensively with its environment. We study the resulting open-ended agent qualitatively. It can reliably follow complex multi-step instructions, store and reuse information across runs, and propose and solve its own tasks, though it remains sensitive to prompt design, prone to repetitive task generation, and unable to form self-representations. These findings illustrate both the promise and current limits of adapting pretrained LLMs toward open-endedness, and point to future directions for training agents to manage memory, explore productively, and pursue abstract long-term goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04575v2">Are LLMs Stable Formal Logic Translators in Logical Reasoning Across Linguistically Diversified Texts?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Logical reasoning with large language models (LLMs) has received growing attention. One mainstream approach translates natural language into formal logic and then applies symbolic solvers for deduction. While effective in many tasks, these LLM-based translators often fail to generate consistent symbolic representations when the same concept appears in different linguistic forms. Such inconsistencies break logical coherence and lead to solver errors. However, most existing benchmarks lack this type of linguistic variation, which frequently occurs in real-world text, leaving the problem underexplored. To address this gap, we present SoLT, a benchmark that systematically rewrites reasoning datasets into diverse yet logically equivalent forms across multiple levels. Beyond evaluation, SoLT also provides a general method to enrich any dataset with linguistic diversity while preserving both meaning and logic. To further enhance the stability of LLM-based reasoning, we propose MenTaL, which explicitly guides models to build a concept-symbol mapping table during translation. By linking equivalent expressions to shared symbols, MenTaL maintains consistency and mitigates symbol drift. Experiments on SoLT demonstrate that LLMs indeed suffer from inconsistent symbol mapping under linguistic variation, leading to significant drops in reasoning accuracy. Meanwhile, applying MenTaL brings clear and stable performance improvements across diverse inputs. Overall, our findings reveal that overlooking linguistic diversity hides key weaknesses in LLM-based translators, and our work offers a step toward more reliable logical reasoning in varied real-world scenarios. Our code is available at https://github.com/wufeiwuwoshihua/LinguDiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14522v1">Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03867v3">Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted for oral presentation at the EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a benchmark dataset of over 1,200+ meticulously curated and diverse examples across English, Mandarin, Spanish, French, Japanese, and Korean. Each example underwent careful expert review to verify its Drivelological characteristics, involving multiple rounds of discussion and adjudication to address disagreements. Using this dataset, we evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss implied rhetorical functions altogether. These findings highlight a deep representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03479v4">Women, Infamous, and Exotic Beings: A Comparative Study of Honorific Usages in Wikipedia and LLMs for Bengali and Hindi</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted and published at EMNLP 2025 (Main)
    </div>
    <details class="paper-abstract">
      The obligatory use of third-person honorifics is a distinctive feature of several South Asian languages, encoding nuanced socio-pragmatic cues such as power, age, gender, fame, and social distance. In this work, (i) We present the first large-scale study of third-person honorific pronoun and verb usage across 10,000 Hindi and Bengali Wikipedia articles with annotations linked to key socio-demographic attributes of the subjects, including gender, age group, fame, and cultural origin. (ii) Our analysis uncovers systematic intra-language regularities but notable cross-linguistic differences: honorifics are more prevalent in Bengali than in Hindi, while non-honorifics dominate while referring to infamous, juvenile, and culturally exotic entities. Notably, in both languages, and more prominently in Hindi, men are more frequently addressed with honorifics than women. (iii) To examine whether large language models (LLMs) internalize similar socio-pragmatic norms, we probe six LLMs using controlled generation and translation tasks over 1,000 culturally balanced entities. We find that LLMs diverge from Wikipedia usage, exhibiting alternative preferences in honorific selection across tasks, languages, and socio-demographic attributes. These discrepancies highlight gaps in the socio-cultural alignment of LLMs and open new directions for studying how LLMs acquire, adapt, or distort social-linguistic norms. Our code and data are publicly available at https://github.com/souro/honorific-wiki-llm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14444v1">A Free Lunch in LLM Compression: Revisiting Retraining after Pruning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      While Neural Network pruning typically requires retraining the model to recover pruning-induced performance degradation, state-of-the-art Large Language Models (LLMs) pruning methods instead solve a layer-wise mask selection and reconstruction problem on a small set of calibration data to avoid full retraining, as it is considered computationally infeasible for LLMs. Reconstructing single matrices in isolation has favorable properties, such as convexity of the objective and significantly reduced memory requirements compared to full retraining. In practice, however, reconstruction is often implemented at coarser granularities, e.g., reconstructing a whole transformer block against its dense activations instead of a single matrix. In this work, we study the key design choices when reconstructing or retraining the remaining weights after pruning. We conduct an extensive computational study on state-of-the-art GPT architectures, and report several surprising findings that challenge common intuitions about retraining after pruning. In particular, we observe a free lunch scenario: reconstructing attention and MLP components separately within each transformer block is nearly the most resource-efficient yet achieves the best perplexity. Most importantly, this Pareto-optimal setup achieves better performance than full retraining, despite requiring only a fraction of the memory. Furthermore, we demonstrate that simple and efficient pruning criteria such as Wanda can outperform much more complex approaches when the reconstruction step is properly executed, highlighting its importance. Our findings challenge the narrative that retraining should be avoided at all costs and provide important insights into post-pruning performance recovery for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12217v2">HALF: Harm-Aware LLM Fairness Evaluation Aligned with Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed across high-impact domains, from clinical decision support and legal analysis to hiring and education, making fairness and bias evaluation before deployment critical. However, existing evaluations lack grounding in real-world scenarios and do not account for differences in harm severity, e.g., a biased decision in surgery should not be weighed the same as a stylistic bias in text summarization. To address this gap, we introduce HALF (Harm-Aware LLM Fairness), a deployment-aligned framework that assesses model bias in realistic applications and weighs the outcomes by harm severity. HALF organizes nine application domains into three tiers (Severe, Moderate, Mild) using a five-stage pipeline. Our evaluation results across eight LLMs show that (1) LLMs are not consistently fair across domains, (2) model size or performance do not guarantee fairness, and (3) reasoning models perform better in medical decision support but worse in education. We conclude that HALF exposes a clear gap between previous benchmarking success and deployment readiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09721v2">A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into software engineering has driven a transition from traditional rule-based systems to autonomous agentic systems capable of solving complex problems. However, systematic progress is hindered by a lack of comprehensive understanding of how benchmarks and solutions interconnect. This survey addresses this gap by providing the first holistic analysis of LLM-powered software engineering, offering insights into evaluation methodologies and solution paradigms. We review over 150 recent papers and propose a taxonomy along two key dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, including tasks such as code generation, translation, and repair. Our analysis highlights the evolution from simple prompt engineering to sophisticated agentic systems incorporating capabilities like planning, reasoning, memory mechanisms, and tool augmentation. To contextualize this progress, we present a unified pipeline illustrating the workflow from task specification to deliverables, detailing how different solution paradigms address various complexity levels. Unlike prior surveys that focus narrowly on specific aspects, this work connects 50+ benchmarks to their corresponding solution strategies, enabling researchers to identify optimal approaches for diverse evaluation criteria. We also identify critical research gaps and propose future directions, including multi-agent collaboration, self-evolving systems, and formal verification integration. This survey serves as a foundational guide for advancing LLM-driven software engineering. We maintain a GitHub repository that continuously updates the reviewed and related papers at https://github.com/lisaGuojl/LLM-Agent-SE-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10959v2">Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 16 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14401v1">The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      A growing body of multi-agent studies with Large Language Models (LLMs) explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without full visibility into payoffs and population, relying instead on heuristics, communication, and punishment. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14392v1">FairBatching: Fairness-Aware Batch Formation for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference systems face a fundamental tension between minimizing Time-to-First-Token (TTFT) latency for new requests and maintaining a high, steady token generation rate (low Time-Per-Output-Token, or TPOT) for ongoing requests. Existing stall-free batching schedulers proposed by Sarathi, while effective at preventing decode stalls, introduce significant computational unfairness. They prioritize decode tasks excessively, simultaneously leading to underutilized decode slack and unnecessary prefill queuing delays, which collectively degrade the system's overall quality of service (QoS). This work identifies the root cause of this unfairness: the non-monotonic nature of Time-Between-Tokens (TBT) as a scheduling metric and the rigid decode-prioritizing policy that fails to adapt to dynamic workload bursts. We therefore propose FairBatching, a novel LLM inference scheduler that enforces fair resource allocation between prefill and decode tasks. It features an adaptive batch capacity determination mechanism, which dynamically adjusts the computational budget to improve the GPU utilization without triggering SLO violations. Its fair and dynamic batch formation algorithm breaks away from the decode-prioritizing paradigm, allowing computation resources to be reclaimed from bursting decode tasks to serve prefill surges, achieving global fairness. Furthermore, FairBatching provides a novel load estimation method, enabling more effective coordination with upper-level schedulers. Implemented and evaluated on realistic traces, FairBatching significantly reduces TTFT tail latency by up to 2.29x while robustly maintaining TPOT SLOs, achieving overall 20.0% improvement in single-node capacity and 54.3% improvement in cluster-level capacity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14387v1">Can MLLMs Absorb Math Reasoning Abilities from LLMs as Free Lunch?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Math reasoning has been one crucial ability of large language models (LLMs), where significant advancements have been achieved in recent years. However, most efforts focus on LLMs by curating high-quality annotation data and intricate training (or inference) paradigms, while the math reasoning performance of multi-modal LLMs (MLLMs) remains lagging behind. Since the MLLM typically consists of an LLM and a vision block, we wonder: Can MLLMs directly absorb math reasoning abilities from off-the-shelf math LLMs without tuning? Recent model-merging approaches may offer insights into this question. However, they overlook the alignment between the MLLM and LLM, where we find that there is a large gap between their parameter spaces, resulting in lower performance. Our empirical evidence reveals two key factors behind this issue: the identification of crucial reasoning-associated layers in the model and the mitigation of the gaps in parameter space. Based on the empirical insights, we propose IP-Merging that first identifies the reasoning-associated parameters in both MLLM and Math LLM, then projects them into the subspace of MLLM, aiming to maintain the alignment, and finally merges parameters in this subspace. IP-Merging is a tuning-free approach since parameters are directly adjusted. Extensive experiments demonstrate that our IP-Merging method can enhance the math reasoning ability of MLLMs directly from Math LLMs without compromising their other capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14381v1">Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13903v2">CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by NeurIPS 2025 Conference
    </div>
    <details class="paper-abstract">
      Proprietary large language models (LLMs) exhibit strong generalization capabilities across diverse tasks and are increasingly deployed on edge devices for efficiency and privacy reasons. However, deploying proprietary LLMs at the edge without adequate protection introduces critical security threats. Attackers can extract model weights and architectures, enabling unauthorized copying and misuse. Even when protective measures prevent full extraction of model weights, attackers may still perform advanced attacks, such as fine-tuning, to further exploit the model. Existing defenses against these threats typically incur significant computational and communication overhead, making them impractical for edge deployment. To safeguard the edge-deployed LLMs, we introduce CoreGuard, a computation- and communication-efficient protection method. CoreGuard employs an efficient protection protocol to reduce computational overhead and minimize communication overhead via a propagation protocol. Extensive experiments show that CoreGuard achieves upper-bound security protection with negligible overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14365v1">On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce \nameshort{}, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and \textit{implicit} versus \textit{explicit} denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14845v3">AAVENUE: Detecting LLM Biases on NLU Tasks in AAVE via a Novel Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Published at NLP4PI @ EMNLP 2024
    </div>
    <details class="paper-abstract">
      Detecting biases in natural language understanding (NLU) for African American Vernacular English (AAVE) is crucial to developing inclusive natural language processing (NLP) systems. To address dialect-induced performance discrepancies, we introduce AAVENUE ({AAVE} {N}atural Language {U}nderstanding {E}valuation), a benchmark for evaluating large language model (LLM) performance on NLU tasks in AAVE and Standard American English (SAE). AAVENUE builds upon and extends existing benchmarks like VALUE, replacing deterministic syntactic and morphological transformations with a more flexible methodology leveraging LLM-based translation with few-shot prompting, improving performance across our evaluation metrics when translating key tasks from the GLUE and SuperGLUE benchmarks. We compare AAVENUE and VALUE translations using five popular LLMs and a comprehensive set of metrics including fluency, BARTScore, quality, coherence, and understandability. Additionally, we recruit fluent AAVE speakers to validate our translations for authenticity. Our evaluations reveal that LLMs consistently perform better on SAE tasks than AAVE-translated versions, underscoring inherent biases and highlighting the need for more inclusive NLP models. We have open-sourced our source code on GitHub and created a website to showcase our work at https://aavenuee.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07452v2">When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 32 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14348v1">Automated Extraction of Protocol State Machines from 3GPP Specifications with Domain-Informed Prompts and LLM Ensembles</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Mobile telecommunication networks are foundational to global infrastructure and increasingly support critical sectors such as manufacturing, transportation, and healthcare. The security and reliability of these networks are essential, yet depend heavily on accurate modeling of underlying protocols through state machines. While most prior work constructs such models manually from 3GPP specifications, this process is labor-intensive, error-prone, and difficult to maintain due to the complexity and frequent updates of the specifications. Recent efforts using natural language processing have shown promise, but remain limited in handling the scale and intricacy of cellular protocols. In this work, we propose SpecGPT, a novel framework that leverages large language models (LLMs) to automatically extract protocol state machines from 3GPP documents. SpecGPT segments technical specifications into meaningful paragraphs, applies domain-informed prompting with chain-of-thought reasoning, and employs ensemble methods to enhance output reliability. We evaluate SpecGPT on three representative 5G protocols (NAS, NGAP, and PFCP) using manually annotated ground truth, and show that it outperforms existing approaches, demonstrating the effectiveness of LLMs for protocol modeling at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04384v2">LLM Based Bayesian Optimization for Prompt Search</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Bayesian Optimization (BO) has been widely used to efficiently optimize expensive black-box functions with limited evaluations. In this paper, we investigate the use of BO for prompt engineering to enhance text classification with Large Language Models (LLMs). We employ an LLM-powered Gaussian Process (GP) as the surrogate model to estimate the performance of different prompt candidates. These candidates are generated by an LLM through the expansion of a set of seed prompts and are subsequently evaluated using an Upper Confidence Bound (UCB) acquisition function in conjunction with the GP posterior. The optimization process iteratively refines the prompts based on a subset of the data, aiming to improve classification accuracy while reducing the number of API calls by leveraging the prediction uncertainty of the LLM-based GP. The proposed BO-LLM algorithm is evaluated on two datasets, and its advantages are discussed in detail in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14331v1">LLM-ERM: Sample-Efficient Program Learning via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We seek algorithms for program learning that are both sample-efficient and computationally feasible. Classical results show that targets admitting short program descriptions (e.g., with short ``python code'') can be learned with a ``small'' number of examples (scaling with the size of the code) via length-first program enumeration, but the search is exponential in description length. Consequently, Gradient-based training avoids this cost yet can require exponentially many samples on certain short-program families. To address this gap, we introduce LLM-ERM, a propose-and-verify framework that replaces exhaustive enumeration with an LLM-guided search over candidate programs while retaining ERM-style selection on held-out data. Specifically, we draw $k$ candidates with a pretrained reasoning-augmented LLM, compile and check each on the data, and return the best verified hypothesis, with no feedback, adaptivity, or gradients. Theoretically, we show that coordinate-wise online mini-batch SGD requires many samples to learn certain short programs. {\em Empirically, LLM-ERM solves tasks such as parity variants, pattern matching, and primality testing with as few as 200 samples, while SGD-trained transformers overfit even with 100,000 samples}. These results indicate that language-guided program synthesis recovers much of the statistical efficiency of finite-class ERM while remaining computationally tractable, offering a practical route to learning succinct hypotheses beyond the reach of gradient-based training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14471v2">Why We Build Local Large Language Models: An Observational Analysis from 35 Japanese and Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted as a spotlight at the 1st workshop on Multilingual and Equitable Language Technologies (MELT), COLM 2025
    </div>
    <details class="paper-abstract">
      Why do we build local large language models (LLMs)? What should a local LLM learn from the target language? Which abilities can be transferred from other languages? Do language-specific scaling laws exist? To explore these research questions, we evaluated 35 Japanese, English, and multilingual LLMs on 19 evaluation benchmarks for Japanese and English, taking Japanese as a local language. Adopting an observational approach, we analyzed correlations of benchmark scores, and conducted principal component analysis (PCA) on the scores to derive \textit{ability factors} of local LLMs. We found that training on English text can improve the scores of academic subjects in Japanese (JMMLU). In addition, it is unnecessary to specifically train on Japanese text to enhance abilities for solving Japanese code generation, arithmetic reasoning, commonsense, and reading comprehension tasks. In contrast, training on Japanese text could improve question-answering tasks about Japanese knowledge and English-Japanese translation, which indicates that abilities for solving these two tasks can be regarded as \textit{Japanese abilities} for LLMs. Furthermore, we confirmed that the Japanese abilities scale with the computational budget for Japanese text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21638v3">R1-Ranker: Teaching LLM Rankers to Reason</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown strong reasoning abilities in domains like mathematics, coding, and scientific problem-solving, yet their potential for ranking tasks, where prime examples include retrieval, recommender systems, and LLM routing, remains underexplored. Ranking requires complex reasoning across heterogeneous candidates, but existing LLM-based rankers are often domain-specific, tied to fixed backbones, and lack iterative refinement, limiting their ability to fully exploit LLMs' reasoning potential. To address these challenges, we propose R1-Ranker, a reasoning-incentive framework built on reinforcement learning, with two complementary designs: DRanker, which generates full rankings in one shot, and IRanker, which decomposes ranking into an iterative elimination process with step-wise rewards to encourage deeper reasoning. We evaluate unified R1-Rankers on nine datasets spanning recommendation, routing, and passage ranking, showing that IRanker-3B consistently achieves state-of-the-art performance, surpasses larger 7B models on some tasks, and yields a 15.7% average relative improvement. Ablation and generalization experiments further confirm the critical role of reinforcement learning and iterative reasoning, with IRanker-3B improving zero-shot performance by over 9% on out-of-domain tasks and reasoning traces boosting other LLMs by up to 22.87%. These results demonstrate that unifying diverse ranking tasks with a single reasoning-driven foundation model is both effective and essential for advancing LLM reasoning in ranking scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14278v1">PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Retrieval plays a central role in multi-hop question answering (QA), where answering complex questions requires gathering multiple pieces of evidence. We introduce an Agentic Retrieval System that leverages large language models (LLMs) in a structured loop to retrieve relevant evidence with high precision and recall. Our framework consists of three specialized agents: a Question Analyzer that decomposes a multi-hop question into sub-questions, a Selector that identifies the most relevant context for each sub-question (focusing on precision), and an Adder that brings in any missing evidence (focusing on recall). The iterative interaction between Selector and Adder yields a compact yet comprehensive set of supporting passages. In particular, it achieves higher retrieval accuracy while filtering out distracting content, enabling downstream QA models to surpass full-context answer accuracy while relying on significantly less irrelevant information. Experiments on four multi-hop QA benchmarks -- HotpotQA, 2WikiMultiHopQA, MuSiQue, and MultiHopRAG -- demonstrates that our approach consistently outperforms strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14277v1">GenLARP: Enabling Immersive Live Action Role-Play through LLM-Generated Worlds and Characters</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We introduce GenLARP, a virtual reality (VR) system that transforms personalized stories into immersive live action role-playing (LARP) experiences. GenLARP enables users to act as both creators and players, allowing them to design characters based on their descriptions and live in the story world. Generative AI and agents powered by Large Language Models (LLMs) enrich these experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08615v3">Iterative LLM-Based Generation and Refinement of Distracting Conditions in Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Mathematical reasoning serves as a crucial testbed for the intelligence of large language models (LLMs), and math word problems (MWPs) are a popular type of math problems. Most MWP datasets consist of problems containing only the necessary information, while problems with distracting and excessive conditions are often overlooked. Prior works have tested popular LLMs and found a dramatic performance drop in the presence of distracting conditions. However, datasets of MWPs with distracting conditions are limited, and most suffer from lower levels of difficulty and out-of-context expressions. This makes distracting conditions easy to identify and exclude, thus reducing the credibility of benchmarking on them. Moreover, when adding distracting conditions, the reasoning and answers may also change, requiring intensive labor to check and write the solutions. To address these issues, we design an iterative framework to generate distracting conditions using LLMs. We develop a set of prompts to revise MWPs from different perspectives and cognitive levels, encouraging the generation of distracting conditions as well as suggestions for further revision. Another advantage is the shared solutions between original and revised problems: we explicitly guide the LLMs to generate distracting conditions that do not alter the original solutions, thus avoiding the need to generate new solutions. This framework is efficient and easy to deploy, reducing the overhead of generating MWPs with distracting conditions while maintaining data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14253v1">Towards Agentic Self-Learning LLMs in Search Environment</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      We study whether self-learning can scale LLM-based agents without relying on human-curated datasets or predefined rule-based rewards. Through controlled experiments in a search-agent setting, we identify two key determinants of scalable agent training: the source of reward signals and the scale of agent task data. We find that rewards from a Generative Reward Model (GRM) outperform rigid rule-based signals for open-domain learning, and that co-evolving the GRM with the policy further boosts performance. Increasing the volume of agent task data-even when synthetically generated-substantially enhances agentic capabilities. Building on these insights, we propose \textbf{Agentic Self-Learning} (ASL), a fully closed-loop, multi-role reinforcement learning framework that unifies task generation, policy execution, and evaluation within a shared tool environment and LLM backbone. ASL coordinates a Prompt Generator, a Policy Model, and a Generative Reward Model to form a virtuous cycle of harder task setting, sharper verification, and stronger solving. Empirically, ASL delivers steady, round-over-round gains, surpasses strong RLVR baselines (e.g., Search-R1) that plateau or degrade, and continues improving under zero-labeled-data conditions, indicating superior sample efficiency and robustness. We further show that GRM verification capacity is the main bottleneck: if frozen, it induces reward hacking and stalls progress; continual GRM training on the evolving data distribution mitigates this, and a small late-stage injection of real verification data raises the performance ceiling. This work establishes reward source and data scale as critical levers for open-domain agent learning and demonstrates the efficacy of multi-role co-evolution for scalable, self-improving agents. The data and code of this paper are released at https://github.com/forangel2014/Towards-Agentic-Self-Learning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13921v2">APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong reasoning and task planning capabilities but remain fundamentally limited in physical interaction modeling. Existing approaches integrate perception via Vision-Language Models (VLMs) or adaptive decision-making through Reinforcement Learning (RL), but they fail to capture dynamic object interactions or require task-specific training, limiting their real-world applicability. We introduce APEX (Anticipatory Physics-Enhanced Execution), a framework that equips LLMs with physics-driven foresight for real-time task planning. APEX constructs structured graphs to identify and model the most relevant dynamic interactions in the environment, providing LLMs with explicit physical state updates. Simultaneously, APEX provides low-latency forward simulations of physically feasible actions, allowing LLMs to select optimal strategies based on predictive outcomes rather than static observations. We evaluate APEX on three benchmarks designed to assess perception, prediction, and decision-making: (1) Physics Reasoning Benchmark, testing causal inference and object motion prediction; (2) Tetris, evaluating whether physics-informed prediction enhances decision-making performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance, assessing the immediate integration of perception and action feasibility analysis. APEX significantly outperforms standard LLMs and VLM-based models, demonstrating the necessity of explicit physics reasoning for bridging the gap between language-based intelligence and real-world task execution. The source code and experiment setup are publicly available at https://github.com/hwj20/APEX_EXP .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25271v2">RADAR: A Risk-Aware Dynamic Multi-Agent Framework for LLM Safety Evaluation via Role-Specialized Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-16
    </div>
    <details class="paper-abstract">
      Existing safety evaluation methods for large language models (LLMs) suffer from inherent limitations, including evaluator bias and detection failures arising from model homogeneity, which collectively undermine the robustness of risk evaluation processes. This paper seeks to re-examine the risk evaluation paradigm by introducing a theoretical framework that reconstructs the underlying risk concept space. Specifically, we decompose the latent risk concept space into three mutually exclusive subspaces: the explicit risk subspace (encompassing direct violations of safety guidelines), the implicit risk subspace (capturing potential malicious content that requires contextual reasoning for identification), and the non-risk subspace. Furthermore, we propose RADAR, a multi-agent collaborative evaluation framework that leverages multi-round debate mechanisms through four specialized complementary roles and employs dynamic update mechanisms to achieve self-evolution of risk concept distributions. This approach enables comprehensive coverage of both explicit and implicit risks while mitigating evaluator bias. To validate the effectiveness of our framework, we construct an evaluation dataset comprising 800 challenging cases. Extensive experiments on our challenging testset and public benchmarks demonstrate that RADAR significantly outperforms baseline evaluation methods across multiple dimensions, including accuracy, stability, and self-evaluation risk sensitivity. Notably, RADAR achieves a 28.87% improvement in risk identification accuracy compared to the strongest baseline evaluation method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14242v1">Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 14 pages, 6 figures, 3 tables, and 1 algorithm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce inconsistent answers when faced with different phrasings of the same prompt. In this paper, we propose Flip-Flop Consistency ($F^2C$), an unsupervised training method that improves robustness to such perturbations. $F^2C$ is composed of two key components. The first, Consensus Cross-Entropy (CCE), uses a majority vote across prompt variations to create a hard pseudo-label. The second is a representation alignment loss that pulls lower-confidence and non-majority predictors toward the consensus established by high-confidence, majority-voting variations. We evaluate our method on 11 datasets spanning four NLP tasks, with 4-15 prompt variations per dataset. On average, $F^2C$ raises observed agreement by 11.62%, improves mean $F_1$ by 8.94%, and reduces performance variance across formats by 3.29%. In out-of-domain evaluations, $F^2C$ generalizes effectively, increasing $\overline{F_1}$ and agreement while decreasing variance across most source-target pairs. Finally, when trained on only a subset of prompt perturbations and evaluated on held-out formats, $F^2C$ consistently improves both performance and agreement while reducing variance. These findings highlight $F^2C$ as an effective unsupervised method for enhancing LLM consistency, performance, and generalization under prompt perturbations. Code is available at https://github.com/ParsaHejabi/Flip-Flop-Consistency-Unsupervised-Training-for-Robustness-to-Prompt-Perturbations-in-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00527v2">Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 Accepted by IEEE/RSJ IROS 2025
    </div>
    <details class="paper-abstract">
      The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v4">LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 11 pages, 6 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. In this paper, we investigate the Soft Thinking capabilities of various LLMs through a systematic analysis of their internal behavior using a suite of probing techniques. Contrary to the prevailing belief that Soft Thinking supports parallel exploration of diverse reasoning paths, our findings reveal that LLMs behave as single-threaded reasoners--they predominantly rely on the token with the highest probability in the soft input to predict the next step. This behavior induces a greedy feedback loop that suppresses alternative reasoning paths and undermines the benefits of transmitting richer information via Soft Tokens. To address this Greedy Pitfall, we propose Stochastic Soft Thinking, which introduces stochasticity to break free from this Greedy Pitfall. Our experiments demonstrate that incorporating randomness--particularly with the Gumbel-Softmax trick--can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking, resulting in superior performance across eight reasoning benchmarks. We further demonstrate that Stochastic Soft Thinking exhibits stronger exploration potential compared to conventional COT. Our findings deepen the understanding of continuous reasoning and establish the foundation for future work on improving Soft Thinking with Reinforcement Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14207v1">Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-16
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01656v3">Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (AsyPPO), a simple and scalable framework that restores the critics role while remaining efficient in large-model settings. AsyPPO employs a set of lightweight mini-critics, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, AsyPPO leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. After training on open-source data with only 5,000 samples, AsyPPO consistently improves learning stability and performance across multiple benchmarks over strong baselines, such as GRPO, achieving performance gains of more than six percent on Qwen3-4b-Base and about three percent on Qwen3-8b-Base and Qwen3-14b-Base over classic PPO, without additional tricks. These results highlight the importance of architectural innovations for scalable, efficient algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13291v1">Higher Satisfaction, Lower Cost: A Technical Report on How LLMs Revolutionize Meituan's Intelligent Interaction Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 36 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Enhancing customer experience is essential for business success, particularly as service demands grow in scale and complexity. Generative artificial intelligence and Large Language Models (LLMs) have empowered intelligent interaction systems to deliver efficient, personalized, and 24/7 support. In practice, intelligent interaction systems encounter several challenges: (1) Constructing high-quality data for cold-start training is difficult, hindering self-evolution and raising labor costs. (2) Multi-turn dialogue performance remains suboptimal due to inadequate intent understanding, rule compliance, and solution extraction. (3) Frequent evolution of business rules affects system operability and transferability, constraining low-cost expansion and adaptability. (4) Reliance on a single LLM is insufficient in complex scenarios, where the absence of multi-agent frameworks and effective collaboration undermines process completeness and service quality. (5) The open-domain nature of multi-turn dialogues, lacking unified golden answers, hampers quantitative evaluation and continuous optimization. To address these challenges, we introduce WOWService, an intelligent interaction system tailored for industrial applications. With the integration of LLMs and multi-agent architectures, WOWService enables autonomous task management and collaborative problem-solving. Specifically, WOWService focuses on core modules including data construction, general capability enhancement, business scenario adaptation, multi-agent coordination, and automated evaluation. Currently, WOWService is deployed on the Meituan App, achieving significant gains in key metrics, e.g., User Satisfaction Metric 1 (USM 1) -27.53% and User Satisfaction Metric 2 (USM 2) +25.51%, demonstrating its effectiveness in capturing user needs and advancing personalized service.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13271v1">Do You Get the Hint? Benchmarking LLMs on the Board Game Concept</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved striking successes on many benchmarks, yet recent studies continue to expose fundamental weaknesses. In particular, tasks that require abstract reasoning remain challenging, often because they use representations such as grids, symbols, or visual patterns that differ from the natural language data LLMs are trained on. In this paper, we introduce Concept, a simple word-guessing board game, as a benchmark for probing abductive reasoning in a representation that is much closer to LLM pre-training data: natural language. Our results show that this game, easily solved by humans (with a success rate of over 90\%), is still very challenging for state-of-the-art LLMs (no model exceeds 40\% success rate). Specifically, we observe that LLMs struggle with interpreting other players' strategic intents, and with correcting initial hypotheses given sequential information updates. In addition, we extend the evaluation across multiple languages, and find that the LLM performance drops further in lower-resource languages (Dutch, French, and Spanish) compared to English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13257v1">GRIDAI: Generating and Repairing Intrusion Detection Rules via Collaboration among Multiple LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Rule-based network intrusion detection systems play a crucial role in the real-time detection of Web attacks. However, most existing works primarily focus on automatically generating detection rules for new attacks, often overlooking the relationships between new attacks and existing rules, which leads to significant redundancy within the ever-expanding ruleset. To address this issue, we propose GRIDAI, a novel end-to-end framework for the automated Generation and Repair of Intrusion Detection rules through collaboration among multiple LLM-based agents. Unlike traditional methods, GRIDAI first assesses the nature of incoming attack samples. If the sample represents a new attack type, it is used to generate a new rule. Otherwise, the sample is identified as a variant of an attack already covered by an existing rule and used to repair the rule by updating the corresponding signature, thereby enhancing its generalization capability. Additionally, to mitigate syntactic and semantic errors in rules caused by LLM hallucinations, we incorporate a tool-based real-time validation mechanism and a representative attack sample maintained for each rule, enabling fully automated rule generation and repair. Comprehensive experiments were conducted on a public dataset containing seven types of attacks and a private dataset with 43 attack types. The results demonstrate that GRIDAI accurately identifies the relationships between new attack samples and existing rules, efficiently generates and repairs rules to handle new attacks and variants, and effectively mitigates the impact of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03567v2">Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13248v1">Automated Network Protocol Testing with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Network protocol testing is fundamental for modern network infrastructure. However, traditional network protocol testing methods are labor-intensive and error-prone, requiring manual interpretation of specifications, test case design, and translation into executable artifacts, typically demanding one person-day of effort per test case. Existing model-based approaches provide partial automation but still involve substantial manual modeling and expert intervention, leading to high costs and limited adaptability to diverse and evolving protocols. In this paper, we propose a first-of-its-kind system called NeTestLLM that takes advantage of multi-agent Large Language Models (LLMs) for end-to-end automated network protocol testing. NeTestLLM employs hierarchical protocol understanding to capture complex specifications, iterative test case generation to improve coverage, a task-specific workflow for executable artifact generation, and runtime feedback analysis for debugging and refinement. NeTestLLM has been deployed in a production environment for several months, receiving positive feedback from domain experts. In experiments, NeTestLLM generated 4,632 test cases for OSPF, RIP, and BGP, covering 41 historical FRRouting bugs compared to 11 by current national standards. The process of generating executable artifacts also improves testing efficiency by a factor of 8.65x compared to manual methods. NeTestLLM provides the first practical LLM-powered solution for automated end-to-end testing of heterogeneous network protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v2">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using LLMs, a paradigm known as "LLM-as-a-judge". However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. Previous studies mainly optimize based on shallow outputs, overlooking rich cross-layer representations. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and task-relevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a post-hoc, plug-and-play framework for improving the alignment of LLM-as-a-Judge point-wise evaluations with human scores by leveraging internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer score-token logits and computing the expected score from a softmax-based distribution, while keeping the LLM backbone frozen and ensuring no impact on the inference process. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the generalization of LAGER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13229v1">Beyond Static LLM Policies: Imitation-Enhanced Reinforcement Learning for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 ICDM 2025 Accepted Paper
    </div>
    <details class="paper-abstract">
      Recommender systems (RecSys) have become critical tools for enhancing user engagement by delivering personalized content across diverse digital platforms. Recent advancements in large language models (LLMs) demonstrate significant potential for improving RecSys, primarily due to their exceptional generalization capabilities and sophisticated contextual understanding, which facilitate the generation of flexible and interpretable recommendations. However, the direct deployment of LLMs as primary recommendation policies presents notable challenges, including persistent latency issues stemming from frequent API calls and inherent model limitations such as hallucinations and biases. To address these issues, this paper proposes a novel offline reinforcement learning (RL) framework that leverages imitation learning from LLM-generated trajectories. Specifically, inverse reinforcement learning is employed to extract robust reward models from LLM demonstrations. This approach negates the need for LLM fine-tuning, thereby substantially reducing computational overhead. Simultaneously, the RL policy is guided by the cumulative rewards derived from these demonstrations, effectively transferring the semantic insights captured by the LLM. Comprehensive experiments conducted on two benchmark datasets validate the effectiveness of the proposed method, demonstrating superior performance when compared against state-of-the-art RL-based and in-context learning baselines. The code can be found at https://github.com/ArronDZhang/IL-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01427v2">A Tale of LLMs and Induced Small Proxies: Scalable Agents for Knowledge Mining</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Code available: https://github.com/LongfeiYun17/falconer
    </div>
    <details class="paper-abstract">
      At the core of Deep Research is knowledge mining, the task of extracting structured information from massive unstructured text in response to user instructions. Large language models (LLMs) excel at interpreting such instructions but are prohibitively expensive to deploy at scale, while traditional pipelines of classifiers and extractors remain efficient yet brittle and unable to generalize to new tasks. We introduce Falconer, a collaborative framework that combines the agentic reasoning of LLMs with lightweight proxy models for scalable knowledge mining. In Falconer, LLMs act as planners, decomposing user instructions into executable pipelines, and as annotators, generating supervision to train small proxies. The framework unifies classification and extraction into two atomic operations, get label and get span, enabling a single instruction-following model to replace multiple task-specific components. To evaluate the consistency between proxy models incubated by Falconer and annotations provided by humans and large models, we construct new benchmarks covering both planning and end-to-end execution. Experiments show that Falconer closely matches state-of-the-art LLMs in instruction-following accuracy while reducing inference cost by up to 90% and accelerating large-scale knowledge mining by more than 20x, offering an efficient and scalable foundation for Deep Research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13223v1">BanaServe: Unified KV Cache and Dynamic Module Migration for Balancing Disaggregated LLM Serving in AI Infrastructure</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in AI infrastructure, driving the need for high throughput, resource efficient serving systems. Disaggregated LLM serving, which separates prompt prefill from auto-regressive decode, has emerged as a promising architecture by isolating their heterogeneous compute and memory demands. However, current disaggregated systems face three key limitations: (i) static resource allocation cannot adapt to highly dynamic workloads, causing over-provisioning that wastes resources or under-provisioning that violates service level objectives (SLOs); (ii) inherent load imbalance between prefill and decode stages, where prefill is compute-bound and decode is memory-bound, causes under-utilization in one tier while the other becomes a bottleneck; and (iii) prefix cache aware routing skews load distribution, as high cache hit rate prefill nodes attract disproportionately more requests, further degrading balance and efficiency. To address these issues, we present BanaServe, a dynamic orchestration framework that continuously rebalances computational and memory resources across prefill and decode instances while eliminating hotspots induced by cache. BanaServe introduces layer level weight migration, attention level Key Value Cache (KV Cache) migration, and Global KV Cache Store sharing with layer wise overlapped transmission, enabling both coarse grained (layer level) and fine grained (attention level) load redistribution with minimal latency overhead. These mechanisms allow routers to perform purely load aware scheduling, unconstrained by cache placement. Compared to vLLM, BanaServe achieves 1.2x-3.9x higher throughput with 3.9%-78.4% lower total processing time, and outperforms DistServe by 1.1x-2.8x in throughput with 1.4%-70.1% latency reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13217v1">LLM-guided Hierarchical Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Modern IR systems are increasingly tasked with answering complex, multi-faceted queries that require deep reasoning rather than simple keyword or semantic matching. While LLM-based IR has shown great promise, the prevailing retrieve-then-rerank paradigm inherits the limitations of embedding-based retrieval; parametric generative approaches are difficult to update with new information; and long-context methods that place the entire corpus in context are computationally infeasible for large document collections. To address these challenges, we introduce LATTICE, a hierarchical retrieval framework that enables an LLM to reason over and navigate large corpora with logarithmic search complexity by imposing a semantic tree structure on the corpus. Our approach consists of two stages: (1) an offline phase that organizes the corpus into a semantic hierarchy via either a bottom-up agglomerative strategy or a top-down divisive strategy using multi-level summaries and (2) an online traversal phase where a search LLM navigates this tree. A central challenge in such LLM-guided search is that the model's relevance judgments are noisy, context-dependent, and unaware of the hierarchy, making cross-branch and cross-level comparisons difficult. To overcome this, we propose a traversal algorithm that estimates calibrated latent relevance scores from local LLM outputs and aggregates them into a global path relevance metric. Our training-free framework achieves state-of-the-art zero-shot performance on the reasoning-intensive BRIGHT benchmark, demonstrating up to 9% improvement in Recall@100 and 5% in nDCG@10 over the next best zero-shot baseline. Furthermore, compared to the fine-tuned SOTA method DIVER-v2, LATTICE attains comparable results on BRIGHT subsets that use a static corpus for evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13202v1">LLM-Guided Synthetic Augmentation (LGSA) for Mitigating Bias in AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 11 pages, 4 figures, 1 Table, submitted to an international conference
    </div>
    <details class="paper-abstract">
      Bias in AI systems, especially those relying on natural language data, raises ethical and practical concerns. Underrepresentation of certain groups often leads to uneven performance across demographics. Traditional fairness methods, such as pre-processing, in-processing, and post-processing, depend on protected-attribute labels, involve accuracy-fairness trade-offs, and may not generalize across datasets. To address these challenges, we propose LLM-Guided Synthetic Augmentation (LGSA), which uses large language models to generate counterfactual examples for underrepresented groups while preserving label integrity. We evaluated LGSA on a controlled dataset of short English sentences with gendered pronouns, professions, and binary classification labels. Structured prompts were used to produce gender-swapped paraphrases, followed by quality control including semantic similarity checks, attribute verification, toxicity screening, and human spot checks. The augmented dataset expanded training coverage and was used to train a classifier under consistent conditions. Results show that LGSA reduces performance disparities without compromising accuracy. The baseline model achieved 96.7 percent accuracy with a 7.2 percent gender bias gap. Simple swap augmentation reduced the gap to 0.7 percent but lowered accuracy to 95.6 percent. LGSA achieved 99.1 percent accuracy with a 1.9 percent bias gap, improving performance on female-labeled examples. These findings demonstrate that LGSA is an effective strategy for bias mitigation, enhancing subgroup balance while maintaining high task accuracy and label fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05258v2">Spatio-Temporal LLM: Reasoning about Environments and Actions</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Code and data are available at https://zoezheng126.github.io/STLLM-website/
    </div>
    <details class="paper-abstract">
      Despite significant recent progress of Multimodal Large Language Models (MLLMs), current MLLMs are challenged by "spatio-temporal" prompts, i.e., prompts that refer to 1) the entirety of an environment encoded in a point cloud that the MLLM should consider; and simultaneously also refer to 2) actions that happened in part of the environment and are encoded in a short ego-centric video clip. However, such a holistic spatio-temporal understanding is important for agents operating in the real world. To address this challenge, we first develop a framework to collect a large-scale dataset. Using the collected "Reasoning about Environments and Actions" (REA) dataset, we show that recent MLLMs indeed struggle to correctly answer "spatio-temporal" prompts. Building on this dataset, we study two spatio-temporal LLM (STLLM) baselines: 1) STLLM-3D, which directly fuses point cloud, video, and text representations as inputs to the LLM; and 2) STLLM-Aligner, which aligns spatial context with video and text before LLM decoding. Both baselines aim to enhance spatial understanding of environments and temporal grounding of egocentric observations. On REA, the STLLM baselines outperform existing models, demonstrating the effectiveness of our designs. Code and data are available at https://zoezheng126.github.io/STLLM-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13195v1">Emotional Cognitive Modeling Framework with Desire-Driven Objective Optimization for LLM-empowered Agent in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has enabled agents to represent virtual humans in societal simulations, facilitating diverse interactions within complex social systems. However, existing LLM-based agents exhibit severe limitations in affective cognition: They fail to simulate the bounded rationality essential for bridging virtual and real-world services; They lack empirically validated integration mechanisms embedding emotions within agent decision architectures. This paper constructs an emotional cognition framework incorporating desire generation and objective management, designed to achieve emotion alignment between LLM-based agents and humans, modeling the complete decision-making process of LLM-based agents, encompassing state evolution, desire generation, objective optimization, decision generation, and action execution. This study implements the proposed framework within our proprietary multi-agent interaction environment. Experimental results demonstrate that agents governed by our framework not only exhibit behaviors congruent with their emotional states but also, in comparative assessments against other agent types, demonstrate superior ecological validity and generate decision outcomes that significantly more closely approximate human behavioral patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13193v1">ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs), with their structured representation capabilities, offer promising avenue for enhancing Retrieval Augmented Generation (RAG) systems, leading to the development of KG-RAG systems. Nevertheless, existing methods often struggle to achieve effective synergy between system effectiveness and cost efficiency, leading to neither unsatisfying performance nor excessive LLM prompt tokens and inference time. To this end, this paper proposes REMINDRAG, which employs an LLM-guided graph traversal featuring node exploration, node exploitation, and, most notably, memory replay, to improve both system effectiveness and cost efficiency. Specifically, REMINDRAG memorizes traversal experience within KG edge embeddings, mirroring the way LLMs "memorize" world knowledge within their parameters, but in a train-free manner. We theoretically and experimentally confirm the effectiveness of REMINDRAG, demonstrating its superiority over existing baselines across various benchmark datasets and LLM backbones. Our code is available at https://github.com/kilgrims/ReMindRAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23227v2">Enabling Few-Shot Alzheimer's Disease Diagnosis on Biomarker Data with Tabular LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 accepted by ACM-BCB'25: ACM Conference on Bioinformatics, Computational Biology, and Health Informatics [ACM SIGBio Best Paper Award]
    </div>
    <details class="paper-abstract">
      Early and accurate diagnosis of Alzheimer's disease (AD), a complex neurodegenerative disorder, requires analysis of heterogeneous biomarkers (e.g., neuroimaging, genetic risk factors, cognitive tests, and cerebrospinal fluid proteins) typically represented in a tabular format. With flexible few-shot reasoning, multimodal integration, and natural-language-based interpretability, large language models (LLMs) offer unprecedented opportunities for prediction with structured biomedical data. We propose a novel framework called TAP-GPT, Tabular Alzheimer's Prediction GPT, that adapts TableGPT2, a multimodal tabular-specialized LLM originally developed for business intelligence tasks, for AD diagnosis using structured biomarker data with small sample sizes. Our approach constructs few-shot tabular prompts using in-context learning examples from structured biomedical data and finetunes TableGPT2 using the parameter-efficient qLoRA adaption for a clinical binary classification task of AD or cognitively normal (CN). The TAP-GPT framework harnesses the powerful tabular understanding ability of TableGPT2 and the encoded prior knowledge of LLMs to outperform more advanced general-purpose LLMs and a tabular foundation model (TFM) developed for prediction tasks. To our knowledge, this is the first application of LLMs to the prediction task using tabular biomarker data, paving the way for future LLM-driven multi-agent frameworks in biomedical informatics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13161v1">Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rates but introduces additional latency overhead exacerbating the speed-accuracy tradeoff. Prior methods (Medusa, Hydra, EAGLE) partially reduce draft cost but either degrade acceptance or introduce overheads that limit scaling. We present Mirror Speculative Decoding (Mirror-SD), an inference algorithm that breaks the latency-acceptance tradeoff. Mirror-SD launches branch-complete rollouts from early-exit signals in parallel with the target model's suffix and explicitly maps computation across heterogeneous accelerators (GPU and NPU) to exploit cross-device parallelism. The draft speculates forward continuations for the target to verify, while the target simultaneously speculates correction paths for the draft, converting speculation into two complementary execution pipelines. To further cut draft latency without weakening acceptance semantics, we add speculative streaming so the draft emits multiple tokens per step. This dual strategy of parallel heterogeneous execution plus multi-token speculative streaming pushes speculative decoding toward its ideal regime of high acceptance with low overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD delivers consistent end-to-end gains, achieving 2.8x-5.8x wall-time speedups across diverse tasks and a 30% average relative improvement over the strongest baseline, EAGLE3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13154v1">I Am Aligned, But With Whom? MENA Values Benchmark for Evaluating Cultural Alignment and Multilingual Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      We introduce MENAValues, a novel benchmark designed to evaluate the cultural alignment and multilingual biases of large language models (LLMs) with respect to the beliefs and values of the Middle East and North Africa (MENA) region, an underrepresented area in current AI evaluation efforts. Drawing from large-scale, authoritative human surveys, we curate a structured dataset that captures the sociocultural landscape of MENA with population-level response distributions from 16 countries. To probe LLM behavior, we evaluate diverse models across multiple conditions formed by crossing three perspective framings (neutral, personalized, and third-person/cultural observer) with two language modes (English and localized native languages: Arabic, Persian, Turkish). Our analysis reveals three critical phenomena: "Cross-Lingual Value Shifts" where identical questions yield drastically different responses based on language, "Reasoning-Induced Degradation" where prompting models to explain their reasoning worsens cultural alignment, and "Logit Leakage" where models refuse sensitive questions while internal probabilities reveal strong hidden preferences. We further demonstrate that models collapse into simplistic linguistic categories when operating in native languages, treating diverse nations as monolithic entities. MENAValues offers a scalable framework for diagnosing cultural misalignment, providing both empirical insights and methodological tools for developing more culturally inclusive AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13143v1">Stable LLM Ensemble: Interaction between Example Representativeness and Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable results in wide range of domains. However, the accuracy and robustness of one-shot LLM predictions remain highly sensitive to the examples and the diversity among ensemble members. This study systematically investigates the effects of example representativeness (one-shot strategy) and output diversity (sampling temperature) on LLM ensemble performance. Two one-shot strategies are compared: centroid-based representative examples (proposed) and randomly sampled examples (baseline) and sampling temperature also is varied. The proposed approach with higher temperature setting significantly outperforms random selection by +7.6% (macro-F1) and -10.5% (RMSE). Furthermore, the proposed model exceeds 5-shot prompting by +21.1% (macro-F1) and -24.0% (RMSE). Our findings demonstrate that combining representative example selection with increased temperature provides the appropriate level of diversity to the ensemble. This work highlights the practical importance of both example selection and controlled diversity in designing effective one-shot LLM ensembles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13139v1">Addressing the alignment problem in transportation policy making: an LLM approach</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      A key challenge in transportation planning is that the collective preferences of heterogeneous travelers often diverge from the policies produced by model-driven decision tools. This misalignment frequently results in implementation delays or failures. Here, we investigate whether large language models (LLMs), noted for their capabilities in reasoning and simulating human decision-making, can help inform and address this alignment problem. We develop a multi-agent simulation in which LLMs, acting as agents representing residents from different communities in a city, participate in a referendum on a set of transit policy proposals. Using chain-of-thought reasoning, LLM agents provide ranked-choice or approval-based preferences, which are aggregated using instant-runoff voting (IRV) to model democratic consensus. We implement this simulation framework with both GPT-4o and Claude-3.5, and apply it for Chicago and Houston. Our findings suggest that LLM agents are capable of approximating plausible collective preferences and responding to local context, while also displaying model-specific behavioral biases and modest divergences from optimization-based benchmarks. These capabilities underscore both the promise and limitations of LLMs as tools for solving the alignment problem in transportation decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13358v4">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23988v2">LLM/Agent-as-Data-Analyst: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 32 page, 11 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) and agent techniques for data analysis (a.k.a LLM/Agent-as-Data-Analyst) have demonstrated substantial impact in both academica and industry. In comparison with traditional rule or small-model based approaches, (agentic) LLMs enable complex data understanding, natural language interfaces, semantic analysis functions, and autonomous pipeline orchestration. The technical evolution further distills five key design goals for intelligent data analysis agents, namely semantic-aware design, modality-hybrid integration, autonomous pipelines, tool-augmented workflows, and support for open-world tasks. From a modality perspective, we review LLM-based techniques for (i) structured data (e.g., table question answering for relational data and NL2GQL for graph data), (ii) semi-structured data (e.g., markup languages understanding and semi-structured table modeling), (iii) unstructured data (e.g., chart understanding, document understanding, programming languages vulnerable detection), and (iv) heterogeneous data (e.g., data retrieval and modality alignment for data lakes). Finally, we outline the remaining challenges and propose several insights and practical directions for advancing LLM/Agent-powered data analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09657v3">Týr-the-Pruner: Structural Pruning LLMs via Global Sparsity Distribution Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Structural pruning enhances hardware-agnostic inference efficiency for large language models (LLMs) yet often fails to maintain comparable performance. Local pruning performs efficient layer-by-layer compression but ignores global topology. Although global pruning aims to identify an optimal sparse model, intuitive methods typically adopt a two-stage paradigm that first evaluates substructure saliency and then applies global pruning, which ignores inter-structure dependencies and fails to achieve end-to-end optimization. To address these limitations, we propose T\'yr-the-Pruner, an efficient end-to-end search-based global structural pruning framework. This framework constructs a supernet by repeatedly applying local pruning across a range of sparsity ratios to each layer in an LLM, with the core goal of determining the optimal sparsity distribution under a target overall sparsity ratio. Concretely, we introduce an effective local pruning and an expectation error accumulation approach to improve supernet construction. Furthermore, we employ an iterative prune-and-search strategy with coarse-to-fine sparsity granularity to ensure efficient search convergence. Experimental results show that T\'yr-the-Pruner achieves state-of-the-art structural pruning, retaining 97% of the dense model's performance while removing a challenging 50% of Llama-3.1-70B's parameters. Code will be available at https://github.com/AMD-AGI/Tyr-the-Pruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13105v1">EgoSocial: Benchmarking Proactive Intervention Ability of Omnimodal LLMs via Egocentric Social Interaction Perception</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      As AR/VR technologies become integral to daily life, there's a growing need for AI that understands human social dynamics from an egocentric perspective. However, current LLMs often lack the social awareness to discern when to intervene as AI assistant. This leads to constant, socially unaware responses that may disrupt natural conversation and negatively impact user focus. To address these limitations, we introduce EgoSocial, a large-scale egocentric dataset with 13,500 social video-question pairs, specifically designed to benchmark intervention in social interaction perception. We also present an in-depth analysis of current omnimodal LLMs (OLLMs) to assess their effectiveness in detecting diverse social contextual cues. Experiments show that OLLMs still struggle to detect the intervention timing (14.4% for Gemini 2.5 Pro). We also propose EgoSoD (EgoSocial Detection), an end-to-end method for robustly discerning social dynamics. Informed by our OLLM analysis, EgoSoD integrates multimodal contextual cues (e.g., audio and visual cues) into a social thinking graph, dynamically modeling participants and interactions. Our method proactively detects intervention timing and social interactions, precisely determining when to intervene. Our EgoSoD improves Phi-4 by 45.6% and Gemini 2.5 Pro by 9.9% on Intervention Timing performance, and improves Phi-4 by 20.4% and Gemini 2.5 Pro by 6.9% on overall Social Interaction performance. We will release the dataset and code soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10425v3">Learning to Think: Information-Theoretic Reinforcement Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex tasks thanks to advances in their reasoning abilities. However, existing methods overlook the trade-off between reasoning effectiveness and efficiency, often encouraging unnecessarily long reasoning chains and wasting tokens. To address this, we propose Learning to Think (L2T), an information-theoretic reinforcement fine-tuning framework for LLMs to make the models achieve optimal reasoning with fewer tokens. Specifically, L2T treats each query-response interaction as a hierarchical session of multiple episodes and proposes a universal dense process reward, i.e., quantifies the episode-wise information gain in parameters, requiring no extra annotations or task-specific evaluators. We propose a method to quickly estimate this reward based on PAC-Bayes bounds and the Fisher information matrix. Theoretical analyses show that it significantly reduces computational complexity with high estimation accuracy. By immediately rewarding each episode's contribution and penalizing excessive updates, L2T optimizes the model via reinforcement learning to maximize the use of each episode and achieve effective updates. Empirical results on various reasoning benchmarks and base models demonstrate the advantage of L2T across different tasks, boosting both reasoning effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23694v3">SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: https://github.com/jianshuod/SafeSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18133v4">Self-Evolving LLMs via Continual Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      In real-world industrial settings, large language models (LLMs) must learn continually to keep pace with diverse and evolving tasks, requiring self-evolution to refine knowledge under dynamic data distributions. However, existing continual learning (CL) approaches, such as replay and parameter isolation, often suffer from catastrophic forgetting: training on new tasks degrades performance on earlier ones by overfitting to the new distribution and weakening generalization.We propose MoE-CL, a parameter-efficient adversarial mixture-of-experts framework for industrial-scale, self-evolving continual instruction tuning of LLMs. MoE-CL uses a dual-expert design: (1) a dedicated LoRA expert per task to preserve task-specific knowledge via parameter independence, mitigating forgetting; and (2) a shared LoRA expert to enable cross-task transfer. To prevent transferring task-irrelevant noise through the shared pathway, we integrate a task-aware discriminator within a GAN. The discriminator encourages the shared expert to pass only task-aligned information during sequential training. Through adversarial learning, the shared expert acquires generalized representations that mimic the discriminator, while dedicated experts retain task-specific details, balancing knowledge retention and cross-task generalization and thereby supporting self-evolution.Extensive experiments on the public MTL5 benchmark and an industrial Tencent3 benchmark validate the effectiveness of MoE-CL for continual instruction tuning. In real-world A/B testing for content compliance review on the Tencent Video platform, MoE-CL reduced manual review costs by 15.3%. These results demonstrate that MoE-CL is practical for large-scale industrial deployment where continual adaptation and stable transfer are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13091v1">Unmasking Hiring Bias: Platform Data Analysis and Controlled Experiments on Bias in Online Freelance Marketplaces via RAG-LLM Generated Contents</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Online freelance marketplaces, a rapidly growing part of the global labor market, are creating a fair environment where professional skills are the main factor for hiring. While these platforms can reduce bias from traditional hiring, the personal information in user profiles raises concerns about ongoing discrimination. Past studies on this topic have mostly used existing data, which makes it hard to control for other factors and clearly see the effect of things like gender or race. To solve these problems, this paper presents a new method that uses Retrieval-Augmented Generation (RAG) with a Large Language Model (LLM) to create realistic, artificial freelancer profiles for controlled experiments. This approach effectively separates individual factors, enabling a clearer statistical analysis of how different variables influence the freelancer project process. In addition to analyzing extracted data with traditional statistical methods for post-project stage analysis, our research utilizes a dataset with highly controlled variables, generated by an RAG-LLM, to conduct a simulated hiring experiment for pre-project stage analysis. The results of our experiments show that, regarding gender, while no significant preference emerged in initial hiring decisions, female freelancers are substantially more likely to receive imperfect ratings post-project stage. Regarding regional bias, a strong and consistent preference favoring US-based freelancers shows that people are more likely to be selected in the simulated experiments, perceived as more leader-like, and receive higher ratings on the live platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10113v3">What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      In this paper, we introduce S-MedQA, an English medical question-answering (QA) dataset for benchmarking large language models (LLMs) in fine-grained clinical specialties. S-MedQA has over 20k examples, covers 15 medical specialties, and QA pairs can have multiple specialty annotations (e.g., when a question is cross-disciplinary), constructed with both machine and expert verification to maximize data availability. We use S-MedQA to investigate the role of clinical specialty data in the knowledge-intensive scenario of medical QA. Our results show that 1) training on data from a clinical specialty does not necessarily lead to best performance on that specialty, and 2) regardless of the specialty the LLM was fine-tuned on, token probabilities of clinically relevant terms increase consistently across all specialties. Thus, we hypothesize improvement gains are derived mostly from domain shifting (e.g., general to medical) rather than specialty-specific knowledge injection, and suggest rethinking the role of fine-tuning data in the medical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02340v2">Can Prompts Rewind Time for LLMs? Evaluating the Effectiveness of Prompted Knowledge Cutoffs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Published at EMNLP 2025; Code and data available at https://github.com/gxx27/time_unlearn
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for temporal prediction, but their reliance on pretraining data raises contamination concerns, as accurate predictions on pre-cutoff test data may reflect memorization rather than reasoning, leading to an overestimation of their generalization capability. With the recent emergence of prompting-based unlearning techniques, a natural question arises: Can LLMs be prompted to simulate an earlier knowledge cutoff? In this work, we investigate the capability of prompting to simulate earlier knowledge cutoff in LLMs. We construct three evaluation datasets to assess the extent to which LLMs can forget (1) direct factual knowledge, (2) semantic shifts, and (3) causally related knowledge. Results demonstrate that while prompt-based simulated knowledge cutoffs show effectiveness when directly queried with the information after that date, they struggle to induce forgetting when the forgotten content is not directly asked but causally related to the query. These findings highlight the need for more rigorous evaluation settings when applying LLMs for temporal prediction tasks. The full dataset and evaluation code are available at https://github.com/gxx27/time_unlearn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14162v1">FinAI Data Assistant: LLM-based Financial Database Query Processing with the OpenAI Function Calling API</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 4 pages, 2 figures, accepted at CIKM 2025 FinAI Workshop
    </div>
    <details class="paper-abstract">
      We present FinAI Data Assistant, a practical approach for natural-language querying over financial databases that combines large language models (LLMs) with the OpenAI Function Calling API. Rather than synthesizing complete SQL via text-to-SQL, our system routes user requests to a small library of vetted, parameterized queries, trading generative flexibility for reliability, low latency, and cost efficiency. We empirically study three questions: (RQ1) whether LLMs alone can reliably recall or extrapolate time-dependent financial data without external retrieval; (RQ2) how well LLMs map company names to stock ticker symbols; and (RQ3) whether function calling outperforms text-to-SQL for end-to-end database query processing. Across controlled experiments on prices and fundamentals, LLM-only predictions exhibit non-negligible error and show look-ahead bias primarily for stock prices relative to model knowledge cutoffs. Ticker-mapping accuracy is near-perfect for NASDAQ-100 constituents and high for S\&P~500 firms. Finally, FinAI Data Assistant achieves lower latency and cost and higher reliability than a text-to-SQL baseline on our task suite. We discuss design trade-offs, limitations, and avenues for deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12491v2">Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v3">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Accepted to EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches on graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23410v2">PATCH: Learnable Tile-level Hybrid Sparsity for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18x-1.38x end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14115v1">David vs. Goliath: A comparative study of different-sized LLMs for code generation in the domain of automotive scenario generation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Scenario simulation is central to testing autonomous driving systems. Scenic, a domain-specific language (DSL) for CARLA, enables precise and reproducible scenarios, but NL-to-Scenic generation with large language models (LLMs) suffers from scarce data, limited reproducibility, and inconsistent metrics. We introduce NL2Scenic, an open dataset and framework with 146 NL/Scenic pairs, a difficulty-stratified 30-case test split, an Example Retriever, and 14 prompting variants (ZS, FS, CoT, SP, MoT). We evaluate 13 models: four proprietary (GPT-4o, GPT-5, Claude-Sonnet-4, Gemini-2.5-pro) and nine open-source code models (Qwen2.5Coder 0.5B-32B; CodeLlama 7B/13B/34B), using text metrics (BLEU, ChrF, EDIT-SIM, CrystalBLEU) and execution metrics (compilation and generation), and compare them with an expert study (n=11). EDIT-SIM correlates best with human judgments; we also propose EDIT-COMP (F1 of EDIT-SIM and compilation) as a robust dataset-level proxy that improves ranking fidelity. GPT-4o performs best overall, while Qwen2.5Coder-14B reaches about 88 percent of its expert score on local hardware. Retrieval-augmented prompting, Few-Shot with Example Retriever (FSER), consistently boosts smaller models, and scaling shows diminishing returns beyond mid-size, with Qwen2.5Coder outperforming CodeLlama at comparable scales. NL2Scenic and EDIT-COMP offer a standardized, reproducible basis for evaluating Scenic code generation and indicate that mid-size open-source models are practical, cost-effective options for autonomous-driving scenario programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13593v3">Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      We introduce time-to-unsafe-sampling, a novel safety measure for generative models, defined as the number of generations required by a large language model (LLM) to trigger an unsafe (e.g., toxic) response. While providing a new dimension for prompt-adaptive safety evaluation, quantifying time-to-unsafe-sampling is challenging: unsafe outputs are often rare in well-aligned models and thus may not be observed under any feasible sampling budget. To address this challenge, we frame this estimation problem as one of survival analysis. We build on recent developments in conformal prediction and propose a novel calibration technique to construct a lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt with rigorous coverage guarantees. Our key technical innovation is an optimized sampling-budget allocation scheme that improves sample efficiency while maintaining distribution-free guarantees. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04101v2">LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Conference paper, 6 pages, 4 figures, 1 table
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is expected to be an integral part of next-generation AI-native 6G networks. With the prevalence of AI, researchers have identified numerous use cases of AI in network security. However, there are very few studies that analyze the suitability of Large Language Models (LLMs) in network security. To fill this gap, we examine the suitability of LLMs in network security, particularly with the case study of STRIDE threat modeling. We utilize four prompting techniques with five LLMs to perform STRIDE classification of 5G threats. From our evaluation results, we point out key findings and detailed insights along with the explanation of the possible underlying factors influencing the behavior of LLMs in the modeling of certain threats. The numerical results and the insights support the necessity for adjusting and fine-tuning LLMs for network security use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09674v5">Probabilistic Reasoning with LLMs for k-anonymity Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 10 pages, Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14036v1">One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Fixing bugs in large programs is a challenging task that demands substantial time and effort. Once a bug is found, it is reported to the project maintainers, who work with the reporter to fix it and eventually close the issue. However, across the program, there are often similar code segments, which may also contain the bug, but were missed during discovery. Finding and fixing each recurring bug instance individually is labor intensive. Even more concerning, bug reports can inadvertently widen the attack surface as they provide attackers with an exploitable pattern that may be unresolved in other parts of the program. In this paper, we explore these Recurring Pattern Bugs (RPBs) that appear repeatedly across various code segments of a program or even in different programs, stemming from a same root cause, but are unresolved. Our investigation reveals that RPBs are widespread and can significantly compromise the security of software programs. This paper introduces BugStone, a program analysis system empowered by LLVM and a Large Language Model (LLM). The key observation is that many RPBs have one patched instance, which can be leveraged to identify a consistent error pattern, such as a specific API misuse. By examining the entire program for this pattern, it is possible to identify similar sections of code that may be vulnerable. Starting with 135 unique RPBs, BugStone identified more than 22K new potential issues in the Linux kernel. Manual analysis of 400 of these findings confirmed that 246 were valid. We also created a dataset from over 1.9K security bugs reported by 23 recent top-tier conference works. We manually annotate the dataset, identify 80 recurring patterns and 850 corresponding fixes. Even with a cost-efficient model choice, BugStone achieved 92.2% precision and 79.1% pairwise accuracy on the dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12349v5">SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 48 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Reasoning and strategic behavior in social interactions is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present Strategic Planning, Interaction, and Negotiation (SPIN-Bench), a new multi-domain evaluation designed to measure the intelligence of strategic planning and social reasoning. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also conceptual inference of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle basic fact retrieval and short-range planning reasonably well, they encounter significant performance bottlenecks in tasks requiring deep multi-hop reasoning over large state spaces and socially adept coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. Project Website: https://spinbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14030v1">Think Globally, Group Locally: Evaluating LLMs Using Multi-Lingual Word Grouping Games</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 EMNLP Main 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can exhibit biases in reasoning capabilities due to linguistic modality, performing better on tasks in one language versus another, even with similar content. Most previous works evaluate this through reasoning tasks where reliance on strategies or knowledge can ensure success, such as in commonsense or math tasks. However, abstract reasoning is vital to reasoning for everyday life, where people apply "out-of-the-box thinking" to identify and use patterns for solutions, without a reliance on formulaic approaches. Comparatively, little work has evaluated linguistic biases in this task type. In this paper, we propose a task inspired by the New York Times Connections: GlobalGroup, that evaluates models in an abstract reasoning task across several languages. We constructed a game benchmark with five linguistic backgrounds -- English, Spanish, Chinese, Hindi, and Arabic -- in both the native language and an English translation for comparison. We also proposed game difficulty measurements to evaluate models on games with similar difficulty, enabling a more controlled comparison, which is particularly important in reasoning evaluations. Through experimentation, we find English modalities largely lead to better performance in this abstract reasoning task, and performance disparities between open- and closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14024v1">Efficiently Executing High-throughput Lightweight LLM Inference Applications on Heterogeneous Opportunistic GPU Clusters with Pervasive Context Management</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      The rise of Generative AI introduces a new class of HPC workloads that integrates lightweight LLMs with traditional high-throughput applications to accelerate scientific discovery. The current design of HPC clusters is inadequate to support this new class however, either incurring long wait times on static batch queues or repeatedly paying expensive LLM startup costs upon resource preemption. To circumvent both the long queues and high startup costs, we propose to "decouple" the LLM initialization context from the actual LLM inferences, and retain the context in GPUs until it is no longer needed, a technique we term "Pervasive Context Management". We transform a fact verification application to enable this technique, allowing it to reduce its execution time by 72.1% (from 3 hours to 48 minutes) using the same amount of GPUs, and scale opportunistically on 32.8% of all GPUs in the cluster and further reduce the execution time to 13 minutes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14008v1">Stop Reducing Responsibility in LLM-Powered Multi-Agent Systems to Local Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      LLM-powered Multi-Agent Systems (LLM-MAS) unlock new potentials in distributed reasoning, collaboration, and task generalization but also introduce additional risks due to unguaranteed agreement, cascading uncertainty, and adversarial vulnerabilities. We argue that ensuring responsible behavior in such systems requires a paradigm shift: from local, superficial agent-level alignment to global, systemic agreement. We conceptualize responsibility not as a static constraint but as a lifecycle-wide property encompassing agreement, uncertainty, and security, each requiring the complementary integration of subjective human-centered values and objective verifiability. Furthermore, a dual-perspective governance framework that combines interdisciplinary design with human-AI collaborative oversight is essential for tracing and ensuring responsibility throughout the lifecycle of LLM-MAS. Our position views LLM-MAS not as loose collections of agents, but as unified, dynamic socio-technical systems that demand principled mechanisms to support each dimension of responsibility and enable ethically aligned, verifiably coherent, and resilient behavior for sustained, system-wide agreement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14005v1">PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 The code is available at https://github.com/weizou52/PIShield
    </div>
    <details class="paper-abstract">
      LLM-integrated applications are vulnerable to prompt injection attacks, where an attacker contaminates the input to inject malicious prompts, causing the LLM to follow the attacker's intent instead of the original user's. Existing prompt injection detection methods often have sub-optimal performance and/or high computational overhead. In this work, we propose PIShield, a detection method that is both effective and efficient. Our key observation is that the internal representation of the final token in a prompt-extracted from a specific layer of the LLM, which we term the injection-critical layer-captures distinguishing features between clean and contaminated prompts. Leveraging this insight, we train a simple linear classifier on these internal representations using a labeled set of clean and contaminated prompts. We compare PIShield against 11 baselines across 5 diverse benchmark datasets and 8 prompt injection attacks. The results demonstrate that PIShield is both highly effective and efficient, substantially outperforming existing methods. Additionally, we show that PIShield resists strong adaptive attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13982v1">Static Sandboxes Are Inadequate: Modeling Societal Complexity Requires Open-Ended Co-Evolution in LLM-Based Multi-Agent Simulations</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      What if artificial agents could not just communicate, but also evolve, adapt, and reshape their worlds in ways we cannot fully predict? With llm now powering multi-agent systems and social simulations, we are witnessing new possibilities for modeling open-ended, ever-changing environments. Yet, most current simulations remain constrained within static sandboxes, characterized by predefined tasks, limited dynamics, and rigid evaluation criteria. These limitations prevent them from capturing the complexity of real-world societies. In this paper, we argue that static, task-specific benchmarks are fundamentally inadequate and must be rethought. We critically review emerging architectures that blend llm with multi-agent dynamics, highlight key hurdles such as balancing stability and diversity, evaluating unexpected behaviors, and scaling to greater complexity, and introduce a fresh taxonomy for this rapidly evolving field. Finally, we present a research roadmap centered on open-endedness, continuous co-evolution, and the development of resilient, socially aligned AI ecosystems. \textbf{We call on the community to move beyond static paradigms and help shape the next generation of adaptive, socially-aware multi-agent simulations.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13940v1">Less is More: Improving LLM Reasoning with Minimal Test-Time Intervention</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Code: https://github.com/EnVision-Research/MTI
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has focused on test-time scaling to improve reasoning via increased inference computation, but often at the cost of efficiency. We revisit test-time behavior and uncover a simple yet underexplored phenomenon: reasoning uncertainty is highly localized-only a small subset of high-entropy tokens dominantly affects output correctness. Motivated by this, we propose Minimal Test-Time Intervention (MTI), a training-free framework that enhances reasoning accuracy and stability with minimal overhead. MTI includes: (i) Selective CFG intervention, applying classifier-free guidance only at uncertain positions; and (ii) Lightweight negative-prompt guidance, reusing the main model's KV cache to approximate unconditional decoding efficiently. MTI yields consistent gains across general, coding, and STEM tasks-e.g., +1.35% average improvement on eight benchmarks for Qwen3-8B-Base and +5% on AIME2024 using Qwen3-32B-Reasoning-while remaining highly efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13931v1">Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions</a></div>
    <div class="paper-meta">
      📅 2025-10-15
      | 💬 Preprint of a paper accepted as a poster at the NeurIPS 2025 Workshop on Generative AI for Health (GenAI4Health). The final camera-ready workshop version may differ. Licensed under CC BY 4.0
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in biomedical domains, yet their reliability in drug-safety prediction remains underexplored. In this work, we investigate whether LLMs incorporate socio-demographic information into adverse event (AE) predictions, despite such attributes being clinically irrelevant. Using structured data from the United States Food and Drug Administration Adverse Event Reporting System (FAERS) and a persona-based evaluation framework, we assess two state-of-the-art models, ChatGPT-4o and Bio-Medical-Llama-3.8B, across diverse personas defined by education, marital status, employment, insurance, language, housing stability, and religion. We further evaluate performance across three user roles (general practitioner, specialist, patient) to reflect real-world deployment scenarios where commercial systems often differentiate access by user type. Our results reveal systematic disparities in AE prediction accuracy. Disadvantaged groups (e.g., low education, unstable housing) were frequently assigned higher predicted AE likelihoods than more privileged groups (e.g., postgraduate-educated, privately insured). Beyond outcome disparities, we identify two distinct modes of bias: explicit bias, where incorrect predictions directly reference persona attributes in reasoning traces, and implicit bias, where predictions are inconsistent, yet personas are not explicitly mentioned. These findings expose critical risks in applying LLMs to pharmacovigilance and highlight the urgent need for fairness-aware evaluation protocols and mitigation strategies before clinical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13928v1">LLMs Can Get "Brain Rot"!</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      We propose and test the LLM Brain Rot Hypothesis: continual exposure to junk web text induces lasting cognitive decline in large language models (LLMs). To causally isolate data quality, we run controlled experiments on real Twitter/X corpora, constructing junk and reversely controlled datasets via two orthogonal operationalizations: M1 (engagement degree) and M2 (semantic quality), with matched token scale and training operations across conditions. Contrary to the control group, continual pre-training of 4 LLMs on the junk dataset causes non-trivial declines (Hedges' $g>0.3$) on reasoning, long-context understanding, safety, and inflating "dark traits" (e.g., psychopathy, narcissism). The gradual mixtures of junk and control datasets also yield dose-response cognition decay: for example, under M1, ARC-Challenge with Chain Of Thoughts drops $74.9 \rightarrow 57.2$ and RULER-CWE $84.4 \rightarrow 52.3$ as junk ratio rises from $0\%$ to $100\%$. Error forensics reveal several key insights. First, we identify thought-skipping as the primary lesion: models increasingly truncate or skip reasoning chains, explaining most of the error growth. Second, partial but incomplete healing is observed: scaling instruction tuning and clean data pre-training improve the declined cognition yet cannot restore baseline capability, suggesting persistent representational drift rather than format mismatch. Finally, we discover that the popularity, a non-semantic metric, of a tweet is a better indicator of the Brain Rot effect than the length in M1. Together, the results provide significant, multi-perspective evidence that data quality is a causal driver of LLM capability decay, reframing curation for continual pretraining as a \textit{training-time safety} problem and motivating routine "cognitive health checks" for deployed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13926v1">BioMedSearch: A Multi-Source Biomedical Retrieval Framework Based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-15
    </div>
    <details class="paper-abstract">
      Biomedical queries often rely on a deep understanding of specialized knowledge such as gene regulatory mechanisms and pathological processes of diseases. They require detailed analysis of complex physiological processes and effective integration of information from multiple data sources to support accurate retrieval and reasoning. Although large language models (LLMs) perform well in general reasoning tasks, their generated biomedical content often lacks scientific rigor due to the inability to access authoritative biomedical databases and frequently fabricates protein functions, interactions, and structural details that deviate from authentic information. Therefore, we present BioMedSearch, a multi-source biomedical information retrieval framework based on LLMs. The method integrates literature retrieval, protein database and web search access to support accurate and efficient handling of complex biomedical queries. Through sub-queries decomposition, keywords extraction, task graph construction, and multi-source information filtering, BioMedSearch generates high-quality question-answering results. To evaluate the accuracy of question answering, we constructed a multi-level dataset, BioMedMCQs, consisting of 3,000 questions. The dataset covers three levels of reasoning: mechanistic identification, non-adjacent semantic integration, and temporal causal reasoning, and is used to assess the performance of BioMedSearch and other methods on complex QA tasks. Experimental results demonstrate that BioMedSearch consistently improves accuracy over all baseline models across all levels. Specifically, at Level 1, the average accuracy increases from 59.1% to 91.9%; at Level 2, it rises from 47.0% to 81.0%; and at the most challenging Level 3, the average accuracy improves from 36.3% to 73.4%. The code and BioMedMCQs are available at: https://github.com/CyL-ucas/BioMed_Search
    </details>
</div>
