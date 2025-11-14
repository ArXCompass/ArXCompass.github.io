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
- [Part 10](papers_10.md)
- Part 11
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01908v2">UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at https://github.com/AI-secure/UDora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05887v1">Explainability in Context: A Multilevel Framework Aligning AI Explanations with Stakeholder with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 22 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The growing application of artificial intelligence in sensitive domains has intensified the demand for systems that are not only accurate but also explainable and trustworthy. Although explainable AI (XAI) methods have proliferated, many do not consider the diverse audiences that interact with AI systems: from developers and domain experts to end-users and society. This paper addresses how trust in AI is influenced by the design and delivery of explanations and proposes a multilevel framework that aligns explanations with the epistemic, contextual, and ethical expectations of different stakeholders. The framework consists of three layers: algorithmic and domain-based, human-centered, and social explainability. We highlight the emerging role of Large Language Models (LLMs) in enhancing the social layer by generating accessible, natural language explanations. Through illustrative case studies, we demonstrate how this approach facilitates technical fidelity, user engagement, and societal accountability, reframing XAI as a dynamic, trust-building process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05853v1">Training-Free Query Optimization via LLM-Based Plan Similarity</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) embeddings offer a promising new avenue for database query optimization. In this paper, we explore how pre-trained execution plan embeddings can guide SQL query execution without the need for additional model training. We introduce LLM-PM (LLM-based Plan Mapping), a framework that embeds the default execution plan of a query, finds its k nearest neighbors among previously executed plans, and recommends database hintsets based on neighborhood voting. A lightweight consistency check validates the selected hint, while a fallback mechanism searches the full hint space when needed. Evaluated on the JOB-CEB benchmark using OpenGauss, LLM-PM achieves an average speed-up of 21% query latency reduction. This work highlights the potential of LLM-powered embeddings to deliver practical improvements in query performance and opens new directions for training-free, embedding-based optimizer guidance systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04768v2">DiMA: An LLM-Powered Ride-Hailing Assistant at DiDi</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 KDD 2025
    </div>
    <details class="paper-abstract">
      On-demand ride-hailing services like DiDi, Uber, and Lyft have transformed urban transportation, offering unmatched convenience and flexibility. In this paper, we introduce DiMA, an LLM-powered ride-hailing assistant deployed in DiDi Chuxing. Its goal is to provide seamless ride-hailing services and beyond through a natural and efficient conversational interface under dynamic and complex spatiotemporal urban contexts. To achieve this, we propose a spatiotemporal-aware order planning module that leverages external tools for precise spatiotemporal reasoning and progressive order planning. Additionally, we develop a cost-effective dialogue system that integrates multi-type dialog repliers with cost-aware LLM configurations to handle diverse conversation goals and trade-off response quality and latency. Furthermore, we introduce a continual fine-tuning scheme that utilizes real-world interactions and simulated dialogues to align the assistant's behavior with human preferred decision-making processes. Since its deployment in the DiDi application, DiMA has demonstrated exceptional performance, achieving 93% accuracy in order planning and 92% in response generation during real-world interactions. Offline experiments further validate DiMA capabilities, showing improvements of up to 70.23% in order planning and 321.27% in response generation compared to three state-of-the-art agent frameworks, while reducing latency by $0.72\times$ to $5.47\times$. These results establish DiMA as an effective, efficient, and intelligent mobile assistant for ride-hailing services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02107v3">TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Code available at: https://github.com/apple/ml-tic-lm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained on historical web data inevitably become outdated. We investigate evaluation strategies and update methods for LLMs as new data becomes available. We introduce a web-scale dataset for time-continual pretraining of LLMs derived from 114 dumps of Common Crawl (CC) - orders of magnitude larger than previous continual language modeling benchmarks. We also design time-stratified evaluations across both general CC data and specific domains (Wikipedia, StackExchange, and code documentation) to assess how well various continual learning methods adapt to new data while retaining past knowledge. Our findings demonstrate that, on general CC data, autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6x). However, the optimal balance between incorporating new data and replaying old data differs as replay is crucial to avoid forgetting on generic web data but less so on specific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01484v2">LLM in the Loop: Creating the ParaDeHate Dataset for Hate Speech Detoxification</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Detoxification, the task of rewriting harmful language into non-toxic text, has become increasingly important amid the growing prevalence of toxic content online. However, high-quality parallel datasets for detoxification, especially for hate speech, remain scarce due to the cost and sensitivity of human annotation. In this paper, we propose a novel LLM-in-the-loop pipeline leveraging GPT-4o-mini for automated detoxification. We first replicate the ParaDetox pipeline by replacing human annotators with an LLM and show that the LLM performs comparably to human annotation. Building on this, we construct ParaDeHate, a large-scale parallel dataset specifically for hatespeech detoxification. We release ParaDeHate as a benchmark of over 8K hate/non-hate text pairs and evaluate a wide range of baseline methods. Experimental results show that models such as BART, fine-tuned on ParaDeHate, achieve better performance in style accuracy, content preservation, and fluency, demonstrating the effectiveness of LLM-generated detoxification text as a scalable alternative to human annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18223v2">IDA-Bench: Evaluating LLMs on Interactive Guided Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise as data analysis agents, but existing benchmarks overlook the iterative nature of the field, where experts' decisions evolve with deeper insights of the dataset. To address this, we introduce IDA-Bench, a novel benchmark evaluating LLM agents in multi-round interactive scenarios. Derived from complex Kaggle notebooks, tasks are presented as sequential natural language instructions by an LLM-simulated user. Agent performance is judged by comparing its final numerical output to the human-derived baseline. Initial results show that even state-of-the-art coding agents (like Claude-3.7-thinking) succeed on < 50% of the tasks, highlighting limitations not evident in single-turn tests. This work underscores the need to improve LLMs' multi-round capabilities for building more reliable data analysis agents, highlighting the necessity of achieving a balance between instruction following and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24757v2">LGAR: Zero-Shot LLM-Guided Neural Ranking for Abstract Screening in Systematic Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted to ACL Findings 2025
    </div>
    <details class="paper-abstract">
      The scientific literature is growing rapidly, making it hard to keep track of the state-of-the-art. Systematic literature reviews (SLRs) aim to identify and evaluate all relevant papers on a topic. After retrieving a set of candidate papers, the abstract screening phase determines initial relevance. To date, abstract screening methods using large language models (LLMs) focus on binary classification settings; existing question answering (QA) based ranking approaches suffer from error propagation. LLMs offer a unique opportunity to evaluate the SLR's inclusion and exclusion criteria, yet, existing benchmarks do not provide them exhaustively. We manually extract these criteria as well as research questions for 57 SLRs, mostly in the medical domain, enabling principled comparisons between approaches. Moreover, we propose LGAR, a zero-shot LLM Guided Abstract Ranker composed of an LLM based graded relevance scorer and a dense re-ranker. Our extensive experiments show that LGAR outperforms existing QA-based methods by 5-10 pp. in mean average precision. Our code and data is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05766v1">BioMol-MQA: A Multi-Modal Question Answering Dataset For LLM Reasoning Over Bio-Molecular Interactions</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Retrieval augmented generation (RAG) has shown great power in improving Large Language Models (LLMs). However, most existing RAG-based LLMs are dedicated to retrieving single modality information, mainly text; while for many real-world problems, such as healthcare, information relevant to queries can manifest in various modalities such as knowledge graph, text (clinical notes), and complex molecular structure. Thus, being able to retrieve relevant multi-modality domain-specific information, and reason and synthesize diverse knowledge to generate an accurate response is important. To address the gap, we present BioMol-MQA, a new question-answering (QA) dataset on polypharmacy, which is composed of two parts (i) a multimodal knowledge graph (KG) with text and molecular structure for information retrieval; and (ii) challenging questions that designed to test LLM capabilities in retrieving and reasoning over multimodal KG to answer questions. Our benchmarks indicate that existing LLMs struggle to answer these questions and do well only when given the necessary background data, signaling the necessity for strong RAG frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12882v3">ProSec: Fortifying Code LLMs with Proactive Security Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 The first two authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      While recent code-specific large language models (LLMs) have greatly enhanced their code generation capabilities, the safety of these models remains under-explored, posing potential risks as insecure code generated by these models may introduce vulnerabilities into real-world systems. Existing methods collect security-focused datasets from real-world vulnerabilities for instruction tuning in order to mitigate such issues. However, they are largely constrained by the data sparsity of vulnerable code, and have limited applicability in the multi-stage post-training workflows of modern LLMs. In this paper, we propose ProSec, a novel proactive security alignment approach designed to align code LLMs with secure coding practices. ProSec systematically exposes the vulnerabilities in a code LLM by synthesizing vulnerability-inducing coding scenarios from Common Weakness Enumerations (CWEs) and generates fixes to vulnerable code snippets, allowing the model to learn secure practices through preference learning objectives. The scenarios synthesized by ProSec trigger 25x more vulnerable code than a normal instruction-tuning dataset, resulting in a security-focused alignment dataset 7x larger than the previous work. Experiments show that models trained with ProSec are 25.2% to 35.4% more secure compared to previous work without degrading models' utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05748v1">Efficient Online RFT with Plug-and-Play LLM Judges: Unlocking State-of-the-Art Performance</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Reward-model training is the cost bottleneck in modern Reinforcement Learning Human Feedback (RLHF) pipelines, often requiring tens of billions of parameters and an offline preference-tuning phase. In the proposed method, a frozen, instruction-tuned 7B LLM is augmented with only a one line JSON rubric and a rank-16 LoRA adapter (affecting just 0.8% of the model's parameters), enabling it to serve as a complete substitute for the previously used heavyweight evaluation models. The plug-and-play judge achieves 96.2% accuracy on RewardBench, outperforming specialized reward networks ranging from 27B to 70B parameters. Additionally, it allows a 7B actor to outperform the top 70B DPO baseline, which scores 61.8%, by achieving 92% exact match accuracy on GSM-8K utilizing online PPO. Thorough ablations indicate that (i) six in context demonstrations deliver the majority of the zero-to-few-shot improvements (+2pp), and (ii) the LoRA effectively addresses the remaining disparity, particularly in the safety and adversarial Chat-Hard segments. The proposed model introduces HH-Rationales, a subset of 10,000 pairs from Anthropic HH-RLHF, to examine interpretability, accompanied by human generated justifications. GPT-4 scoring indicates that our LoRA judge attains approximately = 9/10 in similarity to human explanations, while zero-shot judges score around =5/10. These results indicate that the combination of prompt engineering and tiny LoRA produces a cost effective, transparent, and easily adjustable reward function, removing the offline phase while achieving new state-of-the-art outcomes for both static evaluation and online RLHF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05746v1">LLM-Symbolic Integration for Robust Temporal Tabular Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted to ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Temporal tabular question answering presents a significant challenge for Large Language Models (LLMs), requiring robust reasoning over structured data, which is a task where traditional prompting methods often fall short. These methods face challenges such as memorization, sensitivity to table size, and reduced performance on complex queries. To overcome these limitations, we introduce TempTabQA-C, a synthetic dataset designed for systematic and controlled evaluations, alongside a symbolic intermediate representation that transforms tables into database schemas. This structured approach allows LLMs to generate and execute SQL queries, enhancing generalization and mitigating biases. By incorporating adaptive few-shot prompting with contextually tailored examples, our method achieves superior robustness, scalability, and performance. Experimental results consistently highlight improvements across key challenges, setting a new benchmark for robust temporal reasoning with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05739v1">To Protect the LLM Agent Against the Prompt Injection Attack with Polymorphic Prompt</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 To appear in the Industry Track of the 55th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN 2025)
    </div>
    <details class="paper-abstract">
      LLM agents are widely used as agents for customer support, content generation, and code assistance. However, they are vulnerable to prompt injection attacks, where adversarial inputs manipulate the model's behavior. Traditional defenses like input sanitization, guard models, and guardrails are either cumbersome or ineffective. In this paper, we propose a novel, lightweight defense mechanism called Polymorphic Prompt Assembling (PPA), which protects against prompt injection with near-zero overhead. The approach is based on the insight that prompt injection requires guessing and breaking the structure of the system prompt. By dynamically varying the structure of system prompts, PPA prevents attackers from predicting the prompt structure, thereby enhancing security without compromising performance. We conducted experiments to evaluate the effectiveness of PPA against existing attacks and compared it with other defense methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05735v1">Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13816v2">Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 ACL 2025 main; camera ready
    </div>
    <details class="paper-abstract">
      While understanding the knowledge boundaries of LLMs is crucial to prevent hallucination, research on the knowledge boundaries of LLMs has predominantly focused on English. In this work, we present the first study to analyze how LLMs recognize knowledge boundaries across different languages by probing their internal representations when processing known and unknown questions in multiple languages. Our empirical studies reveal three key findings: 1) LLMs' perceptions of knowledge boundaries are encoded in the middle to middle-upper layers across different languages. 2) Language differences in knowledge boundary perception follow a linear structure, which motivates our proposal of a training-free alignment method that effectively transfers knowledge boundary perception ability across languages, thereby helping reduce hallucination risk in low-resource languages; 3) Fine-tuning on bilingual question pair translation further enhances LLMs' recognition of knowledge boundaries across languages. Given the absence of standard testbeds for cross-lingual knowledge boundary analysis, we construct a multilingual evaluation suite comprising three representative types of knowledge boundary data. Our code and datasets are publicly available at https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22318v3">Online Detection of LLM-Generated Texts via Sequential Hypothesis Testing by Betting</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Developing algorithms to differentiate between machine-generated texts and human-written texts has garnered substantial attention in recent years. Existing methods in this direction typically concern an offline setting where a dataset containing a mix of real and machine-generated texts is given upfront, and the task is to determine whether each sample in the dataset is from a large language model (LLM) or a human. However, in many practical scenarios, sources such as news websites, social media accounts, and online forums publish content in a streaming fashion. Therefore, in this online scenario, how to quickly and accurately determine whether the source is an LLM with strong statistical guarantees is crucial for these media or platforms to function effectively and prevent the spread of misinformation and other potential misuse of LLMs. To tackle the problem of online detection, we develop an algorithm based on the techniques of sequential hypothesis testing by betting that not only builds upon and complements existing offline detection techniques but also enjoys statistical guarantees, which include a controlled false positive rate and the expected time to correctly identify a source as an LLM. Experiments were conducted to demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00845v2">Generalizable LLM Learning of Graph Synthetic Data with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 9 pages, 3 figures, 3 tables. Experimental code and results are publicly available at https://anonymous.4open.science/r/Graph_RL-BF08/readme.md
    </div>
    <details class="paper-abstract">
      Previous research has sought to enhance the graph reasoning capabilities of LLMs by supervised fine-tuning on synthetic graph data. While these led to specialized LLMs better at solving graph algorithm problems, we don't need LLMs for shortest path: we need generalization from synthetic graph data to real-world tasks with implicit graph structures. In this work, we propose to unlock generalizable learning of graph synthetic data with reinforcement learning. We first design solution-based and process-based rewards for synthetic graph problems: instead of rigid memorizing response patterns in direct fine-tuning, we posit that RL would help LLMs grasp the essentials underlying graph reasoning and alleviate overfitting. We employ RL algorithms such as GRPO and DPO, aligning both off-the-shelf LLMs and LLMs fine-tuned on synthetic graph data. We then compare them against existing settings on both in-domain synthetic tasks and out-of-domain real-world tasks with implicit graph structures such as multi-hop QA, structured planning, and more. Extensive experiments demonstrate that our RL recipe leads to statistically significant improvement on 5 datasets, with an average gain of 12.9\% over baseline settings. Further analysis reveals that process-based rewards consistently outperform solution-based rewards, mixing synthetic and real-world task data yields potential gains, while compositionality and explainable intermediate steps remains a critical challenge even after RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05692v1">SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      The code generation capabilities of large language models(LLMs) have emerged as a critical dimension in evaluating their overall performance. However, prior research has largely overlooked the security risks inherent in the generated code. In this work, we introduce \benchmark, a benchmark specifically designed to assess the security of LLM-generated code. The dataset encompasses a wide range of common software development scenarios and vulnerability types. Building upon this benchmark, we develop an automatic evaluation framework that leverages both static application security testing(SAST) and LLM-based judging to assess the presence of security vulnerabilities in model-generated code. Through the empirical evaluation of state-of-the-art LLMs on \benchmark, we reveal notable deficiencies in their ability to produce vulnerability-free code. Our findings highlight pressing challenges and offer actionable insights for future advancements in the secure code generation performance of LLMs. The data and code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04202v2">TracLLM: A Generic Framework for Attributing Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 To appear in USENIX Security Symposium 2025. The code and data are at: https://github.com/Wang-Yanting/TracLLM
    </div>
    <details class="paper-abstract">
      Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17156v2">LLM-based relevance assessment still can't replace human relevance assessment</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 To appear in "11th International Workshop on Evaluating Information Access (EVIA 2025)"
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) for relevance assessment in information retrieval has gained significant attention, with recent studies suggesting that LLM-based judgments provide comparable evaluations to human judgments. Notably, based on TREC 2024 data, Upadhyay et al. make a bold claim that LLM-based relevance assessments, such as those generated by the UMBRELA system, can fully replace traditional human relevance assessments in TREC-style evaluations. This paper critically examines this claim, highlighting practical and theoretical limitations that undermine the validity of this conclusion. First, we question whether the evidence provided by Upadhyay et al. really supports their claim, particularly if a test collection is used asa benchmark for future improvements. Second, through a submission deliberately intended to do so, we demonstrate the ease with which automatic evaluation metrics can be subverted, showing that systems designed to exploit these evaluations can achieve artificially high scores. Theoretical challenges -- such as the inherent narcissism of LLMs, the risk of overfitting to LLM-based metrics, and the potential degradation of future LLM performance -- must be addressed before LLM-based relevance assessments can be considered a viable replacement for human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08503v3">Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted to ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      In this work, we identify the "2D-Cheating" problem in 3D LLM evaluation, where these tasks might be easily solved by VLMs with rendered images of point clouds, exposing ineffective evaluation of 3D LLMs' unique 3D capabilities. We test VLM performance across multiple 3D LLM benchmarks and, using this as a reference, propose principles for better assessing genuine 3D understanding. We also advocate explicitly separating 3D abilities from 1D or 2D aspects when evaluating 3D LLMs. Code and data are available at https://github.com/LLM-class-group/Revisiting-3D-LLM-Benchmarks
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05166v2">Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to exhibit social, demographic, and gender biases, often as a consequence of the data on which they are trained. In this work, we adopt a mechanistic interpretability approach to analyze how such biases are structurally represented within models such as GPT-2 and Llama2. Focusing on demographic and gender biases, we explore different metrics to identify the internal edges responsible for biased behavior. We then assess the stability, localization, and generalizability of these components across dataset and linguistic variations. Through systematic ablations, we demonstrate that bias-related computations are highly localized, often concentrated in a small subset of layers. Moreover, the identified components change across fine-tuning settings, including those unrelated to bias. Finally, we show that removing these components not only reduces biased outputs but also affects other NLP tasks, such as named entity recognition and linguistic acceptability judgment because of the sharing of important components with these tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05671v1">Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05670v1">Can LLMs Express Personality Across Cultures? Introducing CulturalPersonas for Evaluating Trait Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      As LLMs become central to interactive applications, ranging from tutoring to mental health, the ability to express personality in culturally appropriate ways is increasingly important. While recent works have explored personality evaluation of LLMs, they largely overlook the interplay between culture and personality. To address this, we introduce CulturalPersonas, the first large-scale benchmark with human validation for evaluating LLMs' personality expression in culturally grounded, behaviorally rich contexts. Our dataset spans 3,000 scenario-based questions across six diverse countries, designed to elicit personality through everyday scenarios rooted in local values. We evaluate three LLMs, using both multiple-choice and open-ended response formats. Our results show that CulturalPersonas improves alignment with country-specific human personality distributions (over a 20% reduction in Wasserstein distance across models and countries) and elicits more expressive, culturally coherent outputs compared to existing benchmarks. CulturalPersonas surfaces meaningful modulated trait outputs in response to culturally grounded prompts, offering new directions for aligning LLMs to global norms of behavior. By bridging personality expression and cultural nuance, we envision that CulturalPersonas will pave the way for more socially intelligent and globally adaptive LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11014v2">CoopetitiveV: Leveraging LLM-powered Coopetitive Multi-Agent Prompting for High-quality Verilog Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Recent advances in agentic LLMs have demonstrated great capabilities in Verilog code generation. However, existing approaches either use LLM-assisted single-agent prompting or cooperation-only multi-agent learning, which will lead to: (i) Degeneration issue for single-agent learning: characterized by diminished error detection and correction capabilities; (ii) Error propagation in cooperation-only multi-agent learning: erroneous information from the former agent will be propagated to the latter through prompts, which can make the latter agents generate buggy code. In this paper, we propose an LLM-based coopetitive multi-agent prompting framework, in which the agents cannot collaborate with each other to form the generation pipeline, but also create a healthy competitive mechanism to improve the generating quality. Our experimental results show that the coopetitive multi-agent framework can effectively mitigate the degeneration risk and reduce the error propagation while improving code error correction capabilities, resulting in higher quality Verilog code generation. The effectiveness of our approach is validated through extensive experiments. On VerilogEval Machine and Human dataset, CoopetitiveV+GPT-4 achieves 99.2% and 99.1% pass@10 scores, respectively. While on RTLLM, CoopetitiveV+GPT-4 obtains 100% syntax and 99.9% functionality pass@5 scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18574v2">Autocomp: LLM-Driven Code Optimization for Tensor Accelerators</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Hardware accelerators, especially those designed for tensor processing, have become ubiquitous in today's computing landscape. However, even with significant efforts in building compilers, programming these tensor accelerators remains challenging, leaving much of their potential underutilized. Recently, large language models (LLMs), trained on large amounts of code, have shown significant promise in code generation and optimization tasks, but generating low-resource languages like specialized tensor accelerator code still poses a significant challenge. We tackle this challenge with Autocomp, an approach that empowers accelerator programmers to leverage domain knowledge and hardware feedback to optimize code via an automated LLM-driven search. We accomplish this by: 1) formulating each optimization pass as a structured two-phase prompt, divided into planning and code generation phases, 2) inserting domain knowledge during planning via a concise and adaptable optimization menu, and 3) integrating correctness and performance metrics from hardware as feedback at each search iteration. Across three categories of representative workloads and two different accelerators, we demonstrate that Autocomp-optimized code runs 5.6x (GEMM) and 2.7x (convolution) faster than the vendor-provided library, and outperforms expert-level hand-tuned code by 1.4x (GEMM), 1.1x (convolution), and 1.3x (fine-grained linear algebra). Additionally, we demonstrate that optimization schedules generated from Autocomp can be reused across similar tensor operations, improving speedups by up to 24% under a fixed sample budget.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01245v2">Comprehensive Vulnerability Analysis is Necessary for Trustworthy LLM-MAS</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      This paper argues that a comprehensive vulnerability analysis is essential for building trustworthy Large Language Model-based Multi-Agent Systems (LLM-MAS). These systems, which consist of multiple LLM-powered agents working collaboratively, are increasingly deployed in high-stakes applications but face novel security threats due to their complex structures. While single-agent vulnerabilities are well-studied, LLM-MAS introduces unique attack surfaces through inter-agent communication, trust relationships, and tool integration that remain significantly underexplored. We present a systematic framework for vulnerability analysis of LLM-MAS that unifies diverse research. For each type of vulnerability, we define formal threat models grounded in practical attacker capabilities and illustrate them using real-world LLM-MAS applications. This formulation enables rigorous quantification of vulnerability across different architectures and provides a foundation for designing meaningful evaluation benchmarks. Our analysis reveals that LLM-MAS faces elevated risk due to compositional effects -- vulnerabilities in individual components can cascade through agent communication, creating threat models not present in single-agent systems. We conclude by identifying critical open challenges: (1) developing benchmarks specifically tailored to LLM-MAS vulnerability assessment, (2) considering new potential attacks specific to multi-agent architectures, and (3) implementing trust management systems that can enforce security in LLM-MAS. This research provides essential groundwork for future efforts to enhance LLM-MAS trustworthiness as these systems continue their expansion into critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06579v1">Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Recent progress in Language Models (LMs) has dramatically advanced the field of natural language processing (NLP), excelling at tasks like text generation, summarization, and question answering. However, their inference remains computationally expensive and energy intensive, especially in settings with limited hardware, power, or bandwidth. This makes it difficult to deploy LMs in mobile, edge, or cost sensitive environments. To address these challenges, recent approaches have introduced multi LLM intelligent model selection strategies that dynamically allocate computational resources based on query complexity -- using lightweight models for simpler queries and escalating to larger models only when necessary. This survey explores two complementary strategies for efficient LLM inference: (i) routing, which selects the most suitable model based on the query, and (ii) cascading or hierarchical inference (HI), which escalates queries through a sequence of models until a confident response is found. Both approaches aim to reduce computation by using lightweight models for simpler tasks while offloading only when needed. We provide a comparative analysis of these techniques across key performance metrics, discuss benchmarking efforts, and outline open challenges. Finally, we outline future research directions to enable faster response times, adaptive model selection based on task complexity, and scalable deployment across heterogeneous environments, making LLM based systems more efficient and accessible for real world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14182v3">LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is revolutionizing scientific research, yet its growing integration into laboratory environments presents critical safety challenges. While large language models (LLMs) increasingly assist in tasks ranging from procedural guidance to autonomous experiment orchestration, an "illusion of understanding" may lead researchers to overestimate their reliability. Such overreliance is particularly dangerous in high-stakes laboratory settings, where failures in hazard identification or risk assessment can result in severe accidents. To address these concerns, we propose the Laboratory Safety Benchmark (LabSafety Bench), a comprehensive framework that evaluates large language models and vision language models (VLMs) on their ability to identify potential hazards, assess risks, and predict the consequences of unsafe actions in lab environments. LabSafety Bench comprises 765 multiple-choice questions aligned with US Occupational Safety and Health Administration (OSHA) protocols, along with 404 realistic laboratory scenarios featuring dual evaluation tasks: the Hazards Identification Test and the Consequence Identification Test, with 3128 open-ended questions in total. Evaluations across eight proprietary models, seven open-weight LLMs, and four VLMs reveal that, despite advanced performance on structured assessments, no model achieves the safety threshold required for reliable operation -- none scoring above 70% on the Hazards Identification Test. Moreover, while proprietary models tend to excel in multiple-choice evaluations, their performance in open-ended, real-world scenario responses is comparable to that of open-source models. These findings underscore the urgent need for specialized evaluation frameworks to ensure the safe and responsible deployment of AI in laboratory settings.
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
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.04920v1">Simulating LLM-to-LLM Tutoring for Multilingual Math Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Preprint, in submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated the ability to generate formative feedback and instructional hints in English, making them increasingly relevant for AI-assisted education. However, their ability to provide effective instructional support across different languages, especially for mathematically grounded reasoning tasks, remains largely unexamined. In this work, we present the first large-scale simulation of multilingual tutor-student interactions using LLMs. A stronger model plays the role of the tutor, generating feedback in the form of hints, while a weaker model simulates the student. We explore 352 experimental settings across 11 typologically diverse languages, four state-of-the-art LLMs, and multiple prompting strategies to assess whether language-specific feedback leads to measurable learning gains. Our study examines how student input language, teacher feedback language, model choice, and language resource level jointly influence performance. Results show that multilingual hints can significantly improve learning outcomes, particularly in low-resource languages when feedback is aligned with the student's native language. These findings offer practical insights for developing multilingual, LLM-based educational tools that are both effective and inclusive.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16029v2">FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06877v2">The Synergy of LLMs & RL Unlocks Offline Learning of Generalizable Language-Conditioned Policies with Low-fidelity Data</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 Accepted at International Conference on Machine Learning (ICML) 2025
    </div>
    <details class="paper-abstract">
      Developing autonomous agents capable of performing complex, multi-step decision-making tasks specified in natural language remains a significant challenge, particularly in realistic settings where labeled data is scarce and real-time experimentation is impractical. Existing reinforcement learning (RL) approaches often struggle to generalize to unseen goals and states, limiting their applicability. In this paper, we introduce TEDUO, a novel training pipeline for offline language-conditioned policy learning in symbolic environments. Unlike conventional methods, TEDUO operates on readily available, unlabeled datasets and addresses the challenge of generalization to previously unseen goals and states. Our approach harnesses large language models (LLMs) in a dual capacity: first, as automatization tools augmenting offline datasets with richer annotations, and second, as generalizable instruction-following agents. Empirical results demonstrate that TEDUO achieves data-efficient learning of robust language-conditioned policies, accomplishing tasks beyond the reach of conventional RL frameworks or out-of-the-box LLMs alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05981v1">CrimeMind: Simulating Urban Crime with Multi-Modal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Modeling urban crime is an important yet challenging task that requires understanding the subtle visual, social, and cultural cues embedded in urban environments. Previous work has predominantly focused on rule-based agent-based modeling (ABM) and deep learning methods. ABMs offer interpretability of internal mechanisms but exhibit limited predictive accuracy.In contrast, deep learning methods are often effective in prediction but are less interpretable and require extensive training data. Moreover, both lines of work lack the cognitive flexibility to adapt to changing environments. Leveraging the capabilities of large language models (LLMs), we propose CrimeMind, a novel LLM-driven ABM framework for simulating urban crime within a multi-modal urban context.A key innovation of our design is the integration of the Routine Activity Theory (RAT) into the agentic workflow of CrimeMind, enabling it to process rich multi-modal urban features and reason about criminal behavior.However, RAT requires LLM agents to infer subtle cues in evaluating environmental safety as part of assessing guardianship, which can be challenging for LLMs. To address this, we collect a small-scale human-annotated dataset and align CrimeMind's perception with human judgment via a training-free textual gradient method.Experiments across four major U.S. cities demonstrate that CrimeMind outperforms both traditional ABMs and deep learning baselines in crime hotspot prediction and spatial distribution accuracy, achieving up to a 24% improvement over the strongest baseline.Furthermore, we conduct counterfactual simulations of external incidents and policy interventions and it successfully captures the expected changes in crime patterns, demonstrating its ability to reflect counterfactual scenarios.Overall, CrimeMind enables fine-grained modeling of individual behaviors and facilitates evaluation of real-world interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04381v2">TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 ACL 2025 camera-ready Codes and models are available at https://github.com/d223302/TRACT
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm uses large language models (LLMs) for automated text evaluation, where a numerical assessment is assigned by an LLM to the input text following scoring rubrics. Existing methods for LLM-as-a-judge use cross-entropy (CE) loss for fine-tuning, which neglects the numeric nature of score prediction. Recent work addresses numerical prediction limitations of LLM fine-tuning through regression-aware fine-tuning, which, however, does not consider chain-of-thought (CoT) reasoning for score prediction. In this paper, we introduce TRACT (Two-stage Regression-Aware fine-tuning with CoT), a method combining CoT reasoning with regression-aware training. TRACT consists of two stages: first, seed LLM is fine-tuned to generate CoTs, which serve as supervision for the second stage fine-tuning. The training objective of TRACT combines the CE loss for learning the CoT reasoning capabilities, and the regression-aware loss for the score prediction. Experiments across four LLM-as-a-judge datasets and two LLMs show that TRACT significantly outperforms existing methods. Extensive ablation studies validate the importance of each component in TRACT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02958v2">AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML</a></div>
    <div class="paper-meta">
      📅 2025-06-06
      | 💬 ICML 2025, Project Page: https://deepauto-ai.github.io/automl-agent
    </div>
    <details class="paper-abstract">
      Automated machine learning (AutoML) accelerates AI development by automating tasks in the development pipeline, such as optimal model search and hyperparameter tuning. Existing AutoML systems often require technical expertise to set up complex tools, which is in general time-consuming and requires a large amount of human effort. Therefore, recent works have started exploiting large language models (LLM) to lessen such burden and increase the usability of AutoML frameworks via a natural language interface, allowing non-expert users to build their data-driven solutions. These methods, however, are usually designed only for a particular process in the AI development pipeline and do not efficiently use the inherent capacity of the LLMs. This paper proposes AutoML-Agent, a novel multi-agent framework tailored for full-pipeline AutoML, i.e., from data retrieval to model deployment. AutoML-Agent takes user's task descriptions, facilitates collaboration between specialized LLM agents, and delivers deployment-ready models. Unlike existing work, instead of devising a single plan, we introduce a retrieval-augmented planning strategy to enhance exploration to search for more optimal plans. We also decompose each plan into sub-tasks (e.g., data preprocessing and neural network design) each of which is solved by a specialized agent we build via prompting executing in parallel, making the search process more efficient. Moreover, we propose a multi-stage verification to verify executed results and guide the code generation LLM in implementing successful solutions. Extensive experiments on seven downstream tasks using fourteen datasets show that AutoML-Agent achieves a higher success rate in automating the full AutoML process, yielding systems with good performance throughout the diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05925v1">Small Models, Big Support: A Local LLM Framework for Teacher-Centric Content Creation and Assessment using RAG and CAG</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) are increasingly utilized as student-facing educational aids, their potential to directly support educators, particularly through locally deployable and customizable open-source solutions, remains significantly underexplored. Many existing educational solutions rely on cloud-based infrastructure or proprietary tools, which are costly and may raise privacy concerns. Regulated industries with limited budgets require affordable, self-hosted solutions. We introduce an end-to-end, open-source framework leveraging small (3B-7B parameters), locally deployed LLMs for customized teaching material generation and assessment. Our system uniquely incorporates an interactive loop crucial for effective small-model refinement, and an auxiliary LLM verifier to mitigate jailbreaking risks, enhancing output reliability and safety. Utilizing Retrieval and Context Augmented Generation (RAG/CAG), it produces factually accurate, customized pedagogically-styled content. Deployed on-premises for data privacy and validated through an evaluation pipeline and a college physics pilot, our findings show that carefully engineered small LLM systems can offer robust, affordable, practical, and safe educator support, achieving utility comparable to larger models for targeted tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09429v4">From Intention To Implementation: Automating Biomedical Research via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 To appear in SCIENCE CHINA Information Sciences. If you find our work useful, please cite us as: @article{ BioResearcher, author = "Yi Luo and Linghang Shi and Yihao Li and Aobo Zhuang and Yeyun Gong and Ling Liu and Chen Lin", title = "From Intention To Implementation: Automating Biomedical Research via LLMs", journal = "SCIENCE CHINA Information Sciences", year = "2025" }
    </div>
    <details class="paper-abstract">
      Conventional biomedical research is increasingly labor-intensive due to the exponential growth of scientific literature and datasets. Artificial intelligence (AI), particularly Large Language Models (LLMs), has the potential to revolutionize this process by automating various steps. Still, significant challenges remain, including the need for multidisciplinary expertise, logicality of experimental design, and performance measurements. This paper introduces BioResearcher, the first end-to-end automated system designed to streamline the entire biomedical research process involving dry lab experiments. BioResearcher employs a modular multi-agent architecture, integrating specialized agents for search, literature processing, experimental design, and programming. By decomposing complex tasks into logically related sub-tasks and utilizing a hierarchical learning approach, BioResearcher effectively addresses the challenges of multidisciplinary requirements and logical complexity. Furthermore, BioResearcher incorporates an LLM-based reviewer for in-process quality control and introduces novel evaluation metrics to assess the quality and automation of experimental protocols. BioResearcher successfully achieves an average execution success rate of 63.07% across eight previously unmet research objectives. The generated protocols, on average, outperform typical agent systems by 22.0% on five quality metrics. The system demonstrates significant potential to reduce researchers' workloads and accelerate biomedical discoveries, paving the way for future innovations in automated research systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12872v2">Not All Options Are Created Equal: Textual Option Weighting for Token-Efficient LLM-Based Knowledge Tracing</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as promising tools for knowledge tracing (KT) due to their strong reasoning and generalization abilities. While recent LLM-based KT methods have proposed new prompt formats, they struggle to represent the full interaction histories of example learners within a single prompt during in-context learning (ICL), resulting in limited scalability and high computational cost under token constraints. In this work, we present \textit{LLM-based Option-weighted Knowledge Tracing (LOKT)}, a simple yet effective framework that encodes the interaction histories of example learners in context as \textit{textual categorical option weights (TCOW)}. TCOW are semantic labels (e.g., ``inadequate'') assigned to the options selected by learners when answering questions, enhancing the interpretability of LLMs. Experiments on multiple-choice datasets show that LOKT outperforms existing non-LLM and LLM-based KT models in both cold-start and warm-start settings. Moreover, LOKT enables scalable and cost-efficient inference, achieving strong performance even under strict token constraints. Our code is available at \href{https://anonymous.4open.science/r/LOKT_model-3233}{https://anonymous.4open.science/r/LOKT\_model-3233}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24871v2">MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Project Webpage: https://modomodo-rl.github.io/
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for post-training large language models (LLMs), achieving state-of-the-art performance on tasks with structured, verifiable answers. Applying RLVR to Multimodal LLMs (MLLMs) presents significant opportunities but is complicated by the broader, heterogeneous nature of vision-language tasks that demand nuanced visual, logical, and spatial capabilities. As such, training MLLMs using RLVR on multiple datasets could be beneficial but creates challenges with conflicting objectives from interaction among diverse datasets, highlighting the need for optimal dataset mixture strategies to improve generalization and reasoning. We introduce a systematic post-training framework for Multimodal LLM RLVR, featuring a rigorous data mixture problem formulation and benchmark implementation. Specifically, (1) We developed a multimodal RLVR framework for multi-dataset post-training by curating a dataset that contains different verifiable vision-language problems and enabling multi-domain online RL learning with different verifiable rewards; (2) We proposed a data mixture strategy that learns to predict the RL fine-tuning outcome from the data mixture distribution, and consequently optimizes the best mixture. Comprehensive experiments showcase that multi-domain RLVR training, when combined with mixture prediction strategies, can significantly boost MLLM general reasoning capacities. Our best mixture improves the post-trained model's accuracy on out-of-distribution benchmarks by an average of 5.24% compared to the same model post-trained with uniform data mixture, and by a total of 20.74% compared to the pre-finetuning baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03295v2">Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      We have witnessed that strong LLMs like Qwen-Math, MiMo, and Phi-4 possess immense reasoning potential inherited from the pre-training stage. With reinforcement learning (RL), these models can improve dramatically on reasoning tasks. Recent studies have shown that even RL on a single problem can unleash these models' reasoning capabilities. However, RL is not only expensive but also unstable. Even one-shot RL requires hundreds of GPU hours. This raises a critical question: Is there a more efficient way to unleash the reasoning potential of these powerful base LLMs? In this work, we demonstrate that Critique Fine-Tuning (CFT) on only one problem can effectively unleash the reasoning potential of LLMs. Our method constructs critique data by collecting diverse model-generated solutions to a single problem and using teacher LLMs to provide detailed critiques. We fine-tune Qwen and Llama family models, ranging from 1.5B to 14B parameters, on the CFT data and observe significant performance gains across diverse reasoning tasks. For example, with just 5 GPU hours of training, Qwen-Math-7B-CFT show an average improvement of 15% on six math benchmarks and 16% on three logic reasoning benchmarks. These results are comparable to or even surpass the results from RL with 20x less compute. Ablation studies reveal the robustness of one-shot CFT across different prompt problems. These results highlight one-shot CFT as a simple, general, and compute-efficient approach to unleashing the reasoning capabilities of modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04594v1">Intelligent Channel Allocation for IEEE 802.11be Multi-Link Operation: When MAB Meets LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 This work has been accepted by JSAC 2025
    </div>
    <details class="paper-abstract">
      WiFi networks have achieved remarkable success in enabling seamless communication and data exchange worldwide. The IEEE 802.11be standard, known as WiFi 7, introduces Multi-Link Operation (MLO), a groundbreaking feature that enables devices to establish multiple simultaneous connections across different bands and channels. While MLO promises substantial improvements in network throughput and latency reduction, it presents significant challenges in channel allocation, particularly in dense network environments. Current research has predominantly focused on performance analysis and throughput optimization within static WiFi 7 network configurations. In contrast, this paper addresses the dynamic channel allocation problem in dense WiFi 7 networks with MLO capabilities. We formulate this challenge as a combinatorial optimization problem, leveraging a novel network performance analysis mechanism. Given the inherent lack of prior network information, we model the problem within a Multi-Armed Bandit (MAB) framework to enable online learning of optimal channel allocations. Our proposed Best-Arm Identification-enabled Monte Carlo Tree Search (BAI-MCTS) algorithm includes rigorous theoretical analysis, providing upper bounds for both sample complexity and error probability. To further reduce sample complexity and enhance generalizability across diverse network scenarios, we put forth LLM-BAI-MCTS, an intelligent algorithm for the dynamic channel allocation problem by integrating the Large Language Model (LLM) into the BAI-MCTS algorithm. Numerical results demonstrate that the BAI-MCTS algorithm achieves a convergence rate approximately $50.44\%$ faster than the state-of-the-art algorithms when reaching $98\%$ of the optimal value. Notably, the convergence rate of the LLM-BAI-MCTS algorithm increases by over $63.32\%$ in dense networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11283v3">AdvBDGen: Adversarially Fortified Prompt-Specific Fuzzy Backdoor Generator Against LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Published at the Neurips Safe Generative AI Workshop 2024
    </div>
    <details class="paper-abstract">
      With the growing adoption of reinforcement learning with human feedback (RLHF) for aligning large language models (LLMs), the risk of backdoor installation during alignment has increased, leading to unintended and harmful behaviors. Existing backdoor triggers are typically limited to fixed word patterns, making them detectable during data cleaning and easily removable post-poisoning. In this work, we explore the use of prompt-specific paraphrases as backdoor triggers, enhancing their stealth and resistance to removal during LLM alignment. We propose AdvBDGen, an adversarially fortified generative fine-tuning framework that automatically generates prompt-specific backdoors that are effective, stealthy, and transferable across models. AdvBDGen employs a generator-discriminator pair, fortified by an adversary, to ensure the installability and stealthiness of backdoors. It enables the crafting and successful installation of complex triggers using as little as 3% of the fine-tuning data. Once installed, these backdoors can jailbreak LLMs during inference, demonstrate improved stability against perturbations compared to traditional constant triggers, and are more challenging to remove. These findings underscore an urgent need for the research community to develop more robust defenses against adversarial backdoor threats in LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04575v1">Are LLMs Reliable Translators of Logical Reasoning Across Lexically Diversified Contexts?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Neuro-symbolic approaches combining large language models (LLMs) with solvers excels in logical reasoning problems need long reasoning chains. In this paradigm, LLMs serve as translators, converting natural language reasoning problems into formal logic formulas. Then reliable symbolic solvers return correct solutions. Despite their success, we find that LLMs, as translators, struggle to handle lexical diversification, a common linguistic phenomenon, indicating that LLMs as logic translators are unreliable in real-world scenarios. Moreover, existing logical reasoning benchmarks lack lexical diversity, failing to challenge LLMs' ability to translate such text and thus obscuring this issue. In this work, we propose SCALe, a benchmark designed to address this significant gap through **logic-invariant lexical diversification**. By using LLMs to transform original benchmark datasets into lexically diversified but logically equivalent versions, we evaluate LLMs' ability to consistently map diverse expressions to uniform logical symbols on these new datasets. Experiments using SCALe further confirm that current LLMs exhibit deficiencies in this capability. Building directly on the deficiencies identified through our benchmark, we propose a new method, MenTaL, to address this limitation. This method guides LLMs to first construct a table unifying diverse expressions before performing translation. Applying MenTaL through in-context learning and supervised fine-tuning (SFT) significantly improves the performance of LLM translators on lexically diversified text. Our code is now available at https://github.com/wufeiwuwoshihua/LexicalDiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04565v1">From Standalone LLMs to Integrated Intelligence: A Survey of Compound Al Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Compound Al Systems (CAIS) is an emerging paradigm that integrates large language models (LLMs) with external components, such as retrievers, agents, tools, and orchestrators, to overcome the limitations of standalone models in tasks requiring memory, reasoning, real-time grounding, and multimodal understanding. These systems enable more capable and context-aware behaviors by composing multiple specialized modules into cohesive workflows. Despite growing adoption in both academia and industry, the CAIS landscape remains fragmented, lacking a unified framework for analysis, taxonomy, and evaluation. In this survey, we define the concept of CAIS, propose a multi-dimensional taxonomy based on component roles and orchestration strategies, and analyze four foundational paradigms: Retrieval-Augmented Generation (RAG), LLM Agents, Multimodal LLMs (MLLMs), and orchestration-centric architectures. We review representative systems, compare design trade-offs, and summarize evaluation methodologies across these paradigms. Finally, we identify key challenges-including scalability, interoperability, benchmarking, and coordination-and outline promising directions for future research. This survey aims to provide researchers and practitioners with a comprehensive foundation for understanding, developing, and advancing the next generation of system-level artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04557v1">SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for Under-Resourced African Languages?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Evaluating machine translation (MT) quality for under-resourced African languages remains a significant challenge, as existing metrics often suffer from limited language coverage and poor performance in low-resource settings. While recent efforts, such as AfriCOMET, have addressed some of the issues, they are still constrained by small evaluation sets, a lack of publicly available training data tailored to African languages, and inconsistent performance in extremely low-resource scenarios. In this work, we introduce SSA-MTE, a large-scale human-annotated MT evaluation (MTE) dataset covering 13 African language pairs from the News domain, with over 63,000 sentence-level annotations from a diverse set of MT systems. Based on this data, we develop SSA-COMET and SSA-COMET-QE, improved reference-based and reference-free evaluation metrics. We also benchmark prompting-based approaches using state-of-the-art LLMs like GPT-4o and Claude. Our experimental results show that SSA-COMET models significantly outperform AfriCOMET and are competitive with the strongest LLM (Gemini 2.5 Pro) evaluated in our study, particularly on low-resource languages such as Twi, Luo, and Yoruba. All resources are released under open licenses to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12930v2">LEDRO: LLM-Enhanced Design Space Reduction and Optimization for Analog Circuits</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Accepted at IEEE International Conference on LLM-Aided Design (ICLAD), 2025; Accepted as a WIP poster at Design Automation Conference (DAC), 2025
    </div>
    <details class="paper-abstract">
      Traditional approaches for designing analog circuits are time-consuming and require significant human expertise. Existing automation efforts using methods like Bayesian Optimization (BO) and Reinforcement Learning (RL) are sub-optimal and costly to generalize across different topologies and technology nodes. In our work, we introduce a novel approach, LEDRO, utilizing Large Language Models (LLMs) in conjunction with optimization techniques to iteratively refine the design space for analog circuit sizing. LEDRO is highly generalizable compared to other RL and BO baselines, eliminating the need for design annotation or model training for different topologies or technology nodes. We conduct a comprehensive evaluation of our proposed framework and baseline on 22 different Op-Amp topologies across four FinFET technology nodes. Results demonstrate the superior performance of LEDRO as it outperforms our best baseline by an average of 13% FoM improvement with 2.15x speed-up on low complexity Op-Amps and 48% FoM improvement with 1.7x speed-up on high complexity Op-Amps. This highlights LEDRO's effective performance, efficiency, and generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23743v2">What Happened in LLMs Layers when Trained for Fast vs. Slow Thinking: A Gradient Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 ACL2025 main, Camera-ready
    </div>
    <details class="paper-abstract">
      What makes a difference in the post-training of LLMs? We investigate the training patterns of different layers in large language models (LLMs) through the lens of the gradient. We are specifically interested in how fast vs. slow thinking affects the layer-wise gradients, given the recent popularity of training LLMs on reasoning paths such as chain-of-thoughts (CoT) and process rewards. In our study, fast thinking without CoT leads to larger gradients and larger differences of gradients across layers than slow thinking (Detailed CoT), indicating the learning stability brought by the latter. Additionally, we study whether the gradient patterns can reflect the correctness of responses when training different LLMs using slow vs. fast thinking paths. The results show that the gradients of slow thinking can distinguish correct and irrelevant reasoning paths. As a comparison, we conduct similar gradient analyses on non-reasoning knowledge learning tasks, on which, however, trivially increasing the response length does not lead to similar behaviors of slow thinking. Our study strengthens fundamental understandings of LLM training and sheds novel insights on its efficiency and stability, which pave the way towards building a generalizable System-2 agent. Our code, data, and gradient statistics can be found in: https://github.com/MingLiiii/Layer_Gradient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04551v1">PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Traditional offline evaluation methods for recommender systems struggle to capture the complexity of modern platforms due to sparse behavioural signals, noisy data, and limited modelling of user personality traits. While simulation frameworks can generate synthetic data to address these gaps, existing methods fail to replicate behavioural diversity, limiting their effectiveness. To overcome these challenges, we propose the Personality-driven User Behaviour Simulator (PUB), an LLM-based simulation framework that integrates the Big Five personality traits to model personalised user behaviour. PUB dynamically infers user personality from behavioural logs (e.g., ratings, reviews) and item metadata, then generates synthetic interactions that preserve statistical fidelity to real-world data. Experiments on the Amazon review datasets show that logs generated by PUB closely align with real user behaviour and reveal meaningful associations between personality traits and recommendation outcomes. These results highlight the potential of the personality-driven simulator to advance recommender system evaluation, offering scalable, controllable, high-fidelity alternatives to resource-intensive real-world experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04544v1">hdl2v: A Code Translation Dataset for Enhanced LLM Verilog Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are playing an increasingly large role in domains such as code generation, including hardware code generation, where Verilog is the key language. However, the amount of publicly available Verilog code pales in comparison to the amount of code available for software languages like Python. In this work, we present hdl2v ("HDL-to-Verilog"), a dataset which seeks to increase the amount of available human-written Verilog data by translating or compiling three other hardware description languages - VHDL, Chisel, and PyMTL3 - to Verilog. Furthermore, we demonstrate the value of hdl2v in enhancing LLM Verilog generation by improving performance of a 32 billion-parameter open-weight model by up to 23% (pass@10) in VerilogEvalV2, without utilizing any data augmentation or knowledge distillation from larger models. We also show hdl2v's ability to boost the performance of a data augmentation-based fine-tuning approach by 63%. Finally, we characterize and analyze our dataset to better understand which characteristics of HDL-to-Verilog datasets can be expanded upon in future work for even better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04535v1">BSBench: will your LLM find the largest prime number?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 7 + 2 pages
    </div>
    <details class="paper-abstract">
      We propose that benchmarking LLMs on questions which have no reasonable answer actually isn't as silly as it sounds. We also present a benchmark that allows such testing and a method to modify the existing datasets, and discover that existing models demonstrate a performance far from the perfect on such questions. Our code and data artifacts are available at https://github.com/L3G5/impossible-bench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04534v1">Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 To be published in Findings of The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025). The main paper is 5 pages and contains 3 figures and 1 table. In total, the paper is 12 pages and contains 8 figures and 5 tables (References + Appendix)
    </div>
    <details class="paper-abstract">
      Discourse particles are crucial elements that subtly shape the meaning of text. These words, often polyfunctional, give rise to nuanced and often quite disparate semantic/discourse effects, as exemplified by the diverse uses of the particle "just" (e.g., exclusive, temporal, emphatic). This work investigates the capacity of LLMs to distinguish the fine-grained senses of English "just", a well-studied example in formal semantics, using data meticulously created and labeled by expert linguists. Our findings reveal that while LLMs exhibit some ability to differentiate between broader categories, they struggle to fully capture more subtle nuances, highlighting a gap in their understanding of discourse particles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19044v3">Calibrating Translation Decoding with Quality Estimation on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Neural machine translation (NMT) systems typically employ maximum a posteriori (MAP) decoding to select the highest-scoring translation from the distribution mass. However, recent evidence highlights the inadequacy of MAP decoding, often resulting in low-quality or even pathological hypotheses -- the decoding objective is not aligned with real-world translation quality. This paper proposes calibrating hypothesis likelihoods with translation quality from a distribution view by directly optimizing their Pearson correlation -- thereby enhancing the effectiveness of translation decoding. With our method, translation on large language models (LLMs) improves substantially after limited training (2K instances per direction). This improvement is orthogonal to those achieved through supervised fine-tuning, leading to substantial gains across a broad range of metrics and human evaluations -- even when applied to top-performing translation-specialized LLMs fine-tuned on high-quality translation data, such as Tower, or when compared to recent preference optimization methods, like CPO. Moreover, the calibrated translation likelihood can directly serve as a strong proxy for translation quality, closely approximating or even surpassing some state-of-the-art translation quality estimation models, like CometKiwi. Lastly, our in-depth analysis demonstrates that calibration enhances the effectiveness of MAP decoding, thereby enabling greater efficiency in real-world deployment. The resulting state-of-the-art translation model, which covers 10 languages, along with the accompanying code and human evaluation data, has been released to the community: https://github.com/moore3930/calibrating-llm-mt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05623v1">Deployability-Centric Infrastructure-as-Code Generation: An LLM-based Iterative Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions, but current evaluation focuses on syntactic correctness while ignoring deployability, the fatal measure of IaC template utility. We address this gap through two contributions: (1) IaCGen, an LLM-based deployability-centric framework that uses iterative feedback mechanism to generate IaC templates, and (2) DPIaC-Eval, a deployability-centric IaC template benchmark consists of 153 real-world scenarios that can evaluate syntax, deployment, user intent, and security. Our evaluation reveals that state-of-the-art LLMs initially performed poorly, with Claude-3.5 and Claude-3.7 achieving only 30.2% and 26.8% deployment success on the first attempt respectively. However, IaCGen transforms this performance dramatically: all evaluated models reach over 90% passItr@25, with Claude-3.5 and Claude-3.7 achieving 98% success rate. Despite these improvements, critical challenges remain in user intent alignment (25.2% accuracy) and security compliance (8.4% pass rate), highlighting areas requiring continued research. Our work provides the first comprehensive assessment of deployability-centric IaC template generation and establishes a foundation for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01618v4">Rollout Roulette: A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating scaling the computation spent at inference time. Existing inference-time scaling methods, usually with reward models, cast the task as a search problem, which tends to be vulnerable to reward hacking as a consequence of approximation errors in reward models. In this paper, we instead cast inference-time scaling as a probabilistic inference task and leverage sampling-based techniques to explore the typical set of the state distribution of a state-space model with an approximate likelihood, rather than optimize for its mode directly. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods to this task. Our empirical evaluation demonstrates that our methods have a 4-16x better scaling rate over our deterministic search counterparts on various challenging mathematical reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct can surpass GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects the rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code, videos, and further information available at https://probabilistic-inference-scaling.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05606v1">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v6">Prompting is Not All You Need! Evaluating LLM Agent Simulation Methodologies with Real-World Online Customer Behavior Data</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating LLM's objective ``accuracy'' rather than the subjective ``believability'' in simulating human behavior, leveraging a large-scale, real-world dataset collected from customers' online shopping actions. We present the first comprehensive evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web shopping action generation. Our results show that out-of-the-box LLM-generated actions are often misaligned with actual human behavior, whereas fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate accurate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasonings into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work evaluates state-of-the-art LLMs in behavior simulation and provides actionable insights into how real-world action data can enhance the fidelity of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05598v1">SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Recent calls for pluralistic alignment of Large Language Models (LLMs) encourage adapting models to diverse user preferences. However, most prior work on personalized reward models heavily rely on additional identity information, such as demographic details or a predefined set of preference categories. To this end, we introduce SynthesizeMe, an approach to inducing synthetic user personas from user interactions for personalized reward modeling. SynthesizeMe first generates and verifies reasoning to explain user preferences, then induces synthetic user personas from that reasoning, and finally filters to informative prior user interactions in order to build personalized prompts for a particular user. We show that using SynthesizeMe induced prompts improves personalized LLM-as-a-judge accuracy by 4.4% on Chatbot Arena. Combining SynthesizeMe derived prompts with a reward model achieves top performance on PersonalRewardBench: a new curation of user-stratified interactions with chatbots collected from 854 users of Chatbot Arena and PRISM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05594v1">SoK: Are Watermarks in LLMs Ready for Deployment?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11187v3">TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 LLMs, Benchmarking, Large Language Models, Bangla, BanglaLLMs
    </div>
    <details class="paper-abstract">
      In this paper, we present TituLLMs, the first large pretrained Bangla LLMs, available in 1b and 3b parameter sizes. Due to computational constraints during both training and inference, we focused on smaller models. To train TituLLMs, we collected a pretraining dataset of approximately ~37 billion tokens. We extended the Llama-3.2 tokenizer to incorporate language- and culture-specific knowledge, which also enables faster training and inference. There was a lack of benchmarking datasets to benchmark LLMs for Bangla. To address this gap, we developed five benchmarking datasets. We benchmarked various LLMs, including TituLLMs, and demonstrated that TituLLMs outperforms its initial multilingual versions. However, this is not always the case, highlighting the complexities of language adaptation. Our work lays the groundwork for adapting existing multilingual open models to other low-resource languages. To facilitate broader adoption and further research, we have made the TituLLMs models and benchmarking datasets publicly available (https://huggingface.co/collections/hishab/titulm-llama-family-6718d31fc1b83529276f490a).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22118v3">The Impact of Inference Acceleration on Bias of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Last few years have seen unprecedented advances in capabilities of Large Language Models (LLMs). These advancements promise to benefit a vast array of application domains. However, due to their immense size, performing inference with LLMs is both costly and slow. Consequently, a plethora of recent work has proposed strategies to enhance inference efficiency, e.g., quantization, pruning, and caching. These acceleration strategies reduce the inference cost and latency, often by several factors, while maintaining much of the predictive performance measured via common benchmarks. In this work, we explore another critical aspect of LLM performance: demographic bias in model generations due to inference acceleration optimizations. Using a wide range of metrics, we probe bias in model outputs from a number of angles. Analysis of outputs before and after inference acceleration shows significant change in bias. Worryingly, these bias effects are complex and unpredictable. A combination of an acceleration strategy and bias type may show little bias change in one model but may lead to a large effect in another. Our results highlight a need for in-depth and case-by-case evaluation of model bias after it has been modified to accelerate inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05566v1">ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled near-human performance on software coding benchmarks, but their effectiveness in RTL code generation remains limited due to the scarcity of high-quality training data. While prior efforts have fine-tuned LLMs for RTL tasks, they do not fundamentally overcome the data bottleneck and lack support for test-time scaling due to their non-reasoning nature. In this work, we introduce ScaleRTL, the first reasoning LLM for RTL coding that scales up both high-quality reasoning data and test-time compute. Specifically, we curate a diverse set of long chain-of-thought reasoning traces averaging 56K tokens each, resulting in a dataset of 3.5B tokens that captures rich RTL knowledge. Fine-tuning a general-purpose reasoning model on this corpus yields ScaleRTL that is capable of deep RTL reasoning. Subsequently, we further enhance the performance of ScaleRTL through a novel test-time scaling strategy that extends the reasoning process via iteratively reflecting on and self-correcting previous reasoning steps. Experimental results show that ScaleRTL achieves state-of-the-art performance on VerilogEval and RTLLM, outperforming 18 competitive baselines by up to 18.4% on VerilogEval and 12.7% on RTLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05560v1">Improving LLMs with a knowledge from databases</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are achieving significant progress almost every moment now. Many advanced techniques have been introduced and widely accepted, like retrieval-augmentation generation (RAG), agents, and tools. Tools can query the database to answer questions from structured data files or perform groupings or other statistics. This unlocks huge opportunities, such as it can answer any question, but also poses threats, such as safety, because there is no control over the commands that are created. We would like to discuss whether we can create a new method that improves answers based on dataset/database via some interpretable ML methods, namely enhanced association rules. The advantage would be if the method can be also used in some safe technique like RAG. Association rules have a sound history. Since the introduction of CN2 and aproiri, many enhancements have been made. In parallel, enhanced association rules have been introduced and evolved over the last 40 years. The general problem is typically that there are too many rules. There are some techniques for handling it, but when LLM emerged, it turned out to be the best use case for the RAG technique for LLMs. We proposed a method that generates a ruleset based on defined knowledge patterns, then converts rules into text form via a rule-to-text converter, and includes the result as an RAG into LLM. We compared this method with ChatGPT (even with using agents) and we have discovered a significant improvement in answering questions based on the dataset. We have also tried several strategies how much rules to generate. We found this improvement interesting. Moreover, it can also be improved in many ways as future work, like incorporating other patterns, the use of rule mining as an agent, and many others.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14074v3">Investigating Non-Transitivity in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 ICML 2025 (Spotlight)
    </div>
    <details class="paper-abstract">
      Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18839v2">Detect, Explain, Escalate: Low-Carbon Dialogue Breakdown Management for LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) are transforming numerous applications, their susceptibility to conversational breakdowns remains a critical challenge undermining user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing low-carbon operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown 'detector' and 'explainer'. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes well to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an 'escalation' architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and energy consumption. Our fine-tuned model and prompting strategies establish new state-of-the-art results on dialogue breakdown detection benchmarks, outperforming specialized classifiers and significantly narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, offering a scalable, efficient, and more interpretable solution for robust conversational AI in high-impact domains. Code and models will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05451v1">Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 31 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04571v2">EnsemW2S: Can an Ensemble of LLMs be Leveraged to Obtain a Stronger LLM?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 superalignment, weak-to-strong generalization on unseen OOD task; formerly appeared as arXiv:2505.21959v1 which was uploaded as a new submission in error
    </div>
    <details class="paper-abstract">
      How can we harness the collective capabilities of multiple Large Language Models (LLMs) to create an even more powerful model? This question forms the foundation of our research, where we propose an innovative approach to weak-to-strong (w2s) generalization-a critical problem in AI alignment. Our work introduces an easy-to-hard (e2h) framework for studying the feasibility of w2s generalization, where weak models trained on simpler tasks collaboratively supervise stronger models on more complex tasks. This setup mirrors real-world challenges, where direct human supervision is limited. To achieve this, we develop a novel AdaBoost-inspired ensemble method, demonstrating that an ensemble of weak supervisors can enhance the performance of stronger LLMs across classification and generative tasks on difficult QA datasets. In several cases, our ensemble approach matches the performance of models trained on ground-truth data, establishing a new benchmark for w2s generalization. We observe an improvement of up to 14% over existing baselines and average improvements of 5% and 4% for binary classification and generative tasks, respectively. This research points to a promising direction for enhancing AI through collective supervision, especially in scenarios where labeled data is sparse or insufficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06391v1">From Rogue to Safe AI: The Role of Explicit Refusals in Aligning LLMs with International Humanitarian Law</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used across sectors, yet their alignment with International Humanitarian Law (IHL) is not well understood. This study evaluates eight leading LLMs on their ability to refuse prompts that explicitly violate these legal frameworks, focusing also on helpfulness - how clearly and constructively refusals are communicated. While most models rejected unlawful requests, the clarity and consistency of their responses varied. By revealing the model's rationale and referencing relevant legal or safety principles, explanatory refusals clarify the system's boundaries, reduce ambiguity, and help prevent misuse. A standardised system-level safety prompt significantly improved the quality of the explanations expressed within refusals in most models, highlighting the effectiveness of lightweight interventions. However, more complex prompts involving technical language or requests for code revealed ongoing vulnerabilities. These findings contribute to the development of safer, more transparent AI systems and propose a benchmark to evaluate the compliance of LLM with IHL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05346v1">Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Project Page: https://hsiung.cc/llm-similarity-risk/
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have underscored their vulnerability to safety alignment jailbreaks, particularly when subjected to downstream fine-tuning. However, existing mitigation strategies primarily focus on reactively addressing jailbreak incidents after safety guardrails have been compromised, removing harmful gradients during fine-tuning, or continuously reinforcing safety alignment throughout fine-tuning. As such, they tend to overlook a critical upstream factor: the role of the original safety-alignment data. This paper therefore investigates the degradation of safety guardrails through the lens of representation similarity between upstream alignment datasets and downstream fine-tuning tasks. Our experiments demonstrate that high similarity between these datasets significantly weakens safety guardrails, making models more susceptible to jailbreaks. Conversely, low similarity between these two types of datasets yields substantially more robust models and thus reduces harmfulness score by up to 10.33%. By highlighting the importance of upstream dataset design in the building of durable safety guardrails and reducing real-world vulnerability to jailbreak attacks, these findings offer actionable insights for fine-tuning service providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05334v1">Search Arena: Analyzing Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Preprint. Code: https://github.com/lmarena/search-arena. Dataset: https://huggingface.co/datasets/lmarena-ai/search-arena-24k
    </div>
    <details class="paper-abstract">
      Search-augmented language models combine web search with Large Language Models (LLMs) to improve response groundedness and freshness. However, analyzing these systems remains challenging: existing datasets are limited in scale and narrow in scope, often constrained to static, single-turn, fact-checking questions. In this work, we introduce Search Arena, a crowd-sourced, large-scale, human-preference dataset of over 24,000 paired multi-turn user interactions with search-augmented LLMs. The dataset spans diverse intents and languages, and contains full system traces with around 12,000 human preference votes. Our analysis reveals that user preferences are influenced by the number of citations, even when the cited content does not directly support the attributed claims, uncovering a gap between perceived and actual credibility. Furthermore, user preferences vary across cited sources, revealing that community-driven platforms are generally preferred and static encyclopedic sources are not always appropriate and reliable. To assess performance across different settings, we conduct cross-arena analyses by testing search-augmented LLMs in a general-purpose chat environment and conventional LLMs in search-intensive settings. We find that web search does not degrade and may even improve performance in non-search settings; however, the quality in search settings is significantly affected if solely relying on the model's parametric knowledge. We open-sourced the dataset to support future research in this direction. Our dataset and code are available at: https://github.com/lmarena/search-arena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05316v1">Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become an effective approach for fine-tuning large language models (LLMs), particularly to enhance their reasoning capabilities. However, RL fine-tuning remains highly resource-intensive, and existing work has largely overlooked the problem of data efficiency. In this paper, we propose two techniques to improve data efficiency in LLM RL fine-tuning: difficulty-targeted online data selection and rollout replay. We introduce the notion of adaptive difficulty to guide online data selection, prioritizing questions of moderate difficulty that are more likely to yield informative learning signals. To estimate adaptive difficulty efficiently, we develop an attention-based framework that requires rollouts for only a small reference set of questions. The adaptive difficulty of the remaining questions is then estimated based on their similarity to this set. To further reduce rollout cost, we introduce a rollout replay mechanism that reuses recent rollouts, lowering per-step computation while maintaining stable updates. Extensive experiments across 6 LLM-dataset combinations show that our method reduces RL fine-tuning time by 25% to 65% to reach the same level of performance as the original GRPO algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24102v2">Is LLM the Silver Bullet to Low-Resource Languages Machine Translation?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Low-Resource Languages (LRLs) present significant challenges in natural language processing due to their limited linguistic resources and underrepresentation in standard datasets. While recent advances in Large Language Models (LLMs) and Neural Machine Translation have substantially improved translation capabilities for high-resource languages, performance disparities persist for LRLs, particularly impacting privacy-sensitive and resource-constrained scenarios. This paper systematically evaluates current LLMs in 200 languages using the FLORES-200 benchmark and demonstrates their limitations in LRL translation capability. We also explore alternative data sources, including news articles and bilingual dictionaries, and demonstrate how knowledge distillation from large pre-trained teacher models can significantly improve the performance of small LLMs on LRL translation tasks. For example, this approach increases EN->LB with the LLM-as-a-Judge score on the validation set from 0.36 to 0.89 for Llama-3.2-3B. Furthermore, we examine different fine-tuning configurations, providing practical insights on optimal data scale, training efficiency, and the preservation of generalization capabilities of models under study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05309v1">Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      LLMs are used predominantly in synchronous communication, where a human user and a model communicate in alternating turns. In contrast, many real-world settings are inherently asynchronous. For example, in group chats, online team meetings, or social games, there is no inherent notion of turns; therefore, the decision of when to speak forms a crucial part of the participant's decision making. In this work, we develop an adaptive asynchronous LLM-agent which, in addition to determining what to say, also decides when to say it. To evaluate our agent, we collect a unique dataset of online Mafia games, including both human participants, as well as our asynchronous agent. Overall, our agent performs on par with human players, both in game performance, as well as in its ability to blend in with the other human players. Our analysis shows that the agent's behavior in deciding when to speak closely mirrors human patterns, although differences emerge in message content. We release all our data and code to support and encourage further research for more realistic asynchronous communication between LLM agents. This work paves the way for integration of LLMs into realistic human group settings, from assistance in team discussions to educational and professional environments where complex social dynamics must be navigated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05260v1">LeanPO: Lean Preference Optimization for Likelihood Alignment in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Code: https://github.com/Wang-Xiaodong1899/LeanPO
    </div>
    <details class="paper-abstract">
      Most Video Large Language Models (Video-LLMs) adopt preference alignment techniques, e.g., DPO~\citep{rafailov2024dpo}, to optimize the reward margin between a winning response ($y_w$) and a losing response ($y_l$). However, the likelihood displacement observed in DPO indicates that both $\log \pi_\theta (y_w\mid x)$ and $\log \pi_\theta (y_l\mid x) $ often decrease during training, inadvertently boosting the probabilities of non-target responses. In this paper, we systematically revisit this phenomenon from LLMs to Video-LLMs, showing that it intensifies when dealing with the redundant complexity of video content. To alleviate the impact of this phenomenon, we propose \emph{Lean Preference Optimization} (LeanPO), a reference-free approach that reformulates the implicit reward as the average likelihood of the response with respect to the policy model. A key component of LeanPO is the reward-trustworthiness correlated self-generated preference data pipeline, which carefully infuses relevant prior knowledge into the model while continuously refining the preference data via self-reflection. This allows the policy model to obtain high-quality paired data and accurately estimate the newly defined reward, thus mitigating the unintended drop. In addition, we introduce a dynamic label smoothing strategy that mitigates the impact of noise in responses from diverse video content, preventing the model from overfitting to spurious details. Extensive experiments demonstrate that LeanPO significantly enhances the performance of state-of-the-art Video-LLMs, consistently boosting baselines of varying capacities with minimal additional training overhead. Moreover, LeanPO offers a simple yet effective solution for aligning Video-LLM preferences with human trustworthiness, paving the way toward the reliable and efficient Video-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04075v2">Rethinking LLM Advancement: Compute-Dependent and Independent Paths to Progress</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Regulatory efforts to govern large language model (LLM) development have predominantly focused on restricting access to high-performance computational resources. This study evaluates the efficacy of such measures by examining whether LLM capabilities can advance through algorithmic innovation in compute-constrained environments. We propose a novel framework distinguishing compute-dependent innovations--which yield disproportionate benefits at high compute--from compute-independent innovations, which improve efficiency across compute scales. The impact is quantified using Compute-Equivalent Gain (CEG). Experimental validation with nanoGPT models confirms that compute-independent advancements yield significant performance gains (e.g., with combined CEG up to $3.5\times$) across the tested scales. In contrast, compute-dependent advancements were detrimental to performance at smaller experimental scales, but showed improved CEG (on par with the baseline) as model size increased, a trend consistent with their definition of yielding primary benefits at higher compute. Crucially, these findings indicate that restrictions on computational hardware, while potentially slowing LLM progress, are insufficient to prevent all capability gains driven by algorithmic advancements. We argue that effective AI oversight must therefore incorporate mechanisms for understanding, anticipating, and potentially guiding algorithmic research, moving beyond a singular focus on hardware. The proposed framework also serves as an analytical tool for forecasting AI progress.
    </details>
</div>
