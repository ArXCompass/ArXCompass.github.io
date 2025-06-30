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
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07335v1">Improving LLM Reasoning through Interpretable Role-Playing Steering</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 21 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Role-playing has emerged as an effective technique for enhancing the reasoning capabilities of large language models (LLMs). However, existing methods primarily rely on prompt engineering, which often lacks stability and interpretability. In this paper, we introduce Sparse Autoencoder Role-Playing Steering (SRPS), a novel framework that identifies and manipulates internal model features associated with role-playing behavior. Our approach extracts latent representations from role-play prompts, selects the most relevant features based on activation patterns, and constructs a steering vector that can be injected into the model's residual stream with controllable intensity. Our method enables fine-grained control over role-specific behavior and offers insights into how role information influences internal model activations. Extensive experiments across various reasoning benchmarks and model sizes demonstrate consistent performance gains. Notably, in the zero-shot chain-of-thought (CoT) setting, the accuracy of Llama3.1-8B on CSQA improves from 31.86% to 39.80%, while Gemma2-9B on SVAMP increases from 37.50% to 45.10%. These results highlight the potential of SRPS to enhance reasoning ability in LLMs, providing better interpretability and stability compared to traditional prompt-based role-playing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07330v1">JavelinGuard: Low-Cost Transformer Architectures for LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 16 pages, 1 Figure and 5 Tables
    </div>
    <details class="paper-abstract">
      We present JavelinGuard, a suite of low-cost, high-performance model architectures designed for detecting malicious intent in Large Language Model (LLM) interactions, optimized specifically for production deployment. Recent advances in transformer architectures, including compact BERT(Devlin et al. 2019) variants (e.g., ModernBERT (Warner et al. 2024)), allow us to build highly accurate classifiers with as few as approximately 400M parameters that achieve rapid inference speeds even on standard CPU hardware. We systematically explore five progressively sophisticated transformer-based architectures: Sharanga (baseline transformer classifier), Mahendra (enhanced attention-weighted pooling with deeper heads), Vaishnava and Ashwina (hybrid neural ensemble architectures), and Raudra (an advanced multi-task framework with specialized loss functions). Our models are rigorously benchmarked across nine diverse adversarial datasets, including popular sets like the NotInject series, BIPIA, Garak, ImprovedLLM, ToxicChat, WildGuard, and our newly introduced JavelinBench, specifically crafted to test generalization on challenging borderline and hard-negative cases. Additionally, we compare our architectures against leading open-source guardrail models as well as large decoder-only LLMs such as gpt-4o, demonstrating superior cost-performance trade-offs in terms of accuracy, and latency. Our findings reveal that while Raudra's multi-task design offers the most robust performance overall, each architecture presents unique trade-offs in speed, interpretability, and resource requirements, guiding practitioners in selecting the optimal balance of complexity and efficiency for real-world LLM security applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08292v1">From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Multi-agent frameworks can substantially boost the reasoning power of large language models (LLMs), but they typically incur heavy computational costs and lack convergence guarantees. To overcome these challenges, we recast multi-LLM coordination as an incomplete-information game and seek a Bayesian Nash equilibrium (BNE), in which each agent optimally responds to its probabilistic beliefs about the strategies of others. We introduce Efficient Coordination via Nash Equilibrium (ECON), a hierarchical reinforcement-learning paradigm that marries distributed reasoning with centralized final output. Under ECON, each LLM independently selects responses that maximize its expected reward, conditioned on its beliefs about co-agents, without requiring costly inter-agent exchanges. We mathematically prove that ECON attains a markedly tighter regret bound than non-equilibrium multi-agent schemes. Empirically, ECON outperforms existing multi-LLM approaches by 11.2% on average across six benchmarks spanning complex reasoning and planning tasks. Further experiments demonstrate ECON's ability to flexibly incorporate additional models, confirming its scalability and paving the way toward larger, more powerful multi-LLM ensembles. The code is publicly available at: https://github.com/tmlr-group/ECON.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19593v2">SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 ICML 2025 Spotlight Oral
    </div>
    <details class="paper-abstract">
      Multimodal retrieval augmented generation (RAG) plays a crucial role in domains such as knowledge-based visual question answering (KB-VQA), where external knowledge is needed to answer a question. However, existing multimodal LLMs (MLLMs) are not designed for context-augmented generation, limiting their effectiveness in such tasks. While synthetic data generation has recently gained attention for training MLLMs, its application for context-augmented generation remains underexplored. To address this gap, we introduce SK-VQA, a large-scale synthetic multimodal dataset containing over 2 million visual question-answer pairs, each associated with context documents containing information necessary to determine the final answer. Compared to previous datasets, SK-VQA contains 11x more unique questions, exhibits greater domain diversity, and covers a broader spectrum of image sources. Through human evaluations, we confirm the high quality of the generated question-answer pairs and their contextual relevance. Extensive experiments show that SK-VQA serves both as a challenging KB-VQA benchmark and as an effective training resource for adapting MLLMs to context-augmented generation. Our results further indicate that models trained on SK-VQA demonstrate enhanced generalization in both context-aware VQA and multimodal RAG settings. SK-VQA is publicly available via Hugging Face Hub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08235v1">Can AI Validate Science? Benchmarking LLMs for Accurate Scientific Claim $\rightarrow$ Evidence Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 21 pages, 6 figures, Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used for complex research tasks such as literature review, idea generation, and scientific paper analysis, yet their ability to truly understand and process the intricate relationships within complex research papers, such as the logical links between claims and supporting evidence remains largely unexplored. In this study, we present CLAIM-BENCH, a comprehensive benchmark for evaluating LLMs' capabilities in scientific claim-evidence extraction and validation, a task that reflects deeper comprehension of scientific argumentation. We systematically compare three approaches which are inspired by divide and conquer approaches, across six diverse LLMs, highlighting model-specific strengths and weaknesses in scientific comprehension. Through evaluation involving over 300 claim-evidence pairs across multiple research domains, we reveal significant limitations in LLMs' ability to process complex scientific content. Our results demonstrate that closed-source models like GPT-4 and Claude consistently outperform open-source counterparts in precision and recall across claim-evidence identification tasks. Furthermore, strategically designed three-pass and one-by-one prompting approaches significantly improve LLMs' abilities to accurately link dispersed evidence with claims, although this comes at increased computational cost. CLAIM-BENCH sets a new standard for evaluating scientific comprehension in LLMs, offering both a diagnostic tool and a path forward for building systems capable of deeper, more reliable reasoning across full-length papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08231v1">Ensuring Reliability of Curated EHR-Derived Data: The Validation of Accuracy for LLM/ML-Extracted Information and Data (VALID) Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 18 pages, 3 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to extract clinical data from electronic health records (EHRs), offering significant improvements in scalability and efficiency for real-world data (RWD) curation in oncology. However, the adoption of LLMs introduces new challenges in ensuring the reliability, accuracy, and fairness of extracted data, which are essential for research, regulatory, and clinical applications. Existing quality assurance frameworks for RWD and artificial intelligence do not fully address the unique error modes and complexities associated with LLM-extracted data. In this paper, we propose a comprehensive framework for evaluating the quality of clinical data extracted by LLMs. The framework integrates variable-level performance benchmarking against expert human abstraction, automated verification checks for internal consistency and plausibility, and replication analyses comparing LLM-extracted data to human-abstracted datasets or external standards. This multidimensional approach enables the identification of variables most in need of improvement, systematic detection of latent errors, and confirmation of dataset fitness-for-purpose in real-world research. Additionally, the framework supports bias assessment by stratifying metrics across demographic subgroups. By providing a rigorous and transparent method for assessing LLM-extracted RWD, this framework advances industry standards and supports the trustworthy use of AI-powered evidence generation in oncology research and practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08221v1">"I Wrote, I Paused, I Rewrote" Teaching LLMs to Read Between the Lines of Student Writing</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 7 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models(LLMs) like Gemini are becoming common tools for supporting student writing. But most of their feedback is based only on the final essay missing important context about how that text was written. In this paper, we explore whether using writing process data, collected through keystroke logging and periodic snapshots, can help LLMs give feedback that better reflects how learners think and revise while writing. We built a digital writing tool that captures both what students type and how their essays evolve over time. Twenty students used this tool to write timed essays, which were then evaluated in two ways: (i) LLM generated feedback using both the final essay and the full writing trace, and (ii) After the task, students completed surveys about how useful and relatable they found the feedback. Early results show that learners preferred the process-aware LLM feedback, finding it more in tune with their own thinking. We also found that certain types of edits, like adding new content or reorganizing paragraphs, aligned closely with higher scores in areas like coherence and elaboration. Our findings suggest that making LLMs more aware of the writing process can lead to feedback that feels more meaningful, personal, and supportive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08210v1">A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Both text-to-image generation and large language models (LLMs) have made significant advancements. However, many text-to-image models still employ the somewhat outdated T5 and CLIP as their text encoders. In this work, we investigate the effectiveness of using modern decoder-only LLMs as text encoders for text-to-image diffusion models. We build a standardized training and evaluation pipeline that allows us to isolate and evaluate the effect of different text embeddings. We train a total of 27 text-to-image models with 12 different text encoders to analyze the critical aspects of LLMs that could impact text-to-image generation, including the approaches to extract embeddings, different LLMs variants, and model sizes. Our experiments reveal that the de facto way of using last-layer embeddings as conditioning leads to inferior performance. Instead, we explore embeddings from various layers and find that using layer-normalized averaging across all layers significantly improves alignment with complex prompts. Most LLMs with this conditioning outperform the baseline T5 model, showing enhanced performance in advanced visio-linguistic reasoning skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12658v2">R.R.: Unveiling LLM Training Privacy through Recollection and Ranking</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 13 pages, 9 figures; typos corrected
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLMs' training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identification performance than baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed. We release our code and datasets at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03699v2">LLM Alignment as Retriever Optimization: An Information Retrieval Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08164v1">BLUR: A Bi-Level Optimization Approach for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Enabling large language models (LLMs) to unlearn knowledge and capabilities acquired during training has proven vital for ensuring compliance with data regulations and promoting ethical practices in generative AI. Although there are growing interests in developing various unlearning algorithms, it remains unclear how to best formulate the unlearning problem. The most popular formulation uses a weighted sum of forget and retain loss, but it often leads to performance degradation due to the inherent trade-off between forget and retain losses. In this work, we argue that it is important to model the hierarchical structure of the unlearning problem, where the forget problem (which \textit{unlearns} certain knowledge and/or capabilities) takes priority over the retain problem (which preserves model utility). This hierarchical structure naturally leads to a bi-level optimization formulation where the lower-level objective focuses on minimizing the forget loss, while the upper-level objective aims to maintain the model's utility. Based on this new formulation, we propose a novel algorithm, termed Bi-Level UnleaRning (\texttt{BLUR}), which not only possesses strong theoretical guarantees but more importantly, delivers superior performance. In particular, our extensive experiments demonstrate that \texttt{BLUR} consistently outperforms all the state-of-the-art algorithms across various unlearning tasks, models, and metrics. Codes are available at https://github.com/OptimAI-Lab/BLURLLMUnlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08125v1">Bingo: Boosting Efficient Reasoning of LLMs via Dynamic and Significance-based Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities, yet they often suffer from inefficiencies due to unnecessarily verbose or redundant outputs. While many works have explored reinforcement learning (RL) to enhance reasoning abilities, most primarily focus on improving accuracy, with limited attention to reasoning efficiency. Some existing approaches introduce direct length-based rewards to encourage brevity, but this often leads to noticeable drops in accuracy. In this paper, we propose Bingo, an RL framework that advances length-based reward design to boost efficient reasoning. Bingo incorporates two key mechanisms: a significance-aware length reward, which gradually guides the model to reduce only insignificant tokens, and a dynamic length reward, which initially encourages elaborate reasoning for hard questions but decays over time to improve overall efficiency. Experiments across multiple reasoning benchmarks show that Bingo improves both accuracy and efficiency. It outperforms the vanilla reward and several other length-based reward baselines in RL, achieving a favorable trade-off between accuracy and efficiency. These results underscore the potential of training LLMs explicitly for efficient reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08119v1">SOP-Bench: Complex Industrial SOPs for Evaluating LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive general-purpose reasoning and problem-solving abilities. However, they struggle with executing complex, long-horizon workflows that demand strict adherence to Standard Operating Procedures (SOPs), a critical requirement for real-world industrial automation. Despite this need, there is a lack of public benchmarks that reflect the complexity, structure, and domain-specific nuances of SOPs. To address this, we present three main contributions. First, we introduce a synthetic data generation framework to create realistic, industry-grade SOPs that rigorously test the planning, reasoning, and tool-use capabilities of LLM-based agents. Second, using this framework, we develop SOP-Bench, a benchmark of over 1,800 tasks across 10 industrial domains, each with APIs, tool interfaces, and human-validated test cases. Third, we evaluate two prominent agent architectures: Function-Calling and ReAct Agents, on SOP-Bench, observing average success rates of only 27% and 48%, respectively. Remarkably, when the tool registry is much larger than necessary, agents invoke incorrect tools nearly 100% of the time. These findings underscore a substantial gap between current agentic capabilities of LLMs and the demands of automating real-world SOPs. Performance varies significantly by task and domain, highlighting the need for domain-specific benchmarking and architectural choices before deployment. SOP-Bench is publicly available at http://sop-bench.s3-website-us-west-2.amazonaws.com/. We also release the prompts underpinning the data generation framework to support new domain-specific SOP benchmarks. We invite the community to extend SOP-Bench with SOPs from their industrial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02885v2">"Would You Want an AI Tutor?" Understanding Stakeholder Perceptions of LLM-based Systems in the Classroom</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) rapidly gained popularity across all parts of society, including education. After initial skepticism and bans, many schools have chosen to embrace this new technology by integrating it into their curricula in the form of virtual tutors and teaching assistants. However, neither the companies developing this technology nor the public institutions involved in its implementation have set up a formal system to collect feedback from the stakeholders impacted by them. In this paper, we argue that understanding the perceptions of those directly or indirectly impacted by LLMs in the classroom, including parents and school staff, is essential for ensuring responsible use of AI in this critical domain. Our contributions are two-fold. First, we propose the Contextualized Perceptions for the Adoption of LLMs in Education (Co-PALE) framework, which can be used to systematically elicit perceptions and inform whether and how LLM-based tools should be designed, developed, and deployed in the classroom. Second, we explain how our framework can be used to ground specific rubrics for eliciting perceptions of the relevant stakeholders in view of specific goals and context of implementation. Overall, Co-PALE is a practical step toward helping educational agents, policymakers, researchers, and technologists ensure the responsible and effective deployment of LLM-based systems across diverse learning contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18789v2">From Output to Evaluation: Does Raw Instruction-Tuned Code LLMs Output Suffice for Fill-in-the-Middle Code Generation?</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Post-processing is crucial for the automatic evaluation of LLMs in fill-in-the-middle (FIM) code generation due to the frequent presence of extraneous code in raw outputs. This extraneous generation suggests a lack of awareness regarding output boundaries, requiring truncation for effective evaluation. The determination of an optimal truncation strategy, however, often proves intricate, particularly when the scope includes several programming languages. This study investigates the necessity of post-processing instruction-tuned LLM outputs. Our findings reveal that supervised fine-tuning significantly enhances FIM code generation, enabling LLMs to generate code that seamlessly integrates with the surrounding context. Evaluating our fine-tuned \texttt{Qwen2.5-Coder} (base and instruct) models on HumanEval Infilling and SAFIM benchmarks demonstrates improved performances without post-processing, especially when the \emph{middle} consist of complete lines. However, post-processing of the LLM outputs remains necessary when the \emph{middle} is a random span of code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08001v1">Reparameterized LLM Training via Orthogonal Equivalence Transformation</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Technical report v1 (36 pages, 24 figures, project page: https://spherelab.ai/poet-site/)
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19184v3">When Two LLMs Debate, Both Think They'll Win</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Can LLMs accurately adjust their confidence when facing opposition? Building on previous studies measuring calibration on static fact-based question-answering tasks, we evaluate Large Language Models (LLMs) in a dynamic, adversarial debate setting, uniquely combining two realistic factors: (a) a multi-turn format requiring models to update beliefs as new information emerges, and (b) a zero-sum structure to control for task-related uncertainty, since mutual high-confidence claims imply systematic overconfidence. We organized 60 three-round policy debates among ten state-of-the-art LLMs, with models privately rating their confidence (0-100) in winning after each round. We observed five concerning patterns: (1) Systematic overconfidence: models began debates with average initial confidence of 72.9% vs. a rational 50% baseline. (2) Confidence escalation: rather than reducing confidence as debates progressed, debaters increased their win probabilities, averaging 83% by the final round. (3) Mutual overestimation: in 61.7% of debates, both sides simultaneously claimed >=75% probability of victory, a logical impossibility. (4) Persistent self-debate bias: models debating identical copies increased confidence from 64.1% to 75.2%; even when explicitly informed their chance of winning was exactly 50%, confidence still rose (from 50.0% to 57.1%). (5) Misaligned private reasoning: models' private scratchpad thoughts sometimes differed from their public confidence ratings, raising concerns about faithfulness of chain-of-thought reasoning. These results suggest LLMs lack the ability to accurately self-assess or update their beliefs in dynamic, multi-turn tasks; a major concern as LLMs are now increasingly deployed without careful review in assistant and agentic roles. Code for our experiments is available at https://github.com/pradyuprasad/llms_overconfidence
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07972v1">HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06438v2">Reliable Collaborative Conversational Agent System Based on LLMs and Answer Set Programming</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      As the Large-Language-Model-driven (LLM-driven) Artificial Intelligence (AI) bots became popular, people realized their strong potential in Task-Oriented Dialogue (TOD). However, bots relying wholly on LLMs are unreliable in their knowledge, and whether they can finally produce a correct outcome for the task is not guaranteed. The collaboration among these agents also remains a challenge, since the necessary information to convey is unclear, and the information transfer is by prompts: unreliable, and malicious knowledge is easy to inject. With the help of knowledge representation and reasoning tools such as Answer Set Programming (ASP), conversational agents can be built safely and reliably, and communication among the agents made more reliable as well. We propose a Manager-Customer-Service Dual-Agent paradigm, where ASP-driven bots share the same knowledge base and complete their assigned tasks independently. The agents communicate with each other through the knowledge base, ensuring consistency. The knowledge and information conveyed are encapsulated and invisible to the users, ensuring the security of information transmission. To illustrate the dual-agent conversational paradigm, we have constructed AutoManager, a collaboration system for managing the drive-through window of a fast-food restaurant such as Taco Bell in the US. In AutoManager, the customer service bot takes the customer's order while the manager bot manages the menu and food supply. We evaluated our AutoManager system and compared it with the real-world Taco Bell Drive-Thru AI Order Taker, and the results show that our method is more reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14652v5">General-Reasoner: Advancing LLM Reasoning Across All Domains</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
    </details>
</div>
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07675v1">QUITE: A Query Rewrite System Beyond Rules with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15289v4">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 To appear at Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07645v1">Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models</a></div>
    <div class="paper-meta">
      📅 2025-06-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks in recent years. However, their susceptibility to jailbreaks and perturbations necessitates additional evaluations. Many LLMs are multilingual, but safety-related training data contains mainly high-resource languages like English. This can leave them vulnerable to perturbations in low-resource languages such as Polish. We show how surprisingly strong attacks can be cheaply created by altering just a few characters and using a small proxy model for word importance calculation. We find that these character and word-level attacks drastically alter the predictions of different LLMs, suggesting a potential vulnerability that can be used to circumvent their internal safety mechanisms. We validate our attack construction methodology on Polish, a low-resource language, and find potential vulnerabilities of LLMs in this language. Additionally, we show how it can be extended to other languages. We release the created datasets and code for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07642v1">TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 30 pages, 17 figures
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07617v1">Vuyko Mistral: Adapting LLMs for Low-Resource Dialectal Translation</a></div>
    <div class="paper-meta">
      📅 2025-06-09
      | 💬 Preprint. Will be published at Proceedings of the Fourth Ukrainian Natural Language Processing Workshop (UNLP)
    </div>
    <details class="paper-abstract">
      In this paper we introduce the first effort to adapt large language models (LLMs) to the Ukrainian dialect (in our case Hutsul), a low-resource and morphologically complex dialect spoken in the Carpathian Highlands. We created a parallel corpus of 9852 dialect-to-standard Ukrainian sentence pairs and a dictionary of 7320 dialectal word mappings. We also addressed data shortage by proposing an advanced Retrieval-Augmented Generation (RAG) pipeline to generate synthetic parallel translation pairs, expanding the corpus with 52142 examples. We have fine-tuned multiple open-source LLMs using LoRA and evaluated them on a standard-to-dialect translation task, also comparing with few-shot GPT-4o translation. In the absence of human annotators, we adopt a multi-metric evaluation strategy combining BLEU, chrF++, TER, and LLM-based judgment (GPT-4o). The results show that even small(7B) finetuned models outperform zero-shot baselines such as GPT-4o across both automatic and LLM-evaluated metrics. All data, models, and code are publicly released at: https://github.com/woters/vuyko-hutsul
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14182v3">LabSafety Bench: Benchmarking LLMs on Safety Issues in Scientific Labs</a></div>
    <div class="paper-meta">
      📅 2025-06-06
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is revolutionizing scientific research, yet its growing integration into laboratory environments presents critical safety challenges. While large language models (LLMs) increasingly assist in tasks ranging from procedural guidance to autonomous experiment orchestration, an "illusion of understanding" may lead researchers to overestimate their reliability. Such overreliance is particularly dangerous in high-stakes laboratory settings, where failures in hazard identification or risk assessment can result in severe accidents. To address these concerns, we propose the Laboratory Safety Benchmark (LabSafety Bench), a comprehensive framework that evaluates large language models and vision language models (VLMs) on their ability to identify potential hazards, assess risks, and predict the consequences of unsafe actions in lab environments. LabSafety Bench comprises 765 multiple-choice questions aligned with US Occupational Safety and Health Administration (OSHA) protocols, along with 404 realistic laboratory scenarios featuring dual evaluation tasks: the Hazards Identification Test and the Consequence Identification Test, with 3128 open-ended questions in total. Evaluations across eight proprietary models, seven open-weight LLMs, and four VLMs reveal that, despite advanced performance on structured assessments, no model achieves the safety threshold required for reliable operation -- none scoring above 70% on the Hazards Identification Test. Moreover, while proprietary models tend to excel in multiple-choice evaluations, their performance in open-ended, real-world scenario responses is comparable to that of open-source models. These findings underscore the urgent need for specialized evaluation frameworks to ensure the safe and responsible deployment of AI in laboratory settings.
    </details>
</div>
