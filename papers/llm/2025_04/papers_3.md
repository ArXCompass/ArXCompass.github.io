# llm - 2025_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v4">A Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      We propose \emph{LumosCore} to build high-bandwidth and large-scale data center networks for LLM jobs. By replacing the core-layer electrical packet switches by optical circuit switches, \emph{LumosCore} could achieves $2\times$ increase in bandwidth or $8\times$ increase in network size. We offer the detailed design of \emph{LumosCore} at both deployment stage and running stage. At deployment stage, we propose Interleaved Wiring, which is compatible with all possible logical topologies. At running stage, we design polynomial-time algorithms for GPU placement, logical topology generating and OCS reconfiguration to minimize network contention and reduce impact to scheduled jobs. We evaluate \emph{LumosCore} using both testbed experiments and large-scale simulation. Compared to traditional hybrid optical/electrical architectures, \emph{LumosCore} increases the end-to-end training throughput by up to 39.5\% on a 128-node testbed. Compared to the state-of-art Clos architectures, \emph{LumosCore} reduces the average job completion time by up to 34.1\% in a 16k simulation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12022v3">ITERTL: An Iterative Framework for Fine-tuning LLMs for RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated excellent performance, inspiring researchers to explore their use in automating register transfer level (RTL) code generation and improving hardware design efficiency. However, the existing approaches to fine-tune LLMs for RTL generation typically are conducted on fixed datasets, which do not fully stimulate the capability of LLMs and require large amounts of reference data, which are costly to acquire. To mitigate these issues, we innovatively introduce an iterative training paradigm named ITERTL. During each iteration, samples are drawn from the model trained in the previous cycle. Then these new samples are employed for training in current loop. Furthermore, we introduce a plug-and-play data filtering strategy, thereby encouraging the model to generate high-quality, self-contained code. Our model outperforms GPT4 and state-of-the-art (SOTA) open-source models, achieving remarkable 53.8% pass@1 rate on VerilogEval-human benchmark. Under similar conditions of data quantity and quality, our approach significantly outperforms the baseline. Extensive experiments validate the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19572v5">ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Accepted at Conference of the North American Chapter of the Association for Computational Linguistics, Student Research Workshop 2025 (NAACL SRW 2025)
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) systems using large language models (LLMs) often generate inaccurate responses due to the retrieval of irrelevant or loosely related information. Existing methods, which operate at the document level, fail to effectively filter out such content. We propose LLM-driven chunk filtering, ChunkRAG, a framework that enhances RAG systems by evaluating and filtering retrieved information at the chunk level. Our approach employs semantic chunking to divide documents into coherent sections and utilizes LLM-based relevance scoring to assess each chunk's alignment with the user's query. By filtering out less pertinent chunks before the generation phase, we significantly reduce hallucinations and improve factual accuracy. Experiments show that our method outperforms existing RAG models, achieving higher accuracy on tasks requiring precise information retrieval. This advancement enhances the reliability of RAG systems, making them particularly beneficial for applications like fact-checking and multi-hop reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10157v2">SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      Social simulation is transforming traditional social science research by modeling human behavior through interactions between virtual individuals and their environments. With recent advances in large language models (LLMs), this approach has shown growing potential in capturing individual differences and predicting group behaviors. However, existing methods face alignment challenges related to the environment, target users, interaction mechanisms, and behavioral patterns. To this end, we introduce SocioVerse, an LLM-agent-driven world model for social simulation. Our framework features four powerful alignment components and a user pool of 10 million real individuals. To validate its effectiveness, we conducted large-scale simulation experiments across three distinct domains: politics, news, and economics. Results demonstrate that SocioVerse can reflect large-scale population dynamics while ensuring diversity, credibility, and representativeness through standardized procedures and minimal manual adjustments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16429v1">Give LLMs a Security Course: Securing Retrieval-Augmented Code Generation via Knowledge Injection</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Code Generation (RACG) leverages external knowledge to enhance Large Language Models (LLMs) in code synthesis, improving the functional correctness of the generated code. However, existing RACG systems largely overlook security, leading to substantial risks. Especially, the poisoning of malicious code into knowledge bases can mislead LLMs, resulting in the generation of insecure outputs, which poses a critical threat in modern software development. To address this, we propose a security-hardening framework for RACG systems, CodeGuarder, that shifts the paradigm from retrieving only functional code examples to incorporating both functional code and security knowledge. Our framework constructs a security knowledge base from real-world vulnerability databases, including secure code samples and root cause annotations. For each code generation query, a retriever decomposes the query into fine-grained sub-tasks and fetches relevant security knowledge. To prioritize critical security guidance, we introduce a re-ranking and filtering mechanism by leveraging the LLMs' susceptibility to different vulnerability types. This filtered security knowledge is seamlessly integrated into the generation prompt. Our evaluation shows CodeGuarder significantly improves code security rates across various LLMs, achieving average improvements of 20.12\% in standard RACG, and 31.53\% and 21.91\% under two distinct poisoning scenarios without compromising functional correctness. Furthermore, CodeGuarder demonstrates strong generalization, enhancing security even when the targeted language's security knowledge is lacking. This work presents CodeGuarder as a pivotal advancement towards building secure and trustworthy RACG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20770v4">Large Language Model Sentinel: LLM Agent for Adversarial Purification</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15610v2">A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 18 pages, 6 figures (3 graphs + 3 flowchart/architecture diagrams), submitted as a preprint for review consideration in AI for Education or Machine Learning applications in low-resource settings. Includes detailed experiments with LoRA and quantization methods for efficient LLM fine-tuning
    </div>
    <details class="paper-abstract">
      The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13217v2">Sustainability via LLM Right-sizing</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 17 pages, 2 Figures, 6 Tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become increasingly embedded in organizational workflows. This has raised concerns over their energy consumption, financial costs, and data sovereignty. While performance benchmarks often celebrate cutting-edge models, real-world deployment decisions require a broader perspective: when is a smaller, locally deployable model "good enough"? This study offers an empirical answer by evaluating eleven proprietary and open-weight LLMs across ten everyday occupational tasks, including summarizing texts, generating schedules, and drafting emails and proposals. Using a dual-LLM-based evaluation framework, we automated task execution and standardized evaluation across ten criteria related to output quality, factual accuracy, and ethical responsibility. Results show that GPT-4o delivers consistently superior performance but at a significantly higher cost and environmental footprint. Notably, smaller models like Gemma-3 and Phi-4 achieved strong and reliable results on most tasks, suggesting their viability in contexts requiring cost-efficiency, local deployment, or privacy. A cluster analysis revealed three model groups -- premium all-rounders, competent generalists, and limited but safe performers -- highlighting trade-offs between quality, control, and sustainability. Significantly, task type influenced model effectiveness: conceptual tasks challenged most models, while aggregation and transformation tasks yielded better performances. We argue for a shift from performance-maximizing benchmarks to task- and context-aware sufficiency assessments that better reflect organizational priorities. Our approach contributes a scalable method to evaluate AI models through a sustainability lens and offers actionable guidance for responsible LLM deployment in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06845v4">7B Fully Open Source Moxin-LLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training and obtaining the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO), an efficient and effective reinforcement learning algorithm following DeepSeek R1, to finetune our model, leading to the Moxin Reasoning model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00348v2">Attention Tracker: Detecting Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Project page: https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05401v2">Post-hoc Study of Climate Microtargeting on Social Media Ads with LLMs: Thematic Insights and Fairness Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Climate change communication on social media increasingly employs microtargeting strategies to effectively reach and influence specific demographic groups. This study presents a post-hoc analysis of microtargeting practices within climate campaigns by leveraging large language models (LLMs) to examine Facebook advertisements. Our analysis focuses on two key aspects: demographic targeting and fairness. We evaluate the ability of LLMs to accurately predict the intended demographic targets, such as gender and age group, achieving an overall accuracy of 88.55%. Furthermore, we instruct the LLMs to generate explanations for their classifications, providing transparent reasoning behind each decision. These explanations reveal the specific thematic elements used to engage different demographic segments, highlighting distinct strategies tailored to various audiences. Our findings show that young adults are primarily targeted through messages emphasizing activism and environmental consciousness, while women are engaged through themes related to caregiving roles and social advocacy. In addition to evaluating the effectiveness of LLMs in detecting microtargeted messaging, we conduct a comprehensive fairness analysis to identify potential biases in model predictions. Our findings indicate that while LLMs perform well overall, certain biases exist, particularly in the classification of senior citizens and male audiences. By showcasing the efficacy of LLMs in dissecting and explaining targeted communication strategies and by highlighting fairness concerns, this study provides a valuable framework for future research aimed at enhancing transparency, accountability, and inclusivity in social media-driven climate campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17130v1">Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17087v1">Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 12 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being widely applied across various fields, but as tasks become more complex, evaluating their responses is increasingly challenging. Compared to human evaluators, the use of LLMs to support performance evaluation offers a more efficient alternative. However, most studies focus mainly on aligning LLMs' judgments with human preferences, overlooking the existence of biases and mistakes in human judgment. Furthermore, how to select suitable LLM judgments given multiple potential LLM responses remains underexplored. To address these two aforementioned issues, we propose a three-stage meta-judge selection pipeline: 1) developing a comprehensive rubric with GPT-4 and human experts, 2) using three advanced LLM agents to score judgments, and 3) applying a threshold to filter out low-scoring judgments. Compared to methods using a single LLM as both judge and meta-judge, our pipeline introduces multi-agent collaboration and a more comprehensive rubric. Experimental results on the JudgeBench dataset show about 15.55\% improvement compared to raw judgments and about 8.37\% improvement over the single-agent baseline. Our work demonstrates the potential of LLMs as meta-judges and lays the foundation for future research on constructing preference datasets for LLM-as-a-judge reinforcement learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17083v1">How Individual Traits and Language Styles Shape Preferences In Open-ended User-LLM Interaction: A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Accepted at GenAICHI 2025 @ ACM CHI 2025
    </div>
    <details class="paper-abstract">
      What makes an interaction with the LLM more preferable for the user? While it is intuitive to assume that information accuracy in the LLM's responses would be one of the influential variables, recent studies have found that inaccurate LLM's responses could still be preferable when they are perceived to be more authoritative, certain, well-articulated, or simply verbose. These variables interestingly fall under the broader category of language style, implying that the style in the LLM's responses might meaningfully influence users' preferences. This hypothesized dynamic could have double-edged consequences: enhancing the overall user experience while simultaneously increasing their susceptibility to risks such as LLM's misinformation or hallucinations. In this short paper, we present our preliminary studies in exploring this subject. Through a series of exploratory and experimental user studies, we found that LLM's language style does indeed influence user's preferences, but how and which language styles influence the preference varied across different user populations, and more interestingly, moderated by the user's very own individual traits. As a preliminary work, the findings in our studies should be interpreted with caution, particularly given the limitations in our samples, which still need wider demographic diversity and larger sample sizes. Our future directions will first aim to address these limitations, which would enable a more comprehensive joint effect analysis between the language style, individual traits, and preferences, and further investigate the potential causal relationship between and beyond these variables.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17075v1">Agree to Disagree? A Meta-Evaluation of LLM Misgendering</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Numerous methods have been proposed to measure LLM misgendering, including probability-based evaluations (e.g., automatically with templatic sentences) and generation-based evaluations (e.g., with automatic heuristics or human validation). However, it has gone unexamined whether these evaluation methods have convergent validity, that is, whether their results align. Therefore, we conduct a systematic meta-evaluation of these methods across three existing datasets for LLM misgendering. We propose a method to transform each dataset to enable parallel probability- and generation-based evaluation. Then, by automatically evaluating a suite of 6 models from 3 families, we find that these methods can disagree with each other at the instance, dataset, and model levels, conflicting on 20.2% of evaluation instances. Finally, with a human evaluation of 2400 LLM generations, we show that misgendering behaviour is complex and goes far beyond pronouns, which automatic evaluations are not currently designed to capture, suggesting essential disagreement with human evaluations. Based on our findings, we provide recommendations for future evaluations of LLM misgendering. Our results are also more widely relevant, as they call into question broader methodological conventions in LLM evaluation, which often assume that different evaluation methods agree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17070v1">Robo-Troj: Attacking LLM-based Task Planners</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17025v1">Optimizing LLMs for Italian: Reducing Token Fertility and Enhancing Efficiency Through Vocabulary Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      The number of pretrained Large Language Models (LLMs) is increasing steadily, though the majority are designed predominantly for the English language. While state-of-the-art LLMs can handle other languages, due to language contamination or some degree of multilingual pretraining data, they are not optimized for non-English languages, leading to inefficient encoding (high token "fertility") and slower inference speed. In this work, we thoroughly compare a variety of vocabulary adaptation techniques for optimizing English LLMs for the Italian language, and put forward Semantic Alignment Vocabulary Adaptation (SAVA), a novel method that leverages neural mapping for vocabulary substitution. SAVA achieves competitive performance across multiple downstream tasks, enhancing grounded alignment strategies. We adapt two LLMs: Mistral-7b-v0.1, reducing token fertility by 25\%, and Llama-3.1-8B, optimizing the vocabulary and reducing the number of parameters by 1 billion. We show that, following the adaptation of the vocabulary, these models can recover their performance with a relatively limited stage of continual training on the target language. Finally, we test the capabilities of the adapted models on various multi-choice and generative tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17018v1">LLM impact on BLV programming</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Submitted to ASSETS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly becoming integral to a wide range of tools, tasks, and problem-solving processes, especially in software development. Originally designed for natural language processing tasks such as text generation, LLMs are increasingly being used to assist both professionals and students in writing code. This growing reliance on LLM-based tools is reshaping programming workflows and task execution. In this study, we explore the impact of these technologies on blind and low-vision (BLV) developers. Our review of existing literature indicates that while LLMs help mitigate some of the challenges faced by BLV programmers, they also introduce new forms of inaccessibility. We conducted an evaluation of five popular LLM-powered integrated development environments (IDEs), assessing their performance across a comprehensive set of programming tasks. Our findings highlight several unsupported scenarios, instances of incorrect model output, and notable limitations in interaction support for specific tasks. Through observing BLV developers as they engaged in coding activities, we uncovered key interaction barriers that go beyond model accuracy or code generation quality. This paper outlines the challenges and corresponding opportunities for improving accessibility in the context of generative AI-assisted programming. Addressing these issues can meaningfully enhance the programming experience for BLV developers. As the generative AI revolution continues to unfold, it must also address the unique burdens faced by this community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15364v2">KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 8 pages, 14 figures
    </div>
    <details class="paper-abstract">
      In this work, we demonstrate that distinctive keys during LLM inference tend to have high attention scores. We explore this phenomenon and propose KeyDiff, a training-free KV cache eviction method based on key similarity. This method facilitates the deployment of LLM-based application requiring long input prompts in resource-constrained environments with limited memory and compute budgets. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We demonstrate that KeyDiff computes the optimal solution to a KV cache selection problem that maximizes key diversity, providing a theoretical understanding of KeyDiff. Notably,KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. We demonstrate the effectiveness of KeyDiff across diverse tasks and models, illustrating a performance gap of less than 0.04\% with 8K cache budget ($\sim$ 23\% KV cache reduction) from the non-evicting baseline on the LongBench benchmark for Llama 3.1-8B and Llama 3.2-3B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15903v2">Impact of Noise on LLM-Models Performance in Abstraction and Reasoning Corpus (ARC) Tasks with Model Temperature Considerations</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 60 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have generated growing interest in their structured reasoning capabilities, particularly in tasks involving abstraction and pattern recognition. The Abstraction and Reasoning Corpus (ARC) benchmark plays a crucial role in evaluating these capabilities by testing how well AI models generalize to novel problems. While GPT-4o demonstrates strong performance by solving all ARC tasks under zero-noise conditions, other models like DeepSeek R1 and LLaMA 3.2 fail to solve any, suggesting limitations in their ability to reason beyond simple pattern matching. To explore this gap, we systematically evaluate these models across different noise levels and temperature settings. Our results reveal that the introduction of noise consistently impairs model performance, regardless of architecture. This decline highlights a shared vulnerability: current LLMs, despite showing signs of abstract reasoning, remain highly sensitive to input perturbations. Such fragility raises concerns about their real-world applicability, where noise and uncertainty are common. By comparing how different model architectures respond to these challenges, we offer insights into the structural weaknesses of modern LLMs in reasoning tasks. This work underscores the need for developing more robust and adaptable AI systems capable of handling the ambiguity and variability inherent in real-world scenarios. Our findings aim to guide future research toward enhancing model generalization, robustness, and alignment with human-like cognitive flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17824v1">EduBot -- Can LLMs Solve Personalized Learning and Programming Assignments?</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Published at AAAI 2025 AI4EDU Workshop
    </div>
    <details class="paper-abstract">
      The prevalence of Large Language Models (LLMs) is revolutionizing the process of writing code. General and code LLMs have shown impressive performance in generating standalone functions and code-completion tasks with one-shot queries. However, the ability to solve comprehensive programming tasks with recursive requests and bug fixes remains questionable. In this paper, we propose EduBot, an intelligent automated assistant system that combines conceptual knowledge teaching, end-to-end code development, personalized programming through recursive prompt-driven methods, and debugging with limited human interventions powered by LLMs. We show that EduBot can solve complicated programming tasks consisting of sub-tasks with increasing difficulties ranging from conceptual to coding questions by recursive automatic prompt-driven systems without finetuning on LLMs themselves. To further evaluate EduBot's performance, we design and conduct a benchmark suite consisting of 20 scenarios in algorithms, machine learning, and real-world problems. The result shows that EduBot can complete most scenarios in less than 20 minutes. Based on the benchmark suites, we perform a comparative study to take different LLMs as the backbone and to verify EduBot's compatibility and robustness across LLMs with varying capabilities. We believe that EduBot is an exploratory approach to explore the potential of pre-trained LLMs in multi-step reasoning and code generation for solving personalized assignments with knowledge learning and code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18583v1">PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The autoregressive nature of large language models (LLMs) limits inference speed. Each forward pass generates only a single token and is often bottlenecked by memory bandwidth. Speculative decoding alleviates this issue using a draft-then-verify approach to accelerate token generation. However, the overhead introduced during the draft phase and the training cost of the draft model limit the efficiency and adaptability of speculative decoding. In this work, we introduce PARallel Draft (PARD), a novel speculative decoding method that enables low-cost adaptation of autoregressive draft models into parallel draft models. PARD enhances inference efficiency by predicting multiple future tokens in a single forward pass of the draft phase, and incorporates a conditional drop token method to accelerate training. Its target-independence property allows a single draft model to be applied to an entire family of different models, minimizing the adaptation cost. Our proposed conditional drop token method can improves draft model training efficiency by 3x. On our optimized inference framework, PARD accelerates LLaMA3.1-8B inference by 4.08x, achieving 311.5 tokens per second.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15585v1">A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10834v3">MetaLLM: A High-performant and Cost-efficient Dynamic Framework for Wrapping LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The rapid progress in machine learning (ML) has brought forth many large language models (LLMs) that excel in various tasks and areas. These LLMs come with different abilities and costs in terms of computation or pricing. Since the demand for each query can vary, e.g., because of the queried domain or its complexity, defaulting to one LLM in an application is not usually the best choice, whether it is the biggest, priciest, or even the one with the best average test performance. Consequently, picking the right LLM that is both accurate and cost-effective for an application is necessary yet remains a challenge. In this paper, we introduce MetaLLM, a framework that dynamically and intelligently routes each query to the optimal LLM (among several available LLMs) for classification and multi-choice question-answering tasks, achieving significantly improved accuracy and cost-effectiveness. By framing the selection problem as a multi-armed bandit, MetaLLM balances prediction accuracy and cost efficiency under uncertainty. Our experiments, conducted on popular LLM platforms such as OpenAI and Together AI, as well as open-source LLM, showcase MetaLLM's efficacy in real-world scenarios, laying the groundwork for future extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15564v1">A Large-scale Class-level Benchmark Dataset for Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 This paper was submitted to the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE 2025) AI models/data track
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated promising capabilities in code generation tasks. However, most existing benchmarks focus on isolated functions and fail to capture the complexity of real-world, class-level software structures. To address this gap, we introduce a large-scale, Python class-level dataset curated from $13{,}174$ real-world open-source projects. The dataset contains over 842,000 class skeletons, each including class and method signatures, along with associated docstrings when available. We preserve structural and contextual dependencies critical to realistic software development scenarios and enrich the dataset with static code metrics to support downstream analysis. To evaluate the usefulness of this dataset, we use extracted class skeletons as prompts for GPT-4 to generate full class implementations. Results show that the LLM-generated classes exhibit strong lexical and structural similarity to human-written counterparts, with average ROUGE@L, BLEU, and TSED scores of 0.80, 0.59, and 0.73, respectively. These findings confirm that well-structured prompts derived from real-world class skeletons significantly enhance LLM performance in class-level code generation. This dataset offers a valuable resource for benchmarking, training, and improving LLMs in realistic software engineering contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10284v2">Can LLMs Generate Tabular Summaries of Science Papers? Rethinking the Evaluation Protocol</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Literature review tables are essential for summarizing and comparing collections of scientific papers. We explore the task of generating tables that best fulfill a user's informational needs given a collection of scientific papers. Building on recent work (Newman et al., 2024), we extend prior approaches to address real-world complexities through a combination of LLM-based methods and human annotations. Our contributions focus on three key challenges encountered in real-world use: (i) User prompts are often under-specified; (ii) Retrieved candidate papers frequently contain irrelevant content; and (iii) Task evaluation should move beyond shallow text similarity techniques and instead assess the utility of inferred tables for information-seeking tasks (e.g., comparing papers). To support reproducible evaluation, we introduce ARXIV2TABLE, a more realistic and challenging benchmark for this task, along with a novel approach to improve literature review table generation in real-world scenarios. Our extensive experiments on this benchmark show that both open-weight and proprietary LLMs struggle with the task, highlighting its difficulty and the need for further advancements. Our dataset and code are available at https://github.com/JHU-CLSP/arXiv2Table.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15548v1">LLM-based Semantic Augmentation for Harmful Content Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong performance on simple text classification tasks, frequently under zero-shot settings. However, their efficacy declines when tackling complex social media challenges such as propaganda detection, hateful meme classification, and toxicity identification. Much of the existing work has focused on using LLMs to generate synthetic training data, overlooking the potential of LLM-based text preprocessing and semantic augmentation. In this paper, we introduce an approach that prompts LLMs to clean noisy text and provide context-rich explanations, thereby enhancing training sets without substantial increases in data volume. We systematically evaluate on the SemEval 2024 multi-label Persuasive Meme dataset and further validate on the Google Jigsaw toxic comments and Facebook hateful memes datasets to assess generalizability. Our results reveal that zero-shot LLM classification underperforms on these high-context tasks compared to supervised models. In contrast, integrating LLM-based semantic augmentation yields performance on par with approaches that rely on human-annotated data, at a fraction of the cost. These findings underscore the importance of strategically incorporating LLMs into machine learning (ML) pipeline for social media classification tasks, offering broad implications for combating harmful content online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15546v1">A Framework for Testing and Adapting REST APIs as LLM Tools</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15544v1">llm-jp-modernbert: A ModernBERT Model Trained on a Large-Scale Japanese Corpus with Long Context Length</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Encoder-only transformer models like BERT are widely adopted as a pre-trained backbone for tasks like sentence classification and retrieval. However, pretraining of encoder models with large-scale corpora and long contexts has been relatively underexplored compared to decoder-only transformers. In this work, we present llm-jp-modernbert, a ModernBERT model trained on a publicly available, massive Japanese corpus with a context length of 8192 tokens. While our model does not surpass existing baselines on downstream tasks, it achieves good results on fill-mask test evaluations. We also analyze the effect of context length expansion through pseudo-perplexity experiments. Furthermore, we investigate sentence embeddings in detail, analyzing their transitions during training and comparing them with those from other existing models, confirming similar trends with models sharing the same architecture. To support reproducibility and foster the development of long-context BERT, we release our model, along with the training and evaluation code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10150v2">HistLLM: A Unified Framework for LLM-Based Multimodal Recommendation with User History Encoding and Compression</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 We want to withdraw this paper and revise its experimental details. The revised version will be uploaded after further verification
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have proven effective in leveraging textual data for recommendations, their application to multimodal recommendation tasks remains relatively underexplored. Although LLMs can process multimodal information through projection functions that map visual features into their semantic space, recommendation tasks often require representing users' history interactions through lengthy prompts combining text and visual elements, which not only hampers training and inference efficiency but also makes it difficult for the model to accurately capture user preferences from complex and extended prompts, leading to reduced recommendation performance. To address this challenge, we introduce HistLLM, an innovative multimodal recommendation framework that integrates textual and visual features through a User History Encoding Module (UHEM), compressing multimodal user history interactions into a single token representation, effectively facilitating LLMs in processing user preferences. Extensive experiments demonstrate the effectiveness and efficiency of our proposed mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12867v3">EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and chain-of-modality (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at https://yanghaha0908.github.io/EmoVoice/. Dataset, code, and checkpoints will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15509v1">SimulS2S-LLM: Unlocking Simultaneous Inference of Speech LLMs for Speech-to-Speech Translation</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Simultaneous speech translation (SST) outputs translations in parallel with streaming speech input, balancing translation quality and latency. While large language models (LLMs) have been extended to handle the speech modality, streaming remains challenging as speech is prepended as a prompt for the entire generation process. To unlock LLM streaming capability, this paper proposes SimulS2S-LLM, which trains speech LLMs offline and employs a test-time policy to guide simultaneous inference. SimulS2S-LLM alleviates the mismatch between training and inference by extracting boundary-aware speech prompts that allows it to be better matched with text input data. SimulS2S-LLM achieves simultaneous speech-to-speech translation (Simul-S2ST) by predicting discrete output speech tokens and then synthesising output speech using a pre-trained vocoder. An incremental beam search is designed to expand the search space of speech token prediction without increasing latency. Experiments on the CVSS speech data show that SimulS2S-LLM offers a better translation quality-latency trade-off than existing methods that use the same training data, such as improving ASR-BLEU scores by 3 points at similar latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16273v1">Investigating LLMs in Clinical Triage: Promising Capabilities, Persistent Intersectional Biases</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 Accepted to GenAI4Health Workshop @ AAAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in clinical decision support, yet their application to triage remains underexplored. We systematically investigate the capabilities of LLMs in emergency department triage through two key dimensions: (1) robustness to distribution shifts and missing data, and (2) counterfactual analysis of intersectional biases across sex and race. We assess multiple LLM-based approaches, ranging from continued pre-training to in-context learning, as well as machine learning approaches. Our results indicate that LLMs exhibit superior robustness, and we investigate the key factors contributing to the promising LLM-based approaches. Furthermore, in this setting, we identify gaps in LLM preferences that emerge in particular intersections of sex and race. LLMs generally exhibit sex-based differences, but they are most pronounced in certain racial groups. These findings suggest that LLMs encode demographic preferences that may emerge in specific clinical contexts or particular combinations of characteristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16266v1">TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge platforms is challenged by their high computational and memory demands. Although recent low-bit quantization methods (e.g., BitNet, DeepSeek) compress weights to as little as 1.58 bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected latency of the prefill phase. We present TeLLMe, the first ternary LLM accelerator for low-power FPGAs (e.g., AMD KV260) that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. Our contributions include: (1) a table-lookup matrix engine for ternary matmul that merges grouped activations with online precomputation to minimize resource use; (2) a fused, bandwidth-efficient attention module featuring a reversed reordering scheme to accelerate prefill; and (3) a tightly integrated normalization and quantization--dequantization unit optimized for ultra-low-bit inference. Under a 7W power budget, TeLLMe delivers up to 9 tokens/s throughput over 1,024-token contexts and prefill latencies of 0.55--1.15 s for 64--128 token prompts, marking a significant energy-efficiency advance and establishing a new edge FPGA benchmark for generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17604v4">OmniScience: A Domain-Specialized LLM for Scientific Reasoning and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in advancing scientific knowledge and addressing complex challenges. In this work, we introduce OmniScience, a specialized large reasoning model for general science, developed through three key components: (1) domain adaptive pretraining on a carefully curated corpus of scientific literature, (2) instruction tuning on a specialized dataset to guide the model in following domain-specific tasks, and (3) reasoning-based knowledge distillation through fine-tuning to significantly enhance its ability to generate contextually relevant and logically sound responses. We demonstrate the versatility of OmniScience by developing a battery agent that efficiently ranks molecules as potential electrolyte solvents or additives. Comprehensive evaluations reveal that OmniScience is competitive with state-of-the-art large reasoning models on the GPQA Diamond and domain-specific battery benchmarks, while outperforming all public reasoning and non-reasoning models with similar parameter counts. We further demonstrate via ablation experiments that domain adaptive pretraining and reasoning-based knowledge distillation are critical to attain our performance levels, across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08619v4">Analyzing 16,193 LLM Papers for Fun and Profits</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping the landscape of computer science research, driving significant shifts in research priorities across diverse conferences and fields. This study provides a comprehensive analysis of the publication trend of LLM-related papers in 77 top-tier computer science conferences over the past six years (2019-2024). We approach this analysis from four distinct perspectives: (1) We investigate how LLM research is driving topic shifts within major conferences. (2) We adopt a topic modeling approach to identify various areas of LLM-related topic growth and reveal the topics of concern at different conferences. (3) We explore distinct contribution patterns of academic and industrial institutions. (4) We study the influence of national origins on LLM development trajectories. Synthesizing the findings from these diverse analytical angles, we derive ten key insights that illuminate the dynamics and evolution of the LLM research ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13657v2">Why Do Multi-Agent LLM Systems Fail?</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 ArXiv v2
    </div>
    <details class="paper-abstract">
      Despite growing enthusiasm for Multi-Agent LLM Systems (MAS), their performance gains on popular benchmarks often remain minimal compared with single-agent frameworks. This gap highlights the need to systematically analyze the challenges hindering MAS effectiveness. We present MAST (Multi-Agent System Failure Taxonomy), the first empirically grounded taxonomy designed to understand MAS failures. We analyze seven popular MAS frameworks across over 200 tasks, involving six expert human annotators. Through this process, we identify 14 unique failure modes, organized into 3 overarching categories, (i) specification issues, (ii) inter-agent misalignment, and (iii) task verification. MAST emerges iteratively from rigorous inter-annotator agreement studies, achieving a Cohen's Kappa score of 0.88. To support scalable evaluation, we develop a validated LLM-as-a-Judge pipeline integrated with MAST. We leverage two case studies to demonstrate MAST's practical utility in analyzing failures and guiding MAS development. Our findings reveal that identified failures require more complex solutions, highlighting a clear roadmap for future research. We open source our comprehensive dataset and LLM annotator to facilitate further development of MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16155v1">DATETIME: A new benchmark to measure LLM translation and reasoning capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      This paper introduces DATETIME, a new high-quality benchmark designed to evaluate the translation and reasoning abilities of a Large Language Model (LLM) on datetimes. A datetime is simply a date and a time, for example '11th.february.2023 ,1:12:31'. Datetimes are an interesting domain because they are intuitive and straightforward for humans to process but present significant challenges for LLMs. At the time of writing, no publicly available benchmark exists for systematically evaluating LLMs on datetime processing. Our experiments show that state-of-the-art models exhibit significant difficulty with tasks involving reasoning on datetimes, and that General Artificial Intelligence is still a distant aspiration. We hypothesize that working with datetimes necessitates translation and/or computation capabilities, and the tasks of the benchmark are organized accordingly. Significant dispersion in performance across models is observed with surprisingly poor performance even on apparently trivial tasks. Whilst frontier models such as ChatGPT, Claude and Llama3.1 have evidently been built and trained with datetime reasoning abilities, significant improvement is required for the open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16144v1">Detecting Actionable Requests and Offers on Social Media During Crises Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Natural disasters often result in a surge of social media activity, including requests for assistance, offers of help, sentiments, and general updates. To enable humanitarian organizations to respond more efficiently, we propose a fine-grained hierarchical taxonomy to systematically organize crisis-related information about requests and offers into three critical dimensions: supplies, emergency personnel, and actions. Leveraging the capabilities of Large Language Models (LLMs), we introduce Query-Specific Few-shot Learning (QSF Learning) that retrieves class-specific labeled examples from an embedding database to enhance the model's performance in detecting and classifying posts. Beyond classification, we assess the actionability of messages to prioritize posts requiring immediate attention. Extensive experiments demonstrate that our approach outperforms baseline prompting strategies, effectively identifying and prioritizing actionable requests and offers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16078v1">LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The success of Large Language Models (LLMs) has sparked interest in various agentic applications. A key hypothesis is that LLMs, leveraging common sense and Chain-of-Thought (CoT) reasoning, can effectively explore and efficiently solve complex domains. However, LLM agents have been found to suffer from sub-optimal exploration and the knowing-doing gap, the inability to effectively act on knowledge present in the model. In this work, we systematically study why LLMs perform sub-optimally in decision-making scenarios. In particular, we closely examine three prevalent failure modes: greediness, frequency bias, and the knowing-doing gap. We propose mitigation of these shortcomings by fine-tuning via Reinforcement Learning (RL) on self-generated CoT rationales. Our experiments across multi-armed bandits, contextual bandits, and Tic-tac-toe, demonstrate that RL fine-tuning enhances the decision-making abilities of LLMs by increasing exploration and narrowing the knowing-doing gap. Finally, we study both classic exploration mechanisms, such as $\epsilon$-greedy, and LLM-specific approaches, such as self-correction and self-consistency, to enable more effective fine-tuning of LLMs for decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16046v1">Certified Mitigation of Worst-Case LLM Copyright Infringement</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      The exposure of large language models (LLMs) to copyrighted material during pre-training raises concerns about unintentional copyright infringement post deployment. This has driven the development of "copyright takedown" methods, post-training approaches aimed at preventing models from generating content substantially similar to copyrighted ones. While current mitigation approaches are somewhat effective for average-case risks, we demonstrate that they overlook worst-case copyright risks exhibits by the existence of long, verbatim quotes from copyrighted sources. We propose BloomScrub, a remarkably simple yet highly effective inference-time approach that provides certified copyright takedown. Our method repeatedly interleaves quote detection with rewriting techniques to transform potentially infringing segments. By leveraging efficient data sketches (Bloom filters), our approach enables scalable copyright screening even for large-scale real-world corpora. When quotes beyond a length threshold cannot be removed, the system can abstain from responding, offering certified risk reduction. Experimental results show that BloomScrub reduces infringement risk, preserves utility, and accommodates different levels of enforcement stringency with adaptive abstention. Our results suggest that lightweight, inference-time methods can be surprisingly effective for copyright prevention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16032v1">LLMs meet Federated Learning for Scalable and Secure IoT Management</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 This work has been submitted to the IEEE Global Communications Conference (GLOBECOM) 2025 for possible publication
    </div>
    <details class="paper-abstract">
      The rapid expansion of IoT ecosystems introduces severe challenges in scalability, security, and real-time decision-making. Traditional centralized architectures struggle with latency, privacy concerns, and excessive resource consumption, making them unsuitable for modern large-scale IoT deployments. This paper presents a novel Federated Learning-driven Large Language Model (FL-LLM) framework, designed to enhance IoT system intelligence while ensuring data privacy and computational efficiency. The framework integrates Generative IoT (GIoT) models with a Gradient Sensing Federated Strategy (GSFS), dynamically optimizing model updates based on real-time network conditions. By leveraging a hybrid edge-cloud processing architecture, our approach balances intelligence, scalability, and security in distributed IoT environments. Evaluations on the IoT-23 dataset demonstrate that our framework improves model accuracy, reduces response latency, and enhances energy efficiency, outperforming traditional FL techniques (i.e., FedAvg, FedOpt). These findings highlight the potential of integrating LLM-powered federated learning into large-scale IoT ecosystems, paving the way for more secure, scalable, and adaptive IoT management solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16030v1">LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 CVPR 2025. If any references are missing, please contact joyachen@u.nus.edu
    </div>
    <details class="paper-abstract">
      Recent video large language models (Video LLMs) often depend on costly human annotations or proprietary model APIs (e.g., GPT-4o) to produce training data, which limits their training at scale. In this paper, we explore large-scale training for Video LLM with cheap automatic speech recognition (ASR) transcripts. Specifically, we propose a novel streaming training approach that densely interleaves the ASR words and video frames according to their timestamps. Compared to previous studies in vision-language representation with ASR, our method naturally fits the streaming characteristics of ASR, thus enabling the model to learn temporally-aligned, fine-grained vision-language modeling. To support the training algorithm, we introduce a data production pipeline to process YouTube videos and their closed captions (CC, same as ASR), resulting in Live-CC-5M dataset for pre-training and Live-WhisperX-526K dataset for high-quality supervised fine-tuning (SFT). Remarkably, even without SFT, the ASR-only pre-trained LiveCC-7B-Base model demonstrates competitive general video QA performance and exhibits a new capability in real-time video commentary. To evaluate this, we carefully design a new LiveSports-3K benchmark, using LLM-as-a-judge to measure the free-form commentary. Experiments show our final LiveCC-7B-Instruct model can surpass advanced 72B models (Qwen2.5-VL-72B-Instruct, LLaVA-Video-72B) in commentary quality even working in a real-time mode. Meanwhile, it achieves state-of-the-art results at the 7B/8B scale on popular video QA benchmarks such as VideoMME and OVOBench, demonstrating the broad generalizability of our approach. All resources of this paper have been released at https://showlab.github.io/livecc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16027v1">Benchmarking LLM for Code Smells Detection: OpenAI GPT-4.0 vs DeepSeek-V3</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Determining the most effective Large Language Model for code smell detection presents a complex challenge. This study introduces a structured methodology and evaluation matrix to tackle this issue, leveraging a curated dataset of code samples consistently annotated with known smells. The dataset spans four prominent programming languages Java, Python, JavaScript, and C++; allowing for cross language comparison. We benchmark two state of the art LLMs, OpenAI GPT 4.0 and DeepSeek-V3, using precision, recall, and F1 score as evaluation metrics. Our analysis covers three levels of detail: overall performance, category level performance, and individual code smell type performance. Additionally, we explore cost effectiveness by comparing the token based detection approach of GPT 4.0 with the pattern-matching techniques employed by DeepSeek V3. The study also includes a cost analysis relative to traditional static analysis tools such as SonarQube. The findings offer valuable guidance for practitioners in selecting an efficient, cost effective solution for automated code smell detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03621v2">Network-aided Efficient LLM Services With Denoising-inspired Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 This paper has been accepted by SCIENCE CHINA Information Sciences. arXiv admin note: substantial text overlap with arXiv:2411.18010
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, leading to their increasing adoption in diverse services delivered through wireless networks. There is a growing trend toward longer prompts to better leverage LLMs' capabilities and address difficult tasks. However, longer prompts not only increase data transmission costs but also require more computing resources and processing time, which impacts overall system efficiency and user experience. To address this challenge, we propose Joint Power and Prompt Optimization (JPPO), a framework that combines Small Language Model (SLM)-based prompt compression with wireless power allocation optimization. By deploying SLM at edge devices for prompt compression and employing Deep Reinforcement Learning (DRL) for joint optimization of compression ratio and transmission power, JPPO effectively balances service quality with resource efficiency. Furthermore, inspired by denoising diffusion models, we design a denoising-inspired prompt compression approach that iteratively compresses prompts by gradually removing non-critical information, further enhancing the framework's performance. Experimental results with long prompt tokens demonstrate that our framework achieves high service fidelity while optimizing power usage in wireless LLM services, significantly reducing the total service response time. With our DRL-based JPPO, the framework maintains fidelity comparable to the no-compression baseline while still achieving a 17% service time reduction through adaptive compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14489v2">Optimizing SLO-oriented LLM Serving with PD-Multiplexing</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Modern LLM services demand high throughput and stringent SLO guarantees across two distinct inference phases-prefill and decode-and complex multi-turn workflows. However, current systems face a fundamental tradeoff: out-of-place compute partition enables per-phase SLO attainment, while in-place memory sharing maximizes throughput via KV cache reuse. Moreover, existing in-place compute partition also encounters low utilization and high overhead due to phase-coupling design. We present Drift, a new LLM serving framework that resolves this tension via PD multiplexing, enabling in-place and phase-decoupled compute partition. Drift leverages low-level GPU partitioning techniques to multiplex prefill and decode phases spatially and adaptively on shared GPUs, while preserving in-place memory sharing. To fully leverage the multiplexing capability, Drift introduces an adaptive gang scheduling mechanism, a contention-free modeling method, and a SLO-aware dispatching policy. Evaluation shows that Drift achieves an average $5.1\times$ throughput improvement (up to $17.5\times$) over state-of-the-art baselines, while consistently meeting SLO targets under complex LLM workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15965v1">From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 26 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Memory is the process of encoding, storing, and retrieving information, allowing humans to retain experiences, knowledge, skills, and facts over time, and serving as the foundation for growth and effective interaction with the world. It plays a crucial role in shaping our identity, making decisions, learning from past experiences, building relationships, and adapting to changes. In the era of large language models (LLMs), memory refers to the ability of an AI system to retain, recall, and use information from past interactions to improve future responses and interactions. Although previous research and reviews have provided detailed descriptions of memory mechanisms, there is still a lack of a systematic review that summarizes and analyzes the relationship between the memory of LLM-driven AI systems and human memory, as well as how we can be inspired by human memory to construct more powerful memory systems. To achieve this, in this paper, we propose a comprehensive survey on the memory of LLM-driven AI systems. In particular, we first conduct a detailed analysis of the categories of human memory and relate them to the memory of AI systems. Second, we systematically organize existing memory-related work and propose a categorization method based on three dimensions (object, form, and time) and eight quadrants. Finally, we illustrate some open problems regarding the memory of current AI systems and outline possible future directions for memory in the era of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15930v1">StreamRL: Scalable, Heterogeneous, and Elastic RL for LLMs with Disaggregated Stream Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the core post-training technique for large language models (LLMs). RL for LLMs involves two stages: generation and training. The LLM first generates samples online, which are then used to derive rewards for training. The conventional view holds that the colocated architecture, where the two stages share resources via temporal multiplexing, outperforms the disaggregated architecture, in which dedicated resources are assigned to each stage. However, in real-world deployments, we observe that the colocated architecture suffers from resource coupling, where the two stages are constrained to use the same resources. This coupling compromises the scalability and cost-efficiency of colocated RL in large-scale training. In contrast, the disaggregated architecture allows for flexible resource allocation, supports heterogeneous training setups, and facilitates cross-datacenter deployment. StreamRL is designed with disaggregation from first principles and fully unlocks its potential by addressing two types of performance bottlenecks in existing disaggregated RL frameworks: pipeline bubbles, caused by stage dependencies, and skewness bubbles, resulting from long-tail output length distributions. To address pipeline bubbles, StreamRL breaks the traditional stage boundary in synchronous RL algorithms through stream generation and achieves full overlapping in asynchronous RL. To address skewness bubbles, StreamRL employs an output-length ranker model to identify long-tail samples and reduces generation time via skewness-aware dispatching and scheduling. Experiments show that StreamRL improves throughput by up to 2.66x compared to existing state-of-the-art systems, and improves cost-effectiveness by up to 1.33x in a heterogeneous, cross-datacenter setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09597v4">Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09385v2">AI Predicts AGI: Leveraging AGI Forecasting and Peer Review to Explore LLMs' Complex Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 47 pages, 8 figures, 17 tables, appendix with data and code
    </div>
    <details class="paper-abstract">
      We tasked 16 state-of-the-art large language models (LLMs) with estimating the likelihood of Artificial General Intelligence (AGI) emerging by 2030. To assess the quality of these forecasts, we implemented an automated peer review process (LLM-PR). The LLMs' estimates varied widely, ranging from 3% (Reka- Core) to 47.6% (GPT-4o), with a median of 12.5%. These estimates closely align with a recent expert survey that projected a 10% likelihood of AGI by 2027, underscoring the relevance of LLMs in forecasting complex, speculative scenarios. The LLM-PR process demonstrated strong reliability, evidenced by a high Intraclass Correlation Coefficient (ICC = 0.79), reflecting notable consistency in scoring across the models. Among the models, Pplx-70b-online emerged as the top performer, while Gemini-1.5-pro-api ranked the lowest. A cross-comparison with external benchmarks, such as LMSYS Chatbot Arena, revealed that LLM rankings remained consistent across different evaluation methods, suggesting that existing benchmarks may not encapsulate some of the skills relevant for AGI prediction. We further explored the use of weighting schemes based on external benchmarks, optimizing the alignment of LLMs' predictions with human expert forecasts. This analysis led to the development of a new, 'AGI benchmark' designed to highlight performance differences in AGI-related tasks. Our findings offer insights into LLMs' capabilities in speculative, interdisciplinary forecasting tasks and emphasize the growing need for innovative evaluation frameworks for assessing AI performance in complex, uncertain real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15903v1">Impact of Noise on LLM-Models Performance in Abstraction and Reasoning Corpus (ARC) Tasks with Model Temperature Considerations</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 60 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have generated growing interest in their structured reasoning capabilities, particularly in tasks involving abstraction and pattern recognition. The Abstraction and Reasoning Corpus (ARC) benchmark plays a crucial role in evaluating these capabilities by testing how well AI models generalize to novel problems. While GPT-4o demonstrates strong performance by solving all ARC tasks under zero-noise conditions, other models like DeepSeek R1 and LLaMA 3.2 fail to solve any, suggesting limitations in their ability to reason beyond simple pattern matching. To explore this gap, we systematically evaluate these models across different noise levels and temperature settings. Our results reveal that the introduction of noise consistently impairs model performance, regardless of architecture. This decline highlights a shared vulnerability: current LLMs, despite showing signs of abstract reasoning, remain highly sensitive to input perturbations. Such fragility raises concerns about their real-world applicability, where noise and uncertainty are common. By comparing how different model architectures respond to these challenges, we offer insights into the structural weaknesses of modern LLMs in reasoning tasks. This work underscores the need for developing more robust and adaptable AI systems capable of handling the ambiguity and variability inherent in real-world scenarios. Our findings aim to guide future research toward enhancing model generalization, robustness, and alignment with human-like cognitive flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14350v2">Time's Up! An Empirical Study of LLM Reasoning Ability Under Output Length Constraint</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the remarkable potential of Large Language Models (LLMs) in test-time scaling. By making the models think before answering, they are able to achieve much higher accuracy with extra inference computation. However, in many real-world scenarios, models are used under time constraints, where an answer should be given to the user within a certain output length. It is unclear whether and how the reasoning abilities of LLMs remain effective under such constraints. We take a first look at this problem by conducting an in-depth empirical study. Specifically, we test more than 25 LLMs on common reasoning datasets under a wide range of output length budgets, and we analyze the correlation between the inference accuracy and various properties including model type, model size, prompt style, etc. We also consider the mappings between the token budgets and the actual on-device latency budgets. The results have demonstrated several interesting findings regarding the budget-aware LLM reasoning that differ from the unconstrained situation, e.g. the optimal choices of model sizes and prompts change under different budgets. These findings offer practical guidance for users to deploy LLMs under real-world latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15867v1">Inducing Vulnerable Code Generation in LLM Coding Assistants</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Due to insufficient domain knowledge, LLM coding assistants often reference related solutions from the Internet to address programming problems. However, incorporating external information into LLMs' code generation process introduces new security risks. In this paper, we reveal a real-world threat, named HACKODE, where attackers exploit referenced external information to embed attack sequences, causing LLMs to produce code with vulnerabilities such as buffer overflows and incomplete validations. We designed a prototype of the attack, which generates effective attack sequences for potential diverse inputs with various user queries and prompt templates. Through the evaluation on two general LLMs and two code LLMs, we demonstrate that the attack is effective, achieving an 84.29% success rate. Additionally, on a real-world application, HACKODE achieves 75.92% ASR, demonstrating its real-world impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15804v1">Insights from Verification: Training a Verilog Generation LLM with Reinforcement Learning with Testbench Feedback</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance in Verilog generation from natural language description. However, ensuring the functional correctness of the generated code remains a significant challenge. This paper introduces a method that integrates verification insights from testbench into the training of Verilog generation LLMs, aligning the training with the fundamental goal of hardware design: functional correctness. The main obstacle in using LLMs for Verilog code generation is the lack of sufficient functional verification data, particularly testbenches paired with design specifications and code. To address this problem, we introduce an automatic testbench generation pipeline that decomposes the process and uses feedback from the Verilog compiler simulator (VCS) to reduce hallucination and ensure correctness. We then use the testbench to evaluate the generated codes and collect them for further training, where verification insights are introduced. Our method applies reinforcement learning (RL), specifically direct preference optimization (DPO), to align Verilog code generation with functional correctness by training preference pairs based on testbench outcomes. In evaluations on VerilogEval-Machine, VerilogEval-Human, RTLLM v1.1, RTLLM v2, and VerilogEval v2, our approach consistently outperforms state-of-the-art baselines in generating functionally correct Verilog code. We open source all training code, data, and models at https://anonymous.4open.science/r/VeriPrefer-E88B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11761v2">Harnessing Language for Coordination: A Framework and Benchmark for LLM-Driven Multi-Agent Control</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various tasks. Their potential to facilitate human coordination with many agents is a promising but largely under-explored area. Such capabilities would be helpful in disaster response, urban planning, and real-time strategy scenarios. In this work, we introduce (1) a real-time strategy game benchmark designed to evaluate these abilities and (2) a novel framework we term HIVE. HIVE empowers a single human to coordinate swarms of up to 2,000 agents through a natural language dialog with an LLM. We present promising results on this multi-agent benchmark, with our hybrid approach solving tasks such as coordinating agent movements, exploiting unit weaknesses, leveraging human annotations, and understanding terrain and strategic points. Our findings also highlight critical limitations of current models, including difficulties in processing spatial visual information and challenges in formulating long-term strategic plans. This work sheds light on the potential and limitations of LLMs in human-swarm coordination, paving the way for future research in this area. The HIVE project page, hive.syrkis.com, includes videos of the system in action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15785v1">WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 Code is available at https://github.com/elated-sawyer/WALL-E
    </div>
    <details class="paper-abstract">
      Can we build accurate world models out of large language models (LLMs)? How can world models benefit LLM agents? The gap between the prior knowledge of LLMs and the specified environment's dynamics usually bottlenecks LLMs' performance as world models. To bridge the gap, we propose a training-free "world alignment" that learns an environment's symbolic knowledge complementary to LLMs. The symbolic knowledge covers action rules, knowledge graphs, and scene graphs, which are extracted by LLMs from exploration trajectories and encoded into executable codes to regulate LLM agents' policies. We further propose an RL-free, model-based agent "WALL-E 2.0" through the model-predictive control (MPC) framework. Unlike classical MPC requiring costly optimization on the fly, we adopt an LLM agent as an efficient look-ahead optimizer of future steps' actions by interacting with the neurosymbolic world model. While the LLM agent's strong heuristics make it an efficient planner in MPC, the quality of its planned actions is also secured by the accurate predictions of the aligned world model. They together considerably improve learning efficiency in a new environment. On open-world challenges in Mars (Minecraft like) and ALFWorld (embodied indoor environments), WALL-E 2.0 significantly outperforms existing methods, e.g., surpassing baselines in Mars by 16.1%-51.6% of success rate and by at least 61.7% in score. In ALFWorld, it achieves a new record 98% success rate after only 4 iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15993v2">Fearful Falcons and Angry Llamas: Emotion Category Annotations of Arguments by Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 accepted to NLP4DH 2025
    </div>
    <details class="paper-abstract">
      Arguments evoke emotions, influencing the effect of the argument itself. Not only the emotional intensity but also the category influence the argument's effects, for instance, the willingness to adapt stances. While binary emotionality has been studied in arguments, there is no work on discrete emotion categories (e.g., "Anger") in such data. To fill this gap, we crowdsource subjective annotations of emotion categories in a German argument corpus and evaluate automatic LLM-based labeling methods. Specifically, we compare three prompting strategies (zero-shot, one-shot, chain-of-thought) on three large instruction-tuned language models (Falcon-7b-instruct, Llama-3.1-8B-instruct, GPT-4o-mini). We further vary the definition of the output space to be binary (is there emotionality in the argument?), closed-domain (which emotion from a given label set is in the argument?), or open-domain (which emotion is in the argument?). We find that emotion categories enhance the prediction of emotionality in arguments, emphasizing the need for discrete emotion annotations in arguments. Across all prompt settings and models, automatic predictions show a high recall but low precision for predicting anger and fear, indicating a strong bias toward negative emotions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14286v2">SRPO: A Cross-Domain Implementation of Large-Scale Reinforcement Learning on LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Recent advances of reasoning models, exemplified by OpenAI's o1 and DeepSeek's R1, highlight the significant potential of Reinforcement Learning (RL) to enhance the reasoning capabilities of Large Language Models (LLMs). However, replicating these advancements across diverse domains remains challenging due to limited methodological transparency. In this work, we present two-Staged history-Resampling Policy Optimization (SRPO), which surpasses the performance of DeepSeek-R1-Zero-32B on the AIME24 and LiveCodeBench benchmarks. SRPO achieves this using the same base model as DeepSeek (i.e. Qwen2.5-32B), using only about 1/10 of the training steps required by DeepSeek-R1-Zero-32B, demonstrating superior efficiency. Building upon Group Relative Policy Optimization (GRPO), we introduce two key methodological innovations: (1) a two-stage cross-domain training paradigm designed to balance the development of mathematical reasoning and coding proficiency, and (2) History Resampling (HR), a technique to address ineffective samples. Our comprehensive experiments validate the effectiveness of our approach, offering valuable insights into scaling LLM reasoning capabilities across diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15719v1">Implementing Rational Choice Functions with LLMs and Measuring their Alignment with User Preferences</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become integral to intelligent user interfaces (IUIs), their role as decision-making agents raises critical concerns about alignment. Although extensive research has addressed issues such as factuality, bias, and toxicity, comparatively little attention has been paid to measuring alignment to preferences, i.e., the relative desirability of different alternatives, a concept used in decision making, economics, and social choice theory. However, a reliable decision-making agent makes choices that align well with user preferences. In this paper, we generalize existing methods that exploit LLMs for ranking alternative outcomes by addressing alignment with the broader and more flexible concept of user preferences, which includes both strict preferences and indifference among alternatives. To this end, we put forward design principles for using LLMs to implement rational choice functions, and provide the necessary tools to measure preference satisfaction. We demonstrate the applicability of our approach through an empirical study in a practical application of an IUI in the automotive domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04060v2">VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Speech large language models (LLMs) have emerged as a prominent research focus in speech processing. We introduce VocalNet-1B and VocalNet-8B, a series of high-performance, low-latency speech LLMs enabled by a scalable and model-agnostic training framework designed for real-time voice interaction. Central to our contribution is the first application of multi-token prediction (MTP) to speech LLMs. This approach represents a paradigm shift from standard next-token prediction (NTP), offering simultaneous improvements in generation speed and quality. Informed by analysis of MTP's effect on speech generation and experimental comparisons, we designed a straightforward and highly effective MTP implementation. Experiments demonstrate that VocalNet performs on par with mainstream Omni LLMs even with limited training data, and significantly surpasses existing open-source speech LLMs. To foster reproducibility and community advancement, all model weights, inference code, training data, and framework implementations have been made publicly available at https://github.com/SJTU-OmniAgent/VocalNet
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15659v1">VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of over 125,000 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4% respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01532v2">Unmasking Implicit Bias: Evaluating Persona-Prompted LLM Responses in Power-Disparate Social Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 NAACL 2025; 10 pages of main text. For interactive visualisation, see https://inc0mple.github.io/Implicit_Bias_Interactive_Data_Viz
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in simulating human behaviour and social intelligence. However, they risk perpetuating societal biases, especially when demographic information is involved. We introduce a novel framework using cosine distance to measure semantic shifts in responses and an LLM-judged Preference Win Rate (WR) to assess how demographic prompts affect response quality across power-disparate social scenarios. Evaluating five LLMs over 100 diverse social scenarios and nine demographic axes, our findings suggest a "default persona" bias toward middle-aged, able-bodied, native-born, Caucasian, atheistic males with centrist views. Moreover, interactions involving specific demographics are associated with lower-quality responses. Lastly, the presence of power disparities increases variability in response semantics and quality across demographic groups, suggesting that implicit biases may be heightened under power-imbalanced conditions. These insights expose the demographic biases inherent in LLMs and offer potential paths toward future bias mitigation efforts in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15630v1">Exploiting Contextual Knowledge in LLMs through V-usable Information based Layer Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, yet they often struggle with context-faithfulness generations that properly reflect contextual knowledge. While existing approaches focus on enhancing the decoding strategies, they ignore the fundamental mechanism of how contextual information is processed within LLMs' internal states. As a result, LLMs remain limited in their ability to fully leverage contextual knowledge. In this paper, we propose Context-aware Layer Enhancement (CaLE), a novel intervention method that enhances the utilization of contextual knowledge within LLMs' internal representations. By employing V-usable information analysis, CaLE strategically amplifies the growth of contextual information at an optimal layer, thereby enriching representations in the final layer. Our experiments demonstrate that CaLE effectively improves context-faithful generation in Question-Answering tasks, particularly in scenarios involving unknown or conflicting contextual knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15610v1">A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 18 pages, 6 figures (3 graphs + 3 flowchart/architecture diagrams), submitted as a preprint for review consideration in AI for Education or Machine Learning applications in low-resource settings. Includes detailed experiments with LoRA and quantization methods for efficient LLM fine-tuning
    </div>
    <details class="paper-abstract">
      The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15600v1">Research on Navigation Methods Based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-22
    </div>
    <details class="paper-abstract">
      In recent years, the field of indoor navigation has witnessed groundbreaking advancements through the integration of Large Language Models (LLMs). Traditional navigation approaches relying on pre-built maps or reinforcement learning exhibit limitations such as poor generalization and limited adaptability to dynamic environments. In contrast, LLMs offer a novel paradigm for complex indoor navigation tasks by leveraging their exceptional semantic comprehension, reasoning capabilities, and zero-shot generalization properties. We propose an LLM-based navigation framework that leverages function calling capabilities, positioning the LLM as the central controller. Our methodology involves modular decomposition of conventional navigation functions into reusable LLM tools with expandable configurations. This is complemented by a systematically designed, transferable system prompt template and interaction workflow that can be easily adapted across different implementations. Experimental validation in PyBullet simulation environments across diverse scenarios demonstrates the substantial potential and effectiveness of our approach, particularly in achieving context-aware navigation through dynamic tool composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11972v2">LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA</a></div>
    <div class="paper-meta">
      📅 2025-04-22
      | 💬 17 pages; code and data are available at https://github.com/Alab-NII/llm-judge-extract-qa
    </div>
    <details class="paper-abstract">
      Extractive reading comprehension question answering (QA) datasets are typically evaluated using Exact Match (EM) and F1-score, but these metrics often fail to fully capture model performance. With the success of large language models (LLMs), they have been employed in various tasks, including serving as judges (LLM-as-a-judge). In this paper, we reassess the performance of QA models using LLM-as-a-judge across four reading comprehension QA datasets. We examine different families of LLMs and various answer types to evaluate the effectiveness of LLM-as-a-judge in these tasks. Our results show that LLM-as-a-judge is highly correlated with human judgments and can replace traditional EM/F1 metrics. By using LLM-as-a-judge, the correlation with human judgments improves significantly, from 0.22 (EM) and 0.40 (F1-score) to 0.85. These findings confirm that EM and F1 metrics underestimate the true performance of the QA models. While LLM-as-a-judge is not perfect for more difficult answer types (e.g., job), it still outperforms EM/F1, and we observe no bias issues, such as self-preference, when the same model is used for both the QA and judgment tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15263v1">Interpretable Locomotion Prediction in Construction Using a Memory-Driven LLM Agent With Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Construction tasks are inherently unpredictable, with dynamic environments and safety-critical demands posing significant risks to workers. Exoskeletons offer potential assistance but falter without accurate intent recognition across diverse locomotion modes. This paper presents a locomotion prediction agent leveraging Large Language Models (LLMs) augmented with memory systems, aimed at improving exoskeleton assistance in such settings. Using multimodal inputs - spoken commands and visual data from smart glasses - the agent integrates a Perception Module, Short-Term Memory (STM), Long-Term Memory (LTM), and Refinement Module to predict locomotion modes effectively. Evaluation reveals a baseline weighted F1-score of 0.73 without memory, rising to 0.81 with STM, and reaching 0.90 with both STM and LTM, excelling with vague and safety-critical commands. Calibration metrics, including a Brier Score drop from 0.244 to 0.090 and ECE from 0.222 to 0.044, affirm improved reliability. This framework supports safer, high-level human-exoskeleton collaboration, with promise for adaptive assistive systems in dynamic industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15253v1">Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 The first two authors contributed equally. The codebase is at https://github.com/SalesforceAIResearch/jetts-benchmark
    </div>
    <details class="paper-abstract">
      Scaling test-time computation, or affording a generator large language model (LLM) extra compute during inference, typically employs the help of external non-generative evaluators (i.e., reward models). Concurrently, LLM-judges, models trained to generate evaluations and critiques (explanations) in natural language, are becoming increasingly popular in automatic evaluation. Despite judge empirical successes, their effectiveness as evaluators in test-time scaling settings is largely unknown. In this paper, we introduce the Judge Evaluation for Test-Time Scaling (JETTS) benchmark, which evaluates judge performance in three domains (math reasoning, code generation, and instruction following) under three task settings: response reranking, step-level beam search, and critique-based response refinement. We evaluate 10 different judge models (7B-70B parameters) for 8 different base generator models (6.7B-72B parameters). Our benchmark shows that while judges are competitive with outcome reward models in reranking, they are consistently worse than process reward models in beam search procedures. Furthermore, though unique to LLM-judges, their natural language critiques are currently ineffective in guiding the generator towards better responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05291v2">Can LLMs Rank the Harmfulness of Smaller LLMs? We are Not There Yet</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become ubiquitous, thus it is important to understand their risks and limitations. Smaller LLMs can be deployed where compute resources are constrained, such as edge devices, but with different propensity to generate harmful output. Mitigation of LLM harm typically depends on annotating the harmfulness of LLM output, which is expensive to collect from humans. This work studies two questions: How do smaller LLMs rank regarding generation of harmful content? How well can larger LLMs annotate harmfulness? We prompt three small LLMs to elicit harmful content of various types, such as discriminatory language, offensive content, privacy invasion, or negative influence, and collect human rankings of their outputs. Then, we evaluate three state-of-the-art large LLMs on their ability to annotate the harmfulness of these responses. We find that the smaller models differ with respect to harmfulness. We also find that large LLMs show low to moderate agreement with humans. These findings underline the need for further work on harm mitigation in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15210v1">Integrating Symbolic Execution into the Fine-Tuning of Code-Generating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Code-generating Large Language Models (LLMs) have become essential tools in modern software development, enhancing productivity and accelerating development. This paper aims to investigate the fine-tuning of code-generating LLMs using Reinforcement Learning and Direct Preference Optimization, further improving their performance. To achieve this, we enhance the training data for the reward model with the help of symbolic execution techniques, ensuring more comprehensive and objective data. With symbolic execution, we create a custom dataset that better captures the nuances in code evaluation. Our reward models, fine-tuned on this dataset, demonstrate significant improvements over the baseline, CodeRL, in estimating the quality of generated code. Our code-generating LLMs, trained with the help of reward model feedback, achieve similar results compared to the CodeRL benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15208v1">Compute-Optimal LLMs Provably Generalize Better With Scale</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Why do larger language models generalize better? To investigate this question, we develop generalization bounds on the pretraining objective of large language models (LLMs) in the compute-optimal regime, as described by the Chinchilla scaling laws. We introduce a novel, fully empirical Freedman-type martingale concentration inequality that tightens existing bounds by accounting for the variance of the loss function. This generalization bound can be decomposed into three interpretable components: the number of parameters per token, the loss variance, and the quantization error at a fixed bitrate. As compute-optimal language models are scaled up, the number of parameters per data point remains constant; however, both the loss variance and the quantization error decrease, implying that larger models should have smaller generalization gaps. We examine why larger models tend to be more quantizable from an information theoretic perspective, showing that the rate at which they can integrate new information grows more slowly than their capacity on the compute-optimal frontier. From these findings we produce a scaling law for the generalization gap, with bounds that become predictably stronger with scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15205v1">Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted at SIGIR 2025 (short)
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enables large language models (LLMs) to generate answers with citations from source documents containing "ground truth", thereby reducing system hallucinations. A crucial factor in RAG evaluation is "support", whether the information in the cited documents supports the answer. To this end, we conducted a large-scale comparative study of 45 participant submissions on 36 topics to the TREC 2024 RAG Track, comparing an automatic LLM judge (GPT-4o) against human judges for support assessment. We considered two conditions: (1) fully manual assessments from scratch and (2) manual assessments with post-editing of LLM predictions. Our results indicate that for 56% of the manual from-scratch assessments, human and GPT-4o predictions match perfectly (on a three-level scale), increasing to 72% in the manual with post-editing condition. Furthermore, by carefully analyzing the disagreements in an unbiased study, we found that an independent human judge correlates better with GPT-4o than a human judge, suggesting that LLM judges can be a reliable alternative for support assessment. To conclude, we provide a qualitative analysis of human and GPT-4o errors to help guide future iterations of support assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15199v1">Zero-Shot, But at What Cost? Unveiling the Hidden Overhead of MILS's LLM-CLIP Framework for Image Captioning</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 9 pages, 2 tables, 1 figure
    </div>
    <details class="paper-abstract">
      MILS (Multimodal Iterative LLM Solver) is a recently published framework that claims "LLMs can see and hear without any training" by leveraging an iterative, LLM-CLIP based approach for zero-shot image captioning. While this MILS approach demonstrates good performance, our investigation reveals that this success comes at a hidden, substantial computational cost due to its expensive multi-step refinement process. In contrast, alternative models such as BLIP-2 and GPT-4V achieve competitive results through a streamlined, single-pass approach. We hypothesize that the significant overhead inherent in MILS's iterative process may undermine its practical benefits, thereby challenging the narrative that zero-shot performance can be attained without incurring heavy resource demands. This work is the first to expose and quantify the trade-offs between output quality and computational cost in MILS, providing critical insights for the design of more efficient multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09597v3">Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14866v2">LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted by MLSys 2025. Code available at: https://github.com/mit-han-lab/omniserve
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable potential in processing long sequences and complex reasoning tasks, yet efficiently serving these models remains challenging due to the quadratic computational complexity of attention in the prefilling stage and the large memory footprint of the KV cache in the decoding stage. To address these issues, we introduce LServe, an efficient system that accelerates long-sequence LLM serving via hybrid sparse attention. This method unifies different hardware-friendly, structured sparsity patterns for both prefilling and decoding attention into a single framework, where computations on less important tokens are skipped block-wise. LServe demonstrates the compatibility of static and dynamic sparsity in long-context LLM attention. This design enables multiplicative speedups by combining these optimizations. Specifically, we convert half of the attention heads to nearly free streaming heads in both the prefilling and decoding stages. Additionally, we find that only a constant number of KV pages is required to preserve long-context and reasoning capabilities, irrespective of context length. We then design a hierarchical KV page selection policy that dynamically prunes KV pages based on query-centric similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is released at https://github.com/mit-han-lab/omniserve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15080v1">Empowering AI to Generate Better AI Code: Guided Generation of Deep Learning Projects with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have been widely applied to code generation, they struggle with generating entire deep learning projects, which are characterized by complex structures, longer functions, and stronger reliance on domain knowledge than general-purpose code. An open-domain LLM often lacks coherent contextual guidance and domain expertise for specific projects, making it challenging to produce complete code that fully meets user requirements. In this paper, we propose a novel planning-guided code generation method, DLCodeGen, tailored for generating deep learning projects. DLCodeGen predicts a structured solution plan, offering global guidance for LLMs to generate the project. The generated plan is then leveraged to retrieve semantically analogous code samples and subsequently abstract a code template. To effectively integrate these multiple retrieval-augmented techniques, a comparative learning mechanism is designed to generate the final code. We validate the effectiveness of our approach on a dataset we build for deep learning code generation. Experimental results demonstrate that DLCodeGen outperforms other baselines, achieving improvements of 9.7% in CodeBLEU and 3.6% in human evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15077v1">Think2SQL: Reinforce LLM Reasoning Capabilities for Text2SQL</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities in transforming natural language questions about relational databases into SQL queries. Despite recent improvements, small LLMs struggle to handle questions involving multiple tables and complex SQL patterns under a Zero-Shot Learning (ZSL) setting. Supervised Fine-Tuning (SFT) partially compensate the knowledge deficits in pretrained models but falls short while dealing with queries involving multi-hop reasoning. To bridge this gap, different LLM training strategies to reinforce reasoning capabilities have been proposed, ranging from leveraging a thinking process within ZSL, including reasoning traces in SFT, or adopt Reinforcement Learning (RL) strategies. However, the influence of reasoning on Text2SQL performance is still largely unexplored. This paper investigates to what extent LLM reasoning capabilities influence their Text2SQL performance on four benchmark datasets. To this end, it considers the following LLM settings: (1) ZSL, including general-purpose reasoning or not; (2) SFT, with and without task-specific reasoning traces; (3) RL, leveraging execution accuracy as primary reward function; (4) SFT+RL, i.e, a two-stage approach that combines SFT and RL. The results show that general-purpose reasoning under ZSL proves to be ineffective in tackling complex Text2SQL cases. Small LLMs benefit from SFT with reasoning much more than larger ones, bridging the gap of their (weaker) model pretraining. RL is generally beneficial across all tested models and datasets, particularly when SQL queries involve multi-hop reasoning and multiple tables. Small LLMs with SFT+RL excel on most complex datasets thanks to a strategic balance between generality of the reasoning process and optimization of the execution accuracy. Thanks to RL, the7B Qwen-Coder-2.5 model performs on par with 100+ Billion ones on the Bird dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15052v1">Testing LLMs' Capabilities in Annotating Translations Based on an Error Typology Designed for LSP Translation: First Experiments with ChatGPT</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted for publication in the proceedings of MT Summit 2025
    </div>
    <details class="paper-abstract">
      This study investigates the capabilities of large language models (LLMs), specifically ChatGPT, in annotating MT outputs based on an error typology. In contrast to previous work focusing mainly on general language, we explore ChatGPT's ability to identify and categorise errors in specialised translations. By testing two different prompts and based on a customised error typology, we compare ChatGPT annotations with human expert evaluations of translations produced by DeepL and ChatGPT itself. The results show that, for translations generated by DeepL, recall and precision are quite high. However, the degree of accuracy in error categorisation depends on the prompt's specific features and its level of detail, ChatGPT performing very well with a detailed prompt. When evaluating its own translations, ChatGPT achieves significantly poorer results, revealing limitations with self-assessment. These results highlight both the potential and the limitations of LLMs for translation evaluation, particularly in specialised domains. Our experiments pave the way for future research on open-source LLMs, which could produce annotations of comparable or even higher quality. In the future, we also aim to test the practical effectiveness of this automated evaluation in the context of translation training, particularly by optimising the process of human evaluation by teachers and by exploring the impact of annotations by LLMs on students' post-editing and translation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15022v1">LLMs as Data Annotators: How Close Are We to Human Performance</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 27 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In NLP, fine-tuning LLMs is effective for various applications but requires high-quality annotated data. However, manual annotation of data is labor-intensive, time-consuming, and costly. Therefore, LLMs are increasingly used to automate the process, often employing in-context learning (ICL) in which some examples related to the task are given in the prompt for better performance. However, manually selecting context examples can lead to inefficiencies and suboptimal model performance. This paper presents comprehensive experiments comparing several LLMs, considering different embedding models, across various datasets for the Named Entity Recognition (NER) task. The evaluation encompasses models with approximately $7$B and $70$B parameters, including both proprietary and non-proprietary models. Furthermore, leveraging the success of Retrieval-Augmented Generation (RAG), it also considers a method that addresses the limitations of ICL by automatically retrieving contextual examples, thereby enhancing performance. The results highlight the importance of selecting the appropriate LLM and embedding model, understanding the trade-offs between LLM sizes and desired performance, and the necessity to direct research efforts towards more challenging datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15013v1">Stay Hungry, Stay Foolish: On the Extended Reading Articles Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted by iRAISE@AAAI2025
    </div>
    <details class="paper-abstract">
      The process of creating educational materials is both time-consuming and demanding for educators. This research explores the potential of Large Language Models (LLMs) to streamline this task by automating the generation of extended reading materials and relevant course suggestions. Using the TED-Ed Dig Deeper sections as an initial exploration, we investigate how supplementary articles can be enriched with contextual knowledge and connected to additional learning resources. Our method begins by generating extended articles from video transcripts, leveraging LLMs to include historical insights, cultural examples, and illustrative anecdotes. A recommendation system employing semantic similarity ranking identifies related courses, followed by an LLM-based refinement process to enhance relevance. The final articles are tailored to seamlessly integrate these recommendations, ensuring they remain cohesive and informative. Experimental evaluations demonstrate that our model produces high-quality content and accurate course suggestions, assessed through metrics such as Hit Rate, semantic similarity, and coherence. Our experimental analysis highlight the nuanced differences between the generated and existing materials, underscoring the model's capacity to offer more engaging and accessible learning experiences. This study showcases how LLMs can bridge the gap between core content and supplementary learning, providing students with additional recommended resources while also assisting teachers in designing educational materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14985v1">aiXamine: LLM Safety and Security Simplified</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14969v1">Evaluating LLMs on Chinese Topic Constructions: A Research Proposal Inspired by Tian et al. (2024)</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      This paper proposes a framework for evaluating large language models (LLMs) on Chinese topic constructions, focusing on their sensitivity to island constraints. Drawing inspiration from Tian et al. (2024), we outline an experimental design for testing LLMs' grammatical knowledge of Mandarin syntax. While no experiments have been conducted yet, this proposal aims to provide a foundation for future studies and invites feedback on the methodology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14964v1">Evaluating Code Generation of LLMs in Advanced Computer Science Problems</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as GitHub Copilot and ChatGPT have become popular among programming students. Students use LLMs to assist them in programming courses, including generating source code. Previous work has evaluated the ability of LLMs in solving introductory-course programming assignments. The results have shown that LLMs are highly effective in generating code for introductory Computer Science (CS) courses. However, there is a gap in research on evaluating LLMs' ability to generate code that solves advanced programming assignments. In this work, we evaluate the ability of four LLM tools to solve programming assignments from advanced CS courses in three popular programming languages, Java, Python, and C. We manually select 12 problems, three problems from introductory courses as the baseline and nine programming assignments from second- and third-year CS courses. To evaluate the LLM-generated code, we generate a test suite of 1000 test cases per problem and analyze the program output. Our evaluation shows that although LLMs are highly effective in generating source code for introductory programming courses, solving advanced programming assignments is more challenging. Nonetheless, in many cases, LLMs identify the base problem and provide partial solutions that may be useful to CS students. Furthermore, our results may provide useful guidance for teachers of advanced programming courses on how to design programming assignments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06193v3">Can LLMs Replace Human Evaluators? An Empirical Study of LLM-as-a-Judge in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted by ISSTA 2025: https://conf.researchr.org/details/issta-2025/issta-2025-papers/85/Can-LLMs-replace-Human-Evaluators-An-Empirical-Study-of-LLM-as-a-Judge-in-Software-E
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been deployed to tackle various software engineering (SE) tasks like code generation, significantly advancing the automation of SE tasks. However, assessing the quality of these LLM-generated code and text remains challenging. The commonly used Pass@k metric necessitates extensive unit tests and configured environments, demands a high labor cost, and is not suitable for evaluating LLM-generated text. Conventional metrics like BLEU, which measure only lexical rather than semantic similarity, have also come under scrutiny. In response, a new trend has emerged to employ LLMs for automated evaluation, known as LLM-as-a-judge. These LLM-as-a-judge methods are claimed to better mimic human assessment than conventional metrics without relying on high-quality reference answers. Nevertheless, their exact human alignment in SE tasks remains unexplored. In this paper, we empirically explore LLM-as-a-judge methods for evaluating SE tasks, focusing on their alignment with human judgments. We select seven LLM-as-a-judge methods that utilize general-purpose LLMs, alongside two LLMs specifically fine-tuned for evaluation. After generating and manually scoring LLM responses on three recent SE datasets of code translation, code generation, and code summarization, we then prompt these methods to evaluate each response. Finally, we compare the scores generated by these methods with human evaluation. The results indicate that output-based methods reach the highest Pearson correlation of 81.32 and 68.51 with human scores in code translation and generation, achieving near-human evaluation, noticeably outperforming ChrF++, one of the best conventional metrics, at 34.23 and 64.92. Such output-based methods prompt LLMs to output judgments directly, and exhibit more balanced score distributions that resemble human score patterns. Finally, we provide...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14928v1">EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12322v2">A Strategic Coordination Framework of Small LLMs Matches Large LLMs in Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      While data synthesis and distillation are promising strategies to enhance small language models, current approaches heavily rely on Large Language Models (LLMs), which suffer from high computational costs, environmental inefficiency, and potential biases inherited from monolithic architectures. In contrast, smaller LLMs are more accessible and sustainable, but their individual capabilities often fall short in generating high-quality, diverse, and reliable data. Inspired by collaborative human processes (e.g., peer review), we propose a multiple small LLMs involved framework, GRA, that aggregates specialized roles across small LLMs to iterative refinement and quality control typically achieved by a single large LLM. In this collaborative framework, multiple small LLMs assume distinct roles-Generator, Reviewer, and Adjudicator-to simulate a peer-review-inspired data synthesis pipeline. The Generator proposes initial data samples, the Reviewer critiques their quality and diversity, and the Adjudicator resolves conflicts to finalize the output. By decomposing the synthesis process into specialized sub-tasks, collaborative small LLMs can achieve data-level parity with large LLM-based distillation. Through experiments across multiple benchmarks, we demonstrate that GRA-produced data matches or exceeds the quality of single large LLM outputs, e.g., Qwen-2.5-72B-Instruct. Our results challenge the necessity of monolithic large models for high-quality data synthesis, advocating instead for strategic coordination of smaller agents. Our datasets, models, and code are publicly available at https://github.com/GX-XinGao/GRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01354v2">MCGMark: An Encodable and Robust Online Watermark for Tracing LLM-Generated Malicious Code</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      With the advent of large language models (LLMs), numerous software service providers (SSPs) are dedicated to developing LLMs customized for code generation tasks, such as CodeLlama and Copilot. However, these LLMs can be leveraged by attackers to create malicious software, which may pose potential threats to the software ecosystem. For example, they can automate the creation of advanced phishing malware. To address this issue, we first conduct an empirical study and design a prompt dataset, MCGTest, which involves approximately 400 person-hours of work and consists of 406 malicious code generation tasks. Utilizing this dataset, we propose MCGMark, the first robust, code structure-aware, and encodable watermarking approach to trace LLM-generated code. We embed encodable information by controlling the token selection and ensuring the output quality based on probabilistic outliers. Additionally, we enhance the robustness of the watermark by considering the structural features of malicious code, preventing the embedding of the watermark in easily modified positions, such as comments. We validate the effectiveness and robustness of MCGMark on the DeepSeek-Coder. MCGMark achieves an embedding success rate of 88.9% within a maximum output limit of 400 tokens. Furthermore, it also demonstrates strong robustness and has minimal impact on the quality of the output code. Our approach assists SSPs in tracing and holding responsible parties accountable for malicious code generated by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14905v1">CRAVE: A Conflicting Reasoning Approach for Explainable Claim Verification Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      The rapid spread of misinformation, driven by digital media and AI-generated content, has made automatic claim verification essential. Traditional methods, which depend on expert-annotated evidence, are labor-intensive and not scalable. Although recent automated systems have improved, they still struggle with complex claims that require nuanced reasoning. To address this, we propose CRAVE, a Conflicting Reasoning Approach for explainable claim VErification, that verify the complex claims based on the conflicting rationales reasoned by large language models (LLMs). Specifically, CRAVE introduces a three-module framework. Ambiguity Elimination enchanced Evidence Retrieval module performs ambiguity elimination and entity-based search to gather relevant evidence related to claim verification from external sources like Wikipedia. Conflicting Perspective Reasoning and Preliminary Judgment module with LLMs adopts LLMs to reason rationales with conflicting stances about claim verification from retrieved evidence across four dimensions, i.e., direct evidence, semantic relationships, linguistic patterns, and logical reasoning and make a preliminary judgment. Finally, Small Language Model (SLM) based Judge module is fine-tuned to make use of preliminary judgment from LLMs to assess the confidence of the conflicting rationales and make a final authenticity judgment. This methodology allows CRAVE to capture subtle inconsistencies in complex claims, improving both the accuracy and transparency of claim verification. Extensive experiments on two public claim verification datasets demonstrate that our CRAVE model achieves much better performance than state-of-the-art methods and exhibits a superior capacity for finding relevant evidence and explaining the model predictions. The code is provided at https://github.com/8zym/CRAVE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14815v2">Adapting Multilingual LLMs to Low-Resource Languages using Continued Pre-training and Synthetic Corpus</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Multilingual LLMs support a variety of languages; however, their performance is suboptimal for low-resource languages. In this work, we emphasize the importance of continued pre-training of multilingual LLMs and the use of translation-based synthetic pre-training corpora for improving LLMs in low-resource languages. We conduct our study in the context of the low-resource Indic language Hindi. We introduce Nemotron-Mini-Hindi 4B, a bilingual SLM supporting both Hindi and English, based on Nemotron-Mini 4B. The model is trained using a mix of real and synthetic Hindi + English tokens, with continuous pre-training performed on 400B tokens. We demonstrate that both the base and instruct models achieve state-of-the-art results on Hindi benchmarks while remaining competitive on English tasks. Additionally, we observe that the continued pre-training approach enhances the model's overall factual accuracy. We perform an ablation study to highlight the impact of Hindi pre-training, showing significant improvements in Hindi chat capabilities and factual accuracy, which cannot be achieved through Hindi alignment alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09407v2">UXAgent: A System for Simulating Usability Testing of Web Design with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Usability testing is a fundamental research method that user experience (UX) researchers use to evaluate and iterate a web design, but\textbf{ how to evaluate and iterate the usability testing study design } itself? Recent advances in Large Language Model-simulated Agent (\textbf{LLM Agent}) research inspired us to design \textbf{UXAgent} to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human-subject study. Our system features a Persona Generator module, an LLM Agent module, and a Universal Browser Connector module to automatically generate thousands of simulated users to interactively test the target website. The system also provides an Agent Interview Interface and a Video Replay Interface so that the UX researchers can easily review and analyze the generated qualitative and quantitative log data. Through a heuristic evaluation, five UX researcher participants praised the innovation of our system but also expressed concerns about the future of LLM Agent usage in UX studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v4">LLM Agents That Act Like Us: Accurate Human Behavior Simulation with Real-World Data</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating and improving LLM's objective ``accuracy'' rather than the subjective ``believability'' in the web action generation task, leveraging a large-scale, real-world dataset collected from online shopping human actions. We present the first comprehensive quantitative evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web action generation. Our results show that fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasoning traces into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work establishes a new benchmark for evaluating LLMs in behavior simulation and offers actionable insights into how real-world action data and reasoning augmentation can enhance the fidelity of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14852v1">APIRAT: Integrating Multi-source API Knowledge for Enhanced Code Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 accepted by COMPSAC2025
    </div>
    <details class="paper-abstract">
      Code translation is an essential task in software migration, multilingual development, and system refactoring. Recent advancements in large language models (LLMs) have demonstrated significant potential in this task. However, prior studies have highlighted that LLMs often struggle with domain-specific code, particularly in resolving cross-lingual API mappings. To tackle this challenge, we propose APIRAT, a novel code translation method that integrates multi-source API knowledge. APIRAT employs three API knowledge augmentation techniques, including API sequence retrieval, API sequence back-translation, and API mapping, to guide LLMs to translating code, ensuring both the correct structure of API sequences and the accurate usage of individual APIs. Extensive experiments on two public datasets, CodeNet and AVATAR, indicate that APIRAT significantly surpasses existing LLM-based methods, achieving improvements in computational accuracy ranging from 4% to 15.1%. Additionally, our evaluation across different LLMs showcases the generalizability of APIRAT. An ablation study further confirms the individual contributions of each API knowledge component, underscoring the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14810v1">DONOD: Robust and Generalizable Instruction Fine-Tuning for LLMs via Model-Intrinsic Dataset Pruning</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Ad-hoc instruction fine-tuning of large language models (LLMs) is widely adopted for domain-specific adaptation. While domain-specific supervised fine-tuning (SFT) is effective and efficient, it often weakens cross-domain generalization and struggles with noisy training data. To address these challenges, we propose DONOD, a lightweight model-intrinsic data pruning method. Our approach evaluates data using two model-parameter-based metrics: Delta of Norm (DON), which captures the cumulative influence on model weights, and Norm of Delta (NOD), which quantifies weight instability. Moreover, by employing the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) algorithm, we effectively filter noisy, unlearnable, and generalization-harming samples without relying on auxiliary models during the SFT process. Experiments on mathematical tasks demonstrate that data selected by DONOD achieve superior fine-tuning efficiency and improved robustness against noisy data. By filtering out 70% of the full dataset, we improve target-domain accuracy by 14.90% and cross-domain accuracy by 5.67%. Meanwhile, our selected data present superior cross-architecture generalization. Data pruned by smaller models (e.g., Llama 3.1-8B) generalize effectively on larger models (e.g., Llama 2-13B). Compared to existing related methodologies, DONOD demonstrates comparable or superior performance while remaining dataset-agnostic, enabling broader applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15547v2">Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are combined with tools to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or tool's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompts are prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., agent isolation, secure untrusted data processing, and privilege escalation guardrails. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05693v2">Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted by AAAI 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) stand out for their impressive performance in intricate language modeling tasks. However, their demanding computational and memory needs pose obstacles for broad use on edge devices. Quantization is then introduced to boost LLMs' on-device efficiency. Recent works show that 8-bit or lower weight quantization is feasible with minimal impact on end-to-end task performance, while the activation is still not quantized. On the other hand, mainstream commodity edge devices still struggle to execute these sub-8-bit quantized networks effectively. In this paper, we propose Agile-Quant, an activation-guided quantization framework for popular Large Language Models (LLMs), and implement an end-to-end accelerator on multiple edge devices for faster inference. Considering the hardware profiling and activation analysis, we first introduce a basic activation quantization strategy to balance the trade-off of task performance and real inference speed. Then we leverage the activation-aware token pruning technique to reduce the outliers and the adverse impact on attentivity. Ultimately, we utilize the SIMD-based 4-bit multiplier and our efficient TRIP matrix multiplication to implement the accelerator for LLMs on the edge. We apply our framework on different scales of LLMs including LLaMA, OPT, and BLOOM with 4-bit or 8-bit for the activation and 4-bit for the weight quantization. Experiments show that Agile-Quant achieves simultaneous quantization of model weights and activations while maintaining task performance comparable to existing weight-only quantization methods. Moreover, in the 8- and 4-bit scenario, Agile-Quant achieves an on-device speedup of up to 2.55x compared to its FP16 counterparts across multiple edge devices, marking a pioneering advancement in this domain. Code: https://github.com/shawnricecake/agile-quant
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14775v1">gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Pipeline parallelism has emerged as a predominant approach for deploying large language models (LLMs) across distributed nodes, owing to its lower communication overhead compared to tensor parallelism. While demonstrating high throughput in request serving, pipeline parallelism often suffers from performance limitations caused by pipeline bubbles, which are primarily resulted from imbalanced computation delays across batches. Existing methods like Sarathi-Serve attempt to address this through hybrid scheduling of chunked prefill and decode tokens using a fixed token budget. However, such methods may experience significant fluctuations due to either insufficient prefill tokens or uneven distribution of decode tokens, ultimately leading to computational imbalance. To overcome these inefficiencies, we present gLLM, a globally balanced pipeline parallelism system incorporating Token Throttling to effectively mitigate the pipeline bubbles. Our Token Throttling mechanism is a fine-grained scheduling policy that independently regulates the quantities of prefill and decode tokens, thus enabling balanced computation by leveraging global information from the inference system. Specifically, for decode tokens, gLLM maintains near-consistent token count across processing batches. For prefill tokens, it dynamically adjusts batch sizes based on both total pending tokens and the memory utilization rates of key-value cache (KV cache). Furthermore, gLLM runtime adopts an asynchronous execution and message passing architecture specifically optimized for pipeline parallelism characteristics. Experimental evaluations with representative LLMs show that gLLM achieves significant performance improvements, delivering 11% to 398% higher maximum throughput compared to state-of-the-art pipeline or tensor parallelism systems, while simultaneously maintaining lower latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09723v2">AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents Amazon.com, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18780v3">Certifying Counterfactual Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Published at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce biased responses that can cause representational harms. However, conventional studies are insufficient to thoroughly evaluate biases across LLM responses for different demographic groups (a.k.a. counterfactual bias), as they do not scale to large number of inputs and do not provide guarantees. Therefore, we propose the first framework, LLMCert-B that certifies LLMs for counterfactual bias on distributions of prompts. A certificate consists of high-confidence bounds on the probability of unbiased LLM responses for any set of counterfactual prompts - prompts differing by demographic groups, sampled from a distribution. We illustrate counterfactual bias certification for distributions of counterfactual prompts created by applying prefixes sampled from prefix distributions, to a given set of prompts. We consider prefix distributions consisting random token sequences, mixtures of manual jailbreaks, and perturbations of jailbreaks in LLM's embedding space. We generate non-trivial certificates for SOTA LLMs, exposing their vulnerabilities over distributions of prompts generated from computationally inexpensive prefix distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15929v3">Certifying Knowledge Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in safety-critical systems where they provide answers based on in-context information derived from knowledge bases. As LLMs are increasingly envisioned as superhuman agents, their proficiency in knowledge comprehension-extracting relevant information and reasoning over it to answer questions, a key facet of human intelligence-becomes crucial. However, existing evaluations of LLMs on knowledge comprehension are typically conducted on small test sets, but these datasets represent only a tiny fraction of the vast number of possible queries. Simple empirical evaluations on these limited test sets raises concerns about the reliability and generalizability of the results. In this work, we introduce the first specification and certification framework for knowledge comprehension in LLMs, providing formal probabilistic guarantees for reliability. Instead of a fixed dataset, we design novel specifications that mathematically represent prohibitively large probability distributions of knowledge comprehension prompts with natural noise, using knowledge graphs. From these specifications, we generate quantitative certificates that offer high-confidence, tight bounds on the probability that a given LLM correctly answers any question drawn from the specification distribution. We apply our framework to certify SOTA LLMs in two domains: precision medicine and general question-answering. Our results reveal previously unrecognized vulnerabilities in SOTA LLMs due to natural noise in the prompts. Additionally, we establish performance hierarchies with formal guarantees among the SOTA LLMs, particularly in the context of precision medicine question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15476v1">From Reviews to Dialogues: Active Synthesis for Zero-Shot LLM-based Conversational Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Conversational recommender systems (CRS) typically require extensive domain-specific conversational datasets, yet high costs, privacy concerns, and data-collection challenges severely limit their availability. Although Large Language Models (LLMs) demonstrate strong zero-shot recommendation capabilities, practical applications often favor smaller, internally managed recommender models due to scalability, interpretability, and data privacy constraints, especially in sensitive or rapidly evolving domains. However, training these smaller models effectively still demands substantial domain-specific conversational data, which remains challenging to obtain. To address these limitations, we propose an active data augmentation framework that synthesizes conversational training data by leveraging black-box LLMs guided by active learning techniques. Specifically, our method utilizes publicly available non-conversational domain data, including item metadata, user reviews, and collaborative signals, as seed inputs. By employing active learning strategies to select the most informative seed samples, our approach efficiently guides LLMs to generate synthetic, semantically coherent conversational interactions tailored explicitly to the target domain. Extensive experiments validate that conversational data generated by our proposed framework significantly improves the performance of LLM-based CRS models, effectively addressing the challenges of building CRS in no- or low-resource scenarios.
    </details>
</div>
