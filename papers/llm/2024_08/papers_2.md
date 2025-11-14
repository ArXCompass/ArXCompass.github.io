# llm - 2024_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11431v1">Diagnosing and Remedying Knowledge Deficiencies in LLMs via Label-free Curricular Meaningful Learning</a></div>
    <div class="paper-meta">
      📅 2024-08-21
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are versatile and demonstrate impressive generalization ability by mining and learning information from extensive unlabeled text. However, they still exhibit reasoning mistakes, often stemming from knowledge deficiencies, which can affect their trustworthiness and reliability. Although users can provide diverse and comprehensive queries, obtaining sufficient and effective feedback is demanding. Furthermore, evaluating LLMs comprehensively with limited labeled samples is difficult. This makes it a challenge to diagnose and remedy the deficiencies of LLMs through rich label-free user queries. To tackle this challenge, we propose a label-free curricular meaningful learning framework (LaMer). LaMer first employs relative entropy to automatically diagnose and quantify the knowledge deficiencies of LLMs in a label-free setting. Next, to remedy the diagnosed knowledge deficiencies, we apply curricular meaningful learning: first, we adopt meaningful learning to adaptively synthesize augmentation data according to the severity of the deficiencies, and then design a curricular deficiency remedy strategy to remedy the knowledge deficiencies of LLMs progressively. Experiments show that LaMer efficiently and effectively diagnoses and remedies knowledge deficiencies in LLMs, improving various LLMs across seven out-of-distribution (OOD) reasoning and language understanding benchmarks, achieving comparable results to baselines with just 40\% training data. LaMer even surpasses methods that rely on labeled datasets for deficiency diagnosis. In application, our label-free method can offer an effective knowledge deficiency diagnostic tool for efficient LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11428v1">Migrating Existing Container Workload to Kubernetes -- LLM Based Approach and Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-08-21
      | 💬 submitted to ICSME 2024 Industry Track
    </div>
    <details class="paper-abstract">
      Although Kubernetes has become a widespread open-source system that automates the management of containerized applications, its complexity can be a significant barrier, particularly for application developers unfamiliar with it. One approach employs large language models (LLMs) to assist developers in generating Kubernetes manifests; however it is currently impossible to determine whether the output satisfies given specifications and is comprehensible. In this study, we proposed a benchmarking method for evaluating the effectiveness of LLMs in synthesizing manifests, using the Compose specification -- a standard widely adopted by application developers -- as input. The proposed benchmarking method revealed that LLMs generally produce accurate results that compensate for simple specification gaps. However, we also observed that inline comments for readability were often omitted, and completion accuracy was low for atypical inputs with unclear intentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11397v1">EAGLE: Elevating Geometric Reasoning through LLM-empowered Visual Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-08-21
    </div>
    <details class="paper-abstract">
      Multi-modal Large Language Models have recently experienced rapid developments and excel in various multi-modal tasks. However, they still struggle with mathematical geometric problem solving, which requires exceptional visual perception proficiency. Existing MLLMs mostly optimize the LLM backbone to acquire geometric reasoning capabilities, while rarely emphasizing improvements in visual comprehension. In this paper, we first investigate the visual perception performance of MLLMs when facing geometric diagrams. Our findings reveal that current MLLMs severely suffer from inaccurate geometric perception and hallucinations. To address these limitations, we propose EAGLE, a novel two-stage end-to-end visual enhancement MLLM framework designed to ElevAte Geometric reasoning through LLM-Empowered visual instruction tuning. Specifically, in the preliminary stage, we feed geometric image-caption pairs into our MLLM that contains a fully fine-tuning CLIP ViT and a frozen LLM, aiming to endow our model with basic geometric knowledge. In the subsequent advanced stage, we incorporate LoRA modules into the vision encoder and unfreeze the LLM backbone. This enables the model to leverage the inherent CoT rationales within question-answer pairs, guiding the MLLM to focus on nuanced visual cues and enhancing its overall perceptual capacity. Moreover, we optimize the cross-modal projector in both stages to foster adaptive visual-linguistic alignments. After the two-stage visual enhancement, we develop the geometry expert model EAGLE-7B. Extensive experiments on popular benchmarks demonstrate the effectiveness of our model. For example, on the GeoQA benchmark, EAGLE-7B not only surpasses the exemplary G-LLaVA 7B model by 2.9%, but also marginally outperforms the larger G-LLaVA 13B model. On the MathVista benchmark, EAGLE-7B achieves remarkable 3.8% improvements compared with the proprietary model GPT-4V.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11363v1">ProteinGPT: Multimodal LLM for Protein Property Prediction and Structure Understanding</a></div>
    <div class="paper-meta">
      📅 2024-08-21
      | 💬 19 pages, 9 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Understanding biological processes, drug development, and biotechnological advancements requires detailed analysis of protein structures and sequences, a task in protein research that is inherently complex and time-consuming when performed manually. To streamline this process, we introduce ProteinGPT, a state-of-the-art multi-modal protein chat system, that allows users to upload protein sequences and/or structures for comprehensive protein analysis and responsive inquiries. ProteinGPT seamlessly integrates protein sequence and structure encoders with linear projection layers for precise representation adaptation, coupled with a large language model (LLM) to generate accurate and contextually relevant responses. To train ProteinGPT, we construct a large-scale dataset of 132,092 proteins with annotations, and optimize the instruction-tuning process using GPT-4o. This innovative system ensures accurate alignment between the user-uploaded data and prompts, simplifying protein analysis. Experiments show that ProteinGPT can produce promising responses to proteins and their corresponding questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11334v1">BURExtract-Llama: An LLM for Clinical Concept Extraction in Breast Ultrasound Reports</a></div>
    <div class="paper-meta">
      📅 2024-08-21
      | 💬 This paper has been accepted as the oral paper for the HCHM workshop, ACM Multimedia 2024
    </div>
    <details class="paper-abstract">
      Breast ultrasound is essential for detecting and diagnosing abnormalities, with radiology reports summarizing key findings like lesion characteristics and malignancy assessments. Extracting this critical information is challenging due to the unstructured nature of these reports, with varied linguistic styles and inconsistent formatting. While proprietary LLMs like GPT-4 are effective, they are costly and raise privacy concerns when handling protected health information. This study presents a pipeline for developing an in-house LLM to extract clinical information from radiology reports. We first use GPT-4 to create a small labeled dataset, then fine-tune a Llama3-8B model on it. Evaluated on clinician-annotated reports, our model achieves an average F1 score of 84.6%, which is on par with GPT-4. Our findings demonstrate the feasibility of developing an in-house LLM that not only matches GPT-4's performance but also offers cost reductions and enhanced data privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11324v1">HITS: High-coverage LLM-based Unit Test Generation via Method Slicing</a></div>
    <div class="paper-meta">
      📅 2024-08-21
      | 💬 to be published in ASE 24' Research Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have behaved well in generating unit tests for Java projects. However, the performance for covering the complex focal methods within the projects is poor. Complex methods comprise many conditions and loops, requiring the test cases to be various enough to cover all lines and branches. However, existing test generation methods with LLMs provide the whole method-to-test to the LLM without assistance on input analysis. The LLM has difficulty inferring the test inputs to cover all conditions, resulting in missing lines and branches. To tackle the problem, we propose decomposing the focal methods into slices and asking the LLM to generate test cases slice by slice. Our method simplifies the analysis scope, making it easier for the LLM to cover more lines and branches in each slice. We build a dataset comprising complex focal methods collected from the projects used by existing state-of-the-art approaches. Our experiment results show that our method significantly outperforms current test case generation methods with LLMs and the typical SBST method Evosuite regarding both line and branch coverage scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11313v1">Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer</a></div>
    <div class="paper-meta">
      📅 2024-08-21
    </div>
    <details class="paper-abstract">
      Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11294v1">RedWhale: An Adapted Korean LLM Through Efficient Continual Pretraining</a></div>
    <div class="paper-meta">
      📅 2024-08-21
    </div>
    <details class="paper-abstract">
      The field of Natural Language Processing (NLP) has seen significant advancements with the development of Large Language Models (LLMs). However, much of this research remains focused on English, often overlooking low-resource languages like Korean. This oversight presents challenges due to the unique non-alphabetic token structure of Korean and the substantial memory and computational demands required for LLM training, which frequently lead to memory constraints and out-of-memory errors. To address these issues, we present RedWhale, a model specifically tailored for Korean language processing. RedWhale is developed using an efficient continual pretraining approach that includes a comprehensive Korean corpus preprocessing pipeline, a specialized tokenizer, an optimized model initialization technique, and a multistage pretraining strategy. These innovations collectively reduce training time and computational costs while maintaining high levels of accuracy and comprehension. By leveraging cross-lingual transfer learning, which exploits shared linguistic similarities across languages, RedWhale builds on English models to enhance Korean language processing. Experimental results demonstrate that RedWhale outperforms other leading models on Korean NLP benchmarks, including the Korean Balanced Evaluation of Significant Tasks (KoBEST), showing superior understanding and generation of Korean text. Furthermore, RedWhale showed no signs of convergence even after pretraining on 9.7 billion tokens, indicating the potential for further improvements with additional training. This work represents a significant advancement in bridging the linguistic divide, particularly in enhancing NLP capabilities for the Korean language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16276v2">Empowering LLMs with Pseudo-Untrimmed Videos for Audio-Visual Temporal Understanding</a></div>
    <div class="paper-meta">
      📅 2024-08-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in natural language and multimodal domains. By fine-tuning multimodal LLMs with temporal annotations from well-annotated datasets, e.g., dense video captioning datasets, their temporal understanding capacity in video-language tasks can be obtained. However, there is a notable lack of untrimmed audio-visual video datasets with precise temporal annotations for events. This deficiency hinders LLMs from learning the alignment between time, audio-visual events, and text tokens, thus impairing their ability to temporally localize audio-visual events in videos. To address this gap, we introduce PU-VALOR, a comprehensive audio-visual dataset comprising over 114,000 pseudo-untrimmed videos with detailed temporal annotations. PU-VALOR is derived from the large-scale but coarse-annotated audio-visual dataset VALOR, through a subtle method involving event-based video clustering, random temporal scaling, and permutation. By fine-tuning a multimodal LLM on PU-VALOR, we developed AVicuna, a model capable of aligning audio-visual events with temporal intervals and corresponding text tokens. AVicuna excels in temporal localization and time-aware dialogue capabilities. Our experiments demonstrate that AVicuna effectively handles temporal understanding in audio-visual videos and achieves state-of-the-art performance on open-ended video QA, audio-visual QA, and audio-visual event dense localization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11198v1">EPiC: Cost-effective Search-based Prompt Engineering of LLMs for Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 Submitted to TSE
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen increasing use in various software development tasks, especially in code generation. The most advanced recent methods attempt to incorporate feedback from code execution into prompts to help guide LLMs in generating correct code, in an iterative process. While effective, these methods could be costly and time-consuming due to numerous interactions with the LLM and the extensive token usage. To address this issue, we propose an alternative approach named Evolutionary Prompt Engineering for Code (EPiC), which leverages a lightweight evolutionary algorithm to evolve the original prompts toward better ones that produce high-quality code, with minimal interactions with LLM. Our evaluation against state-of-the-art (SOTA) LLM-based code generation models shows that EPiC outperforms all the baselines in terms of cost-effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11053v1">Revisiting VerilogEval: Newer LLMs, In-Context Learning, and Specification-to-RTL Tasks</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 This paper revisits and improves the benchmark first presented in arXiv:2309.07544. Seven pages, three figures
    </div>
    <details class="paper-abstract">
      The application of large-language models (LLMs) to digital hardware code generation is an emerging field. Most LLMs are primarily trained on natural language and software code. Hardware code, such as Verilog, represents only a small portion of the training data and few hardware benchmarks exist. To address this gap, the open-source VerilogEval benchmark was released in 2023, providing a consistent evaluation framework for LLMs on code completion tasks. It was tested on state-of-the-art models at the time including GPT-4. However, VerilogEval and other Verilog generation benchmarks lack failure analysis and, in present form, are not conducive to exploring prompting techniques. Also, since VerilogEval's release, both commercial and open-source models have seen continued development. In this work, we evaluate new commercial and open-source models of varying sizes against an improved VerilogEval benchmark suite. We enhance VerilogEval's infrastructure and dataset by automatically classifying failures, introduce new prompts for supporting in-context learning (ICL) examples, and extend the supported tasks to specification-to-RTL translation. We find a measurable improvement in commercial state-of-the-art models, with GPT-4 Turbo achieving a 59% pass rate on spec-to-RTL tasks. We also study the performance of open-source and domain-specific models that have emerged, and demonstrate that models can benefit substantially from ICL. We find that recently-released Llama 3.1 405B achieves a pass rate of 58%, effectively matching that of GPT-4 Turbo, and that the much smaller domain-specific RTL-Coder 6.7B models achieve an impressive 37% pass rate. However, prompt engineering is key to achieving good pass rates, and varies widely with model and task. A benchmark infrastructure that allows for prompt engineering and failure analysis is key to continued model development and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10830v2">Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 Accepted to KDD 2024 (Research Track)
    </div>
    <details class="paper-abstract">
      It is commonly perceived that fake news and real news exhibit distinct writing styles, such as the use of sensationalist versus objective language. However, we emphasize that style-related features can also be exploited for style-based attacks. Notably, the advent of powerful Large Language Models (LLMs) has empowered malicious actors to mimic the style of trustworthy news sources, doing so swiftly, cost-effectively, and at scale. Our analysis reveals that LLM-camouflaged fake news content significantly undermines the effectiveness of state-of-the-art text-based detectors (up to 38% decrease in F1 Score), implying a severe vulnerability to stylistic variations. To address this, we introduce SheepDog, a style-robust fake news detector that prioritizes content over style in determining news veracity. SheepDog achieves this resilience through (1) LLM-empowered news reframings that inject style diversity into the training process by customizing articles to match different styles; (2) a style-agnostic training scheme that ensures consistent veracity predictions across style-diverse reframings; and (3) content-focused veracity attributions that distill content-centric guidelines from LLMs for debunking fake news, offering supplementary cues and potential intepretability that assist veracity prediction. Extensive experiments on three real-world benchmarks demonstrate SheepDog's style robustness and adaptability to various backbones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00054v1">Automating Knowledge Discovery from Scientific Literature via LLMs: A Dual-Agent Approach with Progressive Ontology Prompting</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 in submission
    </div>
    <details class="paper-abstract">
      To address the challenge of automating knowledge discovery from a vast volume of literature, in this paper, we introduce a novel framework based on large language models (LLMs) that combines a progressive ontology prompting (POP) algorithm with a dual-agent system, named LLM-Duo, designed to enhance the automation of knowledge extraction from scientific articles. The POP algorithm utilizes a prioritized breadth-first search (BFS) across a predefined ontology to generate structured prompt templates and action orders, thereby guiding LLMs to discover knowledge in an automatic manner. Additionally, our LLM-Duo employs two specialized LLM agents: an explorer and an evaluator. These two agents work collaboratively and adversarially to enhance the reliability of the discovery and annotation processes. Experiments demonstrate that our method outperforms advanced baselines, enabling more accurate and complete annotations. To validate the effectiveness of our method in real-world scenarios, we employ our method in a case study of speech-language intervention discovery. Our method identifies 2,421 interventions from 64,177 research articles in the speech-language therapy domain. We curate these findings into a publicly accessible intervention knowledge base that holds significant potential to benefit the speech-language therapy community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09235v2">Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form Text</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) as chat assistants capable of generating human-like conversations has amplified the need for robust evaluation methods, particularly for open-ended tasks. Conventional metrics like BLEU and ROUGE, while useful, are increasingly inadequate for capturing the subtle semantics and contextual richness of such generative outputs. We propose a reference-guided verdict method that automates the evaluation process by leveraging multiple LLMs-as-judges. Through experiments on three open-ended question-answering tasks, we demonstrate that combining multiple LLMs-as-judges significantly improves the reliability and accuracy of evaluations, particularly in complex tasks where a single model might struggle. Our findings reveal a strong correlation with human evaluations, establishing our method as a viable and effective alternative to traditional metrics and human judgments, particularly in the context of LLM-based chat assistants where the complexity and diversity of responses challenge existing benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10886v1">Leveraging LLMs for the Quality Assurance of Software Requirements</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 Accepted for publication at the RE@Next! track of RE 2024
    </div>
    <details class="paper-abstract">
      Successful software projects depend on the quality of software requirements. Creating high-quality requirements is a crucial step toward successful software development. Effective support in this area can significantly reduce development costs and enhance the software quality. In this paper, we introduce and assess the capabilities of a Large Language Model (LLM) to evaluate the quality characteristics of software requirements according to the ISO 29148 standard. We aim to further improve the support of stakeholders engaged in requirements engineering (RE). We show how an LLM can assess requirements, explain its decision-making process, and examine its capacity to propose improved versions of requirements. We conduct a study with software engineers to validate our approach. Our findings emphasize the potential of LLMs for improving the quality of software requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10811v1">Beyond English-Centric LLMs: What Language Do Multilingual Language Models Think in?</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      In this study, we investigate whether non-English-centric LLMs, despite their strong performance, `think' in their respective dominant language: more precisely, `think' refers to how the representations of intermediate layers, when un-embedded into the vocabulary space, exhibit higher probabilities for certain dominant languages during generation. We term such languages as internal $\textbf{latent languages}$. We examine the latent language of three typical categories of models for Japanese processing: Llama2, an English-centric model; Swallow, an English-centric model with continued pre-training in Japanese; and LLM-jp, a model pre-trained on balanced English and Japanese corpora. Our empirical findings reveal that, unlike Llama2 which relies exclusively on English as the internal latent language, Japanese-specific Swallow and LLM-jp employ both Japanese and English, exhibiting dual internal latent languages. For any given target language, the model preferentially activates the latent language most closely related to it. In addition, we explore how intermediate layers respond to questions involving cultural conflicts between latent internal and target output languages. We further explore how the language identity shifts across layers while keeping consistent semantic meaning reflected in the intermediate layer representations. This study deepens the understanding of non-English-centric large language models, highlighting the intricate dynamics of language representation within their intermediate layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10794v1">Tapping in a Remote Vehicle's onboard LLM to Complement the Ego Vehicle's Field-of-View</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 50th Euromicro Conference Series on Software Engineering and Advanced Applications (SEAA) 2024 - WiP
    </div>
    <details class="paper-abstract">
      Today's advanced automotive systems are turning into intelligent Cyber-Physical Systems (CPS), bringing computational intelligence to their cyber-physical context. Such systems power advanced driver assistance systems (ADAS) that observe a vehicle's surroundings for their functionality. However, such ADAS have clear limitations in scenarios when the direct line-of-sight to surrounding objects is occluded, like in urban areas. Imagine now automated driving (AD) systems that ideally could benefit from other vehicles' field-of-view in such occluded situations to increase traffic safety if, for example, locations about pedestrians can be shared across vehicles. Current literature suggests vehicle-to-infrastructure (V2I) via roadside units (RSUs) or vehicle-to-vehicle (V2V) communication to address such issues that stream sensor or object data between vehicles. When considering the ongoing revolution in vehicle system architectures towards powerful, centralized processing units with hardware accelerators, foreseeing the onboard presence of large language models (LLMs) to improve the passengers' comfort when using voice assistants becomes a reality. We are suggesting and evaluating a concept to complement the ego vehicle's field-of-view (FOV) with another vehicle's FOV by tapping into their onboard LLM to let the machines have a dialogue about what the other vehicle ``sees''. Our results show that very recent versions of LLMs, such as GPT-4V and GPT-4o, understand a traffic situation to an impressive level of detail, and hence, they can be used even to spot traffic participants. However, better prompts are needed to improve the detection quality and future work is needed towards a standardised message interchange format between vehicles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04472v2">Can LLMs Beat Humans in Debating? A Dynamic Multi-agent Framework for Competitive Debate</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 12 pages (including appendix), 7 figures
    </div>
    <details class="paper-abstract">
      Competitive debate is a complex task of computational argumentation. Large Language Models (LLMs) suffer from hallucinations and lack competitiveness in this field. To address these challenges, we introduce Agent for Debate (Agent4Debate), a dynamic multi-agent framework based on LLMs designed to enhance their capabilities in competitive debate. Drawing inspiration from human behavior in debate preparation and execution, Agent4Debate employs a collaborative architecture where four specialized agents, involving Searcher, Analyzer, Writer, and Reviewer, dynamically interact and cooperate. These agents work throughout the debate process, covering multiple stages from initial research and argument formulation to rebuttal and summary. To comprehensively evaluate framework performance, we construct the Competitive Debate Arena, comprising 66 carefully selected Chinese debate motions. We recruit ten experienced human debaters and collect records of 200 debates involving Agent4Debate, baseline models, and humans. The evaluation employs the Debatrix automatic scoring system and professional human reviewers based on the established Debatrix-Elo and Human-Elo ranking. Experimental results indicate that the state-of-the-art Agent4Debate exhibits capabilities comparable to those of humans. Furthermore, ablation studies demonstrate the effectiveness of each component in the agent structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10746v1">Pluto and Charon: A Time and Memory Efficient Collaborative Edge AI Framework for Personal LLMs Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 Accepted by The 53rd International Conference on Parallel Processing (ICPP'24)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have unlocked a plethora of powerful applications at the network edge, such as intelligent personal assistants. Data privacy and security concerns have prompted a shift towards edge-based fine-tuning of personal LLMs, away from cloud reliance. However, this raises issues of computational intensity and resource scarcity, hindering training efficiency and feasibility. While current studies investigate parameter-efficient fine-tuning (PEFT) techniques to mitigate resource constraints, our analysis indicates that these techniques are not sufficiently resource-efficient for edge devices. To tackle these challenges, we propose Pluto and Charon (PAC), a time and memory efficient collaborative edge AI framework for personal LLMs fine-tuning. PAC breaks the resource wall of personal LLMs fine-tuning with a sophisticated algorithm-system co-design. (1) Algorithmically, PAC implements a personal LLMs fine-tuning technique that is efficient in terms of parameters, time, and memory. It utilizes Parallel Adapters to circumvent the need for a full backward pass through the LLM backbone. Additionally, an activation cache mechanism further streamlining the process by negating the necessity for repeated forward passes across multiple epochs. (2) Systematically, PAC leverages edge devices in close proximity, pooling them as a collective resource for in-situ personal LLMs fine-tuning, utilizing a hybrid data and pipeline parallelism to orchestrate distributed training. The use of the activation cache eliminates the need for forward pass through the LLM backbone,enabling exclusive fine-tuning of the Parallel Adapters using data parallelism. Extensive evaluation based on prototype implementation demonstrates that PAC remarkably outperforms state-of-the-art approaches, achieving up to 8.64x end-to-end speedup and up to 88.16% reduction in memory footprint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03891v2">AutoBench: Automatic Testbench Generation and Evaluation Using LLMs for HDL Design</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      In digital circuit design, testbenches constitute the cornerstone of simulation-based hardware verification. Traditional methodologies for testbench generation during simulation-based hardware verification still remain partially manual, resulting in inefficiencies in testing various scenarios and requiring expensive time from designers. Large Language Models (LLMs) have demonstrated their potential in automating the circuit design flow. However, directly applying LLMs to generate testbenches suffers from a low pass rate. To address this challenge, we introduce AutoBench, the first LLM-based testbench generator for digital circuit design, which requires only the description of the design under test (DUT) to automatically generate comprehensive testbenches. In AutoBench, a hybrid testbench structure and a self-checking system are realized using LLMs. To validate the generated testbenches, we also introduce an automated testbench evaluation framework to evaluate the quality of generated testbenches from multiple perspectives. Experimental results demonstrate that AutoBench achieves a 57% improvement in the testbench pass@1 ratio compared with the baseline that directly generates testbenches using LLMs. For 75 sequential circuits, AutoBench successfully has a 3.36 times testbench pass@1 ratio compared with the baseline. The source codes and experimental results are open-sourced at this link: https://github.com/AutoBench/AutoBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10646v1">Beneath the Surface of Consistency: Exploring Cross-lingual Knowledge Representation Sharing in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      The veracity of a factoid is largely independent of the language it is written in. However, language models are inconsistent in their ability to answer the same factual question across languages. This raises questions about how LLMs represent a given fact across languages. We explore multilingual factual knowledge through two aspects: the model's ability to answer a query consistently across languages, and the ability to ''store'' answers in a shared representation for several languages. We propose a methodology to measure the extent of representation sharing across languages by repurposing knowledge editing methods. We examine LLMs with various multilingual configurations using a new multilingual dataset. We reveal that high consistency does not necessarily imply shared representation, particularly for languages with different scripts. Moreover, we find that script similarity is a dominant factor in representation sharing. Finally, we observe that if LLMs could fully share knowledge across languages, their accuracy in their best-performing language could benefit an increase of up to 150\% on average. These findings highlight the need for improved multilingual knowledge representation in LLMs and suggest a path for the development of more robust and consistent multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10642v1">Minor SFT loss for LLM fine-tune to increase performance and reduce model deviation</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Instruct LLM provide a paradigm used in large scale language model to align LLM to human preference. The paradigm contains supervised fine tuning and reinforce learning from human feedback. This paradigm is also used in downstream scenarios to adapt LLM to specific corpora and applications. Comparing to SFT, there are many efforts focused on RLHF and several algorithms being proposed, such as PPO, DPO, IPO, KTO, MinorDPO and etc. Meanwhile most efforts for SFT are focused on how to collect, filter and mix high quality data. In this article with insight from DPO and MinorDPO, we propose a training metric for SFT to measure the discrepancy between the optimized model and the original model, and a loss function MinorSFT that can increase the training effectiveness, and reduce the discrepancy between the optimized LLM and original LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10573v1">Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 7 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant capabilities, particularly in the domain of question answering (QA). However, their effectiveness in QA is often undermined by the vagueness of user questions. To address this issue, we introduce single-round instance-level prompt optimization, referred to as question rewriter. By enhancing the intelligibility of human questions for black-box LLMs, our question rewriter improves the quality of generated answers. The rewriter is optimized using direct preference optimization based on feedback collected from automatic criteria for evaluating generated answers; therefore, its training does not require costly human annotations. The experiments across multiple black-box LLMs and long-form question answering (LFQA) datasets demonstrate the efficacy of our method. This paper provides a practical framework for training question rewriters and sets a precedent for future explorations in prompt optimization within LFQA tasks. Code is available at \url{https://github.com/3244we/Question-Rewriter}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09955v2">MegaAgent: A Practical Framework for Autonomous Cooperation in Large-Scale LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      With the emergence of large language models (LLMs), LLM-powered multi-agent systems (LLM-MA systems) have been proposed to tackle real-world tasks. However, their agents mostly follow predefined Standard Operating Procedures (SOPs) that remain unchanged across the whole interaction, lacking autonomy and scalability. Additionally, current solutions often overlook the necessity for effective agent cooperation. To address the above limitations, we propose MegaAgent, a practical framework designed for autonomous cooperation in large-scale LLM Agent systems. MegaAgent leverages the autonomy of agents to dynamically generate agents based on task requirements, incorporating features such as automatically dividing tasks, systematic planning and monitoring of agent activities, and managing concurrent operations. In addition, MegaAgent is designed with a hierarchical structure and employs system-level parallelism to enhance performance and boost communication. We demonstrate the effectiveness of MegaAgent through Gobang game development, showing that it outperforms popular LLM-MA systems; and national policy simulation, demonstrating its high autonomy and potential to rapidly scale up to 590 agents while ensuring effective cooperation among them. Our results indicate that MegaAgent is the first autonomous large-scale LLM-MA system with no pre-defined SOPs, high effectiveness and scalability, paving the way for further research in this field. Our code is at https://anonymous.4open.science/r/MegaAgent-81F3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00981v2">Bayesian Optimization with LLM-Based Acquisition Functions for Natural Language Preference Elicitation</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      Designing preference elicitation (PE) methodologies that can quickly ascertain a user's top item preferences in a cold-start setting is a key challenge for building effective and personalized conversational recommendation (ConvRec) systems. While large language models (LLMs) enable fully natural language (NL) PE dialogues, we hypothesize that monolithic LLM NL-PE approaches lack the multi-turn, decision-theoretic reasoning required to effectively balance the exploration and exploitation of user preferences towards an arbitrary item set. In contrast, traditional Bayesian optimization PE methods define theoretically optimal PE strategies, but cannot generate arbitrary NL queries or reason over content in NL item descriptions -- requiring users to express preferences via ratings or comparisons of unfamiliar items. To overcome the limitations of both approaches, we formulate NL-PE in a Bayesian Optimization (BO) framework that seeks to actively elicit NL feedback to identify the best recommendation. Key challenges in generalizing BO to deal with natural language feedback include determining: (a) how to leverage LLMs to model the likelihood of NL preference feedback as a function of item utilities, and (b) how to design an acquisition function for NL BO that can elicit preferences in the infinite space of language. We demonstrate our framework in a novel NL-PE algorithm, PEBOL, which uses: 1) Natural Language Inference (NLI) between user preference utterances and NL item descriptions to maintain Bayesian preference beliefs, and 2) BO strategies such as Thompson Sampling (TS) and Upper Confidence Bound (UCB) to steer LLM query generation. We numerically evaluate our methods in controlled simulations, finding that after 10 turns of dialogue, PEBOL can achieve an MRR@10 of up to 0.27 compared to the best monolithic LLM baseline's MRR@10 of 0.17, despite relying on earlier and smaller LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04863v5">SCLA: Automated Smart Contract Summarization via LLMs and Semantic Augmentation</a></div>
    <div class="paper-meta">
      📅 2024-08-20
    </div>
    <details class="paper-abstract">
      In the rapidly evolving world of blockchain systems, the efficient development and maintenance of smart contracts has become a critical task. Smart contract code summarization can significantly facilitate the maintenance of smart contracts and mitigate their vulnerabilities. Large Language Models (LLMs), such as GPT-4o and Gemini-1.5-Pro, possess the capability to generate code summarizations from code examples embedded in prompts. However, the performance of LLMs in code summarization remains suboptimal compared to fine-tuning-based models (e.g., CodeT5+, CodeBERT). Therefore, we propose SCLA, a framework leveraging LLMs and semantic augmentation to improve code summarization performance. SCLA constructs the smart contract's Abstract Syntax Tree (AST) to extract latent semantics, thereby forming a semantically augmented prompt. For evaluation, we utilize a large-scale dataset comprising 40,000 real-world contracts. Experimental results demonstrate that SCLA, with its enhanced prompt, significantly improves the quality of code summarizations. SCLA surpasses other state-of-the-art models (e.g., CodeBERT, CodeT5, and CodeT5+), achieving 37.53% BLEU-4, 52.54% METEOR, 56.97% ROUGE-L, and 63.44% BLEURT, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08808v3">Constructing Domain-Specific Evaluation Sets for LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 14 pages, 8 figures, Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized the landscape of machine learning, yet current benchmarks often fall short in capturing the diverse behavior of these models in real-world applications. A benchmark's usefulness is determined by its ability to clearly differentiate between models of varying capabilities (separability) and closely align with human preferences. Existing frameworks like Alpaca-Eval 2.0 LC \cite{dubois2024lengthcontrolledalpacaevalsimpleway} and Arena-Hard v0.1 \cite{li2024crowdsourced} are limited by their focus on general-purpose queries and lack of diversity across domains such as law, medicine, and multilingual contexts. In this paper, we address these limitations by introducing a novel data pipeline that curates diverse, domain-specific evaluation sets tailored for LLM-as-a-Judge frameworks. Our approach leverages a combination of manual curation, semi-supervised learning to generate clusters, and stratified sampling to ensure balanced representation across a wide range of domains and languages. The resulting evaluation set, which includes 1573 samples across 14 categories, demonstrates high separability (84\%) across ten top-ranked models, and agreement (84\%) with Chatbot Arena and (0.915) Spearman correlation. The agreement values are 9\% better than Arena Hard and 20\% better than AlpacaEval 2.0 LC, while the Spearman coefficient is 0.7 more than the next best benchmark, showcasing a significant improvement in the usefulness of the benchmark. We further provide an open-source evaluation tool that enables fine-grained analysis of model performance across user-defined categories, offering valuable insights for practitioners. This work contributes to the ongoing effort to enhance the transparency, diversity, and effectiveness of LLM evaluation methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13712v3">Impact of Guidance and Interaction Strategies for LLM Use on Learner Performance and Perception</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 To appear in CSCW 2024
    </div>
    <details class="paper-abstract">
      Personalized chatbot-based teaching assistants can be crucial in addressing increasing classroom sizes, especially where direct teacher presence is limited. Large language models (LLMs) offer a promising avenue, with increasing research exploring their educational utility. However, the challenge lies not only in establishing the efficacy of LLMs but also in discerning the nuances of interaction between learners and these models, which impact learners' engagement and results. We conducted a formative study in an undergraduate computer science classroom (N=145) and a controlled experiment on Prolific (N=356) to explore the impact of four pedagogically informed guidance strategies on the learners' performance, confidence and trust in LLMs. Direct LLM answers marginally improved performance, while refining student solutions fostered trust. Structured guidance reduced random queries as well as instances of students copy-pasting assignment questions to the LLM. Our work highlights the role that teachers can play in shaping LLM-supported learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.20224v3">Can Editing LLMs Inject Harm?</a></div>
    <div class="paper-meta">
      📅 2024-08-20
      | 💬 The first two authors contributed equally. 9 pages for main paper, 36 pages including appendix. The code, results, dataset for this paper and more resources are on the project website: https://llm-editing.github.io
    </div>
    <details class="paper-abstract">
      Knowledge editing has been increasingly adopted to correct the false or outdated knowledge in Large Language Models (LLMs). Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10428v1">Are LLMs Any Good for High-Level Synthesis?</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 ICCAD '24 Special Session on AI4HLS: New Frontiers in High-Level Synthesis Augmented with Artificial Intelligence
    </div>
    <details class="paper-abstract">
      The increasing complexity and demand for faster, energy-efficient hardware designs necessitate innovative High-Level Synthesis (HLS) methodologies. This paper explores the potential of Large Language Models (LLMs) to streamline or replace the HLS process, leveraging their ability to understand natural language specifications and refactor code. We survey the current research and conduct experiments comparing Verilog designs generated by a standard HLS tool (Vitis HLS) with those produced by LLMs translating C code or natural language specifications. Our evaluation focuses on quantifying the impact on performance, power, and resource utilization, providing an assessment of the efficiency of LLM-based approaches. This study aims to illuminate the role of LLMs in HLS, identifying promising directions for optimized hardware design in applications such as AI acceleration, embedded systems, and high-performance computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10365v1">AI-Driven Review Systems: Evaluating LLMs in Scalable and Bias-Aware Academic Reviews</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Automatic reviewing helps handle a large volume of papers, provides early feedback and quality control, reduces bias, and allows the analysis of trends. We evaluate the alignment of automatic paper reviews with human reviews using an arena of human preferences by pairwise comparisons. Gathering human preference may be time-consuming; therefore, we also use an LLM to automatically evaluate reviews to increase sample efficiency while reducing bias. In addition to evaluating human and LLM preferences among LLM reviews, we fine-tune an LLM to predict human preferences, predicting which reviews humans will prefer in a head-to-head battle between LLMs. We artificially introduce errors into papers and analyze the LLM's responses to identify limitations, use adaptive review questions, meta prompting, role-playing, integrate visual and textual analysis, use venue-specific reviewing materials, and predict human preferences, improving upon the limitations of the traditional review processes. We make the reviews of publicly available arXiv and open-access Nature journal papers available online, along with a free service which helps authors review and revise their research papers and improve their quality. This work develops proof-of-concept LLM reviewing systems that quickly deliver consistent, high-quality reviews and evaluate their quality. We mitigate the risks of misuse, inflated review scores, overconfident ratings, and skewed score distributions by augmenting the LLM with multiple documents, including the review form, reviewer guide, code of ethics and conduct, area chair guidelines, and previous year statistics, by finding which errors and shortcomings of the paper may be detected by automated reviews, and evaluating pairwise reviewer preferences. This work identifies and addresses the limitations of using LLMs as reviewers and evaluators and enhances the quality of the reviewing process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11715v2">Semantic Trajectory Data Mining with LLM-Informed POI Classification</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 7 pages, accepted for the 27th IEEE International Conference on Intelligent Transportation Systems (ITSC 2024)
    </div>
    <details class="paper-abstract">
      Human travel trajectory mining is crucial for transportation systems, enhancing route optimization, traffic management, and the study of human travel patterns. Previous rule-based approaches without the integration of semantic information show a limitation in both efficiency and accuracy. Semantic information, such as activity types inferred from Points of Interest (POI) data, can significantly enhance the quality of trajectory mining. However, integrating these insights is challenging, as many POIs have incomplete feature information, and current learning-based POI algorithms require the integrity of datasets to do the classification. In this paper, we introduce a novel pipeline for human travel trajectory mining. Our approach first leverages the strong inferential and comprehension capabilities of large language models (LLMs) to annotate POI with activity types and then uses a Bayesian-based algorithm to infer activity for each stay point in a trajectory. In our evaluation using the OpenStreetMap (OSM) POI dataset, our approach achieves a 93.4% accuracy and a 96.1% F-1 score in POI classification, and a 91.7% accuracy with a 92.3% F-1 score in activity inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.02138v3">Topic-Based Watermarks for LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 Results for proposed scheme, additional/removal of content (figures and equations), 12 pages
    </div>
    <details class="paper-abstract">
      The indistinguishability of text generated by large language models (LLMs) from human-generated text poses significant challenges. Watermarking algorithms are potential solutions by embedding detectable signatures within LLM-generated outputs. However, current watermarking schemes lack robustness to a range of attacks such as text substitution or manipulation, undermining their reliability. This paper proposes a novel topic-based watermarking algorithm for LLMs, designed to enhance the robustness of watermarking in LLMs. Our approach leverages the topics extracted from input prompts or outputs of non-watermarked LLMs in the generation process of watermarked text. We dynamically utilize token lists on identified topics and adjust token sampling weights accordingly. By using these topic-specific token biases, we embed a topic-sensitive watermarking into the generated text. We outline the theoretical framework of our topic-based watermarking algorithm and discuss its potential advantages in various scenarios. Additionally, we explore a comprehensive range of attacks against watermarking algorithms, including discrete alterations, paraphrasing, and tokenizations. We demonstrate that our proposed watermarking scheme classifies various watermarked text topics with 99.99% confidence and outperforms existing algorithms in terms of z-score robustness and the feasibility of modeling text degradation by potential attackers, while considering the trade-offs between the benefits and losses of watermarking LLM-generated text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06009v3">Detectors for Safe and Reliable LLMs: Implementations, Uses, and Limitations</a></div>
    <div class="paper-meta">
      📅 2024-08-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are susceptible to a variety of risks, from non-faithful output to biased and toxic generations. Due to several limiting factors surrounding LLMs (training cost, API access, data availability, etc.), it may not always be feasible to impose direct safety constraints on a deployed model. Therefore, an efficient and reliable alternative is required. To this end, we present our ongoing efforts to create and deploy a library of detectors: compact and easy-to-build classification models that provide labels for various harms. In addition to the detectors themselves, we discuss a wide range of uses for these detector models - from acting as guardrails to enabling effective AI governance. We also deep dive into inherent challenges in their development and discuss future work aimed at making the detectors more reliable and broadening their scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09946v1">Microscopic Analysis on LLM players via Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 Under review, 10 pages
    </div>
    <details class="paper-abstract">
      Recent studies have begun developing autonomous game players for social deduction games using large language models (LLMs). When building LLM players, fine-grained evaluations are crucial for addressing weaknesses in game-playing abilities. However, existing studies have often overlooked such assessments. Specifically, we point out two issues with the evaluation methods employed. First, game-playing abilities have typically been assessed through game-level outcomes rather than specific event-level skills; Second, error analyses have lacked structured methodologies. To address these issues, we propose an approach utilizing a variant of the SpyFall game, named SpyGame. We conducted an experiment with four LLMs, analyzing their gameplay behavior in SpyGame both quantitatively and qualitatively. For the quantitative analysis, we introduced eight metrics to resolve the first issue, revealing that these metrics are more effective than existing ones for evaluating the two critical skills: intent identification and camouflage. In the qualitative analysis, we performed thematic analysis to resolve the second issue. This analysis identifies four major categories that affect gameplay of LLMs. Additionally, we demonstrate how these categories complement and support the findings from the quantitative analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14362v4">Less but Better: Enabling Generalized Zero-shot Learning Towards Unseen Domains by Intrinsic Learning from Redundant LLM Semantics</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Generalized zero-shot learning (GZSL) focuses on recognizing seen and unseen classes against domain shift problem (DSP) where data of unseen classes may be misclassified as seen classes. However, existing GZSL is still limited to seen domains. In the current work, we pioneer cross-domain GZSL (CDGZSL) which addresses GZSL towards unseen domains. Different from existing GZSL methods which alleviate DSP by generating features of unseen classes with semantics, CDGZSL needs to construct a common feature space across domains and acquire the corresponding intrinsic semantics shared among domains to transfer from seen to unseen domains. Considering the information asymmetry problem caused by redundant class semantics annotated with large language models (LLMs), we present Meta Domain Alignment Semantic Refinement (MDASR). Technically, MDASR consists of two parts: Inter-class Similarity Alignment (ISA), which eliminates the non-intrinsic semantics not shared across all domains under the guidance of inter-class feature relationships, and Unseen-class Meta Generation (UMG), which preserves intrinsic semantics to maintain connectivity between seen and unseen classes by simulating feature generation. MDASR effectively aligns the redundant semantic space with the common feature space, mitigating the information asymmetry in CDGZSL. The effectiveness of MDASR is demonstrated on the Office-Home and Mini-DomainNet, and we have shared the LLM-based semantics for these datasets as the benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13213v2">Multi-Meta-RAG: Improving RAG for Multi-Hop Queries using Database Filtering with LLM-Extracted Metadata</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 Accepted to ICTERI 2024 Posters Track
    </div>
    <details class="paper-abstract">
      The retrieval-augmented generation (RAG) enables retrieval of relevant information from an external knowledge source and allows large language models (LLMs) to answer queries over previously unseen document collections. However, it was demonstrated that traditional RAG applications perform poorly in answering multi-hop questions, which require retrieving and reasoning over multiple elements of supporting evidence. We introduce a new method called Multi-Meta-RAG, which uses database filtering with LLM-extracted metadata to improve the RAG selection of the relevant documents from various sources, relevant to the question. While database filtering is specific to a set of questions from a particular domain and format, we found out that Multi-Meta-RAG greatly improves the results on the MultiHop-RAG benchmark. The code is available at https://github.com/mxpoliakov/Multi-Meta-RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02889v3">InterrogateLLM: Zero-Resource Hallucination Detection in LLM-Generated Answers</a></div>
    <div class="paper-meta">
      📅 2024-08-19
    </div>
    <details class="paper-abstract">
      Despite the many advances of Large Language Models (LLMs) and their unprecedented rapid evolution, their impact and integration into every facet of our daily lives is limited due to various reasons. One critical factor hindering their widespread adoption is the occurrence of hallucinations, where LLMs invent answers that sound realistic, yet drift away from factual truth. In this paper, we present a novel method for detecting hallucinations in large language models, which tackles a critical issue in the adoption of these models in various real-world scenarios. Through extensive evaluations across multiple datasets and LLMs, including Llama-2, we study the hallucination levels of various recent LLMs and demonstrate the effectiveness of our method to automatically detect them. Notably, we observe up to 87% hallucinations for Llama-2 in a specific experiment, where our method achieves a Balanced Accuracy of 81%, all without relying on external knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09757v1">Strategic Demonstration Selection for Improved Fairness in LLM In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2024-08-19
    </div>
    <details class="paper-abstract">
      Recent studies highlight the effectiveness of using in-context learning (ICL) to steer large language models (LLMs) in processing tabular data, a challenging task given the structured nature of such data. Despite advancements in performance, the fairness implications of these methods are less understood. This study investigates how varying demonstrations within ICL prompts influence the fairness outcomes of LLMs. Our findings reveal that deliberately including minority group samples in prompts significantly boosts fairness without sacrificing predictive accuracy. Further experiments demonstrate that the proportion of minority to majority samples in demonstrations affects the trade-off between fairness and prediction accuracy. Based on these insights, we introduce a mitigation technique that employs clustering and evolutionary strategies to curate a diverse and representative sample set from the training data. This approach aims to enhance both predictive performance and fairness in ICL applications. Experimental results validate that our proposed method dramatically improves fairness across various metrics, showing its efficacy in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09742v1">Paired Completion: Flexible Quantification of Issue-framing at Scale with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Detecting and quantifying issue framing in textual discourse - the perspective one takes to a given topic (e.g. climate science vs. denialism, misogyny vs. gender equality) - is highly valuable to a range of end-users from social and political scientists to program evaluators and policy analysts. However, conceptual framing is notoriously challenging for automated natural language processing (NLP) methods since the words and phrases used by either `side' of an issue are often held in common, with only subtle stylistic flourishes separating their use. Here we develop and rigorously evaluate new detection methods for issue framing and narrative analysis within large text datasets. By introducing a novel application of next-token log probabilities derived from generative large language models (LLMs) we show that issue framing can be reliably and efficiently detected in large corpora with only a few examples of either perspective on a given issue, a method we call `paired completion'. Through 192 independent experiments over three novel, synthetic datasets, we evaluate paired completion against prompt-based LLM methods and labelled methods using traditional NLP and recent LLM contextual embeddings. We additionally conduct a cost-based analysis to mark out the feasible set of performant methods at production-level scales, and a model bias analysis. Together, our work demonstrates a feasible path to scalable, accurate and low-bias issue-framing in large corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09701v1">Bridging the Language Gap: Enhancing Multilingual Prompt-Based Code Generation in LLMs via Zero-Shot Cross-Lingual Transfer</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) for program code generation has gained substantial attention, but their biases and limitations with non-English prompts challenge global inclusivity. This paper investigates the complexities of multilingual prompt-based code generation. Our evaluations of LLMs, including CodeLLaMa and CodeGemma, reveal significant disparities in code quality for non-English prompts; we also demonstrate the inadequacy of simple approaches like prompt translation, bootstrapped data augmentation, and fine-tuning. To address this, we propose a zero-shot cross-lingual approach using a neural projection technique, integrating a cross-lingual encoder like LASER artetxe2019massively to map multilingual embeddings from it into the LLM's token space. This method requires training only on English data and scales effectively to other languages. Results on a translated and quality-checked MBPP dataset show substantial improvements in code quality. This research promotes a more inclusive code generation landscape by empowering LLMs with multilingual capabilities to support the diverse linguistic spectrum in programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08694v3">TeaMs-RL: Teaching LLMs to Generate Better Instruction Datasets via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-08-19
    </div>
    <details class="paper-abstract">
      The development of Large Language Models (LLMs) often confronts challenges stemming from the heavy reliance on human annotators in the reinforcement learning with human feedback (RLHF) framework, or the frequent and costly external queries tied to the self-instruct paradigm. In this work, we pivot to Reinforcement Learning (RL) -- but with a twist. Diverging from the typical RLHF, which refines LLMs following instruction data training, we use RL to directly generate the foundational instruction dataset that alone suffices for fine-tuning. Our method, TeaMs-RL, uses a suite of textual operations and rules, prioritizing the diversification of training datasets. It facilitates the generation of high-quality data without excessive reliance on external advanced models, paving the way for a single fine-tuning step and negating the need for subsequent RLHF stages. Our findings highlight key advantages of our approach: reduced need for human involvement and fewer model queries (only $5.73\%$ of the strong baseline's total), along with enhanced capabilities of LLMs in crafting and comprehending complex instructions compared to strong baselines, and substantially improved model privacy protection. Code is available at the link: https://github.com/SafeRL-Lab/TeaMs-RL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21369v2">An LLM-based Readability Measurement for Unit Tests' Context-aware Inputs</a></div>
    <div class="paper-meta">
      📅 2024-08-19
    </div>
    <details class="paper-abstract">
      Automated test techniques usually generate unit tests with higher code coverage than manual tests. However, the readability of automated tests is crucial for code comprehension and maintenance. The readability of unit tests involves many aspects. In this paper, we focus on test inputs. The central limitation of existing studies on input readability is that they focus on test codes alone without taking the tested source codes into consideration, making them either ignore different source codes' different readability requirements or require manual efforts to write readable inputs. However, we observe that the source codes specify the contexts that test inputs must satisfy. Based on such observation, we introduce the \underline{C}ontext \underline{C}onsistency \underline{C}riterion (a.k.a, C3), which is a readability measurement tool that leverages Large Language Models to extract primitive-type (including string-type) parameters' readability contexts from the source codes and checks whether test inputs are consistent with those contexts. We have also proposed EvoSuiteC3. It leverages C3's extracted contexts to help EvoSuite generate readable test inputs. We have evaluated C3's performance on $409$ \java{} classes and compared manual and automated tests' readability under C3 measurement. The results are two-fold. First, The Precision, Recall, and F1-Score of C3's mined readability contexts are \precision{}, \recall{}, and \fone{}, respectively. Second, under C3's measurement, the string-type input readability scores of EvoSuiteC3, ChatUniTest (an LLM-based test generation tool), manual tests, and two traditional tools (EvoSuite and Randoop) are $90\%$, $83\%$, $68\%$, $8\%$, and $8\%$, showing the traditional tools' inability in generating readable string-type inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09639v1">How to Make the Most of LLMs' Grammatical Knowledge for Acceptability Judgments</a></div>
    <div class="paper-meta">
      📅 2024-08-19
    </div>
    <details class="paper-abstract">
      The grammatical knowledge of language models (LMs) is often measured using a benchmark of linguistic minimal pairs, where LMs are presented with a pair of acceptable and unacceptable sentences and required to judge which is acceptable. The existing dominant approach, however, naively calculates and compares the probabilities of paired sentences using LMs. Additionally, large language models (LLMs) have yet to be thoroughly examined in this field. We thus investigate how to make the most of LLMs' grammatical knowledge to comprehensively evaluate it. Through extensive experiments of nine judgment methods in English and Chinese, we demonstrate that a probability readout method, in-template LP, and a prompting-based method, Yes/No probability computing, achieve particularly high performance, surpassing the conventional approach. Our analysis reveals their different strengths, e.g., Yes/No probability computing is robust against token-length bias, suggesting that they harness different aspects of LLMs' grammatical knowledge. Consequently, we recommend using diverse judgment methods to evaluate LLMs comprehensively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09629v1">A Strategy to Combine 1stGen Transformers and Open LLMs for Automatic Text Classification</a></div>
    <div class="paper-meta">
      📅 2024-08-19
      | 💬 13 pages, 3 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Transformer models have achieved state-of-the-art results, with Large Language Models (LLMs), an evolution of first-generation transformers (1stTR), being considered the cutting edge in several NLP tasks. However, the literature has yet to conclusively demonstrate that LLMs consistently outperform 1stTRs across all NLP tasks. This study compares three 1stTRs (BERT, RoBERTa, and BART) with two open LLMs (Llama 2 and Bloom) across 11 sentiment analysis datasets. The results indicate that open LLMs may moderately outperform or match 1stTRs in 8 out of 11 datasets but only when fine-tuned. Given this substantial cost for only moderate gains, the practical applicability of these models in cost-sensitive scenarios is questionable. In this context, a confidence-based strategy that seamlessly integrates 1stTRs with open LLMs based on prediction certainty is proposed. High-confidence documents are classified by the more cost-effective 1stTRs, while uncertain cases are handled by LLMs in zero-shot or few-shot modes, at a much lower cost than fine-tuned versions. Experiments in sentiment analysis demonstrate that our solution not only outperforms 1stTRs, zero-shot, and few-shot LLMs but also competes closely with fine-tuned LLMs at a fraction of the cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06643v3">Effects of diversity incentives on sample diversity and downstream model performance in LLM-based text augmentation</a></div>
    <div class="paper-meta">
      📅 2024-08-18
      | 💬 ACL'24 version, 24 pages
    </div>
    <details class="paper-abstract">
      The latest generative large language models (LLMs) have found their application in data augmentation tasks, where small numbers of text samples are LLM-paraphrased and then used to fine-tune downstream models. However, more research is needed to assess how different prompts, seed data selection strategies, filtering methods, or model settings affect the quality of paraphrased data (and downstream models). In this study, we investigate three text diversity incentive methods well established in crowdsourcing: taboo words, hints by previous outlier solutions, and chaining on previous outlier solutions. Using these incentive methods as part of instructions to LLMs augmenting text datasets, we measure their effects on generated texts lexical diversity and downstream model performance. We compare the effects over 5 different LLMs, 6 datasets and 2 downstream models. We show that diversity is most increased by taboo words, but downstream model performance is highest with hints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10450v2">TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-08-18
      | 💬 Submitted to IEEE TKDE. Our code and dataset will be made available upon acceptance of the paper
    </div>
    <details class="paper-abstract">
      There is a growing interest in utilizing large-scale language models (LLMs) to advance next-generation Recommender Systems (RecSys), driven by their outstanding language understanding and in-context learning capabilities. In this scenario, tokenizing (i.e., indexing) users and items becomes essential for ensuring a seamless alignment of LLMs with recommendations. While several studies have made progress in representing users and items through textual contents or latent representations, challenges remain in efficiently capturing high-order collaborative knowledge into discrete tokens that are compatible with LLMs. Additionally, the majority of existing tokenization approaches often face difficulties in generalizing effectively to new/unseen users or items that were not in the training corpus. To address these challenges, we propose a novel framework called TokenRec, which introduces not only an effective ID tokenization strategy but also an efficient retrieval paradigm for LLM-based recommendations. Specifically, our tokenization strategy, Masked Vector-Quantized (MQ) Tokenizer, involves quantizing the masked user/item representations learned from collaborative filtering into discrete tokens, thus achieving a smooth incorporation of high-order collaborative knowledge and a generalizable tokenization of users and items for LLM-based RecSys. Meanwhile, our generative retrieval paradigm is designed to efficiently recommend top-$K$ items for users to eliminate the need for the time-consuming auto-regressive decoding and beam search processes used by LLMs, thus significantly reducing inference time. Comprehensive experiments validate the effectiveness of the proposed methods, demonstrating that TokenRec outperforms competitive benchmarks, including both traditional recommender systems and emerging LLM-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09326v1">Characterizing and Evaluating the Reliability of LLMs against Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2024-08-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have increasingly become pivotal in content generation with notable societal impact. These models hold the potential to generate content that could be deemed harmful.Efforts to mitigate this risk include implementing safeguards to ensure LLMs adhere to social ethics.However, despite such measures, the phenomenon of "jailbreaking" -- where carefully crafted prompts elicit harmful responses from models -- persists as a significant challenge. Recognizing the continuous threat posed by jailbreaking tactics and their repercussions for the trustworthy use of LLMs, a rigorous assessment of the models' robustness against such attacks is essential. This study introduces an comprehensive evaluation framework and conducts an large-scale empirical experiment to address this need. We concentrate on 10 cutting-edge jailbreak strategies across three categories, 1525 questions from 61 specific harmful categories, and 13 popular LLMs. We adopt multi-dimensional metrics such as Attack Success Rate (ASR), Toxicity Score, Fluency, Token Length, and Grammatical Errors to thoroughly assess the LLMs' outputs under jailbreak. By normalizing and aggregating these metrics, we present a detailed reliability score for different LLMs, coupled with strategic recommendations to reduce their susceptibility to such vulnerabilities. Additionally, we explore the relationships among the models, attack strategies, and types of harmful content, as well as the correlations between the evaluation metrics, which proves the validity of our multifaceted evaluation framework. Our extensive experimental results demonstrate a lack of resilience among all tested LLMs against certain strategies, and highlight the need to concentrate on the reliability facets of LLMs. We believe our study can provide valuable insights into enhancing the security evaluation of LLMs against jailbreak within the domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09304v1">CyberPal.AI: Empowering LLMs with Expert-Driven Cybersecurity Instructions</a></div>
    <div class="paper-meta">
      📅 2024-08-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing (NLP), providing versatile capabilities across various applications. However, their application to complex, domain-specific tasks, such as cyber-security, often faces substantial challenges. In this study, we introduce SecKnowledge and CyberPal.AI to address these challenges and train security-expert LLMs. SecKnowledge is a domain-knowledge-driven cyber-security instruction dataset, meticulously designed using years of accumulated expert knowledge in the domain through a multi-phase generation process. CyberPal.AI refers to a family of LLMs fine-tuned using SecKnowledge, aimed at building security-specialized LLMs capable of answering and following complex security-related instructions. Additionally, we introduce SecKnowledge-Eval, a comprehensive and diverse cyber-security evaluation benchmark, composed of an extensive set of cyber-security tasks we specifically developed to assess LLMs in the field of cyber-security, along with other publicly available security benchmarks. Our results show a significant average improvement of up to 24% over the baseline models, underscoring the benefits of our expert-driven instruction dataset generation process. These findings contribute to the advancement of AI-based cyber-security applications, paving the way for security-expert LLMs that can enhance threat-hunting and investigation processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11865v1">How Susceptible are LLMs to Influence in Prompts?</a></div>
    <div class="paper-meta">
      📅 2024-08-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are highly sensitive to prompts, including additional context provided therein. As LLMs grow in capability, understanding their prompt-sensitivity becomes increasingly crucial for ensuring reliable and robust performance, particularly since evaluating these models becomes more challenging. In this work, we investigate how current models (Llama, Mixtral, Falcon) respond when presented with additional input from another model, mimicking a scenario where a more capable model -- or a system with access to more external information -- provides supplementary information to the target model. Across a diverse spectrum of question-answering tasks, we study how an LLM's response to multiple-choice questions changes when the prompt includes a prediction and explanation from another model. Specifically, we explore the influence of the presence of an explanation, the stated authoritativeness of the source, and the stated confidence of the supplementary input. Our findings reveal that models are strongly influenced, and when explanations are provided they are swayed irrespective of the quality of the explanation. The models are more likely to be swayed if the input is presented as being authoritative or confident, but the effect is small in size. This study underscores the significant prompt-sensitivity of LLMs and highlights the potential risks of incorporating outputs from external sources without thorough scrutiny and further validation. As LLMs continue to advance, understanding and mitigating such sensitivities will be crucial for their reliable and trustworthy deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11863v1">Unraveling Text Generation in LLMs: A Stochastic Differential Equation Approach</a></div>
    <div class="paper-meta">
      📅 2024-08-17
    </div>
    <details class="paper-abstract">
      This paper explores the application of Stochastic Differential Equations (SDE) to interpret the text generation process of Large Language Models (LLMs) such as GPT-4. Text generation in LLMs is modeled as a stochastic process where each step depends on previously generated content and model parameters, sampling the next word from a vocabulary distribution. We represent this generation process using SDE to capture both deterministic trends and stochastic perturbations. The drift term describes the deterministic trends in the generation process, while the diffusion term captures the stochastic variations. We fit these functions using neural networks and validate the model on real-world text corpora. Through numerical simulations and comprehensive analyses, including drift and diffusion analysis, stochastic process property evaluation, and phase space exploration, we provide deep insights into the dynamics of text generation. This approach not only enhances the understanding of the inner workings of LLMs but also offers a novel mathematical perspective on language generation, which is crucial for diagnosing, optimizing, and controlling the quality of generated text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14812v2">As an AI Language Model, "Yes I Would Recommend Calling the Police": Norm Inconsistency in LLM Decision-Making</a></div>
    <div class="paper-meta">
      📅 2024-08-17
      | 💬 To appear in the proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES 2024)
    </div>
    <details class="paper-abstract">
      We investigate the phenomenon of norm inconsistency: where LLMs apply different norms in similar situations. Specifically, we focus on the high-risk application of deciding whether to call the police in Amazon Ring home surveillance videos. We evaluate the decisions of three state-of-the-art LLMs -- GPT-4, Gemini 1.0, and Claude 3 Sonnet -- in relation to the activities portrayed in the videos, the subjects' skin-tone and gender, and the characteristics of the neighborhoods where the videos were recorded. Our analysis reveals significant norm inconsistencies: (1) a discordance between the recommendation to call the police and the actual presence of criminal activity, and (2) biases influenced by the racial demographics of the neighborhoods. These results highlight the arbitrariness of model decisions in the surveillance context and the limitations of current bias detection and mitigation strategies in normative decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09199v1">TC-RAG:Turing-Complete RAG's Case study on Medical LLM Systems</a></div>
    <div class="paper-meta">
      📅 2024-08-17
      | 💬 version 1.0
    </div>
    <details class="paper-abstract">
      In the pursuit of enhancing domain-specific Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) emerges as a promising solution to mitigate issues such as hallucinations, outdated knowledge, and limited expertise in highly specialized queries. However, existing approaches to RAG fall short by neglecting system state variables, which are crucial for ensuring adaptive control, retrieval halting, and system convergence. In this paper, we introduce the TC-RAG through rigorous proof, a novel framework that addresses these challenges by incorporating a Turing Complete System to manage state variables, thereby enabling more efficient and accurate knowledge retrieval. By leveraging a memory stack system with adaptive retrieval, reasoning, and planning capabilities, TC-RAG not only ensures the controlled halting of retrieval processes but also mitigates the accumulation of erroneous knowledge via Push and Pop actions. In the case study of the medical domain, our extensive experiments on real-world healthcare datasets demonstrate the superiority of TC-RAG over existing methods in accuracy by over 7.20\%. Our dataset and code have been available at https://https://github.com/Artessay/SAMA.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09176v1">Cognitive LLMs: Towards Integrating Cognitive Architectures and Large Language Models for Manufacturing Decision-making</a></div>
    <div class="paper-meta">
      📅 2024-08-17
      | 💬 20 pages, 8 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Resolving the dichotomy between the human-like yet constrained reasoning processes of Cognitive Architectures and the broad but often noisy inference behavior of Large Language Models (LLMs) remains a challenging but exciting pursuit, for enabling reliable machine reasoning capabilities in production systems. Because Cognitive Architectures are famously developed for the purpose of modeling the internal mechanisms of human cognitive decision-making at a computational level, new investigations consider the goal of informing LLMs with the knowledge necessary for replicating such processes, e.g., guided perception, memory, goal-setting, and action. Previous approaches that use LLMs for grounded decision-making struggle with complex reasoning tasks that require slower, deliberate cognition over fast and intuitive inference -- reporting issues related to the lack of sufficient grounding, as in hallucination. To resolve these challenges, we introduce LLM-ACTR, a novel neuro-symbolic architecture that provides human-aligned and versatile decision-making by integrating the ACT-R Cognitive Architecture with LLMs. Our framework extracts and embeds knowledge of ACT-R's internal decision-making process as latent neural representations, injects this information into trainable LLM adapter layers, and fine-tunes the LLMs for downstream prediction. Our experiments on novel Design for Manufacturing tasks show both improved task performance as well as improved grounded decision-making capability of our approach, compared to LLM-only baselines that leverage chain-of-thought reasoning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10892v3">Proving membership in LLM pretraining data via data watermarks</a></div>
    <div class="paper-meta">
      📅 2024-08-17
      | 💬 Findings of ACL 2024
    </div>
    <details class="paper-abstract">
      Detecting whether copyright holders' works were used in LLM pretraining is poised to be an important problem. This work proposes using data watermarks to enable principled detection with only black-box model access, provided that the rightholder contributed multiple training documents and watermarked them before public release. By applying a randomly sampled data watermark, detection can be framed as hypothesis testing, which provides guarantees on the false detection rate. We study two watermarks: one that inserts random sequences, and another that randomly substitutes characters with Unicode lookalikes. We first show how three aspects of watermark design -- watermark length, number of duplications, and interference -- affect the power of the hypothesis test. Next, we study how a watermark's detection strength changes under model and dataset scaling: while increasing the dataset size decreases the strength of the watermark, watermarks remain strong if the model size also increases. Finally, we view SHA hashes as natural watermarks and show that we can robustly detect hashes from BLOOM-176B's training data, as long as they occurred at least 90 times. Together, our results point towards a promising future for data watermarks in real world use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09127v1">From Specifications to Prompts: On the Future of Generative LLMs in Requirements Engineering</a></div>
    <div class="paper-meta">
      📅 2024-08-17
      | 💬 Published in IEEE Software 41 (5)
    </div>
    <details class="paper-abstract">
      Generative LLMs, such as GPT, have the potential to revolutionize Requirements Engineering (RE) by automating tasks in new ways. This column explores the novelties and introduces the importance of precise prompts for effective interactions. Human evaluation and prompt engineering are essential in leveraging LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20574v2">Open Ko-LLM Leaderboard: Evaluating Large Language Models in Korean with Ko-H5 Benchmark</a></div>
    <div class="paper-meta">
      📅 2024-08-17
      | 💬 Accepted at ACL 2024 Main
    </div>
    <details class="paper-abstract">
      This paper introduces the Open Ko-LLM Leaderboard and the Ko-H5 Benchmark as vital tools for evaluating Large Language Models (LLMs) in Korean. Incorporating private test sets while mirroring the English Open LLM Leaderboard, we establish a robust evaluation framework that has been well integrated in the Korean LLM community. We perform data leakage analysis that shows the benefit of private test sets along with a correlation study within the Ko-H5 benchmark and temporal analyses of the Ko-H5 score. Moreover, we present empirical support for the need to expand beyond set benchmarks. We hope the Open Ko-LLM Leaderboard sets precedent for expanding LLM evaluation to foster more linguistic diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09611v4">VizAbility: Enhancing Chart Accessibility with LLM-based Conversational Interaction</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Traditional accessibility methods like alternative text and data tables typically underrepresent data visualization's full potential. Keyboard-based chart navigation has emerged as a potential solution, yet efficient data exploration remains challenging. We present VizAbility, a novel system that enriches chart content navigation with conversational interaction, enabling users to use natural language for querying visual data trends. VizAbility adapts to the user's navigation context for improved response accuracy and facilitates verbal command-based chart navigation. Furthermore, it can address queries for contextual information, designed to address the needs of visually impaired users. We designed a large language model (LLM)-based pipeline to address these user queries, leveraging chart data & encoding, user context, and external web knowledge. We conducted both qualitative and quantitative studies to evaluate VizAbility's multimodal approach. We discuss further opportunities based on the results, including improved benchmark testing, incorporation of vision models, and integration with visualization workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20224v3">Can Editing LLMs Inject Harm?</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 The first two authors contributed equally. 9 pages for main paper, 36 pages including appendix. The code, results, dataset for this paper and more resources are on the project website: https://llm-editing.github.io
    </div>
    <details class="paper-abstract">
      Knowledge editing has been increasingly adopted to correct the false or outdated knowledge in Large Language Models (LLMs). Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13717v2">CoDefeater: Using LLMs To Find Defeaters in Assurance Cases</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 ASE 2024 NIER
    </div>
    <details class="paper-abstract">
      Constructing assurance cases is a widely used, and sometimes required, process toward demonstrating that safety-critical systems will operate safely in their planned environment. To mitigate the risk of errors and missing edge cases, the concept of defeaters - arguments or evidence that challenge claims in an assurance case - has been introduced. Defeaters can provide timely detection of weaknesses in the arguments, prompting further investigation and timely mitigations. However, capturing defeaters relies on expert judgment, experience, and creativity and must be done iteratively due to evolving requirements and regulations. This paper proposes CoDefeater, an automated process to leverage large language models (LLMs) for finding defeaters. Initial results on two systems show that LLMs can efficiently find known and unforeseen feasible defeaters to support safety analysts in enhancing the completeness and confidence of assurance cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08660v2">Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification</a></div>
    <div class="paper-meta">
      📅 2024-08-16
    </div>
    <details class="paper-abstract">
      Generative AI offers a simple, prompt-based alternative to fine-tuning smaller BERT-style LLMs for text classification tasks. This promises to eliminate the need for manually labeled training data and task-specific model training. However, it remains an open question whether tools like ChatGPT can deliver on this promise. In this paper, we show that smaller, fine-tuned LLMs (still) consistently and significantly outperform larger, zero-shot prompted models in text classification. We compare three major generative AI models (ChatGPT with GPT-3.5/GPT-4 and Claude Opus) with several fine-tuned LLMs across a diverse set of classification tasks (sentiment, approval/disapproval, emotions, party positions) and text categories (news, tweets, speeches). We find that fine-tuning with application-specific training data achieves superior performance in all cases. To make this approach more accessible to a broader audience, we provide an easy-to-use toolkit alongside this paper. Our toolkit, accompanied by non-technical step-by-step guidance, enables users to select and fine-tune BERT-like LLMs for any classification task with minimal technical and computational effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08781v1">Evaluating the Evaluator: Measuring LLMs' Adherence to Task Evaluation Instructions</a></div>
    <div class="paper-meta">
      📅 2024-08-16
    </div>
    <details class="paper-abstract">
      LLMs-as-a-judge is a recently popularized method which replaces human judgements in task evaluation (Zheng et al. 2024) with automatic evaluation using LLMs. Due to widespread use of RLHF (Reinforcement Learning from Human Feedback), state-of-the-art LLMs like GPT4 and Llama3 are expected to have strong alignment with human preferences when prompted for a quality judgement, such as the coherence of a text. While this seems beneficial, it is not clear whether the assessments by an LLM-as-a-judge constitute only an evaluation based on the instructions in the prompts, or reflect its preference for high-quality data similar to its fine-tune data. To investigate how much influence prompting the LLMs-as-a-judge has on the alignment of AI judgements to human judgements, we analyze prompts with increasing levels of instructions about the target quality of an evaluation, for several LLMs-as-a-judge. Further, we compare to a prompt-free method using model perplexity as a quality measure instead. We aggregate a taxonomy of quality criteria commonly used across state-of-the-art evaluations with LLMs and provide this as a rigorous benchmark of models as judges. Overall, we show that the LLMs-as-a-judge benefit only little from highly detailed instructions in prompts and that perplexity can sometimes align better with human judgements than prompting, especially on textual quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10160v6">Reinforcement Learning from Multi-role Debates as Feedback for Bias Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 The first three authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Bias in LLMs can harm user experience and societal outcomes. However, current bias mitigation methods often require intensive human feedback, lack transferability to other topics or yield overconfident and random outputs. We find that involving LLMs in role-playing scenario boosts their ability to recognize and mitigate biases. Based on this, we propose Reinforcement Learning from Multi-role Debates as Feedback (RLDF), a novel approach for bias mitigation replacing human feedback in traditional RLHF. We utilize LLMs in multi-role debates to create a dataset that includes both high-bias and low-bias instances for training the reward model in reinforcement learning. Our approach comprises two modes: (1) self-reflection, where the same LLM participates in multi-role debates, and (2) teacher-student, where a more advanced LLM like GPT-3.5-turbo guides the LLM to perform this task. Experimental results across different LLMs on BBQ and our datasets demonstrate the effectiveness of our approach in bias mitigation. Our source code and datasets are available at \texttt{https://anonymous.4open.science/r/RLDF-E344}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08676v1">Fine-tuning LLMs for Autonomous Spacecraft Control: A Case Study Using Kerbal Space Program</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 ESA SPAICE Conference 2024. arXiv admin note: text overlap with arXiv:2404.00413
    </div>
    <details class="paper-abstract">
      Recent trends are emerging in the use of Large Language Models (LLMs) as autonomous agents that take actions based on the content of the user text prompt. This study explores the use of fine-tuned Large Language Models (LLMs) for autonomous spacecraft control, using the Kerbal Space Program Differential Games suite (KSPDG) as a testing environment. Traditional Reinforcement Learning (RL) approaches face limitations in this domain due to insufficient simulation capabilities and data. By leveraging LLMs, specifically fine-tuning models like GPT-3.5 and LLaMA, we demonstrate how these models can effectively control spacecraft using language-based inputs and outputs. Our approach integrates real-time mission telemetry into textual prompts processed by the LLM, which then generate control actions via an agent. The results open a discussion about the potential of LLMs for space operations beyond their nominal use for text-related tasks. Future work aims to expand this methodology to other space control tasks and evaluate the performance of different LLM families. The code is available at this URL: \texttt{https://github.com/ARCLab-MIT/kspdg}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08656v1">LLMs Are Biased Towards Output Formats! Systematically Evaluating and Mitigating Output Format Bias of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-16
    </div>
    <details class="paper-abstract">
      We present the first systematic evaluation examining format bias in performance of large language models (LLMs). Our approach distinguishes between two categories of an evaluation metric under format constraints to reliably and accurately assess performance: one measures performance when format constraints are adhered to, while the other evaluates performance regardless of constraint adherence. We then define a metric for measuring the format bias of LLMs and establish effective strategies to reduce it. Subsequently, we present our empirical format bias evaluation spanning four commonly used categories -- multiple-choice question-answer, wrapping, list, and mapping -- covering 15 widely-used formats. Our evaluation on eight generation tasks uncovers significant format bias across state-of-the-art LLMs. We further discover that improving the format-instruction following capabilities of LLMs across formats potentially reduces format bias. Based on our evaluation findings, we study prompting and fine-tuning with synthesized format data techniques to mitigate format bias. Our methods successfully reduce the variance in ChatGPT's performance among wrapping formats from 235.33 to 0.71 (%$^2$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01129v3">Emphasising Structured Information: Integrating Abstract Meaning Representation into LLMs for Enhanced Open-Domain Dialogue Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-08-16
    </div>
    <details class="paper-abstract">
      Automatic open-domain dialogue evaluation has attracted increasing attention. Trainable evaluation metrics, typically trained with true positive and randomly selected negative responses, tend to assign higher scores to responses that share greater content similarity with a given context. However, adversarial negative responses, despite possessing high content similarity with the contexts, are semantically different. Consequently, existing evaluation metrics are not robust enough to evaluate such responses, resulting in low correlations with human judgments. While recent studies have demonstrated the effectiveness of Large Language Models (LLMs) for open-domain dialogue evaluation, they still face challenges in effectively handling adversarial negative examples. In this paper, we propose an effective framework for open-domain dialogue evaluation, which combines domain-specific language models (SLMs) enhanced with Abstract Meaning Representation (AMR) knowledge with LLMs. The SLMs can explicitly incorporate AMR graph information of the dialogue through a gating mechanism for enhanced dialogue semantic representation learning. Both the evaluation result from the SLMs and the AMR graph information are incorporated into the LLM's prompt for enhanced evaluation performance. Experimental results on open-domain dialogue evaluation tasks demonstrate the superiority of our method compared to a wide range of state-of-the-art baselines, especially in discriminating adversarial negative responses. Our code and data are publicly available at https://github.com/Bernard-Yang/SIMAMR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08477v1">Automating Transparency Mechanisms in the Judicial System Using LLMs: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 Accepted at the Seventh AAAI/ACM Conference on AI, Ethics, and Society (AIES 2024)
    </div>
    <details class="paper-abstract">
      Bringing more transparency to the judicial system for the purposes of increasing accountability often demands extensive effort from auditors who must meticulously sift through numerous disorganized legal case files to detect patterns of bias and errors. For example, the high-profile investigation into the Curtis Flowers case took seven reporters a full year to assemble evidence about the prosecutor's history of selecting racially biased juries. LLMs have the potential to automate and scale these transparency pipelines, especially given their demonstrated capabilities to extract information from unstructured documents. We discuss the opportunities and challenges of using LLMs to provide transparency in two important court processes: jury selection in criminal trials and housing eviction cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.20224v3">Can Editing LLMs Inject Harm?</a></div>
    <div class="paper-meta">
      📅 2024-08-16
      | 💬 The first two authors contributed equally. 9 pages for main paper, 36 pages including appendix. The code, results, dataset for this paper and more resources are on the project website: https://llm-editing.github.io
    </div>
    <details class="paper-abstract">
      Knowledge editing has been increasingly adopted to correct the false or outdated knowledge in Large Language Models (LLMs). Meanwhile, one critical but under-explored question is: can knowledge editing be used to inject harm into LLMs? In this paper, we propose to reformulate knowledge editing as a new type of safety threat for LLMs, namely Editing Attack, and conduct a systematic investigation with a newly constructed dataset EditAttack. Specifically, we focus on two typical safety risks of Editing Attack including Misinformation Injection and Bias Injection. For the risk of misinformation injection, we first categorize it into commonsense misinformation injection and long-tail misinformation injection. Then, we find that editing attacks can inject both types of misinformation into LLMs, and the effectiveness is particularly high for commonsense misinformation injection. For the risk of bias injection, we discover that not only can biased sentences be injected into LLMs with high effectiveness, but also one single biased sentence injection can cause a bias increase in general outputs of LLMs, which are even highly irrelevant to the injected sentence, indicating a catastrophic impact on the overall fairness of LLMs. Then, we further illustrate the high stealthiness of editing attacks, measured by their impact on the general knowledge and reasoning capacities of LLMs, and show the hardness of defending editing attacks with empirical evidence. Our discoveries demonstrate the emerging misuse risks of knowledge editing techniques on compromising the safety alignment of LLMs and the feasibility of disseminating misinformation or bias with LLMs as new channels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08546v2">Grounding LLMs For Robot Task Planning Using Closed-loop State Feedback</a></div>
    <div class="paper-meta">
      📅 2024-08-15
      | 💬 This work has been submitted to Autonomous Robots
    </div>
    <details class="paper-abstract">
      Planning algorithms decompose complex problems into intermediate steps that can be sequentially executed by robots to complete tasks. Recent works have employed Large Language Models (LLMs) for task planning, using natural language to generate robot policies in both simulation and real-world environments. LLMs like GPT-4 have shown promising results in generalizing to unseen tasks, but their applicability is limited due to hallucinations caused by insufficient grounding in the robot environment. The robustness of LLMs in task planning can be enhanced with environmental state information and feedback. In this paper, we introduce a novel approach to task planning that utilizes two separate LLMs for high-level planning and low-level control, improving task-related success rates and goal condition recall. Our algorithm, \textit{BrainBody-LLM}, draws inspiration from the human neural system, emulating its brain-body architecture by dividing planning across two LLMs in a structured, hierarchical manner. BrainBody-LLM implements a closed-loop feedback mechanism, enabling learning from simulator errors to resolve execution errors in complex settings. We demonstrate the successful application of BrainBody-LLM in the VirtualHome simulation environment, achieving a 29\% improvement in task-oriented success rates over competitive baselines with the GPT-4 backend. Additionally, we evaluate our algorithm on seven complex tasks using a realistic physics simulator and the Franka Research 3 robotic arm, comparing it with various state-of-the-art LLMs. Our results show advancements in the reasoning capabilities of recent LLMs, which enable them to learn from raw simulator/controller errors to correct plans, making them highly effective in robotic task planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08401v1">Understanding Help-Seeking Behavior of Students Using LLMs vs. Web Search for Writing SQL Queries</a></div>
    <div class="paper-meta">
      📅 2024-08-15
    </div>
    <details class="paper-abstract">
      Growth in the use of large language models (LLMs) in programming education is altering how students write SQL queries. Traditionally, students relied heavily on web search for coding assistance, but this has shifted with the adoption of LLMs like ChatGPT. However, the comparative process and outcomes of using web search versus LLMs for coding help remain underexplored. To address this, we conducted a randomized interview study in a database classroom to compare web search and LLMs, including a publicly available LLM (ChatGPT) and an instructor-tuned LLM, for writing SQL queries. Our findings indicate that using an instructor-tuned LLM required significantly more interactions than both ChatGPT and web search, but resulted in a similar number of edits to the final SQL query. No significant differences were found in the quality of the final SQL queries between conditions, although the LLM conditions directionally showed higher query quality. Furthermore, students using instructor-tuned LLM reported a lower mental demand. These results have implications for learning and productivity in programming education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08146v1">KOALA: Enhancing Speculative Decoding for LLM via Multi-Layer Draft Heads with Adversarial Learning</a></div>
    <div class="paper-meta">
      📅 2024-08-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit high inference latency due to their autoregressive decoding nature. While the draft head in speculative decoding mitigates this issue, its full potential remains unexplored. In this paper, we introduce KOALA (K-layer Optimized Adversarial Learning Architecture), an orthogonal approach to the draft head. By transforming the conventional single-layer draft head into a multi-layer architecture and incorporating adversarial learning into the traditional supervised training, KOALA significantly improves the accuracy of the draft head in predicting subsequent tokens, thus more closely mirroring the functionality of LLMs. Although this improvement comes at the cost of slightly increased drafting overhead, KOALA substantially unlocks the draft head's potential, greatly enhancing speculative decoding. We conducted comprehensive evaluations of KOALA, including both autoregressive and non-autoregressive draft heads across various tasks, demonstrating a latency speedup ratio improvement of 0.24x-0.41x, which is 10.57%-14.09% faster than the original draft heads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00005v1">Csi-LLM: A Novel Downlink Channel Prediction Method Aligned with LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2024-08-15
    </div>
    <details class="paper-abstract">
      Downlink channel temporal prediction is a critical technology in massive multiple-input multiple-output (MIMO) systems. However, existing methods that rely on fixed-step historical sequences significantly limit the accuracy, practicality, and scalability of channel prediction. Recent advances have shown that large language models (LLMs) exhibit strong pattern recognition and reasoning abilities over complex sequences. The challenge lies in effectively aligning wireless communication data with the modalities used in natural language processing to fully harness these capabilities. In this work, we introduce Csi-LLM, a novel LLM-powered downlink channel prediction technique that models variable-step historical sequences. To ensure effective cross-modality application, we align the design and training of Csi-LLM with the processing of natural language tasks, leveraging the LLM's next-token generation capability for predicting the next step in channel state information (CSI). Simulation results demonstrate the effectiveness of this alignment strategy, with Csi-LLM consistently delivering stable performance improvements across various scenarios and showing significant potential in continuous multi-step prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17435v4">Can LLMs Replace Economic Choice Prediction Labs? The Case of Language-based Persuasion Games</a></div>
    <div class="paper-meta">
      📅 2024-08-14
    </div>
    <details class="paper-abstract">
      Human choice prediction in economic contexts is crucial for applications in marketing, finance, public policy, and more. This task, however, is often constrained by the difficulties in acquiring human choice data. With most experimental economics studies focusing on simple choice settings, the AI community has explored whether LLMs can substitute for humans in these predictions and examined more complex experimental economics settings. However, a key question remains: can LLMs generate training data for human choice prediction? We explore this in language-based persuasion games, a complex economic setting involving natural language in strategic interactions. Our experiments show that models trained on LLM-generated data can effectively predict human behavior in these games and even outperform models trained on actual human data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07773v1">MedTsLLM: Leveraging LLMs for Multimodal Medical Time Series Analysis</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 published in Proceedings of Machine Learning Research, MLHC 2024
    </div>
    <details class="paper-abstract">
      The complexity and heterogeneity of data in many real-world applications pose significant challenges for traditional machine learning and signal processing techniques. For instance, in medicine, effective analysis of diverse physiological signals is crucial for patient monitoring and clinical decision-making and yet highly challenging. We introduce MedTsLLM, a general multimodal large language model (LLM) framework that effectively integrates time series data and rich contextual information in the form of text to analyze physiological signals, performing three tasks with clinical relevance: semantic segmentation, boundary detection, and anomaly detection in time series. These critical tasks enable deeper analysis of physiological signals and can provide actionable insights for clinicians. We utilize a reprogramming layer to align embeddings of time series patches with a pretrained LLM's embedding space and make effective use of raw time series, in conjunction with textual context. Given the multivariate nature of medical datasets, we develop methods to handle multiple covariates. We additionally tailor the text prompt to include patient-specific information. Our model outperforms state-of-the-art baselines, including deep learning models, other LLMs, and clinical methods across multiple medical domains, specifically electrocardiograms and respiratory waveforms. MedTsLLM presents a promising step towards harnessing the power of LLMs for medical time series analysis that can elevate data-driven tools for clinicians and improve patient outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03862v3">Persona Inconstancy in Multi-Agent LLM Collaboration: Conformity, Confabulation, and Impersonation</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 16 pages, 8 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Multi-agent AI systems can be used for simulating collective decision-making in scientific and practical applications. They can also be used to introduce a diverse group discussion step in chatbot pipelines, enhancing the cultural sensitivity of the chatbot's responses. These applications, however, are predicated on the ability of AI agents to reliably adopt assigned personas and mimic human interactions. To see whether LLM agents satisfy these requirements, we examine AI agent ensembles engaged in cross-national collaboration and debate by analyzing their private responses and chat transcripts. Our findings suggest that multi-agent discussions can support collective AI decisions that more often reflect diverse perspectives, yet this effect is tempered by the agents' susceptibility to conformity due to perceived peer pressure and occasional challenges in maintaining consistent personas and opinions. Instructions that encourage debate in support of one's opinions rather than collaboration increase the rate of inconstancy. Without addressing the factors we identify, the full potential of multi-agent frameworks for producing more culturally diverse AI outputs or more realistic simulations of group decision-making may remain untapped.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15363v2">Exploring LLM Multi-Agents for ICD Coding</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 12pages
    </div>
    <details class="paper-abstract">
      To address the limitations of Large Language Models (LLMs) in the International Classification of Diseases (ICD) coding task, where they often produce inaccurate and incomplete prediction results due to the high-dimensional and skewed distribution of the ICD codes, and often lack interpretability and reliability as well. We introduce an innovative multi-agent approach for ICD coding which mimics the ICD coding assignment procedure in real-world settings, comprising five distinct agents: the patient, physician, coder, reviewer, and adjuster. Each agent utilizes an LLM-based model tailored to their specific role within the coding process. We also integrate the system with Electronic Health Record (HER)'s SOAP (subjective, objective, assessment and plan) structure to boost the performances. We compare our method with a system of agents designed solely by LLMs and other strong baselines and evaluate it using the Medical Information Mart for Intensive Care III (MIMIC-III) dataset. Our multi-agent coding framework significantly outperforms Zero-shot Chain of Thought (CoT) prompting and self-consistency with CoT (CoT-SC) in coding common and rare ICD codes. An ablation study validates the effectiveness of the designated agent roles. it also outperforms the LLM-designed agent system. Moreover, our method achieves comparable results to state-of-the-art ICD coding methods that require extensive pre-training or fine-tuning, and outperforms them in rare code accuracy, and explainability. Additionally, we demonstrate the method's practical applicability by presenting its performance in scenarios not limited by the common or rare ICD code constraints.The proposed multi-agent method for ICD coding effectively mimics the real-world coding process and improves performance on both common and rare codes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01766v3">LLM Voting: Human Choices and AI Collective Decision Making</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 Accepted in AAAI Conference on AI, Ethics, and Society (AIES)
    </div>
    <details class="paper-abstract">
      This paper investigates the voting behaviors of Large Language Models (LLMs), specifically GPT-4 and LLaMA-2, their biases, and how they align with human voting patterns. Our methodology involved using a dataset from a human voting experiment to establish a baseline for human preferences and conducting a corresponding experiment with LLM agents. We observed that the choice of voting methods and the presentation order influenced LLM voting outcomes. We found that varying the persona can reduce some of these biases and enhance alignment with human choices. While the Chain-of-Thought approach did not improve prediction accuracy, it has potential for AI explainability in the voting process. We also identified a trade-off between preference diversity and alignment accuracy in LLMs, influenced by different temperature settings. Our findings indicate that LLMs may lead to less diverse collective outcomes and biased assumptions when used in voting scenarios, emphasizing the need for cautious integration of LLMs into democratic processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10020v2">Lost in Overlap: Exploring Watermark Collision in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 Long Paper, 7 pages
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) in generating content raises concerns about text copyright. Watermarking methods, particularly logit-based approaches, embed imperceptible identifiers into text to address these challenges. However, the widespread usage of watermarking across diverse LLMs has led to an inevitable issue known as watermark collision during common tasks, such as paraphrasing or translation. In this paper, we introduce watermark collision as a novel and general philosophy for watermark attacks, aimed at enhancing attack performance on top of any other attacking methods. We also provide a comprehensive demonstration that watermark collision poses a threat to all logit-based watermark algorithms, impacting not only specific attack scenarios but also downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01107v2">BioRAG: A RAG-LLM Framework for Biological Question Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The question-answering system for Life science research, which is characterized by the rapid pace of discovery, evolving insights, and complex interactions among knowledge entities, presents unique challenges in maintaining a comprehensive knowledge warehouse and accurate information retrieval. To address these issues, we introduce BioRAG, a novel Retrieval-Augmented Generation (RAG) with the Large Language Models (LLMs) framework. Our approach starts with parsing, indexing, and segmenting an extensive collection of 22 million scientific papers as the basic knowledge, followed by training a specialized embedding model tailored to this domain. Additionally, we enhance the vector retrieval process by incorporating a domain-specific knowledge hierarchy, which aids in modeling the intricate interrelationships among each query and context. For queries requiring the most current information, BioRAG deconstructs the question and employs an iterative retrieval process incorporated with the search engine for step-by-step reasoning. Rigorous experiments have demonstrated that our model outperforms fine-tuned LLM, LLM with search engines, and other scientific RAG frameworks across multiple life science question-answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.21264v2">Model Attribution in LLM-Generated Disinformation: A Domain Generalization Approach with Supervised Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2024-08-14
      | 💬 10 pages, 2 figures, accepted at DSAA 2024
    </div>
    <details class="paper-abstract">
      Model attribution for LLM-generated disinformation poses a significant challenge in understanding its origins and mitigating its spread. This task is especially challenging because modern large language models (LLMs) produce disinformation with human-like quality. Additionally, the diversity in prompting methods used to generate disinformation complicates accurate source attribution. These methods introduce domain-specific features that can mask the fundamental characteristics of the models. In this paper, we introduce the concept of model attribution as a domain generalization problem, where each prompting method represents a unique domain. We argue that an effective attribution model must be invariant to these domain-specific features. It should also be proficient in identifying the originating models across all scenarios, reflecting real-world detection challenges. To address this, we introduce a novel approach based on Supervised Contrastive Learning. This method is designed to enhance the model's robustness to variations in prompts and focuses on distinguishing between different source LLMs. We evaluate our model through rigorous experiments involving three common prompting methods: ``open-ended'', ``rewriting'', and ``paraphrasing'', and three advanced LLMs: ``llama 2'', ``chatgpt'', and ``vicuna''. Our results demonstrate the effectiveness of our approach in model attribution tasks, achieving state-of-the-art performance across diverse and unseen datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07321v1">LLM-Enhanced Static Analysis for Precise Identification of Vulnerable OSS Versions</a></div>
    <div class="paper-meta">
      📅 2024-08-14
    </div>
    <details class="paper-abstract">
      Open-source software (OSS) has experienced a surge in popularity, attributed to its collaborative development model and cost-effective nature. However, the adoption of specific software versions in development projects may introduce security risks when these versions bring along vulnerabilities. Current methods of identifying vulnerable versions typically analyze and trace the code involved in vulnerability patches using static analysis with pre-defined rules. They then use syntactic-level code clone detection to identify the vulnerable versions. These methods are hindered by imprecisions due to (1) the inclusion of vulnerability-irrelevant code in the analysis and (2) the inadequacy of syntactic-level code clone detection. This paper presents Vercation, an approach designed to identify vulnerable versions of OSS written in C/C++. Vercation combines program slicing with a Large Language Model (LLM) to identify vulnerability-relevant code from vulnerability patches. It then backtraces historical commits to gather previous modifications of identified vulnerability-relevant code. We propose semantic-level code clone detection to compare the differences between pre-modification and post-modification code, thereby locating the vulnerability-introducing commit (vic) and enabling to identify the vulnerable versions between the patch commit and the vic. We curate a dataset linking 74 OSS vulnerabilities and 1013 versions to evaluate Vercation. On this dataset, our approach achieves the F1 score of 92.4%, outperforming current state-of-the-art methods. More importantly, Vercation detected 134 incorrect vulnerable OSS versions in NVD reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2408.06993v1">LLMs can Schedule</a></div>
    <div class="paper-meta">
      📅 2024-08-14
    </div>
    <details class="paper-abstract">
      The job shop scheduling problem (JSSP) remains a significant hurdle in optimizing production processes. This challenge involves efficiently allocating jobs to a limited number of machines while minimizing factors like total processing time or job delays. While recent advancements in artificial intelligence have yielded promising solutions, such as reinforcement learning and graph neural networks, this paper explores the potential of Large Language Models (LLMs) for JSSP. We introduce the very first supervised 120k dataset specifically designed to train LLMs for JSSP. Surprisingly, our findings demonstrate that LLM-based scheduling can achieve performance comparable to other neural approaches. Furthermore, we propose a sampling method that enhances the effectiveness of LLMs in tackling JSSP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07238v1">Using Advanced LLMs to Enhance Smaller LLMs: An Interpretable Knowledge Distillation Approach</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      Advanced Large language models (LLMs) like GPT-4 or LlaMa 3 provide superior performance in complex human-like interactions. But they are costly, or too large for edge devices such as smartphones and harder to self-host, leading to security and privacy concerns. This paper introduces a novel interpretable knowledge distillation approach to enhance the performance of smaller, more economical LLMs that firms can self-host. We study this problem in the context of building a customer service agent aimed at achieving high customer satisfaction through goal-oriented dialogues. Unlike traditional knowledge distillation, where the "student" model learns directly from the "teacher" model's responses via fine-tuning, our interpretable "strategy" teaching approach involves the teacher providing strategies to improve the student's performance in various scenarios. This method alternates between a "scenario generation" step and a "strategies for improvement" step, creating a customized library of scenarios and optimized strategies for automated prompting. The method requires only black-box access to both student and teacher models; hence it can be used without manipulating model parameters. In our customer service application, the method improves performance, and the learned strategies are transferable to other LLMs and scenarios beyond the training set. The method's interpretabilty helps safeguard against potential harms through human audit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07137v1">ELLA: Empowering LLMs for Interpretable, Accurate and Informative Legal Advice</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      Despite remarkable performance in legal consultation exhibited by legal Large Language Models(LLMs) combined with legal article retrieval components, there are still cases when the advice given is incorrect or baseless. To alleviate these problems, we propose {\bf ELLA}, a tool for {\bf E}mpowering {\bf L}LMs for interpretable, accurate, and informative {\bf L}egal {\bf A}dvice. ELLA visually presents the correlation between legal articles and LLM's response by calculating their similarities, providing users with an intuitive legal basis for the responses. Besides, based on the users' queries, ELLA retrieves relevant legal articles and displays them to users. Users can interactively select legal articles for LLM to generate more accurate responses. ELLA also retrieves relevant legal cases for user reference. Our user study shows that presenting the legal basis for the response helps users understand better. The accuracy of LLM's responses also improves when users intervene in selecting legal articles for LLM. Providing relevant legal cases also aids individuals in obtaining comprehensive information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07055v1">LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      Current long context large language models (LLMs) can process inputs up to 100,000 tokens, yet struggle to generate outputs exceeding even a modest length of 2,000 words. Through controlled experiments, we find that the model's effective generation length is inherently bounded by the sample it has seen during supervised fine-tuning (SFT). In other words, their output limitation is due to the scarcity of long-output examples in existing SFT datasets. To address this, we introduce AgentWrite, an agent-based pipeline that decomposes ultra-long generation tasks into subtasks, enabling off-the-shelf LLMs to generate coherent outputs exceeding 20,000 words. Leveraging AgentWrite, we construct LongWriter-6k, a dataset containing 6,000 SFT data with output lengths ranging from 2k to 32k words. By incorporating this dataset into model training, we successfully scale the output length of existing models to over 10,000 words while maintaining output quality. We also develop LongBench-Write, a comprehensive benchmark for evaluating ultra-long generation capabilities. Our 9B parameter model, further improved through DPO, achieves state-of-the-art performance on this benchmark, surpassing even much larger proprietary models. In general, our work demonstrates that existing long context LLM already possesses the potential for a larger output window--all you need is data with extended output during model alignment to unlock this capability. Our code & models are at: https://github.com/THUDM/LongWriter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06993v1">LLMs can Schedule</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      The job shop scheduling problem (JSSP) remains a significant hurdle in optimizing production processes. This challenge involves efficiently allocating jobs to a limited number of machines while minimizing factors like total processing time or job delays. While recent advancements in artificial intelligence have yielded promising solutions, such as reinforcement learning and graph neural networks, this paper explores the potential of Large Language Models (LLMs) for JSSP. We introduce the very first supervised 120k dataset specifically designed to train LLMs for JSSP. Surprisingly, our findings demonstrate that LLM-based scheduling can achieve performance comparable to other neural approaches. Furthermore, we propose a sampling method that enhances the effectiveness of LLMs in tackling JSSP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06816v1">MAQA: Evaluating Uncertainty Quantification in LLMs Regarding Data Uncertainty</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) are capable of performing various tasks, they still suffer from producing plausible but incorrect responses. To improve the reliability of LLMs, recent research has focused on uncertainty quantification to predict whether a response is correct or not. However, most uncertainty quantification methods have been evaluated on questions requiring a single clear answer, ignoring the existence of data uncertainty that arises from irreducible randomness. Instead, these methods only consider model uncertainty, which arises from a lack of knowledge. In this paper, we investigate previous uncertainty quantification methods under the presence of data uncertainty. Our contributions are two-fold: 1) proposing a new Multi-Answer Question Answering dataset, MAQA, consisting of world knowledge, mathematical reasoning, and commonsense reasoning tasks to evaluate uncertainty quantification regarding data uncertainty, and 2) assessing 5 uncertainty quantification methods of diverse white- and black-box LLMs. Our findings show that entropy and consistency-based methods estimate the model uncertainty well even under data uncertainty, while other methods for white- and black-box LLMs struggle depending on the tasks. Additionally, methods designed for white-box LLMs suffer from overconfidence in reasoning tasks compared to simple knowledge queries. We believe our observations will pave the way for future work on uncertainty quantification in realistic setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06810v1">HLSPilot: LLM-based High-Level Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have catalyzed an upsurge in automatic code generation, garnering significant attention for register transfer level (RTL) code generation. Despite the potential of RTL code generation with natural language, it remains error-prone and limited to relatively small modules because of the substantial semantic gap between natural language expressions and hardware design intent. In response to the limitations, we propose a methodology that reduces the semantic gaps by utilizing C/C++ for generating hardware designs via High-Level Synthesis (HLS) tools. Basically, we build a set of C-to-HLS optimization strategies catering to various code patterns, such as nested loops and local arrays. Then, we apply these strategies to sequential C/C++ code through in-context learning, which provides the LLMs with exemplary C/C++ to HLS prompts. With this approach, HLS designs can be generated effectively. Since LLMs still face problems in determining the optimized pragma parameters precisely, we have a design space exploration (DSE) tool integrated for pragma parameter tuning. Furthermore, we also employ profiling tools to pinpoint the performance bottlenecks within a program and selectively convert bottleneck components to HLS code for hardware acceleration. By combining the LLM-based profiling, C/C++ to HLS translation, and DSE, we have established HLSPilot, the first LLM-enabled high-level synthesis framework, which can fully automate the high-level application acceleration on hybrid CPU-FPGA architectures. According to our experiments on real-world application benchmarks, HLSPilot achieve comparable performance in general and can even outperform manually crafted counterparts, thereby underscoring the substantial promise of LLM-assisted hardware designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07106v1">"You still have to study" -- On the Security of LLM generated code</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      We witness an increasing usage of AI-assistants even for routine (classroom) programming tasks. However, the code generated on basis of a so called "prompt" by the programmer does not always meet accepted security standards. On the one hand, this may be due to lack of best-practice examples in the training data. On the other hand, the actual quality of the programmers prompt appears to influence whether generated code contains weaknesses or not. In this paper we analyse 4 major LLMs with respect to the security of generated code. We do this on basis of a case study for the Python and Javascript language, using the MITRE CWE catalogue as the guiding security definition. Our results show that using different prompting techniques, some LLMs initially generate 65% code which is deemed insecure by a trained security engineer. On the other hand almost all analysed LLMs will eventually generate code being close to 100% secure with increasing manual guidance of a skilled engineer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06673v1">Pragmatic inference of scalar implicature by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-08-13
      | 💬 This research was presented at the Association for Computational Linguistics conference, held on August 11-16
    </div>
    <details class="paper-abstract">
      This study investigates how Large Language Models (LLMs), particularly BERT (Devlin et al., 2019) and GPT-2 (Radford et al., 2019), engage in pragmatic inference of scalar implicature, such as some. Two sets of experiments were conducted using cosine similarity and next sentence/token prediction as experimental methods. The results in experiment 1 showed that, both models interpret some as pragmatic implicature not all in the absence of context, aligning with human language processing. In experiment 2, in which Question Under Discussion (QUD) was presented as a contextual cue, BERT showed consistent performance regardless of types of QUDs, while GPT-2 encountered processing difficulties since a certain type of QUD required pragmatic inference for implicature. The findings revealed that, in terms of theoretical approaches, BERT inherently incorporates pragmatic implicature not all within the term some, adhering to Default model (Levinson, 2000). In contrast, GPT-2 seems to encounter processing difficulties in inferring pragmatic implicature within context, consistent with Context-driven model (Sperber and Wilson, 2002).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15281v1">LAAG-RV: LLM Assisted Assertion Generation for RTL Design Verification</a></div>
    <div class="paper-meta">
      📅 2024-08-13
      | 💬 3 Figures, 1 Table
    </div>
    <details class="paper-abstract">
      Writing SystemVerilog Assertions (SVA) is an important but complex step in verifying Register Transfer Level (RTL) designs. Conventionally, experts need to understand the design specifications and write the SVA assertions, which is time-consuming and error-prone. However, with the recent advancement of transformer models, the Large Language Models (LLMs) assisted assertion generation for design verification is gaining interest in recent times. Motivated by this, we proposed a novel LLM-based framework, LAAG-RV, to generate SVA from the natural language specifications of the design. Our framework provides a one-time Verilog loop for signal synchronization in the generated SVA to improve the generated assertion quality. For our experiments, we created a custom LLM based on OpenAI GPT-4. Furthermore, we developed test cases to validate the LLM-generated assertions. Initial observations show that some generated assertions contain issues and did not pass all the test cases. However, by iteratively prompting the LLMs using carefully crafted manual prompts derived from test case failures in a simulator, the framework can generate correct SVAs. Our results on OpenTitan designs demonstrate that LLMs significantly simplify the process of generating assertions, making it efficient and less error-prone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08315v2">Application of LLM Agents in Recruitment: A Novel Framework for Resume Screening</a></div>
    <div class="paper-meta">
      📅 2024-08-13
      | 💬 Accept by Journal of Information Processing,(2024), 18 pages, 19 figures
    </div>
    <details class="paper-abstract">
      The automation of resume screening is a crucial aspect of the recruitment process in organizations. Automated resume screening systems often encompass a range of natural language processing (NLP) tasks. This paper introduces a novel Large Language Models (LLMs) based agent framework for resume screening, aimed at enhancing efficiency and time management in recruitment processes. Our framework is distinct in its ability to efficiently summarize and grade each resume from a large dataset. Moreover, it utilizes LLM agents for decision-making. To evaluate our framework, we constructed a dataset from actual resumes and simulated a resume screening process. Subsequently, the outcomes of the simulation experiment were compared and subjected to detailed analysis. The results demonstrate that our automated resume screening framework is 11 times faster than traditional manual methods. Furthermore, by fine-tuning the LLMs, we observed a significant improvement in the F1 score, reaching 87.73\%, during the resume sentence classification phase. In the resume summarization and grading phase, our fine-tuned model surpassed the baseline performance of the GPT-3.5 model. Analysis of the decision-making efficacy of the LLM agents in the final offer stage further underscores the potential of LLM agents in transforming resume screening processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06610v1">CROME: Cross-Modal Adapters for Efficient Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) demonstrate remarkable image-language capabilities, but their widespread use faces challenges in cost-effective training and adaptation. Existing approaches often necessitate expensive language model retraining and limited adaptability. Additionally, the current focus on zero-shot performance improvements offers insufficient guidance for task-specific tuning. We propose CROME, an efficient vision-language instruction tuning framework. It features a novel gated cross-modal adapter that effectively combines visual and textual representations prior to input into a frozen LLM. This lightweight adapter, trained with minimal parameters, enables efficient cross-modal understanding. Notably, CROME demonstrates superior zero-shot performance on standard visual question answering and instruction-following benchmarks. Moreover, it yields fine-tuning with exceptional parameter efficiency, competing with task-specific specialist state-of-the-art methods. CROME demonstrates the potential of pre-LM alignment for building scalable, adaptable, and parameter-efficient multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.00798v4">Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      Recent advancements on Large Language Models (LLMs) enable AI Agents to automatically generate and execute multi-step plans to solve complex tasks. However, since LLM's content generation process is hardly controllable, current LLM-based agents frequently generate invalid or non-executable plans, which jeopardizes the performance of the generated plans and corrupts users' trust in LLM-based agents. In response, this paper proposes a novel "Formal-LLM" framework for LLM-based agents by integrating the expressiveness of natural language and the precision of formal language. Specifically, the framework allows agent developers to express their requirements or constraints for the planning process as an automaton. A stack-based LLM plan generation process is then conducted under the supervision of the automaton to ensure that the generated plan satisfies the constraints, making the planning process controllable. We conduct experiments on both benchmark tasks and practical real-life tasks, and our framework achieves over 50% overall performance increase, which validates the feasibility and effectiveness of employing Formal-LLM to guide the plan generation of agents, preventing the agents from generating invalid and unsuccessful plans. Further, more controllable LLM-based agents can facilitate the broader utilization of LLM in application scenarios where high validity of planning is essential. The source code of this work is available at https://github.com/agiresearch/Formal-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.06993v1">LLMs can Schedule</a></div>
    <div class="paper-meta">
      📅 2024-08-13
    </div>
    <details class="paper-abstract">
      The job shop scheduling problem (JSSP) remains a significant hurdle in optimizing production processes. This challenge involves efficiently allocating jobs to a limited number of machines while minimizing factors like total processing time or job delays. While recent advancements in artificial intelligence have yielded promising solutions, such as reinforcement learning and graph neural networks, this paper explores the potential of Large Language Models (LLMs) for JSSP. We introduce the very first supervised 120k dataset specifically designed to train LLMs for JSSP. Surprisingly, our findings demonstrate that LLM-based scheduling can achieve performance comparable to other neural approaches. Furthermore, we propose a sampling method that enhances the effectiveness of LLMs in tackling JSSP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00798v4">Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-08-12
    </div>
    <details class="paper-abstract">
      Recent advancements on Large Language Models (LLMs) enable AI Agents to automatically generate and execute multi-step plans to solve complex tasks. However, since LLM's content generation process is hardly controllable, current LLM-based agents frequently generate invalid or non-executable plans, which jeopardizes the performance of the generated plans and corrupts users' trust in LLM-based agents. In response, this paper proposes a novel "Formal-LLM" framework for LLM-based agents by integrating the expressiveness of natural language and the precision of formal language. Specifically, the framework allows agent developers to express their requirements or constraints for the planning process as an automaton. A stack-based LLM plan generation process is then conducted under the supervision of the automaton to ensure that the generated plan satisfies the constraints, making the planning process controllable. We conduct experiments on both benchmark tasks and practical real-life tasks, and our framework achieves over 50% overall performance increase, which validates the feasibility and effectiveness of employing Formal-LLM to guide the plan generation of agents, preventing the agents from generating invalid and unsuccessful plans. Further, more controllable LLM-based agents can facilitate the broader utilization of LLM in application scenarios where high validity of planning is essential. The source code of this work is available at https://github.com/agiresearch/Formal-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06318v1">Can We Rely on LLM Agents to Draft Long-Horizon Plans? Let's Take TravelPlanner as an Example</a></div>
    <div class="paper-meta">
      📅 2024-08-12
      | 💬 13 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have brought autonomous agents closer to artificial general intelligence (AGI) due to their promising generalization and emergent capabilities. There is, however, a lack of studies on how LLM-based agents behave, why they could potentially fail, and how to improve them, particularly in demanding real-world planning tasks. In this paper, as an effort to fill the gap, we present our study using a realistic benchmark, TravelPlanner, where an agent must meet multiple constraints to generate accurate plans. We leverage this benchmark to address four key research questions: (1) are LLM agents robust enough to lengthy and noisy contexts when it comes to reasoning and planning? (2) can few-shot prompting adversely impact the performance of LLM agents in scenarios with long context? (3) can we rely on refinement to improve plans, and (4) can fine-tuning LLMs with both positive and negative feedback lead to further improvement? Our comprehensive experiments indicate that, firstly, LLMs often fail to attend to crucial parts of a long context, despite their ability to handle extensive reference information and few-shot examples; secondly, they still struggle with analyzing the long plans and cannot provide accurate feedback for refinement; thirdly, we propose Feedback-Aware Fine-Tuning (FAFT), which leverages both positive and negative feedback, resulting in substantial gains over Supervised Fine-Tuning (SFT). Our findings offer in-depth insights to the community on various aspects related to real-world planning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06285v1">Synthetic Patient-Physician Dialogue Generation from Clinical Notes Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-08-12
    </div>
    <details class="paper-abstract">
      Medical dialogue systems (MDS) enhance patient-physician communication, improve healthcare accessibility, and reduce costs. However, acquiring suitable data to train these systems poses significant challenges. Privacy concerns prevent the use of real conversations, necessitating synthetic alternatives. Synthetic dialogue generation from publicly available clinical notes offers a promising solution to this issue, providing realistic data while safeguarding privacy. Our approach, SynDial, uses a single LLM iteratively with zero-shot prompting and a feedback loop to generate and refine high-quality synthetic dialogues. The feedback consists of weighted evaluation scores for similarity and extractiveness. The iterative process ensures dialogues meet predefined thresholds, achieving superior extractiveness as a result of the feedback loop. Additionally, evaluation shows that the generated dialogues excel in factuality metric compared to the baselines and has comparable diversity scores with GPT4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06186v1">Improving Structural Diversity of Blackbox LLMs via Chain-of-Specification Prompting</a></div>
    <div class="paper-meta">
      📅 2024-08-12
    </div>
    <details class="paper-abstract">
      The capability to generate diverse text is a key challenge facing large language models (LLMs). Thus far, diversity has been studied via metrics such as $n$-gram diversity or diversity of BERT embeddings. However, for these kinds of diversity, the user has little control over the dimensions along which diversity is considered. For example, in the poetry domain, one might desire diversity in terms of rhyme and meter, whereas in the code domain, one might desire diversity in terms of the kinds of expressions used to solve a problem. We propose a diversity metric called structural diversity, where the user provides a mapping from generated text to features capturing the kinds of diversity that they care about. In addition, we propose a novel strategy called chain-of-specification (CoS) prompting for improving diversity by first having the LLM generate a specification encoding one instance of structural features, and then prompting the LLM to generate text that satisfies these features; notably, our strategy works with blackbox LLMs. In our experiments, we show that for structural diversity in the poetry and code domains, CoS significantly improves diversity compared to several baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03745v3">Fakes of Varying Shades: How Warning Affects Human Perception and Engagement Regarding LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2024-08-12
      | 💬 Accepted at COLM 2024
    </div>
    <details class="paper-abstract">
      The widespread adoption and transformative effects of large language models (LLMs) have sparked concerns regarding their capacity to produce inaccurate and fictitious content, referred to as `hallucinations'. Given the potential risks associated with hallucinations, humans should be able to identify them. This research aims to understand the human perception of LLM hallucinations by systematically varying the degree of hallucination (genuine, minor hallucination, major hallucination) and examining its interaction with warning (i.e., a warning of potential inaccuracies: absent vs. present). Participants (N=419) from Prolific rated the perceived accuracy and engaged with content (e.g., like, dislike, share) in a Q/A format. Participants ranked content as truthful in the order of genuine, minor hallucination, and major hallucination, and user engagement behaviors mirrored this pattern. More importantly, we observed that warning improved the detection of hallucination without significantly affecting the perceived truthfulness of genuine content. We conclude by offering insights for future tools to aid human detection of hallucinations. All survey materials, demographic questions, and post-session questions are available at: https://github.com/MahjabinNahar/fakes-of-varying-shades-survey-materials
    </details>
</div>
