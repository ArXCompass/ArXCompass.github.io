# llm - 2025_08

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
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13358v3">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19202v1">Demystifying Scientific Problem-Solving in LLMs by Probing Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 28 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Scientific problem solving poses unique challenges for LLMs, requiring both deep domain knowledge and the ability to apply such knowledge through complex reasoning. While automated scientific reasoners hold great promise for assisting human scientists, there is currently no widely adopted holistic benchmark for evaluating scientific reasoning, and few approaches systematically disentangle the distinct roles of knowledge and reasoning in these tasks. To address these gaps, we introduce SciReas, a diverse suite of existing benchmarks for scientific reasoning tasks, and SciReas-Pro, a selective subset that requires more complex reasoning. Our holistic evaluation surfaces insights about scientific reasoning performance that remain hidden when relying on individual benchmarks alone. We then propose KRUX, a probing framework for studying the distinct roles of reasoning and knowledge in scientific tasks. Combining the two, we conduct an in-depth analysis that yields several key findings: (1) Retrieving task-relevant knowledge from model parameters is a critical bottleneck for LLMs in scientific reasoning; (2) Reasoning models consistently benefit from external knowledge added in-context on top of the reasoning enhancement; (3) Enhancing verbalized reasoning improves LLMs' ability to surface task-relevant knowledge. Finally, we conduct a lightweight analysis, comparing our science-focused data composition with concurrent efforts on long CoT SFT, and release SciLit01, a strong 8B baseline for scientific reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08249v2">GeNet: A Multimodal LLM-Based Co-Pilot for Network Topology and Configuration</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Communication network engineering in enterprise environments is traditionally a complex, time-consuming, and error-prone manual process. Most research on network engineering automation has concentrated on configuration synthesis, often overlooking changes in the physical network topology. This paper introduces GeNet, a multimodal co-pilot for enterprise network engineers. GeNet is a novel framework that leverages a large language model (LLM) to streamline network design workflows. It uses visual and textual modalities to interpret and update network topologies and device configurations based on user intents. GeNet was evaluated on enterprise network scenarios adapted from Cisco certification exercises. Our results demonstrate GeNet's ability to interpret network topology images accurately, potentially reducing network engineers' efforts and accelerating network design processes in enterprise environments. Furthermore, we show the importance of precise topology understanding when handling intents that require modifications to the network's topology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19131v1">ZeST: an LLM-based Zero-Shot Traversability Navigation for Unknown Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The advancement of robotics and autonomous navigation systems hinges on the ability to accurately predict terrain traversability. Traditional methods for generating datasets to train these prediction models often involve putting robots into potentially hazardous environments, posing risks to equipment and safety. To solve this problem, we present ZeST, a novel approach leveraging visual reasoning capabilities of Large Language Models (LLMs) to create a traversability map in real-time without exposing robots to danger. Our approach not only performs zero-shot traversability and mitigates the risks associated with real-world data collection but also accelerates the development of advanced navigation systems, offering a cost-effective and scalable solution. To support our findings, we present navigation results, in both controlled indoor and unstructured outdoor environments. As shown in the experiments, our method provides safer navigation when compared to other state-of-the-art methods, constantly reaching the final goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05123v3">A Survey on Data Selection for LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Published in JAIR (Vol. 83, Article 32, 2025)
    </div>
    <details class="paper-abstract">
      Instruction tuning is a vital step of training large language models (LLMs), so how to enhance the effect of instruction tuning has received increased attention. Existing works indicate that the quality of the dataset is more crucial than the quantity during instruction tuning of LLMs. Therefore, recently a lot of studies focus on exploring the methods of selecting high-quality subset from instruction datasets, aiming to reduce training costs and enhance the instruction-following capabilities of LLMs. This paper presents a comprehensive survey on data selection for LLM instruction tuning. Firstly, we introduce the wildly used instruction datasets. Then, we propose a new taxonomy of the data selection methods and provide a detailed introduction of recent advances, and the evaluation strategies and results of data selection methods are also elaborated in detail. Finally, we emphasize the open challenges and present new frontiers of this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19097v1">Reasoning LLMs in the Medical Domain: A Literature Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The emergence of advanced reasoning capabilities in Large Language Models (LLMs) marks a transformative development in healthcare applications. Beyond merely expanding functional capabilities, these reasoning mechanisms enhance decision transparency and explainability-critical requirements in medical contexts. This survey examines the transformation of medical LLMs from basic information retrieval tools to sophisticated clinical reasoning systems capable of supporting complex healthcare decisions. We provide a thorough analysis of the enabling technological foundations, with a particular focus on specialized prompting techniques like Chain-of-Thought and recent breakthroughs in Reinforcement Learning exemplified by DeepSeek-R1. Our investigation evaluates purpose-built medical frameworks while also examining emerging paradigms such as multi-agent collaborative systems and innovative prompting architectures. The survey critically assesses current evaluation methodologies for medical validation and addresses persistent challenges in field interpretation limitations, bias mitigation strategies, patient safety frameworks, and integration of multimodal clinical data. Through this survey, we seek to establish a roadmap for developing reliable LLMs that can serve as effective partners in clinical practice and medical research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19089v1">It's All About In-Context Learning! Teaching Extremely Low-Resource Languages to LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Extremely low-resource languages, especially those written in rare scripts, as shown in Figure 1, remain largely unsupported by large language models (LLMs). This is due in part to compounding factors such as the lack of training data. This paper delivers the first comprehensive analysis of whether LLMs can acquire such languages purely via in-context learning (ICL), with or without auxiliary alignment signals, and how these methods compare to parameter-efficient fine-tuning (PEFT). We systematically evaluate 20 under-represented languages across three state-of-the-art multilingual LLMs. Our findings highlight the limitation of PEFT when both language and its script are extremely under-represented by the LLM. In contrast, zero-shot ICL with language alignment is impressively effective on extremely low-resource languages, while few-shot ICL or PEFT is more beneficial for languages relatively better represented by LLMs. For LLM practitioners working on extremely low-resource languages, we summarise guidelines grounded by our results on adapting LLMs to low-resource languages, e.g., avoiding fine-tuning a multilingual model on languages of unseen scripts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19087v1">APT-LLM: Exploiting Arbitrary-Precision Tensor Core Computing for LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 To appear in the IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized AI applications, yet their enormous computational demands severely limit deployment and real-time performance. Quantization methods can help reduce computational costs, however, attaining the extreme efficiency associated with ultra-low-bit quantized LLMs at arbitrary precision presents challenges on GPUs. This is primarily due to the limited support for GPU Tensor Cores, inefficient memory management, and inflexible kernel optimizations. To tackle these challenges, we propose a comprehensive acceleration scheme for arbitrary precision LLMs, namely APT-LLM. Firstly, we introduce a novel data format, bipolar-INT, which allows for efficient and lossless conversion with signed INT, while also being more conducive to parallel computation. We also develop a matrix multiplication (MatMul) method allowing for arbitrary precision by dismantling and reassembling matrices at the bit level. This method provides flexible precision and optimizes the utilization of GPU Tensor Cores. In addition, we propose a memory management system focused on data recovery, which strategically employs fast shared memory to substantially increase kernel execution speed and reduce memory access latency. Finally, we develop a kernel mapping method that dynamically selects the optimal configurable hyperparameters of kernels for varying matrix sizes, enabling optimal performance across different LLM architectures and precision settings. In LLM inference, APT-LLM achieves up to a 3.99$\times$ speedup compared to FP16 baselines and a 2.16$\times$ speedup over NVIDIA CUTLASS INT4 acceleration on RTX 3090. On RTX 4090 and H800, APT-LLM achieves up to 2.44$\times$ speedup over FP16 and 1.65$\times$ speedup over CUTLASS integer baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19076v1">HiPlan: Hierarchical Planning for LLM-Based Agents with Adaptive Global-Local Guidance</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents have demonstrated remarkable capabilities in decision-making tasks, but struggle significantly with complex, long-horizon planning scenarios. This arises from their lack of macroscopic guidance, causing disorientation and failures in complex tasks, as well as insufficient continuous oversight during execution, rendering them unresponsive to environmental changes and prone to deviations. To tackle these challenges, we introduce HiPlan, a hierarchical planning framework that provides adaptive global-local guidance to boost LLM-based agents'decision-making. HiPlan decomposes complex tasks into milestone action guides for general direction and step-wise hints for detailed actions. During the offline phase, we construct a milestone library from expert demonstrations, enabling structured experience reuse by retrieving semantically similar tasks and milestones. In the execution phase, trajectory segments from past milestones are dynamically adapted to generate step-wise hints that align current observations with the milestone objectives, bridging gaps and correcting deviations. Extensive experiments across two challenging benchmarks demonstrate that HiPlan substantially outperforms strong baselines, and ablation studies validate the complementary benefits of its hierarchical components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06029v2">SmartBench: Is Your LLM Truly a Good Chinese Smartphone Assistant?</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to daily life, especially advancing as intelligent assistants through on-device deployment on smartphones. However, existing LLM evaluation benchmarks predominantly focus on objective tasks like mathematics and coding in English, which do not necessarily reflect the practical use cases of on-device LLMs in real-world mobile scenarios, especially for Chinese users. To address these gaps, we introduce SmartBench, the first benchmark designed to evaluate the capabilities of on-device LLMs in Chinese mobile contexts. We analyze functionalities provided by representative smartphone manufacturers and divide them into five categories: text summarization, text Q&A, information extraction, content creation, and notification management, further detailed into 20 specific tasks. For each task, we construct high-quality datasets comprising 50 to 200 question-answer pairs that reflect everyday mobile interactions, and we develop automated evaluation criteria tailored for these tasks. We conduct comprehensive evaluations of on-device LLMs and MLLMs using SmartBench and also assess their performance after quantized deployment on real smartphone NPUs. Our contributions provide a standardized framework for evaluating on-device LLMs in Chinese, promoting further development and optimization in this critical area. Code and data will be available at https://github.com/vivo-ai-lab/SmartBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19074v1">An LLM-powered Natural-to-Robotic Language Translation Framework with Correctness Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The Large Language Models (LLM) are increasingly being deployed in robotics to generate robot control programs for specific user tasks, enabling embodied intelligence. Existing methods primarily focus on LLM training and prompt design that utilize LLMs to generate executable programs directly from user tasks in natural language. However, due to the inconsistency of the LLMs and the high complexity of the tasks, such best-effort approaches often lead to tremendous programming errors in the generated code, which significantly undermines the effectiveness especially when the light-weight LLMs are applied. This paper introduces a natural-robotic language translation framework that (i) provides correctness verification for generated control programs and (ii) enhances the performance of LLMs in program generation via feedback-based fine-tuning for the programs. To achieve this, a Robot Skill Language (RSL) is proposed to abstract away from the intricate details of the control programs, bridging the natural language tasks with the underlying robot skills. Then, the RSL compiler and debugger are constructed to verify RSL programs generated by the LLM and provide error feedback to the LLM for refining the outputs until being verified by the compiler. This provides correctness guarantees for the LLM-generated programs before being offloaded to the robots for execution, significantly enhancing the effectiveness of LLM-powered robotic applications. Experiments demonstrate NRTrans outperforms the existing method under a range of LLMs and tasks, and achieves a high success rate for light-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19069v1">Can Structured Templates Facilitate LLMs in Tackling Harder Tasks? : An Exploration of Scaling Laws by Difficulty</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Structured, procedural reasoning is essential for Large Language Models (LLMs), especially in mathematics. While post-training methods have improved LLM performance, they still fall short in capturing deep procedural logic on complex tasks. To tackle the issue, in this paper, we first investigate this limitation and uncover a novel finding: a Scaling Law by Difficulty, which reveals that model performance follows a U-shaped curve with respect to training data complexity -- excessive low-difficulty data impedes abstraction, while high-difficulty data significantly enhances reasoning ability. Motivated by this, we propose the Structured Solution Template (SST) framework, which uses solution templates and a curriculum of varied difficulty to explicitly teach procedural reasoning. Specifically, SST comprises (1) fine-tuning with structured solution-template chains and dynamically weighted loss to prioritize procedural logic, (2) prompt-time injection of solution templates as cognitive scaffolds to guide inference, and (3) integrated curriculum fine-tuning that explicitly teaches the model to self-plan - execute - self-correct. Experiments on GSM8K, AIME24, and new Dynamic En benchmark show that SST significantly improves both accuracy and efficiency, especially on harder problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17178v2">SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/Lza12a/SKA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19042v1">A Concurrent Modular Agent: Framework for Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      We introduce the Concurrent Modular Agent (CMA), a framework that orchestrates multiple Large-Language-Model (LLM)-based modules that operate fully asynchronously yet maintain a coherent and fault-tolerant behavioral loop. This framework addresses long-standing difficulties in agent architectures by letting intention emerge from language-mediated interactions among autonomous processes. This approach enables flexible, adaptive, and context-dependent behavior through the combination of concurrently executed modules that offload reasoning to an LLM, inter-module communication, and a single shared global state.We consider this approach to be a practical realization of Minsky's Society of Mind theory. We demonstrate the viability of our system through two practical use-case studies. The emergent properties observed in our system suggest that complex cognitive phenomena like self-awareness may indeed arise from the organized interaction of simpler processes, supporting Minsky-Society of Mind concept and opening new avenues for artificial intelligence research. The source code for our work is available at: https://github.com/AlternativeMachine/concurrent-modular-agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13500v2">LLM-Enhanced Linear Autoencoders for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Accepted by CIKM 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted to enrich the semantic representation of textual item information in recommender systems. However, existing linear autoencoders (LAEs) that incorporate textual information rely on sparse word co-occurrence patterns, limiting their ability to capture rich textual semantics. To address this, we propose L3AE, the first integration of LLMs into the LAE framework. L3AE effectively integrates the heterogeneous knowledge of textual semantics and user-item interactions through a two-phase optimization strategy. (i) L3AE first constructs a semantic item-to-item correlation matrix from LLM-derived item representations. (ii) It then learns an item-to-item weight matrix from collaborative signals while distilling semantic item correlations as regularization. Notably, each phase of L3AE is optimized through closed-form solutions, ensuring global optimality and computational efficiency. Extensive experiments demonstrate that L3AE consistently outperforms state-of-the-art LLM-enhanced models on three benchmark datasets, achieving gains of 27.6% in Recall@20 and 39.3% in NDCG@20. The source code is available at https://github.com/jaewan7599/L3AE_CIKM2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13972v2">Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Accepted at INLG 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models: being the same model, belonging to the same model family, being independent models, and having an distillation relationship. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, four generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18976v1">The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025
    </div>
    <details class="paper-abstract">
      Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16267v2">From Confidence to Collapse in LLM Factual Robustness</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18947v1">LLMs in the SOC: An Empirical Study of Human-AI Collaboration in Security Operations Centres</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 22 pages, 9 figures, under review
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Security Operations Centres (SOCs) presents a transformative, yet still evolving, opportunity to reduce analyst workload through human-AI collaboration. However, their real-world application in SOCs remains underexplored. To address this gap, we present a longitudinal study of 3,090 analyst queries from 45 SOC analysts over 10 months. Our analysis reveals that analysts use LLMs as on-demand aids for sensemaking and context-building, rather than for making high-stakes determinations, preserving analyst decision authority. The majority of queries are related to interpreting low-level telemetry (e.g., commands) and refining technical communication through short (1-3 turn) interactions. Notably, 93% of queries align with established cybersecurity competencies (NICE Framework), underscoring the relevance of LLM use for SOC-related tasks. Despite variations in tasks and engagement, usage trends indicate a shift from occasional exploration to routine integration, with growing adoption and sustained use among a subset of analysts. We find that LLMs function as flexible, on-demand cognitive aids that augment, rather than replace, SOC expertise. Our study provides actionable guidance for designing context-aware, human-centred AI assistance in security operations, highlighting the need for further in-the-wild research on real-world analyst-LLM collaboration, challenges, and impacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v2">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 This work is still in progress
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. This work is still in progress, and we will release the code, the models, and the datasets soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18918v1">DESAMO: A Device for Elder-Friendly Smart Homes Powered by Embedded LLM with Audio Modality</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 2 pages, 2 figures. Accepted for presentation as a UIST 2025 Poster
    </div>
    <details class="paper-abstract">
      We present DESAMO, an on-device smart home system for elder-friendly use powered by Audio LLM, that supports natural and private interactions. While conventional voice assistants rely on ASR-based pipelines or ASR-LLM cascades, often struggling with the unclear speech common among elderly users and unable to handle non-speech audio, DESAMO leverages an Audio LLM to process raw audio input directly, enabling a robust understanding of user intent and critical events, such as falls or calls for help.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09396v2">The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) has shifted artificial intelligence (AI) research toward agentic systems, motivating the use of weaker and more flexible notions of agency. However, this shift raises key questions about the extent to which LLM-based agents replicate human strategic reasoning, particularly in game-theoretic settings. In this context, we examine the role of agentic sophistication in shaping artificial reasoners' performance by evaluating three agent designs: a simple game-theoretic model, an unstructured LLM-as-agent model, and an LLM integrated into a traditional agentic framework. Using guessing games as a testbed, we benchmarked these agents against human participants across general reasoning patterns and individual role-based objectives. Furthermore, we introduced obfuscated game scenarios to assess agents' ability to generalise beyond training distributions. Our analysis, covering over 2000 reasoning samples across 25 agent configurations, shows that human-inspired cognitive structures can enhance LLM agents' alignment with human strategic behaviour. Still, the relationship between agentic design complexity and human-likeness is non-linear, highlighting a critical dependence on underlying LLM capabilities and suggesting limits to simple architectural augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18872v1">Empowering Computing Education Researchers Through LLM-Assisted Content Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Computing education research (CER) is often instigated by practitioners wanting to improve both their own and the wider discipline's teaching practice. However, the latter is often difficult as many researchers lack the colleagues, resources, or capacity to conduct research that is generalisable or rigorous enough to advance the discipline. As a result, research methods that enable sense-making with larger volumes of qualitative data, while not increasing the burden on the researcher, have significant potential within CER. In this discussion paper, we propose such a method for conducting rigorous analysis on large volumes of textual data, namely a variation of LLM-assisted content analysis (LACA). This method combines content analysis with the use of large language models, empowering researchers to conduct larger-scale research which they would otherwise not be able to perform. Using a computing education dataset, we illustrate how LACA could be applied in a reproducible and rigorous manner. We believe this method has potential in CER, enabling more generalisable findings from a wider range of research. This, together with the development of similar methods, can help to advance both the practice and research quality of the CER discipline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18850v1">ClusterFusion: Expanding Operator Fusion Scope for LLM Inference via Cluster-Level Collective Primitive</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Large language model (LLM) decoding suffers from high latency due to fragmented execution across operators and heavy reliance on off-chip memory for data exchange and reduction. This execution model limits opportunities for fusion and incurs significant memory traffic and kernel launch overhead. While modern architectures such as NVIDIA Hopper provide distributed shared memory and low-latency intra-cluster interconnects, they expose only low-level data movement instructions, lacking structured abstractions for collective on-chip communication. To bridge this software-hardware gap, we introduce two cluster-level communication primitives, ClusterReduce and ClusterGather, which abstract common communication patterns and enable structured, high-speed data exchange and reduction between thread blocks within a cluster, allowing intermediate results to be on-chip without involving off-chip memory. Building on these abstractions, we design ClusterFusion, an execution framework that schedules communication and computation jointly to expand operator fusion scope by composing decoding stages such as QKV Projection, Attention, and Output Projection into a single fused kernels. Evaluations on H100 GPUs show that ClusterFusion outperforms state-of-the-art inference frameworks by 1.61x on average in end-to-end latency across different models and configurations. The source code is available at https://github.com/xinhao-luo/ClusterFusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18819v1">LLM-based Contrastive Self-Supervised AMR Learning with Masked Graph Autoencoders for Fake News Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation in the digital age has led to significant societal challenges. Existing approaches often struggle with capturing long-range dependencies, complex semantic relations, and the social dynamics influencing news dissemination. Furthermore, these methods require extensive labelled datasets, making their deployment resource-intensive. In this study, we propose a novel self-supervised misinformation detection framework that integrates both complex semantic relations using Abstract Meaning Representation (AMR) and news propagation dynamics. We introduce an LLM-based graph contrastive loss (LGCL) that utilizes negative anchor points generated by a Large Language Model (LLM) to enhance feature separability in a zero-shot manner. To incorporate social context, we employ a multi view graph masked autoencoder, which learns news propagation features from social context graph. By combining these semantic and propagation-based features, our approach effectively differentiates between fake and real news in a self-supervised manner. Extensive experiments demonstrate that our self-supervised framework achieves superior performance compared to other state-of-the-art methodologies, even with limited labelled datasets while improving generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18190v2">ST-Raptor: LLM-Powered Semi-Structured Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Extension of our SIGMOD 2026 paper. Please refer to source code available at: https://github.com/weAIDB/ST-Raptor
    </div>
    <details class="paper-abstract">
      Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at https://github.com/weAIDB/ST-Raptor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18736v1">Rethinking Caching for LLM Serving Systems: Beyond Traditional Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) at scale requires meeting strict Service Level Objectives (SLOs) under severe computational and memory constraints. Nevertheless, traditional caching strategies fall short: exact-matching and prefix caches neglect query semantics, while state-of-the-art semantic caches remain confined to traditional intuitions, offering little conceptual departure. Building on this, we present SISO, a semantic caching system that redefines efficiency for LLM serving. SISO introduces centroid-based caching to maximize coverage with minimal memory, locality-aware replacement to preserve high-value entries, and dynamic thresholding to balance accuracy and latency under varying workloads. Across diverse datasets, SISO delivers up to 1.71$\times$ higher hit ratios and consistently stronger SLO attainment compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18721v1">LLM as an Execution Estimator: Recovering Missing Dependency for Practical Time-travelling Debugging</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Dynamic data dependency, answering "why a variable has this value?", is critical for debugging. Given a program step `s` reading a variable `v`, finding the dynamic definition of `v` is challenging. Traditional methods require either (1) exhaustive instrumentation of all possible definitions of `v` in one run or (2) replicating the run to re-examine reads/writes - both costly. If `v` is defined in a library, instrumentation becomes expensive; for non-deterministic programs, replication is infeasible. We propose RecovSlicing, which computes dynamic data dependency in a single run with partial instrumentation. We leverage LLMs to infer program behavior from a partially recorded trace and code context. Given a trace and a slicing criterion (step `s` and variable `v`), RecovSlicing estimates the runtime definition of `v` by recovering the missing execution.It also supports implicit variables, such as those in `list.get(i)`. Technically, RecovSlicing tackles: (1) recovering runtime values and structures, and (2) aligning recovered variables with recorded memory to analyze definitions. We evaluate RecovSlicing on 8,300 data dependencies across three slicing benchmarks, comparing it with Slicer4J, ND-Slicer, LLM Slicer, and re-execution Slicer. RecovSlicing achieves accuracy of 80.3%, 91.1%, and 98.3%, outperforming the best baseline (39.0%, 82.0%, 59.9%), and also leads in recall (91.1%, 91.1%, 98.3% vs. 53.4%, 79.1%, 87.1%). Integrated into a regression bug localizer, it enables finding 16% more regressions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18709v1">Filtering for Creativity: Adaptive Prompting for Multilingual Riddle Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Multilingual riddle generation challenges large language models (LLMs) to balance cultural fluency with creative abstraction. Standard prompting strategies -- zero-shot, few-shot, chain-of-thought -- tend to reuse memorized riddles or perform shallow paraphrasing. We introduce Adaptive Originality Filtering (AOF), a prompting framework that filters redundant generations using cosine-based similarity rejection, while enforcing lexical novelty and cross-lingual fidelity. Evaluated across three LLMs and four language pairs, AOF-enhanced GPT-4o achieves \texttt{0.177} Self-BLEU and \texttt{0.915} Distinct-2 in Japanese, signaling improved lexical diversity and reduced redundancy compared to other prompting methods and language pairs. Our findings show that semantic rejection can guide culturally grounded, creative generation without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08392v3">Multi-Agent LLMs as Ethics Advocates for AI-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18684v1">FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 11 pages, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13811v3">Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18676v1">Utilizing Training Data to Improve LLM Reasoning for Tabular Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Automated tabular understanding and reasoning are essential tasks for data scientists. Recently, Large language models (LLMs) have become increasingly prevalent in tabular reasoning tasks. Previous work focuses on (1) finetuning LLMs using labeled data or (2) Training-free prompting LLM agents using chain-of-thought (CoT). Finetuning offers dataset-specific learning at the cost of generalizability. Training-free prompting is highly generalizable but does not take full advantage of training data. In this paper, we propose a novel prompting-based reasoning approach, Learn then Retrieve: LRTab, which integrates the benefits of both by retrieving relevant information learned from training data. We first use prompting to obtain CoT responses over the training data. For incorrect CoTs, we prompt the LLM to predict Prompt Conditions to avoid the error, learning insights from the data. We validate the effectiveness of Prompt Conditions using validation data. Finally, at inference time, we retrieve the most relevant Prompt Conditions for additional context for table understanding. We provide comprehensive experiments on WikiTQ and Tabfact, showing that LRTab is interpretable, cost-efficient, and can outperform previous baselines in tabular reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18665v1">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17691v2">ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences via Tournament Graph Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Pairwise evaluation of large language models (LLMs) has become the dominant paradigm for benchmarking open-ended tasks, yet non-transitive preferences, where evaluators prefer A over B, B over C, but C over A, fundamentally undermine ranking reliability. We show that this critical issue stems largely from low-quality data that contains inherently ambiguous preference pairs. To address this challenge, we propose ELSPR, a principled graph-theoretic framework that models pairwise preferences as tournament graphs and systematically identifies problematic training data. ELSPR quantifies non-transitivity through strongly connected components (SCCs) analysis and measures overall preference clarity using a novel normalized directed graph structural entropy metric. Our filtering methodology selectively removes preference data that induce non-transitivity while preserving transitive preferences. Extensive experiments on the AlpacaEval benchmark demonstrate that models fine-tuned on ELSPR-filtered data achieve substantial improvements: a 13.8% reduction in non-transitivity, a 0.088 decrease in structural entropy, and significantly enhanced discriminative power in real-world evaluation systems. Human validation confirms that discarded data exhibit dramatically lower inter-annotator agreement (34.4% vs. 52.6%) and model-human consistency (51.2% vs. 80.6%) compared to cleaned data. These findings establish ELSPR as an effective data self-purification approach for developing more robust, consistent, and human-aligned LLM evaluation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18646v1">Beyond Benchmark: LLMs Evaluation with an Anthropomorphic and Value-oriented Roadmap</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      For Large Language Models (LLMs), a disconnect persists between benchmark performance and real-world utility. Current evaluation frameworks remain fragmented, prioritizing technical metrics while neglecting holistic assessment for deployment. This survey introduces an anthropomorphic evaluation paradigm through the lens of human intelligence, proposing a novel three-dimensional taxonomy: Intelligence Quotient (IQ)-General Intelligence for foundational capacity, Emotional Quotient (EQ)-Alignment Ability for value-based interactions, and Professional Quotient (PQ)-Professional Expertise for specialized proficiency. For practical value, we pioneer a Value-oriented Evaluation (VQ) framework assessing economic viability, social impact, ethical alignment, and environmental sustainability. Our modular architecture integrates six components with an implementation roadmap. Through analysis of 200+ benchmarks, we identify key challenges including dynamic assessment needs and interpretability gaps. It provides actionable guidance for developing LLMs that are technically proficient, contextually relevant, and ethically sound. We maintain a curated repository of open-source evaluation resources at: https://github.com/onejune2018/Awesome-LLM-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02502v2">Unveiling the Landscape of LLM Deployment in the Wild: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed through open-source and commercial frameworks, enabling individuals and organizations to self-host advanced LLM capabilities. As LLM deployments become prevalent, particularly in industry, ensuring their secure and reliable operation has become a critical issue. However, insecure defaults and misconfigurations often expose LLM services to the public internet, posing serious security and system engineering risks. This study conducted a large-scale empirical investigation of public-facing LLM deployments, focusing on the prevalence of services, exposure characteristics, systemic vulnerabilities, and associated risks. Through internet-wide measurements, we identified 320,102 public-facing LLM services across 15 frameworks and extracted 158 unique API endpoints, categorized into 12 functional groups based on functionality and security risk. Our analysis found that over 40% of endpoints used plain HTTP, and over 210,000 endpoints lacked valid TLS metadata. API exposure was highly inconsistent: some frameworks, such as Ollama, responded to over 35% of unauthenticated API requests, with about 15% leaking model or system information, while other frameworks implemented stricter controls. We observed widespread use of insecure protocols, poor TLS configurations, and unauthenticated access to critical operations. These security risks, such as model leakage, system compromise, and unauthorized access, are pervasive and highlight the need for a secure-by-default framework and stronger deployment practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18636v1">LaQual: A Novel Framework for Automated Evaluation of LLM App Quality</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      LLM app stores are quickly emerging as platforms that gather a wide range of intelligent applications based on LLMs, giving users many choices for content creation, coding support, education, and more. However, the current methods for ranking and recommending apps in these stores mostly rely on static metrics like user activity and favorites, which makes it hard for users to efficiently find high-quality apps. To address these challenges, we propose LaQual, an automated framework for evaluating the quality of LLM apps. LaQual consists of three main stages: first, it labels and classifies LLM apps in a hierarchical way to accurately match them to different scenarios; second, it uses static indicators, such as time-weighted user engagement and functional capability metrics, to filter out low-quality apps; and third, it conducts a dynamic, scenario-adaptive evaluation, where the LLM itself generates scenario-specific evaluation metrics, scoring rules, and tasks for a thorough quality assessment. Experiments on a popular LLM app store show that LaQual is effective. Its automated scores are highly consistent with human judgments (with Spearman's rho of 0.62 and p=0.006 in legal consulting, and rho of 0.60 and p=0.009 in travel planning). By effectively screening, LaQual can reduce the pool of candidate LLM apps by 66.7% to 81.3%. User studies further confirm that LaQual significantly outperforms baseline systems in decision confidence, comparison efficiency (with average scores of 5.45 compared to 3.30), and the perceived value of its evaluation reports (4.75 versus 2.25). Overall, these results demonstrate that LaQual offers a scalable, objective, and user-centered solution for finding and recommending high-quality LLM apps in real-world use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22316v3">Evaluating Scoring Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18610v1">Scalable Fairness Shaping with LLM-Guided Multi-Agent Reinforcement Learning for Peer-to-Peer Electricity Markets</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      Peer-to-peer (P2P) energy trading is becoming central to modern distribution systems as rooftop PV and home energy management systems become pervasive, yet most existing market and reinforcement learning designs emphasize efficiency or private profit and offer little real-time guidance to ensure equitable outcomes under uncertainty. To address this gap, a fairness-aware multiagent reinforcement learning framework, FairMarket-RL, is proposed in which a large language model (LLM) critic shapes bidding policies within a continuous double auction under partial observability and discrete price-quantity actions. After each trading slot, the LLM returns normalized fairness scores Fairness-to-Grid (FTG), Fairness-Between-Sellers (FBS), and Fairness-of-Pricing (FPP) that are integrated into the reward via ramped coefficients and tunable scaling, so that fairness guidance complements, rather than overwhelms, economic incentives. The environment models realistic residential load and PV profiles and enforce hard constraints on prices, physical feasibility, and policy-update stability. Across a progression of experiments from a small pilot to a larger simulated community and a mixed-asset real-world dataset, the framework shifts exchanges toward local P2P trades, lowers consumer costs relative to grid-only procurement, sustains strong fairness across participants, and preserves utility viability. Sensitivity analyses over solar availability and aggregate demand further indicate robust performance, suggesting a scalable, LLM-guided pathway to decentralized electricity markets that are economically efficient, socially equitable, and technically sound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18600v1">Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate human decision-making, but their intrinsic biases often diverge from real human behavior--limiting their ability to reflect population-level diversity. We address this challenge with a persona-based approach that leverages individual-level behavioral data from behavioral economics to adjust model biases. Applying this method to the ultimatum game--a standard but difficult benchmark for LLMs--we observe improved alignment between simulated and empirical behavior, particularly on the responder side. While further refinement of trait representations is needed, our results demonstrate the promise of persona-conditioned LLMs for simulating human-like decision patterns at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18588v1">History Rhymes: Accelerating LLM Reinforcement Learning with RhymeRL</a></div>
    <div class="paper-meta">
      📅 2025-08-26
    </div>
    <details class="paper-abstract">
      With the rapid advancement of large language models (LLMs), reinforcement learning (RL) has emerged as a pivotal methodology for enhancing the reasoning capabilities of LLMs. Unlike traditional pre-training approaches, RL encompasses multiple stages: rollout, reward, and training, which necessitates collaboration among various worker types. However, current RL systems continue to grapple with substantial GPU underutilization, due to two primary factors: (1) The rollout stage dominates the overall RL process due to test-time scaling; (2) Imbalances in rollout lengths (within the same batch) result in GPU bubbles. While prior solutions like asynchronous execution and truncation offer partial relief, they may compromise training accuracy for efficiency. Our key insight stems from a previously overlooked observation: rollout responses exhibit remarkable similarity across adjacent training epochs. Based on the insight, we introduce RhymeRL, an LLM RL system designed to accelerate RL training with two key innovations. First, to enhance rollout generation, we present HistoSpec, a speculative decoding inference engine that utilizes the similarity of historical rollout token sequences to obtain accurate drafts. Second, to tackle rollout bubbles, we introduce HistoPipe, a two-tier scheduling strategy that leverages the similarity of historical rollout distributions to balance workload among rollout workers. We have evaluated RhymeRL within a real production environment, demonstrating scalability from dozens to thousands of GPUs. Experimental results demonstrate that RhymeRL achieves a 2.6x performance improvement over existing methods, without compromising accuracy or modifying the RL paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18587v1">A Case Study on the Effectiveness of LLMs in Verification with Proof Assistants</a></div>
    <div class="paper-meta">
      📅 2025-08-26
      | 💬 Accepted by LMPL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can potentially help with verification using proof assistants by automating proofs. However, it is unclear how effective LLMs are in this task. In this paper, we perform a case study based on two mature Rocq projects: the hs-to-coq tool and Verdi. We evaluate the effectiveness of LLMs in generating proofs by both quantitative and qualitative analysis. Our study finds that: (1) external dependencies and context in the same source file can significantly help proof generation; (2) LLMs perform great on small proofs but can also generate large proofs; (3) LLMs perform differently on different verification projects; and (4) LLMs can generate concise and smart proofs, apply classical techniques to new definitions, but can also make odd mistakes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18253v1">From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature. To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance. Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18190v1">ST-Raptor: LLM-Powered Semi-Structured Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Extension of our SIGMOD 2026 paper. Please refer to source code available at: https://github.com/weAIDB/ST-Raptor
    </div>
    <details class="paper-abstract">
      Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at https://github.com/weAIDB/ST-Raptor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12964v2">Trust Me, I'm Wrong: LLMs Hallucinate with Certainty Despite Knowing the Answer</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Prior work on large language model (LLM) hallucinations has associated them with model uncertainty or inaccurate knowledge. In this work, we define and investigate a distinct type of hallucination, where a model can consistently answer a question correctly, but a seemingly trivial perturbation, which can happen in real-world settings, causes it to produce a hallucinated response with high certainty. This phenomenon, which we dub CHOKE (Certain Hallucinations Overriding Known Evidence), is particularly concerning in high-stakes domains such as medicine or law, where model certainty is often used as a proxy for reliability. We show that CHOKE examples are consistent across prompts, occur in different models and datasets, and are fundamentally distinct from other hallucinations. This difference leads existing mitigation methods to perform worse on CHOKE examples than on general hallucinations. Finally, we introduce a probing-based mitigation that outperforms existing methods on CHOKE hallucinations. These findings reveal an overlooked aspect of hallucinations, emphasizing the need to understand their origins and improve mitigation strategies to enhance LLM safety. The code is available at https://github.com/technion-cs-nlp/Trust_me_Im_wrong .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12349v4">SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 48 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Reasoning and strategic behavior in social interactions is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present Strategic Planning, Interaction, and Negotiation (SPIN-Bench), a new multi-domain evaluation designed to measure the intelligence of strategic planning and social reasoning. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also conceptual inference of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle basic fact retrieval and short-range planning reasonably well, they encounter significant performance bottlenecks in tasks requiring deep multi-hop reasoning over large state spaces and socially adept coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. Project Website: https://spinbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19134v4">Confidential Prompting: Privacy-preserving LLM Inference on Cloud</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      This paper introduces a vision of confidential prompting: securing user prompts from untrusted, cloud-hosted large language model (LLM) provider while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Obfuscated Secure Partitioned Decoding (OSPD), a system built on two key innovations. First, Secure Partitioned Decoding (SPD) isolates user prompts within per-user processes residing in a confidential virtual machine (CVM) on the cloud, which are inaccessible for the cloud LLM while allowing it to generate tokens efficiently. Second, Prompt Obfuscation (PO) introduces a novel cryptographic technique that enhances SPD resilience against advanced prompt reconstruction attacks. Together, these innovations ensure OSPD protects both prompt and model confidentiality while maintaining service functionality. OSPD enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18118v1">HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      AI-generated content technologies are widely used in content creation. However, current AIGC systems rely heavily on creators' inspiration, rarely generating truly user-personalized content. In real-world applications such as online advertising, a single product may have multiple selling points, with different users focusing on different features. This underscores the significant value of personalized, user-centric creative generation. Effective personalized content generation faces two main challenges: (1) accurately modeling user interests and integrating them into the content generation process while adhering to factual constraints, and (2) ensuring high efficiency and scalability to handle the massive user base in industrial scenarios. Additionally, the scarcity of personalized creative data in practice complicates model training, making data construction another key hurdle. We propose HLLM-Creator, a hierarchical LLM framework for efficient user interest modeling and personalized content generation. During inference, a combination of user clustering and a user-ad-matching-prediction based pruning strategy is employed to significantly enhance generation efficiency and reduce computational overhead, making the approach suitable for large-scale deployment. Moreover, we design a data construction pipeline based on chain-of-thought reasoning, which generates high-quality, user-specific creative titles and ensures factual consistency despite limited personalized data. This pipeline serves as a critical foundation for the effectiveness of our model. Extensive experiments on personalized title generation for Douyin Search Ads show the effectiveness of HLLM-Creator. Online A/B test shows a 0.476% increase on Adss, paving the way for more effective and efficient personalized generation in industrial scenarios. Codes for academic dataset are available at https://github.com/bytedance/HLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02531v2">Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 20 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As general-purpose artificial intelligence systems become increasingly integrated into society and are used for information seeking, content generation, problem solving, textual analysis, coding, and running processes, it is crucial to assess their long-term impact on humans. This research explores the sentiment of large language models (LLMs) and humans toward artificial general intelligence (AGI) using a Likert-scale survey. Seven LLMs, including GPT-4 and Bard, were analyzed and compared with sentiment data from three independent human sample populations. Temporal variations in sentiment were also evaluated over three consecutive days. The results show a diversity in sentiment scores among LLMs, ranging from 3.32 to 4.12 out of 5. GPT-4 recorded the most positive sentiment toward AGI, while Bard leaned toward a neutral sentiment. In contrast, the human samples showed a lower average sentiment of 2.97. The analysis outlines potential conflicts of interest and biases in the sentiment formation of LLMs, and indicates that LLMs could subtly influence societal perceptions. To address the need for regulatory oversight and culturally grounded assessments of AI systems, we introduce the Societal AI Alignment and Sentiment Benchmark (SAAS-AI), which leverages multidimensional prompts and empirically validated societal value frameworks to evaluate language model outputs across temporal, model, and multilingual axes. This benchmark is designed to guide policymakers and AI agencies, including within frameworks such as the EU AI Act, by providing robust, actionable insights into AI alignment with human values, public sentiment, and ethical norms at both national and international levels. Future research should further refine the operationalization of the SAAS-AI benchmark and systematically evaluate its effectiveness through comprehensive empirical testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18093v1">Agri-Query: A Case Study on RAG vs. Long-Context LLMs for Cross-Lingual Technical Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      We present a case study evaluating large language models (LLMs) with 128K-token context windows on a technical question answering (QA) task. Our benchmark is built on a user manual for an agricultural machine, available in English, French, and German. It simulates a cross-lingual information retrieval scenario where questions are posed in English against all three language versions of the manual. The evaluation focuses on realistic "needle-in-a-haystack" challenges and includes unanswerable questions to test for hallucinations. We compare nine long-context LLMs using direct prompting against three Retrieval-Augmented Generation (RAG) strategies (keyword, semantic, hybrid), with an LLM-as-a-judge for evaluation. Our findings for this specific manual show that Hybrid RAG consistently outperforms direct long-context prompting. Models like Gemini 2.5 Flash and the smaller Qwen 2.5 7B achieve high accuracy (over 85%) across all languages with RAG. This paper contributes a detailed analysis of LLM performance in a specialized industrial domain and an open framework for similar evaluations, highlighting practical trade-offs and challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18091v1">Teaching LLMs to Think Mathematically: A Critical Study of Decision-Making via Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      This paper investigates the capabilities of large language models (LLMs) in formulating and solving decision-making problems using mathematical programming. We first conduct a systematic review and meta-analysis of recent literature to assess how well LLMs understand, structure, and solve optimization problems across domains. The analysis is guided by critical review questions focusing on learning approaches, dataset designs, evaluation metrics, and prompting strategies. Our systematic evidence is complemented by targeted experiments designed to evaluate the performance of state-of-the-art LLMs in automatically generating optimization models for problems in computer networks. Using a newly constructed dataset, we apply three prompting strategies: Act-as-expert, chain-of-thought, and self-consistency, and evaluate the obtained outputs based on optimality gap, token-level F1 score, and compilation accuracy. Results show promising progress in LLMs' ability to parse natural language and represent symbolic formulations, but also reveal key limitations in accuracy, scalability, and interpretability. These empirical gaps motivate several future research directions, including structured datasets, domain-specific fine-tuning, hybrid neuro-symbolic approaches, modular multi-agent architectures, and dynamic retrieval via chain-of-RAGs. This paper contributes a structured roadmap for advancing LLM capabilities in mathematical programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18089v1">LLM-Guided Genetic Improvement: Envisioning Semantic Aware Automated Software Evolution</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Genetic Improvement (GI) of software automatically creates alternative software versions that are improved according to certain properties of interests (e.g., running-time). Search-based GI excels at navigating large program spaces, but operates primarily at the syntactic level. In contrast, Large Language Models (LLMs) offer semantic-aware edits, yet lack goal-directed feedback and control (which is instead a strength of GI). As such, we propose the investigation of a new research line on AI-powered GI aimed at incorporating semantic aware search. We take a first step at it by augmenting GI with the use of automated clustering of LLM edits. We provide initial empirical evidence that our proposal, dubbed PatchCat, allows us to automatically and effectively categorize LLM-suggested patches. PatchCat identified 18 different types of software patches and categorized newly suggested patches with high accuracy. It also enabled detecting NoOp edits in advance and, prospectively, to skip test suite execution to save resources in many cases. These results, coupled with the fact that PatchCat works with small, local LLMs, are a promising step toward interpretable, efficient, and green GI. We outline a rich agenda of future work and call for the community to join our vision of building a principled understanding of LLM-driven mutations, guiding the GI search process with semantic signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18076v1">Neither Valid nor Reliable? Investigating the Use of LLMs as Judges</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Prepared for conference submission
    </div>
    <details class="paper-abstract">
      Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18048v1">HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted at the 2nd EARL Workshop on Evaluating and Applying Recommender Systems with Large Language Models (RecSys 2025)
    </div>
    <details class="paper-abstract">
      User queries in real-world recommendation systems often combine structured constraints (e.g., category, attributes) with unstructured preferences (e.g., product descriptions or reviews). We introduce HyST (Hybrid retrieval over Semi-structured Tabular data), a hybrid retrieval framework that combines LLM-powered structured filtering with semantic embedding search to support complex information needs over semi-structured tabular data. HyST extracts attribute-level constraints from natural language using large language models (LLMs) and applies them as metadata filters, while processing the remaining unstructured query components via embedding-based retrieval. Experiments on a semi-structured benchmark show that HyST consistently outperforms tradtional baselines, highlighting the importance of structured filtering in improving retrieval precision, offering a scalable and accurate solution for real-world user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16153v2">Memento: Fine-tuning LLM Agents without Fine-tuning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a novel learning paradigm for Adaptive Large Language Model (LLM) agents that eliminates the need for fine-tuning the underlying LLMs. Existing approaches are often either rigid, relying on static, handcrafted reflection workflows, or computationally intensive, requiring gradient updates of LLM model parameters. In contrast, our method enables low-cost continual adaptation via memory-based online reinforcement learning. We formalise this as a Memory-augmented Markov Decision Process (M-MDP), equipped with a neural case-selection policy to guide action decisions. Past experiences are stored in an episodic memory, either differentiable or non-parametric. The policy is continually updated based on environmental feedback through a memory rewriting mechanism, whereas policy improvement is achieved through efficient memory reading (retrieval). We instantiate our agent model in the deep research setting, namely \emph{Memento}, which attains top-1 on GAIA validation ($87.88\%$ Pass@$3$) and $79.40\%$ on the test set. It reaches $66.6\%$ F1 and $80.4\%$ PM on the DeepResearcher dataset, outperforming the state-of-the-art training-based method, while case-based memory adds $4.7\%$ to $9.6\%$ absolute points on out-of-distribution tasks. Our approach offers a scalable and efficient pathway for developing generalist LLM agents capable of continuous, real-time learning without gradient updates, advancing machine learning towards open-ended skill acquisition and deep research scenarios. The code is available at https://github.com/Agent-on-the-Fly/Memento.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17948v1">Debiasing Multilingual LLMs in Cross-lingual Latent Space</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Debiasing techniques such as SentDebias aim to reduce bias in large language models (LLMs). Previous studies have evaluated their cross-lingual transferability by directly applying these methods to LLM representations, revealing their limited effectiveness across languages. In this work, we therefore propose to perform debiasing in a joint latent space rather than directly on LLM representations. We construct a well-aligned cross-lingual latent space using an autoencoder trained on parallel TED talk scripts. Our experiments with Aya-expanse and two debiasing techniques across four languages (English, French, German, Dutch) demonstrate that a) autoencoders effectively construct a well-aligned cross-lingual latent space, and b) applying debiasing techniques in the learned cross-lingual latent space significantly improves both the overall debiasing performance and cross-lingual transferability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17884v1">PhantomLint: Principled Detection of Hidden LLM Prompts in Structured Documents</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Hidden LLM prompts have appeared in online documents with increasing frequency. Their goal is to trigger indirect prompt injection attacks while remaining undetected from human oversight, to manipulate LLM-powered automated document processing systems, against applications as diverse as r\'esum\'e screeners through to academic peer review processes. Detecting hidden LLM prompts is therefore important for ensuring trust in AI-assisted human decision making. This paper presents the first principled approach to hidden LLM prompt detection in structured documents. We implement our approach in a prototype tool called PhantomLint. We evaluate PhantomLint against a corpus of 3,402 documents, including both PDF and HTML documents, and covering academic paper preprints, CVs, theses and more. We find that our approach is generally applicable against a wide range of methods for hiding LLM prompts from visual inspection, has a very low false positive rate (approx. 0.092%), is practically useful for detecting hidden LLM prompts in real documents, while achieving acceptable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17856v1">MalLoc: Toward Fine-grained Android Malicious Payload Localization via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted at ICSME 2025, NIER Track
    </div>
    <details class="paper-abstract">
      The rapid evolution of Android malware poses significant challenges to the maintenance and security of mobile applications (apps). Traditional detection techniques often struggle to keep pace with emerging malware variants that employ advanced tactics such as code obfuscation and dynamic behavior triggering. One major limitation of these approaches is their inability to localize malicious payloads at a fine-grained level, hindering precise understanding of malicious behavior. This gap in understanding makes the design of effective and targeted mitigation strategies difficult, leaving mobile apps vulnerable to continuously evolving threats. To address this gap, we propose MalLoc, a novel approach that leverages the code understanding capabilities of large language models (LLMs) to localize malicious payloads at a fine-grained level within Android malware. Our experimental results demonstrate the feasibility and effectiveness of using LLMs for this task, highlighting the potential of MalLoc to enhance precision and interpretability in malware analysis. This work advances beyond traditional detection and classification by enabling deeper insights into behavior-level malicious logic and opens new directions for research, including dynamic modeling of localized threats and targeted countermeasure development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17850v1">Group Expectation Policy Optimization for Stable Heterogeneous Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As single-center computing approaches power constraints, decentralized training is becoming essential. Reinforcement Learning (RL) post-training enhances Large Language Models (LLMs) but faces challenges in heterogeneous distributed environments due to its tightly-coupled sampling-learning alternation. We propose HeteroRL, an asynchronous RL architecture that decouples rollout sampling from parameter learning, enabling robust deployment across geographically distributed nodes under network delays. We identify that latency-induced KL divergence causes importance sampling failure due to high variance. To address this, we propose Group Expectation Policy Optimization (GEPO), which reduces importance weight variance through a refined sampling mechanism. Theoretically, GEPO achieves exponential variance reduction. Experiments show it maintains superior stability over methods like GRPO, with less than 3% performance degradation under 1800-second delays, demonstrating strong potential for decentralized RL in heterogeneous networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17814v1">Scalable Engine and the Performance of Different LLM Models in a SLURM based HPC architecture</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted in ESSV 2025 - https://www.essv.de/paper.php?id=1265
    </div>
    <details class="paper-abstract">
      This work elaborates on a High performance computing (HPC) architecture based on Simple Linux Utility for Resource Management (SLURM) [1] for deploying heterogeneous Large Language Models (LLMs) into a scalable inference engine. Dynamic resource scheduling and seamless integration of containerized microservices have been leveraged herein to manage CPU, GPU, and memory allocations efficiently in multi-node clusters. Extensive experiments, using Llama 3.2 (1B and 3B parameters) [2] and Llama 3.1 (8B and 70B) [3], probe throughput, latency, and concurrency and show that small models can handle up to 128 concurrent requests at sub-50 ms latency, while for larger models, saturation happens with as few as two concurrent users, with a latency of more than 2 seconds. This architecture includes Representational State Transfer Application Programming Interfaces (REST APIs) [4] endpoints for single and bulk inferences, as well as advanced workflows such as multi-step "tribunal" refinement. Experimental results confirm minimal overhead from container and scheduling activities and show that the approach scales reliably both for batch and interactive settings. We further illustrate real-world scenarios, including the deployment of chatbots with retrievalaugmented generation, which helps to demonstrate the flexibility and robustness of the architecture. The obtained results pave ways for significantly more efficient, responsive, and fault-tolerant LLM inference on large-scale HPC infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17771v1">Speculating LLMs' Chinese Training Data Pollution from Their Tokens</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Tokens are basic elements in the datasets for LLM training. It is well-known that many tokens representing Chinese phrases in the vocabulary of GPT (4o/4o-mini/o1/o3/4.5/4.1/o4-mini) are indicating contents like pornography or online gambling. Based on this observation, our goal is to locate Polluted Chinese (PoC) tokens in LLMs and study the relationship between PoC tokens' existence and training data. (1) We give a formal definition and taxonomy of PoC tokens based on the GPT's vocabulary. (2) We build a PoC token detector via fine-tuning an LLM to label PoC tokens in vocabularies by considering each token's both semantics and related contents from the search engines. (3) We study the speculation on the training data pollution via PoC tokens' appearances (token ID). Experiments on GPT and other 23 LLMs indicate that tokens widely exist while GPT's vocabulary behaves the worst: more than 23% long Chinese tokens (i.e., a token with more than two Chinese characters) are either porn or online gambling. We validate the accuracy of our speculation method on famous pre-training datasets like C4 and Pile. Then, considering GPT-4o, we speculate that the ratio of "Yui Hatano" related webpages in GPT-4o's training data is around 0.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22316v2">Evaluating Scoring Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01263v2">FlowDubber: Movie Dubbing with LLM-based Semantic-aware Learning and Flow Matching based Voice Enhancing</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Movie Dubbing aims to convert scripts into speeches that align with the given movie clip in both temporal and emotional aspects while preserving the vocal timbre of a given brief reference audio. Existing methods focus primarily on reducing the word error rate while ignoring the importance of lip-sync and acoustic quality. To address these issues, we propose a large language model (LLM) based flow matching architecture for dubbing, named FlowDubber, which achieves high-quality audio-visual sync and pronunciation by incorporating a large speech language model and dual contrastive aligning while achieving better acoustic quality via the proposed voice-enhanced flow matching than previous works. First, we introduce Qwen2.5 as the backbone of LLM to learn the in-context sequence from movie scripts and reference audio. Then, the proposed semantic-aware learning focuses on capturing LLM semantic knowledge at the phoneme level. Next, dual contrastive aligning (DCA) boosts mutual alignment with lip movement, reducing ambiguities where similar phonemes might be confused. Finally, the proposed Flow-based Voice Enhancing (FVE) improves acoustic quality in two aspects, which introduces an LLM-based acoustics flow matching guidance to strengthen clarity and uses affine style prior to enhance identity when recovering noise into mel-spectrograms via gradient vector field prediction. Extensive experiments demonstrate that our method outperforms several state-of-the-art methods on two primary benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17735v1">SMITE: Enhancing Fairness in LLMs through Optimal In-Context Example Selection via Dynamic Validation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for downstream tasks such as tabular classification, where ensuring fairness in their outputs is critical for inclusivity, equal representation, and responsible AI deployment. This study introduces a novel approach to enhancing LLM performance and fairness through the concept of a dynamic validation set, which evolves alongside the test set, replacing the traditional static validation approach. We also propose an iterative algorithm, SMITE, to select optimal in-context examples, with each example set validated against its corresponding dynamic validation set. The in-context set with the lowest total error is used as the final demonstration set. Our experiments across four different LLMs show that our proposed techniques significantly improve both predictive accuracy and fairness compared to baseline methods. To our knowledge, this is the first study to apply dynamic validation in the context of in-context learning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17720v1">RepoTransAgent: Multi-Agent LLM Framework for Repository-Aware Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Repository-aware code translation is critical for modernizing legacy systems, enhancing maintainability, and enabling interoperability across diverse programming languages. While recent advances in large language models (LLMs) have improved code translation quality, existing approaches face significant challenges in practical scenarios: insufficient contextual understanding, inflexible prompt designs, and inadequate error correction mechanisms. These limitations severely hinder accurate and efficient translation of complex, real-world code repositories. To address these challenges, we propose RepoTransAgent, a novel multi-agent LLM framework for repository-aware code translation. RepoTransAgent systematically decomposes the translation process into specialized subtasks-context retrieval, dynamic prompt construction, and iterative code refinement-each handled by dedicated agents. Our approach leverages retrieval-augmented generation (RAG) for contextual information gathering, employs adaptive prompts tailored to varying repository scenarios, and introduces a reflection-based mechanism for systematic error correction. We evaluate RepoTransAgent on hundreds of Java-C# translation pairs from six popular open-source projects. Experimental results demonstrate that RepoTransAgent significantly outperforms state-of-the-art baselines in both compile and pass rates. Specifically, RepoTransAgent achieves up to 55.34% compile rate and 45.84% pass rate. Comprehensive analysis confirms the robustness and generalizability of RepoTransAgent across different LLMs, establishing its effectiveness for real-world repository-aware code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17715v1">How Do LLM-Generated Texts Impact Term-Based Retrieval Models?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As more content generated by large language models (LLMs) floods into the Internet, information retrieval (IR) systems now face the challenge of distinguishing and handling a blend of human-authored and machine-generated texts. Recent studies suggest that neural retrievers may exhibit a preferential inclination toward LLM-generated content, while classic term-based retrievers like BM25 tend to favor human-written documents. This paper investigates the influence of LLM-generated content on term-based retrieval models, which are valued for their efficiency and robust generalization across domains. Our linguistic analysis reveals that LLM-generated texts exhibit smoother high-frequency and steeper low-frequency Zipf slopes, higher term specificity, and greater document-level diversity. These traits are aligned with LLMs being trained to optimize reader experience through diverse and precise expressions. Our study further explores whether term-based retrieval models demonstrate source bias, concluding that these models prioritize documents whose term distributions closely correspond to those of the queries, rather than displaying an inherent source bias. This work provides a foundation for understanding and addressing potential biases in term-based IR systems managing mixed-source content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17711v1">Enhancing LLM-Based Social Bot via an Adversarial Learning Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Developing Large Language Model (LLM) agents that exhibit human-like behavior, encompassing not only individual heterogeneity rooted in unique user profiles but also adaptive response to socially connected neighbors, is a significant research challenge. Social media platforms, with their diverse user data and explicit social structures, provide an ideal testbed for such investigations. This paper introduces EvoBot, an \textbf{Evo}lving LLM-based social \textbf{Bot} that significantly enhances human-like generative capabilities through a novel adversarial learning framework. EvoBot is initialized by Supervised Fine-Tuning (SFT) on representative data from social media and then iteratively refines its generation of sophisticated, human-like content via Direct Preference Optimization (DPO). This refinement is guided by feedback from a co-adapting \textbf{Detector} which concurrently improves its ability to distinguish EvoBot from humans, thereby creating an increasingly challenging learning environment for EvoBot. Experiments demonstrate that EvoBot generates content aligned with diverse user profiles, increasingly bypassing the co-adapting Detector through human-like expression. Moreover, it exhibits strong social responsiveness, more accurately modeling real-world opinion dynamics and information spread in multi-agent simulations. The framework also yields a more robust Detector, underscoring its broader utility for both advanced agent development and related detection tasks. The code is available at https://github.com/kfq20/EvoBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10904v2">A2HCoder: An LLM-Driven Coding Agent for Hierarchical Algorithm-to-HDL Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      In wireless communication systems, stringent requirements such as ultra-low latency and power consumption have significantly increased the demand for efficient algorithm-to-hardware deployment. However, a persistent and substantial gap remains between algorithm design and hardware implementation. Bridging this gap traditionally requires extensive domain expertise and time-consuming manual development, due to fundamental mismatches between high-level programming languages like MATLAB and hardware description languages (HDLs) such as Verilog-in terms of memory access patterns, data processing manners, and datatype representations. To address this challenge, we propose A2HCoder: a Hierarchical Algorithm-to-HDL Coding Agent, powered by large language models (LLMs), designed to enable agile and reliable algorithm-to-hardware translation. A2HCoder introduces a hierarchical framework that enhances both robustness and interpretability while suppressing common hallucination issues in LLM-generated code. In the horizontal dimension, A2HCoder decomposes complex algorithms into modular functional blocks, simplifying code generation and improving consistency. In the vertical dimension, instead of relying on end-to-end generation, A2HCoder performs step-by-step, fine-grained translation, leveraging external toolchains such as MATLAB and Vitis HLS for debugging and circuit-level synthesis. This structured process significantly mitigates hallucinations and ensures hardware-level correctness. We validate A2HCoder through a real-world deployment case in the 5G wireless communication domain, demonstrating its practicality, reliability, and deployment efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01070v4">An Inquiry into Datacenter TCO for LLM Inference with FP8</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale, the high power consumption of AI accelerators in datacenters presents significant challenges, substantially increasing the total cost of ownership (TCO) for cloud service providers (CSPs) that provide LLM inference. In this work, we analyze the computational characteristics of LLM inference from a TCO perspective and present a generalizable framework to compare AI accelerators across diverse operational requirements. Using this model, we investigate key workload characteristics influencing TCO for AI accelerators from Intel (Gaudi 2 & 3) and NVIDIA (H100 & H200), especially thin GEMM utilization and FP8 quantization. In particular, as FP8 emerges as the baseline precision for next-generation LLMs, understanding how different architectures implement and benefit from low-precision computation is increasingly critical. Throughput on thin GEMMs has a greater impact on TCO than theoretical hardware peak throughput because the memory-bound decode phase is dominated by GEMV-like computations. We find that Gaudi HPUs achieve superior utilization on thin GEMMs compared to their counterparts, especially in FP8-quantized models. Our result underscores the importance of empirical, workload-level analysis in evaluating accelerator performance, rather than relying solely on theoretical hardware specifications. By studying the interaction between power consumption, quantization strategies, and hardware architecture, we provide insights to support informed deployment decisions and guide future accelerator designs aimed at improving the TCO of LLM inference workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17692v1">LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 51 pages,10 figures,8 tables. Work in progress
    </div>
    <details class="paper-abstract">
      Recent advances in the intrinsic reasoning capabilities of large language models (LLMs) have given rise to LLM-based agent systems that exhibit near-human performance on a variety of automated tasks. However, although these systems share similarities in terms of their use of LLMs, different reasoning frameworks of the agent system steer and organize the reasoning process in different ways. In this survey, we propose a systematic taxonomy that decomposes agentic reasoning frameworks and analyze how these frameworks dominate framework-level reasoning by comparing their applications across different scenarios. Specifically, we propose an unified formal language to further classify agentic reasoning systems into single-agent methods, tool-based methods, and multi-agent methods. After that, we provide a comprehensive review of their key application scenarios in scientific discovery, healthcare, software engineering, social simulation, and economics. We also analyze the characteristic features of each framework and summarize different evaluation strategies. Our survey aims to provide the research community with a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of different agentic reasoning frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17674v1">Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17644v1">Demographically-Inspired Query Variants Using an LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Published in the proceedings of ICTIR'25, Padua, Italy
    </div>
    <details class="paper-abstract">
      This study proposes a method to diversify queries in existing test collections to reflect some of the diversity of search engine users, aligning with an earlier vision of an 'ideal' test collection. A Large Language Model (LLM) is used to create query variants: alternative queries that have the same meaning as the original. These variants represent user profiles characterised by different properties, such as language and domain proficiency, which are known in the IR literature to influence query formulation. The LLM's ability to generate query variants that align with user profiles is empirically validated, and the variants' utility is further explored for IR system evaluation. Results demonstrate that the variants impact how systems are ranked and show that user profiles experience significantly different levels of system effectiveness. This method enables an alternative perspective on system evaluation where we can observe both the impact of user profiles on system rankings and how system performance varies across users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17627v1">Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enhance complex reasoning tasks by scaling the individual thinking process. However, prior work shows that overthinking can degrade overall performance. Motivated by observed patterns in thinking length and content length, we categorize reasoning into three stages: insufficient exploration stage, compensatory reasoning stage, and reasoning convergence stage. Typically, LLMs produce correct answers in the compensatory reasoning stage, whereas reasoning convergence often triggers overthinking, causing increased resource usage or even infinite loops. Therefore, mitigating overthinking hinges on detecting the end of the compensatory reasoning stage, defined as the Reasoning Completion Point (RCP). RCP typically appears at the end of the first complete reasoning cycle and can be identified by querying the LLM sentence by sentence or monitoring the probability of an end-of-thinking token (e.g., \texttt{</think>}), though these methods lack an efficient and precise balance. To improve this, we mine more sensitive and consistent RCP patterns and develop a lightweight thresholding strategy based on heuristic rules. Experimental evaluations on benchmarks (AIME24, AIME25, GPQA-D) demonstrate that the proposed method reduces token consumption while preserving or enhancing reasoning accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17573v1">Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted in EMNLP main proceedings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18547v1">How do Humans and LLMs Process Confusing Code?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Already today, humans and programming assistants based on large language models (LLMs) collaborate in everyday programming tasks. Clearly, a misalignment between how LLMs and programmers comprehend code can lead to misunderstandings, inefficiencies, low code quality, and bugs. A key question in this space is whether humans and LLMs are confused by the same kind of code. This would not only guide our choices of integrating LLMs in software engineering workflows, but also inform about possible improvements of LLMs. To this end, we conducted an empirical study comparing an LLM to human programmers comprehending clean and confusing code. We operationalized comprehension for the LLM by using LLM perplexity, and for human programmers using neurophysiological responses (in particular, EEG-based fixation-related potentials). We found that LLM perplexity spikes correlate both in terms of location and amplitude with human neurophysiological responses that indicate confusion. This result suggests that LLMs and humans are similarly confused about the code. Based on these findings, we devised a data-driven, LLM-based approach to identify regions of confusion in code that elicit confusion in human programmers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18533v1">A Database-Driven Framework for 3D Level Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Procedural Content Generation for 3D game levels faces challenges in balancing spatial coherence, navigational functionality, and adaptable gameplay progression across multi-floor environments. This paper introduces a novel framework for generating such levels, centered on the offline, LLM-assisted construction of reusable databases for architectural components (facilities and room templates) and gameplay mechanic elements. Our multi-phase pipeline assembles levels by: (1) selecting and arranging instances from the Room Database to form a multi-floor global structure with an inherent topological order; (2) optimizing the internal layout of facilities for each room based on predefined constraints from the Facility Database; and (3) integrating progression-based gameplay mechanics by placing components from a Mechanics Database according to their topological and spatial rules. A subsequent two-phase repair system ensures navigability. This approach combines modular, database-driven design with constraint-based optimization, allowing for systematic control over level structure and the adaptable pacing of gameplay elements. Initial experiments validate the framework's ability in generating diverse, navigable 3D environments and its capability to simulate distinct gameplay pacing strategies through simple parameterization. This research advances PCG by presenting a scalable, database-centric foundation for the automated generation of complex 3D levels with configurable gameplay progression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18467v1">The AI in the Mirror: LLM Self-Recognition in an Iterated Public Goods Game</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As AI agents become increasingly capable of tool use and long-horizon tasks, they have begun to be deployed in settings where multiple agents can interact. However, whereas prior work has mostly focused on human-AI interactions, there is an increasing need to understand AI-AI interactions. In this paper, we adapt the iterated public goods game, a classic behavioral economics game, to analyze the behavior of four reasoning and non-reasoning models across two conditions: models are either told they are playing against "another AI agent" or told their opponents are themselves. We find that, across different settings, telling LLMs that they are playing against themselves significantly changes their tendency to cooperate. While our study is conducted in a toy environment, our results may provide insights into multi-agent settings where agents "unconsciously" discriminating against each other could inexplicably increase or decrease cooperation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16571v2">LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      In this paper, we describe and benchmark a competitor-discovery component used within an agentic AI system for fast drug asset due diligence. A competitor-discovery AI agent, given an indication, retrieves all drugs comprising the competitive landscape of that indication and extracts canonical attributes for these drugs. The competitor definition is investor-specific, and data is paywalled/licensed, fragmented across registries, ontology-mismatched by indication, alias-heavy for drug names, multimodal, and rapidly changing. Although considered the best tool for this problem, the current LLM-based AI systems aren't capable of reliably retrieving all competing drug names, and there is no accepted public benchmark for this task. To address the lack of evaluation, we use LLM-based agents to transform five years of multi-modal, unstructured diligence memos from a private biotech VC fund into a structured evaluation corpus mapping indications to competitor drugs with normalized attributes. We also introduce a competitor validating LLM-as-a-judge agent that filters out false positives from the list of predicted competitors to maximize precision and suppress hallucinations. On this benchmark, our competitor-discovery agent achieves 83% recall, exceeding OpenAI Deep Research (65%) and Perplexity Labs (60%). The system is deployed in production with enterprise users; in a case study with a biotech VC investment fund, analyst turnaround time dropped from 2.5 days to $\sim$3 hours ($\sim$20x) for the competitive analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18444v1">How Reliable are LLMs for Reasoning on the Re-ranking task?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted at FQAS Conference 2024. DOI will be provided in 3 weeks after the conference has published the paper
    </div>
    <details class="paper-abstract">
      With the improving semantic understanding capability of Large Language Models (LLMs), they exhibit a greater awareness and alignment with human values, but this comes at the cost of transparency. Although promising results are achieved via experimental analysis, an in-depth understanding of the LLM's internal workings is unavoidable to comprehend the reasoning behind the re-ranking, which provides end users with an explanation that enables them to make an informed decision. Moreover, in newly developed systems with limited user engagement and insufficient ranking data, accurately re-ranking content remains a significant challenge. While various training methods affect the training of LLMs and generate inference, our analysis has found that some training methods exhibit better explainability than others, implying that an accurate semantic understanding has not been learned through all training methods; instead, abstract knowledge has been gained to optimize evaluation, which raises questions about the true reliability of LLMs. Therefore, in this work, we analyze how different training methods affect the semantic understanding of the re-ranking task in LLMs and investigate whether these models can generate more informed textual reasoning to overcome the challenges of transparency or LLMs and limited training data. To analyze the LLMs for re-ranking tasks, we utilize a relatively small ranking dataset from the environment and the Earth science domain to re-rank retrieved content. Furthermore, we also analyze the explainable information to see if the re-ranking can be reasoned using explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18439v1">A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Vulnerability databases, such as the National Vulnerability Database (NVD), offer detailed descriptions of Common Vulnerabilities and Exposures (CVEs), but often lack information on their real-world impact, such as the tactics, techniques, and procedures (TTPs) that adversaries may use to exploit the vulnerability. However, manually linking CVEs to their corresponding TTPs is a challenging and time-consuming task, and the high volume of new vulnerabilities published annually makes automated support desirable. This paper introduces TRIAGE, a two-pronged automated approach that uses Large Language Models (LLMs) to map CVEs to relevant techniques from the ATT&CK knowledge base. We first prompt an LLM with instructions based on MITRE's CVE Mapping Methodology to predict an initial list of techniques. This list is then combined with the results from a second LLM-based module that uses in-context learning to map a CVE to relevant techniques. This hybrid approach strategically combines rule-based reasoning with data-driven inference. Our evaluation reveals that in-context learning outperforms the individual mapping methods, and the hybrid approach improves recall of exploitation techniques. We also find that GPT-4o-mini performs better than Llama3.3-70B on this task. Overall, our results show that LLMs can be used to automatically predict the impact of cybersecurity vulnerabilities and TRIAGE makes the process of mapping CVEs to ATT&CK more efficient. Keywords: vulnerability impact, CVE, ATT&CK techniques, large language models, automated mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18420v1">LLM-Driven Intrinsic Motivation for Sparse Reward Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 11 pages, 5 figures, Accepted to the ENIAC 2025 conference
    </div>
    <details class="paper-abstract">
      This paper explores the combination of two intrinsic motivation strategies to improve the efficiency of reinforcement learning (RL) agents in environments with extreme sparse rewards, where traditional learning struggles due to infrequent positive feedback. We propose integrating Variational State as Intrinsic Reward (VSIMR), which uses Variational AutoEncoders (VAEs) to reward state novelty, with an intrinsic reward approach derived from Large Language Models (LLMs). The LLMs leverage their pre-trained knowledge to generate reward signals based on environment and goal descriptions, guiding the agent. We implemented this combined approach with an Actor-Critic (A2C) agent in the MiniGrid DoorKey environment, a benchmark for sparse rewards. Our empirical results show that this combined strategy significantly increases agent performance and sampling efficiency compared to using each strategy individually or a standard A2C agent, which failed to learn. Analysis of learning curves indicates that the combination effectively complements different aspects of the environment and task: VSIMR drives exploration of new states, while the LLM-derived rewards facilitate progressive exploitation towards goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18379v1">REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted to EMNLP 2025 (Main Conference). 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong capabilities in document re-ranking, a key component in modern Information Retrieval (IR) systems. However, existing LLM-based approaches face notable limitations, including ranking uncertainty, unstable top-k recovery, and high token cost due to token-intensive prompting. To effectively address these limitations, we propose REALM, an uncertainty-aware re-ranking framework that models LLM-derived relevance as Gaussian distributions and refines them through recursive Bayesian updates. By explicitly capturing uncertainty and minimizing redundant queries, REALM achieves better rankings more efficiently. Experimental results demonstrate that our REALM surpasses state-of-the-art re-rankers while significantly reducing token usage and latency, promoting it as the next-generation re-ranker for modern IR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18914v3">PRISM: Efficient Long-Range Reasoning With Short-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Published as a conference paper at EMNLP 2025. 28 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Long-range tasks demand reasoning over long inputs. However, existing solutions are limited, e.g., long-context models require large compute budgets, parameter-efficient fine-tuning (PEFT) needs training data, and retrieval-augmented generation (RAG) entails complex task-specific designs. Though in-context approaches overcome many of these issues, methods with short-context LLMs are inefficient, trading context for processing more tokens. We introduce PRISM, a highly token-efficient in-context method based on structured schemas that outperforms baselines on diverse tasks with 4x shorter contexts. This approach produces concise outputs and efficiently leverages key-value (KV) caches to reduce costs by up to 54%. PRISM scales down to tiny contexts without increasing costs or sacrificing quality, and generalizes to new tasks with minimal effort by generating schemas from task descriptions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08899v2">From Legal Texts to Defeasible Deontic Logic via LLMs: A Study in Automated Semantic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      We present a novel approach to the automated semantic analysis of legal texts using large language models (LLMs), targeting their transformation into formal representations in Defeasible Deontic Logic (DDL). We propose a structured pipeline that segments complex normative language into atomic snippets, extracts deontic rules, and evaluates them for syntactic and semantic coherence. Our methodology is evaluated across various LLM configurations, including prompt engineering strategies, fine-tuned models, and multi-stage pipelines, focusing on legal norms from the Australian Telecommunications Consumer Protections Code. Empirical results demonstrate promising alignment between machine-generated and expert-crafted formalizations, showing that LLMs - particularly when prompted effectively - can significantly contribute to scalable legal informatics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10142v3">Multi-Turn Puzzles: Evaluating Interactive Reasoning and Strategic Dialogue in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at solving problems with clear and complete statements, but often struggle with nuanced environments or interactive tasks which are common in most real-world scenarios. This highlights the critical need for developing LLMs that can effectively engage in logically consistent multi-turn dialogue, seek information and reason with incomplete data. To this end, we introduce a novel benchmark comprising a suite of multi-turn tasks each designed to test specific reasoning, interactive dialogue, and information-seeking abilities. These tasks have deterministic scoring mechanisms, thus eliminating the need for human intervention. Evaluating frontier models on our benchmark reveals significant headroom. Our analysis shows that most errors emerge from poor instruction following, reasoning failures, and poor planning. This benchmark provides valuable insights into the strengths and weaknesses of current LLMs in handling complex, interactive scenarios and offers a robust platform for future research aimed at improving these critical capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15659v2">VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of 125,777 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4%, respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation. Our code, data, and models are publicly available at https://github.com/Anjiang-Wei/VeriCoder
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00850v2">ICQuant: Index Coding enables Low-bit LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      The rapid deployment of Large Language Models (LLMs) highlights the need for efficient low-bit post-training quantization (PTQ), due to their high memory costs. A key challenge in weight quantization is the presence of outliers, which inflate quantization ranges and lead to large errors. While a number of outlier suppression techniques have been proposed, they either: fail to effectively shrink the quantization range, or incur (relatively) high bit overhead. In this paper, we present ICQuant, a novel framework that leverages outlier statistics to design an efficient index coding scheme for outlier-aware weight-only quantization. Compared to existing outlier suppression techniques requiring $\approx 1$ bit overhead to halve the quantization range, ICQuant requires only $\approx 0.3$ bits; a significant saving in extreme compression regimes (e.g., 2-3 bits per weight). ICQuant can be used on top of any existing quantizers to eliminate outliers, improving the quantization quality. Using just 2.3 bits per weight and simple scalar quantizers, ICQuant improves the zero-shot accuracy of the 2-bit Llama3-70B model by up to 130% and 150% relative to QTIP and QuIP#; and it achieves comparable performance to the best-known fine-tuned quantizer (PV-tuning) without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v5">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/JARVIS-Xs/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12529v2">LLM4MSR: An LLM-Enhanced Paradigm for Multi-Scenario Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 CIKM 2024 Full Research Paper
    </div>
    <details class="paper-abstract">
      As the demand for more personalized recommendation grows and a dramatic boom in commercial scenarios arises, the study on multi-scenario recommendation (MSR) has attracted much attention, which uses the data from all scenarios to simultaneously improve their recommendation performance. However, existing methods tend to integrate insufficient scenario knowledge and neglect learning personalized cross-scenario preferences, thus leading to sub-optimal performance. Meanwhile, though large language model (LLM) has shown great capability of reasoning and capturing semantic information, the high inference latency and high computation cost of tuning hinder its implementation in industrial recommender systems. To fill these gaps, we propose an LLM-enhanced paradigm LLM4MSR in this work. Specifically, we first leverage LLM to uncover multi-level knowledge from the designed scenario- and user-level prompt without fine-tuning the LLM, then adopt hierarchical meta networks to generate multi-level meta layers to explicitly improve the scenario-aware and personalized recommendation capability. Our experiments on KuaiSAR-small, KuaiSAR, and Amazon datasets validate significant advantages of LLM4MSR: (i) the effectiveness and compatibility with different multi-scenario backbone models, (ii) high efficiency and deployability on industrial recommender systems, and (iii) improved interpretability. The implemented code and data is available to ease reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17450v1">Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 To appear at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17435v1">An LLM-LVLM Driven Agent for Iterative and Fine-Grained Image Editing</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Despite the remarkable capabilities of text-to-image (T2I) generation models, real-world applications often demand fine-grained, iterative image editing that existing methods struggle to provide. Key challenges include granular instruction understanding, robust context preservation during modifications, and the lack of intelligent feedback mechanisms for iterative refinement. This paper introduces RefineEdit-Agent, a novel, training-free intelligent agent framework designed to address these limitations by enabling complex, iterative, and context-aware image editing. RefineEdit-Agent leverages the powerful planning capabilities of Large Language Models (LLMs) and the advanced visual understanding and evaluation prowess of Vision-Language Large Models (LVLMs) within a closed-loop system. Our framework comprises an LVLM-driven instruction parser and scene understanding module, a multi-level LLM-driven editing planner for goal decomposition, tool selection, and sequence generation, an iterative image editing module, and a crucial LVLM-driven feedback and evaluation loop. To rigorously evaluate RefineEdit-Agent, we propose LongBench-T2I-Edit, a new benchmark featuring 500 initial images with complex, multi-turn editing instructions across nine visual dimensions. Extensive experiments demonstrate that RefineEdit-Agent significantly outperforms state-of-the-art baselines, achieving an average score of 3.67 on LongBench-T2I-Edit, compared to 2.29 for Direct Re-Prompting, 2.91 for InstructPix2Pix, 3.16 for GLIGEN-based Edit, and 3.39 for ControlNet-XL. Ablation studies, human evaluations, and analyses of iterative refinement, backbone choices, tool usage, and robustness to instruction complexity further validate the efficacy of our agentic design in delivering superior edit fidelity and context preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17402v1">DS@GT at CheckThat! 2025: A Simple Retrieval-First, LLM-Backed Framework for Claim Normalization</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 CLEF 2025 Working Notes, Madrid, Spain
    </div>
    <details class="paper-abstract">
      Claim normalization is an integral part of any automatic fact-check verification system. It parses the typically noisy claim data, such as social media posts into normalized claims, which are then fed into downstream veracity classification tasks. The CheckThat! 2025 Task 2 focuses specifically on claim normalization and spans 20 languages under monolingual and zero-shot conditions. Our proposed solution consists of a lightweight \emph{retrieval-first, LLM-backed} pipeline, in which we either dynamically prompt a GPT-4o-mini with in-context examples, or retrieve the closest normalization from the train dataset directly. On the official test set, the system ranks near the top for most monolingual tracks, achieving first place in 7 out of of the 13 languages. In contrast, the system underperforms in the zero-shot setting, highlighting the limitation of the proposed solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17387v1">Graph-R1: Incentivizing the Zero-Shot Graph Learning Capability in LLMs via Explicit Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Generalizing to unseen graph tasks without task-pecific supervision remains challenging. Graph Neural Networks (GNNs) are limited by fixed label spaces, while Large Language Models (LLMs) lack structural inductive biases. Recent advances in Large Reasoning Models (LRMs) provide a zero-shot alternative via explicit, long chain-of-thought reasoning. Inspired by this, we propose a GNN-free approach that reformulates graph tasks--node classification, link prediction, and graph classification--as textual reasoning problems solved by LRMs. We introduce the first datasets with detailed reasoning traces for these tasks and develop Graph-R1, a reinforcement learning framework that leverages task-specific rethink templates to guide reasoning over linearized graphs. Experiments demonstrate that Graph-R1 outperforms state-of-the-art baselines in zero-shot settings, producing interpretable and effective predictions. Our work highlights the promise of explicit reasoning for graph learning and provides new resources for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17378v1">UI-Level Evaluation of ALLaM 34B: Measuring an Arabic-Centric LLM via HUMAIN Chat</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained primarily on English corpora often struggle to capture the linguistic and cultural nuances of Arabic. To address this gap, the Saudi Data and AI Authority (SDAIA) introduced the $ALLaM$ family of Arabic-focused models. The most capable of these available to the public, $ALLaM-34B$, was subsequently adopted by HUMAIN, who developed and deployed HUMAIN Chat, a closed conversational web service built on this model. This paper presents an expanded and refined UI-level evaluation of $ALLaM-34B$. Using a prompt pack spanning modern standard Arabic, five regional dialects, code-switching, factual knowledge, arithmetic and temporal reasoning, creative generation, and adversarial safety, we collected 115 outputs (23 prompts times 5 runs) and scored each with three frontier LLM judges (GPT-5, Gemini 2.5 Pro, Claude Sonnet-4). We compute category-level means with 95\% confidence intervals, analyze score distributions, and visualize dialect-wise metric heat maps. The updated analysis reveals consistently high performance on generation and code-switching tasks (both averaging 4.92/5), alongside strong results in MSA handling (4.74/5), solid reasoning ability (4.64/5), and improved dialect fidelity (4.21/5). Safety-related prompts show stable, reliable performance of (4.54/5). Taken together, these results position $ALLaM-34B$ as a robust and culturally grounded Arabic LLM, demonstrating both technical strength and practical readiness for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16814v4">Understanding Bias Reinforcement in LLM Agents Debate</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 32 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models $($LLMs$)$ solve complex problems using training-free methods like prompt engineering and in-context learning, yet ensuring reasoning correctness remains challenging. While self-correction methods such as self-consistency and self-refinement aim to improve reliability, they often reinforce biases due to the lack of effective feedback mechanisms. Multi-Agent Debate $($MAD$)$ has emerged as an alternative, but we identify two key limitations: bias reinforcement, where debate amplifies model biases instead of correcting them, and lack of perspective diversity, as all agents share the same model and reasoning patterns, limiting true debate effectiveness. To systematically evaluate these issues, we introduce $\textit{MetaNIM Arena}$, a benchmark designed to assess LLMs in adversarial strategic decision-making, where dynamic interactions influence optimal decisions. To overcome MAD's limitations, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$, a novel framework that $(1)$ refines LLM's strategic prior knowledge to improve reasoning quality and $(2)$ promotes diverse viewpoints within a single model by systematically modifying prompts, reducing bias. Empirical results show that $\textbf{DReaMAD}$ significantly improves decision accuracy, reasoning diversity, and bias mitigation across multiple strategic tasks, establishing it as a more effective approach for LLM-based decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08243v3">Jinx: Unlimited LLMs for Probing Alignment Failures</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 https://huggingface.co/Jinx-org
    </div>
    <details class="paper-abstract">
      Unlimited, or so-called helpful-only language models are trained without safety alignment constraints and never refuse user queries. They are widely used by leading AI companies as internal tools for red teaming and alignment evaluation. For example, if a safety-aligned model produces harmful outputs similar to an unlimited model, this indicates alignment failures that require further attention. Despite their essential role in assessing alignment, such models are not available to the research community. We introduce Jinx, a helpful-only variant of popular open-weight LLMs. Jinx responds to all queries without refusals or safety filtering, while preserving the base model's capabilities in reasoning and instruction following. It provides researchers with an accessible tool for probing alignment failures, evaluating safety boundaries, and systematically studying failure modes in language model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17361v1">Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA). We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17322v1">Chinese Court Simulation with LLM-Based Agent System</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Mock trial has long served as an important platform for legal professional training and education. It not only helps students learn about realistic trial procedures, but also provides practical value for case analysis and judgment prediction. Traditional mock trials are difficult to access by the public because they rely on professional tutors and human participants. Fortunately, the rise of large language models (LLMs) provides new opportunities for creating more accessible and scalable court simulations. While promising, existing research mainly focuses on agent construction while ignoring the systematic design and evaluation of court simulations, which are actually more important for the credibility and usage of court simulation in practice. To this end, we present the first court simulation framework -- SimCourt -- based on the real-world procedure structure of Chinese courts. Our framework replicates all 5 core stages of a Chinese trial and incorporates 5 courtroom roles, faithfully following the procedural definitions in China. To simulate trial participants with different roles, we propose and craft legal agents equipped with memory, planning, and reflection abilities. Experiment on legal judgment prediction show that our framework can generate simulated trials that better guide the system to predict the imprisonment, probation, and fine of each case. Further annotations by human experts show that agents' responses under our simulation framework even outperformed judges and lawyers from the real trials in many scenarios. These further demonstrate the potential of LLM-based court simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17320v1">AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Understanding the internal representations of large language models (LLMs) remains a central challenge for interpretability research. Sparse autoencoders (SAEs) offer a promising solution by decomposing activations into interpretable features, but existing approaches rely on fixed sparsity constraints that fail to account for input complexity. We propose Adaptive Top K Sparse Autoencoders (AdaptiveK), a novel framework that dynamically adjusts sparsity levels based on the semantic complexity of each input. Leveraging linear probes, we demonstrate that context complexity is linearly encoded in LLM representations, and we use this signal to guide feature allocation during training. Experiments across three language models (Pythia-70M, Pythia-160M, and Gemma-2-2B) demonstrate that this complexity-driven adaptation significantly outperforms fixed-sparsity approaches on reconstruction fidelity, explained variance, and cosine similarity metrics while eliminating the computational burden of extensive hyperparameter tuning.
    </details>
</div>
