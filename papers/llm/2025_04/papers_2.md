# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19259v2">Neuro-LIFT: A Neuromorphic, LLM-based Interactive Framework for Autonomous Drone FlighT at the Edge</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 Accepted for publication at the International Joint Conference on Neural Networks (IJCNN) 2025
    </div>
    <details class="paper-abstract">
      The integration of human-intuitive interactions into autonomous systems has been limited. Traditional Natural Language Processing (NLP) systems struggle with context and intent understanding, severely restricting human-robot interaction. Recent advancements in Large Language Models (LLMs) have transformed this dynamic, allowing for intuitive and high-level communication through speech and text, and bridging the gap between human commands and robotic actions. Additionally, autonomous navigation has emerged as a central focus in robotics research, with artificial intelligence (AI) increasingly being leveraged to enhance these systems. However, existing AI-based navigation algorithms face significant challenges in latency-critical tasks where rapid decision-making is critical. Traditional frame-based vision systems, while effective for high-level decision-making, suffer from high energy consumption and latency, limiting their applicability in real-time scenarios. Neuromorphic vision systems, combining event-based cameras and spiking neural networks (SNNs), offer a promising alternative by enabling energy-efficient, low-latency navigation. Despite their potential, real-world implementations of these systems, particularly on physical platforms such as drones, remain scarce. In this work, we present Neuro-LIFT, a real-time neuromorphic navigation framework implemented on a Parrot Bebop2 quadrotor. Leveraging an LLM for natural language processing, Neuro-LIFT translates human speech into high-level planning commands which are then autonomously executed using event-based neuromorphic vision and physics-driven planning. Our framework demonstrates its capabilities in navigating in a dynamic environment, avoiding obstacles, and adapting to human instructions in real-time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18985v1">Tracking the Moving Target: A Framework for Continuous Evaluation of LLM Test Generation in Industry</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 Accepted for publication at the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential in automating software testing tasks, including test generation. However, their rapid evolution poses a critical challenge for companies implementing DevSecOps - evaluations of their effectiveness quickly become outdated, making it difficult to assess their reliability for production use. While academic research has extensively studied LLM-based test generation, evaluations typically provide point-in-time analyses using academic benchmarks. Such evaluations do not address the practical needs of companies who must continuously assess tool reliability and integration with existing development practices. This work presents a measurement framework for the continuous evaluation of commercial LLM test generators in industrial environments. We demonstrate its effectiveness through a longitudinal study at LKS Next. The framework integrates with industry-standard tools like SonarQube and provides metrics that evaluate both technical adequacy (e.g., test coverage) and practical considerations (e.g., maintainability or expert assessment). Our methodology incorporates strategies for test case selection, prompt engineering, and measurement infrastructure, addressing challenges such as data leakage and reproducibility. Results highlight both the rapid evolution of LLM capabilities and critical factors for successful industrial adoption, offering practical guidance for companies seeking to integrate these technologies into their development pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18919v1">Clinical knowledge in LLMs does not translate to human interactions</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 52 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Global healthcare providers are exploring use of large language models (LLMs) to provide medical advice to the public. LLMs now achieve nearly perfect scores on medical licensing exams, but this does not necessarily translate to accurate performance in real-world settings. We tested if LLMs can assist members of the public in identifying underlying conditions and choosing a course of action (disposition) in ten medical scenarios in a controlled study with 1,298 participants. Participants were randomly assigned to receive assistance from an LLM (GPT-4o, Llama 3, Command R+) or a source of their choice (control). Tested alone, LLMs complete the scenarios accurately, correctly identifying conditions in 94.9% of cases and disposition in 56.3% on average. However, participants using the same LLMs identified relevant conditions in less than 34.5% of cases and disposition in less than 44.2%, both no better than the control group. We identify user interactions as a challenge to the deployment of LLMs for medical advice. Standard benchmarks for medical knowledge and simulated patient interactions do not predict the failures we find with human participants. Moving forward, we recommend systematic human user testing to evaluate interactive capabilities prior to public deployments in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17497v2">Combining GCN Structural Learning with LLM Chemical Knowledge for Enhanced Virtual Screening</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Virtual screening plays a critical role in modern drug discovery by enabling the identification of promising candidate molecules for experimental validation. Traditional machine learning methods such, as Support Vector Machines (SVM) and XGBoost, rely on predefined molecular representations, often leading to information loss and potential bias. In contrast, deep learning approaches-particularly Graph Convolutional Networks (GCNs)-offer a more expressive and unbiased alternative by operating directly on molecular graphs. Meanwhile, Large Language Models (LLMs) have recently demonstrated state-of-the-art performance in drug design, thanks to their capacity to capture complex chemical patterns from large-scale data via attention mechanisms. In this paper, we propose a hybrid architecture that integrates GCNs with LLM-derived embeddings to combine localized structural learning with global chemical knowledge. The LLM embeddings can be precomputed and stored in a molecular feature library, removing the need to rerun the LLM during training or inference and thus maintaining computational efficiency. We found that concatenating the LLM embeddings after each GCN layer-rather than only at the final layer-significantly improves performance, enabling deeper integration of global context throughout the network. The resulting model achieves superior results, with an F1-score of (88.8\%), outperforming standalone GCN (87.9%), XGBoost (85.5%), and SVM (85.4%) baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18884v1">A Simple Ensemble Strategy for LLM Inference: Towards More Stable Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 This manuscript has been accepted for the 30th International Conference on Natural Language & Information Systems (NLDB 2025). The final version will appear in the Springer LNCS proceedings. arXiv admin note: text overlap with arXiv:2407.13069
    </div>
    <details class="paper-abstract">
      With the advance of large language models (LLMs), LLMs have been utilized for the various tasks. However, the issues of variability and reproducibility of results from each trial of LLMs have been largely overlooked in existing literature while actual human annotation uses majority voting to resolve disagreements among annotators. Therefore, this study introduces the straightforward ensemble strategy to a sentiment analysis using LLMs. As the results, we demonstrate that the ensemble of multiple inference using medium-sized LLMs produces more robust and accurate results than using a large model with a single attempt with reducing RMSE by 18.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v5">Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16563v2">Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Intelligent agent systems based on Large Language Models (LLMs) have shown great potential in real-world applications. However, existing agent frameworks still face critical limitations in task planning and execution, restricting their effectiveness and generalizability. Specifically, current planning methods often lack clear global goals, leading agents to get stuck in local branches, or produce non-executable plans. Meanwhile, existing execution mechanisms struggle to balance complexity and stability, and their limited action space restricts their ability to handle diverse real-world tasks. To address these limitations, we propose GoalAct, a novel agent framework that introduces a continuously updated global planning mechanism and integrates a hierarchical execution strategy. GoalAct decomposes task execution into high-level skills, including searching, coding, writing and more, thereby reducing planning complexity while enhancing the agents' adaptability across diverse task scenarios. We evaluate GoalAct on LegalAgentBench, a benchmark with multiple types of legal tasks that require the use of multiple types of tools. Experimental results demonstrate that GoalAct achieves state-of-the-art (SOTA) performance, with an average improvement of 12.22% in success rate. These findings highlight GoalAct's potential to drive the development of more advanced intelligent agent systems, making them more effective across complex real-world applications. Our code can be found at https://github.com/cjj826/GoalAct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18838v1">Toward Generalizable Evaluation in the LLM Era: A Survey Beyond Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are advancing at an amazing speed and have become indispensable across academia, industry, and daily applications. To keep pace with the status quo, this survey probes the core challenges that the rise of LLMs poses for evaluation. We identify and analyze two pivotal transitions: (i) from task-specific to capability-based evaluation, which reorganizes benchmarks around core competencies such as knowledge, reasoning, instruction following, multi-modal understanding, and safety; and (ii) from manual to automated evaluation, encompassing dynamic dataset curation and "LLM-as-a-judge" scoring. Yet, even with these transitions, a crucial obstacle persists: the evaluation generalization issue. Bounded test sets cannot scale alongside models whose abilities grow seemingly without limit. We will dissect this issue, along with the core challenges of the above two transitions, from the perspectives of methods, datasets, evaluators, and metrics. Due to the fast evolving of this field, we will maintain a living GitHub repository (links are in each section) to crowd-source updates and corrections, and warmly invite contributors and collaborators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18804v1">Can We Enhance Bug Report Quality Using LLMs?: An Empirical Study of LLM-Based Bug Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Bug reports contain the information developers need to triage and fix software bugs. However, unclear, incomplete, or ambiguous information may lead to delays and excessive manual effort spent on bug triage and resolution. In this paper, we explore whether Instruction fine-tuned Large Language Models (LLMs) can automatically transform casual, unstructured bug reports into high-quality, structured bug reports adhering to a standard template. We evaluate three open-source instruction-tuned LLMs (\emph{Qwen 2.5, Mistral, and Llama 3.2}) against ChatGPT-4o, measuring performance on established metrics such as CTQRS, ROUGE, METEOR, and SBERT. Our experiments show that fine-tuned Qwen 2.5 achieves a CTQRS score of \textbf{77%}, outperforming both fine-tuned Mistral (\textbf{71%}), Llama 3.2 (\textbf{63%}) and ChatGPT in 3-shot learning (\textbf{75%}). Further analysis reveals that Llama 3.2 shows higher accuracy of detecting missing fields particularly Expected Behavior and Actual Behavior, while Qwen 2.5 demonstrates superior performance in capturing Steps-to-Reproduce, with an F1 score of 76%. Additional testing of the models on other popular projects (e.g., Eclipse, GCC) demonstrates that our approach generalizes well, achieving up to \textbf{70%} CTQRS in unseen projects' bug reports. These findings highlight the potential of instruction fine-tuning in automating structured bug report generation, reducing manual effort for developers and streamlining the software maintenance process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13107v3">MatterChat: A Multi-Modal LLM for Material Science</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Understanding and predicting the properties of inorganic materials is crucial for accelerating advancements in materials science and driving applications in energy, electronics, and beyond. Integrating material structure data with language-based information through multi-modal large language models (LLMs) offers great potential to support these efforts by enhancing human-AI interaction. However, a key challenge lies in integrating atomic structures at full resolution into LLMs. In this work, we introduce MatterChat, a versatile structure-aware multi-modal LLM that unifies material structural data and textual inputs into a single cohesive model. MatterChat employs a bridging module to effectively align a pretrained machine learning interatomic potential with a pretrained LLM, reducing training costs and enhancing flexibility. Our results demonstrate that MatterChat significantly improves performance in material property prediction and human-AI interaction, surpassing general-purpose LLMs such as GPT-4. We also demonstrate its usefulness in applications such as more advanced scientific reasoning and step-by-step material synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13961v2">From Single to Multi: How LLMs Hallucinate in Multi-Document Summarization</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 NAACL 2025 - Findings
    </div>
    <details class="paper-abstract">
      Although many studies have investigated and reduced hallucinations in large language models (LLMs) for single-document tasks, research on hallucination in multi-document summarization (MDS) tasks remains largely unexplored. Specifically, it is unclear how the challenges arising from handling multiple documents (e.g., repetition and diversity of information) affect models outputs. In this work, we investigate how hallucinations manifest in LLMs when summarizing topic-specific information from multiple documents. Since no benchmarks exist for investigating hallucinations in MDS, we use existing news and conversation datasets, annotated with topic-specific insights, to create two novel multi-document benchmarks. When evaluating 5 LLMs on our benchmarks, we observe that on average, up to 75% of the content in LLM-generated summary is hallucinated, with hallucinations more likely to occur towards the end of the summaries. Moreover, when summarizing non-existent topic-related information, gpt-3.5-turbo and GPT-4o still generate summaries about 79.35% and 44% of the time, raising concerns about their tendency to fabricate content. To understand the characteristics of these hallucinations, we manually evaluate 700+ insights and find that most errors stem from either failing to follow instructions or producing overly generic insights. Motivated by these observations, we investigate the efficacy of simple post-hoc baselines in mitigating hallucinations but find them only moderately effective. Our results underscore the need for more effective approaches to systematically mitigate hallucinations in MDS. We release our dataset and code at github.com/megagonlabs/Hallucination_MDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11745v2">BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 HPCA 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across various machine learning tasks. Yet the substantial memory footprint of LLMs significantly hinders their deployment. In this paper, we improve the accessibility of LLMs through BitMoD, an algorithm-hardware co-design solution that enables efficient LLM acceleration at low weight precision. On the algorithm side, BitMoD introduces fine-grained data type adaptation that uses a different numerical data type to quantize a group of (e.g., 128) weights. Through the careful design of these new data types, BitMoD is able to quantize LLM weights to very low precision (e.g., 4 bits and 3 bits) while maintaining high accuracy. On the hardware side, BitMoD employs a bit-serial processing element to easily support multiple numerical precisions and data types; our hardware design includes two key innovations: First, it employs a unified representation to process different weight data types, thus reducing the hardware cost. Second, it adopts a bit-serial dequantization unit to rescale the per-group partial sum with minimal hardware overhead. Our evaluation on six representative LLMs demonstrates that BitMoD significantly outperforms state-of-the-art LLM quantization and acceleration methods. For discriminative tasks, BitMoD can quantize LLM weights to 4-bit with $<\!0.5\%$ accuracy loss on average. For generative tasks, BitMoD is able to quantize LLM weights to 3-bit while achieving better perplexity than prior LLM quantization scheme. Combining the superior model performance with an efficient accelerator design, BitMoD achieves an average of $1.69\times$ and $1.48\times$ speedups compared to prior LLM accelerators ANT and OliVe, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18765v1">A Vision for Auto Research with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      This paper introduces Agent-Based Auto Research, a structured multi-agent framework designed to automate, coordinate, and optimize the full lifecycle of scientific research. Leveraging the capabilities of large language models (LLMs) and modular agent collaboration, the system spans all major research phases, including literature review, ideation, methodology planning, experimentation, paper writing, peer review response, and dissemination. By addressing issues such as fragmented workflows, uneven methodological expertise, and cognitive overload, the framework offers a systematic and scalable approach to scientific inquiry. Preliminary explorations demonstrate the feasibility and potential of Auto Research as a promising paradigm for self-improving, AI-driven research processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18762v1">SynLexLM: Scaling Legal LLMs with Synthetic Data and Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 9 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful but often require extensive fine-tuning and large datasets for specialized domains like law. General-purpose pre-training may not capture legal nuances, and acquiring sufficient legal data is challenging. We introduce SynLexLM, a novel approach to efficiently pre-train a legal LLM. Our method employs curriculum learning, progressing from simple to complex legal texts and queries, combined with synthetic data augmentation using models like Gemini Pro to address data scarcity. We aim to achieve improved performance on legal benchmarks (BigLaw-Bench, EUR-Lex-Sum) compared to traditional models and fine-tuned versions. Preliminary work involves generating synthetic QA pairs reflecting legal reasoning. This work aims to enhance legal document analysis and research tools, potentially democratizing access to advanced legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16511v2">QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Quality and diversity are two critical metrics for the training data of large language models (LLMs), positively impacting performance. Existing studies often optimize these metrics separately, typically by first applying quality filtering and then adjusting data proportions. However, these approaches overlook the inherent trade-off between quality and diversity, necessitating their joint consideration. Given a fixed training quota, it is essential to evaluate both the quality of each data point and its complementary effect on the overall dataset. In this paper, we introduce a unified data selection framework called QuaDMix, which automatically optimizes the data distribution for LLM pretraining while balancing both quality and diversity. Specifically, we first propose multiple criteria to measure data quality and employ domain classification to distinguish data points, thereby measuring overall diversity. QuaDMix then employs a unified parameterized data sampling function that determines the sampling probability of each data point based on these quality and diversity related labels. To accelerate the search for the optimal parameters involved in the QuaDMix framework, we conduct simulated experiments on smaller models and use LightGBM for parameters searching, inspired by the RegMix method. Our experiments across diverse models and datasets demonstrate that QuaDMix achieves an average performance improvement of 7.2% across multiple benchmarks. These results outperform the independent strategies for quality and diversity, highlighting the necessity and ability to balance data quality and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18496v1">Facets, Taxonomies, and Syntheses: Navigating Structured Representations in LLM-Assisted Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Comprehensive literature review requires synthesizing vast amounts of research -- a labor intensive and cognitively demanding process. Most prior work focuses either on helping researchers deeply understand a few papers (e.g., for triaging or reading), or retrieving from and visualizing a vast corpus. Deep analysis and synthesis of large paper collections (e.g., to produce a survey paper) is largely conducted manually with little support. We present DimInd, an interactive system that scaffolds literature review across large paper collections through LLM-generated structured representations. DimInd scaffolds literature understanding with multiple levels of compression, from papers, to faceted literature comparison tables with information extracted from individual papers, to taxonomies of concepts, to narrative syntheses. Users are guided through these successive information transformations while maintaining provenance to source text. In an evaluation with 23 researchers, DimInd supported participants in extracting information and conceptually organizing papers with less effort compared to a ChatGPT-assisted baseline workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06018v2">A Picture is Worth A Thousand Numbers: Enabling LLMs Reason about Time Series via Visualization</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), with demonstrated reasoning abilities across multiple domains, are largely underexplored for time-series reasoning (TsR), which is ubiquitous in the real world. In this work, we propose TimerBed, the first comprehensive testbed for evaluating LLMs' TsR performance. Specifically, TimerBed includes stratified reasoning patterns with real-world tasks, comprehensive combinations of LLMs and reasoning strategies, and various supervised models as comparison anchors. We perform extensive experiments with TimerBed, test multiple current beliefs, and verify the initial failures of LLMs in TsR, evidenced by the ineffectiveness of zero shot (ZST) and performance degradation of few shot in-context learning (ICL). Further, we identify one possible root cause: the numerical modeling of data. To address this, we propose a prompt-based solution VL-Time, using visualization-modeled data and language-guided reasoning. Experimental results demonstrate that Vl-Time enables multimodal LLMs to be non-trivial ZST and powerful ICL reasoners for time series, achieving about 140% average performance improvement and 99% average token costs reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09537v2">Efficient Budget Allocation for Large-Scale LLM-Enabled Virtual Screening</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Screening tasks that aim to identify a small subset of top alternatives from a large pool are common in business decision-making processes. These tasks often require substantial human effort to evaluate each alternative's performance, making them time-consuming and costly. Motivated by recent advances in large language models (LLMs), particularly their ability to generate outputs that align well with human evaluations, we consider an LLM-as-human-evaluator approach for conducting screening virtually, thereby reducing the cost burden. To achieve scalability and cost-effectiveness in virtual screening, we identify that the stochastic nature of LLM outputs and their cost structure necessitate efficient budget allocation across all alternatives. To address this, we propose using a top-$m$ greedy evaluation mechanism, a simple yet effective approach that keeps evaluating the current top-$m$ alternatives, and design the explore-first top-$m$ greedy (EFG-$m$) algorithm. We prove that EFG-$m$ is both sample-optimal and consistent in large-scale virtual screening. Surprisingly, we also uncover a bonus ranking effect, where the algorithm naturally induces an indifference-based ranking within the selected subset. To further enhance practicality, we design a suite of algorithm variants to improve screening performance and computational efficiency. Numerical experiments validate our results and demonstrate the effectiveness of our algorithms. Lastly, we conduct a case study on LLM-based virtual screening. The study shows that while LLMs alone may not provide meaningful screening and ranking results when directly queried, integrating them with our sample-optimal algorithms unlocks their potential for cost-effective, large-scale virtual screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18423v1">LLMpatronous: Harnessing the Power of LLMs For Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Despite the transformative impact of Artificial Intelligence (AI) across various sectors, cyber security continues to rely on traditional static and dynamic analysis tools, hampered by high false positive rates and superficial code comprehension. While generative AI offers promising automation capabilities for software development, leveraging Large Language Models (LLMs) for vulnerability detection presents unique challenges. This paper explores the potential and limitations of LLMs in identifying vulnerabilities, acknowledging inherent weaknesses such as hallucinations, limited context length, and knowledge cut-offs. Previous attempts employing machine learning models for vulnerability detection have proven ineffective due to limited real-world applicability, feature engineering challenges, lack of contextual understanding, and the complexities of training models to keep pace with the evolving threat landscape. Therefore, we propose a robust AI-driven approach focused on mitigating these limitations and ensuring the quality and reliability of LLM based vulnerability detection. Through innovative methodologies combining Retrieval-Augmented Generation (RAG) and Mixtureof-Agents (MoA), this research seeks to leverage the strengths of LLMs while addressing their weaknesses, ultimately paving the way for dependable and efficient AI-powered solutions in securing the ever-evolving software landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18415v1">BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Efficient deployment of 1-bit Large Language Models (LLMs) is hindered by activation outliers, which complicate quantization to low bit-widths. We introduce BitNet v2, a novel framework enabling native 4-bit activation quantization for 1-bit LLMs. To tackle outliers in attention and feed-forward network activations, we propose H-BitLinear, a module applying an online Hadamard transformation prior to activation quantization. This transformation smooths sharp activation distributions into more Gaussian-like forms, suitable for low-bit representation. Experiments show BitNet v2 trained from scratch with 8-bit activations matches BitNet b1.58 performance. Crucially, BitNet v2 achieves minimal performance degradation when trained with native 4-bit activations, significantly reducing memory footprint and computational cost for batched inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18412v1">Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Should a large language model (LLM) be used as a therapist? In this paper, we investigate the use of LLMs to *replace* mental health providers, a use case promoted in the tech startup and research space. We conduct a mapping review of therapy guides used by major medical institutions to identify crucial aspects of therapeutic relationships, such as the importance of a therapeutic alliance between therapist and client. We then assess the ability of LLMs to reproduce and adhere to these aspects of therapeutic relationships by conducting several experiments investigating the responses of current LLMs, such as `gpt-4o`. Contrary to best practices in the medical community, LLMs 1) express stigma toward those with mental health conditions and 2) respond inappropriately to certain common (and critical) conditions in naturalistic therapy settings -- e.g., LLMs encourage clients' delusional thinking, likely due to their sycophancy. This occurs even with larger and newer LLMs, indicating that current safety practices may not address these gaps. Furthermore, we note foundational and practical barriers to the adoption of LLMs as therapists, such as that a therapeutic alliance requires human characteristics (e.g., identity and stakes). For these reasons, we conclude that LLMs should not replace therapists, and we discuss alternative roles for LLMs in clinical therapy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18410v1">Can Code Outlove Blood? A LLM-based VR Experience to Prompt Reflection on Parental Verbal Abuse</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 8 pages, 5 figures, accetped by 30th International Symposium on Electronic Art (ISEA 2025)
    </div>
    <details class="paper-abstract">
      Parental verbal abuse leaves lasting emotional impacts, yet current therapeutic approaches often lack immersive self-reflection opportunities. To address this, we developed a VR experience powered by LLMs to foster reflection on parental verbal abuse. Participants with relevant experiences engage in a dual-phase VR experience: first assuming the role of a verbally abusive parent, interacting with an LLM portraying a child, then observing the LLM reframing abusive dialogue into warm, supportive expressions as a nurturing parent. A qualitative study with 12 participants showed that the experience encourages reflection on their past experiences and fosters supportive emotions. However, these effects vary with participants' personal histories, emphasizing the need for greater personalization in AI-driven emotional support. This study explores the use of LLMs in immersive environment to promote emotional reflection, offering insights into the design of AI-driven emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17565v2">DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have recently achieved remarkable performance on various complex reasoning benchmarks, the academic community still lacks an in-depth understanding of base model training processes and data quality. To address this, we construct a large-scale, difficulty-graded reasoning dataset containing approximately 3.34 million unique queries of varying difficulty levels and about 40 million distilled responses generated by multiple models over several passes. Leveraging pass rate and Coefficient of Variation (CV), we precisely select the most valuable training data to enhance reasoning capability. Notably, we observe a training pattern shift, indicating that reasoning-focused training based on base models requires higher learning rates for effective training. Using this carefully selected data, we significantly improve the reasoning capabilities of the base model, achieving a pass rate of 79.2\% on the AIME2024 mathematical reasoning benchmark. This result surpasses most current distilled models and closely approaches state-of-the-art performance. We provide detailed descriptions of our data processing, difficulty assessment, and training methodology, and have publicly released all datasets and methods to promote rapid progress in open-source long-reasoning LLMs. The dataset is available at: https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18333v1">Adversarial Attacks on LLM-as-a-Judge Systems: Insights from Prompt Injections</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      LLM as judge systems used to assess text quality code correctness and argument strength are vulnerable to prompt injection attacks. We introduce a framework that separates content author attacks from system prompt attacks and evaluate five models Gemma 3.27B Gemma 3.4B Llama 3.2 3B GPT 4 and Claude 3 Opus on four tasks with various defenses using fifty prompts per condition. Attacks achieved up to seventy three point eight percent success smaller models proved more vulnerable and transferability ranged from fifty point five to sixty two point six percent. Our results contrast with Universal Prompt Injection and AdvPrompter We recommend multi model committees and comparative scoring and release all code and datasets
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v5">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL) using process rewards and iterative self-play, without supervised fine-tuning (SFT) as a preliminary step. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications. Code and data are available at https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/EPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19662v2">HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Quantization is critical for efficiently deploying large language models (LLMs). Yet conventional methods remain hardware-agnostic, limited to bit-width constraints, and do not account for intrinsic circuit characteristics such as the timing behaviors and energy profiles of Multiply-Accumulate (MAC) units. This disconnect from circuit-level behavior limits the ability to exploit available timing margins and energy-saving opportunities, reducing the overall efficiency of deployment on modern accelerators. To address these limitations, we propose HALO, a versatile framework for Hardware-Aware Post-Training Quantization (PTQ). Unlike traditional methods, HALO explicitly incorporates detailed hardware characteristics, including critical-path timing and power consumption, into its quantization approach. HALO strategically selects weights with low critical-path-delays enabling higher operational frequencies and dynamic frequency scaling without disrupting the architecture's dataflow. Remarkably, HALO achieves these improvements with only a few dynamic voltage and frequency scaling (DVFS) adjustments, ensuring simplicity and practicality in deployment. Additionally, by reducing switching activity within the MAC units, HALO effectively lowers energy consumption. Evaluations on accelerators such as Tensor Processing Units (TPUs) and Graphics Processing Units (GPUs) demonstrate that HALO significantly enhances inference efficiency, achieving average performance improvements of 270% and energy savings of 51% over baseline quantization methods, all with minimal impact on accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18154v1">EcoServe: Enabling Cost-effective LLM Serving with Proactive Intra- and Inter-Instance Orchestration</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Existing LLM serving strategies can be categorized based on whether prefill and decode phases are disaggregated: non-disaggregated (NoDG) or fully disaggregated (FuDG). However, the NoDG strategy leads to strong prefill-decode interference and the FuDG strategy highly relies on high-performance interconnects, making them less cost-effective. We introduce EcoServe, a system that enables cost-effective LLM serving on clusters with commodity interconnects. EcoServe is built on the partially disaggregated (PaDG) strategy, applying temporal disaggregation and rolling activation for proactive intra- and inter-instance scheduling. It first disaggregates the prefill and decode phases along the time dimension within a single instance to mitigate inter-phase interference and enhance throughput. Next, it coordinates multiple instances and cyclically activates them to ensure the continuous availability of prefill processing, thereby improving latency. Thus, EcoServe's basic serving unit is the macro instance, within which multiple instances collaborate. It further integrates an adaptive scheduling algorithm to route requests in a macro instance and a mitosis scaling approach to enable fine-grained capacity scaling. Beyond delivering high goodput, EcoServe excels in load balancing, hardware cost, parallelism compatibility, and even engineering simplicity compared to existing solutions. When serving 30B- and 70B-scale models on a production-level cluster with 32 NVIDIA L20 GPUs using commodity Ethernet, EcoServe averagely improves goodput by 82.49%, 86.17%, 122.76%, and 126.96% over four representative NoDG and FuDG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18147v1">NoEsis: Differentially Private Knowledge Transfer in Modular LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 ICLR 2025 MCDC workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) are typically trained on vast amounts of data from various sources. Even when designed modularly (e.g., Mixture-of-Experts), LLMs can leak privacy on their sources. Conversely, training such models in isolation arguably prohibits generalization. To this end, we propose a framework, NoEsis, which builds upon the desired properties of modularity, privacy, and knowledge transfer. NoEsis integrates differential privacy with a hybrid two-staged parameter-efficient fine-tuning that combines domain-specific low-rank adapters, acting as experts, with common prompt tokens, acting as a knowledge-sharing backbone. Results from our evaluation on CodeXGLUE showcase that NoEsis can achieve provable privacy guarantees with tangible knowledge transfer across domains, and empirically show protection against Membership Inference Attacks. Finally, on code completion tasks, NoEsis bridges at least 77% of the accuracy gap between the non-shared and the non-private baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14657v2">A Case Study Exploring the Current Landscape of Synthetic Medical Record Generation with Commercial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 Accepted at the Conference of Health, Inference, Learning (CHIL 2025) in Berkeley, CA. To appear in PMLR later in 2025
    </div>
    <details class="paper-abstract">
      Synthetic Electronic Health Records (EHRs) offer a valuable opportunity to create privacy preserving and harmonized structured data, supporting numerous applications in healthcare. Key benefits of synthetic data include precise control over the data schema, improved fairness and representation of patient populations, and the ability to share datasets without concerns about compromising real individuals privacy. Consequently, the AI community has increasingly turned to Large Language Models (LLMs) to generate synthetic data across various domains. However, a significant challenge in healthcare is ensuring that synthetic health records reliably generalize across different hospitals, a long standing issue in the field. In this work, we evaluate the current state of commercial LLMs for generating synthetic data and investigate multiple aspects of the generation process to identify areas where these models excel and where they fall short. Our main finding from this work is that while LLMs can reliably generate synthetic health records for smaller subsets of features, they struggle to preserve realistic distributions and correlations as the dimensionality of the data increases, ultimately limiting their ability to generalize across diverse hospital settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18106v1">Comparative Study on the Discourse Meaning of Chinese and English Media in the Paris Olympics Based on LDA Topic Modeling Technology and LLM Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      This study analyzes Chinese and English media reports on the Paris Olympics using topic modeling, Large Language Model (LLM) prompt engineering, and corpus phraseology methods to explore similarities and differences in discourse construction and attitudinal meanings. Common topics include the opening ceremony, athlete performance, and sponsorship brands. Chinese media focus on specific sports, sports spirit, doping controversies, and new technologies, while English media focus on female athletes, medal wins, and eligibility controversies. Chinese reports show more frequent prepositional co-occurrences and positive semantic prosody in describing the opening ceremony and sports spirit. English reports exhibit positive semantic prosody when covering female athletes but negative prosody in predicting opening ceremony reactions and discussing women's boxing controversies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08813v2">Your Weak LLM is Secretly a Strong Teacher for Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      The burgeoning capabilities of large language models (LLMs) have underscored the need for alignment to ensure these models act in accordance with human values and intentions. Existing alignment frameworks present constraints either in the form of expensive human effort or high computational costs. This paper explores a promising middle ground, where we employ a weak LLM that is significantly less resource-intensive than top-tier models, yet offers more automation than purely human feedback. We present a systematic study to evaluate and understand weak LLM's ability to generate feedback for alignment. Our empirical findings demonstrate that weak LLMs can provide feedback that rivals or even exceeds that of fully human-annotated data. Our study indicates a minimized impact of model size on feedback efficacy, shedding light on a scalable and sustainable alignment strategy. To deepen our understanding of alignment under weak LLM feedback, we conduct a series of qualitative and quantitative analyses, offering novel insights into the quality discrepancies between human feedback vs. weak LLM feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18080v1">Stabilizing Reasoning in Medical LLMs with Continued Pretraining and Reasoning Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show potential in medicine, yet clinical adoption is hindered by concerns over factual accuracy, language-specific limitations (e.g., Japanese), and critically, their reliability when required to generate reasoning explanations -- a prerequisite for trust. This paper introduces Preferred-MedLLM-Qwen-72B, a 72B-parameter model optimized for the Japanese medical domain to achieve both high accuracy and stable reasoning. We employ a two-stage fine-tuning process on the Qwen2.5-72B base model: first, Continued Pretraining (CPT) on a comprehensive Japanese medical corpus instills deep domain knowledge. Second, Reasoning Preference Optimization (RPO), a preference-based method, enhances the generation of reliable reasoning pathways while preserving high answer accuracy. Evaluations on the Japanese Medical Licensing Exam benchmark (IgakuQA) show Preferred-MedLLM-Qwen-72B achieves state-of-the-art performance (0.868 accuracy), surpassing strong proprietary models like GPT-4o (0.866). Crucially, unlike baseline or CPT-only models which exhibit significant accuracy degradation (up to 11.5\% and 3.8\% respectively on IgakuQA) when prompted for explanations, our model maintains its high accuracy (0.868) under such conditions. This highlights RPO's effectiveness in stabilizing reasoning generation. This work underscores the importance of optimizing for reliable explanations alongside accuracy. We release the Preferred-MedLLM-Qwen-72B model weights to foster research into trustworthy LLMs for specialized, high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18062v1">LLM-Guided Open RAN: Empowering Hierarchical RAN Intelligent Control</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have led to a significant interest in deploying LLMempowered algorithms for wireless communication networks. Meanwhile, open radio access network (O-RAN) techniques offer unprecedented flexibility, with the non-real-time (non-RT) radio access network (RAN) intelligent controller (RIC) (non-RT RIC) and near-real-time (near-RT) RIC (near-RT RIC) components enabling intelligent resource management across different time scales. In this paper, we propose the LLM empowered hierarchical RIC (LLM-hRIC) framework to improve the collaboration between RICs. This framework integrates LLMs with reinforcement learning (RL) for efficient network resource management. In this framework, LLMs-empowered non-RT RICs provide strategic guidance and high-level policies based on environmental context. Concurrently, RL-empowered near-RT RICs perform low-latency tasks based on strategic guidance and local near-RT observation. We evaluate the LLM-hRIC framework in an integrated access and backhaul (IAB) network setting. Simulation results demonstrate that the proposed framework achieves superior performance. Finally, we discuss the key future challenges in applying LLMs to O-RAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18041v1">RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Efforts to ensure the safety of large language models (LLMs) include safety fine-tuning, evaluation, and red teaming. However, despite the widespread use of the Retrieval-Augmented Generation (RAG) framework, AI safety work focuses on standard LLMs, which means we know little about how RAG use cases change a model's safety profile. We conduct a detailed comparative analysis of RAG and non-RAG frameworks with eleven LLMs. We find that RAG can make models less safe and change their safety profile. We explore the causes of this change and find that even combinations of safe models with safe documents can cause unsafe generations. In addition, we evaluate some existing red teaming methods for RAG settings and show that they are less effective than when used for non-RAG settings. Our work highlights the need for safety research and red-teaming methods specifically tailored for RAG LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12881v2">MIND: Math Informed syNthetic Dialogues for Pretraining LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 31 pages, 5 figures, 14 tables
    </div>
    <details class="paper-abstract">
      The utility of synthetic data to enhance pretraining data quality and hence to improve downstream task accuracy has been widely explored in recent large language models (LLMs). Yet, these approaches fall inadequate in complex, multi-hop and mathematical reasoning tasks as the synthetic data typically fails to add complementary knowledge to the existing raw corpus. In this work, we propose a novel large-scale and diverse Math Informed syNthetic Dialogue (MIND) generation method that improves the mathematical reasoning ability of LLMs. Specifically, using MIND, we generate synthetic conversations based on OpenWebMath (OWM), resulting in a new math corpus, MIND-OWM. Our experiments with different conversational settings reveal that incorporating knowledge gaps between dialog participants is essential for generating high-quality math data. We further identify an effective way to format and integrate synthetic and raw data during pretraining to maximize the gain in mathematical reasoning, emphasizing the need to restructure raw data rather than use it as-is. Compared to pretraining just on raw data, a model pretrained on MIND-OWM shows significant boost in mathematical reasoning (GSM8K: +13.42%, MATH: +2.30%), including superior performance in specialized knowledge (MMLU: +4.55%, MMLU-STEM: +4.28%) and general purpose reasoning tasks (GENERAL REASONING: +2.51%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19325v3">Nearest Neighbor Speculative Decoding for LLM Generation and Attribution</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often hallucinate and lack the ability to provide attribution for their generations. Semi-parametric LMs, such as kNN-LM, approach these limitations by refining the output of an LM for a given prompt using its nearest neighbor matches in a non-parametric data store. However, these models often exhibit slow inference speeds and produce non-fluent texts. In this paper, we introduce Nearest Neighbor Speculative Decoding (NEST), a novel semi-parametric language modeling approach that is capable of incorporating real-world text spans of arbitrary length into the LM generations and providing attribution to their sources. NEST performs token-level retrieval at each inference step to compute a semi-parametric mixture distribution and identify promising span continuations in a corpus. It then uses an approximate speculative decoding procedure that accepts a prefix of the retrieved span or generates a new token. NEST significantly enhances the generation quality and attribution rate of the base LM across a variety of knowledge-intensive tasks, surpassing the conventional kNN-LM method and performing competitively with in-context retrieval augmentation. In addition, NEST substantially improves the generation speed, achieving a 1.8x speedup in inference time when applied to Llama-2-Chat 70B. Code will be released at https://github.com/facebookresearch/NEST/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17999v1">Streaming, Fast and Slow: Cognitive Load-Aware Streaming for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Generative conversational interfaces powered by large language models (LLMs) typically stream output token-by-token at a rate determined by computational budget, often neglecting actual human reading speeds and the cognitive load associated with the content. This mismatch frequently leads to inefficient use of computational resources. For example, in cloud-based services, streaming content faster than users can read appears unnecessary, resulting in wasted computational resources and potential delays for other users, particularly during peak usage periods. To address this issue, we propose an adaptive streaming method that dynamically adjusts the pacing of LLM streaming output in real-time based on inferred cognitive load. Our approach estimates the cognitive load associated with streaming content and strategically slows down the stream during complex or information-rich segments, thereby freeing computational resources for other users. Our statistical analysis of computational savings, combined with crowdsourced user studies, provides insights into the trade-offs between service efficiency and user satisfaction, demonstrating that our method can significantly reduce computational consumption up to 16.8\%. This context-aware computational resource management strategy presents a practical framework for enhancing system efficiency in cloud-based conversational AI interfaces without compromising user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17993v1">Improving LLM Personas via Rationalization with Psychological Scaffolds</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Language models prompted with a user description or persona can predict a user's preferences and opinions, but existing approaches to building personas -- based solely on a user's demographic attributes and/or prior judgments -- fail to capture the underlying reasoning behind said user judgments. We introduce PB&J (Psychology of Behavior and Judgments), a framework that improves LLM personas by incorporating rationales of why a user might make specific judgments. These rationales are LLM-generated, and aim to reason about a user's behavior on the basis of their experiences, personality traits or beliefs. This is done using psychological scaffolds -- structured frameworks grounded in theories such as the Big 5 Personality Traits and Primal World Beliefs -- that help provide structure to the generated rationales. Experiments on public opinion and movie preference prediction tasks demonstrate that LLM personas augmented with PB&J rationales consistently outperform methods using only a user's demographics and/or judgments. Additionally, LLM personas constructed using scaffolds describing user beliefs perform competitively with those using human-written rationales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10370v3">Papers-to-Posts: Supporting Detailed Long-Document Summarization with an Interactive LLM-Powered Source Outline</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 Revised for clearer message
    </div>
    <details class="paper-abstract">
      Compressing long and technical documents (e.g., >10 pages) into shorter-form articles (e.g., <2 pages) is critical for communicating information to different audiences, for example, blog posts of scientific research paper or legal briefs of dense court proceedings. While large language models (LLMs) are powerful tools for condensing large amounts of text, current interfaces to these models lack support for understanding and controlling what content is included in a detailed summarizing article. Such capability is especially important for detail- and technical-oriented domains, in which tactical selection and coherent synthesis of key details is critical for effective communication to the target audience. For this, we present interactive reverse source outlines, a novel mechanism for controllable long-form summarization featuring outline bullet points with automatic point selections that the user can iteratively adjust to obtain an article with the desired content coverage. We implement this mechanism in Papers-to-Posts, a new LLM-powered system for authoring research-paper blog posts. Through a within-subjects lab study (n=20) and a between-subjects deployment study (n=37 blog posts, 26 participants), we compare Papers-to-Posts to a strong baseline tool that provides an LLM-generated draft and access to free-form prompting. Under time constraints, Papers-to-Posts significantly increases writer satisfaction with blog post quality, particularly with respect to content coverage. Furthermore, quantitative results showed an increase in editing power (change in text for an amount of time or writing actions) while using Papers-to-Posts, and qualitative results showed that participants found incorporating key research-paper insights in their blog posts easier while using Papers-to-Posts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18691v1">From Prompts to Propositions: A Logic-Based Lens on Student-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Background and Context. The increasing integration of large language models (LLMs) in computing education presents an emerging challenge in understanding how students use LLMs and craft prompts to solve computational tasks. Prior research has used both qualitative and quantitative methods to analyze prompting behavior, but these approaches lack scalability or fail to effectively capture the semantic evolution of prompts. Objective. In this paper, we investigate whether students prompts can be systematically analyzed using propositional logic constraints. We examine whether this approach can identify patterns in prompt evolution, detect struggling students, and provide insights into effective and ineffective strategies. Method. We introduce Prompt2Constraints, a novel method that translates students prompts into logical constraints. The constraints are able to represent the intent of the prompts in succinct and quantifiable ways. We used this approach to analyze a dataset of 1,872 prompts from 203 students solving introductory programming tasks. Findings. We find that while successful and unsuccessful attempts tend to use a similar number of constraints overall, when students fail, they often modify their prompts more significantly, shifting problem-solving strategies midway. We also identify points where specific interventions could be most helpful to students for refining their prompts. Implications. This work offers a new and scalable way to detect students who struggle in solving natural language programming tasks. This work could be extended to investigate more complex tasks and integrated into programming tools to provide real-time support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18671v1">Proof-of-TBI -- Fine-Tuned Vision Language Model Consortium and OpenAI-o3 Reasoning LLM-Based Medical Diagnosis Support System for Mild Traumatic Brain Injury (TBI) Prediction</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Mild Traumatic Brain Injury (TBI) detection presents significant challenges due to the subtle and often ambiguous presentation of symptoms in medical imaging, making accurate diagnosis a complex task. To address these challenges, we propose Proof-of-TBI, a medical diagnosis support system that integrates multiple fine-tuned vision-language models with the OpenAI-o3 reasoning large language model (LLM). Our approach fine-tunes multiple vision-language models using a labeled dataset of TBI MRI scans, training them to diagnose TBI symptoms effectively. The predictions from these models are aggregated through a consensus-based decision-making process. The system evaluates the predictions from all fine-tuned vision language models using the OpenAI-o3 reasoning LLM, a model that has demonstrated remarkable reasoning performance, to produce the most accurate final diagnosis. The LLM Agents orchestrates interactions between the vision-language models and the reasoning LLM, managing the final decision-making process with transparency, reliability, and automation. This end-to-end decision-making workflow combines the vision-language model consortium with the OpenAI-o3 reasoning LLM, enabled by custom prompt engineering by the LLM agents. The prototype for the proposed platform was developed in collaboration with the U.S. Army Medical Research team in Newport News, Virginia, incorporating five fine-tuned vision-language models. The results demonstrate the transformative potential of combining fine-tuned vision-language model inputs with the OpenAI-o3 reasoning LLM to create a robust, secure, and highly accurate diagnostic system for mild TBI prediction. To the best of our knowledge, this research represents the first application of fine-tuned vision-language models integrated with a reasoning LLM for TBI prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21465v3">ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      With the widespread deployment of long-context large language models (LLMs), there has been a growing demand for efficient support of high-throughput inference. However, as the key-value (KV) cache expands with the sequence length, the increasing memory footprint and the need to access it for each token generation both result in low throughput when serving long-context LLMs. While various dynamic sparse attention methods have been proposed to speed up inference while maintaining generation quality, they either fail to sufficiently reduce GPU memory consumption or introduce significant decoding latency by offloading the KV cache to the CPU. We present ShadowKV, a high-throughput long-context LLM inference system that stores the low-rank key cache and offloads the value cache to reduce the memory footprint for larger batch sizes and longer sequences. To minimize decoding latency, ShadowKV employs an accurate KV selection strategy that reconstructs minimal sparse KV pairs on-the-fly. By evaluating ShadowKV on a broad range of benchmarks, including RULER, LongBench, and Needle In A Haystack, and models like Llama-3.1-8B, Llama-3-8B-1M, GLM-4-9B-1M, Yi-9B-200K, Phi-3-Mini-128K, and Qwen2-7B-128K, we demonstrate that it can support up to 6$\times$ larger batch sizes and boost throughput by up to 3.04$\times$ on an A100 GPU without sacrificing accuracy, even surpassing the performance achievable with infinite batch size under the assumption of infinite GPU memory. The code is available at https://github.com/bytedance/ShadowKV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16543v2">Multi-Agent LLMs Ensemble for Efficient Atrial Fibrillation Annotation of ECG Reports</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 26 pages, 12 figures
    </div>
    <details class="paper-abstract">
      This study introduces a novel multiagent ensemble method powered by LLMs to address a key challenge in ML - data labeling, particularly in large-scale EHR datasets. Manual labeling of such datasets requires domain expertise and is labor-intensive, time-consuming, expensive, and error-prone. To overcome this bottleneck, we developed an ensemble LLMs method and demonstrated its effectiveness in two real-world tasks: (1) labeling a large-scale unlabeled ECG dataset in MIMIC-IV; (2) identifying social determinants of health (SDOH) from the clinical notes of EHR. Trading off benefits and cost, we selected a pool of diverse open source LLMs with satisfactory performance. We treat each LLM's prediction as a vote and apply a mechanism of majority voting with minimal winning threshold for ensemble. We implemented an ensemble LLMs application for EHR data labeling tasks. By using the ensemble LLMs and natural language processing, we labeled MIMIC-IV ECG dataset of 623,566 ECG reports with an estimated accuracy of 98.2%. We applied the ensemble LLMs method to identify SDOH from social history sections of 1,405 EHR clinical notes, also achieving competitive performance. Our experiments show that the ensemble LLMs can outperform individual LLM even the best commercial one, and the method reduces hallucination errors. From the research, we found that (1) the ensemble LLMs method significantly reduces the time and effort required for labeling large-scale EHR data, automating the process with high accuracy and quality; (2) the method generalizes well to other text data labeling tasks, as shown by its application to SDOH identification; (3) the ensemble of a group of diverse LLMs can outperform or match the performance of the best individual LLM; and (4) the ensemble method substantially reduces hallucination errors. This approach provides a scalable and efficient solution to data-labeling challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12984v2">Tilus: A Virtual Machine for Arbitrary Low-Precision GPGPU Computation in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 18 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) is critical for AI-powered applications but demands substantial computational resources, particularly in memory bandwidth and computational throughput. Low-precision computation has emerged as a key technique to improve efficiency while reducing resource consumption. Existing approaches for generating low-precision kernels are limited to weight bit widths that are powers of two and suffer from suboptimal performance due to high-level GPU programming abstractions. These abstractions restrict critical optimizations, such as fine-grained register management and optimized memory access patterns, which are essential for efficient low-precision computations. In this paper, we introduce a virtual machine (VM) designed for General-Purpose GPU (GPGPU) computing, enabling support for low-precision data types with arbitrary bit widths while maintaining GPU programmability. The proposed VM features a thread-block-level programming model, a hierarchical memory space, a novel algebraic layout system, and extensive support for diverse low-precision data types. VM programs are compiled into highly efficient GPU programs with automatic vectorization and instruction selection. Extensive experiments demonstrate that our VM efficiently supports a full spectrum of low-precision data types, and outperforms state-of-the-art low-precision kernels on their supported types. Compared to existing compilers like Triton and Ladder, as well as hand-optimized kernels such as QuantLLM and Marlin, our VM achieves performance improvements of 1.75x, 2.61x, 1.29x and 1.03x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18639v1">Span-Level Hallucination Detection for LLM-Generated Answers</a></div>
    <div class="paper-meta">
      📅 2025-04-25
    </div>
    <details class="paper-abstract">
      Detecting spans of hallucination in LLM-generated answers is crucial for improving factual consistency. This paper presents a span-level hallucination detection framework for the SemEval-2025 Shared Task, focusing on English and Arabic texts. Our approach integrates Semantic Role Labeling (SRL) to decompose the answer into atomic roles, which are then compared with a retrieved reference context obtained via question-based LLM prompting. Using a DeBERTa-based textual entailment model, we evaluate each role semantic alignment with the retrieved context. The entailment scores are further refined through token-level confidence measures derived from output logits, and the combined scores are used to detect hallucinated spans. Experiments on the Mu-SHROOM dataset demonstrate competitive performance. Additionally, hallucinated spans have been verified through fact-checking by prompting GPT-4 and LLaMA. Our findings contribute to improving hallucination detection in LLM-generated responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20092v1">An Integrated Framework for Contextual Personalized LLM-Based Food Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-25
      | 💬 Doctorate Thesis, University of California, Irvine 2024
    </div>
    <details class="paper-abstract">
      Personalized food recommendation systems (Food-RecSys) critically underperform due to fragmented component understanding and the failure of conventional machine learning with vast, imbalanced food data. While Large Language Models (LLMs) offer promise, current generic Recommendation as Language Processing (RLP) strategies lack the necessary specialization for the food domain's complexity. This thesis tackles these deficiencies by first identifying and analyzing the essential components for effective Food-RecSys. We introduce two key innovations: a multimedia food logging platform for rich contextual data acquisition and the World Food Atlas, enabling unique geolocation-based food analysis previously unavailable. Building on this foundation, we pioneer the Food Recommendation as Language Processing (F-RLP) framework - a novel, integrated approach specifically architected for the food domain. F-RLP leverages LLMs in a tailored manner, overcoming the limitations of generic models and providing a robust infrastructure for effective, contextual, and truly personalized food recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17768v1">The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Sparse attention offers a promising strategy to extend long-context capabilities in Transformer LLMs, yet its viability, its efficiency-accuracy trade-offs, and systematic scaling studies remain unexplored. To address this gap, we perform a careful comparison of training-free sparse attention methods at varying model scales, sequence lengths, and sparsity levels on a diverse collection of long-sequence tasks-including novel ones that rely on natural language while remaining controllable and easy to evaluate. Based on our experiments, we report a series of key findings: 1) an isoFLOPS analysis reveals that for very long sequences, larger and highly sparse models are preferable to smaller and dense ones. 2) The level of sparsity attainable while statistically guaranteeing accuracy preservation is higher during decoding than prefilling, and correlates with model size in the former. 3) There is no clear strategy that performs best across tasks and phases, with different units of sparsification or budget adaptivity needed for different scenarios. Even moderate sparsity levels often result in significant performance degradation on at least one task, highlighting that sparse attention is not a universal solution. 4) We introduce and validate novel scaling laws specifically tailored for sparse attention, providing evidence that our findings are likely to hold true beyond our range of experiments. Through these insights, we demonstrate that sparse attention is a key tool to enhance the capabilities of Transformer LLMs for processing longer sequences, but requires careful evaluation of trade-offs for performance-sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17723v1">Towards Robust LLMs: an Adversarial Robustness Measurement Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has revolutionized artificial intelligence, yet these models remain vulnerable to adversarial perturbations, undermining their reliability in high-stakes applications. While adversarial robustness in vision-based neural networks has been extensively studied, LLM robustness remains under-explored. We adapt the Robustness Measurement and Assessment (RoMA) framework to quantify LLM resilience against adversarial inputs without requiring access to model parameters. By comparing RoMA's estimates to those of formal verification methods, we demonstrate its accuracy with minimal error margins while maintaining computational efficiency. Our empirical evaluation reveals that robustness varies significantly not only between different models but also across categories within the same task and between various types of perturbations. This non-uniformity underscores the need for task-specific robustness evaluations, enabling practitioners to compare and select models based on application-specific robustness requirements. Our work provides a systematic methodology to assess LLM robustness, advancing the development of more reliable language models for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17685v1">Ensemble Bayesian Inference: Leveraging Small Language Models to Achieve LLM-level Accuracy in Profile Matching Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      This study explores the potential of small language model(SLM) ensembles to achieve accuracy comparable to proprietary large language models (LLMs). We propose Ensemble Bayesian Inference (EBI), a novel approach that applies Bayesian estimation to combine judgments from multiple SLMs, allowing them to exceed the performance limitations of individual models. Our experiments on diverse tasks(aptitude assessments and consumer profile analysis in both Japanese and English) demonstrate EBI's effectiveness. Notably, we analyze cases where incorporating models with negative Lift values into ensembles improves overall performance, and we examine the method's efficacy across different languages. These findings suggest new possibilities for constructing high-performance AI systems with limited computational resources and for effectively utilizing models with individually lower performance. Building on existing research on LLM performance evaluation, ensemble methods, and open-source LLM utilization, we discuss the novelty and significance of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16871v2">Exploring How LLMs Capture and Represent Domain-Specific Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      We study whether Large Language Models (LLMs) inherently capture domain-specific nuances in natural language. Our experiments probe the domain sensitivity of LLMs by examining their ability to distinguish queries from different domains using hidden states generated during the prefill phase. We reveal latent domain-related trajectories that indicate the model's internal recognition of query domains. We also study the robustness of these domain representations to variations in prompt styles and sources. Our approach leverages these representations for model selection, mapping the LLM that best matches the domain trace of the input query (i.e., the model with the highest performance on similar traces). Our findings show that LLMs can differentiate queries for related domains, and that the fine-tuned model is not always the most accurate. Unlike previous work, our interpretations apply to both closed and open-ended generative tasks
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11242v4">Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Published at ICLR 2025 (Oral)
    </div>
    <details class="paper-abstract">
      LLMs are an integral component of retrieval-augmented generation (RAG) systems. While many studies focus on evaluating the overall quality of end-to-end RAG systems, there is a gap in understanding the appropriateness of LLMs for the RAG task. To address this, we introduce Trust-Score, a holistic metric that evaluates the trustworthiness of LLMs within the RAG framework. Our results show that various prompting methods, such as in-context learning, fail to effectively adapt LLMs to the RAG task as measured by Trust-Score. Consequently, we propose Trust-Align, a method to align LLMs for improved Trust-Score performance. 26 out of 27 models aligned using Trust-Align substantially outperform competitive baselines on ASQA, QAMPARI, and ELI5. Specifically, in LLaMA-3-8b, Trust-Align outperforms FRONT on ASQA (up 12.56), QAMPARI (up 36.04), and ELI5 (up 17.69). Trust-Align also significantly enhances models' ability to correctly refuse and provide quality citations. We also demonstrate the effectiveness of Trust-Align across different open-weight models, including the LLaMA series (1b to 8b), Qwen-2.5 series (0.5b to 7b), and Phi3.5 (3.8b). We release our code at https://github.com/declare-lab/trust-align.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12533v3">To Help or Not to Help: LLM-based Attentive Support for Human-Robot Group Interactions</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024
    </div>
    <details class="paper-abstract">
      How can a robot provide unobtrusive physical support within a group of humans? We present Attentive Support, a novel interaction concept for robots to support a group of humans. It combines scene perception, dialogue acquisition, situation understanding, and behavior generation with the common-sense reasoning capabilities of Large Language Models (LLMs). In addition to following user instructions, Attentive Support is capable of deciding when and how to support the humans, and when to remain silent to not disturb the group. With a diverse set of scenarios, we show and evaluate the robot's attentive behavior, which supports and helps the humans when required, while not disturbing if no help is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17584v1">L3: DIMM-PIM Integrated Architecture and Coordination for Scalable Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 16 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly require processing long text sequences, but GPU memory limitations force difficult trade-offs between memory capacity and bandwidth. While HBM-based acceleration offers high bandwidth, its capacity remains constrained. Offloading data to host-side DIMMs improves capacity but introduces costly data swapping overhead. We identify that the critical memory bottleneck lies in the decoding phase of multi-head attention (MHA) exclusively, which demands substantial capacity for storing KV caches and high bandwidth for attention computation. Our key insight reveals this operation uniquely aligns with modern DIMM-based processing-in-memory (PIM) architectures, which offers scalability of both capacity and bandwidth. Based on this observation and insight, we propose L3, a hardware-software co-designed system integrating DIMM-PIM and GPU devices. L3 introduces three innovations: First, hardware redesigns resolve data layout mismatches and computational element mismatches in DIMM-PIM, enhancing LLM inference utilization. Second, communication optimization enables hiding the data transfer overhead with the computation. Third, an adaptive scheduler coordinates GPU-DIMM-PIM operations to maximize parallelism between devices. Evaluations using real-world traces show L3 achieves up to 6.1$\times$ speedup over state-of-the-art HBM-PIM solutions while significantly improving batch sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07347v2">Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      As demand for Large Language Models (LLMs) and AI agents rapidly grows, optimizing systems for efficient LLM inference becomes critical. While significant efforts have focused on system-level engineering, little is explored from a mathematical modeling and queuing perspective. In this paper, we aim to develop the queuing fundamentals for large language model (LLM) inference, bridging the gap between the queueing theory and LLM system communities. In particular, we study the throughput aspect in LLM inference systems. We prove that a large class of 'work-conserving' scheduling algorithms can achieve maximum throughput for individual inference LLM engine, highlighting 'work-conserving' as a key design principle in practice. In a network of LLM agents, work-conserving scheduling alone is insufficient, particularly when facing specific workload structures and multi-class workflows that require more sophisticated scheduling strategies. Evaluations of real-world systems show that Orca and Sarathi-serve are throughput-optimal, reassuring practitioners, while FasterTransformer and vanilla vLLM are not maximally stable and should be used with caution. Our results highlight the substantial benefits that the queueing community can offer in improving LLM inference systems and call for more interdisciplinary development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17565v1">DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have recently achieved remarkable performance on various complex reasoning benchmarks, the academic community still lacks an in-depth understanding of base model training processes and data quality. To address this, we construct a large-scale, difficulty-graded reasoning dataset containing approximately 3.34 million unique queries of varying difficulty levels and about 40 million distilled responses generated by multiple models over several passes. Leveraging pass rate and Coefficient of Variation (CV), we precisely select the most valuable training data to enhance reasoning capability. Notably, we observe a training pattern shift, indicating that reasoning-focused training based on base models requires higher learning rates for effective training. Using this carefully selected data, we significantly improve the reasoning capabilities of the base model, achieving a pass rate of 79.2\% on the AIME2024 mathematical reasoning benchmark. This result surpasses most current distilled models and closely approaches state-of-the-art performance. We provide detailed descriptions of our data processing, difficulty assessment, and training methodology, and have publicly released all datasets and methods to promote rapid progress in open-source long-reasoning LLMs. The dataset is available at: https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17550v1">HalluLens: LLM Hallucination Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate responses that deviate from user input or training data, a phenomenon known as "hallucination." These hallucinations undermine user trust and hinder the adoption of generative AI systems. Addressing hallucinations is essential for the advancement of LLMs. This paper introduces a comprehensive hallucination benchmark, incorporating both new extrinsic and existing intrinsic evaluation tasks, built upon clear taxonomy of hallucination. A major challenge in benchmarking hallucinations is the lack of a unified framework due to inconsistent definitions and categorizations. We disentangle LLM hallucination from "factuality," proposing a clear taxonomy that distinguishes between extrinsic and intrinsic hallucinations, to promote consistency and facilitate research. Extrinsic hallucinations, where the generated content is not consistent with the training data, are increasingly important as LLMs evolve. Our benchmark includes dynamic test set generation to mitigate data leakage and ensure robustness against such leakage. We also analyze existing benchmarks, highlighting their limitations and saturation. The work aims to: (1) establish a clear taxonomy of hallucinations, (2) introduce new extrinsic hallucination tasks, with data that can be dynamically regenerated to prevent saturation by leakage, (3) provide a comprehensive analysis of existing benchmarks, distinguishing them from factuality evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02868v2">Enhancing LLMs with Smart Preprocessing for EHR Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in natural language processing; however, their application in sensitive domains such as healthcare, especially in processing Electronic Health Records (EHRs), is constrained by limited computational resources and privacy concerns. This paper introduces a compact LLM framework optimized for local deployment in environments with stringent privacy requirements and restricted access to high-performance GPUs. Our approach leverages simple yet powerful preprocessing techniques, including regular expressions (regex) and Retrieval-Augmented Generation (RAG), to extract and highlight critical information from clinical notes. By pre-filtering long, unstructured text, we enhance the performance of smaller LLMs on EHR-related tasks. Our framework is evaluated using zero-shot and few-shot learning paradigms on both private and publicly available datasets (MIMIC-IV), with additional comparisons against fine-tuned LLMs on MIMIC-IV. Experimental results demonstrate that our preprocessing strategy significantly supercharges the performance of smaller LLMs, making them well-suited for privacy-sensitive and resource-constrained applications. This study offers valuable insights into optimizing LLM performance for local, secure, and efficient healthcare applications. It provides practical guidance for real-world deployment for LLMs while tackling challenges related to privacy, computational feasibility, and clinical applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17497v1">Combining GCN Structural Learning with LLM Chemical Knowledge for or Enhanced Virtual Screening</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Virtual screening plays a critical role in modern drug discovery by enabling the identification of promising candidate molecules for experimental validation. Traditional machine learning methods such as support vector machines (SVM) and XGBoost rely on predefined molecular representations, often leading to information loss and potential bias. In contrast, deep learning approaches-particularly Graph Convolutional Networks (GCNs)-offer a more expressive and unbiased alternative by operating directly on molecular graphs. Meanwhile, Large Language Models (LLMs) have recently demonstrated state-of-the-art performance in drug design, thanks to their capacity to capture complex chemical patterns from large-scale data via attention mechanisms. In this paper, we propose a hybrid architecture that integrates GCNs with LLM-derived embeddings to combine localized structural learning with global chemical knowledge. The LLM embeddings can be precomputed and stored in a molecular feature library, removing the need to rerun the LLM during training or inference and thus maintaining computational efficiency. We found that concatenating the LLM embeddings after each GCN layer-rather than only at the final layer-significantly improves performance, enabling deeper integration of global context throughout the network. The resulting model achieves superior results, with an F1-score of (88.8%), outperforming standalone GCN (87.9%), XGBoost (85.5%), and SVM (85.4%) baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17445v1">Creating Targeted, Interpretable Topic Models with LLM-Generated Text Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Presented at IC2S2 2024 in Philadelphia, USA
    </div>
    <details class="paper-abstract">
      Unsupervised machine learning techniques, such as topic modeling and clustering, are often used to identify latent patterns in unstructured text data in fields such as political science and sociology. These methods overcome common concerns about reproducibility and costliness involved in the labor-intensive process of human qualitative analysis. However, two major limitations of topic models are their interpretability and their practicality for answering targeted, domain-specific social science research questions. In this work, we investigate opportunities for using LLM-generated text augmentation to improve the usefulness of topic modeling output. We use a political science case study to evaluate our results in a domain-specific application, and find that topic modeling using GPT-4 augmentations creates highly interpretable categories that can be used to investigate domain-specific research questions with minimal human guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17432v1">Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 13 pages, 8 figures, Project page: https://garygutc.github.io/UniME
    </div>
    <details class="paper-abstract">
      The Contrastive Language-Image Pre-training (CLIP) framework has become a widely used approach for multimodal representation learning, particularly in image-text retrieval and clustering. However, its efficacy is constrained by three key limitations: (1) text token truncation, (2) isolated image-text encoding, and (3) deficient compositionality due to bag-of-words behavior. While recent Multimodal Large Language Models (MLLMs) have demonstrated significant advances in generalized vision-language understanding, their potential for learning transferable multimodal representations remains underexplored.In this work, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that leverages MLLMs to learn discriminative representations for diverse downstream tasks. In the first stage, we perform textual discriminative knowledge distillation from a powerful LLM-based teacher model to enhance the embedding capability of the MLLM\'s language component. In the second stage, we introduce hard negative enhanced instruction tuning to further advance discriminative representation learning. Specifically, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short and long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19151v2">Can LLMs Really Learn to Translate a Low-Resource Language from One Grammar Book?</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Accepted at ICLR 2025 (Spotlight)
    </div>
    <details class="paper-abstract">
      Extremely low-resource (XLR) languages lack substantial corpora for training NLP models, motivating the use of all available resources such as dictionaries and grammar books. Machine Translation from One Book (Tanzer et al., 2024) suggests that prompting long-context LLMs with one grammar book enables English-Kalamang translation, an XLR language unseen by LLMs - a noteworthy case of linguistics helping an NLP task. We investigate the source of this translation ability, finding almost all improvements stem from the book's parallel examples rather than its grammatical explanations. We find similar results for Nepali and Guarani, seen low-resource languages, and we achieve performance comparable to an LLM with a grammar book by simply fine-tuning an encoder-decoder translation model. We then investigate where grammar books help by testing two linguistic tasks, grammaticality judgment and gloss prediction, and we explore what kind of grammatical knowledge helps by introducing a typological feature prompt that achieves leading results on these more relevant tasks. We thus emphasise the importance of task-appropriate data for XLR languages: parallel examples for translation, and grammatical data for linguistic tasks. As we find no evidence that long-context LLMs can make effective use of grammatical explanations for XLR translation, we conclude data collection for multilingual XLR tasks such as translation is best focused on parallel data over linguistic description.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17376v1">On-Device Qwen2.5: Efficient LLM Inference with Model Compression and Hardware Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Transformer-based Large Language Models (LLMs) have significantly advanced AI capabilities but pose considerable challenges for deployment on edge devices due to high computational demands, memory bandwidth constraints, and energy consumption. This paper addresses these challenges by presenting an efficient framework for deploying the Qwen2.5-0.5B model on the Xilinx Kria KV260 edge platform, a heterogeneous system integrating an ARM Cortex-A53 CPU with reconfigurable FPGA logic. Leveraging Activation-aware Weight Quantization (AWQ) with FPGA-accelerated execution pipelines, the proposed approach enhances both model compression rate and system throughput. Additionally, we propose a hybrid execution strategy that intelligently offloads compute-intensive operations to the FPGA while utilizing the CPU for lighter tasks, effectively balancing the computational workload and maximizing overall performance. Our framework achieves a model compression rate of 55.08% compared to the original model and produces output at a rate of 5.1 tokens per second, outperforming the baseline performance of 2.8 tokens per second.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17334v1">DataScout: Automatic Data Fact Retrieval for Statement Augmentation with an LLM-Based Agent</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      A data story typically integrates data facts from multiple perspectives and stances to construct a comprehensive and objective narrative. However, retrieving these facts demands time for data search and challenges the creator's analytical skills. In this work, we introduce DataScout, an interactive system that automatically performs reasoning and stance-based data facts retrieval to augment the user's statement. Particularly, DataScout leverages an LLM-based agent to construct a retrieval tree, enabling collaborative control of its expansion between users and the agent. The interface visualizes the retrieval tree as a mind map that eases users to intuitively steer the retrieval direction and effectively engage in reasoning and analysis. We evaluate the proposed system through case studies and in-depth expert interviews. Our evaluation demonstrates that DataScout can effectively retrieve multifaceted data facts from different stances, helping users verify their statements and enhance the credibility of their stories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17331v1">Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation measures include eye-tracking data analysis, including explainable machine learning through SHAP analysis as well as standardized questionnaires for usability, presence, cybersickness, and cognitive load to examine user attention and engagement. Our findings indicate that the LLM-driven locomotion possesses comparable usability, presence, and cybersickness scores to established methods like teleportation, demonstrating its novel potential as a comfortable, natural language-based, hands-free alternative. In addition, it enhances user attention within the virtual environment, suggesting greater engagement. Complementary to these findings, SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13471v2">From Large to Super-Tiny: End-to-End Optimization for Cost-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have significantly advanced artificial intelligence by optimizing traditional Natural Language Processing (NLP) pipelines, improving performance and generalization. This has spurred their integration into various systems. Many NLP systems, including ours, employ a "one-stage" pipeline directly incorporating LLMs. While effective, this approach incurs substantial costs and latency due to the need for large model parameters to achieve satisfactory outcomes. This paper introduces a three-stage cost-efficient end-to-end LLM deployment pipeline-including prototyping, knowledge transfer, and model compression-to tackle the cost-performance dilemma in LLM-based frameworks. Our approach yields a super tiny model optimized for cost and performance in online systems, simplifying the system architecture. Initially, by transforming complex tasks into a function call-based LLM-driven pipeline, an optimal performance prototype system is constructed to produce high-quality data as a teacher model. The second stage combines techniques like rejection fine-tuning, reinforcement learning, and knowledge distillation to transfer knowledge to a smaller 0.5B student model, delivering effective performance at minimal cost. The final stage applies quantization and pruning to extremely compress models to 0.4B, achieving ultra-low latency and cost. The framework's modular design and cross-domain capabilities suggest potential applicability in other NLP areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17226v1">FLAG: Formal and LLM-assisted SVA Generation for Formal Specifications of On-Chip Communication Protocols</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 9 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Formal specifications of on-chip communication protocols are crucial for system-on-chip (SoC) design and verification. However, manually constructing these formal specifications from informal documents remains a tedious and error-prone task. Although recent efforts have used Large Language Models (LLMs) to generate SystemVerilog Assertion (SVA) properties from design documents for Register-Transfer Level (RTL) design verification, in our experience these approaches have not shown promise in generating SVA properties for communication protocols. Since protocol specification documents are unstructured and ambiguous in nature, LLMs often fail to extract the necessary information and end up generating irrelevant or even incorrect properties. We propose FLAG, a two-stage framework to help construct formal protocol specifications from informal documents. In the first stage, a predefined template set is used to generate candidate SVA properties. To avoid missing necessary properties, we develop a grammar-based approach to generate comprehensive template sets that capture critical signal behaviors for various communication protocols. In the second stage, we utilize unambiguous timing diagrams in conjunction with textual descriptions from the specification documents to filter out incorrect properties. A formal approach is first implemented to check the candidate properties and filter out those inconsistent with the timing diagrams. An LLM is then consulted to further remove incorrect properties with respect to the textual description, obtaining the final property set. Experiments on various open-source communication protocols demonstrate the effectiveness of FLAG in generating SVA properties from informal documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17200v1">A RAG-Based Multi-Agent LLM System for Natural Hazard Resilience and Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are a transformational capability at the frontier of artificial intelligence and machine learning that can support decision-makers in addressing pressing societal challenges such as extreme natural hazard events. As generalized models, LLMs often struggle to provide context-specific information, particularly in areas requiring specialized knowledge. In this work we propose a retrieval-augmented generation (RAG)-based multi-agent LLM system to support analysis and decision-making in the context of natural hazards and extreme weather events. As a proof of concept, we present WildfireGPT, a specialized system focused on wildfire hazards. The architecture employs a user-centered, multi-agent design to deliver tailored risk insights across diverse stakeholder groups. By integrating natural hazard and extreme weather projection data, observational datasets, and scientific literature through an RAG framework, the system ensures both the accuracy and contextual relevance of the information it provides. Evaluation across ten expert-led case studies demonstrates that WildfireGPT significantly outperforms existing LLM-based solutions for decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13192v2">CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Accepted by KDD 2024;
    </div>
    <details class="paper-abstract">
      Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06425v2">Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 Accepted to The ACM Web Conference (WWW) 2025 Short Paper Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized in domains such as finance, healthcare, and interpersonal relationships to provide advice tailored to user traits and contexts. However, this personalization often relies on sensitive data, raising critical privacy concerns and necessitating data minimization. To address these challenges, we propose a framework that integrates zero-knowledge proof (ZKP) technology, specifically zkVM, with LLM-based chatbots. This integration enables privacy-preserving data sharing by verifying user traits without disclosing sensitive information. Our research introduces both an architecture and a prompting strategy for this approach. Through empirical evaluation, we clarify the current constraints and performance limitations of both zkVM and the proposed prompting strategy, thereby demonstrating their practical feasibility in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17967v1">LLM Agent Swarm for Hypothesis-Driven Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Drug discovery remains a formidable challenge: more than 90 percent of candidate molecules fail in clinical evaluation, and development costs often exceed one billion dollars per approved therapy. Disparate data streams, from genomics and transcriptomics to chemical libraries and clinical records, hinder coherent mechanistic insight and slow progress. Meanwhile, large language models excel at reasoning and tool integration but lack the modular specialization and iterative memory required for regulated, hypothesis-driven workflows. We introduce PharmaSwarm, a unified multi-agent framework that orchestrates specialized LLM "agents" to propose, validate, and refine hypotheses for novel drug targets and lead compounds. Each agent accesses dedicated functionality--automated genomic and expression analysis; a curated biomedical knowledge graph; pathway enrichment and network simulation; interpretable binding affinity prediction--while a central Evaluator LLM continuously ranks proposals by biological plausibility, novelty, in silico efficacy, and safety. A shared memory layer captures validated insights and fine-tunes underlying submodels over time, yielding a self-improving system. Deployable on low-code platforms or Kubernetes-based microservices, PharmaSwarm supports literature-driven discovery, omics-guided target identification, and market-informed repurposing. We also describe a rigorous four-tier validation pipeline spanning retrospective benchmarking, independent computational assays, experimental testing, and expert user studies to ensure transparency, reproducibility, and real-world impact. By acting as an AI copilot, PharmaSwarm can accelerate translational research and deliver high-confidence hypotheses more efficiently than traditional pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06621v5">Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 ICLR 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong reasoning and memorization capabilities via pretraining on massive textual corpora. However, this poses risk of privacy and copyright violations, highlighting the need for efficient machine unlearning methods that remove sensitive data without retraining from scratch. While Gradient Ascent (GA) is commonly used to unlearn by reducing the likelihood of generating unwanted content, it leads to unstable optimization and catastrophic forgetting of retrained knowledge. We find that combining GA with low-rank adaptation results in poor trade-offs between computational cost and generative performance. To address these challenges, we propose Low-rank Knowledge Unlearning (LoKU), a novel framework that enables robust and efficient unlearning for LLMs. First, we introduce Inverted Hinge Loss, which suppresses unwanted tokens while maintaining fluency by boosting the probability of the next most likely token. Second, we develop a data-adaptive initialization for LoRA adapters via low-rank approximation weighted with relative Fisher information, thereby focusing updates on parameters critical for removing targeted knowledge. Experiments on the Training Data Extraction Challenge dataset using GPT-Neo models as well as on the TOFU benchmark with Phi-1.5B and Llama2-7B models demonstrate that our approach effectively removes sensitive information while maintaining reasoning and generative capabilities with minimal impact. Our implementation can be found in https://github.com/csm9493/efficient-llm-unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17950v1">Collaborating Action by Action: A Multi-agent LLM Framework for Embodied Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 9 pages of main paper with 6 main figures, overall 28 pages
    </div>
    <details class="paper-abstract">
      Collaboration is ubiquitous and essential in day-to-day life -- from exchanging ideas, to delegating tasks, to generating plans together. This work studies how LLMs can adaptively collaborate to perform complex embodied reasoning tasks. To this end we introduce MINDcraft, an easily extensible platform built to enable LLM agents to control characters in the open-world game of Minecraft; and MineCollab, a benchmark to test the different dimensions of embodied and collaborative reasoning. An experimental study finds that the primary bottleneck in collaborating effectively for current state-of-the-art agents is efficient natural language communication, with agent performance dropping as much as 15% when they are required to communicate detailed task completion plans. We conclude that existing LLM agents are ill-optimized for multi-agent collaboration, especially in embodied scenarios, and highlight the need to employ methods beyond in-context and imitation learning. Our website can be found here: https://mindcraft-minecollab.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17934v1">Toward a Human-Centered Evaluation Framework for Trustworthy LLM-Powered GUI Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has revolutionized Graphical User Interface (GUI) automation through LLM-powered GUI agents, yet their ability to process sensitive data with limited human oversight raises significant privacy and security risks. This position paper identifies three key risks of GUI agents and examines how they differ from traditional GUI automation and general autonomous agents. Despite these risks, existing evaluations focus primarily on performance, leaving privacy and security assessments largely unexplored. We review current evaluation metrics for both GUI and general LLM agents and outline five key challenges in integrating human evaluators for GUI agent assessments. To address these gaps, we advocate for a human-centered evaluation framework that incorporates risk assessments, enhances user awareness through in-context consent, and embeds privacy and security considerations into GUI agent design and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08040v2">Can Reasoning LLMs Enhance Clinical Document Classification?</a></div>
    <div class="paper-meta">
      📅 2025-04-24
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Clinical document classification is essential for converting unstructured medical texts into standardised ICD-10 diagnoses, yet it faces challenges due to complex medical language, privacy constraints, and limited annotated datasets. Large Language Models (LLMs) offer promising improvements in accuracy and efficiency for this task. This study evaluates the performance and consistency of eight LLMs; four reasoning (Qwen QWQ, Deepseek Reasoner, GPT o3 Mini, Gemini 2.0 Flash Thinking) and four non-reasoning (Llama 3.3, GPT 4o Mini, Gemini 2.0 Flash, Deepseek Chat); in classifying clinical discharge summaries using the MIMIC-IV dataset. Using cTAKES to structure clinical narratives, models were assessed across three experimental runs, with majority voting determining final predictions. Results showed that reasoning models outperformed non-reasoning models in accuracy (71% vs 68%) and F1 score (67% vs 60%), with Gemini 2.0 Flash Thinking achieving the highest accuracy (75%) and F1 score (76%). However, non-reasoning models demonstrated greater stability (91% vs 84% consistency). Performance varied across ICD-10 codes, with reasoning models excelling in complex cases but struggling with abstract categories. Findings indicate a trade-off between accuracy and consistency, suggesting that a hybrid approach could optimise clinical coding. Future research should explore multi-label classification, domain-specific fine-tuning, and ensemble methods to enhance model reliability in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16266v2">TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge platforms is challenged by their high computational and memory demands. Although recent low-bit quantization methods (e.g., BitNet, DeepSeek) compress weights to as little as 1.58 bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected latency of the prefill phase. We present TeLLMe, the first ternary LLM accelerator for low-power FPGAs (e.g., AMD KV260) that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. Our contributions include: (1) a table-lookup matrix engine for ternary matmul that merges grouped activations with online precomputation to minimize resource use; (2) a fused, bandwidth-efficient attention module featuring a reversed reordering scheme to accelerate prefill; and (3) a tightly integrated normalization and quantization--dequantization unit optimized for ultra-low-bit inference. Under a 7W power budget, TeLLMe delivers up to 9 tokens/s throughput over 1,024-token contexts and prefill latencies of 0.55--1.15 s for 64--128 token prompts, marking a significant energy-efficiency advance and establishing a new edge FPGA benchmark for generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17833v1">The Role of Open-Source LLMs in Shaping the Future of GeoAI</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming geospatial artificial intelligence (GeoAI), offering new capabilities in data processing, spatial analysis, and decision support. This paper examines the open-source paradigm's pivotal role in this transformation. While proprietary LLMs offer accessibility, they often limit the customization, interoperability, and transparency vital for specialized geospatial tasks. Conversely, open-source alternatives significantly advance Geographic Information Science (GIScience) by fostering greater adaptability, reproducibility, and community-driven innovation. Open frameworks empower researchers to tailor solutions, integrate cutting-edge methodologies (e.g., reinforcement learning, advanced spatial indexing), and align with FAIR principles. However, the growing reliance on any LLM necessitates careful consideration of security vulnerabilities, ethical risks, and robust governance for AI-generated geospatial outputs. Ongoing debates on accessibility, regulation, and misuse underscore the critical need for responsible AI development strategies. This paper argues that GIScience advances best not through a single model type, but by cultivating a diverse, interoperable ecosystem combining open-source foundations for innovation, bespoke geospatial models, and interdisciplinary collaboration. By critically evaluating the opportunities and challenges of open-source LLMs within the broader GeoAI landscape, this work contributes to a nuanced discourse on leveraging AI to effectively advance spatial research, policy, and decision-making in an equitable, sustainable, and scientifically rigorous manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18603v1">Toward Personalizing Quantum Computing Education: An Evolutionary LLM-Powered Approach</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Quantum computing education faces significant challenges due to its complexity and the limitations of current tools; this paper introduces a novel Intelligent Teaching Assistant for quantum computing education and details its evolutionary design process. The system combines a knowledge-graph-augmented architecture with two specialized Large Language Model (LLM) agents: a Teaching Agent for dynamic interaction, and a Lesson Planning Agent for lesson plan generation. The system is designed to adapt to individual student needs, with interactions meticulously tracked and stored in a knowledge graph. This graph represents student actions, learning resources, and relationships, aiming to enable reasoning about effective learning pathways. We describe the implementation of the system, highlighting the challenges encountered and the solutions implemented, including introducing a dual-agent architecture where tasks are separated, all coordinated through a central knowledge graph that maintains system awareness, and a user-facing tag system intended to mitigate LLM hallucination and improve user control. Preliminary results illustrate the system's potential to capture rich interaction data, dynamically adapt lesson plans based on student feedback via a tag system in simulation, and facilitate context-aware tutoring through the integrated knowledge graph, though systematic evaluation is required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18598v1">BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) have emerged as a powerful architecture for large language models (LLMs), enabling efficient scaling of model capacity while maintaining manageable computational costs. The key advantage lies in their ability to route different tokens to different ``expert'' networks within the model, enabling specialization and efficient handling of diverse input. However, the vulnerabilities of MoE-based LLMs still have barely been studied, and the potential for backdoor attacks in this context remains largely unexplored. This paper presents the first backdoor attack against MoE-based LLMs where the attackers poison ``dormant experts'' (i.e., underutilized experts) and activate them by optimizing routing triggers, thereby gaining control over the model's output. We first rigorously prove the existence of a few ``dominating experts'' in MoE models, whose outputs can determine the overall MoE's output. We also show that dormant experts can serve as dominating experts to manipulate model predictions. Accordingly, our attack, namely \textsc{BadMoE}, exploits the unique architecture of MoE models by 1) identifying dormant experts unrelated to the target task, 2) constructing a routing-aware loss to optimize the activation triggers of these experts, and 3) promoting dormant experts to dominating roles via poisoned training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20073v1">RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose StarPO (State-Thinking-Actions-Reward Policy Optimization), a general framework for trajectory-level agent RL, and introduce RAGEN, a modular system for training and evaluating LLM agents. Our study on three stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of Echo Trap where reward variance cliffs and gradient spikes; we address this with StarPO-S, a stabilized variant with trajectory filtering, critic incorporation, and decoupled clipping. Second, we find the shaping of RL rollouts would benefit from diverse initial states, medium interaction granularity and more frequent sampling. Third, we show that without fine-grained, reasoning-aware reward signals, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts. Code and environments are available at https://github.com/RAGEN-AI/RAGEN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20068v1">Tempo: Application-aware LLM Serving with Mixed SLO Requirements</a></div>
    <div class="paper-meta">
      📅 2025-04-24
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into diverse applications, ranging from interactive chatbots and cloud AIOps to intelligent agents, has introduced a wide spectrum of Service Level Objectives (SLOs) for responsiveness. These workloads include latency-sensitive requests focused on per-token latency in streaming chat, throughput-intensive requests that require rapid full responses to invoke tools, and collective requests with dynamic dependencies arising from self-reflection or agent-based reasoning. This workload diversity, amplified by unpredictable request information such as response lengths and runtime dependencies, makes existing schedulers inadequate even within their design envelopes. In this paper, we define service gain as the useful service delivered by completing requests. We observe that as SLO directly reflects the actual performance needs of requests, completing a request much faster than its SLO (e.g., deadline) yields limited additional service gain. Based on this insight, we introduce Tempo, the first systematic SLO-aware scheduler designed to maximize service gain across diverse LLM workloads. Tempo allocates just enough serving bandwidth to meet each SLO, maximizing residual capacity for others best-effort workloads. Instead of assuming request information or none at all, it adopts a hybrid scheduling strategy: using quantile-based response upper bounds and dependency-graph matching for conservative initial estimates, prioritizing requests by service gain density, and refining decisions online as generation progresses. Our evaluation across diverse workloads, including chat, reasoning, and agentic pipelines, shows that Tempo improves end-to-end service gain by up to 8.3$\times$ and achieves up to 10.3$\times$ SLO goodput compared to state-of-the-art designs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16921v1">IberBench: LLM Evaluation on Iberian Languages</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) remain difficult to evaluate comprehensively, particularly for languages other than English, where high-quality data is often limited. Existing benchmarks and leaderboards are predominantly English-centric, with only a few addressing other languages. These benchmarks fall short in several key areas: they overlook the diversity of language varieties, prioritize fundamental Natural Language Processing (NLP) capabilities over tasks of industrial relevance, and are static. With these aspects in mind, we present IberBench, a comprehensive and extensible benchmark designed to assess LLM performance on both fundamental and industry-relevant NLP tasks, in languages spoken across the Iberian Peninsula and Ibero-America. IberBench integrates 101 datasets from evaluation campaigns and recent benchmarks, covering 22 task categories such as sentiment and emotion analysis, toxicity detection, and summarization. The benchmark addresses key limitations in current evaluation practices, such as the lack of linguistic diversity and static evaluation setups by enabling continual updates and community-driven model and dataset submissions moderated by a committee of experts. We evaluate 23 LLMs ranging from 100 million to 14 billion parameters and provide empirical insights into their strengths and limitations. Our findings indicate that (i) LLMs perform worse on industry-relevant tasks than in fundamental ones, (ii) performance is on average lower for Galician and Basque, (iii) some tasks show results close to random, and (iv) in other tasks LLMs perform above random but below shared task systems. IberBench offers open-source implementations for the entire evaluation pipeline, including dataset normalization and hosting, incremental evaluation of LLMs, and a publicly accessible leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16918v1">OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Optimization plays a vital role in scientific research and practical applications, but formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise. We introduce \textbf{OptimAI}, a framework for solving \underline{Optim}ization problems described in natural language by leveraging LLM-powered \underline{AI} agents, achieving superior performance over current state-of-the-art methods. Our framework is built upon four key roles: (1) a \emph{formulator} that translates natural language problem descriptions into precise mathematical formulations; (2) a \emph{planner} that constructs a high-level solution strategy prior to execution; and (3) a \emph{coder} and a \emph{code critic} capable of interacting with the environment and reflecting on outcomes to refine future actions. Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively. Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain. Our design emphasizes multi-agent collaboration, allowing us to conveniently explore the synergistic effect of combining diverse models within a unified system. Our approach attains 88.1\% accuracy on the NLP4LP dataset and 71.2\% on the Optibench (non-linear w/o table) subset, reducing error rates by 58\% and 50\% respectively over prior best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16913v1">Tracing Thought: Using Chain-of-Thought Reasoning to Identify the LLM Behind AI-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 De-Factify 4: 4th Workshop on Multimodal Fact Checking and Hate Speech Detection, co-located with AAAI 2025. Pennsylvania
    </div>
    <details class="paper-abstract">
      In recent years, the detection of AI-generated text has become a critical area of research due to concerns about academic integrity, misinformation, and ethical AI deployment. This paper presents COT Fine-tuned, a novel framework for detecting AI-generated text and identifying the specific language model. responsible for generating the text. We propose a dual-task approach, where Task A involves classifying text as AI-generated or human-written, and Task B identifies the specific LLM behind the text. The key innovation of our method lies in the use of Chain-of-Thought reasoning, which enables the model to generate explanations for its predictions, enhancing transparency and interpretability. Our experiments demonstrate that COT Fine-tuned achieves high accuracy in both tasks, with strong performance in LLM identification and human-AI classification. We also show that the CoT reasoning process contributes significantly to the models effectiveness and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14985v2">aiXamine: Simplified LLM Safety and Security</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16871v1">Exploring How LLMs Capture and Represent Domain-Specific Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      We study whether Large Language Models (LLMs) inherently capture domain-specific nuances in natural language. Our experiments probe the domain sensitivity of LLMs by examining their ability to distinguish queries from different domains using hidden states generated during the prefill phase. We reveal latent domain-related trajectories that indicate the model's internal recognition of query domains. We also study the robustness of these domain representations to variations in prompt styles and sources. Our approach leverages these representations for model selection, mapping the LLM that best matches the domain trace of the input query (i.e., the model with the highest performance on similar traces). Our findings show that LLMs can differentiate queries for related domains, and that the fine-tuned model is not always the most accurate. Unlike previous work, our interpretations apply to both closed and open-ended generative tasks
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16046v2">Certified Mitigation of Worst-Case LLM Copyright Infringement</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      The exposure of large language models (LLMs) to copyrighted material during pre-training raises concerns about unintentional copyright infringement post deployment. This has driven the development of "copyright takedown" methods, post-training approaches aimed at preventing models from generating content substantially similar to copyrighted ones. While current mitigation approaches are somewhat effective for average-case risks, we demonstrate that they overlook worst-case copyright risks exhibits by the existence of long, verbatim quotes from copyrighted sources. We propose BloomScrub, a remarkably simple yet highly effective inference-time approach that provides certified copyright takedown. Our method repeatedly interleaves quote detection with rewriting techniques to transform potentially infringing segments. By leveraging efficient data sketches (Bloom filters), our approach enables scalable copyright screening even for large-scale real-world corpora. When quotes beyond a length threshold cannot be removed, the system can abstain from responding, offering certified risk reduction. Experimental results show that BloomScrub reduces infringement risk, preserves utility, and accommodates different levels of enforcement stringency with adaptive abstention. Our results suggest that lightweight, inference-time methods can be surprisingly effective for copyright prevention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00694v2">On Benchmarking Code LLMs for Android Malware Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 This paper has been accepted to the 34th ACM SIGSOFT ISSTA Companion (LLMSC Workshop 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in various code intelligence tasks. However, their effectiveness for Android malware analysis remains underexplored. Decompiled Android malware code presents unique challenges for analysis, due to the malicious logic being buried within a large number of functions and the frequent lack of meaningful function names. This paper presents CAMA, a benchmarking framework designed to systematically evaluate the effectiveness of Code LLMs in Android malware analysis. CAMA specifies structured model outputs to support key malware analysis tasks, including malicious function identification and malware purpose summarization. Built on these, it integrates three domain-specific evaluation metrics (consistency, fidelity, and semantic relevance), enabling rigorous stability and effectiveness assessment and cross-model comparison. We construct a benchmark dataset of 118 Android malware samples from 13 families collected in recent years, encompassing over 7.5 million distinct functions, and use CAMA to evaluate four popular open-source Code LLMs. Our experiments provide insights into how Code LLMs interpret decompiled code and quantify the sensitivity to function renaming, highlighting both their potential and current limitations in malware analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16833v1">LRASGen: LLM-based RESTful API Specification Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      REpresentation State Transfer (REST) is an architectural style for designing web applications that enable scalable, stateless communication between clients and servers via common HTTP techniques. Web APIs that employ the REST style are known as RESTful (or REST) APIs. When using or testing a RESTful API, developers may need to employ its specification, which is often defined by open-source standards such as the OpenAPI Specification (OAS). However, it can be very time-consuming and error-prone to write and update these specifications, which may negatively impact the use of RESTful APIs, especially when the software requirements change. Many tools and methods have been proposed to solve this problem, such as Respector and Swagger Core. OAS generation can be regarded as a common text-generation task that creates a formal description of API endpoints derived from the source code. A potential solution for this may involve using Large Language Models (LLMs), which have strong capabilities in both code understanding and text generation. Motivated by this, we propose a novel approach for generating the OASs of RESTful APIs using LLMs: LLM-based RESTful API-Specification Generation (LRASGen). To the best of our knowledge, this is the first use of LLMs and API source code to generate OASs for RESTful APIs. Compared with existing tools and methods, LRASGen can generate the OASs, even when the implementation is incomplete (with partial code, and/or missing annotations/comments, etc.). To evaluate the LRASGen performance, we conducted a series of empirical studies on 20 real-world RESTful APIs. The results show that two LLMs (GPT-4o mini and DeepSeek V3) can both support LARSGen to generate accurate specifications, and LRASGen-generated specifications cover an average of 48.85% more missed entities than the developer-provided specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16813v1">LLM-assisted Graph-RAG Information Extraction from IFC Data</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 2025 European Conference on Computing in Construction
    </div>
    <details class="paper-abstract">
      IFC data has become the general building information standard for collaborative work in the construction industry. However, IFC data can be very complicated because it allows for multiple ways to represent the same product information. In this research, we utilise the capabilities of LLMs to parse the IFC data with Graph Retrieval-Augmented Generation (Graph-RAG) technique to retrieve building object properties and their relations. We will show that, despite limitations due to the complex hierarchy of the IFC data, the Graph-RAG parsing enhances generative LLMs like GPT-4o with graph-based knowledge, enabling natural language query-response retrieval without the need for a complex pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05812v2">Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Ongoing work. First released on April 8, 2025. Updated the natural reasoning results on April 23, 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated exceptional capabilities in challenging tasks such as mathematical reasoning, existing methods to enhance reasoning ability predominantly rely on supervised fine-tuning (SFT) followed by reinforcement learning (RL) on reasoning-specific data after pre-training. However, these approaches critically depend on external supervision--such as human-labelled reasoning traces, verified golden answers, or pre-trained reward models--which limits scalability and practical applicability. In this work, we propose Entropy Minimized Policy Optimization (EMPO), which makes an early attempt at fully unsupervised LLM reasoning incentivization. EMPO does not require any supervised information for incentivizing reasoning capabilities (i.e., neither verifiable reasoning traces, problems with golden answers, nor additional pre-trained reward models). By continuously minimizing the predictive entropy of LLMs on unlabeled user queries in a latent semantic space, EMPO enables purely self-supervised evolution of reasoning capabilities with strong flexibility and practicality. Our experiments demonstrate competitive performance of EMPO on both mathematical reasoning and free-form natural reasoning tasks. Specifically, without any supervised signals, \ours boosts the accuracy of Qwen2.5-Math-7B Base from 30.7\% to 48.1\% on mathematical benchmarks and improves the accuracy of Qwen2.5-7B Base from 32.1\% to 50.1\% on MMLU-Pro.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v4">Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15965v2">From Human Memory to AI Memory: A Survey on Memory Mechanisms in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 26 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Memory is the process of encoding, storing, and retrieving information, allowing humans to retain experiences, knowledge, skills, and facts over time, and serving as the foundation for growth and effective interaction with the world. It plays a crucial role in shaping our identity, making decisions, learning from past experiences, building relationships, and adapting to changes. In the era of large language models (LLMs), memory refers to the ability of an AI system to retain, recall, and use information from past interactions to improve future responses and interactions. Although previous research and reviews have provided detailed descriptions of memory mechanisms, there is still a lack of a systematic review that summarizes and analyzes the relationship between the memory of LLM-driven AI systems and human memory, as well as how we can be inspired by human memory to construct more powerful memory systems. To achieve this, in this paper, we propose a comprehensive survey on the memory of LLM-driven AI systems. In particular, we first conduct a detailed analysis of the categories of human memory and relate them to the memory of AI systems. Second, we systematically organize existing memory-related work and propose a categorization method based on three dimensions (object, form, and time) and eight quadrants. Finally, we illustrate some open problems regarding the memory of current AI systems and outline possible future directions for memory in the era of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00513v2">Leveraging LLMs for User Stories in AI Systems: UStAI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11094v2">Reinforcement Learning With LLMs Interaction For Distributed Diffusion Model Services</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Distributed Artificial Intelligence-Generated Content (AIGC) has attracted significant attention, but two key challenges remain: maximizing subjective Quality of Experience (QoE) and improving energy efficiency, which are particularly pronounced in widely adopted Generative Diffusion Model (GDM)-based image generation services. In this paper, we propose a novel user-centric Interactive AI (IAI) approach for service management, with a distributed GDM-based AIGC framework that emphasizes efficient and cooperative deployment. The proposed method restructures the GDM inference process by allowing users with semantically similar prompts to share parts of the denoising chain. Furthermore, to maximize the users' subjective QoE, we propose an IAI approach, i.e., Reinforcement Learning With Large Language Models Interaction (RLLI), which utilizes Large Language Model (LLM)-empowered generative agents to replicate user interaction, providing real-time and subjective QoE feedback aligned with diverse user personalities. Lastly, we present the GDM-based Deep Deterministic Policy Gradient (GDDPG) algorithm, adapted to the proposed RLLI framework, to allocate communication and computing resources effectively while accounting for subjective user traits and dynamic wireless conditions. Simulation results demonstrate that G-DDPG improves total QoE by 15% compared with the standard DDPG algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16563v1">Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Intelligent agent systems based on Large Language Models (LLMs) have shown great potential in real-world applications. However, existing agent frameworks still face critical limitations in task planning and execution, restricting their effectiveness and generalizability. Specifically, current planning methods often lack clear global goals, leading agents to get stuck in local branches, or produce non-executable plans. Meanwhile, existing execution mechanisms struggle to balance complexity and stability, and their limited action space restricts their ability to handle diverse real-world tasks. To address these limitations, we propose GoalAct, a novel agent framework that introduces a continuously updated global planning mechanism and integrates a hierarchical execution strategy. GoalAct decomposes task execution into high-level skills, including searching, coding, writing and more, thereby reducing planning complexity while enhancing the agents' adaptability across diverse task scenarios. We evaluate GoalAct on LegalAgentBench, a benchmark with multiple types of legal tasks that require the use of multiple types of tools. Experimental results demonstrate that GoalAct achieves state-of-the-art (SOTA) performance, with an average improvement of 12.22% in success rate. These findings highlight GoalAct's potential to drive the development of more advanced intelligent agent systems, making them more effective across complex real-world applications. Our code can be found at https://github.com/cjj826/GoalAct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16511v1">QuaDMix: Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Quality and diversity are two critical metrics for the training data of large language models (LLMs), positively impacting performance. Existing studies often optimize these metrics separately, typically by first applying quality filtering and then adjusting data proportions. However, these approaches overlook the inherent trade-off between quality and diversity, necessitating their joint consideration. Given a fixed training quota, it is essential to evaluate both the quality of each data point and its complementary effect on the overall dataset. In this paper, we introduce a unified data selection framework called QuaDMix, which automatically optimizes the data distribution for LLM pretraining while balancing both quality and diversity. Specifically, we first propose multiple criteria to measure data quality and employ domain classification to distinguish data points, thereby measuring overall diversity. QuaDMix then employs a unified parameterized data sampling function that determines the sampling probability of each data point based on these quality and diversity related labels. To accelerate the search for the optimal parameters involved in the QuaDMix framework, we conduct simulated experiments on smaller models and use LightGBM for parameters searching, inspired by the RegMix method. Our experiments across diverse models and datasets demonstrate that QuaDMix achieves an average performance improvement of 7.2% across multiple benchmarks. These results outperform the independent strategies for quality and diversity, highlighting the necessity and ability to balance data quality and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16504v1">Intelligent Depression Prevention via LLM-Based Dialogue Analysis: Overcoming the Limitations of Scale-Dependent Diagnosis through Precise Emotional Pattern Recognition</a></div>
    <div class="paper-meta">
      📅 2025-04-23
    </div>
    <details class="paper-abstract">
      Existing depression screening predominantly relies on standardized questionnaires (e.g., PHQ-9, BDI), which suffer from high misdiagnosis rates (18-34% in clinical studies) due to their static, symptom-counting nature and susceptibility to patient recall bias. This paper presents an AI-powered depression prevention system that leverages large language models (LLMs) to analyze real-time conversational cues--including subtle emotional expressions (e.g., micro-sentiment shifts, self-referential language patterns)--for more accurate and dynamic mental state assessment. Our system achieves three key innovations: (1) Continuous monitoring through natural dialogue, detecting depression-indicative linguistic features (anhedonia markers, hopelessness semantics) with 89% precision (vs. 72% for PHQ-9); (2) Adaptive risk stratification that updates severity levels based on conversational context, reducing false positives by 41% compared to scale-based thresholds; and (3) Personalized intervention strategies tailored to users' emotional granularity, demonstrating 2.3x higher adherence rates than generic advice. Clinical validation with 450 participants shows the system identifies 92% of at-risk cases missed by traditional scales, while its explainable AI interface bridges the gap between automated analysis and clinician judgment. This work establishes conversational AI as a paradigm shift from episodic scale-dependent diagnosis to continuous, emotionally intelligent mental health monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16489v1">Amplified Vulnerabilities: Structured Jailbreak Attacks on LLM-based Multi-Agent Debate</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 33 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Multi-Agent Debate (MAD), leveraging collaborative interactions among Large Language Models (LLMs), aim to enhance reasoning capabilities in complex tasks. However, the security implications of their iterative dialogues and role-playing characteristics, particularly susceptibility to jailbreak attacks eliciting harmful content, remain critically underexplored. This paper systematically investigates the jailbreak vulnerabilities of four prominent MAD frameworks built upon leading commercial LLMs (GPT-4o, GPT-4, GPT-3.5-turbo, and DeepSeek) without compromising internal agents. We introduce a novel structured prompt-rewriting framework specifically designed to exploit MAD dynamics via narrative encapsulation, role-driven escalation, iterative refinement, and rhetorical obfuscation. Our extensive experiments demonstrate that MAD systems are inherently more vulnerable than single-agent setups. Crucially, our proposed attack methodology significantly amplifies this fragility, increasing average harmfulness from 28.14% to 80.34% and achieving attack success rates as high as 80% in certain scenarios. These findings reveal intrinsic vulnerabilities in MAD architectures and underscore the urgent need for robust, specialized defenses prior to real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05000v2">Needle Threading: Can LLMs Follow Threads through Near-Million-Scale Haystacks?</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      As the context limits of Large Language Models (LLMs) increase, the range of possible applications and downstream functions broadens. In many real-world tasks, decisions depend on details scattered across collections of often disparate documents containing mostly irrelevant information. Long-context LLMs appear well-suited to this form of complex information retrieval and reasoning, which has traditionally proven costly and time-consuming. However, although the development of longer context models has seen rapid gains in recent years, our understanding of how effectively LLMs use their context has not kept pace. To address this, we conduct a set of retrieval experiments designed to evaluate the capabilities of 17 leading LLMs, such as their ability to follow threads of information through the context window. Strikingly, we find that many models are remarkably threadsafe: capable of simultaneously following multiple threads without significant loss in performance. Still, for many models, we find the effective context limit is significantly shorter than the supported context length, with accuracy decreasing as the context window grows. Our study also highlights the important point that token counts from different tokenizers should not be directly compared -- they often correspond to substantially different numbers of written characters. We release our code and long-context experimental data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16472v1">Harden and Catch for Just-in-Time Assured LLM-Based Software Testing: Open Research Challenges</a></div>
    <div class="paper-meta">
      📅 2025-04-23
      | 💬 To Appear as keynote paper at FSE 2025
    </div>
    <details class="paper-abstract">
      Despite decades of research and practice in automated software testing, several fundamental concepts remain ill-defined and under-explored, yet offer enormous potential real-world impact. We show that these concepts raise exciting new challenges in the context of Large Language Models for software test generation. More specifically, we formally define and investigate the properties of hardening and catching tests. A hardening test is one that seeks to protect against future regressions, while a catching test is one that catches such a regression or a fault in new functionality introduced by a code change. Hardening tests can be generated at any time and may become catching tests when a future regression is caught. We also define and motivate the Catching `Just-in-Time' (JiTTest) Challenge, in which tests are generated `just-in-time' to catch new faults before they land into production. We show that any solution to Catching JiTTest generation can also be repurposed to catch latent faults in legacy code. We enumerate possible outcomes for hardening and catching tests and JiTTests, and discuss open research problems, deployment options, and initial results from our work on automated LLM-based hardening at Meta. This paper\footnote{Author order is alphabetical. The corresponding author is Mark Harman.} was written to accompany the keynote by the authors at the ACM International Conference on the Foundations of Software Engineering (FSE) 2025.
    </details>
</div>
