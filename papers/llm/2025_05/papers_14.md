# llm - 2025_05

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14401v2">LLM-Driven Usefulness Judgment for Web Search Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      Evaluation is fundamental in optimizing search experiences and supporting diverse user intents in Information Retrieval (IR). Traditional search evaluation methods primarily rely on relevance labels, which assess how well retrieved documents match a user's query. However, relevance alone fails to capture a search system's effectiveness in helping users achieve their search goals, making usefulness a critical evaluation criterion. In this paper, we explore an alternative approach: LLM-generated usefulness labels, which incorporate both implicit and explicit user behavior signals to evaluate document usefulness. We propose Task-aware Rubric-based Usefulness Evaluation (TRUE), a rubric-driven evaluation method that employs iterative sampling and reasoning to model complex search behavior patterns. Our findings show that (i) LLMs can generate moderate usefulness labels by leveraging comprehensive search session history incorporating personalization and contextual understanding, and (ii) fine-tuned LLMs improve usefulness judgments when provided with structured search session contexts. Additionally, we examine whether LLMs can distinguish between relevance and usefulness, particularly in cases where this divergence impacts search success. We also conduct an ablation study to identify key metrics for accurate usefulness label generation, optimizing for token efficiency and cost-effectiveness in real-world applications. This study advances LLM-based usefulness evaluation by refining key user metrics, exploring LLM-generated label reliability, and ensuring feasibility for large-scale search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06877v3">LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help?</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 11 pages. Accepted at SIGIR 2025 (48th International ACM SIGIR Conference on Research and Development in Information Retrieval)
    </div>
    <details class="paper-abstract">
      Test collections are information retrieval tools that allow researchers to quickly and easily evaluate ranking algorithms. While test collections have become an integral part of IR research, the process of data creation involves significant effort in manual annotations, which often makes it very expensive and time-consuming. Thus, test collections could become too small when the budget is limited, which may lead to unstable evaluations. As a cheaper alternative, recent studies have proposed the use of large language models (LLMs) to completely replace human assessors. However, while LLMs seem to somewhat correlate with human judgments, their predictions are not perfect and often show bias. Thus a complete replacement with LLMs is argued to be too risky and not fully reliable. Thus, in this paper, we propose LLM-Assisted Relevance Assessments (LARA), an effective method to balance manual annotations with LLM annotations, which helps to build a rich and reliable test collection even under a low budget. We use the LLM's predicted relevance probabilities to select the most profitable documents to manually annotate under a budget constraint. With theoretical reasoning, LARA effectively guides the human annotation process by actively learning to calibrate the LLM's predicted relevance probabilities. Then, using the calibration model learned from the limited manual annotations, LARA debiases the LLM predictions to annotate the remaining non-assessed data. Empirical evaluations on TREC-7 Ad Hoc, TREC-8 Ad Hoc, TREC Robust 2004, and TREC-COVID datasets show that LARA outperforms alternative solutions under almost any budget constraint.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01259v2">Facilitating Instructors-LLM Collaboration for Problem Design in Introductory Programming Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted at CHI 2025 Workshop on Augmented Educators and AI: Shaping the Future of Human and AI Cooperation in Learning
    </div>
    <details class="paper-abstract">
      Advancements in Large Language Models (LLMs), such as ChatGPT, offer significant opportunities to enhance instructional support in introductory programming courses. While extensive research has explored the effectiveness of LLMs in supporting student learning, limited studies have examined how these models can assist instructors in designing instructional activities. This work investigates how instructors' expertise in effective activity design can be integrated with LLMs' ability to generate novel and targeted programming problems, facilitating more effective activity creation for programming classrooms. To achieve this, we employ a participatory design approach to develop an instructor-authoring tool that incorporates LLM support, fostering collaboration between instructors and AI in generating programming exercises. This tool also allows instructors to specify common student mistakes and misconceptions, which informs the adaptive feedback generation process. We conduct case studies with three instructors, analyzing how they use our system to design programming problems for their introductory courses. Through these case studies, we assess instructors' perceptions of the usefulness and limitations of LLMs in authoring problem statements for instructional purposes. Additionally, we compare the efficiency, quality, effectiveness, and coverage of designed activities when instructors create problems with and without structured LLM prompting guidelines. Our findings provide insights into the potential of LLMs in enhancing instructor workflows and improving programming education and provide guidelines for designing effective AI-assisted problem-authoring interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09798v3">ReadMe.LLM: A Framework to Help LLMs Understand Your Library</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 15 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with code generation tasks involving niche software libraries. Existing code generation techniques with only human-oriented documentation can fail -- even when the LLM has access to web search and the library is documented online. To address this challenge, we propose ReadMe$.$LLM, LLM-oriented documentation for software libraries. By attaching the contents of ReadMe$.$LLM to a query, performance consistently improves to near-perfect accuracy, with one case study demonstrating up to 100% success across all tested models. We propose a software development lifecycle where LLM-specific documentation is maintained alongside traditional software updates. In this study, we present two practical applications of the ReadMe$.$LLM idea with diverse software libraries, highlighting that our proposed approach could generalize across programming domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02309v2">Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-08
      | 💬 Accepted to IEEE COMPSAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. This survey paper provides a comprehensive overview of techniques for compressing LLMs to enable efficient inference in resource-constrained environments. We examine three primary approaches: Knowledge Distillation, Model Quantization, and Model Pruning. For each technique, we discuss the underlying principles, present different variants, and provide examples of successful applications. We also briefly discuss complementary techniques such as mixture-of-experts and early-exit strategies. Finally, we highlight promising future directions, aiming to provide a valuable resource for both researchers and practitioners seeking to optimize LLMs for edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02665v2">A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law</a></div>
    <div class="paper-meta">
      📅 2025-05-08
    </div>
    <details class="paper-abstract">
      This survey explores recent advancements in reasoning large language models (LLMs) designed to mimic "slow thinking" - a reasoning process inspired by human cognition, as described in Kahneman's Thinking, Fast and Slow. These models, like OpenAI's o1, focus on scaling computational resources dynamically during complex tasks, such as math reasoning, visual reasoning, medical diagnosis, and multi-agent debates. We present the development of reasoning LLMs and list their key technologies. By synthesizing over 100 studies, it charts a path toward LLMs that combine human-like deep thinking with scalable efficiency for reasoning. The review breaks down methods into three categories: (1) test-time scaling dynamically adjusts computation based on task complexity via search and sampling, dynamic verification; (2) reinforced learning refines decision-making through iterative improvement leveraging policy networks, reward models, and self-evolution strategies; and (3) slow-thinking frameworks (e.g., long CoT, hierarchical processes) that structure problem-solving with manageable steps. The survey highlights the challenges and further directions of this domain. Understanding and advancing the reasoning abilities of LLMs is crucial for unlocking their full potential in real-world applications, from scientific discovery to decision support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04670v1">LLM Code Customization with Visual Results: A Benchmark on TikZ</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      With the rise of AI-based code generation, customizing existing code out of natural language instructions to modify visual results -such as figures or images -has become possible, promising to reduce the need for deep programming expertise. However, even experienced developers can struggle with this task, as it requires identifying relevant code regions (feature location), generating valid code variants, and ensuring the modifications reliably align with user intent. In this paper, we introduce vTikZ, the first benchmark designed to evaluate the ability of Large Language Models (LLMs) to customize code while preserving coherent visual outcomes. Our benchmark consists of carefully curated vTikZ editing scenarios, parameterized ground truths, and a reviewing tool that leverages visual feedback to assess correctness. Empirical evaluation with stateof-the-art LLMs shows that existing solutions struggle to reliably modify code in alignment with visual intent, highlighting a gap in current AI-assisted code editing approaches. We argue that vTikZ opens new research directions for integrating LLMs with visual feedback mechanisms to improve code customization tasks in various domains beyond TikZ, including image processing, art creation, Web design, and 3D modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04660v1">AI-Generated Fall Data: Assessing LLMs and Diffusion Model for Wearable Fall Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Training fall detection systems is challenging due to the scarcity of real-world fall data, particularly from elderly individuals. To address this, we explore the potential of Large Language Models (LLMs) for generating synthetic fall data. This study evaluates text-to-motion (T2M, SATO, ParCo) and text-to-text models (GPT4o, GPT4, Gemini) in simulating realistic fall scenarios. We generate synthetic datasets and integrate them with four real-world baseline datasets to assess their impact on fall detection performance using a Long Short-Term Memory (LSTM) model. Additionally, we compare LLM-generated synthetic data with a diffusion-based method to evaluate their alignment with real accelerometer distributions. Results indicate that dataset characteristics significantly influence the effectiveness of synthetic data, with LLM-generated data performing best in low-frequency settings (e.g., 20Hz) while showing instability in high-frequency datasets (e.g., 200Hz). While text-to-motion models produce more realistic biomechanical data than text-to-text models, their impact on fall detection varies. Diffusion-based synthetic data demonstrates the closest alignment to real data but does not consistently enhance model performance. An ablation study further confirms that the effectiveness of synthetic data depends on sensor placement and fall representation. These findings provide insights into optimizing synthetic data generation for fall detection models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07852v1">Joint Detection of Fraud and Concept Drift inOnline Conversations with LLM-Assisted Judgment</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Detecting fake interactions in digital communication platforms remains a challenging and insufficiently addressed problem. These interactions may appear as harmless spam or escalate into sophisticated scam attempts, making it difficult to flag malicious intent early. Traditional detection methods often rely on static anomaly detection techniques that fail to adapt to dynamic conversational shifts. One key limitation is the misinterpretation of benign topic transitions referred to as concept drift as fraudulent behavior, leading to either false alarms or missed threats. We propose a two stage detection framework that first identifies suspicious conversations using a tailored ensemble classification model. To improve the reliability of detection, we incorporate a concept drift analysis step using a One Class Drift Detector (OCDD) to isolate conversational shifts within flagged dialogues. When drift is detected, a large language model (LLM) assesses whether the shift indicates fraudulent manipulation or a legitimate topic change. In cases where no drift is found, the behavior is inferred to be spam like. We validate our framework using a dataset of social engineering chat scenarios and demonstrate its practical advantages in improving both accuracy and interpretability for real time fraud detection. To contextualize the trade offs, we compare our modular approach against a Dual LLM baseline that performs detection and judgment using different language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04588v1">ZeroSearch: Incentivize the Search Capability of LLMs without Searching</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a reinforcement learning framework that incentivizes the search capabilities of LLMs without interacting with real search engines. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both relevant and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20984v2">ACE: A Security Architecture for LLM-Integrated App Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 21 pages, 13 figures; clarify relation to indirect prompt injection attacks
    </div>
    <details class="paper-abstract">
      LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution. In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that our system is secure against attacks from the INJECAGENT benchmark, a standard benchmark for control flow integrity in the face of indirect prompt injection attacks, and our newly introduced attacks. Our architecture represents a significant advancement towards hardening LLM-based systems containing system facilities of varying levels of trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03161v2">An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 Accepted by IEEE Communications Magazine
    </div>
    <details class="paper-abstract">
      Recently emerged 6G space-air-ground integrated networks (SAGINs), which integrate satellites, aerial networks, and terrestrial communications, offer ubiquitous coverage for various mobile applications. However, the highly dynamic, open, and heterogeneous nature of SAGINs poses severe security issues. Forming a defense line of SAGINs suffers from two preliminary challenges: 1) accurately understanding massive unstructured multi-dimensional threat information to generate defense strategies against various malicious attacks, 2) rapidly adapting to potential unknown threats to yield more effective security strategies. To tackle the above two challenges, we propose a novel security framework for SAGINs based on Large Language Models (LLMs), which consists of two key ingredients LLM-6GNG and 6G-INST. Our proposed LLM-6GNG leverages refined chain-of-thought (CoT) reasoning and dynamic multi-agent mechanisms to analyze massive unstructured multi-dimensional threat data and generate comprehensive security strategies, thus addressing the first challenge. Our proposed 6G-INST relies on a novel self-evolving method to automatically update LLM-6GNG, enabling it to accommodate unknown threats under dynamic communication environments, thereby addressing the second challenge. Additionally, we prototype the proposed framework with ns-3, OpenAirInterface (OAI), and software-defined radio (SDR). Experiments on three benchmarks demonstrate the effectiveness of our framework. The results show that our framework produces highly accurate security strategies that remain robust against a variety of unknown attacks. We will release our code to contribute to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04521v1">Comparative Analysis of Carbon Footprint in Manual vs. LLM-Assisted Code Development</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) have significantly transformed various domains, including software development. These models assist programmers in generating code, potentially increasing productivity and efficiency. However, the environmental impact of utilising these AI models is substantial, given their high energy consumption during both training and inference stages. This research aims to compare the energy consumption of manual software development versus an LLM-assisted approach, using Codeforces as a simulation platform for software development. The goal is to quantify the environmental impact and propose strategies for minimising the carbon footprint of using LLM in software development. Our results show that the LLM-assisted code generation leads on average to 32.72 higher carbon footprint than the manual one. Moreover, there is a significant correlation between task complexity and the difference in the carbon footprint of the two approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04487v1">A Design Space for the Critical Validation of LLM-Generated Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 To appear at the 16th International EuroVis Workshop on Visual Analytics (EuroVA'25)
    </div>
    <details class="paper-abstract">
      LLM-generated tabular data is creating new opportunities for data-driven applications in academia, business, and society. To leverage benefits like missing value imputation, labeling, and enrichment with context-aware attributes, LLM-generated data needs a critical validation process. The number of pioneering approaches is increasing fast, opening a promising validation space that, so far, remains unstructured. We present a design space for the critical validation of LLM-generated tabular data with two dimensions: First, the Analysis Granularity dimension: from within-attribute (single-item and multi-item) to across-attribute perspectives (1 x 1, 1 x m, and n x n). Second, the Data Source dimension: differentiating between LLM-generated values, ground truth values, explanations, and their combinations. We discuss analysis tasks for each dimension cross-cut, map 19 existing validation approaches, and discuss the characteristics of two approaches in detail, demonstrating descriptive power.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04480v1">TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Trajectory prediction is a crucial task in modeling human behavior, especially in fields as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy, while recently proposed deep learning approaches suffer from computational cost, lack of explainability, and generalization issues that limit their practical adoption. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We introduce a Cross-Generation Elite Sampling to promote population diversity and a Statistics Feedback Loop allowing the LLM to analyze alternative predictions. Our evaluations show TrajEvo outperforms previous heuristic methods on the ETH-UCY datasets, and remarkably outperforms both heuristics and deep learning methods when generalizing to the unseen SDD dataset. TrajEvo represents a first step toward automated design of fast, explainable, and generalizable trajectory prediction heuristics. We make our source code publicly available to foster future research at https://github.com/ai4co/trajevo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04441v1">Towards Effectively Leveraging Execution Traces for Program Repair with Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promising performance on various programming tasks, including Automatic Program Repair (APR). However, most approaches to LLM-based APR are limited to the static analysis of the programs, while disregarding their runtime behavior. Inspired by knowledge-augmented NLP, in this work, we aim to remedy this potential blind spot by augmenting standard APR prompts with program execution traces. We evaluate our approach using the GPT family of models on three popular APR datasets. Our findings suggest that simply incorporating execution traces into the prompt provides a limited performance improvement over trace-free baselines, in only 2 out of 6 tested dataset / model configurations. We further find that the effectiveness of execution traces for APR diminishes as their complexity increases. We explore several strategies for leveraging traces in prompts and demonstrate that LLM-optimized prompts help outperform trace-free prompts more consistently. Additionally, we show trace-based prompting to be superior to finetuning a smaller LLM on a small-scale dataset; and conduct probing studies reinforcing the notion that execution traces can complement the reasoning abilities of the LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v4">Estimating LLM Uncertainty with Logits</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 Fixed some data errors in Table 1
    </div>
    <details class="paper-abstract">
      Over the past few years, Large Language Models (LLMs) have developed rapidly and are widely applied in various domains. However, LLMs face the issue of hallucinations, generating responses that may be unreliable when the models lack relevant knowledge. To be aware of potential hallucinations, uncertainty estimation methods have been introduced, and most of them have confirmed that reliability lies in critical tokens. However, probability-based methods perform poorly in identifying token reliability, limiting their practical utility. In this paper, we reveal that the probability-based method fails to estimate token reliability due to the loss of evidence strength information which is accumulated in the training stage. Therefore, we present Logits-induced token uncertainty (LogTokU), a framework for estimating decoupled token uncertainty in LLMs, enabling real-time uncertainty estimation without requiring multiple sampling processes. We employ evidence modeling to implement LogTokU and use the estimated uncertainty to guide downstream tasks. The experimental results demonstrate that LogTokU has significant effectiveness and promise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04388v1">The Aloe Family Recipe for Open and Specialized Healthcare LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 arXiv admin note: substantial text overlap with arXiv:2405.01886
    </div>
    <details class="paper-abstract">
      Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license. Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results. Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models. Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09410v3">LLM-based Bi-level Multi-interest Learning Framework for Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Sequential recommendation (SR) leverages users' dynamic preferences, with recent advances incorporating multi-interest learning to model diverse user interests. However, most multi-interest SR models rely on noisy, sparse implicit feedback, limiting recommendation accuracy. Large language models (LLMs) offer robust reasoning on low-quality data but face high computational costs and latency challenges for SR integration. We propose a novel LLM-based multi-interest SR framework combining implicit behavioral and explicit semantic perspectives. It includes two modules: the Implicit Behavioral Interest Module (IBIM), which learns from user behavior using a traditional SR model, and the Explicit Semantic Interest Module (ESIM), which uses clustering and prompt-engineered LLMs to extract semantic multi-interest representations from informative samples. Semantic insights from ESIM enhance IBIM's behavioral representations via modality alignment and semantic prediction tasks. During inference, only IBIM is used, ensuring efficient, LLM-free recommendations. Experiments on four real-world datasets validate the framework's effectiveness and practicality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04364v1">Benchmarking LLMs' Swarm intelligence</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict constraints-such as limited local perception and communication, characteristic of natural swarms-remains largely unexplored, particularly concerning the nuances of swarm intelligence. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination that arise when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks within a configurable 2D grid environment, forcing agents to rely primarily on local sensory input (k x k view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Evaluating several leading LLMs in a zero-shot setting, we find significant performance variations across tasks, highlighting the difficulties posed by local information constraints. While some coordination emerges, results indicate limitations in robust planning and strategy formation under uncertainty in these decentralized scenarios. Assessing LLMs under swarm-like conditions is crucial for realizing their potential in future decentralized systems. We release SwarmBench as an open, extensible toolkit-built upon a customizable and scalable physical system with defined mechanical properties. It provides environments, prompts, evaluation scripts, and the comprehensive experimental datasets generated, aiming to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of Embodied MAS. Our code repository is available at https://github.com/x66ccff/swarmbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18884v2">A Simple Ensemble Strategy for LLM Inference: Towards More Stable Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 This manuscript has been accepted for the 30th International Conference on Natural Language \& Information Systems (NLDB 2025) and will appear in Springer Lecture Notes in Computer Science (LNCS)
    </div>
    <details class="paper-abstract">
      With the advance of large language models (LLMs), LLMs have been utilized for the various tasks. However, the issues of variability and reproducibility of results from each trial of LLMs have been largely overlooked in existing literature while actual human annotation uses majority voting to resolve disagreements among annotators. Therefore, this study introduces the straightforward ensemble strategy to a sentiment analysis using LLMs. As the results, we demonstrate that the ensemble of multiple inference using medium-sized LLMs produces more robust and accurate results than using a large model with a single attempt with reducing RMSE by 18.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13184v3">Can LLM be a Good Path Planner based on Prompt Engineering? Mitigating the Hallucination for Path Planning</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 Accepted by ICIC 2025
    </div>
    <details class="paper-abstract">
      Spatial reasoning in Large Language Models (LLMs) is the foundation for embodied intelligence. However, even in simple maze environments, LLMs still encounter challenges in long-term path-planning, primarily influenced by their spatial hallucination and context inconsistency hallucination by long-term reasoning. To address this challenge, this study proposes an innovative model, Spatial-to-Relational Transformation and Curriculum Q-Learning (S2RCQL). To address the spatial hallucination of LLMs, we propose the Spatial-to-Relational approach, which transforms spatial prompts into entity relations and paths representing entity relation chains. This approach fully taps the potential of LLMs in terms of sequential thinking. As a result, we design a path-planning algorithm based on Q-learning to mitigate the context inconsistency hallucination, which enhances the reasoning ability of LLMs. Using the Q-value of state-action as auxiliary information for prompts, we correct the hallucinations of LLMs, thereby guiding LLMs to learn the optimal path. Finally, we propose a reverse curriculum learning technique based on LLMs to further mitigate the context inconsistency hallucination. LLMs can rapidly accumulate successful experiences by reducing task difficulty and leveraging them to tackle more complex tasks. We performed comprehensive experiments based on Baidu's self-developed LLM: ERNIE-Bot 4.0. The results showed that our S2RCQL achieved a 23%--40% improvement in both success and optimality rates compared with advanced prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04260v1">Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) improve in their capacity to serve as personal AI assistants, their ability to output uniquely tailored, personalized responses that align with the soft preferences of their users is essential for enhancing user satisfaction and retention. However, untrained lay users have poor prompt specification abilities and often struggle with conveying their latent preferences to AI assistants. To address this, we leverage activation steering to guide LLMs to align with interpretable preference dimensions during inference. In contrast to memory-based personalization methods that require longer user history, steering is extremely lightweight and can be easily controlled by the user via an linear strength factor. We embed steering into three different interactive chatbot interfaces and conduct a within-subjects user study (n=14) to investigate how end users prefer to personalize their conversations. The results demonstrate the effectiveness of preference-based steering for aligning real-world conversations with hidden user preferences, and highlight further insights on how diverse values around control, usability, and transparency lead users to prefer different interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04254v1">CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      With open-source projects growing in size and complexity, manual compilation becomes tedious and error-prone, highlighting the need for automation to improve efficiency and accuracy. However, the complexity of compilation instruction search and error resolution makes automatic compilation challenging. Inspired by the success of LLM-based agents in various fields, we propose CompileAgent, the first LLM-based agent framework dedicated to repo-level compilation. CompileAgent integrates five tools and a flow-based agent strategy, enabling interaction with software artifacts for compilation instruction search and error resolution. To measure the effectiveness of our method, we design a public repo-level benchmark CompileAgentBench, and we also design two baselines for comparison by combining two compilation-friendly schemes. The performance on this benchmark shows that our method significantly improves the compilation success rate, ranging from 10% to 71%. Meanwhile, we evaluate the performance of CompileAgent under different agent strategies and verify the effectiveness of the flow-based strategy. Additionally, we emphasize the scalability of CompileAgent, further expanding its application prospects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04253v1">LLM-Independent Adaptive RAG: Let the Question Speak for Itself</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 11 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models~(LLMs) are prone to hallucinations, and Retrieval-Augmented Generation (RAG) helps mitigate this, but at a high computational cost while risking misinformation. Adaptive retrieval aims to retrieve only when necessary, but existing approaches rely on LLM-based uncertainty estimation, which remain inefficient and impractical. In this study, we introduce lightweight LLM-independent adaptive retrieval methods based on external information. We investigated 27 features, organized into 7 groups, and their hybrid combinations. We evaluated these methods on 6 QA datasets, assessing the QA performance and efficiency. The results show that our approach matches the performance of complex LLM-based methods while achieving significant efficiency gains, demonstrating the potential of external information for adaptive retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04251v1">Facilitating Trustworthy Human-Agent Collaboration in LLM-based Multi-Agent System oriented Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Multi-agent autonomous systems (MAS) are better at addressing challenges that spans across multiple domains than singular autonomous agents. This holds true within the field of software engineering (SE) as well. The state-of-the-art research on MAS within SE focuses on integrating LLMs at the core of autonomous agents to create LLM-based multi-agent autonomous (LMA) systems. However, the introduction of LMA systems into SE brings a plethora of challenges. One of the major challenges is the strategic allocation of tasks between humans and the LMA system in a trustworthy manner. To address this challenge, a RACI-based framework is proposed in this work in progress article, along with implementation guidelines and an example implementation of the framework. The proposed framework can facilitate efficient collaboration, ensure accountability, and mitigate potential risks associated with LLM-driven automation while aligning with the Trustworthy AI guidelines. The future steps for this work delineating the planned empirical validation method are also presented.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04209v1">To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v3">InfiniteHBD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 takes a middle-ground approach by leveraging Optical Circuit Switches, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfiniteHBD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfiniteHBD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt into variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh) based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfiniteHBD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04174v1">On-Device LLM for Context-Aware Wi-Fi Roaming</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Wireless roaming is a critical yet challenging task for maintaining seamless connectivity in dynamic mobile environments. Conventional threshold-based or heuristic schemes often fail, leading to either sticky or excessive handovers. We introduce the first cross-layer use of an on-device large language model (LLM): high-level reasoning in the application layer that issues real-time actions executed in the PHY/MAC stack. The LLM addresses two tasks: (i) context-aware AP selection, where structured prompts fuse environmental cues (e.g., location, time) to choose the best BSSID; and (ii) dynamic threshold adjustment, where the model adaptively decides when to roam. To satisfy the tight latency and resource budgets of edge hardware, we apply a suite of optimizations-chain-of-thought prompting, parameter-efficient fine-tuning, and quantization. Experiments on indoor and outdoor datasets show that our approach surpasses legacy heuristics and DRL baselines, achieving a strong balance between roaming stability and signal quality. These findings underscore the promise of application-layer LLM reasoning for lower-layer wireless control in future edge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04146v1">Unmasking the Canvas: A Dynamic Benchmark for Image Generation Jailbreaking and LLM Content Safety</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Existing large language models (LLMs) are advancing rapidly and produce outstanding results in image generation tasks, yet their content safety checks remain vulnerable to prompt-based jailbreaks. Through preliminary testing on platforms such as ChatGPT, MetaAI, and Grok, we observed that even short, natural prompts could lead to the generation of compromising images ranging from realistic depictions of forged documents to manipulated images of public figures. We introduce Unmasking the Canvas (UTC Benchmark; UTCB), a dynamic and scalable benchmark dataset to evaluate LLM vulnerability in image generation. Our methodology combines structured prompt engineering, multilingual obfuscation (e.g., Zulu, Gaelic, Base64), and evaluation using Groq-hosted LLaMA-3. The pipeline supports both zero-shot and fallback prompting strategies, risk scoring, and automated tagging. All generations are stored with rich metadata and curated into Bronze (non-verified), Silver (LLM-aided verification), and Gold (manually verified) tiers. UTCB is designed to evolve over time with new data sources, prompt templates, and model behaviors. Warning: This paper includes visual examples of adversarial inputs designed to test model safety. All outputs have been redacted to ensure responsible disclosure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02116v3">Advancements and limitations of LLMs in replicating human color-word associations</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 20 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Color-word associations play a fundamental role in human cognition and design applications. Large Language Models (LLMs) have become widely available and have demonstrated intelligent behaviors in various benchmarks with natural conversation skills. However, their ability to replicate human color-word associations remains understudied. We compared multiple generations of LLMs (from GPT-3 to GPT-4o) against human color-word associations using data collected from over 10,000 Japanese participants, involving 17 colors and 80 words (10 word from eight categories) in Japanese. Our findings reveal a clear progression in LLM performance across generations, with GPT-4o achieving the highest accuracy in predicting the best voted word for each color and category. However, the highest median performance was approximately 50% even for GPT-4o with visual inputs (chance level of 10%). Moreover, we found performance variations across word categories and colors: while LLMs tended to excel in categories such as Rhythm and Landscape, they struggled with categories such as Emotions. Interestingly, color discrimination ability estimated from our color-word association data showed high correlation with human color discrimination patterns, consistent with previous studies. Thus, despite reasonable alignment in basic color discrimination, humans and LLMs still diverge systematically in the words they assign to those colors. Our study highlights both the advancements in LLM capabilities and their persistent limitations, raising the possibility of systematic differences in semantic memory structures between humans and LLMs in representing color-word associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05040v3">SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 Our code, data, and model will be released at https://github.com/InternLM/SWE-Fixer
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency across a variety of complex tasks. One significant application of LLMs is in tackling software engineering challenges, particularly in resolving real-world tasks on GitHub by fixing code based on the issues reported by the users. However, many current approaches rely on proprietary LLMs, which limits reproducibility, accessibility, and transparency. The critical components of LLMs for addressing software engineering issues and how their capabilities can be effectively enhanced remain unclear. To address these challenges, we introduce SWE-Fixer, a novel open-source framework designed to effectively and efficiently resolve GitHub issues. SWE-Fixer comprises two essential modules: a code file retrieval module and a code editing module. The retrieval module employs BM25 along with a lightweight model to achieve coarse-to-fine file retrieval. Subsequently, the code editing module utilizes the other model to generate patches for the identified files. To mitigate the lack of publicly available datasets, we compile an extensive dataset that includes 110K GitHub issues along with their corresponding patches and train the two models of SWE-Fixer separately. We assess our approach on the SWE-Bench Lite and Verified benchmarks, achieving competitive performance among open-source models with scores of 22.0% and 30.2%. Furthermore, SWE-Fixer reaches state-of-the-art performance (24.7% on Lite and 32.8% on Verified) with PASS_TO_PASS (P2P) filtering. Additionally, our approach requires only two model calls per instance, making it significantly more efficient than existing methods. These results highlight the effectiveness of SWE-Fixer in real-world code-fixing scenarios. We will make our model, dataset, and code publicly available at https://github.com/InternLM/SWE-Fixer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04101v1">LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) is expected to be an integral part of next-generation AI-native 6G networks. With the prevalence of AI, researchers have identified numerous use cases of AI in network security. However, there are almost nonexistent studies that analyze the suitability of Large Language Models (LLMs) in network security. To fill this gap, we examine the suitability of LLMs in network security, particularly with the case study of STRIDE threat modeling. We utilize four prompting techniques with five LLMs to perform STRIDE classification of 5G threats. From our evaluation results, we point out key findings and detailed insights along with the explanation of the possible underlying factors influencing the behavior of LLMs in the modeling of certain threats. The numerical results and the insights support the necessity for adjusting and fine-tuning LLMs for network security use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11026v2">SceneLLM: Implicit Language Reasoning in LLM for Dynamic Scene Graph Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 29 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Dynamic scenes contain intricate spatio-temporal information, crucial for mobile robots, UAVs, and autonomous driving systems to make informed decisions. Parsing these scenes into semantic triplets <Subject-Predicate-Object> for accurate Scene Graph Generation (SGG) is highly challenging due to the fluctuating spatio-temporal complexity. Inspired by the reasoning capabilities of Large Language Models (LLMs), we propose SceneLLM, a novel framework that leverages LLMs as powerful scene analyzers for dynamic SGG. Our framework introduces a Video-to-Language (V2L) mapping module that transforms video frames into linguistic signals (scene tokens), making the input more comprehensible for LLMs. To better encode spatial information, we devise a Spatial Information Aggregation (SIA) scheme, inspired by the structure of Chinese characters, which encodes spatial data into tokens. Using Optimal Transport (OT), we generate an implicit language signal from the frame-level token sequence that captures the video's spatio-temporal information. To further improve the LLM's ability to process this implicit linguistic input, we apply Low-Rank Adaptation (LoRA) to fine-tune the model. Finally, we use a transformer-based SGG predictor to decode the LLM's reasoning and predict semantic triplets. Our method achieves state-of-the-art results on the Action Genome (AG) benchmark, and extensive experiments show the effectiveness of SceneLLM in understanding and generating accurate dynamic scene graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04075v1">LLM-e Guess: Can LLMs Capabilities Advance Without Hardware Progress?</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      This paper examines whether large language model (LLM) capabilities can continue to advance without additional compute by analyzing the development and role of algorithms used in state-of-the-art LLMs. Motivated by regulatory efforts that have largely focused on restricting access to high-performance hardware, we ask: Can LLMs progress in a compute-constrained environment, and how do algorithmic innovations perform under such conditions? To address these questions, we introduce a novel classification framework that distinguishes between compute-dependent innovations -- which yield disproportionate benefits at high compute levels (e.g., the Transformer architecture and mixture-of-experts models) and compute-independent innovations, which improve efficiency across all compute scales (e.g., rotary positional encoding, FlashAttention, or layer normalization). We quantify these contributions using a metric called compute-equivalent gain (CEG), which estimates the additional compute that would be required to achieve similar improvements without these algorithmic advancements. To validate this framework, we conduct small-scale training experiments with a scaled-down GPT-2 model. Our results confirm that compute-independent advancements yield meaningful performance gains even in resource-constrained settings, with a CEG of up to $3.5\times$ over a baseline model. By contrast, compute-dependent advancements provided little benefit or even degraded performance at the small scale, reinforcing the importance of compute availability for certain algorithmic gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04072v1">Advancing and Benchmarking Personalized Tool Invocation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 14 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Tool invocation is a crucial mechanism for extending the capabilities of Large Language Models (LLMs) and has recently garnered significant attention. It enables LLMs to solve complex problems through tool calls while accessing up-to-date world knowledge. However, existing work primarily focuses on the fundamental ability of LLMs to invoke tools for problem-solving, without considering personalized constraints in tool invocation. In this work, we introduce the concept of Personalized Tool Invocation and define two key tasks: Tool Preference and Profile-dependent Query. Tool Preference addresses user preferences when selecting among functionally similar tools, while Profile-dependent Query considers cases where a user query lacks certain tool parameters, requiring the model to infer them from the user profile. To tackle these challenges, we propose PTool, a data synthesis framework designed for personalized tool invocation. Additionally, we construct \textbf{PTBench}, the first benchmark for evaluating personalized tool invocation. We then fine-tune various open-source models, demonstrating the effectiveness of our framework and providing valuable insights. Our benchmark is public at https://github.com/hyfshadow/PTBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00831v2">SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 Paper is under review
    </div>
    <details class="paper-abstract">
      Efficient path planning in robotics, particularly within large-scale, dynamic environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability in dynamic scenarios hinder real-time deployment on edge devices. We present SmallPlan -- a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like travel distance and number of trials. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04847v1">Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Hallucinations remain a persistent challenge for LLMs. RAG aims to reduce hallucinations by grounding responses in contexts. However, even when provided context, LLMs still frequently introduce unsupported information or contradictions. This paper presents our efforts to measure LLM hallucinations with a focus on summarization tasks, assessing how often various LLMs introduce hallucinations when summarizing documents. We discuss Vectara's existing LLM hallucination leaderboard, based on the Hughes Hallucination Evaluation Model (HHEM). While HHEM and Vectara's Hallucination Leaderboard have garnered great research interest, we examine challenges faced by HHEM and current hallucination detection methods by analyzing the effectiveness of these methods on existing hallucination datasets. To address these limitations, we propose FaithJudge, an LLM-as-a-judge approach guided by few-shot human hallucination annotations, which substantially improves automated LLM hallucination evaluation over current methods. We introduce an enhanced hallucination leaderboard centered on FaithJudge, alongside our current hallucination leaderboard, enabling more reliable benchmarking of LLMs for hallucinations in RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04842v1">Putting the Value Back in RL: Better Test-Time Scaling by Unifying LLM Reasoners With Verifiers</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Prevalent reinforcement learning~(RL) methods for fine-tuning LLM reasoners, such as GRPO or Leave-one-out PPO, abandon the learned value function in favor of empirically estimated returns. This hinders test-time compute scaling that relies on using the value-function for verification. In this work, we propose RL$^V$ that augments any ``value-free'' RL method by jointly training the LLM as both a reasoner and a generative verifier using RL-generated data, adding verification capabilities without significant overhead. Empirically, RL$^V$ boosts MATH accuracy by over 20\% with parallel sampling and enables $8-32\times$ efficient test-time compute scaling compared to the base RL method. RL$^V$ also exhibits strong generalization capabilities for both easy-to-hard and out-of-domain tasks. Furthermore, RL$^V$ achieves $1.2-1.6\times$ higher performance when jointly scaling parallel and sequential test-time compute with a long reasoning R1 model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04806v1">Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 7 Pages, 6 Figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04736v1">The Promise and Limits of LLMs in Constructing Proofs and Hints for Logic Problems in Intelligent Tutoring Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems have demonstrated effectiveness in teaching formal propositional logic proofs, but their reliance on template-based explanations limits their ability to provide personalized student feedback. While large language models (LLMs) offer promising capabilities for dynamic feedback generation, they risk producing hallucinations or pedagogically unsound explanations. We evaluated the stepwise accuracy of LLMs in constructing multi-step symbolic logic proofs, comparing six prompting techniques across four state-of-the-art LLMs on 358 propositional logic problems. Results show that DeepSeek-V3 achieved superior performance with 84.4% accuracy on stepwise proof construction and excelled particularly in simpler rules. We further used the best-performing LLM to generate explanatory hints for 1,050 unique student problem-solving states from a logic ITS and evaluated them on 4 criteria with both an LLM grader and human expert ratings on a 20% sample. Our analysis finds that LLM-generated hints were 75% accurate and rated highly by human evaluators on consistency and clarity, but did not perform as well explaining why the hint was provided or its larger context. Our results demonstrate that LLMs may be used to augment tutoring systems with logic tutoring hints, but requires additional modifications to ensure accuracy and pedagogical appropriateness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04732v1">QBD-RankedDataGen: Generating Custom Ranked Datasets for Improving Query-By-Document Search Using LLM-Reranking with Reduced Human Effort</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      The Query-By-Document (QBD) problem is an information retrieval problem where the query is a document, and the retrieved candidates are documents that match the query document, often in a domain or query specific manner. This can be crucial for tasks such as patent matching, legal or compliance case retrieval, and academic literature review. Existing retrieval methods, including keyword search and document embeddings, can be optimized with domain-specific datasets to improve QBD search performance. However, creating these domain-specific datasets is often costly and time-consuming. Our work introduces a process to generate custom QBD-search datasets and compares a set of methods to use in this problem, which we refer to as QBD-RankedDatagen. We provide a comparative analysis of our proposed methods in terms of cost, speed, and the human interface with the domain experts. The methods we compare leverage Large Language Models (LLMs) which can incorporate domain expert input to produce document scores and rankings, as well as explanations for human review. The process and methods for it that we present can significantly reduce human effort in dataset creation for custom domains while still obtaining sufficient expert knowledge for tuning retrieval models. We evaluate our methods on QBD datasets from the Text Retrieval Conference (TREC) and finetune the parameters of the BM25 model -- which is used in many industrial-strength search engines like OpenSearch -- using the generated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04723v1">SOAEsV2-7B/72B: Full-Pipeline Optimization for State-Owned Enterprise LLMs via Continual Pre-Training, Domain-Progressive SFT and Distillation-Enhanced Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      This study addresses key challenges in developing domain-specific large language models (LLMs) for Chinese state-owned assets and enterprises (SOAEs), where current approaches face three limitations: 1) constrained model capacity that limits knowledge integration and cross-task adaptability; 2) excessive reliance on domain-specific supervised fine-tuning (SFT) data, which neglects the broader applicability of general language patterns; and 3) inefficient inference acceleration for large models processing long contexts. In this work, we propose SOAEsV2-7B/72B, a specialized LLM series developed via a three-phase framework: 1) continual pre-training integrates domain knowledge while retaining base capabilities; 2) domain-progressive SFT employs curriculum-based learning strategy, transitioning from weakly relevant conversational data to expert-annotated SOAEs datasets to optimize domain-specific tasks; 3) distillation-enhanced speculative decoding accelerates inference via logit distillation between 72B target and 7B draft models, achieving 1.39-1.52$\times$ speedup without quality loss. Experimental results demonstrate that our domain-specific pre-training phase maintains 99.8% of original general language capabilities while significantly improving domain performance, resulting in a 1.08$\times$ improvement in Rouge-1 score and a 1.17$\times$ enhancement in BLEU-4 score. Ablation studies further show that domain-progressive SFT outperforms single-stage training, achieving 1.02$\times$ improvement in Rouge-1 and 1.06$\times$ in BLEU-4. Our work introduces a comprehensive, full-pipeline approach for optimizing SOAEs LLMs, bridging the gap between general language capabilities and domain-specific expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04710v1">Exploring Influence Factors on LLM Suitability for No-Code Development of End User IoT Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-07
    </div>
    <details class="paper-abstract">
      With the increasing popularity of IoT applications, end users demand more personalized and intuitive functionality. A major obstacle for this, however, is that custom IoT functionality today still requires at least some coding skills. To address this, no-code development platforms have been proposed as a solution for empowering non-technical users to create applications. However, such platforms still require a certain level of technical expertise for structuring process steps or defining event-action relations. The advent of LLMs can further enhance no-code platforms by enabling natural language-based interaction, automating of complex tasks, and dynamic code generation. By allowing users to describe their requirements in natural language, LLMs can significantly streamline no-code development. As LLMs vary in performance, architecture, training data used, and the use cases they target, it is still unclear which models are best suited and what are the influence factors determining this fit. In particular, no-code development of IoT applications by non-technical users will have completely different demands on LLMs than, e.g., code generation for more open-ended applications or for supporting professional developers. In this paper, we explore the factors influencing the suitability of LLMs to no-code development of IoT applications. We also examine the role of input prompt language on accuracy and quality of generated applications as well as the influence of LLM training data. By conducting comprehensive experiments with a range of LLMs, we provide valuable insights for optimizing LLM-powered no-code platforms, guiding the selection of the suitable LLMs and their effective application. Our findings contribute to improving the accessibility, efficiency, and user experience of no-code IoT development, ultimately enabling broader adoption of IoT technologies among non-expert users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04673v1">REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-07
      | 💬 13 pages (8 main), to be published in IJCAI 2025
    </div>
    <details class="paper-abstract">
      Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o. We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20953v2">Clean & Clear: Feasibility of Safe LLM Clinical Guidance</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Background: Clinical guidelines are central to safe evidence-based medicine in modern healthcare, providing diagnostic criteria, treatment options and monitoring advice for a wide range of illnesses. LLM-empowered chatbots have shown great promise in Healthcare Q&A tasks, offering the potential to provide quick and accurate responses to medical inquiries. Our main objective was the development and preliminary assessment of an LLM-empowered chatbot software capable of reliably answering clinical guideline questions using University College London Hospital (UCLH) clinical guidelines. Methods: We used the open-weight Llama-3.1-8B LLM to extract relevant information from the UCLH guidelines to answer questions. Our approach highlights the safety and reliability of referencing information over its interpretation and response generation. Seven doctors from the ward assessed the chatbot's performance by comparing its answers to the gold standard. Results: Our chatbot demonstrates promising performance in terms of relevance, with ~73% of its responses rated as very relevant, showcasing a strong understanding of the clinical context. Importantly, our chatbot achieves a recall of 1.00 for extracted guideline lines, substantially minimising the risk of missing critical information. Approximately 78% of responses were rated satisfactory in terms of completeness. A small portion (~14.5%) contained minor unnecessary information, indicating occasional lapses in precision. The chatbot' showed high efficiency, with an average completion time of 10 seconds, compared to 30 seconds for human respondents. Evaluation of clinical reasoning showed that 72% of the chatbot's responses were without flaws. Our chatbot demonstrates significant potential to speed up and improve the process of accessing locally relevant clinical information for healthcare professionals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03678v1">Graph Drawing for LLMs: An Empirical Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Our work contributes to the fast-growing literature on the use of Large Language Models (LLMs) to perform graph-related tasks. In particular, we focus on usage scenarios that rely on the visual modality, feeding the model with a drawing of the graph under analysis. We investigate how the model's performance is affected by the chosen layout paradigm, the aesthetics of the drawing, and the prompting technique used for the queries. We formulate three corresponding research questions and present the results of a thorough experimental analysis. Our findings reveal that choosing the right layout paradigm and optimizing the readability of the input drawing from a human perspective can significantly improve the performance of the model on the given task. Moreover, selecting the most effective prompting technique is a challenging yet crucial task for achieving optimal performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15279v2">Don't Mesh with Me: Generating Constructive Solid Geometry Instead of Meshes by Fine-Tuning a Code-Generation LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 Accepted to the AI for Content Creation Workshop at CVPR 2025
    </div>
    <details class="paper-abstract">
      While recent advancements in machine learning, such as LLMs, are revolutionizing software development and creative industries, they have had minimal impact on engineers designing mechanical parts, which remains largely a manual process. Existing approaches to generating 3D geometry most commonly use meshes as a 3D representation. While meshes are suitable for assets in video games or animations, they lack sufficient precision and adaptability for mechanical engineering purposes. This paper introduces a novel approach for the generation of 3D geometry that generates surface-based Constructive Solid Geometry (CSG) by leveraging a code-generation LLM. First, we create a dataset of 3D mechanical parts represented as code scripts by converting Boundary Representation geometry (BREP) into CSG-based Python scripts. Second, we create annotations in natural language using GPT-4. The resulting dataset is used to fine-tune a code-generation LLM. The fine-tuned LLM can complete geometries based on positional input and natural language in a plausible way, demonstrating geometric understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18991v2">HAIR: Hardness-Aware Inverse Reinforcement Learning with Introspective Reasoning for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 The three authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) with human values remains critical yet hindered by four key challenges: (1) scarcity of balanced safety datasets, (2) alignment tax, (3) vulnerability to jailbreak attacks due to shallow alignment, and (4) inability to dynamically adapt rewards according to task difficulty. To address these limitations, we introduce HAIR (Hardness-Aware Inverse Reinforcement Learning with Introspective Reasoning), a novel alignment approach inspired by shadow models in membership inference attacks. Our approach consists of two main components: (1) construction of a balanced safety Chain-of-Draft (CoD) dataset for seven harmful categories using structured prompts that leverage the introspective reasoning capabilities of LLMs; and (2) training of category-specific reward models with Group Relative Policy Optimization (GRPO), dynamically tuning optimization to task difficulty at both the data and model levels. Comprehensive experiments across four harmlessness and four usefulness benchmarks demonstrate that HAIR achieves state-of-the-art performance, outperforming all baseline methods in safety while maintaining high levels of usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03531v1">Faster MoE LLM Inference for Extremely Large Models</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Sparse Mixture of Experts (MoE) large language models (LLMs) are gradually becoming the mainstream approach for ultra-large-scale models. Existing optimization efforts for MoE models have focused primarily on coarse-grained MoE architectures. With the emergence of DeepSeek Models, fine-grained MoE models are gaining popularity, yet research on them remains limited. Therefore, we want to discuss the efficiency dynamic under different service loads. Additionally, fine-grained models allow deployers to reduce the number of routed experts, both activated counts and total counts, raising the question of how this reduction affects the trade-off between MoE efficiency and performance. Our findings indicate that while deploying MoE models presents greater challenges, it also offers significant optimization opportunities. Reducing the number of activated experts can lead to substantial efficiency improvements in certain scenarios, with only minor performance degradation. Reducing the total number of experts provides limited efficiency gains but results in severe performance degradation. Our method can increase throughput by at least 10\% without any performance degradation. Overall, we conclude that MoE inference optimization remains an area with substantial potential for exploration and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03475v1">am-ELO: A Stable Framework for Arena-based LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 ICML2025 Accepted
    </div>
    <details class="paper-abstract">
      Arena-based evaluation is a fundamental yet significant evaluation paradigm for modern AI models, especially large language models (LLMs). Existing framework based on ELO rating system suffers from the inevitable instability problem due to ranking inconsistency and the lack of attention to the varying abilities of annotators. In this paper, we introduce a novel stable arena framework to address these issues by enhancing the ELO Rating System. Specifically, we replace the iterative update method with a Maximum Likelihood Estimation (MLE) approach, m-ELO, and provide theoretical proof of the consistency and stability of the MLE approach for model ranking. Additionally, we proposed the am-ELO, which modify the Elo Rating's probability function to incorporate annotator abilities, enabling the simultaneous estimation of model scores and annotator reliability. Experiments demonstrate that this method ensures stability, proving that this framework offers a more robust, accurate, and stable evaluation method for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03473v1">Evaluation of LLMs on Long-tail Entity Linking in Historical Documents</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Entity Linking (EL) plays a crucial role in Natural Language Processing (NLP) applications, enabling the disambiguation of entity mentions by linking them to their corresponding entries in a reference knowledge base (KB). Thanks to their deep contextual understanding capabilities, LLMs offer a new perspective to tackle EL, promising better results than traditional methods. Despite the impressive generalization capabilities of LLMs, linking less popular, long-tail entities remains challenging as these entities are often underrepresented in training data and knowledge bases. Furthermore, the long-tail EL task is an understudied problem, and limited studies address it with LLMs. In the present work, we assess the performance of two popular LLMs, GPT and LLama3, in a long-tail entity linking scenario. Using MHERCL v0.1, a manually annotated benchmark of sentences from domain-specific historical texts, we quantitatively compare the performance of LLMs in identifying and linking entities to their corresponding Wikidata entries against that of ReLiK, a state-of-the-art Entity Linking and Relation Extraction framework. Our preliminary experiments reveal that LLMs perform encouragingly well in long-tail EL, indicating that this technology can be a valuable adjunct in filling the gap between head and long-tail EL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04430v3">The Struggles of LLMs in Cross-lingual Code Clone Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 Accepted for publication at the ACM International Conference on the Foundations of Software Engineering (FSE) 2025
    </div>
    <details class="paper-abstract">
      With the involvement of multiple programming languages in modern software development, cross-lingual code clone detection has gained traction within the software engineering community. Numerous studies have explored this topic, proposing various promising approaches. Inspired by the significant advances in machine learning in recent years, particularly Large Language Models (LLMs), which have demonstrated their ability to tackle various tasks, this paper revisits cross-lingual code clone detection. We evaluate the performance of five (05) LLMs and eight prompts (08) for the identification of cross-lingual code clones. Additionally, we compare these results against two baseline methods. Finally, we evaluate a pre-trained embedding model to assess the effectiveness of the generated representations for classifying clone and non-clone pairs. The studies involving LLMs and Embedding models are evaluated using two widely used cross-lingual datasets, XLCoST and CodeNet. Our results show that LLMs can achieve high F1 scores, up to 0.99, for straightforward programming examples. However, they not only perform less well on programs associated with complex programming challenges but also do not necessarily understand the meaning of "code clones" in a cross-lingual setting. We show that embedding models used to represent code fragments from different programming languages in the same representation space enable the training of a basic classifier that outperforms all LLMs by ~1 and ~20 percentage points on the XLCoST and CodeNet datasets, respectively. This finding suggests that, despite the apparent capabilities of LLMs, embeddings provided by embedding models offer suitable representations to achieve state-of-the-art performance in cross-lingual code clone detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03434v1">Procedural Memory Is Not All You Need: Bridging Cognitive Gaps in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 Accepted to the workshop on Hybrid AI for Human-Centric Personalization (HyPer), co-located with ACM UMAP '25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent a landmark achievement in Artificial Intelligence (AI), demonstrating unprecedented proficiency in procedural tasks such as text generation, code completion, and conversational coherence. These capabilities stem from their architecture, which mirrors human procedural memory -- the brain's ability to automate repetitive, pattern-driven tasks through practice. However, as LLMs are increasingly deployed in real-world applications, it becomes impossible to ignore their limitations operating in complex, unpredictable environments. This paper argues that LLMs, while transformative, are fundamentally constrained by their reliance on procedural memory. To create agents capable of navigating ``wicked'' learning environments -- where rules shift, feedback is ambiguous, and novelty is the norm -- we must augment LLMs with semantic memory and associative learning systems. By adopting a modular architecture that decouples these cognitive functions, we can bridge the gap between narrow procedural expertise and the adaptive intelligence required for real-world problem-solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03406v1">Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      This research paper investigates the application of Large Language Models (LLMs) in healthcare, specifically focusing on enhancing medical decision support through Retrieval-Augmented Generation (RAG) integrated with hospital-specific data and fine-tuning using Quantized Low-Rank Adaptation (QLoRA). The system utilizes Llama 3.2-3B-Instruct as its foundation model. By embedding and retrieving context-relevant healthcare information, the system significantly improves response accuracy. QLoRA facilitates notable parameter efficiency and memory optimization, preserving the integrity of medical information through specialized quantization techniques. Our research also shows that our model performs relatively well on various medical benchmarks, indicating that it can be used to make basic medical suggestions. This paper details the system's technical components, including its architecture, quantization methods, and key healthcare applications such as enhanced disease prediction from patient symptoms and medical history, treatment suggestions, and efficient summarization of complex medical reports. We touch on the ethical considerations-patient privacy, data security, and the need for rigorous clinical validation-as well as the practical challenges of integrating such systems into real-world healthcare workflows. Furthermore, the lightweight quantized weights ensure scalability and ease of deployment even in low-resource hospital environments. Finally, the paper concludes with an analysis of the broader impact of LLMs on healthcare and outlines future directions for LLMs in medical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03345v1">Elevating Cyber Threat Intelligence against Disinformation Campaigns with LLM-based Concept Extraction and the FakeCTI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      The swift spread of fake news and disinformation campaigns poses a significant threat to public trust, political stability, and cybersecurity. Traditional Cyber Threat Intelligence (CTI) approaches, which rely on low-level indicators such as domain names and social media handles, are easily evaded by adversaries who frequently modify their online infrastructure. To address these limitations, we introduce a novel CTI framework that focuses on high-level, semantic indicators derived from recurrent narratives and relationships of disinformation campaigns. Our approach extracts structured CTI indicators from unstructured disinformation content, capturing key entities and their contextual dependencies within fake news using Large Language Models (LLMs). We further introduce FakeCTI, the first dataset that systematically links fake news to disinformation campaigns and threat actors. To evaluate the effectiveness of our CTI framework, we analyze multiple fake news attribution techniques, spanning from traditional Natural Language Processing (NLP) to fine-tuned LLMs. This work shifts the focus from low-level artifacts to persistent conceptual structures, establishing a scalable and adaptive approach to tracking and countering disinformation campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03336v1">Avoid Recommending Out-of-Domain Items: Constrained Generative Recommendation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise for generative recommender systems due to their transformative capabilities in user interaction. However, ensuring they do not recommend out-of-domain (OOD) items remains a challenge. We study two distinct methods to address this issue: RecLM-ret, a retrieval-based method, and RecLM-cgen, a constrained generation method. Both methods integrate seamlessly with existing LLMs to ensure in-domain recommendations. Comprehensive experiments on three recommendation datasets demonstrate that RecLM-cgen consistently outperforms RecLM-ret and existing LLM-based recommender models in accuracy while eliminating OOD recommendations, making it the preferred method for adoption. Additionally, RecLM-cgen maintains strong generalist capabilities and is a lightweight plug-and-play module for easy integration into LLMs, offering valuable practical benefits for the community. Source code is available at https://github.com/microsoft/RecAI
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03293v1">Ψ-Arena: Interactive Assessment and Optimization of LLM-based Psychological Counselors with Tripartite Feedback</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in providing scalable mental health support, while evaluating their counseling capability remains crucial to ensure both efficacy and safety. Existing evaluations are limited by the static assessment that focuses on knowledge tests, the single perspective that centers on user experience, and the open-loop framework that lacks actionable feedback. To address these issues, we propose {\Psi}-Arena, an interactive framework for comprehensive assessment and optimization of LLM-based counselors, featuring three key characteristics: (1) Realistic arena interactions that simulate real-world counseling through multi-stage dialogues with psychologically profiled NPC clients, (2) Tripartite evaluation that integrates assessments from the client, counselor, and supervisor perspectives, and (3) Closed-loop optimization that iteratively improves LLM counselors using diagnostic feedback. Experiments across eight state-of-the-art LLMs show significant performance variations in different real-world scenarios and evaluation perspectives. Moreover, reflection-based optimization results in up to a 141% improvement in counseling performance. We hope PsychoArena provides a foundational resource for advancing reliable and human-aligned LLM applications in mental healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10913v3">ASHABot: An LLM-Powered Chatbot to Support the Informational Needs of Community Health Workers</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Community health workers (CHWs) provide last-mile healthcare services but face challenges due to limited medical knowledge and training. This paper describes the design, deployment, and evaluation of ASHABot, an LLM-powered, experts-in-the-loop, WhatsApp-based chatbot to address the information needs of CHWs in India. Through interviews with CHWs and their supervisors and log analysis, we examine factors affecting their engagement with ASHABot, and ASHABot's role in addressing CHWs' informational needs. We found that ASHABot provided a private channel for CHWs to ask rudimentary and sensitive questions they hesitated to ask supervisors. CHWs trusted the information they received on ASHABot and treated it as an authoritative resource. CHWs' supervisors expanded their knowledge by contributing answers to questions ASHABot failed to answer, but were concerned about demands on their workload and increased accountability. We emphasize positioning LLMs as supplemental fallible resources within the community healthcare ecosystem, instead of as replacements for supervisor support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03275v1">RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle to effectively utilize a growing number of external tools, such as those defined by the Model Context Protocol (MCP)\cite{IntroducingMCP}, due to prompt bloat and selection complexity. We introduce RAG-MCP, a Retrieval-Augmented Generation framework that overcomes this challenge by offloading tool discovery. RAG-MCP uses semantic retrieval to identify the most relevant MCP(s) for a given query from an external index before engaging the LLM. Only the selected tool descriptions are passed to the model, drastically reducing prompt size and simplifying decision-making. Experiments, including an MCP stress test, demonstrate RAG-MCP significantly cuts prompt tokens (e.g., by over 50%) and more than triples tool selection accuracy (43.13% vs 13.62% baseline) on benchmark tasks. RAG-MCP enables scalable and accurate tool integration for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02579v2">EMORL: Ensemble Multi-Objective Reinforcement Learning for Efficient and Flexible LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 13 pages, 9 figures, submitted to SIGDIAL 2025 conference
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) for large language model (LLM) fine-tuning show promise in addressing multi-objective tasks but still face significant challenges, including complex objective balancing, low training efficiency, poor scalability, and limited explainability. Leveraging ensemble learning principles, we introduce an Ensemble Multi-Objective RL (EMORL) framework that fine-tunes multiple models with individual objectives while optimizing their aggregation after the training to improve efficiency and flexibility. Our method is the first to aggregate the last hidden states of individual models, incorporating contextual information from multiple objectives. This approach is supported by a hierarchical grid search algorithm that identifies optimal weighted combinations. We evaluate EMORL on counselor reflection generation tasks, using text-scoring LLMs to evaluate the generations and provide rewards during RL fine-tuning. Through comprehensive experiments on the PAIR and Psych8k datasets, we demonstrate the advantages of EMORL against existing baselines: significantly lower and more stable training consumption ($17,529\pm 1,650$ data points and $6,573\pm 147.43$ seconds), improved scalability and explainability, and comparable performance across multiple objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03196v1">A Trustworthy Multi-LLM Network: Challenges,Solutions, and A Use Case</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong potential across a variety of tasks in communications and networking due to their advanced reasoning capabilities. However, because different LLMs have different model structures and are trained using distinct corpora and methods, they may offer varying optimization strategies for the same network issues. Moreover, the limitations of an individual LLM's training data, aggravated by the potential maliciousness of its hosting device, can result in responses with low confidence or even bias. To address these challenges, we propose a blockchain-enabled collaborative framework that connects multiple LLMs into a Trustworthy Multi-LLM Network (MultiLLMN). This architecture enables the cooperative evaluation and selection of the most reliable and high-quality responses to complex network optimization problems. Specifically, we begin by reviewing related work and highlighting the limitations of existing LLMs in collaboration and trust, emphasizing the need for trustworthiness in LLM-based systems. We then introduce the workflow and design of the proposed Trustworthy MultiLLMN framework. Given the severity of False Base Station (FBS) attacks in B5G and 6G communication systems and the difficulty of addressing such threats through traditional modeling techniques, we present FBS defense as a case study to empirically validate the effectiveness of our approach. Finally, we outline promising future research directions in this emerging area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03179v1">Bridging Expertise Gaps: The Role of LLMs in Human-AI Collaboration for Cybersecurity</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 20 pages, 10 figures, 2 tables, under review
    </div>
    <details class="paper-abstract">
      This study investigates whether large language models (LLMs) can function as intelligent collaborators to bridge expertise gaps in cybersecurity decision-making. We examine two representative tasks-phishing email detection and intrusion detection-that differ in data modality, cognitive complexity, and user familiarity. Through a controlled mixed-methods user study, n = 58 (phishing, n = 34; intrusion, n = 24), we find that human-AI collaboration improves task performance,reducing false positives in phishing detection and false negatives in intrusion detection. A learning effect is also observed when participants transition from collaboration to independent work, suggesting that LLMs can support long-term skill development. Our qualitative analysis shows that interaction dynamics-such as LLM definitiveness, explanation style, and tone-influence user trust, prompting strategies, and decision revision. Users engaged in more analytic questioning and showed greater reliance on LLM feedback in high-complexity settings. These results provide design guidance for building interpretable, adaptive, and trustworthy human-AI teaming systems, and demonstrate that LLMs can meaningfully support non-experts in reasoning through complex cybersecurity problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03171v1">CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Neurosymbolic approaches integrating large language models with formal reasoning have recently achieved human-level performance on mathematics competition problems in algebra, geometry and number theory. In comparison, combinatorics remains a challenging domain, characterized by a lack of appropriate benchmarks and theorem libraries. To address this gap, we introduce CombiBench, a comprehensive benchmark comprising 100 combinatorial problems, each formalized in Lean~4 and paired with its corresponding informal statement. The problem set covers a wide spectrum of difficulty levels, ranging from middle school to IMO and university level, and span over ten combinatorial topics. CombiBench is suitable for testing IMO solving capabilities since it includes all IMO combinatorial problems since 2000 (except IMO 2004 P3 as its statement contain an images). Furthermore, we provide a comprehensive and standardized evaluation framework, dubbed Fine-Eval (for $\textbf{F}$ill-in-the-blank $\textbf{in}$ L$\textbf{e}$an Evaluation), for formal mathematics. It accommodates not only proof-based problems but also, for the first time, the evaluation of fill-in-the-blank questions. Using Fine-Eval as the evaluation method and Kimina Lean Server as the backend, we benchmark several LLMs on CombiBench and observe that their capabilities for formally solving combinatorial problems remain limited. Among all models tested (none of which has been trained for this particular task), Kimina-Prover attains the best results, solving 7 problems (out of 100) under both ``with solution'' and ``without solution'' scenarios. We open source the benchmark dataset alongside with the code of the proposed evaluation method at https://github.com/MoonshotAI/CombiBench/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03161v1">An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Recently emerged 6G space-air-ground integrated networks (SAGINs), which integrate satellites, aerial networks, and terrestrial communications, offer ubiquitous coverage for various mobile applications. However, the highly dynamic, open, and heterogeneous nature of SAGINs poses severe security issues. Forming a defense line of SAGINs suffers from two preliminary challenges: 1) accurately understanding massive unstructured multi-dimensional threat information to generate defense strategies against various malicious attacks, 2) rapidly adapting to potential unknown threats to yield more effective security strategies. To tackle the above two challenges, we propose a novel security framework for SAGINs based on Large Language Models (LLMs), which consists of two key ingredients LLM-6GNG and 6G-INST. Our proposed LLM-6GNG leverages refined chain-of-thought (CoT) reasoning and dynamic multi-agent mechanisms to analyze massive unstructured multi-dimensional threat data and generate comprehensive security strategies, thus addressing the first challenge. Our proposed 6G-INST relies on a novel self-evolving method to automatically update LLM-6GNG, enabling it to accommodate unknown threats under dynamic communication environments, thereby addressing the second challenge. Additionally, we prototype the proposed framework with ns-3, OpenAirInterface (OAI), and software-defined radio (SDR). Experiments on three benchmarks demonstrate the effectiveness of our framework. The results show that our framework produces highly accurate security strategies that remain robust against a variety of unknown attacks. We will release our code to contribute to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08717v4">Social Opinions Prediction Utilizes Fusing Dynamics Equation with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      In the context where social media emerges as a pivotal platform for social movements and shaping public opinion, accurately simulating and predicting the dynamics of user opinions is of significant importance. Such insights are vital for understanding social phenomena, informing policy decisions, and guiding public opinion. Unfortunately, traditional algorithms based on idealized models and disregarding social data often fail to capture the complexity and nuance of real-world social interactions. This study proposes the Fusing Dynamics Equation-Large Language Model (FDE-LLM) algorithm. This innovative approach aligns the actions and evolution of opinions in Large Language Models (LLMs) with the real-world data on social networks. The FDE-LLM devides users into two roles: opinion leaders and followers. Opinion leaders use LLM for role-playing and employ Cellular Automata(CA) to constrain opinion changes. In contrast, opinion followers are integrated into a dynamic system that combines the CA model with the Susceptible-Infectious-Recovered (SIR) model. This innovative design significantly improves the accuracy of the simulation. Our experiments utilized four real-world datasets from Weibo. The result demonstrates that the FDE-LLM significantly outperforms traditional Agent-Based Modeling (ABM) algorithms and LLM-based algorithms. Additionally, our algorithm accurately simulates the decay and recovery of opinions over time, underscoring LLMs potential to revolutionize the understanding of social media dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03112v1">Plug-and-Play AMC: Context Is King in Training-Free, Open-Set Modulation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Automatic Modulation Classification (AMC) is critical for efficient spectrum management and robust wireless communications. However, AMC remains challenging due to the complex interplay of signal interference and noise. In this work, we propose an innovative framework that integrates traditional signal processing techniques with Large-Language Models (LLMs) to address AMC. Our approach leverages higher-order statistics and cumulant estimation to convert quantitative signal features into structured natural language prompts. By incorporating exemplar contexts into these prompts, our method exploits the LLM's inherent familiarity with classical signal processing, enabling effective one-shot classification without additional training or preprocessing (e.g., denoising). Experimental evaluations on synthetically generated datasets, spanning both noiseless and noisy conditions, demonstrate that our framework achieves competitive performance across diverse modulation schemes and Signal-to-Noise Ratios (SNRs). Moreover, our approach paves the way for robust foundation models in wireless communications across varying channel conditions, significantly reducing the expense associated with developing channel-specific models. This work lays the foundation for scalable, interpretable, and versatile signal classification systems in next-generation wireless networks. The source code is available at https://github.com/RU-SIT/context-is-king
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03100v1">Towards a standardized methodology and dataset for evaluating LLM-based digital forensic timeline analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have seen widespread adoption in many domains including digital forensics. While prior research has largely centered on case studies and examples demonstrating how LLMs can assist forensic investigations, deeper explorations remain limited, i.e., a standardized approach for precise performance evaluations is lacking. Inspired by the NIST Computer Forensic Tool Testing Program, this paper proposes a standardized methodology to quantitatively evaluate the application of LLMs for digital forensic tasks, specifically in timeline analysis. The paper describes the components of the methodology, including the dataset, timeline generation, and ground truth development. Additionally, the paper recommends using BLEU and ROUGE metrics for the quantitative evaluation of LLMs through case studies or tasks involving timeline analysis. Experimental results using ChatGPT demonstrate that the proposed methodology can effectively evaluate LLM-based forensic timeline analysis. Finally, we discuss the limitations of applying LLMs to forensic timeline analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03096v1">Assessing and Enhancing the Robustness of LLM-based Multi-Agent Systems Through Chaos Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      This study explores the application of chaos engineering to enhance the robustness of Large Language Model-Based Multi-Agent Systems (LLM-MAS) in production-like environments under real-world conditions. LLM-MAS can potentially improve a wide range of tasks, from answering questions and generating content to automating customer support and improving decision-making processes. However, LLM-MAS in production or preproduction environments can be vulnerable to emergent errors or disruptions, such as hallucinations, agent failures, and agent communication failures. This study proposes a chaos engineering framework to proactively identify such vulnerabilities in LLM-MAS, assess and build resilience against them, and ensure reliable performance in critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04249v3">DiffSpec: Differential Testing with LLMs using Natural Language Specifications and Code Artifacts</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Differential testing can be an effective way to find bugs in software systems with multiple implementations that conform to the same specification, like compilers, network protocol parsers, or language runtimes. Specifications for such systems are often standardized in natural language documents, like Instruction Set Architecture (ISA) specifications or IETF RFC's. Large Language Models (LLMs) have demonstrated potential in both generating tests and handling large volumes of natural language text, making them well-suited for analyzing artifacts like specification documents, bug reports, and code implementations. In this work, we leverage natural language and code artifacts to guide LLMs to generate targeted tests that highlight meaningful behavioral differences between implementations, including those corresponding to bugs. We introduce DiffSpec, a framework for generating differential tests with LLMs using prompt chaining. We demonstrate DiffSpec's efficacy on two different (extensively tested) systems, eBPF runtimes and Wasm validators. Using DiffSpec, we generated 1901 differentiating tests, uncovering at least four distinct and confirmed bugs in eBPF, including a kernel memory leak, inconsistent behavior in jump instructions, undefined behavior when using the stack pointer, and tests with infinite loops that hang the verifier in ebpf-for-windows. We also found 299 differentiating tests in Wasm validators pointing to two confirmed and fixed bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11863v2">Context-aware LLM-based Safe Control Against Latent Risks</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Autonomous control systems face significant challenges in performing complex tasks in the presence of latent risks. To address this, we propose an integrated framework that combines Large Language Models (LLMs), numerical optimization, and optimization-based control to facilitate efficient subtask learning while ensuring safety against latent risks. The framework decomposes complex tasks into a sequence of context-aware subtasks that account for latent risks. These subtasks and their parameters are then refined through a multi-time-scale process: high-layer multi-turn in-context learning, mid-layer LLM Chain-of-Thought reasoning and numerical optimization, and low-layer model predictive control. The framework iteratively improves decisions by leveraging qualitative feedback and optimized trajectory data from lower-layer optimization processes and a physics simulator. We validate the proposed framework through simulated case studies involving robot arm and autonomous vehicle scenarios. The experiments demonstrate that the proposed framework can mediate actions based on the context and latent risks and learn complex behaviors efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04021v1">Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) is expensive, especially for providers hosting many models, making cost reduction essential. The unique workload patterns of serving multiple LLMs (i.e., multi-LLM serving) create new opportunities and challenges for this task. The long-tail popularity of models and their long idle periods present opportunities to improve utilization through GPU sharing. However, existing GPU sharing systems lack the ability to adjust their resource allocation and sharing policies at runtime, making them ineffective at meeting latency service-level objectives (SLOs) under rapidly fluctuating workloads. This paper presents Prism, a multi-LLM serving system that unleashes the full potential of GPU sharing to achieve both cost efficiency and SLO attainment. At its core, Prism tackles a key limitation of existing systems$\unicode{x2014}$the lack of $\textit{cross-model memory coordination}$, which is essential for flexibly sharing GPU memory across models under dynamic workloads. Prism achieves this with two key designs. First, it supports on-demand memory allocation by dynamically mapping physical to virtual memory pages, allowing flexible memory redistribution among models that space- and time-share a GPU. Second, it improves memory efficiency through a two-level scheduling policy that dynamically adjusts sharing strategies based on models' runtime demands. Evaluations on real-world traces show that Prism achieves more than $2\times$ cost savings and $3.3\times$ SLO attainment compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02044v3">Towards Universal and Black-Box Query-Response Only Attack on LLMs with QROA</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Models (LLMs) has exposed critical security and ethical vulnerabilities, particularly their susceptibility to adversarial manipulations. This paper introduces QROA, a novel black-box jailbreak method designed to identify adversarial suffixes that can bypass LLM alignment safeguards when appended to a malicious instruction. Unlike existing suffix-based jailbreak approaches, QROA does not require access to the model's logit or any other internal information. It also eliminates reliance on human-crafted templates, operating solely through the standard query-response interface of LLMs. By framing the attack as an optimization bandit problem, QROA employs a surrogate model and token level optimization to efficiently explore suffix variations. Furthermore, we propose QROA-UNV, an extension that identifies universal adversarial suffixes for individual models, enabling one-query jailbreaks across a wide range of instructions. Testing on multiple models demonstrates Attack Success Rate (ASR) greater than 80\%. These findings highlight critical vulnerabilities, emphasize the need for advanced defenses, and contribute to the development of more robust safety evaluations for secure AI deployment. The code is made public on the following link: https://github.com/qroa/QROA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03973v1">Divide, Optimize, Merge: Fine-Grained LLM Agent Optimization at Scale</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      LLM-based optimization has shown remarkable potential in enhancing agentic systems. However, the conventional approach of prompting LLM optimizer with the whole training trajectories on training dataset in a single pass becomes untenable as datasets grow, leading to context window overflow and degraded pattern recognition. To address these challenges, we propose Fine-Grained Optimization (FGO), a scalable framework that divides large optimization tasks into manageable subsets, performs targeted optimizations, and systematically combines optimized components through progressive merging. Evaluation across ALFWorld, LogisticsQA, and GAIA benchmarks demonstrate that FGO outperforms existing approaches by 1.6-8.6% while reducing average prompt token consumption by 56.3%. Our framework provides a practical solution for scaling up LLM-based optimization of increasingly sophisticated agent systems. Further analysis demonstrates that FGO achieves the most consistent performance gain in all training dataset sizes, showcasing its scalability and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03961v1">The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete</a></div>
    <div class="paper-meta">
      📅 2025-05-06
      | 💬 16 pages, 8 figures. Code available at https://github.com/storyagents25/story-agents
    </div>
    <details class="paper-abstract">
      According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04654v1">A Comparative Analysis of Ethical and Safety Gaps in LLMs using Relative Danger Coefficient</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) and Large Language Models (LLMs) have rapidly evolved in recent years, showcasing remarkable capabilities in natural language understanding and generation. However, these advancements also raise critical ethical questions regarding safety, potential misuse, discrimination and overall societal impact. This article provides a comparative analysis of the ethical performance of various AI models, including the brand new DeepSeek-V3(R1 with reasoning and without), various GPT variants (4o, 3.5 Turbo, 4 Turbo, o1/o3 mini) and Gemini (1.5 flash, 2.0 flash and 2.0 flash exp) and highlights the need for robust human oversight, especially in situations with high stakes. Furthermore, we present a new metric for calculating harm in LLMs called Relative Danger Coefficient (RDC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03733v1">WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-05-06
    </div>
    <details class="paper-abstract">
      LLM-based agents have demonstrated great potential in generating and managing code within complex codebases. In this paper, we introduce WebGen-Bench, a novel benchmark designed to measure an LLM-based agent's ability to create multi-file website codebases from scratch. It contains diverse instructions for website generation, created through the combined efforts of human annotators and GPT-4o. These instructions span three major categories and thirteen minor categories, encompassing nearly all important types of web applications. To assess the quality of the generated websites, we use GPT-4o to generate test cases targeting each functionality described in the instructions, and then manually filter, adjust, and organize them to ensure accuracy, resulting in 647 test cases. Each test case specifies an operation to be performed on the website and the expected result after the operation. To automate testing and improve reproducibility, we employ a powerful web-navigation agent to execute tests on the generated websites and determine whether the observed responses align with the expected results. We evaluate three high-performance code-agent frameworks, Bolt.diy, OpenHands, and Aider, using multiple proprietary and open-source LLMs as engines. The best-performing combination, Bolt.diy powered by DeepSeek-R1, achieves only 27.8\% accuracy on the test cases, highlighting the challenging nature of our benchmark. Additionally, we construct WebGen-Instruct, a training set consisting of 6,667 website-generation instructions. Training Qwen2.5-Coder-32B-Instruct on Bolt.diy trajectories generated from a subset of this training set achieves an accuracy of 38.2\%, surpassing the performance of the best proprietary model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02376v1">LAMeD: LLM-generated Annotations for Memory Leak Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Static analysis tools are widely used to detect software bugs and vulnerabilities but often struggle with scalability and efficiency in complex codebases. Traditional approaches rely on manually crafted annotations -- labeling functions as sources or sinks -- to track data flows, e.g., ensuring that allocated memory is eventually freed, and code analysis tools such as CodeQL, Infer, or Cooddy can use function specifications, but manual annotation is laborious and error-prone, especially for large or third-party libraries. We present LAMeD (LLM-generated Annotations for Memory leak Detection), a novel approach that leverages large language models (LLMs) to automatically generate function-specific annotations. When integrated with analyzers such as Cooddy, LAMeD significantly improves memory leak detection and reduces path explosion. We also suggest directions for extending LAMeD to broader code analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00985v2">Position: Enough of Scaling LLMs! Lets Focus on Downscaling</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      We challenge the dominant focus on neural scaling laws and advocate for a paradigm shift toward downscaling in the development of large language models (LLMs). While scaling laws have provided critical insights into performance improvements through increasing model and dataset size, we emphasize the significant limitations of this approach, particularly in terms of computational inefficiency, environmental impact, and deployment constraints. To address these challenges, we propose a holistic framework for downscaling LLMs that seeks to maintain performance while drastically reducing resource demands. This paper outlines practical strategies for transitioning away from traditional scaling paradigms, advocating for a more sustainable, efficient, and accessible approach to LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00626v2">The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them)</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) that integrate multiple input roles (e.g., system instructions, user queries, external tool outputs) are increasingly prevalent in practice. Ensuring that the model accurately distinguishes messages from each role -- a concept we call \emph{role separation} -- is crucial for consistent multi-role behavior. Although recent work often targets state-of-the-art prompt injection defenses, it remains unclear whether such methods truly teach LLMs to differentiate roles or merely memorize known triggers. In this paper, we examine \emph{role-separation learning}: the process of teaching LLMs to robustly distinguish system and user tokens. Through a \emph{simple, controlled experimental framework}, we find that fine-tuned models often rely on two proxies for role identification: (1) task type exploitation, and (2) proximity to begin-of-text. Although data augmentation can partially mitigate these shortcuts, it generally leads to iterative patching rather than a deeper fix. To address this, we propose reinforcing \emph{invariant signals} that mark role boundaries by adjusting token-wise cues in the model's input encoding. In particular, manipulating position IDs helps the model learn clearer distinctions and reduces reliance on superficial proxies. By focusing on this mechanism-centered perspective, our work illuminates how LLMs can more reliably maintain consistent multi-role behavior without merely memorizing known prompts or triggers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02324v1">From Course to Skill: Evaluating LLM Performance in Curricular Analytics</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Curricular analytics (CA) -- systematic analysis of curricula data to inform program and course refinement -- becomes an increasingly valuable tool to help institutions align academic offerings with evolving societal and economic demands. Large language models (LLMs) are promising for handling large-scale, unstructured curriculum data, but it remains uncertain how reliably LLMs can perform CA tasks. In this paper, we systematically evaluate four text alignment strategies based on LLMs or traditional NLP methods for skill extraction, a core task in CA. Using a stratified sample of 400 curriculum documents of different types and a human-LLM collaborative evaluation framework, we find that retrieval-augmented generation (RAG) to be the top-performing strategy across all types of curriculum documents, while zero-shot prompting performs worse than traditional NLP methods in most cases. Our findings highlight the promise of LLMs in analyzing brief and abstract curriculum documents, but also reveal that their performance can vary significantly depending on model selection and prompting strategies. This underscores the importance of carefully evaluating the performance of LLM-based strategies before large-scale deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02322v1">HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 arXiv admin note: text overlap with arXiv:2406.14228 by other authors
    </div>
    <details class="paper-abstract">
      Recent advancements have significantly enhanced the performance of large language models (LLMs) in tackling complex reasoning tasks, achieving notable success in domains like mathematical and logical reasoning. However, these methods encounter challenges with complex planning tasks, primarily due to extended reasoning steps, diverse constraints, and the challenge of handling multiple distinct sub-tasks. To address these challenges, we propose HyperTree Planning (HTP), a novel reasoning paradigm that constructs hypertree-structured planning outlines for effective planning. The hypertree structure enables LLMs to engage in hierarchical thinking by flexibly employing the divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. We further introduce an autonomous planning framework that completes the planning process by iteratively refining and expanding the hypertree-structured planning outlines. Experiments demonstrate the effectiveness of HTP, achieving state-of-the-art accuracy on the TravelPlanner benchmark with Gemini-1.5-Pro, resulting in a 3.6 times performance improvement over o1-preview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02309v1">Optimizing LLMs for Resource-Constrained Environments: A Survey of Model Compression Techniques</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted to IEEE COMPSAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. This survey paper provides a comprehensive overview of techniques for compressing LLMs to enable efficient inference in resource-constrained environments. We examine three primary approaches: Knowledge Distillation, Model Quantization, and Model Pruning. For each technique, we discuss the underlying principles, present different variants, and provide examples of successful applications. We also briefly discuss complementary techniques such as mixture-of-experts and early-exit strategies. Finally, we highlight promising future directions, aiming to provide a valuable resource for both researchers and practitioners seeking to optimize LLMs for edge deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03059v1">Improving Model Alignment Through Collective Intelligence of Open-Source LLMS</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Building helpful and harmless large language models (LLMs) requires effective model alignment approach based on human instructions and feedback, which necessitates high-quality human-labeled data. Constructing such datasets is often expensive and hard to scale, and may face potential limitations on diversity and generalization. To address these challenges, we introduce Mixture of Agents Alignment (MoAA), that leverages the collective strengths of various language models to provide high-quality data for model alignment. By employing MoAA, we enhance both supervised fine-tuning and preference optimization, leading to improved performance compared to using a single model alone to generate alignment data (e.g. using GPT-4o alone). Evaluation results show that our approach can improve win rate of LLaMA-3.1-8B-Instruct from 19.5 to 48.3 on Arena-Hard and from 22.33 to 57.23 on AlpacaEval2, highlighting a promising direction for model alignment through this new scalable and diverse synthetic data recipe. Furthermore, we demonstrate that MoAA enables a self-improvement pipeline, where models finetuned on MoA-generated data surpass their own initial capabilities, providing evidence that our approach can push the frontier of open-source LLMs without reliance on stronger external supervision. Data and code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03049v1">34 Examples of LLM Applications in Materials Science and Chemistry: Towards Automation, Assistants, Agents, and Accelerated Scientific Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.15221
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping many aspects of materials science and chemistry research, enabling advances in molecular property prediction, materials design, scientific automation, knowledge extraction, and more. Recent developments demonstrate that the latest class of models are able to integrate structured and unstructured data, assist in hypothesis generation, and streamline research workflows. To explore the frontier of LLM capabilities across the research lifecycle, we review applications of LLMs through 34 total projects developed during the second annual Large Language Model Hackathon for Applications in Materials Science and Chemistry, a global hybrid event. These projects spanned seven key research areas: (1) molecular and material property prediction, (2) molecular and material design, (3) automation and novel interfaces, (4) scientific communication and education, (5) research data management and automation, (6) hypothesis generation and evaluation, and (7) knowledge extraction and reasoning from the scientific literature. Collectively, these applications illustrate how LLMs serve as versatile predictive models, platforms for rapid prototyping of domain-specific tools, and much more. In particular, improvements in both open source and proprietary LLM performance through the addition of reasoning, additional training data, and new techniques have expanded effectiveness, particularly in low-data environments and interdisciplinary research. As LLMs continue to improve, their integration into scientific workflows presents both new opportunities and new challenges, requiring ongoing exploration, continued refinement, and further research to address reliability, interpretability, and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03030v1">UCSC at SemEval-2025 Task 3: Context, Models and Prompt Optimization for Automated Hallucination Detection in LLM Output</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Hallucinations pose a significant challenge for large language models when answering knowledge-intensive queries. As LLMs become more widely adopted, it is crucial not only to detect if hallucinations occur but also to pinpoint exactly where in the LLM output they occur. SemEval 2025 Task 3, Mu-SHROOM: Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes, is a recent effort in this direction. This paper describes the UCSC system submission to the shared Mu-SHROOM task. We introduce a framework that first retrieves relevant context, next identifies false content from the answer, and finally maps them back to spans in the LLM output. The process is further enhanced by automatically optimizing prompts. Our system achieves the highest overall performance, ranking #1 in average position across all languages. We release our code and experiment results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14498v3">LLaSA: A Multimodal LLM for Human Activity Analysis Through Wearable and Smartphone Sensors</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Wearables generate rich motion data, yet current systems only classify what happened - failing to support natural questions about why it happened or what it means. We introduce LLaSA (Large Language and Sensor Assistant), a compact 13B model that enables ask-anything, open-ended question answering grounded in raw IMU data. LLaSA supports conversational, context-aware reasoning - explaining the causes of sensor-detected behaviors and answering free-form questions in real-world scenarios. It is tuned for scientific accuracy, coherence, and response reliability. To advance this new task of sensor-based QA, we release three large-scale datasets: SensorCaps, OpenSQA, and Tune-OpenSQA. Together, these resources define a new benchmark for sensor-language models. LLaSA consistently produces interpretable, causal answers and outperforms commercial LLMs across both public and real-world settings. Our code repository and datasets can be found at https://github.com/BASHLab/LLaSA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14307v2">LLM-3D Print: Large Language Models To Monitor and Control 3D Printing</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Industry 4.0 has revolutionized manufacturing by driving digitalization and shifting the paradigm toward additive manufacturing (AM). Fused Deposition Modeling (FDM), a key AM technology, enables the creation of highly customized, cost-effective products with minimal material waste through layer-by-layer extrusion, posing a significant challenge to traditional subtractive methods. However, the susceptibility of material extrusion techniques to errors often requires expert intervention to detect and mitigate defects that can severely compromise product quality. While automated error detection and machine learning models exist, their generalizability across diverse 3D printer setups, firmware, and sensors is limited, and deep learning methods require extensive labeled datasets, hindering scalability and adaptability. To address these challenges, we present a process monitoring and control framework that leverages pre-trained Large Language Models (LLMs) alongside 3D printers to detect and address printing defects. The LLM evaluates print quality by analyzing images captured after each layer or print segment, identifying failure modes and querying the printer for relevant parameters. It then generates and executes a corrective action plan. We validated the effectiveness of the proposed framework in identifying defects by comparing it against a control group of engineers with diverse AM expertise. Our evaluation demonstrated that LLM-based agents not only accurately identify common 3D printing errors, such as inconsistent extrusion, stringing, warping, and layer adhesion, but also effectively determine the parameters causing these failures and autonomously correct them without any need for human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01822v3">Firewalls to Secure Dynamic LLM Agentic Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Future LLM agents are likely to communicate on behalf of users with other entity-representing agents on tasks that entail long-horizon plans with interdependent goals. Current work does not focus on such agentic networks, nor does it address their challenges. Thus, we first identify the required properties of agents' communication, which should be proactive and adaptable. It needs to satisfy 1) privacy: agents should not share more than what is needed for the task, and 2) security: the communication must preserve integrity and maintain utility against selfish entities. We design a use case (travel planning) as a testbed that exemplifies these requirements, and we show examples of how this can go wrong. Next, we propose a practical design, inspired by established network security principles, for constrained LLM agentic networks that balance adaptability, security, and privacy. Our framework automatically constructs and updates task-specific rules from prior simulations to build firewalls. We offer layers of defense to 1) convert free-form input to a task-specific protocol, 2) dynamically abstract users' data to a task-specific degree of permissiveness, and 3) self-correct the agents' trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03019v1">Memorization or Interpolation ? Detecting LLM Memorization through Input Perturbation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) achieve remarkable performance through training on massive datasets, they can exhibit concerning behaviors such as verbatim reproduction of training data rather than true generalization. This memorization phenomenon raises significant concerns about data privacy, intellectual property rights, and the reliability of model evaluations. This paper introduces PEARL, a novel approach for detecting memorization in LLMs. PEARL assesses how sensitive an LLM's performance is to input perturbations, enabling memorization detection without requiring access to the model's internals. We investigate how input perturbations affect the consistency of outputs, enabling us to distinguish between true generalization and memorization. Our findings, following extensive experiments on the Pythia open model, provide a robust framework for identifying when the model simply regurgitates learned information. Applied on the GPT 4o models, the PEARL framework not only identified cases of memorization of classic texts from the Bible or common code from HumanEval but also demonstrated that it can provide supporting evidence that some data, such as from the New York Times news articles, were likely part of the training data of a given model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07963v3">Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 22 pages, 12 figures, 4 tables, CHIL 2025
    </div>
    <details class="paper-abstract">
      Medical research faces well-documented challenges in translating novel treatments into clinical practice. Publishing incentives encourage researchers to present "positive" findings, even when empirical results are equivocal. Consequently, it is well-documented that authors often spin study results, especially in article abstracts. Such spin can influence clinician interpretation of evidence and may affect patient care decisions. In this study, we ask whether the interpretation of trial results offered by Large Language Models (LLMs) is similarly affected by spin. This is important since LLMs are increasingly being used to trawl through and synthesize published medical evidence. We evaluated 22 LLMs and found that they are across the board more susceptible to spin than humans. They might also propagate spin into their outputs: We find evidence, e.g., that LLMs implicitly incorporate spin into plain language summaries that they generate. We also find, however, that LLMs are generally capable of recognizing spin, and can be prompted in a way to mitigate spin's impact on LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00153v3">Beyond Single Concept Vector: Modeling Concept Subspace in LLMs with Gaussian Distribution</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Probing learned concepts in large language models (LLMs) is crucial for understanding how semantic knowledge is encoded internally. Training linear classifiers on probing tasks is a principle approach to denote the vector of a certain concept in the representation space. However, the single vector identified for a concept varies with both data and training, making it less robust and weakening its effectiveness in real-world applications. To address this challenge, we propose an approach to approximate the subspace representing a specific concept. Built on linear probing classifiers, we extend the concept vectors into Gaussian Concept Subspace (GCS). We demonstrate GCS's effectiveness through measuring its faithfulness and plausibility across multiple LLMs with different sizes and architectures. Additionally, we use representation intervention tasks to showcase its efficacy in real-world applications such as emotion steering. Experimental results indicate that GCS concept vectors have the potential to balance steering performance and maintaining the fluency in natural language generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01122v2">CFBench: A Comprehensive Constraints-Following Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 15 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The adeptness of Large Language Models (LLMs) in comprehending and following natural language instructions is critical for their deployment in sophisticated real-world applications. Existing evaluations mainly focus on fragmented constraints or narrow scenarios, but they overlook the comprehensiveness and authenticity of constraints from the user's perspective. To bridge this gap, we propose CFBench, a large-scale Comprehensive Constraints Following Benchmark for LLMs, featuring 1,000 curated samples that cover more than 200 real-life scenarios and over 50 NLP tasks. CFBench meticulously compiles constraints from real-world instructions and constructs an innovative systematic framework for constraint types, which includes 10 primary categories and over 25 subcategories, and ensures each constraint is seamlessly integrated within the instructions. To make certain that the evaluation of LLM outputs aligns with user perceptions, we propose an advanced methodology that integrates multi-dimensional assessment criteria with requirement prioritization, covering various perspectives of constraints, instructions, and requirement fulfillment. Evaluating current leading LLMs on CFBench reveals substantial room for improvement in constraints following, and we further investigate influencing factors and enhancement strategies. The data and code are publicly available at https://github.com/PKU-Baichuan-MLSystemLab/CFBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02922v1">RetroInfer: A Vector-Storage Approach for Scalable Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      The growing context lengths of large language models (LLMs) pose significant challenges for efficient inference, primarily due to GPU memory and bandwidth constraints. We present RetroInfer, a novel system that reconceptualizes the key-value (KV) cache as a vector storage system which exploits the inherent attention sparsity to accelerate long-context LLM inference. At its core is the wave index, an Attention-aWare VEctor index that enables efficient and accurate retrieval of critical tokens through techniques such as tripartite attention approximation, accuracy-bounded attention estimation, and segmented clustering. Complementing this is the wave buffer, which coordinates KV cache placement and overlaps computation and data transfer across GPU and CPU to sustain high throughput. Unlike prior sparsity-based methods that struggle with token selection and hardware coordination, RetroInfer delivers robust performance without compromising model accuracy. Experiments on long-context benchmarks show up to 4.5X speedup over full attention within GPU memory limits and up to 10.5X over sparse attention baselines when KV cache is extended to CPU memory, all while preserving full-attention-level accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02881v1">Rewriting Pre-Training Data Boosts LLM Performance in Math and Code</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 27pages(including appendix), 10 figures
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04640v1">A Comparative Benchmark of a Moroccan Darija Toxicity Detection Model (Typica.ai) and Major LLM-Based Moderation APIs (OpenAI, Mistral, Anthropic)</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 GitHub repository with reproducibility materials and evaluation notebook available at: https://github.com/assoudi-typica-ai/darija-toxicity-benchmark
    </div>
    <details class="paper-abstract">
      This paper presents a comparative benchmark evaluating the performance of Typica.ai's custom Moroccan Darija toxicity detection model against major LLM-based moderation APIs: OpenAI (omni-moderation-latest), Mistral (mistral-moderation-latest), and Anthropic Claude (claude-3-haiku-20240307). We focus on culturally grounded toxic content, including implicit insults, sarcasm, and culturally specific aggression often overlooked by general-purpose systems. Using a balanced test set derived from the OMCD_Typica.ai_Mix dataset, we report precision, recall, F1-score, and accuracy, offering insights into challenges and opportunities for moderation in underrepresented languages. Our results highlight Typica.ai's superior performance, underlining the importance of culturally adapted models for reliable content moderation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05494v1">An Automated LLM-based Pipeline for Asset-Level Database Creation to Assess Deforestation Impact</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Accepted to ACL ClimateNLP 2025
    </div>
    <details class="paper-abstract">
      The European Union Deforestation Regulation (EUDR) requires companies to prove their products do not contribute to deforestation, creating a critical demand for precise, asset-level environmental impact data. Current databases lack the necessary detail, relying heavily on broad financial metrics and manual data collection, which limits regulatory compliance and accurate environmental modeling. This study presents an automated, end-to-end data extraction pipeline that uses LLMs to create, clean, and validate structured databases, specifically targeting sectors with a high risk of deforestation. The pipeline introduces Instructional, Role-Based, Zero-Shot Chain-of-Thought (IRZ-CoT) prompting to enhance data extraction accuracy and a Retrieval-Augmented Validation (RAV) process that integrates real-time web searches for improved data reliability. Applied to SEC EDGAR filings in the Mining, Oil & Gas, and Utilities sectors, the pipeline demonstrates significant improvements over traditional zero-shot prompting approaches, particularly in extraction accuracy and validation coverage. This work advances NLP-driven automation for regulatory compliance, CSR (Corporate Social Responsibility), and ESG, with broad sectoral applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02802v1">Generating HomeAssistant Automations Using an LLM-based Chatbot</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      To combat climate change, individuals are encouraged to adopt sustainable habits, in particular, with their household, optimizing their electrical consumption. Conversational agents, such as Smart Home Assistants, hold promise as effective tools for promoting sustainable practices within households. Our research investigated the application of Large Language Models (LLM) in enhancing smart home automation and promoting sustainable household practices, specifically using the HomeAssistant framework. In particular, it highlights the potential of GPT models in generating accurate automation routines. While the LLMs showed proficiency in understanding complex commands and creating valid JSON outputs, challenges such as syntax errors and message malformations were noted, indicating areas for further improvement. Still, despite minimal quantitative differences between "green" and "no green" prompts, qualitative feedback highlighted a positive shift towards sustainability in the routines generated with environmentally focused prompts. Then, an empirical evaluation (N=56) demonstrated that the system was well-received and found engaging by users compared to its traditional rule-based counterpart. Our findings highlight the role of LLMs in advancing smart home technologies and suggest further research to refine these models for broader, real-world applications to support sustainable living.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21239v4">Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the determinant of the Gram matrix of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring internal access to LLMs. We conduct extensive experiments on both external and internal uncertainty detection, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20834v2">Token-Efficient RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-05
      | 💬 Title updated to "Token-Efficient RL for LLM Reasoning" to better reflect algorithmic focus. Revised abstract, intro, and conclusion. Paper shortened and typos fixed
    </div>
    <details class="paper-abstract">
      We propose reinforcement learning (RL) strategies tailored for reasoning in large language models (LLMs) under strict memory and compute limits, with a particular focus on compatibility with LoRA fine-tuning. Rather than relying on full-sequence updates or separate critic networks, we design critic-free methods that operate on a small, informative subset of output tokens to reduce memory usage and stabilize training. We introduce S-GRPO, a stochastic variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching approach for fine-grained credit assignment. Applied to Qwen2-1.5B, our methods raise accuracy on the SVAMP benchmark from 46% to over 70% and show strong performance on multi-digit multiplication. Surprisingly, full-token GRPO under LoRA fails to improve over the base model, suggesting that selective token-level optimization may act as an implicit regularizer in low-parameter training regimes.
    </details>
</div>
