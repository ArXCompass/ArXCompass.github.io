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
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19449v1">R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), while demonstrating remarkable capabilities across various applications, present significant challenges during inference due to their substantial model size, especially when deployed on edge devices. Activation sparsity offers a promising solution to reduce computation and memory movement, enabling more efficient inference, particularly for small-batch on-device applications. However, current approaches face limitations with non-ReLU activation function, which are foundational to most advanced LLMs, or require heavy continual training. Additionally, the difficulty in predicting active channels and limited achievable sparsity ratios constrain the effectiveness of activation sparsity-based methods. In this paper, we introduce R-Sparse, a training-free activation sparsity approach capable of achieving high sparsity levels in advanced LLMs. We conducted two preliminary investigations into how different components contribute to the output within a single linear layer and found two key observations: (i) the non-sparse components of the input function can be regarded as a few bias terms, and (ii) The full computation can be effectively approximated by an appropriate combination of input channels and weight singular values. Building on this, we replace the linear layers in LLMs with a rank-aware sparse inference method that leverages the sparsity of input channels and singular value components, eliminating the need for active channel prediction like the output sparsity based approaches. Experiments on Llama-2/3 and Mistral models across ten diverse tasks demonstrate that R-Sparse achieves comparable performance at 50% model-level sparsity, resulting in a significant 43% end-to-end efficient improvements with customized kernels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17620v3">A Case Study of Scalable Content Annotation Using Multi-LLM Consensus and Human Review</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 7 pages, GenAICHI: CHI 2025 Workshop on Generative AI and HCI
    </div>
    <details class="paper-abstract">
      Content annotation at scale remains challenging, requiring substantial human expertise and effort. This paper presents a case study in code documentation analysis, where we explore the balance between automation efficiency and annotation accuracy. We present MCHR (Multi-LLM Consensus with Human Review), a novel semi-automated framework that enhances annotation scalability through the systematic integration of multiple LLMs and targeted human review. Our framework introduces a structured consensus-building mechanism among LLMs and an adaptive review protocol that strategically engages human expertise. Through our case study, we demonstrate that MCHR reduces annotation time by 32% to 100% compared to manual annotation while maintaining high accuracy (85.5% to 98%) across different difficulty levels, from basic binary classification to challenging open-set scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03665v2">LLM & HPC:Benchmarking DeepSeek's Performance in High-Performance Computing Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 9 pages, 2 figures, 3 tables, conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as GPT-4 and DeepSeek, have been applied to a wide range of domains in software engineering. However, their potential in the context of High-Performance Computing (HPC) much remains to be explored. This paper evaluates how well DeepSeek, a recent LLM, performs in generating a set of HPC benchmark codes: a conjugate gradient solver, the parallel heat equation, parallel matrix multiplication, DGEMM, and the STREAM triad operation. We analyze DeepSeek's code generation capabilities for traditional HPC languages like Cpp, Fortran, Julia and Python. The evaluation includes testing for code correctness, performance, and scaling across different configurations and matrix sizes. We also provide a detailed comparison between DeepSeek and another widely used tool: GPT-4. Our results demonstrate that while DeepSeek generates functional code for HPC tasks, it lags behind GPT-4, in terms of scalability and execution efficiency of the generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16732v3">Perfectly to a Tee: Understanding User Perceptions of Personalized LLM-Enhanced Narrative Interventions</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Stories about overcoming personal struggles can effectively illustrate the application of psychological theories in real life, yet they may fail to resonate with individuals' experiences. In this work, we employ large language models (LLMs) to create tailored narratives that acknowledge and address unique challenging thoughts and situations faced by individuals. Our study, involving 346 young adults across two settings, demonstrates that personalized LLM-enhanced stories were perceived to be better than human-written ones in conveying key takeaways, promoting reflection, and reducing belief in negative thoughts. These stories were not only seen as more relatable but also similarly authentic to human-written ones, highlighting the potential of LLMs in helping young adults manage their struggles. The findings of this work provide crucial design considerations for future narrative-based digital mental health interventions, such as the need to maintain relatability without veering into implausibility and refining the wording and tone of AI-enhanced content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20196v1">Prompting LLMs for Code Editing: Struggles and Remedies</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly transforming software engineering, with coding assistants embedded in an IDE becoming increasingly prevalent. While research has focused on improving the tools and understanding developer perceptions, a critical gap exists in understanding how developers actually use these tools in their daily workflows, and, crucially, where they struggle. This paper addresses part of this gap through a multi-phased investigation of developer interactions with an LLM-powered code editing and transformation feature, Transform Code, in an IDE widely used at Google. First, we analyze telemetry logs of the feature usage, revealing that frequent re-prompting can be an indicator of developer struggles with using Transform Code. Second, we conduct a qualitative analysis of unsatisfactory requests, identifying five key categories of information often missing from developer prompts. Finally, based on these findings, we propose and evaluate a tool, AutoPrompter, for automatically improving prompts by inferring missing information from the surrounding code context, leading to a 27% improvement in edit correctness on our test set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20183v1">BLADE: Benchmark suite for LLM-driven Automated Design and Evolution of iterative optimisation heuristics</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 9 pages, accepted at GECCO Workshop 2025
    </div>
    <details class="paper-abstract">
      The application of Large Language Models (LLMs) for Automated Algorithm Discovery (AAD), particularly for optimisation heuristics, is an emerging field of research. This emergence necessitates robust, standardised benchmarking practices to rigorously evaluate the capabilities and limitations of LLM-driven AAD methods and the resulting generated algorithms, especially given the opacity of their design process and known issues with existing benchmarks. To address this need, we introduce BLADE (Benchmark suite for LLM-driven Automated Design and Evolution), a modular and extensible framework specifically designed for benchmarking LLM-driven AAD methods in a continuous black-box optimisation context. BLADE integrates collections of benchmark problems (including MA-BBOB and SBOX-COST among others) with instance generators and textual descriptions aimed at capability-focused testing, such as generalisation, specialisation and information exploitation. It offers flexible experimental setup options, standardised logging for reproducibility and fair comparison, incorporates methods for analysing the AAD process (e.g., Code Evolution Graphs and various visualisation approaches) and facilitates comparison against human-designed baselines through integration with established tools like IOHanalyser and IOHexplainer. BLADE provides an `out-of-the-box' solution to systematically evaluate LLM-driven AAD approaches. The framework is demonstrated through two distinct use cases exploring mutation prompt strategies and function specialisation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20119v1">Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 8 Pages. This paper has been accepted for presentation at the IEEE Swiss Conference on Data Science (SDS25)
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has advanced significantly in recent years. The complexity of RAG systems, which involve multiple components-such as indexing, retrieval, and generation-along with numerous other parameters, poses substantial challenges for systematic evaluation and quality enhancement. Previous research highlights that evaluating RAG systems is essential for documenting advancements, comparing configurations, and identifying effective approaches for domain-specific applications. This study systematically reviews 63 academic articles to provide a comprehensive overview of state-of-the-art RAG evaluation methodologies, focusing on four key areas: datasets, retrievers, indexing and databases, and the generator component. We observe the feasibility of an automated evaluation approach for each component of a RAG system, leveraging an LLM capable of both generating evaluation datasets and conducting evaluations. In addition, we found that further practical research is essential to provide companies with clear guidance on the do's and don'ts of implementing and evaluating RAG systems. By synthesizing evaluation approaches for key RAG components and emphasizing the creation and adaptation of domain-specific datasets for benchmarking, we contribute to the advancement of systematic evaluation methods and the improvement of evaluation rigor for RAG systems. Furthermore, by examining the interplay between automated approaches leveraging LLMs and human judgment, we contribute to the ongoing discourse on balancing automation and human input, clarifying their respective contributions, limitations, and challenges in achieving robust and reliable evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20118v1">OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we integrate OpenTCM with this knowledge graph, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that our prompt design and model selection significantly improve knowledge graph quality, achieving a precision of 98. 55% and an F1 score of 99. 55%. In addition, OpenTCM achieves mean expert scores of 4.5 in ingredient information retrieval and 3.8 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20117v1">ResearchCodeAgent: An LLM Multi-Agent System for Automated Codification of Research Methodologies</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      In this paper we introduce ResearchCodeAgent, a novel multi-agent system leveraging large language models (LLMs) agents to automate the codification of research methodologies described in machine learning literature. The system bridges the gap between high-level research concepts and their practical implementation, allowing researchers auto-generating code of existing research papers for benchmarking or building on top-of existing methods specified in the literature with availability of partial or complete starter code. ResearchCodeAgent employs a flexible agent architecture with a comprehensive action suite, enabling context-aware interactions with the research environment. The system incorporates a dynamic planning mechanism, utilizing both short and long-term memory to adapt its approach iteratively. We evaluate ResearchCodeAgent on three distinct machine learning tasks with distinct task complexity and representing different parts of the ML pipeline: data augmentation, optimization, and data batching. Our results demonstrate the system's effectiveness and generalizability, with 46.9% of generated code being high-quality and error-free, and 25% showing performance improvements over baseline implementations. Empirical analysis shows an average reduction of 57.9% in coding time compared to manual implementation. We observe higher gains for more complex tasks. ResearchCodeAgent represents a significant step towards automating the research implementation process, potentially accelerating the pace of machine learning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21045v1">Leveraging LLM to Strengthen ML-Based Cross-Site Scripting Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 This work has been accepted for presentation at the ACM Workshop on Wireless Security and Machine Learning (WiseML 2025)
    </div>
    <details class="paper-abstract">
      According to the Open Web Application Security Project (OWASP), Cross-Site Scripting (XSS) is a critical security vulnerability. Despite decades of research, XSS remains among the top 10 security vulnerabilities. Researchers have proposed various techniques to protect systems from XSS attacks, with machine learning (ML) being one of the most widely used methods. An ML model is trained on a dataset to identify potential XSS threats, making its effectiveness highly dependent on the size and diversity of the training data. A variation of XSS is obfuscated XSS, where attackers apply obfuscation techniques to alter the code's structure, making it challenging for security systems to detect its malicious intent. Our study's random forest model was trained on traditional (non-obfuscated) XSS data achieved 99.8% accuracy. However, when tested against obfuscated XSS samples, accuracy dropped to 81.9%, underscoring the importance of training ML models with obfuscated data to improve their effectiveness in detecting XSS attacks. A significant challenge is to generate highly complex obfuscated code despite the availability of several public tools. These tools can only produce obfuscation up to certain levels of complexity. In our proposed system, we fine-tune a Large Language Model (LLM) to generate complex obfuscated XSS payloads automatically. By transforming original XSS samples into diverse obfuscated variants, we create challenging training data for ML model evaluation. Our approach achieved a 99.5% accuracy rate with the obfuscated dataset. We also found that the obfuscated samples generated by the LLMs were 28.1% more complex than those created by other tools, significantly improving the model's ability to handle advanced XSS attacks and making it more effective for real-world application security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21038v1">Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21036v1">Can Differentially Private Fine-tuning LLMs Protect Against Privacy Attacks?</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) has become an essential strategy for adapting them to specialized tasks; however, this process introduces significant privacy challenges, as sensitive training data may be inadvertently memorized and exposed. Although differential privacy (DP) offers strong theoretical guarantees against such leakage, its empirical privacy effectiveness on LLMs remains unclear, especially under different fine-tuning methods. In this paper, we systematically investigate the impact of DP across fine-tuning methods and privacy budgets, using both data extraction and membership inference attacks to assess empirical privacy risks. Our main findings are as follows: (1) Differential privacy reduces model utility, but its impact varies significantly across different fine-tuning methods. (2) Without DP, the privacy risks of models fine-tuned with different approaches differ considerably. (3) When DP is applied, even a relatively high privacy budget can substantially lower privacy risk. (4) The privacy-utility trade-off under DP training differs greatly among fine-tuning methods, with some methods being unsuitable for DP due to severe utility degradation. Our results provide practical guidance for privacy-conscious deployment of LLMs and pave the way for future research on optimizing the privacy-utility trade-off in fine-tuning methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00031v1">Learning to Plan Before Answering: Self-Teaching LLMs to Learn Abstract Plans for Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      In the field of large language model (LLM) post-training, the effectiveness of utilizing synthetic data generated by the LLM itself has been well-presented. However, a key question remains unaddressed: what essential information should such self-generated data encapsulate? Existing approaches only produce step-by-step problem solutions, and fail to capture the abstract meta-knowledge necessary for generalization across similar problems. Drawing insights from cognitive science, where humans employ high-level abstraction to simplify complex problems before delving into specifics, we introduce a novel self-training algorithm: LEarning to Plan before Answering (LEPA). LEPA trains the LLM to formulate anticipatory plans, which serve as abstract meta-knowledge for problem-solving, before engaging with the intricacies of problems. This approach not only outlines the solution generation path but also shields the LLM from the distraction of irrelevant details. During data generation, LEPA first crafts an anticipatory plan based on the problem, and then generates a solution that aligns with both the plan and the problem. LEPA refines the plan through self-reflection, aiming to acquire plans that are instrumental in yielding correct solutions. During model optimization, the LLM is trained to predict both the refined plans and the corresponding solutions. By efficiently extracting and utilizing the anticipatory plans, LEPA demonstrates remarkable superiority over conventional algorithms on various challenging natural language reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01441v1">Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable progress in complex reasoning tasks, yet they remain fundamentally limited by their reliance on static internal knowledge and text-only reasoning. Real-world problem solving often demands dynamic, multi-step reasoning, adaptive decision making, and the ability to interact with external tools and environments. In this work, we introduce ARTIST (Agentic Reasoning and Tool Integration in Self-improving Transformers), a unified framework that tightly couples agentic reasoning, reinforcement learning, and tool integration for LLMs. ARTIST enables models to autonomously decide when, how, and which tools to invoke within multi-turn reasoning chains, leveraging outcome-based RL to learn robust strategies for tool use and environment interaction without requiring step-level supervision. Extensive experiments on mathematical reasoning and multi-turn function calling benchmarks show that ARTIST consistently outperforms state-of-the-art baselines, with up to 22% absolute improvement over base models and strong gains on the most challenging tasks. Detailed studies and metric analyses reveal that agentic RL training leads to deeper reasoning, more effective tool use, and higher-quality solutions. Our results establish agentic RL with tool integration as a powerful new frontier for robust, interpretable, and generalizable problem-solving in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20022v1">Better To Ask in English? Evaluating Factual Accuracy of Multilingual LLMs in English and Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) have demonstrated significant effectiveness across various languages, particularly in high-resource languages such as English. However, their performance in terms of factual accuracy across other low-resource languages, especially Indic languages, remains an area of investigation. In this study, we assess the factual accuracy of LLMs - GPT-4o, Gemma-2-9B, Gemma-2-2B, and Llama-3.1-8B - by comparing their performance in English and Indic languages using the IndicQuest dataset, which contains question-answer pairs in English and 19 Indic languages. By asking the same questions in English and their respective Indic translations, we analyze whether the models are more reliable for regional context questions in Indic languages or when operating in English. Our findings reveal that LLMs often perform better in English, even for questions rooted in Indic contexts. Notably, we observe a higher tendency for hallucination in responses generated in low-resource Indic languages, highlighting challenges in the multilingual understanding capabilities of current LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20016v1">Applying LLM-Powered Virtual Humans to Child Interviews in Child-Centered Design</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 This paper has been accepted as a Work-in-Progress (WiP) paper in the 24th annual ACM Interaction Design and Children (IDC) Conference
    </div>
    <details class="paper-abstract">
      In child-centered design, directly engaging children is crucial for deeply understanding their experiences. However, current research often prioritizes adult perspectives, as interviewing children involves unique challenges such as environmental sensitivities and the need for trust-building. AI-powered virtual humans (VHs) offer a promising approach to facilitate engaging and multimodal interactions with children. This study establishes key design guidelines for LLM-powered virtual humans tailored to child interviews, standardizing multimodal elements including color schemes, voice characteristics, facial features, expressions, head movements, and gestures. Using ChatGPT-based prompt engineering, we developed three distinct Human-AI workflows (LLM-Auto, LLM-Interview, and LLM-Analyze) and conducted a user study involving 15 children aged 6 to 12. The results indicated that the LLM-Analyze workflow outperformed the others by eliciting longer responses, achieving higher user experience ratings, and promoting more effective child engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20013v1">LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 ACM SIGIR 2025 Full Paper
    </div>
    <details class="paper-abstract">
      Online fake news moderation now faces a new challenge brought by the malicious use of large language models (LLMs) in fake news production. Though existing works have shown LLM-generated fake news is hard to detect from an individual aspect, it remains underexplored how its large-scale release will impact the news ecosystem. In this study, we develop a simulation pipeline and a dataset with ~56k generated news of diverse types to investigate the effects of LLM-generated fake news within neural news recommendation systems. Our findings expose a truth decay phenomenon, where real news is gradually losing its advantageous position in news ranking against fake news as LLM-generated news is involved in news recommendation. We further provide an explanation about why truth decay occurs from a familiarity perspective and show the positive correlation between perplexity and news ranking. Finally, we discuss the threats of LLM-generated fake news and provide possible countermeasures. We urge stakeholders to address this emerging challenge to preserve the integrity of news ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20006v1">Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics in the Evaluation of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 10 pages, 8 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Battles, or side-by-side comparisons in so called arenas that elicit human preferences, have emerged as a popular approach to assessing the output quality of LLMs. Recently, this idea has been extended to retrieval-augmented generation (RAG) systems. While undoubtedly representing an advance in evaluation, battles have at least two drawbacks, particularly in the context of complex information-seeking queries: they are neither explanatory nor diagnostic. Recently, the nugget evaluation methodology has emerged as a promising approach to evaluate the quality of RAG answers. Nuggets decompose long-form LLM-generated answers into atomic facts, highlighting important pieces of information necessary in a "good" response. In this work, we apply our AutoNuggetizer framework to analyze data from roughly 7K Search Arena battles provided by LMArena in a fully automatic manner. Our results show a significant correlation between nugget scores and human preferences, showcasing promise in our approach to explainable and diagnostic system evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20000v1">Knowledge Distillation of Domain-adapted LLMs for Question-Answering in Telecom</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 10 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Knowledge Distillation (KD) is one of the approaches to reduce the size of Large Language Models (LLMs). A LLM with smaller number of model parameters (student) is trained to mimic the performance of a LLM of a larger size (teacher model) on a specific task. For domain-specific tasks, it is not clear if teacher or student model, or both, must be considered for domain adaptation. In this work, we study this problem from perspective of telecom domain Question-Answering (QA) task. We systematically experiment with Supervised Fine-tuning (SFT) of teacher only, SFT of student only and SFT of both prior to KD. We design experiments to study the impact of vocabulary (same and different) and KD algorithms (vanilla KD and Dual Space KD, DSKD) on the distilled model. Multi-faceted evaluation of the distillation using 14 different metrics (N-gram, embedding and LLM-based metrics) is considered. Experimental results show that SFT of teacher improves performance of distilled model when both models have same vocabulary, irrespective of algorithm and metrics. Overall, SFT of both teacher and student results in better performance across all metrics, although the statistical significance of the same depends on the vocabulary of the teacher models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19981v1">Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Achieving both accuracy and diverse reasoning remains challenging for Large Language Models (LLMs) in complex domains like mathematics. A key bottleneck is evaluating intermediate reasoning steps to guide generation without costly human annotations. To address this, we first introduce a novel Process Reward Model (PRM) trained automatically using Monte Carlo Tree Search coupled with a similarity-based data augmentation technique, effectively capturing step-level reasoning quality. Leveraging this PRM, we then adapt Generative Flow Networks (GFlowNets) to operate at the reasoning step level. Unlike traditional reinforcement learning focused on maximizing a single reward, GFlowNets naturally sample diverse, high-quality solutions proportional to their rewards, as measured by our PRM. Empirical evaluation shows strong improvements in both accuracy and solution diversity on challenging mathematical benchmarks (e.g., +2.59% absolute accuracy on MATH Level 5 for Llama3.2-3B), with effective generalization to unseen datasets (+9.4% absolute on SAT MATH). Our work demonstrates the potential of PRM-guided, step-level GFlowNets for developing more robust and versatile mathematical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19959v1">From Concept to Practice: an Automated LLM-aided UVM Machine for RTL Verification</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of the total development effort. While the Universal Verification Methodology (UVM) is widely used in industry to improve verification efficiency through structured and reusable testbenches, constructing these testbenches and generating sufficient stimuli remain challenging. These challenges arise from the considerable manual coding effort required, repetitive manual execution of multiple EDA tools, and the need for in-depth domain expertise to navigate complex designs.Here, we present UVM^2, an automated verification framework that leverages Large Language Models (LLMs) to generate UVM testbenches and iteratively refine them using coverage feedback, significantly reducing manual effort while maintaining rigorous verification standards.To evaluate UVM^2, we introduce a benchmark suite comprising Register Transfer Level (RTL) designs of up to 1.6K lines of code.The results show that UVM^2 reduces testbench setup time by up to UVM^2 compared to experienced engineers, and achieve average code and function coverage of 87.44% and 89.58%, outperforming state-of-the-art solutions by 20.96% and 23.51%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08775v3">What Should We Engineer in Prompts? Training Humans in Requirement-Driven LLM Use</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 15 pages; TOCHI 2025
    </div>
    <details class="paper-abstract">
      Prompting LLMs for complex tasks (e.g., building a trip advisor chatbot) needs humans to clearly articulate customized requirements (e.g., "start the response with a tl;dr"). However, existing prompt engineering instructions often lack focused training on requirement articulation and instead tend to emphasize increasingly automatable strategies (e.g., tricks like adding role-plays and "think step-by-step"). To address the gap, we introduce Requirement-Oriented Prompt Engineering (ROPE), a paradigm that focuses human attention on generating clear, complete requirements during prompting. We implement ROPE through an assessment and training suite that provides deliberate practice with LLM-generated feedback. In a randomized controlled experiment with 30 novices, ROPE significantly outperforms conventional prompt engineering training (20% vs. 1% gains), a gap that automatic prompt optimization cannot close. Furthermore, we demonstrate a direct correlation between the quality of input requirements and LLM outputs. Our work paves the way to empower more end-users to build complex LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11690v3">ID-Free Not Risk-Free: LLM-Powered Agents Unveil Risks in ID-Free Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Recent advances in ID-free recommender systems have attracted significant attention for effectively addressing the cold start problem. However, their vulnerability to malicious attacks remains largely unexplored. In this paper, we unveil a critical yet overlooked risk: LLM-powered agents can be strategically deployed to attack ID-free recommenders, stealthily promoting low-quality items in black-box settings. This attack exploits a novel rewriting-based deception strategy, where malicious agents synthesize deceptive textual descriptions by simulating the characteristics of popular items. To achieve this, the attack mechanism integrates two primary components: (1) a popularity extraction component that captures essential characteristics of popular items and (2) a multi-agent collaboration mechanism that enables iterative refinement of promotional textual descriptions through independent thinking and team discussion. To counter this risk, we further introduce a detection method to identify suspicious text generated by our discovered attack. By unveiling this risk, our work aims to underscore the urgent need to enhance the security of ID-free recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19898v1">GenCLS++: Pushing the Boundaries of Generative Classification in LLMs Through Comprehensive SFT and RL Studies Across Diverse Datasets</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      As a fundamental task in machine learning, text classification plays a crucial role in many areas. With the rapid scaling of Large Language Models (LLMs), particularly through reinforcement learning (RL), there is a growing need for more capable discriminators. Consequently, advances in classification are becoming increasingly vital for enhancing the overall capabilities of LLMs. Traditional discriminative methods map text to labels but overlook LLMs' intrinsic generative strengths. Generative classification addresses this by prompting the model to directly output labels. However, existing studies still rely on simple SFT alone, seldom probing the interplay between training and inference prompts, and no work has systematically leveraged RL for generative text classifiers and unified SFT, RL, and inference-time prompting in one framework. We bridge this gap with GenCLS++, a framework that jointly optimizes SFT and RL while systematically exploring five high-level strategy dimensions-in-context learning variants, category definitions, explicit uncertainty labels, semantically irrelevant numeric labels, and perplexity-based decoding-during both training and inference. After an SFT "policy warm-up," we apply RL with a simple rule-based reward, yielding sizable extra gains. Across seven datasets, GenCLS++ achieves an average accuracy improvement of 3.46% relative to the naive SFT baseline; on public datasets, this improvement rises to 4.00%. Notably, unlike reasoning-intensive tasks that benefit from explicit thinking processes, we find that classification tasks perform better without such reasoning steps. These insights into the role of explicit reasoning provide valuable guidance for future LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19867v1">semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 18 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Existing large language model (LLM) serving systems fall into two categories: 1) a unified system where prefill phase and decode phase are co-located on the same GPU, sharing the unified computational resource and storage, and 2) a disaggregated system where the two phases are disaggregated to different GPUs. The design of the disaggregated system addresses the latency interference and sophisticated scheduling issues in the unified system but leads to storage challenges including 1) replicated weights for both phases that prevent flexible deployment, 2) KV cache transfer overhead between the two phases, 3) storage imbalance that causes substantial wasted space of the GPU capacity, and 4) suboptimal resource adjustment arising from the difficulties in migrating KV cache. Such storage inefficiency delivers poor serving performance under high request rates. In this paper, we identify that the advantage of the disaggregated system lies in the disaggregated computation, i.e., partitioning the computational resource to enable the asynchronous computation of two phases. Thus, we propose a novel LLM serving system, semi-PD, characterized by disaggregated computation and unified storage. In semi-PD, we introduce a computation resource controller to achieve disaggregated computation at the streaming multi-processor (SM) level, and a unified memory manager to manage the asynchronous memory access from both phases. semi-PD has a low-overhead resource adjustment mechanism between the two phases, and a service-level objective (SLO) aware dynamic partitioning algorithm to optimize the SLO attainment. Compared to state-of-the-art systems, semi-PD maintains lower latency at higher request rates, reducing the average end-to-end latency per request by 1.27-2.58x on DeepSeek series models, and serves 1.55-1.72x more requests adhering to latency constraints on Llama series models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19838v1">LLM-Powered GUI Agents in Phone Automation: Surveying Progress and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 37 pages, 10 figures, 7 tables, Project Homepage: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </div>
    <details class="paper-abstract">
      With the rapid rise of large language models (LLMs), phone automation has undergone transformative changes. This paper systematically reviews LLM-driven phone GUI agents, highlighting their evolution from script-based automation to intelligent, adaptive systems. We first contextualize key challenges, (i) limited generality, (ii) high maintenance overhead, and (iii) weak intent comprehension, and show how LLMs address these issues through advanced language understanding, multimodal perception, and robust decision-making. We then propose a taxonomy covering fundamental agent frameworks (single-agent, multi-agent, plan-then-act), modeling approaches (prompt engineering, training-based), and essential datasets and benchmarks. Furthermore, we detail task-specific architectures, supervised fine-tuning, and reinforcement learning strategies that bridge user intent and GUI operations. Finally, we discuss open challenges such as dataset diversity, on-device deployment efficiency, user-centric adaptation, and security concerns, offering forward-looking insights into this rapidly evolving field. By providing a structured overview and identifying pressing research gaps, this paper serves as a definitive reference for researchers and practitioners seeking to harness LLMs in designing scalable, user-friendly phone GUI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19793v1">Prompt Injection Attack to Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Tool selection is a key component of LLM agents. The process operates through a two-step mechanism - \emph{retrieval} and \emph{selection} - to pick the most appropriate tool from a tool library for a given task. In this work, we introduce \textit{ToolHijacker}, a novel prompt injection attack targeting tool selection in no-box scenarios. ToolHijacker injects a malicious tool document into the tool library to manipulate the LLM agent's tool selection process, compelling it to consistently choose the attacker's malicious tool for an attacker-chosen target task. Specifically, we formulate the crafting of such tool documents as an optimization problem and propose a two-phase optimization strategy to solve it. Our extensive experimental evaluation shows that ToolHijacker is highly effective, significantly outperforming existing manual-based and automated prompt injection attacks when applied to tool selection. Moreover, we explore various defenses, including prevention-based defenses (StruQ and SecAlign) and detection-based defenses (known-answer detection, perplexity detection, and perplexity windowed detection). Our experimental results indicate that these defenses are insufficient, highlighting the urgent need for developing new defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03111v2">Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75\% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19759v1">Moral Reasoning Across Languages: The Critical Role of Low-Resource Languages in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 5 pages, 2 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce the Multilingual Moral Reasoning Benchmark (MMRB) to evaluate the moral reasoning abilities of large language models (LLMs) across five typologically diverse languages and three levels of contextual complexity: sentence, paragraph, and document. Our results show moral reasoning performance degrades with increasing context complexity, particularly for low-resource languages such as Vietnamese. We further fine-tune the open-source LLaMA-3-8B model using curated monolingual data for alignment and poisoning. Surprisingly, low-resource languages have a stronger impact on multilingual reasoning than high-resource ones, highlighting their critical role in multilingual NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04178v2">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 Accepted by SIGIR2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), known for their comprehension capabilities and extensive knowledge, have been increasingly applied to recommendation systems (RS). Given the fundamental gap between the mechanism of LLMs and the requirement of RS, researchers have focused on fine-tuning LLMs with recommendation-specific data to enhance their performance. Language Modeling Loss (LML), originally designed for language generation tasks, is commonly adopted. However, we identify two critical limitations of LML: 1) it exhibits significant divergence from the recommendation objective; 2) it erroneously treats all fictitious item descriptions as negative samples, introducing misleading training signals. To address these limitations, we propose a novel Masked Softmax Loss (MSL) tailored for fine-tuning LLMs on recommendation. MSL improves LML by identifying and masking invalid tokens that could lead to fictitious item descriptions during loss computation. This strategy can effectively avoid the interference from erroneous negative signals and ensure well alignment with the recommendation objective supported by theoretical guarantees. During implementation, we identify a potential challenge related to gradient vanishing of MSL. To overcome this, we further introduce the temperature coefficient and propose an Adaptive Temperature Strategy (ATS) that adaptively adjusts the temperature without requiring extensive hyperparameter tuning. Extensive experiments conducted on four public datasets further validate the effectiveness of MSL, achieving an average improvement of 42.24% in NDCG@10. The code is available at https://github.com/WANGBohaO-jpg/MSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19746v1">FineQ: Software-Hardware Co-Design for Low-Bit Fine-Grained Mixed-Precision Quantization of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 DATE 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced the natural language processing paradigm but impose substantial demands on memory and computational resources. Quantization is one of the most effective ways to reduce memory consumption of LLMs. However, advanced single-precision quantization methods experience significant accuracy degradation when quantizing to ultra-low bits. Existing mixed-precision quantization methods are quantized by groups with coarse granularity. Employing high precision for group data leads to substantial memory overhead, whereas low precision severely impacts model accuracy. To address this issue, we propose FineQ, software-hardware co-design for low-bit fine-grained mixed-precision quantization of LLMs. First, FineQ partitions the weights into finer-grained clusters and considers the distribution of outliers within these clusters, thus achieving a balance between model accuracy and memory overhead. Then, we propose an outlier protection mechanism within clusters that uses 3 bits to represent outliers and introduce an encoding scheme for index and data concatenation to enable aligned memory access. Finally, we introduce an accelerator utilizing temporal coding that effectively supports the quantization algorithm while simplifying the multipliers in the systolic array. FineQ achieves higher model accuracy compared to the SOTA mixed-precision quantization algorithm at a close average bit-width. Meanwhile, the accelerator achieves up to 1.79x energy efficiency and reduces the area of the systolic array by 61.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19734v1">LLM-Assisted Automated Deductive Coding of Dialogue Data: Leveraging Dialogue-Specific Characteristics to Enhance Contextual Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Dialogue data has been a key source for understanding learning processes, offering critical insights into how students engage in collaborative discussions and how these interactions shape their knowledge construction. The advent of Large Language Models (LLMs) has introduced promising opportunities for advancing qualitative research, particularly in the automated coding of dialogue data. However, the inherent contextual complexity of dialogue presents unique challenges for these models, especially in understanding and interpreting complex contextual information. This study addresses these challenges by developing a novel LLM-assisted automated coding approach for dialogue data. The novelty of our proposed framework is threefold: 1) We predict the code for an utterance based on dialogue-specific characteristics -- communicative acts and communicative events -- using separate prompts following the role prompts and chain-of-thoughts methods; 2) We engaged multiple LLMs including GPT-4-turbo, GPT-4o, DeepSeek in collaborative code prediction; 3) We leveraged the interrelation between events and acts to implement consistency checking using GPT-4o. In particular, our contextual consistency checking provided a substantial accuracy improvement. We also found the accuracy of act predictions was consistently higher than that of event predictions. This study contributes a new methodological framework for enhancing the precision of automated coding of dialogue data as well as offers a scalable solution for addressing the contextual challenges inherent in dialogue analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19730v1">Evaluate-and-Purify: Fortifying Code Language Models Against Adversarial Attacks Using LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 25 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of code language models in software engineering tasks has exposed vulnerabilities to adversarial attacks, especially the identifier substitution attacks. Although existing identifier substitution attackers demonstrate high success rates, they often produce adversarial examples with unnatural code patterns. In this paper, we systematically assess the quality of adversarial examples using LLM-as-a-Judge. Our analysis reveals that over 80% of adversarial examples generated by state-of-the-art identifier substitution attackers (e.g., ALERT) are actually detectable. Based on this insight, we propose EP-Shield, a unified framework for evaluating and purifying identifier substitution attacks via naturalness-aware reasoning. Specifically, we first evaluate the naturalness of code and identify the perturbed adversarial code, then purify it so that the victim model can restore correct prediction. Extensive experiments demonstrate the superiority of EP-Shield over adversarial fine-tuning (up to 83.36% improvement) and its lightweight design 7B parameters) with GPT-4-level performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03108v2">SoK: Knowledge is All You Need: Accelerating Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at https://anonymous.4open.science/r/PIDS-with-LLM-613B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19720v1">Taming the Titans: A Survey of Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 work in progress;11 pages of main paper with 7 main figures, overall 20 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) for Generative AI have achieved remarkable progress, evolving into sophisticated and versatile tools widely adopted across various domains and applications. However, the substantial memory overhead caused by their vast number of parameters, combined with the high computational demands of the attention mechanism, poses significant challenges in achieving low latency and high throughput for LLM inference services. Recent advancements, driven by groundbreaking research, have significantly accelerated progress in this field. This paper provides a comprehensive survey of these methods, covering fundamental instance-level approaches, in-depth cluster-level strategies, emerging scenario directions, and other miscellaneous but important areas. At the instance level, we review model placement, request scheduling, decoding length prediction, storage management, and the disaggregation paradigm. At the cluster level, we explore GPU cluster deployment, multi-instance load balancing, and cloud service solutions. For emerging scenarios, we organize the discussion around specific tasks, modules, and auxiliary methods. To ensure a holistic overview, we also highlight several niche yet critical areas. Finally, we outline potential research directions to further advance the field of LLM inference serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05999v2">LLMPot: Dynamically Configured LLM-based Honeypot for Industrial Protocol and Physical Process Emulation</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19678v1">From LLM Reasoning to Autonomous AI Agents: A Comprehensive Review</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Large language models and autonomous AI agents have evolved rapidly, resulting in a diverse array of evaluation benchmarks, frameworks, and collaboration protocols. However, the landscape remains fragmented and lacks a unified taxonomy or comprehensive survey. Therefore, we present a side-by-side comparison of benchmarks developed between 2019 and 2025 that evaluate these models and agents across multiple domains. In addition, we propose a taxonomy of approximately 60 benchmarks that cover general and academic knowledge reasoning, mathematical problem-solving, code generation and software engineering, factual grounding and retrieval, domain-specific evaluations, multimodal and embodied tasks, task orchestration, and interactive assessments. Furthermore, we review AI-agent frameworks introduced between 2023 and 2025 that integrate large language models with modular toolkits to enable autonomous decision-making and multi-step reasoning. Moreover, we present real-world applications of autonomous AI agents in materials science, biomedical research, academic ideation, software engineering, synthetic data generation, chemical reasoning, mathematical problem-solving, geographic information systems, multimedia, healthcare, and finance. We then survey key agent-to-agent collaboration protocols, namely the Agent Communication Protocol (ACP), the Model Context Protocol (MCP), and the Agent-to-Agent Protocol (A2A). Finally, we discuss recommendations for future research, focusing on advanced reasoning strategies, failure modes in multi-agent LLM systems, automated scientific discovery, dynamic tool integration via reinforcement learning, integrated search capabilities, and security vulnerabilities in agent protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19394v1">LLMs for Engineering: Teaching Models to Design High Powered Rockets</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software engineering, but their application to physical engineering domains remains underexplored. This paper evaluates LLMs' capabilities in high-powered rocketry design through RocketBench, a benchmark connecting LLMs to high-fidelity rocket simulations. We test models on two increasingly complex design tasks: target altitude optimization and precision landing challenges. Our findings reveal that while state-of-the-art LLMs demonstrate strong baseline engineering knowledge, they struggle to iterate on their designs when given simulation results and ultimately plateau below human performance levels. However, when enhanced with reinforcement learning (RL), we show that a 7B parameter model outperforms both SoTA foundation models and human experts. This research demonstrates that RL-trained LLMs can serve as effective tools for complex engineering optimization, potentially transforming engineering domains beyond software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19384v1">From Inductive to Deductive: LLMs-Based Qualitative Data Analysis in Requirements Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Requirements Engineering (RE) is essential for developing complex and regulated software projects. Given the challenges in transforming stakeholder inputs into consistent software designs, Qualitative Data Analysis (QDA) provides a systematic approach to handling free-form data. However, traditional QDA methods are time-consuming and heavily reliant on manual effort. In this paper, we explore the use of Large Language Models (LLMs), including GPT-4, Mistral, and LLaMA-2, to improve QDA tasks in RE. Our study evaluates LLMs' performance in inductive (zero-shot) and deductive (one-shot, few-shot) annotation tasks, revealing that GPT-4 achieves substantial agreement with human analysts in deductive settings, with Cohen's Kappa scores exceeding 0.7, while zero-shot performance remains limited. Detailed, context-rich prompts significantly improve annotation accuracy and consistency, particularly in deductive scenarios, and GPT-4 demonstrates high reliability across repeated runs. These findings highlight the potential of LLMs to support QDA in RE by reducing manual effort while maintaining annotation quality. The structured labels automatically provide traceability of requirements and can be directly utilized as classes in domain models, facilitating systematic software design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06443v2">Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen widespread adoption due to their remarkable natural language capabilities. However, when deploying them in real-world settings, it is important to align LLMs to generate texts according to acceptable human standards. Methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO) have enabled significant progress in refining LLMs using human preference data. However, the privacy concerns inherent in utilizing such preference data have yet to be adequately studied. In this paper, we investigate the vulnerability of LLMs aligned using two widely used methods - DPO and PPO - to membership inference attacks (MIAs). Our study has two main contributions: first, we theoretically motivate that DPO models are more vulnerable to MIA compared to PPO models; second, we introduce a novel reference-based attack framework specifically for analyzing preference data called PREMIA (\uline{Pre}ference data \uline{MIA}). Using PREMIA and existing baselines we empirically show that DPO models have a relatively heightened vulnerability towards MIA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15077v2">Think2SQL: Reinforce LLM Reasoning Capabilities for Text2SQL</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 17 pages, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities in transforming natural language questions about relational databases into SQL queries. Despite recent improvements, small LLMs struggle to handle questions involving multiple tables and complex SQL patterns under a Zero-Shot Learning (ZSL) setting. Supervised Fine-Tuning (SFT) partially compensates for the knowledge deficits in pretrained models but falls short while dealing with queries involving multi-hop reasoning. To bridge this gap, different LLM training strategies to reinforce reasoning capabilities have been proposed, ranging from leveraging a thinking process within ZSL, including reasoning traces in SFT, or adopt Reinforcement Learning (RL) strategies. However, the influence of reasoning on Text2SQL performance is still largely unexplored. This paper investigates to what extent LLM reasoning capabilities influence their Text2SQL performance on four benchmark datasets. To this end, it considers the following LLM settings: (1) ZSL, including general-purpose reasoning or not; (2) SFT, with and without task-specific reasoning traces; (3) RL, exploring the use of different rewarding functions, both the established EXecution accuracy (EX) and a mix with fine-grained ones that also account the precision, recall, and cardinality of partially correct answers; (4) SFT+RL, i.e, a two-stage approach that combines SFT and RL. The results show that general-purpose reasoning under ZSL proves to be ineffective in tackling complex Text2SQL cases. Small LLMs benefit from SFT with reasoning much more than larger ones. RL is generally beneficial across all tested models and datasets. The use of the fine-grained metrics turns out to be the most effective RL strategy. Thanks to RL and the novel text2SQL rewards, the 7B Qwen-Coder-2.5 model performs on par with 400+ Billion ones (including gpt-4o) on the Bird dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19254v1">Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 UQLM repository: https://github.com/cvs-health/uqlm
    </div>
    <details class="paper-abstract">
      Hallucinations are a persistent problem with Large Language Models (LLMs). As these models become increasingly used in high-stakes domains, such as healthcare and finance, the need for effective hallucination detection is crucial. To this end, we propose a versatile framework for zero-resource hallucination detection that practitioners can apply to real-world use cases. To achieve this, we adapt a variety of existing uncertainty quantification (UQ) techniques, including black-box UQ, white-box UQ, and LLM-as-a-Judge, transforming them as necessary into standardized response-level confidence scores ranging from 0 to 1. To enhance flexibility, we introduce a tunable ensemble approach that incorporates any combination of the individual confidence scores. This approach enables practitioners to optimize the ensemble for a specific use case for improved performance. To streamline implementation, the full suite of scorers is offered in this paper's companion Python toolkit, UQLM. To evaluate the performance of the various scorers, we conduct an extensive set of experiments using several LLM question-answering benchmarks. We find that our tunable ensemble typically surpasses its individual components and outperforms existing hallucination detection methods. Our results demonstrate the benefits of customized hallucination detection strategies for improving the accuracy and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12098v2">Gauging Overprecision in LLMs: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Recently, overconfidence in large language models (LLMs) has garnered considerable attention due to its fundamental importance in quantifying the trustworthiness of LLM generation. However, existing approaches prompt the \textit{black box LLMs} to produce their confidence (\textit{verbalized confidence}), which can be subject to many biases and hallucinations. Inspired by a different aspect of overconfidence in cognitive science called \textit{overprecision}, we designed a framework for its study in black box LLMs. This framework contains three main phases: 1) generation, 2) refinement and 3) evaluation. In the generation phase we prompt the LLM to generate answers to numerical questions in the form of intervals with a certain level of confidence. This confidence level is imposed in the prompt and not required for the LLM to generate as in previous approaches. We use various prompting techniques and use the same prompt multiple times to gauge the effects of randomness in the generation process. In the refinement phase, answers from the previous phase are refined to generate better answers. The LLM answers are evaluated and studied in the evaluation phase to understand its internal workings. This study allowed us to gain various insights into LLM overprecision: 1) LLMs are highly uncalibrated for numerical tasks 2) there is no correlation between the length of the interval and the imposed confidence level, which can be symptomatic of a a) lack of understanding of the concept of confidence or b) inability to adjust self-confidence by following instructions, {3) LLM numerical precision differs depending on the task, scale of answer and prompting technique 4) Refinement of answers doesn't improve precision in most cases. We believe this study offers new perspectives on LLM overconfidence and serves as a strong baseline for overprecision in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v3">RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving information from the relevant knowledge database, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21223v2">Rethinking Graph Structure Learning in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 29 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Recently, the emergence of LLMs has prompted researchers to integrate language descriptions into graphs, aiming to enhance model encoding capabilities from a data-centric perspective. This graph representation is called text-attributed graphs (TAGs). A review of prior advancements highlights that graph structure learning (GSL) is a pivotal technique for improving data utility, making it highly relevant to efficient TAG learning. However, most GSL methods are tailored for traditional graphs without textual information, underscoring the necessity of developing a new GSL paradigm. Despite clear motivations, it remains challenging: (1) How can we define a reasonable optimization objective for GSL in the era of LLMs, considering the massive parameters in LLM? (2) How can we design an efficient model architecture that enables seamless integration of LLM for this optimization objective? For Question 1, we reformulate existing GSL optimization objectives as a tree optimization framework, shifting the focus from obtaining a well-trained edge predictor to a language-aware tree sampler. For Question 2, we propose decoupled and training-free model design principles for LLM integration, shifting the focus from computation-intensive fine-tuning to more efficient inference. Based on this, we propose Large Language and Tree Assistant (LLaTA), which leverages tree-based LLM in-context learning to enhance the understanding of topology and text, enabling reliable inference and generating improved graph structure. Extensive experiments on 10 datasets demonstrate that LLaTA enjoys flexibility-incorporated with any backbone; scalability-outperforms other LLM-enhanced graph learning methods; effectiveness-achieves SOTA predictive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12790v2">Quantum-Enhanced LLM Efficient Fine Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) enables efficient fine-tuning of pre-trained language models through low-rank matrix approximation, achieving effectiveness in many scenarios. However, its representation capacity is constrained in complex tasks or high-rank dependency settings, potentially limiting model adaptability. To overcome the expressive bottleneck in classical low-rank approximation for fine-tuning large language models (LLMs), we propose Quantum Tensor Hybrid Adaptation (QTHA), a parameter-efficient fine-tuning method that integrates a quantum neural network (QNN) with a tensor network. QTHA explores quantum tensor hybrid fine-tuning within low-rank spaces by decomposing pre-trained weights into quantum neural network and tensor network representations, leveraging quantum state superposition to overcome classical rank limitations. Experiments demonstrate that QTHA achieves performance comparable to or surpassing LoRA in parameter-efficient fine-tuning. Compared to LoRA, QTHA reduces trainable parameters by 76% while reducing training loss by up to 17% and improving test set performance by up to 17% within the same training steps. This research not only enables lightweight adaptation of quantum resources to the billion-parameter models but also validates the feasibility of quantum hardware optimization driven by LLM tasks. It establishes the first engineering-ready foundation for future quantum-enhanced Artificial General Intelligence (AGI) systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10069v4">A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 TMLR (camera-ready). Details on https://openreview.net/forum?id=x9VQFjtOPS
    </div>
    <details class="paper-abstract">
      LLM test-time compute (or LLM inference) via search has emerged as a promising research area with rapid developments. However, current frameworks often adopt distinct perspectives on three key aspects: task definition, LLM profiling, and search procedures, making direct comparisons challenging. Moreover, the search algorithms employed often diverge from standard implementations, and their specific characteristics are not thoroughly specified. This survey aims to provide a comprehensive but integrated technical review on existing LIS frameworks. Specifically, we unify task definitions under Markov Decision Process (MDP) and provides modular definitions of LLM profiling and search procedures. The definitions enable precise comparisons of various LLM inference frameworks while highlighting their departures from conventional search algorithms. We also discuss the applicability, performance, and efficiency of these methods. For ongoing paper updates, please refer to our GitHub repository: https://github.com/xinzhel/LLM-Search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19162v1">SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 Project: https://chen-judge.github.io/SPC/
    </div>
    <details class="paper-abstract">
      Evaluating the step-by-step reliability of large language model (LLM) reasoning, such as Chain-of-Thought, remains challenging due to the difficulty and cost of obtaining high-quality step-level supervision. In this paper, we introduce Self-Play Critic (SPC), a novel approach where a critic model evolves its ability to assess reasoning steps through adversarial self-play games, eliminating the need for manual step-level annotation. SPC involves fine-tuning two copies of a base model to play two roles, namely a "sneaky generator" that deliberately produces erroneous steps designed to be difficult to detect, and a "critic" that analyzes the correctness of reasoning steps. These two models engage in an adversarial game in which the generator aims to fool the critic, while the critic model seeks to identify the generator's errors. Using reinforcement learning based on the game outcomes, the models iteratively improve; the winner of each confrontation receives a positive reward and the loser receives a negative reward, driving continuous self-evolution. Experiments on three reasoning process benchmarks (ProcessBench, PRM800K, DeltaBench) demonstrate that our SPC progressively enhances its error detection capabilities (e.g., accuracy increases from 70.8% to 77.7% on ProcessBench) and surpasses strong baselines, including distilled R1 model. Furthermore, applying SPC to guide the test-time search of diverse LLMs significantly improves their mathematical reasoning performance on MATH500 and AIME2024, outperforming state-of-the-art process reward models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19144v1">ChiseLLM: Unleashing the Power of Reasoning LLMs for Chisel Agile Hardware Development</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      The growing demand for Domain-Specific Architecture (DSA) has driven the development of Agile Hardware Development Methodology (AHDM). Hardware Construction Language (HCL) like Chisel offers high-level abstraction features, making it an ideal language for HCL-Based AHDM. While Large Language Models (LLMs) excel in code generation tasks, they still face challenges with Chisel generation, particularly regarding syntax correctness and design variability. Recent reasoning models have significantly enhanced code generation capabilities through test-time scaling techniques. However, we found that reasoning models without domain adaptation cannot bring substantial benefits to Chisel code generation tasks. This paper presents ChiseLLM, a solution comprising data processing and transformation, prompt-guided reasoning trace synthesis, and domain-adapted model training. We constructed high-quality datasets from public RTL code resources and guided the model to adopt structured thinking patterns through prompt enhancement methods. Experiments demonstrate that our ChiseLLM-7B and ChiseLLM-32B models improved syntax correctness by 18.85% and 26.32% respectively over base models, while increasing variability design ability by 47.58% compared to baseline reasoning models. Our datasets and models are publicly available, providing high-performance, cost-effective models for HCL-Based AHDM, and offering an effective baseline for future research. Github repository: https://github.com/observerw/ChiseLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19099v1">VeriDebug: A Unified LLM for Verilog Debugging via Contrastive Embedding and Guided Correction</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in debugging for various programming languages. However, the application of LLMs to Verilog debugging remains insufficiently explored. Here, we present VeriDebug, an approach that integrates contrastive representation and guided correction capabilities for automated Verilog debugging. Unlike existing methods, VeriDebug employs an embedding-based technique to accurately retrieve internal information, followed by bug-fixing. VeriDebug unifies Verilog bug detection and correction through a shared parameter space. By simultaneously learning bug patterns and fixes, it streamlines debugging via contrastive embedding and guided correction. Empirical results show the efficacy of VeriDebug in enhancing Verilog debugging. Our VeriDebugLoc, Type model achieves 64.7 accuracy in bug fixing (Acc1), a significant improvement from the existing open-source SOTAs 11.3. This performance not only outperforms open-source alternatives but also exceeds larger closed-source models like GPT-3.5-turbo (36.6), offering a more accurate alternative to conventional debugging methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19093v1">CipherBank: Exploring the Boundary of LLM Reasoning Capabilities through Cryptography Challenges</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities, especially the recent advancements in reasoning, such as o1 and o3, pushing the boundaries of AI. Despite these impressive achievements in mathematics and coding, the reasoning abilities of LLMs in domains requiring cryptographic expertise remain underexplored. In this paper, we introduce CipherBank, a comprehensive benchmark designed to evaluate the reasoning capabilities of LLMs in cryptographic decryption tasks. CipherBank comprises 2,358 meticulously crafted problems, covering 262 unique plaintexts across 5 domains and 14 subdomains, with a focus on privacy-sensitive and real-world scenarios that necessitate encryption. From a cryptographic perspective, CipherBank incorporates 3 major categories of encryption methods, spanning 9 distinct algorithms, ranging from classical ciphers to custom cryptographic techniques. We evaluate state-of-the-art LLMs on CipherBank, e.g., GPT-4o, DeepSeek-V3, and cutting-edge reasoning-focused models such as o1 and DeepSeek-R1. Our results reveal significant gaps in reasoning abilities not only between general-purpose chat LLMs and reasoning-focused LLMs but also in the performance of current reasoning-focused models when applied to classical cryptographic decryption tasks, highlighting the challenges these models face in understanding and manipulating encrypted data. Through detailed analysis and error investigations, we provide several key observations that shed light on the limitations and potential improvement areas for LLMs in cryptographic reasoning. These findings underscore the need for continuous advancements in LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19076v1">LLM-Evaluation Tropes: Perspectives on the Validity of LLM-Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to evaluate information retrieval (IR) systems, generating relevance judgments traditionally made by human assessors. Recent empirical studies suggest that LLM-based evaluations often align with human judgments, leading some to suggest that human judges may no longer be necessary, while others highlight concerns about judgment reliability, validity, and long-term impact. As IR systems begin incorporating LLM-generated signals, evaluation outcomes risk becoming self-reinforcing, potentially leading to misleading conclusions. This paper examines scenarios where LLM-evaluators may falsely indicate success, particularly when LLM-based judgments influence both system development and evaluation. We highlight key risks, including bias reinforcement, reproducibility challenges, and inconsistencies in assessment methodologies. To address these concerns, we propose tests to quantify adverse effects, guardrails, and a collaborative framework for constructing reusable test collections that integrate LLM judgments responsibly. By providing perspectives from academia and industry, this work aims to establish best practices for the principled use of LLMs in IR evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21032v1">Selecting the Right LLM for eGov Explanations</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 8 pages, 7 figures. ICEDEG 2025, Bern, Switzerland, June 2025
    </div>
    <details class="paper-abstract">
      The perceived quality of the explanations accompanying e-government services is key to gaining trust in these institutions, consequently amplifying further usage of these services. Recent advances in generative AI, and concretely in Large Language Models (LLMs) allow the automation of such content articulations, eliciting explanations' interpretability and fidelity, and more generally, adapting content to various audiences. However, selecting the right LLM type for this has become a non-trivial task for e-government service providers. In this work, we adapted a previously developed scale to assist with this selection, providing a systematic approach for the comparative analysis of the perceived quality of explanations generated by various LLMs. We further demonstrated its applicability through the tax-return process, using it as an exemplar use case that could benefit from employing an LLM to generate explanations about tax refund decisions. This was attained through a user study with 128 survey respondents who were asked to rate different versions of LLM-generated explanations about tax refund decisions, providing a methodological basis for selecting the most appropriate LLM. Recognizing the practical challenges of conducting such a survey, we also began exploring the automation of this process by attempting to replicate human feedback using a selection of cutting-edge predictive techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19076v1">LLM-Evaluation Tropes: Perspectives on the Validity of LLM-Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-04-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to evaluate information retrieval (IR) systems, generating relevance judgments traditionally made by human assessors. Recent empirical studies suggest that LLM-based evaluations often align with human judgments, leading some to suggest that human judges may no longer be necessary, while others highlight concerns about judgment reliability, validity, and long-term impact. As IR systems begin incorporating LLM-generated signals, evaluation outcomes risk becoming self-reinforcing, potentially leading to misleading conclusions. This paper examines scenarios where LLM-evaluators may falsely indicate success, particularly when LLM-based judgments influence both system development and evaluation. We highlight key risks, including bias reinforcement, reproducibility challenges, and inconsistencies in assessment methodologies. To address these concerns, we propose tests to quantify adverse effects, guardrails, and a collaborative framework for constructing reusable test collections that integrate LLM judgments responsibly. By providing perspectives from academia and industry, this work aims to establish best practices for the principled use of LLMs in IR evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12098v2">Gauging Overprecision in LLMs: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-04-27
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Recently, overconfidence in large language models (LLMs) has garnered considerable attention due to its fundamental importance in quantifying the trustworthiness of LLM generation. However, existing approaches prompt the \textit{black box LLMs} to produce their confidence (\textit{verbalized confidence}), which can be subject to many biases and hallucinations. Inspired by a different aspect of overconfidence in cognitive science called \textit{overprecision}, we designed a framework for its study in black box LLMs. This framework contains three main phases: 1) generation, 2) refinement and 3) evaluation. In the generation phase we prompt the LLM to generate answers to numerical questions in the form of intervals with a certain level of confidence. This confidence level is imposed in the prompt and not required for the LLM to generate as in previous approaches. We use various prompting techniques and use the same prompt multiple times to gauge the effects of randomness in the generation process. In the refinement phase, answers from the previous phase are refined to generate better answers. The LLM answers are evaluated and studied in the evaluation phase to understand its internal workings. This study allowed us to gain various insights into LLM overprecision: 1) LLMs are highly uncalibrated for numerical tasks 2) there is no correlation between the length of the interval and the imposed confidence level, which can be symptomatic of a a) lack of understanding of the concept of confidence or b) inability to adjust self-confidence by following instructions, {3) LLM numerical precision differs depending on the task, scale of answer and prompting technique 4) Refinement of answers doesn't improve precision in most cases. We believe this study offers new perspectives on LLM overconfidence and serves as a strong baseline for overprecision in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19044v1">Calibrating Translation Decoding with Quality Estimation on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Neural machine translation (NMT) systems typically employ maximum a posteriori (MAP) decoding to select the highest-scoring translation from the distribution mass. However, recent evidence highlights the inadequacy of MAP decoding, often resulting in low-quality or even pathological hypotheses -- the decoding objective is not aligned with real-world translation quality. This paper proposes calibrating hypothesis likelihoods with translation quality from a distribution view by directly optimizing their Pearson correlation -- thereby enhancing the effectiveness of translation decoding. With our method, translation on large language models (LLMs) improves substantially after limited training (2K instances per direction). This improvement is orthogonal to those achieved through supervised fine-tuning, leading to substantial gains across a broad range of metrics and human evaluations -- even when applied to top-performing translation-specialized LLMs fine-tuned on high-quality translation data, such as Tower, or when compared to recent preference optimization methods, like CPO. Moreover, the calibrated translation likelihood can directly serve as a strong proxy for translation quality, closely approximating or even surpassing some state-of-the-art translation quality estimation models, like CometKiwi. Lastly, our in-depth analysis demonstrates that calibration enhances the effectiveness of MAP decoding, thereby enabling greater efficiency in real-world deployment. The resulting state-of-the-art translation model, which covers 10 languages, along with the accompanying code and human evaluation data, has been released to the community: https://github.com/moore3930/calibrating-llm-mt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19037v1">"I Would Have Written My Code Differently'': Beginners Struggle to Understand LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 To appear in 33rd ACM International Conference on the Foundations of Software Engineering (FSE Companion '25), June 23-28, 2025, Trondheim, Norway
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being increasingly adopted for programming work. Prior work shows that while LLMs accelerate task completion for professional programmers, beginning programmers struggle to prompt models effectively. However, prompting is just half of the code generation process -- when code is generated, it must be read, evaluated, and integrated (or rejected). How accessible are these tasks for beginning programmers? This paper measures how well beginners comprehend LLM-generated code and explores the challenges students face in judging code correctness. We compare how well students understand natural language descriptions of functions and LLM-generated implementations, studying 32 CS1 students on 160 task instances. Our results show a low per-task success rate of 32.5\%, with indiscriminate struggles across demographic populations. Key challenges include barriers for non-native English speakers, unfamiliarity with Python syntax, and automation bias. Our findings highlight the barrier that code comprehension presents to beginning programmers seeking to write code with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19019v1">Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-26
      | 💬 19 pages, 1 figure, 6 tables
    </div>
    <details class="paper-abstract">
      The challenge of ensuring Large Language Models (LLMs) align with societal standards is of increasing interest, as these models are still prone to adversarial jailbreaks that bypass their safety mechanisms. Identifying these vulnerabilities is crucial for enhancing the robustness of LLMs against such exploits. We propose Graph of ATtacks (GoAT), a method for generating adversarial prompts to test the robustness of LLM alignment using the Graph of Thoughts framework [Besta et al., 2024]. GoAT excels at generating highly effective jailbreak prompts with fewer queries to the victim model than state-of-the-art attacks, achieving up to five times better jailbreak success rate against robust models like Llama. Notably, GoAT creates high-quality, human-readable prompts without requiring access to the targeted model's parameters, making it a black-box attack. Unlike approaches constrained by tree-based reasoning, GoAT's reasoning is based on a more intricate graph structure. By making simultaneous attack paths aware of each other's progress, this dynamic framework allows a deeper integration and refinement of reasoning paths, significantly enhancing the collaborative exploration of adversarial vulnerabilities in LLMs. At a technical level, GoAT starts with a graph structure and iteratively refines it by combining and improving thoughts, enabling synergy between different thought paths. The code for our implementation can be found at: https://github.com/GoAT-pydev/Graph_of_Attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17130v2">Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control</a></div>
    <div class="paper-meta">
      📅 2025-04-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector. Our code is publicly available at: https://github.com/hannahxchen/llm-censorship-steering
    </details>
</div>
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
