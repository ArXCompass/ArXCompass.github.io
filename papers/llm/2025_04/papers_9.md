# llm - 2025_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06511v1">GTS-LUM: Reshaping User Behavior Modeling with LLMs in Telecommunications Industry</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      As telecommunication service providers shifting their focus to analyzing user behavior for package design and marketing interventions, a critical challenge lies in developing a unified, end-to-end framework capable of modeling long-term and periodic user behavior sequences with diverse time granularities, multi-modal data inputs, and heterogeneous labels. This paper introduces GTS-LUM, a novel user behavior model that redefines modeling paradigms in telecommunication settings. GTS-LUM adopts a (multi-modal) encoder-adapter-LLM decoder architecture, enhanced with several telecom-specific innovations. Specifically, the model incorporates an advanced timestamp processing method to handle varying time granularities. It also supports multi-modal data inputs -- including structured tables and behavior co-occurrence graphs -- and aligns these with semantic information extracted by a tokenizer using a Q-former structure. Additionally, GTS-LUM integrates a front-placed target-aware mechanism to highlight historical behaviors most relevant to the target. Extensive experiments on industrial dataset validate the effectiveness of this end-to-end framework and also demonstrate that GTS-LUM outperforms LLM4Rec approaches which are popular in recommendation systems, offering an effective and generalizing solution for user behavior modeling in telecommunications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11871v4">Generative AI Voting: Fair Collective Choice is Resilient to LLM Biases and Inconsistencies</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 23 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Scaling up deliberative and voting participation is a longstanding endeavor -- a cornerstone for direct democracy and legitimate collective choice. Recent breakthroughs in generative artificial intelligence (AI) and large language models (LLMs) unravel new capabilities for AI personal assistants to overcome cognitive bandwidth limitations of humans, providing decision support or even direct representation of human voters at large scale. However, the quality of this representation and what underlying biases manifest when delegating collective decision-making to LLMs is an alarming and timely challenge to tackle. By rigorously emulating with high realism more than >50K LLM voting personas in 306 real-world voting elections, we disentangle the nature of different biases in LLMS (GPT 3, GPT 3.5, and Llama2). Complex preferential ballot formats exhibit significant inconsistencies compared to simpler majoritarian elections that show higher consistency. Strikingly though, by demonstrating for the first time in real-world a proportional representation of voters in direct democracy, we are also able to show that fair ballot aggregation methods, such as equal shares, prove to be a win-win: fairer voting outcomes for humans with fairer AI representation, especially for voters who are likely to abstain. This novel underlying relationship proves paramount for democratic resilience in progressives scenarios with low voters turnout and voter fatigue supported by AI representatives: abstained voters are mitigated by recovering highly representative voting outcomes that are fairer. These interdisciplinary insights provide remarkable foundations for science, policymakers, and citizens to develop safeguards and resilience for AI risks in democratic innovations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06265v2">GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can encode complex relationships in their latent spaces, yet harnessing them for optimization under uncertainty remains challenging. We address this gap with a novel architecture that reframes LLM finetuning as Gaussian process (GP) marginal likelihood optimization via deep kernel methods. We introduce LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both - LLMs to provide a rich and flexible input space for Bayesian optimization and - GPs to model this space with predictive uncertainty for more efficient sampling. Applied to Buchwald-Hartwig reaction optimization, our method nearly doubles the discovery rate of high-performing reactions compared to static LLM embeddings (from 24% to 43% coverage of the top 5% reactions in just 50 optimization iterations). We also observe a 14% improvement over domain-specific representations without requiring specialized features. Extensive empirical evaluation across 19 benchmarks - ranging from general chemistry to reaction and molecular property optimization - demonstrates our method's robustness, generality, and consistent improvements across: (1) tasks, (2) LLM architectures (encoder, decoder, encoder-decoder), (3) pretraining domains (chemistry-related or general-purpose) and (4) hyperparameter settings (tuned once on a single dataset). Finally, we explain these improvements: joint LLM-GP optimization through marginal likelihood implicitly performs contrastive learning, aligning representations to produce (1) better-structured embedding spaces, (2) improved uncertainty calibration, and (3) more efficient sampling - without requiring any external loss. This work provides both practical advances in sample-efficient optimization and insights into what makes effective Bayesian optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07336v1">Zeus: Zero-shot LLM Instruction for Union Segmentation in Multimodal Medical Imaging</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 21 pages, 4 figures, In Press by a journal
    </div>
    <details class="paper-abstract">
      Medical image segmentation has achieved remarkable success through the continuous advancement of UNet-based and Transformer-based foundation backbones. However, clinical diagnosis in the real world often requires integrating domain knowledge, especially textual information. Conducting multimodal learning involves visual and text modalities shown as a solution, but collecting paired vision-language datasets is expensive and time-consuming, posing significant challenges. Inspired by the superior ability in numerous cross-modal tasks for Large Language Models (LLMs), we proposed a novel Vision-LLM union framework to address the issues. Specifically, we introduce frozen LLMs for zero-shot instruction generation based on corresponding medical images, imitating the radiology scanning and report generation process. {To better approximate real-world diagnostic processes}, we generate more precise text instruction from multimodal radiology images (e.g., T1-w or T2-w MRI and CT). Based on the impressive ability of semantic understanding and rich knowledge of LLMs. This process emphasizes extracting special features from different modalities and reunion the information for the ultimate clinical diagnostic. With generated text instruction, our proposed union segmentation framework can handle multimodal segmentation without prior collected vision-language datasets. To evaluate our proposed method, we conduct comprehensive experiments with influential baselines, the statistical results and the visualized case study demonstrate the superiority of our novel method.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07303v1">Modeling Response Consistency in Multi-Agent LLM Systems: A Comparative Analysis of Shared and Separate Context Approaches</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized in multi-agent systems (MAS) to enhance collaborative problem-solving and interactive reasoning. Recent advancements have enabled LLMs to function as autonomous agents capable of understanding complex interactions across multiple topics. However, deploying LLMs in MAS introduces challenges related to context management, response consistency, and scalability, especially when agents must operate under memory limitations and handle noisy inputs. While prior research has explored optimizing context sharing and response latency in LLM-driven MAS, these efforts often focus on either fully centralized or decentralized configurations, each with distinct trade-offs. In this paper, we develop a probabilistic framework to analyze the impact of shared versus separate context configurations on response consistency and response times in LLM-based MAS. We introduce the Response Consistency Index (RCI) as a metric to evaluate the effects of context limitations, noise, and inter-agent dependencies on system performance. Our approach differs from existing research by focusing on the interplay between memory constraints and noise management, providing insights into optimizing scalability and response times in environments with interdependent topics. Through this analysis, we offer a comprehensive understanding of how different configurations impact the efficiency of LLM-driven multi-agent systems, thereby guiding the design of more robust architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21934v3">Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Recent math benchmarks for large language models (LLMs) such as MathArena indicate that state-of-the-art reasoning models achieve impressive performance on mathematical competitions like AIME, with the leading model, Gemini-2.5-Pro, achieving scores comparable to top human competitors. However, these benchmarks evaluate models solely based on final numerical answers, neglecting rigorous reasoning and proof generation which are essential for real-world mathematical tasks. To address this, we introduce the first comprehensive evaluation of full-solution reasoning for challenging mathematical problems. Using expert human annotators, we evaluated several state-of-the-art reasoning models on the six problems from the 2025 USAMO within hours of their release. Our results reveal that all tested models struggled significantly: only Gemini-2.5-Pro achieves a non-trivial score of 25%, while all other models achieve less than 5%. Through detailed analysis of reasoning traces, we identify the most common failure modes and find several unwanted artifacts arising from the optimization strategies employed during model training. Overall, our results suggest that current LLMs are inadequate for rigorous mathematical reasoning tasks, highlighting the need for substantial improvements in reasoning and proof generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07278v1">A Multi-Phase Analysis of Blood Culture Stewardship: Machine Learning Prediction, Expert Recommendation Assessment, and LLM Automation</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 10 pages, 2 figures, 2 tables, conference
    </div>
    <details class="paper-abstract">
      Blood cultures are often over ordered without clear justification, straining healthcare resources and contributing to inappropriate antibiotic use pressures worsened by the global shortage. In study of 135483 emergency department (ED) blood culture orders, we developed machine learning (ML) models to predict the risk of bacteremia using structured electronic health record (EHR) data and provider notes via a large language model (LLM). The structured models AUC improved from 0.76 to 0.79 with note embeddings and reached 0.81 with added diagnosis codes. Compared to an expert recommendation framework applied by human reviewers and an LLM-based pipeline, our ML approach offered higher specificity without compromising sensitivity. The recommendation framework achieved sensitivity 86%, specificity 57%, while the LLM maintained high sensitivity (96%) but over classified negatives, reducing specificity (16%). These findings demonstrate that ML models integrating structured and unstructured data can outperform consensus recommendations, enhancing diagnostic stewardship beyond existing standards of care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07199v1">SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 10 pages, 4 figures, Accepted as SemEval 2025 Task 5 description paper
    </div>
    <details class="paper-abstract">
      We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13453v3">Adaptive Augmentation Policy Optimization with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 15 pages, 4 tables, 3 figures submitted for consideration to 2025 Medical Image Understanding and Analysis Conference (MIUA)
    </div>
    <details class="paper-abstract">
      Data augmentation is a critical component of deep learning pipelines, enhancing model generalization by increasing dataset diversity. Traditional augmentation strategies rely on manually designed transformations, stochastic sampling, or automated search-based approaches. Although automated methods improve performance, they often require extensive computational resources and are tailored to specific datasets. In this work, we propose a Large Language Model (LLM)-guided augmentation optimization strategy that refines augmentation policies based on model performance feedback. We introduce two approaches: (1) LLM-Guided Augmentation Policy Optimization, where augmentation policies are selected by an LLM prior to training and iteratively refined across multiple training cycles, and (2) Adaptive LLM-Guided Augmentation Policy Optimization, where policies adapt in real-time based on performance metrics. This in-training approach eliminates the need for full model retraining before receiving LLM feedback, thereby reducing computational costs while improving performance. Our methodology employs an LLM to dynamically select augmentation transformations based on dataset characteristics, model architecture, and prior training outcomes. Unlike traditional search-based methods, our approach leverages the contextual knowledge of LLMs, particularly in specialized domains like medical imaging, to recommend augmentation strategies tailored to domain-specific data. We evaluate our approach on multiple domain-specific image classification datasets where augmentation is key to model robustness. Results show that LLM-guided augmentation optimization outperforms traditional methods, improving model accuracy. These findings highlight the potential of LLMs in automating and adapting deep learning training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08002v1">More diverse more adaptive: Comprehensive Multi-task Learning for Improved LLM Domain Adaptation in E-commerce</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 Accepted by KDD workshop 2024
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have been widely applied across various domains due to their powerful domain adaptation capabilities. Previous studies have suggested that diverse, multi-modal data can enhance LLMs' domain adaptation performance. However, this hypothesis remains insufficiently validated in the e-commerce sector. To address this gap, we propose a comprehensive e-commerce multi-task framework and design empirical experiments to examine the impact of diverse data and tasks on LLMs from two perspectives: "capability comprehensiveness" and "task comprehensiveness." Specifically, we observe significant improvements in LLM performance by progressively introducing tasks related to new major capability areas and by continuously adding subtasks within different major capability domains. Furthermore, we observe that increasing model capacity amplifies the benefits of diversity, suggesting a synergistic relationship between model capacity and data diversity. Finally, we validate the best-performing model from our empirical experiments in the KDD Cup 2024, achieving a rank 5 in Task 1. This outcome demonstrates the significance of our research for advancing LLMs in the e-commerce domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08820v1">CAReDiO: Cultural Alignment of LLM via Representativeness and Distinctiveness Guided Data Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) more deeply integrate into human life across various regions, aligning them with pluralistic cultures is crucial for improving user experience and mitigating cultural conflicts. Existing approaches develop culturally aligned LLMs primarily through fine-tuning with massive carefully curated culture-specific corpora. Nevertheless, inspired by culture theories, we identify two key challenges faced by these datasets: (1) Representativeness: These corpora fail to fully capture the target culture's core characteristics with redundancy, causing computation waste; (2) Distinctiveness: They struggle to distinguish the unique nuances of a given culture from shared patterns across other relevant ones, hindering precise cultural modeling. To handle these challenges, we introduce CAReDiO, a novel cultural data construction framework. Specifically, CAReDiO utilizes powerful LLMs to automatically generate cultural conversation data, where both the queries and responses are further optimized by maximizing representativeness and distinctiveness. Using CAReDiO, we construct a small yet effective dataset, covering five cultures, and compare it with several recent cultural corpora. Extensive experiments demonstrate that our method generates more effective data and enables cultural alignment with as few as 100 training samples, enhancing both performance and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08808v1">Exploring the Effectiveness and Interpretability of Texts in LLM-based Time Series Models</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been applied to time series forecasting tasks, leveraging pre-trained language models as the backbone and incorporating textual data to purportedly enhance the comprehensive capabilities of LLMs for time series. However, are these texts really helpful for interpretation? This study seeks to investigate the actual efficacy and interpretability of such textual incorporations. Through a series of empirical experiments on textual prompts and textual prototypes, our findings reveal that the misalignment between two modalities exists, and the textual information does not significantly improve time series forecasting performance in many cases. Furthermore, visualization analysis indicates that the textual representations learned by existing frameworks lack sufficient interpretability when applied to time series data. We further propose a novel metric named Semantic Matching Index (SMI) to better evaluate the matching degree between time series and texts during our post hoc interpretability investigation. Our analysis reveals the misalignment and limited interpretability of texts in current time-series LLMs, and we hope this study can raise awareness of the interpretability of texts for time series. The code is available at https://github.com/zachysun/TS-Lang-Exp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10509v1">Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion</a></div>
    <div class="paper-meta">
      📅 2025-04-09
    </div>
    <details class="paper-abstract">
      This study presents a comprehensive reproducibility and extension analysis of the Setwise prompting methodology for zero-shot ranking with Large Language Models (LLMs), as proposed by Zhuang et al. We evaluate its effectiveness and efficiency compared to traditional Pointwise, Pairwise, and Listwise approaches in document ranking tasks. Our reproduction confirms the findings of Zhuang et al., highlighting the trade-offs between computational efficiency and ranking effectiveness in Setwise methods. Building on these insights, we introduce Setwise Insertion, a novel approach that leverages the initial document ranking as prior knowledge, reducing unnecessary comparisons and uncertainty by focusing on candidates more likely to improve the ranking results. Experimental results across multiple LLM architectures (Flan-T5, Vicuna, and Llama) show that Setwise Insertion yields a 31% reduction in query time, a 23% reduction in model inferences, and a slight improvement in reranking effectiveness compared to the original Setwise method. These findings highlight the practical advantage of incorporating prior ranking knowledge into Setwise prompting for efficient and accurate zero-shot document reranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07097v1">Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 25 pages, 13 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07087v1">KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 To be presented at NAACL-HLT, KnowledgeNLP Workshop (2025)
    </div>
    <details class="paper-abstract">
      Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.07015v1">LLM-IFT: LLM-Powered Information Flow Tracking for Secure Hardware</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 This paper is presented at IEEE VLSI Test Symposium (VTS) 2025
    </div>
    <details class="paper-abstract">
      As modern hardware designs grow in complexity and size, ensuring security across the confidentiality, integrity, and availability (CIA) triad becomes increasingly challenging. Information flow tracking (IFT) is a widely-used approach to tracing data propagation, identifying unauthorized activities that may compromise confidentiality or/and integrity in hardware. However, traditional IFT methods struggle with scalability and adaptability, particularly in high-density and interconnected architectures, leading to tracing bottlenecks that limit applicability in large-scale hardware. To address these limitations and show the potential of transformer-based models in integrated circuit (IC) design, this paper introduces LLM-IFT that integrates large language models (LLM) for the realization of the IFT process in hardware. LLM-IFT exploits LLM-driven structured reasoning to perform hierarchical dependency analysis, systematically breaking down even the most complex designs. Through a multi-step LLM invocation, the framework analyzes both intra-module and inter-module dependencies, enabling comprehensive IFT assessment. By focusing on a set of Trust-Hub vulnerability test cases at both the IP level and the SoC level, our experiments demonstrate a 100\% success rate in accurate IFT analysis for confidentiality and integrity checks in hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.07087v1">KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-09
      | 💬 To be presented at NAACL-HLT, KnowledgeNLP Workshop (2025)
    </div>
    <details class="paper-abstract">
      Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06426v1">S'MoRE: Structural Mixture of Residual Experts for LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained large language models (LLMs) presents a dual challenge of balancing parameter efficiency and model capacity. Existing methods like low-rank adaptations (LoRA) are efficient but lack flexibility, while Mixture-of-Experts (MoE) architectures enhance model capacity at the cost of more & under-utilized parameters. To address these limitations, we propose Structural Mixture of Residual Experts (S'MoRE), a novel framework that seamlessly integrates the efficiency of LoRA with the flexibility of MoE. Specifically, S'MoRE employs hierarchical low-rank decomposition of expert weights, yielding residuals of varying orders interconnected in a multi-layer structure. By routing input tokens through sub-trees of residuals, S'MoRE emulates the capacity of many experts by instantiating and assembling just a few low-rank matrices. We craft the inter-layer propagation of S'MoRE's residuals as a special type of Graph Neural Network (GNN), and prove that under similar parameter budget, S'MoRE improves "structural flexibility" of traditional MoE (or Mixture-of-LoRA) by exponential order. Comprehensive theoretical analysis and empirical results demonstrate that S'MoRE achieves superior fine-tuning performance, offering a transformative approach for efficient LLM adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03038v3">Towards Federated RLHF with Aggregated Client Preference for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 ICLR'25
    </div>
    <details class="paper-abstract">
      Reinforcement learning with human feedback (RLHF) fine-tunes a pretrained large language model (LLM) using user preference data, enabling it to generate content aligned with human preferences. However, due to privacy concerns, users may be reluctant to share sensitive preference data. To address this, we propose utilizing Federated Learning (FL) techniques, allowing large-scale preference collection from diverse real-world users without requiring them to transmit data to a central server. Our federated RLHF methods (i.e., FedBis and FedBiscuit) encode each client's preferences into binary selectors and aggregate them to capture common preferences. In particular, FedBiscuit overcomes key challenges, such as preference heterogeneity and reward hacking, through innovative solutions like grouping clients with similar preferences to reduce heterogeneity and using multiple binary selectors to enhance LLM output quality. To evaluate the performance of the proposed methods, we establish the first federated RLHF benchmark with a heterogeneous human preference dataset. Experimental results show that by integrating the LLM with aggregated client preferences, FedBis and FedBiscuit significantly enhance the professionalism and readability of the generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06356v1">Query Understanding in LLM-based Conversational Information Seeking</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 WWW'25 Tutorial
    </div>
    <details class="paper-abstract">
      Query understanding in Conversational Information Seeking (CIS) involves accurately interpreting user intent through context-aware interactions. This includes resolving ambiguities, refining queries, and adapting to evolving information needs. Large Language Models (LLMs) enhance this process by interpreting nuanced language and adapting dynamically, improving the relevance and precision of search results in real-time. In this tutorial, we explore advanced techniques to enhance query understanding in LLM-based CIS systems. We delve into LLM-driven methods for developing robust evaluation metrics to assess query understanding quality in multi-turn interactions, strategies for building more interactive systems, and applications like proactive query management and query reformulation. We also discuss key challenges in integrating LLMs for query understanding in conversational search systems and outline future research directions. Our goal is to deepen the audience's understanding of LLM-based conversational query understanding and inspire discussions to drive ongoing advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06324v1">From Stability to Inconsistency: A Study of Moral Preferences in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly integrate into our daily lives, it becomes crucial to understand their implicit biases and moral tendencies. To address this, we introduce a Moral Foundations LLM dataset (MFD-LLM) grounded in Moral Foundations Theory, which conceptualizes human morality through six core foundations. We propose a novel evaluation method that captures the full spectrum of LLMs' revealed moral preferences by answering a range of real-world moral dilemmas. Our findings reveal that state-of-the-art models have remarkably homogeneous value preferences, yet demonstrate a lack of consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06323v1">Mosaic: Composite Projection Pruning for Resource-efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Extensive compute and memory requirements limit the deployment of large language models (LLMs) on any hardware. Compression methods, such as pruning, can reduce model size, which in turn reduces resource requirements. State-of-the-art pruning is based on coarse-grained methods. They are time-consuming and inherently remove critical model parameters, adversely impacting the quality of the pruned model. This paper introduces projection pruning, a novel fine-grained method for pruning LLMs. In addition, LLM projection pruning is enhanced by a new approach we refer to as composite projection pruning - the synergistic combination of unstructured pruning that retains accuracy and structured pruning that reduces model size. We develop Mosaic, a novel system to create and deploy pruned LLMs using composite projection pruning. Mosaic is evaluated using a range of performance and quality metrics on multiple hardware platforms, LLMs, and datasets. Mosaic is 7.19x faster in producing models than existing approaches. Mosaic models achieve up to 84.2% lower perplexity and 31.4% higher accuracy than models obtained from coarse-grained pruning. Up to 67% faster inference and 68% lower GPU memory use is noted for Mosaic models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06319v1">Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit pronounced memory-bound characteristics during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In this paper, we propose an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM inference through computation-load overlap. By strategically scheduling idle memory bandwidth during active computation windows, our method proactively prefetches required KV Cache into GPU L2 cache, enabling high-speed L2 cache hits for subsequent accesses and effectively hiding HBM access latency within computational cycles. Extensive experiments on NVIDIA H20 GPUs demonstrate that the proposed method achieves 2.15x improvement in attention kernel efficiency and up to 1.97x end-to-end throughput enhancement, surpassing state-of-the-art baseline FlashAttention-3. Notably, our solution maintains orthogonality to existing optimization techniques and can be integrated with current inference frameworks, providing a scalable latency-hiding solution for next-generation LLM inference engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06265v1">GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can encode complex relationships in their latent spaces, yet harnessing them for optimization under uncertainty remains challenging. We address this gap with a novel architecture that reframes LLM finetuning as Gaussian process (GP) marginal likelihood optimization via deep kernel methods. We introduce LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both - LLMs to provide a rich and flexible input space for Bayesian optimization and - GPs to model this space with predictive uncertainty for more efficient sampling. Applied to Buchwald-Hartwig reaction optimization, our method nearly doubles the discovery rate of high-performing reactions compared to static LLM embeddings (from 24% to 43% coverage of the top 5% reactions in just 50 optimization iterations). We also observe a 14% improvement over domain-specific representations without requiring specialized features. Extensive empirical evaluation across 19 benchmarks - ranging from general chemistry to reaction and molecular property optimization - demonstrates our method's robustness, generality, and consistent improvements across: (1) tasks, (2) LLM architectures (encoder, decoder, encoder-decoder), (3) pretraining domains (chemistry-related or general-purpose) and (4) hyperparameter settings (tuned once on a single dataset). Finally, we explain these improvements: joint LLM-GP optimization through marginal likelihood implicitly performs contrastive learning, aligning representations to produce (1) better-structured embedding spaces, (2) improved uncertainty calibration, and (3) more efficient sampling - without requiring any external loss. This work provides both practical advances in sample-efficient optimization and insights into what makes effective Bayesian optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06261v1">Hogwild! Inference: Parallel LLM Generation via Concurrent Attention</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's partial progress in the concurrent cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's generated tokens. Hogwild! inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22250v2">Modeling Challenging Patient Interactions: LLMs for Medical Communication Training</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Effective patient communication is pivotal in healthcare, yet traditional medical training often lacks exposure to diverse, challenging interpersonal dynamics. To bridge this gap, this study proposes the use of Large Language Models (LLMs) to simulate authentic patient communication styles, specifically the "accuser" and "rationalizer" personas derived from the Satir model, while also ensuring multilingual applicability to accommodate diverse cultural contexts and enhance accessibility for medical professionals. Leveraging advanced prompt engineering, including behavioral prompts, author's notes, and stubbornness mechanisms, we developed virtual patients (VPs) that embody nuanced emotional and conversational traits. Medical professionals evaluated these VPs, rating their authenticity (accuser: $3.8 \pm 1.0$; rationalizer: $3.7 \pm 0.8$ on a 5-point Likert scale (from one to five)) and correctly identifying their styles. Emotion analysis revealed distinct profiles: the accuser exhibited pain, anger, and distress, while the rationalizer displayed contemplation and calmness, aligning with predefined, detailed patient description including medical history. Sentiment scores (on a scale from zero to nine) further validated these differences in the communication styles, with the accuser adopting negative ($3.1 \pm 0.6$) and the rationalizer more neutral ($4.0 \pm 0.4$) tone. These results underscore LLMs' capability to replicate complex communication styles, offering transformative potential for medical education. This approach equips trainees to navigate challenging clinical scenarios by providing realistic, adaptable patient interactions, enhancing empathy and diagnostic acumen. Our findings advocate for AI-driven tools as scalable, cost-effective solutions to cultivate nuanced communication skills, setting a foundation for future innovations in healthcare training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02879v2">Position: LLM Unlearning Benchmarks are Weak Measures of Progress</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Appears in IEEE Secure and Trustworthy Machine Learning (SaTML) '25
    </div>
    <details class="paper-abstract">
      Unlearning methods have the potential to improve the privacy and safety of large language models (LLMs) by removing sensitive or harmful information post hoc. The LLM unlearning research community has increasingly turned toward empirical benchmarks to assess the effectiveness of such methods. In this paper, we find that existing benchmarks provide an overly optimistic and potentially misleading view on the effectiveness of candidate unlearning methods. By introducing simple, benign modifications to a number of popular benchmarks, we expose instances where supposedly unlearned information remains accessible, or where the unlearning process has degraded the model's performance on retained information to a much greater extent than indicated by the original benchmark. We identify that existing benchmarks are particularly vulnerable to modifications that introduce even loose dependencies between the forget and retain information. Further, we show that ambiguity in unlearning targets in existing benchmarks can easily lead to the design of methods that overfit to the given test queries. Based on our findings, we urge the community to be cautious when interpreting benchmark results as reliable measures of progress, and we provide several recommendations to guide future LLM unlearning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15341v3">GenoTEX: An LLM Agent Benchmark for Automated Gene Expression Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 31 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in machine learning have significantly improved the identification of disease-associated genes from gene expression datasets. However, these processes often require extensive expertise and manual effort, limiting their scalability. Large Language Model (LLM)-based agents have shown promise in automating these tasks due to their increasing problem-solving abilities. To support the evaluation and development of such methods, we introduce GenoTEX, a benchmark dataset for the automated analysis of gene expression data. GenoTEX provides analysis code and results for solving a wide range of gene-trait association problems, encompassing dataset selection, preprocessing, and statistical analysis, in a pipeline that follows computational genomics standards. The benchmark includes expert-curated annotations from bioinformaticians to ensure accuracy and reliability. To provide baselines for these tasks, we present GenoAgent, a team of LLM-based agents that adopt a multi-step programming workflow with flexible self-correction, to collaboratively analyze gene expression datasets. Our experiments demonstrate the potential of LLM-based methods in analyzing genomic data, while error analysis highlights the challenges and areas for future improvement. We propose GenoTEX as a promising resource for benchmarking and enhancing automated methods for gene expression data analysis. The benchmark is available at https://github.com/Liu-Hy/GenoTEX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06219v1">Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      The increasing adoption of web crawling opt-outs by copyright holders of online content raises critical questions about the impact of data compliance on large language model (LLM) performance. However, little is known about how these restrictions (and the resultant filtering of pretraining datasets) affect the capabilities of models trained using these corpora. In this work, we conceptualize this effect as the $\textit{data compliance gap}$ (DCG), which quantifies the performance difference between models trained on datasets that comply with web crawling opt-outs, and those that do not. We measure the data compliance gap in two settings: pretraining models from scratch and continual pretraining from existing compliant models (simulating a setting where copyrighted data could be integrated later in pretraining). Our experiments with 1.5B models show that, as of January 2025, compliance with web data opt-outs does not degrade general knowledge acquisition (close to 0\% DCG). However, in specialized domains such as biomedical research, excluding major publishers leads to performance declines. These findings suggest that while general-purpose LLMs can be trained to perform equally well using fully open data, performance in specialized domains may benefit from access to high-quality copyrighted sources later in training. Our study provides empirical insights into the long-debated trade-off between data compliance and downstream model performance, informing future discussions on AI training practices and policy decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06196v1">TxGemma: Efficient and Agentic LLMs for Therapeutics</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Therapeutic development is a costly and high-risk endeavor that is often plagued by high failure rates. To address this, we introduce TxGemma, a suite of efficient, generalist large language models (LLMs) capable of therapeutic property prediction as well as interactive reasoning and explainability. Unlike task-specific models, TxGemma synthesizes information from diverse sources, enabling broad application across the therapeutic development pipeline. The suite includes 2B, 9B, and 27B parameter models, fine-tuned from Gemma-2 on a comprehensive dataset of small molecules, proteins, nucleic acids, diseases, and cell lines. Across 66 therapeutic development tasks, TxGemma achieved superior or comparable performance to the state-of-the-art generalist model on 64 (superior on 45), and against state-of-the-art specialist models on 50 (superior on 26). Fine-tuning TxGemma models on therapeutic downstream tasks, such as clinical trial adverse event prediction, requires less training data than fine-tuning base LLMs, making TxGemma suitable for data-limited applications. Beyond these predictive capabilities, TxGemma features conversational models that bridge the gap between general LLMs and specialized property predictors. These allow scientists to interact in natural language, provide mechanistic reasoning for predictions based on molecular structure, and engage in scientific discussions. Building on this, we further introduce Agentic-Tx, a generalist therapeutic agentic system powered by Gemini 2.5 that reasons, acts, manages diverse workflows, and acquires external domain knowledge. Agentic-Tx surpasses prior leading models on the Humanity's Last Exam benchmark (Chemistry & Biology) with 52.3% relative improvement over o3-mini (high) and 26.7% over o3-mini (high) on GPQA (Chemistry) and excels with improvements of 6.3% (ChemBench-Preference) and 2.4% (ChemBench-Mini) over o3-mini (high).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06160v1">Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06143v1">ARLO: A Tailorable Approach for Transforming Natural Language Software Requirements into Architecture using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Software requirements expressed in natural language (NL) frequently suffer from verbosity, ambiguity, and inconsistency. This creates a range of challenges, including selecting an appropriate architecture for a system and assessing different architectural alternatives. Relying on human expertise to accomplish the task of mapping NL requirements to architecture is time-consuming and error-prone. This paper proposes ARLO, an approach that automates this task by leveraging (1) a set of NL requirements for a system, (2) an existing standard that specifies architecturally relevant software quality attributes, and (3) a readily available Large Language Model (LLM). Specifically, ARLO determines the subset of NL requirements for a given system that is architecturally relevant and maps that subset to a tailorable matrix of architectural choices. ARLO applies integer linear programming on the architectural-choice matrix to determine the optimal architecture for the current requirements. We demonstrate ARLO's efficacy using a set of real-world examples. We highlight ARLO's ability (1) to trace the selected architectural choices to the requirements and (2) to isolate NL requirements that exert a particular influence on a system's architecture. This allows the identification, comparative assessment, and exploration of alternative architectural choices based on the requirements and constraints expressed therein.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06095v1">Nonuniform-Tensor-Parallelism: Mitigating GPU failure impact for Scaled-up LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      LLM training is scaled up to 10Ks of GPUs by a mix of data-(DP) and model-parallel (MP) execution. Critical to achieving efficiency is tensor-parallel (TP; a form of MP) execution within tightly-coupled subsets of GPUs, referred to as a scale-up domain, and the larger the scale-up domain the better the performance. New datacenter architectures are emerging with more GPUs able to be tightly-coupled in a scale-up domain, such as moving from 8 GPUs to 72 GPUs connected via NVLink. Unfortunately, larger scale-up domains increase the blast-radius of failures, with a failure of single GPU potentially impacting TP execution on the full scale-up domain, which can degrade overall LLM training throughput dramatically. With as few as 0.1% of GPUs being in a failed state, a high TP-degree job can experience nearly 10% reduction in LLM training throughput. We propose nonuniform-tensor-parallelism (NTP) to mitigate this amplified impact of GPU failures. In NTP, a DP replica that experiences GPU failures operates at a reduced TP degree, contributing throughput equal to the percentage of still-functional GPUs. We also propose a rack-design with improved electrical and thermal capabilities in order to sustain power-boosting of scale-up domains that have experienced failures; combined with NTP, this can allow the DP replica with the reduced TP degree (i.e., with failed GPUs) to keep up with the others, thereby achieving near-zero throughput loss for large-scale LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09516v3">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06006v1">Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning?</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Optimal hyperparameter selection is critical for maximizing neural network performance, especially as models grow in complexity. This work investigates the viability of using large language models (LLMs) for hyperparameter optimization by employing a fine-tuned version of Code Llama. Through parameter-efficient fine-tuning using LoRA, we adapt the LLM to generate accurate and efficient hyperparameter recommendations tailored to diverse neural network architectures. Unlike traditional methods such as Optuna, which rely on exhaustive trials, the proposed approach achieves competitive or superior results in terms of Root Mean Square Error (RMSE) while significantly reducing computational overhead. Our approach highlights that LLM-based optimization not only matches state-of-the-art methods like Tree-structured Parzen Estimators but also accelerates the tuning process. This positions LLMs as a promising alternative to conventional optimization techniques, particularly for rapid experimentation. Furthermore, the ability to generate hyperparameters in a single inference step makes this method particularly well-suited for resource-constrained environments such as edge devices and mobile applications, where computational efficiency is paramount. The results confirm that LLMs, beyond their efficiency, offer substantial time savings and comparable stability, underscoring their value in advancing machine learning workflows. All generated hyperparameters are included in the LEMUR Neural Network (NN) Dataset, which is publicly available and serves as an open-source benchmark for hyperparameter optimization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05995v1">NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has raised concerns about cultural bias, fairness, and their applicability in diverse linguistic and underrepresented regional contexts. To enhance and benchmark the capabilities of LLMs, there is a need to develop large-scale resources focused on multilingual, local, and cultural contexts. In this study, we propose a framework, NativQA, that can seamlessly construct large-scale, culturally and regionally aligned QA datasets in native languages. The framework utilizes user-defined seed queries and leverages search engines to collect location-specific, everyday information. It has been evaluated across 39 locations in 24 countries and in 7 languages, ranging from extremely low-resource to high-resource languages, which resulted over 300K Question Answer (QA) pairs. The developed resources can be used for LLM benchmarking and further fine-tuning. The framework has been made publicly available for the community (https://gitlab.com/nativqa/nativqa-framework).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05946v1">InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware Control</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Model Predictive Control~(MPC) is a powerful control strategy widely utilized in domains like energy management, building control, and autonomous systems. However, its effectiveness in real-world settings is challenged by the need to incorporate context-specific predictions and expert instructions, which traditional MPC often neglects. We propose \IMPC, a novel framework that addresses this gap by integrating real-time human instructions through a Large Language Model~(LLM) to produce context-aware predictions for MPC. Our method employs a Language-to-Distribution~(L2D) module to translate contextual information into predictive disturbance trajectories, which are then incorporated into the MPC optimization. Unlike existing context-aware and language-based MPC models, \IMPC enables dynamic human-LLM interaction and fine-tunes the L2D module in a closed loop with theoretical performance guarantees, achieving a regret bound of $O(\sqrt{T\log T})$ for linear dynamics when optimized via advanced fine-tuning methods such as Direct Preference Optimization~(DPO) using a tailored loss function.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08574v2">Themes of Building LLM-based Applications for Production: A Practitioner's View</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Background: Large language models (LLMs) have become a paramount interest of researchers and practitioners alike, yet a comprehensive overview of key considerations for those developing LLM-based systems is lacking. This study addresses this gap by collecting and mapping the topics practitioners discuss online, offering practical insights into where priorities lie in developing LLM-based applications. Method: We collected 189 videos from 2022 to 2024 from practitioners actively developing such systems and discussing various aspects they encounter during development and deployment of LLMs in production. We analyzed the transcripts using BERTopic, then manually sorted and merged the generated topics into themes, leading to a total of 20 topics in 8 themes. Results: The most prevalent topics fall within the theme Design & Architecture, with a strong focus on retrieval-augmented generation (RAG) systems. Other frequently discussed topics include model capabilities and enhancement techniques (e.g., fine-tuning, prompt engineering), infrastructure and tooling, and risks and ethical challenges. Implications: Our results highlight current discussions and challenges in deploying LLMs in production. This way, we provide a systematic overview of key aspects practitioners should be aware of when developing LLM-based applications. We further pale off topics of interest for academics where further research is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13453v2">Adaptive Augmentation Policy Optimization with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 15 pages, 4 tables, 3 figures submitted for consideration to 2025 Medical Image Understanding and Analysis Conference (MIUA)
    </div>
    <details class="paper-abstract">
      Data augmentation is a critical component of deep learning pipelines, enhancing model generalization by increasing dataset diversity. Traditional augmentation strategies rely on manually designed transformations, stochastic sampling, or automated search-based approaches. Although automated methods improve performance, they often require extensive computational resources and are tailored to specific datasets. In this work, we propose a Large Language Model (LLM)-guided augmentation optimization strategy that refines augmentation policies based on model performance feedback. We introduce two approaches: (1) LLM-Guided Augmentation Policy Optimization, where augmentation policies are selected by an LLM prior to training and iteratively refined across multiple training cycles, and (2) Adaptive LLM-Guided Augmentation Policy Optimization, where policies adapt in real-time based on performance metrics. This in-training approach eliminates the need for full model retraining before receiving LLM feedback, thereby reducing computational costs while improving performance. Our methodology employs an LLM to dynamically select augmentation transformations based on dataset characteristics, model architecture, and prior training outcomes. Unlike traditional search-based methods, our approach leverages the contextual knowledge of LLMs, particularly in specialized domains like medical imaging, to recommend augmentation strategies tailored to domain-specific data. We evaluate our approach on multiple domain-specific image classification datasets where augmentation is key to model robustness. Results show that LLM-guided augmentation optimization outperforms traditional methods, improving model accuracy. These findings highlight the potential of LLMs in automating and adapting deep learning training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05898v1">Assessing Thai Dialect Performance in LLMs with Automatic Benchmarks and Human Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Datasets and codes are available at https://github.com/mrpeerat/Thai_local_benchmark
    </div>
    <details class="paper-abstract">
      Large language models show promising results in various NLP tasks. Despite these successes, the robustness and consistency of LLMs in underrepresented languages remain largely unexplored, especially concerning local dialects. Existing benchmarks also focus on main dialects, neglecting LLMs' ability on local dialect texts. In this paper, we introduce a Thai local dialect benchmark covering Northern (Lanna), Northeastern (Isan), and Southern (Dambro) Thai, evaluating LLMs on five NLP tasks: summarization, question answering, translation, conversation, and food-related tasks. Furthermore, we propose a human evaluation guideline and metric for Thai local dialects to assess generation fluency and dialect-specific accuracy. Results show that LLM performance declines significantly in local Thai dialects compared to standard Thai, with only proprietary models like GPT-4o and Gemini2 demonstrating some fluency
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08424v3">Comparing Apples to Oranges: LLM-powered Multimodal Intention Prediction in an Object Categorization Task</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Published in the Proceedings of the 16th International Conference on Social Robotics (ICSR) 2024,15 pages,5 figures,2 tables; work was co-funded by Horizon Europe project TERAIS under Grant agreement number 101079338
    </div>
    <details class="paper-abstract">
      Human intention-based systems enable robots to perceive and interpret user actions to interact with humans and adapt to their behavior proactively. Therefore, intention prediction is pivotal in creating a natural interaction with social robots in human-designed environments. In this paper, we examine using Large Language Models (LLMs) to infer human intention in a collaborative object categorization task with a physical robot. We propose a novel multimodal approach that integrates user non-verbal cues, like hand gestures, body poses, and facial expressions, with environment states and user verbal cues to predict user intentions in a hierarchical architecture. Our evaluation of five LLMs shows the potential for reasoning about verbal and non-verbal user cues, leveraging their context-understanding and real-world knowledge to support intention prediction while collaborating on a task with a social robot. Video: https://youtu.be/tBJHfAuzohI
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17875v3">Understanding Layer Significance in LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) through supervised fine-tuning is essential for tailoring them to specific applications. Recent studies suggest that alignment primarily adjusts a model's presentation style rather than its foundational knowledge, indicating that only certain components of the model are significantly impacted. To uncover how alignment affects model behavior at a granular level, we propose identifying which layers within LLMs are most critical to the alignment process. Our approach, named ILA, involves learning a binary mask for the parameter changes in each layer during alignment, as an indicator of layer significance. Experimental results reveal that, despite substantial differences in alignment datasets, the important layers of a model identified by ILA exhibit nearly 90\% overlap, highlighting fundamental patterns in LLM alignment. The results also indicate that freezing non-essential layers improves overall model performance, while selectively tuning the most critical layers significantly enhances fine-tuning efficiency with minimal performance loss. Finally, we discuss how these findings extend from LLM alignment to reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05831v1">Leveraging Robust Optimization for LLM Alignment under Distribution Shifts</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly rely on preference alignment methods to steer outputs toward human values, yet these methods are often constrained by the scarcity of high-quality human-annotated data. To tackle this, recent approaches have turned to synthetic data generated by LLMs as a scalable alternative. However, synthetic data can introduce distribution shifts, compromising the nuanced human preferences that are essential for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment in the presence of such shifts. Our approach first estimates the likelihood ratios between the target and training distributions leveraging a learned classifier, then it minimizes the worst-case loss over data regions that reflect the target human-preferred distribution. By explicitly prioritizing the target distribution during optimization, our method mitigates the adverse effects of distributional variation and enhances the generation of responses that faithfully reflect human values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05812v1">Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Ongoing work
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated exceptional capabilities in challenging tasks such as mathematical reasoning, existing methods to enhance reasoning ability predominantly rely on supervised fine-tuning (SFT) followed by reinforcement learning (RL) on reasoning-specific data after pre-training. However, these approaches critically depend on external supervisions--such as human labelled reasoning traces, verified golden answers, or pre-trained reward models--which limits scalability and practical applicability. In this work, we propose Entropy Minimized Policy Optimization (EMPO), which makes an early attempt at fully unsupervised LLM reasoning incentivization. EMPO does not require any supervised information for incentivizing reasoning capabilities (i.e., neither verifiable reasoning traces, problems with golden answers, nor additional pre-trained reward models). By continuously minimizing the predictive entropy of LLMs on unlabeled user queries in a latent semantic space, EMPO enables purely self-supervised evolution of reasoning capabilities with strong flexibility and practicality. Our experiments demonstrate competitive performance of EMPO on both mathematical reasoning and free-form commonsense reasoning tasks. Specifically, without any supervised signals, EMPO boosts the accuracy of Qwen2.5-Math-7B Base from 30.7\% to 48.1\% on mathematical benchmarks and improves truthfulness accuracy of Qwen2.5-7B Instruct from 87.16\% to 97.25\% on TruthfulQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v2">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly contributing to the creation of content on the Internet. This creates a feedback loop as subsequent generations of models will be trained on this generated, synthetic data. This phenomenon is receiving increasing interest, in particular because previous studies have shown that it may lead to distribution shift - models misrepresent and forget the true underlying distributions of human data they are expected to approximate (e.g. resulting in a drastic loss of quality). In this study, we study the impact of human data properties on distribution shift dynamics in iterated training loops. We first confirm that the distribution shift dynamics greatly vary depending on the human data by comparing four datasets (two based on Twitter and two on Reddit). We then test whether data quality may influence the rate of this shift. We find that it does on the twitter, but not on the Reddit datasets. We then focus on a Reddit dataset and conduct a more exhaustive evaluation of a large set of dataset properties. This experiment associated lexical diversity with larger, and semantic diversity with smaller detrimental shifts, suggesting that incorporating text with high lexical (but limited semantic) diversity could exacerbate the degradation of generated text. We then focus on the evolution of political bias, and find that the type of shift observed (bias reduction, amplification or inversion) depends on the political lean of the human (true) distribution. Overall, our work extends the existing literature on the consequences of recursive fine-tuning by showing that this phenomenon is highly dependent on features of the human data on which training occurs. This suggests that different parts of internet (e.g. GitHub, Reddit) may undergo different types of shift depending on their properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05804v1">StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05786v1">How to Enable LLM with 3D Capacity? A Survey of Spatial Reasoning in LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      3D spatial understanding is essential in real-world applications such as robotics, autonomous vehicles, virtual reality, and medical imaging. Recently, Large Language Models (LLMs), having demonstrated remarkable success across various domains, have been leveraged to enhance 3D understanding tasks, showing potential to surpass traditional computer vision methods. In this survey, we present a comprehensive review of methods integrating LLMs with 3D spatial understanding. We propose a taxonomy that categorizes existing methods into three branches: image-based methods deriving 3D understanding from 2D visual data, point cloud-based methods working directly with 3D representations, and hybrid modality-based methods combining multiple data streams. We systematically review representative methods along these categories, covering data representations, architectural modifications, and training strategies that bridge textual and 3D modalities. Finally, we discuss current limitations, including dataset scarcity and computational challenges, while highlighting promising research directions in spatial perception, multi-modal fusion, and real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05764v1">Layer-Aware Embedding Fusion for LLMs in Text Classifications</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 11 pages, 3 figures, Preprint
    </div>
    <details class="paper-abstract">
      Embedding fusion has emerged as an effective approach for enhancing performance across various NLP tasks. However, systematic guidelines for selecting optimal layers and developing effective fusion strategies for the integration of LLMs remain underexplored. In this study, we propose a layer-aware embedding selection method and investigate how to quantitatively evaluate different layers to identify the most important ones for downstream NLP tasks, showing that the critical layers vary depending on the dataset. We also explore how combining embeddings from multiple LLMs, without requiring model fine-tuning, can improve performance. Experiments on four English text classification datasets (SST-2, MR, R8, and R52) demonstrate that different layers in LLMs exhibit varying degrees of representational strength for classification, and that combining embeddings from different models can enhance performance if the models exhibit complementary characteristics. Additionally, we discuss resources overhead (memory and inference time) to provide a balanced perspective on the real world feasibility of embedding fusion. Future work will explore multilingual and domain specific datasets, as well as techniques for automating layer selection, to improve both performance and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05738v1">LLM-assisted Mutation for Whitebox API Testing</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Cloud applications heavily rely on APIs to communicate with each other and exchange data. To ensure the reliability of cloud applications, cloud providers widely adopt API testing techniques. Unfortunately, existing API testing approaches are insufficient to reach strict conditions, a problem known as fitness plateaus, due to the lack of gradient provided by coverage metrics. To address this issue, we propose MioHint, a novel white-box API testing approach that leverages the code comprehension capabilities of Large Language Model (LLM) to boost API testing. The key challenge of LLM-based API testing lies in system-level testing, which emphasizes the dependencies between requests and targets across functions and files, thereby making the entire codebase the object of analysis. However, feeding the entire codebase to an LLM is impractical due to its limited context length and short memory. MioHint addresses this challenge by synergizing static analysis with LLMs. We retrieve relevant code with data-dependency analysis at the statement level, including def-use analysis for variables used in the target and function expansion for subfunctions called by the target. To evaluate the effectiveness of our method, we conducted experiments across 16 real-world REST API services. The findings reveal that MioHint achieves an average increase of 4.95% absolute in line coverage compared to the baseline, EvoMaster, alongside a remarkable factor of 67x improvement in mutation accuracy. Furthermore, our method successfully covers over 57% of hard-to-cover targets while in baseline the coverage is less than 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05732v1">LLM$\times$MapReduce-V2: Entropy-Driven Convolutional Test-Time Scaling for Generating Long-Form Articles from Extremely Long Resources</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Long-form generation is crucial for a wide range of practical applications, typically categorized into short-to-long and long-to-long generation. While short-to-long generations have received considerable attention, generating long texts from extremely long resources remains relatively underexplored. The primary challenge in long-to-long generation lies in effectively integrating and analyzing relevant information from extensive inputs, which remains difficult for current large language models (LLMs). In this paper, we propose LLM$\times$MapReduce-V2, a novel test-time scaling strategy designed to enhance the ability of LLMs to process extremely long inputs. Drawing inspiration from convolutional neural networks, which iteratively integrate local features into higher-level global representations, LLM$\times$MapReduce-V2 utilizes stacked convolutional scaling layers to progressively expand the understanding of input materials. Both quantitative and qualitative experimental results demonstrate that our approach substantially enhances the ability of LLMs to process long inputs and generate coherent, informative long-form articles, outperforming several representative baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02810v3">StateAct: Enhancing LLM Base Agents via Self-prompting and State-tracking</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 9 pages, 5 pages appendix, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as autonomous agents, tackling tasks from robotics to web navigation. Their performance depends on the underlying base agent. Existing methods, however, struggle with long-context reasoning and goal adherence. We introduce StateAct, a novel and efficient base agent that enhances decision-making through (1) self-prompting, which reinforces task goals at every step, and (2) chain-of-states, an extension of chain-of-thought that tracks state information over time. StateAct outperforms ReAct, the previous best base agent, by over 10% on Alfworld, 30% on Textcraft, and 7% on Webshop across multiple frontier LLMs. We also demonstrate that StateAct can be used as a drop-in replacement for ReAct with advanced LLM agent methods such as test-time scaling, yielding an additional 12% gain on Textcraft. By improving efficiency and long-range reasoning without requiring additional training or retrieval, StateAct provides a scalable foundation for LLM agents. We open source our code to support further research at https://github.com/ai-nikolai/stateact .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05716v1">Single-Agent vs. Multi-Agent LLM Strategies for Automated Student Reflection Assessment</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 To be published in Proceedings of the 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2025)
    </div>
    <details class="paper-abstract">
      We explore the use of Large Language Models (LLMs) for automated assessment of open-text student reflections and prediction of academic performance. Traditional methods for evaluating reflections are time-consuming and may not scale effectively in educational settings. In this work, we employ LLMs to transform student reflections into quantitative scores using two assessment strategies (single-agent and multi-agent) and two prompting techniques (zero-shot and few-shot). Our experiments, conducted on a dataset of 5,278 reflections from 377 students over three academic terms, demonstrate that the single-agent with few-shot strategy achieves the highest match rate with human evaluations. Furthermore, models utilizing LLM-assessed reflection scores outperform baselines in both at-risk student identification and grade prediction tasks. These findings suggest that LLMs can effectively automate reflection assessment, reduce educators' workload, and enable timely support for students who may need additional assistance. Our work emphasizes the potential of integrating advanced generative AI technologies into educational practices to enhance student engagement and academic success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07300v3">CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Deep learning (e.g., Transformer) has been widely and successfully used in multivariate time series forecasting (MTSF). Unlike existing methods that focus on training models from a single modal of time series input, large language models (LLMs) based MTSF methods with cross-modal text and time series input have recently shown great superiority, especially with limited temporal data. However, current LLM-based MTSF methods usually focus on adapting and fine-tuning LLMs, while neglecting the distribution discrepancy between textual and temporal input tokens, thus leading to sub-optimal performance. To address this issue, we propose a novel Cross-Modal LLM Fine-Tuning (CALF) framework for MTSF by reducing the distribution discrepancy between textual and temporal data, which mainly consists of the temporal target branch with temporal input and the textual source branch with aligned textual input. To reduce the distribution discrepancy, we develop the cross-modal match module to first align cross-modal input distributions. Additionally, to minimize the modality distribution gap in both feature and output spaces, feature regularization loss is developed to align the intermediate features between the two branches for better weight updates, while output consistency loss is introduced to allow the output representations of both branches to correspond effectively. Thanks to the modality alignment, CALF establishes state-of-the-art performance for both long-term and short-term forecasting tasks with low computational complexity, and exhibiting favorable few-shot and zero-shot abilities similar to that in LLMs. Code is available at https://github.com/Hank0626/LLaTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05711v1">Automated Archival Descriptions with Federated Intelligence of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Enforcing archival standards requires specialized expertise, and manually creating metadata descriptions for archival materials is a tedious and error-prone task. This work aims at exploring the potential of agentic AI and large language models (LLMs) in addressing the challenges of implementing a standardized archival description process. To this end, we introduce an agentic AI-driven system for automated generation of high-quality metadata descriptions of archival materials. We develop a federated optimization approach that unites the intelligence of multiple LLMs to construct optimal archival metadata. We also suggest methods to overcome the challenges associated with using LLMs for consistent metadata generation. To evaluate the feasibility and effectiveness of our techniques, we conducted extensive experiments using a real-world dataset of archival materials, which covers a variety of document types and data formats. The evaluation results demonstrate the feasibility of our techniques and highlight the superior performance of the federated optimization approach compared to single-model solutions in metadata quality and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05683v1">Towards Smarter Hiring: Are Zero-Shot and Few-Shot Pre-trained LLMs Ready for HR Spoken Interview Transcript Analysis?</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 32 pages, 24 figures
    </div>
    <details class="paper-abstract">
      This research paper presents a comprehensive analysis of the performance of prominent pre-trained large language models (LLMs), including GPT-4 Turbo, GPT-3.5 Turbo, text-davinci-003, text-babbage-001, text-curie-001, text-ada-001, llama-2-7b-chat, llama-2-13b-chat, and llama-2-70b-chat, in comparison to expert human evaluators in providing scores, identifying errors, and offering feedback and improvement suggestions to candidates during mock HR (Human Resources) interviews. We introduce a dataset called HURIT (Human Resource Interview Transcripts), which comprises 3,890 HR interview transcripts sourced from real-world HR interview scenarios. Our findings reveal that pre-trained LLMs, particularly GPT-4 Turbo and GPT-3.5 Turbo, exhibit commendable performance and are capable of producing evaluations comparable to those of expert human evaluators. Although these LLMs demonstrate proficiency in providing scores comparable to human experts in terms of human evaluation metrics, they frequently fail to identify errors and offer specific actionable advice for candidate performance improvement in HR interviews. Our research suggests that the current state-of-the-art pre-trained LLMs are not fully conducive for automatic deployment in an HR interview assessment. Instead, our findings advocate for a human-in-the-loop approach, to incorporate manual checks for inconsistencies and provisions for improving feedback quality as a more suitable strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05673v1">VC-LLM: Automated Advertisement Video Creation from Raw Footage using Multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      As short videos have risen in popularity, the role of video content in advertising has become increasingly significant. Typically, advertisers record a large amount of raw footage about the product and then create numerous different short-form advertisement videos based on this raw footage. Creating such videos mainly involves editing raw footage and writing advertisement scripts, which requires a certain level of creative ability. It is usually challenging to create many different video contents for the same product, and manual efficiency is often low. In this paper, we present VC-LLM, a framework powered by Large Language Models for the automatic creation of high-quality short-form advertisement videos. Our approach leverages high-resolution spatial input and low-resolution temporal input to represent video clips more effectively, capturing both fine-grained visual details and broader temporal dynamics. In addition, during training, we incorporate supplementary information generated by rewriting the ground truth text, ensuring that all key output information can be directly traced back to the input, thereby reducing model hallucinations. We also designed a benchmark to evaluate the quality of the created videos. Experiments show that VC-LLM based on GPT-4o can produce videos comparable to those created by humans. Furthermore, we collected numerous high-quality short advertisement videos to create a pre-training dataset and manually cleaned a portion of the data to construct a high-quality fine-tuning dataset. Experiments indicate that, on the benchmark, the VC-LLM based on fine-tuned LLM can produce videos with superior narrative logic compared to those created by the VC-LLM based on GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03661v2">MILLION: Mastering Long-Context LLM Inference Via Outlier-Immunized KV Product Quantization</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 7 pages, 7 figures and 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized for complex tasks requiring longer context lengths, with some models supporting up to 128K or 1M tokens. This trend, however, presents significant challenges in inference speed and memory management. Quantization emerges as a promising approach to address the widening gap between LLM size and memory capacity. However, traditional quantization schemes often yield suboptimal compression results for KV caches due to two key factors: i) On-the-fly quantization and de-quantization, causing significant performance overhead; ii) Prevalence of outliers in KV values, challenging low-bitwidth uniform quantization. To this end, we propose MILLION, a novel quantization framework achieving low-bitwidth KV cache through product quantization. First, we conduct a thorough analysis of KV cache distribution, revealing the limitations of existing quantization schemes. Second, we introduce a non-uniform quantization algorithm based on product quantization, which efficiently compresses data while preserving accuracy. Third, we develop a high-performance GPU inference framework with efficient attention kernel and pipeline design for MILLION that leverages sparse computation and asynchronous quantization, significantly enhancing inference speed. Comprehensive evaluation results demonstrate that MILLION can achieve 4 bits quantization with trivial perplexity and accuracy loss, and achieve 2.09x end-to-end performance gains at 32K context length. Code is released at https://github.com/ZongwuWang/MILLION.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01698v2">ToM-RL: Reinforcement Learning Unlocks Theory of Mind in Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Recent advancements in rule-based reinforcement learning (RL), applied during the post-training phase of large language models (LLMs), have significantly enhanced their capabilities in structured reasoning tasks such as mathematics and logical inference. However, the effectiveness of RL in social reasoning, particularly in Theory of Mind (ToM), the ability to infer others' mental states, remains largely unexplored. In this study, we demonstrate that RL methods effectively unlock ToM reasoning capabilities even in small-scale LLMs (0.5B to 7B parameters). Using a modest dataset comprising 3200 questions across diverse scenarios, our RL-trained 7B model achieves 84.50\% accuracy on the Hi-ToM benchmark, surpassing models like GPT-4o and DeepSeek-v3 despite significantly fewer parameters. While smaller models ($\leq$3B parameters) suffer from reasoning collapse, larger models (7B parameters) maintain stable performance through consistent belief tracking. Additionally, our RL-based models demonstrate robust generalization to higher-order, out-of-distribution ToM problems, novel textual presentations, and previously unseen datasets. These findings highlight RL's potential to enhance social cognitive reasoning, bridging the gap between structured problem-solving and nuanced social inference in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05652v1">Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17867v4">Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 International Joint Conference on Neural Networks 2025 (IJCNN 2025)
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries. Our dataset and code are publicly available at https://mcxiaoxiao.github.io/MMSQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v2">Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05614v1">Two Intermediate Translations Are Better Than One: Fine-tuning LLMs for Document-level Translation Refinement</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Recent research has shown that large language models (LLMs) can enhance translation quality through self-refinement. In this paper, we build on this idea by extending the refinement from sentence-level to document-level translation, specifically focusing on document-to-document (Doc2Doc) translation refinement. Since sentence-to-sentence (Sent2Sent) and Doc2Doc translation address different aspects of the translation process, we propose fine-tuning LLMs for translation refinement using two intermediate translations, combining the strengths of both Sent2Sent and Doc2Doc. Additionally, recognizing that the quality of intermediate translations varies, we introduce an enhanced fine-tuning method with quality awareness that assigns lower weights to easier translations and higher weights to more difficult ones, enabling the model to focus on challenging translation cases. Experimental results across ten translation tasks with LLaMA-3-8B-Instruct and Mistral-Nemo-Instruct demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05607v1">FactGuard: Leveraging Multi-Agent Systems to Generate Answerable and Unanswerable Questions for Enhanced Long-Context LLM Extraction</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Extractive reading comprehension systems are designed to locate the correct answer to a question within a given text. However, a persistent challenge lies in ensuring these models maintain high accuracy in answering questions while reliably recognizing unanswerable queries. Despite significant advances in large language models (LLMs) for reading comprehension, this issue remains critical, particularly as the length of supported contexts continues to expand. To address this challenge, we propose an innovative data augmentation methodology grounded in a multi-agent collaborative framework. Unlike traditional methods, such as the costly human annotation process required for datasets like SQuAD 2.0, our method autonomously generates evidence-based question-answer pairs and systematically constructs unanswerable questions. Using this methodology, we developed the FactGuard-Bench dataset, which comprises 25,220 examples of both answerable and unanswerable question scenarios, with context lengths ranging from 8K to 128K. Experimental evaluations conducted on seven popular LLMs reveal that even the most advanced models achieve only 61.79% overall accuracy. Furthermore, we emphasize the importance of a model's ability to reason about unanswerable questions to avoid generating plausible but incorrect answers. By implementing efficient data selection and generation within the multi-agent collaborative framework, our method significantly reduces the traditionally high costs associated with manual annotation and provides valuable insights for the training and optimization of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05605v1">ShadowCoT: Cognitive Hijacking for Stealthy Reasoning Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Zhao et al., 16 pages, 2025, uploaded by Hanzhou Wu, Shanghai University
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) enhances an LLM's ability to perform complex reasoning tasks, but it also introduces new security issues. In this work, we present ShadowCoT, a novel backdoor attack framework that targets the internal reasoning mechanism of LLMs. Unlike prior token-level or prompt-based attacks, ShadowCoT directly manipulates the model's cognitive reasoning path, enabling it to hijack multi-step reasoning chains and produce logically coherent but adversarial outcomes. By conditioning on internal reasoning states, ShadowCoT learns to recognize and selectively disrupt key reasoning steps, effectively mounting a self-reflective cognitive attack within the target model. Our approach introduces a lightweight yet effective multi-stage injection pipeline, which selectively rewires attention pathways and perturbs intermediate representations with minimal parameter overhead (only 0.15% updated). ShadowCoT further leverages reinforcement learning and reasoning chain pollution (RCP) to autonomously synthesize stealthy adversarial CoTs that remain undetectable to advanced defenses. Extensive experiments across diverse reasoning benchmarks and LLMs show that ShadowCoT consistently achieves high Attack Success Rate (94.4%) and Hijacking Success Rate (88.4%) while preserving benign performance. These results reveal an emergent class of cognition-level threats and highlight the urgent need for defenses beyond shallow surface-level consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09529v3">How Well Can Modern LLMs Act as Agent Cores in Radiology Environments?</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      We introduce RadA-BenchPlat, an evaluation platform that benchmarks the performance of large language models (LLMs) act as agent cores in radiology environments using 2,200 radiologist-verified synthetic patient records covering six anatomical regions, five imaging modalities, and 2,200 disease scenarios, resulting in 24,200 question-answer pairs that simulate diverse clinical situations. The platform also defines ten categories of tools for agent-driven task solving and evaluates seven leading LLMs, revealing that while models like Claude-3.7-Sonnet can achieve a 67.1% task completion rate in routine settings, they still struggle with complex task understanding and tool coordination, limiting their capacity to serve as the central core of automated radiology systems. By incorporating four advanced prompt engineering strategies--where prompt-backpropagation and multi-agent collaboration contributed 16.8% and 30.7% improvements, respectively--the performance for complex tasks was enhanced by 48.2% overall. Furthermore, automated tool building was explored to improve robustness, achieving a 65.4% success rate, thereby offering promising insights for the future integration of fully automated radiology applications into clinical practice. All of our code and data are openly available at https://github.com/MAGIC-AI4Med/RadABench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14567v3">ELOQ: Resources for Enhancing LLM Detection of Out-of-Scope Questions</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 Accepted by SIGIR'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in Conversational AI systems to generate responses to user inquiries. However, many natural questions lack well-defined answers. While existing studies primarily focus on question types such as false premises, they often overlook out-of-scope questions, where the provided document is semantically highly similar to the query but does not contain the required answer. In this paper, we propose a guided hallucination-based method to efficiently generate a diverse set of out-of-scope questions from a given document corpus. We then evaluate multiple LLMs based on their effectiveness in confusion detection and appropriate response generation. Furthermore, we introduce an improved method for detecting such out-of-scope questions, enhancing the reliability of LLM-based question-answering systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06460v1">Can LLMs Simulate Personas with Reversed Performance? A Benchmark for Counterfactual Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now increasingly widely used to simulate personas in virtual environments, leveraging their instruction-following capability. However, we discovered that even state-of-the-art LLMs cannot simulate personas with reversed performance (e.g., student personas with low proficiency in educational settings), which impairs the simulation diversity and limits the practical applications of the simulated environments. In this work, using mathematical reasoning as a representative scenario, we propose the first benchmark dataset for evaluating LLMs on simulating personas with reversed performance, a capability that we dub "counterfactual instruction following". We evaluate both open-weight and closed-source LLMs on this task and find that LLMs, including the OpenAI o1 reasoning model, all struggle to follow counterfactual instructions for simulating reversedly performing personas. Intersectionally simulating both the performance level and the race population of a persona worsens the effect even further. These results highlight the challenges of counterfactual instruction following and the need for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05108v2">Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-08
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Discovering efficient algorithms for solving complex problems has been an outstanding challenge in mathematics and computer science, requiring substantial human expertise over the years. Recent advancements in evolutionary search with large language models (LLMs) have shown promise in accelerating the discovery of algorithms across various domains, particularly in mathematics and optimization. However, existing approaches treat the LLM as a static generator, missing the opportunity to update the model with the signal obtained from evolutionary exploration. In this work, we propose to augment LLM-based evolutionary search by continuously refining the search operator - the LLM - through reinforcement learning (RL) fine-tuning. Our method leverages evolutionary search as an exploration strategy to discover improved algorithms, while RL optimizes the LLM policy based on these discoveries. Our experiments on three combinatorial optimization tasks - bin packing, traveling salesman, and the flatpack problem - show that combining RL and evolutionary search improves discovery efficiency of improved algorithms, showcasing the potential of RL-enhanced evolutionary strategies to assist computer scientists and mathematicians for more efficient algorithm design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08688v2">Randomness, Not Representation: The Unreliability of Evaluating Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-08
    </div>
    <details class="paper-abstract">
      Research on the 'cultural alignment' of Large Language Models (LLMs) has emerged in response to growing interest in understanding representation across diverse stakeholders. Current approaches to evaluating cultural alignment through survey-based assessments that borrow from social science methodologies often overlook systematic robustness checks. Here, we identify and test three assumptions behind current survey-based evaluation methods: (1) Stability: that cultural alignment is a property of LLMs rather than an artifact of evaluation design, (2) Extrapolability: that alignment with one culture on a narrow set of issues predicts alignment with that culture on others, and (3) Steerability: that LLMs can be reliably prompted to represent specific cultural perspectives. Through experiments examining both explicit and implicit preferences of leading LLMs, we find a high level of instability across presentation formats, incoherence between evaluated versus held-out cultural dimensions, and erratic behavior under prompt steering. We show that these inconsistencies can cause the results of an evaluation to be very sensitive to minor variations in methodology. Finally, we demonstrate in a case study on evaluation design that narrow experiments and a selective assessment of evidence can be used to paint an incomplete picture of LLMs' cultural alignment properties. Overall, these results highlight significant limitations of current survey-based approaches to evaluating the cultural alignment of LLMs and highlight a need for systematic robustness checks and red-teaming for evaluation results. Data and code are available at https://huggingface.co/datasets/akhan02/cultural-dimension-cover-letters and https://github.com/ariba-k/llm-cultural-alignment-evaluation, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04745v1">Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 13 pages, 23 figures. Submitted to XLLM @ ACL 2025
    </div>
    <details class="paper-abstract">
      This paper evaluates the ability of Large Language Models (LLMs) to leverage contextual information in the form of structured linguistic representations. Specifically, we examine the impact of encoding both short and long contexts using Abstract Meaning Representation (AMR) structures across a diverse set of language tasks. We perform our analysis using 8-bit quantized and instruction-tuned versions of Llama 3.1 (8B), Phi-3, and Mistral 7B. Our results indicate that, for tasks involving short contexts, augmenting the prompt with the AMR of the original language context often degrades the performance of the underlying LLM. However, for tasks that involve long contexts, such as dialogue summarization in the SAMSum dataset, this enhancement improves LLM performance, for example, by increasing the zero-shot cosine similarity score of Llama 3.1 from 66.2% to 76%. This improvement is more evident in the newer and larger LLMs, but does not extend to the older or smaller ones. In addition, we observe that LLMs can effectively reconstruct the original text from a linearized AMR, achieving a cosine similarity of 81.3% in the best-case scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12501v2">Crowd Comparative Reasoning: Unlocking Comprehensive Evaluations for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge, which generates chain-of-thought (CoT) judgments, has become a widely adopted auto-evaluation method. However, its reliability is compromised by the CoT reasoning's inability to capture comprehensive and deeper details, often leading to incomplete outcomes. Existing methods mainly rely on majority voting or criteria expansion, which is insufficient to address the limitation in CoT. We propose Crowd-based Comparative Evaluation, which introduces additional crowd responses to compare with the candidate responses, thereby exposing deeper and more comprehensive details within the candidate responses. This process effectively guides LLM-as-a-Judge to provide a more detailed CoT judgment. Extensive experiments demonstrate that our approach enhances evaluation reliability, achieving an average accuracy gain of 6.7% across five benchmarks. Moreover, our method produces higher-quality CoTs that facilitate judge distillation and exhibit superior performance in rejection sampling for supervised fine-tuning (SFT), referred to as crowd rejection sampling, thereby enabling more efficient SFT. Our analysis confirms that CoTs generated by ours are more comprehensive and of higher quality, and evaluation accuracy improves as inference scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03444v2">LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 This paper is accepted by 45th IEEE International Conference on Distributed Computing Systems (ICDCS 2025)
    </div>
    <details class="paper-abstract">
      Developing compound Large Language Model (LLM) applications is becoming an increasingly prevalent approach to solving real-world problems. In these applications, an LLM collaborates with various external modules, including APIs and even other LLMs, to realize complex intelligent services. However, we reveal that the intrinsic duration and structural uncertainty in compound LLM applications pose great challenges for LLM service providers in serving and scheduling them efficiently. In this paper, we propose LLMSched, an uncertainty-aware scheduling framework for emerging compound LLM applications. In LLMSched, we first design a novel DAG-based model to describe the uncertain compound LLM applications. Then, we adopt the Bayesian network to comprehensively profile compound LLM applications and identify uncertainty-reducing stages, along with an entropy-based mechanism to quantify their uncertainty reduction. Combining an uncertainty reduction strategy and a job completion time (JCT)-efficient scheme, we further propose an efficient scheduler to reduce the average JCT. Evaluation of both simulation and testbed experiments on various representative compound LLM applications shows that compared to existing state-of-the-art scheduling schemes, LLMSched can reduce the average JCT by 14~79%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09654v3">Do LLMs Understand Visual Anomalies? Uncovering LLM's Capabilities in Zero-shot Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted by MM'24 (Oral)
    </div>
    <details class="paper-abstract">
      Large vision-language models (LVLMs) are markedly proficient in deriving visual representations guided by natural language. Recent explorations have utilized LVLMs to tackle zero-shot visual anomaly detection (VAD) challenges by pairing images with textual descriptions indicative of normal and abnormal conditions, referred to as anomaly prompts. However, existing approaches depend on static anomaly prompts that are prone to cross-semantic ambiguity, and prioritize global image-level representations over crucial local pixel-level image-to-text alignment that is necessary for accurate anomaly localization. In this paper, we present ALFA, a training-free approach designed to address these challenges via a unified model. We propose a run-time prompt adaptation strategy, which first generates informative anomaly prompts to leverage the capabilities of a large language model (LLM). This strategy is enhanced by a contextual scoring mechanism for per-image anomaly prompt adaptation and cross-semantic ambiguity mitigation. We further introduce a novel fine-grained aligner to fuse local pixel-level semantics for precise anomaly localization, by projecting the image-text alignment from global to local semantic spaces. Extensive evaluations on MVTec and VisA datasets confirm ALFA's effectiveness in harnessing the language potential for zero-shot VAD, achieving significant PRO improvements of 12.1% on MVTec and 8.9% on VisA compared to state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04726v1">Can LLM-Driven Hard Negative Sampling Empower Collaborative Filtering? Findings and Potentials</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Hard negative samples can accelerate model convergence and optimize decision boundaries, which is key to improving the performance of recommender systems. Although large language models (LLMs) possess strong semantic understanding and generation capabilities, systematic research has not yet been conducted on how to generate hard negative samples effectively. To fill this gap, this paper introduces the concept of Semantic Negative Sampling and exploreshow to optimize LLMs for high-quality, hard negative sampling. Specifically, we design an experimental pipeline that includes three main modules, profile generation, semantic negative sampling, and semantic alignment, to verify the potential of LLM-driven hard negative sampling in enhancing the accuracy of collaborative filtering (CF). Experimental results indicate that hard negative samples generated based on LLMs, when semantically aligned and integrated into CF, can significantly improve CF performance, although there is still a certain gap compared to traditional negative sampling methods. Further analysis reveals that this gap primarily arises from two major challenges: noisy samples and lack of behavioral constraints. To address these challenges, we propose a framework called HNLMRec, based on fine-tuning LLMs supervised by collaborative signals. Experimental results show that this framework outperforms traditional negative sampling and other LLM-driven recommendation methods across multiple datasets, providing new solutions for empowering traditional RS with LLMs. Additionally, we validate the excellent generalization ability of the LLM-based semantic negative sampling method on new datasets, demonstrating its potential in alleviating issues such as data sparsity, popularity bias, and the problem of false hard negative samples. Our implementation code is available at https://github.com/user683/HNLMRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04715v1">Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) accessed via black-box APIs introduces a significant trust challenge: users pay for services based on advertised model capabilities (e.g., size, performance), but providers may covertly substitute the specified model with a cheaper, lower-quality alternative to reduce operational costs. This lack of transparency undermines fairness, erodes trust, and complicates reliable benchmarking. Detecting such substitutions is difficult due to the black-box nature, typically limiting interaction to input-output queries. This paper formalizes the problem of model substitution detection in LLM APIs. We systematically evaluate existing verification techniques, including output-based statistical tests, benchmark evaluations, and log probability analysis, under various realistic attack scenarios like model quantization, randomized substitution, and benchmark evasion. Our findings reveal the limitations of methods relying solely on text outputs, especially against subtle or adaptive attacks. While log probability analysis offers stronger guarantees when available, its accessibility is often limited. We conclude by discussing the potential of hardware-based solutions like Trusted Execution Environments (TEEs) as a pathway towards provable model integrity, highlighting the trade-offs between security, performance, and provider adoption. Code is available at https://github.com/sunblaze-ucb/llm-api-audit
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04698v1">scAgent: Universal Single-Cell Annotation via a LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Cell type annotation is critical for understanding cellular heterogeneity. Based on single-cell RNA-seq data and deep learning models, good progress has been made in annotating a fixed number of cell types within a specific tissue. However, universal cell annotation, which can generalize across tissues, discover novel cell types, and extend to novel cell types, remains less explored. To fill this gap, this paper proposes scAgent, a universal cell annotation framework based on Large Language Models (LLMs). scAgent can identify cell types and discover novel cell types in diverse tissues; furthermore, it is data efficient to learn novel cell types. Experimental studies in 160 cell types and 35 tissues demonstrate the superior performance of scAgent in general cell-type annotation, novel cell discovery, and extensibility to novel cell type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05527v1">Bridging Industrial Expertise and XR with LLM-Powered Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 7 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This paper introduces a novel integration of Retrieval-Augmented Generation (RAG) enhanced Large Language Models (LLMs) with Extended Reality (XR) technologies to address knowledge transfer challenges in industrial environments. The proposed system embeds domain-specific industrial knowledge into XR environments through a natural language interface, enabling hands-free, context-aware expert guidance for workers. We present the architecture of the proposed system consisting of an LLM Chat Engine with dynamic tool orchestration and an XR application featuring voice-driven interaction. Performance evaluation of various chunking strategies, embedding models, and vector databases reveals that semantic chunking, balanced embedding models, and efficient vector stores deliver optimal performance for industrial knowledge retrieval. The system's potential is demonstrated through early implementation in multiple industrial use cases, including robotic assembly, smart infrastructure maintenance, and aerospace component servicing. Results indicate potential for enhancing training efficiency, remote assistance capabilities, and operational guidance in alignment with Industry 5.0's human-centric and resilient approach to industrial development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08527v2">Scaling Laws for Predicting Downstream Performance in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      Precise estimation of downstream performance in large language models (LLMs) prior to training is essential for guiding their development process. Scaling laws analysis utilizes the statistics of a series of significantly smaller sampling language models (LMs) to predict the performance of the target LLM. For downstream performance prediction, the critical challenge lies in the emergent abilities in LLMs that occur beyond task-specific computational thresholds. In this work, we focus on the pre-training loss as a more computation-efficient metric for performance estimation. Our two-stage approach FLP consists of first estimating a function that maps computational resources (e.g., FLOPs) to the pre-training Loss using a series of fully-converged sampling models, followed by mapping the pre-training loss to downstream task Performance using the intermediate models with emerged performance. In our experiments, this FLP solution accurately predicts the performance of LLMs with 7B and 13B parameters using a series of sampling LMs up to 3B, achieving error margins of 5% and 10%, respectively, and significantly outperforming the FLOPs-to-Performance approach. Further, we present FLP-M, a fundamental approach for performance prediction that addresses the practical need to integrate datasets from multiple sources during pre-training. FLP-M extends the power law analytical function to predict domain-specific pre-training loss based on FLOPs across data sources, and employs a two-layer neural network to model the non-linear relationship between multiple domain-specific loss and downstream performance. By utilizing a 3B LLM trained on a specific ratio and a series of smaller sampling LMs, FLP-M can effectively forecast the performance of 3B and 7B LLMs across various data mixtures for most benchmarks within 10% error margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05522v1">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Exploration, the act of broadening user experiences beyond their established preferences, is challenging in large-scale recommendation systems due to feedback loops and limited signals on user exploration patterns. Large Language Models (LLMs) offer potential by leveraging their world knowledge to recommend novel content outside these loops. A key challenge is aligning LLMs with user preferences while preserving their knowledge and reasoning. While using LLMs to plan for the next novel user interest, this paper introduces a novel approach combining hierarchical planning with LLM inference-time scaling to improve recommendation relevancy without compromising novelty. We decouple novelty and user-alignment, training separate LLMs for each objective. We then scale up the novelty-focused LLM's inference and select the best-of-n predictions using the user-aligned LLM. Live experiments demonstrate efficacy, showing significant gains in both user satisfaction (measured by watch activity and active user counts) and exploration diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09722v3">Optimized Multi-Token Joint Decoding with Auxiliary Model for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across diverse tasks, yet their inference processes are hindered by substantial time and energy demands due to single-token generation at each decoding step. While previous methods such as speculative decoding mitigate these inefficiencies by producing multiple tokens per step, each token is still generated by its single-token distribution, thereby enhancing speed without improving effectiveness. In contrast, our work simultaneously enhances inference speed and improves the output effectiveness. We consider multi-token joint decoding (MTJD), which generates multiple tokens from their joint distribution at each iteration, theoretically reducing perplexity and enhancing task performance. However, MTJD suffers from the high cost of sampling from the joint distribution of multiple tokens. Inspired by speculative decoding, we introduce multi-token assisted decoding (MTAD), a novel framework designed to accelerate MTJD. MTAD leverages a smaller auxiliary model to approximate the joint distribution of a larger model, incorporating a verification mechanism that not only ensures the accuracy of this approximation, but also improves the decoding efficiency over conventional speculative decoding. Theoretically, we demonstrate that MTAD closely approximates exact MTJD with bounded error. Empirical evaluations using Llama-2 and OPT models ranging from 13B to 70B parameters across various tasks reveal that MTAD reduces perplexity by 21.2% and improves downstream performance compared to standard single-token sampling. Furthermore, MTAD achieves a 1.42x speed-up and consumes 1.54x less energy than conventional speculative decoding methods. These results highlight MTAD's ability to make multi-token joint decoding both effective and efficient, promoting more sustainable and high-performance deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09017v2">Diversity Enhances an LLM's Performance in RAG and Long-context Task</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have highlighted the challenge of context window limitations, primarily due to the quadratic time complexity of the self-attention mechanism (\(O(N^2)\), where \(N\) denotes the context window length). This constraint impacts tasks such as retrieval-augmented generation (RAG) in question answering (Q\&A) and long context summarization. A common approach involves selecting content with the highest similarity to the query; however, this often leads to redundancy and the exclusion of diverse yet relevant information. Building on principles from Maximal Marginal Relevance (MMR) and Farthest Point Sampling (FPS), we integrate diversity into the content selection process. Our findings reveal that incorporating diversity substantially increases the recall of selecting relevant sentences or chunks before LLM-based Q\&A and summarization. These results highlight the importance of maintaining diversity in future LLM applications to further improve summarization and Q\&A outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05500v1">Prism: Dynamic and Flexible Benchmarking of LLMs Code Generation with Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has outpaced traditional evaluation methods. Static benchmarks fail to capture the depth and breadth of LLM capabilities and eventually become obsolete, while most dynamic approaches either rely too heavily on LLM-based evaluation or remain constrained by predefined test sets. We introduce Prism, a flexible, dynamic benchmarking framework designed for comprehensive LLM assessment. Prism builds on three key components: (1) a tree-based state representation that models evaluation as a Markov Decision Process, (2) a Monte Carlo Tree Search algorithm adapted to uncover challenging evaluation scenarios, and (3) a multi-agent evaluation pipeline that enables simultaneous assessment of diverse capabilities. To ensure robust evaluation, Prism integrates structural measurements of tree exploration patterns with performance metrics across difficulty levels, providing detailed diagnostics of error patterns, test coverage, and solution approaches. Through extensive experiments on five state-of-the-art LLMs, we analyze how model architecture and scale influence code generation performance across varying task difficulties. Our results demonstrate Prism's effectiveness as a dynamic benchmark that evolves with model advancements while offering deeper insights into their limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12531v2">GSCE: A Prompt Framework with Enhanced Reasoning for Reliable LLM-driven Drone Control</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into robotic control, including drones, has the potential to revolutionize autonomous systems. Research studies have demonstrated that LLMs can be leveraged to support robotic operations. However, when facing tasks with complex reasoning, concerns and challenges are raised about the reliability of solutions produced by LLMs. In this paper, we propose a prompt framework with enhanced reasoning to enable reliable LLM-driven control for drones. Our framework consists of novel technical components designed using Guidelines, Skill APIs, Constraints, and Examples, namely GSCE. GSCE is featured by its reliable and constraint-compliant code generation. We performed thorough experiments using GSCE for the control of drones with a wide level of task complexities. Our experiment results demonstrate that GSCE can significantly improve task success rates and completeness compared to baseline approaches, highlighting its potential for reliable LLM-driven autonomous drone systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05050v2">A Unified Framework with Novel Metrics for Evaluating the Effectiveness of XAI Techniques in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 arXiv admin note: substantial text overlap with arXiv:2501.15374
    </div>
    <details class="paper-abstract">
      The increasing complexity of LLMs presents significant challenges to their transparency and interpretability, necessitating the use of eXplainable AI (XAI) techniques to enhance trustworthiness and usability. This study introduces a comprehensive evaluation framework with four novel metrics for assessing the effectiveness of five XAI techniques across five LLMs and two downstream tasks. We apply this framework to evaluate several XAI techniques LIME, SHAP, Integrated Gradients, Layer-wise Relevance Propagation (LRP), and Attention Mechanism Visualization (AMV) using the IMDB Movie Reviews and Tweet Sentiment Extraction datasets. The evaluation focuses on four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. Our results show that LIME consistently achieves high scores across multiple LLMs and evaluation metrics, while AMV demonstrates superior Robustness and near-perfect Consistency. LRP excels in Contrastivity, particularly with more complex models. Our findings provide valuable insights into the strengths and limitations of different XAI methods, offering guidance for developing and selecting appropriate XAI techniques for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05491v1">REEF: Relevance-Aware and Efficient LLM Adapter for Video Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted at CVPRW'25
    </div>
    <details class="paper-abstract">
      Integrating vision models into large language models (LLMs) has sparked significant interest in creating vision-language foundation models, especially for video understanding. Recent methods often utilize memory banks to handle untrimmed videos for video-level understanding. However, they typically compress visual memory using similarity-based greedy approaches, which can overlook the contextual importance of individual tokens. To address this, we introduce an efficient LLM adapter designed for video-level understanding of untrimmed videos that prioritizes the contextual relevance of spatio-temporal tokens. Our framework leverages scorer networks to selectively compress the visual memory bank and filter spatial tokens based on relevance, using a differentiable Top-K operator for end-to-end training. Across three key video-level understanding tasks$\unicode{x2013}$ untrimmed video classification, video question answering, and video captioning$\unicode{x2013}$our method achieves competitive or superior results on four large-scale datasets while reducing computational overhead by up to 34%. The code will be available soon on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11007v2">Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Compared to traditional machine learning models, recent large language models (LLMs) can exhibit multi-task-solving capabilities through multiple dialogues and multi-modal data sources. These unique characteristics of LLMs, together with their large model size, make their deployment more challenging. Specifically, (i) deploying LLMs on local devices faces computational, memory, and energy resource issues, while (ii) deploying them in the cloud cannot guarantee real-time service and incurs communication/usage costs. In this paper, we design TMO, a local-cloud LLM inference system with Three-M Offloading: Multi-modal, Multi-task, and Multi-dialogue. TMO incorporates (i) a lightweight local LLM that can process simple tasks at high speed and (ii) a large-scale cloud LLM that can handle multi-modal data sources. We develop a resource-constrained reinforcement learning (RCRL) strategy for TMO that optimizes the inference location (i.e., local vs. cloud) and multi-modal data sources to use for each task/dialogue, aiming to maximize the long-term reward (response quality, latency, and usage cost) while adhering to resource constraints. We also contribute M4A1, a new dataset we curated that contains reward and cost metrics across multiple modality, task, dialogue, and LLM configurations, enabling evaluation of offloading decisions. We demonstrate the effectiveness of TMO compared to several exploration-decision and LLM-as-Agent baselines, showing significant improvements in latency, cost, and response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05370v1">EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced smart education in the Artificial General Intelligence (AGI) era. A promising application lies in the automatic generalization of instructional design for curriculum and learning activities, focusing on two key aspects: (1) Customized Generation: generating niche-targeted teaching content based on students' varying learning abilities and states, and (2) Intelligent Optimization: iteratively optimizing content based on feedback from learning effectiveness or test scores. Currently, a single large LLM cannot effectively manage the entire process, posing a challenge for designing intelligent teaching plans. To address these issues, we developed EduPlanner, an LLM-based multi-agent system comprising an evaluator agent, an optimizer agent, and a question analyst, working in adversarial collaboration to generate customized and intelligent instructional design for curriculum and learning activities. Taking mathematics lessons as our example, EduPlanner employs a novel Skill-Tree structure to accurately model the background mathematics knowledge of student groups, personalizing instructional design for curriculum and learning activities according to students' knowledge levels and learning abilities. Additionally, we introduce the CIDDP, an LLM-based five-dimensional evaluation module encompassing clarity, Integrity, Depth, Practicality, and Pertinence, to comprehensively assess mathematics lesson plan quality and bootstrap intelligent optimization. Experiments conducted on the GSM8K and Algebra datasets demonstrate that EduPlanner excels in evaluating and optimizing instructional design for curriculum and learning activities. Ablation studies further validate the significance and effectiveness of each component within the framework. Our code is publicly available at https://github.com/Zc0812/Edu_Planner
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05352v1">Achieving binary weight and activation for LLMs using Post-Training Quantization</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Quantizing large language models (LLMs) to 1-bit precision significantly reduces computational costs, but existing quantization techniques suffer from noticeable performance degradation when using weight and activation precisions below 4 bits (W4A4). In this paper, we propose a post-training quantization framework with W(1+1)A(1*4) configuration, where weights are quantized to 1 bit with an additional 1 bit for fine-grain grouping and activations are quantized to 1 bit with a 4-fold increase in the number of channels. For weight quantization, we propose utilizing Hessian-aware fine-grained grouping along with an EM-based quantization scheme. For activation quantization, we decompose INT4-quantized activations into a 4 * INT1 format equivalently and simultaneously smooth the scaling factors based on quantization errors, which further reduces the quantization errors in activations. Our method surpasses state-of-the-art (SOTA) LLM quantization baselines on W2A4 across multiple tasks, pushing the boundaries of existing LLM quantization methods toward fully binarized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07137v1">Large Language Model (LLM) for Software Security: Code Analysis, Malware Analysis, Reverse Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as powerful tools in cybersecurity, offering advanced capabilities in malware detection, generation, and real-time monitoring. Numerous studies have explored their application in cybersecurity, demonstrating their effectiveness in identifying novel malware variants, analyzing malicious code structures, and enhancing automated threat analysis. Several transformer-based architectures and LLM-driven models have been proposed to improve malware analysis, leveraging semantic and structural insights to recognize malicious intent more accurately. This study presents a comprehensive review of LLM-based approaches in malware code analysis, summarizing recent advancements, trends, and methodologies. We examine notable scholarly works to map the research landscape, identify key challenges, and highlight emerging innovations in LLM-driven cybersecurity. Additionally, we emphasize the role of static analysis in malware detection, introduce notable datasets and specialized LLM models, and discuss essential datasets supporting automated malware research. This study serves as a valuable resource for researchers and cybersecurity professionals, offering insights into LLM-powered malware detection and defence strategies while outlining future directions for strengthening cybersecurity resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07135v1">SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      In the era of rapidly evolving large language models (LLMs), state-of-the-art rumor detection systems, particularly those based on Message Propagation Trees (MPTs), which represent a conversation tree with the post as its root and the replies as its descendants, are facing increasing threats from adversarial attacks that leverage LLMs to generate and inject malicious messages. Existing methods are based on the assumption that different nodes exhibit varying degrees of influence on predictions. They define nodes with high predictive influence as important nodes and target them for attacks. If the model treats nodes' predictive influence more uniformly, attackers will find it harder to target high predictive influence nodes. In this paper, we propose Similarizing the predictive Influence of Nodes with Contrastive Learning (SINCon), a defense mechanism that encourages the model to learn graph representations where nodes with varying importance have a more uniform influence on predictions. Extensive experiments on the Twitter and Weibo datasets demonstrate that SINCon not only preserves high classification accuracy on clean data but also significantly enhances resistance against LLM-driven message injection attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05276v1">Enhancing LLM-Based Short Answer Grading with Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Short answer assessment is a vital component of science education, allowing evaluation of students' complex three-dimensional understanding. Large language models (LLMs) that possess human-like ability in linguistic tasks are increasingly popular in assisting human graders to reduce their workload. However, LLMs' limitations in domain knowledge restrict their understanding in task-specific requirements and hinder their ability to achieve satisfactory performance. Retrieval-augmented generation (RAG) emerges as a promising solution by enabling LLMs to access relevant domain-specific knowledge during assessment. In this work, we propose an adaptive RAG framework for automated grading that dynamically retrieves and incorporates domain-specific knowledge based on the question and student answer context. Our approach combines semantic search and curated educational sources to retrieve valuable reference materials. Experimental results in a science education dataset demonstrate that our system achieves an improvement in grading accuracy compared to baseline LLM approaches. The findings suggest that RAG-enhanced grading systems can serve as reliable support with efficient performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05262v1">Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Despite high benchmark scores, Large Language Models (LLMs) often fail simple problem, raising a critical question: Do LLMs learn mathematical principles or merely memorize patterns? Rather than designing increasingly complex benchmarks like recent works, we investigate this using elementary two-integer addition ($0$ to $2^{64}$), probing two core properties: commutativity ($A+B=B+A$) and compositional generalization (via isomorphic symbolic mappings, e.g., $7 \rightarrow y$). While state-of-the-art LLMs achieve 73.8-99.8\% accuracy on numerical addition, performance collapses to $\leq$7.5\% under symbolic mapping, indicating failure to generalize learned rules. Non-monotonic performance scaling with digit count and frequent commutativity violations (over 1,700 cases of $A+B \neq B+A$) further support this. Explicitly providing addition rules degrades performance by 81.2\% on average, while self-explanation maintains baseline accuracy, suggesting LLM arithmetic processing is misaligned with human-defined principles. Our findings indicate current LLMs rely on memory pattern over genuine rule learning, highlighting architectural limitations and the need for new approaches to achieve true mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05259v1">How to evaluate control measures for LLM agents? A trajectory from today to superintelligence</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      As LLM agents grow more capable of causing harm autonomously, AI developers will rely on increasingly sophisticated control measures to prevent possibly misaligned agents from causing harm. AI developers could demonstrate that their control measures are sufficient by running control evaluations: testing exercises in which a red team produces agents that try to subvert control measures. To ensure control evaluations accurately capture misalignment risks, the affordances granted to this red team should be adapted to the capability profiles of the agents to be deployed under control measures. In this paper we propose a systematic framework for adapting affordances of red teams to advancing AI capabilities. Rather than assuming that agents will always execute the best attack strategies known to humans, we demonstrate how knowledge of an agents's actual capability profile can inform proportional control evaluations, resulting in more practical and cost-effective control measures. We illustrate our framework by considering a sequence of five fictional models (M1-M5) with progressively advanced capabilities, defining five distinct AI control levels (ACLs). For each ACL, we provide example rules for control evaluation, control measures, and safety cases that could be appropriate. Finally, we show why constructing a compelling AI control safety case for superintelligent LLM agents will require research breakthroughs, highlighting that we might eventually need alternative approaches to mitigating misalignment risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05239v1">LLM-based Automated Grading with Human-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The rise of artificial intelligence (AI) technologies, particularly large language models (LLMs), has brought significant advancements to the field of education. Among various applications, automatic short answer grading (ASAG), which focuses on evaluating open-ended textual responses, has seen remarkable progress with the introduction of LLMs. These models not only enhance grading performance compared to traditional ASAG approaches but also move beyond simple comparisons with predefined "golden" answers, enabling more sophisticated grading scenarios, such as rubric-based evaluation. However, existing LLM-powered methods still face challenges in achieving human-level grading performance in rubric-based assessments due to their reliance on fully automated approaches. In this work, we explore the potential of LLMs in ASAG tasks by leveraging their interactive capabilities through a human-in-the-loop (HITL) approach. Our proposed framework, GradeHITL, utilizes the generative properties of LLMs to pose questions to human experts, incorporating their insights to refine grading rubrics dynamically. This adaptive process significantly improves grading accuracy, outperforming existing methods and bringing ASAG closer to human-level evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09439v2">Spider: Any-to-Many Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) have emerged as an extension of Large Language Models (LLMs), enabling the integration of various modalities. However, Any-to-Any MLLMs are limited to generating pairwise modalities 'Text + X' within a single response, such as Text + {Image or Audio or Video}. To address this limitation, we introduce Spider, a novel efficient Any-to-Many Modalities Generation (AMMG) framework, which can generate an arbitrary combination of modalities 'Text + Xs', such as Text + {Image and Audio and Video}. To achieve efficient AMMG, our Spider integrates three core components: a Base Model for basic X-to-X (i.e., Any-to-Any) modality processing, an Any-to-Many Instruction Template designed for producing Xs signal prompts, and a novel Efficient Decoders-Controller for controlling multimodal Decoders to generate Xs (many-modal) contents. To train Spider, we constructed a novel Text-formatted Many-Modal (TMM) dataset, which facilitates learning the X-to-Xs (i.e., Any-to-Many) capability necessary for AMMG. Ultimately, the well-trained Spider generates a pseudo X-to-Xs dataset, the first-ever X-to-Xs many-modal dataset, enhancing the potential for AMMG tasks in future research. Overall, this work not only pushes the boundary of multimodal interaction but also provides rich data support for advancing the field. Code: https://github.com/Layjins/Spider
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v1">Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05216v1">Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07045v2">Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 6 pages, 0 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Insider threats wield an outsized influence on organizations, disproportionate to their small numbers. This is due to the internal access insiders have to systems, information, and infrastructure. %One example of this influence is where anonymous respondents submit web-based job search site reviews, an insider threat risk to organizations. Signals for such risks may be found in anonymous submissions to public web-based job search site reviews. This research studies the potential for large language models (LLMs) to analyze and detect insider threat sentiment within job site reviews. Addressing ethical data collection concerns, this research utilizes synthetic data generation using LLMs alongside existing job review datasets. A comparative analysis of sentiment scores generated by LLMs is benchmarked against expert human scoring. Findings reveal that LLMs demonstrate alignment with human evaluations in most cases, thus effectively identifying nuanced indicators of threat sentiment. The performance is lower on human-generated data than synthetic data, suggesting areas for improvement in evaluating real-world data. Text diversity analysis found differences between human-generated and LLM-generated datasets, with synthetic data exhibiting somewhat lower diversity. Overall, the results demonstrate the applicability of LLMs to insider threat detection, and a scalable solution for insider sentiment testing by overcoming ethical and logistical barriers tied to data acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05204v1">Quantum Program Linting with LLMs: Emerging Results from a Comparative Study</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Ensuring the quality of quantum programs is increasingly important; however, traditional static analysis techniques are insufficient due to the unique characteristics of quantum computing. Quantum-specific linting tools, such as LintQ, have been developed to detect quantum-specific programming problems; however, they typically rely on manually crafted analysis queries. The manual effort required to update these tools limits their adaptability to evolving quantum programming practices. To address this challenge, this study investigates the feasibility of employing Large Language Models (LLMs) to develop a novel linting technique for quantum software development and explores potential avenues to advance linting approaches. We introduce LintQ-LLM, an LLM-based linting tool designed to detect quantum-specific problems comparable to those identified by LintQ. Through an empirical comparative study using real-world Qiskit programs, our results show that LintQ-LLM is a viable solution that complements LintQ, with particular strengths in problem localization, explanation clarity, and adaptability potential for emerging quantum programming frameworks, thus providing a basis for further research. Furthermore, this study discusses several research opportunities for developing more advanced, adaptable, and feedback-aware quantum software quality assurance methods by leveraging LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05147v1">Pr$εε$mpt: Sanitizing Sensitive Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has introduced new privacy challenges, particularly during inference where sensitive information in prompts may be exposed to proprietary LLM APIs. In this paper, we address the problem of formally protecting the sensitive information contained in a prompt while maintaining response quality. To this end, first, we introduce a cryptographically inspired notion of a prompt sanitizer which transforms an input prompt to protect its sensitive tokens. Second, we propose Pr$\epsilon\epsilon$mpt, a novel system that implements a prompt sanitizer. Pr$\epsilon\epsilon$mpt categorizes sensitive tokens into two types: (1) those where the LLM's response depends solely on the format (such as SSNs, credit card numbers), for which we use format-preserving encryption (FPE); and (2) those where the response depends on specific values, (such as age, salary) for which we apply metric differential privacy (mDP). Our evaluation demonstrates that Pr$\epsilon\epsilon$mpt is a practical method to achieve meaningful privacy guarantees, while maintaining high utility compared to unsanitized prompts, and outperforming prior methods
    </details>
</div>
