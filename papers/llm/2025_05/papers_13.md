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
- Part 13
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10948v1">The Way We Prompt: Conceptual Blending, Neural Dynamics, and Prompt-Induced Transitions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), inspired by neuroscience, exhibit behaviors that often evoke a sense of personality and intelligence-yet the mechanisms behind these effects remain elusive. Here, we operationalize Conceptual Blending Theory (CBT) as an experimental framework, using prompt-based methods to reveal how LLMs blend and compress meaning. By systematically investigating Prompt-Induced Transitions (PIT) and Prompt-Induced Hallucinations (PIH), we uncover structural parallels and divergences between artificial and biological cognition. Our approach bridges linguistics, neuroscience, and empirical AI research, demonstrating that human-AI collaboration can serve as a living prototype for the future of cognitive science. This work proposes prompt engineering not just as a technical tool, but as a scientific method for probing the deep structure of meaning itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10940v1">Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Recommender systems filter contents/items valuable to users by inferring preferences from user features and historical behaviors. Mainstream approaches follow the learning-to-rank paradigm, which focus on discovering and modeling item topics (e.g., categories), and capturing user preferences on these topics based on historical interactions. However, this paradigm often neglects the modeling of user characteristics and their social roles, which are logical confounders influencing the correlated interest and user preference transition. To bridge this gap, we introduce the user role identification task and the behavioral logic modeling task that aim to explicitly model user roles and learn the logical relations between item topics and user social roles. We show that it is possible to explicitly solve these tasks through an efficient integration framework of Large Language Model (LLM) and recommendation systems, for which we propose TagCF. On the one hand, the exploitation of the LLM's world knowledge and logic inference ability produces a virtual logic graph that reveals dynamic and expressive knowledge of users, augmenting the recommendation performance. On the other hand, the user role aligns the user behavioral logic with the observed user feedback, refining our understanding of user behaviors. Additionally, we also show that the extracted user-item logic graph is empirically a general knowledge that can benefit a wide range of recommendation tasks, and conduct experiments on industrial and several public datasets as verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10939v1">GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Accepted to ACL 2025 (main conference, short paper), 10 pages
    </div>
    <details class="paper-abstract">
      Large language models often struggle with zero-shot generalization, and several modular approaches have been proposed to address this challenge. Yet, we hypothesize that a key limitation remains: the entanglement of general knowledge and task-specific adaptations. To overcome this, we propose a modular framework that disentangles these components by constructing a library of task-specific LoRA modules alongside a general-domain LoRA. By subtracting this general knowledge component from each task-specific module, we obtain residual modules that focus more exclusively on task-relevant information, a method we call general knowledge subtraction (GenKnowSub). Leveraging the refined task-specific modules and the Arrow routing algorithm \citep{ostapenko2024towards}, we dynamically select and combine modules for new inputs without additional training. Our studies on the Phi-3 model and standard Arrow as baselines reveal that using general knowledge LoRAs derived from diverse languages, including English, French, and German, yields consistent performance gains in both monolingual and cross-lingual settings across a wide set of benchmarks. Further experiments on Phi-2 demonstrate how GenKnowSub generalizes to weaker LLMs. The complete code and data are available at https://github.com/saharsamr/Modular-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10936v1">Connecting the Dots: A Chain-of-Collaboration Prompting Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 34 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11865v2">From Image to Video, what do we need in multimodal LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Covering from Image LLMs to the more complex Video LLMs, the Multimodal Large Language Models (MLLMs) have demonstrated profound capabilities in comprehending cross-modal information as numerous studies have illustrated. Previous methods delve into designing comprehensive Video LLMs through integrating video foundation models with primitive LLMs. Despite its effectiveness, such paradigm renders Video LLM's structure verbose and typically requires substantial video data for pre-training. Crucially, it neglects leveraging the foundational contributions of ready-made Image LLMs. In this paper, we introduce RED-VILLM, a Resource-Efficient Development pipeline which builds robust Video LLMs through leveraging the prior knowledge of Image LLMs. Specifically, since a video is naturally a combination of images along the temporal dimension, we devise a temporal adaptation plug-and-play structure, endowing the backbone Image LLM with the capability to grasp temporal information. Moreover, through applying this pipeline, we achieve the first Video LLM within the Chinese-speaking community. Extensive experiments demonstrate that Video LLMs developed through our approach surpass conventional Video LLMs, requiring minimal instructional data and training resources. Our approach highlights the potential for a more cost-effective and scalable advancement in multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v4">MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 36 pages, 11 figures. The last version adds more experiments and modifies name for better summary of the work
    </div>
    <details class="paper-abstract">
      The ability to recognize patterns from examples and apply them to new ones is a primal ability for general intelligence, and is widely studied by psychology and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually <10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations often focus on classification, and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context reasoning benchmark for pattern recognition that asks LLM to predict output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for many-shot in-context reasoning, and acquired many insightful findings including scaling effect, robustness, inductive vs. transductive reasoning, retrieval Augmented Generation (RAG), coding for inductive reasoning, cross-domain generalizability, etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10900v1">Explain What You Mean: Intent Augmented Knowledge Graph Recommender Built With LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Interaction sparsity is the primary obstacle for recommendation systems. Sparsity manifests in environments with disproportional cardinality of groupings of entities, such as users and products in an online marketplace. It also is found for newly introduced entities, described as the cold-start problem. Recent efforts to mitigate this sparsity issue shifts the performance bottleneck to other areas in the computational pipeline. Those that focus on enriching sparse representations with connectivity data from other external sources propose methods that are resource demanding and require careful domain expert aided addition of this newly introduced data. Others that turn to Large Language Model (LLM) based recommenders will quickly encounter limitations surrounding data quality and availability. In this work, we propose LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that leverages retrieval-augmented generation and an encoding approach to construct and densify a knowledge graph. IKGR learns latent user-item affinities from an interaction knowledge graph and further densifies it through mutual intent connectivity. This addresses sparsity issues and allows the model to make intent-grounded recommendations with an interpretable embedding translation layer. Through extensive experiments on real-world datasets, we demonstrate that IKGR overcomes knowledge gaps and achieves substantial gains over state-of-the-art baselines on both publicly available and our internal recommendation datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11507v4">TestAgent: A Framework for Domain-Adaptive Evaluation of LLMs via Dynamic Benchmark Construction and Exploratory Interaction</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed to various vertical domains, automatically evaluating their performance across different domains remains a critical challenge. Current evaluation methods often rely on static and resource-intensive datasets that are not aligned with real-world requirements and lack cross-domain adaptability. To address these limitations, we revisit the evaluation process and introduce two key concepts: \textbf{Benchmark+}, which extends the traditional question-answer benchmark into a more flexible ``strategy-criterion'' format; and \textbf{Assessment+}, which enhances the interaction process to facilitate deeper exploration and comprehensive analysis from multiple perspectives. We propose \textbf{\textsc{TestAgent}}, an agent-based evaluation framework that implements these concepts using retrieval-augmented generation and reinforcement learning. \textsc{TestAgent} enables automatic dynamic benchmark generation and in-depth assessment across diverse vertical domains. Experiments on tasks ranging from constructing multiple vertical domain evaluations to transforming static benchmarks into dynamic forms demonstrate the effectiveness of \textsc{TestAgent}. This work provides a novel perspective on automatic evaluation methods for domain-specific LLMs, offering a pathway for domain-adaptive dynamic benchmark construction and exploratory assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06326v5">Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle to provide up-to-date information due to their one-time training and the constantly evolving nature of the world. To keep LLMs current, existing approaches typically involve continued pre-training on new documents. However, they frequently face difficulties in extracting stored knowledge. Motivated by the remarkable success of the Feynman Technique in efficient human learning, we introduce Self-Tuning, a learning framework aimed at improving an LLM's ability to effectively acquire new knowledge from unseen raw documents through self-teaching. Specifically, we develop a Self-Teaching strategy that augments the documents with a set of knowledge-intensive tasks created in a self-supervised manner, focusing on three crucial aspects: memorization, comprehension, and self-reflection. Additionally, we introduce three Wiki-Newpages-2023-QA datasets to facilitate an in-depth analysis of an LLM's knowledge acquisition ability concerning memorization, extraction, and reasoning. Extensive experimental results on various models, e.g., Llama2-7B reveal that Self-Tuning consistently exhibits superior performance across all knowledge acquisition tasks and excels in preserving previous knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10861v1">Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 31 pages (9 for the main paper), 27 figures, NeurIPS 25 submission
    </div>
    <details class="paper-abstract">
      We investigate the usage of Large Language Model (LLM) in collecting high-quality data to warm-start Reinforcement Learning (RL) algorithms for learning in some classical Markov Decision Process (MDP) environments. In this work, we focus on using LLM to generate an off-policy dataset that sufficiently covers state-actions visited by optimal policies, then later using an RL algorithm to explore the environment and improve the policy suggested by the LLM. Our algorithm, LORO, can both converge to an optimal policy and have a high sample efficiency thanks to the LLM's good starting policy. On multiple OpenAI Gym environments, such as CartPole and Pendulum, we empirically demonstrate that LORO outperforms baseline algorithms such as pure LLM-based policies, pure RL, and a naive combination of the two, achieving up to $4 \times$ the cumulative rewards of the pure RL baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07623v6">The Curious Case of Class Accuracy Imbalance in LLMs: Post-hoc Debiasing via Nonlinear Integer Programming</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are good knowledge bases but struggle to perform equally well for all classes in text classification. This paper investigates the case of class accuracy imbalance in LLMs, where deeply entangled pretraining biases and prompt-specific cues contribute to the imbalance. To overcome the difficulty in bias identification and inaccessibility of retraining, we post-hoc balance class accuracy using only output probabilities. This is enabled by reformulating debiasing as a combinatorial optimization problem. In details, we first motivate a post-hoc bias metric, the Contextual Oddity Bias (COBias), to quantify the over-/under-prediction (a tendency to over-predict some classes while under-predicting others) in LLMs. We then propose the Debiasing as Nonlinear Integer Programming (DNIP) method to reweight LLM output class probabilities towards minimizing COBias and max12 imizing overall accuracy, without being constrained by bias sources or updating LLM parameters. Since the DNIP model contains non-differentiable elements, we use simulated annealing to efficiently solve it. Evaluations on five LLMs across NLP classification benchmarks show that DNIP simultaneously achieves signifi16 cant COBias reduction (61% relative reduction) and accuracy improvement (18% relative increase) under different LLM prompting setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10838v1">LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Efficient red-teaming method to uncover vulnerabilities in Large Language Models (LLMs) is crucial. While recent attacks often use LLMs as optimizers, the discrete language space make gradient-based methods struggle. We introduce LARGO (Latent Adversarial Reflection through Gradient Optimization), a novel latent self-reflection attack that reasserts the power of gradient-based optimization for generating fluent jailbreaking prompts. By operating within the LLM's continuous latent space, LARGO first optimizes an adversarial latent vector and then recursively call the same LLM to decode the latent into natural language. This methodology yields a fast, effective, and transferable attack that produces fluent and stealthy prompts. On standard benchmarks like AdvBench and JailbreakBench, LARGO surpasses leading jailbreaking techniques, including AutoDAN, by 44 points in attack success rate. Our findings demonstrate a potent alternative to agentic LLM prompting, highlighting the efficacy of interpreting and attacking LLM internals through gradient optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v1">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. We open source our code at \href{https://github.com/uiuctml/MergeBench}{https://github.com/uiuctml/MergeBench}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10829v1">Enhancing Low-Resource Minority Language Translation with LLMs and Retrieval-Augmented Generation for Cultural Nuances</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Accepted to IntelliSys 2025
    </div>
    <details class="paper-abstract">
      This study investigates the challenges of translating low-resource languages by integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). Various model configurations were tested on Hakka translations, with BLEU scores ranging from 12% (dictionary-only) to 31% (RAG with Gemini 2.0). The best-performing model (Model 4) combined retrieval and advanced language modeling, improving lexical coverage, particularly for specialized or culturally nuanced terms, and enhancing grammatical coherence. A two-stage method (Model 3) using dictionary outputs refined by Gemini 2.0 achieved a BLEU score of 26%, highlighting iterative correction's value and the challenges of domain-specific expressions. Static dictionary-based approaches struggled with context-sensitive content, demonstrating the limitations of relying solely on predefined resources. These results emphasize the need for curated resources, domain knowledge, and ethical collaboration with local communities, offering a framework that improves translation accuracy and fluency while supporting cultural preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09665v2">Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Corrected capitalization errors in the section subtitle 3.4, 4.3, step 1 in section 3.3.2, and Supplementary Information. Fix typo with "Weighting" for step 4 in section 3.3.2
    </div>
    <details class="paper-abstract">
      Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11885v4">Med-R$^2$: Crafting Trustworthy LLM Physicians via Retrieval and Reasoning of Evidence-Based Medicine</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. Despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.74\% improvement over vanilla RAG methods and even a 3.32\% enhancement compared to fine-tuning strategies, without incurring additional training costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12632v2">What External Knowledge is Preferred by LLMs? Characterizing and Exploring Chain of Evidence in Imperfect Context</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Incorporating external knowledge into large language models (LLMs) has emerged as a promising approach to mitigate outdated knowledge and hallucination in LLMs. However, external knowledge is often imperfect. In addition to useful knowledge, external knowledge is rich in irrelevant or misinformation in the context that can impair the reliability of LLM responses. This paper focuses on LLMs' preferred external knowledge in imperfect contexts when handling multi-hop QA. Inspired by criminal procedural law's Chain of Evidence (CoE), we characterize that knowledge preferred by LLMs should maintain both relevance to the question and mutual support among knowledge pieces. Accordingly, we propose an automated CoE discrimination approach and evaluate LLMs' effectiveness, faithfulness and robustness with CoE, including its application in the Retrieval-Augmented Generation (RAG). Tests on five LLMs show CoE improves generation accuracy, answer faithfulness, robustness to knowledge conflicts, and boosts the performance of existing approaches in three practical RAG scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10782v1">EdgeMM: Multi-Core CPU with Heterogeneous AI-Extension and Activation-aware Weight Pruning for Multimodal LLMs at Edge</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Accepted by DAC 2025
    </div>
    <details class="paper-abstract">
      Emerging multimodal LLMs (MLLMs) exhibit strong cross-modality perception and reasoning capabilities and hold great potential for various applications at edge. However, MLLMs typically consist of a compute-intensive modality encoder and a memory-bound LLM decoder, leading to distinct bottlenecks for hardware designs. In this work, we present a multi-core CPU solution with heterogeneous AI extensions, which are based on either the compute-centric systolic array or memory-centric digital compute-in-memory (CIM) co-processors. In addition, dynamic activation-aware weight pruning and bandwidth management are developed to enhance bandwidth efficiency and core utilization, improving overall performance. We implemented our solution using commercial 22nm technology. For representative MLLMs, our evaluations show EdgeMM can achieve 2.84x performance speedup compared to laptop 3060 GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03049v2">34 Examples of LLM Applications in Materials Science and Chemistry: Towards Automation, Assistants, Agents, and Accelerated Scientific Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.15221. This paper is a refinement and analysis of the raw project submissions from arXiv:2411.15221
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping many aspects of materials science and chemistry research, enabling advances in molecular property prediction, materials design, scientific automation, knowledge extraction, and more. Recent developments demonstrate that the latest class of models are able to integrate structured and unstructured data, assist in hypothesis generation, and streamline research workflows. To explore the frontier of LLM capabilities across the research lifecycle, we review applications of LLMs through 34 total projects developed during the second annual Large Language Model Hackathon for Applications in Materials Science and Chemistry, a global hybrid event. These projects spanned seven key research areas: (1) molecular and material property prediction, (2) molecular and material design, (3) automation and novel interfaces, (4) scientific communication and education, (5) research data management and automation, (6) hypothesis generation and evaluation, and (7) knowledge extraction and reasoning from the scientific literature. Collectively, these applications illustrate how LLMs serve as versatile predictive models, platforms for rapid prototyping of domain-specific tools, and much more. In particular, improvements in both open source and proprietary LLM performance through the addition of reasoning, additional training data, and new techniques have expanded effectiveness, particularly in low-data environments and interdisciplinary research. As LLMs continue to improve, their integration into scientific workflows presents both new opportunities and new challenges, requiring ongoing exploration, continued refinement, and further research to address reliability, interpretability, and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03266v2">LLM Content Moderation and User Satisfaction: Evidence from Response Refusals in Chatbot Arena</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      LLM safety and ethical alignment are widely discussed, but the impact of content moderation on user satisfaction remains underexplored. In particular, little is known about how users respond when models refuse to answer a prompt-one of the primary mechanisms used to enforce ethical boundaries in LLMs. We address this gap by analyzing nearly 50,000 model comparisons from Chatbot Arena, a platform where users indicate their preferred LLM response in pairwise matchups, providing a large-scale setting for studying real-world user preferences. Using a novel RoBERTa-based refusal classifier fine-tuned on a hand-labeled dataset, we distinguish between refusals due to ethical concerns and technical limitations. Our results reveal a substantial refusal penalty: ethical refusals yield significantly lower win rates than both technical refusals and standard responses, indicating that users are especially dissatisfied when models decline a task for ethical reasons. However, this penalty is not uniform. Refusals receive more favorable evaluations when the underlying prompt is highly sensitive (e.g., involving illegal content), and when the refusal is phrased in a detailed and contextually aligned manner. These findings underscore a core tension in LLM design: safety-aligned behaviors may conflict with user expectations, calling for more adaptive moderation strategies that account for context and presentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10774v1">Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Time series forecasting is important for applications spanning energy markets, climate analysis, and traffic management. However, existing methods struggle to effectively integrate exogenous texts and align them with the probabilistic nature of large language models (LLMs). Current approaches either employ shallow text-time series fusion via basic prompts or rely on deterministic numerical decoding that conflict with LLMs' token-generation paradigm, which limits contextual awareness and distribution modeling. To address these limitations, we propose CAPTime, a context-aware probabilistic multimodal time series forecasting method that leverages text-informed abstraction and autoregressive LLM decoding. Our method first encodes temporal patterns using a pretrained time series encoder, then aligns them with textual contexts via learnable interactions to produce joint multimodal representations. By combining a mixture of distribution experts with frozen LLMs, we enable context-aware probabilistic forecasting while preserving LLMs' inherent distribution modeling capabilities. Experiments on diverse time series forecasting tasks demonstrate the superior accuracy and generalization of CAPTime, particularly in multimodal scenarios. Additional analysis highlights its robustness in data-scarce scenarios through hybrid probabilistic decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11718v1">REMOR: Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 18 pages, 6 figures
    </div>
    <details class="paper-abstract">
      AI-based peer review systems tend to produce shallow and overpraising suggestions compared to human feedback. Here, we evaluate how well a reasoning LLM trained with multi-objective reinforcement learning (REMOR) can overcome these limitations. We start by designing a multi-aspect reward function that aligns with human evaluation of reviews. The aspects are related to the review itself (e.g., criticisms, novelty) and the relationship between the review and the manuscript (i.e., relevance). First, we perform supervised fine-tuning of DeepSeek-R1-Distill-Qwen-7B using LoRA on PeerRT, a new dataset of high-quality top AI conference reviews enriched with reasoning traces. We then apply Group Relative Policy Optimization (GRPO) to train two models: REMOR-H (with the human-aligned reward) and REMOR-U (with a uniform reward). Interestingly, the human-aligned reward penalizes aspects typically associated with strong reviews, leading REMOR-U to produce qualitatively more substantive feedback. Our results show that REMOR-U and REMOR-H achieve more than twice the average rewards of human reviews, non-reasoning state-of-the-art agentic multi-modal AI review systems, and general commercial LLM baselines. We found that while the best AI and human reviews are comparable in quality, REMOR avoids the long tail of low-quality human reviews. We discuss how reasoning is key to achieving these improvements and release the Human-aligned Peer Review Reward (HPRR) function, the Peer Review Reasoning-enriched Traces (PeerRT) dataset, and the REMOR models, which we believe can help spur progress in the area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11715v1">ConflictLens: LLM-Based Conflict Resolution Training in Romantic Relationship</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Romantic conflicts are often rooted in deep psychological factors such as coping styles, emotional responses, and communication habits. Existing systems tend to address surface-level behaviors or isolated events, offering limited support for understanding the underlying dynamics. We present ConflictLens, an interactive system that leverages psychological theory and large language models (LLMs) to help individuals analyze and reflect on the deeper mechanisms behind their conflicts. The system provides multi-level strategy recommendations and guided dialogue exercises, including annotation, rewriting, and continuation tasks. A case study demonstrates how ConflictLens supports emotional insight, improves relational understanding, and fosters more constructive communication. This work offers a novel approach to supporting self-awareness and growth in romantic relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00234v3">Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Improving Large Language Model (LLM) agents for sequential decision-making tasks typically requires extensive task-specific knowledge engineering--custom prompts, curated examples, and specialized observation/action spaces. We investigate a different approach where agents automatically improve by learning from their own successful experiences without human intervention. Our method constructs and refines a database of self-generated trajectories that serve as in-context examples for future tasks. Even naive accumulation of successful trajectories yields substantial performance gains across three diverse benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%). These improvements exceed those achieved by upgrading from gpt-4o-mini to gpt-4o and match the performance of allowing multiple attempts per task. We further enhance this approach with two innovations: database-level curation using population-based training to propagate high-performing example collections, and exemplar-level curation that selectively retains trajectories based on their empirical utility as in-context examples. With these enhancements, our method achieves 93% success on ALFWorld--surpassing approaches that use more powerful LLMs and hand-crafted components. Our trajectory bootstrapping technique demonstrates that agents can autonomously improve through experience, offering a scalable alternative to labor-intensive knowledge engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11701v1">DMN-Guided Prompting: A Low-Code Framework for Controlling LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Large Language Models, Decision Model and Notation, Prompt Engineering, Automated Feedback
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown considerable potential in automating decision logic within knowledge-intensive processes. However, their effectiveness largely depends on the strategy and quality of prompting. Since decision logic is typically embedded in prompts, it becomes challenging for end users to modify or refine it. Decision Model and Notation (DMN) offers a standardized graphical approach for defining decision logic in a structured, user-friendly manner. This paper introduces a DMN-guided prompting framework that breaks down complex decision logic into smaller, manageable components, guiding LLMs through structured decision pathways. We implemented the framework in a graduate-level course where students submitted assignments. The assignments and DMN models representing feedback instructions served as inputs to our framework. The instructor evaluated the generated feedback and labeled it for performance assessment. Our approach demonstrated promising results, outperforming chain-of-thought (CoT) prompting. Students also responded positively to the generated feedback, reporting high levels of perceived usefulness in a survey based on the Technology Acceptance Model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11618v1">Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-16
    </div>
    <details class="paper-abstract">
      Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11584v1">LLM Agents Are Hypersensitive to Nudges</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 33 pages, 28 figures
    </div>
    <details class="paper-abstract">
      LLMs are being set loose in complex, real-world environments involving sequential decision-making and tool use. Often, this involves making choices on behalf of human users. However, not much is known about the distribution of such choices, and how susceptible they are to different choice architectures. We perform a case study with a few such LLM models on a multi-attribute tabular decision-making problem, under canonical nudges such as the default option, suggestions, and information highlighting, as well as additional prompting strategies. We show that, despite superficial similarities to human choice distributions, such models differ in subtle but important ways. First, they show much higher susceptibility to the nudges. Second, they diverge in points earned, being affected by factors like the idiosyncrasy of available prizes. Third, they diverge in information acquisition strategies: e.g. incurring substantial cost to reveal too much information, or selecting without revealing any. Moreover, we show that simple prompt strategies like zero-shot chain of thought (CoT) can shift the choice distribution, and few-shot prompting with human data can induce greater alignment. Yet, none of these methods resolve the sensitivity of these models to nudges. Finally, we show how optimal nudges optimized with a human resource-rational model can similarly increase LLM performance for some models. All these findings suggest that behavioral tests are needed before deploying models as agents or assistants acting on behalf of users in complex environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11565v1">ACSE-Eval: Can LLMs threat model real-world cloud infrastructure?</a></div>
    <div class="paper-meta">
      📅 2025-05-16
      | 💬 Submitted to the 39th Annual Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      While Large Language Models have shown promise in cybersecurity applications, their effectiveness in identifying security threats within cloud deployments remains unexplored. This paper introduces AWS Cloud Security Engineering Eval, a novel dataset for evaluating LLMs cloud security threat modeling capabilities. ACSE-Eval contains 100 production grade AWS deployment scenarios, each featuring detailed architectural specifications, Infrastructure as Code implementations, documented security vulnerabilities, and associated threat modeling parameters. Our dataset enables systemic assessment of LLMs abilities to identify security risks, analyze attack vectors, and propose mitigation strategies in cloud environments. Our evaluations on ACSE-Eval demonstrate that GPT 4.1 and Gemini 2.5 Pro excel at threat identification, with Gemini 2.5 Pro performing optimally in 0-shot scenarios and GPT 4.1 showing superior results in few-shot settings. While GPT 4.1 maintains a slight overall performance advantage, Claude 3.7 Sonnet generates the most semantically sophisticated threat models but struggles with threat categorization and generalization. To promote reproducibility and advance research in automated cybersecurity threat analysis, we open-source our dataset, evaluation metrics, and methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09986v2">From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Motivated by the remarkable success of artificial intelligence (AI) across diverse fields, the application of AI to solve scientific problems, often formulated as partial differential equations (PDEs), has garnered increasing attention. While most existing research concentrates on theoretical properties (such as well-posedness, regularity, and continuity) of the solutions, alongside direct AI-driven methods for solving PDEs, the challenge of uncovering symbolic relationships within these equations remains largely unexplored. In this paper, we propose leveraging large language models (LLMs) to learn such symbolic relationships. Our results demonstrate that LLMs can effectively predict the operators involved in PDE solutions by utilizing the symbolic information in the PDEs both theoretically and numerically. Furthermore, we show that discovering these symbolic relationships can substantially improve both the efficiency and accuracy of symbolic machine learning for finding analytical approximation of PDE solutions, delivering a fully interpretable solution pipeline. This work opens new avenues for understanding the symbolic structure of scientific problems and advancing their solution processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14609v2">DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 11 pages, 6 figures. SIGIR '25 Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval July 13--18, 2025 Padua, Italy
    </div>
    <details class="paper-abstract">
      Conversational Search (CS) involves retrieving relevant documents from a corpus while considering the conversational context, integrating retrieval with context modeling. Recent advancements in Large Language Models (LLMs) have significantly enhanced CS by enabling query rewriting based on conversational context. However, employing LLMs during inference poses efficiency challenges. Existing solutions mitigate this issue by distilling embeddings derived from human-rewritten queries, focusing primarily on learning the context modeling task. These methods, however, often separate the contrastive retrieval task from the distillation process, treating it as an independent loss term. To overcome these limitations, we introduce DiSCo (Distillation of Sparse Conversational retrieval), a novel approach that unifies retrieval and context modeling through a relaxed distillation objective. Instead of relying exclusively on representation learning, our method distills similarity scores between conversations and documents, providing more freedom in the representation space and better leveraging the contrastive nature of document relevance. Extensive experiments on Learned Sparse Retrieval (LSR) across five CS datasets demonstrate that DiSCo achieves substantial improvements in both in-domain and out-of-domain retrieval tasks, achieving up to a six-point gain in recall for out-of-domain datasets over state-of-the-art methods. Additionally, DiSCo employs a multi-teacher distillation strategy, using multiple LLMs as teachers, further enhancing performance and surpassing the individual teachers in in-domain settings. Furthermore, analysis of model sparsity reveals that DiSCo allows for more effective control over the sparsity of the trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10708v1">SafeTrans: LLM-assisted Transpilation from C to Rust</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Rust is a strong contender for a memory-safe alternative to C as a "systems" programming language, but porting the vast amount of existing C code to Rust is a daunting task. In this paper, we evaluate the potential of large language models (LLMs) to automate the transpilation of C code to idiomatic Rust, while ensuring that the generated code mitigates any memory-related vulnerabilities present in the original code. To that end, we present the design and implementation of SafeTrans, a framework that uses LLMs to i) transpile C code into Rust and ii) iteratively fix any compilation and runtime errors in the resulting code. A key novelty of our approach is the introduction of a few-shot guided repair technique for translation errors, which provides contextual information and example code snippets for specific error types, guiding the LLM toward the correct solution. Another novel aspect of our work is the evaluation of the security implications of the transpilation process, i.e., whether potential vulnerabilities in the original C code have been properly addressed in the translated Rust code. We experimentally evaluated SafeTrans with six leading LLMs and a set of 2,653 C programs accompanied by comprehensive unit tests, which were used for validating the correctness of the translated code. Our results show that our iterative repair strategy improves the rate of successful translations from 54% to 80% for the best-performing LLM (GPT-4o), and that all types of identified vulnerabilities in the original C code are effectively mitigated in the translated Rust code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v2">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      This paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: Although AI is becoming cheaper and faster, its global adoption drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10681v1">Towards an LLM-powered Social Digital Twinning Platform</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 13 pages, 3 figures, 23rd International Conference on Practical applications of Agents and Multi-Agent Systems (PAAMS 2025)
    </div>
    <details class="paper-abstract">
      We present Social Digital Twinner, an innovative social simulation tool for exploring plausible effects of what-if scenarios in complex adaptive social systems. The architecture is composed of three seamlessly integrated parts: a data infrastructure featuring real-world data and a multi-dimensionally representative synthetic population of citizens, an LLM-enabled agent-based simulation engine, and a user interface that enable intuitive, natural language interactions with the simulation engine and the artificial agents (i.e. citizens). Social Digital Twinner facilitates real-time engagement and empowers stakeholders to collaboratively design, test, and refine intervention measures. The approach is promoting a data-driven and evidence-based approach to societal problem-solving. We demonstrate the tool's interactive capabilities by addressing the critical issue of youth school dropouts in Kragero, Norway, showcasing its ability to create and execute a dedicated social digital twin using natural language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10670v1">Interpretable Risk Mitigation in LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Autonomous agents powered by large language models (LLMs) enable novel use cases in domains where responsible action is increasingly important. Yet the inherent unpredictability of LLMs raises safety concerns about agent reliability. In this work, we explore agent behaviour in a toy, game-theoretic environment based on a variation of the Iterated Prisoner's Dilemma. We introduce a strategy-modification method-independent of both the game and the prompt-by steering the residual stream with interpretable features extracted from a sparse autoencoder latent space. Steering with the good-faith negotiation feature lowers the average defection probability by 28 percentage points. We also identify feasible steering ranges for several open-source LLM agents. Finally, we hypothesise that game-theoretic evaluation of LLM agents, combined with representation-steering alignment, can generalise to real-world applications on end-user devices and embodied platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10597v1">Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Reward models (RMs) are essential for aligning large language models (LLMs) with human values. However, noisy preferences in human feedback often lead to reward misgeneralization, where RMs overfit to spurious patterns and provide misleading signals during policy optimization. We systematically analyze the training dynamics of preference pairs and identify that noisy examples are harder to fit and introduce instability. Empirical evidence shows that LLMs optimized using reward models trained on full noisy datasets perform worse than those trained on filtered, high-quality preferences. To address this, we propose Collaborative Reward Modeling (CRM), an online framework that enhances robustness by combining peer review and curriculum learning. Two reward models are trained in parallel and assess each other's data selections to filter out potential noise. Curriculum learning structures the preference data from easy to hard, ensuring synchronized training and stable feedback. Extensive experiments demonstrate that CRM improves generalization, with up to 9.94 points of accuracy gain on RewardBench under 40 percent label noise. CRM is also compatible with implicit-reward alignment methods, offering a practical and versatile strategy for robust alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10593v1">LLM-Explorer: Towards Efficient and Affordable LLM-based Exploration for Mobile Apps</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Accepted by MobiCom 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have opened new opportunities for automated mobile app exploration, an important and challenging problem that used to suffer from the difficulty of generating meaningful UI interactions. However, existing LLM-based exploration approaches rely heavily on LLMs to generate actions in almost every step, leading to a huge cost of token fees and computational resources. We argue that such extensive usage of LLMs is neither necessary nor effective, since many actions during exploration do not require, or may even be biased by the abilities of LLMs. Further, based on the insight that a precise and compact knowledge plays the central role for effective exploration, we introduce LLM-Explorer, a new exploration agent designed for efficiency and affordability. LLM-Explorer uses LLMs primarily for maintaining the knowledge instead of generating actions, and knowledge is used to guide action generation in a LLM-less manner. Based on a comparison with 5 strong baselines on 20 typical apps, LLM-Explorer was able to achieve the fastest and highest coverage among all automated app explorers, with over 148x lower cost than the state-of-the-art LLM-based approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10495v1">RouteNator: A Router-Based Multi-Modal Architecture for Generating Synthetic Training Data for Function Calling LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing
    </div>
    <details class="paper-abstract">
      This paper addresses fine-tuning Large Language Models (LLMs) for function calling tasks when real user interaction data is unavailable. In digital content creation tools, where users express their needs through natural language queries that must be mapped to API calls, the lack of real-world task-specific data and privacy constraints for training on it necessitate synthetic data generation. Existing approaches to synthetic data generation fall short in diversity and complexity, failing to replicate real-world data distributions and leading to suboptimal performance after LLM fine-tuning. We present a novel router-based architecture that leverages domain resources like content metadata and structured knowledge graphs, along with text-to-text and vision-to-text language models to generate high-quality synthetic training data. Our architecture's flexible routing mechanism enables synthetic data generation that matches observed real-world distributions, addressing a fundamental limitation of traditional approaches. Evaluation on a comprehensive set of real user queries demonstrates significant improvements in both function classification accuracy and API parameter selection. Models fine-tuned with our synthetic data consistently outperform traditional approaches, establishing new benchmarks for function calling tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10490v1">Campus AI vs Commercial AI: A Late-Breaking Study on How LLM As-A-Service Customizations Shape Trust and Usage Patterns</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      As the use of Large Language Models (LLMs) by students, lecturers and researchers becomes more prevalent, universities - like other organizations - are pressed to develop coherent AI strategies. LLMs as-a-Service (LLMaaS) offer accessible pre-trained models, customizable to specific (business) needs. While most studies prioritize data, model, or infrastructure adaptations (e.g., model fine-tuning), we focus on user-salient customizations, like interface changes and corporate branding, which we argue influence users' trust and usage patterns. This study serves as a functional prequel to a large-scale field study in which we examine how students and employees at a German university perceive and use their institution's customized LLMaaS compared to ChatGPT. The goals of this prequel are to stimulate discussions on psychological effects of LLMaaS customizations and refine our research approach through feedback. Our forthcoming findings will deepen the understanding of trust dynamics in LLMs, providing practical guidance for organizations considering LLMaaS deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v4">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10425v1">Learning to Think: Information-Theoretic Reinforcement Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at complex tasks thanks to advances in reasoning abilities. However, existing methods overlook the trade-off between reasoning effectiveness and computational efficiency, often encouraging unnecessarily long reasoning chains and wasting tokens. To address this, we propose Learning to Think (L2T), an information-theoretic reinforcement fine-tuning framework for LLMs to make the models achieve optimal reasoning with fewer tokens. Specifically, L2T treats each query-response interaction as a hierarchical session of multiple episodes and proposes a universal dense process reward, i.e., quantifies the episode-wise information gain in parameters, requiring no extra annotations or task-specific evaluators. We propose a method to quickly estimate this reward based on PAC-Bayes bounds and the Fisher information matrix. Theoretical analyses show that it significantly reduces computational complexity with high estimation accuracy. By immediately rewarding each episode's contribution and penalizing excessive updates, L2T optimizes the model via reinforcement learning to maximize the use of each episode and achieve effective updates. Empirical results on various reasoning benchmarks and base models demonstrate the advantage of L2T across different tasks, boosting both reasoning effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10409v1">Are LLM-generated plain language summaries truly understandable? A large-scale crowdsourced evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Plain language summaries (PLSs) are essential for facilitating effective communication between clinicians and patients by making complex medical information easier for laypeople to understand and act upon. Large language models (LLMs) have recently shown promise in automating PLS generation, but their effectiveness in supporting health information comprehension remains unclear. Prior evaluations have generally relied on automated scores that do not measure understandability directly, or subjective Likert-scale ratings from convenience samples with limited generalizability. To address these gaps, we conducted a large-scale crowdsourced evaluation of LLM-generated PLSs using Amazon Mechanical Turk with 150 participants. We assessed PLS quality through subjective Likert-scale ratings focusing on simplicity, informativeness, coherence, and faithfulness; and objective multiple-choice comprehension and recall measures of reader understanding. Additionally, we examined the alignment between 10 automated evaluation metrics and human judgments. Our findings indicate that while LLMs can generate PLSs that appear indistinguishable from human-written ones in subjective evaluations, human-written PLSs lead to significantly better comprehension. Furthermore, automated evaluation metrics fail to reflect human judgment, calling into question their suitability for evaluating PLSs. This is the first study to systematically evaluate LLM-generated PLSs based on both reader preferences and comprehension outcomes. Our findings highlight the need for evaluation frameworks that move beyond surface-level quality and for generation methods that explicitly optimize for layperson comprehension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10402v1">Rethinking Repetition Problems of LLMs in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      With the advent of neural language models, the performance of code generation has been significantly boosted. However, the problem of repetitions during the generation process continues to linger. Previous work has primarily focused on content repetition, which is merely a fraction of the broader repetition problem in code generation. A more prevalent and challenging problem is structural repetition. In structural repetition, the repeated code appears in various patterns but possesses a fixed structure, which can be inherently reflected in grammar. In this paper, we formally define structural repetition and propose an efficient decoding approach called RPG, which stands for Repetition Penalization based on Grammar, to alleviate the repetition problems in code generation for LLMs. Specifically, RPG first leverages grammar rules to identify repetition problems during code generation, and then strategically decays the likelihood of critical tokens that contribute to repetitions, thereby mitigating them in code generation. To facilitate this study, we construct a new dataset CodeRepetEval to comprehensively evaluate approaches for mitigating the repetition problems in code generation. Extensive experimental results demonstrate that RPG substantially outperforms the best-performing baselines on CodeRepetEval dataset as well as HumanEval and MBPP benchmarks, effectively reducing repetitions and enhancing the quality of generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06046v2">Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 24 pages, 10 pages main text
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, currently little is known about LLM knowledge of UK Government public health information. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest private LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Importantly we find in both setups LLMs have higher accuracy on guidance intended for the general public. Therefore, there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, but additional safeguards or tools may still be needed when providing free form responses on public health topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10321v1">AutoPentest: Enhancing Vulnerability Management With Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 24 pages, 1 figure, for implementation, see https://github.com/JuliusHenke/autopentest
    </div>
    <details class="paper-abstract">
      A recent area of increasing research is the use of Large Language Models (LLMs) in penetration testing, which promises to reduce costs and thus allow for higher frequency. We conduct a review of related work, identifying best practices and common evaluation issues. We then present AutoPentest, an application for performing black-box penetration tests with a high degree of autonomy. AutoPentest is based on the LLM GPT-4o from OpenAI and the LLM agent framework LangChain. It can perform complex multi-step tasks, augmented by external tools and knowledge bases. We conduct a study on three capture-the-flag style Hack The Box (HTB) machines, comparing our implementation AutoPentest with the baseline approach of manually using the ChatGPT-4o user interface. Both approaches are able to complete 15-25 % of the subtasks on the HTB machines, with AutoPentest slightly outperforming ChatGPT. We measure a total cost of \$96.20 US when using AutoPentest across all experiments, while a one-month subscription to ChatGPT Plus costs \$20. The results show that further implementation efforts and the use of more powerful LLMs released in the future are likely to make this a viable part of vulnerability management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10320v1">J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 10 pages, 8 tables, 11 figures
    </div>
    <details class="paper-abstract">
      The progress of AI is bottlenecked by the quality of evaluation, and powerful LLM-as-a-Judge models have proved to be a core solution. Improved judgment ability is enabled by stronger chain-of-thought reasoning, motivating the need to find the best recipes for training such models to think. In this work we introduce J1, a reinforcement learning approach to training such models. Our method converts both verifiable and non-verifiable prompts to judgment tasks with verifiable rewards that incentivize thinking and mitigate judgment bias. In particular, our approach outperforms all other existing 8B or 70B models when trained at those sizes, including models distilled from DeepSeek-R1. J1 also outperforms o1-mini, and even R1 on some benchmarks, despite training a smaller model. We provide analysis and ablations comparing Pairwise-J1 vs Pointwise-J1 models, offline vs online training recipes, reward strategies, seed prompts, and variations in thought length and content. We find that our models make better judgments by learning to outline evaluation criteria, comparing against self-generated reference answers, and re-evaluating the correctness of model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10260v1">Comparing LLM Text Annotation Skills: A Study on Human Rights Violations in Social Media Data</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      In the era of increasingly sophisticated natural language processing (NLP) systems, large language models (LLMs) have demonstrated remarkable potential for diverse applications, including tasks requiring nuanced textual understanding and contextual reasoning. This study investigates the capabilities of multiple state-of-the-art LLMs - GPT-3.5, GPT-4, LLAMA3, Mistral 7B, and Claude-2 - for zero-shot and few-shot annotation of a complex textual dataset comprising social media posts in Russian and Ukrainian. Specifically, the focus is on the binary classification task of identifying references to human rights violations within the dataset. To evaluate the effectiveness of these models, their annotations are compared against a gold standard set of human double-annotated labels across 1000 samples. The analysis includes assessing annotation performance under different prompting conditions, with prompts provided in both English and Russian. Additionally, the study explores the unique patterns of errors and disagreements exhibited by each model, offering insights into their strengths, limitations, and cross-linguistic adaptability. By juxtaposing LLM outputs with human annotations, this research contributes to understanding the reliability and applicability of LLMs for sensitive, domain-specific tasks in multilingual contexts. It also sheds light on how language models handle inherently subjective and context-dependent judgments, a critical consideration for their deployment in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10259v1">SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Efficient LLM inference on resource-constrained devices presents significant challenges in compute and memory utilization. Due to limited GPU memory, existing systems offload model weights to CPU memory, incurring substantial I/O overhead between the CPU and GPU. This leads to two major inefficiencies: (1) GPU cores are underutilized, often remaining idle while waiting for data to be loaded; and (2) GPU memory has low impact on performance, as reducing its capacity has minimal effect on overall throughput.In this paper, we propose SpecOffload, a high-throughput inference engine that embeds speculative decoding into offloading. Our key idea is to unlock latent GPU resources for storing and executing a draft model used for speculative decoding, thus accelerating inference at near-zero additional cost. To support this, we carefully orchestrate the interleaved execution of target and draft models in speculative decoding within the offloading pipeline, and propose a planner to manage tensor placement and select optimal parameters. Compared to the best baseline, SpecOffload improves GPU core utilization by 4.49x and boosts inference throughput by 2.54x. Our code is available at https://github.com/MobiSense/SpecOffload .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15674v2">TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Accpeted for IEEE International Joint Conference on Neural Networks (IJCNN 2025). The code is available at https://github.com/guyuxuan9/TensorLLM
    </div>
    <details class="paper-abstract">
      The reasoning abilities of Large Language Models (LLMs) can be improved by structurally denoising their weights, yet existing techniques primarily focus on denoising the feed-forward network (FFN) of the transformer block, and can not efficiently utilise the Multi-head Attention (MHA) block, which is the core of transformer architectures. To address this issue, we propose a novel intuitive framework that, at its very core, performs MHA compression through a multi-head tensorisation process and the Tucker decomposition. This enables both higher-dimensional structured denoising and compression of the MHA weights, by enforcing a shared higher-dimensional subspace across the weights of the multiple attention heads. We demonstrate that this approach consistently enhances the reasoning capabilities of LLMs across multiple benchmark datasets, and for both encoder-only and decoder-only architectures, while achieving compression rates of up to $\sim 250$ times in the MHA weights, all without requiring any additional data, training, or fine-tuning. Furthermore, we show that the proposed method can be seamlessly combined with existing FFN-only-based denoising techniques to achieve further improvements in LLM reasoning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10218v1">RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Role-playing conversational agents (RPCAs) face persistent challenges in maintaining role consistency. To address this, we propose RAIDEN-R1, a novel reinforcement learning framework that integrates Verifiable Role-Awareness Reward (VRAR). The method introduces both singular and multi-term mining strategies to generate quantifiable rewards by assessing role-specific keys. Additionally, we construct a high-quality, role-aware Chain-of-Thought dataset through multi-LLM collaboration, and implement experiments to enhance reasoning coherence. Experiments on the RAIDEN benchmark demonstrate RAIDEN-R1's superiority: our 14B-GRPO model achieves 88.04% and 88.65% accuracy on Script-Based Knowledge and Conversation Memory metrics, respectively, outperforming baseline models while maintaining robustness. Case analyses further reveal the model's enhanced ability to resolve conflicting contextual cues and sustain first-person narrative consistency. This work bridges the non-quantifiability gap in RPCA training and provides insights into role-aware reasoning patterns, advancing the development of RPCAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10213v1">Informed Forecasting: Leveraging Auxiliary Knowledge to Boost LLM Performance on Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), there is a growing need to establish best practices for leveraging their capabilities beyond traditional natural language tasks. In this paper, a novel cross-domain knowledge transfer framework is proposed to enhance the performance of LLMs in time series forecasting -- a task of increasing relevance in fields such as energy systems, finance, and healthcare. The approach systematically infuses LLMs with structured temporal information to improve their forecasting accuracy. This study evaluates the proposed method on a real-world time series dataset and compares it to a naive baseline where the LLM receives no auxiliary information. Results show that knowledge-informed forecasting significantly outperforms the uninformed baseline in terms of predictive accuracy and generalization. These findings highlight the potential of knowledge transfer strategies to bridge the gap between LLMs and domain-specific forecasting tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10212v1">Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly central to recommendation scenarios due to their remarkable natural language understanding and generation capabilities. Although significant research has explored the use of LLMs for various recommendation tasks, little effort has been dedicated to verifying whether they have memorized public recommendation dataset as part of their training data. This is undesirable because memorization reduces the generalizability of research findings, as benchmarking on memorized datasets does not guarantee generalization to unseen datasets. Furthermore, memorization can amplify biases, for example, some popular items may be recommended more frequently than others. In this work, we investigate whether LLMs have memorized public recommendation datasets. Specifically, we examine two model families (GPT and Llama) across multiple sizes, focusing on one of the most widely used dataset in recommender systems: MovieLens-1M. First, we define dataset memorization as the extent to which item attributes, user profiles, and user-item interactions can be retrieved by prompting the LLMs. Second, we analyze the impact of memorization on recommendation performance. Lastly, we examine whether memorization varies across model families and model sizes. Our results reveal that all models exhibit some degree of memorization of MovieLens-1M, and that recommendation performance is related to the extent of memorization. We have made all the code publicly available at: https://github.com/sisinflab/LLM-MemoryInspector
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10325v2">Collaborative Speculative Inference for Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Speculative inference is a promising paradigm employing small speculative models (SSMs) as drafters to generate draft tokens, which are subsequently verified in parallel by the target large language model (LLM). This approach enhances the efficiency of inference serving by reducing LLM inference latency and costs while preserving generation quality. However, existing speculative methods face critical challenges, including inefficient resource utilization and limited draft acceptance, which constrain their scalability and overall effectiveness. To overcome these obstacles, we present CoSine, a novel speculative inference system that decouples sequential speculative decoding from parallel verification, enabling efficient collaboration among multiple nodes. Specifically, CoSine routes inference requests to specialized drafters based on their expertise and incorporates a confidence-based token fusion mechanism to synthesize outputs from cooperating drafters, ensuring high-quality draft generation. Additionally, CoSine dynamically orchestrates the execution of speculative decoding and verification in a pipelined manner, employing batch scheduling to selectively group requests and adaptive speculation control to minimize idle periods. By optimizing parallel workflows through heterogeneous node collaboration, CoSine balances draft generation and verification throughput in real-time, thereby maximizing resource utilization. Experimental results demonstrate that CoSine achieves superior performance compared to state-of-the-art speculative approaches. Notably, with equivalent resource costs, CoSine achieves up to a 23.2% decrease in latency and a 32.5% increase in throughput compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10182v1">Mining Hidden Thoughts from Texts: Evaluating Continual Pretraining with Synthetic Data for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant improvements in reasoning capabilities through supervised fine-tuning and reinforcement learning. However, when training reasoning models, these approaches are primarily applicable to specific domains such as mathematics and programming, which imposes fundamental constraints on the breadth and scalability of training data. In contrast, continual pretraining (CPT) offers the advantage of not requiring task-specific signals. Nevertheless, how to effectively synthesize training data for reasoning and how such data affect a wide range of domains remain largely unexplored. This study provides a detailed evaluation of Reasoning CPT, a form of CPT that uses synthetic data to reconstruct the hidden thought processes underlying texts, based on the premise that texts are the result of the author's thinking process. Specifically, we apply Reasoning CPT to Gemma2-9B using synthetic data with hidden thoughts derived from STEM and Law corpora, and compare it to standard CPT on the MMLU benchmark. Our analysis reveals that Reasoning CPT consistently improves performance across all evaluated domains. Notably, reasoning skills acquired in one domain transfer effectively to others; the performance gap with conventional methods widens as problem difficulty increases, with gains of up to 8 points on the most challenging problems. Furthermore, models trained with hidden thoughts learn to adjust the depth of their reasoning according to problem difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01797v3">LLM A*: Human in the Loop Large Language Models Enabled A* Search for Robotics</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 7 figures, 8 pages
    </div>
    <details class="paper-abstract">
      This research focuses on how Large Language Models (LLMs) can help with (path) planning for mobile embodied agents such as robots, in a human-in-the-loop and interactive manner. A novel framework named LLM A*, aims to leverage the commonsense of LLMs, and the utility-optimal A* is proposed to facilitate few-shot near-optimal path planning. Prompts are used for two main purposes: 1) to provide LLMs with essential information like environments, costs, heuristics, etc.; 2) to communicate human feedback on intermediate planning results to LLMs. This approach takes human feedback on board and renders the entire planning process transparent (akin to a `white box') to humans. Moreover, it facilitates code-free path planning, thereby fostering the accessibility and inclusiveness of artificial intelligence techniques to communities less proficient in coding. Comparative analysis against A* and RL demonstrates that LLM A* exhibits greater efficiency in terms of search space and achieves paths comparable to A* while outperforming RL. The interactive nature of LLM A* also makes it a promising tool for deployment in collaborative human-robot tasks. Codes and Supplemental Materials can be found at GitHub: https://github.com/speedhawk/LLM-A-.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04363v3">AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 Code for this work is avaliable at https://github.com/AIRI-Institute/AriGraph
    </div>
    <details class="paper-abstract">
      Advancements in the capabilities of Large Language Models (LLMs) have created a promising foundation for developing autonomous agents. With the right tools, these agents could learn to solve tasks in new environments by accumulating and updating their knowledge. Current LLM-based agents process past experiences using a full history of observations, summarization, retrieval augmentation. However, these unstructured memory representations do not facilitate the reasoning and planning essential for complex decision-making. In our study, we introduce AriGraph, a novel method wherein the agent constructs and updates a memory graph that integrates semantic and episodic memories while exploring the environment. We demonstrate that our Ariadne LLM agent, consisting of the proposed memory architecture augmented with planning and decision-making, effectively handles complex tasks within interactive text game environments difficult even for human players. Results show that our approach markedly outperforms other established memory methods and strong RL baselines in a range of problems of varying complexity. Additionally, AriGraph demonstrates competitive performance compared to dedicated knowledge graph-based methods in static multi-hop question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10143v1">GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 5 pages, 4 figures, accepted to IJCAI2025 demo track
    </div>
    <details class="paper-abstract">
      Large Language Models are now key assistants in human decision-making processes. However, a common note always seems to follow: "LLMs can make mistakes. Be careful with important info." This points to the reality that not all outputs from LLMs are dependable, and users must evaluate them manually. The challenge deepens as hallucinated responses, often presented with seemingly plausible explanations, create complications and raise trust issues among users. To tackle such issue, this paper proposes GE-Chat, a knowledge Graph enhanced retrieval-augmented generation framework to provide Evidence-based response generation. Specifically, when the user uploads a material document, a knowledge graph will be created, which helps construct a retrieval-augmented agent, enhancing the agent's responses with additional knowledge beyond its training corpus. Then we leverage Chain-of-Thought (CoT) logic generation, n-hop sub-graph searching, and entailment-based sentence generation to realize accurate evidence retrieval. We demonstrate that our method improves the existing models' performance in terms of identifying the exact evidence in a free-form context, providing a reliable way to examine the resources of LLM's conclusion and help with the judgment of the trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12636v3">MORepair: Teaching LLMs to Repair Code via Multi-Objective Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Within the realm of software engineering, specialized tasks on code, such as program repair, present unique challenges, necessitating fine-tuning Large language models~(LLMs) to unlock state-of-the-art performance. Fine-tuning approaches proposed in the literature for LLMs on program repair tasks generally overlook the need to reason about the logic behind code changes, beyond syntactic patterns in the data. High-performing fine-tuning experiments also usually come at very high computational costs. With MORepair, we propose a novel perspective on the learning focus of LLM fine-tuning for program repair: we not only adapt the LLM parameters to the syntactic nuances of the task of code transformation (objective 1), but we also specifically fine-tune the LLM with respect to the logical reason behind the code change in the training data (objective 2). Such a multi-objective fine-tuning will instruct LLMs to generate high-quality patches. We apply MORepair to fine-tune four open-source LLMs with different sizes and architectures. Experimental results on function-level and repository-level repair benchmarks show that the implemented fine-tuning effectively boosts LLM repair performance by 11.4% to 56.0%. We further show that our fine-tuning strategy yields superior performance compared to the state-of-the-art approaches, including standard fine-tuning, Fine-tune-CoT, and RepairLLaMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10066v1">Dark LLMs: The Growing Threat of Unaligned AI Models</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09427v2">SafePath: Conformal Prediction for Safe LLM-Based Autonomous Navigation</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show growing promise in autonomous driving by reasoning over complex traffic scenarios to generate path plans. However, their tendencies toward overconfidence, and hallucinations raise critical safety concerns. We introduce SafePath, a modular framework that augments LLM-based path planning with formal safety guarantees using conformal prediction. SafePath operates in three stages. In the first stage, we use an LLM that generates a set of diverse candidate paths, exploring possible trajectories based on agent behaviors and environmental cues. In the second stage, SafePath filters out high-risk trajectories while guaranteeing that at least one safe option is included with a user-defined probability, through a multiple-choice question-answering formulation that integrates conformal prediction. In the final stage, our approach selects the path with the lowest expected collision risk when uncertainty is low or delegates control to a human when uncertainty is high. We theoretically prove that SafePath guarantees a safe trajectory with a user-defined probability, and we show how its human delegation rate can be tuned to balance autonomy and safety. Extensive experiments on nuScenes and Highway-env show that SafePath reduces planning uncertainty by 77\% and collision rates by up to 70\%, demonstrating effectiveness in making LLM-driven path planning more safer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19159v3">A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Compared to width-wise pruning, depth-wise pruning can significantly accelerate inference in resource-constrained scenarios. However, treating the entire Transformer layer as the minimum pruning unit may degrade model performance by indiscriminately discarding the entire information of the layer. This paper reveals the ``Patch-like'' feature relationship between layers in large language models by analyzing the correlation of the outputs of different layers in the reproducing kernel Hilbert space. Building on this observation, we propose a sliding layer merging method that dynamically selects and fuses consecutive layers from top to bottom according to a pre-defined similarity threshold, thereby simplifying the model structure while maintaining its performance. Extensive experiments on LLMs with various architectures and different parameter scales show that our method outperforms existing pruning techniques in both zero-shot inference performance and retraining recovery quality after pruning. In particular, in the experiment with 35% pruning on the Vicuna-7B model, our method achieved a 1.654% improvement in average performance on zero-shot tasks compared to the existing method. Moreover, we further reveal the potential of combining depth pruning with width pruning to enhance the pruning effect. Our codes are available at https://github.com/920927/SLM-a-sliding-layer-merging-method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10013v1">DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have risen in prominence over the past few years, there has been concern over the potential biases in LLMs inherited from the training data. Previous studies have examined how LLMs exhibit implicit bias, such as when response generation changes when different social contexts are introduced. We argue that this implicit bias is not only an ethical, but also a technical issue, as it reveals an inability of LLMs to accommodate extraneous information. However, unlike other measures of LLM intelligence, there are no standard methods to benchmark this specific subset of LLM bias. To bridge this gap, we developed a method for calculating an easily interpretable benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM logic and math problem datasets with sociodemographic personas. We demonstrate that this method can statistically validate the presence of implicit bias in LLM behavior and find an inverse trend between question answering accuracy and implicit bias, supporting our argument.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10008v1">SVA-ICL: Improving LLM-based Software Vulnerability Assessment via In-Context Learning and Information Fusion</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Context: Software vulnerability assessment (SVA) is critical for identifying, evaluating, and prioritizing security weaknesses in software applications. Objective: Despite the increasing application of large language models (LLMs) in various software engineering tasks, their effectiveness in SVA remains underexplored. Method: To address this gap, we introduce a novel approach SVA-ICL, which leverages in-context learning (ICL) to enhance LLM performance. Our approach involves the selection of high-quality demonstrations for ICL through information fusion, incorporating both source code and vulnerability descriptions. For source code, we consider semantic, lexical, and syntactic similarities, while for vulnerability descriptions, we focus on textual similarity. Based on the selected demonstrations, we construct context prompts and consider DeepSeek-V2 as the LLM for SVA-ICL. Results: We evaluate the effectiveness of SVA-ICL using a large-scale dataset comprising 12,071 C/C++ vulnerabilities. Experimental results demonstrate that SVA-ICL outperforms state-of-the-art SVA baselines in terms of Accuracy, F1-score, and MCC measures. Furthermore, ablation studies highlight the significance of component customization in SVA-ICL, such as the number of demonstrations, the demonstration ordering strategy, and the optimal fusion ratio of different modalities. Conclusion: Our findings suggest that leveraging ICL with information fusion can effectively improve the effectiveness of LLM-based SVA, warranting further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09974v1">Analysing Safety Risks in LLMs Fine-Tuned with Pseudo-Malicious Cyber Security Data</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. We present a systematic evaluation of safety risks in fine-tuned LLMs for cyber security applications. Using the OWASP Top 10 for LLM Applications framework, we assessed seven open-source LLMs: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. Our evaluation shows that fine-tuning reduces safety resilience across all tested LLMs (e.g., the safety score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). We propose and evaluate a safety alignment approach that carefully rewords instruction-response pairs to include explicit safety precautions and ethical considerations. This approach demonstrates that it is possible to maintain or even improve model safety while preserving technical utility, offering a practical path forward for developing safer fine-tuning methodologies. This work offers a systematic evaluation for safety risks in LLMs, enabling safer adoption of generative AI in sensitive domains, and contributing towards the development of secure, trustworthy, and ethically aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09970v1">Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      The ReAct (Reasoning + Action) capability in large language models (LLMs) has become the foundation of modern agentic systems. Recent LLMs, such as DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through the generation of ample intermediate tokens, which help build a strong premise before producing the final output tokens. In this paper, we introduce Pre-Act, a novel approach that enhances the agent's performance by creating a multi-step execution plan along with the detailed reasoning for the given user input. This plan incrementally incorporates previous steps and tool outputs, refining itself after each step execution until the final response is obtained. Our approach is applicable to both conversational and non-conversational agents. To measure the performance of task-oriented agents comprehensively, we propose a two-level evaluation framework: (1) turn level and (2) end-to-end. Our turn-level evaluation, averaged across five models, shows that our approach, Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While this approach is effective for larger models, smaller models crucial for practical applications, where latency and cost are key constraints, often struggle with complex reasoning tasks required for agentic systems. To address this limitation, we fine-tune relatively small models such as Llama 3.1 (8B & 70B) using the proposed Pre-Act approach. Our experiments show that the fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action accuracy (turn-level) and a 28% improvement in goal completion rate (end-to-end) on the Almita (out-of-domain) dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07440v2">Model Utility Law: Evaluating LLMs beyond Performance through Mechanism Interpretable Metric</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. One core challenge of evaluation in the large language model (LLM) era is the generalization issue: how to infer a model's near-unbounded abilities from inevitably bounded benchmarks. We address this challenge by proposing Model Utilization Index (MUI), a mechanism interpretability enhanced metric that complements traditional performance scores. MUI quantifies the effort a model expends on a task, defined as the proportion of activated neurons or features during inference. Intuitively, a truly capable model should achieve higher performance with lower effort. Extensive experiments across popular LLMs reveal a consistent inverse logarithmic relationship between MUI and performance, which we formulate as the Utility Law. From this law we derive four practical corollaries that (i) guide training diagnostics, (ii) expose data contamination issue, (iii) enable fairer model comparisons, and (iv) design model-specific dataset diversity. Our code can be found at https://github.com/ALEX-nlp/MUI-Eva.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09921v1">PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at \href{https://github.com/redwyd/PrivacyJailbreak}{https://github.com/redwyd/PrivacyJailbreak}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01698v3">Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models</a></div>
    <div class="paper-meta">
      📅 2025-05-15
      | 💬 19 Pages, https://github.com/abhibambhaniya/GenZ-LLM-Analyzer, https://genz-llm-analyzer.streamlit.app/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance across a wide range of applications, often outperforming human experts. However, deploying these gigantic models efficiently for diverse inference use cases requires carefully designed hardware platforms with ample computing, memory, and network resources. With constant innovation in LLM serving optimizations and model architecture evolving at breakneck speed, the hardware requirements to meet Service Level Objectives (SLOs) remain an open research question. To answer the question, we present an analytical tool, GenZ, to efficiently navigate the relationship between diverse LLM model architectures(Dense, GQA, MoE, Mamba), LLM serving optimizations(Chunking, Speculative decoding, quanitization), and AI platform design parameters. Our tool estimates LLM inference performance metrics for the given scenario. We have validated against real hardware platforms running various different LLM models, achieving a max geomean error of 5.82.We use GenZ to identify compute, memory capacity, memory bandwidth, network latency, and network bandwidth requirements across diverse LLM inference use cases. We also study diverse architectural choices in use today (inspired by LLM serving platforms from several vendors) to help inform computer architects designing next-generation AI hardware accelerators and platforms. The trends and insights derived from GenZ can guide AI engineers deploying LLMs as well as computer architects designing next-generation hardware accelerators and platforms. Ultimately, this work sheds light on the platform design considerations for unlocking the full potential of large language models across a spectrum of applications. The source code is available at https://github.com/abhibambhaniya/GenZ-LLM-Analyzer . Users can also be tried it on at https://genz-llm-analyzer.streamlit.app/ without any setup on your web browser.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09901v1">Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making tasks. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) tasks introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how explicit reasoning, through both prompting strategies and reasoning-enhanced models, shapes LLM decision-making. We find that reasoning shifts LLMs toward more human-like behavior, characterized by a mix of random and directed exploration. In simple stationary tasks, reasoning-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas of improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.06018v4">TARGET: Automated Scenario Generation from Traffic Rules for Testing Autonomous Vehicles via Validated LLM-Guided Knowledge Extraction</a></div>
    <div class="paper-meta">
      📅 2025-05-15
    </div>
    <details class="paper-abstract">
      Recent incidents with autonomous vehicles highlight the need for rigorous testing to ensure safety and robustness. Constructing test scenarios for autonomous driving systems (ADSs), however, is labor-intensive. We propose TARGET, an end-to-end framework that automatically generates test scenarios from traffic rules. To address complexity, we leverage a Large Language Model (LLM) to extract knowledge from traffic rules. To mitigate hallucinations caused by large context during input processing, we introduce a domain-specific language (DSL) designed to be syntactically simple and compositional. This design allows the LLM to learn and generate test scenarios in a modular manner while enabling syntactic and semantic validation for each component. Based on these validated representations, TARGET synthesizes executable scripts to render scenarios in simulation. Evaluated seven ADSs with 284 scenarios derived from 54 traffic rules, TARGET uncovered 610 rule violations, collisions, and other issues. For each violation, TARGET generates scenario recordings and detailed logs, aiding root cause analysis. Two identified issues were confirmed by ADS developers: one linked to an existing bug report and the other to limited ADS functionality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09602v1">Adversarial Suffix Filtering: a Defense Pipeline for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in autonomous systems and public-facing environments, yet they remain susceptible to jailbreak vulnerabilities that may undermine their security and trustworthiness. Adversarial suffixes are considered to be the current state-of-the-art jailbreak, consistently outperforming simpler methods and frequently succeeding even in black-box settings. Existing defenses rely on access to the internal architecture of models limiting diverse deployment, increase memory and computation footprints dramatically, or can be bypassed with simple prompt engineering methods. We introduce $\textbf{Adversarial Suffix Filtering}$ (ASF), a lightweight novel model-agnostic defensive pipeline designed to protect LLMs against adversarial suffix attacks. ASF functions as an input preprocessor and sanitizer that detects and filters adversarially crafted suffixes in prompts, effectively neutralizing malicious injections. We demonstrate that ASF provides comprehensive defense capabilities across both black-box and white-box attack settings, reducing the attack efficacy of state-of-the-art adversarial suffix generation methods to below 4%, while only minimally affecting the target model's capabilities in non-adversarial scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v1">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) spread across industries, understanding their environmental footprint at the inference level is no longer optional; it is essential. However, most existing studies exclude proprietary models, overlook infrastructural variability and overhead, or focus solely on training, even as inference increasingly dominates AI's environmental impact. To bridge this gap, this paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: although individual queries are efficient, their global scale drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09439v1">Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU benchmark. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09427v1">SafePath: Conformal Prediction for Safe LLM-Based Autonomous Navigation</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show growing promise in autonomous driving by reasoning over complex traffic scenarios to generate path plans. However, their tendencies toward overconfidence, and hallucinations raise critical safety concerns. We introduce SafePath, a modular framework that augments LLM-based path planning with formal safety guarantees using conformal prediction. SafePath operates in three stages. In the first stage, we use an LLM that generates a set of diverse candidate paths, exploring possible trajectories based on agent behaviors and environmental cues. In the second stage, SafePath filters out high-risk trajectories while guaranteeing that at least one safe option is included with a user-defined probability, through a multiple-choice question-answering formulation that integrates conformal prediction. In the final stage, our approach selects the path with the lowest expected collision risk when uncertainty is low or delegates control to a human when uncertainty is high. We theoretically prove that SafePath guarantees a safe trajectory with a user-defined probability, and we show how its human delegation rate can be tuned to balance autonomy and safety. Extensive experiments on nuScenes and Highway-env show that SafePath reduces planning uncertainty by 77\% and collision rates by up to 70\%, demonstrating effectiveness in making LLM-driven path planning more safer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09396v1">The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) has shifted artificial intelligence (AI) research toward agentic systems, motivating the use of weaker and more flexible notions of agency. However, this shift raises key questions about the extent to which LLM-based agents replicate human strategic reasoning, particularly in game-theoretic settings. In this context, we examine the role of agentic sophistication in shaping artificial reasoners' performance by evaluating three agent designs: a simple game-theoretic model, an unstructured LLM-as-agent model, and an LLM integrated into a traditional agentic framework. Using guessing games as a testbed, we benchmarked these agents against human participants across general reasoning patterns and individual role-based objectives. Furthermore, we introduced obfuscated game scenarios to assess agents' ability to generalise beyond training distributions. Our analysis, covering over 2000 reasoning samples across 25 agent configurations, shows that human-inspired cognitive structures can enhance LLM agents' alignment with human strategic behaviour. Still, the relationship between agentic design complexity and human-likeness is non-linear, highlighting a critical dependence on underlying LLM capabilities and suggesting limits to simple architectural augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03343v2">What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train probes to classify successful from unsuccessful jailbreaks using the latent representations corresponding to prompt tokens. Notably, we find that even when probes achieve high accuracy in predicting the success of jailbreaks, their performance often fails to generalize to unseen attack methods. This reveals that different jailbreaking strategies exploit different non-linear, non-universal features. Next, we demonstrate that non-linear probes provide a powerful tool for steering model behavior. Specifically, we use these probes to guide targeted latent space perturbations, enabling us to effectively modulate the model's robustness against jailbreaks. Overall, our findings challenge the assumption that jailbreaks can be fully understood through linear or simple universal prompt features alone, highlighting the importance of a nuanced understanding of the mechanisms behind LLM vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09319v1">Statistical Modeling and Uncertainty Estimation of LLM Inference Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference systems present significant challenges in statistical performance characterization due to dynamic workload variations, diverse hardware architectures, and complex interactions between model size, batch processing, and throughput requirements. Accurate statistical characterization enables better workload scheduling, adaptive resource provisioning, and cost-aware inference optimization, making it crucial for improving efficiency in large-scale AI deployments. Traditional analytical models provide explainability but cannot cover the vast diversity of real-world workloads, making it impossible to benchmark every scenario in advance. Machine learning (ML) approaches effectively predict performance for non-benchmarked cases but struggle when extrapolating beyond their observed training space. To address these limitations for LLM inference systems, we propose an Analytical with Learning Augmentation (ALA) framework that bridges analytical modeling with \ml for robust statistical prediction and uncertainty estimation in LLM inference workloads. Our method employs an analytical throughput model with parameters estimated for benchmarked workloads, then extends to unobserved configurations using \ml predictions. We enhance this with simulated annealing to exploit subsets of the workload data point combinations and develop an error predictor. Finally, we quantify uncertainty based on vector space similarity between new and observed workloads to ensure robust generalization. Through extensive experimentation on diverse LLM inference workloads, we demonstrate that our framework achieves low median errors while maintaining adaptability to new inference scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09289v1">Reproducibility Study of "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents"</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 11 Tables, 9 Figures
    </div>
    <details class="paper-abstract">
      This study evaluates and extends the findings made by Piatti et al., who introduced GovSim, a simulation framework designed to assess the cooperative decision-making capabilities of large language models (LLMs) in resource-sharing scenarios. By replicating key experiments, we validate claims regarding the performance of large models, such as GPT-4-turbo, compared to smaller models. The impact of the universalization principle is also examined, with results showing that large models can achieve sustainable cooperation, with or without the principle, while smaller models fail without it. In addition, we provide multiple extensions to explore the applicability of the framework to new settings. We evaluate additional models, such as DeepSeek-V3 and GPT-4o-mini, to test whether cooperative behavior generalizes across different architectures and model sizes. Furthermore, we introduce new settings: we create a heterogeneous multi-agent environment, study a scenario using Japanese instructions, and explore an "inverse environment" where agents must cooperate to mitigate harmful resource distributions. Our results confirm that the benchmark can be applied to new models, scenarios, and languages, offering valuable insights into the adaptability of LLMs in complex cooperative tasks. Moreover, the experiment involving heterogeneous multi-agent systems demonstrates that high-performing models can influence lower-performing ones to adopt similar behaviors. This finding has significant implications for other agent-based applications, potentially enabling more efficient use of computational resources and contributing to the development of more effective cooperative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16472v2">Harden and Catch for Just-in-Time Assured LLM-Based Software Testing: Open Research Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 To Appear as keynote paper at FSE 2025
    </div>
    <details class="paper-abstract">
      Despite decades of research and practice in automated software testing, several fundamental concepts remain ill-defined and under-explored, yet offer enormous potential real-world impact. We show that these concepts raise exciting new challenges in the context of Large Language Models for software test generation. More specifically, we formally define and investigate the properties of hardening and catching tests. A hardening test is one that seeks to protect against future regressions, while a catching test is one that catches such a regression or a fault in new functionality introduced by a code change. Hardening tests can be generated at any time and may become catching tests when a future regression is caught. We also define and motivate the Catching 'Just-in-Time' (JiTTest) Challenge, in which tests are generated 'just-in-time' to catch new faults before they land into production. We show that any solution to Catching JiTTest generation can also be repurposed to catch latent faults in legacy code. We enumerate possible outcomes for hardening and catching tests and JiTTests, and discuss open research problems, deployment options, and initial results from our work on automated LLM-based hardening at Meta. This paper was written to accompany the keynote by the authors at the ACM International Conference on the Foundations of Software Engineering (FSE) 2025. Author order is alphabetical. The corresponding author is Mark Harman.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01383v3">LLM-based NLG Evaluation: Current Status and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Evaluating natural language generation (NLG) is a vital but challenging problem in natural language processing. Traditional evaluation metrics mainly capturing content (e.g. n-gram) overlap between system outputs and references are far from satisfactory, and large language models (LLMs) such as ChatGPT have demonstrated great potential in NLG evaluation in recent years. Various automatic evaluation methods based on LLMs have been proposed, including metrics derived from LLMs, prompting LLMs, fine-tuning LLMs, and human-LLM collaborative evaluation. In this survey, we first give a taxonomy of LLM-based NLG evaluation methods, and discuss their pros and cons, respectively. Lastly, we discuss several open problems in this area and point out future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09142v1">ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 13 pages, 5 figures. Cloud-native LLM scheduling system with latency-aware inference optimization
    </div>
    <details class="paper-abstract">
      We propose ELIS, a serving system for Large Language Models (LLMs) featuring an Iterative Shortest Remaining Time First (ISRTF) scheduler designed to efficiently manage inference tasks with the shortest remaining tokens. Current LLM serving systems often employ a first-come-first-served scheduling strategy, which can lead to the "head-of-line blocking" problem. To overcome this limitation, it is necessary to predict LLM inference times and apply a shortest job first scheduling strategy. However, due to the auto-regressive nature of LLMs, predicting the inference latency is challenging. ELIS addresses this challenge by training a response length predictor for LLMs using the BGE model, an encoder-based state-of-the-art model. Additionally, we have devised the ISRTF scheduling strategy, an optimization of shortest remaining time first tailored to existing LLM iteration batching. To evaluate our work in an industrial setting, we simulate streams of requests based on our study of real-world user LLM serving trace records. Furthermore, we implemented ELIS as a cloud-native scheduler system on Kubernetes to evaluate its performance in production environments. Our experimental results demonstrate that ISRTF reduces the average job completion time by up to 19.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18599v2">Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 16 pages, 14 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      Modern Large Language Model serving system batches multiple requests to achieve high throughput, while batching attention operations is challenging, rendering memory bandwidth a critical bottleneck. The community relies on high-end GPUs with multiple high-bandwidth memory channels. Unfortunately, HBM's high bandwidth often comes at the expense of limited memory capacity, which reduces core utilization and increases costs. Recent advancements enabling longer contexts for LLMs have substantially increased the key-value cache size, further intensifying the pressures on memory capacity. The literature has explored KV cache quantization techniques, which commonly use low bitwidth for most values, selectively using higher bitwidth for outlier values. While this approach helps achieve high accuracy and low bitwidth simultaneously, it comes with the limitation that cost for online outlier detection is excessively high, negating the advantages. We propose Oaken, an acceleration solution that achieves high accuracy and high performance simultaneously through co-designing algorithm and hardware. To effectively find a sweet spot in the accuracy-performance trade-off space of KV cache quantization, Oaken employs an online-offline hybrid approach, setting outlier thresholds offline, which are then used to determine the quantization scale online. To translate the proposed algorithmic technique into tangible performance gains, Oaken also comes with custom quantization engines and memory management units that can be integrated with any LLM accelerators. We built an Oaken accelerator on top of an LLM accelerator, LPU, and conducted a comprehensive evaluation. Our experiments show that for a batch size of 256, Oaken achieves up to 1.58x throughput improvement over NVIDIA A100 GPU, incurring a minimal accuracy loss of only 0.54\% on average, compared to state-of-the-art KV cache quantization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09082v1">CEC-Zero: Chinese Error Correction Solution Based on LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) demonstrate exceptional Chinese text processing capabilities, particularly in Chinese Spelling Correction (CSC). While LLMs outperform traditional BERT-based models in accuracy and robustness, challenges persist in reliability and generalization. This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework enabling LLMs to self-correct through autonomous error strategy learning without external supervision. By integrating RL with LLMs' generative power, the method eliminates dependency on annotated data or auxiliary models. Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and superior cross-domain generalization, offering a scalable solution for reliability optimization in Chinese NLP applications. This breakthrough facilitates LLM deployment in practical Chinese text correction scenarios while establishing a new paradigm for self-improving language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09116v2">P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) showcase varied multilingual capabilities across tasks like translation, code generation, and reasoning. Previous assessments often limited their scope to fundamental natural language processing (NLP) or isolated capability-specific tasks. To alleviate this drawback, we aim to present a comprehensive multilingual multitask benchmark. First, we introduce P-MMEval, a large-scale benchmark covering effective fundamental and capability-specialized datasets. Furthermore, P-MMEval delivers consistent language coverage across various datasets and provides parallel samples. Finally, we conduct extensive experiments on representative multilingual model series to compare performances across models and tasks, explore the relationship between multilingual performances and factors such as tasks, model sizes, languages, and prompts, and examine the effectiveness of knowledge transfer from English to other languages. The resulting insights are intended to offer valuable guidance for future research. The dataset is available at https://huggingface.co/datasets/Qwen/P-MMEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09852v1">Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance across natural language tasks, but their ability to forecast violent conflict remains underexplored. We investigate whether LLMs possess meaningful parametric knowledge-encoded in their pretrained weights-to predict conflict escalation and fatalities without external data. This is critical for early warning systems, humanitarian planning, and policy-making. We compare this parametric knowledge with non-parametric capabilities, where LLMs access structured and unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent news reports via Retrieval-Augmented Generation (RAG). Incorporating external information could enhance model performance by providing up-to-date context otherwise missing from pretrained weights. Our two-part evaluation framework spans 2020-2024 across conflict-prone regions in the Horn of Africa and the Middle East. In the parametric setting, LLMs predict conflict trends and fatalities relying only on pretrained knowledge. In the non-parametric setting, models receive summaries of recent conflict events, indicators, and geopolitical developments. We compare predicted conflict trend labels (e.g., Escalate, Stable Conflict, De-escalate, Peace) and fatalities against historical data. Our findings highlight the strengths and limitations of LLMs for conflict forecasting and the benefits of augmenting them with structured external knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09810v1">Lossless Compression for LLM Tensor Incremental Snapshots</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      During the training of Large Language Models (LLMs), tensor data is periodically "checkpointed" to persistent storage to allow recovery of work done in the event of failure. The volume of data that must be copied during each checkpoint, even when using reduced-precision representations such as bfloat16, often reaches hundreds of gigabytes. Furthermore, the data must be moved across a network and written to a storage system before the next epoch occurs. With a view to ultimately building an optimized checkpointing solution, this paper presents experimental analysis of checkpoint data used to derive a design that maximizes the use of lossless compression to reduce the volume of data. We examine how tensor data and its compressibility evolve during model training and evaluate the efficacy of existing common off-the-shelf general purpose compression engines combined with known data optimization techniques such as byte-grouping and incremental delta compression. Leveraging our analysis we have built an effective compression solution, known as Language Model Compressor (LMC), which is based on byte-grouping and Huffman encoding. LMC offers more compression performance than the best alternative (BZ2) but with an order-of-magnitude reduction in the time needed to perform the compression. We show that a 16-core parallel implementation of LMC can attain compression and decompression throughput of 2.78 GiB/s and 3.76 GiB/s respectively. This increase in performance ultimately reduces the CPU resources needed and provides more time to copy the data to the storage system before the next epoch thus allowing for higher-frequency checkpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09807v1">Exploring the generalization of LLM truth directions on conversational formats</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Several recent works argue that LLMs have a universal truth direction where true and false statements are linearly separable in the activation space of the model. It has been demonstrated that linear probes trained on a single hidden state of the model already generalize across a range of topics and might even be used for lie detection in LLM conversations. In this work we explore how this truth direction generalizes between various conversational formats. We find good generalization between short conversations that end on a lie, but poor generalization to longer formats where the lie appears earlier in the input prompt. We propose a solution that significantly improves this type of generalization by adding a fixed key phrase at the end of each conversation. Our results highlight the challenges towards reliable LLM lie detectors that generalize to new settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09806v1">Learn, Explore and Reflect by Chatting: Understanding the Value of an LLM-Based Voting Advice Application Chatbot</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 Accepted to ACM CUI 2025
    </div>
    <details class="paper-abstract">
      Voting advice applications (VAAs), which have become increasingly prominent in European elections, are seen as a successful tool for boosting electorates' political knowledge and engagement. However, VAAs' complex language and rigid presentation constrain their utility to less-sophisticated voters. While previous work enhanced VAAs' click-based interaction with scripted explanations, a conversational chatbot's potential for tailored discussion and deliberate political decision-making remains untapped. Our exploratory mixed-method study investigates how LLM-based chatbots can support voting preparation. We deployed a VAA chatbot to 331 users before Germany's 2024 European Parliament election, gathering insights from surveys, conversation logs, and 10 follow-up interviews. Participants found the VAA chatbot intuitive and informative, citing its simple language and flexible interaction. We further uncovered VAA chatbots' role as a catalyst for reflection and rationalization. Expanding on participants' desire for transparency, we provide design recommendations for building interactive and trustworthy VAA chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09724v1">An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 31 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09665v1">Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08783v1">CodePDE: An Inference Framework for LLM-driven PDE Solver Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). Leveraging advanced inference-time algorithms and scaling strategies, CodePDE unlocks critical capacities of LLM for PDE solving: reasoning, debugging, selfrefinement, and test-time scaling -- all without task-specific tuning. CodePDE achieves superhuman performance across a range of representative PDE problems. We also present a systematic empirical analysis of LLM generated solvers, analyzing their accuracy, efficiency, and numerical scheme choices. Our findings highlight the promise and the current limitations of LLMs in PDE solving, offering a new perspective on solver design and opportunities for future model development. Our code is available at https://github.com/LithiumDA/CodePDE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02870v2">AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Accepted by CVPR 2025 Workshop
    </div>
    <details class="paper-abstract">
      Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v3">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00651v2">Open-Source LLM-Driven Federated Transformer for Predictive IoV Management</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Preprint version; submitted for academic peer review
    </div>
    <details class="paper-abstract">
      The proliferation of connected vehicles within the Internet of Vehicles (IoV) ecosystem presents critical challenges in ensuring scalable, real-time, and privacy-preserving traffic management. Existing centralized IoV solutions often suffer from high latency, limited scalability, and reliance on proprietary Artificial Intelligence (AI) models, creating significant barriers to widespread deployment, particularly in dynamic and privacy-sensitive environments. Meanwhile, integrating Large Language Models (LLMs) in vehicular systems remains underexplored, especially concerning prompt optimization and effective utilization in federated contexts. To address these challenges, we propose the Federated Prompt-Optimized Traffic Transformer (FPoTT), a novel framework that leverages open-source LLMs for predictive IoV management. FPoTT introduces a dynamic prompt optimization mechanism that iteratively refines textual prompts to enhance trajectory prediction. The architecture employs a dual-layer federated learning paradigm, combining lightweight edge models for real-time inference with cloud-based LLMs to retain global intelligence. A Transformer-driven synthetic data generator is incorporated to augment training with diverse, high-fidelity traffic scenarios in the Next Generation Simulation (NGSIM) format. Extensive evaluations demonstrate that FPoTT, utilizing EleutherAI Pythia-1B, achieves 99.86% prediction accuracy on real-world data while maintaining high performance on synthetic datasets. These results underscore the potential of open-source LLMs in enabling secure, adaptive, and scalable IoV management, offering a promising alternative to proprietary solutions in smart mobility ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08704v1">LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 IEEE 26th International Conference on Information Reuse and Integration for Data Science (IRI 2025), San Jose, CA, USA
    </div>
    <details class="paper-abstract">
      Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08662v1">Revealing economic facts: LLMs know more than they say</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 34 pages, 17 figures
    </div>
    <details class="paper-abstract">
      We investigate whether the hidden states of large language models (LLMs) can be used to estimate and impute economic and financial statistics. Focusing on county-level (e.g. unemployment) and firm-level (e.g. total assets) variables, we show that a simple linear model trained on the hidden states of open-source LLMs outperforms the models' text outputs. This suggests that hidden states capture richer economic information than the responses of the LLMs reveal directly. A learning curve analysis indicates that only a few dozen labelled examples are sufficient for training. We also propose a transfer learning method that improves estimation accuracy without requiring any labelled data for the target variable. Finally, we demonstrate the practical utility of hidden-state representations in super-resolution and data imputation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08600v1">Automatic Task Detection and Heterogeneous LLM Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 10 pages, 10 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Speculative decoding, which combines a draft model with a target model, has emerged as an effective approach to accelerate large language model (LLM) inference. However, existing methods often face a trade-off between the acceptance rate and decoding speed in downstream tasks due to the limited capacity of the draft model, making it difficult to ensure efficiency across diverse tasks. To address this problem, we propose a speculative decoding algorithm tailored for downstream task optimization. It includes an automatic task partitioning and assigning method, which automatically categorizes downstream tasks into different sub-tasks and assigns them to a set of heterogeneous draft models. Each draft model is aligned with the target model using task-specific data, thereby enhancing the consistency of inference results. In addition, our proposed method incorporates an online lightweight prompt classifier to dynamically route prompts to the appropriate draft model. Experimental results demonstrate that the proposed method improves draft accuracy by 6% to 50% over vanilla speculative decoding, while achieving a speedup of 1.10x to 2.64x in LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08590v1">Enhancing Thyroid Cytology Diagnosis with RAG-Optimized LLMs and Pa-thology Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Advancements in artificial intelligence (AI) are transforming pathology by integrat-ing large language models (LLMs) with retrieval-augmented generation (RAG) and domain-specific foundation models. This study explores the application of RAG-enhanced LLMs coupled with pathology foundation models for thyroid cytology diagnosis, addressing challenges in cytological interpretation, standardization, and diagnostic accuracy. By leveraging a curated knowledge base, RAG facilitates dy-namic retrieval of relevant case studies, diagnostic criteria, and expert interpreta-tion, improving the contextual understanding of LLMs. Meanwhile, pathology foun-dation models, trained on high-resolution pathology images, refine feature extrac-tion and classification capabilities. The fusion of these AI-driven approaches en-hances diagnostic consistency, reduces variability, and supports pathologists in dis-tinguishing benign from malignant thyroid lesions. Our results demonstrate that integrating RAG with pathology-specific LLMs significantly improves diagnostic efficiency and interpretability, paving the way for AI-assisted thyroid cytopathology, with foundation model UNI achieving AUC 0.73-0.93 for correct prediction of surgi-cal pathology diagnosis from thyroid cytology samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08542v1">Guiding LLM-based Smart Contract Generation with Finite State Machine</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Smart contract is a kind of self-executing code based on blockchain technology with a wide range of application scenarios, but the traditional generation method relies on manual coding and expert auditing, which has a high threshold and low efficiency. Although Large Language Models (LLMs) show great potential in programming tasks, they still face challenges in smart contract generation w.r.t. effectiveness and security. To solve these problems, we propose FSM-SCG, a smart contract generation framework based on finite state machine (FSM) and LLMs, which significantly improves the quality of the generated code by abstracting user requirements to generate FSM, guiding LLMs to generate smart contracts, and iteratively optimizing the code with the feedback of compilation and security checks. The experimental results show that FSM-SCG significantly improves the quality of smart contract generation. Compared to the best baseline, FSM-SCG improves the compilation success rate of generated smart contract code by at most 48%, and reduces the average vulnerability risk score by approximately 68%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00782v2">TradExpert: Revolutionizing Trading with Mixture of Expert LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      The integration of Artificial Intelligence (AI) in the financial domain has opened new avenues for quantitative trading, particularly through the use of Large Language Models (LLMs). However, the challenge of effectively synthesizing insights from diverse data sources and integrating both structured and unstructured data persists. This paper presents TradeExpert, a novel framework that employs a mix of experts (MoE) approach, using four specialized LLMs, each analyzing distinct sources of financial data, including news articles, market data, alpha factors, and fundamental data. The insights of these expert LLMs are further synthesized by a General Expert LLM to make a final prediction or decision. With specific prompts, TradeExpert can be switched between the prediction mode and the ranking mode for stock movement prediction and quantitative stock trading, respectively. In addition to existing benchmarks, we also release a large-scale financial dataset to comprehensively evaluate TradeExpert's effectiveness. Our experimental results demonstrate TradeExpert's superior performance across all trading scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08492v1">Achieving Scalable Robot Autonomy via neurosymbolic planning using lightweight local LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 19 pages, 3 figures, 4 tables, accepted at IAS 2025
    </div>
    <details class="paper-abstract">
      PDDL-based symbolic task planning remains pivotal for robot autonomy yet struggles with dynamic human-robot collaboration due to scalability, re-planning demands, and delayed plan availability. Although a few neurosymbolic frameworks have previously leveraged LLMs such as GPT-3 to address these challenges, reliance on closed-source, remote models with limited context introduced critical constraints: third-party dependency, inconsistent response times, restricted plan length and complexity, and multi-domain scalability issues. We present Gideon, a novel framework that enables the transition to modern, smaller, local LLMs with extended context length. Gideon integrates a novel problem generator to systematically generate large-scale datasets of realistic domain-problem-plan tuples for any domain, and adapts neurosymbolic planning for local LLMs, enabling on-device execution and extended context for multi-domain support. Preliminary experiments in single-domain scenarios performed on Qwen-2.5 1.5B and trained on 8k-32k samples, demonstrate a valid plan percentage of 66.1% (32k model) and show that the figure can be further scaled through additional data. Multi-domain tests on 16k samples yield an even higher 70.6% planning validity rate, proving extensibility across domains and signaling that data variety can have a positive effect on learning efficiency. Although long-horizon planning and reduced model size make Gideon training much less efficient than baseline models based on larger LLMs, the results are still significant considering that the trained model is about 120x smaller than baseline and that significant advantages can be achieved in inference efficiency, scalability, and multi-domain adaptability, all critical factors in human-robot collaboration. Training inefficiency can be mitigated by Gideon's streamlined data generation pipeline.
    </details>
</div>
