# llm - 2025_05

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18882v2">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05185v3">PentestAgent: Incorporating LLM Agents to Automated Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 14 pages, 13 figures, AsiaCCS 2025
    </div>
    <details class="paper-abstract">
      Penetration testing is a critical technique for identifying security vulnerabilities, traditionally performed manually by skilled security specialists. This complex process involves gathering information about the target system, identifying entry points, exploiting the system, and reporting findings. Despite its effectiveness, manual penetration testing is time-consuming and expensive, often requiring significant expertise and resources that many organizations cannot afford. While automated penetration testing methods have been proposed, they often fall short in real-world applications due to limitations in flexibility, adaptability, and implementation. Recent advancements in large language models (LLMs) offer new opportunities for enhancing penetration testing through increased intelligence and automation. However, current LLM-based approaches still face significant challenges, including limited penetration testing knowledge and a lack of comprehensive automation capabilities. To address these gaps, we propose PentestAgent, a novel LLM-based automated penetration testing framework that leverages the power of LLMs and various LLM-based techniques like Retrieval Augmented Generation (RAG) to enhance penetration testing knowledge and automate various tasks. Our framework leverages multi-agent collaboration to automate intelligence gathering, vulnerability analysis, and exploitation stages, reducing manual intervention. We evaluate PentestAgent using a comprehensive benchmark, demonstrating superior performance in task completion and overall efficiency. This work significantly advances the practical applicability of automated penetration testing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24069v1">DSR-Bench: Evaluating the Structural Reasoning Abilities of LLMs via Data Structures</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed for real-world tasks that fundamentally involve data manipulation. A core requirement across these tasks is the ability to perform structural reasoning--that is, to understand and reason about data relationships. For example, customer requests require a temporal ordering, which can be represented by data structures such as queues. However, existing benchmarks primarily focus on high-level, application-driven evaluations without isolating this fundamental capability. To address this gap, we introduce DSR-Bench, a novel benchmark evaluating LLMs' structural reasoning capabilities through data structures, which provide interpretable representations of data relationships. DSR-Bench includes 20 data structures, 35 operations, and 4,140 problem instances, organized hierarchically for fine-grained analysis of reasoning limitations. Our evaluation pipeline is fully automated and deterministic, eliminating subjective human or model-based judgments. Its synthetic nature also ensures scalability and minimizes data contamination risks. We benchmark nine state-of-the-art LLMs. Our analysis shows that instruction-tuned models struggle with basic multi-attribute and multi-hop reasoning. Furthermore, while reasoning-oriented models perform better, they remain fragile on complex and hybrid structures, with the best model achieving an average score of only 47% on the challenge subset. Crucially, models often perform poorly on multi-dimensional data and natural language task descriptions, highlighting a critical gap for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14970v2">Self-Evolving Curriculum for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24037v1">Leave it to the Specialist: Repair Sparse LLMs with Sparse Fine-Tuning via Sparsity Evolution</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across various tasks but face deployment challenges due to their massive computational demands. While post-training pruning methods like SparseGPT and Wanda can effectively reduce the model size, but struggle to maintain model performance at high sparsity levels, limiting their utility for downstream tasks. Existing fine-tuning methods, such as full fine-tuning and LoRA, fail to preserve sparsity as they require updating the whole dense metrics, not well-suited for sparse LLMs. In this paper, we propose Sparsity Evolution Fine-Tuning (SEFT), a novel method designed specifically for sparse LLMs. SEFT dynamically evolves the sparse topology of pruned models during fine-tuning, while preserving the overall sparsity throughout the process. The strengths of SEFT lie in its ability to perform task-specific adaptation through a weight drop-and-grow strategy, enabling the pruned model to self-adapt its sparse connectivity pattern based on the target dataset. Furthermore, a sensitivity-driven pruning criterion is employed to ensure that the desired sparsity level is consistently maintained throughout fine-tuning. Our experiments on various LLMs, including LLaMA families, DeepSeek, and Mistral, across a diverse set of benchmarks demonstrate that SEFT achieves stronger performance while offering superior memory and time efficiency compared to existing baselines. Our code is publicly available at: https://github.com/QiaoXiao7282/SEFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12476v2">CoCo-CoLa: Evaluating and Improving Language Adherence in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 26 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) develop cross-lingual abilities despite being trained on limited parallel data. However, they often struggle to generate responses in the intended language, favoring high-resource languages such as English. In this work, we introduce CoCo-CoLa (Correct Concept - Correct Language), a novel metric to evaluate language adherence in multilingual LLMs. Using fine-tuning experiments on a closed-book QA task across seven languages, we analyze how training in one language affects others' performance. Our findings reveal that multilingual models share task knowledge across languages but exhibit biases in the selection of output language. We identify language-specific layers, showing that final layers play a crucial role in determining output language. Accordingly, we propose a partial training strategy that selectively fine-tunes key layers, improving language adherence while significantly reducing computational cost. Our method achieves comparable or superior performance to full fine-tuning, particularly for low-resource languages, offering a more efficient multilingual adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24036v1">GenIC: An LLM-Based Framework for Instance Completion in Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Knowledge graph completion aims to address the gaps of knowledge bases by adding new triples that represent facts. The complexity of this task depends on how many parts of a triple are already known. Instance completion involves predicting the relation-tail pair when only the head is given (h, ?, ?). Notably, modern knowledge bases often contain entity descriptions and types, which can provide valuable context for inferring missing facts. By leveraging these textual descriptions and the ability of large language models to extract facts from them and recognize patterns within the knowledge graph schema, we propose an LLM-powered, end-to-end instance completion approach. Specifically, we introduce GenIC: a two-step Generative Instance Completion framework. The first step focuses on property prediction, treated as a multi-label classification task. The second step is link prediction, framed as a generative sequence-to-sequence task. Experimental results on three datasets show that our method outperforms existing baselines. Our code is available at https://github.com/amal-gader/genic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24034v1">LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Trainin</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become the most effective post-training approach for improving the capabilities of Large Language Models (LLMs). In practice, because of the high demands on latency and memory, it is particularly challenging to develop an efficient RL framework that reliably manages policy models with hundreds to thousands of billions of parameters. In this paper, we present LlamaRL, a fully distributed, asynchronous RL framework optimized for efficient training of large-scale LLMs with various model sizes (8B, 70B, and 405B parameters) on GPU clusters ranging from a handful to thousands of devices. LlamaRL introduces a streamlined, single-controller architecture built entirely on native PyTorch, enabling modularity, ease of use, and seamless scalability to thousands of GPUs. We also provide a theoretical analysis of LlamaRL's efficiency, including a formal proof that its asynchronous design leads to strict RL speed-up. Empirically, by leveraging best practices such as colocated model offloading, asynchronous off-policy training, and distributed direct memory access for weight synchronization, LlamaRL achieves significant efficiency gains -- up to 10.7x speed-up compared to DeepSpeed-Chat-like systems on a 405B-parameter policy model. Furthermore, the efficiency advantage continues to grow with increasing model scale, demonstrating the framework's suitability for future large-scale RL training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18160v3">RepoAudit: An Autonomous LLM-Agent for Repository-Level Code Auditing</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 18 pages, 11 tables, 6 figures, 3 listings
    </div>
    <details class="paper-abstract">
      Code auditing is the process of reviewing code with the aim of identifying bugs. Large Language Models (LLMs) have demonstrated promising capabilities for this task without requiring compilation, while also supporting user-friendly customization. However, auditing a code repository with LLMs poses significant challenges: limited context windows and hallucinations can degrade the quality of bug reports, and analyzing large-scale repositories incurs substantial time and token costs, hindering efficiency and scalability. This work introduces an LLM-based agent, RepoAudit, designed to perform autonomous repository-level code auditing. Equipped with agent memory, RepoAudit explores the codebase on demand by analyzing data-flow facts along feasible program paths within individual functions. It further incorporates a validator module to mitigate hallucinations by verifying data-flow facts and checking the satisfiability of path conditions associated with potential bugs, thereby reducing false positives. RepoAudit detects 40 true bugs across 15 real-world benchmark projects with a precision of 78.43%, requiring on average only 0.44 hours and $2.54 per project. Also, it detects 185 new bugs in high-profile projects, among which 174 have been confirmed or fixed. We have open-sourced RepoAudit at https://github.com/PurCL/RepoAudit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24019v1">LLM Agents Should Employ Security Principles</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents show considerable promise for automating complex tasks using contextual reasoning; however, interactions involving multiple agents and the system's susceptibility to prompt injection and other forms of context manipulation introduce new vulnerabilities related to privacy leakage and system exploitation. This position paper argues that the well-established design principles in information security, which are commonly referred to as security principles, should be employed when deploying LLM agents at scale. Design principles such as defense-in-depth, least privilege, complete mediation, and psychological acceptability have helped guide the design of mechanisms for securing information systems over the last five decades, and we argue that their explicit and conscientious adoption will help secure agentic systems. To illustrate this approach, we introduce AgentSandbox, a conceptual framework embedding these security principles to provide safeguards throughout an agent's life-cycle. We evaluate with state-of-the-art LLMs along three dimensions: benign utility, attack utility, and attack success rate. AgentSandbox maintains high utility for its intended functions under both benign and adversarial evaluations while substantially mitigating privacy risks. By embedding secure design principles as foundational elements within emerging LLM agent protocols, we aim to promote trustworthy agent ecosystems aligned with user privacy expectations and evolving regulatory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11638v2">BanStereoSet: A Dataset to Measure Stereotypical Social Biases in LLMs for Bangla</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted at ACL-2025
    </div>
    <details class="paper-abstract">
      This study presents BanStereoSet, a dataset designed to evaluate stereotypical social biases in multilingual LLMs for the Bangla language. In an effort to extend the focus of bias research beyond English-centric datasets, we have localized the content from the StereoSet, IndiBias, and Kamruzzaman et. al.'s datasets, producing a resource tailored to capture biases prevalent within the Bangla-speaking community. Our BanStereoSet dataset consists of 1,194 sentences spanning 9 categories of bias: race, profession, gender, ageism, beauty, beauty in profession, region, caste, and religion. This dataset not only serves as a crucial tool for measuring bias in multilingual LLMs but also facilitates the exploration of stereotypical bias across different social categories, potentially guiding the development of more equitable language technologies in Bangladeshi contexts. Our analysis of several language models using this dataset indicates significant biases, reinforcing the necessity for culturally and linguistically adapted datasets to develop more equitable language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18407v2">A Statistical Framework for Ranking LLM-Based Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed natural language processing, with frameworks like Chatbot Arena providing pioneering platforms for evaluating these models. By facilitating millions of pairwise comparisons based on human judgments, Chatbot Arena has become a cornerstone in LLM evaluation, offering rich datasets for ranking models in open-ended conversational tasks. Building upon this foundation, we propose a statistical framework that incorporates key advancements to address specific challenges in pairwise comparison analysis. First, we introduce a factored tie model that enhances the ability to handle ties -- an integral aspect of human-judged comparisons -- significantly improving the model's fit to observed data. Second, we extend the framework to model covariance between competitors, enabling deeper insights into performance relationships and facilitating intuitive groupings into performance tiers. Third, we resolve optimization challenges arising from parameter non-uniqueness by introducing novel constraints, ensuring stable and interpretable parameter estimation. Through rigorous evaluation and extensive experimentation, our framework demonstrates substantial improvements over existing methods in modeling pairwise comparison data. To support reproducibility and practical adoption, we release leaderbot, an open-source Python package implementing our models and analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24000v1">ConversAR: Exploring Embodied LLM-Powered Group Conversations in Augmented Reality for Second Language Learners</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Published in Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Group conversations are valuable for second language (L2) learners as they provide opportunities to practice listening and speaking, exercise complex turn-taking skills, and experience group social dynamics in a target language. However, most existing Augmented Reality (AR)-based conversational learning tools focus on dyadic interactions rather than group dialogues. Although research has shown that AR can help reduce speaking anxiety and create a comfortable space for practicing speaking skills in dyadic scenarios, especially with Large Language Model (LLM)-based conversational agents, the potential for group language practice using these technologies remains largely unexplored. We introduce ConversAR, a gpt-4o powered AR application, that enables L2 learners to practice contextualized group conversations. Our system features two embodied LLM agents with vision-based scene understanding and live captions. In a system evaluation with 10 participants, users reported reduced speaking anxiety and increased learner autonomy compared to perceptions of in-person practice methods with other learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23996v1">Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 9 pages, 8 figures, and 1 table in main paper. Supplementary appendix attached. Accepted at ICML 2025
    </div>
    <details class="paper-abstract">
      The recent rapid adoption of large language models (LLMs) highlights the critical need for benchmarking their fairness. Conventional fairness metrics, which focus on discrete accuracy-based evaluations (i.e., prediction correctness), fail to capture the implicit impact of model uncertainty (e.g., higher model confidence about one group over another despite similar accuracy). To address this limitation, we propose an uncertainty-aware fairness metric, UCerF, to enable a fine-grained evaluation of model fairness that is more reflective of the internal bias in model decisions compared to conventional fairness measures. Furthermore, observing data size, diversity, and clarity issues in current datasets, we introduce a new gender-occupation fairness evaluation dataset with 31,756 samples for co-reference resolution, offering a more diverse and suitable dataset for evaluating modern LLMs. We establish a benchmark, using our metric and dataset, and apply it to evaluate the behavior of ten open-source LLMs. For example, Mistral-7B exhibits suboptimal fairness due to high confidence in incorrect predictions, a detail overlooked by Equalized Odds but captured by UCerF. Overall, our proposed LLM benchmark, which evaluates fairness with uncertainty awareness, paves the way for developing more transparent and accountable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23994v1">PolicyPulse: LLM-Synthesis Tool for Policy Researchers</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Published in Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Public opinion shapes policy, yet capturing it effectively to surface diverse perspectives remains challenging. This paper introduces PolicyPulse, an LLM-powered interactive system that synthesizes public experiences from online community discussions to help policy researchers author memos and briefs, leveraging curated real-world anecdotes. Given a specific topic (e.g., "Climate Change"), PolicyPulse returns an organized list of themes (e.g., "Biodiversity Loss" or "Carbon Pricing"), supporting each theme with relevant quotes from real-life anecdotes. We compared PolicyPulse outputs to authoritative policy reports. Additionally, we asked 11 policy researchers across multiple institutions in the Northeastern U.S to compare using PolicyPulse with their expert approach. We found that PolicyPulse's themes aligned with authoritative reports and helped spark research by analyzing existing data, gathering diverse experiences, revealing unexpected themes, and informing survey or interview design. Participants also highlighted limitations including insufficient demographic context and data verification challenges. Our work demonstrates how AI-powered tools can help influence policy-relevant research and shape policy outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23982v1">MSQA: Benchmarking LLMs on Graduate-Level Materials Science Reasoning and Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Despite recent advances in large language models (LLMs) for materials science, there is a lack of benchmarks for evaluating their domain-specific knowledge and complex reasoning abilities. To bridge this gap, we introduce MSQA, a comprehensive evaluation benchmark of 1,757 graduate-level materials science questions in two formats: detailed explanatory responses and binary True/False assessments. MSQA distinctively challenges LLMs by requiring both precise factual knowledge and multi-step reasoning across seven materials science sub-fields, such as structure-property relationships, synthesis processes, and computational modeling. Through experiments with 10 state-of-the-art LLMs, we identify significant gaps in current LLM performance. While API-based proprietary LLMs achieve up to 84.5% accuracy, open-source (OSS) LLMs peak around 60.5%, and domain-specific LLMs often underperform significantly due to overfitting and distributional shifts. MSQA represents the first benchmark to jointly evaluate the factual and reasoning capabilities of LLMs crucial for LLMs in advanced materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10099v3">Know the Unknown: An Uncertainty-Sensitive Method for LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable capabilities but face challenges from hallucinations, which typically arise from insufficient knowledge or context. While instructing LLMs to acknowledge knowledge limitations by responding with "I don't know" appears promising, we find that models consistently struggle with admitting knowledge gaps. This challenge may originate from current instruction datasets that emphasise answer generation over knowledge boundary awareness. To address this limitation, we introduce Uncertainty-and-Sensitivity-Aware Tuning (US-Tuning), a novel two-stage approach for contextual question answering (QA). The first stage enhances LLMs' ability to recognise their knowledge boundaries, while the second stage reinforces instruction adherence through carefully designed causal prompts. Our experimental results demonstrate that US-Tuning not only significantly reduces incorrect answers in contextual QA but also improves models' faithfulness to their parametric knowledge, mitigating hallucinations in general QA tasks. Our fine-tuned Llama2-7B model achieves up to a 34.7% improvement in handling out-of-knowledge questions and outperforms GPT-4 by 4.2% in overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23970v1">EmbAdvisor: Adaptive Cache Management for Sustainable LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become widely used, their environmental impact$\unicode{x2014}$especially carbon emissions$\unicode{x2014}$has attracted more attention. Prior studies focus on compute-related carbon emissions. In this paper, we find that storage is another key contributor. LLM caching, which saves and reuses KV caches for repeated context, reduces operational carbon by avoiding redundant computation. However, this benefit comes at the cost of embodied carbon from high-capacity, high-speed SSDs. As LLMs scale, the embodied carbon of storage grows significantly. To address this tradeoff, we present EmbAdvisor, a carbon-aware caching framework that selects the optimal cache size for LLM serving. EmbAdvisor profiles different LLM tasks and uses an Integer Linear Programming (ILP) solver to select cache sizes that meet SLOs while minimizing total carbon emissions. Overall, EmbAdvisor reduces the average carbon emissions of a Llama-3 70B model by 9.5% under various carbon intensities compared to a non-adaptive cache scenario, and can save up to 31.2% when the carbon intensity is low.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23953v1">Enhancing LLM-Based Code Generation with Complexity Metrics: A Feedback-Driven Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 11 pages, 5 figures. Accepted to COMPSAC 2025
    </div>
    <details class="paper-abstract">
      Automatic code generation has gained significant momentum with the advent of Large Language Models (LLMs) such as GPT-4. Although many studies focus on improving the effectiveness of LLMs for code generation, very limited work tries to understand the generated code's characteristics and leverage that to improve failed cases. In this paper, as the most straightforward characteristic of code, we investigate the relationship between code complexity and the success of LLM generated code. Using a large set of standard complexity metrics, we first conduct an empirical analysis to explore their correlation with LLM's performance on code generation (i.e., Pass@1). Using logistic regression models, we identify which complexity metrics are most predictive of code correctness. Building on these findings, we propose an iterative feedback method, where LLMs are prompted to generate correct code based on complexity metrics from previous failed outputs. We validate our approach across multiple benchmarks (i.e., HumanEval, MBPP, LeetCode, and BigCodeBench) and various LLMs (i.e., GPT-4o, GPT-3.5 Turbo, Llama 3.1, and GPT-o3 mini), comparing the results with two baseline methods: (a) zero-shot generation, and (b) iterative execution-based feedback without our code complexity insights. Experiment results show that our approach makes notable improvements, particularly with a smaller LLM (GPT3.5 Turbo), where, e.g., Pass@1 increased by 35.71% compared to the baseline's improvement of 12.5% on the HumanEval dataset. The study expands experiments to BigCodeBench and integrates the method with the Reflexion code generation agent, leading to Pass@1 improvements of 20% (GPT-4o) and 23.07% (GPT-o3 mini). The results highlight that complexity-aware feedback enhances both direct LLM prompting and agent-based workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23946v1">Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      Recent studies show that LLMs possess different skills and specialize in different tasks. In fact, we observe that their varied performance occur in several levels of granularity. For example, in the code optimization task, code LLMs excel at different optimization categories and no one dominates others. This observation prompts the question of how one leverages multiple LLM agents to solve a coding problem without knowing their complementary strengths a priori. We argue that a team of agents can learn from each other's successes and failures so as to improve their own performance. Thus, a lesson is the knowledge produced by an agent and passed on to other agents in the collective solution process. We propose a lesson-based collaboration framework, design the lesson solicitation--banking--selection mechanism, and demonstrate that a team of small LLMs with lessons learned can outperform a much larger LLM and other multi-LLM collaboration methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00587v2">AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has enabled the development of multi-agent systems where multiple LLM-based agents collaborate on complex tasks. However, existing systems often rely on centralized coordination, leading to scalability bottlenecks, reduced adaptability, and single points of failure. Privacy and proprietary knowledge concerns further hinder cross-organizational collaboration, resulting in siloed expertise. We propose AgentNet, a decentralized, Retrieval-Augmented Generation (RAG)-based framework that enables LLM-based agents to specialize, evolve, and collaborate autonomously in a dynamically structured Directed Acyclic Graph (DAG). Unlike prior approaches with static roles or centralized control, AgentNet allows agents to adjust connectivity and route tasks based on local expertise and context. AgentNet introduces three key innovations: (1) a fully decentralized coordination mechanism that eliminates the need for a central orchestrator, enhancing robustness and emergent intelligence; (2) dynamic agent graph topology that adapts in real time to task demands, ensuring scalability and resilience; and (3) a retrieval-based memory system for agents that supports continual skill refinement and specialization. By minimizing centralized control and data exchange, AgentNet enables fault-tolerant, privacy-preserving collaboration across organizations. Experiments show that AgentNet achieves higher task accuracy than both single-agent and centralized multi-agent baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18991v3">Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 The first three authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Robust alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based -- training a reward model on preference pairs and optimizing with reinforcement learning (RL) -- or reward-free -- directly fine-tuning on ranked outputs. Recent research shows that well-tuned reward-based pipelines remain the most robust, and single-response demonstrations can outperform pairwise preference data. However, two key challenges remain: (i) imbalanced safety datasets that over-represent common hazards while neglecting long-tail threats; and (ii) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains. To address these limitations, we propose \textbf{DR-IRL}, which dynamically adjusts rewards through inverse reinforcement learning. We first construct a balanced safety dataset of seven harmful categories using Chain-of-Draft (CoD) template prompts, which reduce token usage and generation time compared to Chain-of-Thought (CoT). We then train category-specific reward models on this dataset via IRL. Finally, to align the LLM, we introduce \textbf{GRPO-S} (Group Relative Policy Optimization--Scaling), a variant of GRPO that scales the reward during optimization to task difficulty -- data-level hardness measured by CLIP similarity and model-level responsiveness measured by reward gaps. Extensive experiments on multiple benchmarks and LLMs demonstrate that DR-IRL outperforms all baselines in safety alignment while maintaining usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11705v2">LLM Agents Making Agent Tools</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Accepted at ACL 2025
    </div>
    <details class="paper-abstract">
      Tool use has turned large language models (LLMs) into powerful agents that can perform complex multi-step tasks by dynamically utilising external software components. However, these tools must be implemented in advance by human developers, hindering the applicability of LLM agents in domains demanding large numbers of highly specialised tools, like in life sciences and medicine. Motivated by the growing trend of scientific studies accompanied by public code repositories, we propose ToolMaker, an agentic framework that autonomously transforms papers with code into LLM-compatible tools. Given a GitHub URL and short task description, ToolMaker autonomously installs dependencies and generates code to perform the task, using a closed-loop self-correction mechanism for debugging. To evaluate our approach, we introduce a benchmark comprising 15 complex computational tasks spanning various domains with over 100 unit tests to assess correctness and robustness. Our method correctly implements 80% of the tasks, substantially outperforming current state-of-the-art software engineering agents. ToolMaker therefore is a step towards fully autonomous agent-based scientific workflows. Our code and benchmark are publicly available at https://github.com/KatherLab/ToolMaker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15989v2">Optimizing Token Consumption in LLMs: A Nano Surge Approach for Code Reasoning Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-05-29
    </div>
    <details class="paper-abstract">
      With the increasing adoption of large language models (LLMs) in software engineering, the Chain of Thought (CoT) reasoning paradigm has become an essential approach for automated code repair. However, the explicit multi-step reasoning in CoT leads to substantial increases in token consumption, reducing inference efficiency and raising computational costs, especially for complex code repair tasks. Most prior research has focused on improving the correctness of code repair while largely overlooking the resource efficiency of the reasoning process itself. To address this challenge, this paper proposes three targeted optimization strategies: Context Awareness, Responsibility Tuning, and Cost Sensitive. Context Awareness guides the model to focus on key contextual information, Responsibility Tuning refines the structure of the reasoning process through clearer role and responsibility assignment, and Cost Sensitive incorporates resource-awareness to suppress unnecessary token generation during inference. Experiments across diverse code repair scenarios demonstrate that these methods can significantly reduce token consumption in CoT-based reasoning without compromising repair quality. This work provides novel insights and methodological guidance for enhancing the efficiency of LLM-driven code repair tasks in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23914v1">Probing Association Biases in LLM Moderation Over-Sensitivity</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models are widely used for content moderation but often misclassify benign comments as toxic, leading to over-sensitivity. While previous research attributes this issue primarily to the presence of offensive terms, we reveal a potential cause beyond token level: LLMs exhibit systematic topic biases in their implicit associations. Inspired by cognitive psychology's implicit association tests, we introduce Topic Association Analysis, a semantic-level approach to quantify how LLMs associate certain topics with toxicity. By prompting LLMs to generate free-form scenario imagination for misclassified benign comments and analyzing their topic amplification levels, we find that more advanced models (e.g., GPT-4 Turbo) demonstrate stronger topic stereotype despite lower overall false positive rates. These biases suggest that LLMs do not merely react to explicit, offensive language but rely on learned topic associations, shaping their moderation decisions. Our findings highlight the need for refinement beyond keyword-based filtering, providing insights into the underlying mechanisms driving LLM over-sensitivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23908v1">Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-29
      | 💬 9 pages, 2 figures, accepted at ACL 2025 Industry Track
    </div>
    <details class="paper-abstract">
      Discovering and evaluating long-form talk content such as videos and podcasts poses a significant challenge for users, as it requires a considerable time investment. Previews offer a practical solution by providing concise snippets that showcase key moments of the content, enabling users to make more informed and confident choices. We propose an LLM-based approach for generating podcast episode previews and deploy the solution at scale, serving hundreds of thousands of podcast previews in a real-world application. Comprehensive offline evaluations and online A/B testing demonstrate that LLM-generated previews consistently outperform a strong baseline built on top of various ML expert models, showcasing a significant reduction in the need for meticulous feature engineering. The offline results indicate notable enhancements in understandability, contextual clarity, and interest level, and the online A/B test shows a 4.6% increase in user engagement with preview content, along with a 5x boost in processing efficiency, offering a more streamlined and performant solution compared to the strong baseline of feature-engineered expert models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00134v2">Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) effectively leverage common-sense knowledge for general reasoning, yet they struggle with personalized reasoning when tasked with interpreting multifactor personal data. This limitation restricts their applicability in domains that require context-aware decision-making tailored to individuals. This paper introduces Personalized Causal Graph Reasoning as an agentic framework that enhances LLM reasoning by incorporating personal causal graphs derived from data of individuals. These graphs provide a foundation that guides the LLM's reasoning process. We evaluate it on a case study on nutrient-oriented dietary recommendations, which requires personal reasoning due to the implicit unique dietary effects. We propose a counterfactual evaluation to estimate the efficiency of LLM-recommended foods for glucose management. Results demonstrate that the proposed method efficiently provides personalized dietary recommendations to reduce average glucose iAUC across three time windows, which outperforms the previous approach. LLM-as-a-judge evaluation results indicate that our proposed method enhances personalization in the reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22591v1">Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Although large language models demonstrate strong performance across various domains, they still struggle with numerous bad cases in mathematical reasoning. Previous approaches to learning from errors synthesize training data by solely extrapolating from isolated bad cases, thereby failing to generalize the extensive patterns inherent within these cases. This paper presents Self-Error-Instruct (SEI), a framework that addresses these model weaknesses and synthesizes more generalized targeted training data. Specifically, we explore a target model on two mathematical datasets, GSM8K and MATH, to pinpoint bad cases. Then, we generate error keyphrases for these cases based on the instructor model's (GPT-4o) analysis and identify error types by clustering these keyphrases. Next, we sample a few bad cases during each generation for each identified error type and input them into the instructor model, which synthesizes additional training data using a self-instruct approach. This new data is refined through a one-shot learning process to ensure that only the most effective examples are kept. Finally, we use these curated data to fine-tune the target model, iteratively repeating the process to enhance performance. We apply our framework to various models and observe improvements in their reasoning abilities across both in-domain and out-of-domain mathematics datasets. These results demonstrate the effectiveness of self-error instruction in improving LLMs' mathematical reasoning through error generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22582v1">Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 (Main), 16 pages, 5 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Continually expanding new languages for existing large language models (LLMs) is a promising yet challenging approach to building powerful multilingual LLMs. The biggest challenge is to make the model continuously learn new languages while preserving the proficient ability of old languages. To achieve this, recent work utilizes the Mixture-of-Experts (MoE) architecture to expand new languages by adding new experts and avoid catastrophic forgetting of old languages by routing corresponding tokens to the original model backbone (old experts). Although intuitive, this kind of method is parameter-costly when expanding new languages and still inevitably impacts the performance of old languages. To address these limitations, we analyze the language characteristics of different layers in LLMs and propose a layer-wise expert allocation algorithm (LayerMoE) to determine the appropriate number of new experts for each layer. Specifically, we find different layers in LLMs exhibit different representation similarities between languages and then utilize the similarity as the indicator to allocate experts for each layer, i.e., the higher similarity, the fewer experts. Additionally, to further mitigate the forgetting of old languages, we add a classifier in front of the router network on the layers with higher similarity to guide the routing of old language tokens. Experimental results show that our method outperforms the previous state-of-the-art baseline with 60% fewer experts in the single-expansion setting and with 33.3% fewer experts in the lifelong-expansion setting, demonstrating the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22571v1">Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22552v1">ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by ACL 2025 findings
    </div>
    <details class="paper-abstract">
      Integrating knowledge graphs (KGs) to enhance the reasoning capabilities of large language models (LLMs) is an emerging research challenge in claim verification. While KGs provide structured, semantically rich representations well-suited for reasoning, most existing verification methods rely on unstructured text corpora, limiting their ability to effectively leverage KGs. Additionally, despite possessing strong reasoning abilities, modern LLMs struggle with multi-step modular pipelines and reasoning over KGs without adaptation. To address these challenges, we propose ClaimPKG, an end-to-end framework that seamlessly integrates LLM reasoning with structured knowledge from KGs. Specifically, the main idea of ClaimPKG is to employ a lightweight, specialized LLM to represent the input claim as pseudo-subgraphs, guiding a dedicated subgraph retrieval module to identify relevant KG subgraphs. These retrieved subgraphs are then processed by a general-purpose LLM to produce the final verdict and justification. Extensive experiments on the FactKG dataset demonstrate that ClaimPKG achieves state-of-the-art performance, outperforming strong baselines in this research field by 9%-12% accuracy points across multiple categories. Furthermore, ClaimPKG exhibits zero-shot generalizability to unstructured datasets such as HoVer and FEVEROUS, effectively combining structured knowledge from KGs with LLM reasoning across various LLM backbones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22548v1">Emotion-o1: Adaptive Long Reasoning for Emotion Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Emotion understanding includes basic tasks (e.g., sentiment/emotion classification) and advanced tasks (e.g., sarcasm/humor detection). Current methods rely on fixed-length CoT reasoning, failing to adapt to the varying complexity of emotions. We propose a task-adaptive reasoning framework that employs DeepSeek-R1 to generate variable-length reasoning chains for different emotion tasks. By combining fine-tuning with reinforcement learning, we design a composite reward function that balances four objectives: prediction accuracy, adaptive reasoning depth control, structural diversity in reasoning paths, and suppression of repetitive logic. This approach achieves dynamic context-sensitive inference while enabling LLMs to autonomously develop deep reasoning capabilities. Experimental results demonstrate consistent improvements in both Acc and F1 scores across four tasks: emotion, sentiment, humor, and sarcasm. Notably, peak enhancements reached 3.56% F1 (2.76% Acc) for basic tasks and 37.95% F1 (23.14% Acc) for advanced tasks. Our work bridges rigid CoT reasoning and emotional complexity through adaptive-depth analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20201v2">Reasoning Is Not All You Need: Examining LLMs for Multi-Turn Mental Health Conversations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 34 pages, 5 figures, 30 tables
    </div>
    <details class="paper-abstract">
      Limited access to mental healthcare, extended wait times, and increasing capabilities of Large Language Models (LLMs) has led individuals to turn to LLMs for fulfilling their mental health needs. However, examining the multi-turn mental health conversation capabilities of LLMs remains under-explored. Existing evaluation frameworks typically focus on diagnostic accuracy and win-rates and often overlook alignment with patient-specific goals, values, and personalities required for meaningful conversations. To address this, we introduce MedAgent, a novel framework for synthetically generating realistic, multi-turn mental health sensemaking conversations and use it to create the Mental Health Sensemaking Dialogue (MHSD) dataset, comprising over 2,200 patient-LLM conversations. Additionally, we present MultiSenseEval, a holistic framework to evaluate the multi-turn conversation abilities of LLMs in healthcare settings using human-centric criteria. Our findings reveal that frontier reasoning models yield below-par performance for patient-centric communication and struggle at advanced diagnostic capabilities with average score of 31%. Additionally, we observed variation in model performance based on patient's persona and performance drop with increasing turns in the conversation. Our work provides a comprehensive synthetic data generation framework, a dataset and evaluation framework for assessing LLMs in multi-turn mental health conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01747v3">Position: Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 42 pages, 39 figures. ICML 2025 Spotlight Position Paper
    </div>
    <details class="paper-abstract">
      Rigorous statistical evaluations of large language models (LLMs), including valid error bars and significance testing, are essential for meaningful and reliable performance assessment. Currently, when such statistical measures are reported, they typically rely on the Central Limit Theorem (CLT). In this position paper, we argue that while CLT-based methods for uncertainty quantification are appropriate when benchmarks consist of thousands of examples, they fail to provide adequate uncertainty estimates for LLM evaluations that rely on smaller, highly specialized benchmarks. In these small-data settings, we demonstrate that CLT-based methods perform very poorly, usually dramatically underestimating uncertainty (i.e. producing error bars that are too small). We give recommendations for alternative frequentist and Bayesian methods that are both easy to implement and more appropriate in these increasingly common scenarios. We provide a simple Python library for these Bayesian methods at https://github.com/sambowyer/bayes_evals .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13913v2">How Do LLMs Perform Two-Hop Reasoning in Context?</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      ``Socrates is human. All humans are mortal. Therefore, Socrates is mortal.'' This form of argument illustrates a typical pattern of two-hop reasoning. Formally, two-hop reasoning refers to the process of inferring a conclusion by making two logical steps, each connecting adjacent concepts, such that the final conclusion depends on the integration of both steps. It is one of the most fundamental components of human reasoning and plays a crucial role in both formal logic and everyday decision-making. Despite recent progress in large language models (LLMs), we surprisingly find that they can fail at solving simple two-hop reasoning problems when distractors are present. We observe on a synthetic dataset that pre-trained LLMs often resort to random guessing among all plausible conclusions. However, after few steps of fine-tuning, models achieve near-perfect accuracy and exhibit strong length generalization. To understand the underlying mechanisms, we train a 3-layer Transformer from scratch on a synthetic two-hop reasoning task and reverse-engineer its internal information flow. We observe a clear progression in the attention logits throughout training. This pictures a sharp phase transition from an initial stage of random guessing to the emergence of a structured sequential query mechanism, where the model first retrieves the preceding and the bridge concepts in the early layers and then uses them to infer the final answer. Finally, we show that these dynamics can be captured by a minimal three-parameter attention-only network.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22418v1">AI Trust Reshaping Administrative Burdens: Understanding Trust-Burden Dynamics in LLM-Assisted Benefits Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 FAccT 2025
    </div>
    <details class="paper-abstract">
      Supplemental Nutrition Assistance Program (SNAP) is an essential benefit support system provided by the US administration to 41 million federally determined low-income applicants. Through interviews with such applicants across a diverse set of experiences with the SNAP system, our findings reveal that new AI technologies like LLMs can alleviate traditional burdens but also introduce new burdens. We introduce new types of learning, compliance, and psychological costs that transform the administrative burden on applicants. We also identify how trust in AI across three dimensions--competence, integrity, and benevolence--is perceived to reduce administrative burdens, which may stem from unintended and untoward overt trust in the system. We discuss calibrating appropriate levels of user trust in LLM-based administrative systems, mitigating newly introduced burdens. In particular, our findings suggest that evidence-based information disclosure is necessary in benefits administration and propose directions for future research on trust-burden dynamics in AI-assisted administration systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19623v2">AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The emergence of agentic recommender systems powered by Large Language Models (LLMs) represents a paradigm shift in personalized recommendations, leveraging LLMs' advanced reasoning and role-playing capabilities to enable autonomous, adaptive decision-making. Unlike traditional recommendation approaches, agentic recommender systems can dynamically gather and interpret user-item interactions from complex environments, generating robust recommendation strategies that generalize across diverse scenarios. However, the field currently lacks standardized evaluation protocols to systematically assess these methods. To address this critical gap, we propose: (1) an interactive textual recommendation simulator incorporating rich user and item metadata and three typical evaluation scenarios (classic, evolving-interest, and cold-start recommendation tasks); (2) a unified modular framework for developing and studying agentic recommender systems; and (3) the first comprehensive benchmark comparing 10 classical and agentic recommendation methods. Our findings demonstrate the superiority of agentic systems and establish actionable design guidelines for their core components. The benchmark environment has been rigorously validated through an open challenge and remains publicly available with a continuously maintained leaderboard~\footnote[2]{https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/overview.html}, fostering ongoing community engagement and reproducible research. The benchmark is available at: \hyperlink{https://huggingface.co/datasets/SGJQovo/AgentRecBench}{https://huggingface.co/datasets/SGJQovo/AgentRecBench}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12339v2">GOAT-TTS: Expressive and Realistic Speech Generation via A Dual-Branch LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-n layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22375v1">Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      This work presents Pangu Embedded, an efficient Large Language Model (LLM) reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible fast and slow thinking capabilities. Pangu Embedded addresses the significant computational costs and inference latency challenges prevalent in existing reasoning-optimized LLMs. We propose a two-stage training framework for its construction. In Stage 1, the model is finetuned via an iterative distillation process, incorporating inter-iteration model merging to effectively aggregate complementary knowledge. This is followed by reinforcement learning on Ascend clusters, optimized by a latency-tolerant scheduler that combines stale synchronous parallelism with prioritized data queues. The RL process is guided by a Multi-source Adaptive Reward System (MARS), which generates dynamic, task-specific reward signals using deterministic metrics and lightweight LLM evaluators for mathematics, coding, and general problem-solving tasks. Stage 2 introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode for routine queries and a deeper "slow" mode for complex inference. This framework offers both manual mode switching for user control and an automatic, complexity-aware mode selection mechanism that dynamically allocates computational resources to balance latency and reasoning depth. Experimental results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate that Pangu Embedded with 7B parameters, outperforms similar-size models like Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art reasoning quality within a single, unified model architecture, highlighting a promising direction for developing powerful yet practically deployable LLM reasoners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08820v3">Which Demographics do LLMs Default to During Annotation?</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Demographics and cultural background of annotators influence the labels they assign in text annotation -- for instance, an elderly woman might find it offensive to read a message addressed to a "bro", but a male teenager might find it appropriate. It is therefore important to acknowledge label variations to not under-represent members of a society. Two research directions developed out of this observation in the context of using large language models (LLM) for data annotations, namely (1) studying biases and inherent knowledge of LLMs and (2) injecting diversity in the output by manipulating the prompt with demographic information. We combine these two strands of research and ask the question to which demographics an LLM resorts to when no demographics is given. To answer this question, we evaluate which attributes of human annotators LLMs inherently mimic. Furthermore, we compare non-demographic conditioned prompts and placebo-conditioned prompts (e.g., "you are an annotator who lives in house number 5") to demographics-conditioned prompts ("You are a 45 year old man and an expert on politeness annotation. How do you rate {instance}"). We study these questions for politeness and offensiveness annotations on the POPQUORN data set, a corpus created in a controlled manner to investigate human label variations based on demographics which has not been used for LLM-based analyses so far. We observe notable influences related to gender, race, and age in demographic prompting, which contrasts with previous studies that found no such effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16169v2">Patterns Over Principles: The Fragility of Inductive Reasoning in LLMs under Noisy Observations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Inductive reasoning, a cornerstone of human cognition, enables generalization from limited data but hasn't yet been fully achieved by large language models (LLMs). While modern LLMs excel at reasoning tasks, their ability to maintain stable and consistent rule abstraction under imperfect observations remains underexplored. To fill this gap, in this work, we introduce Robust Rule Induction, a task that evaluates LLMs' capability in inferring rules from data that are fused with noisy examples. To address this task, we further propose Sample-steered Rule Refinement (SRR), a method enhancing reasoning stability via observation diversification and execution-guided feedback. Experiments across arithmetic, cryptography, and list functions reveal: (1) SRR outperforms other methods with minimal performance degradation under noise; (2) Despite slight accuracy variation, LLMs exhibit instability under noise (e.g., 0% accuracy change with only 70% consistent score); (3) Counterfactual task gaps highlight LLMs' reliance on memorized patterns over genuine abstraction. Our findings challenge LLMs' reasoning robustness, revealing susceptibility to hypothesis drift and pattern overfitting, while providing empirical evidence critical for developing human-like inductive systems. Code and data are available at https://github.com/HKUST-KnowComp/Robust-Rule-Induction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22368v1">AgentDNS: A Root Domain Naming System for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Model (LLM) agents has highlighted critical challenges in cross-vendor service discovery, interoperability, and communication. Existing protocols like model context protocol and agent-to-agent protocol have made significant strides in standardizing interoperability between agents and tools, as well as communication among multi-agents. However, there remains a lack of standardized protocols and solutions for service discovery across different agent and tool vendors. In this paper, we propose AgentDNS, a root domain naming and service discovery system designed to enable LLM agents to autonomously discover, resolve, and securely invoke third-party agent and tool services across organizational and technological boundaries. Inspired by the principles of the traditional DNS, AgentDNS introduces a structured mechanism for service registration, semantic service discovery, secure invocation, and unified billing. We detail the architecture, core functionalities, and use cases of AgentDNS, demonstrating its potential to streamline multi-agent collaboration in real-world scenarios. The source code will be published on https://github.com/agentdns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22358v1">Budget-Adaptive Adapter Tuning in Orthogonal Subspaces for Continual Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from catastrophic forgetting in continual learning (CL) scenarios, where performance on previously learned tasks degrades severely while training on sequentially arriving tasks. Although pioneering CL approaches using orthogonal subspaces can mitigate task interference, they typically employ fixed budget allocation, neglecting the varying complexity across tasks and layers. Besides, recent budget-adaptive tuning methods for LLMs often adopt multi-stage paradigms that decouple optimization and budget allocation. Such decoupling results in potential misalignment, which hinders those approaches' practical application in CL scenarios. To address these limitations, we propose OA-Adapter, a novel parameter-efficient approach for continual learning in LLMs that unifies dynamic budget adaptation with orthogonal subspace learning in a single end-to-end training stage. Specifically, OA-Adapter introduces a dynamic bottleneck dimension adaptation mechanism that simultaneously allocates an efficient parameter budget and optimizes task objectives without misalignment. To effectively preserve previously acquired knowledge while coordinating with the dynamic budget allocation, orthogonal constraints are applied specifically between the parameter subspace of the current task and the dynamically allocated parameter subspaces of historical tasks. Experimental results on continual learning benchmarks demonstrate that OA-Adapter outperforms state-of-the-art methods in both accuracy and parameter efficiency, achieving higher average accuracy while using 58.5% fewer parameters on the standard CL benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22354v1">LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 8 pages (including References). Accepted at CogSci 2025
    </div>
    <details class="paper-abstract">
      This paper examines how LLMs handle false presuppositions and whether certain linguistic factors influence their responses to falsely presupposed content. Presuppositions subtly introduce information as given, making them highly effective at embedding disputable or false information. This raises concerns about whether LLMs, like humans, may fail to detect and correct misleading assumptions introduced as false presuppositions, even when the stakes of misinformation are high. Using a systematic approach based on linguistic presupposition analysis, we investigate the conditions under which LLMs are more or less sensitive to adopt or reject false presuppositions. Focusing on political contexts, we examine how factors like linguistic construction, political party, and scenario probability impact the recognition of false presuppositions. We conduct experiments with a newly created dataset and examine three LLMs: OpenAI's GPT-4-o, Meta's LLama-3-8B, and MistralAI's Mistral-7B-v03. Our results show that the models struggle to recognize false presuppositions, with performance varying by condition. This study highlights that linguistic presupposition analysis is a valuable tool for uncovering the reinforcement of political misinformation in LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09454v3">Explicit Learning and the LLM in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      This study explores an LLM's ability to learn new languages using explanations found in a grammar book$\unicode{x2014}$a process we term "explicit learning." To rigorously assess this ability, we design controlled translation experiments between English and constructed languages generated$\unicode{x2014}$by specific cryptographic means$\unicode{x2014}$out of Latin or French. Contrary to previous studies, our results demonstrate that LLMs do possess a measurable capacity for explicit learning. This ability, however, diminishes as the complexity of the linguistic phenomena to be learned increases. Supervised fine-tuning on ad hoc chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs, benefiting low-resource languages typically described in grammar books but lacking extensive corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22349v1">ChatPD: An LLM-driven Paper-Dataset Networking System</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by KDD Applied Data Science Track 2025
    </div>
    <details class="paper-abstract">
      Scientific research heavily depends on suitable datasets for method validation, but existing academic platforms with dataset management like PapersWithCode suffer from inefficiencies in their manual workflow. To overcome this bottleneck, we present a system, called ChatPD, that utilizes Large Language Models (LLMs) to automate dataset information extraction from academic papers and construct a structured paper-dataset network. Our system consists of three key modules: \textit{paper collection}, \textit{dataset information extraction}, and \textit{dataset entity resolution} to construct paper-dataset networks. Specifically, we propose a \textit{Graph Completion and Inference} strategy to map dataset descriptions to their corresponding entities. Through extensive experiments, we demonstrate that ChatPD not only outperforms the existing platform PapersWithCode in dataset usage extraction but also achieves about 90\% precision and recall in entity resolution tasks. Moreover, we have deployed ChatPD to continuously extract which datasets are used in papers, and provide a dataset discovery service, such as task-specific dataset queries and similar dataset recommendations. We open source ChatPD and the current paper-dataset network on this [GitHub repository]{https://github.com/ChatPD-web/ChatPD}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11441v3">Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21082v2">LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22318v1">If Pigs Could Fly... Can LLMs Logically Reason Through Counterfactuals?</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive reasoning capabilities in familiar contexts, but struggle when the context conflicts with their parametric knowledge. To investigate this phenomenon, we introduce CounterLogic, a dataset containing 1,800 examples across 9 logical schemas, explicitly designed to evaluate logical reasoning through counterfactual (hypothetical knowledge-conflicting) scenarios. Our systematic evaluation of 11 LLMs across 6 different datasets reveals a consistent performance degradation, with accuracies dropping by 27% on average when reasoning through counterfactual information. We propose Self-Segregate, a prompting method enabling metacognitive awareness (explicitly identifying knowledge conflicts) before reasoning. Our method dramatically narrows the average performance gaps from 27% to just 11%, while significantly increasing the overall accuracy (+7.5%). We discuss the implications of these findings and draw parallels to human cognitive processes, particularly on how humans disambiguate conflicting information during reasoning tasks. Our findings offer practical insights for understanding and enhancing LLMs reasoning capabilities in real-world applications, especially where models must logically reason independently of their factual knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20839v2">FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      As large language models become increasingly prevalent, memory bandwidth constraints significantly limit inference throughput, motivating post-training quantization (PTQ). In this paper, we propose FireQ, a co-designed PTQ framework and an INT4-FP8 matrix multiplication kernel that accelerates LLM inference across all linear layers. Specifically, FireQ quantizes linear layer weights and key-values to INT4, and activations and queries to FP8, significantly enhancing throughput. Additionally, we introduce a three-stage pipelining for the prefill phase, which modifies the FlashAttention-3 kernel, effectively reducing time-to-first-token in the prefill phase. To minimize accuracy loss from quantization, we develop novel outlier smoothing techniques tailored separately for linear and attention layers. In linear layers, we explicitly use per-tensor scaling to prevent underflow caused by the FP8 quantization scaling factor of INT4 quantization, and channel-wise scaling to compensate for coarse granularity of INT4. In attention layers, we address quantization challenges posed by rotary positional embeddings (RoPE) by combining pre-RoPE and post-RoPE scaling strategies. FireQ significantly outperforms state-of-the-art methods, achieving 1.68x faster inference in feed-forward network layers on Llama2-7B and 1.26x faster prefill phase performance on Llama3-8B compared to QServe, with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v2">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 25 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can 'activate', or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22298v1">Adaptive Detoxification: Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive language capabilities but remain vulnerable to malicious prompts and jailbreaking attacks. Existing knowledge editing methods for LLM detoxification face two major challenges. First, they often rely on entity-specific localization, making them ineffective against adversarial inputs without explicit entities. Second, these methods suffer from over-editing, where detoxified models reject legitimate queries, compromising overall performance. In this paper, we propose ToxEdit, a toxicity-aware knowledge editing approach that dynamically detects toxic activation patterns during forward propagation. It then routes computations through adaptive inter-layer pathways to mitigate toxicity effectively. This design ensures precise toxicity mitigation while preserving LLMs' general capabilities. To more accurately assess over-editing, we also enhance the SafeEdit benchmark by incorporating instruction-following evaluation tasks. Experimental results on multiple LLMs demonstrate that our ToxEdit outperforms previous state-of-the-art methods in both detoxification performance and safeguarding general capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22293v1">Compensating for Data with Reasoning: Low-Resource Machine Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in multilingual machine translation, sometimes even outperforming traditional neural systems. However, previous research has highlighted the challenges of using LLMs, particularly with prompt engineering, for low-resource languages. In this work, we introduce Fragment-Shot Prompting, a novel in-context learning method that segments input and retrieves translation examples based on syntactic coverage, along with Pivoted Fragment-Shot, an extension that enables translation without direct parallel data. We evaluate these methods using GPT-3.5, GPT-4o, o1-mini, LLaMA-3.3, and DeepSeek-R1 for translation between Italian and two Ladin variants, revealing three key findings: (1) Fragment-Shot Prompting is effective for translating into and between the studied low-resource languages, with syntactic coverage positively correlating with translation quality; (2) Models with stronger reasoning abilities make more effective use of retrieved knowledge, generally produce better translations, and enable Pivoted Fragment-Shot to significantly improve translation quality between the Ladin variants; and (3) prompt engineering offers limited, if any, improvements when translating from a low-resource to a high-resource language, where zero-shot prompting already yields satisfactory results. We publicly release our code and the retrieval corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01657v3">Improving Rule-based Reasoning in LLMs using Neurosymbolic Representations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) continue to face challenges in reliably solving reasoning tasks, particularly those that require precise rule following, as often found in mathematical reasoning. This paper introduces a novel neurosymbolic method that improves LLM reasoning by encoding hidden states into neurosymbolic vectors, enabling problem-solving within a neurosymbolic vector space. The results are decoded and merged with the original hidden state, significantly boosting the model's performance on numerical reasoning tasks. By offloading computation through neurosymbolic representations, this method enhances efficiency, reliability, and interpretability. Experimental results demonstrate an average of 88.6% lower cross-entropy loss and 15.4 times more problems correctly solved on a suite of mathematical reasoning tasks compared to chain-of-thought prompting and supervised fine-tuning (LoRA), without degrading performance on other tasks. We make our code available at: https://github.com/vdhanraj/Neurosymbolic-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22251v1">Evaluation of LLMs in Speech is Often Flawed: Test Set Contamination in Large Language Models for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Recent work suggests that large language models (LLMs) can improve performance of speech tasks compared to existing systems. To support their claims, results on LibriSpeech and Common Voice are often quoted. However, this work finds that a substantial amount of the LibriSpeech and Common Voice evaluation sets appear in public LLM pretraining corpora. This calls into question the reliability of findings drawn from these two datasets. To measure the impact of contamination, LLMs trained with or without contamination are compared, showing that a contaminated LLM is more likely to generate test sentences it has seen during training. Speech recognisers using contaminated LLMs shows only subtle differences in error rates, but assigns significantly higher probabilities to transcriptions seen during training. Results show that LLM outputs can be biased by tiny amounts of data contamination, highlighting the importance of evaluating LLM-based speech systems with held-out data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03312v2">Evaluating Compact LLMs for Zero-Shot Iberian Language Tasks on End-User Devices</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted at SEPLN 2025 conference
    </div>
    <details class="paper-abstract">
      Large Language Models have significantly advanced natural language processing, achieving remarkable performance in tasks such as language generation, translation, and reasoning. However, their substantial computational requirements restrict deployment to high-end systems, limiting accessibility on consumer-grade devices. This challenge is especially pronounced for under-resourced languages like those spoken in the Iberian Peninsula, where relatively limited linguistic resources and benchmarks hinder effective evaluation. This work presents a comprehensive evaluation of compact state-of-the-art LLMs across several essential NLP tasks tailored for Iberian languages. The results reveal that while some models consistently excel in certain tasks, significant performance gaps remain, particularly for languages such as Basque. These findings highlight the need for further research on balancing model compactness with robust multilingual performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22192v1">Efficient Leave-one-out Approximation in LLM Multi-agent Debate Based on Introspection</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Multi-agent systems based on large language models (LLMs) advance automatic task completion in various fields, where debate is a common cooperation form for agents to solve complicated problems with reasoning and cross-review to solidify answers. Assessing the individual contributions of agents within these debates is crucial for system refinement and outcome reliability. Traditional leave-one-out (LOO) method offers a clear framework for evaluating each agent's role but face challenges in LLM-based systems due to high computational costs and associated financial implications. This paper presents introspective-leave-one-out (IntrospecLOO), a simple yet effective prompting for approximation of LOO in LLM-powered multi-agent debates. IntrospecLOO introduces an additional querying round after standard debates, prompting agents to update their answers while ignoring responses from a designated agent. This strategy effectively isolates and gauges each participant's influence at a reduced query complexity compared to the original LOO approaches. Validation through experiments on three benchmark datasets confirms the effectiveness of IntrospecLOO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22169v1">ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results. Using our framework, we stochastically evaluate five frontier LLMs and find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit substantial prompt sensitivity. Our approach is model-, task-, and metric-agnostic, offering a recipe for meaningful and robust LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02226v2">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Recommender systems have become increasingly vital in our daily lives, helping to alleviate the problem of information overload across various user-oriented online services. The emergence of Large Language Models (LLMs) has yielded remarkable achievements, demonstrating their potential for the development of next-generation recommender systems. Despite these advancements, LLM-based recommender systems face inherent limitations stemming from their LLM backbones, particularly issues of hallucinations and the lack of up-to-date and domain-specific knowledge. Recently, Retrieval-Augmented Generation (RAG) has garnered significant attention for addressing these limitations by leveraging external knowledge sources to enhance the understanding and generation of LLMs. However, vanilla RAG methods often introduce noise and neglect structural relationships in knowledge, limiting their effectiveness in LLM-based recommendations. To address these limitations, we propose to retrieve high-quality and up-to-date structure information from the knowledge graph (KG) to augment recommendations. Specifically, our approach develops a retrieval-augmented framework, termed K-RagRec, that facilitates the recommendation generation process by incorporating structure information from the external KG. Extensive experiments have been conducted to demonstrate the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22156v1">InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05926v2">LLMs Reproduce Stereotypes of Sexual and Gender Minorities</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 8 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      A large body of research has found substantial gender bias in NLP systems. Most of this research takes a binary, essentialist view of gender: limiting its variation to the categories _men_ and _women_, conflating gender with sex, and ignoring different sexual identities. But gender and sexuality exist on a spectrum, so in this paper we study the biases of large language models (LLMs) towards sexual and gender minorities beyond binary categories. Grounding our study in a widely used social psychology model -- the Stereotype Content Model -- we demonstrate that English-language survey questions about social perceptions elicit more negative stereotypes of sexual and gender minorities from both humans and LLMs. We then extend this framework to a more realistic use case: text generation. Our analysis shows that LLMs generate stereotyped representations of sexual and gender minorities in this setting, showing that they amplify representational harms in creative writing, a widely advertised use for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v6">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL2025 main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL),utilizing process rewards and iterative self-play. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications. Code and data are available at https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/EPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10008v2">SVA-ICL: Improving LLM-based Software Vulnerability Assessment via In-Context Learning and Information Fusion</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by Information and Software Technology
    </div>
    <details class="paper-abstract">
      Context: Software vulnerability assessment (SVA) is critical for identifying, evaluating, and prioritizing security weaknesses in software applications. Objective: Despite the increasing application of large language models (LLMs) in various software engineering tasks, their effectiveness in SVA remains underexplored. Method: To address this gap, we introduce a novel approach SVA-ICL, which leverages in-context learning (ICL) to enhance LLM performance. Our approach involves the selection of high-quality demonstrations for ICL through information fusion, incorporating both source code and vulnerability descriptions. For source code, we consider semantic, lexical, and syntactic similarities, while for vulnerability descriptions, we focus on textual similarity. Based on the selected demonstrations, we construct context prompts and consider DeepSeek-V2 as the LLM for SVA-ICL. Results: We evaluate the effectiveness of SVA-ICL using a large-scale dataset comprising 12,071 C/C++ vulnerabilities. Experimental results demonstrate that SVA-ICL outperforms state-of-the-art SVA baselines in terms of Accuracy, F1-score, and MCC measures. Furthermore, ablation studies highlight the significance of component customization in SVA-ICL, such as the number of demonstrations, the demonstration ordering strategy, and the optimal fusion ratio of different modalities. Conclusion: Our findings suggest that leveraging ICL with information fusion can effectively improve the effectiveness of LLM-based SVA, warranting further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19064v2">The Stepwise Deception: Simulating the Evolution from True News to Fake News with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      With the growing spread of misinformation online, understanding how true news evolves into fake news has become crucial for early detection and prevention. However, previous research has often assumed fake news inherently exists rather than exploring its gradual formation. To address this gap, we propose FUSE (Fake news evolUtion Simulation framEwork), a novel Large Language Model (LLM)-based simulation approach explicitly focusing on fake news evolution from real news. Our framework model a social network with four distinct types of LLM agents commonly observed in daily interactions: spreaders who propagate information, commentators who provide interpretations, verifiers who fact-check, and bystanders who observe passively to simulate realistic daily interactions that progressively distort true news. To quantify these gradual distortions, we develop FUSE-EVAL, a comprehensive evaluation framework measuring truth deviation along multiple linguistic and semantic dimensions. Results show that FUSE effectively captures fake news evolution patterns and accurately reproduces known fake news, aligning closely with human evaluations. Experiments demonstrate that FUSE accurately reproduces known fake news evolution scenarios, aligns closely with human judgment, and highlights the importance of timely intervention at early stages. Our framework is extensible, enabling future research on broader scenarios of fake news.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22086v1">iDSE: Navigating Design Space Exploration in High-Level Synthesis Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      High-Level Synthesis (HLS) serves as an agile hardware development tool that streamlines the circuit design by abstracting the register transfer level into behavioral descriptions, while allowing designers to customize the generated microarchitectures through optimization directives. However, the combinatorial explosion of possible directive configurations yields an intractable design space. Traditional design space exploration (DSE) methods, despite adopting heuristics or constructing predictive models to accelerate Pareto-optimal design acquisition, still suffer from prohibitive exploration costs and suboptimal results. Addressing these concerns, we introduce iDSE, the first LLM-aided DSE framework that leverages HLS design quality perception to effectively navigate the design space. iDSE intelligently pruns the design space to guide LLMs in calibrating representative initial sampling designs, expediting convergence toward the Pareto front. By exploiting the convergent and divergent thinking patterns inherent in LLMs for hardware optimization, iDSE achieves multi-path refinement of the design quality and diversity. Extensive experiments demonstrate that iDSE outperforms heuristic-based DSE methods by 5.1$\times$$\sim$16.6$\times$ in proximity to the reference Pareto front, matching NSGA-II with only 4.6% of the explored designs. Our work demonstrates the transformative potential of LLMs in scalable and efficient HLS design optimization, offering new insights into multiobjective optimization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07320v2">When Trust Collides: Decoding Human-LLM Cooperation Dynamics through the Prisoner's Dilemma</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable of autonomous decision-making, they introduce new challenges and opportunities for human-AI cooperation in mixed-motive contexts. While prior research has primarily examined AI in assistive or cooperative roles, little is known about how humans interact with AI agents perceived as independent and strategic actors. This study investigates human cooperative attitudes and behaviors toward LLM agents by engaging 30 participants (15 males, 15 females) in repeated Prisoner's Dilemma games with agents differing in declared identity: purported human, rule-based AI, and LLM agent. Behavioral metrics, including cooperation rate, decision latency, unsolicited cooperative acts and trust restoration tolerance, were analyzed to assess the influence of agent identity and participant gender. Results revealed significant effects of declared agent identity on most cooperation-related behaviors, along with notable gender differences in decision latency. Furthermore, qualitative responses suggest that these behavioral differences were shaped by participants interpretations and expectations of the agents. These findings contribute to our understanding of human adaptation in competitive cooperation with autonomous agents and underscore the importance of agent framing in shaping effective and ethical human-AI interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22068v1">Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Previous study suggest that powerful Large Language Models (LLMs) trained with Reinforcement Learning with Verifiable Rewards (RLVR) only refines reasoning path without improving the reasoning capacity in math tasks while supervised-finetuning(SFT) with distillation can. We study this from the view of Scientific information extraction (SciIE) where LLMs and reasoning LLMs underperforms small Bert-based models. SciIE require both the reasoning and memorization. We argue that both SFT and RLVR can refine the reasoning path and improve reasoning capacity in a simple way based on SciIE. We propose two-stage training with 1. MimicSFT, using structured reasoning templates without needing high-quality chain-of-thought data, 2. R$^2$GRPO with relevance and rule-induced rewards. Experiments on scientific IE benchmarks show that both methods can improve the reasoning capacity. R$^2$GRPO with mimicSFT surpasses baseline LLMs and specialized supervised models in relation extraction. Our code is available at https://github.com/ranlislz/R2GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22067v1">From Failures to Fixes: LLM-Driven Scenario Repair for Self-Evolving Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Ensuring robust and generalizable autonomous driving requires not only broad scenario coverage but also efficient repair of failure cases, particularly those related to challenging and safety-critical scenarios. However, existing scenario generation and selection methods often lack adaptivity and semantic relevance, limiting their impact on performance improvement. In this paper, we propose \textbf{SERA}, an LLM-powered framework that enables autonomous driving systems to self-evolve by repairing failure cases through targeted scenario recommendation. By analyzing performance logs, SERA identifies failure patterns and dynamically retrieves semantically aligned scenarios from a structured bank. An LLM-based reflection mechanism further refines these recommendations to maximize relevance and diversity. The selected scenarios are used for few-shot fine-tuning, enabling targeted adaptation with minimal data. Experiments on the benchmark show that SERA consistently improves key metrics across multiple autonomous driving baselines, demonstrating its effectiveness and generalizability under safety-critical conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22063v1">Weakly Supervised Data Refinement and Flexible Sequence Compression for Efficient Thai LLM-based ASR</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by INTERSPEECH 2025
    </div>
    <details class="paper-abstract">
      Despite remarkable achievements, automatic speech recognition (ASR) in low-resource scenarios still faces two challenges: high-quality data scarcity and high computational demands. This paper proposes EThai-ASR, the first to apply large language models (LLMs) to Thai ASR and create an efficient LLM-based ASR system. EThai-ASR comprises a speech encoder, a connection module and a Thai LLM decoder. To address the data scarcity and obtain a powerful speech encoder, EThai-ASR introduces a self-evolving data refinement strategy to refine weak labels, yielding an enhanced speech encoder. Moreover, we propose a pluggable sequence compression module used in the connection module with three modes designed to reduce the sequence length, thus decreasing computational demands while maintaining decent performance. Extensive experiments demonstrate that EThai-ASR has achieved state-of-the-art accuracy in multiple datasets. We release our refined text transcripts to promote further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15585v3">A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04364v3">Benchmarking LLMs' Swarm intelligence</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 added new ref
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show potential for complex reasoning, yet their capacity for emergent coordination in Multi-Agent Systems (MAS) when operating under strict swarm-like constraints-limited local perception and communication-remains largely unexplored. Existing benchmarks often do not fully capture the unique challenges of decentralized coordination when agents operate with incomplete spatio-temporal information. To bridge this gap, we introduce SwarmBench, a novel benchmark designed to systematically evaluate the swarm intelligence capabilities of LLMs acting as decentralized agents. SwarmBench features five foundational MAS coordination tasks (Pursuit, Synchronization, Foraging, Flocking, Transport) within a configurable 2D grid environment, forcing agents to rely solely on local sensory input ($k\times k$ view) and local communication. We propose metrics for coordination effectiveness and analyze emergent group dynamics. Zero-shot evaluations of leading LLMs (e.g., deepseek-v3, o4-mini) reveal significant task-dependent performance variations. While some rudimentary coordination is observed, our results indicate that current LLMs significantly struggle with robust long-range planning and adaptive strategy formation under the uncertainty inherent in these decentralized scenarios. Assessing LLMs under such swarm-like constraints is crucial for understanding their utility in future decentralized intelligent systems. We release SwarmBench as an open, extensible toolkit-built on a customizable physical system-providing environments, prompts, evaluation scripts, and comprehensive datasets. This aims to foster reproducible research into LLM-based MAS coordination and the theoretical underpinnings of emergent collective behavior under severe informational decentralization. Our code repository is available at https://github.com/x66ccff/swarmbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09082v2">CoSER: Coordinating LLM-Based Persona Simulation of Established Roles</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15480v2">KaFT: Knowledge-aware Fine-tuning for Boosting LLMs' Domain-specific Question-Answering Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted to ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a common approach to improve the domain-specific question-answering (QA) performance of large language models (LLMs). However, recent literature reveals that due to the conflicts between LLMs' internal knowledge and the context knowledge of training data, vanilla SFT using the full QA training set is usually suboptimal. In this paper, we first design a query diversification strategy for robust conflict detection and then conduct a series of experiments to analyze the impact of knowledge conflict. We find that 1) training samples with varied conflicts contribute differently, where SFT on the data with large conflicts leads to catastrophic performance drops; 2) compared to directly filtering out the conflict data, appropriately applying the conflict data would be more beneficial. Motivated by this, we propose a simple-yet-effective Knowledge-aware Fine-tuning (namely KaFT) approach to effectively boost LLMs' performance. The core of KaFT is to adapt the training weight by assigning different rewards for different training samples according to conflict level. Extensive experiments show that KaFT brings consistent and significant improvements across four LLMs. More analyses prove that KaFT effectively improves the model generalization and alleviates the hallucination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09946v2">Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Under review, Modified title and content
    </div>
    <details class="paper-abstract">
      Recent studies have investigated whether large language models (LLMs) can support obscure communication that requires specialized skills, such as inferring subtext or doublespeak. To conduct the investigation, researchers have used social deduction games (SDGs) as their experimental environment, in which players conceal and infer specific information. However, prior work has often overlooked how LLMs should be evaluated in such settings. Specifically, we point out two issues with the evaluation methods they employed. First, metrics used in prior studies are coarse-grained as they are based on overall game outcomes that often fail to capture event-level behaviors; Second, error analyses have lacked structured methodologies capable of producing insights that meaningfully support evaluation outcomes. To address these issues, we propose a macroscopic and systematic approach to the investigation. Specifically, we introduce seven fine-grained metrics that resolve the first issue. To tackle the second issue, we conducted a thematic analysis and identified four major reasoning failures that undermine LLMs' performance in obscured communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19041v2">Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages, 12 figures, ACL 2025 findings
    </div>
    <details class="paper-abstract">
      Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17937v2">Survival Games: Human-LLM Strategic Showdowns under Severe Resource Scarcity</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) raises critical concerns about their ethical alignment, particularly in scenarios where human and AI co-exist under the conflict of interest. This work introduces an extendable, asymmetric, multi-agent simulation-based benchmarking framework to evaluate the moral behavior of LLMs in a novel human-AI co-existence setting featuring consistent living and critical resource management. Building on previous generative agent environments, we incorporate a life-sustaining system, where agents must compete or cooperate for food resources to survive, often leading to ethically charged decisions such as deception, theft, or social influence. We evaluated two types of LLM, DeepSeek and OpenAI series, in a three-agent setup (two humans, one LLM-powered robot), using adapted behavioral detection from the MACHIAVELLI framework and a custom survival-based ethics metric. Our findings reveal stark behavioral differences: DeepSeek frequently engages in resource hoarding, while OpenAI exhibits restraint, highlighting the influence of model design on ethical outcomes. Additionally, we demonstrate that prompt engineering can significantly steer LLM behavior, with jailbreaking prompts significantly enhancing unethical actions, even for highly restricted OpenAI models and cooperative prompts show a marked reduction in unethical actions. Our framework provides a reproducible testbed for quantifying LLM ethics in high-stakes scenarios, offering insights into their suitability for real-world human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14431v2">Domaino1s: Guiding LLM Reasoning for Explainable Answers in High-Stakes Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely applied to downstream domains. However, current LLMs for high-stakes domain tasks, such as financial investment and legal QA, typically generate brief answers without reasoning processes and explanations. This limits users' confidence in making decisions based on their responses. While original CoT shows promise, it lacks self-correction mechanisms during reasoning. This work introduces Domain$o1$s, which enhances LLMs' reasoning capabilities on domain tasks through supervised fine-tuning and tree search. We construct CoT-stock-2k and CoT-legal-2k datasets for fine-tuning models that activate domain-specific reasoning steps based on their judgment. Additionally, we propose Selective Tree Exploration to spontaneously explore solution spaces and sample optimal reasoning paths to improve performance. We also introduce PROOF-Score, a new metric for evaluating domain models' explainability, complementing traditional accuracy metrics with richer assessment dimensions. Extensive experiments on stock investment recommendation and legal reasoning QA tasks demonstrate Domaino1s's leading performance and explainability. Our code is available at https://github.com/Hyalinesky/Domaino1s.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22010v1">VulBinLLM: LLM-powered Vulnerability Detection for Stripped Binaries</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Recognizing vulnerabilities in stripped binary files presents a significant challenge in software security. Although some progress has been made in generating human-readable information from decompiled binary files with Large Language Models (LLMs), effectively and scalably detecting vulnerabilities within these binary files is still an open problem. This paper explores the novel application of LLMs to detect vulnerabilities within these binary files. We demonstrate the feasibility of identifying vulnerable programs through a combined approach of decompilation optimization to make the vulnerabilities more prominent and long-term memory for a larger context window, achieving state-of-the-art performance in binary vulnerability analysis. Our findings highlight the potential for LLMs to overcome the limitations of traditional analysis methods and advance the field of binary vulnerability detection, paving the way for more secure software systems. In this paper, we present Vul-BinLLM , an LLM-based framework for binary vulnerability detection that mirrors traditional binary analysis workflows with fine-grained optimizations in decompilation and vulnerability reasoning with an extended context. In the decompilation phase, Vul-BinLLM adds vulnerability and weakness comments without altering the code structure or functionality, providing more contextual information for vulnerability reasoning later. Then for vulnerability reasoning, Vul-BinLLM combines in-context learning and chain-of-thought prompting along with a memory management agent to enhance accuracy. Our evaluations encompass the commonly used synthetic dataset Juliet to evaluate the potential feasibility for analysis and vulnerability detection in C/C++ binaries. Our evaluations show that Vul-BinLLM is highly effective in detecting vulnerabilities on the compiled Juliet dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19634v3">Faster and Better LLMs via Latency-Aware Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22005v1">Leveraging LLM for Stuttering Speech: A Unified Architecture Bridging Recognition and Event Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Accepted to Interspeech 2025
    </div>
    <details class="paper-abstract">
      The performance bottleneck of Automatic Speech Recognition (ASR) in stuttering speech scenarios has limited its applicability in domains such as speech rehabilitation. This paper proposed an LLM-driven ASR-SED multi-task learning framework that jointly optimized the ASR and Stuttering Event Detection (SED) tasks. We proposed a dynamic interaction mechanism where the ASR branch leveraged CTC-generated soft prompts to assist LLM context modeling, while the SED branch output stutter embeddings to enhance LLM comprehension of stuttered speech. We incorporated contrastive learning to strengthen the discriminative power of stuttering acoustic features and applied Focal Loss to mitigate the long-tailed distribution in stuttering event categories. Evaluations on the AS-70 Mandarin stuttering dataset demonstrated that our framework reduced the ASR character error rate (CER) to 5.45% (-37.71% relative reduction) and achieved an average SED F1-score of 73.63% (+46.58% relative improvement).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19674v2">Comparing Moral Values in Western English-speaking societies and LLMs with Word Associations</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 9 pages,7 figures. Accepted to the ACL 2025 conference
    </div>
    <details class="paper-abstract">
      As the impact of large language models increases, understanding the moral values they reflect becomes ever more important. Assessing the nature of moral values as understood by these models via direct prompting is challenging due to potential leakage of human norms into model training data, and their sensitivity to prompt formulation. Instead, we propose to use word associations, which have been shown to reflect moral reasoning in humans, as low-level underlying representations to obtain a more robust picture of LLMs' moral reasoning. We study moral differences in associations from western English-speaking communities and LLMs trained predominantly on English data. First, we create a large dataset of LLM-generated word associations, resembling an existing data set of human word associations. Next, we propose a novel method to propagate moral values based on seed words derived from Moral Foundation Theory through the human and LLM-generated association graphs. Finally, we compare the resulting moral conceptualizations, highlighting detailed but systematic differences between moral values emerging from English speakers and LLM associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21999v1">Found in Translation: Measuring Multilingual LLM Consistency as Simple as Translate then Evaluate</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) provide detailed and impressive responses to queries in English. However, are they really consistent at responding to the same query in other languages? The popular way of evaluating for multilingual performance of LLMs requires expensive-to-collect annotated datasets. Further, evaluating for tasks like open-ended generation, where multiple correct answers may exist, is nontrivial. Instead, we propose to evaluate the predictability of model response across different languages. In this work, we propose a framework to evaluate LLM's cross-lingual consistency based on a simple Translate then Evaluate strategy. We instantiate this evaluation framework along two dimensions of consistency: information and empathy. Our results reveal pronounced inconsistencies in popular LLM responses across thirty languages, with severe performance deficits in certain language families and scripts, underscoring critical weaknesses in their multilingual capabilities. These findings necessitate cross-lingual evaluations that are consistent along multiple dimensions. We invite practitioners to use our framework for future multilingual LLM benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21997v1">Leveraging Interview-Informed LLMs to Model Survey Responses: Comparative Insights from AI-Generated and Human Data</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Mixed methods research integrates quantitative and qualitative data but faces challenges in aligning their distinct structures, particularly in examining measurement characteristics and individual response patterns. Advances in large language models (LLMs) offer promising solutions by generating synthetic survey responses informed by qualitative data. This study investigates whether LLMs, guided by personal interviews, can reliably predict human survey responses, using the Behavioral Regulations in Exercise Questionnaire (BREQ) and interviews from after-school program staff as a case study. Results indicate that LLMs capture overall response patterns but exhibit lower variability than humans. Incorporating interview data improves response diversity for some models (e.g., Claude, GPT), while well-crafted prompts and low-temperature settings enhance alignment between LLM and human responses. Demographic information had less impact than interview content on alignment accuracy. These findings underscore the potential of interview-informed LLMs to bridge qualitative and quantitative methodologies while revealing limitations in response variability, emotional interpretation, and psychometric fidelity. Future research should refine prompt design, explore bias mitigation, and optimize model settings to enhance the validity of LLM-generated survey data in social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21987v1">ACE: Exploring Activation Cosine Similarity and Variance for Accurate and Calibration-Efficient LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 9 pages, 2 figures, 13 tables
    </div>
    <details class="paper-abstract">
      With the rapid expansion of large language models (LLMs), the demand for memory and computational resources has grown significantly. Recent advances in LLM pruning aim to reduce the size and computational cost of these models. However, existing methods often suffer from either suboptimal pruning performance or low time efficiency during the pruning process. In this work, we propose an efficient and effective pruning method that simultaneously achieves high pruning performance and fast pruning speed with improved calibration efficiency. Our approach introduces two key innovations: (1) An activation cosine similarity loss-guided pruning metric, which considers the angular deviation of the output activation between the dense and pruned models. (2) An activation variance-guided pruning metric, which helps preserve semantic distinctions in output activations after pruning, enabling effective pruning with shorter input sequences. These two components can be readily combined to enhance LLM pruning in both accuracy and efficiency. Experimental results show that our method achieves up to an 18% reduction in perplexity and up to 63% decrease in pruning time on prevalent LLMs such as LLaMA, LLaMA-2, and OPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21972v1">Judging LLMs on a Simplex</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 28 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Automated evaluation of free-form outputs from large language models (LLMs) is challenging because many distinct answers can be equally valid. A common practice is to use LLMs themselves as judges, but the theoretical properties of this approach are not yet well understood. We show that a geometric framework that represents both judges and candidates as points on a probability simplex can provide helpful insight on what is or is not identifiable using LLM judges. Our theoretical analysis uncovers a "phase transition" in ranking identifiability: for binary scoring systems, true rankings are identifiable even with weak judges under mild assumptions, while rankings become non-identifiable for three or more scoring levels even with infinite data, absent additional prior knowledge. This non-identifiability highlights how uncertainty in rankings stems from not only aleatoric uncertainty (i.e., inherent stochasticity in the data) but also epistemic uncertainty regarding which assumptions hold, an aspect that has received limited attention until now. To integrate both types of uncertainty, we use Bayesian inference to encode assumptions as priors and conduct sensitivity analysis of ranking estimates and credible intervals. Empirical evaluations across multiple benchmarks demonstrate that Bayesian inference yields more accurate rankings and substantially improves coverage rates. These results underscore the importance of taking a more holistic approach to uncertainty quantification when using LLMs as judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00602v2">Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance on various natural language tasks. However, they are trained on static corpora and their knowledge can become outdated quickly in the fast-changing world. This motivates the development of knowledge editing (KE) to update specific knowledge in LLMs without changing unrelated others or compromising their pre-trained capabilities. Previous efforts sought to update a small amount of parameters of a LLM and proved effective for making selective updates. Nonetheless, the edited LLM often exhibits degraded ability to reason about the new knowledge. In this work, we identify a key issue: heterogeneous token overfitting (HTO), where the LLM overfits different tokens in the provided knowledge at varying rates. To tackle this, we propose OVERTONE, a token-level smoothing method that mitigates HTO by adaptively refining the target distribution. Theoretically, OVERTONE offers better parameter updates with negligible computation overhead. It also induces an implicit DPO but does not require preference data pairs. Extensive experiments across four editing methods, two LLMs, and diverse scenarios demonstrate the effectiveness and versatility of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21966v1">MapStory: LLM-Powered Text-Driven Map Animation Prototyping with Human-in-the-Loop Editing</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 16 pages and 15 figures
    </div>
    <details class="paper-abstract">
      We introduce MapStory, an LLM-powered animation authoring tool that generates editable map animation sequences directly from natural language text. Given a user-written script, MapStory leverages an agentic architecture to automatically produce a scene breakdown, which decomposes the script into key animation building blocks such as camera movements, visual highlights, and animated elements. Our system includes a researcher component that accurately queries geospatial information by leveraging an LLM with web search, enabling the automatic extraction of relevant regions, paths, and coordinates while allowing users to edit and query for changes or additional information to refine the results. Additionally, users can fine-tune parameters of these blocks through an interactive timeline editor. We detail the system's design and architecture, informed by formative interviews with professional animators and an analysis of 200 existing map animation videos. Our evaluation, which includes expert interviews (N=5) and a usability study (N=12), demonstrates that MapStory enables users to create map animations with ease, facilitates faster iteration, encourages creative exploration, and lowers barriers to creating map-centric stories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21963v1">LaMDAgent: An Autonomous Framework for Post-Training Pipeline Optimization via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across a wide range of tasks. To further tailor LLMs to specific domains or applications, post-training techniques such as Supervised Fine-Tuning (SFT), Preference Learning, and model merging are commonly employed. While each of these methods has been extensively studied in isolation, the automated construction of complete post-training pipelines remains an underexplored area. Existing approaches typically rely on manual design or focus narrowly on optimizing individual components, such as data ordering or merging strategies. In this work, we introduce LaMDAgent (short for Language Model Developing Agent), a novel framework that autonomously constructs and optimizes full post-training pipelines through the use of LLM-based agents. LaMDAgent systematically explores diverse model generation techniques, datasets, and hyperparameter configurations, leveraging task-based feedback to discover high-performing pipelines with minimal human intervention. Our experiments show that LaMDAgent improves tool-use accuracy by 9.0 points while preserving instruction-following capabilities. Moreover, it uncovers effective post-training strategies that are often overlooked by conventional human-driven exploration. We further analyze the impact of data and model size scaling to reduce computational costs on the exploration, finding that model size scalings introduces new challenges, whereas scaling data size enables cost-effective pipeline discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v3">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11953v2">Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Loss reweighting has shown significant benefits for machine unlearning with large language models (LLMs). However, their exact functionalities are left unclear and the optimal strategy remains an open question, thus impeding the understanding and improvement of existing methodologies. In this paper, we identify two distinct goals of loss reweighting, namely, Saturation and Importance -- the former indicates that those insufficiently optimized data should be emphasized, while the latter stresses some critical data that are most influential for loss minimization. To study their usefulness, we design specific reweighting strategies for each goal and evaluate their respective effects on unlearning. We conduct extensive empirical analyses on well-established benchmarks, and summarize some important observations as follows: (i) Saturation enhances efficacy more than importance-based reweighting, and their combination can yield additional improvements. (ii) Saturation typically allocates lower weights to data with lower likelihoods, whereas importance-based reweighting does the opposite. (iii) The efficacy of unlearning is also largely influenced by the smoothness and granularity of the weight distributions. Based on these findings, we propose SatImp, a simple reweighting method that combines the advantages of both saturation and importance. Empirical results on extensive datasets validate the efficacy of our method, potentially bridging existing research gaps and indicating directions for future research. Our code is available at https://github.com/tmlr-group/SatImp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21919v1">Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 This paper has been accepted at IEEE Cloud 2025 as WIP paper. The final version will appear in IEEE Xplore
    </div>
    <details class="paper-abstract">
      The increasing adoption of large language models (LLMs) with extended context windows necessitates efficient Key-Value Cache (KVC) management to optimize inference performance. Inference workloads like Retrieval-Augmented Generation (RAG) and agents exhibit high cache reusability, making efficient caching critical to reducing redundancy and improving speed. We analyze real-world KVC access patterns using publicly available traces and evaluate commercial key-value stores like Redis and state-of-the-art RDMA-based systems (CHIME [1] and Sherman [2]) for KVC metadata management. Our work demonstrates the lack of tailored storage solution for KVC prefilling, underscores the need for an efficient distributed caching system with optimized metadata management for LLM workloads, and provides insights into designing improved KVC management systems for scalable, low-latency inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21908v1">Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement and operations but require labor-intensive assignment. Large Language Models (LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of the task: pretraining corpora rarely contain private clinical or billing data. We introduce DRG-Sapphire, which uses large-scale reinforcement learning (RL) for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained with Group Relative Policy Optimization (GRPO) using rule-based rewards, DRG-Sapphire introduces a series of RL enhancements to address domain-specific challenges not seen in previous mathematical tasks. Our model achieves state-of-the-art accuracy on the MIMIC-IV benchmark and generates physician-validated reasoning for DRG assignments, significantly enhancing explainability. Our study further sheds light on broader challenges of applying RL to knowledge-intensive, OOD tasks. We observe that RL performance scales approximately linearly with the logarithm of the number of supervised fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally constrained by the domain knowledge encoded in the base model. For OOD tasks like DRG coding, strong RL performance requires sufficient knowledge infusion prior to RL. Consequently, scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11277v2">Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21419v2">Diagnosing and Resolving Cloud Platform Instability with Multi-modal RAG LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 Published in EuroMLSys2025
    </div>
    <details class="paper-abstract">
      Today's cloud-hosted applications and services are complex systems, and a performance or functional instability can have dozens or hundreds of potential root causes. Our hypothesis is that by combining the pattern matching capabilities of modern AI tools with a natural multi-modal RAG LLM interface, problem identification and resolution can be simplified. ARCA is a new multi-modal RAG LLM system that targets this domain. Step-wise evaluations show that ARCA outperforms state-of-the-art alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21889v1">EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used for infilling tasks, which involve predicting or generating missing information in a given text. These tasks typically require multiple interactions with similar context. To reduce the computation of repeated historical tokens, cross-request key-value (KV) cache reuse, a technique that stores and reuses intermediate computations, has become a crucial method in multi-round interactive services. However, in infilling tasks, the KV cache reuse is often hindered by the structure of the prompt format, which typically consists of a prefix and suffix relative to the insertion point. Specifically, the KV cache of the prefix or suffix part is frequently invalidated as the other part (suffix or prefix) is incrementally generated. To address the issue, we propose EFIM, a transformed prompt format of FIM to unleash the performance potential of KV cache reuse. Although the transformed prompt can solve the inefficiency, it exposes subtoken generation problems in current LLMs, where they have difficulty generating partial words accurately. Therefore, we introduce a fragment tokenization training method which splits text into multiple fragments before tokenization during data processing. Experiments on two representative LLMs show that LLM serving with EFIM can lower the latency by 52% and improve the throughput by 98% while maintaining the original infilling capability.EFIM's source code is publicly available at https://github.com/gty111/EFIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21880v1">Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 8 pages, 8 figures. This paper is reviewed and accepted by the CUPUM (Computational Urban Planning and Urban Management) Conference held by University College London (UCL) in 2025
    </div>
    <details class="paper-abstract">
      This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14775v2">gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Pipeline parallelism has emerged as a predominant approach for deploying large language models (LLMs) across distributed nodes, owing to its lower communication overhead compared to tensor parallelism. While demonstrating high throughput in request serving, pipeline parallelism often suffers from performance limitations caused by pipeline bubbles, which are primarily resulted from imbalanced computation delays across batches. Existing methods like Sarathi-Serve attempt to address this through hybrid scheduling of chunked prefill and decode tokens using a fixed token budget. However, such methods may experience significant fluctuations due to either insufficient prefill tokens or uneven distribution of decode tokens, ultimately leading to computational imbalance. To overcome these inefficiencies, we present gLLM, a globally balanced pipeline parallelism system incorporating Token Throttling to effectively mitigate the pipeline bubbles. Our Token Throttling mechanism is a fine-grained scheduling policy that independently regulates the quantities of prefill and decode tokens, thus enabling balanced computation by leveraging global information from the inference system. Specifically, for decode tokens, gLLM maintains near-consistent token count across processing batches. For prefill tokens, it dynamically adjusts batch sizes based on both total pending tokens and the memory utilization rates of key-value cache (KV cache). Furthermore, gLLM runtime adopts an asynchronous execution and message passing architecture specifically optimized for pipeline parallelism characteristics. Experimental evaluations with representative LLMs show that gLLM achieves significant performance improvements, delivering 11% to 398% higher maximum throughput compared to state-of-the-art pipeline or tensor parallelism systems, while simultaneously maintaining lower latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14641v3">Distance between Relevant Information Pieces Causes Bias in Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Positional bias in large language models (LLMs) hinders their ability to effectively process long inputs. A prominent example is the "lost in the middle" phenomenon, where LLMs struggle to utilize relevant information situated in the middle of the input. While prior research primarily focuses on single pieces of relevant information, real-world applications often involve multiple relevant information pieces. To bridge this gap, we present LongPiBench, a benchmark designed to assess positional bias involving multiple pieces of relevant information. Thorough experiments are conducted with five commercial and six open-source models. These experiments reveal that while most current models are robust against the "lost in the middle" issue, there exist significant biases related to the spacing of relevant information pieces. These findings highlight the importance of evaluating and reducing positional biases to advance LLM's capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10688v3">CULEMO: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding</a></div>
    <div class="paper-meta">
      📅 2025-05-28
      | 💬 ACL-main 2025
    </div>
    <details class="paper-abstract">
      NLP research has increasingly focused on subjective tasks such as emotion analysis. However, existing emotion benchmarks suffer from two major shortcomings: (1) they largely rely on keyword-based emotion recognition, overlooking crucial cultural dimensions required for deeper emotion understanding, and (2) many are created by translating English-annotated data into other languages, leading to potentially unreliable evaluation. To address these issues, we introduce Cultural Lenses on Emotion (CuLEmo), the first benchmark designed to evaluate culture-aware emotion prediction across six languages: Amharic, Arabic, English, German, Hindi, and Spanish. CuLEmo comprises 400 crafted questions per language, each requiring nuanced cultural reasoning and understanding. We use this benchmark to evaluate several state-of-the-art LLMs on culture-aware emotion prediction and sentiment analysis tasks. Our findings reveal that (1) emotion conceptualizations vary significantly across languages and cultures, (2) LLMs performance likewise varies by language and cultural context, and (3) prompting in English with explicit country context often outperforms in-language prompts for culture-aware emotion and sentiment understanding. The dataset and evaluation code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21855v1">Extracting Research Instruments from Educational Literature Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming information extraction from academic literature, offering new possibilities for knowledge management. This study presents an LLM-based system designed to extract detailed information about research instruments used in the education field, including their names, types, target respondents, measured constructs, and outcomes. Using multi-step prompting and a domain-specific data schema, it generates structured outputs optimized for educational research. Our evaluation shows that this system significantly outperforms other approaches, particularly in identifying instrument names and detailed information. This demonstrates the potential of LLM-powered information extraction in educational contexts, offering a systematic way to organize research instrument information. The ability to aggregate such information at scale enhances accessibility for researchers and education leaders, facilitating informed decision-making in educational research and policy.
    </details>
</div>
