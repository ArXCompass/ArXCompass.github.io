# llm - 2025_02

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v2">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14304v2">MASTER: A Multi-Agent System with LLM Specialized MCTS</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted by main NAACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) are increasingly being explored for problem-solving tasks. However, their strategic planning capability is often viewed with skepticism. Recent studies have incorporated the Monte Carlo Tree Search (MCTS) algorithm to augment the planning capacity of LLM. Despite its potential, MCTS relies on extensive sampling simulations to approximate the true reward distribution, which leads to two primary issues. Firstly, MCTS is effective for tasks like the Game of Go, where simulation results can yield objective rewards (e.g., 1 for a win and 0 for a loss). However, for tasks such as question answering, the result of a simulation is the answer to the question, which cannot yield an objective reward without the ground truth. Secondly, obtaining statistically significant reward estimations typically requires a sample size exceeding 30 simulations, resulting in excessive token usage and time consumption. To address these challenges, we present the Multi-Agent System with Tactical Execution and Reasoning using LLM Specialized MCTS (MASTER), a novel framework that coordinates agent recruitment and communication through LLM specialized MCTS. This system autonomously adjusts the number of agents based on task complexity and ensures focused communication among them. Comprehensive experiments across various tasks demonstrate the effectiveness of our proposed framework. It achieves 76% accuracy on HotpotQA and 80% on WebShop, setting new state-of-the-art performance on these datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00412v2">Adapting While Learning: Grounding LLMs for Scientific Problems with Intelligent Tool Usage Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 32 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate promising capabilities in solving simple scientific problems but, even with domain-specific fine-tuning, often produce hallucinations for complex ones. While integrating LLMs with tools can mitigate this reliability issue, models finetuned on tool usage only often over-rely on them, incurring unnecessary costs from resource-intensive scientific tools even for simpler problems. Inspired by how human experts assess the complexity of the problem before choosing the solutions, we propose a novel two-component fine-tuning method, Adapting While Learning (AWL). In the first component, World Knowledge Learning (WKL), LLMs internalize scientific knowledge by learning from tools-generated solutions. In the second component, Tool Usage Adaptation (TUA), we classify questions as easy or hard based on the WKL-trained model's accuracy, and train it to maintain direct reasoning for simple problems while switching to tools for challenging ones. We validate our method on 6 scientific benchmark datasets in climate science, epidemiology, and mathematics. Compared to the base 8B model, our trained models achieve 28.27% higher answer accuracy and 13.76% better tool usage accuracy, even surpassing state-of-the-art models including GPT-4 and Claude-3.5 on 4 custom-created datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02009v1">LLMSecConfig: An LLM-Based Approach for Fixing Software Container Misconfigurations</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Security misconfigurations in Container Orchestrators (COs) can pose serious threats to software systems. While Static Analysis Tools (SATs) can effectively detect these security vulnerabilities, the industry currently lacks automated solutions capable of fixing these misconfigurations. The emergence of Large Language Models (LLMs), with their proven capabilities in code understanding and generation, presents an opportunity to address this limitation. This study introduces LLMSecConfig, an innovative framework that bridges this gap by combining SATs with LLMs. Our approach leverages advanced prompting techniques and Retrieval-Augmented Generation (RAG) to automatically repair security misconfigurations while preserving operational functionality. Evaluation of 1,000 real-world Kubernetes configurations achieved a 94\% success rate while maintaining a low rate of introducing new misconfigurations. Our work makes a promising step towards automated container security management, reducing the manual effort required for configuration maintenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01992v1">FinRLlama: A Solution to LLM-Engineered Signals Challenge at FinRL Contest 2024</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Competition Track FinRL, ICAIF 2024
    </div>
    <details class="paper-abstract">
      In response to Task II of the FinRL Challenge at ACM ICAIF 2024, this study proposes a novel prompt framework for fine-tuning large language models (LLM) with Reinforcement Learning from Market Feedback (RLMF). Our framework incorporates market-specific features and short-term price dynamics to generate more precise trading signals. Traditional LLMs, while competent in sentiment analysis, lack contextual alignment for financial market applications. To bridge this gap, we fine-tune the LLaMA-3.2-3B-Instruct model using a custom RLMF prompt design that integrates historical market data and reward-based feedback. Our evaluation shows that this RLMF-tuned framework outperforms baseline methods in signal consistency and achieving tighter trading outcomes; awarded as winner of Task II. You can find the code for this project on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01991v1">Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted at 17th ACM Web Science Conference 2025 (WebSci'25)
    </div>
    <details class="paper-abstract">
      Nowadays, social media is pivotal in shaping public discourse, especially on polarizing issues like vaccination, where diverse moral perspectives influence individual opinions. In NLP, data scarcity and complexity of psycholinguistic tasks such as identifying morality frames makes relying solely on human annotators costly, time-consuming, and prone to inconsistency due to cognitive load. To address these issues, we leverage large language models (LLMs), which are adept at adapting new tasks through few-shot learning, utilizing a handful of in-context examples coupled with explanations that connect examples to task principles. Our research explores LLMs' potential to assist human annotators in identifying morality frames within vaccination debates on social media. We employ a two-step process: generating concepts and explanations with LLMs, followed by human evaluation using a "think-aloud" tool. Our study shows that integrating LLMs into the annotation process enhances accuracy, reduces task difficulty, lowers cognitive load, suggesting a promising avenue for human-AI collaboration in complex psycholinguistic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17017v3">Reasoning Aware Self-Consistency: Leveraging Reasoning Paths for Efficient LLM Sampling</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Self-Consistency mitigates hallucinations in Large Language Models (LLMs) by sampling multiple reasoning paths,but it lacks a systematic approach to determine the optimal number of samples or select the most faithful rationale. To address this limitation, we introduce Reasoning-Aware Self-Consistency (RASC), a novel framework that enhances sampling efficiency and reasoning faithfulness by dynamically evaluating both outputs and rationales. RASC assesses the quality of reasoning and the consistency of answers for each generated sample, using these assessments to guide early stopping decisions and rationale selection. The framework employs criteria-based stopping and weighted majority voting, enabling more informed choices on when to halt sampling and which rationale to select. Our comprehensive experiments across diverse question-answering datasets demonstrate that RASC outperforms existing methods, reducing sample usage by approximately 70% while maintaining accuracy. Moreover, RASC facilitates the selection of high-fidelity rationales, thereby improving the faithfulness of LLM outputs. Our approach effectively addresses the efficiency-accuracy trade-off in LLM reasoning tasks, offering a new perspective for more nuanced, faithful, and effective utilization of LLMs in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01066v2">From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 15 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      LLMs when used with Retrieval Augmented Generation (RAG), are greatly improving the SOTA of translating natural language queries to structured and correct SQL. Unlike previous reviews, this survey provides a comprehensive study of the evolution of LLM-based text-to-SQL systems, from early rule-based models to advanced LLM approaches that use (RAG) systems. We discuss benchmarks, evaluation methods, and evaluation metrics. Also, we uniquely study the use of Graph RAGs for better contextual accuracy and schema linking in these systems. Finally, we highlight key challenges such as computational efficiency, model robustness, and data privacy toward improvements of LLM-based text-to-SQL systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01977v1">AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      User interface understanding with vision-language models has received much attention due to its potential for enabling next-generation software automation. However, existing UI datasets either only provide large-scale context-free element annotations or contextualized functional descriptions for elements at a much smaller scale. In this work, we propose the \methodname{} pipeline for automatically annotating UI elements with detailed functionality descriptions at scale. Specifically, we leverage large language models (LLMs) to infer element functionality by comparing the UI content changes before and after simulated interactions with specific UI elements. To improve annotation quality, we propose LLM-aided rejection and verification, eliminating invalid and incorrect annotations without human labor. We construct an \methodname{}-704k dataset using the proposed pipeline, featuring multi-resolution, multi-device screenshots, diverse data domains, and detailed functionality annotations that have never been provided by previous datasets. Human evaluation shows that the AutoGUI pipeline achieves annotation correctness comparable to trained human annotators. Extensive experimental results show that our \methodname{}-704k dataset remarkably enhances VLM's UI grounding capabilities, exhibits significant scaling effects, and outperforms existing web pre-training data types. We envision AutoGUI as a scalable pipeline for generating massive data to build GUI-oriented VLMs. AutoGUI dataset can be viewed at this anonymous URL: https://autogui-project.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01968v1">Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recent studies show that in supervised fine-tuning (SFT) of large language models (LLMs), data quality matters more than quantity. While most data cleaning methods concentrate on filtering entire samples, the quality of individual tokens within a sample can vary significantly. After pre-training, even in high-quality samples, patterns or phrases that are not task-related can be redundant or uninformative. Continuing to fine-tune on these patterns may offer limited benefit and even degrade downstream task performance. In this paper, we investigate token quality from a noisy-label perspective and propose a generic token cleaning pipeline for SFT tasks. Our method filters out uninformative tokens while preserving those carrying key task-specific information. Specifically, we first evaluate token quality by examining the influence of model updates on each token, then apply a threshold-based separation. The token influence can be measured in a single pass with a fixed reference model or iteratively with self-evolving reference models. The benefits and limitations of both methods are analyzed theoretically by error upper bounds. Extensive experiments show that our framework consistently improves performance across multiple downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.00948v3">Large Linguistic Models: Investigating LLMs' metalinguistic abilities</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) has recently improved to the point where the models can perform well on many language tasks. We show here that -- for the first time -- the models can also generate valid metalinguistic analyses of language data. We outline a research program where the behavioral interpretability of LLMs on these tasks is tested via prompting. LLMs are trained primarily on text -- as such, evaluating their metalinguistic abilities improves our understanding of their general capabilities and sheds new light on theoretical models in linguistics. We show that OpenAI's o1 vastly outperforms other models on tasks involving drawing syntactic trees and phonological generalization. We speculate that OpenAI o1's unique advantage over other models may result from the model's chain-of-thought mechanism, which mimics the structure of human reasoning used in complex cognitive tasks, such as linguistic analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01941v1">Can LLMs Maintain Fundamental Abilities under KV Cache Compression?</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      This paper investigates an under-explored challenge in large language models (LLMs): the impact of KV cache compression methods on LLMs' fundamental capabilities. While existing methods achieve impressive compression ratios on long-context benchmarks, their effects on core model capabilities remain understudied. We present a comprehensive empirical study evaluating prominent KV cache compression methods across diverse tasks, spanning world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and generation.Our analysis reveals that KV cache compression methods exhibit task-specific performance degradation. Arithmetic reasoning tasks prove particularly sensitive to aggressive compression, with different methods showing performance drops of $17.4\%$-$43.3\%$. Notably, the DeepSeek R1 Distill model exhibits more robust compression tolerance compared to instruction-tuned models, showing only $9.67\%$-$25.53\%$ performance degradation. Based on our analysis of attention patterns and cross-task compression performance, we propose ShotKV, a novel compression approach that distinctly handles prefill and decoding phases while maintaining shot-level semantic coherence. Empirical results show that ShotKV achieves $9\%$-$18\%$ performance improvements on long-context generation tasks under aggressive compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09615v2">SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Conventional model compression techniques for LLMs address high memory consumption and slow inference challenges but typically require computationally expensive retraining to preserve accuracy. In contrast, one-shot compression methods eliminate retraining cost, but struggle to achieve accuracy comparable to dense models. This paper presents SLIM, a new one-shot compression framework that holistically integrates hardware-friendly quantization, sparsity, and low-rank approximation into a unified process. First, we formulate the quantization process using a probabilistic approach (SLIM-Quant) that enables us to apply uniform quantization. Then, we use an existing one-shot pruning method to apply semi-structured sparsity on top of the quantized weights. Finally, to compensate for the introduced aggregated quantization and sparsity error, we use a novel saliency function with unique invertible and additive features that enables us to mathematically compute the value of low-rank adapters. SLIM improves model accuracy by up to 5.66% (LLaMA-2-7B) for 2:4 sparsity with 4-bit weight quantization, outperforming prior methods. Models compressed with SLIM achieve up to 3.78x and 3.75x layer-wise speedup on Nvidia RTX3060 and A100 GPUs, respectively. We also propose an optional PEFT recipe that further improves accuracy by up to 1.66% (LLaMA-2-13B) compared to SLIM without fine-tuning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02773v1">SD++: Enhancing Standard Definition Maps by Incorporating Road Knowledge using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      High-definition maps (HD maps) are detailed and informative maps capturing lane centerlines and road elements. Although very useful for autonomous driving, HD maps are costly to build and maintain. Furthermore, access to these high-quality maps is usually limited to the firms that build them. On the other hand, standard definition (SD) maps provide road centerlines with an accuracy of a few meters. In this paper, we explore the possibility of enhancing SD maps by incorporating information from road manuals using LLMs. We develop SD++, an end-to-end pipeline to enhance SD maps with location-dependent road information obtained from a road manual. We suggest and compare several ways of using LLMs for such a task. Furthermore, we show the generalization ability of SD++ by showing results from both California and Japan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05534v2">Can LLMs Replace Manual Annotation of Software Engineering Artifacts?</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Experimental evaluations of software engineering innovations, e.g., tools and processes, often include human-subject studies as a component of a multi-pronged strategy to obtain greater generalizability of the findings. However, human-subject studies in our field are challenging, due to the cost and difficulty of finding and employing suitable subjects, ideally, professional programmers with varying degrees of experience. Meanwhile, large language models (LLMs) have recently started to demonstrate human-level performance in several areas. This paper explores the possibility of substituting costly human subjects with much cheaper LLM queries in evaluations of code and code-related artifacts. We study this idea by applying six state-of-the-art LLMs to ten annotation tasks from five datasets created by prior work, such as judging the accuracy of a natural language summary of a method or deciding whether a code change fixes a static analysis warning. Our results show that replacing some human annotation effort with LLMs can produce inter-rater agreements equal or close to human-rater agreement. To help decide when and how to use LLMs in human-subject studies, we propose model-model agreement as a predictor of whether a given task is suitable for LLMs at all, and model confidence as a means to select specific samples where LLMs can safely replace human annotators. Overall, our work is the first step toward mixed human-LLM evaluations in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02743v1">LLM Bandit: Cost-Efficient LLM Generation via Preference-Conditioned Dynamic Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The rapid advancement in large language models (LLMs) has brought forth a diverse range of models with varying capabilities that excel in different tasks and domains. However, selecting the optimal LLM for user queries often involves a challenging trade-off between accuracy and cost, a problem exacerbated by the diverse demands of individual queries. In this work, we present a novel framework that formulates the LLM selection process as a multi-armed bandit problem, enabling dynamic and intelligent routing of queries to the most appropriate model. Our approach incorporates a preference-conditioned dynamic routing mechanism, allowing users to specify their preferences at inference time, thereby offering a customizable balance between performance and cost. Additionally, our selection policy is designed to generalize to unseen LLMs, ensuring adaptability to new models as they emerge. Experimental results demonstrate that our method achieves significant improvements in both accuracy and cost-effectiveness across various LLM platforms, showcasing the potential of our framework to adaptively optimize LLM selection in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13284v2">Learning to Route LLMs with Confidence Tokens</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance on several tasks and are increasingly deployed in real-world applications. However, especially in high-stakes settings, it becomes vital to know when the output of an LLM may be unreliable. Depending on whether an answer is trustworthy, a system can then choose to route the question to another expert, or otherwise fall back on a safe default behavior. In this work, we study the extent to which LLMs can reliably indicate confidence in their answers, and how this notion of confidence can translate into downstream accuracy gains. We propose Self-REF, a lightweight training strategy to teach LLMs to express confidence in whether their answers are correct in a reliable manner. Self-REF introduces confidence tokens into the LLM, from which a confidence score can be extracted. Compared to conventional approaches such as verbalizing confidence and examining token probabilities, we demonstrate empirically that confidence tokens show significant improvements in downstream routing and rejection learning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02723v1">Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      We provide a new LLM-compression solution via SVD, unlocking new possibilities for LLM compression beyond quantization and pruning. We point out that the optimal use of SVD lies in truncating activations, rather than merely using activations as an optimization distance. Building on this principle, we address three critical challenges in SVD-based LLM compression: including (1) How can we determine the optimal activation truncation position for each weight matrix in LLMs? (2) How can we efficiently reconstruct the weight matrices based on truncated activations? (3) How can we address the inherent "injection" nature that results in the information loss of the SVD? We propose Dobi-SVD, which establishes a new, principled approach to SVD-based LLM compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02715v1">An Analysis of LLM Fine-Tuning and Few-Shot Learning for Flaky Test Detection and Classification</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Flaky tests exhibit non-deterministic behavior during execution and they may pass or fail without any changes to the program under test. Detecting and classifying these flaky tests is crucial for maintaining the robustness of automated test suites and ensuring the overall reliability and confidence in the testing. However, flaky test detection and classification is challenging due to the variability in test behavior, which can depend on environmental conditions and subtle code interactions. Large Language Models (LLMs) offer promising approaches to address this challenge, with fine-tuning and few-shot learning (FSL) emerging as viable techniques. With enough data fine-tuning a pre-trained LLM can achieve high accuracy, making it suitable for organizations with more resources. Alternatively, we introduce FlakyXbert, an FSL approach that employs a Siamese network architecture to train efficiently with limited data. To understand the performance and cost differences between these two methods, we compare fine-tuning on larger datasets with FSL in scenarios restricted by smaller datasets. Our evaluation involves two existing flaky test datasets, FlakyCat and IDoFT. Our results suggest that while fine-tuning can achieve high accuracy, FSL provides a cost-effective approach with competitive accuracy, which is especially beneficial for organizations or projects with limited historical data available for training. These findings underscore the viability of both fine-tuning and FSL in flaky test detection and classification with each suited to different organizational needs and resource availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02705v4">Certifying LLM Safety against Adversarial Prompting</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted at COLM 2024: https://openreview.net/forum?id=9Ik05cycLq
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08854v3">Hybrid LLM-DDQN based Joint Optimization of V2I Communication and Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted by IEEE Wireless Communications Letters
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have received considerable interest recently due to their outstanding reasoning and comprehension capabilities. This work explores applying LLMs to vehicular networks, aiming to jointly optimize vehicle-to-infrastructure (V2I) communications and autonomous driving (AD) policies. We deploy LLMs for AD decision-making to maximize traffic flow and avoid collisions for road safety, and a double deep Q-learning algorithm (DDQN) is used for V2I optimization to maximize the received data rate and reduce frequent handovers. In particular, for LLM-enabled AD, we employ the Euclidean distance to identify previously explored AD experiences, and then LLMs can learn from past good and bad decisions for further improvement. Then, LLM-based AD decisions will become part of states in V2I problems, and DDQN will optimize the V2I decisions accordingly. After that, the AD and V2I decisions are iteratively optimized until convergence. Such an iterative optimization approach can better explore the interactions between LLMs and conventional reinforcement learning techniques, revealing the potential of using LLMs for network optimization and management. Finally, the simulations demonstrate that our proposed hybrid LLM-DDQN approach outperforms the conventional DDQN algorithm, showing faster convergence and higher average rewards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02675v1">Exploring LLMs Impact on Student-Created User Stories and Acceptance Testing in Software Development</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 3m pages
    </div>
    <details class="paper-abstract">
      In Agile software development methodology, a user story describes a new feature or functionality from an end user's perspective. The user story details may also incorporate acceptance testing criteria, which can be developed through negotiation with users. When creating stories from user feedback, the software engineer may maximize their usefulness by considering story attributes, including scope, independence, negotiability, and testability. This study investigates how LLMs (large language models), with guided instructions, affect undergraduate software engineering students' ability to transform user feedback into user stories. Students, working individually, were asked to analyze user feedback comments, appropriately group related items, and create user stories following the principles of INVEST, a framework for assessing user stories. We found that LLMs help students develop valuable stories with well-defined acceptance criteria. However, students tend to perform better without LLMs when creating user stories with an appropriate scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02659v1">A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation (GALI)</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 9 pages, under review in the conference
    </div>
    <details class="paper-abstract">
      Transformer-based Large Language Models (LLMs) struggle to process inputs exceeding their training context window, with performance degrading due to positional out-of-distribution (O.O.D.) that disrupt attention computations. Existing solutions, fine-tuning and training-free methods, are limited by computational inefficiency, attention logit outliers or loss of local positional information. To address this, we propose Greedy Attention Logit Interpolation (GALI), a training-free length extrapolation method that maximizes the utilization of pretrained positional intervals while avoiding attention logit outliers through attention logit interpolation. The result demonstrates that GALI consistently outperforms state-of-the-art training-free methods. Our findings reveal that LLMs interpret positional intervals unevenly within their training context window, suggesting that extrapolating within a smaller positional interval range yields superior results-even for short-context tasks. GALI represents a significant step toward resolving the positional O.O.D. challenge, enabling more reliable long-text understanding in LLMs. Our implementation of GALI, along with the experiments from our paper, is open-sourced at https://github.com/AcademyCityL/GALI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04358v1">Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 12 pages including references
    </div>
    <details class="paper-abstract">
      Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04351v1">NER4all or Context is All You Need: Using LLMs for low-effort, high-performance NER on historical texts. A humanities informed approach</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Named entity recognition (NER) is a core task for historical research in automatically establishing all references to people, places, events and the like. Yet, do to the high linguistic and genre diversity of sources, only limited canonisation of spellings, the level of required historical domain knowledge, and the scarcity of annotated training data, established approaches to natural language processing (NLP) have been both extremely expensive and yielded only unsatisfactory results in terms of recall and precision. Our paper introduces a new approach. We demonstrate how readily-available, state-of-the-art LLMs significantly outperform two leading NLP frameworks, spaCy and flair, for NER in historical documents by seven to twentytwo percent higher F1-Scores. Our ablation study shows how providing historical context to the task and a bit of persona modelling that turns focus away from a purely linguistic approach are core to a successful prompting strategy. We also demonstrate that, contrary to our expectations, providing increasing numbers of examples in few-shot approaches does not improve recall or precision below a threshold of 16-shot. In consequence, our approach democratises access to NER for all historians by removing the barrier of scripting languages and computational skills required for established NLP tools and instead leveraging natural language prompts and consumer-grade tools and frontends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04349v1">Dynamic benchmarking framework for LLM-based conversational data capture</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has transformed conversational agents, enabling complex human-machine interactions. However, evaluation frameworks often focus on single tasks, failing to capture the dynamic nature of multi-turn dialogues. This paper introduces a dynamic benchmarking framework to assess LLM-based conversational agents through interactions with synthetic users. The framework integrates generative agent simulation to evaluate performance on key dimensions: information extraction, context awareness, and adaptive engagement. By simulating various aspects of user behavior, our work provides a scalable, automated, and flexible benchmarking approach. Experimental evaluation - within a loan application use case - demonstrates the framework's effectiveness under one-shot and few-shot extraction conditions. Results show that adaptive strategies improve data extraction accuracy, especially when handling ambiguous responses. Future work will extend its applicability to broader domains and incorporate additional metrics (e.g., conversational coherence, user engagement). This study contributes a structured, scalable approach to evaluating LLM-based conversational agents, facilitating real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04347v1">SCALM: Detecting Bad Practices in Smart Contracts Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      As the Ethereum platform continues to mature and gain widespread usage, it is crucial to maintain high standards of smart contract writing practices. While bad practices in smart contracts may not directly lead to security issues, they do elevate the risk of encountering problems. Therefore, to understand and avoid these bad practices, this paper introduces the first systematic study of bad practices in smart contracts, delving into over 35 specific issues. Specifically, we propose a large language models (LLMs)-based framework, SCALM. It combines Step-Back Prompting and Retrieval-Augmented Generation (RAG) to identify and address various bad practices effectively. Our extensive experiments using multiple LLMs and datasets have shown that SCALM outperforms existing tools in detecting bad practices in smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04346v1">Multi-Lingual Cyber Threat Detection in Tweets/X Using ML, DL, and LLM: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Cyber threat detection has become an important area of focus in today's digital age due to the growing spread of fake information and harmful content on social media platforms such as Twitter (now 'X'). These cyber threats, often disguised within tweets, pose significant risks to individuals, communities, and even nations, emphasizing the need for effective detection systems. While previous research has explored tweet-based threats, much of the work is limited to specific languages, domains, or locations, or relies on single-model approaches, reducing their applicability to diverse real-world scenarios. To address these gaps, our study focuses on multi-lingual tweet cyber threat detection using a variety of advanced models. The research was conducted in three stages: (1) We collected and labeled tweet datasets in four languages English, Chinese, Russian, and Arabic employing both manual and polarity-based labeling methods to ensure high-quality annotations. (2) Each dataset was analyzed individually using machine learning (ML) and deep learning (DL) models to assess their performance on distinct languages. (3) Finally, we combined all four datasets into a single multi-lingual dataset and applied DL and large language model (LLM) architectures to evaluate their efficacy in identifying cyber threats across various languages. Our results show that among machine learning models, Random Forest (RF) attained the highest performance; however, the Bi-LSTM architecture consistently surpassed other DL and LLM architectures across all datasets. These findings underline the effectiveness of Bi-LSTM in multilingual cyber threat detection. The code for this paper can be found at this link: https://github.com/Mmurrad/Tweet-Data-Classification.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01100v1">ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Website: https://huggingface.co/spaces/WildEval/ZebraLogic
    </div>
    <details class="paper-abstract">
      We investigate the logical reasoning capabilities of large language models (LLMs) and their scalability in complex non-monotonic reasoning. To this end, we introduce ZebraLogic, a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs). ZebraLogic enables the generation of puzzles with controllable and quantifiable complexity, facilitating a systematic study of the scaling limits of models such as Llama, o1 models, and DeepSeek-R1. By encompassing a broad range of search space complexities and diverse logical constraints, ZebraLogic provides a structured environment to evaluate reasoning under increasing difficulty. Our results reveal a significant decline in accuracy as problem complexity grows -- a phenomenon we term the curse of complexity. This limitation persists even with larger models and increased inference-time computation, suggesting inherent constraints in current LLM reasoning capabilities. Additionally, we explore strategies to enhance logical reasoning, including Best-of-N sampling, backtracking mechanisms, and self-verification prompts. Our findings offer critical insights into the scalability of LLM reasoning, highlight fundamental limitations, and outline potential directions for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01083v1">Tool Unlearning for Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 https://clu-uml.github.io/MU-Bench-Project-Page/
    </div>
    <details class="paper-abstract">
      Tool-augmented large language models (LLMs) are often trained on datasets of query-response pairs, which embed the ability to use tools or APIs directly into the parametric knowledge of LLMs. Tool-augmented LLMs need the ability to forget learned tools due to security vulnerabilities, privacy regulations, or tool deprecations. However, ``tool unlearning'' has not been investigated in unlearning literature. We introduce this novel task, which requires addressing distinct challenges compared to traditional unlearning: knowledge removal rather than forgetting individual samples, the high cost of optimizing LLMs, and the need for principled evaluation metrics. To bridge these gaps, we propose ToolDelete, the first approach for unlearning tools from tool-augmented LLMs. It implements three key properties to address the above challenges for effective tool unlearning and introduces a new membership inference attack (MIA) model for effective evaluation. Extensive experiments on multiple tool learning datasets and tool-augmented LLMs show that ToolDelete effectively unlearns randomly selected tools, while preserving the LLM's knowledge on non-deleted tools and maintaining performance on general tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01070v1">An Investigation of FP8 Across Accelerators for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      The introduction of 8-bit floating-point (FP8) computation units in modern AI accelerators has generated significant interest in FP8-based large language model (LLM) inference. Unlike 16-bit floating-point formats, FP8 in deep learning requires a shared scaling factor. Additionally, while E4M3 and E5M2 are well-defined at the individual value level, their scaling and accumulation methods remain unspecified and vary across hardware and software implementations. As a result, FP8 behaves more like a quantization format than a standard numeric representation. In this work, we provide the first comprehensive analysis of FP8 computation and acceleration on two AI accelerators: the NVIDIA H100 and Intel Gaudi 2. Our findings highlight that the Gaudi 2, by leveraging FP8, achieves higher throughput-to-power efficiency during LLM inference, offering valuable insights into the practical implications of FP8 adoption for datacenter-scale LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00988v1">PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Scientific data visualization is pivotal for transforming raw data into comprehensible visual representations, enabling pattern recognition, forecasting, and the presentation of data-driven insights. However, novice users often face difficulties due to the complexity of selecting appropriate tools and mastering visualization techniques. Large Language Models (LLMs) have recently demonstrated potential in assisting code generation, though they struggle with accuracy and require iterative debugging. In this paper, we propose PlotGen, a novel multi-agent framework aimed at automating the creation of precise scientific visualizations. PlotGen orchestrates multiple LLM-based agents, including a Query Planning Agent that breaks down complex user requests into executable steps, a Code Generation Agent that converts pseudocode into executable Python code, and three retrieval feedback agents - a Numeric Feedback Agent, a Lexical Feedback Agent, and a Visual Feedback Agent - that leverage multimodal LLMs to iteratively refine the data accuracy, textual labels, and visual correctness of generated plots via self-reflection. Extensive experiments show that PlotGen outperforms strong baselines, achieving a 4-6 percent improvement on the MatPlotBench dataset, leading to enhanced user trust in LLM-generated visualizations and improved novice productivity due to a reduction in debugging time needed for plot errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00963v1">PDE-Controller: LLMs for Autoformalization and Reasoning of PDEs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      While recent AI-for-math has made strides in pure mathematics, areas of applied mathematics, particularly PDEs, remain underexplored despite their significant real-world applications. We present PDE-Controller, a framework that enables large language models (LLMs) to control systems governed by partial differential equations (PDEs). Our approach enables LLMs to transform informal natural language instructions into formal specifications, and then execute reasoning and planning steps to improve the utility of PDE control. We build a holistic solution comprising datasets (both human-written cases and 2 million synthetic samples), math-reasoning models, and novel evaluation metrics, all of which require significant effort. Our PDE-Controller significantly outperforms prompting the latest open-source and GPT models in reasoning, autoformalization, and program synthesis, achieving up to a 62% improvement in utility gain for PDE control. By bridging the gap between language generation and PDE systems, we demonstrate the potential of LLMs in addressing complex scientific and engineering challenges. We will release all data, model checkpoints, and code at https://pde-controller.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18626v2">The TIP of the Iceberg: Revealing a Hidden Class of Task-In-Prompt Adversarial Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies. Warning: this paper contains examples of unethical inquiries used solely for research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07959v2">COMPL-AI Framework: A Technical Interpretation and LLM Benchmarking Suite for the EU Artificial Intelligence Act</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      The EU's Artificial Intelligence Act (AI Act) is a significant step towards responsible AI development, but lacks clear technical interpretation, making it difficult to assess models' compliance. This work presents COMPL-AI, a comprehensive framework consisting of (i) the first technical interpretation of the EU AI Act, translating its broad regulatory requirements into measurable technical requirements, with the focus on large language models (LLMs), and (ii) an open-source Act-centered benchmarking suite, based on thorough surveying and implementation of state-of-the-art LLM benchmarks. By evaluating 12 prominent LLMs in the context of COMPL-AI, we reveal shortcomings in existing models and benchmarks, particularly in areas like robustness, safety, diversity, and fairness. This work highlights the need for a shift in focus towards these aspects, encouraging balanced development of LLMs and more comprehensive regulation-aligned benchmarks. Simultaneously, COMPL-AI for the first time demonstrates the possibilities and difficulties of bringing the Act's obligations to a more concrete, technical level. As such, our work can serve as a useful first step towards having actionable recommendations for model providers, and contributes to ongoing efforts of the EU to enable application of the Act, such as the drafting of the GPAI Code of Practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03376v2">Log Parsing using LLMs with Self-Generated In-Context Learning and Self-Correction</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Accepted by the 33rd IEEE/ACM International Conference on Program Comprehension (ICPC'25)
    </div>
    <details class="paper-abstract">
      Log parsing transforms log messages into structured formats, serving as a crucial step for log analysis. Despite a variety of log parsers that have been proposed, their performance on evolving log data remains unsatisfactory due to reliance on human-crafted rules or learning-based models with limited training data. The recent emergence of large language models (LLMs) has demonstrated strong abilities in understanding natural language and code, making it promising to apply LLMs for log parsing. Consequently, several studies have proposed LLM-based log parsers. However, LLMs may produce inaccurate templates, and existing LLM-based log parsers directly use the template generated by the LLM as the parsing result, hindering the accuracy of log parsing. Furthermore, these log parsers depend heavily on historical log data as demonstrations, which poses challenges in maintaining accuracy when dealing with scarce historical log data or evolving log data. To address these challenges, we propose AdaParser, an effective and adaptive log parsing framework using LLMs with self-generated in-context learning (SG-ICL) and self-correction. To facilitate accurate log parsing, AdaParser incorporates a novel component, a template corrector, which utilizes the LLM to correct potential parsing errors in the templates it generates. In addition, AdaParser maintains a dynamic candidate set composed of previously generated templates as demonstrations to adapt evolving log data. Extensive experiments on public large-scale datasets indicate that AdaParser outperforms state-of-the-art methods across all metrics, even in zero-shot scenarios. Moreover, when integrated with different LLMs, AdaParser consistently enhances the performance of the utilized LLMs by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14744v3">Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to face hallucination issues due to the data they trained on often containing human bias; whether this is reflected in the decision-making process of LLM Agents remains under-explored. As LLM Agents are increasingly employed in intricate social environments, a pressing and natural question emerges: Can we utilize LLM Agents' systematic hallucinations to mirror human cognitive biases, thus exhibiting irrational social intelligence? In this paper, we probe the irrational behavior among contemporary LLM Agents by melding practical social science experiments with theoretical insights. Specifically, We propose CogMir, an open-ended Multi-LLM Agents framework that utilizes hallucination properties to assess and enhance LLM Agents' social intelligence through cognitive biases. Experimental results on CogMir subsets show that LLM Agents and humans exhibit high consistency in irrational and prosocial decision-making under uncertain conditions, underscoring the prosociality of LLM Agents as social entities and highlighting the significance of hallucination properties. Additionally, the CogMir framework demonstrates its potential as a valuable platform for encouraging more research into the social intelligence of LLM Agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14569v3">When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 20 pages, To appear in Usenix Security 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, generated impersonation posts where 93.9% of them were deemed authentic, and boosted click rate of phishing links in spear phishing emails by 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for robust security measures to prevent the misuse of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18712v2">Invisible Traces: Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Fingerprinting refers to the process of identifying underlying Machine Learning (ML) models of AI Systemts, such as Large Language Models (LLMs), by analyzing their unique characteristics or patterns, much like a human fingerprint. The fingerprinting of Large Language Models (LLMs) has become essential for ensuring the security and transparency of AI-integrated applications. While existing methods primarily rely on access to direct interactions with the application to infer model identity, they often fail in real-world scenarios involving multi-agent systems, frequent model updates, and restricted access to model internals. In this paper, we introduce a novel fingerprinting framework designed to address these challenges by integrating static and dynamic fingerprinting techniques. Our approach identifies architectural features and behavioral traits, enabling accurate and robust fingerprinting of LLMs in dynamic environments. We also highlight new threat scenarios where traditional fingerprinting methods are ineffective, bridging the gap between theoretical techniques and practical application. To validate our framework, we present an extensive evaluation setup that simulates real-world conditions and demonstrate the effectiveness of our methods in identifying and monitoring LLMs in Gen-AI applications. Our results highlight the framework's adaptability to diverse and evolving deployment contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12433v4">LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Reranking is a critical component in recommender systems, playing an essential role in refining the output of recommendation algorithms. Traditional reranking models have focused predominantly on accuracy, but modern applications demand consideration of additional criteria such as diversity and fairness. Existing reranking approaches often fail to harmonize these diverse criteria effectively at the model level. Moreover, these models frequently encounter challenges with scalability and personalization due to their complexity and the varying significance of different reranking criteria in diverse scenarios. In response, we introduce a comprehensive reranking framework enhanced by LLM, designed to seamlessly integrate various reranking criteria while maintaining scalability and facilitating personalized recommendations. This framework employs a fully connected graph structure, allowing the LLM to simultaneously consider multiple aspects such as accuracy, diversity, and fairness through a coherent Chain-of-Thought (CoT) process. A customizable input mechanism is also integrated, enabling the tuning of the language model's focus to meet specific reranking needs. We validate our approach using three popular public datasets, where our framework demonstrates superior performance over existing state-of-the-art reranking models in balancing multiple criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01853v1">Security and Quality in LLM-Generated Code: A Multi-Language, Multi-Model Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 12 pages, 10 tables. In submission to IEEE Transactions on Dependable and Secure Computing
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI)-driven code generation tools are increasingly used throughout the software development lifecycle to accelerate coding tasks. However, the security of AI-generated code using Large Language Models (LLMs) remains underexplored, with studies revealing various risks and weaknesses. This paper analyzes the security of code generated by LLMs across different programming languages. We introduce a dataset of 200 tasks grouped into six categories to evaluate the performance of LLMs in generating secure and maintainable code. Our research shows that while LLMs can automate code creation, their security effectiveness varies by language. Many models fail to utilize modern security features in recent compiler and toolkit updates, such as Java 17. Moreover, outdated methods are still commonly used, particularly in C++. This highlights the need for advancing LLMs to enhance security and quality while incorporating emerging best practices in programming languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01827v1">Relatively-Secure LLM-Based Steganography via Constrained Markov Decision Processes</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Linguistic steganography aims to conceal information within natural language text without being detected. An effective steganography approach should encode the secret message into a minimal number of language tokens while preserving the natural appearance and fluidity of the stego-texts. We present a new framework to enhance the embedding efficiency of stego-texts generated by modifying the output of a large language model (LLM). The novelty of our approach is in abstracting the sequential steganographic embedding process as a Constrained Markov Decision Process (CMDP), which takes into consideration the long-term dependencies instead of merely the immediate effects. We constrain the solution space such that the discounted accumulative total variation divergence between the selected probability distribution and the original distribution given by the LLM is below a threshold. To find the optimal policy, we first show that the functional optimization problem can be simplified to a convex optimization problem with a finite number of variables. A closed-form solution for the optimal policy is then presented to this equivalent problem. It is remarkable that the optimal policy is deterministic and resembles water-filling in some cases. The solution suggests that usually adjusting the probability distribution for the state that has the least random transition probability should be prioritized, but the choice should be made by taking into account the transition probabilities at all states instead of only the current state.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01822v1">Firewalls to Secure Dynamic LLM Agentic Networks</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Future LLM agents are likely to communicate on behalf of users with other entity-representing agents on tasks that entail long-horizon plans with interdependent goals. Current work does not focus on such agentic networks, nor does it address their challenges. Thus, we first identify the required properties of agents' communication, which should be proactive and adaptable. It needs to satisfy 1) privacy: agents should not share more than what is needed for the task, and 2) security: the communication must preserve integrity and maintain utility against selfish entities. We design a use case (travel planning) as a testbed that exemplifies these requirements, and we show examples of how this can go wrong. Next, we propose a practical design, inspired by established network security principles, for constrained LLM agentic networks that balance adaptability, security, and privacy. Our framework automatically constructs and updates task-specific rules from prior simulations to build firewalls. We offer layers of defense to 1) convert free-form input to a task-specific protocol, 2) dynamically abstract users' data to a task-specific degree of permissiveness, and 3) self-correct the agents' trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01801v1">MemPal: Leveraging Multimodal AI and LLMs for Voice-Activated Object Retrieval in Homes of Older Adults</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Older adults have increasing difficulty with retrospective memory, hindering their abilities to perform daily activities and posing stress on caregivers to ensure their wellbeing. Recent developments in Artificial Intelligence (AI) and large context-aware multimodal models offer an opportunity to create memory support systems that assist older adults with common issues like object finding. This paper discusses the development of an AI-based, wearable memory assistant, MemPal, that helps older adults with a common problem, finding lost objects at home, and presents results from tests of the system in older adults' own homes. Using visual context from a wearable camera, the multimodal LLM system creates a real-time automated text diary of the person's activities for memory support purposes, offering object retrieval assistance using a voice-based interface. The system is designed to support additional use cases like context-based proactive safety reminders and recall of past actions. We report on a quantitative and qualitative study with N=15 older adults within their own homes that showed improved performance of object finding with audio-based assistance compared to no aid and positive overall user perceptions on the designed system. We discuss further applications of MemPal's design as a multi-purpose memory aid and future design guidelines to adapt memory assistants to older adults' unique needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11709v2">Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential in software development, especially for issue resolution. However, despite their widespread use, significant challenges persist in the quality of LLM responses to issue resolution queries. LLM interactions often yield incorrect, incomplete, or ambiguous information, largely due to knowledge gaps in prompt design, which can lead to unproductive exchanges and reduced developer productivity. In this paper, we analyze 433 developer-ChatGPT conversations within GitHub issue threads to examine the impact of prompt knowledge gaps and conversation styles on issue resolution. We identify four main knowledge gaps in developer prompts: Missing Context, Missing Specifications, Multiple Context, and Unclear Instructions. Assuming that conversations within closed issues contributed to successful resolutions while those in open issues did not, we find that ineffective conversations contain knowledge gaps in 44.6% of prompts, compared to only 12.6% in effective ones. Additionally, we observe seven distinct conversational styles, with Directive Prompting, Chain of Thought, and Responsive Feedback being the most prevalent. We find that knowledge gaps are present in all styles of conversations, with Missing Context being the most repeated challenge developers face in issue-resolution conversations. Based on our analysis, we identify key textual and code-related heuristics (Specificity, Contextual Richness, and Clarity) that are associated with successful issue closure and help assess prompt quality. These heuristics lay the foundation for an automated tool that can dynamically flag unclear prompts and suggest structured improvements. To test feasibility, we developed a lightweight browser extension prototype for detecting prompt gaps, that can be easily adapted to other tools within developer workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01630v1">TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Temporal reasoning in multi-session dialogues presents a significant challenge which has been under-studied in previous temporal reasoning benchmarks. To bridge this gap, we propose a new evaluation task for temporal reasoning in multi-session dialogues and introduce an approach to construct a new benchmark by augmenting dialogues from LoCoMo and creating multi-choice QAs. Furthermore, we present TReMu, a new framework aimed at enhancing the temporal reasoning capabilities of LLM-agents in this context. Specifically, the framework employs \textit{time-aware memorization} through timeline summarization, generating retrievable memory by summarizing events in each dialogue session with their inferred dates. Additionally, we integrate \textit{neuro-symbolic temporal reasoning}, where LLMs generate Python code to perform temporal calculations and select answers. Experimental evaluations on popular LLMs demonstrate that our benchmark is challenging, and the proposed framework significantly improves temporal reasoning performance compared to baseline methods, raising from 29.83 on GPT-4o via standard prompting to 77.67 via our approach and highlighting its effectiveness in addressing temporal reasoning in multi-session dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10590v2">LLM-Mediated Domain-Specific Voice Agents: The Case of TextileBot</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Published at Behaviour and Information Technology Journal
    </div>
    <details class="paper-abstract">
      Developing domain-specific conversational agents (CAs) has been challenged by the need for extensive domain-focused data. Recent advancements in Large Language Models (LLMs) make them a viable option as a knowledge backbone. LLMs behaviour can be enhanced through prompting, instructing them to perform downstream tasks in a zero-shot fashion (i.e. without training). To this end, we incorporated structural knowledge into prompts and used prompted LLMs to prototyping domain-specific CAs. We demonstrate a case study in a specific domain-textile circularity - TextileBot, we present the design, development, and evaluation of the TextileBot. Specially, we conducted an in-person user study (N=30) with Free Chat and Information-Gathering tasks with TextileBots to gather insights from the interaction. We analyse the human-agent interactions, combining quantitative and qualitative methods. Our results suggest that participants engaged in multi-turn conversations, and their perceptions of the three variation agents and respective interactions varied demonstrating the effectiveness of our prompt-based LLM approach. We discuss the dynamics of these interactions and their implications for designing future voice-based CAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01620v1">LLM-TA: An LLM-Enhanced Thematic Analysis Pipeline for Transcripts from Parents of Children with Congenital Heart Disease</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Accepted by GenAI for Health Workshop @ AAAI 2025, Philadelphia
    </div>
    <details class="paper-abstract">
      Thematic Analysis (TA) is a fundamental method in healthcare research for analyzing transcript data, but it is resource-intensive and difficult to scale for large, complex datasets. This study investigates the potential of large language models (LLMs) to augment the inductive TA process in high-stakes healthcare settings. Focusing on interview transcripts from parents of children with Anomalous Aortic Origin of a Coronary Artery (AAOCA), a rare congenital heart disease, we propose an LLM-Enhanced Thematic Analysis (LLM-TA) pipeline. Our pipeline integrates an affordable state-of-the-art LLM (GPT-4o mini), LangChain, and prompt engineering with chunking techniques to analyze nine detailed transcripts following the inductive TA framework. We evaluate the LLM-generated themes against human-generated results using thematic similarity metrics, LLM-assisted assessments, and expert reviews. Results demonstrate that our pipeline outperforms existing LLM-assisted TA methods significantly. While the pipeline alone has not yet reached human-level quality in inductive TA, it shows great potential to improve scalability, efficiency, and accuracy while reducing analyst workload when working collaboratively with domain experts. We provide practical recommendations for incorporating LLMs into high-stakes TA workflows and emphasize the importance of close collaboration with domain experts to address challenges related to real-world applicability and dataset complexity. https://github.com/jiaweixu98/LLM-TA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01586v1">SubTrack your Grad: Gradient Subspace Tracking for Memory and Time Efficient Full-Parameter LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) demand significant time and computational resources due to their large model sizes and optimizer states. To overcome these challenges, recent methods, such as BAdam, employ partial weight updates to enhance time and memory efficiency, though sometimes at the cost of performance. Others, like GaLore, focus on maintaining performance while optimizing memory usage through full parameter training, but may incur higher time complexity. By leveraging the low-rank structure of the gradient and the Grassmannian geometry, we propose SubTrack-Grad, a subspace tracking-based optimization method that efficiently tracks the evolving gradient subspace by incorporating estimation errors and previously identified subspaces. SubTrack-Grad delivers better or on-par results compared to GaLore, while significantly outperforming BAdam, which, despite being time-efficient, compromises performance. SubTrack-Grad reduces wall-time by up to 20.57% on GLUE tasks (15% average reduction) and up to 65% on SuperGLUE tasks (22% average reduction) compared to GaLore. Notably, for a 3B parameter model, GaLore incurred a substantial 157% increase in wall-time compared to full-rank training, whereas SubTrack-Grad exhibited a 31% increase, representing a 49% reduction in wall-time, while enjoying the same memory reductions as GaLore.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01573v1">Next Steps in LLM-Supported Java Verification</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Accepted to NSE 2025, 1st International Workshop on Neuro-Symbolic Software Engineering (ICSE Workshop), 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Recent work has shown that Large Language Models (LLMs) are not only a suitable tool for code generation but also capable of generating annotation-based code specifications. Scaling these methodologies may allow us to deduce provable correctness guarantees for large-scale software systems. In comparison to other LLM tasks, the application field of deductive verification has the notable advantage of providing a rigorous toolset to check LLM-generated solutions. This short paper provides early results on how this rigorous toolset can be used to reliably elicit correct specification annotations from an unreliable LLM oracle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01564v1">MeetMap: Real-Time Collaborative Dialogue Mapping with LLMs in Online Meetings</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 CSCW2025 Accepted
    </div>
    <details class="paper-abstract">
      Video meeting platforms display conversations linearly through transcripts or summaries. However, ideas during a meeting do not emerge linearly. We leverage LLMs to create dialogue maps in real time to help people visually structure and connect ideas. Balancing the need to reduce the cognitive load on users during the conversation while giving them sufficient control when using AI, we explore two system variants that encompass different levels of AI assistance. In Human-Map, AI generates summaries of conversations as nodes, and users create dialogue maps with the nodes. In AI-Map, AI produces dialogue maps where users can make edits. We ran a within-subject experiment with ten pairs of users, comparing the two MeetMap variants and a baseline. Users preferred MeetMap over traditional methods for taking notes, which aligned better with their mental models of conversations. Users liked the ease of use for AI-Map due to the low effort demands and appreciated the hands-on opportunity in Human-Map for sense-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01534v1">Preference Leakage: A Contamination Problem in LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) as judges and LLM-based data synthesis have emerged as two fundamental LLM-driven data annotation methods in model development. While their combination significantly enhances the efficiency of model training and evaluation, little attention has been given to the potential contamination brought by this new model development paradigm. In this work, we expose preference leakage, a contamination problem in LLM-as-a-judge caused by the relatedness between the synthetic data generators and LLM-based evaluators. To study this issue, we first define three common relatednesses between data generator LLM and judge LLM: being the same model, having an inheritance relationship, and belonging to the same model family. Through extensive experiments, we empirically confirm the bias of judges towards their related student models caused by preference leakage across multiple LLM baselines and benchmarks. Further analysis suggests that preference leakage is a pervasive issue that is harder to detect compared to previously identified biases in LLM-as-a-judge scenarios. All of these findings imply that preference leakage is a widespread and challenging problem in the area of LLM-as-a-judge. We release all codes and data at: https://github.com/David-Li0406/Preference-Leakage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01477v1">Position: Empowering Time Series Reasoning with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Understanding time series data is crucial for multiple real-world applications. While large language models (LLMs) show promise in time series tasks, current approaches often rely on numerical data alone, overlooking the multimodal nature of time-dependent information, such as textual descriptions, visual data, and audio signals. Moreover, these methods underutilize LLMs' reasoning capabilities, limiting the analysis to surface-level interpretations instead of deeper temporal and multimodal reasoning. In this position paper, we argue that multimodal LLMs (MLLMs) can enable more powerful and flexible reasoning for time series analysis, enhancing decision-making and real-world applications. We call on researchers and practitioners to leverage this potential by developing strategies that prioritize trust, interpretability, and robust reasoning in MLLMs. Lastly, we highlight key research directions, including novel reasoning paradigms, architectural innovations, and domain-specific applications, to advance time series reasoning with MLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01714v1">Position: Towards a Responsible LLM-empowered Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      The rise of Agent AI and Large Language Model-powered Multi-Agent Systems (LLM-MAS) has underscored the need for responsible and dependable system operation. Tools like LangChain and Retrieval-Augmented Generation have expanded LLM capabilities, enabling deeper integration into MAS through enhanced knowledge retrieval and reasoning. However, these advancements introduce critical challenges: LLM agents exhibit inherent unpredictability, and uncertainties in their outputs can compound across interactions, threatening system stability. To address these risks, a human-centered design approach with active dynamic moderation is essential. Such an approach enhances traditional passive oversight by facilitating coherent inter-agent communication and effective system governance, allowing MAS to achieve desired outcomes more efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18645v2">Layered Chain-of-Thought Prompting for Multi-Agent LLM Systems: A Comprehensive Approach to Explainable Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) leverage chain-of-thought (CoT) prompting to provide step-by-step rationales, improving performance on complex tasks. Despite its benefits, vanilla CoT often fails to fully verify intermediate inferences and can produce misleading explanations. In this work, we propose Layered Chain-of-Thought (Layered-CoT) Prompting, a novel framework that systematically segments the reasoning process into multiple layers, each subjected to external checks and optional user feedback. We expand on the key concepts, present three scenarios -- medical triage, financial risk assessment, and agile engineering -- and demonstrate how Layered-CoT surpasses vanilla CoT in terms of transparency, correctness, and user engagement. By integrating references from recent arXiv papers on interactive explainability, multi-agent frameworks, and agent-based collaboration, we illustrate how Layered-CoT paves the way for more reliable and grounded explanations in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01450v1">Simulating Rumor Spreading in Social Networks using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      With the rise of social media, misinformation has become increasingly prevalent, fueled largely by the spread of rumors. This study explores the use of Large Language Model (LLM) agents within a novel framework to simulate and analyze the dynamics of rumor propagation across social networks. To this end, we design a variety of LLM-based agent types and construct four distinct network structures to conduct these simulations. Our framework assesses the effectiveness of different network constructions and agent behaviors in influencing the spread of rumors. Our results demonstrate that the framework can simulate rumor spreading across more than one hundred agents in various networks with thousands of edges. The evaluations indicate that network structure, personas, and spreading schemes can significantly influence rumor dissemination, ranging from no spread to affecting 83\% of agents in iterations, thereby offering a realistic simulation of rumor spread in social networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01390v1">Plan-Then-Execute: An Empirical Study of User Trust and Team Performance When Using LLM Agents As A Daily Assistant</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 conditionally accepted to CHI 2025
    </div>
    <details class="paper-abstract">
      Since the explosion in popularity of ChatGPT, large language models (LLMs) have continued to impact our everyday lives. Equipped with external tools that are designed for a specific purpose (e.g., for flight booking or an alarm clock), LLM agents exercise an increasing capability to assist humans in their daily work. Although LLM agents have shown a promising blueprint as daily assistants, there is a limited understanding of how they can provide daily assistance based on planning and sequential decision making capabilities. We draw inspiration from recent work that has highlighted the value of 'LLM-modulo' setups in conjunction with humans-in-the-loop for planning tasks. We conducted an empirical study (N = 248) of LLM agents as daily assistants in six commonly occurring tasks with different levels of risk typically associated with them (e.g., flight ticket booking and credit card payments). To ensure user agency and control over the LLM agent, we adopted LLM agents in a plan-then-execute manner, wherein the agents conducted step-wise planning and step-by-step execution in a simulation environment. We analyzed how user involvement at each stage affects their trust and collaborative team performance. Our findings demonstrate that LLM agents can be a double-edged sword -- (1) they can work well when a high-quality plan and necessary user involvement in execution are available, and (2) users can easily mistrust the LLM agents with plans that seem plausible. We synthesized key insights for using LLM agents as daily assistants to calibrate user trust and achieve better overall task outcomes. Our work has important implications for the future design of daily assistants and human-AI collaboration with LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01387v1">TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Although Deep Reinforcement Learning (DRL) and Large Language Models (LLMs) each show promise in addressing decision-making challenges in autonomous driving, DRL often suffers from high sample complexity, while LLMs have difficulty ensuring real-time decision making. To address these limitations, we propose TeLL-Drive, a hybrid framework that integrates an Teacher LLM to guide an attention-based Student DRL policy. By incorporating risk metrics, historical scenario retrieval, and domain heuristics into context-rich prompts, the LLM produces high-level driving strategies through chain-of-thought reasoning. A self-attention mechanism then fuses these strategies with the DRL agent's exploration, accelerating policy convergence and boosting robustness across diverse driving conditions. Our experimental results, evaluated across multiple traffic scenarios, show that TeLL-Drive outperforms existing baseline methods, including other LLM-based approaches, in terms of success rates, average returns, and real-time feasibility. Ablation studies underscore the importance of each model component, especially the synergy between the attention mechanism and LLM-driven guidance. These findings suggest that TeLL-Drive significantly enhances both the adaptability and safety of autonomous driving systems, while offering a more efficient and scalable approach for policy learning. Full validation results are available on our website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01349v1">Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized product recommendation systems, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making these adversarial manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive experiments on LLMs of varying scales, we reveal significant vulnerabilities in their use as recommenders, providing critical insights into safeguarding these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01705v1">Progressive Binarization with Semi-Structured Pruning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing tasks, but their high computational and memory demands pose challenges for deployment on resource-constrained devices. Binarization, as an efficient compression method that reduces model weights to just 1 bit, significantly lowers both computational and memory requirements. Despite this, the binarized LLM still contains redundancy, which can be further compressed. Semi-structured pruning provides a promising approach to achieve this, which offers a better trade-off between model performance and hardware efficiency. However, simply combining binarization with semi-structured pruning can lead to a significant performance drop. To address this issue, we propose a Progressive Binarization with Semi-Structured Pruning (PBS$^2$P) method for LLM compression. We first propose a Stepwise semi-structured Pruning with Binarization Optimization (SPBO). Our optimization strategy significantly reduces the total error caused by pruning and binarization, even below that of the no-pruning scenario. Furthermore, we design a Coarse-to-Fine Search (CFS) method to select pruning elements more effectively. Extensive experiments demonstrate that PBS$^2$P achieves superior accuracy across various LLM families and evaluation metrics, noticeably outperforming state-of-the-art (SOTA) binary PTQ methods. The code and models will be available at https://github.com/XIANGLONGYAN/PBS2P.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01298v1">Augmented Knowledge Graph Querying leveraging LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Adopting Knowledge Graphs (KGs) as a structured, semantic-oriented, data representation model has significantly improved data integration, reasoning, and querying capabilities across different domains. This is especially true in modern scenarios such as Industry 5.0, in which the integration of data produced by humans, smart devices, and production processes plays a crucial role. However, the management, retrieval, and visualization of data from a KG using formal query languages can be difficult for non-expert users due to their technical complexity, thus limiting their usage inside industrial environments. For this reason, we introduce SparqLLM, a framework that utilizes a Retrieval-Augmented Generation (RAG) solution, to enhance the querying of Knowledge Graphs (KGs). SparqLLM executes the Extract, Transform, and Load (ETL) pipeline to construct KGs from raw data. It also features a natural language interface powered by Large Language Models (LLMs) to enable automatic SPARQL query generation. By integrating template-based methods as retrieved-context for the LLM, SparqLLM enhances query reliability and reduces semantic errors, ensuring more accurate and efficient KG interactions. Moreover, to improve usability, the system incorporates a dynamic visualization dashboard that adapts to the structure of the retrieved data, presenting the query results in an intuitive format. Rigorous experimental evaluations demonstrate that SparqLLM achieves high query accuracy, improved robustness, and user-friendly interaction with KGs, establishing it as a scalable solution to access semantic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01273v1">Analysis of Student-LLM Interaction in a Software Engineering Project</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly competent across various domains, educators are showing a growing interest in integrating these LLMs into the learning process. Especially in software engineering, LLMs have demonstrated qualitatively better capabilities in code summarization, code generation, and debugging. Despite various research on LLMs for software engineering tasks in practice, limited research captures the benefits of LLMs for pedagogical advancements and their impact on the student learning process. To this extent, we analyze 126 undergraduate students' interaction with an AI assistant during a 13-week semester to understand the benefits of AI for software engineering learning. We analyze the conversations, code generated, code utilized, and the human intervention levels to integrate the code into the code base. Our findings suggest that students prefer ChatGPT over CoPilot. Our analysis also finds that ChatGPT generates responses with lower computational complexity compared to CoPilot. Furthermore, conversational-based interaction helps improve the quality of the code generated compared to auto-generated code. Early adoption of LLMs in software engineering is crucial to remain competitive in the rapidly developing landscape. Hence, the next generation of software engineers must acquire the necessary skills to interact with AI to improve productivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01205v1">OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 To be published in RESOURCEFUL 2025
    </div>
    <details class="paper-abstract">
      Optical Character Recognition (OCR) systems often introduce errors when transcribing historical documents, leaving room for post-correction to improve text quality. This study evaluates the use of open-weight LLMs for OCR error correction in historical English and Finnish datasets. We explore various strategies, including parameter optimization, quantization, segment length effects, and text continuation methods. Our results demonstrate that while modern LLMs show promise in reducing character error rates (CER) in English, a practically useful performance for Finnish was not reached. Our findings highlight the potential and limitations of LLMs in scaling OCR post-correction for large historical corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01116v1">Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as powerful tools for addressing a wide range of general inquiries and tasks. Despite this, fine-tuning aligned LLMs on smaller, domain-specific datasets, critical to adapting them to specialized tasks, can inadvertently degrade their safety alignment, even when the datasets are benign. This phenomenon makes models more susceptible to providing inappropriate responses. In this study, we systematically examine the factors contributing to safety alignment degradation in benign fine-tuning scenarios. Our analysis identifies three critical factors affecting aligned LLMs: answer structure, identity calibration, and role-play. Additionally, we evaluate the reliability of state-of-the-art reward models (RMs), which are often used to guide alignment processes. Our findings reveal that these RMs frequently fail to accurately reflect human preferences regarding safety, underscoring their limitations in practical applications. By uncovering these challenges, our work highlights the complexities of maintaining safety alignment during fine-tuning and offers guidance to help developers balance utility and safety in LLMs. Datasets and fine-tuning code used in our experiments can be found in https://github.com/GuanlinLee/llm_instruction_tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14931v2">Do LLMs Dream of Ontologies?</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across diverse natural language processing tasks, yet their ability to memorize structured knowledge remains underexplored. In this paper, we investigate the extent to which general-purpose pre-trained LLMs retain and correctly reproduce concept identifier (ID)-label associations from publicly available ontologies. We conduct a systematic evaluation across multiple ontological resources, including the Gene Ontology, Uberon, Wikidata, and ICD-10, using LLMs such as Pythia-12B, Gemini-1.5-Flash, GPT-3.5, and GPT-4. Our findings reveal that only a small fraction of ontological concepts is accurately memorized, with GPT-4 demonstrating the highest performance. To understand why certain concepts are memorized more effectively than others, we analyze the relationship between memorization accuracy and concept popularity on the Web. Our results indicate a strong correlation between the frequency of a concept's occurrence online and the likelihood of accurately retrieving its ID from the label. This suggests that LLMs primarily acquire such knowledge through indirect textual exposure rather than directly from structured ontological resources. Furthermore, we introduce new metrics to quantify prediction invariance, demonstrating that the stability of model responses across variations in prompt language and temperature settings can serve as a proxy for estimating memorization robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07927v2">Gandalf the Red: Adaptive Security for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally
    </div>
    <details class="paper-abstract">
      Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14427v2">GraphSOS: Graph Sampling and Order Selection to Help LLMs Understand Graphs Better</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The success of Large Language Models (LLMs) in various domains has led researchers to apply them to graph-related problems by converting graph data into natural language text. However, unlike graph data, natural language inherently has sequential order. We observe a counter-intuitive fact that when the order of nodes or edges in the natural language description of a graph is shuffled, despite describing the same graph, model performance fluctuates between high performance and random guessing. Additionally, due to LLMs' limited input context length, current methods typically randomly sample neighbors of target nodes as representatives of their neighborhood, which may not always be effective for accurate reasoning. To address these gaps, we introduce GraphSOS (Graph Sampling and Order Selection). This novel model framework features an Order Selector Module to ensure proper serialization order of the graph and a Subgraph Sampling Module to sample subgraphs with better structure for better reasoning. Furthermore, we propose Graph CoT obtained through distillation, and enhance LLM's reasoning and zero-shot learning capabilities for graph tasks through instruction tuning. Experiments on multiple datasets for node classification and graph question-answering demonstrate that GraphSOS improves LLMs' performance and generalization ability on graph tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16383v2">RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Key-Value (KV) cache facilitates efficient large language models (LLMs) inference by avoiding recomputation of past KVs. As the batch size and context length increase, the oversized KV caches become a significant memory bottleneck, highlighting the need for efficient compression. Existing KV quantization rely on fine-grained quantization or the retention of a significant portion of high bit-widths caches, both of which compromise compression ratio and often fail to maintain robustness at extremely low average bit-widths. In this work, we explore the potential of rotation technique for 2-bit KV quantization and propose RotateKV, which achieves accurate and robust performance through the following innovations: (i) Outlier-Aware Rotation, which utilizes channel-reordering to adapt the rotations to varying channel-wise outlier distributions without sacrificing the computational efficiency of the fast Walsh-Hadamard transform (FWHT); (ii) Pre-RoPE Grouped-Head Rotation, which mitigates the impact of rotary position embedding (RoPE) on proposed outlier-aware rotation and further smooths outliers across heads; (iii) Attention-Sink-Aware Quantization, which leverages the massive activations to precisely identify and protect attention sinks. RotateKV achieves less than 0.3 perplexity (PPL) degradation with 2-bit quantization on WikiText-2 using LLaMA-2-13B, maintains strong CoT reasoning and long-context capabilities, with less than 1.7\% degradation on GSM8K, outperforming existing methods even at lower average bit-widths. RotateKV also showcases a 3.97x reduction in peak memory usage, supports 5.75x larger batch sizes, and achieves a 2.32x speedup in decoding stage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07288v2">LLM-Net: Democratizing LLMs-as-a-Service through Blockchain-based Expert Networks</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The centralization of Large Language Models (LLMs) development has created significant barriers to AI advancement, limiting the democratization of these powerful technologies. This centralization, coupled with the scarcity of high-quality training data and mounting complexity of maintaining comprehensive expertise across rapidly expanding knowledge domains, poses critical challenges to the continued growth of LLMs. While solutions like Retrieval-Augmented Generation (RAG) offer potential remedies, maintaining up-to-date expert knowledge across diverse domains remains a significant challenge, particularly given the exponential growth of specialized information. This paper introduces LLMs Networks (LLM-Net), a blockchain-based framework that democratizes LLMs-as-a-Service through a decentralized network of specialized LLM providers. By leveraging collective computational resources and distributed domain expertise, LLM-Net incorporates fine-tuned expert models for various specific domains, ensuring sustained knowledge growth while maintaining service quality through collaborative prompting mechanisms. The framework's robust design includes blockchain technology for transparent transaction and performance validation, establishing an immutable record of service delivery. Our simulation, built on top of state-of-the-art LLMs such as Claude 3.5 Sonnet, Llama 3.1, Grok-2, and GPT-4o, validates the effectiveness of the reputation-based mechanism in maintaining service quality by selecting high-performing respondents (LLM providers). Thereby it demonstrates the potential of LLM-Net to sustain AI advancement through the integration of decentralized expertise and blockchain-based accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00922v1">Huff-LLM: End-to-End Lossless Compression for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      As they become more capable, large language models (LLMs) have continued to rapidly increase in size. This has exacerbated the difficulty in running state of the art LLMs on small, edge devices. Standard techniques advocate solving this problem through lossy compression techniques such as quantization or pruning. However, such compression techniques are lossy, and have been shown to change model behavior in unpredictable manners. We propose Huff-LLM, an \emph{end-to-end, lossless} model compression method that lets users store LLM weights in compressed format \emph{everywhere} -- cloud, disk, main memory, and even in on-chip memory/buffers. This allows us to not only load larger models in main memory, but also reduces bandwidth required to load weights on chip, and makes more efficient use of on-chip weight buffers. In addition to the memory savings achieved via compression, we also show latency and energy efficiency improvements when performing inference with the compressed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00916v1">The Accuracy, Robustness, and Readability of LLM-Generated Sustainability-Related Word Definitions</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 NLP4Ecology Workshop 2025
    </div>
    <details class="paper-abstract">
      A common language with standardized definitions is crucial for effective climate discussions. However, concerns exist about LLMs misrepresenting climate terms. We compared 300 official IPCC glossary definitions with those generated by GPT-4o-mini, Llama3.1 8B, and Mistral 7B, analyzing adherence, robustness, and readability using SBERT sentence embeddings. The LLMs scored an average adherence of $0.57-0.59 \pm 0.15$, and their definitions proved harder to read than the originals. Model-generated definitions vary mainly among words with multiple or ambiguous definitions, showing the potential to highlight terms that need standardization. The results show how LLMs could support environmental discourse while emphasizing the need to align model outputs with established terminology for clarity and consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00899v1">HASSLE-free: A unified Framework for Sparse plus Low-Rank Matrix Decomposition for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The impressive capabilities of large foundation models come at a cost of substantial computing resources to serve them. Compressing these pre-trained models is of practical interest as it can democratize deploying them to the machine learning community at large by lowering the costs associated with inference. A promising compression scheme is to decompose foundation models' dense weights into a sum of sparse plus low-rank matrices. In this paper, we design a unified framework coined HASSLE-free for (semi-structured) sparse plus low-rank matrix decomposition of foundation models. Our framework introduces the local layer-wise reconstruction error objective for this decomposition, we demonstrate that prior work solves a relaxation of this optimization problem; and we provide efficient and scalable methods to minimize the exact introduced optimization problem. HASSLE-free substantially outperforms state-of-the-art methods in terms of the introduced objective and a wide range of LLM evaluation benchmarks. For the Llama3-8B model with a 2:4 sparsity component plus a 64-rank component decomposition, a compression scheme for which recent work shows important inference acceleration on GPUs, HASSLE-free reduces the test perplexity by 12% for the WikiText-2 dataset and reduces the gap (compared to the dense model) of the average of eight popular zero-shot tasks by 15% compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00894v1">MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Tokenization is fundamental to Natural Language Processing (NLP), directly impacting model efficiency and linguistic fidelity. While Byte Pair Encoding (BPE) is widely used in Large Language Models (LLMs), it often disregards morpheme boundaries, leading to suboptimal segmentation, particularly in morphologically rich languages. We introduce MorphBPE, a morphology-aware extension of BPE that integrates linguistic structure into subword tokenization while preserving statistical efficiency. Additionally, we propose two morphology-based evaluation metrics: (i) Morphological Consistency F1-Score, which quantifies the consistency between morpheme sharing and token sharing, contributing to LLM training convergence, and (ii) Morphological Edit Distance, which measures alignment between morphemes and tokens concerning interpretability. Experiments on English, Russian, Hungarian, and Arabic across 300M and 1B parameter LLMs demonstrate that MorphBPE consistently reduces cross-entropy loss, accelerates convergence, and improves morphological alignment scores. Fully compatible with existing LLM pipelines, MorphBPE requires minimal modifications for integration. The MorphBPE codebase and tokenizer playground will be available at: https://github.com/llm-lab-org/MorphBPE and https://tokenizer.llm-lab.org
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00840v1">Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defense</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have showcased remarkable capabilities across various domains. Accompanying the evolving capabilities and expanding deployment scenarios of LLMs, their deployment challenges escalate due to their sheer scale and the advanced yet complex activation designs prevalent in notable model series, such as Llama, Gemma, and Mistral. These challenges have become particularly pronounced in resource-constrained deployment scenarios, where mitigating inference efficiency bottlenecks is imperative. Among various recent efforts, activation approximation has emerged as a promising avenue for pursuing inference efficiency, sometimes considered indispensable in applications such as private inference. Despite achieving substantial speedups with minimal impact on utility, even appearing sound and practical for real-world deployment, the safety implications of activation approximations remain unclear. In this work, we fill this critical gap in LLM safety by conducting the first systematic safety evaluation of activation approximations. Our safety vetting spans seven sota techniques across three popular categories, revealing consistent safety degradation across ten safety-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00829v1">A Comprehensive Analysis on LLM-based Node Classification Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Node classification is a fundamental task in graph analysis, with broad applications across various fields. Recent breakthroughs in Large Language Models (LLMs) have enabled LLM-based approaches for this task. Although many studies demonstrate the impressive performance of LLM-based methods, the lack of clear design guidelines may hinder their practical application. In this work, we aim to establish such guidelines through a fair and systematic comparison of these algorithms. As a first step, we developed LLMNodeBed, a comprehensive codebase and testbed for node classification using LLMs. It includes ten datasets, eight LLM-based algorithms, and three learning paradigms, and is designed for easy extension with new methods and datasets. Subsequently, we conducted extensive experiments, training and evaluating over 2,200 models, to determine the key settings (e.g., learning paradigms and homophily) and components (e.g., model size) that affect performance. Our findings uncover eight insights, e.g., (1) LLM-based methods can significantly outperform traditional methods in a semi-supervised setting, while the advantage is marginal in a supervised setting; (2) Graph Foundation Models can beat open-source LLMs but still fall short of strong LLMs like GPT-4o in a zero-shot setting. We hope that the release of LLMNodeBed, along with our insights, will facilitate reproducible research and inspire future studies in this field. Codes and datasets are released at \href{https://llmnodebed.github.io/}{https://llmnodebed.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00808v1">Synthetic Artifact Auditing: Tracing LLM-Generated Synthetic Data Usage in Downstream Applications</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 To Appear in the 34th USENIX Security Symposium, August 13-15, 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have facilitated the generation of high-quality, cost-effective synthetic data for developing downstream models and conducting statistical analyses in various domains. However, the increased reliance on synthetic data may pose potential negative impacts. Numerous studies have demonstrated that LLM-generated synthetic data can perpetuate and even amplify societal biases and stereotypes, and produce erroneous outputs known as ``hallucinations'' that deviate from factual knowledge. In this paper, we aim to audit artifacts, such as classifiers, generators, or statistical plots, to identify those trained on or derived from synthetic data and raise user awareness, thereby reducing unexpected consequences and risks in downstream applications. To this end, we take the first step to introduce synthetic artifact auditing to assess whether a given artifact is derived from LLM-generated synthetic data. We then propose an auditing framework with three methods including metric-based auditing, tuning-based auditing, and classification-based auditing. These methods operate without requiring the artifact owner to disclose proprietary training details. We evaluate our auditing framework on three text classification tasks, two text summarization tasks, and two data visualization tasks across three training scenarios. Our evaluation demonstrates the effectiveness of all proposed auditing methods across all these tasks. For instance, black-box metric-based auditing can achieve an average accuracy of $0.868 \pm 0.071$ for auditing classifiers and $0.880 \pm 0.052$ for auditing generators using only 200 random queries across three scenarios. We hope our research will enhance model transparency and regulatory compliance, ensuring the ethical and responsible use of synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00792v1">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Accepted by WWW 2025
    </div>
    <details class="paper-abstract">
      Real-Time Bidding (RTB) enables advertisers to place competitive bids on impression opportunities instantaneously, striving for cost-effectiveness in a highly competitive landscape. Although RTB has widely benefited from the utilization of technologies such as deep learning and reinforcement learning, the reliability of related methods often encounters challenges due to the discrepancies between online and offline environments and the rapid fluctuations of online bidding. To handle these challenges, RTBAgent is proposed as the first RTB agent system based on large language models (LLMs), which synchronizes real competitive advertising bidding environments and obtains bidding prices through an integrated decision-making process. Specifically, obtaining reasoning ability through LLMs, RTBAgent is further tailored to be more professional for RTB via involved auxiliary modules, i.e., click-through rate estimation model, expert strategy knowledge, and daily reflection. In addition, we propose a two-step decision-making process and multi-memory retrieval mechanism, which enables RTBAgent to review historical decisions and transaction records and subsequently make decisions more adaptive to market changes in real-time bidding. Empirical testing with real advertising datasets demonstrates that RTBAgent significantly enhances profitability. The RTBAgent code will be publicly accessible at: https://github.com/CaiLeng/RTBAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00735v1">From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00728v1">Meta-Prompt Optimization for LLM-Based Sequential Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently been employed as agents to solve sequential decision-making tasks such as Bayesian optimization and multi-armed bandits (MAB). These works usually adopt an LLM for sequential action selection by providing it with a fixed, manually designed meta-prompt. However, numerous previous works have found that the prompt has a significant impact on the performance of the LLM, which calls for a method to automatically optimize the meta-prompt for LLM-based agents. Unfortunately, the non-stationarity in the reward observations during LLM-based sequential decision-making makes meta-prompt optimization highly challenging. To address this challenge, we draw inspirations from adversarial bandit algorithms, which are inherently capable of handling non-stationary reward observations. Building on this foundation, we propose our EXPonential-weight algorithm for prompt Optimization} (EXPO) to automatically optimize the task description and meta-instruction in the meta-prompt for LLM-based agents. We also extend EXPO to additionally optimize the exemplars (i.e., history of interactions) in the meta-prompt to further enhance the performance, hence introducing our EXPO-ES algorithm. We use extensive experiments to show that our algorithms significantly improve the performance of LLM-based sequential decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00722v1">Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to increasingly diverse requests, accompanied with varying resource (compute and memory) demands to serve them. However, this in turn degrades the cost-efficiency of LLM serving as common practices primarily rely on homogeneous GPU resources. In response to this problem, this work conducts a thorough study about serving LLMs over heterogeneous GPU resources on cloud platforms. The rationale is that different GPU types exhibit distinct compute and memory characteristics, aligning well with the divergent resource demands of diverse requests. Particularly, through comprehensive benchmarking, we discover that the cost-efficiency of LLM serving can be substantially optimized by meticulously determining GPU composition, deployment configurations, and workload assignments. Subsequently, we design a scheduling algorithm via mixed-integer linear programming, aiming at deducing the most cost-efficient serving plan under the constraints of price budget and real-time GPU availability. Remarkably, our approach effectively outperforms homogeneous and heterogeneous baselines under a wide array of scenarios, covering diverse workload traces, varying GPU availablilities, and multi-model serving. This casts new light on more accessible and efficient LLM serving over heterogeneous cloud resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01683v1">LLM-Powered Benchmark Factory: Reliable, Generic, and Efficient</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has led to a surge in both model supply and application demands. To facilitate effective matching between them, reliable, generic and efficient benchmark generators are widely needed. However, human annotators are constrained by inefficiency, and current LLM benchmark generators not only lack generalizability but also struggle with limited reliability, as they lack a comprehensive evaluation framework for validation and optimization. To fill this gap, we first propose an automated and unbiased evaluation framework, structured around four dimensions and ten criteria. Under this framework, we carefully analyze the advantages and weaknesses of directly prompting LLMs as generic benchmark generators. To enhance the reliability, we introduce a series of methods to address the identified weaknesses and integrate them as BenchMaker. Experiments across multiple LLMs and tasks confirm that BenchMaker achieves superior or comparable performance to human-annotated benchmarks on all metrics, highlighting its generalizability and reliability. More importantly, it delivers highly consistent evaluation results across 12 LLMs (0.967 Pearson correlation against MMLU-Pro), while taking only $0.005 and 0.38 minutes per sample.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00689v1">Leveraging LLMs for Dynamic IoT Systems Generation through Mixed-Initiative Interaction</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      IoT systems face significant challenges in adapting to user needs, which are often under-specified and evolve with changing environmental contexts. To address these complexities, users should be able to explore possibilities, while IoT systems must learn and support users in the process of providing proper services, e.g., to serve novel experiences. The IoT-Together paradigm aims to meet this demand through the Mixed-Initiative Interaction (MII) paradigm that facilitates a collaborative synergy between users and IoT systems, enabling the co-creation of intelligent and adaptive solutions that are precisely aligned with user-defined goals. This work advances IoT-Together by integrating Large Language Models (LLMs) into its architecture. Our approach enables intelligent goal interpretation through a multi-pass dialogue framework and dynamic service generation at runtime according to user needs. To demonstrate the efficacy of our methodology, we design and implement the system in the context of a smart city tourism case study. We evaluate the system's performance using agent-based simulation and user studies. Results indicate efficient and accurate service identification and high adaptation quality. The empirical evidence indicates that the integration of Large Language Models (LLMs) into IoT architectures can significantly enhance the architectural adaptability of the system while ensuring real-world usability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00677v1">LLM-based event log analysis techniques: A survey</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Event log analysis is an important task that security professionals undertake. Event logs record key information on activities that occur on computing devices, and due to the substantial number of events generated, they consume a large amount of time and resources to analyse. This demanding and repetitive task is also prone to errors. To address these concerns, researchers have developed automated techniques to improve the event log analysis process. Large Language Models (LLMs) have recently demonstrated the ability to successfully perform a wide range of tasks that individuals would usually partake in, to high standards, and at a pace and degree of complexity that outperform humans. Due to this, researchers are rapidly investigating the use of LLMs for event log analysis. This includes fine-tuning, Retrieval-Augmented Generation (RAG) and in-context learning, which affect performance. These works demonstrate good progress, yet there is a need to understand the developing body of knowledge, identify commonalities between works, and identify key challenges and potential solutions to further developments in this domain. This paper aims to survey LLM-based event log analysis techniques, providing readers with an in-depth overview of the domain, gaps identified in previous research, and concluding with potential avenues to explore in future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00657v1">LLM Safety Alignment is Divergence Estimation in Disguise</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      We propose a theoretical framework demonstrating that popular Large Language Model (LLM) alignment methods, including Reinforcement Learning from Human Feedback (RLHF) and alternatives, fundamentally function as divergence estimators between aligned (preferred or safe) and unaligned (less-preferred or harmful) distributions. This explains the separation phenomenon between safe and harmful prompts in the model hidden representation after alignment. Inspired by the theoretical results, we identify that some alignment methods are better than others in terms of separation and, introduce a new method, KLDO, and further demonstrate the implication of our theories. We advocate for compliance-refusal datasets over preference datasets to enhance safety alignment, supported by both theoretical reasoning and empirical evidence. Additionally, to quantify safety separation, we leverage a distance metric in the representation space and statistically validate its efficacy as a statistical significant indicator of LLM resilience against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00602v1">Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance on various natural language tasks. However, they are trained on static corpora and their knowledge can become outdated quickly in the fast-changing world. This motivates the development of knowledge editing (KE) to update specific knowledge in LLMs without changing unrelated others or compromising their pre-trained capabilities. Previous efforts sought to update a small amount of parameters of a LLM and proved effective for making selective updates. Nonetheless, the edited LLM often exhibits degraded ability to reason about the new knowledge. In this work, we identify a key issue: heterogeneous token overfitting (HTO), where the LLM overfits different tokens in the provided knowledge at varying rates. To tackle this, we propose OVERTONE, a token-level smoothing method that mitigates HTO by adaptively refining the target distribution. Theoretically, OVERTONE offers better parameter updates with negligible computation overhead. It also induces an implicit DPO but does not require preference data pairs. Extensive experiments across four editing methods, two LLMs, and diverse scenarios demonstrate the effectiveness and versatility of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02626v3">Time-Reversal Provides Unsupervised Feedback to LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Accepted as a spotlight in NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12004v2">The Open Source Advantage in Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 9 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly advanced natural language processing, driving significant breakthroughs in tasks such as text generation, machine translation, and domain-specific reasoning. The field now faces a critical dilemma in its approach: closed-source models like GPT-4 deliver state-of-the-art performance but restrict reproducibility, accessibility, and external oversight, while open-source frameworks like LLaMA and Mixtral democratize access, foster collaboration, and support diverse applications, achieving competitive results through techniques like instruction tuning and LoRA. Hybrid approaches address challenges like bias mitigation and resource accessibility by combining the scalability of closed-source systems with the transparency and inclusivity of open-source framework. However, in this position paper, we argue that open-source remains the most robust path for advancing LLM research and ethical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05541v3">Customizable LLM-Powered Chatbot for Behavioral Science Research</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The rapid advancement of Artificial Intelligence has resulted in the advent of Large Language Models (LLMs) with the capacity to produce text that closely resembles human communication. These models have been seamlessly integrated into diverse applications, enabling interactive and responsive communication across multiple platforms. The potential utility of chatbots transcends these traditional applications, particularly in research contexts, wherein they can offer valuable insights and facilitate the design of innovative experiments. In this study, we present a Customizable LLM-Powered Chatbot (CLPC), a web-based chatbot system designed to assist in behavioral science research. The system is meticulously designed to function as an experimental instrument rather than a conventional chatbot, necessitating users to input a username and experiment code upon access. This setup facilitates precise data cross-referencing, thereby augmenting the integrity and applicability of the data collected for research purposes. It can be easily expanded to accommodate new basic events as needed; and it allows researchers to integrate their own logging events without the necessity of implementing a separate logging mechanism. It is worth noting that our system was built to assist primarily behavioral science research but is not limited to it, it can easily be adapted to assist information retrieval research or interacting with chat bot agents in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00510v1">Who's the MVP? A Game-Theoretic Evaluation Benchmark for Modular Attribution in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents frameworks often employ modular architectures, incorporating components such as planning, reasoning, action execution, and reflection to tackle complex tasks. However, quantifying the contribution of each module to overall system performance remains a significant challenge, impeding optimization and interpretability. To address this, we introduce CapaBench (Capability-level Assessment Benchmark), an evaluation framework grounded in cooperative game theory's Shapley Value, which systematically measures the marginal impact of individual modules and their interactions within an agent's architecture. By replacing default modules with test variants across all possible combinations, CapaBench provides a principle method for attributing performance contributions. Key contributions include: (1) We are the first to propose a Shapley Value-based methodology for quantifying the contributions of capabilities in LLM agents; (2) Modules with high Shapley Values consistently lead to predictable performance gains when combined, enabling targeted optimization; and (3) We build a multi-round dataset of over 1,000 entries spanning diverse domains and practical task scenarios, enabling comprehensive evaluation of agent capabilities. CapaBench bridges the gap between component-level evaluation and holistic system assessment, providing actionable insights for optimizing modular LLM agents and advancing their deployment in complex, real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00439v1">UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 11 pages, 4 figures. Preprint, under review
    </div>
    <details class="paper-abstract">
      Post-training is essential for adapting Large Language Models (LLMs) to real-world applications. Deploying post-trained models faces significant challenges due to substantial memory overhead and noticeable inference latency. Existing work has identified significant redundancies in LLMs and proposed efficient architectures, namely intra-layer KV sharing and cross-layer KV sharing. However, intra-layer KV sharing still results in high inference costs, while cross-layer KV sharing leads to significant performance degradation. As a result, both methods remain suboptimal for post-training pre-trained LLMs. In this paper, we identify that the \texttt{Softmax} operation is a primary bottleneck for LLM inference and discover that it is actually highly redundant during post-training. We propose Softmax \textbf{Uni}fication in \textbf{Att}e\textbf{n}tion (\textbf{UniAttn}), a novel post-training method that unifies Softmax activations across transformer blocks to reduce LLM inference costs. Additionally, UniAttn adopts a linear projection to compensate for the errors induced by Softmax unification. Experiments show that UniAttn matches the performance of standard post-training while significantly reducing inference costs, outperforming existing efficient architectures during post-training. Our code will be available at \url{https://github.com/Bostoncake/UniAttn}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00415v1">MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 25 pages, 7 figures, Under review at Financial Innovation (FIN)
    </div>
    <details class="paper-abstract">
      MarketSenseAI is a novel framework for holistic stock analysis which leverages Large Language Models (LLMs) to process financial news, historical prices, company fundamentals and the macroeconomic environment to support decision making in stock analysis and selection. In this paper, we present the latest advancements on MarketSenseAI, driven by rapid technological expansion in LLMs. Through a novel architecture combining Retrieval-Augmented Generation and LLM agents, the framework processes SEC filings and earnings calls, while enriching macroeconomic analysis through systematic processing of diverse institutional reports. We demonstrate a significant improvement in fundamental analysis accuracy over the previous version. Empirical evaluation on S\&P 100 stocks over two years (2023-2024) shows MarketSenseAI achieving cumulative returns of 125.9% compared to the index return of 73.5%, while maintaining comparable risk profiles. Further validation on S\&P 500 stocks during 2024 demonstrates the framework's scalability, delivering a 33.8% higher Sortino ratio than the market. This work marks a significant advancement in applying LLM technology to financial analysis, offering insights into the robustness of LLM-driven investment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00406v1">ALU: Agentic LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Information removal or suppression in large language models (LLMs) is a desired functionality, useful in AI regulation, legal compliance, safety, and privacy. LLM unlearning methods aim to remove information on demand from LLMs. Current LLM unlearning methods struggle to balance the unlearning efficacy and utility due to the competing nature of these objectives. Keeping the unlearning process computationally feasible without assuming access to the model weights is an overlooked area. We present the first agentic LLM unlearning (ALU) method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning that achieves effective unlearning while preserving the utility. Our ALU framework unlearns by involving multiple LLM agents, each designed for a specific step in the unlearning process, without the need to update model weights for any of the agents in the framework. Users can easily request any set of unlearning instances in any sequence, and ALU seamlessly adapts in real time. This is facilitated without requiring any changes in the underlying LLM model. Through extensive experiments on established benchmarks (TOFU, WMDP, WPU) and jailbreaking techniques (many shot, target masking, other languages), we demonstrate that ALU consistently stands out as the most robust LLM unlearning framework among current state-of-the-art methods while incurring a low constant-time cost. We further highlight ALU's superior performance compared to existing methods when evaluated at scale. Specifically, ALU is assessed on up to 1000 unlearning targets, exceeding the evaluation scope of all previously proposed LLM unlearning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00350v1">OrcaLoca: An LLM Agent Framework for Software Issue Localization</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00339v1">Challenges and Innovations in LLM-Powered Fake News Detection: A Synthesis of Approaches and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      The pervasiveness of the dissemination of fake news through social media platforms poses critical risks to the trust of the general public, societal stability, and democratic institutions. This challenge calls for novel methodologies in detection, which can keep pace with the dynamic and multi-modal nature of misinformation. Recent works include powering the detection using large language model advances in multimodal frameworks, methodologies using graphs, and adversarial training in the literature of fake news. Based on the different approaches which can bring success, some key highlights will be underlined: enhanced LLM-improves accuracy through more advanced semantics and cross-modality fusion for robust detections. The review further identifies critical gaps in adaptability to dynamic social media trends, real-time, and cross-platform detection capabilities, as well as the ethical challenges thrown up by the misuse of LLMs. Future directions underline the development of style-agnostic models, cross-lingual detection frameworks, and robust policies with a view to mitigating LLM-driven misinformation. This synthesis thus lays a concrete foundation for those researchers and practitioners committed to reinforcing fake news detection systems with complications that keep on growing in the digital landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v1">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      To reduce memory costs in long-context inference with Large Language Models (LLMs), many recent works focus on compressing the key-value (KV) cache of different tokens. However, we identify that the previous KV cache compression methods measure token importance individually, neglecting the dependency between different tokens in the real-world language characterics. In light of this, we introduce ChunkKV, grouping the tokens in a chunk as a basic compressing unit, and retaining the most informative semantic chunks while discarding the less important ones. Furthermore, observing that ChunkKV exhibits higher similarity in the preserved indices across different layers, we propose layer-wise index reuse to further reduce computational overhead. We evaluated ChunkKV on cutting-edge long-context benchmarks including LongBench and Needle-In-A-HayStack, as well as the GSM8K and JailbreakV in-context learning benchmark. Our experiments with instruction tuning and multi-step reasoning (O1 and R1) LLMs, achieve up to 10\% performance improvement under aggressive compression ratios compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v1">Estimating LLM Uncertainty with Logits</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have seen remarkable advancements and have been extensively integrated across various fields. Despite their progress, LLMs are prone to hallucinations, producing responses that may not be dependable if the models lack sufficient grounding knowledge. To mitigate this issue, methods for estimating uncertainty have been adopted, with a focus on critical tokens as indicators of reliability. Nevertheless, probability-based approaches have shown limitations in assessing token-level reliability due to the erosion of evidence strength information acquired during training. In this paper, we introduce Logits-induced Token Uncertainty (LogU), a novel framework designed to estimate token-specific uncertainty in LLMs in real time, without the need for multiple sampling rounds. By leveraging evidence modeling for the implementation of LogU, we utilize the derived uncertainty measures to steer downstream tasks. Our experimental findings highlight the substantial effectiveness and potential of LogU, marking a significant advancement in addressing the challenge of model hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00258v1">ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance in natural language processing tasks, yet their massive size makes serving them inefficient and costly. Semi-structured pruning has emerged as an effective method for model acceleration, but existing approaches are suboptimal because they focus on local, layer-wise optimizations using heuristic rules, failing to leverage global feedback. We present ProxSparse, a learning-based framework for mask selection enabled by regularized optimization. ProxSparse transforms the rigid, non-differentiable mask selection process into a smoother optimization procedure, allowing gradual mask exploration with flexibility. ProxSparse does not involve additional weight updates once the mask is determined. Our extensive evaluations on 7 widely used models show that ProxSparse consistently outperforms previously proposed semi-structured mask selection methods with significant improvement, demonstrating the effectiveness of our learned approach towards semi-structured pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14654v3">Comparative Analysis of Pooling Mechanisms in LLMs: A Sentiment Analysis Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Accepted to ISMSI'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP) by delivering state-of-the-art performance across a variety of tasks. Among these, Transformer-based models like BERT and GPT rely on pooling layers to aggregate token-level embeddings into sentence-level representations. Common pooling mechanisms such as Mean, Max, and Weighted Sum play a pivotal role in this aggregation process. Despite their widespread use, the comparative performance of these strategies on different LLM architectures remains underexplored. To address this gap, this paper investigates the effects of these pooling mechanisms on two prominent LLM families -- BERT and GPT, in the context of sentence-level sentiment analysis. Comprehensive experiments reveal that each pooling mechanism exhibits unique strengths and weaknesses depending on the task's specific requirements. Our findings underline the importance of selecting pooling methods tailored to the demands of particular applications, prompting a re-evaluation of common assumptions regarding pooling operations. By offering actionable insights, this study contributes to the optimization of LLM-based models for downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02625v2">HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Quantized training of Large Language Models (LLMs) remains an open challenge, as maintaining accuracy while performing all matrix multiplications in low precision has proven difficult. This is particularly the case when fine-tuning pre-trained models, which can have large weight and activation outlier values that make lower-precision optimization difficult. To address this, we present HALO, a novel quantization-aware training approach for Transformers that enables accurate and efficient low-precision training by combining 1) strategic placement of Hadamard rotations in both forward and backward passes, which mitigate outliers, 2) high-performance kernel support, and 3) FSDP integration for low-precision communication. Our approach ensures that all large matrix multiplications during the forward and backward passes are executed in lower precision. Applied to LLAMA-family models, HALO achieves near-full-precision-equivalent results during fine-tuning on various tasks, while delivering up to 1.41x end-to-end speedup for full fine-tuning on RTX 4090 GPUs. HALO efficiently supports both standard and parameterefficient fine-tuning (PEFT). Our results demonstrate the first practical approach to fully quantized LLM fine-tuning that maintains accuracy in 8-bit precision, while delivering performance benefits. Code is available at \url{https://github.com/IST-DASLab/HALO}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07267v2">Transforming Role Classification in Scientific Teams Using LLMs and Advanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 16 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Scientific team dynamics are critical in determining the nature and impact of research outputs. However, existing methods for classifying author roles based on self-reports and clustering lack comprehensive contextual analysis of contributions. Thus, we present a transformative approach to classifying author roles in scientific teams using advanced large language models (LLMs), which offers a more refined analysis compared to traditional clustering methods. Specifically, we seek to complement and enhance these traditional methods by utilizing open source and proprietary LLMs, such as GPT-4, Llama3 70B, Llama2 70B, and Mistral 7x8B, for role classification. Utilizing few-shot prompting, we categorize author roles and demonstrate that GPT-4 outperforms other models across multiple categories, surpassing traditional approaches such as XGBoost and BERT. Our methodology also includes building a predictive deep learning model using 10 features. By training this model on a dataset derived from the OpenAlex database, which provides detailed metadata on academic publications -- such as author-publication history, author affiliation, research topics, and citation counts -- we achieve an F1 score of 0.76, demonstrating robust classification of author roles.
    </details>
</div>
