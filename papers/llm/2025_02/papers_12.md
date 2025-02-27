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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- Part 12
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01042v2">Internal Activation as the Polar Star for Steering Unsafe LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities across a wide range of tasks but also pose significant risks due to their potential to generate harmful content. Although existing safety mechanisms can improve model safety, they often lead to overly cautious behavior and fail to fully utilize LLMs' internal cognitive processes. Drawing inspiration from cognitive science, where humans rely on reflective reasoning (System 2 thinking) to regulate language and behavior, we empirically demonstrate that LLMs also possess a similar capacity for internal assessment and regulation, which can be actively detected. Building on this insight, we introduce SafeSwitch, a framework that dynamically regulates unsafe outputs by monitoring and utilizing the model's internal states. Our empirical results show that SafeSwitch reduces harmful outputs by over 80% on safety benchmarks while maintaining strong utility. Compared to traditional safety alignment methods, SafeSwitch delivers more informative and context-aware refusals, demonstrates resilience to unseen queries, and achieves these benefits while only tuning less than 6% of the original parameters. These features make SafeSwitch a promising approach for implementing nuanced safety controls in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00903v2">Embracing Dialectic Intersubjectivity: Coordination of Different Perspectives in Content Analysis with LLM Persona Simulation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      This study attempts to advancing content analysis methodology from consensus-oriented to coordination-oriented practices, thereby embracing diverse coding outputs and exploring the dynamics among differential perspectives. As an exploratory investigation of this approach, we evaluate six GPT-4o configurations to analyze sentiment in Fox News and MSNBC transcripts on Biden and Trump during the 2020 U.S. presidential campaign, examining patterns across these models. By assessing each model's alignment with ideological perspectives, we explore how partisan selective processing could be identified in LLM-Assisted Content Analysis (LACA). Findings reveal that partisan persona LLMs exhibit stronger ideological biases when processing politically congruent content. Additionally, intercoder reliability is higher among same-partisan personas compared to cross-partisan pairs. This approach enhances the nuanced understanding of LLM outputs and advances the integrity of AI-driven social science research, enabling simulations of real-world implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12091v3">Is poisoning a real threat to LLM alignment? Maybe more so than you think</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01618v2">A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant performance gains via scaling up model sizes and/or data. However, recent evidence suggests diminishing returns from such approaches, motivating scaling the computation spent at inference time. Existing inference-time scaling methods, usually with reward models, cast the task as a search problem, which tends to be vulnerable to reward hacking as a consequence of approximation errors in reward models. In this paper, we instead cast inference-time scaling as a probabilistic inference task and leverage sampling-based techniques to explore the typical set of the state distribution of a state-space model with an approximate likelihood, rather than optimize for its mode directly. We propose a novel inference-time scaling approach by adapting particle-based Monte Carlo methods to this task. Our empirical evaluation demonstrates that our methods have a 4-16x better scaling rate over our deterministic search counterparts on various challenging mathematical reasoning tasks. Using our approach, we show that Qwen2.5-Math-1.5B-Instruct can surpass GPT-4o accuracy in only 4 rollouts, while Qwen2.5-Math-7B-Instruct scales to o1 level accuracy in only 32 rollouts. Our work not only presents an effective method to inference-time scaling, but also connects the rich literature in probabilistic inference with inference-time scaling of LLMs to develop more robust algorithms in future work. Code and further information is available at https://probabilistic-inference-scaling.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19297v2">Analysis of LLMs vs Human Experts in Requirements Engineering</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 8 pages, 15 figures
    </div>
    <details class="paper-abstract">
      The majority of research around Large Language Models (LLM) application to software development has been on the subject of code generation. There is little literature on LLMs' impact on requirements engineering (RE), which deals with the process of developing and verifying the system requirements. Within RE, there is a subdiscipline of requirements elicitation, which is the practice of discovering and documenting requirements for a system from users, customers, and other stakeholders. In this analysis, we compare LLM's ability to elicit requirements of a software system, as compared to that of a human expert in a time-boxed and prompt-boxed study. We found LLM-generated requirements were evaluated as more aligned (+1.12) than human-generated requirements with a trend of being more complete (+10.2%). Conversely, we found users tended to believe that solutions they perceived as more aligned had been generated by human experts. Furthermore, while LLM-generated documents scored higher and performed at 720x the speed, their cost was, on average, only 0.06% that of a human expert. Overall, these findings indicate that LLMs will play an increasingly important role in requirements engineering by improving requirements definitions, enabling more efficient resource allocation, and reducing overall project timelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02412v1">AI-Powered, But Power-Hungry? Energy Efficiency of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used in software development to assist in various tasks, e.g., code generation and code completion, but empirical evaluations of the quality of the results produced by these models focus on correctness and ignore other relevant aspects, such as their performance and energy efficiency. Studying the performance of LLM-produced programs is essential to understand how well LLMs can support the construction of performance- and energy-critical software, such as operating systems, servers, and mobile applications. This paper presents the first study analyzing the energy efficiency and performance of LLM-generated code for three programming languages Python, Java, and C++, on two platforms, a Mac and a PC, leveraging three frontier LLMs, Github Copilot, GPT-4o, and the recently-released OpenAI o1-mini, and targeting ``hard'' programming problems from LeetCode. Our results show that the models are much more successful in generating Python and Java than C++ code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07668v3">Towards Evaluation Guidelines for Empirical Studies involving LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 4 pages, 2nd IEEE/ACM International Workshop on Methodological Issues with Empirical Studies in Software Engineering (WSESE 2025)
    </div>
    <details class="paper-abstract">
      In the short period since the release of ChatGPT, large language models (LLMs) have changed the software engineering research landscape. While there are numerous opportunities to use LLMs for supporting research or software engineering tasks, solid science needs rigorous empirical evaluations. However, so far, there are no specific guidelines for conducting and assessing studies involving LLMs in software engineering research. Our focus is on empirical studies that either use LLMs as part of the research process or studies that evaluate existing or new tools that are based on LLMs. This paper contributes the first set of holistic guidelines for such studies. Our goal is to start a discussion in the software engineering research community to reach a common understanding of our standards for high-quality empirical studies involving LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02368v1">Evaluating the Effectiveness of LLMs in Fixing Maintainability Issues in Real-World Projects</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained attention for addressing coding problems, but their effectiveness in fixing code maintainability remains unclear. This study evaluates LLMs capability to resolve 127 maintainability issues from 10 GitHub repositories. We use zero-shot prompting for Copilot Chat and Llama 3.1, and few-shot prompting with Llama only. The LLM-generated solutions are assessed for compilation errors, test failures, and new maintainability problems. Llama with few-shot prompting successfully fixed 44.9% of the methods, while Copilot Chat and Llama zero-shot fixed 32.29% and 30%, respectively. However, most solutions introduced errors or new maintainability issues. We also conducted a human study with 45 participants to evaluate the readability of 51 LLM-generated solutions. The human study showed that 68.63% of participants observed improved readability. Overall, while LLMs show potential for fixing maintainability issues, their introduction of errors highlights their current limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02362v1">Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances mathematical reasoning in large language models (LLMs) by enabling detailed step-by-step solutions. However, due to the verbosity of LLMs, the resulting reasoning chains can be long, making it harder to verify the reasoning steps and trace issues resulting from dependencies between the steps that may be farther away in the sequence of steps. Importantly, mathematical reasoning allows each step to be derived from a small set of premises, which are a subset of the preceding steps in the reasoning chain. In this paper, we present a framework that identifies the premises for each step, to improve the evaluation of reasoning. We restructure conventional linear reasoning chains into Premise Augmented Reasoning Chains (PARC) by introducing premise links, resulting in a directed acyclic graph where the nodes are the steps and the edges are the premise links. Through experiments with a PARC-based dataset that we built, namely PERL (Premises and ERrors identification in LLMs), we demonstrate that LLMs can reliably identify premises within complex reasoning chains. In particular, even open-source LLMs achieve 90% recall in premise identification. We also show that PARC helps to identify errors in reasoning chains more reliably. The accuracy of error identification improves by 6% to 16% absolute when step-by-step verification is carried out in PARC under the premises. Our findings highlight the utility of premise-centric representations in addressing complex problem-solving tasks and open new avenues for improving the reliability of LLM-based reasoning evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01600v2">Reinforcement Learning for Long-Horizon Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Interactive digital agents (IDAs) leverage APIs of stateful digital environments to perform tasks in response to user requests. While IDAs powered by instruction-tuned large language models (LLMs) can react to feedback from interface invocations in multi-step exchanges, they have not been trained in their respective digital environments. Prior methods accomplish less than half of tasks in sophisticated benchmarks such as AppWorld. We present a reinforcement learning (RL) approach that trains IDAs directly in their target environments. We formalize this training as a partially observable Markov decision process and derive LOOP, a data- and memory-efficient variant of proximal policy optimization. LOOP uses no value network and maintains exactly one copy of the underlying LLM in memory, making its implementation straightforward and as memory-efficient as fine-tuning a single LLM. A 32-billion-parameter agent trained with LOOP in the AppWorld environment outperforms the much larger OpenAI o1 agent by 9 percentage points (15% relative). To our knowledge, this is the first reported application of RL to IDAs that interact with a stateful, multi-domain, multi-app environment via direct API calls. Our analysis sheds light on the effectiveness of RL in this area, showing that the agent learns to consult the API documentation, avoid unwarranted assumptions, minimize confabulation, and recover from setbacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02342v1">SHIELD: APT Detection and Intelligent Explanation Using LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Advanced persistent threats (APTs) are sophisticated cyber attacks that can remain undetected for extended periods, making their mitigation particularly challenging. Given their persistence, significant effort is required to detect them and respond effectively. Existing provenance-based attack detection methods often lack interpretability and suffer from high false positive rates, while investigation approaches are either supervised or limited to known attacks. To address these challenges, we introduce SHIELD, a novel approach that combines statistical anomaly detection and graph-based analysis with the contextual analysis capabilities of large language models (LLMs). SHIELD leverages the implicit knowledge of LLMs to uncover hidden attack patterns in provenance data, while reducing false positives and providing clear, interpretable attack descriptions. This reduces analysts' alert fatigue and makes it easier for them to understand the threat landscape. Our extensive evaluation demonstrates SHIELD's effectiveness and computational efficiency in real-world scenarios. SHIELD was shown to outperform state-of-the-art methods, achieving higher precision and recall. SHIELD's integration of anomaly detection, LLM-driven contextual analysis, and advanced graph-based correlation establishes a new benchmark for APT detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02337v1">Rule-ATT&CK Mapper (RAM): Mapping SIEM Rules to TTPs Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      The growing frequency of cyberattacks has heightened the demand for accurate and efficient threat detection systems. SIEM platforms are important for analyzing log data and detecting adversarial activities through rule-based queries, also known as SIEM rules. The efficiency of the threat analysis process relies heavily on mapping these SIEM rules to the relevant attack techniques in the MITRE ATT&CK framework. Inaccurate annotation of SIEM rules can result in the misinterpretation of attacks, increasing the likelihood that threats will be overlooked. Existing solutions for annotating SIEM rules with MITRE ATT&CK technique labels have notable limitations: manual annotation of SIEM rules is both time-consuming and prone to errors, and ML-based approaches mainly focus on annotating unstructured free text sources rather than structured data like SIEM rules. Structured data often contains limited information, further complicating the annotation process and making it a challenging task. To address these challenges, we propose Rule-ATT&CK Mapper (RAM), a novel framework that leverages LLMs to automate the mapping of structured SIEM rules to MITRE ATT&CK techniques. RAM's multi-stage pipeline, which was inspired by the prompt chaining technique, enhances mapping accuracy without requiring LLM pre-training or fine-tuning. Using the Splunk Security Content dataset, we evaluate RAM's performance using several LLMs, including GPT-4-Turbo, Qwen, IBM Granite, and Mistral. Our evaluation highlights GPT-4-Turbo's superior performance, which derives from its enriched knowledge base, and an ablation study emphasizes the importance of external contextual knowledge in overcoming the limitations of LLMs' implicit knowledge for domain-specific tasks. These findings demonstrate RAM's potential in automating cybersecurity workflows and provide valuable insights for future advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00029v2">AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Financial metrics like the Sharpe ratio are pivotal in evaluating investment performance by balancing risk and return. However, traditional metrics often struggle with robustness and generalization, particularly in dynamic and volatile market conditions. This paper introduces AlphaSharpe, a novel framework leveraging large language models (LLMs) to iteratively evolve and optimize financial metrics to discover enhanced risk-return metrics that outperform traditional approaches in robustness and correlation with future performance metrics by employing iterative crossover, mutation, and evaluation. Key contributions of this work include: (1) a novel use of LLMs to generate and refine financial metrics with implicit domain-specific knowledge, (2) a scoring mechanism to ensure that evolved metrics generalize effectively to unseen data, and (3) an empirical demonstration of 3x predictive power for future risk-returns, and 2x portfolio performance. Experimental results in a real-world dataset highlight the superiority of discovered metrics, making them highly relevant to portfolio managers and financial decision-makers. This framework not only addresses the limitations of existing metrics but also showcases the potential of LLMs in advancing financial analytics, paving the way for informed and robust investment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02329v1">ReSpark: Leveraging Previous Data Reports as References to Generate New Reports with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Creating data reports is time-consuming, as it requires iterative exploration and understanding of data, followed by summarizing the insights. While large language models (LLMs) are powerful tools for data processing and text generation, they often struggle to produce complete data reports that fully meet user expectations. One significant challenge is effectively communicating the entire analysis logic to LLMs. Moreover, determining a comprehensive analysis logic can be mentally taxing for users. To address these challenges, we propose ReSpark, an LLM-based method that leverages existing data reports as references for creating new ones. Given a data table, ReSpark searches for similar-topic reports, parses them into interdependent segments corresponding to analytical objectives, and executes them with new data. It identifies inconsistencies and customizes the objectives, data transformations, and textual descriptions. ReSpark allows users to review real-time outputs, insert new objectives, and modify report content. Its effectiveness was evaluated through comparative and user studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14630v2">Extracting Problem Structure with LLMs for Optimized SAT Local Search</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Local search preprocessing makes Conflict-Driven Clause Learning (CDCL) solvers faster by providing high-quality starting points and modern SAT solvers have incorporated this technique into their preprocessing steps. However, these tools rely on basic strategies that miss the structural patterns in problems. We present a method that applies Large Language Models (LLMs) to analyze Python-based encoding code. This reveals hidden structural patterns in how problems convert into SAT. Our method automatically generates specialized local search algorithms that find these patterns and use them to create strong initial assignments. This works for any problem instance from the same encoding type. Our tests show encouraging results, achieving faster solving times compared to baseline preprocessing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02201v1">Can You Move These Over There? An LLM-based VR Mover for Supporting Object Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 64 pages (30 in main text), 22 figures (19 in main text)
    </div>
    <details class="paper-abstract">
      In our daily lives, we can naturally convey instructions for the spatial manipulation of objects using words and gestures. Transposing this form of interaction into virtual reality (VR) object manipulation can be beneficial. We propose VR Mover, an LLM-empowered solution that can understand and interpret the user's vocal instruction to support object manipulation. By simply pointing and speaking, the LLM can manipulate objects without structured input. Our user study demonstrates that VR Mover enhances user usability, overall experience and performance on multi-object manipulation, while also reducing workload and arm fatigue. Users prefer the proposed natural interface for broad movements and may complementarily switch to gizmos or virtual hands for finer adjustments. These findings are believed to contribute to design implications for future LLM-based object manipulation interfaces, highlighting the potential for more intuitive and efficient user interactions in VR environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02199v1">When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable success in language modelling due to scaling laws found in model size and the hidden dimension of the model's text representation. Yet, we demonstrate that compressed representations of text can yield better performance in LLM-based regression tasks. In this paper, we compare the relative performance of embedding compression in three different signal-to-noise contexts: financial return prediction, writing quality assessment and review scoring. Our results show that compressing embeddings, in a minimally supervised manner using an autoencoder's hidden representation, can mitigate overfitting and improve performance on noisy tasks, such as financial return prediction; but that compression reduces performance on tasks that have high causal dependencies between the input and target data. Our results suggest that the success of interpretable compressed representations such as sentiment may be due to a regularising effect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18712v3">Invisible Traces: Using Hybrid Fingerprinting to identify underlying LLMs in GenAI Apps</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Fingerprinting refers to the process of identifying underlying Machine Learning (ML) models of AI Systemts, such as Large Language Models (LLMs), by analyzing their unique characteristics or patterns, much like a human fingerprint. The fingerprinting of Large Language Models (LLMs) has become essential for ensuring the security and transparency of AI-integrated applications. While existing methods primarily rely on access to direct interactions with the application to infer model identity, they often fail in real-world scenarios involving multi-agent systems, frequent model updates, and restricted access to model internals. In this paper, we introduce a novel fingerprinting framework designed to address these challenges by integrating static and dynamic fingerprinting techniques. Our approach identifies architectural features and behavioral traits, enabling accurate and robust fingerprinting of LLMs in dynamic environments. We also highlight new threat scenarios where traditional fingerprinting methods are ineffective, bridging the gap between theoretical techniques and practical application. To validate our framework, we present an extensive evaluation setup that simulates real-world conditions and demonstrate the effectiveness of our methods in identifying and monitoring LLMs in Gen-AI applications. Our results highlight the framework's adaptability to diverse and evolving deployment contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14202v2">Rationale Behind Essay Scores: Enhancing S-LLM's Multi-Trait Essay Scoring with Rationale Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Existing automated essay scoring (AES) has solely relied on essay text without using explanatory rationales for the scores, thereby forgoing an opportunity to capture the specific aspects evaluated by rubric indicators in a fine-grained manner. This paper introduces Rationale-based Multiple Trait Scoring (RMTS), a novel approach for multi-trait essay scoring that integrates prompt-engineering-based large language models (LLMs) with a fine-tuning-based essay scoring model using a smaller large language model (S-LLM). RMTS uses an LLM-based trait-wise rationale generation system where a separate LLM agent generates trait-specific rationales based on rubric guidelines, which the scoring model uses to accurately predict multi-trait scores. Extensive experiments on benchmark datasets, including ASAP, ASAP++, and Feedback Prize, show that RMTS significantly outperforms state-of-the-art models and vanilla S-LLMs in trait-specific scoring. By assisting quantitative assessment with fine-grained qualitative rationales, RMTS enhances the trait-wise reliability, providing partial explanations about essays. The code is available at https://github.com/BBeeChu/RMTS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17178v2">Tuning LLM Judge Design Decisions for 1/1000 of the Cost</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) often requires costly human annotations. To address this, LLM-based judges have been proposed, which compare the outputs of two LLMs enabling the ranking of models without human intervention. While several approaches have been proposed, many confounding factors are present between different papers. For instance the model, the prompt and other hyperparameters are typically changed at the same time making apple-to-apple comparisons challenging. In this paper, we propose to systematically analyze and tune hyperparameter of LLM judges. To alleviate the high cost of evaluating a judge, we propose to leverage multi-objective multi-fidelity which allows to find judges that trades accuracy for cost and also reduce significantly the cost of the search. Our method identifies judges that not only outperform existing benchmarks in accuracy and cost-efficiency but also utilize open-weight models, ensuring greater accessibility and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19200v2">On Behalf of the Stakeholders: Trends in NLP Model Interpretability in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Recent advancements in NLP systems, particularly with the introduction of LLMs, have led to widespread adoption of these systems by a broad spectrum of users across various domains, impacting decision-making, the job market, society, and scientific research. This surge in usage has led to an explosion in NLP model interpretability and analysis research, accompanied by numerous technical surveys. Yet, these surveys often overlook the needs and perspectives of explanation stakeholders. In this paper, we address three fundamental questions: Why do we need interpretability, what are we interpreting, and how? By exploring these questions, we examine existing interpretability paradigms, their properties, and their relevance to different stakeholders. We further explore the practical implications of these paradigms by analyzing trends from the past decade across multiple research fields. To this end, we retrieved thousands of papers and employed an LLM to characterize them. Our analysis reveals significant disparities between NLP developers and non-developer users, as well as between research fields, underscoring the diverse needs of stakeholders. For example, explanations of internal model components are rarely used outside the NLP field. We hope this paper informs the future design, development, and application of methods that align with the objectives and requirements of various stakeholders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14175v2">QMOS: Enhancing LLMs for Telecommunication with Question Masked loss and Option Shuffling</a></div>
    <div class="paper-meta">
      📅 2025-02-04
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have brought about substantial advancements in the field of Question Answering (QA) systems. These models do remarkably well in addressing intricate inquiries in a variety of disciplines. However, because of domain-specific vocabulary, complex technological concepts, and the requirement for exact responses applying LLMs to specialized sectors like telecommunications presents additional obstacles. GPT-3.5 has been used in recent work, to obtain noteworthy accuracy for telecom-related questions in a Retrieval Augmented Generation (RAG) framework. Notwithstanding these developments, the practical use of models such as GPT-3.5 is restricted by their proprietary nature and high computing demands. This paper introduces QMOS, an innovative approach which uses a Question-Masked loss and Option Shuffling trick to enhance the performance of LLMs in answering Multiple-Choice Questions in the telecommunications domain. Our focus was on using opensource, smaller language models (Phi-2 and Falcon-7B) within an enhanced RAG framework. Our multi-faceted approach involves several enhancements to the whole LLM-RAG pipeline of finetuning, retrieval, prompt engineering and inference. Our approaches significantly outperform existing results, achieving accuracy improvements from baselines of 24.70% to 49.30% with Falcon-7B and from 42.07% to 84.65% with Phi-2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02067v1">AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinement</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025
    </div>
    <details class="paper-abstract">
      Embodied agents assisting humans are often asked to complete a new task in a new scenario. An agent preparing a particular dish in the kitchen based on a known recipe may be asked to prepare a new dish or to perform cleaning tasks in the storeroom. There may not be sufficient resources, e.g., time or labeled examples, to train the agent for these new situations. Large Language Models (LLMs) trained on considerable knowledge across many domains are able to predict a sequence of abstract actions for such new tasks and scenarios, although it may not be possible for the agent to execute this action sequence due to task-, agent-, or domain-specific constraints. Our framework addresses these challenges by leveraging the generic predictions provided by LLM and the prior domain-specific knowledge encoded in a Knowledge Graph (KG), enabling an agent to quickly adapt to new tasks and scenarios. The robot also solicits and uses human input as needed to refine its existing knowledge. Based on experimental evaluation over cooking and cleaning tasks in simulation domains, we demonstrate that the interplay between LLM, KG, and human input leads to substantial performance gains compared with just using the LLM output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02066v1">Anticipate & Act : Integrating LLMs and Classical Planning for Efficient Task Execution in Household Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2024
    </div>
    <details class="paper-abstract">
      Assistive agents performing household tasks such as making the bed or cooking breakfast often compute and execute actions that accomplish one task at a time. However, efficiency can be improved by anticipating upcoming tasks and computing an action sequence that jointly achieves these tasks. State-of-the-art methods for task anticipation use data-driven deep networks and Large Language Models (LLMs), but they do so at the level of high-level tasks and/or require many training examples. Our framework leverages the generic knowledge of LLMs through a small number of prompts to perform high-level task anticipation, using the anticipated tasks as goals in a classical planning system to compute a sequence of finer-granularity actions that jointly achieve these goals. We ground and evaluate our framework's abilities in realistic scenarios in the VirtualHome environment and demonstrate a 31% reduction in execution time compared with a system that does not consider upcoming tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20242v4">BadRobot: Jailbreaking Embodied LLMs in the Physical World</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 Accepted to ICLR 2025. Project page: https://Embodied-LLMs-Safety.github.io
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00212v2">STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</a></div>
    <div class="paper-meta">
      📅 2025-02-04
      | 💬 22 pages, 5 figures
    </div>
    <details class="paper-abstract">
      A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating between LLM generating proofs and finetuning them on correctly generated ones, performance quickly plateaus due to the scarcity of correct proofs (sparse rewards). To keep improving the models with limited data, we draw inspiration from mathematicians, who continuously develop new results, partly by proposing novel conjectures or exercises (which are often variants of known results) and attempting to solve them. We design the Self-play Theorem Prover (STP) that simultaneously takes on two roles, conjecturer and prover, each providing training signals to the other. The conjecturer is trained iteratively on previously generated conjectures that are barely provable by the current prover, which incentivizes it to generate increasingly challenging conjectures over time. The prover attempts to prove the conjectures with standard expert iteration. We evaluate STP with both Lean and Isabelle formal versifiers. With 19.8 billion tokens generated during the training in Lean, STP proves 26.3% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration. The final model achieves state-of-the-art performance among whole-proof generation methods on miniF2F-test (61.1%, pass@3200), Proofnet-test (23.1%, pass@3200) and PutnamBench (8/644, pass@64).
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14808v2">HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 15 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have facilitated a wide range of applications with distinct service-level objectives (SLOs), from latency-sensitive online tasks like interactive chatbots to throughput-oriented offline workloads like document summarization. The existing deployment model, which dedicates machines to each workload, simplifies SLO management but often leads to poor resource utilization. This paper introduces HyGen, an interference-aware LLM serving system that enables efficient co-location of online and offline workloads while preserving latency requirements. HyGen incorporates two key innovations: (1) performance control mechanisms, including a latency predictor to estimate batch execution time and an SLO-aware profiler to quantify latency interference, and (2) SLO-aware offline scheduling policies that maximize serving throughput and prevent starvation, without compromising online serving latency. Our evaluation on production workloads shows that HyGen achieves up to 3.87x overall throughput and 5.84x offline throughput gains over online and hybrid serving baselines, respectively, while strictly satisfying latency SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13865v2">Breaking Information Cocoons: A Hyperbolic Graph-LLM Framework for Exploration and Exploitation in Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Modern recommender systems often create information cocoons, restricting users' exposure to diverse content. A key challenge lies in balancing content exploration and exploitation while allowing users to adjust their recommendation preferences. Intuitively, this balance can be modeled as a tree-structured representation, where depth search facilitates exploitation and breadth search enables exploration. However, existing approaches face two fundamental limitations: Euclidean methods struggle to capture hierarchical structures, while hyperbolic methods, despite their superior hierarchical modeling, lack semantic understanding of user and item profiles and fail to provide a principled mechanism for balancing exploration and exploitation. To address these challenges, we propose HERec, a hyperbolic graph-LLM framework that effectively balances exploration and exploitation in recommender systems. Our framework introduces two key innovations: (1) a hierarchical-aware graph-LLM mechanism that jointly aligns textual descriptions with user-item collaborative information in hyperbolic space, and (2) a hierarchical representation structure that enables user-adjustable exploration-exploitation trade-offs. Extensive experiments demonstrate that HERec consistently outperforms both Euclidean and hyperbolic baselines, achieving up to 5.49% improvement in utility metrics and 11.39% increase in diversity metrics, effectively mitigating information cocoons. We open-source our model implementation at https://github.com/Martin-qyma/HERec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17183v2">LLM Evaluation Based on Aerospace Manufacturing Expertise: Automated Generation and Multi-Model Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 conference paper
    </div>
    <details class="paper-abstract">
      Aerospace manufacturing demands exceptionally high precision in technical parameters. The remarkable performance of Large Language Models (LLMs), such as GPT-4 and QWen, in Natural Language Processing has sparked industry interest in their application to tasks including process design, material selection, and tool information retrieval. However, LLMs are prone to generating "hallucinations" in specialized domains, producing inaccurate or false information that poses significant risks to the quality of aerospace products and flight safety. This paper introduces a set of evaluation metrics tailored for LLMs in aerospace manufacturing, aiming to assess their accuracy by analyzing their performance in answering questions grounded in professional knowledge. Firstly, key information is extracted through in-depth textual analysis of classic aerospace manufacturing textbooks and guidelines. Subsequently, utilizing LLM generation techniques, we meticulously construct multiple-choice questions with multiple correct answers of varying difficulty. Following this, different LLM models are employed to answer these questions, and their accuracy is recorded. Experimental results demonstrate that the capabilities of LLMs in aerospace professional knowledge are in urgent need of improvement. This study provides a theoretical foundation and practical guidance for the application of LLMs in aerospace manufacturing, addressing a critical gap in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15594v4">A Survey on LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Corrected typos & more discussion on reasoning models 33 pages, 9 figures. arXiv admin note: text overlap with arXiv:2310.05470 by other authors
    </div>
    <details class="paper-abstract">
      Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17626v2">Tracking the Feature Dynamics in LLM Training: A Mechanistic Study</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Understanding training dynamics and feature evolution is crucial for the mechanistic interpretability of large language models (LLMs). Although sparse autoencoders (SAEs) have been used to identify features within LLMs, a clear picture of how these features evolve during training remains elusive. In this study, we: (1) introduce SAE-Track, a novel method to efficiently obtain a continual series of SAEs; (2) mechanistically investigate feature formation and develop a progress measure for it ; and (3) analyze and visualize feature drift during training. Our work provides new insights into the dynamics of features in LLMs, enhancing our understanding of training mechanisms and feature evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13382v3">VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 AAAI 2025
    </div>
    <details class="paper-abstract">
      Video Temporal Grounding (VTG) strives to accurately pinpoint event timestamps in a specific video using linguistic queries, significantly impacting downstream tasks like video browsing and editing. Unlike traditional task-specific models, Video Large Language Models (video LLMs) can handle multiple tasks concurrently in a zero-shot manner. Consequently, exploring the application of video LLMs for VTG tasks has become a burgeoning research area. However, despite considerable advancements in video content understanding, video LLMs often struggle to accurately pinpoint timestamps within videos, limiting their effectiveness in VTG tasks. To address this, we introduce VTG-LLM, a model designed to enhance video LLMs' timestamp localization abilities. Our approach includes: (1) effectively integrating timestamp knowledge into visual tokens; (2) incorporating absolute-time tokens to manage timestamp knowledge without concept shifts; and (3) introducing a lightweight, high-performance, slot-based token compression technique designed to accommodate the demands of a large number of frames to be sampled for VTG tasks. Additionally, we present VTG-IT-120K, a collection of publicly available VTG datasets that we have re-annotated to improve upon low-quality annotations. Our comprehensive experiments demonstrate the superior performance of VTG-LLM in comparison to other video LLM methods across a variety of VTG tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03993v2">Assessing LLMs for Zero-shot Abstractive Summarization Through the Lens of Relevance Paraphrasing</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Accepted to NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved state-of-the-art performance at zero-shot generation of abstractive summaries for given articles. However, little is known about the robustness of such a process of zero-shot summarization. To bridge this gap, we propose relevance paraphrasing, a simple strategy that can be used to measure the robustness of LLMs as summarizers. The relevance paraphrasing approach identifies the most relevant sentences that contribute to generating an ideal summary, and then paraphrases these inputs to obtain a minimally perturbed dataset. Then, by evaluating model performance for summarization on both the original and perturbed datasets, we can assess the LLM's one aspect of robustness. We conduct extensive experiments with relevance paraphrasing on 4 diverse datasets, as well as 4 LLMs of different sizes (GPT-3.5-Turbo, Llama-2-13B, Mistral-7B, and Dolly-v2-7B). Our results indicate that LLMs are not consistent summarizers for the minimally perturbed articles, necessitating further improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v2">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00511v1">Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities. However, single-shot inference often yields unreliable results for complex reasoning tasks, leading researchers to explore multiple reasoning paths through methods such as perplexity and self-consistency. In this paper, we present the first theoretical error decomposition analysis of these techniques, breaking down their error into estimation error and model error. Our analysis reveals a fundamental trade-off: perplexity methods suffer from substantial model error due to the absence of a proper consistency function, while self-consistency exhibits high estimation error due to a slow error convergence rate. To overcome these limitations, we propose Reasoning-Pruning Perplexity Consistency (RPC). This approach combines Perplexity Consistency, which seamlessly integrates LLM perplexity with self-consistency, and Reasoning Pruning, which eliminates low-probability reasoning paths to effectively prevent the degeneration of estimation error reduction. Theoretical analysis demonstrates that RPC not only accelerates the convergence rate of estimation error to an exponential level but also holds strong potential for further reducing model error. Extensive empirical evaluations on seven benchmark datasets confirm that RPC can significantly improve reasoning performance, sample efficiency, and confidence reliability.
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
