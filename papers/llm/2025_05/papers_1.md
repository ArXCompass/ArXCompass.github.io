# llm - 2025_05

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11163v2">Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      We propose Model Swarms, a collaborative search algorithm to adapt LLMs via swarm intelligence, the collective behavior guiding individual systems. Specifically, Model Swarms starts with a pool of LLM experts and a utility function. Guided by the best-found checkpoints across models, diverse LLM experts collaboratively move in the weight space and optimize a utility function representing model adaptation objectives. Compared to existing model composition approaches, Model Swarms offers tuning-free model adaptation, works in low-data regimes with as few as 200 examples, and does not require assumptions about specific experts in the swarm or how they should be composed. Extensive experiments demonstrate that Model Swarms could flexibly adapt LLM experts to a single task, multi-task domains, reward models, as well as diverse human interests, improving over 12 model composition baselines by up to 21.0% across tasks and contexts. Further analysis reveals that LLM experts discover previously unseen capabilities in initial checkpoints and that Model Swarms enable the weak-to-strong transition of experts through the collaborative search process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11767v2">From Selection to Generation: A Survey of LLM-based Active Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Active Learning (AL) has been a powerful paradigm for improving model efficiency and performance by selecting the most informative data points for labeling and training. In recent active learning frameworks, Large Language Models (LLMs) have been employed not only for selection but also for generating entirely new data instances and providing more cost-effective annotations. Motivated by the increasing importance of high-quality data and efficient model training in the era of LLMs, we present a comprehensive survey on LLM-based Active Learning. We introduce an intuitive taxonomy that categorizes these techniques and discuss the transformative roles LLMs can play in the active learning loop. We further examine the impact of AL on LLM learning paradigms and its applications across various domains. Finally, we identify open challenges and propose future research directions. This survey aims to serve as an up-to-date resource for researchers and practitioners seeking to gain an intuitive understanding of LLM-based AL techniques and deploy them to new applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17701v2">From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 25 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Evacuation decision prediction is critical for efficient and effective wildfire response by helping emergency management anticipate traffic congestion and bottlenecks, allocate resources, and minimize negative impacts. Traditional statistical methods for evacuation decision prediction fail to capture the complex and diverse behavioral logic of different individuals. In this work, for the first time, we introduce FLARE, short for facilitating LLM for advanced reasoning on wildfire evacuation decision prediction, a Large Language Model (LLM)-based framework that integrates behavioral theories and models to streamline the Chain-of-Thought (CoT) reasoning and subsequently integrate with memory-based Reinforcement Learning (RL) module to provide accurate evacuation decision prediction and understanding. Our proposed method addresses the limitations of using existing LLMs for evacuation behavioral predictions, such as limited survey data, mismatching with behavioral theory, conflicting individual preferences, implicit and complex mental states, and intractable mental state-behavior mapping. Experiments on three post-wildfire survey datasets show an average of 20.47% performance improvement over traditional theory-informed behavioral models, with strong cross-event generalizability. Our complete code is publicly available at https://github.com/SusuXu-s-Lab/FLARE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13667v2">Exploring Multi-Modal Data with Tool-Augmented LLM Agents for Precise Causal Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      Causal discovery is an imperative foundation for decision-making across domains, such as smart health, AI for drug discovery and AIOps. Traditional statistical causal discovery methods, while well-established, predominantly rely on observational data and often overlook the semantic cues inherent in cause-and-effect relationships. The advent of Large Language Models (LLMs) has ushered in an affordable way of leveraging the semantic cues for knowledge-driven causal discovery, but the development of LLMs for causal discovery lags behind other areas, particularly in the exploration of multi-modal data. To bridge the gap, we introduce MATMCD, a multi-agent system powered by tool-augmented LLMs. MATMCD has two key agents: a Data Augmentation agent that retrieves and processes modality-augmented data, and a Causal Constraint agent that integrates multi-modal data for knowledge-driven reasoning. The proposed design of the inner-workings ensures successful cooperation of the agents. Our empirical study across seven datasets suggests the significant potential of multi-modality enhanced causal discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20576v5">OmniRouter: Budget and Performance Controllable Multi-LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver superior performance but require substantial computational resources and operate with relatively low efficiency, while smaller models can efficiently handle simpler tasks with fewer resources. LLM routing is a crucial paradigm that dynamically selects the most suitable large language models from a pool of candidates to process diverse inputs, ensuring optimal resource utilization while maintaining response quality. Existing routing frameworks typically model this as a locally optimal decision-making problem, selecting the presumed best-fit LLM for each query individually, which overlook global budget constraints, resulting in ineffective resource allocation. To tackle this problem, we introduce OmniRouter, a fundamentally controllable routing framework for multi-LLM serving. Instead of making per-query greedy choices, OmniRouter models the routing task as a constrained optimization problem, assigning models that minimize total cost while ensuring the required performance level. Specifically, a hybrid retrieval-augmented predictor is designed to predict the capabilities and costs of LLMs and a constrained optimizer is employed to control globally optimal query-model allocation. Experiments show that OmniRouter achieves up to 6.30% improvement in response accuracy while simultaneously reducing computational costs by at least 10.15% compared to competitive router baselines. The code and the dataset are available at https://github.com/agiresearch/OmniRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00982v2">Are LLMs effective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practice</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      In psychological practices, standardized questionnaires serve as essential tools for assessing mental health through structured, clinically-validated questions (i.e., items). While social media platforms offer rich data for mental health screening, computational approaches often bypass these established clinical assessment tools in favor of black-box classification. We propose a novel questionnaire-guided screening framework that bridges psychological practice and computational methods through adaptive Retrieval-Augmented Generation (\textit{aRAG}). Our approach links unstructured social media content and standardized clinical assessments by retrieving relevant posts for each questionnaire item and using Large Language Models (LLMs) to complete validated psychological instruments. Our findings demonstrate two key advantages of questionnaire-guided screening: First, when completing the Beck Depression Inventory-II (BDI-II), our approach matches or outperforms state-of-the-art performance on Reddit-based benchmarks without requiring training data. Second, we show that guiding LLMs through standardized questionnaires can yield superior results compared to directly prompting them for depression screening, while also providing a more interpretable assessment by linking model outputs to clinically validated diagnostic criteria. Additionally, we show, as a proof-of-concept, how our questionnaire-based methodology can be extended to other mental conditions' screening, highlighting the promising role of LLMs as psychological assessors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03793v3">LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 Accepted by ICML'2025
    </div>
    <details class="paper-abstract">
      The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks? In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. In particular, we first derive a PAC-Bayesian Generalization Bound that unveils fine-tuning dynamics of LLMs and then introduce LENSLLM, a Neural Tangent Kernel (NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency. Extensive empirical results on 3 large-scale benchmarks demonstrate that our model achieves up to 91.1% accuracy and reduces up to 88.5% computational cost in LLM selection, outperforming 5 state-of-the-art methods. We open-source our proposed LENSLLM model and corresponding results at LensLLM.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20445v2">TrajAgent: An LLM-based Agent Framework for Automated Trajectory Modeling via Collaboration of Large and Small Models</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 the code will be openly accessible at: https://github.com/tsinghua-fib-lab/TrajAgent
    </div>
    <details class="paper-abstract">
      Trajectory modeling, which includes research on trajectory data pattern mining and future prediction, has widespread applications in areas such as life services, urban transportation, and public administration. Numerous methods have been proposed to address specific problems within trajectory modeling. However, the heterogeneity of data and the diversity of trajectory tasks make effective and reliable trajectory modeling an important yet highly challenging endeavor, even for domain experts. In this paper, we propose \textit{TrajAgent}, a agent framework powered by large language models (LLMs), designed to facilitate robust and efficient trajectory modeling through automation modeling. This framework leverages and optimizes diverse specialized models to address various trajectory modeling tasks across different datasets effectively. In \textit{TrajAgent}, we first develop \textit{UniEnv}, an execution environment with a unified data and model interface, to support the execution and training of various models. Building on \textit{UniEnv}, we introduce an agentic workflow designed for automatic trajectory modeling across various trajectory tasks and data. Furthermore, we introduce collaborative learning schema between LLM-based agents and small speciallized models, to enhance the performance of the whole framework effectively. Extensive experiments on four tasks using four real-world datasets demonstrate the effectiveness of \textit{TrajAgent} in automated trajectory modeling, achieving a performance improvement of 2.38\%-34.96\% over baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03079v2">Utilizing Precise and Complete Code Context to Guide LLM in Automatic False Positive Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Static Application Security Testing (SAST) tools are critical to software quality, identifying potential code issues early in development. However, they often produce false positive warnings that require manual review, slowing down development. Thus, automating false positive mitigation (FPM) is essential. The advent of Large Language Models (LLMs), with their strong abilities in natural language and code understanding, offers promising avenues for FPM. Yet current LLM-based FPM method faces two major limitations: 1. The warning-related code snippets extracted are overly broad and cluttered with irrelevant control/data flows, reducing precision; 2. Critical code contexts are missing, leading to incomplete representations that can mislead LLMs and cause inaccurate assessments. To overcome these limitations, we propose LLM4FPM , a lightweight and efficient false positive mitigation framework. It features eCPG-Slicer, which builds an extended code property graph (eCPG) to extract precise line-level code contexts for warnings. Furthermore, the integrated FARF algorithm builds a file reference graph to identify all files that are relevant to warnings in linear time. This enables eCPG-Slicer to obtain rich contextual information without resorting to expensive whole-program analysis. LLM4FPM outperforms the existing method on the Juliet dataset (F1 > 99% across various Common Weakness Enumerations) and improves label accuracy on the D2A dataset to 86%. By leveraging a lightweight open-source LLM, LLM4FPM can significantly save inspection costs up to \$2758 per run (\$0.384 per warning) on Juliet with an average inspection time of 4.7s per warning. Moreover, real-world tests on popular C/C++ projects demonstrate its practicality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13031v2">HPSS: Heuristic Prompting Strategy Search for LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Since the adoption of large language models (LLMs) for text evaluation has become increasingly prevalent in the field of natural language processing (NLP), a series of existing works attempt to optimize the prompts for LLM evaluators to improve their alignment with human judgment. However, their efforts are limited to optimizing individual factors of evaluation prompts, such as evaluation criteria or output formats, neglecting the combinatorial impact of multiple factors, which leads to insufficient optimization of the evaluation pipeline. Nevertheless, identifying well-behaved prompting strategies for adjusting multiple factors requires extensive enumeration. To this end, we comprehensively integrate 8 key factors for evaluation prompts and propose a novel automatic prompting strategy optimization method called Heuristic Prompting Strategy Search (HPSS). Inspired by the genetic algorithm, HPSS conducts an iterative search to find well-behaved prompting strategies for LLM evaluators. A heuristic function is employed to guide the search process, enhancing the performance of our algorithm. Extensive experiments across four evaluation tasks demonstrate the effectiveness of HPSS, consistently outperforming both human-designed evaluation prompts and existing automatic prompt optimization methods. Our code is available at https://github.com/thu-coai/HPSS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05551v4">FRAME: Boosting LLMs with A Four-Quadrant Multi-Stage Pretraining Strategy</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced human language understanding and generation, with pretraining data quality and organization being crucial to their performance. Multi-stage pretraining is a promising approach, but existing methods often lack quantitative criteria for data partitioning and instead rely on intuitive heuristics. In this paper, we propose the novel Four-quadRAnt Multi-stage prEtraining strategy (FRAME), guided by the established principle of organizing the pretraining process into four stages to achieve significant loss reductions four times. This principle is grounded in two key findings: first, training on high Perplexity (PPL) data followed by low PPL data, and second, training on low PPL difference (PD) data followed by high PD data, both causing the loss to drop significantly twice and performance enhancements. By partitioning data into four quadrants and strategically organizing them, FRAME achieves a remarkable 16.8% average improvement over random across MMLU and CMMLU for the 3B model, effectively boosting LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13109v2">Visualizationary: Automating Design Feedback for Visualization Designers using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 Minor Revision Status, IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      Interactive visualization editors empower users to author visualizations without writing code, but do not provide guidance on the art and craft of effective visual communication. In this paper, we explore the potential of using an off-the-shelf large language models (LLMs) to provide actionable and customized feedback to visualization designers. Our implementation, VISUALIZATIONARY, demonstrates how ChatGPT can be used for this purpose through two key components: a preamble of visualization design guidelines and a suite of perceptual filters that extract salient metrics from a visualization image. We present findings from a longitudinal user study involving 13 visualization designers-6 novices, 4 intermediates, and 3 experts-who authored a new visualization from scratch over several days. Our results indicate that providing guidance in natural language via an LLM can aid even seasoned designers in refining their visualizations. All our supplemental materials are available at https://osf.io/v7hu8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01743v3">Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Interpreting the law is always essential for the law to adapt to the ever-changing society. It is a critical and challenging task even for legal practitioners, as it requires meticulous and professional annotations and summarizations by legal experts, which are admittedly time-consuming and expensive to collect at scale. To alleviate the burden on legal experts, we propose a method for automated legal interpretation. Specifically, by emulating doctrinal legal research, we introduce a novel framework, ATRIE, to address Legal Concept Interpretation, a typical task in legal interpretation. ATRIE utilizes large language models (LLMs) to AuTomatically Retrieve concept-related information, Interpret legal concepts, and Evaluate generated interpretations, eliminating dependence on legal experts. ATRIE comprises a legal concept interpreter and a legal concept interpretation evaluator. The interpreter uses LLMs to retrieve relevant information from previous cases and interpret legal concepts. The evaluator uses performance changes on Legal Concept Entailment, a downstream task we propose, as a proxy of interpretation quality. Automated and multifaceted human evaluations indicate that the quality of our interpretations is comparable to those written by legal experts, with superior comprehensiveness and readability. Although there remains a slight gap in accuracy, it can already assist legal practitioners in improving the efficiency of legal interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13173v2">A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 Accepted to ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across diverse tasks and languages. In this study, we focus on natural language understanding in three classical languages -- Sanskrit, Ancient Greek and Latin -- to investigate the factors affecting cross-lingual zero-shot generalization. First, we explore named entity recognition and machine translation into English. While LLMs perform equal to or better than fine-tuned baselines on out-of-domain data, smaller models often struggle, especially with niche or abstract entity types. In addition, we concentrate on Sanskrit by presenting a factoid question-answering (QA) dataset and show that incorporating context via retrieval-augmented generation approach significantly boosts performance. In contrast, we observe pronounced performance drops for smaller LLMs across these QA tasks. These results suggest model scale as an important factor influencing cross-lingual generalization. Assuming that models used such as GPT-4o and Llama-3.1 are not instruction fine-tuned on classical languages, our findings provide insights into how LLMs may generalize on these languages and their consequent utility in classical studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22086v2">iDSE: Navigating Design Space Exploration in High-Level Synthesis Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      High-Level Synthesis (HLS) serves as an agile hardware development tool that streamlines the circuit design by abstracting the register transfer level into behavioral descriptions, while allowing designers to customize the generated microarchitectures through optimization directives. However, the combinatorial explosion of possible directive configurations yields an intractable design space. Traditional design space exploration (DSE) methods, despite adopting heuristics or constructing predictive models to accelerate Pareto-optimal design acquisition, still suffer from prohibitive exploration costs and suboptimal results. Addressing these concerns, we introduce iDSE, the first LLM-aided DSE framework that leverages HLS design quality perception to effectively navigate the design space. iDSE intelligently pruns the design space to guide LLMs in calibrating representative initial sampling designs, expediting convergence toward the Pareto front. By exploiting the convergent and divergent thinking patterns inherent in LLMs for hardware optimization, iDSE achieves multi-path refinement of the design quality and diversity. Extensive experiments demonstrate that iDSE outperforms heuristic-based DSE methods by 5.1$\times$$\sim$16.6$\times$ in proximity to the reference Pareto front, matching NSGA-II with only 4.6% of the explored designs. Our work demonstrates the transformative potential of LLMs in scalable and efficient HLS design optimization, offering new insights into multiobjective optimization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18626v4">The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 Accepted to the Main Track of ACL 2025
    </div>
    <details class="paper-abstract">
      We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies. Warning: this paper contains examples of unethical inquiries used solely for research purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11934v4">Stepwise Reasoning Error Disruption Attack of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications. Our code is available at: https://github.com/Applied-Machine-Learning-Lab/SEED-Attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10304v2">LLM-Powered Test Case Generation for Detecting Bugs in Plausible Programs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 Accepted by the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025) Main Track
    </div>
    <details class="paper-abstract">
      Detecting tricky bugs in plausible programs, those that pass existing test suites yet still contain bugs, remains a significant challenge in software testing. To address this problem, we propose TrickCatcher, an LLM-powered approach to generating test cases for uncovering bugs in plausible programs. TrickCatcher operates in three stages: First, it uses an LLM to generate program variants based on the program under test (PUT) and its specification. Second, it employs an LLM to construct an input generator from the specification for producing test inputs. Finally, these inputs are executed on both the PUT and its program variants to detect inconsistencies in their outputs. We evaluate TrickCatcher on two datasets, TrickyBugs and EvalPlus, which include 366 human-written and 151 AI-generated plausible programs with tricky bugs. TrickCatcher achieves recall, precision, and F1 scores that are 1.80x, 2.65x, and 1.66x those of the state-of-the-art baselines, respectively. Code and data used are available at https://github.com/RinCloud/TrickCatcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11093v2">RAEmoLLM: Retrieval Augmented LLMs for Cross-Domain Misinformation Detection Using In-Context Learning Based on Emotional Information</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 Accepted by ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Misinformation is prevalent in various fields such as education, politics, health, etc., causing significant harm to society. However, current methods for cross-domain misinformation detection rely on effort- and resource-intensive fine-tuning and complex model structures. With the outstanding performance of LLMs, many studies have employed them for misinformation detection. Unfortunately, they focus on in-domain tasks and do not incorporate significant sentiment and emotion features (which we jointly call {\em affect}). In this paper, we propose RAEmoLLM, the first retrieval augmented (RAG) LLMs framework to address cross-domain misinformation detection using in-context learning based on affective information. RAEmoLLM includes three modules. (1) In the index construction module, we apply an emotional LLM to obtain affective embeddings from all domains to construct a retrieval database. (2) The retrieval module uses the database to recommend top K examples (text-label pairs) from source domain data for target domain contents. (3) These examples are adopted as few-shot demonstrations for the inference module to process the target domain content. The RAEmoLLM can effectively enhance the general performance of LLMs in cross-domain misinformation detection tasks through affect-based retrieval, without fine-tuning. We evaluate our framework on three misinformation benchmarks. Results show that RAEmoLLM achieves significant improvements compared to the other few-shot methods on three datasets, with the highest increases of 15.64%, 31.18%, and 15.73% respectively. This project is available at https://github.com/lzw108/RAEmoLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19165v2">OrgAccess: A Benchmark for Role Based Access Control in Organization Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 56 Pages
    </div>
    <details class="paper-abstract">
      Role-based access control (RBAC) and hierarchical structures are foundational to how information flows and decisions are made within virtually all organizations. As the potential of Large Language Models (LLMs) to serve as unified knowledge repositories and intelligent assistants in enterprise settings becomes increasingly apparent, a critical, yet under explored, challenge emerges: \textit{can these models reliably understand and operate within the complex, often nuanced, constraints imposed by organizational hierarchies and associated permissions?} Evaluating this crucial capability is inherently difficult due to the proprietary and sensitive nature of real-world corporate data and access control policies. We introduce a synthetic yet representative \textbf{OrgAccess} benchmark consisting of 40 distinct types of permissions commonly relevant across different organizational roles and levels. We further create three types of permissions: 40,000 easy (1 permission), 10,000 medium (3-permissions tuple), and 20,000 hard (5-permissions tuple) to test LLMs' ability to accurately assess these permissions and generate responses that strictly adhere to the specified hierarchical rules, particularly in scenarios involving users with overlapping or conflicting permissions. Our findings reveal that even state-of-the-art LLMs struggle significantly to maintain compliance with role-based structures, even with explicit instructions, with their performance degrades further when navigating interactions involving two or more conflicting permissions. Specifically, even \textbf{GPT-4.1 only achieves an F1-Score of 0.27 on our hardest benchmark}. This demonstrates a critical limitation in LLMs' complex rule following and compositional reasoning capabilities beyond standard factual or STEM-based benchmarks, opening up a new paradigm for evaluating their fitness for practical, structured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11711v3">The Hitchhiker's Guide to Program Analysis, Part II: Deep Thoughts by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      Static analysis plays a crucial role in software vulnerability detection, yet faces a persistent precision-scalability tradeoff. In large codebases like the Linux kernel, traditional static analysis tools often generate excessive false positives due to simplified vulnerability modeling and overapproximation of path and data constraints. While large language models (LLMs) demonstrate promising code understanding capabilities, their direct application to program analysis remains unreliable due to inherent reasoning limitations. We introduce BugLens, a post-refinement framework that significantly enhances static analysis precision for bug detection. BugLens guides LLMs through structured reasoning steps to assess security impact and validate constraints from the source code. When evaluated on Linux kernel taint-style bugs detected by static analysis tools, BugLens improves precision approximately 7-fold (from 0.10 to 0.72), substantially reducing false positives while uncovering four previously unreported vulnerabilities. Our results demonstrate that a well-structured, fully automated LLM-based workflow can effectively complement and enhance traditional static analysis techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04420v4">KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-31
      | 💬 Accepted by ICML25. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we theoretically analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is generally more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 21.25\% compared with KIVI-KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14070v2">Enhancing LLMs via High-Knowledge Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-05-31
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) is intrinsically linked to the quality of its training data. Although several studies have proposed methods for high-quality data selection, they do not consider the importance of knowledge richness in text corpora. In this paper, we propose a novel and gradient-free High-Knowledge Scorer (HKS) to select high-quality data from the dimension of knowledge, to alleviate the problem of knowledge scarcity in the pre-trained corpus. We propose a comprehensive multi-domain knowledge element pool and introduce knowledge density and coverage as metrics to assess the knowledge content of the text. Based on this, we propose a comprehensive knowledge scorer to select data with intensive knowledge, which can also be utilized for domain-specific high-knowledge data selection by restricting knowledge elements to the specific domain. We train models on a high-knowledge bilingual dataset, and experimental results demonstrate that our scorer improves the model's performance in knowledge-intensive and general comprehension tasks, and is effective in enhancing both the generic and domain-specific capabilities of the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24878v1">Open CaptchaWorld: A Comprehensive Web-based Platform for Testing and Benchmarking Multimodal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Code at: https://github.com/MetaAgentX/OpenCaptchaWorld
    </div>
    <details class="paper-abstract">
      CAPTCHAs have been a critical bottleneck for deploying web agents in real-world applications, often blocking them from completing end-to-end automation tasks. While modern multimodal LLM agents have demonstrated impressive performance in static perception tasks, their ability to handle interactive, multi-step reasoning challenges like CAPTCHAs is largely untested. To address this gap, we introduce Open CaptchaWorld, the first web-based benchmark and platform specifically designed to evaluate the visual reasoning and interaction capabilities of MLLM-powered agents through diverse and dynamic CAPTCHA puzzles. Our benchmark spans 20 modern CAPTCHA types, totaling 225 CAPTCHAs, annotated with a new metric we propose: CAPTCHA Reasoning Depth, which quantifies the number of cognitive and motor steps required to solve each puzzle. Experimental results show that humans consistently achieve near-perfect scores, state-of-the-art MLLM agents struggle significantly, with success rates at most 40.0% by Browser-Use Openai-o3, far below human-level performance, 93.3%. This highlights Open CaptchaWorld as a vital benchmark for diagnosing the limits of current multimodal agents and guiding the development of more robust multimodal reasoning systems. Code and Data are available at this https URL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24871v1">MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Project Webpage: https://modomodo-rl.github.io/
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a powerful paradigm for post-training large language models (LLMs), achieving state-of-the-art performance on tasks with structured, verifiable answers. Applying RLVR to Multimodal LLMs (MLLMs) presents significant opportunities but is complicated by the broader, heterogeneous nature of vision-language tasks that demand nuanced visual, logical, and spatial capabilities. As such, training MLLMs using RLVR on multiple datasets could be beneficial but creates challenges with conflicting objectives from interaction among diverse datasets, highlighting the need for optimal dataset mixture strategies to improve generalization and reasoning. We introduce a systematic post-training framework for Multimodal LLM RLVR, featuring a rigorous data mixture problem formulation and benchmark implementation. Specifically, (1) We developed a multimodal RLVR framework for multi-dataset post-training by curating a dataset that contains different verifiable vision-language problems and enabling multi-domain online RL learning with different verifiable rewards; (2) We proposed a data mixture strategy that learns to predict the RL fine-tuning outcome from the data mixture distribution, and consequently optimizes the best mixture. Comprehensive experiments showcase that multi-domain RLVR training, when combined with mixture prediction strategies, can significantly boost MLLM general reasoning capacities. Our best mixture improves the post-trained model's accuracy on out-of-distribution benchmarks by an average of 5.24% compared to the same model post-trained with uniform data mixture, and by a total of 20.74% compared to the pre-finetuning baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24858v1">MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      A critical component in the trustworthiness of LLMs is reliable uncertainty communication, yet LLMs often use assertive language when conveying false claims, leading to over-reliance and eroded trust. We present the first systematic study of $\textit{faithful confidence calibration}$ of LLMs, benchmarking models' ability to use linguistic expressions of uncertainty that $\textit{faithfully reflect}$ their intrinsic uncertainty, across a comprehensive array of models, datasets, and prompting strategies. Our results demonstrate that LLMs largely fail at this task, and that existing interventions are insufficient: standard prompt approaches provide only marginal gains, and existing, factuality-based calibration techniques can even harm faithful calibration. To address this critical gap, we introduce MetaFaith, a novel prompt-based calibration approach inspired by human metacognition. We show that MetaFaith robustly improves faithful calibration across diverse models and task domains, enabling up to 61% improvement in faithfulness and achieving an 83% win rate over original generations as judged by humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13467v3">LlamaDuo: LLMOps Pipeline for Seamless Migration from Service LLMs to Small-Scale Local LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 The first three authors contributed equally to this work; Accepted by ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      The widespread adoption of cloud-based proprietary large language models (LLMs) has introduced significant challenges, including operational dependencies, privacy concerns, and the necessity of continuous internet connectivity. In this work, we introduce an LLMOps pipeline, "LlamaDuo", for the seamless migration of knowledge and abilities from service-oriented LLMs to smaller, locally manageable models. This pipeline is crucial for ensuring service continuity in the presence of operational failures, strict privacy policies, or offline requirements. Our LlamaDuo involves fine-tuning a small language model against the service LLM using a synthetic dataset generated by the latter. If the performance of the fine-tuned model falls short of expectations, it is automatically improved through additional fine-tuning using extra similar data generated by the service LLM. This multi-turn process guarantees that the smaller model can eventually match or even surpass the service LLM's capabilities in specific downstream tasks, offering a practical and scalable solution for managing AI deployments in constrained environments. Extensive experiments with leading-edge LLMs are conducted to demonstrate the effectiveness, adaptability, and affordability of LlamaDuo across various downstream tasks. Our pipeline implementation is available at https://github.com/deep-diver/llamaduo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24850v1">Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 27 pages, 10 figures. Code available at https://github.com/Tim-Siu/reinforcement-distillation
    </div>
    <details class="paper-abstract">
      Recent advances in model distillation demonstrate that data from advanced reasoning models (e.g., DeepSeek-R1, OpenAI's o1) can effectively transfer complex reasoning abilities to smaller, efficient student models. However, standard practices employ rejection sampling, discarding incorrect reasoning examples -- valuable, yet often underutilized data. This paper addresses the critical question: How can both positive and negative distilled reasoning traces be effectively leveraged to maximize LLM reasoning performance in an offline setting? To this end, We propose Reinforcement Distillation (REDI), a two-stage framework. Stage 1 learns from positive traces via Supervised Fine-Tuning (SFT). Stage 2 further refines the model using both positive and negative traces through our proposed REDI objective. This novel objective is a simple, reference-free loss function that outperforms established methods like DPO and SimPO in this distillation context. Our empirical evaluations demonstrate REDI's superiority over baseline Rejection Sampling SFT or SFT combined with DPO/SimPO on mathematical reasoning tasks. Notably, the Qwen-REDI-1.5B model, post-trained on just 131k positive and negative examples from the open Open-R1 dataset, achieves an 83.1% score on MATH-500 (pass@1). Its performance matches or surpasses that of DeepSeek-R1-Distill-Qwen-1.5B (a model post-trained on 800k proprietary data) across various mathematical reasoning benchmarks, establishing a new state-of-the-art for 1.5B models post-trained offline with openly available data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08972v2">RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      This paper introduces RuleArena, a novel and challenging benchmark designed to evaluate the ability of large language models (LLMs) to follow complex, real-world rules in reasoning. Covering three practical domains -- airline baggage fees, NBA transactions, and tax regulations -- RuleArena assesses LLMs' proficiency in handling intricate natural language instructions that demand long-context understanding, logical reasoning, and accurate mathematical computation. Two key attributes distinguish RuleArena from traditional rule-based reasoning benchmarks: (1) it extends beyond standard first-order logic representations, and (2) it is grounded in authentic, practical scenarios, providing insights into the suitability and reliability of LLMs for real-world applications. Our findings reveal several notable limitations in LLMs: (1) they struggle to identify and apply the appropriate rules, frequently becoming confused by similar but distinct regulations, (2) they cannot consistently perform accurate mathematical computations, even when they correctly identify the relevant rules, and (3) in general, they perform poorly in the benchmark. We also observe a significant performance boost when LLMs are provided with external tools for oracle math and logic operations. These results highlight significant challenges and promising research directions in advancing LLMs' rule-guided reasoning capabilities in real-life applications. Our codes and data are publicly available on https://github.com/skyriver-2000/RuleArena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24840v1">Vision LLMs Are Bad at Hierarchical Visual Understanding, and LLMs Are the Bottleneck</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 28 pages, 13 figures
    </div>
    <details class="paper-abstract">
      This paper reveals that many state-of-the-art large language models (LLMs) lack hierarchical knowledge about our visual world, unaware of even well-established biology taxonomies. This shortcoming makes LLMs a bottleneck for vision LLMs' hierarchical visual understanding (e.g., recognizing Anemone Fish but not Vertebrate). We arrive at these findings using about one million four-choice visual question answering (VQA) tasks constructed from six taxonomies and four image datasets. Interestingly, finetuning a vision LLM using our VQA tasks reaffirms LLMs' bottleneck effect to some extent because the VQA tasks improve the LLM's hierarchical consistency more than the vision LLM's. We conjecture that one cannot make vision LLMs understand visual concepts fully hierarchical until LLMs possess corresponding taxonomy knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02355v3">"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Quantization is a powerful tool for accelerating large language model (LLM) inference, but the accuracy-performance trade-offs across different formats remain unclear. In this paper, we conduct the most comprehensive empirical study to date, evaluating FP8, INT8, and INT4 quantization across academic benchmarks and real-world tasks on the entire Llama-3.1 model family. Through over 500,000 evaluations, our investigation yields several key findings: (1) FP8 (W8A8-FP) is effectively lossless across all model scales, (2) well-tuned INT8 (W8A8-INT) achieves surprisingly low (1-3\%) accuracy degradation, and (3) INT4 weight-only (W4A16-INT) is more competitive than expected, rivaling 8-bit quantization. Further, we investigate the optimal quantization format for different deployments by analyzing inference performance through the popular vLLM framework. Our analysis provides clear deployment recommendations: W4A16 is the most cost-efficient for synchronous setups, while W8A8 dominates in asynchronous continuous batching. For mixed workloads, the optimal choice depends on the specific use case. Our findings offer practical, data-driven guidelines for deploying quantized LLMs at scale -- ensuring the best balance between speed, efficiency, and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24830v1">Improving Reliability and Explainability of Medical Question Answering through Atomic Fact Checking in Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit extensive medical knowledge but are prone to hallucinations and inaccurate citations, which pose a challenge to their clinical adoption and regulatory compliance. Current methods, such as Retrieval Augmented Generation, partially address these issues by grounding answers in source documents, but hallucinations and low fact-level explainability persist. In this work, we introduce a novel atomic fact-checking framework designed to enhance the reliability and explainability of LLMs used in medical long-form question answering. This method decomposes LLM-generated responses into discrete, verifiable units called atomic facts, each of which is independently verified against an authoritative knowledge base of medical guidelines. This approach enables targeted correction of errors and direct tracing to source literature, thereby improving the factual accuracy and explainability of medical Q&A. Extensive evaluation using multi-reader assessments by medical experts and an automated open Q&A benchmark demonstrated significant improvements in factual accuracy and explainability. Our framework achieved up to a 40% overall answer improvement and a 50% hallucination detection rate. The ability to trace each atomic fact back to the most relevant chunks from the database provides a granular, transparent explanation of the generated responses, addressing a major gap in current medical AI applications. This work represents a crucial step towards more trustworthy and reliable clinical applications of LLMs, addressing key prerequisites for clinical application and fostering greater confidence in AI-assisted healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09284v3">SparQLe: Speech Queries to Text Translation Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that combines self-supervised speech representations with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English speech data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising approach for various speech understanding applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24826v1">LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 10 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in legal applications, current evaluation benchmarks tend to focus mainly on factual accuracy while largely neglecting important linguistic quality aspects such as clarity, coherence, and terminology. To address this gap, we propose three steps: First, we develop a regression model to evaluate the quality of legal texts based on clarity, coherence, and terminology. Second, we create a specialized set of legal questions. Third, we analyze 49 LLMs using this evaluation framework. Our analysis identifies three key findings: First, model quality levels off at 14 billion parameters, with only a marginal improvement of $2.7\%$ noted at 72 billion parameters. Second, engineering choices such as quantization and context length have a negligible impact, as indicated by statistical significance thresholds above 0.016. Third, reasoning models consistently outperform base architectures. A significant outcome of our research is the release of a ranking list and Pareto analysis, which highlight the Qwen3 series as the optimal choice for cost-performance tradeoffs. This work not only establishes standardized evaluation protocols for legal LLMs but also uncovers fundamental limitations in current training data refinement approaches. Code and models are available at: https://github.com/lyxx3rd/LegalEval-Q.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14830v2">Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (https://github.com/dannigt/mid-align).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06824v2">Combining Domain and Alignment Vectors to Achieve Better Knowledge-Safety Trade-offs in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      There is a growing interest in training domain-expert LLMs that excel in specific technical fields compared to their general-purpose instruction-tuned counterparts. However, these expert models often experience a loss in their safety abilities in the process, making them capable of generating harmful content. As a solution, we introduce an efficient and effective merging-based alignment method called \textsc{MergeAlign} that interpolates the domain and alignment vectors, creating safer domain-specific models while preserving their utility. We apply \textsc{MergeAlign} on Llama3 variants that are experts in medicine and finance, obtaining substantial alignment improvements with minimal to no degradation on domain-specific benchmarks. We study the impact of model merging through model similarity metrics and contributions of individual models being merged. We hope our findings open new research avenues and inspire more efficient development of safe expert LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24757v1">LGAR: Zero-Shot LLM-Guided Neural Ranking for Abstract Screening in Systematic Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The scientific literature is growing rapidly, making it hard to keep track of the state-of-the-art. Systematic literature reviews (SLRs) aim to identify and evaluate all relevant papers on a topic. After retrieving a set of candidate papers, the abstract screening phase determines initial relevance. To date, abstract screening methods using large language models (LLMs) focus on binary classification settings; existing question answering (QA) based ranking approaches suffer from error propagation. LLMs offer a unique opportunity to evaluate the SLR's inclusion and exclusion criteria, yet, existing benchmarks do not provide them exhaustively. We manually extract these criteria as well as research questions for 57 SLRs, mostly in the medical domain, enabling principled comparisons between approaches. Moreover, we propose LGAR, a zero-shot LLM Guided Abstract Ranker composed of an LLM based graded relevance scorer and a dense re-ranker. Our extensive experiments show that LGAR outperforms existing QA-based methods by 5-10 pp. in mean average precision. Our code and data is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24749v1">SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Low-rank gradient-based optimization methods have significantly improved memory efficiency during the training of large language models (LLMs), enabling operations within constrained hardware without sacrificing performance. However, these methods primarily emphasize memory savings, often overlooking potential acceleration in convergence due to their reliance on standard isotropic steepest descent techniques, which can perform suboptimally in the highly anisotropic landscapes typical of deep networks, particularly LLMs. In this paper, we propose SUMO (Subspace-Aware Moment-Orthogonalization), an optimizer that employs exact singular value decomposition (SVD) for moment orthogonalization within a dynamically adapted low-dimensional subspace, enabling norm-inducing steepest descent optimization steps. By explicitly aligning optimization steps with the spectral characteristics of the loss landscape, SUMO effectively mitigates approximation errors associated with commonly used methods like Newton-Schulz orthogonalization approximation. We theoretically establish an upper bound on these approximation errors, proving their dependence on the condition numbers of moments, conditions we analytically demonstrate are encountered during LLM training. Furthermore, we both theoretically and empirically illustrate that exact orthogonalization via SVD substantially improves convergence rates while reducing overall complexity. Empirical evaluations confirm that SUMO accelerates convergence, enhances stability, improves performance, and reduces memory requirements by up to 20% compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24701v1">Multi-Domain ABSA Conversation Dataset Generation via LLMs for Real-World Evaluation and Model Comparison</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 11 pages, 3 figures, 5 tables, 6th International Conference on Natural Language Computing and AI (NLCAI 2025), ISBN : 978-1-923107-59-5, Computer Science & Information Technology (CS & IT), ISSN : 2231 - 5403, Volume 15, Number 10, May 2025
    </div>
    <details class="paper-abstract">
      Aspect-Based Sentiment Analysis (ABSA) offers granular insights into opinions but often suffers from the scarcity of diverse, labeled datasets that reflect real-world conversational nuances. This paper presents an approach for generating synthetic ABSA data using Large Language Models (LLMs) to address this gap. We detail the generation process aimed at producing data with consistent topic and sentiment distributions across multiple domains using GPT-4o. The quality and utility of the generated data were evaluated by assessing the performance of three state-of-the-art LLMs (Gemini 1.5 Pro, Claude 3.5 Sonnet, and DeepSeek-R1) on topic and sentiment classification tasks. Our results demonstrate the effectiveness of the synthetic data, revealing distinct performance trade-offs among the models: DeepSeekR1 showed higher precision, Gemini 1.5 Pro and Claude 3.5 Sonnet exhibited strong recall, and Gemini 1.5 Pro offered significantly faster inference. We conclude that LLM-based synthetic data generation is a viable and flexible method for creating valuable ABSA resources, facilitating research and model evaluation without reliance on limited or inaccessible real-world labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17439v2">LEMMA: Learning from Errors for MatheMatical Advancement in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ACL'25 Findings, Code is available at https://github.com/pzs19/LEMMA
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capability in solving mathematical problems. However, existing approaches primarily focus on improving the quality of correct training data, e.g., distilling high-quality correct solutions from advanced models, neglecting the value contained in error data, potentially hindering the model's reflective ability. Though some studies attempt to leverage error data, they often involve complex mechanisms, such as Monte Carlo Tree Search (MCTS) to explore error nodes. In this work, we propose to enhance LLMs' reasoning ability by Learning from Errors for Mathematical Advancement (LEMMA). LEMMA constructs data consisting of an incorrect solution with an erroneous step and a reflection connection to a correct solution for fine-tuning. Specifically, we systematically analyze the model-generated error types and introduce an error-type grounded mistake augmentation method to collect diverse and representative errors. Correct solutions are either from fixing the errors or generating a fresh start. Through a model-aware smooth reflection connection, the erroneous solution is transferred to the correct one. By fine-tuning on the constructed dataset, the model is able to self-correct errors autonomously within the generation process without relying on external critique models. Experimental results demonstrate that LEMMA achieves significant performance improvements over other strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03621v4">JPPO++: Joint Power and Denoising-inspired Prompt Optimization for Mobile LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 arXiv admin note: text overlap with arXiv:2411.18010
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into mobile services over wireless networks to support complex user requests. This trend has led to longer prompts, which improve LLMs' performance but increase data transmission costs and require more processing time, thereby reducing overall system efficiency and negatively impacting user experience. To address these challenges, we propose Joint Prompt and Power Optimization (JPPO), a framework that jointly optimizes prompt compression and wireless transmission power for mobile LLM services. JPPO leverages a Small Language Model (SLM) deployed at edge devices to perform lightweight prompt compression, reducing communication load before transmission to the cloud-based LLM. A Deep Reinforcement Learning (DRL) agent dynamically adjusts both the compression ratio and transmission power based on network conditions and service constraints, aiming to minimize service time while preserving response fidelity. We further extend the framework to JPPO++, which introduces a denoising-inspired compression scheme. This design performs iterative prompt refinement by progressively removing less informative tokens, allowing for more aggressive yet controlled compression. Experimental results show that JPPO++ reduces service time by 17% compared to the no-compression baseline while maintaining output quality. Under compression-prioritized settings, a reduction of up to 16x in prompt length can be achieved with an acceptable loss in accuracy. Specifically, JPPO with a 16x ratio reduces total service time by approximately 42.3%, and JPPO++ further improves this reduction to 46.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24671v1">Multiple LLM Agents Debate for Equitable Cultural Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 37 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17229v2">FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as knowledge graphs consisting of facts in the form of triples. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sentence-level sampling-based methods while providing more detailed insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35.5% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only a 10.6% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content. Additionally, we contribute a new dataset for evaluating sampling-based methods - FavaMultiSamples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23473v2">EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently refuse to respond to pseudo-malicious instructions: semantically harmless input queries triggering unnecessary LLM refusals due to conservative safety alignment, significantly impairing user experience. Collecting such instructions is crucial for evaluating and mitigating over-refusals, but existing instruction curation methods, like manual creation or instruction rewriting, either lack scalability or fail to produce sufficiently diverse and effective refusal-inducing prompts. To address these limitations, we introduce EVOREFUSE, a prompt optimization approach that generates diverse pseudo-malicious instructions consistently eliciting confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm exploring the instruction space in more diverse directions than existing methods via mutation strategies and recombination, and iteratively evolves seed instructions to maximize evidence lower bound on LLM refusal probability. Using EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582 pseudo-malicious instructions that outperforms the next-best benchmark with 140.41% higher average refusal triggering rate across 9 LLMs, 34.86% greater lexical diversity, and 40.03% improved LLM response confidence scores; and EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with responses for supervised and preference-based alignment training. LLAMA3.1-8B-INSTRUCT supervisedly fine-tuned on EVOREFUSE-ALIGN achieves up to 14.31% fewer over-refusals than models trained on the second-best alignment dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals models trigger over-refusals by overly focusing on sensitive keywords while ignoring broader context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24658v1">Can LLMs and humans be friends? Uncovering factors affecting human-AI intimacy formation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 30 pages, 2figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used in conversational roles, yet little is known about how intimacy emerges in human-LLM interactions. Although previous work emphasized the importance of self-disclosure in human-chatbot interaction, it is questionable whether gradual and reciprocal self-disclosure is also helpful in human-LLM interaction. Thus, this study examined three possible aspects contributing to intimacy formation: gradual self-disclosure, reciprocity, and naturalness. Study 1 explored the impact of mutual, gradual self-disclosure with 29 users and a vanilla LLM. Study 2 adopted self-criticism methods for more natural responses and conducted a similar experiment with 53 users. Results indicate that gradual self-disclosure significantly enhances perceived social intimacy, regardless of persona reciprocity. Moreover, participants perceived utterances generated with self-criticism as more natural compared to those of vanilla LLMs; self-criticism fostered higher intimacy in early stages. Also, we observed that excessive empathetic expressions occasionally disrupted immersion, pointing to the importance of response calibration during intimacy formation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21701v2">Do We Know What LLMs Don't Know? A Study of Consistency in Knowledge Probing</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The reliability of large language models (LLMs) is greatly compromised by their tendency to hallucinate, underscoring the need for precise identification of knowledge gaps within LLMs. Various methods for probing such gaps exist, ranging from calibration-based to prompting-based methods. To evaluate these probing methods, in this paper, we propose a new process based on using input variations and quantitative metrics. Through this, we expose two dimensions of inconsistency in knowledge gap probing. (1) Intra-method inconsistency: Minimal non-semantic perturbations in prompts lead to considerable variance in detected knowledge gaps within the same probing method; e.g., the simple variation of shuffling answer options can decrease agreement to around 40%. (2) Cross-method inconsistency: Probing methods contradict each other on whether a model knows the answer. Methods are highly inconsistent -- with decision consistency across methods being as low as 7% -- even though the model, dataset, and prompt are all the same. These findings challenge existing probing methods and highlight the urgent need for perturbation-robust probing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21693v2">MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge. We publicly release our code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24643v1">Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for analyzing sorting algorithms in pairwise ranking prompting (PRP), re-centering the cost model around LLM inferences rather than traditional pairwise comparisons. While classical metrics based on comparison counts have traditionally been used to gauge efficiency, our analysis reveals that expensive LLM inferences overturn these predictions; accordingly, our framework encourages strategies such as batching and caching to mitigate inference costs. We show that algorithms optimal in the classical setting can lose efficiency when LLM inferences dominate the cost under certain optimizations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01986v2">TuRTLe: A Unified Evaluation of LLMs for RTL Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The rapid advancements in LLMs have driven the adoption of generative AI in various domains, including Electronic Design Automation (EDA). Unlike traditional software development, EDA presents unique challenges, as generated RTL code must not only be syntactically correct and functionally accurate but also synthesizable by hardware generators while meeting performance, power, and area constraints. These additional requirements introduce complexities that existing code-generation benchmarks often fail to capture, limiting their effectiveness in evaluating LLMs for RTL generation. To address this gap, we propose TuRTLe, a unified evaluation framework designed to systematically assess LLMs across key RTL generation tasks. TuRTLe integrates multiple existing benchmarks and automates the evaluation process, enabling a comprehensive assessment of LLM performance in syntax correctness, functional correctness, synthesis, PPA optimization, and exact line completion. Using this framework, we benchmark a diverse set of open LLMs and analyze their strengths and weaknesses in EDA-specific tasks. Our results show that reasoning-based models, such as DeepSeek R1, consistently outperform others across multiple evaluation criteria, but at the cost of increased computational overhead and inference latency. Additionally, base models are better suited in module completion tasks, while instruct-tuned models perform better in specification-to-RTL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24622v1">Random Rule Forest (RRF): Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Predicting startup success requires models that are both accurate and interpretable. We present a lightweight ensemble framework that combines YES/NO questions generated by large language models (LLMs), forming a transparent decision-making system. Each question acts as a weak heuristic, and by filtering, ranking, and aggregating them through a threshold-based voting mechanism, we construct a strong ensemble predictor. On a test set where 10% of startups are classified as successful, our approach achieves a precision rate of 50%, representing a 5x improvement over random selection, while remaining fully transparent. When we incorporate expert-guided heuristics into the generation process, performance improves further to 54% precision. These results highlight the value of combining LLM reasoning with human insight and demonstrate that simple, interpretable ensembles can support high-stakes decisions in domains such as venture capital (VC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24616v1">Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 179 pages
    </div>
    <details class="paper-abstract">
      We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09823v3">NativQA: Multilingual Culturally-Aligned Natural Query for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 LLMs, Native, Multilingual, Language Diversity, Contextual Understanding, Minority Languages, Culturally Informed, Foundation Models, Large Language Models
    </div>
    <details class="paper-abstract">
      Natural Question Answering (QA) datasets play a crucial role in evaluating the capabilities of large language models (LLMs), ensuring their effectiveness in real-world applications. Despite the numerous QA datasets that have been developed and some work has been done in parallel, there is a notable lack of a framework and large scale region-specific datasets queried by native users in their own languages. This gap hinders the effective benchmarking and the development of fine-tuned models for regional and cultural specificities. In this study, we propose a scalable, language-independent framework, NativQA, to seamlessly construct culturally and regionally aligned QA datasets in native languages, for LLM evaluation and tuning. We demonstrate the efficacy of the proposed framework by designing a multilingual natural QA dataset, MultiNativQA, consisting of ~64k manually annotated QA pairs in seven languages, ranging from high to extremely low resource, based on queries from native speakers from 9 regions covering 18 topics. We benchmark open- and closed-source LLMs with the MultiNativQA dataset. We made the MultiNativQA dataset(https://huggingface.co/datasets/QCRI/MultiNativQA), and other experimental scripts(https://gitlab.com/nativqa/multinativqa) publicly available for the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12929v2">Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). Flow-of-Options enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic framework developed for autonomously solving Machine Learning (ML) tasks. FoO enforces diversity in LLM solutions through compressed and interpretable task representations, resulting in improvements of 38.2% - 69.2% on standard data science tasks, and 37.4% - 47.9% on therapeutic chemistry tasks, as compared to state-of-the-art baselines. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications. Going beyond tabular classification and regression, we show the broader applicability of our FoO-based agentic system to tasks such as reinforcement learning and image generation. Our code is open-sourced at: https://github.com/flagshippioneering/Flow-of-Options.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18919v2">TheaterGen: Character Management with LLM for Consistent Multi-turn Image Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion models can generate high-quality and stunning images from text. However, multi-turn image generation, which is of high demand in real-world scenarios, still faces challenges in maintaining semantic consistency between images and texts, as well as contextual consistency of the same subject across multiple interactive turns. To address this issue, we introduce TheaterGen, a training-free framework that integrates large language models (LLMs) and text-to-image (T2I) models to provide the capability of multi-turn image generation. Within this framework, LLMs, acting as a "Screenwriter", engage in multi-turn interaction, generating and managing a standardized prompt book that encompasses prompts and layout designs for each character in the target image. Based on these, Theatergen generate a list of character images and extract guidance information, akin to the "Rehearsal". Subsequently, through incorporating the prompt book and guidance information into the reverse denoising process of T2I diffusion models, Theatergen generate the final image, as conducting the "Final Performance". With the effective management of prompt books and character images, TheaterGen significantly improves semantic and contextual consistency in synthesized images. Furthermore, we introduce a dedicated benchmark, CMIGBench (Consistent Multi-turn Image Generation Benchmark) with 8000 multi-turn instructions. Different from previous multi-turn benchmarks, CMIGBench does not define characters in advance. Both the tasks of story generation and multi-turn editing are included on CMIGBench for comprehensive evaluation. Extensive experimental results show that TheaterGen outperforms state-of-the-art methods significantly. It raises the performance bar of the cutting-edge Mini DALLE 3 model by 21% in average character-character similarity and 19% in average text-image similarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24575v1">NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to the main track of ACL 2025
    </div>
    <details class="paper-abstract">
      Summarizing long-form narratives--such as books, movies, and TV scripts--requires capturing intricate plotlines, character interactions, and thematic coherence, a task that remains challenging for existing LLMs. We introduce NexusSum, a multi-agent LLM framework for narrative summarization that processes long-form text through a structured, sequential pipeline--without requiring fine-tuning. Our approach introduces two key innovations: (1) Dialogue-to-Description Transformation: A narrative-specific preprocessing method that standardizes character dialogue and descriptive text into a unified format, improving coherence. (2) Hierarchical Multi-LLM Summarization: A structured summarization pipeline that optimizes chunk processing and controls output length for accurate, high-quality summaries. Our method establishes a new state-of-the-art in narrative summarization, achieving up to a 30.0% improvement in BERTScore (F1) across books, movies, and TV scripts. These results demonstrate the effectiveness of multi-agent LLMs in handling long-form content, offering a scalable approach for structured summarization in diverse storytelling domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11277v2">From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit excellent performance in natural language processing (NLP), but remain highly sensitive to the quality of input queries, especially when these queries contain misleading or inaccurate information. Existing methods focus on correcting the output, but they often overlook the potential of improving the ability of LLMs to detect and correct misleading content in the input itself. In this paper, we propose a novel three-stage fine-tuning method that enhances the ability of LLMs to detect and correct misleading information in the input, further improving response accuracy and reducing hallucinations. Specifically, the three stages include (1) training LLMs to identify misleading information, (2) training LLMs to correct the misleading information using built-in or external knowledge, and (3) training LLMs to generate accurate answers based on the corrected queries. To evaluate our method, we conducted experiments on three datasets for the hallucination detection task and the question answering~(QA) task, as well as two datasets containing misleading information that we constructed. The experimental results demonstrate that our method significantly improves the accuracy and factuality of LLM responses, while also enhancing the ability to detect hallucinations and reducing the generation of hallucinations in the output, particularly when the query contains misleading information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24553v1">CREFT: Sequential Multi-Agent LLM for Character Relation Extraction</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Understanding complex character relations is crucial for narrative analysis and efficient script evaluation, yet existing extraction methods often fail to handle long-form narratives with nuanced interactions. To address this challenge, we present CREFT, a novel sequential framework leveraging specialized Large Language Model (LLM) agents. First, CREFT builds a base character graph through knowledge distillation, then iteratively refines character composition, relation extraction, role identification, and group assignments. Experiments on a curated Korean drama dataset demonstrate that CREFT significantly outperforms single-agent LLM baselines in both accuracy and completeness. By systematically visualizing character networks, CREFT streamlines narrative comprehension and accelerates script review -- offering substantial benefits to the entertainment, publishing, and educational sectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11844v3">BaxBench: Can LLMs Generate Correct and Secure Backends?</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Automatic program generation has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 62% on code correctness; (ii) on average, we could successfully execute security exploits on around half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08120v3">DeepSeek-R1 vs. o3-mini: How Well can Reasoning LLMs Evaluate MT and Summarization?</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Reasoning-enabled large language models (LLMs) excel in logical tasks, yet their utility for evaluating natural language generation remains unexplored. This study systematically compares reasoning LLMs with non-reasoning counterparts across machine translation and text summarization evaluation tasks. We evaluate eight models spanning state-of-the-art reasoning models (DeepSeek-R1, OpenAI o3), their distilled variants (8B-70B parameters), and equivalent non-reasoning LLMs. Experiments on WMT23 and SummEval benchmarks reveal architecture and task-dependent benefits: OpenAI o3-mini models show improved performance with increased reasoning on MT, while DeepSeek-R1 and generally underperforms compared to its non-reasoning variant except in summarization consistency evaluation. Correlation analysis demonstrates that reasoning token usage correlates with evaluation quality only in specific models, while almost all models generally allocate more reasoning tokens when identifying more quality issues. Distillation maintains reasonable performance up to 32B parameter models but degrades substantially at 8B scale. This work provides the first assessment of reasoning LLMs for NLG evaluation and comparison to non-reasoning models. We share our code to facilitate further research: https://github.com/NL2G/reasoning-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13942v2">A Rose by Any Other Name: LLM-Generated Explanations Are Good Proxies for Human Explanations to Collect Label Distributions on NLI</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted by ACL 2025 Findings, 25 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Disagreement in human labeling is ubiquitous, and can be captured in human judgment distributions (HJDs). Recent research has shown that explanations provide valuable information for understanding human label variation (HLV) and large language models (LLMs) can approximate HJD from a few human-provided label-explanation pairs. However, collecting explanations for every label is still time-consuming. This paper examines whether LLMs can be used to replace humans in generating explanations for approximating HJD. Specifically, we use LLMs as annotators to generate model explanations for a few given human labels. We test ways to obtain and combine these label-explanations with the goal to approximate human judgment distributions. We further compare the resulting human with model-generated explanations, and test automatic and human explanation selection. Our experiments show that LLM explanations are promising for NLI: to estimate HJDs, generated explanations yield comparable results to human's when provided with human labels. Importantly, our results generalize from datasets with human explanations to i) datasets where they are not available and ii) challenging out-of-distribution test sets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01349v2">Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized product recommenders, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making such manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive evaluation across models of varying scale, we find that certain biases, such as social proof, consistently boost product recommendation rate and ranking, while others, like scarcity and exclusivity, surprisingly reduce visibility. Our results demonstrate that cognitive biases are deeply embedded in state-of-the-art LLMs, leading to highly unpredictable behavior in product recommendations and posing significant challenges for effective mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24539v1">Localizing Persona Representations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      We present a study on how and where personas -- defined by distinct sets of human characteristics, values, and beliefs -- are encoded in the representation space of large language models (LLMs). Using a range of dimension reduction and pattern recognition methods, we first identify the model layers that show the greatest divergence in encoding these representations. We then analyze the activations within a selected layer to examine how specific personas are encoded relative to others, including their shared and distinct embedding spaces. We find that, across multiple pre-trained decoder-only LLMs, the analyzed personas show large differences in representation space only within the final third of the decoder layers. We observe overlapping activations for specific ethical perspectives -- such as moral nihilism and utilitarianism -- suggesting a degree of polysemy. In contrast, political ideologies like conservatism and liberalism appear to be represented in more distinct regions. These findings help to improve our understanding of how LLMs internally represent information and can inform future efforts in refining the modulation of specific human traits in LLM outputs. Warning: This paper includes potentially offensive sample statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24532v1">DEEPQUESTION: Systematic Generation of Real-World Challenges for Evaluating LLMs Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      LLMs often excel on standard benchmarks but falter on real-world tasks. We introduce DeepQuestion, a scalable automated framework that augments existing datasets based on Bloom's taxonomy and creates novel questions that trace original solution paths to probe evaluative and creative skills. Extensive experiments across ten open-source and proprietary models, covering both general-purpose and reasoning LLMs, reveal substantial performance drops (even up to 70% accuracy loss) on higher-order tasks, underscoring persistent gaps in deep reasoning. Our work highlights the need for cognitively diverse benchmarks to advance LLM progress. DeepQuestion and related datasets will be released upon acceptance of the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24511v1">Can Slow-thinking LLMs Reason Over Time? Empirical Studies in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Time series forecasting (TSF) is a fundamental and widely studied task, spanning methods from classical statistical approaches to modern deep learning and multimodal language modeling. Despite their effectiveness, these methods often follow a fast thinking paradigm emphasizing pattern extraction and direct value mapping, while overlooking explicit reasoning over temporal dynamics and contextual dependencies. Meanwhile, emerging slow-thinking LLMs (e.g., ChatGPT-o1, DeepSeek-R1) have demonstrated impressive multi-step reasoning capabilities across diverse domains, suggesting a new opportunity for reframing TSF as a structured reasoning task. This motivates a key question: can slow-thinking LLMs effectively reason over temporal patterns to support time series forecasting, even in zero-shot manner? To investigate this, in this paper, we propose TimeReasoner, an extensive empirical study that formulates TSF as a conditional reasoning task. We design a series of prompting strategies to elicit inference-time reasoning from pretrained slow-thinking LLMs and evaluate their performance across diverse TSF benchmarks. Our findings reveal that slow-thinking LLMs exhibit non-trivial zero-shot forecasting capabilities, especially in capturing high-level trends and contextual shifts. While preliminary, our study surfaces important insights into the reasoning behaviors of LLMs in temporal domains highlighting both their potential and limitations. We hope this work catalyzes further research into reasoning-based forecasting paradigms and paves the way toward more interpretable and generalizable TSF frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24500v1">TimeHC-RL: Temporal-aware Hierarchical Cognitive Reinforcement Learning for Enhancing LLMs' Social Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 22 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have made significant progress in IQ-related domains that require careful thinking, such as mathematics and coding. However, enhancing LLMs' cognitive development in social domains, particularly from a post-training perspective, remains underexplored. Recognizing that the social world follows a distinct timeline and requires a richer blend of cognitive modes (from intuitive reactions (System 1) and surface-level thinking to deliberate thinking (System 2)) than mathematics, which primarily relies on System 2 cognition (careful, step-by-step reasoning), we introduce Temporal-aware Hierarchical Cognitive Reinforcement Learning (TimeHC-RL) for enhancing LLMs' social intelligence. In our experiments, we systematically explore improving LLMs' social intelligence and validate the effectiveness of the TimeHC-RL method, through five other post-training paradigms and two test-time intervention paradigms on eight datasets with diverse data patterns. Experimental results reveal the superiority of our proposed TimeHC-RL method compared to the widely adopted System 2 RL method. It gives the 7B backbone model wings, enabling it to rival the performance of advanced models like DeepSeek-R1 and OpenAI-O3. Additionally, the systematic exploration from post-training and test-time interventions perspectives to improve LLMs' social intelligence has uncovered several valuable insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02839v4">An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to Findings of ACL2025
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing trend of utilizing Large Language Model (LLM) to evaluate the quality of other LLMs. Many studies have fine-tuned judge models based on open-source LLMs for evaluation. While the fine-tuned judge models are claimed to achieve comparable evaluation capability with GPT-4, in this work, we conduct an empirical study of LLM-as-a-Judge. Our findings indicate that although the fine-tuned judge models achieve high performance on in-domain test sets, even surpassing GPT-4, they underperform GPT-4 across several dimensions, including generalizability, fairness and adaptability. We also reveal that the fine-tuned judge model inherently operates as a task-specific classifier, consequently imposing the limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24493v1">MELT: Towards Automated Multimodal Emotion Data Annotation by Leveraging LLM Embedded Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Although speech emotion recognition (SER) has advanced significantly with deep learning, annotation remains a major hurdle. Human annotation is not only costly but also subject to inconsistencies annotators often have different preferences and may lack the necessary contextual knowledge, which can lead to varied and inaccurate labels. Meanwhile, Large Language Models (LLMs) have emerged as a scalable alternative for annotating text data. However, the potential of LLMs to perform emotional speech data annotation without human supervision has yet to be thoroughly investigated. To address these problems, we apply GPT-4o to annotate a multimodal dataset collected from the sitcom Friends, using only textual cues as inputs. By crafting structured text prompts, our methodology capitalizes on the knowledge GPT-4o has accumulated during its training, showcasing that it can generate accurate and contextually relevant annotations without direct access to multimodal inputs. Therefore, we propose MELT, a multimodal emotion dataset fully annotated by GPT-4o. We demonstrate the effectiveness of MELT by fine-tuning four self-supervised learning (SSL) backbones and assessing speech emotion recognition performance across emotion datasets. Additionally, our subjective experiments\' results demonstrate a consistence performance improvement on SER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11147v2">RaaS: Reasoning-Aware Attention Sparsity for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities across various domains, with recent advancements in challenging reasoning tasks such as mathematics and programming. However, solving reasoning tasks often requires an LLM to generate long sequences, incurring $O(N)$ time and memory complexities per token, where $N$ is the current sequence length. To reduce complexities, existing sparsity-based algorithms propose to retain Key-Value (KV) vectors, the intermediate representations of only the most critical tokens. However, these algorithms struggle with the "impossible trinity" of accuracy, time, and memory. For example, the state-of-the-art algorithm, Quest, achieves high accuracy with $O(L)$ time but $O(N)$ memory ($L$ is the cache budget, $L \ll N$). To address the "impossible trinity", in this paper, we identify a new attention pattern during the decode stage of reasoning tasks, where milestone tokens (analogous to lemmas in mathematical proofs) emerge, are utilized, and then become unimportant afterward. Based on this pattern, we propose a new algorithm RaaS that identifies milestone tokens and retains their KV vectors until they are no longer needed, achieving high accuracy with $O(L)$ time and $O(L)$ memory complexities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24479v1">Leveraging Knowledge Graphs and LLMs for Structured Generation of Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The rapid spread of misinformation, further amplified by recent advances in generative AI, poses significant threats to society, impacting public opinion, democratic stability, and national security. Understanding and proactively assessing these threats requires exploring methodologies that enable structured and scalable misinformation generation. In this paper, we propose a novel approach that leverages knowledge graphs (KGs) as structured semantic resources to systematically generate fake triplets. By analyzing the structural properties of KGs, such as the distance between entities and their predicates, we identify plausibly false relationships. These triplets are then used to guide large language models (LLMs) in generating misinformation statements with varying degrees of credibility. By utilizing structured semantic relationships, our deterministic approach produces misinformation inherently challenging for humans to detect, drawing exclusively upon publicly available KGs (e.g., WikiGraphs). Additionally, we investigate the effectiveness of LLMs in distinguishing between genuine and artificially generated misinformation. Our analysis highlights significant limitations in current LLM-based detection methods, underscoring the necessity for enhanced detection strategies and a deeper exploration of inherent biases in generative models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24478v1">Optimizing the Interface Between Knowledge Graphs and LLMs for Complex Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 This is a preliminary version. A revised and expanded version is in preparation
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) with Knowledge Graphs (KGs) results in complex systems with numerous hyperparameters that directly affect performance. While such systems are increasingly common in retrieval-augmented generation, the role of systematic hyperparameter optimization remains underexplored. In this paper, we study this problem in the context of Cognee, a modular framework for end-to-end KG construction and retrieval. Using three multi-hop QA benchmarks (HotPotQA, TwoWikiMultiHop, and MuSiQue) we optimize parameters related to chunking, graph construction, retrieval, and prompting. Each configuration is scored using established metrics (exact match, F1, and DeepEval's LLM-based correctness metric). Our results demonstrate that meaningful gains can be achieved through targeted tuning. While the gains are consistent, they are not uniform, with performance varying across datasets and metrics. This variability highlights both the value of tuning and the limitations of standard evaluation measures. While demonstrating the immediate potential of hyperparameter tuning, we argue that future progress will depend not only on architectural advances but also on clearer frameworks for optimization and evaluation in complex, modular systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16520v3">Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24458v1">SEAR: A Multimodal Dataset for Analyzing AR-LLM-Driven Social Engineering Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The SEAR Dataset is a novel multimodal resource designed to study the emerging threat of social engineering (SE) attacks orchestrated through augmented reality (AR) and multimodal large language models (LLMs). This dataset captures 180 annotated conversations across 60 participants in simulated adversarial scenarios, including meetings, classes and networking events. It comprises synchronized AR-captured visual/audio cues (e.g., facial expressions, vocal tones), environmental context, and curated social media profiles, alongside subjective metrics such as trust ratings and susceptibility assessments. Key findings reveal SEAR's alarming efficacy in eliciting compliance (e.g., 93.3% phishing link clicks, 85% call acceptance) and hijacking trust (76.7% post-interaction trust surge). The dataset supports research in detecting AR-driven SE attacks, designing defensive frameworks, and understanding multimodal adversarial manipulation. Rigorous ethical safeguards, including anonymization and IRB compliance, ensure responsible use. The SEAR dataset is available at https://github.com/INSLabCN/SEAR-Dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24451v1">LPASS: Linear Probes as Stepping Stones for vulnerability detection using compressed LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being extensively used for cybersecurity purposes. One of them is the detection of vulnerable codes. For the sake of efficiency and effectiveness, compression and fine-tuning techniques are being developed, respectively. However, they involve spending substantial computational efforts. In this vein, we analyse how Linear Probes (LPs) can be used to provide an estimation on the performance of a compressed LLM at an early phase -- before fine-tuning. We also show their suitability to set the cut-off point when applying layer pruning compression. Our approach, dubbed $LPASS$, is applied in BERT and Gemma for the detection of 12 of MITRE's Top 25 most dangerous vulnerabilities on 480k C/C++ samples. LPs can be computed in 142.97 s. and provide key findings: (1) 33.3 \% and 72.2\% of layers can be removed, respectively, with no precision loss; (2) they provide an early estimate of the post-fine-tuning and post-compression model effectiveness, with 3\% and 8.68\% as the lowest and average precision errors, respectively. $LPASS$-based LLMs outperform the state of the art, reaching 86.9\% of accuracy in multi-class vulnerability detection. Interestingly, $LPASS$-based compressed versions of Gemma outperform the original ones by 1.6\% of F1-score at a maximum while saving 29.4 \% and 23.8\% of training and inference time and 42.98\% of model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24409v1">LLMs Are Globally Multilingual Yet Locally Monolingual: Exploring Knowledge Transfer via Language and Thought Theory</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) open up new possibilities for leveraging information across languages, but their factual knowledge recall remains inconsistent depending on the input language. While previous studies have attempted to address this issue through English-based prompting and evaluation, we explore non-English to English transfer via Language and Thought Theory. This perspective allows us to examine language-thought binding in LLMs and uncover why factual knowledge often fails to transfer effectively. We propose the Language-to-Thought (L2T) prompting strategy, which analyzes the relationship between input language, internal cognitive processes, and knowledge. Experimental results challenge the assumption that English-based approaches consistently outperform other languages and offer a novel insight that aligning the model's internal thought with the knowledge required for the task is critical for successful cross-lingual transfer. Furthermore, we show that applying L2T during training can alleviate LLMs' reliance on the input language and facilitate cross-linguistic knowledge integration without translation-based learning. Code and datasets will be available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24377v1">LLM Inference Enhanced by External Knowledge: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enhanced natural-language reasoning. However, their limited parametric memory and susceptibility to hallucination present persistent challenges for tasks requiring accurate, context-based inference. To overcome these limitations, an increasing number of studies have proposed leveraging external knowledge to enhance LLMs. This study offers a systematic exploration of strategies for using external knowledge to enhance LLMs, beginning with a taxonomy that categorizes external knowledge into unstructured and structured data. We then focus on structured knowledge, presenting distinct taxonomies for tables and knowledge graphs (KGs), detailing their integration paradigms with LLMs, and reviewing representative methods. Our comparative analysis further highlights the trade-offs among interpretability, scalability, and performance, providing insights for developing trustworthy and generalizable knowledge-enhanced LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18410v2">Can Code Outlove Blood? An LLM-based VR Experience to Prompt Reflection on Parental Verbal Abuse</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 8 pages, 5 figures, accetped by 30th International Symposium on Electronic Art (ISEA 2025)
    </div>
    <details class="paper-abstract">
      Parental verbal abuse leaves lasting emotional impacts, yet current therapeutic approaches often lack immersive self-reflection opportunities. To address this, we developed a VR experience powered by LLMs to foster reflection on parental verbal abuse. Participants with relevant experiences engage in a dual-phase VR experience: first assuming the role of a verbally abusive parent, interacting with an LLM portraying a child, then observing the LLM reframing abusive dialogue into warm, supportive expressions as a nurturing parent. A qualitative study with 12 participants showed that the experience encourages reflection on their past experiences and fosters supportive emotions. However, these effects vary with participants' personal histories, emphasizing the need for greater personalization in AI-driven emotional support. This study explores the use of LLMs in immersive environment to promote emotional reflection, offering insights into the design of AI-driven emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17115v2">Swarm Intelligence Enhanced Reasoning: A Density-Driven Framework for LLM-Based Multi-Agent Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Recently, many approaches, such as Chain-of-Thought (CoT) prompting and Multi-Agent Debate (MAD), have been proposed to further enrich Large Language Models' (LLMs) complex problem-solving capacities in reasoning scenarios. However, these methods may fail to solve complex problems due to the lack of ability to find optimal solutions. Swarm Intelligence has been serving as a powerful tool for finding optima in the field of traditional optimization problems. To this end, we propose integrating swarm intelligence into the reasoning process by introducing a novel Agent-based Swarm Intelligence (ASI) paradigm. In this paradigm, we formulate LLM reasoning as an optimization problem and use a swarm intelligence scheme to guide a group of LLM-based agents in collaboratively searching for optimal solutions. To avoid swarm intelligence getting trapped in local optima, we further develop a Swarm Intelligence Enhancing Reasoning (SIER) framework, which develops a density-driven strategy to enhance the reasoning ability. To be specific, we propose to perform kernel density estimation and non-dominated sorting to optimize both solution quality and diversity simultaneously. In this case, SIER efficiently enhances solution space exploration through expanding the diversity of the reasoning path. Besides, a step-level quality evaluation is used to help agents improve solution quality by correcting low-quality intermediate steps. Then, we use quality thresholds to dynamically control the termination of exploration and the selection of candidate steps, enabling a more flexible and efficient reasoning process. Extensive experiments are ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24362v1">Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      We investigate whether the success of a zero-shot Chain-of-Thought (CoT) process can be predicted before completion. We discover that a probing classifier, based on LLM representations, performs well \emph{even before a single token is generated}, suggesting that crucial information about the reasoning process is already present in the initial steps representations. In contrast, a strong BERT-based baseline, which relies solely on the generated tokens, performs worse, likely because it depends on shallow linguistic cues rather than deeper reasoning dynamics. Surprisingly, using later reasoning steps does not always improve classification. When additional context is unhelpful, earlier representations resemble later ones more, suggesting LLMs encode key information early. This implies reasoning can often stop early without loss. To test this, we conduct early stopping experiments, showing that truncating CoT reasoning still improves performance over not using CoT at all, though a gap remains compared to full reasoning. However, approaches like supervised learning or reinforcement learning designed to shorten CoT chains could leverage our classifier's guidance to identify when early stopping is effective. Our findings provide insights that may support such methods, helping to optimize CoT's efficiency while preserving its benefits.\footnote{Code and data is available at \href{https://github.com/anum94/CoTpred}{\texttt{github.com/anum94/CoTpred}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07217v2">ReelWave: Multi-Agentic Movie Sound Generation through Multimodal LLM Conversation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Current audio generation conditioned by text or video focuses on aligning audio with text/video modalities. Despite excellent alignment results, these multimodal frameworks still cannot be directly applied to compelling movie storytelling involving multiple scenes, where "on-screen" sounds require temporally-aligned audio generation, while "off-screen" sounds contribute to appropriate environment sounds accompanied by background music when applicable. Inspired by professional movie production, this paper proposes a multi-agentic framework for audio generation supervised by an autonomous Sound Director agent, engaging multi-turn conversations with other agents for on-screen and off-screen sound generation through multimodal LLM. To address on-screen sound generation, after detecting any talking humans in videos, we capture semantically and temporally synchronized sound by training a prediction model that forecasts interpretable, time-varying audio control signals: loudness, pitch, and timbre, which are used by a Foley Artist agent to condition a cross-attention module in the sound generation. The Foley Artist works cooperatively with the Composer and Voice Actor agents, and together they autonomously generate off-screen sound to complement the overall production. Each agent takes on specific roles similar to those of a movie production team. To temporally ground audio language models, in ReelWave, text/video conditions are decomposed into atomic, specific sound generation instructions synchronized with visuals when applicable. Consequently, our framework can generate rich and relevant audio content conditioned on video clips extracted from movies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24347v1">Fewer Hallucinations, More Verification: A Three-Stage LLM-Based Framework for ASR Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) error correction aims to correct recognition errors while preserving accurate text. Although traditional approaches demonstrate moderate effectiveness, LLMs offer a paradigm that eliminates the need for training and labeled data. However, directly using LLMs will encounter hallucinations problem, which may lead to the modification of the correct text. To address this problem, we propose the Reliable LLM Correction Framework (RLLM-CF), which consists of three stages: (1) error pre-detection, (2) chain-of-thought sub-tasks iterative correction, and (3) reasoning process verification. The advantage of our method is that it does not require additional information or fine-tuning of the model, and ensures the correctness of the LLM correction under multi-pass programming. Experiments on AISHELL-1, AISHELL-2, and Librispeech show that the GPT-4o model enhanced by our framework achieves 21%, 11%, 9%, and 11.4% relative reductions in CER/WER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24324v1">SwiftEval: Developing a Language-Specific Benchmark for LLM-generated Code Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted to FORGE'25 Benchmarking on 15.01.2025, to be published by IEEE under the CC BY-NC-ND 4.0 license. This is the accepted version of the article (5 pages, 2 figures, 1 table). DOI will be added upon publication
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have showcased significant advancements in code generation. However, most evaluation benchmarks are primarily oriented towards Python, making it difficult to evaluate other programming languages, such as Swift, with high quality. By examining widely established multilingual benchmarks like HumanEval-XL and MultiPL-E, we identified critical issues specific to their Swift components, making them insufficient or even irrelevant for assessing LLM coding capabilities on Swift. Unlike these existing approaches, which prioritize rapid scaling and generalization by automatically translating Python-centric benchmarks with LLMs, we adopt a quality-over-quantity methodology. We present SwiftEval, the first Swift-oriented benchmark consisting of 28 carefully hand-crafted problems, and evaluate 44 popular Code LLMs on it. Our results show significant LLM scores drop for problems requiring language-specific features, most noticeable in the models of smaller sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11471v4">GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Accepted by ACL2025(Findings)
    </div>
    <details class="paper-abstract">
      Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24315v1">InteractAnything: Zero-shot Human Object Interaction Synthesis via LLM Feedback and Object Affordance Parsing</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in 3D human-aware generation have made significant progress. However, existing methods still struggle with generating novel Human Object Interaction (HOI) from text, particularly for open-set objects. We identify three main challenges of this task: precise human-object relation reasoning, affordance parsing for any object, and detailed human interaction pose synthesis aligning description and object geometry. In this work, we propose a novel zero-shot 3D HOI generation framework without training on specific datasets, leveraging the knowledge from large-scale pre-trained models. Specifically, the human-object relations are inferred from large language models (LLMs) to initialize object properties and guide the optimization process. Then we utilize a pre-trained 2D image diffusion model to parse unseen objects and extract contact points, avoiding the limitations imposed by existing 3D asset knowledge. The initial human pose is generated by sampling multiple hypotheses through multi-view SDS based on the input text and object geometry. Finally, we introduce a detailed optimization to generate fine-grained, precise, and natural interaction, enforcing realistic 3D contact between the 3D object and the involved body parts, including hands in grasping. This is achieved by distilling human-level feedback from LLMs to capture detailed human-object relations from the text instruction. Extensive experiments validate the effectiveness of our approach compared to prior works, particularly in terms of the fine-grained nature of interactions and the ability to handle open-set 3D objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24306v1">GridRoute: A Benchmark for LLM-Based Route Planning with Cardinal Movement in Grid Environments</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have demonstrated their potential in planning and reasoning tasks, offering a flexible alternative to classical pathfinding algorithms. However, most existing studies focus on LLMs' independent reasoning capabilities and overlook the potential synergy between LLMs and traditional algorithms. To fill this gap, we propose a comprehensive evaluation benchmark GridRoute to assess how LLMs can take advantage of traditional algorithms. We also propose a novel hybrid prompting technique called Algorithm of Thought (AoT), which introduces traditional algorithms' guidance into prompting. Our benchmark evaluates six LLMs ranging from 7B to 72B parameters across various map sizes, assessing their performance in correctness, optimality, and efficiency in grid environments with varying sizes. Our results show that AoT significantly boosts performance across all model sizes, particularly in larger or more complex environments, suggesting a promising approach to addressing path planning challenges. Our code is open-sourced at https://github.com/LinChance/GridRoute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17139v3">EarthSE: A Benchmark for Evaluating Earth Scientific Exploration Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Advancements in Large Language Models (LLMs) drive interest in scientific applications, necessitating specialized benchmarks such as Earth science. Existing benchmarks either present a general science focus devoid of Earth science specificity or cover isolated subdomains, lacking holistic evaluation. Furthermore, current benchmarks typically neglect the assessment of LLMs' capabilities in open-ended scientific exploration. In this paper, we present a comprehensive and professional benchmark for the Earth sciences, designed to evaluate the capabilities of LLMs in scientific exploration within this domain, spanning from fundamental to advanced levels. Leveraging a corpus of 100,000 research papers, we first construct two Question Answering (QA) datasets: Earth-Iron, which offers extensive question coverage for broad assessment, and Earth-Silver, which features a higher level of difficulty to evaluate professional depth. These datasets encompass five Earth spheres, 114 disciplines, and 11 task categories, assessing foundational knowledge crucial for scientific exploration. Most notably, we introduce Earth-Gold with new metrics, a dataset comprising open-ended multi-turn dialogues specifically designed to evaluate the advanced capabilities of LLMs in scientific exploration, including methodology induction, limitation analysis, and concept proposal. Extensive experiments reveal limitations in 11 leading LLMs across different domains and tasks, highlighting considerable room for improvement in their scientific exploration capabilities. The benchmark is available on https://huggingface.co/ai-earth .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24292v1">Mind the Quote: Enabling Quotation-Aware Dialogue in LLMs via Plug-and-Play Modules</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Human-AI conversation frequently relies on quoting earlier text-"check it with the formula I just highlighted"-yet today's large language models (LLMs) lack an explicit mechanism for locating and exploiting such spans. We formalise the challenge as span-conditioned generation, decomposing each turn into the dialogue history, a set of token-offset quotation spans, and an intent utterance. Building on this abstraction, we introduce a quotation-centric data pipeline that automatically synthesises task-specific dialogues, verifies answer correctness through multi-stage consistency checks, and yields both a heterogeneous training corpus and the first benchmark covering five representative scenarios. To meet the benchmark's zero-overhead and parameter-efficiency requirements, we propose QuAda, a lightweight training-based method that attaches two bottleneck projections to every attention head, dynamically amplifying or suppressing attention to quoted spans at inference time while leaving the prompt unchanged and updating < 2.8% of backbone weights. Experiments across models show that QuAda is suitable for all scenarios and generalises to unseen topics, offering an effective, plug-and-play solution for quotation-aware dialogue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24282v1">LLM-powered Query Expansion for Enhancing Boundary Prediction in Language-driven Action Localization</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Language-driven action localization in videos requires not only semantic alignment between language query and video segment, but also prediction of action boundaries. However, the language query primarily describes the main content of an action and usually lacks specific details of action start and end boundaries, which increases the subjectivity of manual boundary annotation and leads to boundary uncertainty in training data. In this paper, on one hand, we propose to expand the original query by generating textual descriptions of the action start and end boundaries through LLMs, which can provide more detailed boundary cues for localization and thus reduce the impact of boundary uncertainty. On the other hand, to enhance the tolerance to boundary uncertainty during training, we propose to model probability scores of action boundaries by calculating the semantic similarities between frames and the expanded query as well as the temporal distances between frames and the annotated boundary frames. They can provide more consistent boundary supervision, thus improving the stability of training. Our method is model-agnostic and can be seamlessly and easily integrated into any existing models of language-driven action localization in an off-the-shelf manner. Experimental results on several datasets demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24273v1">How Much Backtracking is Enough? Exploring the Interplay of SFT and RL in Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in large language models (LLMs) have effectively improved their reasoning abilities, particularly on mathematical and logical problems that have verifiable answers, through techniques such as supervised finetuning (SFT) and reinforcement learning (RL). Prior research indicates that RL effectively internalizes search strategies, enabling long chain-of-thought (CoT) reasoning, with backtracking emerging naturally as a learned capability. However, the precise benefits of backtracking, specifically, how significantly it contributes to reasoning improvements and the optimal extent of its use, remain poorly understood. In this work, we systematically investigate the dynamics between SFT and RL on eight reasoning tasks: Countdown, Sudoku, Arc 1D, Geometry, Color Cube Rotation, List Functions, Zebra Puzzles, and Self Reference. Our findings highlight that short CoT sequences used in SFT as a warm-up do have moderate contribution to RL training, compared with cold-start RL; however such contribution diminishes when tasks become increasingly difficult. Motivated by this observation, we construct synthetic datasets varying systematically in the number of backtracking steps and conduct controlled experiments to isolate the influence of either the correctness (content) or the structure (i.e., backtrack frequency). We find that (1) longer CoT with backtracks generally induce better and more stable RL training, (2) more challenging problems with larger search space tend to need higher numbers of backtracks during the SFT stage. Additionally, we demonstrate through experiments on distilled data that RL training is largely unaffected by the correctness of long CoT sequences, suggesting that RL prioritizes structural patterns over content correctness. Collectively, our results offer practical insights into designing optimal training strategies to effectively scale reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24264v1">Faithful and Robust LLM-Driven Theorem Proving for NLI Explanations</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 Camera-ready for ACL 2025
    </div>
    <details class="paper-abstract">
      Natural language explanations play a fundamental role in Natural Language Inference (NLI) by revealing how premises logically entail hypotheses. Recent work has shown that the interaction of large language models (LLMs) with theorem provers (TPs) can help verify and improve the validity of NLI explanations. However, TPs require translating natural language into machine-verifiable formal representations, a process that introduces the risk of semantic information loss and unfaithful interpretation, an issue compounded by LLMs' challenges in capturing critical logical structures with sufficient precision. Moreover, LLMs are still limited in their capacity for rigorous and robust proof construction within formal verification frameworks. To mitigate issues related to faithfulness and robustness, this paper investigates strategies to (1) alleviate semantic loss during autoformalisation, (2) efficiently identify and correct syntactic errors in logical representations, (3) explicitly use logical expressions to guide LLMs in generating structured proof sketches, and (4) increase LLMs' capacity of interpreting TP's feedback for iterative refinement. Our empirical results on e-SNLI, QASC and WorldTree using different LLMs demonstrate that the proposed strategies yield significant improvements in autoformalisation (+18.46%, +34.2%, +39.77%) and explanation refinement (+29.5%, +51.5%, +41.25%) over the state-of-the-art model. Moreover, we show that specific interventions on the hybrid LLM-TP architecture can substantially improve efficiency, drastically reducing the number of iterations required for successful verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24263v1">Simulating Training Data Leakage in Multiple-Choice Benchmarks for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) continues to improve, as reflected in rising scores on standard benchmarks. However, the lack of transparency around training data raises concerns about potential overlap with evaluation sets and the fairness of reported results. Although prior work has proposed methods for detecting data leakage, these approaches primarily focus on identifying outliers and have not been evaluated under controlled simulated leakage conditions. In this work, we compare existing leakage detection techniques, namely permutation and n-gram-based methods, under a continual pretraining setup that simulates real-world leakage scenarios, and additionally explore a lightweight method we call semi-half question. Although semi-half offers a low-cost alternative, our analysis shows that the n-gram method consistently achieves the highest F1-score. We also refine these techniques to support instance-level detection and reduce computational overhead. Leveraging the best-performing method, we create cleaned versions of MMLU and HellaSwag, and re-evaluate several LLMs. Our findings present a practical path toward more reliable and transparent evaluations, and we recommend contamination checks as a standard step before releasing benchmark results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15268v3">Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 8 pages of content
    </div>
    <details class="paper-abstract">
      The rapid growth of social media platforms has raised significant concerns regarding online content toxicity. When Large Language Models (LLMs) are used for toxicity detection, two key challenges emerge: 1) the absence of domain-specific toxic knowledge leads to false negatives; 2) the excessive sensitivity of LLMs to toxic speech results in false positives, limiting freedom of speech. To address these issues, we propose a novel method called MetaTox, leveraging graph search on a meta-toxic knowledge graph to enhance hatred and toxicity detection. First, we construct a comprehensive meta-toxic knowledge graph by utilizing LLMs to extract toxic information through a three-step pipeline, with toxic benchmark datasets serving as corpora. Second, we query the graph via retrieval and ranking processes to supplement accurate, relevant toxic knowledge. Extensive experiments and in-depth case studies across multiple datasets demonstrate that our MetaTox significantly decreases the false positive rate while boosting overall toxicity detection performance. Our code is available at https://github.com/YiboZhao624/MetaTox.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24255v1">Effects of Theory of Mind and Prosocial Beliefs on Steering Human-Aligned Behaviors of LLMs in Ultimatum Games</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 17 pages, 1 figure, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown potential in simulating human behaviors and performing theory-of-mind (ToM) reasoning, a crucial skill for complex social interactions. In this study, we investigate the role of ToM reasoning in aligning agentic behaviors with human norms in negotiation tasks, using the ultimatum game as a controlled environment. We initialized LLM agents with different prosocial beliefs (including Greedy, Fair, and Selfless) and reasoning methods like chain-of-thought (CoT) and varying ToM levels, and examined their decision-making processes across diverse LLMs, including reasoning models like o3-mini and DeepSeek-R1 Distilled Qwen 32B. Results from 2,700 simulations indicated that ToM reasoning enhances behavior alignment, decision-making consistency, and negotiation outcomes. Consistent with previous findings, reasoning models exhibit limited capability compared to models with ToM reasoning, different roles of the game benefits with different orders of ToM reasoning. Our findings contribute to the understanding of ToM's role in enhancing human-AI interaction and cooperative decision-making. The code used for our experiments can be found at https://github.com/Stealth-py/UltimatumToM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14561v2">Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches relying on domain-specific pre-trained models like SciBERT, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero-, one-, few-, and many-shot prompting. Our experimental study identifies the top-performing model and prompting parameters through extensive in-context learning experiments. We then demonstrate the significant impact of task-specific adaptation by fine-tuning this model, achieving a relative F1-score improvement of 8% on the SciCite dataset and 4.3% on the ACL-ARC dataset compared to the instruction-tuned baseline. These findings provide valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24239v1">An Adversary-Resistant Multi-Agent LLM System via Credibility Scoring</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      While multi-agent LLM systems show strong capabilities in various domains, they are highly vulnerable to adversarial and low-performing agents. To resolve this issue, in this paper, we introduce a general and adversary-resistant multi-agent LLM framework based on credibility scoring. We model the collaborative query-answering process as an iterative game, where the agents communicate and contribute to a final system output. Our system associates a credibility score that is used when aggregating the team outputs. The credibility scores are learned gradually based on the past contributions of each agent in query answering. Our experiments across multiple tasks and settings demonstrate our system's effectiveness in mitigating adversarial influence and enhancing the resilience of multi-agent cooperation, even in the adversary-majority settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02659v2">A Training-Free Length Extrapolation Approach for LLMs: Greedy Attention Logit Interpolation (GALI)</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 9 pages, under review in the conference
    </div>
    <details class="paper-abstract">
      Transformer-based Large Language Models (LLMs) struggle with inputs exceeding their training context window due to positional out-of-distribution (O.O.D.) issues that disrupt attention. Existing solutions, including fine-tuning and training-free methods, face challenges like inefficiency, redundant interpolation, logit outliers, or loss of local positional information. We propose Greedy Attention Logit Interpolation (GALI), a training-free method that improves length extrapolation by greedily reusing pretrained positional intervals and interpolating attention logit to eliminate outliers. GALI achieves stable and superior performance across a wide range of long-context tasks without requiring input-length-specific tuning. Our analysis further reveals that LLMs interpret positional intervals unevenly and that restricting interpolation to narrower ranges improves performance, even on short-context tasks. GALI represents a step toward more robust and generalizable long-text processing in LLMs. Our implementation of GALI, along with the experiments from our paper, is open-sourced at https://github.com/adlnlp/Gali.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24217v1">Semi-structured LLM Reasoners Can Be Rigorously Audited</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly capable at reasoning, the problem of "faithfulness" persists: LLM "reasoning traces" can contain errors and omissions that are difficult to detect, and may obscure biases in model outputs. To address these limitations, we introduce Semi-Structured Reasoning Models (SSRMs), which internalize a semi-structured Chain-of-Thought (CoT) reasoning format within the model. Our SSRMs generate reasoning traces in a Pythonic syntax. While SSRM traces are not executable, they adopt a restricted, task-specific vocabulary to name distinct reasoning steps, and to mark each step's inputs and outputs. Through extensive evaluation on ten benchmarks, SSRMs demonstrate strong performance and generality: they outperform comparably sized baselines by nearly ten percentage points on in-domain tasks while remaining competitive with specialized models on out-of-domain medical benchmarks. Furthermore, we show that semi-structured reasoning is more amenable to analysis: in particular, they can be automatically audited to identify reasoning flaws. We explore both hand-crafted structured audits, which detect task-specific problematic reasoning patterns, and learned typicality audits, which apply probabilistic models over reasoning patterns, and show that both audits can be used to effectively flag probable reasoning errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v4">AlphaPO: Reward Shape Matters for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 26 pages, 16 figures. Accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce \textbf{AlphaPO}, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15\% to 50\% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07067v2">DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 ICML2025 Spotlight
    </div>
    <details class="paper-abstract">
      Despite the success of distillation in large language models (LLMs), most prior work applies identical loss functions to both teacher- and student-generated data. These strategies overlook the synergy between loss formulations and data types, leading to a suboptimal performance boost in student models. To address this, we propose DistiLLM-2, a contrastive approach that simultaneously increases the likelihood of teacher responses and decreases that of student responses by harnessing this synergy. Our extensive experiments show that DistiLLM-2 not only builds high-performing student models across a wide range of tasks, including instruction-following and code generation, but also supports diverse applications, such as preference alignment and vision-language extensions. These findings highlight the potential of a contrastive approach to enhance the efficacy of LLM distillation by effectively aligning teacher and student models across varied data types.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24208v1">Bootstrapping LLM Robustness for VLM Safety via Reducing the Pretraining Modality Gap</a></div>
    <div class="paper-meta">
      📅 2025-05-30
    </div>
    <details class="paper-abstract">
      Ensuring Vision-Language Models (VLMs) generate safe outputs is crucial for their reliable deployment. However, LVLMs suffer from drastic safety degradation compared to their LLM backbone. Even blank or irrelevant images can trigger LVLMs to generate harmful responses to prompts that would otherwise be refused in text-only contexts. The modality gap between image and text representations has been recently hypothesized to contribute to safety degradation of LVLMs. However, if and how the amount of modality gap affects LVLMs' safety is not studied. In this work, we show that the amount of modality gap is highly inversely correlated with VLMs' safety. Then, we show that this modality gap is introduced during pretraining LVLMs and persists through fine-tuning. Inspired by this observation, we propose a regularization to reduce the modality gap during pretraining. Our extensive experiments on LLaVA v1.5, ShareGPT4V, and MiniGPT-4 show that our method substantially improves safety alignment of LVLMs, reducing unsafe rate by up to 16.3% without compromising performance, and can further boost existing defenses by up to 18.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15401v2">Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-30
      | 💬 19 pages, 6 figures, ACL 2025 findings, camera-ready version
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) can significantly enhance the complex reasoning capabilities of large language models (LLMs), with the key lying in the selection and ordering of demonstration examples. Previous methods typically relied on simple features to measure the relevance between examples. We argue that these features are not sufficient to reflect the intrinsic connections between examples. In this study, we propose a curriculum ICL strategy guided by problem-solving logic. We select demonstration examples by analyzing the problem-solving logic and order them based on curriculum learning. Specifically, we constructed a problem-solving logic instruction set based on the BREAK dataset and fine-tuned a language model to analyze the problem-solving logic of examples. Subsequently, we selected appropriate demonstration examples based on problem-solving logic and assessed their difficulty according to the number of problem-solving steps. In accordance with the principles of curriculum learning, we ordered the examples from easy to hard to serve as contextual prompts. Experimental results on multiple benchmarks indicate that our method outperforms previous ICL approaches in terms of performance and efficiency, effectively enhancing the complex reasoning capabilities of LLMs. Our project will be released at https://github.com/maxuetao/CurriculumICL
    </details>
</div>
