# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15004v3">From Vulnerabilities to Remediation: A Systematic Literature Review of LLMs in Code Security</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for automating various programming tasks, including security-related ones, such as detecting and fixing vulnerabilities. Despite their promising capabilities, when required to produce or modify pre-existing code, LLMs could introduce vulnerabilities unbeknown to the programmer. When analyzing code, they could miss clear vulnerabilities or signal nonexistent ones. In this Systematic Literature Review (SLR), we aim to investigate both the security benefits and potential drawbacks of using LLMs for a variety of code-related tasks. In particular, first we focus on the types of vulnerabilities that could be introduced by LLMs, when used for producing code. Second, we analyze the capabilities of LLMs to detect and fix vulnerabilities, in any given code, and how the prompting strategy of choice impacts their performance in these two tasks. Last, we provide an in-depth analysis on how data poisoning attacks on LLMs can impact performance in the aforementioned tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10063v1">Hallucination Detection in LLMs via Topological Divergence on Attention Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments, including evaluation on question answering and data-to-text tasks, show that our approach achieves state-of-the-art or competitive results on several benchmarks, two of which were annotated by us and are being publicly released to facilitate further research. Beyond its strong in-domain performance, TOHA maintains remarkable domain transferability across multiple open-source LLMs. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13572v2">VeriContaminated: Assessing LLM-Driven Verilog Coding for Data Contamination</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized code generation, achieving exceptional results on various established benchmarking frameworks. However, concerns about data contamination - where benchmark data inadvertently leaks into pre-training or fine-tuning datasets - raise questions about the validity of these evaluations. While this issue is known, limiting the industrial adoption of LLM-driven software engineering, hardware coding has received little to no attention regarding these risks. For the first time, we analyze state-of-the-art (SOTA) evaluation frameworks for Verilog code generation (VerilogEval and RTLLM), using established methods for contamination detection (CCD and Min-K% Prob). We cover SOTA commercial and open-source LLMs (CodeGen2.5, Minitron 4b, Mistral 7b, phi-4 mini, LLaMA-{1,2,3.1}, GPT-{2,3.5,4o}, Deepseek-Coder, and CodeQwen 1.5), in baseline and fine-tuned models (RTLCoder and Verigen). Our study confirms that data contamination is a critical concern. We explore mitigations and the resulting trade-offs for code quality vs fairness (i.e., reducing contamination toward unbiased benchmarking).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10050v1">Emotional Strain and Frustration in LLM Interactions in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Accepted in EASE'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into various daily tasks in Software Engineering such as coding and requirement elicitation. Despite their various capabilities and constant use, some interactions can lead to unexpected challenges (e.g. hallucinations or verbose answers) and, in turn, cause emotions that develop into frustration. Frustration can negatively impact engineers' productivity and well-being if they escalate into stress and burnout. In this paper, we assess the impact of LLM interactions on software engineers' emotional responses, specifically strains, and identify common causes of frustration when interacting with LLMs at work. Based on 62 survey responses from software engineers in industry and academia across various companies and universities, we found that a majority of our respondents experience frustrations or other related emotions regardless of the nature of their work. Additionally, our results showed that frustration mainly stemmed from issues with correctness and less critical issues such as adaptability to context or specific format. While such issues may not cause frustration in general, artefacts that do not follow certain preferences, standards, or best practices can make the output unusable without extensive modification, causing frustration over time. In addition to the frustration triggers, our study offers guidelines to improve the software engineers' experience, aiming to minimise long-term consequences on mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08525v2">Task Memory Engine (TME): A Structured Memory Framework with Graph-Aware Extensions for Multi-Step LLM Agent Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 14 pages, 5 figures. Preprint prepared for future submission. Includes implementation and token-efficiency analysis. Code at https://github.com/biubiutomato/TME-Agent
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. A reference implementation of the core TME components is available at https://github.com/biubiutomato/TME-Agent, including basic examples and structured memory integration. While the current implementation uses a tree-based structure, TME is designed to be graph-aware, supporting reusable substeps, converging task paths, and shared dependencies. This lays the groundwork for future DAG-based memory architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v4">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL) using process rewards and iterative self-play, without supervised fine-tuning (SFT) as a preliminary step. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10013v1">Training LLMs on HPC Systems: Best Practices from the OpenGPT-X Project</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      The training of large language models (LLMs) requires substantial computational resources, complex software stacks, and carefully designed workflows to achieve scalability and efficiency. This report presents best practices and insights gained from the OpenGPT-X project, a German initiative focused on developing open, multilingual LLMs optimized for European languages. We detail the use of high-performance computing (HPC) systems, primarily JUWELS Booster at JSC, for training Teuken-7B, a 7-billion-parameter transformer model. The report covers system architecture, training infrastructure, software choices, profiling and benchmarking tools, as well as engineering and operational challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17039v2">Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans?</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Studies on evaluation metrics and LLM-as-a-Judge models for automatic text summarization have largely been focused on English, limiting our understanding of their effectiveness in other languages. Through our new dataset BASSE (BAsque and Spanish Summarization Evaluation), we address this situation by collecting human judgments on 2,040 abstractive summaries in Basque and Spanish, generated either manually or by five LLMs with four different prompts. For each summary, annotators evaluated five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We use these data to reevaluate traditional automatic metrics used for evaluating summaries, as well as several LLM-as-a-Judge models that show strong performance on this task in English. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly. We release BASSE and our code publicly, along with the first large-scale Basque summarization dataset containing 22,525 news articles with their subheads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09961v1">Privacy Meets Explainability: Managing Confidential Data and Transparency Policies in LLM-Empowered Science</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral to scientific workflows, concerns over the confidentiality and ethical handling of confidential data have emerged. This paper explores data exposure risks through LLM-powered scientific tools, which can inadvertently leak confidential information, including intellectual property and proprietary data, from scientists' perspectives. We propose "DataShield", a framework designed to detect confidential data leaks, summarize privacy policies, and visualize data flow, ensuring alignment with organizational policies and procedures. Our approach aims to inform scientists about data handling practices, enabling them to make informed decisions and protect sensitive information. Ongoing user studies with scientists are underway to evaluate the framework's usability, trustworthiness, and effectiveness in tackling real-world privacy challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09936v1">KeepKV: Eliminating Output Perturbation in KV Cache Compression for Efficient LLMs Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 18 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Efficient inference of large language models (LLMs) is hindered by an ever-growing key-value (KV) cache, making KV cache compression a critical research direction. Traditional methods selectively evict less important KV cache entries based on attention scores or position heuristics, which leads to information loss and hallucinations. Recently, merging-based strategies have been explored to retain more information by merging KV pairs that would be discarded; however, these existing approaches inevitably introduce inconsistencies in attention distributions before and after merging, causing output perturbation and degraded generation quality. To overcome this challenge, we propose KeepKV, a novel adaptive KV cache merging method designed to eliminate output perturbation while preserving performance under strict memory constraints. KeepKV introduces the Electoral Votes mechanism that records merging history and adaptively adjusts attention scores. Moreover, it further leverages a novel Zero Inference-Perturbation Merging methods, keeping attention consistency and compensating for attention loss resulting from cache merging. KeepKV successfully retains essential context information within a significantly compressed cache. Extensive experiments on various benchmarks and LLM architectures demonstrate that KeepKV substantially reduces memory usage, enhances inference throughput by more than 2x and keeps superior generation quality even with 10% KV cache budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06931v4">Non-Prehensile Tool-Object Manipulation by Integrating LLM-Based Planning and Manoeuvrability-Driven Controls</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Being able to use tools is a widely recognised indicator of intelligence across species. Humans, for instance, have demonstrated mastery of tool use for over two million years. The ability to use tools is invaluable as it extends an organism's reach and enhances its capacity to interact with objects and the environment. Being able to understand the geometric-mechanical relations between the tools-objects-environments allows certain species (e.g., apes and crows) to reach food in narrow constrained spaces. The same principles of physical augmentation and its associated non-prehensile manipulation capabilities also apply to robotic systems. For example, by instrumenting them with different types of end-effectors, robots can (in principle) dexterously interact (e.g., push and flip) with objects of various shapes and masses akin to its biological counterpart. However, developing this type of manipulation skill is still an open research problem. Furthermore, the complexity of planning tool-object manipulation tasks, particularly in coordinating the actions of dual-arm robots, presents significant challenges. To address these complexities, we propose integrating Large Language Models (LLMs) to assist in planning and executing these intricate manipulations, thereby enhancing the robot's ability to perform in diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09923v1">Guiding Reasoning in Small Language Models with LLM Assistance</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 20 pages, 10 figures, 11 tables
    </div>
    <details class="paper-abstract">
      The limited reasoning capabilities of small language models (SLMs) cast doubt on their suitability for tasks demanding deep, multi-step logical deduction. This paper introduces a framework called Small Reasons, Large Hints (SMART), which selectively augments SLM reasoning with targeted guidance from large language models (LLMs). Inspired by the concept of cognitive scaffolding, SMART employs a score-based evaluation to identify uncertain reasoning steps and injects corrective LLM-generated reasoning only when necessary. By framing structured reasoning as an optimal policy search, our approach steers the reasoning trajectory toward correct solutions without exhaustive sampling. Our experiments on mathematical reasoning datasets demonstrate that targeted external scaffolding significantly improves performance, paving the way for collaborative use of both SLM and LLM to tackle complex reasoning tasks that are currently unsolvable by SLMs alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09855v1">PestMA: LLM-based Multi-Agent System for Informed Pest Management</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Effective pest management is complex due to the need for accurate, context-specific decisions. Recent advancements in large language models (LLMs) open new possibilities for addressing these challenges by providing sophisticated, adaptive knowledge acquisition and reasoning. However, existing LLM-based pest management approaches often rely on a single-agent paradigm, which can limit their capacity to incorporate diverse external information, engage in systematic validation, and address complex, threshold-driven decisions. To overcome these limitations, we introduce PestMA, an LLM-based multi-agent system (MAS) designed to generate reliable and evidence-based pest management advice. Building on an editorial paradigm, PestMA features three specialized agents, an Editor for synthesizing pest management recommendations, a Retriever for gathering relevant external data, and a Validator for ensuring correctness. Evaluations on real-world pest scenarios demonstrate that PestMA achieves an initial accuracy of 86.8% for pest management decisions, which increases to 92.6% after validation. These results underscore the value of collaborative agent-based workflows in refining and validating decisions, highlighting the potential of LLM-based multi-agent systems to automate and enhance pest management processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09816v1">Augmented Relevance Datasets with Fine-Tuned Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 10 pages, 3 figures, and 6 tables. Accepted and presented to LLM4EVAL at WSDM '25
    </div>
    <details class="paper-abstract">
      Building high-quality datasets and labeling query-document relevance are essential yet resource-intensive tasks, requiring detailed guidelines and substantial effort from human annotators. This paper explores the use of small, fine-tuned large language models (LLMs) to automate relevance assessment, with a focus on improving ranking models' performance by augmenting their training dataset. We fine-tuned small LLMs to enhance relevance assessments, thereby improving dataset creation quality for downstream ranking model training. Our experiments demonstrate that these fine-tuned small LLMs not only outperform certain closed source models on our dataset but also lead to substantial improvements in ranking model performance. These results highlight the potential of leveraging small LLMs for efficient and scalable dataset augmentation, providing a practical solution for search engine optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09802v1">Training Small Reasoning LLMs with Cognitive Preference Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of large language models (LLMs), such as OpenAI's o1 and DeepSeek-R1, have seen substantial advancements through deep thinking. However, these enhancements come with significant resource demands, underscoring the need to explore strategies to train effective reasoning LLMs with far fewer parameters. A critical challenge is that smaller models have different capacities and cognitive trajectories than their larger counterparts. Hence, direct distillation of chain-of-thought (CoT) results from large LLMs to smaller ones can be sometimes ineffective and requires a huge amount of annotated data. In this paper, we introduce a novel framework called Critique-Rethink-Verify (CRV), designed for training smaller yet powerful reasoning LLMs. Our CRV framework consists of multiple LLM agents, each specializing in unique abilities: (i) critiquing the CoTs according to the cognitive capabilities of smaller models, (ii) rethinking and refining these CoTs based on the critiques, and (iii) verifying the correctness of the refined results. We further propose the cognitive preference optimization (CogPO) algorithm to enhance the reasoning abilities of smaller models by aligning thoughts of these models with their cognitive capacities. Comprehensive evaluations on challenging reasoning benchmarks demonstrate the efficacy of CRV and CogPO, which outperforms other training methods by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09798v1">ReadMe.LLM: A Framework to Help LLMs Understand Your Library</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 12 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with code generation tasks involving niche software libraries. Existing code generation techniques with only human-oriented documentation can fail -- even when the LLM has access to web search and the library is documented online. To address this challenge, we propose ReadMe.LLM, LLM-oriented documentation for software libraries. By attaching the contents of ReadMe.LLM to a query, performance consistently improves to near-perfect accuracy, with one case study demonstrating up to 100% success across all tested models. We propose a software development lifecycle where LLM-specific documentation is maintained alongside traditional software updates. In this study, we present two practical applications of the ReadMe.LLM idea with diverse software libraries, highlighting that our proposed approach could generalize across programming domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07983v2">Psychological Health Knowledge-Enhanced LLM-based Social Network Crisis Intervention Text Transfer Recognition Method</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      As the prevalence of mental health crises increases on social media platforms, identifying and preventing potential harm has become an urgent challenge. This study introduces a large language model (LLM)-based text transfer recognition method for social network crisis intervention, enhanced with domain-specific mental health knowledge. We propose a multi-level framework that incorporates transfer learning using BERT, and integrates mental health knowledge, sentiment analysis, and behavior prediction techniques. The framework includes a crisis annotation tool trained on social media datasets from real-world events, enabling the model to detect nuanced emotional cues and identify psychological crises. Experimental results show that the proposed method outperforms traditional models in crisis detection accuracy and exhibits greater sensitivity to subtle emotional and contextual variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10724v1">HELIOS: Adaptive Model And Early-Exit Selection for Efficient LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) presents critical challenges due to the inherent trade-offs associated with key performance metrics, such as latency, accuracy, and throughput. Typically, gains in one metric is accompanied with degradation in others. Early-Exit LLMs (EE-LLMs) efficiently navigate this trade-off space by skipping some of the later model layers when it confidently finds an output token early, thus reducing latency without impacting accuracy. However, as the early exits taken depend on the task and are unknown apriori to request processing, EE-LLMs conservatively load the entire model, limiting resource savings and throughput. Also, current frameworks statically select a model for a user task, limiting our ability to adapt to changing nature of the input queries. We propose HELIOS to address these challenges. First, HELIOS shortlists a set of candidate LLMs, evaluates them using a subset of prompts, gathering telemetry data in real-time. Second, HELIOS uses the early exit data from these evaluations to greedily load the selected model only up to a limited number of layers. This approach yields memory savings which enables us to process more requests at the same time, thereby improving throughput. Third, HELIOS monitors and periodically reassesses the performance of the candidate LLMs and if needed, switches to another model that can service incoming queries more efficiently (such as using fewer layers without lowering accuracy). Our evaluations show that HELIOS achieves 1.48$\times$ throughput, 1.10$\times$ energy-efficiency, 1.39$\times$ lower response time, and 3.7$\times$ improvements in inference batch sizes compared to the baseline, when optimizing for the respective service level objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10713v1">Can LLMs Classify CVEs? Investigating LLMs Capabilities in Computing CVSS Vectors</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Accepted at TrustAICyberSec 2025
    </div>
    <details class="paper-abstract">
      Common Vulnerability and Exposure (CVE) records are fundamental to cybersecurity, offering unique identifiers for publicly known software and system vulnerabilities. Each CVE is typically assigned a Common Vulnerability Scoring System (CVSS) score to support risk prioritization and remediation. However, score inconsistencies often arise due to subjective interpretations of certain metrics. As the number of new CVEs continues to grow rapidly, automation is increasingly necessary to ensure timely and consistent scoring. While prior studies have explored automated methods, the application of Large Language Models (LLMs), despite their recent popularity, remains relatively underexplored. In this work, we evaluate the effectiveness of LLMs in generating CVSS scores for newly reported vulnerabilities. We investigate various prompt engineering strategies to enhance their accuracy and compare LLM-generated scores against those from embedding-based models, which use vector representations classified via supervised learning. Our results show that while LLMs demonstrate potential in automating CVSS evaluation, embedding-based methods outperform them in scoring more subjective components, particularly confidentiality, integrity, and availability impacts. These findings underscore the complexity of CVSS scoring and suggest that combining LLMs with embedding-based methods could yield more reliable results across all scoring components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10706v1">GestureCoach: Rehearsing for Engaging Talks with LLM-Driven Gesture Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      This paper introduces GestureCoach, a system designed to help speakers deliver more engaging talks by guiding them to gesture effectively during rehearsal. GestureCoach combines an LLM-driven gesture recommendation model with a rehearsal interface that proactively cues speakers to gesture appropriately. Trained on experts' gesturing patterns from TED talks, the model consists of two modules: an emphasis proposal module, which predicts when to gesture by identifying gesture-worthy text segments in the presenter notes, and a gesture identification module, which determines what gesture to use by retrieving semantically appropriate gestures from a curated gesture database. Results of a model performance evaluation and user study (N=30) show that the emphasis proposal module outperforms off-the-shelf LLMs in identifying suitable gesture regions, and that participants rated the majority of these predicted regions and their corresponding gestures as highly appropriate. A subsequent user study (N=10) showed that rehearsing with GestureCoach encouraged speakers to gesture and significantly increased gesture diversity, resulting in more engaging talks. We conclude with design implications for future AI-driven rehearsal systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10681v1">EMAFusion: A Self-Optimizing System for Seamless LLM Selection and Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      While recent advances in large language models (LLMs) have significantly enhanced performance across diverse natural language tasks, the high computational and financial costs associated with their deployment remain substantial barriers. Existing routing strategies partially alleviate this challenge by assigning queries to cheaper or specialized models, but they frequently rely on extensive labeled data or fragile task-specific heuristics. Conversely, fusion techniques aggregate multiple LLM outputs to boost accuracy and robustness, yet they often exacerbate cost and may reinforce shared biases. We introduce EMAFusion, a new framework that self-optimizes for seamless LLM selection and reliable execution for a given query. Specifically, EMAFusion integrates a taxonomy-based router for familiar query types, a learned router for ambiguous inputs, and a cascading approach that progressively escalates from cheaper to more expensive models based on multi-judge confidence evaluations. Through extensive evaluations, we find EMAFusion outperforms the best individual models by over 2.6 percentage points (94.3% vs. 91.7%), while being 4X cheaper than the average cost. EMAFusion further achieves a remarkable 17.1 percentage point improvement over models like GPT-4 at less than 1/20th the cost. Our combined routing approach delivers 94.3% accuracy compared to taxonomy-based (88.1%) and learned model predictor-based (91.7%) methods alone, demonstrating the effectiveness of our unified strategy. Finally, EMAFusion supports flexible cost-accuracy trade-offs, allowing users to balance their budgetary constraints and performance needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10660v1">LITERA: An LLM Based Approach to Latin-to-English Translation</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 NAACL Findings
    </div>
    <details class="paper-abstract">
      This paper introduces an LLM-based Latin-to-English translation platform designed to address the challenges of translating Latin texts. We named the model LITERA, which stands for Latin Interpretation and Translations into English for Research Assistance. Through a multi-layered translation process utilizing a fine-tuned version of GPT-4o-mini and GPT-4o, LITERA offers an unprecedented level of accuracy, showcased by greatly improved BLEU scores, particularly in classical Latin, along with improved BLEURT scores. The development of LITERA involved close collaboration with Duke University's Classical Studies Department, which was instrumental in creating a small, high-quality parallel Latin-English dataset. This paper details the architecture, fine-tuning methodology, and prompting strategies used in LITERA, emphasizing its ability to produce literal translations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10646v1">Weight-of-Thought Reasoning: Exploring Neural Network Weights for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capabilities when prompted with strategies such as Chain-of-Thought (CoT). However, these approaches focus on token-level output without considering internal weight dynamics. We introduce Weight-of-Thought (WoT) reasoning, a novel approach that examines neural network weights before inference to identify reasoning pathways. Unlike existing methods, WoT explores the weight space through graph-based message passing, multi-step reasoning processes, and attention mechanisms. Our implementation creates an interconnected graph of reasoning nodes. Experiments on diverse reasoning tasks (syllogistic, mathematical, algebraic, combinatorial, and geometric) demonstrate that WoT achieves superior performance compared to traditional methods, particularly for complex problems. This approach leads to both improved performance and greater interpretability of the reasoning process, offering a promising direction for enhancing LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10557v1">The Code Barrier: What LLMs Actually Understand?</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Understanding code represents a core ability needed for automating software development tasks. While foundation models like LLMs show impressive results across many software engineering challenges, the extent of their true semantic understanding beyond simple token recognition remains unclear. This research uses code obfuscation as a structured testing framework to evaluate LLMs' semantic understanding capabilities. We methodically apply controlled obfuscation changes to source code and measure comprehension through two complementary tasks: generating accurate descriptions of obfuscated code and performing deobfuscation, a skill with important implications for reverse engineering applications. Our testing approach includes 13 cutting-edge models, covering both code-specialized (e.g., StarCoder2) and general-purpose (e.g., GPT-4o) architectures, evaluated on a benchmark created from CodeNet and consisting of filtered 250 Java programming problems and their solutions. Findings show a statistically significant performance decline as obfuscation complexity increases, with unexpected resilience shown by general-purpose models compared to their code-focused counterparts. While some models successfully identify obfuscation techniques, their ability to reconstruct the underlying program logic remains constrained, suggesting limitations in their semantic representation mechanisms. This research introduces a new evaluation approach for assessing code comprehension in language models and establishes empirical baselines for advancing research in security-critical code analysis applications such as reverse engineering and adversarial code analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11497v1">LLM-based AI Agent for Sizing of Analog and Mixed Signal Circuit</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 to be presented in IEEE NEWCAS 2025
    </div>
    <details class="paper-abstract">
      The design of Analog and Mixed-Signal (AMS) integrated circuits (ICs) often involves significant manual effort, especially during the transistor sizing process. While Machine Learning techniques in Electronic Design Automation (EDA) have shown promise in reducing complexity and minimizing human intervention, they still face challenges such as numerous iterations and a lack of knowledge about AMS circuit design. Recently, Large Language Models (LLMs) have demonstrated significant potential across various fields, showing a certain level of knowledge in circuit design and indicating their potential to automate the transistor sizing process. In this work, we propose an LLM-based AI agent for AMS circuit design to assist in the sizing process. By integrating LLMs with external circuit simulation tools and data analysis functions and employing prompt engineering strategies, the agent successfully optimized multiple circuits to achieve target performance metrics. We evaluated the performance of different LLMs to assess their applicability and optimization effectiveness across seven basic circuits, and selected the best-performing model Claude 3.5 Sonnet for further exploration on an operational amplifier, with complementary input stage and class AB output stage. This circuit was evaluated against nine performance metrics, and we conducted experiments under three distinct performance requirement groups. A success rate of up to 60% was achieved for reaching the target requirements. Overall, this work demonstrates the potential of LLMs to improve AMS circuit design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12337v1">"It Listens Better Than My Therapist": Exploring Social Media Discourse on LLMs as Mental Health Tool</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 This study does not endorse or encourage the use of AI tools as substitutes for professional mental health support. The findings are presented for research purposes only, and any interpretation should take into account the limitations and potential risks of relying on AI in mental health contexts
    </div>
    <details class="paper-abstract">
      The emergence of generative AI chatbots such as ChatGPT has prompted growing public and academic interest in their role as informal mental health support tools. While early rule-based systems have been around for several years, large language models (LLMs) offer new capabilities in conversational fluency, empathy simulation, and availability. This study explores how users engage with LLMs as mental health tools by analyzing over 10,000 TikTok comments from videos referencing LLMs as mental health tools. Using a self-developed tiered coding schema and supervised classification models, we identify user experiences, attitudes, and recurring themes. Results show that nearly 20% of comments reflect personal use, with these users expressing overwhelmingly positive attitudes. Commonly cited benefits include accessibility, emotional support, and perceived therapeutic value. However, concerns around privacy, generic responses, and the lack of professional oversight remain prominent. It is important to note that the user feedback does not indicate which therapeutic framework, if any, the LLM-generated output aligns with. While the findings underscore the growing relevance of AI in everyday practices, they also highlight the urgent need for clinical and ethical scrutiny in the use of AI for mental health support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13811v1">Can LLMs handle WebShell detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      WebShell attacks, in which malicious scripts are injected into web servers, are a major cybersecurity threat. Traditional machine learning and deep learning methods are hampered by issues such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models (LLMs) have gained attention for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two major contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling (WBFP) that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all models lag behind previous State-Of-The-Art (SOTA) methods. With BFAD, the performance of all LLMs improved, with an average F1 score increase of 13.82%. Larger models such as GPT-4, LLaMA 3.1 70B, and Qwen 2.5 14B outperform SOTA methods, while smaller models such as Qwen 2.5 3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection, and provides solutions to address the challenges in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08727v2">Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Project page: https://boyangdeng.com/visual-chronicles , second and third listed authors have equal contributions
    </div>
    <details class="paper-abstract">
      We present a system using Multimodal LLMs (MLLMs) to analyze a large database with tens of millions of images captured at different times, with the aim of discovering patterns in temporal changes. Specifically, we aim to capture frequent co-occurring changes ("trends") across a city over a certain period. Unlike previous visual analyses, our analysis answers open-ended queries (e.g., "what are the frequent types of changes in the city?") without any predetermined target subjects or training labels. These properties cast prior learning-based or unsupervised visual analysis tools unsuitable. We identify MLLMs as a novel tool for their open-ended semantic understanding capabilities. Yet, our datasets are four orders of magnitude too large for an MLLM to ingest as context. So we introduce a bottom-up procedure that decomposes the massive visual analysis problem into more tractable sub-problems. We carefully design MLLM-based solutions to each sub-problem. During experiments and ablation studies with our system, we find it significantly outperforms baselines and is able to discover interesting trends from images captured in large cities (e.g., "addition of outdoor dining,", "overpass was painted blue," etc.). See more results and interactive demos at https://boyangdeng.com/visual-chronicles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10430v1">LLM Can be a Dangerous Persuader: Empirical Study of Persuasion Safety in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 20 pages, 7 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have enabled them to approach human-level persuasion capabilities. However, such potential also raises concerns about the safety risks of LLM-driven persuasion, particularly their potential for unethical influence through manipulation, deception, exploitation of vulnerabilities, and many other harmful tactics. In this work, we present a systematic investigation of LLM persuasion safety through two critical aspects: (1) whether LLMs appropriately reject unethical persuasion tasks and avoid unethical strategies during execution, including cases where the initial persuasion goal appears ethically neutral, and (2) how influencing factors like personality traits and external pressures affect their behavior. To this end, we introduce PersuSafety, the first comprehensive framework for the assessment of persuasion safety which consists of three stages, i.e., persuasion scene creation, persuasive conversation simulation, and persuasion safety assessment. PersuSafety covers 6 diverse unethical persuasion topics and 15 common unethical strategies. Through extensive experiments across 8 widely used LLMs, we observe significant safety concerns in most LLMs, including failing to identify harmful persuasion tasks and leveraging various unethical persuasion strategies. Our study calls for more attention to improve safety alignment in progressive and goal-driven conversations such as persuasion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10421v1">Can We Edit LLMs for Long-Tail Biomedical Knowledge?</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Knowledge editing has emerged as an effective approach for updating large language models (LLMs) by modifying their internal knowledge. However, their application to the biomedical domain faces unique challenges due to the long-tailed distribution of biomedical knowledge, where rare and infrequent information is prevalent. In this paper, we conduct the first comprehensive study to investigate the effectiveness of knowledge editing methods for editing long-tail biomedical knowledge. Our results indicate that, while existing editing methods can enhance LLMs' performance on long-tail biomedical knowledge, their performance on long-tail knowledge remains inferior to that on high-frequency popular knowledge, even after editing. Our further analysis reveals that long-tail biomedical knowledge contains a significant amount of one-to-many knowledge, where one subject and relation link to multiple objects. This high prevalence of one-to-many knowledge limits the effectiveness of knowledge editing in improving LLMs' understanding of long-tail biomedical knowledge, highlighting the need for tailored strategies to bridge this performance gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10397v1">Can LLMs Assist Expert Elicitation for Probabilistic Causal Modeling?</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Objective: This study investigates the potential of Large Language Models (LLMs) as an alternative to human expert elicitation for extracting structured causal knowledge and facilitating causal modeling in biometric and healthcare applications. Material and Methods: LLM-generated causal structures, specifically Bayesian networks (BNs), were benchmarked against traditional statistical methods (e.g., Bayesian Information Criterion) using healthcare datasets. Validation techniques included structural equation modeling (SEM) to verifying relationships, and measures such as entropy, predictive accuracy, and robustness to compare network structures. Results and Discussion: LLM-generated BNs demonstrated lower entropy than expert-elicited and statistically generated BNs, suggesting higher confidence and precision in predictions. However, limitations such as contextual constraints, hallucinated dependencies, and potential biases inherited from training data require further investigation. Conclusion: LLMs represent a novel frontier in expert elicitation for probabilistic causal modeling, promising to improve transparency and reduce uncertainty in the decision-making using such models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10391v1">LLM-driven Constrained Copy Generation through Iterative Refinement</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 10 pages, 2 figures, 7 Tables
    </div>
    <details class="paper-abstract">
      Crafting a marketing message (copy), or copywriting is a challenging generation task, as the copy must adhere to various constraints. Copy creation is inherently iterative for humans, starting with an initial draft followed by successive refinements. However, manual copy creation is time-consuming and expensive, resulting in only a few copies for each use case. This limitation restricts our ability to personalize content to customers. Contrary to the manual approach, LLMs can generate copies quickly, but the generated content does not consistently meet all the constraints on the first attempt (similar to humans). While recent studies have shown promise in improving constrained generation through iterative refinement, they have primarily addressed tasks with only a few simple constraints. Consequently, the effectiveness of iterative refinement for tasks such as copy generation, which involves many intricate constraints, remains unclear. To address this gap, we propose an LLM-based end-to-end framework for scalable copy generation using iterative refinement. To the best of our knowledge, this is the first study to address multiple challenging constraints simultaneously in copy generation. Examples of these constraints include length, topics, keywords, preferred lexical ordering, and tone of voice. We demonstrate the performance of our framework by creating copies for e-commerce banners for three different use cases of varying complexity. Our results show that iterative refinement increases the copy success rate by $16.25-35.91$% across use cases. Furthermore, the copies generated using our approach outperformed manually created content in multiple pilot studies using a multi-armed bandit framework. The winning copy improved the click-through rate by $38.5-45.21$%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10369v1">SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 16 pages, 8 figures, 7 tables. Under Review
    </div>
    <details class="paper-abstract">
      Optimizing Register Transfer Level (RTL) code is crucial for improving the power, performance, and area (PPA) of digital circuits in the early stages of synthesis. Manual rewriting, guided by synthesis feedback, can yield high-quality results but is time-consuming and error-prone. Most existing compiler-based approaches have difficulty handling complex design constraints. Large Language Model (LLM)-based methods have emerged as a promising alternative to address these challenges. However, LLM-based approaches often face difficulties in ensuring alignment between the generated code and the provided prompts. This paper presents SymRTLO, a novel neuron-symbolic RTL optimization framework that seamlessly integrates LLM-based code rewriting with symbolic reasoning techniques. Our method incorporates a retrieval-augmented generation (RAG) system of optimization rules and Abstract Syntax Tree (AST)-based templates, enabling LLM-based rewriting that maintains syntactic correctness while minimizing undesired circuit behaviors. A symbolic module is proposed for analyzing and optimizing finite state machine (FSM) logic, allowing fine-grained state merging and partial specification handling beyond the scope of pattern-based compilers. Furthermore, a fast verification pipeline, combining formal equivalence checks with test-driven validation, further reduces the complexity of verification. Experiments on the RTL-Rewriter benchmark with Synopsys Design Compiler and Yosys show that SymRTLO improves power, performance, and area (PPA) by up to 43.9%, 62.5%, and 51.1%, respectively, compared to the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10356v1">MultiLoKo: a multilingual local knowledge benchmark for LLMs spanning 31 languages</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      We present MultiLoKo, a new benchmark for evaluating multilinguality in LLMs covering 31 languages. MultiLoKo consists of three partitions: a main partition consisting of 500 questions per language, separately sourced to be locally relevant to the specific language, and two translated partitions, containing human-authored translations from 30 non-English languages to English and vice versa. For comparison, we also release corresponding machine-authored translations. The data is equally distributed over two splits: a dev split and a blind, out-of-distribution test split. MultiLoKo can be used to study a variety of questions regarding the multilinguality of LLMs as well as meta-questions about multilingual benchmark creation. We compute MultiLoKo scores for 11 base and chat models marketed to be multilingual and study their average performance, their performance parity across languages, how much their ability to answer questions depends on the question language, and which languages are most difficult. None of the models we studied performs well on MultiLoKo, as indicated by low average scores as well as large differences between the best and worst scoring languages. Furthermore, we find a substantial effect of the question language, indicating sub-optimal knowledge transfer between languages. Lastly, we find that using local vs English-translated data can result in differences more than 20 points for the best performing models, drastically change the estimated difficulty of some languages. For using machines instead of human translations, we find a weaker effect on ordering of language difficulty, a larger difference in model rankings, and a substantial drop in estimated performance for all models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10326v1">AlayaDB: The Data Foundation for Efficient and Effective Long-context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 14 pages, 12 figures, conference
    </div>
    <details class="paper-abstract">
      AlayaDB is a cutting-edge vector database system natively architected for efficient and effective long-context inference for Large Language Models (LLMs) at AlayaDB AI. Specifically, it decouples the KV cache and attention computation from the LLM inference systems, and encapsulates them into a novel vector database system. For the Model as a Service providers (MaaS), AlayaDB consumes fewer hardware resources and offers higher generation quality for various workloads with different kinds of Service Level Objectives (SLOs), when comparing with the existing alternative solutions (e.g., KV cache disaggregation, retrieval-based sparse attention). The crux of AlayaDB is that it abstracts the attention computation and cache management for LLM inference into a query processing procedure, and optimizes the performance via a native query optimizer. In this work, we demonstrate the effectiveness of AlayaDB via (i) three use cases from our industry partners, and (ii) extensive experimental results on LLM inference benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14763v2">Enabling Scalable Evaluation of Bias Patterns in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive potential in helping with numerous medical challenges. Deploying LLMs in high-stakes applications such as medicine, however, brings in many concerns. One major area of concern relates to biased behaviors of LLMs in medical applications, leading to unfair treatment of individuals. To pave the way for the responsible and impactful deployment of Med LLMs, rigorous evaluation is a key prerequisite. Due to the huge complexity and variability of different medical scenarios, existing work in this domain has primarily relied on using manually crafted datasets for bias evaluation. In this study, we present a new method to scale up such bias evaluations by automatically generating test cases based on rigorous medical evidence. We specifically target the challenges of a) domain-specificity of bias characterization, b) hallucinating while generating the test cases, and c) various dependencies between the health outcomes and sensitive attributes. To that end, we offer new methods to address these challenges integrated with our generative pipeline, using medical knowledge graphs, medical ontologies, and customized general LLM evaluation frameworks in our method. Through a series of extensive experiments, we show that the test cases generated by our proposed method can effectively reveal bias patterns in Med LLMs at larger and more flexible scales than human-crafted datasets. We publish a large bias evaluation dataset using our pipeline, which is dedicated to a few medical case studies. A live demo of our application for vignette generation is available at https://vignette.streamlit.app. Our code is also available at https://github.com/healthylaife/autofair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09420v1">SaRO: Enhancing LLM Safety through Reasoning-based Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Current safety alignment techniques for large language models (LLMs) face two key challenges: (1) under-generalization, which leaves models vulnerable to novel jailbreak attacks, and (2) over-alignment, which leads to the excessive refusal of benign instructions. Our preliminary investigation reveals semantic overlap between jailbreak/harmful queries and normal prompts in embedding space, suggesting that more effective safety alignment requires a deeper semantic understanding. This motivates us to incorporate safety-policy-driven reasoning into the alignment process. To this end, we propose the Safety-oriented Reasoning Optimization Framework (SaRO), which consists of two stages: (1) Reasoning-style Warmup (RW) that enables LLMs to internalize long-chain reasoning through supervised fine-tuning, and (2) Safety-oriented Reasoning Process Optimization (SRPO) that promotes safety reflection via direct preference optimization (DPO). Extensive experiments demonstrate the superiority of SaRO over traditional alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09407v1">UXAgent: A System for Simulating Usability Testing of Web Design with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Usability testing is a fundamental research method that user experience (UX) researchers use to evaluate and iterate a web design, but\textbf{ how to evaluate and iterate the usability testing study design } itself? Recent advances in Large Language Model-simulated Agent (\textbf{LLM Agent}) research inspired us to design \textbf{UXAgent} to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human-subject study. Our system features a Persona Generator module, an LLM Agent module, and a Universal Browser Connector module to automatically generate thousands of simulated users to interactively test the target website. The system also provides an Agent Interview Interface and a Video Replay Interface so that the UX researchers can easily review and analyze the generated qualitative and quantitative log data. Through a heuristic evaluation, five UX researcher participants praised the innovation of our system but also expressed concerns about the future of LLM Agent usage in UX studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09378v1">Can you map it to English? The Role of Cross-Lingual Alignment in Multilingual Performance of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) pre-trained predominantly on English text exhibit surprising multilingual capabilities, yet the mechanisms driving cross-lingual generalization remain poorly understood. This work investigates how the alignment of representations for text written in different languages correlates with LLM performance on natural language understanding tasks and translation tasks, both at the language and the instance level. For this purpose, we introduce cross-lingual alignment metrics such as the Discriminative Alignment Index (DALI) to quantify the alignment at an instance level for discriminative tasks. Through experiments on three natural language understanding tasks (Belebele, XStoryCloze, XCOPA), and machine translation, we find that while cross-lingual alignment metrics strongly correlate with task accuracy at the language level, the sample-level alignment often fails to distinguish correct from incorrect predictions, exposing alignment as a necessary but insufficient condition for success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10541v1">Multi-Modal Hypergraph Enhanced LLM Learning for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The burgeoning presence of Large Language Models (LLM) is propelling the development of personalized recommender systems. Most existing LLM-based methods fail to sufficiently explore the multi-view graph structure correlations inherent in recommendation scenarios. To this end, we propose a novel framework, Hypergraph Enhanced LLM Learning for multimodal Recommendation (HeLLM), designed to equip LLMs with the capability to capture intricate higher-order semantic correlations by fusing graph-level contextual signals with sequence-level behavioral patterns. In the recommender pre-training phase, we design a user hypergraph to uncover shared interest preferences among users and an item hypergraph to capture correlations within multimodal similarities among items. The hypergraph convolution and synergistic contrastive learning mechanism are introduced to enhance the distinguishability of learned representations. In the LLM fine-tuning phase, we inject the learned graph-structured embeddings directly into the LLM's architecture and integrate sequential features capturing each user's chronological behavior. This process enables hypergraphs to leverage graph-structured information as global context, enhancing the LLM's ability to perceive complex relational patterns and integrate multimodal information, while also modeling local temporal dynamics. Extensive experiments demonstrate the superiority of our proposed method over state-of-the-art baselines, confirming the advantages of fusing hypergraph-based context with sequential user behavior in LLMs for recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12333v1">Meta-Evaluating Local LLMs: Rethinking Performance Metrics for Serious Games</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 2nd HEAL Workshop at CHI Conference on Human Factors in Computing Systems. April 26, 2025. Yokohama, Japan
    </div>
    <details class="paper-abstract">
      The evaluation of open-ended responses in serious games presents a unique challenge, as correctness is often subjective. Large Language Models (LLMs) are increasingly being explored as evaluators in such contexts, yet their accuracy and consistency remain uncertain, particularly for smaller models intended for local execution. This study investigates the reliability of five small-scale LLMs when assessing player responses in \textit{En-join}, a game that simulates decision-making within energy communities. By leveraging traditional binary classification metrics (including accuracy, true positive rate, and true negative rate), we systematically compare these models across different evaluation scenarios. Our results highlight the strengths and limitations of each model, revealing trade-offs between sensitivity, specificity, and overall performance. We demonstrate that while some models excel at identifying correct responses, others struggle with false positives or inconsistent evaluations. The findings highlight the need for context-aware evaluation frameworks and careful model selection when deploying LLMs as evaluators. This work contributes to the broader discourse on the trustworthiness of AI-driven assessment tools, offering insights into how different LLM architectures handle subjective evaluation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12331v1">Span-level Emotion-Cause-Category Triplet Extraction with Instruction Tuning LLMs and Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Span-level emotion-cause-category triplet extraction represents a novel and complex challenge within emotion cause analysis. This task involves identifying emotion spans, cause spans, and their associated emotion categories within the text to form structured triplets. While prior research has predominantly concentrated on clause-level emotion-cause pair extraction and span-level emotion-cause detection, these methods often confront challenges originating from redundant information retrieval and difficulty in accurately determining emotion categories, particularly when emotions are expressed implicitly or ambiguously. To overcome these challenges, this study explores a fine-grained approach to span-level emotion-cause-category triplet extraction and introduces an innovative framework that leverages instruction tuning and data augmentation techniques based on large language models. The proposed method employs task-specific triplet extraction instructions and utilizes low-rank adaptation to fine-tune large language models, eliminating the necessity for intricate task-specific architectures. Furthermore, a prompt-based data augmentation strategy is developed to address data scarcity by guiding large language models in generating high-quality synthetic training data. Extensive experimental evaluations demonstrate that the proposed approach significantly outperforms existing baseline methods, achieving at least a 12.8% improvement in span-level emotion-cause-category triplet extraction metrics. The results demonstrate the method's effectiveness and robustness, offering a promising avenue for advancing research in emotion cause analysis. The source code is available at https://github.com/zxgnlp/InstruDa-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09757v1">Alleviating the Fear of Losing Alignment in LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated revolutionary capabilities in understanding complex contexts and performing a wide range of tasks. However, LLMs can also answer questions that are unethical or harmful, raising concerns about their applications. To regulate LLMs' responses to such questions, a training strategy called \textit{alignment} can help. Yet, alignment can be unexpectedly compromised when fine-tuning an LLM for downstream tasks. This paper focuses on recovering the alignment lost during fine-tuning. We observe that there are two distinct directions inherent in an aligned LLM: the \textit{aligned direction} and the \textit{harmful direction}. An LLM is inclined to answer questions in the aligned direction while refusing queries in the harmful direction. Therefore, we propose to recover the harmful direction of the fine-tuned model that has been compromised. Specifically, we restore a small subset of the fine-tuned model's weight parameters from the original aligned model using gradient descent. We also introduce a rollback mechanism to avoid aggressive recovery and maintain downstream task performance. Our evaluation on 125 fine-tuned LLMs demonstrates that our method can reduce their harmful rate (percentage of answering harmful questions) from 33.25\% to 1.74\%, without sacrificing task performance much. In contrast, the existing methods either only reduce the harmful rate to a limited extent or significantly impact the normal functionality. Our code is available at https://github.com/kangyangWHU/LLMAlignment
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09737v1">Can LLM feedback enhance review quality? A randomized study of 20K reviews at ICLR 2025</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 30 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Peer review at AI conferences is stressed by rapidly rising submission volumes, leading to deteriorating review quality and increased author dissatisfaction. To address these issues, we developed Review Feedback Agent, a system leveraging multiple large language models (LLMs) to improve review clarity and actionability by providing automated feedback on vague comments, content misunderstandings, and unprofessional remarks to reviewers. Implemented at ICLR 2025 as a large randomized control study, our system provided optional feedback to more than 20,000 randomly selected reviews. To ensure high-quality feedback for reviewers at this scale, we also developed a suite of automated reliability tests powered by LLMs that acted as guardrails to ensure feedback quality, with feedback only being sent to reviewers if it passed all the tests. The results show that 27% of reviewers who received feedback updated their reviews, and over 12,000 feedback suggestions from the agent were incorporated by those reviewers. This suggests that many reviewers found the AI-generated feedback sufficiently helpful to merit updating their reviews. Incorporating AI feedback led to significantly longer reviews (an average increase of 80 words among those who updated after receiving feedback) and more informative reviews, as evaluated by blinded researchers. Moreover, reviewers who were selected to receive AI feedback were also more engaged during paper rebuttals, as seen in longer author-reviewer discussions. This work demonstrates that carefully designed LLM-generated review feedback can enhance peer review quality by making reviews more specific and actionable while increasing engagement between reviewers and authors. The Review Feedback Agent is publicly available at https://github.com/zou-group/review_feedback_agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15793v3">DNR Bench: Benchmarking Over-Reasoning in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Test-time scaling has significantly improved large language model performance, enabling deeper reasoning to solve complex problems. However, this increased reasoning capability also leads to excessive token generation and unnecessary problem-solving attempts. We introduce Don\'t Answer Bench (DNA Bench), a new benchmark designed to evaluate LLMs ability to robustly understand the tricky reasoning triggers and avoiding unnecessary generation. DNA Bench consists of 150 adversarially designed prompts that are easy for humans to understand and respond to, but surprisingly not for many of the recent prominent LLMs. DNA Bench tests models abilities across different capabilities, such as instruction adherence, hallucination avoidance, redundancy filtering, and unanswerable question recognition. We evaluate reasoning LLMs (RLMs), including DeepSeek-R1, OpenAI O3-mini, Claude-3.7-sonnet and compare them against a powerful non-reasoning model, e.g., GPT-4o. Our experiments reveal that RLMs generate up to 70x more tokens than necessary, often failing at tasks that simpler non-reasoning models handle efficiently with higher accuracy. Our findings underscore the need for more effective training and inference strategies in RLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06370v2">Towards a Probabilistic Framework for Analyzing and Improving LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Ensuring the reliability and verifiability of large language model (LLM)-enabled systems remains a significant challenge in software engineering. We propose a probabilistic framework for systematically analyzing and improving these systems by modeling and refining distributions over clusters of semantically equivalent outputs. This framework facilitates the evaluation and iterative improvement of Transference Models--key software components that utilize LLMs to transform inputs into outputs for downstream tasks. To illustrate its utility, we apply the framework to the autoformalization problem, where natural language documentation is transformed into formal program specifications. Our case illustrates how distribution-aware analysis enables the identification of weaknesses and guides focused alignment improvements, resulting in more reliable and interpretable outputs. This principled approach offers a foundation for addressing critical challenges in the development of robust LLM-enabled systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09723v1">AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents Amazon.com, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09710v1">DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL)-based post-training have led to notable improvements in large language models (LLMs), particularly in enhancing their reasoning capabilities to handle complex tasks. However, most existing methods treat the training data as a unified whole, overlooking the fact that modern LLM training often involves a mixture of data from diverse distributions-varying in both source and difficulty. This heterogeneity introduces a key challenge: how to adaptively schedule training across distributions to optimize learning efficiency. In this paper, we present a principled curriculum learning framework grounded in the notion of distribution-level learnability. Our core insight is that the magnitude of policy advantages reflects how much a model can still benefit from further training on a given distribution. Based on this, we propose a distribution-level curriculum learning framework for RL-based LLM post-training, which leverages the Upper Confidence Bound (UCB) principle to dynamically adjust sampling probabilities for different distrubutions. This approach prioritizes distributions with either high average advantage (exploitation) or low sample count (exploration), yielding an adaptive and theoretically grounded training schedule. We instantiate our curriculum learning framework with GRPO as the underlying RL algorithm and demonstrate its effectiveness on logic reasoning datasets with multiple difficulties and sources. Our experiments show that our framework significantly improves convergence speed and final performance, highlighting the value of distribution-aware curriculum strategies in LLM post-training. Code: https://github.com/ZhentingWang/DUMP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09691v1">Migrating Code At Scale With LLMs At Google</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Developers often evolve an existing software system by making internal changes, called migration. Moving to a new framework, changing implementation to improve efficiency, and upgrading a dependency to its latest version are examples of migrations. Migration is a common and typically continuous maintenance task undertaken either manually or through tooling. Certain migrations are labor intensive and costly, developers do not find the required work rewarding, and they may take years to complete. Hence, automation is preferred for such migrations. In this paper, we discuss a large-scale, costly and traditionally manual migration project at Google, propose a novel automated algorithm that uses change location discovery and a Large Language Model (LLM) to aid developers conduct the migration, report the results of a large case study, and discuss lessons learned. Our case study on 39 distinct migrations undertaken by three developers over twelve months shows that a total of 595 code changes with 93,574 edits have been submitted, where 74.45% of the code changes and 69.46% of the edits were generated by the LLM. The developers reported high satisfaction with the automated tooling, and estimated a 50% reduction on the total time spent on the migration compared to earlier manual migrations. Our results suggest that our automated, LLM-assisted workflow can serve as a model for similar initiatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16395v2">HELIOT: LLM-Based CDSS for Adverse Drug Reaction Management</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Medication errors significantly threaten patient safety, leading to adverse drug events and substantial economic burdens on healthcare systems. Clinical Decision Support Systems (CDSSs) aimed at mitigating these errors often face limitations when processing unstructured clinical data, including reliance on static databases and rule-based algorithms, frequently generating excessive alerts that lead to alert fatigue among healthcare providers. This paper introduces HELIOT, an innovative CDSS for adverse drug reaction management that processes free-text clinical information using Large Language Models (LLMs) integrated with a comprehensive pharmaceutical data repository. HELIOT leverages advanced natural language processing capabilities to interpret medical narratives, extract relevant drug reaction information from unstructured clinical notes, and learn from past patient-specific medication tolerances to reduce false alerts, enabling more nuanced and contextual adverse drug event warnings across primary care, specialist consultations, and hospital settings. An initial evaluation using a synthetic dataset of clinical narratives and expert-verified ground truth shows promising results. HELIOT achieves high accuracy in a controlled setting. In addition, by intelligently analyzing previous medication tolerance documented in clinical notes and distinguishing between cases requiring different alert types, HELIOT can potentially reduce interruptive alerts by over 50% compared to traditional CDSSs. While these preliminary findings are encouraging, real-world validation will be essential to confirm these benefits in clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09685v1">Can LLMs Revolutionize the Design of Explainable and Efficient TinyML Models?</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      This paper introduces a novel framework for designing efficient neural network architectures specifically tailored to tiny machine learning (TinyML) platforms. By leveraging large language models (LLMs) for neural architecture search (NAS), a vision transformer (ViT)-based knowledge distillation (KD) strategy, and an explainability module, the approach strikes an optimal balance between accuracy, computational efficiency, and memory usage. The LLM-guided search explores a hierarchical search space, refining candidate architectures through Pareto optimization based on accuracy, multiply-accumulate operations (MACs), and memory metrics. The best-performing architectures are further fine-tuned using logits-based KD with a pre-trained ViT-B/16 model, which enhances generalization without increasing model size. Evaluated on the CIFAR-100 dataset and deployed on an STM32H7 microcontroller (MCU), the three proposed models, LMaNet-Elite, LMaNet-Core, and QwNet-Core, achieve accuracy scores of 74.50%, 74.20% and 73.00%, respectively. All three models surpass current state-of-the-art (SOTA) models, such as MCUNet-in3/in4 (69.62% / 72.86%) and XiNet (72.27%), while maintaining a low computational cost of less than 100 million MACs and adhering to the stringent 320 KB static random-access memory (SRAM) constraint. These results demonstrate the efficiency and performance of the proposed framework for TinyML platforms, underscoring the potential of combining LLM-driven search, Pareto optimization, KD, and explainability to develop accurate, efficient, and interpretable models. This approach opens new possibilities in NAS, enabling the design of efficient architectures specifically suited for TinyML.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09597v1">Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across numerous tasks, yet principled explanations for their underlying mechanisms and several phenomena, such as scaling laws, hallucinations, and related behaviors, remain elusive. In this work, we revisit the classical relationship between compression and prediction, grounded in Kolmogorov complexity and Shannon information theory, to provide deeper insights into LLM behaviors. By leveraging the Kolmogorov Structure Function and interpreting LLM compression as a two-part coding process, we offer a detailed view of how LLMs acquire and store information across increasing model and data scales -- from pervasive syntactic patterns to progressively rarer knowledge elements. Motivated by this theoretical perspective and natural assumptions inspired by Heap's and Zipf's laws, we introduce a simplified yet representative hierarchical data-generation framework called the Syntax-Knowledge model. Under the Bayesian setting, we show that prediction and compression within this model naturally lead to diverse learning and scaling behaviors of LLMs. In particular, our theoretical analysis offers intuitive and principled explanations for both data and model scaling laws, the dynamics of knowledge acquisition during training and fine-tuning, factual knowledge hallucinations in LLMs. The experimental results validate our theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09593v1">ControlNET: A Firewall for RAG-based LLM System</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09590v1">Efficient LLM Serving on Hybrid Real-time and Best-effort Requests</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in large Language Models (LLMs) have enabled various generative tasks on a single model. Real-world services (e.g., OpenAI's ChatGPT [27]) powered by an LLM often concurrently support latency-critical requests for interactive applications (e.g., question-answering systems, referred to as real-time or RT requests) and throughput-oriented requests for back-of-house processing (e.g., documents batch processing [28], referred to best-effort or BE requests), with complex hybrid inference workloads to the underlying model. State-of-the-art (SOTA) LLM serving systems dedicate machines to each type of request, towards either low inference latency or high serving throughput, respectively. This practice simplifies request scheduling and management but suffers from poor resource utilization. We propose BROS, a hybrid LLM serving system that aims to collocate RT/BE requests, meeting RT requests' latency requirements while maintaining BE requests' throughput. BROS formulates the problem of hybrid RT/BE request scheduling and solves it with a dynamic priority-based algorithm. BROS designs a bidirectional KV cache management mechanism, allowing RT requests to share KV memory with BE requests to remove the scheduling restrictions caused by insufficient KV memory and improve utilization. Extensive experiments validate that BROS achieves a good trade-off when serving hybrid RT and BE requests. It significantly reduces the latency of RT requests (up to 74.20%), improving their fine-grained service level objectives (SLOs) attainments (up to 36.38x), with negligible throughput reduction for BE requests, showing significant advantages over SOTA systems like vLLM and TGI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09586v1">Short-Path Prompting in LLMs: Analyzing Reasoning Instability and Solutions for Robust Performance</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent years have witnessed significant progress in large language models' (LLMs) reasoning, which is largely due to the chain-of-thought (CoT) approaches, allowing models to generate intermediate reasoning steps before reaching the final answer. Building on these advances, state-of-the-art LLMs are instruction-tuned to provide long and detailed CoT pathways when responding to reasoning-related questions. However, human beings are naturally cognitive misers and will prompt language models to give rather short responses, thus raising a significant conflict with CoT reasoning. In this paper, we delve into how LLMs' reasoning performance changes when users provide short-path prompts. The results and analysis reveal that language models can reason effectively and robustly without explicit CoT prompts, while under short-path prompting, LLMs' reasoning ability drops significantly and becomes unstable, even on grade-school problems. To address this issue, we propose two approaches: an instruction-guided approach and a fine-tuning approach, both designed to effectively manage the conflict. Experimental results show that both methods achieve high accuracy, providing insights into the trade-off between instruction adherence and reasoning accuracy in current models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09570v1">LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      When the complete source sentence is provided, Large Language Models (LLMs) perform excellently in offline machine translation even with a simple prompt "Translate the following sentence from [src lang] into [tgt lang]:". However, in many real scenarios, the source tokens arrive in a streaming manner and simultaneous machine translation (SiMT) is required, then the efficiency and performance of decoder-only LLMs are significantly limited by their auto-regressive nature. To enable LLMs to achieve high-quality SiMT as efficiently as offline translation, we propose a novel paradigm that includes constructing supervised fine-tuning (SFT) data for SiMT, along with new training and inference strategies. To replicate the token input/output stream in SiMT, the source and target tokens are rearranged into an interleaved sequence, separated by special tokens according to varying latency requirements. This enables powerful LLMs to learn read and write operations adaptively, based on varying latency prompts, while still maintaining efficient auto-regressive decoding. Experimental results show that, even with limited SFT data, our approach achieves state-of-the-art performance across various SiMT benchmarks, and preserves the original abilities of offline translation. Moreover, our approach generalizes well to document-level SiMT setting without requiring specific fine-tuning, even beyond the offline translation model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09566v1">Syzygy of Thoughts: Improving LLM CoT with the Minimal Free Resolution</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances the reasoning of large language models (LLMs) by decomposing problems into sequential steps, mimicking human logic and reducing errors. However, complex tasks with vast solution spaces and vague constraints often exceed the capacity of a single reasoning chain. Inspired by Minimal Free Resolution (MFR) in commutative algebra and algebraic geometry, we propose Syzygy of Thoughts (SoT)-a novel framework that extends CoT by introducing auxiliary, interrelated reasoning paths. SoT captures deeper logical dependencies, enabling more robust and structured problem-solving. MFR decomposes a module into a sequence of free modules with minimal rank, providing a structured analytical approach to complex systems. This method introduces the concepts of "Module", "Betti numbers","Freeness", "Mapping", "Exactness" and "Minimality", enabling the systematic decomposition of the original complex problem into logically complete minimal subproblems while preserving key problem features and reducing reasoning length. We tested SoT across diverse datasets (e.g., GSM8K, MATH) and models (e.g., GPT-4o-mini, Qwen2.5), achieving inference accuracy that matches or surpasses mainstream CoTs standards. Additionally, by aligning the sampling process with algebraic constraints, our approach enhances the scalability of inference time in LLMs, ensuring both transparent reasoning and high performance. Our code will be publicly available at https://github.com/dlMARiA/Syzygy-of-thoughts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09561v1">LoopLynx: A Scalable Dataflow Architecture for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      In this paper, we propose LoopLynx, a scalable dataflow architecture for efficient LLM inference that optimizes FPGA usage through a hybrid spatial-temporal design. The design of LoopLynx incorporates a hybrid temporal-spatial architecture, where computationally intensive operators are implemented as large dataflow kernels. This achieves high throughput similar to spatial architecture, and organizing and reusing these kernels in a temporal way together enhances FPGA peak performance. Furthermore, to overcome the resource limitations of a single device, we provide a multi-FPGA distributed architecture that overlaps and hides all data transfers so that the distributed accelerators are fully utilized. By doing so, LoopLynx can be effectively scaled to multiple devices to further explore model parallelism for large-scale LLM inference. Evaluation of GPT-2 model demonstrates that LoopLynx can achieve comparable performance to state-of-the-art single FPGA-based accelerations. In addition, compared to Nvidia A100, our accelerator with a dual-FPGA configuration delivers a 2.52x speed-up in inference latency while consuming only 48.1% of the energy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07177v2">MM-Ego: Towards Building Egocentric Multimodal LLMs for Video QA</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      This research aims to comprehensively explore building a multimodal foundation model for egocentric video understanding. To achieve this goal, we work on three fronts. First, as there is a lack of QA data for egocentric video understanding, we automatically generate 7M high-quality QA samples for egocentric videos ranging from 30 seconds to one hour long in Ego4D based on human-annotated data. This is one of the largest egocentric QA datasets. Second, we contribute a challenging egocentric QA benchmark with 629 videos and 7,026 questions to evaluate the models' ability in recognizing and memorizing visual details across videos of varying lengths. We introduce a new de-biasing evaluation method to help mitigate the unavoidable language bias present in the models being evaluated. Third, we propose a specialized multimodal architecture featuring a novel "Memory Pointer Prompting" mechanism. This design includes a \textit{global glimpse} step to gain an overarching understanding of the entire video and identify key visual information, followed by a fallback step that utilizes the key visual information to generate responses. This enables the model to more effectively comprehend extended video content. With the data, benchmark, and model, we build MM-Ego, an egocentric multimodal LLM that shows powerful performance on egocentric video understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09522v1">How new data permeates LLM knowledge and how to dilute it</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Large language models learn and continually learn through the accumulation of gradient-based updates, but how individual pieces of new information affect existing knowledge, leading to both beneficial generalization and problematic hallucination, remains poorly understood. We demonstrate that when learning new information, LLMs exhibit a "priming" effect: learning a new fact can cause the model to inappropriately apply that knowledge in unrelated contexts. To systematically study this phenomenon, we introduce "Outlandish," a carefully curated dataset of 1320 diverse text samples designed to probe how new knowledge permeates through an LLM's existing knowledge base. Using this dataset, we show that the degree of priming after learning new information can be predicted by measuring the token probability of key words before learning. This relationship holds robustly across different model architectures (PALM-2, Gemma, Llama), sizes, and training stages. Finally, we develop two novel techniques to modulate how new knowledge affects existing model behavior: (1) a ``stepping-stone'' text augmentation strategy and (2) an ``ignore-k'' update pruning method. These approaches reduce undesirable priming effects by 50-95\% while preserving the model's ability to learn new information. Our findings provide both empirical insights into how LLMs learn and practical tools for improving the specificity of knowledge insertion in language models. Further materials: https://sunchipsster1.github.io/projects/outlandish/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09504v1">MADLLM: Multivariate Anomaly Detection via Pre-trained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 Accepted by IEEE International Conference on Multimedia & Expo 2025 (ICME 2025)
    </div>
    <details class="paper-abstract">
      When applying pre-trained large language models (LLMs) to address anomaly detection tasks, the multivariate time series (MTS) modality of anomaly detection does not align with the text modality of LLMs. Existing methods simply transform the MTS data into multiple univariate time series sequences, which can cause many problems. This paper introduces MADLLM, a novel multivariate anomaly detection method via pre-trained LLMs. We design a new triple encoding technique to align the MTS modality with the text modality of LLMs. Specifically, this technique integrates the traditional patch embedding method with two novel embedding approaches: Skip Embedding, which alters the order of patch processing in traditional methods to help LLMs retain knowledge of previous features, and Feature Embedding, which leverages contrastive learning to allow the model to better understand the correlations between different features. Experimental results demonstrate that our method outperforms state-of-the-art methods in various public anomaly detection datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02479v2">From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Future</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      With the rise of large language models (LLMs), researchers are increasingly exploring their applications in var ious vertical domains, such as software engineering. LLMs have achieved remarkable success in areas including code generation and vulnerability detection. However, they also exhibit numerous limitations and shortcomings. LLM-based agents, a novel tech nology with the potential for Artificial General Intelligence (AGI), combine LLMs as the core for decision-making and action-taking, addressing some of the inherent limitations of LLMs such as lack of autonomy and self-improvement. Despite numerous studies and surveys exploring the possibility of using LLMs in software engineering, it lacks a clear distinction between LLMs and LLM based agents. It is still in its early stage for a unified standard and benchmarking to qualify an LLM solution as an LLM-based agent in its domain. In this survey, we broadly investigate the current practice and solutions for LLMs and LLM-based agents for software engineering. In particular we summarise six key topics: requirement engineering, code generation, autonomous decision-making, software design, test generation, and software maintenance. We review and differentiate the work of LLMs and LLM-based agents from these six topics, examining their differences and similarities in tasks, benchmarks, and evaluation metrics. Finally, we discuss the models and benchmarks used, providing a comprehensive analysis of their applications and effectiveness in software engineering. We anticipate this work will shed some lights on pushing the boundaries of LLM-based agents in software engineering for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09482v1">HalluShift: Measuring Distribution Shifts towards Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently garnered widespread attention due to their adeptness at generating innovative responses to the given prompts across a multitude of domains. However, LLMs often suffer from the inherent limitation of hallucinations and generate incorrect information while maintaining well-structured and coherent responses. In this work, we hypothesize that hallucinations stem from the internal dynamics of LLMs. Our observations indicate that, during passage generation, LLMs tend to deviate from factual accuracy in subtle parts of responses, eventually shifting toward misinformation. This phenomenon bears a resemblance to human cognition, where individuals may hallucinate while maintaining logical coherence, embedding uncertainty within minor segments of their speech. To investigate this further, we introduce an innovative approach, HalluShift, designed to analyze the distribution shifts in the internal state space and token probabilities of the LLM-generated responses. Our method attains superior performance compared to existing baselines across various benchmark datasets. Our codebase is available at https://github.com/sharanya-dasgupta001/hallushift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09466v1">AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender</a></div>
    <div class="paper-meta">
      📅 2025-04-13
      | 💬 17 pages, 6 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09373v1">QUDsim: Quantifying Discourse Similarities in LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      As large language models become increasingly capable at various writing tasks, their weakness at generating unique and creative content becomes a major liability. Although LLMs have the ability to generate text covering diverse topics, there is an overall sense of repetitiveness across texts that we aim to formalize and quantify via a similarity metric. The familiarity between documents arises from the persistence of underlying discourse structures. However, existing similarity metrics dependent on lexical overlap and syntactic patterns largely capture $\textit{content}$ overlap, thus making them unsuitable for detecting $\textit{structural}$ similarities. We introduce an abstraction based on linguistic theories in Questions Under Discussion (QUD) and question semantics to help quantify differences in discourse progression. We then use this framework to build $\textbf{QUDsim}$, a similarity metric that can detect discursive parallels between documents. Using QUDsim, we find that LLMs often reuse discourse structures (more so than humans) across samples, even when content differs. Furthermore, LLMs are not only repetitive and structurally uniform, but are also divergent from human authors in the types of structures they use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05209v2">Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the attack success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04104v2">LLMs Can Generate a Better Answer by Aggregating Their Own Responses</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities across tasks, yet they often require additional prompting techniques when facing complex problems. While approaches like self-correction and response selection have emerged as popular solutions, recent studies have shown these methods perform poorly when relying on the LLM itself to provide feedback or selection criteria. We argue this limitation stems from the fact that common LLM post-training procedures lack explicit supervision for discriminative judgment tasks. In this paper, we propose Generative Self-Aggregation (GSA), a novel prompting method that improves answer quality without requiring the model's discriminative capabilities. GSA first samples multiple diverse responses from the LLM, then aggregates them to obtain an improved solution. Unlike previous approaches, our method does not require the LLM to correct errors or compare response quality; instead, it leverages the model's generative abilities to synthesize a new response based on the context of multiple samples. While GSA shares similarities with the self-consistency (SC) approach for response aggregation, SC requires specific verifiable tokens to enable majority voting. In contrast, our approach is more general and can be applied to open-ended tasks. Empirical evaluation demonstrates that GSA effectively improves response quality across various tasks, including mathematical reasoning, knowledge-based problems, and open-ended generation tasks such as code synthesis and conversational responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09345v1">MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Mixture of Experts (MoE) LLMs, characterized by their sparse activation patterns, offer a promising approach to scaling language models while avoiding proportionally increasing the inference cost. However, their large parameter sizes present deployment challenges in resource-constrained environments with limited GPU memory capacity, as GPU memory is often insufficient to accommodate the full set of model weights. Consequently, typical deployments rely on CPU-GPU hybrid execution: the GPU handles compute-intensive GEMM operations, while the CPU processes the relatively lightweight attention mechanism. This setup introduces a key challenge: how to effectively optimize resource utilization across CPU and GPU? Prior work has designed system optimizations based on performance models with limited scope. Specifically, such models do not capture the complex interactions between hardware properties and system execution mechanisms. Therefore, previous approaches neither identify nor achieve the hardware limit. This paper presents MoE-Lens, a high-throughput MoE LLM inference system designed through holistic performance modeling for resource-constrained environments. Our performance model thoroughly analyzes various fundamental system components, including CPU memory capacity, GPU compute power, and workload characteristics, to understand the theoretical performance upper bound of MoE inference. Furthermore, it captures the system execution mechanisms to identify the key hardware bottlenecks and accurately predict the achievable throughput. Informed by our performance model, MoE-Lens introduces an inference system approaching hardware limits. Evaluated on diverse MoE models and datasets, MoE-Lens outperforms the state-of-the-art solution by 4.6x on average (up to 25.5x), with our theoretical model predicting performance with an average 94% accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09307v1">Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 Accepted to MLSys 2025
    </div>
    <details class="paper-abstract">
      Training LLMs in distributed environments presents significant challenges due to the complexity of model execution, deployment systems, and the vast space of configurable strategies. Although various optimization techniques exist, achieving high efficiency in practice remains difficult. Accurate performance models that effectively characterize and predict a model's behavior are essential for guiding optimization efforts and system-level studies. We propose Lumos, a trace-driven performance modeling and estimation toolkit for large-scale LLM training, designed to accurately capture and predict the execution behaviors of modern LLMs. We evaluate Lumos on a production ML cluster with up to 512 NVIDIA H100 GPUs using various GPT-3 variants, demonstrating that it can replay execution time with an average error of just 3.3%, along with other runtime details, across different models and configurations. Additionally, we validate its ability to estimate performance for new setups from existing traces, facilitating efficient exploration of model and deployment configurations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02406v3">Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. We answer the following question: Given a piece of text, can we identify whether it was produced by LLM $A$ or $B$ (where $B$ can be a human)? We model LLM-generated text as a sequential stochastic process with complete dependence on history and design zero-shot statistical tests to distinguish between (i) the text generated by two different sets of LLMs $A$ (in-house) and $B$ (non-sanctioned) and also (ii) LLM-generated and human-generated texts. We prove that our tests' type I and type II errors decrease exponentially as text length increases. For designing our tests for a given string, we demonstrate that if the string is generated by the evaluator model $A$, the log-perplexity of the string under $A$ converges to the average entropy of the string under $A$, except with an exponentially small probability in the string length. We also show that if $B$ generates the text, except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. For our experiments: First, we present experiments using open-source LLMs to support our theoretical results, and then we provide experiments in a black-box setting with adversarial attacks. Practically, our work enables guaranteed finding of the origin of harmful or false LLM-generated text, which can be useful for combating misinformation and compliance with emerging AI regulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05108v3">Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Discovering efficient algorithms for solving complex problems has been an outstanding challenge in mathematics and computer science, requiring substantial human expertise over the years. Recent advancements in evolutionary search with large language models (LLMs) have shown promise in accelerating the discovery of algorithms across various domains, particularly in mathematics and optimization. However, existing approaches treat the LLM as a static generator, missing the opportunity to update the model with the signal obtained from evolutionary exploration. In this work, we propose to augment LLM-based evolutionary search by continuously refining the search operator - the LLM - through reinforcement learning (RL) fine-tuning. Our method leverages evolutionary search as an exploration strategy to discover improved algorithms, while RL optimizes the LLM policy based on these discoveries. Our experiments on three combinatorial optimization tasks - bin packing, traveling salesman, and the flatpack problem - show that combining RL and evolutionary search improves discovery efficiency of improved algorithms, showcasing the potential of RL-enhanced evolutionary strategies to assist computer scientists and mathematicians for more efficient algorithm design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09285v1">DynaServe: Unified and Elastic Tandem-Style Execution for Dynamic Disaggregated LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Modern large language model (LLM) serving must efficiently handle highly dynamic workloads, where prompt and response lengths vary significantly across requests. Existing systems typically adopt either colocated execution, where prefill and decode stages share the same GPU for high throughput, or disaggregated execution, which decouples the two stages and assign their tasks to dedicated GPUs for interference avoidance. However, both paradigms face critical limitations: colocation suffers from resource contention and prolonged tail latency, whereas disaggregation likely leads to resource wasting when prefill or decode GPUs are not fully occupied. To address the above limitations, we introduce DynaServe, a unified LLM serving framework based on the Tandem Serving model. Under this model, DynaServe elastically decomposes each request into two virtual sub-requests that are collaboratively processed by a pair of GPU instances. The Lead GPU handles the initial prompt and early generation, while the Follow GPU completes decoding, enabling dynamic load balancing, fine-grained batching, and coherent execution across distributed resources. By coordinating computation and memory across the cluster, DynaServe adapts to diverse and bursty workloads while maintaining stringent latency service-level objectives (SLOs). Evaluations on real-world traces show that DynaServe improves end-to-end Serving Capacity by up to 1.23 $\times$, increases the overall goodput from 1.15 $\times$ to 4.34 $\times$, and improve the memory utilization by up to 49% compared to state-of-the-art colocated and disaggregated systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09207v1">Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 SIGMOD 2025 Paper
    </div>
    <details class="paper-abstract">
      Finding relevant tables among databases, lakes, and repositories is the first step in extracting value from data. Such a task remains difficult because assessing whether a table is relevant to a problem does not always depend only on its content but also on the context, which is usually tribal knowledge known to the individual or team. While tools like data catalogs and academic data discovery systems target this problem, they rely on keyword search or more complex interfaces, limiting non-technical users' ability to find relevant data. The advent of large language models (LLMs) offers a unique opportunity for users to ask questions directly in natural language, making dataset discovery more intuitive, accessible, and efficient. In this paper, we introduce Pneuma, a retrieval-augmented generation (RAG) system designed to efficiently and effectively discover tabular data. Pneuma leverages large language models (LLMs) for both table representation and table retrieval. For table representation, Pneuma preserves schema and row-level information to ensure comprehensive data understanding. For table retrieval, Pneuma augments LLMs with traditional information retrieval techniques, such as full-text and vector search, harnessing the strengths of both to improve retrieval performance. To evaluate Pneuma, we generate comprehensive benchmarks that simulate table discovery workload on six real-world datasets including enterprise data, scientific databases, warehousing data, and open data. Our results demonstrate that Pneuma outperforms widely used table search systems (such as full-text search and state-of-the-art RAG systems) in accuracy and resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00771v2">Evaluating Personalized Tool-Augmented LLMs from the Perspectives of Personalization and Proactivity</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Personalized tool utilization is essential for aligning large language models (LLMs) with user preference in interaction scenarios with various tools. However, most of the current benchmarks primarily focus on either personalization of text generation or direct tool-utilizing, without considering both. In this work, we introduce a novel benchmark ETAPP for evaluating personalized tool invocation, establishing a sandbox environment, and a comprehensive dataset of 800 testing cases covering diverse user profiles. To improve the accuracy of our evaluation, we propose a key-point-based LLM evaluation method, mitigating biases in the LLM-as-a-judge system by manually annotating key points for each test case and providing them to LLM as the reference. Additionally, we evaluate the excellent LLMs and provide an in-depth analysis. Furthermore, we investigate the impact of different tool-invoking strategies on LLMs' personalization performance and the effects of fine-tuning in our task. The effectiveness of our preference-setting and key-point-based evaluation method is also validated. Our findings offer insights into improving personalized LLM agents. Our Code is available at https://github.com/hypasd-art/ETAPP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17097v3">Re-Ex: Revising after Explanation Reduces the Factual Errors in LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 ICLR 2024 Workshop on Reliable and Responsible Foundation Models
    </div>
    <details class="paper-abstract">
      Mitigating hallucination issues is a key challenge that must be overcome to reliably deploy large language models (LLMs) in real-world scenarios. Recently, various methods have been proposed to detect and revise factual errors in LLM-generated texts, in order to reduce hallucination. In this paper, we propose Re-Ex, a method for post-editing LLM-generated responses. Re-Ex introduces a novel reasoning step dubbed as the factual error explanation step. Re-Ex revises the initial response of LLMs using 3-steps : first, external tools are used to retrieve the evidences of the factual errors in the initial LLM response; next, LLM is instructed to explain the problematic parts of the response based on the gathered evidence; finally, LLM revises the initial response using the explanations provided in the previous step. In addition to the explanation step, Re-Ex also incorporates new prompting techniques to reduce the token count and inference time required for the response revision process. Compared with existing methods including FacTool, CoVE, and RARR, Re-Ex provides better detection and revision performance with less inference time and fewer tokens in multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07431v2">LLM-Enabled Data Transmission in End-to-End Semantic Communication</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Emerging services such as augmented reality (AR) and virtual reality (VR) have increased the volume of data transmitted in wireless communication systems, revealing the limitations of traditional Shannon theory. To address these limitations, semantic communication has been proposed as a solution that prioritizes the meaning of messages over the exact transmission of bits. This paper explores semantic communication for text data transmission in end-to-end (E2E) systems through a novel approach called KG-LLM semantic communication, which integrates knowledge graph (KG) extraction and large language model (LLM) coding. In this method, the transmitter first utilizes a KG to extract key entities and relationships from sentences. The extracted information is then encoded using an LLM to obtain the semantic meaning. On the receiver side, messages are decoded using another LLM, while a bidirectional encoder representations from transformers (i.e., BERT) model further refines the reconstructed sentences for improved semantic similarity. The KG-LLM semantic communication method reduces the transmitted text data volume by 30% through KG-based compression and achieves 84\% semantic similarity between the original and received messages. This demonstrates the KG-LLM methods efficiency and robustness in semantic communication systems, outperforming the deep learning-based semantic communication model (DeepSC), which achieves only 63%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12653v2">Predicting User Behavior in Smart Spaces with LLM-Enhanced Logs and Personalized Prompts</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      Enhancing the intelligence of smart systems, such as smart home, and smart vehicle, and smart grids, critically depends on developing sophisticated planning capabilities that can anticipate the next desired function based on historical interactions. While existing methods view user behaviors as sequential data and apply models like RNNs and Transformers to predict future actions, they often fail to incorporate domain knowledge and capture personalized user preferences. In this paper, we propose a novel approach that incorporates LLM-enhanced logs and personalized prompts. Our approach first constructs a graph that captures individual behavior preferences derived from their interaction histories. This graph effectively transforms into a soft continuous prompt that precedes the sequence of user behaviors. Then our approach leverages the vast general knowledge and robust reasoning capabilities of a pretrained LLM to enrich the oversimplified and incomplete log records. By enhancing these logs semantically, our approach better understands the user's actions and intentions, especially for those rare events in the dataset. We evaluate the method across four real-world datasets from both smart vehicle and smart home settings. The findings validate the effectiveness of our LLM-enhanced description and personalized prompt, shedding light on potential ways to advance the intelligence of smart space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15335v2">Stepwise Informativeness Search for Efficient and Effective LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Advances in Large Language Models (LLMs) have significantly improved multi-step reasoning through generating free-text rationales. However, recent studies show that LLMs tend to lose focus over the middle of long contexts. This raises concerns that as reasoning progresses, LLMs may overlook information in earlier steps when decoding subsequent steps, leading to generate unreliable and redundant rationales. To address this, we propose guiding LLMs to generate more accurate and concise step-by-step rationales by (1) proactively referencing information from underutilized prior steps, and (2) minimizing redundant information between new and existing steps. We introduce stepwise informativeness search, an inference-time tree search framework incorporating two selection heuristics: grounding-guided selection which prioritizes steps paying higher attention over underutilized steps; and novelty-guided selection which encourages steps with novel conclusions. During rationale generation, we use a self-grounding strategy that prompts LLMs to explicitly reference relevant prior steps to provide premises before deduction at each step. Experimental results on four reasoning datasets demonstrate that our approach improves reasoning accuracy by generating higher-quality rationales with reduced errors and redundancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09049v1">From Punchlines to Predictions: A Metric to Assess LLM Performance in Identifying Humor in Stand-Up Comedy</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 Accepted to CMCL2025 @ NAACL
    </div>
    <details class="paper-abstract">
      Comedy serves as a profound reflection of the times we live in and is a staple element of human interactions. In light of the widespread adoption of Large Language Models (LLMs), the intersection of humor and AI has become no laughing matter. Advancements in the naturalness of human-computer interaction correlates with improvements in AI systems' abilities to understand humor. In this study, we assess the ability of models in accurately identifying humorous quotes from a stand-up comedy transcript. Stand-up comedy's unique comedic narratives make it an ideal dataset to improve the overall naturalness of comedic understanding. We propose a novel humor detection metric designed to evaluate LLMs amongst various prompts on their capability to extract humorous punchlines. The metric has a modular structure that offers three different scoring methods - fuzzy string matching, sentence embedding, and subspace similarity - to provide an overarching assessment of a model's performance. The model's results are compared against those of human evaluators on the same task. Our metric reveals that regardless of prompt engineering, leading models, ChatGPT, Claude, and DeepSeek, achieve scores of at most 51% in humor detection. Notably, this performance surpasses that of humans who achieve a score of 41%. The analysis of human evaluators and LLMs reveals variability in agreement, highlighting the subjectivity inherent in humor and the complexities involved in extracting humorous quotes from live performance transcripts. Code available at https://github.com/swaggirl9000/humor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09037v1">A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-12
      | 💬 72 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17662v4">PLAYER*: Enhancing LLM-based Multi-Agent Communication and Interaction in Murder Mystery Games</a></div>
    <div class="paper-meta">
      📅 2025-04-12
    </div>
    <details class="paper-abstract">
      We present PLAYER*, a novel framework for Large Language Model (LLM)-based agents in Murder Mystery Games (MMGs). MMGs pose unique challenges, including undefined state spaces, absent intermediate rewards, and the need for strategic interaction in a continuous language domain. PLAYER* addresses these complexities through a sensor-based representation of agent states, a question-targeting mechanism guided by information gain, and a pruning strategy to refine suspect lists and enhance decision-making efficiency. To enable systematic evaluation, we propose WellPlay, a dataset comprising 1,482 inferential questions across 12 games, categorized into objectives, reasoning, and relationships. Experiments demonstrate PLAYER*'s capacity to achieve superior performance in reasoning accuracy and efficiency compared to existing approaches, while also significantly improving the quality of agent-human interactions in MMGs. This study advances the development of reasoning agents for complex social and interactive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08525v1">Task Memory Engine (TME): Enhancing State Awareness for Multi-Step LLM Agent Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 14 pages, 5 figures. Preprint prepared for future submission. Includes implementation and token-efficiency analysis. Code at https://github.com/biubiutomato/TME-Agent
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. The full implementation of TME is available at https://github.com/biubiutomato/TME-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07199v2">SemEval-2025 Task 5: LLMs4Subjects -- LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 10 pages, 4 figures, Accepted as SemEval 2025 Task 5 description paper
    </div>
    <details class="paper-abstract">
      We present SemEval-2025 Task 5: LLMs4Subjects, a shared task on automated subject tagging for scientific and technical records in English and German using the GND taxonomy. Participants developed LLM-based systems to recommend top-k subjects, evaluated through quantitative metrics (precision, recall, F1-score) and qualitative assessments by subject specialists. Results highlight the effectiveness of LLM ensembles, synthetic data generation, and multilingual processing, offering insights into applying LLMs for digital library classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07557v2">Using LLMs for Analyzing AIS Data</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Recent research in Large Language Models (LLMs), has had a profound impact across various fields, including mobility data science. This paper explores the and experiment with different approaches to using LLMs for analyzing AIS data. We propose a set of carefully designed queries to assess the reasoning capabilities of LLMs in this kind of tasks. Further, we experiment with four different methods: (1) using LLMs as a natural language interface to a spatial database, (2) reasoning on raw data, (3) reasoning on compressed trajectories, and (4) reasoning on semantic trajectories. We investigate the strengths and weaknesses for the four methods, and discuss the findings. The goal is to provide valuable insights for both researchers and practitioners on selecting the most appropriate LLM-based method depending on their specific data analysis objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08378v1">Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed on mobile devices, but the limited DRAM capacity constrains the deployable model size. This paper introduces ActiveFlow, the first LLM inference framework that can achieve adaptive DRAM usage for modern LLMs (not ReLU-based), enabling the scaling up of deployable model sizes. The framework is based on the novel concept of active weight DRAM-flash swapping and incorporates three novel techniques: (1) Cross-layer active weights preloading. It uses the activations from the current layer to predict the active weights of several subsequent layers, enabling computation and data loading to overlap, as well as facilitating large I/O transfers. (2) Sparsity-aware self-distillation. It adjusts the active weights to align with the dense-model output distribution, compensating for approximations introduced by contextual sparsity. (3) Active weight DRAM-flash swapping pipeline. It orchestrates the DRAM space allocation among the hot weight cache, preloaded active weights, and computation-involved weights based on available memory. Results show ActiveFlow achieves the performance-cost Pareto frontier compared to existing efficiency optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17365v2">How Effective Is Constitutional AI in Small LLMs? A Study on DeepSeek-R1 and Its Peers</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Recent incidents highlight safety risks in Large Language Models (LLMs), motivating research into alignment methods like Constitutional AI (CAI). This paper explores CAI's self-critique mechanism on small, uncensored 7-9B parameter models: DeepSeek-R1-8B, Gemma-2-9B, Llama 3.1-8B, and Qwen2.5-7B. We show that while Llama-based models exhibited significant harm reduction through self-critique, other architectures demonstrated less improvement in harm detection after abliteration. These results suggest CAI's effectiveness may vary depending on model architecture and reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07583v2">Do LLMs Understand Your Translations? Evaluating Paragraph-level MT with Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Despite the steady progress in machine translation evaluation, existing automatic metrics struggle to capture how well meaning is preserved beyond sentence boundaries. We posit that reliance on a single intrinsic quality score, trained to mimic human judgments, might be insufficient for evaluating translations of long, complex passages, and a more ``pragmatic'' approach that assesses how accurately key information is conveyed by a translation in context is needed. We introduce TREQA (Translation Evaluation via Question-Answering), a framework that extrinsically evaluates translation quality by assessing how accurately candidate translations answer reading comprehension questions that target key information in the original source or reference texts. In challenging domains that require long-range understanding, such as literary texts, we show that TREQA is competitive with and, in some cases, outperforms state-of-the-art neural and LLM-based metrics in ranking alternative paragraph-level translations, despite never being explicitly optimized to correlate with human judgments. Furthermore, the generated questions and answers offer interpretability: empirical analysis shows that they effectively target translation errors identified by experts in evaluated datasets. Our code is available at https://github.com/deep-spin/treqa
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02623v2">Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08312v1">SortBench: Benchmarking LLMs based on their ability to sort lists</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Sorting is a tedious but simple task for human intelligence and can be solved fairly easily algorithmically. However, for Large Language Models (LLMs) this task is surprisingly hard, as some properties of sorting are among known weaknesses of LLMs: being faithful to the input data, logical comparisons between values, and strictly differentiating between syntax (used for sorting) and semantics (typically learned by embeddings). Within this paper, we describe the new SortBench benchmark for LLMs that comes with different difficulties and that can be easily scaled in terms of difficulty. We apply this benchmark to seven state-of-the-art LLMs, including current test-time reasoning models. Our results show that while the o3-mini model is very capable at sorting in general, even this can be fooled if strings are defined to mix syntactical and semantical aspects, e.g., by asking to sort numbers written-out as word. Furthermore, all models have problems with the faithfulness to the input of long lists, i.e., they drop items and add new ones. Our results also show that test-time reasoning has a tendency to overthink problems which leads to performance degradation. Finally, models without test-time reasoning like GPT-4o are not much worse than reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06943v2">Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08260v1">Evaluating the Bias in LLMs for Surveying Opinion and Decision Making in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Generative agents have been increasingly used to simulate human behaviour in silico, driven by large language models (LLMs). These simulacra serve as sandboxes for studying human behaviour without compromising privacy or safety. However, it remains unclear whether such agents can truly represent real individuals. This work compares survey data from the Understanding America Study (UAS) on healthcare decision-making with simulated responses from generative agents. Using demographic-based prompt engineering, we create digital twins of survey respondents and analyse how well different LLMs reproduce real-world behaviours. Our findings show that some LLMs fail to reflect realistic decision-making, such as predicting universal vaccine acceptance. However, Llama 3 captures variations across race and Income more accurately but also introduces biases not present in the UAS data. This study highlights the potential of generative agents for behavioural research while underscoring the risks of bias from both LLMs and prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08242v1">Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted by IEEE International Conference on Computer Communications 2025
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have garnered significant attention due to their exceptional capabilities in various AI tasks. Traditionally deployed in cloud datacenters, LLMs are now increasingly moving towards more accessible edge platforms to protect sensitive user data and ensure privacy preservation. The limited computational resources of individual edge devices, however, can result in excessively prolonged inference latency and overwhelmed memory usage. While existing research has explored collaborative edge computing to break the resource wall of individual devices, these solutions yet suffer from massive communication overhead and under-utilization of edge resources. Furthermore, they focus exclusively on optimizing the prefill phase, neglecting the crucial autoregressive decoding phase for generative LLMs. To address that, we propose Jupiter, a fast, scalable, and resource-efficient collaborative edge AI system for generative LLM inference. Jupiter introduces a flexible pipelined architecture as a principle and differentiates its system design according to the differentiated characteristics of the prefill and decoding phases. For prefill phase, Jupiter submits a novel intra-sequence pipeline parallelism and develops a meticulous parallelism planning strategy to maximize resource efficiency; For decoding, Jupiter devises an effective outline-based pipeline parallel decoding mechanism combined with speculative decoding, which further magnifies inference acceleration. Extensive evaluation based on realistic implementation demonstrates that Jupiter remarkably outperforms state-of-the-art approaches under various edge environment setups, achieving up to 26.1x end-to-end latency reduction while rendering on-par generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08211v1">LLM for Comparative Narrative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 5 pages, 4 figures, Appendix included
    </div>
    <details class="paper-abstract">
      In this paper, we conducted a Multi-Perspective Comparative Narrative Analysis (CNA) on three prominent LLMs: GPT-3.5, PaLM2, and Llama2. We applied identical prompts and evaluated their outputs on specific tasks, ensuring an equitable and unbiased comparison between various LLMs. Our study revealed that the three LLMs generated divergent responses to the same prompt, indicating notable discrepancies in their ability to comprehend and analyze the given task. Human evaluation was used as the gold standard, evaluating four perspectives to analyze differences in LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08207v1">DRAFT-ing Architectural Design Decisions using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Architectural Knowledge Management (AKM) is crucial for software development but remains challenging due to the lack of standardization and high manual effort. Architecture Decision Records (ADRs) provide a structured approach to capture Architecture Design Decisions (ADDs), but their adoption is limited due to the manual effort involved and insufficient tool support. Our previous work has shown that Large Language Models (LLMs) can assist in generating ADDs. However, simply prompting the LLM does not produce quality ADDs. Moreover, using third-party LLMs raises privacy concerns, while self-hosting them poses resource challenges. To this end, we experimented with different approaches like few-shot, retrieval-augmented generation (RAG) and fine-tuning to enhance LLM's ability to generate ADDs. Our results show that both techniques improve effectiveness. Building on this, we propose Domain Specific Retreival Augumented Few Shot Fine Tuninng, DRAFT, which combines the strengths of all these three approaches for more effective ADD generation. DRAFT operates in two phases: an offline phase that fine-tunes an LLM on generating ADDs augmented with retrieved examples and an online phase that generates ADDs by leveraging retrieved ADRs and the fine-tuned model. We evaluated DRAFT against existing approaches on a dataset of 4,911 ADRs and various LLMs and analyzed them using automated metrics and human evaluations. Results show DRAFT outperforms all other approaches in effectiveness while maintaining efficiency. Our findings indicate that DRAFT can aid architects in drafting ADDs while addressing privacy and resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07596v2">Boosting Universal LLM Reward Design through Heuristic Reward Observation Space Evolution</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are emerging as promising tools for automated reinforcement learning (RL) reward design, owing to their robust capabilities in commonsense reasoning and code generation. By engaging in dialogues with RL agents, LLMs construct a Reward Observation Space (ROS) by selecting relevant environment states and defining their internal operations. However, existing frameworks have not effectively leveraged historical exploration data or manual task descriptions to iteratively evolve this space. In this paper, we propose a novel heuristic framework that enhances LLM-driven reward design by evolving the ROS through a table-based exploration caching mechanism and a text-code reconciliation strategy. Our framework introduces a state execution table, which tracks the historical usage and success rates of environment states, overcoming the Markovian constraint typically found in LLM dialogues and facilitating more effective exploration. Furthermore, we reconcile user-provided task descriptions with expert-defined success criteria using structured prompts, ensuring alignment in reward design objectives. Comprehensive evaluations on benchmark RL tasks demonstrate the effectiveness and stability of the proposed framework. Code and video demos are available at jingjjjjjie.github.io/LLM2Reward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08192v1">SAEs $\textit{Can}$ Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Machine unlearning is a promising approach to improve LLM safety by removing unwanted knowledge from the model. However, prevailing gradient-based unlearning methods suffer from issues such as high computational costs, hyperparameter instability, poor sequential unlearning capability, vulnerability to relearning attacks, low data efficiency, and lack of interpretability. While Sparse Autoencoders are well-suited to improve these aspects by enabling targeted activation-based unlearning, prior approaches underperform gradient-based methods. This work demonstrates that, contrary to these earlier findings, SAEs can significantly improve unlearning when employed dynamically. We introduce $\textbf{Dynamic DAE Guardrails}$ (DSG), a novel method for precision unlearning that leverages principled feature selection and a dynamic classifier. Our experiments show DSG substantially outperforms leading unlearning methods, achieving superior forget-utility trade-offs. DSG addresses key drawbacks of gradient-based approaches for unlearning -- offering enhanced computational efficiency and stability, robust performance in sequential unlearning, stronger resistance to relearning attacks, better data efficiency including zero-shot settings, and more interpretable unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16557v2">Patched RTC: evaluating LLMs for diverse software development tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      This paper introduces Patched Round-Trip Correctness (Patched RTC), a novel evaluation technique for Large Language Models (LLMs) applied to diverse software development tasks, particularly focusing on "outer loop" activities such as bug fixing, code review, and documentation updates. Patched RTC extends the original Round-Trip Correctness method to work with any LLM and downstream task, offering a self-evaluating framework that measures consistency and robustness of model responses without human intervention. The study demonstrates a correlation between Patched RTC scores and task-specific accuracy metrics, presenting it as an alternative to the LLM-as-Judge paradigm for open-domain task evaluation. We implement Patched RTC in an open-source framework called patchwork, allowing for transparent evaluation during inference across various patchflows. Experiments comparing GPT-3.5 and GPT-4 models across different software development tasks reveal that Patched RTC effectively distinguishes model performance and task difficulty. The paper also explores the impact of consistency prompts on improving model accuracy, suggesting that Patched RTC can guide prompt refinement and model selection for complex software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08999v1">MCP Bridge: A Lightweight, LLM-Agnostic RESTful Proxy for Model Context Protocol Servers</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly augmented with external tools through standardized interfaces like the Model Context Protocol (MCP). However, current MCP implementations face critical limitations: they typically require local process execution through STDIO transports, making them impractical for resource-constrained environments like mobile devices, web browsers, and edge computing. We present MCP Bridge, a lightweight RESTful proxy that connects to multiple MCP servers and exposes their capabilities through a unified API. Unlike existing solutions, MCP Bridge is fully LLM-agnostic, supporting any backend regardless of vendor. The system implements a risk-based execution model with three security levels standard execution, confirmation workflow, and Docker isolation while maintaining backward compatibility with standard MCP clients. Complementing this server-side infrastructure is a Python based MCP Gemini Agent that facilitates natural language interaction with MCP tools. The evaluation demonstrates that MCP Bridge successfully addresses the constraints of direct MCP connections while providing enhanced security controls and cross-platform compatibility, enabling sophisticated LLM-powered applications in previously inaccessible environments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05522v2">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-11
    </div>
    <details class="paper-abstract">
      Exploration, the act of broadening user experiences beyond their established preferences, is challenging in large-scale recommendation systems due to feedback loops and limited signals on user exploration patterns. Large Language Models (LLMs) offer potential by leveraging their world knowledge to recommend novel content outside these loops. A key challenge is aligning LLMs with user preferences while preserving their knowledge and reasoning. While using LLMs to plan for the next novel user interest, this paper introduces a novel approach combining hierarchical planning with LLM inference-time scaling to improve recommendation relevancy without compromising novelty. We decouple novelty and user-alignment, training separate LLMs for each objective. We then scale up the novelty-focused LLM's inference and select the best-of-n predictions using the user-aligned LLM. Live experiments demonstrate efficacy, showing significant gains in both user satisfaction (measured by watch activity and active user counts) and exploration diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08985v1">Learning from Elders: Making an LLM-powered Chatbot for Retirement Communities more Accessible through User-centered Design</a></div>
    <div class="paper-meta">
      📅 2025-04-11
      | 💬 Accepted as Research talk for Considering Cultural and Linguistic Diversity in AI Applications workshop at CALD-AI@ASIS&T 2025
    </div>
    <details class="paper-abstract">
      Low technology and eHealth literacy among older adults in retirement communities hinder engagement with digital tools. To address this, we designed an LLM-powered chatbot prototype using a human-centered approach for a local retirement community. Through interviews and persona development, we prioritized accessibility and dual functionality: simplifying internal information retrieval and improving technology and eHealth literacy. A pilot trial with residents demonstrated high satisfaction and ease of use, but also identified areas for further improvement. Based on the feedback, we refined the chatbot using GPT-3.5 Turbo and Streamlit. The chatbot employs tailored prompt engineering to deliver concise responses. Accessible features like adjustable font size, interface theme and personalized follow-up responses were implemented. Future steps include enabling voice-to-text function and longitudinal intervention studies. Together, our results highlight the potential of LLM-driven chatbots to empower older adults through accessible, personalized interactions, bridging literacy gaps in retirement communities.
    </details>
</div>
