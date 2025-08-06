# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03686v1">CompassVerifier: A Unified and Robust Verifier for LLMs Evaluation and Outcome Reward</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Technical Report; 31 Pages
    </div>
    <details class="paper-abstract">
      Answer verification is crucial not only for evaluating large language models (LLMs) by matching their unstructured outputs against standard answers, but also serves as the reward model to guide LLM optimization. Most evaluation frameworks rely on regularized matching or employ general LLMs for answer verification, which demands extensive, repetitive customization for regex rules or evaluation prompts. Two fundamental limitations persist in current methodologies: 1) the absence of comprehensive benchmarks that systematically evaluate verification capabilities across different LLMs; and 2) the nascent stage of verifier development, where existing approaches lack both the robustness to handle complex edge cases and the generalizability across different domains. In this work, we develop CompassVerifier, an accurate and robust lightweight verifier model for evaluation and outcome reward. It demonstrates multi-domain competency spanning math, knowledge, and diverse reasoning tasks, with the capability to process various answer types, including multi-subproblems, formulas, and sequence answers, while effectively identifying abnormal/invalid responses. We introduce VerifierBench benchmark comprising model outputs collected from multiple data sources, augmented through manual analysis of metaerror patterns to enhance CompassVerifier. We anticipate that CompassVerifier and VerifierBench will facilitate answer verification, evaluation protocols, and reinforcement learning research. Code and dataset are available at https://github.com/open-compass/CompassVerifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03685v1">No LLM Solved Yu Tsumura's 554th Problem</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 67 pages
    </div>
    <details class="paper-abstract">
      We show, contrary to the optimism about LLM's problem-solving abilities, fueled by the recent gold medals that were attained, that a problem exists -- Yu Tsumura's 554th problem -- that a) is within the scope of an IMO problem in terms of proof sophistication, b) is not a combinatorics problem which has caused issues for LLMs, c) requires fewer proof techniques than typical hard IMO problems, d) has a publicly available solution (likely in the training data of LLMs), and e) that cannot be readily solved by any existing off-the-shelf LLM (commercial or open-source).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03678v1">More Than a Score: Probing the Impact of Prompt Specificity on LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      State-of-the-art Large Language Models (LLMs) achieve high pass@1 on general benchmarks like HumanEval but underperform on specialized suites such as ParEval. Is this due to LLMs missing domain knowledge or insufficient prompt detail is given? To answer this, we introduce PartialOrderEval, which augments any code generation benchmark with a partial order of prompts from minimal to maximally detailed. Applying it to HumanEval and both serial and OpenMP subsets of ParEval, we measure how pass@1 scales with prompt specificity. Our experiments with Llama-3.x and Qwen2.5-Coder demonstrate varying degrees of prompt sensitivity across different tasks, and a qualitative analysis highlights explicit I/O specifications, edge-case handling, and stepwise breakdowns as the key drivers of prompt detail improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v2">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address for distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03628v1">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations at eBay</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Sellers at eBay are recommended keyphrases to bid on to enhance the performance of their advertising campaigns. The relevance of these keyphrases is crucial in avoiding the overcrowding of search systems with irrelevant items and maintaining a positive seller perception. It is essential that keyphrase recommendations align with both seller and Search judgments regarding auctions. Due to the difficulty in procuring negative human judgment at scale, employing LLM-as-a-judge to mimic seller judgment has been established as the norm in several studies. This study introduces a novel two-step LLM distillation process from a LLM-judge used to debias our Embedding Based Retrieval (EBR) model from the various biases that exist in click-data. We distill from an LLM teacher via a cross-encoder assistant into a bi-encoder student using a multi-task training approach, ultimately employing the student bi-encoder to retrieve relevant advertiser keyphrases. We show that integrating a knowledge distillation process from LLMs in a multi-task training setup enhances bi-encoder performance in retrieving relevant advertiser keyphrases at eBay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v4">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03622v1">Refining Critical Thinking in LLM Code Generation: A Faulty Premise-based Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      With the advancement of code generation capabilities in large language models (LLMs), their reliance on input premises has intensified. When users provide inputs containing faulty premises, the probability of code generation hallucinations rises significantly, exposing deficiencies in their self-scrutiny capabilities. This paper proposes Faulty Premises Bench (FPBench), the first code generation evaluation framework targeting faulty premises. By systematically constructing three categories of faulty premises and integrating multi-dimensional evaluation metrics, it conducts in-depth assessments of 15 representative LLMs. The key findings are as follows: (1) Most models exhibit poor reasoning abilities and suboptimal code generation performance under faulty premises, heavily relying on explicit prompts for error detection, with limited self-scrutiny capabilities; (2) Faulty premises trigger a point of diminishing returns in resource investment, leading to blindly increasing length fails to enhance quality; (3) The three types of faulty premises respectively activate distinct defect patterns in models, revealing a triple dissociation in the cognitive mechanisms of code generation models. This study not only highlights the urgent need for LLMs to proactively verify premises in code generation but also, through the proposed FPBench framework and multi-dimensional evaluation system, provides a theoretical foundation and practical pathway for developing reliable, human-centric code generation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03611v1">Block: Balancing Load in LLM Serving with Context, Knowledge and Predictive Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 12 pages, 8 figures excluding appendix
    </div>
    <details class="paper-abstract">
      This paper presents Block, a distributed scheduling framework designed to optimize load balancing and auto-provisioning across instances in large language model serving frameworks by leveraging contextual information from incoming requests. Unlike popular model serving systems that rely on monolithic and heuristic task schedulers, Block operates as a fully distributed, stateless, and predictive scheduling system to achieve low overhead, reliability, and scalability. It leverages the deterministic and predictable characteristics of LLM inferences, such as host configurations, response lengths, and hardware performance, to make scheduling decisions based on accurately predicted metrics. Evaluation on a 12 GPUs cluster shows that Block significantly outperforms heuristic schedulers, boosting serving capacity by up to 16.7\% and reducing P99 tail latency by up to 49.5\%. These performance gains remain consistent across diverse models, workloads and configurations. Code and data are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10323v4">ELFuzz: Efficient Input Generation via LLM-driven Synthesis Over Fuzzer Space</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted by USENIX Security'25 Cycle 2
    </div>
    <details class="paper-abstract">
      Generation-based fuzzing produces appropriate testing cases according to specifications of input grammars and semantic constraints to test systems and software. However, these specifications require significant manual efforts to construct. This paper proposes a new approach, ELFuzz (Evolution Through Large Language Models for Fuzzing), that automatically synthesizes generation-based fuzzers tailored to a system under test (SUT) via LLM-driven synthesis over fuzzer space. At a high level, it starts with minimal seed fuzzers and propels the synthesis by fully automated LLM-driven evolution with coverage guidance. Compared to previous approaches, ELFuzz can 1) seamlessly scale to SUTs of real-world sizes -- up to 1,791,104 lines of code in our evaluation -- and 2) synthesize efficient fuzzers that catch interesting grammatical structures and semantic constraints in a human-understandable way. Our evaluation compared ELFuzz with specifications manually written by domain experts and synthesized by state-of-the-art approaches. It shows that ELFuzz achieves up to 434.8% more coverage and triggers up to 174.0% more artificially injected bugs. We also used ELFuzz to conduct a real-world fuzzing campaign on the newest version of cvc5 for 14 days, and encouragingly, it found five 0-day bugs (three are exploitable). Moreover, we conducted an ablation study, which shows that the fuzzer space model, the key component of ELFuzz, contributes the most (up to 62.5%) to the effectiveness of ELFuzz. Further analysis of the fuzzers synthesized by ELFuzz confirms that they catch interesting grammatical structures and semantic constraints in a human-understandable way. The results present the promising potential of ELFuzz for more automated, efficient, and extensible input generation for fuzzing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03603v1">ReFuzzer: Feedback-Driven Approach to Enhance Validity of LLM-Generated Test Programs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Existing LLM-based compiler fuzzers often produce syntactically or semantically invalid test programs, limiting their effectiveness in exercising compiler optimizations and backend components. We introduce ReFuzzer, a framework for refining LLM-generated test programs by systematically detecting and correcting compilation and runtime violations (e.g. division by zero or array out-of-bounds accesses). ReFuzzer employs a feedback loop with a local LLM to validate and filter erroneous programs before execution, improving fuzzing effectiveness beyond crash detection and enabling the generation of diverse yet valid test programs. We evaluated ReFuzzer's effectiveness across black-, grey- and white-box fuzzing approaches targeting LLVM/Clang. ReFuzzer improved test programs' validity from 47.0-49.4% to 96.6-97.3%, with an average processing time of 2.9-3.5 s per test program on a dual-GPU machine. Further, refuzzing significantly increased code coverage in critical optimization and IR generation components. For example, vectorization coverage had an absolute improvement of 9.2%, 2.3%, and 7.1% in black-, grey-, and white-box fuzzing, enhancing testing effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06219v2">Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 COLM 2025 Camera Ready version
    </div>
    <details class="paper-abstract">
      The increasing adoption of web crawling opt-outs by copyright holders of online content raises critical questions about the impact of data compliance on large language model (LLM) performance. However, little is known about how these restrictions (and the resultant filtering of pretraining datasets) affect the capabilities of models trained using these corpora. In this work, we conceptualize this effect as the $\textit{data compliance gap}$ (DCG), which quantifies the performance difference between models trained on datasets that comply with web crawling opt-outs, and those that do not. We measure the data compliance gap in two settings: pretraining models from scratch and continual pretraining from existing compliant models (simulating a setting where copyrighted data could be integrated later in pretraining). Our experiments with 1.5B models show that, as of January 2025, compliance with web data opt-outs does not degrade general knowledge acquisition (close to 0\% DCG). However, in specialized domains such as biomedical research, excluding major publishers leads to performance declines. These findings suggest that while general-purpose LLMs can be trained to perform equally well using fully open data, performance in specialized domains may benefit from access to high-quality copyrighted sources later in training. Our study provides empirical insights into the long-debated trade-off between data compliance and downstream model performance, informing future discussions on AI training practices and policy decisions. Our website is available at https://data-compliance.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03571v1">Tackling Distribution Shift in LLM via KILO: Knowledge-Instructed Learning for Continual Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often suffer from performance degradation when faced with domain shifts, primarily due to catastrophic forgetting. In this work, we propose KILO (Knowledge-Instructed Learning for Continual Adaptation), a novel continual learning framework that integrates dynamic knowledge graphs with instruction tuning. By leveraging retrieved domain-specific knowledge as guidance during training, KILO enhances both adaptability to new domains and retention of previously acquired knowledge. We pretrain our model on WikiText-103 and evaluate sequential adaptation across four diverse target domains: BioASQ, SciQ, TweetEval, and MIND. Our experiments demonstrate that KILO consistently outperforms strong baselines, including continual fine-tuning, ERNIE 2.0, and CPT, in terms of backward transfer, forward transfer, F1 score, retention rate, and training efficiency. These results highlight the effectiveness of combining structured knowledge retrieval and instruction prompting to overcome domain shift challenges in continual learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03558v1">SAGE-HLS: Syntax-Aware AST-Guided LLM for High-Level Synthesis Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted to the IEEE International Conference on Computer Design (ICCD 2025)
    </div>
    <details class="paper-abstract">
      In today's rapidly evolving field of electronic design automation (EDA), the complexity of hardware designs is increasing, necessitating more sophisticated automation solutions. High-level synthesis (HLS), as a pivotal solution, automates hardware designs from high-level abstractions (e.g., C/C++). However, it faces significant challenges, particularly in design space exploration and optimization. While large language models (LLMs) have shown notable capabilities in code generation, their application to HLS has been limited due to the scarcity of (publicly) available HLS code datasets. Hence, research in this domain has primarily focused on techniques such as prompt engineering and retrieval-augmented generation (RAG). To overcome this limitation, this paper introduces SAGE-HLS, the first-of-its-kind fine-tuned LLM specifically for HLS code generation. Our method includes three key advancements: (i) We implement Verilog-to-C/C++ porting, converting verified and synthesizable Verilog codes into corresponding C, creating a dataset of 16.7K HLS codes; (ii) We implement a fine-tuning strategy, which is based on instruction prompting to code generation guided by abstract syntax tree (AST); (iii) We develop a semi-automated evaluation framework using VerilogEval to assess the functionality of the generated HLS code. Our experiments show that SAGE-HLS, fined-tuned on the QwenCoder (2.5) 7B model, achieves a near 100% success rate in code synthesizability and a 75% success rate in functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03550v1">Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      The growing scale of evaluation tasks has led to the widespread adoption of automated evaluation using large language models, a paradigm known as "LLMas-a-judge." However, improving its alignment with human preferences without complex prompts or fine-tuning remains challenging. In this work, motivated by preliminary findings that middle-to-upper layers encode semantically and taskrelevant representations that are often more aligned with human judgments than the final layer, we propose LAGER, a lightweight and efficient framework for enhancing LLM-as-a-Judge alignment with human scoring, via internal representations. LAGER produces fine-grained judgment scores by aggregating cross-layer scoretoken logits and computing the expected score from a softmax-based distribution, with the LLM backbone kept frozen. LAGER fully leverages the complementary information across different layers, overcoming the limitations of relying solely on the final layer. We evaluate our method on the standard alignment benchmarks Flask, HelpSteer, and BIGGen using Spearman correlation, and find that LAGER achieves improvements of up to 7.5% over the best baseline across these benchmarks. Without reasoning steps, LAGER matches or outperforms reasoning-based methods. Experiments on downstream applications, such as data selection and emotional understanding, further show the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03547v1">Guided Reality: Generating Visually-Enriched AR Task Guidance with LLMs and Vision Models</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 To appear at UIST 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the automatic generation of step-by-step augmented reality (AR) instructions for a wide range of physical tasks. However, existing LLM-based AR guidance often lacks rich visual augmentations to effectively embed instructions into spatial context for a better user understanding. We present Guided Reality, a fully automated AR system that generates embedded and dynamic visual guidance based on step-by-step instructions. Our system integrates LLMs and vision models to: 1) generate multi-step instructions from user queries, 2) identify appropriate types of visual guidance, 3) extract spatial information about key interaction points in the real world, and 4) embed visual guidance in physical space to support task execution. Drawing from a corpus of user manuals, we define five categories of visual guidance and propose an identification strategy based on the current step. We evaluate the system through a user study (N=16), completing real-world tasks and exploring the system in the wild. Additionally, four instructors shared insights on how Guided Reality could be integrated into their training workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03523v1">FilBench: Can LLMs Understand and Generate Filipino?</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of LLMs on English-based tasks, little is known about their capabilities in specific languages such as Filipino. In this work, we address this gap by introducing FilBench, a Filipino-centric benchmark designed to evaluate LLMs across a diverse set of tasks and capabilities in Filipino, Tagalog, and Cebuano. We carefully curate the tasks in FilBench to reflect the priorities and trends of NLP research in the Philippines such as Cultural Knowledge, Classical NLP, Reading Comprehension, and Generation. By evaluating 27 state-of-the-art LLMs on FilBench, we find that several LLMs suffer from reading comprehension and translation capabilities. Our results indicate that FilBench is challenging, with the best model, GPT-4o, achieving only a score of 72.23%. Moreover, we also find that models trained specifically for Southeast Asian languages tend to underperform on FilBench, with the highest-performing model, SEA-LION v3 70B, achieving only a score of 61.07%. Our work demonstrates the value of curating language-specific LLM benchmarks to aid in driving progress on Filipino NLP and increasing the inclusion of Philippine languages in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17432v2">Breaking the Modality Barrier: Universal Embedding Learning with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 13 pages, 8 figures, Accepted by ACM MM2025, Project page: https://garygutc.github.io/UniME
    </div>
    <details class="paper-abstract">
      The Contrastive Language-Image Pre-training (CLIP) framework has become a widely used approach for multimodal representation learning, particularly in image-text retrieval and clustering. However, its efficacy is constrained by three key limitations: (1) text token truncation, (2) isolated image-text encoding, and (3) deficient compositionality due to bag-of-words behavior. While recent Multimodal Large Language Models (MLLMs) have demonstrated significant advances in generalized vision-language understanding, their potential for learning transferable multimodal representations remains underexplored.In this work, we present UniME (Universal Multimodal Embedding), a novel two-stage framework that leverages MLLMs to learn discriminative representations for diverse downstream tasks. In the first stage, we perform textual discriminative knowledge distillation from a powerful LLM-based teacher model to enhance the embedding capability of the MLLM\'s language component. In the second stage, we introduce hard negative enhanced instruction tuning to further advance discriminative representation learning. Specifically, we initially mitigate false negative contamination and then sample multiple hard negatives per instance within each batch, forcing the model to focus on challenging samples. This approach not only improves discriminative power but also enhances instruction-following ability in downstream tasks. We conduct extensive experiments on the MMEB benchmark and multiple retrieval tasks, including short and long caption retrieval and compositional retrieval. Results demonstrate that UniME achieves consistent performance improvement across all tasks, exhibiting superior discriminative and compositional capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03487v1">BitsAI-Fix: LLM-Driven Approach for Automated Lint Error Resolution in Practice</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      As enterprise codebases continue to grow in scale and complexity, the volume of lint errors far exceeds engineers' manual remediation capacity, leading to continuous accumulation of technical debt and hindered development efficiency. This paper presents BitsAI-Fix, an automated lint error remediation workflow based on Large Language Models (LLMs), designed to address this critical challenge in industrial-scale environments. BitsAI-Fix employs tree-sitter for context expansion and generates search-and-replace format patches through specially trained LLMs, followed by lint scan re-verification to output final remediation results. Additionally, our approach introduces an innovative progressive reinforcement learning (RL) training strategy that can automatically acquire verifiable training data during the project cold-start phase and continuously iterate the model by collecting online samples through feedback after system deployment. Furthermore, we designed a targeted rule-based reward mechanism that combines format rewards and correctness rewards while penalizing redundant modifications. We also propose a "code diff matching" methodology to continuously track online effectiveness. In production deployment at ByteDance, our solution has supported over 5,000 engineers, resolved more than 12,000 static analysis issues, achieved approximately 85% remediation accuracy, with around 1,000 weekly active adopters. This work demonstrates the practical feasibility of LLM-based code remediation solutions in enterprise environments and serves as a reference for automated code fix in large-scale industrial scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03464v1">Learning to Incentivize: LLM-Empowered Contract for AIGC Offloading in Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      With the rapid growth in demand for AI-generated content (AIGC), edge AIGC service providers (ASPs) have become indispensable. However, designing incentive mechanisms that motivate ASPs to deliver high-quality AIGC services remains a challenge, especially in the presence of information asymmetry. In this paper, we address bonus design between a teleoperator and an edge ASP when the teleoperator cannot observe the ASP's private settings and chosen actions (diffusion steps). We formulate this as an online learning contract design problem and decompose it into two subproblems: ASP's settings inference and contract derivation. To tackle the NP-hard setting-inference subproblem with unknown variable sizes, we introduce a large language model (LLM)-empowered framework that iteratively refines a naive seed solver using the LLM's domain expertise. Upon obtaining the solution from the LLM-evolved solver, we directly address the contract derivation problem using convex optimization techniques and obtain a near-optimal contract. Simulation results on our Unity-based teleoperation platform show that our method boosts the teleoperator's utility by $5 \sim 40\%$ compared to benchmarks, while preserving positive incentives for the ASP. The code is available at https://github.com/Zijun0819/llm4contract.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06787v4">Bridging LLMs and KGs without Fine-Tuning: Intermediate Probing Meets Subgraph-Aware Entity Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Traditional knowledge graph completion (KGC) methods rely solely on structural information, struggling with the inherent sparsity of knowledge graphs (KGs). By contrast, Large Language Models (LLMs) encapsulate extensive world knowledge and exhibit powerful context modeling capabilities, making them promising for mitigating the limitations of traditional methods. However, direct fine-tuning of LLMs for KGC, though effective, imposes substantial computational and memory overheads, while utilizing non-fine-tuned LLMs is efficient but yields suboptimal performance. In this work, we propose a novel framework that synergizes the strengths of LLMs with robust knowledge representation to enable effective and efficient KGC. We extract the context-aware hidden states of knowledge triples from the intermediate layers of LLMs, thereby capturing rich semantic and relational nuances. These representations are then utilized to train a data-efficient classifier tailored specifically for KGC tasks. To bridge the semantic gaps between LLMs and KGs, we employ subgraph sampling on KGs to generate model-friendly entity descriptions. We further adopt sliced mutual information (SMI) as a principled metric to quantify the task-specific information encoded in these representations. Extensive experiments on standard benchmarks validate the efficiency and effectiveness of our approach. We achieve a 47\% relative improvement over previous methods based on non-fine-tuned LLMs and, to our knowledge, are the first to achieve classification performance comparable to fine-tuned LLMs while enhancing GPU memory efficiency by $188\times$ and accelerating training and inference by $26.11\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v1">LLMs Have a Heart of Stone: Demystifying the Soft Thinking Ability of Large Reasoning Models</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 10 pages, 7 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06931v5">Non-Prehensile Tool-Object Manipulation by Integrating LLM-Based Planning and Manoeuvrability-Driven Controls</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Being able to use tools is a widely recognised indicator of intelligence across species. Humans, for instance, have demonstrated mastery of tool use for over two million years. The ability to use tools is invaluable as it extends an organism's reach and enhances its capacity to interact with objects and the environment. Being able to understand the geometric-mechanical relations between the tools-objects-environments allows certain species (e.g., apes and crows) to reach food in narrow constrained spaces. The same principles of physical augmentation and its associated non-prehensile manipulation capabilities also apply to robotic systems. For example, by instrumenting them with different types of end-effectors, robots can (in principle) dexterously interact (e.g., push and flip) with objects of various shapes and masses akin to its biological counterpart. However, developing this type of manipulation skill is still an open research problem. Furthermore, the complexity of planning tool-object manipulation tasks, particularly in coordinating the actions of dual-arm robots, presents significant challenges. To address these complexities, we propose integrating Large Language Models (LLMs) to assist in planning and executing these intricate manipulations, thereby enhancing the robot's ability to perform in diverse scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02823v1">NeuroSync: Intent-Aware Code-Based Problem Solving via Direct LLM Understanding Modification</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted in UIST 2025
    </div>
    <details class="paper-abstract">
      Conversational LLMs have been widely adopted by domain users with limited programming experience to solve domain problems. However, these users often face misalignment between their intent and generated code, resulting in frustration and rounds of clarification. This work first investigates the cause of this misalignment, which dues to bidirectional ambiguity: both user intents and coding tasks are inherently nonlinear, yet must be expressed and interpreted through linear prompts and code sequences. To address this, we propose direct intent-task matching, a new human-LLM interaction paradigm that externalizes and enables direct manipulation of the LLM understanding, i.e., the coding tasks and their relationships inferred by the LLM prior to code generation. As a proof-of-concept, this paradigm is then implemented in NeuroSync, which employs a knowledge distillation pipeline to extract LLM understanding, user intents, and their mappings, and enhances the alignment by allowing users to intuitively inspect and edit them via visualizations. We evaluate the algorithmic components of NeuroSync via technical experiments, and assess its overall usability and effectiveness via a user study (N=12). The results show that it enhances intent-task alignment, lowers cognitive effort, and improves coding efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03396v1">Hide and Seek with LLMs: An Adversarial Game for Sneaky Error Generation and Self-Improving Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in reasoning and generation across domains, but still struggle with identifying and diagnosing complex errors. This stems mainly from training objectives that prioritize correct answers, limiting exposure to and learning from errors. While recent studies have begun to address this by introducing error signals, most rely on shallow, static errors, restricting improvement in deep diagnostic ability. To overcome this, we propose Hide and Seek Game (HSG), a dynamic adversarial framework for error generation and diagnosis, and evaluate it on mathematical problem-solving. HSG involves two adversarial roles: Sneaky, which "hides" by generating subtle, deceptive reasoning errors, and Diagnosis, which "seeks" to accurately detect them. Through adversarial co-evolution, both error stealth and diagnostic precision are enhanced. Experiments on several math reasoning tasks show that HSG significantly boosts error diagnosis, achieving 16.8\%--31.4\% higher accuracy than baselines like GPT-4o. We also release a challenging dataset of deceptive errors and diagnostic annotations as a benchmark for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03346v1">Compressing Chain-of-Thought in LLMs via Step Entropy</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) using Chain-of-Thought (CoT) prompting excel at complex reasoning but generate verbose thought processes with considerable redundancy, leading to increased inference costs and reduced efficiency. We introduce a novel CoT compression framework based on step entropy, a metric that quantifies the informational contribution of individual reasoning steps to identify redundancy. Through theoretical analysis and extensive empirical validation on mathematical reasoning benchmarks, we demonstrate that steps with low entropy are indeed highly redundant. Our experiments reveal that an astonishing 80\% of low-entropy intermediate steps can be pruned with minor degradation in the final answer accuracy across DeepSeek-R1-7B, 14B and Qwen3-8B. This finding sharply contrasts with random or high-entropy pruning, which severely impairs reasoning performance. Building on this, we propose a novel two-stage training strategy combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) reinforcement learning. This approach enables LLMs to autonomously learn to generate compressed COTs during inference by strategically incorporating [SKIP] tokens. Our method significantly enhances LLM inference efficiency while rigorously preserving accuracy, offering profound implications for practical LLM deployment and a deeper understanding of reasoning structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02497v3">PennyLang: Pioneering LLM-Based Quantum Code Generation with a Novel PennyLane-Centric Dataset</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 8 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer powerful capabilities in code generation, natural language understanding, and domain-specific reasoning. Their application to quantum software development remains limited, in part because of the lack of high-quality datasets both for LLM training and as dependable knowledge sources. To bridge this gap, we introduce PennyLang, an off-the-shelf, high-quality dataset of 3,347 PennyLane-specific quantum code samples with contextual descriptions, curated from textbooks, official documentation, and open-source repositories. Our contributions are threefold: (1) the creation and open-source release of PennyLang, a purpose-built dataset for quantum programming with PennyLane; (2) a framework for automated quantum code dataset construction that systematizes curation, annotation, and formatting to maximize downstream LLM usability; and (3) a baseline evaluation of the dataset across multiple open-source models, including ablation studies, all conducted within a retrieval-augmented generation (RAG) pipeline. Using PennyLang with RAG substantially improves performance: for example, Qwen 7B's success rate rises from 8.7% without retrieval to 41.7% with full-context augmentation, and LLaMa 4 improves from 78.8% to 84.8%, while also reducing hallucinations and enhancing quantum code correctness. Moving beyond Qiskit-focused studies, we bring LLM-based tools and reproducible methods to PennyLane for advancing AI-assisted quantum development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01273v2">KCR: Resolving Long-Context Knowledge Conflicts via Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Knowledge conflicts commonly arise across diverse sources, and their prevalence has increased with the advent of LLMs. When dealing with conflicts between multiple contexts, also known as \emph{inter-context knowledge conflicts}, LLMs are often confused by lengthy and conflicting contexts. To address this challenge, we propose the Knowledge Conflict Reasoning (KCR) framework, which enhances the ability of LLMs to resolve conflicting knowledge. The key idea of KCR is to train backbone LLMs to establish a correct reasoning process by rewarding them for selecting and adhering to the context with stronger logical consistency when presented with conflicting contexts. Specifically, we first extract reasoning paths, represented by either text or local knowledge graphs, from the conflicting long contexts. Subsequently, we employ Reinforcement Learning to encourage the model to learn the paradigm of reasoning process that follows correct reasoning paths rather than the incorrect counterparts. This enables the backbone models to genuinely acquire the capability to resolve inter-context knowledge conflicts within long contexts. Experimental results demonstrate that our framework significantly improves the ability of various backbone models to resolve knowledge conflicts in long-context scenarios, yielding substantial performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18784v3">LLM-Generated Heuristics for AI Planning: Do We Even Need Domain-Independence Anymore?</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Domain-independent heuristics have long been a cornerstone of AI planning, offering general solutions applicable across a wide range of tasks without requiring domain-specific engineering. However, the advent of large language models (LLMs) presents an opportunity to generate heuristics tailored to specific planning problems, potentially challenging the necessity of domain independence as a strict design principle. In this paper, we explore the use of LLMs to automatically derive planning heuristics from task descriptions represented as successor generators and goal tests written in general purpose programming language. We investigate the trade-offs between domain-specific LLM-generated heuristics and traditional domain-independent methods in terms of computational efficiency and explainability. Our experiments demonstrate that LLMs can create heuristics that achieve state-of-the-art performance on some standard IPC domains, as well as their ability to solve problems that lack an adequate Planning Domain Definition Language ({\sc pddl}) representation. We discuss whether these results signify a paradigm shift and how they can complement existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03329v1">Industrial LLM-based Code Optimization under Regulation: A Mixture-of-Agents Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Submitted to ASE'25 Industry Showcase
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) for code optimization have enabled industrial platforms to automate software performance engineering at unprecedented scale and speed. Yet, organizations in regulated industries face strict constraints on which LLMs they can use - many cannot utilize commercial models due to data privacy regulations and compliance requirements, creating a significant challenge for achieving high-quality code optimization while maintaining cost-effectiveness. We address this by implementing a Mixture-of-Agents (MoA) approach that directly synthesizes code from multiple specialized LLMs, comparing it against TurinTech AI's vanilla Genetic Algorithm (GA)-based ensemble system and individual LLM optimizers using real-world industrial codebases. Our key contributions include: (1) First MoA application to industrial code optimization using real-world codebases; (2) Empirical evidence that MoA excels with open-source models, achieving 14.3% to 22.2% cost savings and 28.6% to 32.2% faster optimization times for regulated environments; (3) Deployment guidelines demonstrating GA's advantage with commercial models while both ensembles outperform individual LLMs; and (4) Real-world validation across 50 code snippets and seven LLM combinations, generating over 8,700 variants, addresses gaps in industrial LLM ensemble evaluation. This provides actionable guidance for organizations balancing regulatory compliance with optimization performance in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19794v3">Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate model behavior across three core dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. Code is available at https://github.com/zjunlp/DataMind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03298v1">GUI-ReRank: Enhancing GUI Retrieval with Multi-Modal LLM-based Reranking</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      GUI prototyping is a fundamental component in the development of modern interactive systems, which are now ubiquitous across diverse application domains. GUI prototypes play a critical role in requirements elicitation by enabling stakeholders to visualize, assess, and refine system concepts collaboratively. Moreover, prototypes serve as effective tools for early testing, iterative evaluation, and validation of design ideas with both end users and development teams. Despite these advantages, the process of constructing GUI prototypes remains resource-intensive and time-consuming, frequently demanding substantial effort and expertise. Recent research has sought to alleviate this burden through NL-based GUI retrieval approaches, which typically rely on embedding-based retrieval or tailored ranking models for specific GUI repositories. However, these methods often suffer from limited retrieval performance and struggle to generalize across arbitrary GUI datasets. In this work, we present GUI-ReRank, a novel framework that integrates rapid embedding-based constrained retrieval models with highly effective MLLM-based reranking techniques. GUI-ReRank further introduces a fully customizable GUI repository annotation and embedding pipeline, enabling users to effortlessly make their own GUI repositories searchable, which allows for rapid discovery of relevant GUIs for inspiration or seamless integration into customized LLM-based RAG workflows. We evaluated our approach on an established NL-based GUI retrieval benchmark, demonstrating that GUI-ReRank significantly outperforms SOTA tailored LTR models in both retrieval accuracy and generalizability. Additionally, we conducted a comprehensive cost and efficiency analysis of employing MLLMs for reranking, providing valuable insights regarding the trade-offs between retrieval effectiveness and computational resources. Video: https://youtu.be/_7x9UCh82ug
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01191v2">Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has been shown to improve Large Language Model (LLM) performance on various tasks. With this approach, LLMs appear to produce human-like reasoning steps before providing answers (a.k.a., CoT reasoning), which often leads to the perception that they engage in deliberate inferential processes. However, some initial findings suggest that CoT reasoning may be more superficial than it appears, motivating us to explore further. In this paper, we study CoT reasoning via a data distribution lens and investigate if CoT reasoning reflects a structured inductive bias learned from in-distribution data, allowing the model to conditionally generate reasoning paths that approximate those seen during training. Thus, its effectiveness is fundamentally bounded by the degree of distribution discrepancy between the training data and the test queries. With this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To investigate each dimension, we design DataAlchemy, an isolated and controlled environment to train LLMs from scratch and systematically probe them under various distribution conditions. Our results reveal that CoT reasoning is a brittle mirage that vanishes when it is pushed beyond training distributions. This work offers a deeper understanding of why and when CoT reasoning fails, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03292v1">Investigating Gender Bias in LLM-Generated Stories via Psychological Stereotypes</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly used across different applications, concerns about their potential to amplify gender biases in various tasks are rising. Prior research has often probed gender bias using explicit gender cues as counterfactual, or studied them in sentence completion and short question answering tasks. These formats might overlook more implicit forms of bias embedded in generative behavior of longer content. In this work, we investigate gender bias in LLMs using gender stereotypes studied in psychology (e.g., aggressiveness or gossiping) in an open-ended task of narrative generation. We introduce a novel dataset called StereoBias-Stories containing short stories either unconditioned or conditioned on (one, two, or six) random attributes from 25 psychological stereotypes and three task-related story endings. We analyze how the gender contribution in the overall story changes in response to these attributes and present three key findings: (1) While models, on average, are highly biased towards male in unconditioned prompts, conditioning on attributes independent from gender stereotypes mitigates this bias. (2) Combining multiple attributes associated with the same gender stereotype intensifies model behavior, with male ones amplifying bias and female ones alleviating it. (3) Model biases align with psychological ground-truth used for categorization, and alignment strength increases with model size. Together, these insights highlight the importance of psychology-grounded evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03275v1">LECTOR: LLM-Enhanced Concept-based Test-Oriented Repetition for Adaptive Spaced Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 15 pages, 4 figures, 1 table
    </div>
    <details class="paper-abstract">
      Spaced repetition systems are fundamental to efficient learning and memory retention, but existing algorithms often struggle with semantic interference and personalized adaptation. We present LECTOR (\textbf{L}LM-\textbf{E}nhanced \textbf{C}oncept-based \textbf{T}est-\textbf{O}riented \textbf{R}epetition), a novel adaptive scheduling algorithm specifically designed for test-oriented learning scenarios, particularly language examinations where success rate is paramount. LECTOR leverages large language models for semantic analysis while incorporating personalized learning profiles, addressing the critical challenge of semantic confusion in vocabulary learning by utilizing LLM-powered semantic similarity assessment and integrating it with established spaced repetition principles. Our comprehensive evaluation against six baseline algorithms (SSP-MMC, SM2, HLR, FSRS, ANKI, THRESHOLD) across 100 simulated learners over 100 days demonstrates significant improvements: LECTOR achieves a 90.2\% success rate compared to 88.4\% for the best baseline (SSP-MMC), representing a 2.0\% relative improvement. The algorithm shows particular strength in handling semantically similar concepts, reducing confusion-induced errors while maintaining computational efficiency. Our results establish LECTOR as a promising direction for intelligent tutoring systems and adaptive learning platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03262v1">Pay What LLM Wants: Can LLM Simulate Economics Experiment with 522 Real-human Persona?</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have generated significant interest in their capacity to simulate human-like behaviors, yet most studies rely on fictional personas rather than actual human data. We address this limitation by evaluating LLMs' ability to predict individual economic decision-making using Pay-What-You-Want (PWYW) pricing experiments with real 522 human personas. Our study systematically compares three state-of-the-art multimodal LLMs using detailed persona information from 522 Korean participants in cultural consumption scenarios. We investigate whether LLMs can accurately replicate individual human choices and how persona injection methods affect prediction performance. Results reveal that while LLMs struggle with precise individual-level predictions, they demonstrate reasonable group-level behavioral tendencies. Also, we found that commonly adopted prompting techniques are not much better than naive prompting methods; reconstruction of personal narrative nor retrieval augmented generation have no significant gain against simple prompting method. We believe that these findings can provide the first comprehensive evaluation of LLMs' capabilities on simulating economic behavior using real human data, offering empirical guidance for persona-based simulation in computational social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03258v1">SmartLLMs Scheduler: A Framework for Cost-Effective LLMs Utilization</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as GPT-4 and Llama have shown remarkable capabilities in a variety of software engineering tasks. Despite the advancements, their practical deployment faces challenges, including high financial costs, long response time, and varying performance, especially when handling a large number of queries (jobs). Existing optimization strategies for deploying LLMs for diverse tasks focus on static scheduling, which requires extensive training data for performance prediction, increasing the computational costs and limiting the applicability and flexibility. In this paper, we propose the SmartLLMs Scheduler (SLS), a dynamic and cost-effective scheduling solution. The key idea is to learn LLMs' performance on diverse tasks and incorporate their real-time feedback to update strategies periodically. Specifically, SLS incorporates three key components, including an Adaptive Cache Manager, a Performance-Cost Optimized Scheduler, and a Dynamic Update Manager. The Cache Manager stores the outputs of previously processed queries and employs an adaptive strategy to reduce redundant computations and minimize response times. For queries not found in the cache, the Scheduler dynamically allocates them to the most suitable LLM based on the predicted performance and cost from models that take both query-specific and LLM-specific features as input. The Update Manager continuously refines the cache and scheduling strategies based on real-time feedback from the assigned queries to enhance decision-making and adapt to evolving task characteristics. To evaluate the effectiveness of SLS, we conduct extensive experiments on two LLM-based software engineering tasks, including log parsing and code generation. The results show that SLS significantly outperforms the baseline methods, achieving an average performance improvement of 198.82% and an average processing time reduction of 63.28%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03247v1">Somatic in the East, Psychological in the West?: Investigating Clinically-Grounded Cross-Cultural Depression Symptom Expression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Prior clinical psychology research shows that Western individuals with depression tend to report psychological symptoms, while Eastern individuals report somatic ones. We test whether Large Language Models (LLMs), which are increasingly used in mental health, reproduce these cultural patterns by prompting them with Western or Eastern personas. Results show that LLMs largely fail to replicate the patterns when prompted in English, though prompting in major Eastern languages (i.e., Chinese, Japanese, and Hindi) improves alignment in several configurations. Our analysis pinpoints two key reasons for this failure: the models' low sensitivity to cultural personas and a strong, culturally invariant symptom hierarchy that overrides cultural cues. These findings reveal that while prompt language is important, current general-purpose LLMs lack the robust, culture-aware capabilities essential for safe and effective mental health applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00282v2">Mind the Gap: The Divergence Between Human and LLM-Generated Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied goals. We conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12537v2">What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the role of speech tokenizer designs in LLM-centric SLMs, augmented by speech heads and speaker modeling. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02089v3">SALAD: Systematic Assessment of Machine Unlearning on LLM-Aided Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative capabilities for hardware design automation, particularly in Verilog code generation. However, they also pose significant data security challenges, including Verilog evaluation data contamination, intellectual property (IP) design leakage, and the risk of malicious Verilog generation. We introduce SALAD, a comprehensive assessment that leverages machine unlearning to mitigate these threats. Our approach enables the selective removal of contaminated benchmarks, sensitive IP and design artifacts, or malicious code patterns from pre-trained LLMs, all without requiring full retraining. Through detailed case studies, we demonstrate how machine unlearning techniques effectively reduce data security risks in LLM-aided hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04856v3">M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted to ACL 2025 (Main Track). Camera-ready version
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12260v3">LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-based text retrieval retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full LLM on an A800 GPU, our method achieves over 1000x speedup in query encoding and over 10x increase in end-to-end retrieval throughput. Extensive experiments on large-scale retrieval benchmarks show that LightRetriever generalizes well across diverse tasks, maintaining an average of 95% retrieval performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03178v1">Light-IF: Endowing LLMs with Generalizable Reasoning via Preview and Self-Checking for Complex Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 12 pages, 10 figures, 7 tables
    </div>
    <details class="paper-abstract">
      While advancements in the reasoning abilities of LLMs have significantly enhanced their performance in solving mathematical problems, coding tasks, and general puzzles, their effectiveness in accurately adhering to instructions remains inconsistent, particularly with more complex directives. Our investigation identifies lazy reasoning during the thinking stage as the primary factor contributing to poor instruction adherence. To mitigate this issue, we propose a comprehensive framework designed to enable rigorous reasoning processes involving preview and self-checking, essential for satisfying strict instruction constraints. Specifically, we first generate instructions with complex constraints and apply a filtering process to obtain valid prompts, resulting in three distinct prompt datasets categorized as hard, easy, and pass. Then, we employ rejection sampling on the pass prompts to curate a small yet high-quality dataset, enabling a cold-start initialization of the model and facilitating its adaptation to effective reasoning patterns. Subsequently, we employ an entropy-preserving supervised fine-tuning (Entropy-SFT) strategy coupled with token-wise entropy-adaptive (TEA-RL) reinforcement learning guided by rule-based dense rewards. This approach encourages the model to transform its reasoning mechanism, ultimately fostering generalizable reasoning abilities that encompass preview and self-checking. Extensive experiments conducted on instruction-following benchmarks demonstrate remarkable performance improvements across various model scales. Notably, our Light-IF-32B model surpasses both larger open-source models such as DeepSeek-R1 and closed-source models like Doubao-1.6.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03153v1">Estimating Worst-Case Frontier Risks of Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      In this paper, we study the worst-case frontier risks of releasing gpt-oss. We introduce malicious fine-tuning (MFT), where we attempt to elicit maximum capabilities by fine-tuning gpt-oss to be as capable as possible in two domains: biology and cybersecurity. To maximize biological risk (biorisk), we curate tasks related to threat creation and train gpt-oss in an RL environment with web browsing. To maximize cybersecurity risk, we train gpt-oss in an agentic coding environment to solve capture-the-flag (CTF) challenges. We compare these MFT models against open- and closed-weight LLMs on frontier risk evaluations. Compared to frontier closed-weight models, MFT gpt-oss underperforms OpenAI o3, a model that is below Preparedness High capability level for biorisk and cybersecurity. Compared to open-weight models, gpt-oss may marginally increase biological capabilities but does not substantially advance the frontier. Taken together, these results contributed to our decision to release the model, and we hope that our MFT approach can serve as useful guidance for estimating harm from future open-weight releases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03148v1">Frontier: Simulating the Next Generation of LLM Inference Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference is growing increasingly complex with the rise of Mixture-of-Experts (MoE) models and disaggregated architectures that decouple components like prefill/decode (PD) or attention/FFN (AF) for heterogeneous scaling. Existing simulators, architected for co-located, dense models, are unable to capture the intricate system dynamics of these emerging paradigms. We present Frontier, a high-fidelity simulator designed from the ground up for this new landscape. Frontier introduces a unified framework to model both co-located and disaggregated systems, providing native support for MoE inference with expert parallelism (EP). It enables the simulation of complex workflows like cross-cluster expert routing and advanced pipelining strategies for latency hiding. To ensure fidelity and usability, Frontier incorporates refined operator models for improved accuracy. Frontier empowers the community to design and optimize the future of LLM inference at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03125v1">Attack the Messages, Not the Agents: A Multi-round Adaptive Stealthy Tampering Framework for LLM-MAS</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Large language model-based multi-agent systems (LLM-MAS) effectively accomplish complex and dynamic tasks through inter-agent communication, but this reliance introduces substantial safety vulnerabilities. Existing attack methods targeting LLM-MAS either compromise agent internals or rely on direct and overt persuasion, which limit their effectiveness, adaptability, and stealthiness. In this paper, we propose MAST, a Multi-round Adaptive Stealthy Tampering framework designed to exploit communication vulnerabilities within the system. MAST integrates Monte Carlo Tree Search with Direct Preference Optimization to train an attack policy model that adaptively generates effective multi-round tampering strategies. Furthermore, to preserve stealthiness, we impose dual semantic and embedding similarity constraints during the tampering process. Comprehensive experiments across diverse tasks, communication architectures, and LLMs demonstrate that MAST consistently achieves high attack success rates while significantly enhancing stealthiness compared to baselines. These findings highlight the effectiveness, stealthiness, and adaptability of MAST, underscoring the need for robust communication safeguards in LLM-MAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03097v1">VFLAIR-LLM: A Comprehensive Framework and Benchmark for Split Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 12 pages, 10 figures, published in KDD2025
    </div>
    <details class="paper-abstract">
      With the advancement of Large Language Models (LLMs), LLM applications have expanded into a growing number of fields. However, users with data privacy concerns face limitations in directly utilizing LLM APIs, while private deployments incur significant computational demands. This creates a substantial challenge in achieving secure LLM adaptation under constrained local resources. To address this issue, collaborative learning methods, such as Split Learning (SL), offer a resource-efficient and privacy-preserving solution for adapting LLMs to private domains. In this study, we introduce VFLAIR-LLM (available at https://github.com/FLAIR-THU/VFLAIR-LLM), an extensible and lightweight split learning framework for LLMs, enabling privacy-preserving LLM inference and fine-tuning in resource-constrained environments. Our library provides two LLM partition settings, supporting three task types and 18 datasets. In addition, we provide standard modules for implementing and evaluating attacks and defenses. We benchmark 5 attacks and 9 defenses under various Split Learning for LLM(SL-LLM) settings, offering concrete insights and recommendations on the choice of model partition configurations, defense strategies, and relevant hyperparameters for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03094v1">Augmenting Continual Learning of Diseases with LLM-Generated Visual Concepts</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Continual learning is essential for medical image classification systems to adapt to dynamically evolving clinical environments. The integration of multimodal information can significantly enhance continual learning of image classes. However, while existing approaches do utilize textual modality information, they solely rely on simplistic templates with a class name, thereby neglecting richer semantic information. To address these limitations, we propose a novel framework that harnesses visual concepts generated by large language models (LLMs) as discriminative semantic guidance. Our method dynamically constructs a visual concept pool with a similarity-based filtering mechanism to prevent redundancy. Then, to integrate the concepts into the continual learning process, we employ a cross-modal image-concept attention module, coupled with an attention loss. Through attention, the module can leverage the semantic knowledge from relevant visual concepts and produce class-representative fused features for classification. Experiments on medical and natural image datasets show our method achieves state-of-the-art performance, demonstrating the effectiveness and superiority of our method. We will release the code publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03092v1">Toward Verifiable Misinformation Detection: A Multi-Tool LLM Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Models (LLMs), the detection of misinformation has become increasingly important and complex. This research proposes an innovative verifiable misinformation detection LLM agent that goes beyond traditional true/false binary judgments. The agent actively verifies claims through dynamic interaction with diverse web sources, assesses information source credibility, synthesizes evidence, and provides a complete verifiable reasoning process. Our designed agent architecture includes three core tools: precise web search tool, source credibility assessment tool and numerical claim verification tool. These tools enable the agent to execute multi-step verification strategies, maintain evidence logs, and form comprehensive assessment conclusions. We evaluate using standard misinformation datasets such as FakeNewsNet, comparing with traditional machine learning models and LLMs. Evaluation metrics include standard classification metrics, quality assessment of reasoning processes, and robustness testing against rewritten content. Experimental results show that our agent outperforms baseline methods in misinformation detection accuracy, reasoning transparency, and resistance to information rewriting, providing a new paradigm for trustworthy AI-assisted fact-checking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03082v1">EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Automated Heuristic Design (AHD) using Large Language Models (LLMs) has achieved notable success in recent years. Despite the effectiveness of existing approaches, they only design a single heuristic to serve all problem instances, often inducing poor generalization across different distributions or settings. To address this issue, we propose Automated Heuristic Set Design (AHSD), a new formulation for LLM-driven AHD. The aim of AHSD is to automatically generate a small-sized complementary heuristic set to serve diverse problem instances, such that each problem instance could be optimized by at least one heuristic in this set. We show that the objective function of AHSD is monotone and supermodular. Then, we propose Evolution of Heuristic Set (EoH-S) to apply the AHSD formulation for LLM-driven AHD. With two novel mechanisms of complementary population management and complementary-aware memetic search, EoH-S could effectively generate a set of high-quality and complementary heuristics. Comprehensive experimental results on three AHD tasks with diverse instances spanning various sizes and distributions demonstrate that EoH-S consistently outperforms existing state-of-the-art AHD methods and achieves up to 60\% performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03080v1">ContractEval: Benchmarking LLMs for Clause-Level Legal Risk Identification in Commercial Contracts</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      The potential of large language models (LLMs) in specialized domains such as legal risk analysis remains underexplored. In response to growing interest in locally deploying open-source LLMs for legal tasks while preserving data confidentiality, this paper introduces ContractEval, the first benchmark to thoroughly evaluate whether open-source LLMs could match proprietary LLMs in identifying clause-level legal risks in commercial contracts. Using the Contract Understanding Atticus Dataset (CUAD), we assess 4 proprietary and 15 open-source LLMs. Our results highlight five key findings: (1) Proprietary models outperform open-source models in both correctness and output effectiveness, though some open-source models are competitive in certain specific dimensions. (2) Larger open-source models generally perform better, though the improvement slows down as models get bigger. (3) Reasoning ("thinking") mode improves output effectiveness but reduces correctness, likely due to over-complicating simpler tasks. (4) Open-source models generate "no related clause" responses more frequently even when relevant clauses are present. This suggests "laziness" in thinking or low confidence in extracting relevant content. (5) Model quantization speeds up inference but at the cost of performance drop, showing the tradeoff between efficiency and accuracy. These findings suggest that while most LLMs perform at a level comparable to junior legal assistants, open-source models require targeted fine-tuning to ensure correctness and effectiveness in high-stakes legal settings. ContractEval offers a solid benchmark to guide future development of legal-domain LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14220v2">Enhancing Spectral Graph Neural Networks with LLM-Predicted Homophily</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Spectral Graph Neural Networks (SGNNs) have achieved remarkable performance in tasks such as node classification due to their ability to learn flexible filters. Typically, these filters are learned under the supervision of downstream tasks, enabling SGNNs to adapt to diverse structural patterns. However, in scenarios with limited labeled data, SGNNs often struggle to capture the optimal filter shapes, resulting in degraded performance, especially on graphs with heterophily. Meanwhile, the rapid progress of Large Language Models (LLMs) has opened new possibilities for enhancing graph learning without modifying graph structure or requiring task-specific training. In this work, we propose a novel framework that leverages LLMs to estimate the homophily level of a graph and uses this global structural prior to guide the construction of spectral filters. Specifically, we design a lightweight and plug-and-play pipeline where a small set of labeled node pairs is formatted as natural language prompts for the LLM, which then predicts the graph's homophily ratio. This estimated value informs the spectral filter basis, enabling SGNNs to adapt more effectively to both homophilic and heterophilic structures. Extensive experiments on multiple benchmark datasets demonstrate that our LLM-assisted spectral framework consistently improves performance over strong SGNN baselines. Importantly, this enhancement incurs negligible computational and monetary cost, making it a practical solution for real-world graph applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09037v3">A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 72 pages, 6 figures. Accepted to TMLR, with Survey Certification award
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02999v1">AGENTiGraph: A Multi-Agent Knowledge Graph Framework for Interactive, Domain-Specific LLM Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 CIKM 2025, Demo Track
    </div>
    <details class="paper-abstract">
      AGENTiGraph is a user-friendly, agent-driven system that enables intuitive interaction and management of domain-specific data through the manipulation of knowledge graphs in natural language. It gives non-technical users a complete, visual solution to incrementally build and refine their knowledge bases, allowing multi-round dialogues and dynamic updates without specialized query languages. The flexible design of AGENTiGraph, including intent classification, task planning, and automatic knowledge integration, ensures seamless reasoning between diverse tasks. Evaluated on a 3,500-query benchmark within an educational scenario, the system outperforms strong zero-shot baselines (achieving 95.12% classification accuracy, 90.45% execution success), indicating potential scalability to compliance-critical or multi-step queries in legal and medical domains, e.g., incorporating new statutes or research on the fly. Our open-source demo offers a powerful new paradigm for multi-turn enterprise knowledge management that bridges LLMs and structured graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02994v1">When AIs Judge AIs: The Rise of Agent-as-a-Judge Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in capability and autonomy, evaluating their outputs-especially in open-ended and complex tasks-has become a critical bottleneck. A new paradigm is emerging: using AI agents as the evaluators themselves. This "agent-as-a-judge" approach leverages the reasoning and perspective-taking abilities of LLMs to assess the quality and safety of other models, promising calable and nuanced alternatives to human evaluation. In this review, we define the agent-as-a-judge concept, trace its evolution from single-model judges to dynamic multi-agent debate frameworks, and critically examine their strengths and shortcomings. We compare these approaches across reliability, cost, and human alignment, and survey real-world deployments in domains such as medicine, law, finance, and education. Finally, we highlight pressing challenges-including bias, robustness, and meta evaluation-and outline future research directions. By bringing together these strands, our review demonstrates how agent-based judging can complement (but not replace) human oversight, marking a step toward trustworthy, scalable evaluation for next-generation LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02979v1">Unified Tool Integration for LLMs: A Protocol-Agnostic Approach to Function Calling</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 arXiv admin note: substantial text overlap with arXiv:2507.10593
    </div>
    <details class="paper-abstract">
      The proliferation of tool-augmented Large Language Models (LLMs) has created a fragmented ecosystem where developers must navigate multiple protocols, manual schema definitions, and complex execution workflows. We address this challenge by proposing a unified approach to tool integration that abstracts protocol differences while optimizing execution performance. Our solution demonstrates how protocol-agnostic design principles can significantly reduce development overhead through automated schema generation, dual-mode concurrent execution, and seamless multi-source tool management. Experimental results show 60-80% code reduction across integration scenarios, performance improvements up to 3.1x through optimized concurrency, and full compatibility with existing function calling standards. This work contributes both theoretical insights into tool integration architecture and practical solutions for real-world LLM application development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02635v1">Test Set Quality in Multilingual LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Accepted at the 1st Workshop on Multilingual Data Quality Signals, COLM 2025, Short paper. 10 pages in total
    </div>
    <details class="paper-abstract">
      Several multilingual benchmark datasets have been developed in a semi-automatic manner in the recent past to measure progress and understand the state-of-the-art in the multilingual capabilities of Large Language Models. However, there is not a lot of attention paid to the quality of the datasets themselves, despite the existence of previous work in identifying errors in even fully human-annotated test sets. In this paper, we manually analyze recent multilingual evaluation sets in two languages - French and Telugu, identifying several errors in the process. We compare the performance difference across several LLMs with the original and revised versions of the datasets and identify large differences (almost 10% in some cases) in both languages). Based on these results, we argue that test sets should not be considered immutable and should be revisited, checked for correctness, and potentially versioned. We end with some recommendations for both the dataset creators as well as consumers on addressing the dataset quality issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02601v1">StructSynth: Leveraging LLMs for Structure-Aware Tabular Data Synthesis in Low-Data Regimes</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      The application of machine learning on tabular data in specialized domains is severely limited by data scarcity. While generative models offer a solution, traditional methods falter in low-data regimes, and recent Large Language Models (LLMs) often ignore the explicit dependency structure of tabular data, leading to low-fidelity synthetics. To address these limitations, we introduce StructSynth, a novel framework that integrates the generative power of LLMs with robust structural control. StructSynth employs a two-stage architecture. First, it performs explicit structure discovery to learn a Directed Acyclic Graph (DAG) from the available data. Second, this learned structure serves as a high-fidelity blueprint to steer the LLM's generation process, forcing it to adhere to the learned feature dependencies and thereby ensuring the generated data respects the underlying structure by design. Our extensive experiments demonstrate that StructSynth produces synthetic data with significantly higher structural integrity and downstream utility than state-of-the-art methods. It proves especially effective in challenging low-data scenarios, successfully navigating the trade-off between privacy preservation and statistical fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07927v3">Gandalf the Red: Adaptive Security for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally
    </div>
    <details class="paper-abstract">
      Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02573v1">Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Verbatim memorization in Large Language Models (LLMs) is a multifaceted phenomenon involving distinct underlying mechanisms. We introduce a novel method to analyze the different forms of memorization described by the existing taxonomy. Specifically, we train Convolutional Neural Networks (CNNs) on the attention weights of the LLM and evaluate the alignment between this taxonomy and the attention weights involved in decoding. We find that the existing taxonomy performs poorly and fails to reflect distinct mechanisms within the attention blocks. We propose a new taxonomy that maximizes alignment with the attention weights, consisting of three categories: memorized samples that are guessed using language modeling abilities, memorized samples that are recalled due to high duplication in the training set, and non-memorized samples. Our results reveal that few-shot verbatim memorization does not correspond to a distinct attention mechanism. We also show that a significant proportion of extractable samples are in fact guessed by the model and should therefore be studied separately. Finally, we develop a custom visual interpretability technique to localize the regions of the attention weights involved in each form of memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03336v2">Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02518v1">AnalogCoder-Pro: Unifying Analog Circuit Generation and Optimization via Multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Despite advances in analog design automation, analog front-end design still heavily depends on expert intuition and iterative simulations, underscoring critical gaps in fully automated optimization for performance-critical applications. Recently, the rapid development of Large Language Models (LLMs) has brought new promise to analog design automation. However, existing work remains in its early stages, and holistic joint optimization for practical end-to-end solutions remains largely unexplored. We propose AnalogCoder-Pro, a unified multimodal LLM-based framework that integrates generative capabilities and optimization techniques to jointly explore circuit topologies and optimize device sizing, automatically generating performance-specific, fully sized schematic netlists. AnalogCoder-Pro employs rejection sampling for fine-tuning LLMs on high-quality synthesized circuit data and introduces a multimodal diagnosis and repair workflow based on functional specifications and waveform images. By leveraging LLMs to interpret generated circuit netlists, AnalogCoder-Pro automates the extraction of critical design parameters and the formulation of parameter spaces, establishing an end-to-end workflow for simultaneous topology generation and device sizing optimization. Extensive experiments demonstrate that these orthogonal approaches significantly improve the success rate of analog circuit design and enhance circuit performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02515v1">PoeTone: A Framework for Constrained Generation of Structured Chinese Songci with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      This paper presents a systematic investigation into the constrained generation capabilities of large language models (LLMs) in producing Songci, a classical Chinese poetry form characterized by strict structural, tonal, and rhyme constraints defined by Cipai templates. We first develop a comprehensive, multi-faceted evaluation framework that includes: (i) a formal conformity score, (ii) automated quality assessment using LLMs, (iii) human evaluation, and (iv) classification-based probing tasks. Using this framework, we evaluate the generative performance of 18 LLMs, including 3 proprietary models and 15 open-source models across four families, under five prompting strategies: zero-shot, one-shot, completion-based, instruction-tuned, and chain-of-thought. Finally, we propose a Generate-Critic architecture in which the evaluation framework functions as an automated critic. Leveraging the critic's feedback as a reward signal, we fine-tune three lightweight open-source LLMs via supervised fine-tuning (SFT), resulting in improvements of up to 5.88% in formal conformity. Our findings offer new insights into the generative strengths and limitations of LLMs in producing culturally significant and formally constrained literary texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02503v1">OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, an LLM-based framework that produces high-quality solvers for optimization problems from natural-language descriptions without iterative self-correction. OptiHive uses a single batched LLM query to generate diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Taking into account the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5\% to 92\% on the most complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02480v1">MindShot: Multi-Shot Video Reconstruction from fMRI with LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Reconstructing dynamic videos from fMRI is important for understanding visual cognition and enabling vivid brain-computer interfaces. However, current methods are critically limited to single-shot clips, failing to address the multi-shot nature of real-world experiences. Multi-shot reconstruction faces fundamental challenges: fMRI signal mixing across shots, the temporal resolution mismatch between fMRI and video obscuring rapid scene changes, and the lack of dedicated multi-shot fMRI-video datasets. To overcome these limitations, we propose a novel divide-and-decode framework for multi-shot fMRI video reconstruction. Our core innovations are: (1) A shot boundary predictor module explicitly decomposing mixed fMRI signals into shot-specific segments. (2) Generative keyframe captioning using LLMs, which decodes robust textual descriptions from each segment, overcoming temporal blur by leveraging high-level semantics. (3) Novel large-scale data synthesis (20k samples) from existing datasets. Experimental results demonstrate our framework outperforms state-of-the-art methods in multi-shot reconstruction fidelity. Ablation studies confirm the critical role of fMRI decomposition and semantic captioning, with decomposition significantly improving decoded caption CLIP similarity by 71.8%. This work establishes a new paradigm for multi-shot fMRI reconstruction, enabling accurate recovery of complex visual narratives through explicit decomposition and semantic prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12601v3">CCSBench: Evaluating Compositional Controllability in LLMs for Scientific Document Summarization</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Accepted to KDD 2025 SciSoc LLM Workshop: Large Language Models for Scientific and Societal Advances
    </div>
    <details class="paper-abstract">
      To broaden the dissemination of scientific knowledge to diverse audiences, it is desirable for scientific document summarization systems to simultaneously control multiple attributes such as length and empirical focus. However, existing research typically focuses on controlling single attributes, leaving the compositional control of multiple attributes underexplored. To address this gap, we introduce CCSBench, the first evaluation benchmark for compositional controllable summarization in the scientific domain. Our benchmark enables fine-grained control over both explicit attributes (e.g., length), which are objective and straightforward, and implicit attributes (e.g., conceptual or empirical focus), which are more subjective and abstract. We conduct extensive experiments using various large language models (LLMs) under various settings, including in-context learning, parameter-efficient fine-tuning, and two-stage modular methods for balancing control over different attributes. Our findings reveal significant limitations in LLMs capabilities in balancing trade-offs between control attributes, especially implicit ones that require deeper understanding and abstract reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05108v4">Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 34 pages
    </div>
    <details class="paper-abstract">
      Discovering efficient algorithms for solving complex problems has been an outstanding challenge in mathematics and computer science, requiring substantial human expertise over the years. Recent advancements in evolutionary search with large language models (LLMs) have shown promise in accelerating the discovery of algorithms across various domains, particularly in mathematics and optimization. However, existing approaches treat the LLM as a static generator, missing the opportunity to update the model with the signal obtained from evolutionary exploration. In this work, we propose to augment LLM-based evolutionary search by continuously refining the search operator - the LLM - through reinforcement learning (RL) fine-tuning. Our method leverages evolutionary search as an exploration strategy to discover improved algorithms, while RL optimizes the LLM policy based on these discoveries. Our experiments on combinatorial optimization tasks demonstrate that integrating RL with evolutionary search accelerates the discovery of superior algorithms, showcasing the potential of RL-enhanced evolutionary strategies for algorithm design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02381v1">Beyond Manually Designed Pruning Policies with Second-Level Performance Prediction: A Pruning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Non-uniform structured network pruning methods can effectively reduce Large Language Model (LLM) size by eliminating redundant channels or layers, offering lower performance degradation than uniform strategies. However, existing non-uniform methods rely heavily on manually designed pruning policies (e.g., layer importance and scaling factors), and therefore cannot efficiently adapt to scenarios with dynamic pruning ratio requirements. Additionly, a critical bottleneck -- the time-consuming evaluation of pruning policies -- further limits the feasibility of iteratively and dynamically finding optimal pruning policies. To address these limitations, we propose PPF (Predictive Pruning Framework), a novel pruning framework for LLMs that eliminates manual design dependencies via second-level performance prediction. PPF not only supports real-time pruning decisions under dynamic pruning ratios but is also applicable to static pruning scenarios. It employs an agent for producing adaptive and real-time pruning actions, while a lightweight performance predictor that can evaluate a pruning policy in seconds, significantly speeding up the iterative optimization process. Experiments on Llama2-7B and Llama3-8B show that PPF can generate dynamic/static pruning policies and it reduces perplexity by up to 33.4% (dynamic pruning) and 84.78% (static pruning) over existing methods, outperforming manually designed pruning policies. The performance predictor achieves second-level performance prediction with high accuracy (prediction error < 0.0011). It reduces the mean evaluation latency from minute-level (1 minute and 38.02 seconds of test-set evaluation methods) to second-level (1.52 second), achieving over 64 times speedup. Our code will be available at https://github.com/Ma-zx/PPF .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v2">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts due to discrepancies between pre-trained parametric knowledge and real-time market data. These conflicts become particularly problematic when LLMs are deployed in real-world investment services, where misalignment between a model's embedded preferences and those of the financial institution can lead to unreliable recommendations. Yet little research has examined what investment views LLMs actually hold. We propose an experimental framework to investigate such conflicts, offering the first quantitative analysis of confirmation bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract the latent preferences of models and measure their persistence. Focusing on sector, size, and momentum, our analysis reveals distinct, model-specific tendencies. In particular, we observe a consistent preference for large-cap stocks and contrarian strategies across most models. These preferences often harden into confirmation bias, with models clinging to initial judgments despite counter-evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02344v1">Traffic-R1: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Traffic signal control (TSC) is vital for mitigating congestion and sustaining urban mobility. In this paper, we introduce Traffic-R1, a foundation model with human-like reasoning for TSC systems. Our model is developed through self-exploration and iteration of reinforced large language models (LLMs) with expert guidance in a simulated traffic environment. Compared to traditional reinforcement learning (RL) and recent LLM-based methods, Traffic-R1 offers three significant advantages. First, Traffic-R1 delivers zero-shot generalisation, transferring unchanged to new road networks and out-of-distribution incidents by utilizing its internal traffic control policies and human-like reasoning. Second, its 3B-parameter architecture is lightweight enough for real-time inference on mobile-class chips, enabling large-scale edge deployment. Third, Traffic-R1 provides an explainable TSC process and facilitates multi-intersection communication through its self-iteration and a new synchronous communication network. Extensive benchmarks demonstrate that Traffic-R1 sets a new state of the art, outperforming strong baselines and training-intensive RL controllers. In practice, the model now manages signals for more than 55,000 drivers daily, shortening average queues by over 5% and halving operator workload. Our checkpoint is available at https://huggingface.co/Season998/Traffic-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08113v2">Test Amplification for REST APIs via Single and Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      REST APIs (Representational State Transfer Application Programming Interfaces) play a vital role in modern cloud-native applications. As these APIs grow in complexity and scale, ensuring their correctness and robustness becomes increasingly important. Automated testing is essential for identifying hidden bugs, particularly those that appear in edge cases or under unexpected inputs. However, creating comprehensive and effective test suites for REST APIs is challenging and often demands significant effort. In this paper, we investigate the use of large language model (LLM) systems, both single-agent and multi-agent setups, for amplifying existing REST API test suites. These systems generate additional test cases that aim to push the boundaries of the API, uncovering behaviors that might otherwise go untested. We present a comparative evaluation of the two approaches across several dimensions, including test coverage, bug detection effectiveness, and practical considerations such as computational cost and energy usage. Our evaluation demonstrates increased API coverage, identification of numerous bugs in the API under test, and insights into the computational cost and energy consumption of both approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02298v1">CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback, helping to mitigate reward hacking. However, current RLVR methods typically treat whole responses as single actions, assigning the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies and inefficient learning. Methods like PPO provide credit assignment through value estimation, but often yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-by-step judgments for each reasoning step, but they require high-quality process supervision labels and are time-consuming when applied in online reinforcement learning (RL). To overcome these limitations, we introduce a simple but efficient method Credit Assignment Policy Optimization (CAPO). Given a reasoning response rollout from the policy model, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass, thereby providing verifiable token-level rewards to refine the tokens that were originally assigned identical rule-based rewards. This enables more fine-grained credit assignment in an effective way. Furthermore, to enhance the accuracy and robustness of CAPO, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments using different backbones like Llama and Qwen models and in different sizes show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across six challenging mathematical benchmarks and three out-of-domain benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10939v2">GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Accepted to ACL 2025 (main conference, short paper), 10 pages
    </div>
    <details class="paper-abstract">
      Large language models often struggle with zero-shot generalization, and several modular approaches have been proposed to address this challenge. Yet, we hypothesize that a key limitation remains: the entanglement of general knowledge and task-specific adaptations. To overcome this, we propose a modular framework that disentangles these components by constructing a library of task-specific LoRA modules alongside a general-domain LoRA. By subtracting this general knowledge component from each task-specific module, we obtain residual modules that focus more exclusively on task-relevant information, a method we call general knowledge subtraction (GenKnowSub). Leveraging the refined task-specific modules and the Arrow routing algorithm \citep{ostapenko2024towards}, we dynamically select and combine modules for new inputs without additional training. Our studies on the Phi-3 model and standard Arrow as baselines reveal that using general knowledge LoRAs derived from diverse languages, including English, French, and German, yields consistent performance gains in both monolingual and cross-lingual settings across a wide set of benchmarks. Further experiments on Phi-2 demonstrate how GenKnowSub generalizes to weaker LLMs. The complete code and data are available at https://github.com/saharsamr/Modular-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02233v1">A Methodological Framework for LLM-Based Mining of Software Repositories</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software engineering research, offering new opportunities for automating repository mining tasks. However, despite their growing popularity, the methodological integration of LLMs into Mining Software Repositories (MSR) remains poorly understood. Existing studies tend to focus on specific capabilities or performance benchmarks, providing limited insight into how researchers utilize LLMs across the full research pipeline. To address this gap, we conduct a mixed-method study that combines a rapid review and questionnaire survey in the field of LLM4MSR. We investigate (1) the approaches and (2) the threats that affect the empirical rigor of researchers involved in this field. Our findings reveal 15 methodological approaches, nine main threats, and 25 mitigation strategies. Building on these findings, we present PRIMES 2.0, a refined empirical framework organized into six stages, comprising 23 methodological substeps, each mapped to specific threats and corresponding mitigation strategies, providing prescriptive and adaptive support throughout the lifecycle of LLM-based MSR studies. Our work contributes to establishing a more transparent and reproducible foundation for LLM-based MSR research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02232v1">Eye2Recall: Exploring the Design of Enhancing Reminiscence Activities via Eye Tracking-Based LLM-Powered Interaction Experience for Older Adults</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Photo-based reminiscence has the potential to have a positive impact on older adults' reconnection with their personal history and improve their well-being. Supporting reminiscence in older adults through technological implementations is becoming an increasingly important area of research in the fields of HCI and CSCW. However, the impact of integrating gaze and speech as mixed-initiative interactions in LLM-powered reminiscence conversations remains under-explored. To address this, we conducted expert interviews to understand the challenges that older adults face with LLM-powered, photo-based reminiscence experiences. Based on these design considerations, we developed Eye2Recall, a system that integrates eye tracking for detecting visual interest with natural language interaction to create a mixed-initiative reminiscence experience. We evaluated its effectiveness through a user study involving ten older adults. The results have important implications for the future design of more accessible and empowering reminiscence technologies that better align with older adults' natural interaction patterns and enhance their positive aging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00737v2">How LLMs are Shaping the Future of Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Virtual Reality (VR) games marks a paradigm shift in the design of immersive, adaptive, and intelligent digital experiences. This paper presents a comprehensive review of recent research at the intersection of LLMs and VR, examining how these models are transforming narrative generation, non-player character (NPC) interactions, accessibility, personalization, and game mastering. Drawing from an analysis of 62 peer reviewed studies published between 2018 and 2025, we identify key application domains ranging from emotionally intelligent NPCs and procedurally generated storytelling to AI-driven adaptive systems and inclusive gameplay interfaces. We also address the major challenges facing this convergence, including real-time performance constraints, memory limitations, ethical risks, and scalability barriers. Our findings highlight that while LLMs significantly enhance realism, creativity, and user engagement in VR environments, their effective deployment requires robust design strategies that integrate multimodal interaction, hybrid AI architectures, and ethical safeguards. The paper concludes by outlining future research directions in multimodal AI, affective computing, reinforcement learning, and open-source development, aiming to guide the responsible advancement of intelligent and inclusive VR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02209v1">Balancing Information Accuracy and Response Timeliness in Networked LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have transformed many fields including scientific discovery, content generation, biomedical text mining, and educational technology. However, the substantial requirements for training data, computational resources, and energy consumption pose significant challenges for their practical deployment. A promising alternative is to leverage smaller, specialized language models and aggregate their outputs to improve overall response quality. In this work, we investigate a networked LLM system composed of multiple users, a central task processor, and clusters of topic-specialized LLMs. Each user submits categorical binary (true/false) queries, which are routed by the task processor to a selected cluster of $m$ LLMs. After gathering individual responses, the processor returns a final aggregated answer to the user. We characterize both the information accuracy and response timeliness in this setting, and formulate a joint optimization problem to balance these two competing objectives. Our extensive simulations demonstrate that the aggregated responses consistently achieve higher accuracy than those of individual LLMs. Notably, this improvement is more significant when the participating LLMs exhibit similar standalone performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13026v3">Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at mathematical reasoning and logical problem-solving. The current popular training paradigms primarily use supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the models' reasoning abilities. However, when using SFT or RL alone, there are respective challenges: SFT may suffer from overfitting, while RL is prone to mode collapse. The state-of-the-art methods have proposed hybrid training schemes. However, static switching faces challenges such as poor generalization across different tasks and high dependence on data quality. In response to these challenges, inspired by the curriculum learning-quiz mechanism in human reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training framework that theoretically unifies SFT and RL and dynamically balances the two throughout optimization. SASR uses SFT for initial warm-up to establish basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm based on gradient norm and divergence relative to the original distribution to seamlessly integrate SFT with the online RL method GRPO. By monitoring the training status of LLMs and adjusting the training process in sequence, SASR ensures a smooth transition between training schemes, maintaining core reasoning abilities while exploring different paths. Experimental results demonstrate that SASR outperforms SFT, RL, and static hybrid training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02110v1">Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface: adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. Our attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even under prompt-level defenses and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface, highlighting the need for execution-level security mechanisms that go beyond prompt-level defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02096v1">Evaluating User Experience in Conversational Recommender Systems: A Systematic Review Across Classical and LLM-Powered Approaches</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Accepted at OZCHI 2025. 23 pages, 1 figure, 5 tables
    </div>
    <details class="paper-abstract">
      Conversational Recommender Systems (CRSs) are receiving growing research attention across domains, yet their user experience (UX) evaluation remains limited. Existing reviews largely overlook empirical UX studies, particularly in adaptive and large language model (LLM)-based CRSs. To address this gap, we conducted a systematic review following PRISMA guidelines, synthesising 23 empirical studies published between 2017 and 2025. We analysed how UX has been conceptualised, measured, and shaped by domain, adaptivity, and LLM. Our findings reveal persistent limitations: post hoc surveys dominate, turn-level affective UX constructs are rarely assessed, and adaptive behaviours are seldom linked to UX outcomes. LLM-based CRSs introduce further challenges, including epistemic opacity and verbosity, yet evaluations infrequently address these issues. We contribute a structured synthesis of UX metrics, a comparative analysis of adaptive and nonadaptive systems, and a forward-looking agenda for LLM-aware UX evaluation. These findings support the development of more transparent, engaging, and user-centred CRS evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v1">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/wanghuacan/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02076v1">Everyone Contributes! Incentivizing Strategic Cooperation in Multi-LLM Systems via Sequential Public Goods Games</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Coordinating multiple large language models (LLMs) to solve complex tasks collaboratively poses a fundamental trade-off between the computation costs and collective performance compared with individual model. We introduce a novel, game-theoretically grounded reinforcement learning (RL) framework, the Multi-Agent Cooperation Sequential Public Goods Game (MAC-SPGG), to systematically incentivize cooperation in multi-LLM ensembles. In MAC-SPGG, LLM agents move in sequence, observing predecessors' outputs and updating beliefs to condition their own contributions. By redesigning the public-goods reward, effortful contributions become the unique Subgame Perfect Nash Equilibrium (SPNE), which eliminates free-riding under traditional SPGG or PGG. Its sequential protocol replaces costly round-based information exchanges with a streamlined decision flow, cutting communication overhead while retaining strategic depth. We prove the existence and uniqueness of the SPNE under realistic parameters, and empirically show that MAC-SPGG-trained ensembles outperform single-agent baselines, chain-of-thought prompting, and other cooperative methods, even achieving comparable performance to large-scale models across reasoning, math, code generation, and NLP tasks. Our results highlight the power of structured, incentive-aligned MAC-SPGG cooperation for scalable and robust multi-agent language generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02066v1">MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large Language Models(LLMs) have demonstrated remarkable performance across various domains, yet their capabilities in molecular reasoning remain insufficiently explored. Current approaches tend to rely heavily on general-purpose prompting, which lacks domain-specific molecular semantics, while those that use fine-tuning strategies often face challenges with interpretability and reasoning depth. To address these issues, we introduce MolReasoner, a two-stage framework designed to transition LLMs from memorization towards chemical reasoning. First, we propose Mol-SFT, which initializes the model's reasoning abilities via synthetic Chain-of-Thought(CoT) samples generated by GPT-4o and verified for chemical accuracy. Subsequently, Mol-RL applies reinforcement learning with specialized reward functions designed explicitly to align chemical structures with linguistic descriptions, thereby enhancing molecular reasoning capabilities. Our approach notably enhances interpretability, improving the model 's molecular understanding and enabling better generalization. Extensive experiments demonstrate that MolReasoner outperforms existing methods, and marking a significant shift from memorization-based outputs to robust chemical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02063v1">TRACEALIGN -- Tracing the Drift: Attributing Alignment Failures to Training-Time Belief Sources in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuned to align with human values often exhibit alignment drift, producing unsafe or policy-violating completions when exposed to adversarial prompts, decoding perturbations, or paraphrased jailbreaks. While prior work has behaviorally characterized alignment failure, little is known about the training-time belief sources underlying these failures. We introduce TraceAlign, a unified framework for tracing unsafe completions back to their root causes in the model's training corpus. Central to our approach is the Belief Conflict Index (BCI), which quantifies semantic inconsistency between generated spans and aligned policies, based on retrieved training documents using suffix-array matching. We propose three complementary interventions: (i) TraceShield, an inference-time safety filter that refuses completions with high-BCI spans, (ii) Contrastive Belief Deconfliction Loss, a contrastive fine-tuning objective penalizing high-BCI continuations during DPO, and (iii) Prov-Decode, a provenance-aware decoding strategy that vetoes beam expansions predicted to yield high-BCI spans. Together, these defenses reduce alignment drift by up to 85% on our curated Alignment Drift Benchmark (ADB) while preserving utility on standard tasks, with delta less than 0.2 and improved refusal quality. We further derive a theoretical upper bound on drift likelihood via suffix-array span statistics, linking memorization frequency and length to adversarial reactivation risk. TraceAlign thus provides the first scalable, traceable, and grounded toolkit for understanding and mitigating alignment failures at source. To encourage further exploration and development, we open-source our implementation at: https://anonymous.4open.science/r/tracealign-2DA7
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02053v1">ProCut: LLM Prompt Compression via Attribution Estimation</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      In large-scale industrial LLM systems, prompt templates often expand to thousands of tokens as teams iteratively incorporate sections such as task instructions, few-shot examples, and heuristic rules to enhance robustness and coverage. This expansion leads to bloated prompts that are difficult to maintain and incur significant inference latency and serving costs. To address this, we introduce Prompt Compression via Attribution Estimation (ProCut), a flexible, LLM-agnostic, training-free framework that compresses prompts through attribution analysis. ProCut segments prompt templates into semantically meaningful units, quantifies their impact on task performance, and prunes low-utility components. Through extensive experiments on five public benchmark datasets and real-world industrial prompts, we show that ProCut achieves substantial prompt size reductions (78% fewer tokens in production) while maintaining or even slightly improving task performance (up to 62% better than alternative methods). We further introduce an LLM-driven attribution estimator that reduces compression latency by over 50%, and demonstrate that ProCut integrates seamlessly with existing prompt-optimization frameworks to produce concise, high-performing prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02035v1">PhishParrot: LLM-Driven Adaptive Crawling to Unveil Cloaked Phishing Sites</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Accepted for publication at IEEE GLOBECOM 2025
    </div>
    <details class="paper-abstract">
      Phishing attacks continue to evolve, with cloaking techniques posing a significant challenge to detection efforts. Cloaking allows attackers to display phishing sites only to specific users while presenting legitimate pages to security crawlers, rendering traditional detection systems ineffective. This research proposes PhishParrot, a novel crawling environment optimization system designed to counter cloaking techniques. PhishParrot leverages the contextual analysis capabilities of Large Language Models (LLMs) to identify potential patterns in crawling information, enabling the construction of optimal user profiles capable of bypassing cloaking mechanisms. The system accumulates information on phishing sites collected from diverse environments. It then adapts browser settings and network configurations to match the attacker's target user conditions based on information extracted from similar cases. A 21-day evaluation showed that PhishParrot improved detection accuracy by up to 33.8% over standard analysis systems, yielding 91 distinct crawling environments for diverse conditions targeted by attackers. The findings confirm that the combination of similar-case extraction and LLM-based context analysis is an effective approach for detecting cloaked phishing attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06258v3">Emergent Response Planning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Published at ICML 2025. Code available at: https://github.com/niconi19/Emergent-Response-Planning-in-LLMs
    </div>
    <details class="paper-abstract">
      In this work, we argue that large language models (LLMs), though trained to predict only the next token, exhibit emergent planning behaviors: $\textbf{their hidden representations encode future outputs beyond the next token}$. Through simple probing, we demonstrate that LLM prompt representations encode global attributes of their entire responses, including $\textit{structure attributes}$ (e.g., response length, reasoning steps), $\textit{content attributes}$ (e.g., character choices in storywriting, multiple-choice answers at the end of response), and $\textit{behavior attributes}$ (e.g., answer confidence, factual consistency). In addition to identifying response planning, we explore how it scales with model size across tasks and how it evolves during generation. The findings that LLMs plan ahead for the future in their hidden representations suggest potential applications for improving transparency and generation control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v6">InfiniteHBD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism. However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs (e.g., TPUv4) take a middle-ground approach, but the fault explosion radius remains large. We propose InfiniteHBD, a transceiver-centric HBD architecture that integrates connectivity and dynamic switching at the transceiver level by embedding Optical Circuit Switching (OCS) within each transceiver. It enables reconfigurable point-to-multipoint communication and scalable variable-size ring topologies. InfiniteHBD achieves datacenter-scale scalability without cost explosion, fault isolation at the node level, and full bandwidth utilization for healthy GPUs. Key innovations include a Silicon Photonic-based OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology, and an HBD-DCN orchestration algorithm. The evaluation demonstrates that InfiniteHBD reduces cost to 31% of NVL-72, achieves a near-zero GPU waste ratio (over 10x lower than NVL-72 and TPUv4), maintains near-zero cross-ToR traffic under 7% node fault ratio, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs/node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02252v2">Compressing KV Cache for Long-Context LLM Inference with Inter-Layer Attention Similarity</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 14 pages, 7 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The rapid expansion of context window sizes in Large Language Models~(LLMs) has enabled them to tackle increasingly complex tasks involving lengthy documents. However, this progress comes at the cost of a substantial increase in memory usage during inference, primarily due to the linear growth of the key-value~(KV) cache. Existing KV cache compression methods often discard less relevant tokens, which can lead to significant performance degradation when critical information is lost. In this paper, we propose \textsc{PoD}~(Proximal tokens over Distant tokens), a novel KV cache compression framework that allocates memory according to token importance, retaining less important tokens in a more compact, shared form rather than discarding them entirely. Our approach is motivated by two key observations: (1) proximal tokens -- those at the beginning and end of the context -- are significantly more important for next-token prediction, and (2) attention scores for distant tokens are highly redundant across consecutive layers. Leveraging these insights, \textsc{PoD} preserves the full KV cache for proximal tokens, while for distant tokens, it shares key states across layers. Since attention scores are determined by both queries and keys, sharing key states enables multiple layers to reuse a single set of keys for distant tokens, substantially reducing KV cache memory without discarding essential context. We further introduce a lightweight post-training adaptation to enable the model to adjust to this new attention-sharing structure. Extensive experiments on both synthetic~(Needle in a Haystack) and real-world long-context benchmarks demonstrate that \textsc{PoD} reduces KV cache memory usage by up to 35\% without compromising performance. Our method is orthogonal to existing token-selection-based techniques and can be combined with them for further KV cache compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01989v1">Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 17 pages, 19 figures
    </div>
    <details class="paper-abstract">
      An ongoing debate considers whether prefill-decode (PD) aggregation or disaggregation is superior for serving large language models (LLMs). This has driven optimizations for both approaches, each showing distinct advantages. This paper compares PD aggregation and disaggregation, showing that each excels under different service-level objectives (SLOs): aggregation is optimal for tight time-to-first-token (TTFT) and relaxed time-per-output-token (TPOT), while disaggregation excels for strict TPOT and relaxed TTFT. However, under balanced TTFT and TPOT SLOs, neither approach delivers optimal goodput. This paper proposes TaiChi, an LLM serving system that unifies PD disaggregation and aggregation for optimal goodput under any combination of TTFT and TPOT SLOs. TaiChi uses a unified disaggregation-aggregation architecture with differentiated-capability GPU instances: prefill-heavy (fast prefill, high-interference decode) and decode-heavy (low-interference decode, slow prefill). Three configurable sliders control the ratio between these instances and their chunk sizes. TaiChi adapts to various SLO regimes by adjusting sliders. When TTFT constraints are tight, TaiChi resembles a PD aggregation configuration; when TPOT dominates, it adapts toward PD disaggregation. Crucially, under balanced SLOs, TaiChi enables a hybrid mode for superior goodput. The key innovation behind this hybrid mode is latency shifting: selectively reallocating GPU resources from requests that meet SLOs to those at risk of violation, maximizing the number of SLO-satisfied requests. This fine-grained latency shifting is orchestrated by two scheduling mechanisms: flowing decode scheduling to control TPOTs and length-aware prefill scheduling to manage TTFTs, which jointly optimize request assignment. Our experiments show TaiChi improves goodput by up to 77% over state-of-the-art systems under balanced TTFT and TPOT SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00198v2">Testing the Untestable? An Empirical Study on the Testing Process of LLM-Powered Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Background: Software systems powered by large language models are becoming a routine part of everyday technologies, supporting applications across a wide range of domains. In software engineering, many studies have focused on how LLMs support tasks such as code generation, debugging, and documentation. However, there has been limited focus on how full systems that integrate LLMs are tested during development. Aims: This study explores how LLM-powered systems are tested in the context of real-world application development. Method: We conducted an exploratory case study using 99 individual reports written by students who built and deployed LLM-powered applications as part of a university course. Each report was independently analyzed using thematic analysis, supported by a structured coding process. Results: Testing strategies combined manual and automated methods to evaluate both system logic and model behavior. Common practices included exploratory testing, unit testing, and prompt iteration. Reported challenges included integration failures, unpredictable outputs, prompt sensitivity, hallucinations, and uncertainty about correctness. Conclusions: Testing LLM-powered systems required adaptations to traditional verification methods, blending source-level reasoning with behavior-aware evaluations. These findings provide evidence on the practical context of testing generative components in software systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01969v1">Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly relied upon for solving complex reasoning tasks in domains such as mathematics, logic, and multi-step question answering. A growing line of work seeks to improve reasoning quality by scaling inference time compute particularly through Process Reward Models (PRMs), used to reward the reasoning at intermediate steps. While effective, these methods introduce substantial computational overhead, especially when generating large numbers of solutions in parallel. In this paper, we investigate whether PRMs can be used mid-generation to provide early signals that enable the rejection of suboptimal candidates before full generation of step is complete. We introduce the hypothesis that PRMs are also Partial Reward Models, meaning that the scores they assign to partially completed reasoning step are predictive of final output quality. This allows for principled early rejection based on intermediate token-level signals. We support this hypothesis both theoretically, by proving that the risk of discarding optimal beams decreases exponentially with generation length and empirically, by demonstrating a strong correlation between partial and final rewards across multiple reward models. On math reasoning benchmarks, our method achieves up to 1.4$\times$-9$\times$ reduction in inference FLOPs without degrading final performance. These results suggest that early rejection is a powerful mechanism for improving the compute-efficiency of reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02961v1">Defend LLMs Through Self-Consciousness</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 Presented at KDD Workshop on Ethical Artificial Intelligence: Methods and Applications (EAI) 2025
    </div>
    <details class="paper-abstract">
      This paper introduces a novel self-consciousness defense mechanism for Large Language Models (LLMs) to combat prompt injection attacks. Unlike traditional approaches that rely on external classifiers, our method leverages the LLM's inherent reasoning capabilities to perform self-protection. We propose a framework that incorporates Meta-Cognitive and Arbitration Modules, enabling LLMs to evaluate and regulate their own outputs autonomously. Our approach is evaluated on seven state-of-the-art LLMs using two datasets: AdvBench and Prompt-Injection-Mixed-Techniques-2024. Experiment results demonstrate significant improvements in defense success rates across models and datasets, with some achieving perfect and near-perfect defense in Enhanced Mode. We also analyze the trade-off between defense success rate improvement and computational overhead. This self-consciousness method offers a lightweight, cost-effective solution for enhancing LLM ethics, particularly beneficial for GenAI use cases across various platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13453v4">Adaptive Augmentation Policy Optimization with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-04
      | 💬 9 pages, 2 algorithms, 2 tables, submitted for consideration to 2025 International Conference on Machine Vision
    </div>
    <details class="paper-abstract">
      Data augmentation is a critical component of deep learning pipelines, enhancing model generalization by increasing dataset diversity. Traditional augmentation strategies rely on manually designed transformations, stochastic sampling, or automated search-based approaches. Although automated methods improve performance, they often require extensive computational resources and are specifically designed for certain datasets. In this work, we propose a Large Language Model (LLM)-guided augmentation optimization strategy that refines augmentation policies based on model performance feedback. We propose two approaches: (1) LLM-Guided Augmentation Policy Optimization, where augmentation policies selected by LLM are refined iteratively across training cycles, and (2) Adaptive LLM-Guided Augmentation Policy Optimization, which adjusts policies at each iteration based on performance metrics. This in-training approach eliminates the need for full model retraining before getting LLM feedback, reducing computational costs while increasing performance. Our methodology employs an LLM to dynamically select augmentation transformations based on dataset characteristics, model architecture, and prior training performance. Leveraging LLMs' contextual knowledge, especially in domain-specific tasks like medical imaging, our method selects augmentations tailored to dataset characteristics and model performance. Experiments across domain-specific image classification datasets show consistent accuracy improvements over traditional methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02931v1">Can LLMs Generate High-Quality Task-Specific Conversations?</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      This paper introduces a parameterization framework for controlling conversation quality in large language models. We explore nine key parameters across six dimensions that enable precise specification of dialogue properties. Through experiments with state-of-the-art LLMs, we demonstrate that parameter-based control produces statistically significant differences in generated conversation properties. Our approach addresses challenges in conversation generation, including topic coherence, knowledge progression, character consistency, and control granularity. The framework provides a standardized method for conversation quality control with applications in education, therapy, customer service, and entertainment. Future work will focus on implementing additional parameters through architectural modifications and developing benchmark datasets for evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12326v2">Reconstructing Sepsis Trajectories from Clinical Case Reports using LLMs: the Textual Time Series Corpus for Sepsis</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Clinical case reports and discharge summaries may be the most complete and accurate summarization of patient encounters, yet they are finalized, i.e., timestamped after the encounter. Complementary data structured streams become available sooner but suffer from incompleteness. To train models and algorithms on more complete and temporally fine-grained data, we construct a pipeline to phenotype, extract, and annotate time-localized findings within case reports using large language models. We apply our pipeline to generate an open-access textual time series corpus for Sepsis-3 comprising 2,139 case reports from the Pubmed-Open Access (PMOA) Subset. To validate our system, we apply it on PMOA and timeline annotations from I2B2/MIMIC-IV and compare the results to physician-expert annotations. We show high recovery rates of clinical findings (event match rates: O1-preview--0.755, Llama 3.3 70B Instruct--0.753) and strong temporal ordering (concordance: O1-preview--0.932, Llama 3.3 70B Instruct--0.932). Our work characterizes the ability of LLMs to time-localize clinical findings in text, illustrating the limitations of LLM use for temporal reconstruction and providing several potential avenues of improvement via multimodal integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11881v2">GPT is Devastated and LLaMA is Content: Emotion Representation Alignment in LLMs for Keyword-based Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      In controlled text generation using large language models (LLMs), gaps arise between the language model's interpretation of concepts and people's expectations. We introduce the human evaluation task of Representation Alignment for measuring this gap. We selected four emotion representations: Words, Valence-Arousal-Dominance (VAD) dimensions expressed in both Lexical and Numeric forms, and Emojis and evaluate them in the context of keyword-guided sentence generation using both GPT-4 and LLaMA-3. In addition to Representation Alignment, we also measure people's judgments of the accuracy and realism of the generated sentences. While representations like VAD break emotions into easy-to-compute components, our findings show that people agree more with how LLMs generate when conditioned on English words (e.g., ``angry'') rather than VAD scales. This difference is especially visible when comparing Numeric VAD to words. Furthermore, we found that the perception of how much a generated sentence conveys an emotion is dependent on both the representation type and which emotion it is.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04562v3">Evaluating LLMs on Real-World Forecasting Against Expert Forecasters</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against top forecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02827v1">Automated Validation of LLM-based Evaluators for Software Engineering Artifacts</a></div>
    <div class="paper-meta">
      📅 2025-08-04
    </div>
    <details class="paper-abstract">
      Automation in software engineering increasingly relies on large language models (LLMs) to generate, review, and assess code artifacts. However, establishing LLMs as reliable evaluators remains an open challenge: human evaluations are costly, subjective and non scalable, while existing automated methods fail to discern fine grained variations in artifact quality. We introduce REFINE (Ranking Evaluators for FIne grained Nuanced Evaluation), an automated framework for benchmarking LLM based evaluators across software engineering tasks. REFINE comprises of two modules: Hierarchy Dataset Builder applies novel generation techniques to automatically synthesize artifacts with progressively reduced quality, and Evaluator Tester quantifies each candidate evaluator configuration by measuring how closely its rankings align with expected ordering. A key feature of REFINE is controllability: users can tune the granularity of degradation to progressively refine evaluator configurations, from coarse filtering to stress testing on subtle quality gaps. While the methodology is general, we focus on coding tasks reflecting the practical demands in our production setting. REFINE was integrated into IBM's internal development workflows and applied to code generation, translation, and summarization for COBOL, an enterprise critical programming language, using industrial data. It was used to identify LLM as a Judge configurations that lifted alignment scores from below $0.7$ to above $0.9$ in some coding tasks. These nuance sensitive evaluators are now actively used by model training teams to support model release decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23035v2">KLLM: Fast LLM Inference with K-Means Quantization</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference poses significant challenges due to its intensive memory and computation demands. Weight and activation quantization (WAQ) offers a promising solution by reducing both memory footprint and arithmetic complexity. Traditional WAQ designs rely on uniform integer quantization for hardware efficiency, but often suffer from significant model performance degradation at low precision. In contrast, K-Means quantization, a non-uniform technique, achieves higher accuracy by aligning with the Gaussian-like distributions of weights and activations in LLMs. However, two key challenges prevent the efficient deployment of K-Means-based WAQ designs for LLM inference: (1) The non-uniform structure of K-Means-quantized data precludes direct execution on low-precision compute units, necessitating dequantization and floating-point matrix multiplications (MatMuls) during inference. (2) Activation outliers hinder effective low-precision quantization. Offline thresholding methods for outlier detection degrade model performance substantially, while existing online detection techniques introduce significant runtime overhead. To address the aforementioned challenges and fully unleash the potential of K-Means-based WAQ for LLM inference, in this paper, we propose KLLM, an LLM inference accelerator for efficient execution with K-Means-quantized weights and activations. KLLM features an index-based computation scheme for efficient execution of MatMuls and nonlinear operations on K-Means-quantized data, which avoids most of the dequantization and full-precision computations. Moreover, KLLM incorporates a lightweight outlier detection engine, Orizuru, that efficiently identifies the top-$k$ largest and smallest elements in the activation data stream during online inference.
    </details>
</div>
