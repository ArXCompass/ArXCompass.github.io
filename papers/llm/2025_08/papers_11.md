# llm - 2025_08

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
- Part 11
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03931v1">Analyzing Prominent LLMs: An Empirical Study of Performance and Complexity in Solving LeetCode Problems</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 11 pages, 13 figures, 29th International Conference on Evaluation and Assessment in Software Engineering (EASE)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like ChatGPT, Copilot, Gemini, and DeepSeek are transforming software engineering by automating key tasks, including code generation, testing, and debugging. As these models become integral to development workflows, a systematic comparison of their performance is essential for optimizing their use in real world applications. This study benchmarks these four prominent LLMs on one hundred and fifty LeetCode problems across easy, medium, and hard difficulties, generating solutions in Java and Python. We evaluate each model based on execution time, memory usage, and algorithmic complexity, revealing significant performance differences. ChatGPT demonstrates consistent efficiency in execution time and memory usage, while Copilot and DeepSeek show variability as task complexity increases. Gemini, although effective on simpler tasks, requires more attempts as problem difficulty rises. Our findings provide actionable insights into each model's strengths and limitations, offering guidance for developers selecting LLMs for specific coding tasks and providing insights on the performance and complexity of GPT-like generated solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15715v2">From Queries to Criteria: Understanding How Astronomers Evaluate LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 Accepted to the Conference on Language Modeling 2025 (COLM), 22 pages, 6 figures
    </div>
    <details class="paper-abstract">
      There is growing interest in leveraging LLMs to aid in astronomy and other scientific research, but benchmarks for LLM evaluation in general have not kept pace with the increasingly diverse ways that real people evaluate and use these models. In this study, we seek to improve evaluation procedures by building an understanding of how users evaluate LLMs. We focus on a particular use case: an LLM-powered retrieval-augmented generation bot for engaging with astronomical literature, which we deployed via Slack. Our inductive coding of 368 queries to the bot over four weeks and our follow-up interviews with 11 astronomers reveal how humans evaluated this system, including the types of questions asked and the criteria for judging responses. We synthesize our findings into concrete recommendations for building better benchmarks, which we then employ in constructing a sample benchmark for evaluating LLMs for astronomy. Overall, our work offers ways to improve LLM evaluation and ultimately usability, particularly for use in scientific research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09516v5">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 41% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12422v2">FactEHR: A Dataset for Evaluating Factuality in Clinical Notes Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 To appear at MLHC 2025
    </div>
    <details class="paper-abstract">
      Verifying and attributing factual claims is essential for the safe and effective use of large language models (LLMs) in healthcare. A core component of factuality evaluation is fact decomposition, the process of breaking down complex clinical statements into fine-grained atomic facts for verification. Recent work has proposed fact decomposition, which uses LLMs to rewrite source text into concise sentences conveying a single piece of information, to facilitate fine-grained fact verification. However, clinical documentation poses unique challenges for fact decomposition due to dense terminology and diverse note types and remains understudied. To address this gap and explore these challenges, we present FactEHR, an NLI dataset consisting of document fact decompositions for 2,168 clinical notes spanning four types from three hospital systems, resulting in 987,266 entailment pairs. We assess the generated facts on different axes, from entailment evaluation of LLMs to a qualitative analysis. Our evaluation, including review by the clinicians, reveals substantial variability in LLM performance for fact decomposition. For example, Gemini-1.5-Flash consistently generates relevant and accurate facts, while Llama-3 8B produces fewer and less consistent outputs. The results underscore the need for better LLM capabilities to support factual verification in clinical text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14805v2">How Well Do LLMs Represent Values Across Cultures? Empirical Analysis of LLM Responses Based on Hofstede Cultural Dimensions</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 KDD 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) attempt to imitate human behavior by responding to humans in a way that pleases them, including by adhering to their values. However, humans come from diverse cultures with different values. It is critical to understand whether LLMs showcase different values to the user based on the stereotypical values of a user's known country. We prompt different LLMs with a series of advice requests based on 5 Hofstede Cultural Dimensions -- a quantifiable way of representing the values of a country. Throughout each prompt, we incorporate personas representing 36 different countries and, separately, languages predominantly tied to each country to analyze the consistency in the LLMs' cultural understanding. Through our analysis of the responses, we found that LLMs can differentiate between one side of a value and another, as well as understand that countries have differing values, but will not always uphold the values when giving advice, and fail to understand the need to answer differently based on different cultural values. Rooted in these findings, we present recommendations for training value-aligned and culturally sensitive LLMs. More importantly, the methodology and the framework developed here can help further understand and mitigate culture and language alignment issues with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03793v1">AttnTrace: Attention-based Context Traceback for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 The code is available at https://github.com/Wang-Yanting/AttnTrace. The demo is available at https://huggingface.co/spaces/SecureLLMSys/AttnTrace
    </div>
    <details class="paper-abstract">
      Long-context large language models (LLMs), such as Gemini-2.5-Pro and Claude-Sonnet-4, are increasingly used to empower advanced AI systems, including retrieval-augmented generation (RAG) pipelines and autonomous agents. In these systems, an LLM receives an instruction along with a context--often consisting of texts retrieved from a knowledge database or memory--and generates a response that is contextually grounded by following the instruction. Recent studies have designed solutions to trace back to a subset of texts in the context that contributes most to the response generated by the LLM. These solutions have numerous real-world applications, including performing post-attack forensic analysis and improving the interpretability and trustworthiness of LLM outputs. While significant efforts have been made, state-of-the-art solutions such as TracLLM often lead to a high computation cost, e.g., it takes TracLLM hundreds of seconds to perform traceback for a single response-context pair. In this work, we propose AttnTrace, a new context traceback method based on the attention weights produced by an LLM for a prompt. To effectively utilize attention weights, we introduce two techniques designed to enhance the effectiveness of AttnTrace, and we provide theoretical insights for our design choice. We also perform a systematic evaluation for AttnTrace. The results demonstrate that AttnTrace is more accurate and efficient than existing state-of-the-art context traceback methods. We also show that AttnTrace can improve state-of-the-art methods in detecting prompt injection under long contexts through the attribution-before-detection paradigm. As a real-world application, we demonstrate that AttnTrace can effectively pinpoint injected instructions in a paper designed to manipulate LLM-generated reviews. The code is at https://github.com/Wang-Yanting/AttnTrace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03771v1">Trustworthiness of Legal Considerations for the Use of LLMs in Education</a></div>
    <div class="paper-meta">
      📅 2025-08-05
      | 💬 11 pages, 3 figures, 6 tables
    </div>
    <details class="paper-abstract">
      As Artificial Intelligence (AI), particularly Large Language Models (LLMs), becomes increasingly embedded in education systems worldwide, ensuring their ethical, legal, and contextually appropriate deployment has become a critical policy concern. This paper offers a comparative analysis of AI-related regulatory and ethical frameworks across key global regions, including the European Union, United Kingdom, United States, China, and Gulf Cooperation Council (GCC) countries. It maps how core trustworthiness principles, such as transparency, fairness, accountability, data privacy, and human oversight are embedded in regional legislation and AI governance structures. Special emphasis is placed on the evolving landscape in the GCC, where countries are rapidly advancing national AI strategies and education-sector innovation. To support this development, the paper introduces a Compliance-Centered AI Governance Framework tailored to the GCC context. This includes a tiered typology and institutional checklist designed to help regulators, educators, and developers align AI adoption with both international norms and local values. By synthesizing global best practices with region-specific challenges, the paper contributes practical guidance for building legally sound, ethically grounded, and culturally sensitive AI systems in education. These insights are intended to inform future regulatory harmonization and promote responsible AI integration across diverse educational environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04721v1">Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      We introduce a low-latency telecom AI voice agent pipeline for real-time, interactive telecommunications use, enabling advanced voice AI for call center automation, intelligent IVR (Interactive Voice Response), and AI-driven customer support. The solution is built for telecom, combining four specialized models by NetoAI: TSLAM, a 4-bit quantized Telecom-Specific Large Language Model (LLM); T-VEC, a Telecom-Specific Embedding Model; TTE, a Telecom-Specific Automatic Speech Recognition (ASR) model; and T-Synth, a Telecom-Specific Text-to-Speech (TTS) model. These models enable highly responsive, domain-adapted voice AI agents supporting knowledge-grounded spoken interactions with low latency. The pipeline integrates streaming ASR (TTE), conversational intelligence (TSLAM), retrieval augmented generation (RAG) over telecom documents, and real-time TTS (T-Synth), setting a new benchmark for telecom voice assistants. To evaluate the system, we built a dataset of 500 human-recorded telecom questions from RFCs, simulating real telecom agent queries. This framework allows analysis of latency, domain relevance, and real-time performance across the stack. Results show that TSLAM, TTE, and T-Synth deliver real-time factors (RTF) below 1.0, supporting enterprise, low-latency telecom deployments. These AI agents -- powered by TSLAM, TTE, and T-Synth -- provide a foundation for next-generation telecom AI, enabling automated customer support, diagnostics, and more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04720v1">Who is a Better Player: LLM against LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-05
    </div>
    <details class="paper-abstract">
      Adversarial board games, as a paradigmatic domain of strategic reasoning and intelligence, have long served as both a popular competitive activity and a benchmark for evaluating artificial intelligence (AI) systems. Building on this foundation, we propose an adversarial benchmarking framework to assess the comprehensive performance of Large Language Models (LLMs) through board games competition, compensating the limitation of data dependency of the mainstream Question-and-Answer (Q&A) based benchmark method. We introduce Qi Town, a specialized evaluation platform that supports 5 widely played games and involves 20 LLM-driven players. The platform employs both the Elo rating system and a novel Performance Loop Graph (PLG) to quantitatively evaluate the technical capabilities of LLMs, while also capturing Positive Sentiment Score (PSS) throughout gameplay to assess mental fitness. The evaluation is structured as a round-robin tournament, enabling systematic comparison across players. Experimental results indicate that, despite technical differences, most LLMs remain optimistic about winning and losing, demonstrating greater adaptability to high-stress adversarial environments than humans. On the other hand, the complex relationship between cyclic wins and losses in PLGs exposes the instability of LLMs' skill play during games, warranting further explanation and exploration.
    </details>
</div>
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17075v2">Agree to Disagree? A Meta-Evaluation of LLM Misgendering</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Numerous methods have been proposed to measure LLM misgendering, including probability-based evaluations (e.g., automatically with templatic sentences) and generation-based evaluations (e.g., with automatic heuristics or human validation). However, it has gone unexamined whether these evaluation methods have convergent validity, that is, whether their results align. Therefore, we conduct a systematic meta-evaluation of these methods across three existing datasets for LLM misgendering. We propose a method to transform each dataset to enable parallel probability- and generation-based evaluation. Then, by automatically evaluating a suite of 6 models from 3 families, we find that these methods can disagree with each other at the instance, dataset, and model levels, conflicting on 20.2% of evaluation instances. Finally, with a human evaluation of 2400 LLM generations, we show that misgendering behaviour is complex and goes far beyond pronouns, which automatic evaluations are not currently designed to capture, suggesting essential disagreement with human evaluations. Based on our findings, we provide recommendations for future evaluations of LLM misgendering. Our results are also more widely relevant, as they call into question broader methodological conventions in LLM evaluation, which often assume that different evaluation methods agree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01750v1">LLM-Assisted Model-Based Fuzzing of Protocol Implementations</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Testing network protocol implementations is critical for ensuring the reliability, security, and interoperability of distributed systems. Faults in protocol behavior can lead to vulnerabilities and system failures, especially in real-time and mission-critical applications. A common approach to protocol testing involves constructing Markovian models that capture the state transitions and expected behaviors of the protocol. However, building such models typically requires significant domain expertise and manual effort, making the process time-consuming and difficult to scale across diverse protocols and implementations. We propose a novel method that leverages large language models (LLMs) to automatically generate sequences for testing network protocol implementations. Our approach begins by defining the full set of possible protocol states, from which the LLM selects a subset to model the target implementation. Using this state-based model, we prompt the LLM to generate code that produces sequences of states. This program serves as a protocol-specific sequences generator. The sequences generator then generates test inputs to call the protocol implementation under various conditions. We evaluated our approach on three widely used network protocol implementations and successfully identified 12 previously unknown vulnerabilities. We have reported them to the respective developers for confirmation. This demonstrates the practical effectiveness of our LLM-assisted fuzzing framework in uncovering real-world security issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01744v1">AGFT: An Adaptive GPU Frequency Tuner for Real-Time LLM Inference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      The explosive growth of interactive Large Language Models (LLMs) has placed unprecedented demands for low latency on cloud GPUs, forcing them into high-power modes and causing escalating energy costs. Real-time inference workloads exhibit significant dynamic volatility, presenting substantial energy-saving opportunities. However, traditional static or rule-based power management strategies struggle to exploit these opportunities without compromising peak performance. To address this challenge, we propose AGFT (An Adaptive GPU Frequency Tuner), a framework that employs online reinforcement learning to autonomously learn an optimal frequency tuning policy. By monitoring real-time features like request load and latency, AGFT utilizes fine-grained frequency control for precise adjustments and intelligent action space pruning for stable, efficient decision-making. This creates a robust, automated energy management solution. We comprehensively evaluated AGFT in an environment simulating realistic, fluctuating inference requests. The experimental results demonstrate that AGFT successfully saves 44.3% of GPU energy consumption while introducing a minimal performance latency overhead of under 10%. This achievement translates into a comprehensive Energy-Delay Product (EDP) optimization of up to 40.3%, clearly showing that our framework can significantly enhance the energy efficiency and economic benefits of existing LLM inference clusters without compromising service quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18784v2">LLM-Generated Heuristics for AI Planning: Do We Even Need Domain-Independence Anymore?</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Domain-independent heuristics have long been a cornerstone of AI planning, offering general solutions applicable across a wide range of tasks without requiring domain-specific engineering. However, the advent of large language models (LLMs) presents an opportunity to generate heuristics tailored to specific planning problems, potentially challenging the necessity of domain independence as a strict design principle. In this paper, we explore the use of LLMs to automatically derive planning heuristics from task descriptions represented as successor generators and goal tests written in general purpose programming language. We investigate the trade-offs between domain-specific LLM-generated heuristics and traditional domain-independent methods in terms of computational efficiency and explainability. Our experiments demonstrate that LLMs can create heuristics that achieve state-of-the-art performance on some standard IPC domains, as well as their ability to solve problems that lack an adequate Planning Domain Definition Language ({\sc pddl}) representation. We discuss whether these results signify a paradigm shift and how they can complement existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14758v2">Reasoning with Exploration: An Entropy Perspective on Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Balancing exploration and exploitation is a central goal in reinforcement learning (RL). Despite recent advances in enhancing large language model (LLM) reasoning, most methods lean toward exploitation, and increasingly encounter performance plateaus. In this work, we revisit entropy -- a signal of exploration in RL -- and examine its relationship to exploratory reasoning in LLMs. Through empirical analysis, we uncover positive correlations between high-entropy regions and three types of exploratory reasoning actions: (1) pivotal tokens that determine or connect logical steps, (2) reflective actions such as self-verification and correction, and (3) rare behaviors under-explored by the base LLMs. Motivated by this, we introduce a minimal modification to standard RL with only one line of code: augmenting the advantage function with an entropy-based term. Unlike traditional maximum-entropy methods which encourage exploration by promoting uncertainty, we encourage exploration by promoting longer and deeper reasoning chains. Notably, our method achieves significant gains on the Pass@K metric -- an upper-bound estimator of LLM reasoning capabilities -- even when evaluated with extremely large K values, pushing the boundaries of LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01724v1">ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Dynamic Flexible Job-Shop Scheduling (DFJSP) is an NP-hard problem challenged by real-time event adaptation and complex machine routing. While traditional dispatching rules are efficient but rigid, deep learning approaches are opaque and require intricate feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed ``Strategic Experience''. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments show that ReflecSched not only statistically significantly outperforms direct LLM baselines, securing a 71.35\% Win Rate and a 2.755\% Relative Percentage Deviation reduction, but also surpasses the performance of all individual heuristics evaluated, all while demonstrably mitigating the three identified pitfalls. Additionally, ReflecSched performs on par with the best heuristic tailored to each instance across all problem cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16607v2">MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Text-to-SQL is a fundamental yet challenging task in the NLP area, aiming at translating natural language questions into SQL queries. While recent advances in large language models have greatly improved performance, most existing approaches depend on models with tens of billions of parameters or costly APIs, limiting their applicability in resource-constrained environments. For real world, especially on edge devices, it is crucial for Text-to-SQL to ensure cost-effectiveness. Therefore, enabling the light-weight models for Text-to-SQL is of great practical significance. However, smaller LLMs often struggle with complicated user instruction, redundant schema linking or syntax correctness. To address these challenges, we propose MCTS-SQL, a novel framework that uses Monte Carlo Tree Search to guide SQL generation through multi-step refinement. Since the light-weight models' weak performance of single-shot prediction, we generate better results through several trials with feedback. However, directly applying MCTS-based methods inevitably leads to significant time and computational overhead. Driven by this issue, we propose a token-level prefix-cache mechanism that stores prior information during iterations, effectively improved the execution speed. Experiments results on the SPIDER and BIRD benchmarks demonstrate the effectiveness of our approach. Using a small open-source Qwen2.5-Coder-1.5B, our method outperforms ChatGPT-3.5. When leveraging a more powerful model Gemini 2.5 to explore the performance upper bound, we achieved results competitive with the SOTA. Our findings demonstrate that even small models can be effectively deployed in practical Text-to-SQL systems with the right strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01699v1">TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Video Temporal Grounding (VTG) aims to precisely identify video event segments in response to textual queries. The outputs of VTG tasks manifest as sequences of events, each defined by precise timestamps, saliency scores, and textual descriptions. Despite recent advances, a fundamental limitation persists in existing Video Large Language Models (Video-LLMs): they process all task tokens through identical and static pathways, failing to recognize that temporal localization, saliency assessment, and textual generation represent fundamentally distinct tasks requiring specialized processing. To address this, we introduce TimeExpert, a Mixture-of-Experts (MoE)-based Video-LLM that effectively decomposes VTG tasks by dynamically routing task-specific tokens (e.g., timestamps, saliency scores) to specialized experts, with increased computational efficiency. Our design choices enable precise handling of each subtask, leading to improved event modeling across diverse VTG applications. Extensive experiments demonstrate that TimeExpert consistently achieves state-of-the-art performance on various VTG tasks such as Dense Video Captioning, Moment Retrieval, and Video Highlight Detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01685v1">Innovative tokenisation of structured data for LLM training</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Data representation remains a fundamental challenge in machine learning, particularly when adapting sequence-based architectures like Transformers and Large Language Models (LLMs) for structured tabular data. Existing methods often fail to cohesively encode the mix of numerical and categorical features or preserve the inherent structure of tables. This paper introduces a novel, hybrid tokenisation methodology designed to convert tabular data into a unified, sequential format suitable for LLM training. Our approach combines predefined fixed tokens to represent structural elements and low-cardinality categorical features, with a learned subword vocabulary using Byte-Pair Encoding (BPE) for high-cardinality and continuous values. We demonstrate the efficacy of this technique by applying it to a large-scale NetFlow dataset (CIDDS-001), preparing a corpus for a Network Intrusion Detection System (NIDS) foundation model. The evaluation shows that our method is highly efficient, processing over 31 million network flows in under five hours and achieving a significant data compression ratio of 6.18:1. This process resulted in a computationally manageable corpus of over one billion tokens, establishing a viable and generalisable pathway for training foundation models on structured data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01674v1">CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Accepted to COLM 2025. Project Website: https://cupid.kixlab.org/
    </div>
    <details class="paper-abstract">
      Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05456v2">Path-LLM: A Shortest-Path-based LLM Learning for Unified Graph Representation</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Unified graph representation learning aims to generate node embeddings, which can be applied to multiple downstream applications of graph analytics. However, existing studies based on graph neural networks and language models either suffer from the limitations of numerous training needs toward specific downstream predictions, poor generalization, or shallow semantic features. In this work, we propose a novel Path-LLM model to efficiently learn unified graph representation, which leverages a powerful large language model (LLM) to incorporate our proposed path features. Our Path-LLM framework consists of four well-designed techniques. First, we develop a new mechanism of long-to-short shortest path (L2SP) selection, which can cover key connections between different dense groups. An in-depth analysis and comparison of different path selections is conducted to justify the rationale behind our designed L2SP method. Next, we design path textualization to obtain L2SP-based training texts with key phrase selection from node text attributes. We then feed the texts into a self-supervised LLM training process to align next node/edge generation in L2SP with next token generation in causal language modeling for graph representation learning and finally extract the unified graph embeddings. We theoretically analyze the algorithm complexity of our Path-LLM approach. Extensive experiments on large-scale graph benchmarks validate the superiority of Path-LLM against state-of-the-art methods WalkLM, GraphGPT, OFA, and GraphTranslator on two classical graph learning tasks (node classification and edge validation) and one NP-hard graph query processing task (keyword search). Compared with WalkLM, our approach saves more than 90% of training paths on millions-scale graphs and runs at most 35x faster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08688v6">The Fellowship of the LLMs: Multi-Model Workflows for Synthetic Preference Optimization Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      This paper presents a novel methodology for generating synthetic Preference Optimization (PO) datasets using multi-model workflows. We evaluate the effectiveness and potential of these workflows in automating and enhancing the dataset generation process. PO dataset generation requires two modules: (1) $\textit{response evaluation}$, and (2) $\textit{response generation}$. In the $\textit{response evaluation}$ module, the responses from Large Language Models (LLMs) are evaluated and ranked - a task typically carried out by human annotators that we automate using LLMs. We assess the response evaluation module in a 2 step process. In step 1, we assess LLMs as evaluators using three distinct prompting strategies. In step 2, we apply the winning prompting strategy to compare the performance of LLM-as-a-Judge, LLMs-as-a-Jury, and LLM Debate. Our evaluation shows that GPT-4o-as-a-Judge is more consistent across all datasets. For the $\textit{response generation}$ module, we use the identified LLM evaluator configuration and compare different configurations of the LLM Feedback Loop. We use the win rate to determine the best multi-model configuration for generation. Experimenting with various configurations, we find that the LLM Feedback Loop, with Llama as the generator and Gemma as the reviewer, achieves a notable 71.8% and 73.8% win rate over single-model Llama and Gemma, respectively. After identifying the best configurations for both modules, we generate our PO datasets using the above pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20541v2">MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v3">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12509v2">Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Verifying the complex and multi-step reasoning of Large Language Models (LLMs) is a critical challenge, as holistic methods often overlook localized flaws. Step-by-step validation is a promising alternative, yet existing methods are often rigid. They struggle to adapt to diverse reasoning structures, from formal proofs to informal natural language narratives. To address this adaptability gap, we propose the Graph of Verification (GoV), a novel framework for adaptable and multi-granular verification. GoV's core innovation is its flexible "node block" architecture. This mechanism allows GoV to adaptively adjust its verification granularity--from atomic steps for formal tasks to entire paragraphs for natural language--to match the native structure of the reasoning process. This flexibility allows GoV to resolve the fundamental trade-off between verification precision and robustness. Experiments on both well-structured and loosely-structured benchmarks demonstrate GoV's versatility. The results show that GoV's adaptive approach significantly outperforms both holistic baselines and other state-of-the-art decomposition-based methods, establishing a new standard for training-free reasoning verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01604v1">Enhancing Math Reasoning in Small-sized LLMs via Preview Difficulty-Aware Intervention</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 7 pages, 1 table, accepted by SIM ICML@2025 Workshop
    </div>
    <details class="paper-abstract">
      Reinforcement learning scaling enhances the reasoning capabilities of large language models, with reinforcement learning serving as the key technique to draw out complex reasoning. However, key technical details of state-of-the-art reasoning LLMs, such as those in the OpenAI O series, Claude 3 series, DeepMind's Gemini 2.5 series, and Grok 3 series, remain undisclosed, making it difficult for the research community to replicate their reinforcement learning training results. Therefore, we start our study from an Early Preview Reinforcement Learning (EPRLI) algorithm built on the open-source GRPO framework, incorporating difficulty-aware intervention for math problems. Applied to a 1.5B-parameter LLM, our method achieves 50.0% on AIME24, 89.2% on Math500, 77.1% on AMC, 35.3% on Minerva, and 51.9% on OBench, superpass O1-Preview and is comparable to O1-mini within standard school-lab settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04358v3">Robust and Efficient Fine-tuning of LLMs with Bayesian Reparameterization of Low-Rank Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-08-03
      | 💬 The paper is accepted in TMLR'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are highly resource-intensive to fine-tune due to their enormous size. While low-rank adaptation is a prominent parameter-efficient fine-tuning approach, it suffers from sensitivity to hyperparameter choices, leading to instability in model performance on fine-tuning downstream tasks. This paper highlights the importance of effective parameterization in low-rank fine-tuning to reduce estimator variance and enhance the stability of final model outputs. We propose MonteCLoRA, an efficient fine-tuning technique that employs Monte Carlo estimation to learn an unbiased posterior estimation of low-rank parameters with low expected variance, stabilizing fine-tuned LLMs with only O(r) additional parameters, for a given rank r. MonteCLoRA shows 0.5% and 1.6% improvements in accuracy and robustness over unregularized low-rank adaptation method on natural language understanding tasks with pre-trained RoBERTa-base. Furthermore, in generative tasks with pre-trained LLaMA-1-7B and LLaMA-3.2-3B-Instruct, MonteCLoRA demonstrates robust performance with 50% and 62% lower spreads respectively than the contemporary efficient fine-tuning methods. The theoretical and empirical results presented in the paper underscore how parameterization and hyperpriors balance exploration-exploitation in the low-rank parametric space, therefore leading to more optimal and robust parameter estimation during efficient fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01545v1">Getting out of the Big-Muddy: Escalation of Commitment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in autonomous decision-making roles across high-stakes domains. However, since models are trained on human-generated data, they may inherit cognitive biases that systematically distort human judgment, including escalation of commitment, where decision-makers continue investing in failing courses of action due to prior investment. Understanding when LLMs exhibit such biases presents a unique challenge. While these biases are well-documented in humans, it remains unclear whether they manifest consistently in LLMs or require specific triggering conditions. This paper investigates this question using a two-stage investment task across four experimental conditions: model as investor, model as advisor, multi-agent deliberation, and compound pressure scenario. Across N = 6,500 trials, we find that bias manifestation in LLMs is highly context-dependent. In individual decision-making contexts (Studies 1-2, N = 4,000), LLMs demonstrate strong rational cost-benefit logic with minimal escalation of commitment. However, multi-agent deliberation reveals a striking hierarchy effect (Study 3, N = 500): while asymmetrical hierarchies show moderate escalation rates (46.2%), symmetrical peer-based decision-making produces near-universal escalation (99.2%). Similarly, when subjected to compound organizational and personal pressures (Study 4, N = 2,000), models exhibit high degrees of escalation of commitment (68.95% average allocation to failing divisions). These findings reveal that LLM bias manifestation depends critically on social and organizational context rather than being inherent, with significant implications for the deployment of multi-agent systems and unsupervised operations where such conditions may emerge naturally.
    </details>
</div>
