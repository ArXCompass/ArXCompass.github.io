# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00717v1">LLMDR: LLM-Driven Deadlock Detection and Resolution in Multi-Agent Pathfinding</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Multi-Agent Pathfinding (MAPF) is a core challenge in multi-agent systems. Existing learning-based MAPF methods often struggle with scalability, particularly when addressing complex scenarios that are prone to deadlocks. To address these challenges, we introduce LLMDR (LLM-Driven Deadlock Detection and Resolution), an approach designed to resolve deadlocks and improve the performance of learnt MAPF models. LLMDR integrates the inference capabilities of large language models (LLMs) with learnt MAPF models and prioritized planning, enabling it to detect deadlocks and provide customized resolution strategies. We evaluate LLMDR on standard MAPF benchmark maps with varying agent numbers, measuring its performance when combined with several base models. The results demonstrate that LLMDR improves the performance of learnt MAPF models, particularly in deadlock-prone scenarios, with notable improvements in success rates. These findings show the potential of integrating LLMs to improve the scalability of learning-based MAPF methods. The source code for LLMDR is available at: https://github.com/ssbacc/llmdr-dhc
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00715v1">Exploring the Design Space of Real-time LLM Knowledge Support Systems: A Case Study of Jargon Explanations</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 This work is accepted and will be presented at CHI25
    </div>
    <details class="paper-abstract">
      Knowledge gaps often arise during communication due to diverse backgrounds, knowledge bases, and vocabularies. With recent LLM developments, providing real-time knowledge support is increasingly viable, but is challenging due to shared and individual cognitive limitations (e.g., attention, memory, and comprehension) and the difficulty in understanding the user's context and internal knowledge. To address these challenges, we explore the key question of understanding how people want to receive real-time knowledge support. We built StopGap -- a prototype that provides real-time knowledge support for explaining jargon words in videos -- to conduct a design probe study (N=24) that explored multiple visual knowledge representation formats. Our study revealed individual differences in preferred representations and highlighted the importance of user agency, personalization, and mixed-initiative assistance. Based on our findings, we map out six key design dimensions for real-time LLM knowledge support systems and offer insights for future research in this space.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00681v1">From Prompting to Partnering: Personalization Features for Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as ChatGPT, exhibit advanced capabilities in generating text, images, and videos. However, their effective use remains constrained by challenges in prompt formulation, personalization, and opaque decision-making processes. To investigate these challenges and identify design opportunities, we conducted a two-phase qualitative study. In Phase 1, we performed in-depth interviews with eight everyday LLM users after they engaged in structured tasks using ChatGPT across both familiar and unfamiliar domains. Our findings revealed key user difficulties in constructing effective prompts, iteratively refining AI-generated responses, and assessing response reliability especially in domains beyond users' expertise. Informed by these insights, we designed a high-fidelity prototype incorporating Reflective Prompting, Section Regeneration, Input-Output Mapping, Confidence Indicators, and a Customization Panel. In Phase 2, user testing of the prototype indicated that these interface-level improvements may prove useful for reducing cognitive load, increasing transparency, and fostering more intuitive and collaborative human-AI interactions. Our study contributes to the growing discourse on human-centred AI, advocating for human-LLM interactions that enhance user agency, transparency, and co-creative interaction, ultimately supporting more intuitive, accessible, and trustworthy generative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11856v2">Automatically Improving LLM-based Verilog Generation using EDA Tool Feedback</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Traditionally, digital hardware designs are written in the Verilog hardware description language (HDL) and debugged manually by engineers. This can be time-consuming and error-prone for complex designs. Large Language Models (LLMs) are emerging as a potential tool to help generate fully functioning HDL code, but most works have focused on generation in the single-shot capacity: i.e., run and evaluate, a process that does not leverage debugging and, as such, does not adequately reflect a realistic development process. In this work, we evaluate the ability of LLMs to leverage feedback from electronic design automation (EDA) tools to fix mistakes in their own generated Verilog. To accomplish this, we present an open-source, highly customizable framework, AutoChip, which combines conversational LLMs with the output from Verilog compilers and simulations to iteratively generate and repair Verilog. To determine the success of these LLMs we leverage the VerilogEval benchmark set. We evaluate four state-of-the-art conversational LLMs, focusing on readily accessible commercial models. EDA tool feedback proved to be consistently more effective than zero-shot prompting only with GPT-4o, the most computationally complex model we evaluated. In the best case, we observed a 5.8% increase in the number of successful designs with a 34.2% decrease in cost over the best zero-shot results. Mixing smaller models with this larger model at the end of the feedback iterations resulted in equally as much success as with GPT-4o using feedback, but incurred 41.9% lower cost (corresponding to an overall decrease in cost over zero-shot by 89.6%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.10279v3">We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 To appear in the 2025 USENIX Security Symposium. 22 pages, 14 figures, 8 tables. Edited from original version for submission to a different conference. No change to original results or findings
    </div>
    <details class="paper-abstract">
      The reliance of popular programming languages such as Python and JavaScript on centralized package repositories and open-source software, combined with the emergence of code-generating Large Language Models (LLMs), has created a new type of threat to the software supply chain: package hallucinations. These hallucinations, which arise from fact-conflicting errors when generating code using LLMs, represent a novel form of package confusion attack that poses a critical threat to the integrity of the software supply chain. This paper conducts a rigorous and comprehensive evaluation of package hallucinations across different programming languages, settings, and parameters, exploring how a diverse set of models and configurations affect the likelihood of generating erroneous package recommendations and identifying the root causes of this phenomenon. Using 16 popular LLMs for code generation and two unique prompt datasets, we generate 576,000 code samples in two programming languages that we analyze for package hallucinations. Our findings reveal that that the average percentage of hallucinated packages is at least 5.2% for commercial models and 21.7% for open-source models, including a staggering 205,474 unique examples of hallucinated package names, further underscoring the severity and pervasiveness of this threat. To overcome this problem, we implement several hallucination mitigation strategies and show that they are able to significantly reduce the number of package hallucinations while maintaining code quality. Our experiments and findings highlight package hallucinations as a persistent and systemic phenomenon while using state-of-the-art LLMs for code generation, and a significant challenge which deserves the research community's urgent attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v2">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20138v5">TradingAgents: Multi-Agents LLM Financial Trading Framework</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 Multi-Agent AI in the Real World @ AAAI 2025
    </div>
    <details class="paper-abstract">
      Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. TradingAgents is available at https://github.com/PioneerFintech.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07137v2">Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 ICLR 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Automatic LLM benchmarks, such as AlpacaEval 2.0, Arena-Hard-Auto, and MT-Bench, have become popular for evaluating language models due to their cost-effectiveness and scalability compared to human evaluation. Achieving high win rates on these benchmarks can significantly boost the promotional impact of newly released language models. This promotional benefit may motivate tricks, such as manipulating model output length or style to game win rates, even though several mechanisms have been developed to control length and disentangle style to reduce gameability. Nonetheless, we show that even a "null model" that always outputs a constant response (irrelevant to input instructions) can cheat automatic benchmarks and achieve top-ranked win rates: an 86.5% LC win rate on AlpacaEval 2.0; an 83.0 score on Arena-Hard-Auto; and a 9.55 score on MT-Bench. Moreover, the crafted cheating outputs are transferable because we assume that the instructions of these benchmarks (e.g., 805 samples of AlpacaEval 2.0) are private and cannot be accessed. While our experiments are primarily proof-of-concept, an adversary could use LLMs to generate more imperceptible cheating responses, unethically benefiting from high win rates and promotional impact. Our findings call for the development of anti-cheating mechanisms for reliable automatic benchmarks. The code is available at https://github.com/sail-sg/Cheating-LLM-Benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08305v2">Large Language Model(LLM) assisted End-to-End Network Health Management based on Multi-Scale Semanticization</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Network device and system health management is the foundation of modern network operations and maintenance. Traditional health management methods, relying on expert identification or simple rule-based algorithms, struggle to cope with the dynamic heterogeneous networks (DHNs) environment. Moreover, current state-of-the-art distributed anomaly detection methods, which utilize specific machine learning techniques, lack multi-scale adaptivity for heterogeneous device information, resulting in unsatisfactory diagnostic accuracy for DHNs. In this paper, we develop an LLM-assisted end-to-end intelligent network health management framework. The framework first proposes a Multi-Scale Semanticized Anomaly Detection Model (MSADM), incorporating semantic rule trees with an attention mechanism to address the multi-scale anomaly detection problem in DHNs. Secondly, a chain-of-thought-based large language model is embedded in downstream to adaptively analyze the fault detection results and produce an analysis report with detailed fault information and optimization strategies. Experimental results show that the accuracy of our proposed MSADM for heterogeneous network entity anomaly detection is as high as 91.31\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02956v2">Curriculum-style Data Augmentation for LLM-based Metaphor Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Recently, utilizing large language models (LLMs) for metaphor detection has achieved promising results. However, these methods heavily rely on the capabilities of closed-source LLMs, which come with relatively high inference costs and latency. To address this, we propose a method for metaphor detection by fine-tuning open-source LLMs, effectively reducing inference costs and latency with a single inference step. Furthermore, metaphor detection suffers from a severe data scarcity problem, which hinders effective fine-tuning of LLMs. To tackle this, we introduce Curriculum-style Data Augmentation (CDA). Specifically, before fine-tuning, we evaluate the training data to identify correctly predicted instances for fine-tuning, while incorrectly predicted instances are used as seed data for data augmentation. This approach enables the model to quickly learn simpler knowledge and progressively acquire more complex knowledge, thereby improving performance incrementally. Experimental results demonstrate that our method achieves state-of-the-art performance across all baselines. Additionally, we provide detailed ablation studies to validate the effectiveness of CDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01281v2">Path-Consistency: Prefix Enhancement for Efficient Inference in LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      To enhance the reasoning capabilities of large language models (LLMs), self-consistency has gained significant popularity by combining multiple sampling with majority voting. However, the state-of-the-art self-consistency approaches consume substantial computational resources and lead to significant additional time costs due to the multiple sampling. This prevents its full potential from being realized in scenarios where computational resources are critical. To improve the inference efficiency, this paper introduces \textit{path-consistency}, a method that leverages the confidence of answers generated in earlier branches to identify the prefix of the most promising path. By dynamically guiding the generation of subsequent branches based on this prefix, the \textit{path-consistency} mitigates both the errors and redundancies from random or less useful sampling in self-consistency. As a result, it can significantly accelerate the inference process by reducing the number of tokens generated. Our extensive empirical evaluation shows that the \textit{path-consistency} achieves significant acceleration in inference latency ranging from $7.8\%$ to $40.5\%$, while maintaining or even improving task accuracy across different datasets, including mathematical reasoning, common sense reasoning, symbolic reasoning, and code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01902v2">SeqAR: Jailbreak LLMs with Sequential Auto-Generated Characters</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 Accepted by NAACL 2025
    </div>
    <details class="paper-abstract">
      The widespread applications of large language models (LLMs) have brought about concerns regarding their potential misuse. Although aligned with human preference data before release, LLMs remain vulnerable to various malicious attacks. In this paper, we adopt a red-teaming strategy to enhance LLM safety and introduce SeqAR, a simple yet effective framework to design jailbreak prompts automatically. The SeqAR framework generates and optimizes multiple jailbreak characters and then applies sequential jailbreak characters in a single query to bypass the guardrails of the target LLM. Different from previous work which relies on proprietary LLMs or seed jailbreak templates crafted by human expertise, SeqAR can generate and optimize the jailbreak prompt in a cold-start scenario using open-sourced LLMs without any seed jailbreak templates. Experimental results show that SeqAR achieves attack success rates of 88% and 60% in bypassing the safety alignment of GPT-3.5-1106 and GPT-4, respectively. Furthermore, we extensively evaluate the transferability of the generated templates across different LLMs and held-out malicious requests, while also exploring defense strategies against the jailbreak attack designed by SeqAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09044v3">MiLoRA: Harnessing Minor Singular Components for Parameter-Efficient LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 This paper has been accepted at NAACL 2025. Code is available at: https://github.com/sufenlp/MiLoRA
    </div>
    <details class="paper-abstract">
      Efficient finetuning of large language models (LLMs) aims to adapt the LLMs with reduced computational and memory cost. Previous LoRA-based approaches initialize the low-rank matrices with Gaussian distribution and zero values while keeping the original weight matrices frozen. However, the trainable model parameters optimized in an unguided subspace might interfere with the well-learned subspace of the pretrained weight matrices. In this paper, we propose MiLoRA, a simple yet effective LLM finetuning approach that only updates the minor singular components of the weight matrix while keeping the principal singular components frozen. It is observed that the minor matrix corresponds to the noisy or long-tail information, while the principal matrix contains important knowledge. The MiLoRA initializes the low-rank matrices within a subspace that is orthogonal to the principal matrix, thus the pretrained knowledge is expected to be well preserved. During finetuning, MiLoRA makes the most use of the less-optimized subspace for learning the labeled dataset. Extensive experiments on commonsense reasoning, math reasoning, instruction following and visual instruction following benchmarks present the superior performance of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03247v2">End User Authoring of Personalized Content Classifiers: Comparing Example Labeling, Rule Writing, and LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 Accepted by CHI'25
    </div>
    <details class="paper-abstract">
      Existing tools for laypeople to create personal classifiers often assume a motivated user working uninterrupted in a single, lengthy session. However, users tend to engage with social media casually, with many short sessions on an ongoing, daily basis. To make creating personal classifiers for content curation easier for such users, tools should support rapid initialization and iterative refinement. In this work, we compare three strategies -- (1) example labeling, (2) rule writing, and (3) large language model (LLM) prompting -- for end users to build personal content classifiers. From an experiment with 37 non-programmers tasked with creating personalized moderation filters, we found that participants preferred different initializing strategies in different contexts, despite LLM prompting's better performance. However, all strategies faced challenges with iterative refinement. To overcome challenges in iterating on their prompts, participants even adopted hybrid approaches such as providing examples as in-context examples or writing rule-like prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07295v2">IterGen: Iterative Semantic-aware Structured LLM Generation with Backtracking</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for tasks such as natural language and code generation, but their outputs often suffer from issues like hallucination, toxicity, and incorrect results. Current libraries for structured LLM generation rely on left-to-right decoding without support for backtracking, limiting the ability to correct or refine outputs mid-generation. To address this, we introduce IterGen, a user-friendly library for iterative, grammar-guided LLM generation that enables users to move both forward and backward within the generated output based on grammar symbols. By leveraging a symbol-to-position mapping and maintaining the key-value (KV) cache state, IterGen ensures efficient and structured generation while allowing for corrections during the process. We demonstrate IterGen's effectiveness in two important applications: reducing privacy leakage in LLM outputs and improving the accuracy of LLM-generated SQL and Vega-Lite queries. Our code and additional resources are available at https://structuredllm.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.09051v3">Canvil: Designerly Adaptation for LLM-Powered User Experiences</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 CHI 2025 paper
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) are sparking a proliferation of LLM-powered user experiences (UX). In product teams, designers often craft UX to meet user needs, but it is unclear how they engage with LLMs as a novel design material. Through a formative study with 12 designers, we find that designers seek a translational process that enables design requirements to shape and be shaped by LLM behavior, motivating a need for designerly adaptation to facilitate this translation. We then built Canvil, a Figma widget that operationalizes designerly adaptation. We used Canvil as a probe to study designerly adaptation in a group-based design study (6 groups, N=17), finding that designers constructively iterated on both adaptation approaches and interface designs to enhance end-user interaction with LLMs. Furthermore, designers identified promising collaborative workflows for designerly adaptation. Our work opens new avenues for processes and tools that foreground designers' human-centered expertise when developing LLM-powered applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01052v1">ALinFiK: Learning to Approximate Linearized Future Influence Kernel for Scalable Third-Parity LLM Data Valuation</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 Accepted to NAACL 2025. Keywords: Influence Function, Data Valuation, Influence Estimation
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) heavily rely on high-quality training data, making data valuation crucial for optimizing model performance, especially when working within a limited budget. In this work, we aim to offer a third-party data valuation approach that benefits both data providers and model developers. We introduce a linearized future influence kernel (LinFiK), which assesses the value of individual data samples in improving LLM performance during training. We further propose ALinFiK, a learning strategy to approximate LinFiK, enabling scalable data valuation. Our comprehensive evaluations demonstrate that this approach surpasses existing baselines in effectiveness and efficiency, demonstrating significant scalability advantages as LLM parameters increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00392v1">Progressive Sparse Attention: Algorithm and System Co-design for Efficient Attention in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Processing long contexts has become a critical capability for modern large language models (LLMs). However, serving long-context LLMs comes with significant inference costs due to the high memory overhead of the key-value (KV) cache. Existing work leverages dynamic sparse attention algorithms (DSAes) to mitigate the KV cache overhead, but these algorithms rely on top-$k$ KV cache selection, which results in a trade-off between accuracy and efficiency. A larger $k$ improves accuracy but decreases efficiency, while a smaller $k$ boosts efficiency but compromises accuracy. To overcome this trade-off, this paper presents PSA, a $\underline{P}$rogressive $\underline{S}$parse $\underline{A}$ttention mechanism that integrates algorithmic innovations with system co-design to achieve both high inference accuracy and improved efficiency in LLM serving. The PSA algorithm adaptively adjusts the KV cache budget of different tokens and layers according to their real attention weight distributions, rather than relying on a fixed budget $k$. This enables high accuracy while minimizing KV cache usage. To further enhance execution efficiency, we introduce a pipelined iteration scheme that reduces CPU-GPU interleaving and synchronization overhead during PSA computation. Additionally, we implement unified GPU memory management that optimizes PSA's memory utilization by accounting for uneven memory requirements across different model layers. Extensive experimental results demonstrate that PSA reduces KV cache usage for attention computation by up to 2.4$\times$ and 8.8$\times$, and increases end-to-end serving throughput by up to 1.4$\times$ and 2.0$\times$, compared to state-of-the-art DSAes and systems without sparse attention, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00353v1">U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have expanded their context windows to unprecedented lengths, sparking debates about the necessity of Retrieval-Augmented Generation (RAG). To address the fragmented evaluation paradigms and limited cases in existing Needle-in-a-Haystack (NIAH), this paper introduces U-NIAH, a unified framework that systematically compares LLMs and RAG methods in controlled long context settings. Our framework extends beyond traditional NIAH by incorporating multi-needle, long-needle, and needle-in-needle configurations, along with different retrieval settings, while leveraging the synthetic Starlight Academy dataset-a fictional magical universe-to eliminate biases from pre-trained knowledge. Through extensive experiments, we investigate three research questions: (1) performance trade-offs between LLMs and RAG, (2) error patterns in RAG, and (3) RAG's limitations in complex settings. Our findings show that RAG significantly enhances smaller LLMs by mitigating the "lost-in-the-middle" effect and improving robustness, achieving an 82.58% win-rate over LLMs. However, we observe that retrieval noise and reverse chunk ordering degrade performance, while surprisingly, advanced reasoning LLMs exhibit reduced RAG compatibility due to sensitivity to semantic distractors. We identify typical error patterns including omission due to noise, hallucination under high noise critical condition, and self-doubt behaviors. Our work not only highlights the complementary roles of RAG and LLMs, but also provides actionable insights for optimizing deployments. Code: https://github.com/Tongji-KGLLM/U-NIAH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00330v1">How Deep is Love in LLMs' Hearts? Exploring Semantic Size in Human-like Cognition</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      How human cognitive abilities are formed has long captivated researchers. However, a significant challenge lies in developing meaningful methods to measure these complex processes. With the advent of large language models (LLMs), which now rival human capabilities in various domains, we are presented with a unique testbed to investigate human cognition through a new lens. Among the many facets of cognition, one particularly crucial aspect is the concept of semantic size, the perceived magnitude of both abstract and concrete words or concepts. This study seeks to investigate whether LLMs exhibit similar tendencies in understanding semantic size, thereby providing insights into the underlying mechanisms of human cognition. We begin by exploring metaphorical reasoning, comparing how LLMs and humans associate abstract words with concrete objects of varying sizes. Next, we examine LLMs' internal representations to evaluate their alignment with human cognitive processes. Our findings reveal that multi-modal training is crucial for LLMs to achieve more human-like understanding, suggesting that real-world, multi-modal experiences are similarly vital for human cognitive development. Lastly, we examine whether LLMs are influenced by attention-grabbing headlines with larger semantic sizes in a real-world web shopping scenario. The results show that multi-modal LLMs are more emotionally engaged in decision-making, but this also introduces potential biases, such as the risk of manipulation through clickbait headlines. Ultimately, this study offers a novel perspective on how LLMs interpret and internalize language, from the smallest concrete objects to the most profound abstract concepts like love. The insights gained not only improve our understanding of LLMs but also provide new avenues for exploring the cognitive abilities that define human intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00309v1">Pseudo-Knowledge Graph: Meta-Path Guided Retrieval and In-Graph Text for RAG-Equipped LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized natural language processing. However, these models face challenges in retrieving precise information from vast datasets. Retrieval-Augmented Generation (RAG) was developed to combining LLMs with external information retrieval systems to enhance the accuracy and context of responses. Despite improvements, RAG still struggles with comprehensive retrieval in high-volume, low-information-density databases and lacks relational awareness, leading to fragmented answers. To address this, this paper introduces the Pseudo-Knowledge Graph (PKG) framework, designed to overcome these limitations by integrating Meta-path Retrieval, In-graph Text and Vector Retrieval into LLMs. By preserving natural language text and leveraging various retrieval techniques, the PKG offers a richer knowledge representation and improves accuracy in information retrieval. Extensive evaluations using Open Compass and MultiHop-RAG benchmarks demonstrate the framework's effectiveness in managing large volumes of data and complex relationships.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01345v2">Towards Generalizable Vision-Language Robotic Manipulation: A Benchmark and LLM-guided 3D Policy</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 ICRA 2025
    </div>
    <details class="paper-abstract">
      Generalizing language-conditioned robotic policies to new tasks remains a significant challenge, hampered by the lack of suitable simulation benchmarks. In this paper, we address this gap by introducing GemBench, a novel benchmark to assess generalization capabilities of vision-language robotic manipulation policies. GemBench incorporates seven general action primitives and four levels of generalization, spanning novel placements, rigid and articulated objects, and complex long-horizon tasks. We evaluate state-of-the-art approaches on GemBench and also introduce a new method. Our approach 3D-LOTUS leverages rich 3D information for action prediction conditioned on language. While 3D-LOTUS excels in both efficiency and performance on seen tasks, it struggles with novel tasks. To address this, we present 3D-LOTUS++, a framework that integrates 3D-LOTUS's motion planning capabilities with the task planning capabilities of LLMs and the object grounding accuracy of VLMs. 3D-LOTUS++ achieves state-of-the-art performance on novel tasks of GemBench, setting a new standard for generalization in robotic manipulation. The benchmark, codes and trained models are available at https://www.di.ens.fr/willow/research/gembench/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19607v2">Revisiting Word Embeddings in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 This work was intended as a replacement of the older version, arXiv:2402.11094, and any subsequent updates will appear there
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown remarkable advancement in various NLP tasks. As such, a popular trend has emerged lately where NLP researchers extract word/sentence/document embeddings from these large decoder-only models and use them for various inference tasks with promising results. However, it is still unclear whether the performance improvement of LLM-induced embeddings is merely because of scale or whether underlying embeddings they produce significantly differ from classical encoding models like Word2Vec, GloVe, Sentence-BERT (SBERT) or Universal Sentence Encoder (USE). This is the central question we investigate in the paper by systematically comparing classical decontextualized and contextualized word embeddings with the same for LLM-induced embeddings. Our results show that LLMs cluster semantically related words more tightly and perform better on analogy tasks in decontextualized settings. However, in contextualized settings, classical models like SimCSE often outperform LLMs in sentence-level similarity assessment tasks, highlighting their continued relevance for fine-grained semantics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11094v3">Revisiting Word Embeddings in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 This is an updated version of the older version: 2402.11094. We accidentally submitted this article as a new submission (2502.19607), which we have requested to withdraw. This version has 30 pages and 22 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown remarkable advancement in various NLP tasks. As such, a popular trend has emerged lately where NLP researchers extract word/sentence/document embeddings from these large decoder-only models and use them for various inference tasks with promising results. However, it is still unclear whether the performance improvement of LLM-induced embeddings is merely because of scale or whether underlying embeddings they produce significantly differ from classical encoding models like Word2Vec, GloVe, Sentence-BERT (SBERT) or Universal Sentence Encoder (USE). This is the central question we investigate in the paper by systematically comparing classical decontextualized and contextualized word embeddings with the same for LLM-induced embeddings. Our results show that LLMs cluster semantically related words more tightly and perform better on analogy tasks in decontextualized settings. However, in contextualized settings, classical models like SimCSE often outperform LLMs in sentence-level similarity assessment tasks, highlighting their continued relevance for fine-grained semantics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08694v4">TeaMs-RL: Teaching LLMs to Generate Better Instruction Datasets via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      The development of Large Language Models (LLMs) often confronts challenges stemming from the heavy reliance on human annotators in the reinforcement learning with human feedback (RLHF) framework, or the frequent and costly external queries tied to the self-instruct paradigm. In this work, we pivot to Reinforcement Learning (RL) -- but with a twist. Diverging from the typical RLHF, which refines LLMs following instruction data training, we use RL to directly generate the foundational instruction dataset that alone suffices for fine-tuning. Our method, TeaMs-RL, uses a suite of textual operations and rules, prioritizing the diversification of training datasets. It facilitates the generation of high-quality data without excessive reliance on external advanced models, paving the way for a single fine-tuning step and negating the need for subsequent RLHF stages. Our findings highlight key advantages of our approach: reduced need for human involvement and fewer model queries (only 5.73% of the strong baseline's total), along with enhanced capabilities of LLMs in crafting and comprehending complex instructions compared to strong baselines, and substantially improved model privacy protection. Code is available at the link: https://github.com/SafeRL-Lab/TeaMs-RL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09102v2">Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are susceptible to security and safety threats, such as prompt injection, prompt extraction, and harmful requests. One major cause of these vulnerabilities is the lack of an instruction hierarchy. Modern LLM architectures treat all inputs equally, failing to distinguish between and prioritize various types of instructions, such as system messages, user prompts, and data. As a result, lower-priority user prompts may override more critical system instructions, including safety protocols. Existing approaches to achieving instruction hierarchy, such as delimiters and instruction-based training, do not address this issue at the architectural level. We introduce the Instructional Segment Embedding (ISE) technique, inspired by BERT, to modern large language models, which embeds instruction priority information directly into the model. This approach enables models to explicitly differentiate and prioritize various instruction types, significantly improving safety against malicious prompts that attempt to override priority rules. Our experiments on the Structured Query and Instruction Hierarchy benchmarks demonstrate an average robust accuracy increase of up to 15.75% and 18.68%, respectively. Furthermore, we observe an improvement in instruction-following capability of up to 4.1% evaluated on AlpacaEval. Overall, our approach offers a promising direction for enhancing the safety and effectiveness of LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17637v2">On Limitations of LLM as Annotator for Low Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Low-resource languages face significant challenges due to the lack of sufficient linguistic data, resources, and tools for tasks such as supervised learning, annotation, and classification. This shortage hinders the development of accurate models and datasets, making it difficult to perform critical NLP tasks like sentiment analysis or hate speech detection. To bridge this gap, Large Language Models (LLMs) present an opportunity for potential annotators, capable of generating datasets and resources for these underrepresented languages. In this paper, we focus on Marathi, a low-resource language, and evaluate the performance of both closed-source and open-source LLMs as annotators, while also comparing these results with fine-tuned BERT models. We assess models such as GPT-4o and Gemini 1.0 Pro, Gemma 2 (2B and 9B), and Llama 3.1 (8B and 405B) on classification tasks including sentiment analysis, news classification, and hate speech detection. Our findings reveal that while LLMs excel in annotation tasks for high-resource languages like English, they still fall short when applied to Marathi. Even advanced models like GPT-4o and Llama 3.1 405B underperform compared to fine-tuned BERT-based baselines, with GPT-4o and Llama 3.1 405B trailing fine-tuned BERT by accuracy margins of 10.2% and 14.1%, respectively. This highlights the limitations of LLMs as annotators for low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04286v2">Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      The efficacy of detectors for texts generated by large language models (LLMs) substantially depends on the availability of large-scale training data. However, white-box zero-shot detectors, which require no such data, are limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose a simple yet effective black-box zero-shot detection approach based on the observation that, from the perspective of LLMs, human-written texts typically contain more grammatical errors than LLM-generated texts. This approach involves calculating the Grammar Error Correction Score (GECScore) for the given text to differentiate between human-written and LLM-generated text. Experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.62% across XSum and Writing Prompts dataset. Additionally, our approach demonstrates strong reliability in the wild, exhibiting robust generalization and resistance to paraphrasing attacks. Data and code are available at: https://github.com/NLP2CT/GECScore.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18448v3">Multi-objective Representation for Numbers in Clinical Narratives: A CamemBERT-Bio-Based Alternative to Large-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Under the revision. arXiv admin note: substantial text overlap with arXiv:2404.10171
    </div>
    <details class="paper-abstract">
      The processing of numerical values is a rapidly developing area in the field of Language Models (LLMs). Despite numerous advancements achieved by previous research, significant challenges persist, particularly within the healthcare domain. This paper investigates the limitations of Transformer models in understanding numerical values. \textit{Objective:} this research aims to categorize numerical values extracted from medical documents into eight specific physiological categories using CamemBERT-bio. \textit{Methods:} In a context where scalable methods and Large Language Models (LLMs) are emphasized, we explore lifting the limitations of transformer-based models. We examine two strategies: fine-tuning CamemBERT-bio on a small medical dataset, integrating Label Embedding for Self-Attention (LESA), and combining LESA with additional enhancement techniques such as Xval. Given that CamemBERT-bio is already pre-trained on a large medical dataset, the first approach aims to update its encoder with the newly added label embeddings technique. In contrast, the second approach seeks to develop multiple representations of numbers (contextual and magnitude-based) to achieve more robust number embeddings. \textit{Results:} As anticipated, fine-tuning the standard CamemBERT-bio on our small medical dataset did not improve F1 scores. However, significant improvements were observed with CamemBERT-bio + LESA, resulting in an over 13\% increase. Similar enhancements were noted when combining LESA with Xval, outperforming conventional methods and giving comparable results to GPT-4 \textit{Conclusions and Novelty:} This study introduces two innovative techniques for handling numerical data, which are also applicable to other modalities. We illustrate how these techniques can improve the performance of Transformer-based models, achieving more reliable classification results even with small datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10596v2">Post-training an LLM for RAG? Train on Self-Generated Demonstrations</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with knowledge intensive NLP tasks, such as answering "Who won the latest World Cup?" because the knowledge they learn during training may be insufficient or outdated. Conditioning generation on retrieved documents -- a technique known as retrieval augmented generation (RAG) -- mitigates these shortcomings by allowing the model to leverage in-context information. Practitioners can improve LLM RAG performance by fine-tuning on retrieval-augmented instructions, but must beware that this can cause undesirable model behaviors like hallucinations. We attribute this degradation to the fact that the training data is likely to be out-of-distribution for the model and may suffer from quality issues, such as misalignment between retrievals and target responses (since retrievals are frequently added post-hoc). We propose a recipe for training RAG-enabled LLMs using self-generated demonstrations, thereby avoiding training on out-of-distribution text and integrating retrievals into the LLM responses. We evaluate our method on knowledge intensive question answering (QA) tasks and show that our method teaches LLMs to properly handle in-context retrievals and abstain from questions it will likely get wrong. Compared to conventional RA-IT methods, our method prevents model degradation in non-RAG settings while exhibiting superior QA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08332v2">Modification and Generated-Text Detection: Achieving Dual Detection Capabilities for the Outputs of LLM by Watermark</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) has raised concerns about potential misuse. One practical solution is to embed a watermark in the text, allowing ownership verification through watermark extraction. Existing methods primarily focus on defending against modification attacks, often neglecting other spoofing attacks. For example, attackers can alter the watermarked text to produce harmful content without compromising the presence of the watermark, which could lead to false attribution of this malicious content to the LLM. This situation poses a serious threat to the LLMs service providers and highlights the significance of achieving modification detection and generated-text detection simultaneously. Therefore, we propose a technique to detect modifications in text for unbiased watermark which is sensitive to modification. We introduce a new metric called ``discarded tokens", which measures the number of tokens not included in watermark detection. When a modification occurs, this metric changes and can serve as evidence of the modification. Additionally, we improve the watermark detection process and introduce a novel method for unbiased watermark. Our experiments demonstrate that we can achieve effective dual detection capabilities: modification detection and generated-text detection by watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03723v2">Speaking the Language of Teamwork: LLM-Guided Credit Assignment in Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 11 pages, 6 figures. Added the acknowledgement section
    </div>
    <details class="paper-abstract">
      Credit assignment, the process of attributing credit or blame to individual agents for their contributions to a team's success or failure, remains a fundamental challenge in multi-agent reinforcement learning (MARL), particularly in environments with sparse rewards. Commonly-used approaches such as value decomposition often lead to suboptimal policies in these settings, and designing dense reward functions that align with human intuition can be complex and labor-intensive. In this work, we propose a novel framework where a large language model (LLM) generates dense, agent-specific rewards based on a natural language description of the task and the overall team goal. By learning a potential-based reward function over multiple queries, our method reduces the impact of ranking errors while allowing the LLM to evaluate each agent's contribution to the overall task. Through extensive experiments, we demonstrate that our approach achieves faster convergence and higher policy returns compared to state-of-the-art MARL baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04650v2">Building Trust in Mental Health Chatbots: Safety Metrics and LLM-Based Evaluation Tools</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Objective: This study aims to develop and validate an evaluation framework to ensure the safety and reliability of mental health chatbots, which are increasingly popular due to their accessibility, human-like interactions, and context-aware support. Materials and Methods: We created an evaluation framework with 100 benchmark questions and ideal responses, and five guideline questions for chatbot responses. This framework, validated by mental health experts, was tested on a GPT-3.5-turbo-based chatbot. Automated evaluation methods explored included large language model (LLM)-based scoring, an agentic approach using real-time data, and embedding models to compare chatbot responses against ground truth standards. Results: The results highlight the importance of guidelines and ground truth for improving LLM evaluation accuracy. The agentic method, dynamically accessing reliable information, demonstrated the best alignment with human assessments. Adherence to a standardized, expert-validated framework significantly enhanced chatbot response safety and reliability. Discussion: Our findings emphasize the need for comprehensive, expert-tailored safety evaluation metrics for mental health chatbots. While LLMs have significant potential, careful implementation is necessary to mitigate risks. The superior performance of the agentic approach underscores the importance of real-time data access in enhancing chatbot reliability. Conclusion: The study validated an evaluation framework for mental health chatbots, proving its effectiveness in improving safety and reliability. Future work should extend evaluations to accuracy, bias, empathy, and privacy to ensure holistic assessment and responsible integration into healthcare. Standardized evaluations will build trust among users and professionals, facilitating broader adoption and improved mental health support through technology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00596v1">BadJudge: Backdoor Vulnerabilities of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Published to ICLR 2025
    </div>
    <details class="paper-abstract">
      This paper proposes a novel backdoor threat attacking the LLM-as-a-Judge evaluation regime, where the adversary controls both the candidate and evaluator model. The backdoored evaluator victimizes benign users by unfairly assigning inflated scores to adversary. A trivial single token backdoor poisoning 1% of the evaluator training data triples the adversary's score with respect to their legitimate score. We systematically categorize levels of data access corresponding to three real-world settings, (1) web poisoning, (2) malicious annotator, and (3) weight poisoning. These regimes reflect a weak to strong escalation of data access that highly correlates with attack severity. Under the weakest assumptions - web poisoning (1), the adversary still induces a 20% score inflation. Likewise, in the (3) weight poisoning regime, the stronger assumptions enable the adversary to inflate their scores from 1.5/5 to 4.9/5. The backdoor threat generalizes across different evaluator architectures, trigger designs, evaluation tasks, and poisoning rates. By poisoning 10% of the evaluator training data, we control toxicity judges (Guardrails) to misclassify toxic prompts as non-toxic 89% of the time, and document reranker judges in RAG to rank the poisoned document first 97% of the time. LLM-as-a-Judge is uniquely positioned at the intersection of ethics and technology, where social implications of mislead model selection and evaluation constrain the available defensive tools. Amidst these challenges, model merging emerges as a principled tool to offset the backdoor, reducing ASR to near 0% whilst maintaining SOTA performance. Model merging's low computational cost and convenient integration into the current LLM Judge training pipeline position it as a promising avenue for backdoor mitigation in the LLM-as-a-Judge setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01917v1">How to Steer LLM Latents for Hallucination Detection?</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 ICLR Workshop on Quantify Uncertainty and Hallucination in Foundation Models (QUESTION), 2025
    </div>
    <details class="paper-abstract">
      Hallucinations in LLMs pose a significant concern to their safe deployment in real-world applications. Recent approaches have leveraged the latent space of LLMs for hallucination detection, but their embeddings, optimized for linguistic coherence rather than factual accuracy, often fail to clearly separate truthful and hallucinated content. To this end, we propose the Truthfulness Separator Vector (TSV), a lightweight and flexible steering vector that reshapes the LLM's representation space during inference to enhance the separation between truthful and hallucinated outputs, without altering model parameters. Our two-stage framework first trains TSV on a small set of labeled exemplars to form compact and well-separated clusters. It then augments the exemplar set with unlabeled LLM generations, employing an optimal transport-based algorithm for pseudo-labeling combined with a confidence-based filtering process. Extensive experiments demonstrate that TSV achieves state-of-the-art performance with minimal labeled data, exhibiting strong generalization across datasets and providing a practical solution for real-world LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00590v1">Characterizing LLM-Empowered Personalized Story-Reading and Interaction for Children: Insights from Multi-Stakeholder Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 Accepted at CHI 2025
    </div>
    <details class="paper-abstract">
      Personalized interaction is highly valued by parents in their story-reading activities with children. While AI-empowered story-reading tools have been increasingly used, their abilities to support personalized interaction with children are still limited. Recent advances in large language models (LLMs) show promise in facilitating personalized interactions, but little is known about how to effectively and appropriately use LLMs to enhance children's personalized story-reading experiences. This work explores this question through a design-based study. Drawing on a formative study, we designed and developed StoryMate, an LLM-empowered personalized interactive story-reading tool for children, following an empirical study with children, parents, and education experts. Our participants valued the personalized features in StoryMate, and also highlighted the need to support personalized content, guiding mechanisms, reading context variations, and interactive interfaces. Based on these findings, we propose a series of design recommendations for better using LLMs to empower children's personalized story reading and interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00527v1">Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00502v1">Interact, Instruct to Improve: A LLM-Driven Parallel Actor-Reasoner Framework for Enhancing Autonomous Vehicle Interactions</a></div>
    <div class="paper-meta">
      📅 2025-03-01
    </div>
    <details class="paper-abstract">
      Autonomous Vehicles (AVs) have entered the commercialization stage, but their limited ability to interact and express intentions still poses challenges in interactions with Human-driven Vehicles (HVs). Recent advances in large language models (LLMs) enable bidirectional human-machine communication, but the conflict between slow inference speed and the need for real-time decision-making challenges practical deployment. To address these issues, this paper introduces a parallel Actor-Reasoner framework designed to enable explicit bidirectional AV-HV interactions across multiple scenarios. First, by facilitating interactions between the LLM-driven Reasoner and heterogeneous simulated HVs during training, an interaction memory database, referred to as the Actor, is established. Then, by introducing the memory partition module and the two-layer memory retrieval module, the Actor's ability to handle heterogeneous HVs is significantly enhanced. Ablation studies and comparisons with other decision-making methods demonstrate that the proposed Actor-Reasoner framework significantly improves safety and efficiency. Finally, with the combination of the external Human-Machine Interface (eHMI) information derived from Reasoner's reasoning and the feasible action solutions retrieved from the Actor, the effectiveness of the proposed Actor-Reasoner is confirmed in multi-scenario field interactions. Our code is available at https://github.com/FanGShiYuu/Actor-Reasoner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00491v1">Tutorial Proposal: Speculative Decoding for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-01
      | 💬 COLING 2025 Tutorial. Our homepage: https://speculative-decoding.github.io/
    </div>
    <details class="paper-abstract">
      This tutorial presents a comprehensive introduction to Speculative Decoding (SD), an advanced technique for LLM inference acceleration that has garnered significant research interest in recent years. SD is introduced as an innovative decoding paradigm to mitigate the high inference latency stemming from autoregressive decoding in LLMs. At each decoding step, SD efficiently drafts several future tokens and then verifies them in parallel. This approach, unlike traditional autoregressive decoding, facilitates the simultaneous decoding of multiple tokens per step, thereby achieving promising 2x-4x speedups in LLM inference while maintaining original distributions. This tutorial delves into the latest techniques in SD, including draft model architectures and verification strategies. Additionally, it explores the acceleration potential and future research directions in this promising field. We aim for this tutorial to elucidate the current research landscape and offer insights for researchers interested in Speculative Decoding, ultimately contributing to more efficient LLM inference.
    </details>
</div>
