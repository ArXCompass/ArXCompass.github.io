# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14810v1">DONOD: Robust and Generalizable Instruction Fine-Tuning for LLMs via Model-Intrinsic Dataset Pruning</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Ad-hoc instruction fine-tuning of large language models (LLMs) is widely adopted for domain-specific adaptation. While domain-specific supervised fine-tuning (SFT) is effective and efficient, it often weakens cross-domain generalization and struggles with noisy training data. To address these challenges, we propose DONOD, a lightweight model-intrinsic data pruning method. Our approach evaluates data using two model-parameter-based metrics: Delta of Norm (DON), which captures the cumulative influence on model weights, and Norm of Delta (NOD), which quantifies weight instability. Moreover, by employing the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) algorithm, we effectively filter noisy, unlearnable, and generalization-harming samples without relying on auxiliary models during the SFT process. Experiments on mathematical tasks demonstrate that data selected by DONOD achieve superior fine-tuning efficiency and improved robustness against noisy data. By filtering out 70% of the full dataset, we improve target-domain accuracy by 14.90% and cross-domain accuracy by 5.67%. Meanwhile, our selected data present superior cross-architecture generalization. Data pruned by smaller models (e.g., Llama 3.1-8B) generalize effectively on larger models (e.g., Llama 2-13B). Compared to existing related methodologies, DONOD demonstrates comparable or superior performance while remaining dataset-agnostic, enabling broader applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15547v2">Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are combined with tools to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or tool's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompts are prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., agent isolation, secure untrusted data processing, and privilege escalation guardrails. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05693v2">Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted by AAAI 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) stand out for their impressive performance in intricate language modeling tasks. However, their demanding computational and memory needs pose obstacles for broad use on edge devices. Quantization is then introduced to boost LLMs' on-device efficiency. Recent works show that 8-bit or lower weight quantization is feasible with minimal impact on end-to-end task performance, while the activation is still not quantized. On the other hand, mainstream commodity edge devices still struggle to execute these sub-8-bit quantized networks effectively. In this paper, we propose Agile-Quant, an activation-guided quantization framework for popular Large Language Models (LLMs), and implement an end-to-end accelerator on multiple edge devices for faster inference. Considering the hardware profiling and activation analysis, we first introduce a basic activation quantization strategy to balance the trade-off of task performance and real inference speed. Then we leverage the activation-aware token pruning technique to reduce the outliers and the adverse impact on attentivity. Ultimately, we utilize the SIMD-based 4-bit multiplier and our efficient TRIP matrix multiplication to implement the accelerator for LLMs on the edge. We apply our framework on different scales of LLMs including LLaMA, OPT, and BLOOM with 4-bit or 8-bit for the activation and 4-bit for the weight quantization. Experiments show that Agile-Quant achieves simultaneous quantization of model weights and activations while maintaining task performance comparable to existing weight-only quantization methods. Moreover, in the 8- and 4-bit scenario, Agile-Quant achieves an on-device speedup of up to 2.55x compared to its FP16 counterparts across multiple edge devices, marking a pioneering advancement in this domain. Code: https://github.com/shawnricecake/agile-quant
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14775v1">gLLM: Global Balanced Pipeline Parallelism System for Distributed LLM Serving with Token Throttling</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Pipeline parallelism has emerged as a predominant approach for deploying large language models (LLMs) across distributed nodes, owing to its lower communication overhead compared to tensor parallelism. While demonstrating high throughput in request serving, pipeline parallelism often suffers from performance limitations caused by pipeline bubbles, which are primarily resulted from imbalanced computation delays across batches. Existing methods like Sarathi-Serve attempt to address this through hybrid scheduling of chunked prefill and decode tokens using a fixed token budget. However, such methods may experience significant fluctuations due to either insufficient prefill tokens or uneven distribution of decode tokens, ultimately leading to computational imbalance. To overcome these inefficiencies, we present gLLM, a globally balanced pipeline parallelism system incorporating Token Throttling to effectively mitigate the pipeline bubbles. Our Token Throttling mechanism is a fine-grained scheduling policy that independently regulates the quantities of prefill and decode tokens, thus enabling balanced computation by leveraging global information from the inference system. Specifically, for decode tokens, gLLM maintains near-consistent token count across processing batches. For prefill tokens, it dynamically adjusts batch sizes based on both total pending tokens and the memory utilization rates of key-value cache (KV cache). Furthermore, gLLM runtime adopts an asynchronous execution and message passing architecture specifically optimized for pipeline parallelism characteristics. Experimental evaluations with representative LLMs show that gLLM achieves significant performance improvements, delivering 11% to 398% higher maximum throughput compared to state-of-the-art pipeline or tensor parallelism systems, while simultaneously maintaining lower latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09723v2">AgentA/B: Automated and Scalable Web A/BTesting with Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      A/B testing experiment is a widely adopted method for evaluating UI/UX design decisions in modern web applications. Yet, traditional A/B testing remains constrained by its dependence on the large-scale and live traffic of human participants, and the long time of waiting for the testing result. Through formative interviews with six experienced industry practitioners, we identified critical bottlenecks in current A/B testing workflows. In response, we present AgentA/B, a novel system that leverages Large Language Model-based autonomous agents (LLM Agents) to automatically simulate user interaction behaviors with real webpages. AgentA/B enables scalable deployment of LLM agents with diverse personas, each capable of navigating the dynamic webpage and interactively executing multi-step interactions like search, clicking, filtering, and purchasing. In a demonstrative controlled experiment, we employ AgentA/B to simulate a between-subject A/B testing with 1,000 LLM agents Amazon.com, and compare agent behaviors with real human shopping behaviors at a scale. Our findings suggest AgentA/B can emulate human-like behavior patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18780v3">Certifying Counterfactual Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Published at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can produce biased responses that can cause representational harms. However, conventional studies are insufficient to thoroughly evaluate biases across LLM responses for different demographic groups (a.k.a. counterfactual bias), as they do not scale to large number of inputs and do not provide guarantees. Therefore, we propose the first framework, LLMCert-B that certifies LLMs for counterfactual bias on distributions of prompts. A certificate consists of high-confidence bounds on the probability of unbiased LLM responses for any set of counterfactual prompts - prompts differing by demographic groups, sampled from a distribution. We illustrate counterfactual bias certification for distributions of counterfactual prompts created by applying prefixes sampled from prefix distributions, to a given set of prompts. We consider prefix distributions consisting random token sequences, mixtures of manual jailbreaks, and perturbations of jailbreaks in LLM's embedding space. We generate non-trivial certificates for SOTA LLMs, exposing their vulnerabilities over distributions of prompts generated from computationally inexpensive prefix distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15929v3">Certifying Knowledge Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in safety-critical systems where they provide answers based on in-context information derived from knowledge bases. As LLMs are increasingly envisioned as superhuman agents, their proficiency in knowledge comprehension-extracting relevant information and reasoning over it to answer questions, a key facet of human intelligence-becomes crucial. However, existing evaluations of LLMs on knowledge comprehension are typically conducted on small test sets, but these datasets represent only a tiny fraction of the vast number of possible queries. Simple empirical evaluations on these limited test sets raises concerns about the reliability and generalizability of the results. In this work, we introduce the first specification and certification framework for knowledge comprehension in LLMs, providing formal probabilistic guarantees for reliability. Instead of a fixed dataset, we design novel specifications that mathematically represent prohibitively large probability distributions of knowledge comprehension prompts with natural noise, using knowledge graphs. From these specifications, we generate quantitative certificates that offer high-confidence, tight bounds on the probability that a given LLM correctly answers any question drawn from the specification distribution. We apply our framework to certify SOTA LLMs in two domains: precision medicine and general question-answering. Our results reveal previously unrecognized vulnerabilities in SOTA LLMs due to natural noise in the prompts. Additionally, we establish performance hierarchies with formal guarantees among the SOTA LLMs, particularly in the context of precision medicine question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15476v1">From Reviews to Dialogues: Active Synthesis for Zero-Shot LLM-based Conversational Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Conversational recommender systems (CRS) typically require extensive domain-specific conversational datasets, yet high costs, privacy concerns, and data-collection challenges severely limit their availability. Although Large Language Models (LLMs) demonstrate strong zero-shot recommendation capabilities, practical applications often favor smaller, internally managed recommender models due to scalability, interpretability, and data privacy constraints, especially in sensitive or rapidly evolving domains. However, training these smaller models effectively still demands substantial domain-specific conversational data, which remains challenging to obtain. To address these limitations, we propose an active data augmentation framework that synthesizes conversational training data by leveraging black-box LLMs guided by active learning techniques. Specifically, our method utilizes publicly available non-conversational domain data, including item metadata, user reviews, and collaborative signals, as seed inputs. By employing active learning strategies to select the most informative seed samples, our approach efficiently guides LLMs to generate synthetic, semantically coherent conversational interactions tailored explicitly to the target domain. Extensive experiments validate that conversational data generated by our proposed framework significantly improves the performance of LLM-based CRS models, effectively addressing the challenges of building CRS in no- or low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18869v3">Reimagining Memory Access for LLM Inference: Compression-Aware Memory Controller Design</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 9 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The efficiency of Large Language Model~(LLM) inference is often constrained by substantial memory bandwidth and capacity demands. Existing techniques, such as pruning, quantization, and mixture of experts/depth, reduce memory capacity and/or bandwidth consumption at the cost of slight degradation in inference quality. This paper introduces a design solution that further alleviates memory bottlenecks by enhancing the on-chip memory controller in AI accelerators to achieve two main objectives: (1) significantly reducing memory capacity and bandwidth usage through lossless block compression~(e.g., LZ4 and ZSTD) of model weights and key-value (KV) cache without compromising inference quality, and (2) enabling memory bandwidth and energy consumption to scale proportionally with context-dependent dynamic quantization. These goals are accomplished by equipping the on-chip memory controller with mechanisms to improve fine-grained bit-level accessibility and compressibility of weights and KV cache through LLM-aware configuration of in-memory placement and representation. Experimental results on publicly available LLMs demonstrate the effectiveness of this approach, showing memory footprint reductions of 25.2\% for model weights and 46.9\% for KV cache. In addition, our hardware prototype at 4\,GHz and 32 lanes (7\,nm) achieves 8\,TB/s throughput with a modest area overhead (under 3.8\,mm\(^2\)), which underscores the viability of LLM-aware memory control as a key to efficient large-scale inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15440v1">Demand for LLMs: Descriptive Evidence on Substitution, Market Expansion, and Multihoming</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      This paper documents three stylized facts about the demand for Large Language Models (LLMs) using data from OpenRouter, a prominent LLM marketplace. First, new models experience rapid initial adoption that stabilizes within weeks. Second, model releases differ substantially in whether they primarily attract new users or substitute demand from competing models. Third, multihoming, using multiple models simultaneously, is common among apps. These findings suggest significant horizontal and vertical differentiation in the LLM market, implying opportunities for providers to maintain demand and pricing power despite rapid technological advances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15439v1">Combating Toxic Language: A Review of LLM-Based Strategies for Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to software engineering (SE), where they are increasingly used in development workflows. However, their widespread use raises concerns about the presence and propagation of toxic language--harmful or offensive content that can foster exclusionary environments. This paper provides a comprehensive review of recent research on toxicity detection and mitigation, focusing on both SE-specific and general-purpose datasets. We examine annotation and preprocessing techniques, assess detection methodologies, and evaluate mitigation strategies, particularly those leveraging LLMs. Additionally, we conduct an ablation study demonstrating the effectiveness of LLM-based rewriting for reducing toxicity. By synthesizing existing work and identifying open challenges, this review highlights key areas for future research to ensure the responsible deployment of LLMs in SE and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15432v1">Feeding LLM Annotations to BERT Classifiers at Your Own Risk</a></div>
    <div class="paper-meta">
      📅 2025-04-21
    </div>
    <details class="paper-abstract">
      Using LLM-generated labels to fine-tune smaller encoder-only models for text classification has gained popularity in various settings. While this approach may be justified in simple and low-stakes applications, we conduct empirical analysis to demonstrate how the perennial curse of training on synthetic data manifests itself in this specific setup. Compared to models trained on gold labels, we observe not only the expected performance degradation in accuracy and F1 score, but also increased instability across training runs and premature performance plateaus. These findings cast doubts on the reliability of such approaches in real-world applications. We contextualize the observed phenomena through the lens of error propagation and offer several practical mitigation strategies, including entropy-based filtering and ensemble techniques. Although these heuristics offer partial relief, they do not fully resolve the inherent risks of propagating non-random errors from LLM annotations to smaller classifiers, underscoring the need for caution when applying this workflow in high-stakes text classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15424v1">LLM-Assisted Translation of Legacy FORTRAN Codes to C++: A Cross-Platform Study</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 12 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being leveraged for generating and translating scientific computer codes by both domain-experts and non-domain experts. Fortran has served as one of the go to programming languages in legacy high-performance computing (HPC) for scientific discoveries. Despite growing adoption, LLM-based code translation of legacy code-bases has not been thoroughly assessed or quantified for its usability. Here, we studied the applicability of LLM-based translation of Fortran to C++ as a step towards building an agentic-workflow using open-weight LLMs on two different computational platforms. We statistically quantified the compilation accuracy of the translated C++ codes, measured the similarity of the LLM translated code to the human translated C++ code, and statistically quantified the output similarity of the Fortran to C++ translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01005v2">FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted by MLSys 2025, code available at http://github.com/flashinfer-ai/flashinfer
    </div>
    <details class="paper-abstract">
      Transformers, driven by attention mechanisms, form the foundation of large language models (LLMs). As these models scale up, efficient GPU attention kernels become essential for high-throughput and low-latency inference. Diverse LLM applications demand flexible and high-performance attention solutions. We present FlashInfer: a customizable and efficient attention engine for LLM serving. FlashInfer tackles KV-cache storage heterogeneity using block-sparse format and composable formats to optimize memory access and reduce redundancy. It also offers a customizable attention template, enabling adaptation to various settings through Just-In-Time (JIT) compilation. Additionally, FlashInfer's load-balanced scheduling algorithm adjusts to dynamism of user requests while maintaining compatibility with CUDAGraph which requires static configuration. FlashInfer have been integrated into leading LLM serving frameworks like SGLang, vLLM and MLC-Engine. Comprehensive kernel-level and end-to-end evaluations demonstrate FlashInfer's ability to significantly boost kernel performance across diverse inference scenarios: compared to state-of-the-art LLM serving solutions, FlashInfer achieve 29-69% inter-token-latency reduction compared to compiler backends for LLM serving benchmark, 28-30% latency reduction for long-context inference, and 13-17% speedup for LLM serving with parallel generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15392v1">Tell Me What You Know About Sexism: Expert-LLM Interaction Strategies and Co-Created Definitions for Zero-Shot Sexism Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 Accepted and published at Findings of NAACL 2025: cite published version whenever possible
    </div>
    <details class="paper-abstract">
      This paper investigates hybrid intelligence and collaboration between researchers of sexism and Large Language Models (LLMs), with a four-component pipeline. First, nine sexism researchers answer questions about their knowledge of sexism and of LLMs. They then participate in two interactive experiments involving an LLM (GPT3.5). The first experiment has experts assessing the model's knowledge about sexism and suitability for use in research. The second experiment tasks them with creating three different definitions of sexism: an expert-written definition, an LLM-written one, and a co-created definition. Lastly, zero-shot classification experiments use the three definitions from each expert in a prompt template for sexism detection, evaluating GPT4o on 2.500 texts sampled from five sexism benchmarks. We then analyze the resulting 67.500 classification decisions. The LLM interactions lead to longer and more complex definitions of sexism. Expert-written definitions on average perform poorly compared to LLM-generated definitions. However, some experts do improve classification performance with their co-created definitions of sexism, also experts who are inexperienced in using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15364v1">KeDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 8 pages, 14 figures
    </div>
    <details class="paper-abstract">
      In this work, we demonstrate that distinctive keys during LLM inference tend to have high attention scores. We explore this phenomenon and propose KeyDiff, a training-free KV cache eviction method based on key similarity. This method facilitates the deployment of LLM-based application requiring long input prompts in resource-constrained environments with limited memory and compute budgets. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We demonstrate that KeyDiff computes the optimal solution to a KV cache selection problem that maximizes key diversity, providing a theoretical understanding of KeyDiff. Notably,KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. We demonstrate the effectiveness of KeyDiff across diverse tasks and models, illustrating a performance gap of less than 0.04\% with 8K cache budget ($\sim$ 23\% KV cache reduction) from the non-evicting baseline on the LongBench benchmark for Llama 3.1-8B and Llama 3.2-3B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13055v2">LAMD: Context-driven Android Malware Detection and Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-21
      | 💬 accepted by 2025 46th IEEE Symposium on Security and Privacy Workshops (SPW)
    </div>
    <details class="paper-abstract">
      The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17116v2">Star Attention: Efficient LLM Inference over Long Sequences</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 Code: https://github.com/NVIDIA/Star-Attention
    </div>
    <details class="paper-abstract">
      Inference with Transformer-based Large Language Models (LLMs) on long sequences is both costly and slow due to the quadratic complexity of the self-attention mechanism. We introduce Star Attention, a two-phase block-sparse approximation that improves computational efficiency by sharding attention across multiple hosts while minimizing communication overhead. In the first phase, the context is processed using blockwise-local attention across hosts, in parallel. In the second phase, query and response tokens attend to all prior cached tokens through sequence-global attention. Star Attention integrates seamlessly with most Transformer-based LLMs trained with global attention, reducing memory requirements and inference time by up to 11x while preserving 97-100% of accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14716v1">Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as proxies for human labelers in both training (Reinforcement Learning from AI Feedback) and large-scale response evaluation (LLM-as-a-judge). Alignment and evaluation are critical components in the development of reliable LLMs, and the choice of feedback protocol plays a central role in both but remains understudied. In this work, we show that the choice of feedback protocol (absolute scores versus relative preferences) can significantly affect evaluation reliability and induce systematic biases. In particular, we show that pairwise evaluation protocols are more vulnerable to distracted evaluation. Generator models can exploit spurious attributes (or distractor features) favored by the LLM judge, resulting in inflated scores for lower-quality outputs and misleading training signals. We find that absolute scoring is more robust to such manipulation, producing judgments that better reflect response quality and are less influenced by distractor features. Our results demonstrate that generator models can flip preferences by embedding distractor features, skewing LLM-as-a-judge comparisons and leading to inaccurate conclusions about model quality in benchmark evaluations. Pairwise preferences flip in about 35% of the cases, compared to only 9% for absolute scores. We offer recommendations for choosing feedback protocols based on dataset characteristics and evaluation objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03477v2">CrowdGenUI: Aligning LLM-Based UI Generation with Crowdsourced User Preferences</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential across various design domains, including user interface (UI) generation. However, current LLMs for UI generation tend to offer generic solutions that lack a nuanced understanding of task context and user preferences. We present CrowdGenUI, a framework that enhances LLM-based UI generation with crowdsourced user preferences. This framework addresses the limitations by guiding LLM reasoning with real user preferences, enabling the generation of UI widgets that reflect user needs and task-specific requirements. We evaluate our framework in the image editing domain by collecting a library of 720 user preferences from 50 participants, covering preferences such as predictability, efficiency, and explorability of various UI widgets. A user study (N=78) demonstrates that UIs generated with our preference-guided framework can better match user intentions compared to those generated by LLMs alone, highlighting the effectiveness of our proposed framework. We further discuss the study findings and present insights for future research on LLM-based user-centered UI generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14681v1">An LLM-enabled Multi-Agent Autonomous Mechatronics Design Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 Accepted by CVPR 2025 Workshop
    </div>
    <details class="paper-abstract">
      Existing LLM-enabled multi-agent frameworks are predominantly limited to digital or simulated environments and confined to narrowly focused knowledge domain, constraining their applicability to complex engineering tasks that require the design of physical embodiment, cross-disciplinary integration, and constraint-aware reasoning. This work proposes a multi-agent autonomous mechatronics design framework, integrating expertise across mechanical design, optimization, electronics, and software engineering to autonomously generate functional prototypes with minimal direct human design input. Operating primarily through a language-driven workflow, the framework incorporates structured human feedback to ensure robust performance under real-world constraints. To validate its capabilities, the framework is applied to a real-world challenge involving autonomous water-quality monitoring and sampling, where traditional methods are labor-intensive and ecologically disruptive. Leveraging the proposed system, a fully functional autonomous vessel was developed with optimized propulsion, cost-effective electronics, and advanced control. The design process was carried out by specialized agents, including a high-level planning agent responsible for problem abstraction and dedicated agents for structural, electronics, control, and software development. This approach demonstrates the potential of LLM-based multi-agent systems to automate real-world engineering workflows and reduce reliance on extensive domain expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14657v1">A Case Study Exploring the Current Landscape of Synthetic Medical Record Generation with Commercial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 Accepted at the Conference of Health, Inference, Learning (CHIL 2025) in Berkeley, CA. To appear in PMLR later in 2025
    </div>
    <details class="paper-abstract">
      Synthetic Electronic Health Records (EHRs) offer a valuable opportunity to create privacy preserving and harmonized structured data, supporting numerous applications in healthcare. Key benefits of synthetic data include precise control over the data schema, improved fairness and representation of patient populations, and the ability to share datasets without concerns about compromising real individuals privacy. Consequently, the AI community has increasingly turned to Large Language Models (LLMs) to generate synthetic data across various domains. However, a significant challenge in healthcare is ensuring that synthetic health records reliably generalize across different hospitals, a long standing issue in the field. In this work, we evaluate the current state of commercial LLMs for generating synthetic data and investigate multiple aspects of the generation process to identify areas where these models excel and where they fall short. Our main finding from this work is that while LLMs can reliably generate synthetic health records for smaller subsets of features, they struggle to preserve realistic distributions and correlations as the dimensionality of the data increases, ultimately limiting their ability to generalize across diverse hospital settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14655v1">LeetCodeDataset: A Temporal Dataset for Robust Evaluation and Efficient Training of Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      We introduce LeetCodeDataset, a high-quality benchmark for evaluating and training code-generation models, addressing two key challenges in LLM research: the lack of reasoning-focused coding benchmarks and self-contained training testbeds. By curating LeetCode Python problems with rich metadata, broad coverage, 100+ test cases per problem, and temporal splits (pre/post July 2024), our dataset enables contamination-free evaluation and efficient supervised fine-tuning (SFT). Experiments show reasoning models significantly outperform non-reasoning counterparts, while SFT with only 2.6K model-generated solutions achieves performance comparable to 110K-sample counterparts. The dataset and evaluation framework are available on Hugging Face and Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14650v1">A Framework for Benchmarking and Aligning Task-Planning Safety in LLM-Based Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit substantial promise in enhancing task-planning capabilities within embodied agents due to their advanced reasoning and comprehension. However, the systemic safety of these agents remains an underexplored frontier. In this study, we present Safe-BeAl, an integrated framework for the measurement (SafePlan-Bench) and alignment (Safe-Align) of LLM-based embodied agents' behaviors. SafePlan-Bench establishes a comprehensive benchmark for evaluating task-planning safety, encompassing 2,027 daily tasks and corresponding environments distributed across 8 distinct hazard categories (e.g., Fire Hazard). Our empirical analysis reveals that even in the absence of adversarial inputs or malicious intent, LLM-based agents can exhibit unsafe behaviors. To mitigate these hazards, we propose Safe-Align, a method designed to integrate physical-world safety knowledge into LLM-based embodied agents while maintaining task-specific performance. Experiments across a variety of settings demonstrate that Safe-BeAl provides comprehensive safety validation, improving safety by 8.55 - 15.22%, compared to embodied agents based on GPT-4, while ensuring successful task completion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14641v1">HLSTester: Efficient Testing of Behavioral Discrepancies with LLMs for High-Level Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      In high-level synthesis (HLS), C/C++ programs with synthesis directives are used to generate circuits for FPGA implementations. However, hardware-specific and platform-dependent characteristics in these implementations can introduce behavioral discrepancies between the original C/C++ programs and the circuits after high-level synthesis. Existing methods for testing behavioral discrepancies in HLS are still immature, and the testing workflow requires significant human efforts. To address this challenge, we propose HLSTester, a large language model (LLM) aided testing framework that efficiently detects behavioral discrepancies in HLS. To mitigate hallucinations in LLMs and enhance prompt quality, the testbenches for original C/C++ programs are leveraged to guide LLMs in generating HLS-compatible testbenches, effectively eliminating certain traditional C/C++ constructs that are incompatible with HLS tools. Key variables are pinpointed through a backward slicing technique in both C/C++ and HLS programs to monitor their runtime spectra, enabling an in-depth analysis of the discrepancy symptoms. To reduce test time, a testing input generation mechanism is introduced to integrate dynamic mutation with insights from an LLM-based progressive reasoning chain. In addition, repetitive hardware testing is skipped by a redundancy-aware filtering technique for the generated test inputs. Experimental results demonstrate that the proposed LLM-aided testing framework significantly accelerates the testing workflow while achieving higher testbench simulation pass rates compared with the traditional method and the direct use of LLMs on the same HLS programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14640v1">Risk Assessment Framework for Code LLMs via Leveraging Internal States</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 To appear in the 33rd ACM International Conference on the Foundations of Software Engineering (FSE Companion'25 Industry Track), June 23-28, 2025, Trondheim, Norway. This work was supported by Fujitsu Limited
    </div>
    <details class="paper-abstract">
      The pre-training paradigm plays a key role in the success of Large Language Models (LLMs), which have been recognized as one of the most significant advancements of AI recently. Building on these breakthroughs, code LLMs with advanced coding capabilities bring huge impacts on software engineering, showing the tendency to become an essential part of developers' daily routines. However, the current code LLMs still face serious challenges related to trustworthiness, as they can generate incorrect, insecure, or unreliable code. Recent exploratory studies find that it can be promising to detect such risky outputs by analyzing LLMs' internal states, akin to how the human brain unconsciously recognizes its own mistakes. Yet, most of these approaches are limited to narrow sub-domains of LLM operations and fall short of achieving industry-level scalability and practicability. To address these challenges, in this paper, we propose PtTrust, a two-stage risk assessment framework for code LLM based on internal state pre-training, designed to integrate seamlessly with the existing infrastructure of software companies. The core idea is that the risk assessment framework could also undergo a pre-training process similar to LLMs. Specifically, PtTrust first performs unsupervised pre-training on large-scale unlabeled source code to learn general representations of LLM states. Then, it uses a small, labeled dataset to train a risk predictor. We demonstrate the effectiveness of PtTrust through fine-grained, code line-level risk assessment and demonstrate that it generalizes across tasks and different programming languages. Further experiments also reveal that PtTrust provides highly intuitive and interpretable features, fostering greater user trust. We believe PtTrust makes a promising step toward scalable and trustworthy assurance for code LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14633v1">Harnessing Generative LLMs for Enhanced Financial Event Entity Extraction Performance</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Financial event entity extraction is a crucial task for analyzing market dynamics and building financial knowledge graphs, yet it presents significant challenges due to the specialized language and complex structures in financial texts. Traditional approaches often rely on sequence labeling models, which can struggle with long-range dependencies and the inherent complexity of extracting multiple, potentially overlapping entities. Motivated by the advanced language understanding and generative capabilities of Large Language Models (LLMs), we propose a novel method that reframes financial event entity extraction as a text-to-structured-output generation task. Our approach involves fine-tuning a pre-trained LLM using Parameter-Efficient Fine-Tuning (PEFT) to directly generate a structured representation, such as a JSON object, containing the extracted entities and their precise character spans from the input text. We evaluate our method on the challenging CCKS 2019 Financial Event Entity Extraction dataset, comparing its performance against strong sequence labeling baselines, including SEBERTNets and sebertNets. Experimental results demonstrate that our generative LLM method achieves a new state-of-the-art F1 score on this benchmark, significantly outperforming previous methods. Through detailed quantitative analysis across event types, entity types, and instance complexity, as well as human evaluation, we show that our approach is more effective at handling the nuances of financial text and extracting high-quality entities. This work validates the potential of applying generative LLMs directly to complex, domain-specific information extraction tasks requiring structured output.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04994v2">Following the Whispers of Values: Unraveling Neural Mechanisms Behind Value-Oriented Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of large language models (LLMs), they can present unintended biases and harmful behaviors driven by encoded values, emphasizing the urgent need to understand the value mechanisms behind them. However, current research primarily evaluates these values through external responses with a focus on AI safety, lacking interpretability and failing to assess social values in real-world contexts. In this paper, we propose a novel framework called ValueExploration, which aims to explore the behavior-driven mechanisms of National Social Values within LLMs at the neuron level. As a case study, we focus on Chinese Social Values and first construct C-voice, a large-scale bilingual benchmark for identifying and evaluating Chinese Social Values in LLMs. By leveraging C-voice, we then identify and locate the neurons responsible for encoding these values according to activation difference. Finally, by deactivating these neurons, we analyze shifts in model behavior, uncovering the internal mechanism by which values influence LLM decision-making. Extensive experiments on four representative LLMs validate the efficacy of our framework. The benchmark and code will be available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10050v3">Emotional Strain and Frustration in LLM Interactions in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 Accepted in EASE'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into various daily tasks in Software Engineering such as coding and requirement elicitation. Despite their various capabilities and constant use, some interactions can lead to unexpected challenges (e.g. hallucinations or verbose answers) and, in turn, cause emotions that develop into frustration. Frustration can negatively impact engineers' productivity and well-being if they escalate into stress and burnout. In this paper, we assess the impact of LLM interactions on software engineers' emotional responses, specifically strains, and identify common causes of frustration when interacting with LLMs at work. Based on 62 survey responses from software engineers in industry and academia across various companies and universities, we found that a majority of our respondents experience frustrations or other related emotions regardless of the nature of their work. Additionally, our results showed that frustration mainly stemmed from issues with correctness and less critical issues such as adaptability to context or specific format. While such issues may not cause frustration in general, artefacts that do not follow certain preferences, standards, or best practices can make the output unusable without extensive modification, causing frustration over time. In addition to the frustration triggers, our study offers guidelines to improve the software engineers' experience, aiming to minimise long-term consequences on mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14557v1">Enhancing LLM-based Quantum Code Generation with Multi-Agent Optimization and Quantum Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 Paper accepted by DAC'25
    </div>
    <details class="paper-abstract">
      Multi-agent frameworks with Large Language Models (LLMs) have become promising tools for generating general-purpose programming languages using test-driven development, allowing developers to create more accurate and robust code. However, their potential has not been fully unleashed for domain-specific programming languages, where specific domain exhibits unique optimization opportunities for customized improvement. In this paper, we take the first step in exploring multi-agent code generation for quantum programs. By identifying the unique optimizations in quantum designs such as quantum error correction, we introduce a novel multi-agent framework tailored to generating accurate, fault-tolerant quantum code. Each agent in the framework focuses on distinct optimizations, iteratively refining the code using a semantic analyzer with multi-pass inference, alongside an error correction code decoder. We also examine the effectiveness of inference-time techniques, like Chain-of-Thought (CoT) and Retrieval-Augmented Generation (RAG) in the context of quantum programming, uncovering observations that are different from general-purpose code generation. To evaluate our approach, we develop a test suite to measure the impact each optimization has on the accuracy of the generated code. Our findings indicate that techniques such as structured CoT significantly improve the generation of quantum algorithms by up to 50%. In contrast, we have also found that certain techniques such as RAG show limited improvement, yielding an accuracy increase of only 4%. Moreover, we showcase examples of AI-assisted quantum error prediction and correction, demonstrating the effectiveness of our multi-agent framework in reducing the quantum errors of generated quantum programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14556v1">LLM-Enabled In-Context Learning for Data Collection Scheduling in UAV-assisted Sensor Networks</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 8 pages, 7 figures,
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) are increasingly being used in various private and commercial applications, e.g. traffic control, package delivery, and Search and Rescue (SAR) operations. Machine Learning (ML) methods used in UAV-assisted Sensor Networks (UASNETs) and especially in Deep Reinforcement Learning (DRL) face challenges such as complex and lengthy model training, gaps between simulation and reality, and low sample efficiency, which conflict with the urgency of emergencies such as SAR operations. This paper proposes In-Context Learning (ICL)-based Data Collection Scheduling (ICLDC) scheme, as an alternative to DRL in emergencies. The UAV collects and transmits logged sensory data, to an LLM, to generate a task description in natural language, from which it obtains a data collection schedule to be executed by the UAV. The system continuously adapts by adding feedback to task descriptions and utilizing feedback for future decisions. This method is tested against jailbreaking attacks, where task description is manipulated to undermine network performance, highlighting the vulnerability of LLMs to such attacks. The proposed ICLDC outperforms the Maximum Channel Gain by reducing cumulative packet loss by approximately 56\%. ICLDC presents a promising direction for intelligent scheduling and control in UAV-assisted data collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.01519v4">Exploring the Frontiers of LLMs in Psychological Applications: A Comprehensive Review</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      This paper explores the frontiers of large language models (LLMs) in psychology applications. Psychology has undergone several theoretical changes, and the current use of Artificial Intelligence (AI) and Machine Learning, particularly LLMs, promises to open up new research directions. We provide a detailed exploration of how LLMs like ChatGPT are transforming psychological research. It discusses the impact of LLMs across various branches of psychology, including cognitive and behavioral, clinical and counseling, educational and developmental, and social and cultural psychology, highlighting their potential to simulate aspects of human cognition and behavior. The paper delves into the capabilities of these models to emulate human-like text generation, offering innovative tools for literature review, hypothesis generation, experimental design, experimental subjects, data analysis, academic writing, and peer review in psychology. While LLMs are essential in advancing research methodologies in psychology, the paper also cautions about their technical and ethical challenges. There are issues like data privacy, the ethical implications of using LLMs in psychological research, and the need for a deeper understanding of these models' limitations. Researchers should responsibly use LLMs in psychological studies, adhering to ethical standards and considering the potential consequences of deploying these technologies in sensitive areas. Overall, the article provides a comprehensive overview of the current state of LLMs in psychology, exploring potential benefits and challenges. It serves as a call to action for researchers to leverage LLMs' advantages responsibly while addressing associated risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14520v1">Meta-Thinking in LLMs via Multi-Agent Reinforcement Learning: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 Submitted to IEEE Transactions on Artificial Intelligence
    </div>
    <details class="paper-abstract">
      This survey explores the development of meta-thinking capabilities in Large Language Models (LLMs) from a Multi-Agent Reinforcement Learning (MARL) perspective. Meta-thinking self-reflection, assessment, and control of thinking processes is an important next step in enhancing LLM reliability, flexibility, and performance, particularly for complex or high-stakes tasks. The survey begins by analyzing current LLM limitations, such as hallucinations and the lack of internal self-assessment mechanisms. It then talks about newer methods, including RL from human feedback (RLHF), self-distillation, and chain-of-thought prompting, and each of their limitations. The crux of the survey is to talk about how multi-agent architectures, namely supervisor-agent hierarchies, agent debates, and theory of mind frameworks, can emulate human-like introspective behavior and enhance LLM robustness. By exploring reward mechanisms, self-play, and continuous learning methods in MARL, this survey gives a comprehensive roadmap to building introspective, adaptive, and trustworthy LLMs. Evaluation metrics, datasets, and future research avenues, including neuroscience-inspired architectures and hybrid symbolic reasoning, are also discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14519v1">SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Pipeline Parallelism (PP) serves as a crucial technique for training Large Language Models (LLMs), owing to its capability to alleviate memory pressure from model states with relatively low communication overhead. However, in long-context scenarios, existing pipeline parallelism methods fail to address the substantial activation memory pressure, primarily due to the peak memory consumption resulting from the accumulation of activations across multiple microbatches. Moreover, these approaches inevitably introduce considerable pipeline bubbles, further hindering efficiency. To tackle these challenges, we propose SlimPipe, a novel approach to fine-grained pipeline parallelism that employs uniform sequence slicing coupled with one-forward-one-backward (1F1B) schedule. It reduces the accumulated activations from several microbatches to just one, which is split into several slices. Although the slices are evenly partitioned, the computation cost is not equal across slices due to causal attention. We develop a sophisticated workload redistribution technique to address this load imbalance. SlimPipe achieves (1) near-zero memory overhead and (2) minimal pipeline bubbles simultaneously. The effectiveness of SlimPipe has been proven by thorough testing with diverse model architectures, context window sizes, and SlimPipe-specific configurations. For example, on the Llama 70B model, compared to state-of-the-art methods, SlimPipe significantly boosts the Model FLOPs Utilization (MFU) to up to $1.57\times$ for a context length of 512K. More notably, for a context length of 2048K, it maintains over 45% utilization on 256 NVIDIA Hopper 80GB GPUs, while other approaches either suffer significant performance drops or fail entirely due to memory constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08858v2">Decoding Secret Memorization in Code LLMs Through Token-Level Characterization</a></div>
    <div class="paper-meta">
      📅 2025-04-20
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Code Large Language Models (LLMs) have demonstrated remarkable capabilities in generating, understanding, and manipulating programming code. However, their training process inadvertently leads to the memorization of sensitive information, posing severe privacy risks. Existing studies on memorization in LLMs primarily rely on prompt engineering techniques, which suffer from limitations such as widespread hallucination and inefficient extraction of the target sensitive information. In this paper, we present a novel approach to characterize real and fake secrets generated by Code LLMs based on token probabilities. We identify four key characteristics that differentiate genuine secrets from hallucinated ones, providing insights into distinguishing real and fake secrets. To overcome the limitations of existing works, we propose DESEC, a two-stage method that leverages token-level features derived from the identified characteristics to guide the token decoding process. DESEC consists of constructing an offline token scoring model using a proxy Code LLM and employing the scoring model to guide the decoding process by reassigning token likelihoods. Through extensive experiments on four state-of-the-art Code LLMs using a diverse dataset, we demonstrate the superior performance of DESEC in achieving a higher plausible rate and extracting more real secrets compared to existing baselines. Our findings highlight the effectiveness of our token-level approach in enabling an extensive assessment of the privacy leakage risks associated with Code LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14492v1">FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to capturing biases from training corpus, leading to potential negative social impacts. Existing prompt-based debiasing methods exhibit instability due to their sensitivity to prompt changes, while fine-tuning-based techniques incur substantial computational overhead and catastrophic forgetting. In this paper, we propose FairSteer, a novel inference-time debiasing framework without requiring customized prompt design or model retraining. Motivated by the linear representation hypothesis, our preliminary investigation demonstrates that fairness-related features can be encoded into separable directions in the hidden activation space. FairSteer operates in three steps: biased activation detection, debiasing steering vector (DSV) computation, and dynamic activation steering. Specifically, it first trains a lightweight linear classifier to detect bias signatures in activations, and then computes DSVs as intervention directions derived from small contrastive prompt pairs. Subsequently, it performs debiasing by adjusting activations with DSVs in the inference stage. Comprehensive evaluation with six LLMs demonstrates the superiority of FairSteer across question-answering, counterfactual input evaluation and open-ended text generation tasks. Code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14489v1">Optimizing SLO-oriented LLM Serving with PD-Multiplexing</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Modern LLM services demand high throughput and stringent SLO guarantees across two distinct inference phases-prefill and decode-and complex multi-turn workflows. However, current systems face a fundamental tradeoff: out-of-place compute partition enables per-phase SLO attainment, while in-place memory sharing maximizes throughput via KV cache reuse. Moreover, existing in-place compute partition also encounters low utilization and high overhead due to phase-coupling design. We present Yoda, a new LLM serving framework that resolves this tension via PD multiplexing, enabling in-place and phase-decoupled compute partition. Yoda leverages low-level GPU partitioning techniques to multiplex prefill and decode phases spatially and adaptively on shared GPUs, while preserving in-place memory sharing. To fully leverage the multiplexing capability, Yoda introduces an adaptive gang scheduling mechanism, a contention-free modeling method, and a SLO-aware dispatching policy. Evaluation shows that Yoda achieves an average $5.1\times$ throughput improvement (up to $17.5\times$) over state-of-the-art baselines, while consistently meeting SLO targets under complex LLM workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14439v1">LoRe: Personalizing LLMs via Low-Rank Reward Modeling</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) to accommodate diverse user preferences is essential for enhancing alignment and user satisfaction. Traditional reinforcement learning from human feedback (RLHF) approaches often rely on monolithic value representations, limiting their ability to adapt to individual preferences. We introduce a novel framework that leverages low-rank preference modeling to efficiently learn and generalize user-specific reward functions. By representing reward functions in a low-dimensional subspace and modeling individual preferences as weighted combinations of shared basis functions, our approach avoids rigid user categorization while enabling scalability and few-shot adaptation. We validate our method on multiple preference datasets, demonstrating superior generalization to unseen users and improved accuracy in preference prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14432v1">ResNetVLLM -- Multi-modal Vision LLM for the Video Understanding Task</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      In this paper, we introduce ResNetVLLM (ResNet Vision LLM), a novel cross-modal framework for zero-shot video understanding that integrates a ResNet-based visual encoder with a Large Language Model (LLM. ResNetVLLM addresses the challenges associated with zero-shot video models by avoiding reliance on pre-trained video understanding models and instead employing a non-pretrained ResNet to extract visual features. This design ensures the model learns visual and semantic representations within a unified architecture, enhancing its ability to generate accurate and contextually relevant textual descriptions from video inputs. Our experimental results demonstrate that ResNetVLLM achieves state-of-the-art performance in zero-shot video understanding (ZSVU) on several benchmarks, including MSRVTT-QA, MSVD-QA, TGIF-QA FrameQA, and ActivityNet-QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09300v3">Nudging: Inference-time Alignment of LLMs via Guided Decoding</a></div>
    <div class="paper-meta">
      📅 2025-04-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require alignment to effectively and safely follow user instructions. This process necessitates training an aligned version for every base model, resulting in significant computational overhead. In this work, we propose nudging, a simple, plug-and-play, and training-free algorithm that aligns any base model at inference time using a small aligned model. Nudging is motivated by recent findings that alignment primarily alters the model's behavior on a small subset of stylistic tokens (e.g., discourse markers). We find that base models are significantly more uncertain when generating these tokens. Building on this insight, nudging employs a small aligned model to generate nudging tokens to guide the base model's output during decoding when the base model's uncertainty is high. We evaluate nudging across 3 model families on a diverse range of open-instruction tasks. Without any training, nudging a large base model with a 7x-14x smaller aligned model achieves zero-shot performance comparable to, and sometimes surpassing, that of large aligned models. By operating at the token level, nudging enables off-the-shelf collaboration between model families. For instance, nudging Gemma-2-27b with Llama-2-7b-chat outperforms Llama-2-70b-chat on various tasks. Overall, our work offers a modular and cost-efficient solution to LLM alignment. Our project website: https://fywalter.github.io/nudging/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v3">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14401v1">LLM-Driven Usefulness Judgment for Web Search Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Evaluation is fundamental in optimizing search experiences and supporting diverse user intents in Information Retrieval (IR). Traditional search evaluation methods primarily rely on relevance labels, which assess how well retrieved documents match a user's query. However, relevance alone fails to capture a search system's effectiveness in helping users achieve their search goals, making usefulness a critical evaluation criterion. In this paper, we explore an alternative approach: LLM-generated usefulness labels, which incorporate both implicit and explicit user behavior signals to evaluate document usefulness. We propose Task-aware Rubric-based Usefulness Evaluation (TRUE), a rubric-driven evaluation method that employs iterative sampling and reasoning to model complex search behavior patterns. Our findings show that (i) LLMs can generate moderate usefulness labels by leveraging comprehensive search session history incorporating personalization and contextual understanding, and (ii) fine-tuned LLMs improve usefulness judgments when provided with structured search session contexts. Additionally, we examine whether LLMs can distinguish between relevance and usefulness, particularly in cases where this divergence impacts search success. We also conduct an ablation study to identify key metrics for accurate usefulness label generation, optimizing for token efficiency and cost-effectiveness in real-world applications. This study advances LLM-based usefulness evaluation by refining key user metrics, exploring LLM-generated label reliability, and ensuring feasibility for large-scale search systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09284v2">SparQLe: Speech Queries to Text Translation Through LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      With the growing influence of Large Language Models (LLMs), there is increasing interest in integrating speech representations with them to enable more seamless multi-modal processing and speech understanding. This study introduces a novel approach that leverages self-supervised speech representations in combination with instruction-tuned LLMs for speech-to-text translation. The proposed approach leverages a modality adapter to align extracted speech features with instruction-tuned LLMs using English-language data. Our experiments demonstrate that this method effectively preserves the semantic content of the input speech and serves as an effective bridge between self-supervised speech models and instruction-tuned LLMs, offering a promising solution for various speech understanding applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02051v2">Self-Resource Allocation in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      With the development of LLMs as agents, there is a growing interest in connecting multiple agents into multi-agent systems to solve tasks concurrently, focusing on their role in task assignment and coordination. This paper explores how LLMs can effectively allocate computational tasks among multiple agents, considering factors such as cost, efficiency, and performance. In this work, we address key questions, including the effectiveness of LLMs as orchestrators and planners, comparing their effectiveness in task assignment and coordination. Our experiments demonstrate that LLMs can achieve high validity and accuracy in resource allocation tasks. We find that the planner method outperforms the orchestrator method in handling concurrent actions, resulting in improved efficiency and better utilization of agents. Additionally, we show that providing explicit information about worker capabilities enhances the allocation strategies of planners, particularly when dealing with suboptimal workers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14365v1">Accelerating LLM Inference with Flexible N:M Sparsity via A Fully Digital Compute-in-Memory Accelerator</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) pruning with fixed N:M structured sparsity significantly limits the expressivity of the sparse model, yielding sub-optimal performance. In contrast, supporting multiple N:M patterns to provide sparse representational freedom introduces costly overhead in hardware. To address these challenges for LLMs, we first present a flexible layer-wise outlier-density-aware N:M sparsity (FLOW) selection method. FLOW enables the identification of optimal layer-wise N and M values (from a given range) by simultaneously accounting for the presence and distribution of outliers, allowing a higher degree of representational freedom. To deploy sparse models with such N:M flexibility, we then introduce a flexible, low-overhead digital compute-in-memory architecture (FlexCiM). FlexCiM supports diverse sparsity patterns by partitioning a digital CiM (DCiM) macro into smaller sub-macros, which are adaptively aggregated and disaggregated through distribution and merging mechanisms for different N and M values. Extensive experiments on both transformer-based and recurrence-based state space foundation models (SSMs) demonstrate that FLOW outperforms existing alternatives with an accuracy improvement of up to 36%, while FlexCiM achieves up to 1.75x lower inference latency and 1.5x lower energy consumption compared to existing sparse accelerators. Code is available at: https://github.com/FLOW-open-project/FLOW
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14363v1">Improving RL Exploration for LLM Reasoning through Retrospective Replay</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has increasingly become a pivotal technique in the post-training of large language models (LLMs). The effective exploration of the output space is essential for the success of RL. We observe that for complex problems, during the early stages of training, the model exhibits strong exploratory capabilities and can identify promising solution ideas. However, its limited capability at this stage prevents it from successfully solving these problems. The early suppression of these potentially valuable solution ideas by the policy gradient hinders the model's ability to revisit and re-explore these ideas later. Consequently, although the LLM's capabilities improve in the later stages of training, it still struggles to effectively address these complex problems. To address this exploration issue, we propose a novel algorithm named Retrospective Replay-based Reinforcement Learning (RRL), which introduces a dynamic replay mechanism throughout the training process. RRL enables the model to revisit promising states identified in the early stages, thereby improving its efficiency and effectiveness in exploration. To evaluate the effectiveness of RRL, we conduct extensive experiments on complex reasoning tasks, including mathematical reasoning and code generation, and general dialogue tasks. The results indicate that RRL maintains high exploration efficiency throughout the training period, significantly enhancing the effectiveness of RL in optimizing LLMs for complicated reasoning tasks. Moreover, it also improves the performance of RLHF, making the model both safer and more helpful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19456v2">SSFF: Investigating LLM Predictive Capabilities for Startup Success through a Multi-Agent Framework with Enhanced Explainability and Performance</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 For relevant code: https://github.com/Xisen-Wang/Startup-Success-Forecasting-Framework
    </div>
    <details class="paper-abstract">
      LLM based agents have recently demonstrated strong potential in automating complex tasks, yet accurately predicting startup success remains an open challenge with few benchmarks and tailored frameworks. To address these limitations, we propose the Startup Success Forecasting Framework, an autonomous system that emulates the reasoning of venture capital analysts through a multi agent collaboration model. Our framework integrates traditional machine learning methods such as random forests and neural networks within a retrieval augmented generation framework composed of three interconnected modules: a prediction block, an analysis block, and an external knowledge block. We evaluate our framework and identify three main findings. First, by leveraging founder segmentation, startups led by L5 founders are 3.79 times more likely to succeed than those led by L1 founders. Second, baseline large language models consistently overpredict startup success and struggle under realistic class imbalances largely due to overreliance on founder claims. Third, our framework significantly enhances prediction accuracy, yielding a 108.3 percent relative improvement over GPT 4o mini and a 30.8 percent relative improvement over GPT 4o. These results demonstrate the value of a multi agent approach combined with discriminative machine learning in mitigating the limitations of standard large language model based prediction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14350v1">Time Up! An Empirical Study of LLM Reasoning Ability Under Output Length Constraint</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the remarkable potential of Large Language Models (LLMs) in test-time scaling. By making the models think before answering, they are able to achieve much higher accuracy with extra inference computation. However, in many real-world scenarios, models are used under time constraints, where an answer should be given to the user within a certain output length. It is unclear whether and how the reasoning abilities of LLMs remain effective under such constraints. We take a first look at this problem by conducting an in-depth empirical study. Specifically, we test more than 25 LLMs on common reasoning datasets under a wide range of output length budgets, and we analyze the correlation between the inference accuracy and various properties including model type, model size, prompt style, etc. We also consider the mappings between the token budgets and the actual on-device latency budgets. The results have demonstrated several interesting findings regarding the budget-aware LLM reasoning that differ from the unconstrained situation, e.g. the optimal choices of model sizes and prompts change under different budgets. These findings offer practical guidance for users to deploy LLMs under real-world latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14345v1">Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 Presented at the ICLR 2025 Workshop on Financial AI (https://sites.google.com/view/financialaiiclr25/home)
    </div>
    <details class="paper-abstract">
      Portfolio optimization faces challenges due to the sensitivity in traditional mean-variance models. The Black-Litterman model mitigates this by integrating investor views, but defining these views remains difficult. This study explores the integration of large language models (LLMs) generated views into portfolio optimization using the Black-Litterman framework. Our method leverages LLMs to estimate expected stock returns from historical prices and company metadata, incorporating uncertainty through the variance in predictions. We conduct a backtest of the LLM-optimized portfolios from June 2024 to February 2025, rebalancing biweekly using the previous two weeks of price data. As baselines, we compare against the S&P 500, an equal-weighted portfolio, and a traditional mean-variance optimized portfolio constructed using the same set of stocks. Empirical results suggest that different LLMs exhibit varying levels of predictive optimism and confidence stability, which impact portfolio performance. The source code and data are available at https://github.com/youngandbin/LLM-MVO-BLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05216v2">Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14286v1">SRPO: A Cross-Domain Implementation of Large-Scale Reinforcement Learning on LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Recent advances of reasoning models, exemplified by OpenAI's o1 and DeepSeek's R1, highlight the significant potential of Reinforcement Learning (RL) to enhance the reasoning capabilities of Large Language Models (LLMs). However, replicating these advancements across diverse domains remains challenging due to limited methodological transparency. In this work, we present two-Staged history-Resampling Policy Optimization (SRPO), which successfully surpasses the performance of DeepSeek-R1-Zero-32B on the AIME24 and LiveCodeBench benchmarks. SRPO achieves this using the same base model as DeepSeek (i.e. Qwen2.5-32B) and relies solely on RL, without prior Supervised Fine-Tuning (SFT). Building upon Group Relative Policy Optimization (GRPO), we introduce two key methodological innovations: (1) a two-stage cross-domain training paradigm designed to balance the development of mathematical reasoning and coding proficiency, and (2) History Resampling (HR), a technique to address ineffective samples. Our comprehensive experiments validate the effectiveness of our approach, dedicating to offer valuable insights into scaling LLM reasoning capabilities across diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14225v1">Know Me, Respond to Me: Benchmarking LLMs for Dynamic User Profiling and Personalized Responses at Scale</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as personalized assistants for users across a wide range of tasks -- from offering writing support to delivering tailored recommendations or consultations. Over time, the interaction history between a user and an LLM can provide extensive information about an individual's traits and preferences. However, open questions remain on how well LLMs today can effectively leverage such history to (1) internalize the user's inherent traits and preferences, (2) track how the user profiling and preferences evolve over time, and (3) generate personalized responses accordingly in new scenarios. In this work, we introduce the PERSONAMEM benchmark. PERSONAMEM features curated user profiles with over 180 simulated user-LLM interaction histories, each containing up to 60 sessions of multi-turn conversations across 15 real-world tasks that require personalization. Given an in-situ user query, i.e. query issued by the user from the first-person perspective, we evaluate LLM chatbots' ability to identify the most suitable response according to the current state of the user's profile. We observe that current LLMs still struggle to recognize the dynamic evolution in users' profiles over time through direct prompting approaches. As a consequence, LLMs often fail to deliver responses that align with users' current situations and preferences, with frontier models such as GPT-4.1, o4-mini, GPT-4.5, o1, or Gemini-2.0 achieving only around 50% overall accuracy, suggesting room for improvement. We hope that PERSONAMEM, along with the user profile and conversation simulation pipeline, can facilitate future research in the development of truly user-aware chatbots. Code and data are available at github.com/bowen-upenn/PersonaMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14223v1">SimplifyMyText: An LLM-Based System for Inclusive Plain Language Text Simplification</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 accepted at ECIR 2025
    </div>
    <details class="paper-abstract">
      Text simplification is essential for making complex content accessible to diverse audiences who face comprehension challenges. Yet, the limited availability of simplified materials creates significant barriers to personal and professional growth and hinders social inclusion. Although researchers have explored various methods for automatic text simplification, none fully leverage large language models (LLMs) to offer tailored customization for different target groups and varying levels of simplicity. Moreover, despite its proven benefits for both consumers and organizations, the well-established practice of plain language remains underutilized. In this paper, we https://simplifymytext.org, the first system designed to produce plain language content from multiple input formats, including typed text and file uploads, with flexible customization options for diverse audiences. We employ GPT-4 and Llama-3 and evaluate outputs across multiple metrics. Overall, our work contributes to research on automatic text simplification and highlights the importance of tailored communication in promoting inclusivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17987v2">Reason2Attack: Jailbreaking Text-to-Image Models via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 This paper includes model-generated content that may contain offensive or distressing material
    </div>
    <details class="paper-abstract">
      Text-to-Image(T2I) models typically deploy safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods manually design prompts for the LLM to generate adversarial prompts, which effectively bypass safety filters while producing sensitive images, exposing safety vulnerabilities of T2I models. However, due to the LLM's limited understanding of the T2I model and its safety filters, existing methods require numerous queries to achieve a successful attack, limiting their practical applicability. To address this issue, we propose Reason2Attack(R2A), which aims to enhance the LLM's reasoning capabilities in generating adversarial prompts by incorporating the jailbreaking attack into the post-training process of the LLM. Specifically, we first propose a CoT example synthesis pipeline based on Frame Semantics, which generates adversarial prompts by identifying related terms and corresponding context illustrations. Using CoT examples generated by the pipeline, we fine-tune the LLM to understand the reasoning path and format the output structure. Subsequently, we incorporate the jailbreaking attack task into the reinforcement learning process of the LLM and design an attack process reward that considers prompt length, prompt stealthiness, and prompt effectiveness, aiming to further enhance reasoning accuracy. Extensive experiments on various T2I models show that R2A achieves a better attack success ratio while requiring fewer queries than baselines. Moreover, our adversarial prompts demonstrate strong attack transferability across both open-source and commercial T2I models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14177v1">Direct Advantage Regression: Aligning LLMs with Online AI Reward</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Online AI Feedback (OAIF) presents a promising alternative to Reinforcement Learning from Human Feedback (RLHF) by utilizing online AI preference in aligning language models (LLMs). However, the straightforward replacement of humans with AI deprives LLMs from learning more fine-grained AI supervision beyond binary signals. In this paper, we propose Direct Advantage Regression (DAR), a simple alignment algorithm using online AI reward to optimize policy improvement through weighted supervised fine-tuning. As an RL-free approach, DAR maintains theoretical consistency with online RLHF pipelines while significantly reducing implementation complexity and improving learning efficiency. Our empirical results underscore that AI reward is a better form of AI supervision consistently achieving higher human-AI agreement as opposed to AI preference. Additionally, evaluations using GPT-4-Turbo and MT-bench show that DAR outperforms both OAIF and online RLHF baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14175v1">Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion</a></div>
    <div class="paper-meta">
      📅 2025-04-19
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Query expansion methods powered by large language models (LLMs) have demonstrated effectiveness in zero-shot retrieval tasks. These methods assume that LLMs can generate hypothetical documents that, when incorporated into a query vector, enhance the retrieval of real evidence. However, we challenge this assumption by investigating whether knowledge leakage in benchmarks contributes to the observed performance gains. Using fact verification as a testbed, we analyzed whether the generated documents contained information entailed by ground truth evidence and assessed their impact on performance. Our findings indicate that performance improvements occurred consistently only for claims whose generated documents included sentences entailed by ground truth evidence. This suggests that knowledge leakage may be present in these benchmarks, inflating the perceived performance of LLM-based query expansion methods, particularly in real-world scenarios that require retrieving niche or novel knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14165v1">Self-Correction Makes LLMs Better Parsers</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across various natural language processing (NLP) tasks. However, recent studies suggest that they still face challenges in performing fundamental NLP tasks essential for deep language understanding, particularly syntactic parsing. In this paper, we conduct an in-depth analysis of LLM parsing capabilities, delving into the specific shortcomings of their parsing results. We find that LLMs may stem from limitations to fully leverage grammar rules in existing treebanks, which restricts their capability to generate valid syntactic structures. To help LLMs acquire knowledge without additional training, we propose a self-correction method that leverages grammar rules from existing treebanks to guide LLMs in correcting previous errors. Specifically, we automatically detect potential errors and dynamically search for relevant rules, offering hints and examples to guide LLMs in making corrections themselves. Experimental results on three datasets with various LLMs, demonstrate that our method significantly improves performance in both in-domain and cross-domain settings on the English and Chinese datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14152v1">FGMP: Fine-Grained Mixed-Precision Weight and Activation Quantization for Hardware-Accelerated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Quantization is a powerful tool to improve large language model (LLM) inference efficiency by utilizing more energy-efficient low-precision datapaths and reducing memory footprint. However, accurately quantizing LLM weights and activations to low precision is challenging without degrading model accuracy. We propose fine-grained mixed precision (FGMP) quantization, a post-training mixed-precision quantization hardware-software co-design methodology that maintains accuracy while quantizing the majority of weights and activations to reduced precision. Our work makes the following contributions: 1) We develop a policy that uses the perturbation in each value, weighted by the Fisher information, to select which weight and activation blocks to keep in higher precision. This approach preserves accuracy by identifying which weight and activation blocks need to be retained in higher precision to minimize the perturbation in the model loss. 2) We also propose a sensitivity-weighted clipping approach for fine-grained quantization which helps retain accuracy for blocks that are quantized to low precision. 3) We then propose hardware augmentations to leverage the efficiency benefits of FGMP quantization. Our hardware implementation encompasses i) datapath support for FGMP at block granularity, and ii) a mixed-precision activation quantization unit to assign activation blocks to high or low precision on the fly with minimal runtime and energy overhead. Our design, prototyped using NVFP4 (an FP4 format with microscaling) as the low-precision datatype and FP8 as the high-precision datatype, facilitates efficient FGMP quantization, attaining <1% perplexity degradation on Wikitext-103 for the Llama-2-7B model relative to an all-FP8 baseline design while consuming 14% less energy during inference and requiring 30% less weight memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14119v1">CODECRASH: Stress Testing LLM Reasoning under Structural and Semantic Perturbations</a></div>
    <div class="paper-meta">
      📅 2025-04-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently showcased strong capabilities in code-related tasks, yet their robustness in code comprehension and reasoning remains underexplored. In this paper, we present CodeCrash, a unified benchmark that evaluates LLM robustness under code structural and textual distraction perturbations, applied to two established benchmarks -- CRUXEval and LiveCodeBench -- across both input and output prediction tasks. We evaluate seventeen LLMs using direct and Chain-of-Thought inference to systematically analyze their robustness, identify primary reasons for performance degradation, and highlight failure modes. Our findings reveal the fragility of LLMs under structural noise and the inherent reliance on natural language cues, highlighting critical robustness issues of LLMs in code execution and understanding. Additionally, we examine three Large Reasoning Models (LRMs) and discover the severe vulnerability of self-reflective reasoning mechanisms that lead to reasoning collapse. CodeCrash provides a principled framework for stress-testing LLMs in code understanding, offering actionable directions for future evaluation and benchmarking. The code of CodeCrash and the robustness leaderboard are publicly available at https://donaldlamnl.github.io/CodeCrash/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13837v1">Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 24 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning capabilities of LLMs, particularly in mathematics and programming tasks. It is widely believed that RLVR enables LLMs to continuously self-improve, thus acquiring novel reasoning abilities that exceed corresponding base models' capacity. In this study, however, we critically re-examines this assumption by measuring the pass@\textit{k} metric with large values of \textit{k} to explore the reasoning capability boundary of the models across a wide range of model families and benchmarks. Surprisingly, the RL does \emph{not}, in fact, elicit fundamentally new reasoning patterns. While RL-trained models outperform their base models at smaller values of $k$ (\eg, $k$=1), base models can achieve a comparable or even higher pass@$k$ score compared to their RL counterparts at large $k$ values. The reasoning paths generated by RL-trained models are already included in the base models' sampling distribution, suggesting that most reasoning abilities manifested in RL-trained models are already obtained by base models. Further analysis shows that RL training boosts the performance by biasing the model's output distribution toward paths that are more likely to yield rewards, therefore sampling correct responses more efficiently. But this also results in a narrower reasoning capability boundary compared to base models. Similar results are observed in visual reasoning tasks trained with RLVR. Moreover, we find that distillation can genuinely introduce new knowledge into the model, different from RLVR. These findings underscore a critical limitation of RLVR in advancing LLM reasoning abilities which requires us to fundamentally rethink the impact of RL training in reasoning LLMs and the need of a better paradigm. Project Page: https://limit-of-RLVR.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13816v1">Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      While understanding the knowledge boundaries of LLMs is crucial to prevent hallucination, research on knowledge boundaries of LLMs has predominantly focused on English. In this work, we present the first study to analyze how LLMs recognize knowledge boundaries across different languages by probing their internal representations when processing known and unknown questions in multiple languages. Our empirical studies reveal three key findings: 1) LLMs' perceptions of knowledge boundaries are encoded in the middle to middle-upper layers across different languages. 2) Language differences in knowledge boundary perception follow a linear structure, which motivates our proposal of a training-free alignment method that effectively transfers knowledge boundary perception ability across languages, thereby helping reduce hallucination risk in low-resource languages; 3) Fine-tuning on bilingual question pair translation further enhances LLMs' recognition of knowledge boundaries across languages. Given the absence of standard testbeds for cross-lingual knowledge boundary analysis, we construct a multilingual evaluation suite comprising three representative types of knowledge boundary data. Our code and datasets are publicly available at https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v5">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13774v1">DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 49 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently revolutionized language processing tasks but have also brought ethical and legal issues. LLMs have a tendency to memorize potentially private or copyrighted information present in the training data, which might then be delivered to end users at inference time. When this happens, a naive solution is to retrain the model from scratch after excluding the undesired data. Although this guarantees that the target data have been forgotten, it is also prohibitively expensive for LLMs. Approximate unlearning offers a more efficient alternative, as it consists of ex post modifications of the trained model itself to prevent undesirable results, but it lacks forgetting guarantees because it relies solely on empirical evidence. In this work, we present DP2Unlearning, a novel LLM unlearning framework that offers formal forgetting guarantees at a significantly lower cost than retraining from scratch on the data to be retained. DP2Unlearning involves training LLMs on textual data protected using {\epsilon}-differential privacy (DP), which later enables efficient unlearning with the guarantees against disclosure associated with the chosen {\epsilon}. Our experiments demonstrate that DP2Unlearning achieves similar model performance post-unlearning, compared to an LLM retraining from scratch on retained data -- the gold standard exact unlearning -- but at approximately half the unlearning cost. In addition, with a reasonable computational cost, it outperforms approximate unlearning methods at both preserving the utility of the model post-unlearning and effectively forgetting the targeted information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13769v1">Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 The paper has been peer-reviewed and accepted for publication to the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE 2025)
    </div>
    <details class="paper-abstract">
      Malicious software packages in open-source ecosystems, such as PyPI, pose growing security risks. Unlike traditional vulnerabilities, these packages are intentionally designed to deceive users, making detection challenging due to evolving attack methods and the lack of structured datasets. In this work, we empirically evaluate the effectiveness of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and few-shot learning for detecting malicious source code. We fine-tune LLMs on curated datasets and integrate YARA rules, GitHub Security Advisories, and malicious code snippets with the aim of enhancing classification accuracy. We came across a counterintuitive outcome: While RAG is expected to boost up the prediction performance, it fails in the performed evaluation, obtaining a mediocre accuracy. In contrast, few-shot learning is more effective as it significantly improves the detection of malicious code, achieving 97% accuracy and 95% balanced accuracy, outperforming traditional RAG approaches. Thus, future work should expand structured knowledge bases, refine retrieval models, and explore hybrid AI-driven cybersecurity solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19413v2">Project Alexandria: Towards Freeing Scientific Knowledge from Copyright Burdens via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Paywalls, licenses and copyright rules often restrict the broad dissemination and reuse of scientific knowledge. We take the position that it is both legally and technically feasible to extract the scientific knowledge in scholarly texts. Current methods, like text embeddings, fail to reliably preserve factual content, and simple paraphrasing may not be legally sound. We propose a new idea for the community to adopt: convert scholarly documents into knowledge preserving, but style agnostic representations we term Knowledge Units using LLMs. These units use structured data capturing entities, attributes and relationships without stylistic content. We provide evidence that Knowledge Units (1) form a legally defensible framework for sharing knowledge from copyrighted research texts, based on legal analyses of German copyright law and U.S. Fair Use doctrine, and (2) preserve most (~95\%) factual knowledge from original text, measured by MCQ performance on facts from the original copyrighted text across four research domains. Freeing scientific knowledge from copyright promises transformative benefits for scientific research and education by allowing language models to reuse important facts from copyrighted text. To support this, we share open-source tools for converting research documents into Knowledge Units. Overall, our work posits the feasibility of democratizing access to scientific knowledge while respecting copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09024v3">AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      The robustness of LLMs to jailbreak attacks, where users design prompts to circumvent safety measures and misuse model capabilities, has been studied primarily for LLMs acting as simple chatbots. Meanwhile, LLM agents -- which use external tools and can execute multi-stage tasks -- may pose a greater risk if misused, but their robustness remains underexplored. To facilitate research on LLM agent misuse, we propose a new benchmark called AgentHarm. The benchmark includes a diverse set of 110 explicitly malicious agent tasks (440 with augmentations), covering 11 harm categories including fraud, cybercrime, and harassment. In addition to measuring whether models refuse harmful agentic requests, scoring well on AgentHarm requires jailbroken agents to maintain their capabilities following an attack to complete a multi-step task. We evaluate a range of leading LLMs, and find (1) leading LLMs are surprisingly compliant with malicious agent requests without jailbreaking, (2) simple universal jailbreak templates can be adapted to effectively jailbreak agents, and (3) these jailbreaks enable coherent and malicious multi-step agent behavior and retain model capabilities. To enable simple and reliable evaluation of attacks and defenses for LLM-based agents, we publicly release AgentHarm at https://huggingface.co/datasets/ai-safety-institute/AgentHarm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11005v3">A Theory of LLM Sampling: Part Descriptive and Part Prescriptive</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized in autonomous decision-making, where they sample options from vast action spaces. However, the heuristics that guide this sampling process remain under-explored. We study this sampling behavior and show that this underlying heuristics resembles that of human decision-making: comprising a descriptive component (reflecting statistical norm) and a prescriptive component (implicit ideal encoded in the LLM) of a concept. We show that this deviation of a sample from the statistical norm towards a prescriptive component consistently appears in concepts across diverse real-world domains like public health, and economic trends. To further illustrate the theory, we demonstrate that concept prototypes in LLMs are affected by prescriptive norms, similar to the concept of normality in humans. Through case studies and comparison with human studies, we illustrate that in real-world applications, the shift of samples toward an ideal value in LLMs' outputs can result in significantly biased decision-making, raising ethical concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12319v4">The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences of LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used as evaluators for natural language generation tasks, ensuring unbiased assessments is essential. However, LLM evaluators often display biased preferences, such as favoring verbosity and authoritative tones. Our empirical analysis reveals that these biases are exacerbated in pairwise evaluation, where LLMs directly compare two outputs and easily prioritize superficial attributes. In contrast, pointwise evaluation, which assesses outputs independently, is less susceptible to such bias because each output is judged in isolation. To address the limitations of the pairwise evaluation, we introduce a novel evaluation method, PRePair, which integrates pointwise reasoning within a pairwise framework. PRePair effectively alleviates biased preference, improving performance on the adversarial benchmark (LLMBar) while outperforming pointwise evaluation on the standard benchmark (MT-Bench).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19874v3">Is In-Context Learning Sufficient for Instruction Following in LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted at ICLR 2025. This camera-ready version v3 adds multi-turn alignment via ICL, revisiting main results on instruct models, and simple mechanistic study. Updates in the v2: experiment with decoding schemes, scaling in-context alignment, ICL vs IFT for instruction following. Code at https://github.com/tml-epfl/icl-alignment
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) allows LLMs to learn from examples without changing their weights: this is a particularly promising capability for long-context LLMs that can potentially learn from many examples. Recently, Lin et al. (2024) proposed URIAL, a method using only three in-context examples to align base LLMs, achieving non-trivial instruction following performance. In this work, we show that, while effective, ICL alignment with URIAL still underperforms compared to instruction fine-tuning on the established benchmark MT-Bench, especially with more capable base LLMs. We then uncover the most relevant elements for successful in-context alignment, finding the crucial role of the decoding parameters. Based on these insights, we show that the approach of URIAL can indeed be improved by adding high-quality, potentially carefully selected via greedy search, demonstrations in context, getting closer to the performance of instruct models. Finally, we provide the first, to our knowledge, systematic comparison of ICL and instruction fine-tuning (IFT) for instruction following in the low data regime, where ICL can be a viable alternative to IFT. Overall, our work advances the understanding of ICL as an alignment technique and its relationship to IFT. We provide our code at https://github.com/tml-epfl/icl-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13629v1">Divergent LLM Adoption and Heterogeneous Convergence Paths in Research Writing</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as ChatGPT, are reshaping content creation and academic writing. This study investigates the impact of AI-assisted generative revisions on research manuscripts, focusing on heterogeneous adoption patterns and their influence on writing convergence. Leveraging a dataset of over 627,000 academic papers from arXiv, we develop a novel classification framework by fine-tuning prompt- and discipline-specific large language models to detect the style of ChatGPT-revised texts. Our findings reveal substantial disparities in LLM adoption across academic disciplines, gender, native language status, and career stage, alongside a rapid evolution in scholarly writing styles. Moreover, LLM usage enhances clarity, conciseness, and adherence to formal writing conventions, with improvements varying by revision type. Finally, a difference-in-differences analysis shows that while LLMs drive convergence in academic writing, early adopters, male researchers, non-native speakers, and junior scholars exhibit the most pronounced stylistic shifts, aligning their writing more closely with that of established researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18337v2">Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
    </div>
    <details class="paper-abstract">
      Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20999v3">MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. Typically, LLMs are first pre-trained on large corpora and subsequently fine-tuned on task-specific datasets. However, during fine-tuning, LLMs may forget some knowledge acquired in the pre-training stage, leading to a decline in general capabilities. Existing approaches to mitigate forgetting often rely on access to pre-training data, which may be unavailable in many real-world scenarios--such as fine-tuning checkpoint-only open-source LLMs. To address this challenge, we propose a new fine-tuning algorithm termed Momentum-Filtered Optimizer (MoFO). MoFO is an extension of greedy block coordinate descent (BCD) methods: in each iteration, MoFO only updates the model parameters with the largest momentum magnitudes, while keeping all other parameters fixed. MoFO achieves similar fine-tuning performance to the default fine-tuning algorithm while effectively mitigating knowledge forgetting. We validate MoFO through rigorous convergence analysis and extensive experiments, demonstrating its effectiveness in mitigating forgetting without pre-training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13562v1">DETAM: Defending LLMs Against Jailbreak Attacks via Targeted Attention Modification</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), jailbreak attacks have become an increasingly pressing safety concern. While safety-aligned LLMs can effectively defend against normal harmful queries, they remain vulnerable to such attacks. Existing defense methods primarily rely on fine-tuning or input modification, which often suffer from limited generalization and reduced utility. To address this, we introduce DETAM, a finetuning-free defense approach that improves the defensive capabilities against jailbreak attacks of LLMs via targeted attention modification. Specifically, we analyze the differences in attention scores between successful and unsuccessful defenses to identify the attention heads sensitive to jailbreak attacks. During inference, we reallocate attention to emphasize the user's core intention, minimizing interference from attack tokens. Our experimental results demonstrate that DETAM outperforms various baselines in jailbreak defense and exhibits robust generalization across different attacks and models, maintaining its effectiveness even on in-the-wild jailbreak data. Furthermore, in evaluating the model's utility, we incorporated over-defense datasets, which further validate the superior performance of our approach. The code will be released immediately upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13557v1">Integrating LLMs for Grading and Appeal Resolution in Computer Science Education</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This study explores the integration of Large Language Models (LLMs) into the grading and appeal resolution process in computer science education. We introduce AI-PAT, an AI-powered assessment tool that leverages LLMs to evaluate computer science exams, generate feedback, and address student appeals. AI-PAT was used to assess over 850 exam submissions and handle 185 appeal cases. Our multi-model comparison (ChatGPT, Gemini) reveals strong correlations between model outputs, though significant variability persists depending on configuration and prompt design. Human graders, while internally consistent, showed notable inter-rater disagreement, further highlighting subjectivity in manual evaluation. The appeal process led to grade changes in 74% of cases, indicating the need for continued refinement of AI evaluation strategies. While students appreciated the speed and detail of AI feedback, survey responses revealed trust and fairness concerns. We conclude that AI-PAT offers scalable benefits for formative assessment and feedback, but must be accompanied by transparent grading rubrics, human oversight, and appeal mechanisms to ensure equitable outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12867v2">EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and chain-of-modality (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at https://anonymous.4open.science/r/EmoVoice-DF55. Dataset, code, and checkpoints will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15545v5">SciLitLLM: How to Adapt LLMs for Scientific Literature Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Scientific literature understanding is crucial for extracting targeted information and garnering insights, thereby significantly advancing scientific discovery. Despite the remarkable success of Large Language Models (LLMs), they face challenges in scientific literature understanding, primarily due to (1) a lack of scientific knowledge and (2) unfamiliarity with specialized scientific tasks. To develop an LLM specialized in scientific literature understanding, we propose a hybrid strategy that integrates continual pre-training (CPT) and supervised fine-tuning (SFT), to simultaneously infuse scientific domain knowledge and enhance instruction-following capabilities for domain-specific tasks.cIn this process, we identify two key challenges: (1) constructing high-quality CPT corpora, and (2) generating diverse SFT instructions. We address these challenges through a meticulous pipeline, including PDF text extraction, parsing content error correction, quality filtering, and synthetic instruction creation. Applying this strategy, we present a suite of LLMs: SciLitLLM, specialized in scientific literature understanding. These models demonstrate promising performance on scientific literature understanding benchmarks. Our contributions are threefold: (1) We present an effective framework that integrates CPT and SFT to adapt LLMs to scientific literature understanding, which can also be easily adapted to other domains. (2) We propose an LLM-based synthesis method to generate diverse and high-quality scientific instructions, resulting in a new instruction set -- SciLitIns -- for supervised fine-tuning in less-represented scientific domains. (3) SciLitLLM achieves promising performance improvements on scientific literature understanding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11056v2">Large Language Models are Good Multi-lingual Learners : When LLMs Meet Cross-lingual Prompts</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      With the advent of Large Language Models (LLMs), generating rule-based data for real-world applications has become more accessible. Due to the inherent ambiguity of natural language and the complexity of rule sets, especially in long contexts, LLMs often struggle to follow all specified rules, frequently omitting at least one. To enhance the reasoning and understanding of LLMs on long and complex contexts, we propose a novel prompting strategy Multi-Lingual Prompt, namely MLPrompt, which automatically translates the error-prone rule that an LLM struggles to follow into another language, thus drawing greater attention to it. Experimental results on public datasets across various tasks have shown MLPrompt can outperform state-of-the-art prompting methods such as Chain of Thought, Tree of Thought, and Self-Consistency. Additionally, we introduce a framework integrating MLPrompt with an auto-checking mechanism for structured data generation, with a specific case study in text-to-MIP instances. Further, we extend the proposed framework for text-to-SQL to demonstrate its generation ability towards structured data synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13843v2">AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted by SIGIR 2025, 6 pages
    </div>
    <details class="paper-abstract">
      LLM-based user agents, which simulate user interaction behavior, are emerging as a promising approach to enhancing recommender systems. In real-world scenarios, users' interactions often exhibit cross-domain characteristics and are influenced by others. However, the memory design in current methods causes user agents to introduce significant irrelevant information during decision-making in cross-domain scenarios and makes them unable to recognize the influence of other users' interactions, such as popularity factors. To tackle this issue, we propose a dual-layer memory architecture combined with a two-step fusion mechanism. This design avoids irrelevant information during decision-making while ensuring effective integration of cross-domain preferences. We also introduce the concepts of interest groups and group-shared memory to better capture the influence of popularity factors on users with similar interests. Comprehensive experiments validate the effectiveness of AgentCF++. Our code is available at https://github.com/jhliu0807/AgentCF-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02497v2">PennyLang: Pioneering LLM-Based Quantum Code Generation with a Novel PennyLane-Centric Dataset</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 10 pages, 7 figures, 7 tables, submitted for review under QCE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer remarkable capabilities in code generation, natural language processing, and domain-specific reasoning. However, their application in quantum software development remains underexplored, particularly for PennyLane-a leading framework for hybrid quantum-classical computing. To address this gap, we introduce a novel, high-quality dataset comprising 3,347 PennyLane-specific quantum code samples and contextual descriptions, specifically curated to support LLM training and fine-tuning for quantum code assistance. Our contributions are threefold: (1) the automatic construction and open-source release of a comprehensive PennyLane dataset derived from textbooks, official documentation, and open-source repositories; (2) a structured methodology for data curation, annotation, and formatting to enhance LLM usability and relevance; and (3) a rigorous evaluation of code generation capabilities using both baseline Retrieval-Augmented Generation (RAG) and a GraphRAG-enhanced pipeline. Using the PennyLang framework, we demonstrate that GraphRAG, when applied to a GPT-4o Mini model, substantially outperforms standard prompting and baseline RAG. Accuracy improves from 20.5% (without RAG) to 58.2% with GraphRAG, showcasing its effectiveness in reducing hallucinations and improving code correctness in quantum programming tasks. Compared to prior efforts focused largely on Qiskit, our work expands LLM-based assistance to the PennyLane ecosystem, contributing practical tools and reproducible methodologies for advancing AI-assisted quantum software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13845v2">Improving LLM-powered Recommendations with Personalized Information</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 Accepted by SIGIR 2025, 7 pages
    </div>
    <details class="paper-abstract">
      Due to the lack of explicit reasoning modeling, existing LLM-powered recommendations fail to leverage LLMs' reasoning capabilities effectively. In this paper, we propose a pipeline called CoT-Rec, which integrates two key Chain-of-Thought (CoT) processes -- user preference analysis and item perception analysis -- into LLM-powered recommendations, thereby enhancing the utilization of LLMs' reasoning abilities. CoT-Rec consists of two stages: (1) personalized information extraction, where user preferences and item perception are extracted, and (2) personalized information utilization, where this information is incorporated into the LLM-powered recommendation process. Experimental results demonstrate that CoT-Rec shows potential for improving LLM-powered recommendations. The implementation is publicly available at https://github.com/jhliu0807/CoT-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00383v2">Unified Parameter-Efficient Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized natural language processing, enabling advanced understanding and reasoning capabilities across a variety of tasks. Fine-tuning these models for specific domains, particularly through Parameter-Efficient Fine-Tuning (PEFT) strategies like LoRA, has become a prevalent practice due to its efficiency. However, this raises significant privacy and security concerns, as models may inadvertently retain and disseminate sensitive or undesirable information. To address these issues, we introduce a novel instance-wise unlearning framework, LLMEraser, which systematically categorizes unlearning tasks and applies precise parameter adjustments using influence functions. Unlike traditional unlearning techniques that are often limited in scope and require extensive retraining, LLMEraser is designed to handle a broad spectrum of unlearning tasks without compromising model performance. Extensive experiments on benchmark datasets demonstrate that LLMEraser excels in efficiently managing various unlearning scenarios while maintaining the overall integrity and efficacy of the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13475v1">LLM Sensitivity Evaluation Framework for Clinical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance across various domains. However, for clinical diagnosis, higher expectations are required for LLM's reliability and sensitivity: thinking like physicians and remaining sensitive to key medical information that affects diagnostic reasoning, as subtle variations can lead to different diagnosis results. Yet, existing works focus mainly on investigating the sensitivity of LLMs to irrelevant context and overlook the importance of key information. In this paper, we investigate the sensitivity of LLMs, i.e. GPT-3.5, GPT-4, Gemini, Claude3 and LLaMA2-7b, to key medical information by introducing different perturbation strategies. The evaluation results highlight the limitations of current LLMs in remaining sensitive to key medical information for diagnostic decision-making. The evolution of LLMs must focus on improving their reliability, enhancing their ability to be sensitive to key information, and effectively utilizing this information. These improvements will enhance human trust in LLMs and facilitate their practical application in real-world scenarios. Our code and dataset are available at https://github.com/chenwei23333/DiagnosisQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13474v1">Everything You Wanted to Know About LLM-based Vulnerability Detection But Were Afraid to Ask</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models are a promising tool for automated vulnerability detection, thanks to their success in code generation and repair. However, despite widespread adoption, a critical question remains: Are LLMs truly effective at detecting real-world vulnerabilities? Current evaluations, which often assess models on isolated functions or files, ignore the broader execution and data-flow context essential for understanding vulnerabilities. This oversight leads to two types of misleading outcomes: incorrect conclusions and flawed rationales, collectively undermining the reliability of prior assessments. Therefore, in this paper, we challenge three widely held community beliefs: that LLMs are (i) unreliable, (ii) insensitive to code patches, and (iii) performance-plateaued across model scales. We argue that these beliefs are artifacts of context-deprived evaluations. To address this, we propose CORRECT (Context-Rich Reasoning Evaluation of Code with Trust), a new evaluation framework that systematically incorporates contextual information into LLM-based vulnerability detection. We construct a context-rich dataset of 2,000 vulnerable-patched program pairs spanning 99 CWEs and evaluate 13 LLMs across four model families. Our framework elicits both binary predictions and natural-language rationales, which are further validated using LLM-as-a-judge techniques. Our findings overturn existing misconceptions. When provided with sufficient context, SOTA LLMs achieve significantly improved performance (e.g., 0.7 F1-score on key CWEs), with 0.8 precision. We show that most false positives stem from reasoning errors rather than misclassification, and that while model and test-time scaling improve performance, they introduce diminishing returns and trade-offs in recall. Finally, we uncover new flaws in current LLM-based detection systems, such as limited generalization and overthinking biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13471v1">From Large to Super-Tiny: End-to-End Optimization for Cost-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have significantly advanced artificial intelligence by optimizing traditional Natural Language Processing (NLP) pipelines, improving performance and generalization. This has spurred their integration into various systems. Many NLP systems, including ours, employ a "one-stage" pipeline directly incorporating LLMs. While effective, this approach incurs substantial costs and latency due to the need for large model parameters to achieve satisfactory outcomes. This paper introduces a three-stage cost-efficient end-to-end LLM deployment pipeline-including prototyping, knowledge transfer, and model compression-to tackle the cost-performance dilemma in LLM-based frameworks. Our approach yields a super tiny model optimized for cost and performance in online systems, simplifying the system architecture. Initially, by transforming complex tasks into a function call-based LLM-driven pipeline, an optimal performance prototype system is constructed to produce high-quality data as a teacher model. The second stage combine techniques like rejection fine-tuning, reinforcement learning and knowledge distillation to transfer knowledge to a smaller 0.5B student model, delivering effective performance at minimal cost. The final stage applies quantization and pruning to extremely compress model to 0.4B, achieving ultra-low latency and cost. The framework's modular design and cross-domain capabilities suggest potential applicability in other NLP areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15288v2">If LLMs Would Just Look: Simple Line-by-line Checking Improves Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      The rapid expansion of software systems and the growing number of reported vulnerabilities have emphasized the importance of accurately identifying vulnerable code segments. Traditional methods for vulnerability localization, such as manual code audits or rule-based tools, are often time-consuming and limited in scope, typically focusing on specific programming languages or types of vulnerabilities. In recent years, the introduction of large language models (LLMs) such as GPT and LLaMA has opened new possibilities for automating vulnerability detection. However, while LLMs show promise in this area, they face challenges, particularly in maintaining accuracy over longer code contexts. This paper introduces LOVA, a novel framework leveraging the self-attention mechanisms inherent in LLMs to enhance vulnerability localization. Our key insight is that self-attention mechanisms assign varying importance to different parts of the input, making it possible to track how much attention the model focuses on specific lines of code. In the context of vulnerability localization, the hypothesis is that vulnerable lines of code will naturally attract higher attention weights because they have a greater influence on the model's output. By systematically tracking changes in attention weights and focusing on specific lines of code, LOVA improves the precision of identifying vulnerable lines across various programming languages. Through rigorous experimentation and evaluation, we demonstrate that LOVA significantly outperforms existing LLM-based approaches, achieving up to a 5.3x improvement in F1-scores. LOVA also demonstrated strong scalability, with up to a 14.6x improvement in smart contract vulnerability localization across languages like C, Python, Java, and Solidity. Its robustness was proven through consistent performance across different LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12943v2">Customizing Emotional Support: How Do Individuals Construct and Interact With LLM-Powered Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 20 pages, 3 figures, 3 tables. Accepted to CHI 2025, ACM Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Personalized support is essential to fulfill individuals' emotional needs and sustain their mental well-being. Large language models (LLMs), with great customization flexibility, hold promises to enable individuals to create their own emotional support agents. In this work, we developed ChatLab, where users could construct LLM-powered chatbots with additional interaction features including voices and avatars. Using a Research through Design approach, we conducted a week-long field study followed by interviews and design activities (N = 22), which uncovered how participants created diverse chatbot personas for emotional reliance, confronting stressors, connecting to intellectual discourse, reflecting mirrored selves, etc. We found that participants actively enriched the personas they constructed, shaping the dynamics between themselves and the chatbot to foster open and honest conversations. They also suggested other customizable features, such as integrating online activities and adjustable memory settings. Based on these findings, we discuss opportunities for enhancing personalized emotional support through emerging AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15793v4">DNR Bench: Benchmarking Over-Reasoning in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Test-time scaling has significantly improved large language model performance, enabling deeper reasoning to solve complex problems. However, this increased reasoning capability also leads to excessive token generation and unnecessary problem-solving attempts. We introduce Don\'t Reason Bench (DNR Bench), a new benchmark designed to evaluate LLMs ability to robustly understand the tricky reasoning triggers and avoiding unnecessary generation. DNR Bench consists of 150 adversarially designed prompts that are easy for humans to understand and respond to, but surprisingly not for many of the recent prominent LLMs. DNR Bench tests models abilities across different capabilities, such as instruction adherence, hallucination avoidance, redundancy filtering, and unanswerable question recognition. We evaluate reasoning LLMs (RLMs), including DeepSeek-R1, OpenAI O3-mini, Claude-3.7-sonnet and compare them against a powerful non-reasoning model, e.g., GPT-4o. Our experiments reveal that RLMs generate up to 70x more tokens than necessary, often failing at tasks that simpler non-reasoning models handle efficiently with higher accuracy. Our findings underscore the need for more effective training and inference strategies in RLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14096v1">VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduce VideoPASTA (Preference Alignment with Spatio-Temporal-Cross Frame Adversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully generated adversarial examples that deliberately violate spatial, temporal, or cross-frame relations. By applying Direct Preference Optimization to just 7,020 preference pairs, VideoPASTA learns robust representations that capture fine-grained spatial relationships and long-range temporal dynamics. Experiments on standard video benchmarks show significant relative performance gains of 3.05% on VideoMME, 1.97% on NeXTQA, and 1.31% on LongVideoBench, over the baseline Qwen2.5-VL model. These results demonstrate that targeted alignment, rather than massive pretraining or architectural modifications, effectively addresses core video-language challenges. Notably, VideoPASTA achieves these improvements without human annotation or captioning, relying on just 32-frame sampling, compared to the 96-frame, multi-GPU setups of prior work. This efficiency makes our approach a scalable, plug-and-play solution that seamlessly integrates with existing models while preserving their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09002v2">KNighter: Transforming Static Analysis with LLM-Synthesized Checkers</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Static analysis is a powerful technique for bug detection in critical systems like operating system kernels. However, designing and implementing static analyzers is challenging, time-consuming, and typically limited to predefined bug patterns. While large language models (LLMs) have shown promise for static analysis, directly applying them to scan large systems remains impractical due to computational constraints and contextual limitations. We present KNighter, the first approach that unlocks scalable LLM-based static analysis by automatically synthesizing static analyzers from historical bug patterns. Rather than using LLMs to directly analyze massive systems, our key insight is leveraging LLMs to generate specialized static analyzers guided by historical patch knowledge. KNighter implements this vision through a multi-stage synthesis pipeline that validates checker correctness against original patches and employs an automated refinement process to iteratively reduce false positives. Our evaluation on the Linux kernel demonstrates that KNighter generates high-precision checkers capable of detecting diverse bug patterns overlooked by existing human-written analyzers. To date, KNighter-synthesized checkers have discovered 92 new, critical, long-latent bugs (average 4.3 years) in the Linux kernel; 77 are confirmed, 57 fixed, and 16 have been assigned CVE numbers. This work establishes an entirely new paradigm for scalable, reliable, and traceable LLM-based static analysis for real-world systems via checker synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13145v2">Exploring Expert Failures Improves LLM Agent Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown tremendous potential as agents, excelling at tasks that require multiple rounds of reasoning and interactions. Rejection Sampling Fine-Tuning (RFT) has emerged as an effective method for finetuning LLMs as agents: it first imitates expert-generated successful trajectories and further improves agentic skills through iterative fine-tuning on successful, self-generated trajectories. However, since the expert (e.g., GPT-4) succeeds primarily on simpler subtasks and RFT inherently favors simpler scenarios, many complex subtasks remain unsolved and persistently out-of-distribution (OOD). Upon investigating these challenging subtasks, we discovered that previously failed expert trajectories can often provide valuable guidance, e.g., plans and key actions, that can significantly improve agent exploration efficiency and acquisition of critical skills. Motivated by these observations, we propose Exploring Expert Failures (EEF), which identifies beneficial actions from failed expert trajectories and integrates them into the training dataset. Potentially harmful actions are meticulously excluded to prevent contamination of the model learning process. By leveraging the beneficial actions in expert failures, EEF successfully solves some previously unsolvable subtasks and improves agent tuning performance. Remarkably, our approach achieved a 62\% win rate in WebShop, outperforming RFT (53. 6\%) and GPT-4 (35. 6\%), and to the best of our knowledge, setting a new state-of-the-art as the first method to surpass a score of 0.81 in WebShop and exceed 81 in SciWorld.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19314v2">LiveBench: A Challenging, Contamination-Limited LLM Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 ICLR 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Test set contamination, wherein test data from a benchmark ends up in a newer model's training set, is a well-documented obstacle for fair LLM evaluation and can quickly render benchmarks obsolete. To mitigate this, many recent benchmarks crowdsource new prompts and evaluations from human or LLM judges; however, these can introduce significant biases, and break down when scoring hard questions. In this work, we introduce a new benchmark for LLMs designed to be resistant to both test set contamination and the pitfalls of LLM judging and human crowdsourcing. We release LiveBench, the first benchmark that (1) contains frequently-updated questions from recent information sources, (2) scores answers automatically according to objective ground-truth values, and (3) contains a wide variety of challenging tasks, spanning math, coding, reasoning, language, instruction following, and data analysis. To achieve this, LiveBench contains questions that are based on recently-released math competitions, arXiv papers, news articles, and datasets, and it contains harder, contamination-limited versions of tasks from previous benchmarks such as Big-Bench Hard, AMPS, and IFEval. We evaluate many prominent closed-source models, as well as dozens of open-source models ranging from 0.5B to 405B in size. LiveBench is difficult, with top models achieving below 70% accuracy. We release all questions, code, and model answers. Questions are added and updated on a monthly basis, and we release new tasks and harder versions of tasks over time so that LiveBench can distinguish between the capabilities of LLMs as they improve in the future. We welcome community engagement and collaboration for expanding the benchmark tasks and models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14039v1">MEQA: A Meta-Evaluation Framework for Question & Answer LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) advance, their potential for widespread societal impact grows simultaneously. Hence, rigorous LLM evaluations are both a technical necessity and social imperative. While numerous evaluation benchmarks have been developed, there remains a critical gap in meta-evaluation: effectively assessing benchmarks' quality. We propose MEQA, a framework for the meta-evaluation of question and answer (QA) benchmarks, to provide standardized assessments, quantifiable scores, and enable meaningful intra-benchmark comparisons. We demonstrate this approach on cybersecurity benchmarks, using human and LLM evaluators, highlighting the benchmarks' strengths and weaknesses. We motivate our choice of test domain by AI models' dual nature as powerful defensive tools and security threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14038v1">Flowco: Rethinking Data Analysis in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Conducting data analysis typically involves authoring code to transform, visualize, analyze, and interpret data. Large language models (LLMs) are now capable of generating such code for simple, routine analyses. LLMs promise to democratize data science by enabling those with limited programming expertise to conduct data analyses, including in scientific research, business, and policymaking. However, analysts in many real-world settings must often exercise fine-grained control over specific analysis steps, verify intermediate results explicitly, and iteratively refine their analytical approaches. Such tasks present barriers to building robust and reproducible analyses using LLMs alone or even in conjunction with existing authoring tools (e.g., computational notebooks). This paper introduces Flowco, a new mixed-initiative system to address these challenges. Flowco leverages a visual dataflow programming model and integrates LLMs into every phase of the authoring process. A user study suggests that Flowco supports analysts, particularly those with less programming experience, in quickly authoring, debugging, and refining data analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14024v1">Simplicity by Obfuscation: Evaluating LLM-Driven Code Transformation with Semantic Elasticity</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 The paper has been peer-reviewed and accepted for publication in the proceedings of the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE 2025)
    </div>
    <details class="paper-abstract">
      Code obfuscation is the conversion of original source code into a functionally equivalent but less readable form, aiming to prevent reverse engineering and intellectual property theft. This is a challenging task since it is crucial to maintain functional correctness of the code while substantially disguising the input code. The recent development of large language models (LLMs) paves the way for practical applications in different domains, including software engineering. This work performs an empirical study on the ability of LLMs to obfuscate Python source code and introduces a metric (i.e., semantic elasticity) to measure the quality degree of obfuscated code. We experimented with 3 leading LLMs, i.e., Claude-3.5-Sonnet, Gemini-1.5, GPT-4-Turbo across 30 Python functions from diverse computational domains. Our findings reveal GPT-4-Turbo's remarkable effectiveness with few-shot prompting (81% pass rate versus 29% standard prompting), significantly outperforming both Gemini-1.5 (39%) and Claude-3.5-Sonnet (30%). Notably, we discovered a counter-intuitive "obfuscation by simplification" phenomenon where models consistently reduce rather than increase cyclomatic complexity. This study provides a methodological framework for evaluating AI-driven obfuscation while highlighting promising directions for leveraging LLMs in software security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13993v1">CPR: Leveraging LLMs for Topic and Phrase Suggestion to Facilitate Comprehensive Product Reviews</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Consumers often heavily rely on online product reviews, analyzing both quantitative ratings and textual descriptions to assess product quality. However, existing research hasn't adequately addressed how to systematically encourage the creation of comprehensive reviews that capture both customers sentiment and detailed product feature analysis. This paper presents CPR, a novel methodology that leverages the power of Large Language Models (LLMs) and Topic Modeling to guide users in crafting insightful and well-rounded reviews. Our approach employs a three-stage process: first, we present users with product-specific terms for rating; second, we generate targeted phrase suggestions based on these ratings; and third, we integrate user-written text through topic modeling, ensuring all key aspects are addressed. We evaluate CPR using text-to-text LLMs, comparing its performance against real-world customer reviews from Walmart. Our results demonstrate that CPR effectively identifies relevant product terms, even for new products lacking prior reviews, and provides sentiment-aligned phrase suggestions, saving users time and enhancing reviews quality. Quantitative analysis reveals a 12.3% improvement in BLEU score over baseline methods, further supported by manual evaluation of generated phrases. We conclude by discussing potential extensions and future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13989v1">Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40\% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley algorithm. We theoretically demonstrates the superiority of Hadamard matrices in reducing outliers.We achieved 3-bit quantization for weights, activations, and KV cache, significantly enhancing model performance. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15303v1">High-Throughput LLM inference on Heterogeneous Clusters</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Nowadays, many companies possess various types of AI accelerators, forming heterogeneous clusters. Efficiently leveraging these clusters for high-throughput large language model (LLM) inference services can significantly reduce costs and expedite task processing. However, LLM inference on heterogeneous clusters presents two main challenges. Firstly, different deployment configurations can result in vastly different performance. The number of possible configurations is large, and evaluating the effectiveness of a specific setup is complex. Thus, finding an optimal configuration is not an easy task. Secondly, LLM inference instances within a heterogeneous cluster possess varying processing capacities, leading to different processing speeds for handling inference requests. Evaluating these capacities and designing a request scheduling algorithm that fully maximizes the potential of each instance is challenging. In this paper, we propose a high-throughput inference service system on heterogeneous clusters. First, the deployment configuration is optimized by modeling the resource amount and expected throughput and using the exhaustive search method. Second, a novel mechanism is proposed to schedule requests among instances, which fully considers the different processing capabilities of various instances. Extensive experiments show that the proposed scheduler improves throughput by 122.5% and 33.6% on two heterogeneous clusters, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16116v1">DMind Benchmark: The First Comprehensive Benchmark for LLM Evaluation in the Web3 Domain</a></div>
    <div class="paper-meta">
      📅 2025-04-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have led to significant progress on a wide range of natural language processing tasks. However, their effectiveness in specialized and rapidly evolving domains such as Web3 remains underexplored. In this paper, we introduce DMind Benchmark, a novel framework that systematically tests LLMs across nine key categories encompassing blockchain fundamentals, infrastructure, smart contract analysis, decentralized finance (DeFi), decentralized autonomous organizations (DAOs), non-fungible tokens (NFTs), token economics, meme concepts, and security vulnerabilities. DMind Benchmark goes beyond conventional multiple-choice questions by incorporating domain-specific subjective tasks (e.g., smart contract code auditing and repair, numeric reasoning on on-chain data, and fill-in assessments), thereby capturing real-world complexities and stress-testing model adaptability. We evaluate fifteen popular LLMs (from ChatGPT, DeepSeek, Claude, and Gemini series) on DMind Benchmark, uncovering performance gaps in Web3-specific reasoning and application, particularly in emerging areas like token economics and meme concepts. Even the strongest models face significant challenges in identifying subtle security vulnerabilities and analyzing complex DeFi mechanisms. To foster progress in this area, we publicly release our benchmark dataset, evaluation pipeline, and annotated results at http://www.dmind.ai, offering a valuable resource for advancing specialized domain adaptation and the development of more robust Web3-enabled LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16112v1">HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing</a></div>
    <div class="paper-meta">
      📅 2025-04-18
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The attention layer, a core component of Transformer-based LLMs, brings out inefficiencies in current GPU systems due to its low operational intensity and the substantial memory requirements of KV caches. We propose a High-bandwidth Processing Unit (HPU), a memoryintensive co-processor that enhances GPU resource utilization during large-batched LLM inference. By offloading memory-bound operations, the HPU allows the GPU to focus on compute-intensive tasks, increasing overall efficiency. Also, the HPU, as an add-on card, scales out to accommodate surging memory demands driven by large batch sizes and extended sequence lengths. In this paper, we show the HPU prototype implemented with PCIe-based FPGA cards mounted on a GPU system. Our novel GPU-HPU heterogeneous system demonstrates up to 4.1x performance gains and 4.6x energy efficiency improvements over a GPUonly system, providing scalability without increasing the number of GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12553v1">ELAB: Extensive LLM Alignment Benchmark in Persian Language</a></div>
    <div class="paper-meta">
      📅 2025-04-17
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive evaluation framework for aligning Persian Large Language Models (LLMs) with critical ethical dimensions, including safety, fairness, and social norms. It addresses the gaps in existing LLM evaluation frameworks by adapting them to Persian linguistic and cultural contexts. This benchmark creates three types of Persian-language benchmarks: (i) translated data, (ii) new data generated synthetically, and (iii) new naturally collected data. We translate Anthropic Red Teaming data, AdvBench, HarmBench, and DecodingTrust into Persian. Furthermore, we create ProhibiBench-fa, SafeBench-fa, FairBench-fa, and SocialBench-fa as new datasets to address harmful and prohibited content in indigenous culture. Moreover, we collect extensive dataset as GuardBench-fa to consider Persian cultural norms. By combining these datasets, our work establishes a unified framework for evaluating Persian LLMs, offering a new approach to culturally grounded alignment evaluation. A systematic evaluation of Persian LLMs is performed across the three alignment aspects: safety (avoiding harmful content), fairness (mitigating biases), and social norms (adhering to culturally accepted behaviors). We present a publicly available leaderboard that benchmarks Persian LLMs with respect to safety, fairness, and social norms at: https://huggingface.co/spaces/MCILAB/LLM_Alignment_Evaluation.
    </details>
</div>
